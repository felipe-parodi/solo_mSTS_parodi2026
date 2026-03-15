"""Utilities for lagged decoding around bout onsets in mid-STS.

This module factors out reusable computation from scripts/lagged_decoding_midsts.py
so the script can focus on orchestration (I/O, CLI, and plotting).

Key functions:
- bin_and_smooth: bin spike times and Gaussian-smooth rates
- make_trials: extract per-bout trial tensors around onsets
- build_samples_at_time: build positive/negative samples for decoding
- make_contiguous_fold_edges: contiguous fold boundaries (optionally snapped)
- assign_fold_id_for_center: fold assignment with margins
- clf_cv_accuracy: cross-validated LinearSVC balanced accuracy
- decode_curve: full lagged decoding curve with null baselines

Example:
    >>> from data_utils.midsts_decoding import decode_curve, Recording
    >>> times_ms, *curves = decode_curve(
    ...     t, rates, unit_mask, bout_onsets_s,
    ...     window_s=(-1.0, 2.0), step_ms=125, decode_win_ms=250,
    ...     n_folds=5, n_boot=200, null_shift_s=120.0,
    ...     min_pos_trials=15, seed=0, cv_mode="stratified",
    ...     bin_ms=250, smooth_ms=125,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


__all__ = [
    "Recording",
    "bin_and_smooth",
    "make_trials",
    "window_mean",
    "nonbout_mask",
    "build_samples_at_time",
    "make_contiguous_fold_edges",
    "assign_fold_id_for_center",
    "compute_sample_weights",
    "clf_cv_accuracy",
    "assert_no_cross_fold_leakage",
    "decode_curve",
    "rebinned_t_rates",
]


@dataclass
class Recording:
    """Per-subject recording container used by lagged decoding.

    Attributes:
        t: Time centers in seconds for binned/smoothed rates, shape (T,)
        rates: Firing rates (Hz), shape (T, U)
        bank: Per-unit bank labels (U,) in {'dorsal','ventral','unknown'}
        bouts_self: Bout onset times (seconds) where the subject is the actor
        bouts_partner: Bout onset times (seconds) where the partner acts
        segments_self: Behavior segments (start, end) seconds for self
        segments_partner: Behavior segments (start, end) seconds for partner
        duration: Total duration in seconds
        spikes_by_unit: Optional raw spike times per unit (seconds) for re-binning
        t_start: Start time used when re-binning
        t_end: End time used when re-binning
    """

    t: np.ndarray
    rates: np.ndarray
    bank: np.ndarray
    bouts_self: np.ndarray
    bouts_partner: np.ndarray
    segments_self: np.ndarray
    segments_partner: np.ndarray
    duration: float
    spikes_by_unit: Optional[List[np.ndarray]] = None
    t_start: float = 0.0
    t_end: float = 0.0
    # Boolean mask of length U_raw indicating which units were kept in rates/bank.
    # When present, any re-binning should apply the same mask to keep alignment.
    unit_keep_mask: Optional[np.ndarray] = None


def bin_and_smooth(
    spike_times_by_unit: Optional[List[np.ndarray]],
    *,
    t_start: float,
    t_end: float,
    bin_ms: int,
    smooth_ms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin spikes into rates and Gaussian-smooth along time.

    Args:
        spike_times_by_unit: List of spike time arrays (seconds) per unit; None → empty outputs.
        t_start: Inclusive start time in seconds.
        t_end: Inclusive end time in seconds.
        bin_ms: Bin width in milliseconds.
        smooth_ms: Gaussian sigma (ms) for smoothing on time axis.

    Returns:
        Tuple (t_centers, rates_hz):
            t_centers: (T,) bin centers in seconds
            rates_hz: (T, U) smoothed firing rates in Hz
    """
    if spike_times_by_unit is None or len(spike_times_by_unit) == 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)

    bin_s = float(bin_ms) / 1000.0
    edges = np.arange(float(t_start), float(t_end) + 1e-9, bin_s, dtype=float)
    if edges.size < 2:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)
    t_centers = (edges[:-1] + edges[1:]) * 0.5

    U = len(spike_times_by_unit)
    counts = np.zeros((edges.size - 1, U), dtype=float)
    for u, st in enumerate(spike_times_by_unit):
        if st is None:
            continue
        st_arr = np.asarray(st, dtype=float)
        if st_arr.size == 0:
            continue
        hist, _ = np.histogram(st_arr, bins=edges)
        counts[:, u] = hist.astype(float)

    rates = counts / bin_s
    sigma_bins = (float(smooth_ms) / 1000.0) / bin_s if smooth_ms > 0 else 0.0
    if sigma_bins > 0:
        rates = gaussian_filter1d(rates, sigma=sigma_bins, axis=0, mode="nearest")
    return t_centers, rates


def make_trials(
    t: np.ndarray,
    rates: np.ndarray,
    bout_starts_s: np.ndarray,
    window_s: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (N_trials, T_win, U) trials around each bout onset within bounds.

    Returns trials tensor and kept onset times. Empty arrays when no valid trials.
    """
    if t.size == 0 or rates.size == 0:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0,), dtype=float)

    tmin_rel, tmax_rel = float(window_s[0]), float(window_s[1])
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if dt <= 0:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0,), dtype=float)

    trials: List[np.ndarray] = []
    kept: List[float] = []
    for onset in np.asarray(bout_starts_s, dtype=float).ravel():
        t0 = onset + tmin_rel
        t1 = onset + tmax_rel
        if t0 < t[0] or t1 > t[-1]:
            continue
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="right"))
        if i1 <= i0:
            continue
        seg = rates[i0:i1, :]
        trials.append(seg)
        kept.append(float(onset))

    if not trials:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0,), dtype=float)

    L = min(tr.shape[0] for tr in trials)
    trials = [tr[:L, :] for tr in trials]
    return np.stack(trials, axis=0), np.asarray(kept, dtype=float)


def window_mean(
    t: np.ndarray,
    rates: np.ndarray,
    center_s: float,
    width_ms: int,
    unit_mask: np.ndarray,
) -> np.ndarray:
    """Average rates within a centered window; returns (U_masked,) or empty if OOB.

    Args:
        t: Time vector (T,)
        rates: Rate matrix (T, U)
        center_s: Center time (seconds)
        width_ms: Window width (ms)
        unit_mask: Boolean mask over units (U,)
    """
    if t.size == 0 or rates.size == 0 or not np.any(unit_mask):
        return np.zeros((0,), dtype=float)
    half_w = (float(width_ms) / 1000.0) * 0.5
    t0 = center_s - half_w
    t1 = center_s + half_w
    if t0 < t[0] or t1 > t[-1]:
        return np.zeros((0,), dtype=float)
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    if i1 <= i0:
        return np.zeros((0,), dtype=float)
    sel = rates[i0:i1, :]
    if sel.size == 0:
        return np.zeros((0,), dtype=float)
    v = sel.mean(axis=0)
    return v[unit_mask]


def nonbout_mask(
    duration_s: float,
    t: np.ndarray,
    bout_centers_s: np.ndarray,
    decode_win_ms: int,
) -> np.ndarray:
    """Boolean mask over time centers that excludes ±(decode_win/2) around onsets.

    Args:
        duration_s: Approx. duration of recording (for periodic wrap safety)
        t: Time centers (T,)
        bout_centers_s: Bout onset centers (N,)
        decode_win_ms: Decode window width in ms
    """
    if t.size == 0:
        return np.zeros((0,), dtype=bool)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if dt <= 0:
        return np.zeros((0,), dtype=bool)
    half_w = (float(decode_win_ms) / 1000.0) * 0.5
    mask = np.ones(t.shape[0], dtype=bool)
    centers = np.asarray(bout_centers_s, dtype=float)
    for c in centers:
        left = c - half_w
        right = c + half_w
        i0 = int(np.searchsorted(t, left, side="left"))
        i1 = int(np.searchsorted(t, right, side="right"))
        i0 = max(0, min(i0, t.shape[0]))
        i1 = max(0, min(i1, t.shape[0]))
        if i1 > i0:
            mask[i0:i1] = False
    return mask


def build_samples_at_time(
    t: np.ndarray,
    rates: np.ndarray,
    unit_mask: np.ndarray,
    bout_onsets_s: np.ndarray,
    rel_time_s: float,
    decode_win_ms: int,
    rng: np.random.Generator,
    *,
    exclude_neg_segments: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build positive/negative samples at a relative time offset.

    Returns X_pos, X_neg, t_centers_pos, t_centers_neg.
    """
    if t.size == 0 or rates.size == 0 or not np.any(unit_mask):
        z = np.zeros((0, 0), dtype=float)
        return z, z, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    # Positive samples from windows around onset + rel_time
    pos_vecs: List[np.ndarray] = []
    t_pos: List[float] = []
    for onset in np.asarray(bout_onsets_s, dtype=float).ravel():
        c = float(onset) + float(rel_time_s)
        v = window_mean(t, rates, c, int(decode_win_ms), unit_mask)
        if v.size > 0:
            pos_vecs.append(v)
            t_pos.append(c)
    if not pos_vecs:
        z = np.zeros((0, 0), dtype=float)
        return z, z, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    X_pos = np.stack(pos_vecs, axis=0)
    t_pos_arr = np.asarray(t_pos, dtype=float)

    # Negatives: sample from candidate bins with valid window extent
    # Start from non-bout mask (excludes ±decode_win/2 around event centers)
    non_mask = nonbout_mask(float(t[-1] - t[0]) + (t[1] - t[0]), t, bout_onsets_s, int(decode_win_ms))

    # Optionally enforce strict negatives: exclude any time that lies inside provided segments
    if exclude_neg_segments is not None and np.asarray(exclude_neg_segments).size > 0:
        seg = np.asarray(exclude_neg_segments, dtype=float).reshape(-1, 2)
        strict_mask = np.ones_like(non_mask, dtype=bool)
        for s0, s1 in seg:
            i0 = int(np.searchsorted(t, float(s0), side="left"))
            i1 = int(np.searchsorted(t, float(s1), side="right"))
            i0 = max(0, min(i0, t.shape[0]))
            i1 = max(0, min(i1, t.shape[0]))
            if i1 > i0:
                strict_mask[i0:i1] = False
        non_mask &= strict_mask
    idx_candidates = np.nonzero(non_mask)[0]
    if idx_candidates.size == 0:
        return X_pos, np.zeros((0, X_pos.shape[1]), dtype=float), t_pos_arr, np.zeros((0,), dtype=float)
    half_w_s = (float(decode_win_ms) / 1000.0) * 0.5
    valid_idx = []
    for i in idx_candidates:
        c = float(t[i])
        if (c - half_w_s) >= t[0] and (c + half_w_s) <= t[-1]:
            valid_idx.append(i)
    valid_idx = np.asarray(valid_idx, dtype=int)
    if valid_idx.size == 0:
        return X_pos, np.zeros((0, X_pos.shape[1]), dtype=float), t_pos_arr, np.zeros((0,), dtype=float)
    choose = rng.choice(valid_idx, size=X_pos.shape[0], replace=(valid_idx.size < X_pos.shape[0]))
    neg_vecs: List[np.ndarray] = []
    t_neg: List[float] = []
    for i in choose:
        c = float(t[i])
        v = window_mean(t, rates, c, int(decode_win_ms), unit_mask)
        if v.size > 0:
            neg_vecs.append(v)
            t_neg.append(c)
    if not neg_vecs:
        return X_pos, np.zeros((0, X_pos.shape[1]), dtype=float), t_pos_arr, np.zeros((0,), dtype=float)
    X_neg = np.stack(neg_vecs, axis=0)
    return X_pos, X_neg, t_pos_arr, np.asarray(t_neg, dtype=float)


def make_contiguous_fold_edges(
    t: np.ndarray,
    k: int,
    *,
    bin_ms: int,
    decode_win_ms: int,
    smooth_ms: int,
    segments: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute K contiguous fold edges across [t[0], t[-1]]; snap to segments if given.

    Returns an array of length k+1. Falls back to evenly spaced edges on degenerate
    inputs or insufficient boundaries.
    """
    if t.size < 2 or k < 2:
        return np.array([0.0, 1.0], dtype=float)
    t0 = float(t[0])
    t1 = float(t[-1])
    even = np.linspace(t0, t1, num=int(k) + 1, dtype=float)
    if segments is None or np.asarray(segments).size == 0:
        return even
    seg = np.asarray(segments, dtype=float).reshape(-1, 2)
    cand = np.unique(np.clip(seg.ravel(), t0, t1))
    cand = np.unique(np.concatenate([[t0], cand, [t1]]))
    if cand.size < (k + 1):
        return even
    idx = np.round(np.linspace(0, cand.size - 1, num=int(k) + 1)).astype(int)
    idx[0] = 0
    idx[-1] = cand.size - 1
    for j in range(1, idx.size):
        if idx[j] <= idx[j - 1]:
            idx[j] = min(idx[j - 1] + 1, cand.size - (idx.size - j))
    edges = cand[idx]
    if np.any(np.diff(edges) <= 0):
        return even
    return edges


def assign_fold_id_for_center(
    center: float,
    edges: np.ndarray,
    *,
    bin_ms: int,
    decode_win_ms: int,
    smooth_ms: int,
) -> int:
    """Return fold index for a centered window, or -1 if margin containment fails.

    Margin seconds = ((1 + smooth_ms/bin_ms) * bin_ms)/1000.
    """
    half_w = (float(decode_win_ms) / 1000.0) * 0.5
    bin_s = float(bin_ms) / 1000.0
    margin = (1.0 + (float(smooth_ms) / float(max(1, bin_ms)))) * bin_s
    left = center - half_w
    right = center + half_w
    for j in range(edges.size - 1):
        a = float(edges[j]) + margin
        b = float(edges[j + 1]) - margin
        if left >= a and right <= b:
            return int(j)
    return -1


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so each class contributes equally in total weight.

    Handles both binary and multiclass problems.
    """
    y = np.asarray(y, dtype=int).ravel()
    n = y.shape[0]
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = np.ones_like(y, dtype=float)
    for cls in range(n_classes):
        c = counts[cls]
        if c > 0:
            weights[y == cls] = float(n) / (float(n_classes) * float(c))
    return weights


def clf_cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    *,
    fold_ids: Optional[np.ndarray] = None,
    mode: str = "stratified",
    n_boot_train: Optional[int] = None,
    shuffle_train_labels: bool = False,
) -> float:
    """Cross-validated balanced accuracy for LinearSVC.

    - mode="stratified": StratifiedKFold with shuffle.
    - mode="contiguous": blocked folds provided via fold_ids. Optional bootstrap of
      training set size via n_boot_train. Optionally shuffle training labels for nulls.
    """
    if mode == "stratified" or fold_ids is None:
        n_folds_eff = int(max(2, min(n_folds, y.size)))
        skf = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=seed)
        accs: List[float] = []
        for tr, te in skf.split(X, y):
            if X[tr].size == 0 or X[te].size == 0:
                continue
            ytr = y[tr]
            yte = y[te]
            if (np.unique(ytr).size < 2) or (np.unique(yte).size < 2):
                continue
            sw = compute_sample_weights(ytr)
            clf = LinearSVC(loss="squared_hinge", C=1.0, class_weight=None, max_iter=20000, tol=1e-4, random_state=seed)
            clf.fit(X[tr], ytr, sample_weight=sw)
            y_pred = clf.predict(X[te])
            accs.append(float(balanced_accuracy_score(yte, y_pred)))
        return float(np.mean(accs)) if accs else float("nan")

    k = int(n_folds)
    if k < 2:
        return float("nan")
    fold_ids = np.asarray(fold_ids, dtype=int)
    if fold_ids.shape[0] != y.shape[0]:
        return float("nan")
    accs: List[float] = []
    rng_global = np.random.default_rng(seed)
    for f in range(k):
        te_mask = (fold_ids == f)
        tr_mask = ~te_mask
        if not np.any(te_mask) or not np.any(tr_mask):
            continue
        Xtr, ytr = X[tr_mask], y[tr_mask]
        Xte, yte = X[te_mask], y[te_mask]
        if (np.unique(ytr).size < 2) or (np.unique(yte).size < 2):
            continue
        n = Xtr.shape[0]
        m: Optional[int]
        if n_boot_train is None:
            m = n
        elif n_boot_train > 0:
            m = int(n_boot_train)
        else:
            m = None
        if m is not None and m > 0:
            idx = rng.integers(low=0, high=n, size=m, endpoint=False)
            Xtr = Xtr[idx]
            ytr = ytr[idx]
        if shuffle_train_labels:
            ytr = rng.permutation(ytr)
        sw = compute_sample_weights(ytr)
        clf = LinearSVC(loss="squared_hinge", C=1.0, class_weight=None, max_iter=20000, tol=1e-4, random_state=seed)
        clf.fit(Xtr, ytr, sample_weight=sw)
        y_pred = clf.predict(Xte)
        accs.append(float(balanced_accuracy_score(yte, y_pred)))
    return float(np.mean(accs)) if accs else float("nan")


def assert_no_cross_fold_leakage(
    centers: np.ndarray,
    fold_ids: np.ndarray,
    edges: np.ndarray,
    *,
    bin_ms: int,
    decode_win_ms: int,
    smooth_ms: int,
) -> None:
    """Validate that each sample window lies strictly inside its assigned fold.

    Raises AssertionError on violation.
    """
    centers = np.asarray(centers, dtype=float)
    fold_ids = np.asarray(fold_ids, dtype=int)
    if centers.shape[0] != fold_ids.shape[0]:
        raise AssertionError("centers and fold_ids must have matching lengths")
    for c, f in zip(centers, fold_ids):
        if int(f) < 0:
            raise AssertionError("found negative fold id")
        fid = assign_fold_id_for_center(
            float(c), edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
        )
        if int(fid) != int(f):
            raise AssertionError("sample window not strictly contained within its fold with margins")


def decode_curve(
    t: np.ndarray,
    rates: np.ndarray,
    unit_mask: np.ndarray,
    bout_onsets_s: np.ndarray,
    *,
    window_s: Tuple[float, float],
    step_ms: int,
    decode_win_ms: int,
    n_folds: int,
    n_boot: int,
    null_shift_s: float,
    min_pos_trials: int,
    seed: int,
    cv_mode: str = "stratified",
    split_manifest_path: Optional[Path] = None,
    segments_for_edge_snap: Optional[np.ndarray] = None,
    bin_ms: int,
    smooth_ms: int,
    max_pos_per_time: Optional[int] = None,
    exclude_neg_segments: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute lagged decoding accuracy and two null baselines across time.

    Returns time_ms, acc_mean/lo/hi, null_shift_mean/lo/hi, null_shuffle_mean/lo/hi.
    """
    rng = np.random.default_rng(seed)
    step_s = float(step_ms) / 1000.0
    times = np.arange(window_s[0], window_s[1] + 1e-12, step_s, dtype=float)
    if times.size == 0:
        # Always return 10 arrays in the documented order
        z = np.zeros((0,), dtype=float)
        return z, z, z, z, z, z, z, z, z, z

    acc_mean = np.full(times.shape, np.nan, dtype=float)
    acc_lo = np.full(times.shape, np.nan, dtype=float)
    acc_hi = np.full(times.shape, np.nan, dtype=float)
    null_mean = np.full(times.shape, np.nan, dtype=float)
    null_lo = np.full(times.shape, np.nan, dtype=float)
    null_hi = np.full(times.shape, np.nan, dtype=float)
    null_shuf_mean = np.full(times.shape, np.nan, dtype=float)
    null_shuf_lo = np.full(times.shape, np.nan, dtype=float)
    null_shuf_hi = np.full(times.shape, np.nan, dtype=float)

    # Shifted-null onsets (circular shift in [t0, t1])
    if bout_onsets_s is None:
        bout_onsets_s = np.zeros((0,), dtype=float)
    if bout_onsets_s.size > 0:
        dur = float(t[-1] - t[0]) + (t[1] - t[0])
        null_onsets = ((np.asarray(bout_onsets_s) - float(t[0]) + float(null_shift_s)) % dur) + float(t[0])
    else:
        null_onsets = np.asarray([], dtype=float)

    edges: Optional[np.ndarray] = None
    if cv_mode == "contiguous" and t.size:
        edges = make_contiguous_fold_edges(
            t,
            int(n_folds),
            bin_ms=int(bin_ms),
            decode_win_ms=int(decode_win_ms),
            smooth_ms=int(smooth_ms),
            segments=segments_for_edge_snap,
        )
        if split_manifest_path is not None:
            manifest = {
                "edges": [float(x) for x in edges],
                "bin_ms": int(bin_ms),
                "decode_win_ms": int(decode_win_ms),
                "smooth_ms": int(smooth_ms),
                "k": int(n_folds),
                "mode": "contiguous",
                "snapped_to_segments": bool(segments_for_edge_snap is not None and np.asarray(segments_for_edge_snap).size > 0),
            }
            split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

    for i, rel in enumerate(times):
        # Derive a deterministic seed per relative time so results at a given lag
        # are invariant to the overall window range and iteration order.
        rel_ms = int(round(float(rel) * 1000.0))
        time_seed = int(seed) * 1_000_003 + int(rel_ms)
        rng = np.random.default_rng(time_seed)
        X_pos, X_neg, t_pos_cent, t_neg_cent = build_samples_at_time(
            t, rates, unit_mask, bout_onsets_s, float(rel), int(decode_win_ms), rng,
            exclude_neg_segments=exclude_neg_segments,
        )
        if X_pos.shape[0] < int(min_pos_trials) or X_neg.shape[0] < int(min_pos_trials):
            continue

        # Cap the number of positive (and matched negative) samples per timepoint to control runtime
        if max_pos_per_time is not None and X_pos.shape[0] > int(max_pos_per_time):
            m = int(max_pos_per_time)
            idx_sub = rng.choice(X_pos.shape[0], size=m, replace=False)
            X_pos = X_pos[idx_sub]
            t_pos_cent = t_pos_cent[idx_sub]
            # Match negatives by the same subsample size
            if X_neg.shape[0] >= m:
                X_neg = X_neg[idx_sub]
                t_neg_cent = t_neg_cent[idx_sub]
            else:
                # If negatives are fewer (shouldn't happen), resample with replacement
                rep_idx = rng.choice(X_neg.shape[0], size=m, replace=True)
                X_neg = X_neg[rep_idx]
                t_neg_cent = t_neg_cent[rep_idx]

        if cv_mode == "contiguous" and edges is not None:
            fpos = np.array(
                [
                    assign_fold_id_for_center(
                        c, edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
                    )
                    for c in t_pos_cent
                ],
                dtype=int,
            )
            fneg = np.array(
                [
                    assign_fold_id_for_center(
                        c, edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
                    )
                    for c in t_neg_cent
                ],
                dtype=int,
            )
            kp = (fpos >= 0)
            kn = (fneg >= 0)
            if not np.any(kp) or not np.any(kn):
                continue
            Xp = X_pos[kp]
            Xn = X_neg[kn]
            tp = t_pos_cent[kp]
            tn = t_neg_cent[kn]
            fold_ids = np.hstack([fpos[kp], fneg[kn]])
            X = np.vstack([Xp, Xn])
            y = np.hstack([np.ones(Xp.shape[0], dtype=int), np.zeros(Xn.shape[0], dtype=int)])
            assert_no_cross_fold_leakage(
                np.hstack([tp, tn]), fold_ids, edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
            )
            acc_boot: List[float] = []
            for b in range(int(n_boot)):
                a = clf_cv_accuracy(
                    X,
                    y,
                    n_folds=int(n_folds),
                    seed=int(time_seed + b),
                    fold_ids=fold_ids,
                    mode="contiguous",
                    n_boot_train=None,
                )
                if not np.isnan(a):
                    acc_boot.append(a)
            if acc_boot:
                acc_mean[i] = float(np.mean(acc_boot))
                lo, hi = np.percentile(acc_boot, [2.5, 97.5])
                acc_lo[i], acc_hi[i] = float(lo), float(hi)
        else:
            m_pos = int(X_pos.shape[0])
            m_neg = int(X_neg.shape[0])
            acc_boot: List[float] = []
            for b in range(int(n_boot)):
                ip = rng.integers(low=0, high=m_pos, size=m_pos, endpoint=False)
                in_ = rng.integers(low=0, high=m_neg, size=m_neg, endpoint=False)
                Xb = np.vstack([X_pos[ip], X_neg[in_]])
                yb = np.hstack([np.ones(m_pos, dtype=int), np.zeros(m_neg, dtype=int)])
                a = clf_cv_accuracy(Xb, yb, n_folds=int(n_folds), seed=int(time_seed + b), mode="stratified")
                if not np.isnan(a):
                    acc_boot.append(a)
            if acc_boot:
                acc_mean[i] = float(np.mean(acc_boot))
                lo, hi = np.percentile(acc_boot, [2.5, 97.5])
                acc_lo[i], acc_hi[i] = float(lo), float(hi)

        # Shifted-onset null
        if null_onsets.size > 0:
            # Use a distinct but deterministic RNG stream for the shifted-null branch
            rng_null = np.random.default_rng(time_seed + 97)
            Xp0, Xn0, tp0, tn0 = build_samples_at_time(
                t, rates, unit_mask, null_onsets, float(rel), int(decode_win_ms), rng_null,
                exclude_neg_segments=exclude_neg_segments,
            )
            if Xp0.size and Xn0.size:
                # Apply same capping to shifted null if present
                if max_pos_per_time is not None and Xp0.shape[0] > int(max_pos_per_time):
                    m0 = int(max_pos_per_time)
                    idx0 = rng_null.choice(Xp0.shape[0], size=m0, replace=False)
                    Xp0 = Xp0[idx0]
                    tp0 = tp0[idx0]
                    if Xn0.shape[0] >= m0:
                        Xn0 = Xn0[idx0]
                        tn0 = tn0[idx0]
                    else:
                        rep_idx0 = rng_null.choice(Xn0.shape[0], size=m0, replace=True)
                        Xn0 = Xn0[rep_idx0]
                        tn0 = tn0[rep_idx0]

                if cv_mode == "contiguous" and edges is not None:
                    fp0 = np.array(
                        [
                            assign_fold_id_for_center(
                                c, edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
                            )
                            for c in tp0
                        ],
                        dtype=int,
                    )
                    fn0 = np.array(
                        [
                            assign_fold_id_for_center(
                                c, edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
                            )
                            for c in tn0
                        ],
                        dtype=int,
                    )
                    kp = (fp0 >= 0)
                    kn = (fn0 >= 0)
                    if np.any(kp) and np.any(kn):
                        Xp0b = Xp0[kp]
                        Xn0b = Xn0[kn]
                        tp0b = tp0[kp]
                        tn0b = tn0[kn]
                        fold_ids0 = np.hstack([fp0[kp], fn0[kn]])
                        X0 = np.vstack([Xp0b, Xn0b])
                        y0 = np.hstack([np.ones(Xp0b.shape[0], dtype=int), np.zeros(Xn0b.shape[0], dtype=int)])
                        assert_no_cross_fold_leakage(
                            np.hstack([tp0b, tn0b]), fold_ids0, edges, bin_ms=int(bin_ms), decode_win_ms=int(decode_win_ms), smooth_ms=int(smooth_ms)
                        )
                        null_boot: List[float] = []
                        for b in range(int(n_boot)):
                            a0 = clf_cv_accuracy(
                                X0,
                                y0,
                                n_folds=int(n_folds),
                                seed=int(time_seed + 17 + b),
                                fold_ids=fold_ids0,
                                mode="contiguous",
                                n_boot_train=None,
                            )
                            if not np.isnan(a0):
                                null_boot.append(a0)
                        if null_boot:
                            null_mean[i] = float(np.mean(null_boot))
                            lo0, hi0 = np.percentile(null_boot, [2.5, 97.5])
                            null_lo[i], null_hi[i] = float(lo0), float(hi0)
                else:
                    mpos0 = int(Xp0.shape[0])
                    mneg0 = int(Xn0.shape[0])
                    null_boot: List[float] = []
                    for b in range(int(n_boot)):
                        ip = rng_null.integers(low=0, high=mpos0, size=mpos0, endpoint=False)
                        in_ = rng_null.integers(low=0, high=mneg0, size=mneg0, endpoint=False)
                        X0 = np.vstack([Xp0[ip], Xn0[in_]])
                        y0 = np.hstack([np.ones(mpos0, dtype=int), np.zeros(mneg0, dtype=int)])
                        a0 = clf_cv_accuracy(X0, y0, n_folds=int(n_folds), seed=int(time_seed + 17 + b), mode="stratified")
                        if not np.isnan(a0):
                            null_boot.append(a0)
                    if null_boot:
                        null_mean[i] = float(np.mean(null_boot))
                        lo0, hi0 = np.percentile(null_boot, [2.5, 97.5])
                        null_lo[i], null_hi[i] = float(lo0), float(hi0)

        # Within-train label-shuffle null
        if cv_mode == "contiguous" and edges is not None and 'X' in locals():
            null_shuf_boot: List[float] = []
            for b in range(int(n_boot)):
                a_shuf = clf_cv_accuracy(
                    X,
                    y,
                    n_folds=int(n_folds),
                    seed=int(time_seed + 33 + b),
                    fold_ids=fold_ids,
                    mode="contiguous",
                    n_boot_train=None,
                    shuffle_train_labels=True,
                )
                if not np.isnan(a_shuf):
                    null_shuf_boot.append(a_shuf)
            if null_shuf_boot:
                null_shuf_mean[i] = float(np.mean(null_shuf_boot))
                lo_s, hi_s = np.percentile(null_shuf_boot, [2.5, 97.5])
                null_shuf_lo[i], null_shuf_hi[i] = float(lo_s), float(hi_s)
        elif X_pos.size and X_neg.size:
            m_pos = int(X_pos.shape[0])
            m_neg = int(X_neg.shape[0])
            null_shuf_boot: List[float] = []
            for b in range(int(n_boot)):
                ip = rng.integers(low=0, high=m_pos, size=m_pos, endpoint=False)
                in_ = rng.integers(low=0, high=m_neg, size=m_neg, endpoint=False)
                Xb = np.vstack([X_pos[ip], X_neg[in_]])
                yb = np.hstack([np.ones(m_pos, dtype=int), np.zeros(m_neg, dtype=int)])
                a_shuf = clf_cv_accuracy(
                    Xb,
                    yb,
                    n_folds=int(n_folds),
                    seed=int(time_seed + 33 + b),
                    mode="stratified",
                    shuffle_train_labels=True,
                )
                if not np.isnan(a_shuf):
                    null_shuf_boot.append(a_shuf)
            if null_shuf_boot:
                null_shuf_mean[i] = float(np.mean(null_shuf_boot))
                lo_s, hi_s = np.percentile(null_shuf_boot, [2.5, 97.5])
                null_shuf_lo[i], null_shuf_hi[i] = float(lo_s), float(hi_s)

    return (
        (times * 1000.0).astype(float),
        acc_mean,
        acc_lo,
        acc_hi,
        null_mean,
        null_lo,
        null_hi,
        null_shuf_mean,
        null_shuf_lo,
        null_shuf_hi,
    )


def rebinned_t_rates(rec: Recording, *, bin_ms: int, smooth_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t, rates) rebinned from raw spikes when available, aligned to kept units.

    - If rec.spikes_by_unit is missing or invalid, returns existing (rec.t, rec.rates).
    - When re-binning from spikes, applies rec.unit_keep_mask (if provided) so that
      the number and ordering of units match rec.rates/rec.bank. This prevents
      downstream shape mismatches when a unit mask built from rec.bank is applied
      to the rebinned rates.
    """
    if rec.spikes_by_unit is None or len(rec.spikes_by_unit) == 0:
        return rec.t, rec.rates
    t0 = float(rec.t_start)
    t1 = float(rec.t_end)
    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 <= t0):
        return rec.t, rec.rates

    t_centers, rates = bin_and_smooth(
        rec.spikes_by_unit, t_start=t0, t_end=t1, bin_ms=int(bin_ms), smooth_ms=int(smooth_ms)
    )

    # Align to kept units if a mask was recorded; otherwise trust the current rates
    if rec.unit_keep_mask is not None:
        m = np.asarray(rec.unit_keep_mask, dtype=bool)
        # Only apply the mask when its length matches the raw unit count
        if m.ndim == 1 and m.shape[0] == rates.shape[1]:
            rates = rates[:, m]
        else:
            # Fallback: shapes disagree; prefer precomputed alignment to avoid errors
            return rec.t, rec.rates
    return t_centers, rates
