"""Spike QC — compute-first utilities.

This module focuses on fast, vectorized computations for per-unit and
population QC, leaving visualization to notebooks. It provides:

- Per-unit metrics: mean FR, ISI violation rate (< ms threshold), presence
  ratio (fraction of time with spikes), FR stability (CV across long bins).
- Time series outputs ready for plotting: FR matrices over time (e.g., 5-min
  bins), active-unit counts per bin, population FR traces.

The functions avoid hidden defaults; thresholds are parameters. Use this with
logger- or video-time spikes alike (seconds domain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..transforms.spikes import bin_events, firing_rate, zscore_by_unit


@dataclass
class SpikesQCParams:
    """Parameters for spike QC computation.

    Attributes:
        refractory_threshold_s: ISI threshold (s) for refractory violations.
        silence_threshold_s: Max ISI threshold (s) for prolonged silence flag.
        sparse_fr_hz: Median FR (Hz) threshold for sparse units.
        sparse_zero_frac: Fraction of zero bins to flag sparsity.
        drift_rel_thresh: Relative drift (|ΔFR|/mean FR) threshold to flag drift.
        pop_z_event_thresh: Z threshold for population events.
        pop_event_refrac_s: Refractory window (s) for population event detection.
        fano_windows_s: Window sizes (s) to compute Fano factor.
    """

    refractory_threshold_s: float = 0.002
    silence_threshold_s: float = 300.0
    sparse_fr_hz: float = 0.02
    sparse_zero_frac: float = 0.98
    drift_rel_thresh: float = 2.0
    pop_z_event_thresh: float = 3.0
    pop_event_refrac_s: float = 2.0
    fano_windows_s: Tuple[float, ...] = (1.0, 10.0, 60.0)


@dataclass
class SpikesQCBins:
    """Binned spike data for QC time-series plots.

    Attributes:
        edges: (T+1,) bin edges (s)
        centers: (T,) bin centers (s)
        counts: (T, U) spike counts per bin
        fr: (T, U) firing rate per bin (Hz)
        bin_s: Bin width (s)
    """

    edges: np.ndarray
    centers: np.ndarray
    counts: np.ndarray
    fr: np.ndarray
    bin_s: float


@dataclass
class SpikesQCMetrics:
    """Per-unit QC metrics.

    Attributes:
        mean_fr_hz: (U,) mean firing rate across session.
        isi_violation_rate: (U,) fraction of ISIs below threshold.
        presence_ratio: (U,) fraction of bins with at least one spike.
        fr_stability_cv: (U,) coefficient of variation across long bins.
    """

    mean_fr_hz: np.ndarray
    isi_violation_rate: np.ndarray
    presence_ratio: np.ndarray
    fr_stability_cv: np.ndarray


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=float)
    return (e[:-1] + e[1:]) / 2.0


def mean_fr_per_unit(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> np.ndarray:
    """Mean firing rate per unit over the session duration.

    Args:
        t: Spike times (s).
        unit_index: Unit indices.
        t_start: Optional start time (s). Defaults to min(t).
        t_end: Optional end time (s). Defaults to max(t).

    Returns:
        (U,) array of mean FR in Hz.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if t.size == 0 or unit_index.size == 0:
        return np.zeros((0,), dtype=float)
    if t_start is None:
        t_start = float(np.nanmin(t))
    if t_end is None:
        t_end = float(np.nanmax(t))
    dur = max(t_end - t_start, 1e-9)
    n_units = int(unit_index.max()) + 1
    # Count spikes per unit
    counts = np.bincount(unit_index, minlength=n_units).astype(float)
    return counts / float(dur)


def presence_ratio(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    bin_s: float = 1.0,
) -> np.ndarray:
    """Fraction of session bins with at least one spike per unit.

    Args:
        t: Spike times (s).
        unit_index: Unit indices.
        t_start: Optional start time (s). Defaults to min(t).
        t_end: Optional end time (s). Defaults to max(t).
        bin_s: Bin width in seconds (default 1.0 s).

    Returns:
        (U,) presence ratios in [0,1]. Units with zero spikes get 0.0.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if t.size == 0 or unit_index.size == 0:
        return np.zeros((0,), dtype=float)
    if t_start is None:
        t_start = float(np.nanmin(t))
    if t_end is None:
        t_end = float(np.nanmax(t))
    n_units = int(unit_index.max()) + 1
    edges, counts = bin_events(t, unit_index, t_start=t_start, t_end=t_end, bin_s=float(bin_s), n_units=n_units)
    active = (counts > 0).astype(float)
    denom = max(active.shape[0], 1)
    return active.sum(axis=0) / float(denom)


def fr_stability_cv(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    window_s: float = 300.0,
) -> np.ndarray:
    """Coefficient of variation of FR across long bins (e.g., 5-min windows).

    Args:
        t: Spike times (s).
        unit_index: Unit indices.
        t_start: Optional start time (s). Defaults to min(t).
        t_end: Optional end time (s). Defaults to max(t).
        window_s: Aggregation window size in seconds.

    Returns:
        (U,) CV values (std/mean). Units with mean FR ~ 0 get NaN.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if t.size == 0 or unit_index.size == 0:
        return np.zeros((0,), dtype=float)
    if t_start is None:
        t_start = float(np.nanmin(t))
    if t_end is None:
        t_end = float(np.nanmax(t))
    n_units = int(unit_index.max()) + 1
    edges, counts = bin_events(t, unit_index, t_start=t_start, t_end=t_end, bin_s=float(window_s), n_units=n_units)
    fr = firing_rate(counts, window_s)
    mu = np.nanmean(fr, axis=0)
    sd = np.nanstd(fr, axis=0)
    mu = np.where(np.abs(mu) <= 1e-12, np.nan, mu)
    return sd / mu


def fr_time_matrix(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    window_s: float = 300.0,
) -> SpikesQCBins:
    """Firing rate (Hz) per long bin, per unit — ready for heatmaps.

    Args:
        t: Spike times (s).
        unit_index: Unit indices.
        t_start: Optional start time (s). Defaults to min(t).
        t_end: Optional end time (s). Defaults to max(t).
        window_s: Aggregation window size in seconds (default 300s).

    Returns:
        SpikesQCBins with edges, centers, counts[T,U], fr[T,U].
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if t.size == 0 or unit_index.size == 0:
        return SpikesQCBins(
            edges=np.asarray([0.0, float(window_s)]),
            centers=np.asarray([0.5 * float(window_s)]),
            counts=np.zeros((1, 0), dtype=np.int32),
            fr=np.zeros((1, 0), dtype=float),
            bin_s=float(window_s),
        )
    if t_start is None:
        t_start = float(np.nanmin(t))
    if t_end is None:
        t_end = float(np.nanmax(t))
    n_units = int(unit_index.max()) + 1
    edges, counts = bin_events(t, unit_index, t_start=t_start, t_end=t_end, bin_s=float(window_s), n_units=n_units)
    fr = firing_rate(counts, float(window_s))
    centers = _bin_centers(edges)
    return SpikesQCBins(edges=edges, centers=centers, counts=counts, fr=fr, bin_s=float(window_s))


def active_units_over_time(fr: np.ndarray, *, threshold_hz: float = 0.1) -> np.ndarray:
    """Count of units per bin with FR ≥ threshold.

    Args:
        fr: (T, U) firing rate matrix.
        threshold_hz: FR threshold for activity.

    Returns:
        (T,) active unit counts per bin.
    """
    X = np.asarray(fr, dtype=float)
    return np.sum(X >= float(threshold_hz), axis=1)


def _gaussian_kernel(sigma_bins: float, radius_mult: float = 4.0) -> np.ndarray:
    sigma = float(max(sigma_bins, 1e-6))
    R = int(max(1, round(radius_mult * sigma)))
    x = np.arange(-R, R + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def population_fr_trace(fr: np.ndarray, *, smooth_sigma_bins: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Population firing rate over time (sum over units), optional Gaussian smoothing.

    Args:
        fr: (T, U) firing rate matrix.
        smooth_sigma_bins: Optional Gaussian sigma in bins. If None, returns raw only.

    Returns:
        (raw, smoothed_or_None)
    """
    X = np.asarray(fr, dtype=float)
    raw = np.nansum(X, axis=1)
    if smooth_sigma_bins is None or smooth_sigma_bins <= 0:
        return raw, None
    k = _gaussian_kernel(float(smooth_sigma_bins))
    sm = np.convolve(raw, k, mode="same")
    return raw, sm


def compute_qc_metrics_bundle(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    isi_threshold_s: float = 0.0015,
    presence_bin_s: float = 1.0,
    fr_window_s: float = 300.0,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[SpikesQCMetrics, SpikesQCBins]:
    """Compute core QC metrics and FR time matrix for plots.

    Args:
        t: Spike times (s).
        unit_index: Unit indices.
        isi_threshold_s: Refractory threshold for violation rate (s), default 1.5 ms.
        presence_bin_s: Bin size for presence ratio.
        fr_window_s: Window for FR heatmap and stability (default 300 s).
        t_start: Optional session start time. Defaults to min(t).
        t_end: Optional session end time. Defaults to max(t).

    Returns:
        (metrics, bins) where metrics are per-unit and bins contain FR over time.
    """
    # ISI metrics at the requested threshold
    isi = compute_isi_metrics(t, unit_index, refractory_threshold_s=float(isi_threshold_s))
    isi_rate = isi["refractory_violation_rate"].astype(float)

    mu_fr = mean_fr_per_unit(t, unit_index, t_start=t_start, t_end=t_end)
    pres = presence_ratio(t, unit_index, t_start=t_start, t_end=t_end, bin_s=float(presence_bin_s))
    cv = fr_stability_cv(t, unit_index, t_start=t_start, t_end=t_end, window_s=float(fr_window_s))
    bins = fr_time_matrix(t, unit_index, t_start=t_start, t_end=t_end, window_s=float(fr_window_s))

    metrics = SpikesQCMetrics(
        mean_fr_hz=mu_fr,
        isi_violation_rate=isi_rate,
        presence_ratio=pres,
        fr_stability_cv=cv,
    )
    return metrics, bins


def _qc_outdir_for_session(session) -> "np.ndarray":  # type: ignore[name-defined]
    """Resolve the QC output directory for a session.

    Prefers session.data_files['output_dir']/qc; falls back to results/<session_id>/qc.
    """
    from pathlib import Path

    try:
        base = session.data_files.get("output_dir") or session.metadata.get("output_dir")
    except Exception:
        base = None
    if base:
        return Path(str(base)).expanduser() / "qc"
    return Path("results") / session.session_id / "qc"


def load_keep_mask_for_session(
    session,
    tag: str,
    *,
    U: int,
    unit_ids: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Load per-unit QC keep mask for a given logger tag.

    Strategy mirrors analysis scripts:
      1) Try keep_units_{sid}_{tag}.json
      2) Try qc_spikes_{sid}_{tag}.json
      3) Try qc_spikes_{sid}.json

    If mask length mismatches U, attempt alignment via provided unit_ids
    when the JSON also contains unit_ids.

    Args:
        session: Session object from freemat_data.registry.load_session_yaml.
        tag: 'm1' or 'm2'.
        U: Expected number of units.
        unit_ids: Optional unit identifiers to align mismatched masks.

    Returns:
        (mask, path) where mask is a (U,) boolean array or None if not found.
    """
    import json
    import numpy as _np

    sid = session.session_id
    qc_dir = _qc_outdir_for_session(session)
    candidates = [
        qc_dir / f"keep_units_{sid}_{tag}.json",
        qc_dir / f"qc_spikes_{sid}_{tag}.json",
        qc_dir / f"qc_spikes_{sid}.json",
    ]
    for p in candidates:
        try:
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            keep_list = None
            if isinstance(obj, dict):
                if "mask" in obj and isinstance(obj["mask"], list):
                    keep_list = obj["mask"]
                elif (
                    "keep" in obj
                    and isinstance(obj["keep"], dict)
                    and isinstance(obj["keep"].get("mask"), list)
                ):
                    keep_list = obj["keep"]["mask"]
            if keep_list is None:
                continue
            keep = _np.asarray(keep_list, dtype=bool)
            if keep.size == int(U):
                return keep, str(p)
            # Attempt unit_ids alignment when sizes differ
            ids_qc = None
            if isinstance(obj, dict):
                if "unit_ids" in obj and isinstance(obj["unit_ids"], list):
                    ids_qc = _np.asarray(obj["unit_ids"], dtype=object)
                elif (
                    "keep" in obj
                    and isinstance(obj["keep"], dict)
                    and isinstance(obj["keep"].get("unit_ids"), list)
                ):
                    ids_qc = _np.asarray(obj["keep"]["unit_ids"], dtype=object)
            if unit_ids is not None and ids_qc is not None and ids_qc.size == keep.size:
                lut = {str(u): i for i, u in enumerate(ids_qc)}
                mask = _np.zeros((int(U),), dtype=bool)
                for i in range(int(U)):
                    key = str(unit_ids[i])
                    j = lut.get(key)
                    if j is not None:
                        mask[i] = bool(keep[j])
                return mask, str(p)
        except Exception:
            continue
    return None, None


def _per_unit_slices(unit_index: np.ndarray, n_units: int) -> List[np.ndarray]:
    """Return boolean masks per unit for indexing events.

    For performance, unit-wise loops are acceptable (U ~ few hundred). Events
    can be very large, so per-unit slicing avoids heavy duplication.
    """
    masks = []
    for u in range(n_units):
        masks.append(unit_index == u)
    return masks


def compute_isi_metrics(
    t: np.ndarray,
    unit_index: np.ndarray,
    n_units: Optional[int] = None,
    *,
    refractory_threshold_s: float = 0.002,
) -> Dict[str, np.ndarray]:
    """Compute ISI-based metrics per unit.

    Returns arrays of length U: isi_median, isi_mean, cv2, lv, ref_violation_rate.
    Units with < 2 spikes get NaNs for metrics.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if n_units is None:
        n_units = int(unit_index.max()) + 1 if unit_index.size else 0
    isi_median = np.full((n_units,), np.nan, dtype=float)
    isi_mean = np.full((n_units,), np.nan, dtype=float)
    cv2 = np.full((n_units,), np.nan, dtype=float)
    lv = np.full((n_units,), np.nan, dtype=float)
    rv_rate = np.full((n_units,), np.nan, dtype=float)

    for u in range(n_units):
        mask = unit_index == u
        if not np.any(mask):
            continue
        tu = np.sort(t[mask])
        if tu.size < 2:
            continue
        isi = np.diff(tu)
        isi_median[u] = float(np.median(isi))
        isi_mean[u] = float(np.mean(isi))
        # CV2 = 2|ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i)
        if isi.size >= 2:
            num = 2.0 * np.abs(isi[1:] - isi[:-1])
            den = (isi[1:] + isi[:-1]) + 1e-12
            cv2[u] = float(np.mean(num / den))
            # LV (local variation) ~= 3 * Σ((ISI_{i+1} - ISI_i)^2 / (ISI_{i+1} + ISI_i)^2) / (N-1)
            lv[u] = float(3.0 * np.mean(((isi[1:] - isi[:-1]) ** 2) / (den ** 2)))
        rv_rate[u] = float(np.mean(isi < refractory_threshold_s))
    return {
        "isi_median": isi_median,
        "isi_mean": isi_mean,
        "cv2": cv2,
        "lv": lv,
        "refractory_violation_rate": rv_rate,
    }


def _quartile_means(fr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_all, mean_q1, mean_q4) per unit from FR[T,U]."""
    T, U = fr.shape
    mean_all = np.nanmean(fr, axis=0)
    q = int(T // 4)
    if q == 0:
        return mean_all, mean_all.copy(), mean_all.copy()
    mean_q1 = np.nanmean(fr[:q, :], axis=0)
    mean_q4 = np.nanmean(fr[-q:, :], axis=0)
    return mean_all, mean_q1, mean_q4


def compute_drift_metrics(fr: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute drift metrics from FR[T,U]: delta between first and last quartile.

    Returns per-unit arrays: drift_delta_hz, drift_delta_rel.
    """
    mu, q1, q4 = _quartile_means(fr)
    delta = q4 - q1
    denom = np.where(np.abs(mu) > 1e-12, np.abs(mu), 1.0)
    rel = delta / denom
    return {"drift_delta_hz": delta, "drift_delta_rel": rel}


def compute_population_metrics(
    fr: np.ndarray,
    *,
    bin_s: float,
    banks: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute population-level metrics and participation ratio (optionally per bank).

    - pop_rate: sum FR over units (T,)
    - pop_rate_z: z-scored across time (T,)
    - pop_events_idx: indices of detected population events (after refractory)
    - participation_ratio_all and per bank (if provided)
    - population_coupling per unit (correlation with pop excluding unit)
    """
    T, U = fr.shape
    pop_rate = np.nansum(fr, axis=1)
    # z-score across time
    pr_mu = float(np.nanmean(pop_rate))
    pr_sd = float(np.nanstd(pop_rate)) or 1.0
    pop_rate_z = (pop_rate - pr_mu) / pr_sd

    # Participation ratio via eigenvalues of covariance of FR_z
    Z = zscore_by_unit(fr)
    C = (Z.T @ Z) / max(T - 1, 1)
    try:
        evals = np.linalg.eigvalsh(C)
    except np.linalg.LinAlgError:
        evals = np.linalg.eigvals(C).real
    s1 = float(np.sum(evals))
    s2 = float(np.sum(evals ** 2)) or 1.0
    pr_all = (s1 * s1) / s2

    # Per-bank PR if provided
    pr_by_bank: Dict[str, float] = {}
    if banks is not None:
        banks = np.asarray(banks)
        for label in ("dorsal", "ventral"):
            sel = np.where(banks == label)[0]
            if sel.size >= 2:
                Zb = Z[:, sel]
                Cb = (Zb.T @ Zb) / max(T - 1, 1)
                try:
                    ev = np.linalg.eigvalsh(Cb)
                except np.linalg.LinAlgError:
                    ev = np.linalg.eigvals(Cb).real
                s1b = float(np.sum(ev))
                s2b = float(np.sum(ev ** 2)) or 1.0
                pr_by_bank[label] = (s1b * s1b) / s2b

    # Population coupling (vectorized)
    if U >= 2:
        sumZ = np.nansum(Z, axis=1)  # (T,)
        denom = np.sqrt(max(U - 1, 1))
        P = (sumZ[:, None] - Z) / denom  # (T, U)
        coupling = np.einsum("tu,tu->u", Z, P) / float(T)
    else:
        coupling = np.zeros((U,), dtype=float)

    return {
        "pop_rate": pop_rate,
        "pop_rate_z": pop_rate_z,
        "participation_ratio": pr_all,
        "participation_ratio_by_bank": pr_by_bank,
        "population_coupling": coupling.astype(float, copy=False),
    }


def detect_population_events(
    pop_rate_z: np.ndarray,
    *,
    bin_s: float,
    z_thresh: float,
    refrac_s: float,
) -> np.ndarray:
    """Return indices of population events after applying z-threshold and refractory.

    Uses a simple greedy refractory: once a timepoint is selected as an event,
    suppress the next R bins, where R = ceil(refrac_s / bin_s).
    """
    x = np.asarray(pop_rate_z, dtype=float).copy()
    T = x.shape[0]
    thr = float(z_thresh)
    R = int(np.ceil(float(refrac_s) / float(bin_s)))
    if R < 1:
        R = 1
    sel = np.zeros((T,), dtype=bool)
    i = 0
    while i < T:
        if x[i] >= thr:
            sel[i] = True
            i += R
        else:
            i += 1
    return np.where(sel)[0]


def fano_factor_counts(
    counts: np.ndarray,
    *,
    bin_s: float,
    windows_s: Iterable[float],
    min_blocks: int = 3,
    ddof: int = 1,
) -> Dict[float, np.ndarray]:
    """Compute Fano factor per unit for several window sizes with guardrails.

    Args:
        counts: (T, U) integer spike counts at base bin size `bin_s`.
        bin_s: Base bin width (seconds) for `counts`.
        windows_s: Iterable of window sizes (seconds) to aggregate over.
        min_blocks: Minimum number of non-overlapping blocks required. If the
            number of blocks B < min_blocks for a given window, returns NaN for
            all units at that window.
        ddof: Degrees of freedom for variance across blocks (default 1). With
            B <= ddof, variance is NaN by definition and thus Fano becomes NaN.

    Returns:
        Dict[window_s -> (U,) Fano values], computed as var(block sums)/mean(block sums).
    """
    counts = np.asarray(counts, dtype=float)
    T, U = counts.shape
    out: Dict[float, np.ndarray] = {}
    for W in windows_s:
        n = int(round(float(W) / float(bin_s)))
        if n <= 0 or n > T:
            out[float(W)] = np.full((U,), np.nan, dtype=float)
            continue
        # Trim to multiple of n to form non-overlapping blocks
        M = (T // n) * n
        if M < n:
            out[float(W)] = np.full((U,), np.nan, dtype=float)
            continue
        C = counts[:M, :]
        B = M // n
        if B < int(min_blocks):
            out[float(W)] = np.full((U,), np.nan, dtype=float)
            continue
        # Reshape to (blocks, n, U) and sum over n
        Cs = C.reshape(B, n, U).sum(axis=1)
        mu = np.mean(Cs, axis=0)
        va = np.var(Cs, axis=0, ddof=int(ddof))
        ff = np.where(mu > 0, va / mu, np.nan)
        out[float(W)] = ff
    return out


@dataclass
class KeepThresholds:
    """Thresholds for QC keep-list policy.

    Attributes:
        min_mean_fr_hz: Minimum mean FR to consider a unit active.
        max_isi_violation_rate: Maximum fraction of ISIs below refractory threshold.
        min_presence_ratio: Minimum fraction of time bins with spikes.
        max_fr_stability_cv: Maximum FR coefficient of variation across long bins.
    """

    min_mean_fr_hz: float = 0.02
    max_isi_violation_rate: float = 0.01
    min_presence_ratio: float = 0.5
    max_fr_stability_cv: float = 2.0


def deduplicate_isi_ties(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    isi_threshold_s: float = 0.0015,
) -> np.ndarray:
    """Recompute ISI violation rate after removing exact-tie timestamps.

    Unit_rasters expand binned counts into events at bin edges, producing
    many exact-tie timestamps (ISI = 0). This function deduplicates per-unit
    timestamps before computing violation rates, avoiding inflated metrics.

    Args:
        t: Spike times (s).
        unit_index: Unit indices.
        isi_threshold_s: Refractory threshold (s) for violation rate.

    Returns:
        (U,) array of ISI violation rates after dedup. Units with < 2 unique
        timestamps get NaN.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    n_units = int(np.max(unit_index)) + 1 if unit_index.size else 0
    isi_rate = np.full((n_units,), np.nan, dtype=float)
    thr_sec = float(isi_threshold_s)
    for u in range(n_units):
        m = unit_index == u
        if not np.any(m):
            continue
        tt = np.sort(t[m])
        if tt.size < 2:
            continue
        # Remove exact ties (bin expansion artifacts)
        keep = np.ones(tt.size, dtype=bool)
        keep[1:] = tt[1:] > tt[:-1]
        ttu = tt[keep]
        if ttu.size < 2:
            continue
        isi = np.diff(ttu)
        isi_rate[u] = float(np.mean(isi < thr_sec)) if isi.size else np.nan
    return isi_rate


def compute_keep_mask(
    metrics: SpikesQCMetrics,
    thresholds: KeepThresholds,
) -> np.ndarray:
    """Return boolean mask of units to keep based on threshold criteria.

    A unit is kept only if it passes ALL four criteria simultaneously.

    Args:
        metrics: Per-unit QC metrics (mean FR, ISI violation, presence, CV).
        thresholds: Threshold values for each metric.

    Returns:
        Boolean array of shape (n_units,). True = keep.
    """
    keep = (
        (metrics.mean_fr_hz >= float(thresholds.min_mean_fr_hz))
        & (metrics.isi_violation_rate <= float(thresholds.max_isi_violation_rate))
        & (metrics.presence_ratio >= float(thresholds.min_presence_ratio))
        & (
            np.isfinite(metrics.fr_stability_cv)
            & (metrics.fr_stability_cv <= float(thresholds.max_fr_stability_cv))
        )
    )
    return keep.astype(bool, copy=False)


def compute_breakdown(
    metrics: SpikesQCMetrics,
    keep_mask: np.ndarray,
    thresholds: KeepThresholds,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compute drop-reason breakdown and exclusive primary assignment.

    Args:
        metrics: Per-unit QC metrics.
        keep_mask: Boolean keep mask from compute_keep_mask().
        thresholds: Threshold values used.

    Returns:
        (breakdown, exclusive_counts) where:
        - breakdown: non-exclusive counts per reason + kept/dropped/total
        - exclusive_counts: each dropped unit assigned to exactly one
          primary reason (priority: low_fr > high_isi > low_presence >
          high_cv > cv_nan)
    """
    low_mean_fr = metrics.mean_fr_hz < float(thresholds.min_mean_fr_hz)
    high_isi = metrics.isi_violation_rate > float(thresholds.max_isi_violation_rate)
    low_presence = metrics.presence_ratio < float(thresholds.min_presence_ratio)
    cv_nan = ~np.isfinite(metrics.fr_stability_cv)
    high_fr_cv = np.isfinite(metrics.fr_stability_cv) & (
        metrics.fr_stability_cv > float(thresholds.max_fr_stability_cv)
    )
    dropped_mask = low_mean_fr | high_isi | low_presence | high_fr_cv | cv_nan

    breakdown = {
        "low_mean_fr": int(np.sum(low_mean_fr)),
        "high_isi_violation": int(np.sum(high_isi)),
        "low_presence": int(np.sum(low_presence)),
        "high_fr_cv": int(np.sum(high_fr_cv)),
        "cv_nan": int(np.sum(cv_nan)),
        "kept": int(np.sum(keep_mask)),
        "dropped": int(np.sum(dropped_mask)),
        "total": int(keep_mask.size),
    }

    reason_order = [
        ("low_mean_fr", low_mean_fr),
        ("high_isi_violation", high_isi),
        ("low_presence", low_presence),
        ("high_fr_cv", high_fr_cv),
        ("cv_nan", cv_nan),
    ]
    primary = np.full_like(keep_mask, fill_value="", dtype=object)
    for name, mask in reason_order:
        assign = (primary == "") & dropped_mask & mask
        primary[assign] = name
    exclusive_counts: Dict[str, int] = {
        k: int(np.sum(primary == k)) for k, _ in reason_order
    }
    return breakdown, exclusive_counts


def compute_qc(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    bin_s: float = 1.0,
    n_units: Optional[int] = None,
    unit_channels: Optional[np.ndarray] = None,
    banks: Optional[np.ndarray] = None,
    params: Optional[SpikesQCParams] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Compute per-unit and population QC metrics from events.

    Args:
        t: Event times (s) shape (E,).
        unit_index: Unit indices (E,) in [0,U).
        t_start/t_end: Optional bounds for binning; defaults to min/max of t.
        bin_s: Bin width for derived metrics (counts, FR, drift, fano, etc.).
        n_units: Optional number of units; inferred if None.
        unit_channels: Optional per-unit channels (U,).
        banks: Optional per-unit bank labels (U,).
        params: Optional SpikesQCParams; defaults to conservative thresholds.

    Returns:
        (per_unit, summary) dictionaries:
          per_unit: keys include unit_index, bank, channel (if provided),
                    metrics arrays and flags (including dropped).
          summary: global/population metrics and parameter metadata.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if n_units is None:
        n_units = int(unit_index.max()) + 1 if unit_index.size else 0
    if params is None:
        params = SpikesQCParams()

    if t.size:
        t0 = float(np.min(t)) if t_start is None else float(t_start)
        t1 = float(np.max(t)) if t_end is None else float(t_end)
    else:
        t0 = float(t_start or 0.0)
        t1 = float(t_end or (t0 + 1.0))

    edges, counts = bin_events(
        t,
        unit_index,
        t_start=t0,
        t_end=t1,
        bin_s=float(bin_s),
        n_units=n_units,
    )
    fr = firing_rate(counts, float(bin_s))  # (T,U)

    # Per-unit metrics
    isi = compute_isi_metrics(t, unit_index, n_units=n_units, refractory_threshold_s=params.refractory_threshold_s)
    drift = compute_drift_metrics(fr)
    zero_bin_frac = np.mean(counts == 0, axis=0)
    median_fr = np.median(fr, axis=0)
    max_isi = np.full((n_units,), np.nan, dtype=float)
    for u in range(n_units):
        mask = unit_index == u
        if np.any(mask):
            tu = np.sort(t[mask])
            if tu.size >= 2:
                max_isi[u] = float(np.max(np.diff(tu)))

    # Flags and conservative drop preset
    refractory_flag = isi["refractory_violation_rate"] >= 0.01
    too_sparse = (median_fr < params.sparse_fr_hz) & (zero_bin_frac > params.sparse_zero_frac)
    prolonged_silence = (max_isi >= params.silence_threshold_s) & (median_fr < 0.05)
    extreme_drift = np.abs(drift["drift_delta_rel"]) >= params.drift_rel_thresh
    bursty = isi["lv"] >= 1.5
    dropped = refractory_flag | too_sparse | prolonged_silence | extreme_drift

    # Population metrics
    pop = compute_population_metrics(fr, bin_s=float(bin_s), banks=banks)
    pop_events_idx = detect_population_events(
        pop["pop_rate_z"],
        bin_s=float(bin_s),
        z_thresh=params.pop_z_event_thresh,
        refrac_s=params.pop_event_refrac_s,
    )

    per_unit: Dict[str, np.ndarray] = {
        "unit_index": np.arange(n_units, dtype=int),
        "median_fr": median_fr.astype(float, copy=False),
        "zero_bin_frac": zero_bin_frac.astype(float, copy=False),
        "max_isi": max_isi,
        "drift_delta_hz": drift["drift_delta_hz"].astype(float, copy=False),
        "drift_delta_rel": drift["drift_delta_rel"].astype(float, copy=False),
        "refractory_violation_rate": isi["refractory_violation_rate"].astype(float, copy=False),
        "isi_median": isi["isi_median"].astype(float, copy=False),
        "isi_mean": isi["isi_mean"].astype(float, copy=False),
        "cv2": isi["cv2"].astype(float, copy=False),
        "lv": isi["lv"].astype(float, copy=False),
        "bursty": bursty.astype(bool, copy=False),
        "too_sparse": too_sparse.astype(bool, copy=False),
        "prolonged_silence": prolonged_silence.astype(bool, copy=False),
        "refractory_flag": refractory_flag.astype(bool, copy=False),
        "extreme_drift": extreme_drift.astype(bool, copy=False),
        "dropped": dropped.astype(bool, copy=False),
        "population_coupling": np.asarray(pop["population_coupling"], dtype=float),
    }
    if unit_channels is not None:
        per_unit["channel"] = np.asarray(unit_channels).astype(int, copy=False)
    if banks is not None:
        per_unit["bank"] = np.asarray(banks)

    summary: Dict[str, object] = {
        "bin_s": float(bin_s),
        "t_start": t0,
        "t_end": t1,
        "n_units": int(n_units),
        "n_dropped": int(np.sum(dropped)),
        "params": params.__dict__.copy(),
        "pop_rate": np.asarray(pop["pop_rate"], dtype=float),
        "pop_rate_z": np.asarray(pop["pop_rate_z"], dtype=float),
        "pop_events_idx": np.asarray(pop_events_idx, dtype=int),
        "participation_ratio": float(pop["participation_ratio"]),
        "participation_ratio_by_bank": dict(pop.get("participation_ratio_by_bank") or {}),
    }
    return per_unit, summary
