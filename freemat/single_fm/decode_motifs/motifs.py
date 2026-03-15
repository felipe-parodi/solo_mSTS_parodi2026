"""Motif loading, labeling, smoothing, and onset detection."""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _intervals_to_binned_labels(
    motif_intervals: np.ndarray,
    *,
    dt_motif: float,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Convert raw motif intervals (start_idx, end_idx, motif_id) to per-bin labels.

    Args:
        motif_intervals: (M,3) array of [start_idx, end_idx, motif_id]; end exclusive
        dt_motif: Seconds per raw index (e.g., pose_dt)
        bin_edges: Target FR bin edges (seconds)

    Returns:
        y (T,) int labels at bin centers
    """
    M = np.asarray(motif_intervals)
    if M.size == 0:
        return np.zeros((len(bin_edges) - 1,), dtype=int)
    if M.shape[1] != 3:
        raise ValueError("motif_intervals must have shape (M,3)")
    starts = M[:, 0].astype(float) * float(dt_motif)
    ends = M[:, 1].astype(float) * float(dt_motif)
    ids = M[:, 2].astype(int)
    T = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y = np.zeros((T,), dtype=int)
    for s, e, k in zip(starts, ends, ids):
        m = (centers >= s) & (centers < e)
        y[m] = int(k)
    return y


def _map_motifs_to_bins(t_motif: np.ndarray, motif_id: np.ndarray, *, bin_edges: np.ndarray) -> np.ndarray:
    """Map per-frame motif IDs at times t_motif to labels at bin centers.

    Uses nearest-neighbor in time.
    """
    T = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    t = np.asarray(t_motif, float).reshape(-1)
    y = np.asarray(motif_id, int).reshape(-1)
    if t.size == 0 or y.size == 0:
        return np.zeros((T,), dtype=int)
    idx_right = np.searchsorted(t, centers, side="left")
    idx_left = np.maximum(idx_right - 1, 0)
    idx_right = np.minimum(idx_right, t.size - 1)
    # choose closer
    left_dist = np.abs(centers - t[idx_left])
    right_dist = np.abs(centers - t[idx_right])
    pick_right = right_dist < left_dist
    idx = np.where(pick_right, idx_right, idx_left)
    return y[idx]


def _smooth_short_motif_runs(y_raw: np.ndarray, *, Lmin_frames: int) -> np.ndarray:
    """Smooth raw per-frame motif labels by filling short motif runs (<Lmin) with neighbors.

    Only applies to positive motifs (>0). Background (0) runs are left unchanged.

    Args:
        y_raw: Per-frame motif IDs (pose/frame rate)
        Lmin_frames: Minimum frames to keep a motif run; shorter runs are reassigned

    Returns:
        Smoothed per-frame motif labels
    """
    y = np.asarray(y_raw, int).copy()
    T = y.size
    if T == 0 or Lmin_frames <= 1:
        return y
    i = 0
    # Precompute run starts/ends
    while i < T:
        j = i + 1
        while j < T and y[j] == y[i]:
            j += 1
        run_len = j - i
        label = y[i]
        if label > 0 and run_len < Lmin_frames:
            # Choose neighbor label from left/right (prefer nonzero; prefer longer neighbor run if both >0)
            left_label = y[i - 1] if i > 0 else 0
            right_label = y[j] if j < T else 0
            pick = 0
            if left_label > 0 and right_label > 0 and left_label != right_label:
                # Compare neighbor run lengths
                # left run length
                a = i - 1
                while a >= 0 and y[a] == left_label:
                    a -= 1
                left_len = (i - 1) - a
                # right run length
                b = j
                while b < T and y[b] == right_label:
                    b += 1
                right_len = b - j
                pick = left_label if left_len >= right_len else right_label
            elif left_label > 0:
                pick = left_label
            elif right_label > 0:
                pick = right_label
            # Reassign if we have a nonzero neighbor; else leave as is
            if pick > 0:
                y[i:j] = pick
        i = j
    return y


def _debug_print_ethogram(eth_path: Optional[str], run_dir: Optional[str], frames_map_override: Optional[str]) -> None:
    try:
        print("[motifs] ethogram path:", eth_path)
        print("[motifs] run_dir:", run_dir)
        if eth_path and os.path.isfile(eth_path):
            df = pd.read_parquet(eth_path)
            print("[motifs] columns:", list(df.columns))
            print("[motifs] head:\n", df.head(5))
            if "session" in df.columns:
                print("[motifs] unique sessions (first 10):", df["session"].astype(str).unique()[:10])
            else:
                print("[motifs] 'session' column not present in ethogram parquet")
        else:
            print("[motifs] ethogram parquet missing or not a file.")
        # frames_map / embedding
        eff_dir = os.path.dirname(eth_path) if eth_path else run_dir
        fmap = frames_map_override or (os.path.join(eff_dir, "frames_map.parquet") if eff_dir else None)
        emb = os.path.join(eff_dir, "embedding.parquet") if eff_dir else None
        print("[motifs] frames_map:", fmap, ", exists:", (os.path.isfile(fmap) if fmap else False))
        print("[motifs] embedding:", emb, ", exists:", (os.path.isfile(emb) if emb else False))
        if fmap and os.path.isfile(fmap):
            dfm = pd.read_parquet(fmap)
            print("[frames_map] columns:", list(dfm.columns))
            if "session" in dfm.columns:
                print("[frames_map] sessions (first 10):", dfm["session"].astype(str).unique()[:10])
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as e:
        print("[motifs] debug failed:", repr(e))


def _manual_load_motifs_from_parquet(eth_path: str, session_id: Optional[str], frames_map_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Manual fallback loader to extract (t_sec, motif_id) from ethogram and frames_map/embedding.

    If ethogram has 't_sec', uses it directly; else joins frames_map or embedding to get t_sec.
    """
    if not os.path.isfile(eth_path):
        raise FileNotFoundError(f"Ethogram parquet not found: {eth_path}")
    eth = pd.read_parquet(eth_path)
    # Drop invalid frames if present to mirror loader behavior
    if "valid" in eth.columns:
        eth = eth.loc[eth["valid"] == True].copy()  # noqa: E712
    # Optional session filter; relax if it produces empty and only one session exists
    if "session" in eth.columns and session_id is not None:
        eth_f = eth.loc[eth["session"].astype(str) == str(session_id)].copy()
        if eth_f.empty:
            # If the ethogram contains a single session, use it to avoid dead-end
            uniq = eth["session"].astype(str).unique() if "session" in eth.columns else []
            if isinstance(uniq, np.ndarray) and uniq.size == 1:
                warnings.warn(
                    f"Session '{session_id}' not found in ethogram; using only available session '{uniq[0]}'",
                    RuntimeWarning,
                )
                eth = eth.copy()
            else:
                raise ValueError(
                    "Ethogram has no rows after session filter (manual fallback). "
                    f"Requested session_id='{session_id}', available={list(uniq) if isinstance(uniq, np.ndarray) else 'unknown'}"
                )
        else:
            eth = eth_f
    if eth.empty:
        raise ValueError("Ethogram parquet has no rows to load")
    if ("t_sec" in eth.columns) and ("llac" in eth.columns):
        t = eth["t_sec"].astype(float).to_numpy()
        y = eth["llac"].astype(int).to_numpy()
        return t, y
    # Else join frames_map or embedding
    eff_dir = os.path.dirname(eth_path)
    fmap = frames_map_path if frames_map_path else os.path.join(eff_dir, "frames_map.parquet")
    emb = os.path.join(eff_dir, "embedding.parquet")
    join_src = None
    if os.path.isfile(fmap):
        join_src = pd.read_parquet(fmap)
    elif os.path.isfile(emb):
        join_src = pd.read_parquet(emb)
    else:
        raise ValueError("Neither frames_map.parquet nor embedding.parquet with t_sec found for ethogram")
    keys = [c for c in ["session", "frame_id"] if c in eth.columns and c in join_src.columns]
    if "t_sec" not in join_src.columns:
        raise ValueError("Join source missing 't_sec' column")
    if not keys:
        raise ValueError("No common join keys to map times (need 'frame_id' and optionally 'session')")
    joined = eth.merge(join_src[[*keys, "t_sec"]], on=keys, how="left")
    if "llac" not in joined.columns:
        raise ValueError("Ethogram missing 'llac' column for motif id")
    t = joined["t_sec"].astype(float).to_numpy()
    y = joined["llac"].astype(int).to_numpy()
    return t, y


def _enforce_min_run_length(y: np.ndarray, *, Lmin: int) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce minimum run length; return (y_kept, keep_mask).

    Drops frames in runs shorter than Lmin by marking them invalid in mask.
    """
    y = np.asarray(y, int)
    T = y.size
    keep = np.ones(T, dtype=bool)
    if T == 0 or Lmin <= 1:
        return y.copy(), keep
    i = 0
    while i < T:
        j = i + 1
        while j < T and y[j] == y[i]:
            j += 1
        run_len = j - i
        if run_len < Lmin:
            keep[i:j] = False
        i = j
    return y.copy(), keep


def _derive_onsets(y: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices where label changes (t>0), optionally masked to valid frames."""
    y = np.asarray(y, int)
    if y.size <= 1:
        return np.array([], dtype=int)
    ch = (y[1:] != y[:-1]).astype(bool)
    idx = np.where(ch)[0] + 1
    if mask is not None and mask.size == y.size:
        idx = idx[mask[idx]]
    return idx.astype(int)
