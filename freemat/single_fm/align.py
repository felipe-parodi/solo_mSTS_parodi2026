"""Synchronization and alignment helpers for Single-FM analyses.

Wrap freemat_data sync utilities and provide small helpers to compute
time windows and binned outputs without duplicating logic in notebooks.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from freemat_data import (
    SessionRef,
    SpikesEvents,
    get_syncmap_from_session,
    infer_logger,
    map_spikes_to_video,
)
from freemat_data import (
    Pose3D,
    pose_time_seconds,
    video_time_to_pose_axis,
    times_to_frame_index,
    compute_center_of_mass,
    compute_speed_from_com,
)
from .spikes import bin_to_tidy
from .io.save import save_aligned_parquet


def to_video_time(sess: SessionRef, spikes: SpikesEvents) -> np.ndarray:
    """Map neural event times to absolute video seconds using session sync.

    Args:
        sess: Session reference
        spikes: Event-coded spikes in neural time

    Returns:
        Video-time event array (E,)
    """

    sync = get_syncmap_from_session(sess)
    logger = (infer_logger(sess) or "m1").strip().lower()
    has_m1 = bool(getattr(sync, "has_m1", False))
    has_m2 = bool(getattr(sync, "has_m2", False))

    if logger == "m1" and (not has_m1) and has_m2:
        logger = "m2"
    elif logger == "m2" and (not has_m2) and has_m1:
        logger = "m1"
    elif logger not in ("m1", "m2"):
        if has_m1:
            logger = "m1"
        elif has_m2:
            logger = "m2"
        else:
            raise ValueError("to_video_time: sync_map.json missing both m1 and m2 mappings")

    return map_spikes_to_video(spikes, sync, logger=logger, session=sess)


def pick_window(
    t_spk_vid: Optional[np.ndarray] = None,
    *,
    t_pose: Optional[np.ndarray] = None,
    bins: Sequence[float] = (1.0, 0.5),
) -> Tuple[float, float]:
    """Choose a conservative time window for binning/alignment.

    Args:
        t_spk_vid: Spike event times in video seconds
        t_pose: Optional pose sample times (seconds)
        bins: Bin sizes to ensure the window spans at least one bin

    Returns:
        (t_start, t_end) time bounds
    """

    candidates = []
    if t_spk_vid is not None and t_spk_vid.size:
        candidates.append((float(np.nanmin(t_spk_vid)), float(np.nanmax(t_spk_vid))))
    if t_pose is not None and t_pose.size:
        candidates.append((float(np.nanmin(t_pose)), float(np.nanmax(t_pose))))

    if not candidates:
        # Default to [0, max_bin]
        t0, t1 = 0.0, max(bins)
    else:
        t0 = min(a for a, _ in candidates)
        t1 = max(b for _, b in candidates)
        if t1 <= t0:
            t1 = t0 + max(bins)
    return (t0, t1)


def bin_aligned(
    sess: SessionRef,
    spikes: SpikesEvents,
    *,
    session_id: str,
    bins: Sequence[float] = (1.0, 0.5),
    t_window: Optional[Tuple[float, float]] = None,
    keep_units: Optional[np.ndarray] = None,
) -> Dict[float, pd.DataFrame]:
    """Map to video time and bin spikes at requested resolutions.

    Args:
        sess: Session reference
        spikes: Event-coded spikes
        session_id: Session ID to annotate outputs
        bins: Bin widths (seconds)
        t_window: Optional (t_start, t_end) bounds; computed if None
        keep_units: Optional per-unit inclusion mask applied to events

    Returns:
        Mapping bin_s -> tidy dataframe
    """

    t_evt = to_video_time(sess, spikes)
    u_evt = spikes.unit_index.astype(int)
    if keep_units is not None:
        m = keep_units[u_evt]
        t_evt = t_evt[m]
        u_evt = u_evt[m]
    if t_window is None:
        t_window = pick_window(t_evt, bins=bins)

    out: Dict[float, pd.DataFrame] = {}
    for b in bins:
        df = bin_to_tidy(t_evt, u_evt, session_id=session_id, bin_s=float(b), t_start=t_window[0], t_end=t_window[1])
        out[float(b)] = df
    return out


__all__ = [
    "to_video_time",
    "pick_window",
    "bin_aligned",
    "write_aligned_bins",
]


def merge_bins_with_pose(
    df_aligned: pd.DataFrame,
    pose: Pose3D,
    *,
    add_speed: bool = True,
) -> pd.DataFrame:
    """Aggregate per-bin population FR and add pose features by nearest sampling.

    This function expects a tidy aligned spikes dataframe (per-unit rows) with
    at least columns: ``['bin_idx','t','unit_index','fr']`` and optional
    ``'bin_s'`` and ``'session_id'``. It computes population mean firing rate
    per bin and joins pose-derived features sampled at the nearest pose frame to
    each bin center time.

    Args:
        df_aligned: Tidy spikes dataframe (possibly multiple bins); one row per unit per bin.
        pose: Processed or upright pose (world coordinates consistent across time).
        add_speed: Whether to include CoM speed (m/s) alongside CoM coordinates.

    Returns:
        DataFrame with one row per bin including columns:
          - 'bin_idx', 't', optional 'bin_s'/'session_id'
          - 'fr_pop_mean', 'fr_pop_z'
          - 'com_x','com_y','com_z' and optional 'speed'

    Raises:
        ValueError: If required columns are missing or inputs are empty.
    """

    req = {"bin_idx", "t", "unit_index", "fr"}
    if df_aligned is None or df_aligned.empty or not req.issubset(df_aligned.columns):
        raise ValueError("merge_bins_with_pose: aligned dataframe missing required columns or is empty")

    # Group by bin (and bin_s/session) to aggregate population FR
    by = [c for c in ["session_id", "bin_s", "bin_idx"] if c in df_aligned.columns]
    if "bin_idx" not in by:
        by.append("bin_idx")
    g = df_aligned.groupby(by, sort=True, as_index=False)
    agg = g.agg({"t": "mean", "fr": "mean"}).rename(columns={"fr": "fr_pop_mean"})

    # Pose-derived features sampled at nearest pose frame to each bin center
    t_bins = agg["t"].to_numpy(dtype=float)
    # Map absolute video seconds to pose-centered axis
    t_pose_q = video_time_to_pose_axis(t_bins, pose)
    idx = times_to_frame_index(t_pose_q, pose)
    com = compute_center_of_mass(pose.X)
    com_s = np.asarray(com, dtype=float)
    com_s = com_s[np.clip(idx, 0, pose.T - 1)] if com_s.shape[0] else np.zeros((t_bins.size, 3), dtype=float)
    agg["com_x"] = com_s[:, 0]
    agg["com_y"] = com_s[:, 1]
    agg["com_z"] = com_s[:, 2]
    if add_speed:
        speed = compute_speed_from_com(com, float(pose.fps)) if pose.fps else np.zeros((pose.T,), dtype=float)
        agg["speed"] = speed[np.clip(idx, 0, pose.T - 1)] if speed.size else np.zeros((t_bins.size,), dtype=float)

    # Z-score of population FR over time (per session/bin_s group if present)
    if "bin_s" in agg.columns:
        agg["fr_pop_z"] = agg.groupby([c for c in ["session_id", "bin_s"] if c in agg.columns])["fr_pop_mean"].transform(
            lambda x: (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + 1e-12)
        )
    else:
        m = float(np.nanmean(agg["fr_pop_mean"]))
        s = float(np.nanstd(agg["fr_pop_mean"])) + 1e-12
        agg["fr_pop_z"] = (agg["fr_pop_mean"] - m) / s

    return agg


def _fmt_bin_suffix(bin_s: float) -> str:
    """Format bin size (seconds) into filename-friendly suffix.

    Examples:
        1.0   -> '1p0'
        0.5   -> '0p5'
        0.25  -> '0p25'
        0.05  -> '0p05'
        0.025 -> '0p025'
    """
    # Use up to 3 decimals, strip trailing zeros, replace '.' with 'p'
    s = f"{bin_s:.3f}".rstrip('0').rstrip('.')
    return s.replace('.', 'p')


def write_aligned_bins(
    df_by_bin: Dict[float, pd.DataFrame],
    out_dir: str,
    *,
    session_id: str,
) -> Dict[float, str]:
    """Persist multiple aligned dataframes (different bin sizes) to parquet.

    Args:
        df_by_bin: Mapping from bin width (s) to tidy aligned dataframe
        out_dir: Output directory for aligned parquets
        session_id: Session identifier for filenames and metadata

    Returns:
        Mapping from bin width to written parquet path

    Raises:
        ValueError: If any dataframe is empty
        OSError: If writing fails
    """
    paths: Dict[float, str] = {}
    for b, df in df_by_bin.items():
        if df is None or df.empty:
            continue  # skip empties silently; caller can decide
        suf = _fmt_bin_suffix(float(b))
        fname = f"session_{session_id}_bin{suf}.parquet"
        path = os.path.join(out_dir, fname)
        save_aligned_parquet(df, path, metadata={"session_id": session_id, "bin_s": float(b)})
        paths[float(b)] = path
    return paths
