"""Data loading, normalization, and QC filtering for Step-A encoding.

Functions extracted from the monolithic ``encoding_stepA.py`` for testability.
Handles neural parquet normalization, pose geometry derivation from NPZ,
predictor coverage checks, and unit variance filtering.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import pandas as pd

from freemat_data import (
    compute_head_forward,
    load_pose3d,
    orient_pose_with_floor,
    times_to_frame_index,
    video_time_to_pose_axis,
)
from freemat_data.banks import load_banks_config, map_channels_to_bank
from freemat_data.calibration import rotation_from_floor_json
from freemat.single_fm.viz.kin_features import (
    _canonicalize_joint_names as _canon_names,
    _compute_head_forward_robust as _fwd_robust,
)

PRED_COLS: Sequence[str] = (
    "speed",
    "v_parallel",
    "v_perp",
    "v_z",
    "theta_look",
    "yaw_rate",
    "align_cos",
    "sin1",
    "cos1",
    "sin2",
    "cos2",
    "sin3",
    "cos3",
)


def _sanity_predictor_coverage(df: pd.DataFrame) -> None:
    """Log warnings for predictors with <90% finite coverage.

    Args:
        df: DataFrame containing predictor columns from ``PRED_COLS``.
    """
    frac = {c: float(np.isfinite(df[c]).mean()) for c in PRED_COLS if c in df.columns}
    bad = {k: v for k, v in frac.items() if v < 0.9}
    if bad:
        logging.warning("Predictor finite coverage <0.9 for: %s", bad)
    # Per-session coverage summary
    try:
        if "session_id" in df.columns:
            for sid, g in df.groupby("session_id", sort=False):
                frac_s = {
                    c: float(np.isfinite(g[c]).mean())
                    for c in PRED_COLS
                    if c in g.columns
                }
                logging.info(
                    "[coverage] session=%s %s",
                    str(sid),
                    {k: round(v, 3) for k, v in frac_s.items()},
                )
    except (KeyError, TypeError):
        pass


def _variance_ok_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Drop units with near-zero firing-rate variance.

    Args:
        df: DataFrame with ``unit_id`` and ``fr`` columns.

    Returns:
        Filtered DataFrame retaining only units with variance > 1e-8.
    """
    keep_units: list[int] = []
    for uid, g in df.groupby("unit_id", sort=False):
        v = float(np.nanvar(g["fr"].to_numpy(float)))
        if v > 1e-8:
            keep_units.append(uid)
    if len(keep_units) != df["unit_id"].nunique():
        logging.info(
            "Skipping %d units with near-zero FR variance",
            df["unit_id"].nunique() - len(keep_units),
        )
    return df[df["unit_id"].isin(keep_units)].copy()


def _pose_df_from_npz_for_session(
    pose_npz: str, neural_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive required pose geometry per bin from an upright NPZ and neural bins.

    Assumes ``neural_df`` contains a single session. Uses its ``t_bin_s`` to
    sample pose at nearest frames. If ``floor_plane.json`` is present next to
    the NPZ, uses its plane normal; otherwise defaults to Z-up.

    Args:
        pose_npz: Path to upright/leveled pose NPZ.
        neural_df: Neural table for a single session.

    Returns:
        DataFrame with pose geometry columns aligned to neural bins.

    Raises:
        ValueError: If neural_df is empty or contains multiple sessions.
    """
    if neural_df is None or neural_df.empty:
        raise ValueError("Neural dataframe is empty; cannot derive pose bins")
    sessions = pd.unique(neural_df["session_id"].astype(str))
    if sessions.size != 1:
        raise ValueError(
            f"--pose-npz path is single-session but neural contains {sessions.size} sessions; "
            "run per-session or provide a multi-session pose parquet"
        )
    sid = str(sessions[0])
    # Time bins for this session (unique centers)
    t_bins = (
        neural_df.loc[neural_df["session_id"].astype(str) == sid, "t_bin_s"]
        .dropna()
        .unique()
    )
    t_bins = np.sort(t_bins.astype(float))
    if t_bins.size == 0:
        raise ValueError("No t_bin_s values found in neural for the session")

    pose0 = load_pose3d(pose_npz)
    pose, _ = orient_pose_with_floor(pose0)

    # Map video-time bins to pose index
    t_pose_q = video_time_to_pose_axis(t_bins, pose)
    idx = times_to_frame_index(t_pose_q, pose)
    idx = np.clip(idx, 0, pose.T - 1)

    # NaN-aware CoM: per-frame mean over available joints
    X = np.asarray(pose.X, dtype=float)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError("pose.X must have shape (T, J, 3)")
    fin = np.isfinite(X).all(axis=2)  # (T, J)
    fin_all = fin.all(axis=1)  # (T,)
    cnt = fin.sum(axis=1)  # (T,)
    sum_xyz = np.where(fin[:, :, None], X, 0.0).sum(axis=1)  # (T, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        com_all = sum_xyz / np.where(cnt[:, None] > 0, cnt[:, None], np.nan)
    com_s = np.asarray(com_all, dtype=float)[idx]

    # Robust head-forward using project-specific helper; fallback to generic
    try:
        n_hat = None
        try:
            floor_json = os.path.join(
                os.path.dirname(os.path.abspath(pose_npz)), "floor_plane.json"
            )
            if os.path.isfile(floor_json):
                _, plane_upr, _ = rotation_from_floor_json(floor_json)
                n_hat = np.array(
                    [float(plane_upr["a"]), float(plane_upr["b"]), float(plane_upr["c"])],
                    dtype=float,
                )
        except (FileNotFoundError, KeyError, ValueError):
            n_hat = None
        names = _canon_names(pose.joint_names or [])
        if n_hat is None:
            n_hat = np.array([0.0, 0.0, 1.0], dtype=float)
        fwd = _fwd_robust(pose.X, names, n_hat)
    except (KeyError, ValueError, IndexError):
        fwd = compute_head_forward(pose.X, pose.joint_names)
    fwd_s = np.asarray(fwd, dtype=float)[idx]
    # Pose full-valid mask sampled to bins
    pose_full_valid = np.asarray(fin_all, dtype=bool)[idx]

    # Floor normal from floor_plane.json if present
    try:
        floor_json = os.path.join(
            os.path.dirname(os.path.abspath(pose_npz)), "floor_plane.json"
        )
        if os.path.isfile(floor_json):
            _, plane_upr, _ = rotation_from_floor_json(floor_json)
            n_vec = np.array(
                [float(plane_upr["a"]), float(plane_upr["b"]), float(plane_upr["c"])],
                dtype=float,
            )
        else:
            n_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    except (FileNotFoundError, KeyError, ValueError):
        n_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    N = np.tile(n_vec[None, :], (t_bins.size, 1))

    df_pose = pd.DataFrame(
        {
            "session_id": sid,
            "t_bin_s": t_bins,
            "P_x": com_s[:, 0],
            "P_y": com_s[:, 1],
            "P_z": com_s[:, 2],
            "fhead_x": fwd_s[:, 0],
            "fhead_y": fwd_s[:, 1],
            "fhead_z": fwd_s[:, 2],
            "n_x": N[:, 0],
            "n_y": N[:, 1],
            "n_z": N[:, 2],
        }
    )
    df_pose["pose_full_valid"] = pose_full_valid
    return df_pose


def _normalize_neural_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize neural dataframe columns expected by Step-A.

    Renames ``t`` -> ``t_bin_s``, ``unit_index`` -> ``unit_id`` if needed.
    Fills ``bank`` from channel mapping or defaults to ``'unknown'``.
    Ensures ``fr`` column exists via fallbacks (``zfr``, ``rate``).

    Args:
        df: Raw neural DataFrame.

    Returns:
        Normalized copy of the DataFrame.
    """
    df = df.copy()
    if "t_bin_s" not in df.columns:
        if "t" in df.columns:
            df = df.rename(columns={"t": "t_bin_s"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "t_bin_s"})
    if "unit_id" not in df.columns and "unit_index" in df.columns:
        df = df.rename(columns={"unit_index": "unit_id"})
    # Ensure firing rate column
    if "fr" not in df.columns:
        for alt in ("zfr", "rate", "fr_pop_mean"):
            if alt in df.columns:
                df = df.rename(columns={alt: "fr"})
                break
    # Bank mapping
    if "bank" not in df.columns:
        if "channel" in df.columns:
            try:
                banks_cfg = load_banks_config()
                df["bank"] = map_channels_to_bank(df["channel"].to_numpy(), banks_cfg)
            except (FileNotFoundError, KeyError, ValueError):
                df["bank"] = "unknown"
        else:
            df["bank"] = "unknown"
    return df


def _apply_qc_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows using a boolean keep column if present.

    Detects one of ``['qc_keep_analysis', 'keep_final', 'keep_conservative',
    'keep']`` and retains only True rows.

    Args:
        df: DataFrame potentially containing a QC keep column.

    Returns:
        Filtered DataFrame (or original if no QC column found).
    """
    for c in ("qc_keep_analysis", "keep_final", "keep_conservative", "keep"):
        if c in df.columns and df[c].dtype != object:
            m = df[c].astype(bool)
            return df.loc[m].copy()
    return df


def _longest_run_mask(df_sess: pd.DataFrame) -> pd.Series:
    """Find the longest contiguous True run in ``pose_full_valid``.

    Args:
        df_sess: DataFrame for a single session, must contain
            ``t_bin_s`` and ``pose_full_valid`` columns.

    Returns:
        Boolean Series aligned to ``df_sess.index`` marking the longest run.
    """
    g = df_sess.sort_values("t_bin_s", kind="mergesort")
    v = g["pose_full_valid"].to_numpy(dtype=bool)
    if not v.any():
        return pd.Series(False, index=g.index)
    best_len = 0
    best_start = 0
    i = 0
    n = len(v)
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n and v[j]:
            j += 1
        if (j - i) > best_len:
            best_len = j - i
            best_start = i
        i = j
    m = np.zeros(n, dtype=bool)
    if best_len > 0:
        m[best_start : best_start + best_len] = True
    return pd.Series(m, index=g.index)
