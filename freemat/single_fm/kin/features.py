from __future__ import annotations

"""KIN feature assembly for Single-FM.

Build compact, head-frame kinematic predictors from upright 3D pose and
align them to a user-specified bin grid (or an existing parquet's `t_bin_s`).
This mirrors the VIS pipeline organization without adding project-agnostic IO
or geometry: those come from `freemat_data`.

Example:
    >>> from freemat.single_fm.kin.features import KinParams, build_kin_features
    >>> df, qc = build_kin_features(
    ...     'configs/session_001.yaml',
    ...     params=KinParams(bin_s=0.05, smooth_win_s=1.0)
    ... )

Notes:
    - Defaults implement head-only frame, 1 s smoothing, POS excluded, and
      per-session z-scoring of predictors; lagged variants can be saved by the CLI.
    - If an aligned or VIS parquet is provided via `bins_parquet`, its `t_bin_s`
      column is used for exact time alignment.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd

from freemat_data import (
    load_session_yaml,
    load_pose3d_from_session,
    load_pose3d,
    orient_pose_with_floor,
    pose_time_seconds,
)
from freemat_data.calibration import rotation_from_floor_json
from freemat_data import (
    frame_valid_fraction,
    longest_contiguous_segment,
    compute_head_forward,
)
from freemat.single_fm.pose import find_nonempty_frame_index

# Reuse the tested computation utilities
from freemat.single_fm.viz.kin_features import (
    compute_kin_features,
    KinParams as _ComputeParams,
    make_lagged as _make_lagged,
)


@dataclass
class KinParams:
    """Parameters for KIN feature assembly.

    Args:
        bin_s: Target bin size in seconds when `bins_parquet` is not provided.
        smooth_win_s: Moving-average smoothing on COM/yaw before derivatives.
        zscore_per_session: Apply per-session z-scoring to predictors.
        include_speed: Include `speed` column (POS stays excluded by default).
        include_acceleration: Include `acceleration` column (3D acceleration magnitude).
        include_pos: Include position features (pos_x, pos_y, pos_z, distance_to_center).
        include_gaze: Include gaze direction components (gaze_x, gaze_y, gaze_z).
        start_stop_tsv: Path to TSV with inference start-stop times (e.g., docs/inf_start_stop_times.tsv).
            If provided, filters pose data to the valid time window before feature extraction.
    """

    bin_s: float = 0.05
    smooth_win_s: float = 1.0
    zscore_per_session: bool = True
    include_speed: bool = False
    include_acceleration: bool = False
    include_pos: bool = False
    include_gaze: bool = False
    min_valid_frac: float = 0.9
    conf_thresh: float = 0.0
    debug: bool = True
    ignore_confidence_for_validity: bool = True
    min_forward_norm: float = 0.05
    start_stop_tsv: Optional[str] = None


def _prefer_processed_pose(session) -> Optional[str]:
    """Return processed upright pose path if present for `session_id`.

    Looks for `results/<sid>/pose_processed.npz`.
    """
    sid = getattr(session, "session_id", None)
    if not sid:
        return None
    cand = os.path.join("results", "single_fm", str(sid), "pose_processed.npz")
    return cand if os.path.isfile(cand) else None


def _resolve_floor_json(npz_path: Optional[str]) -> Optional[str]:
    """Locate `floor_plane.json` next to a pose NPZ, if available."""
    if not npz_path:
        return None
    base = os.path.dirname(os.path.abspath(npz_path))
    fp = os.path.join(base, "floor_plane.json")
    return fp if os.path.isfile(fp) else None


def _bin_centers_from_pose(t_pose: np.ndarray, bin_s: float) -> np.ndarray:
    """Construct bin centers covering [t0, t1] using step `bin_s`.

    Centers are midpoints of edges t0..t1 with uniform spacing; last center
    is <= t1 - 0.5*bin_s.
    """
    t0 = float(t_pose[0])
    t1 = float(t_pose[-1])
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("bin_s must be positive and finite")
    edges = np.arange(t0, t1 + 1e-9, float(bin_s))
    if len(edges) < 2:
        return np.array([(t0 + t1) / 2.0], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(float)


def _collect_predictor_columns(df: pd.DataFrame) -> Sequence[str]:
    """Return predictor column list (exclude time, session_id)."""
    return [c for c in df.columns if c not in ("t_bin_s", "session_id")]


def build_kin_features(
    session_yaml: str,
    *,
    params: Optional[KinParams] = None,
    bins_parquet: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Assemble per-bin KIN features (head-frame) for a session.

    Args:
        session_yaml: Path to per-session YAML.
        params: KinParams controlling bin size, smoothing, and z-scoring.
        bins_parquet: Optional existing parquet with a `t_bin_s` column to align to
            (e.g., aligned FR or VIS features). If provided, alignment uses this.

    Returns:
        (df, qc) where df columns include:
            - t_bin_s, session_id,
            - v_parallel, v_perp, v_z (BODY_KIN)
            - yaw_rate, pitch_rate, roll_rate, delta_yaw (HEAD_KIN)
            - gaze_elevation, head_roll (orientation)
            - (optional) speed if enabled.
        qc: dict with metadata (session_id, bin_s, smooth_win_s, n_bins).

    Feature categories (for unique variance analysis):
        BODY_KIN: v_parallel, v_perp, v_z, speed, acceleration
        HEAD_KIN: yaw_rate, pitch_rate, roll_rate, delta_yaw

    Raises:
        FileNotFoundError: if session YAML or required assets are missing.
        RuntimeError: if pose cannot be loaded or oriented.
    """
    if params is None:
        params = KinParams()
    sref = load_session_yaml(session_yaml)
    sid = str(getattr(sref, "session_id", "session"))

    # Resolve pose, preferring processed NPZ if present
    pose_npz_path = _prefer_processed_pose(sref)
    pose0 = load_pose3d(pose_npz_path) if pose_npz_path else load_pose3d_from_session(sref)
    if pose0 is None:
        raise RuntimeError("Pose not resolved from session YAML or processed NPZ not found")
    pose, _ = orient_pose_with_floor(pose0)
    t_pose = pose_time_seconds(pose)
    X = pose.X

    # Apply start-stop time filtering if configured
    start_stop_applied = False
    if params.start_stop_tsv is not None and os.path.isfile(params.start_stop_tsv):
        try:
            from inference_time_filter import InferenceTimeFilter

            time_filter = InferenceTimeFilter.from_tsv(params.start_stop_tsv)
            fps = pose.fps if pose.fps > 0 else 5.0

            # Get frame range for this session
            start_frame, end_frame = time_filter.get_frame_range(sid, fps=fps)

            if start_frame is not None and end_frame is not None:
                # Filter by frame ID
                fids = pose.fids
                mask = (fids >= start_frame) & (fids <= end_frame)

                if mask.any():
                    X = X[mask]
                    t_pose = t_pose[mask]
                    if pose.C is not None:
                        pose = pose._replace(X=X, fids=fids[mask], C=pose.C[mask])
                    else:
                        pose = pose._replace(X=X, fids=fids[mask])
                    start_stop_applied = True

                    if params.debug:
                        print(f"[KIN] Applied start-stop filter: frames {start_frame}-{end_frame}, "
                              f"kept {mask.sum()}/{len(mask)} frames")
                else:
                    if params.debug:
                        print(f"[KIN][warn] Start-stop filter yielded no frames; using full pose")
        except ImportError:
            if params.debug:
                print("[KIN][warn] inference_time_filter not available; skipping start-stop filtering")
        except Exception as e:
            if params.debug:
                print(f"[KIN][warn] Start-stop filtering failed: {e}")
    # Resolve plane normal for yaw reference (used by forward + features)
    floor_json = _resolve_floor_json(pose0.path)
    plane_dict = None
    if floor_json:
        _, plane_upr, _ = rotation_from_floor_json(floor_json)
        plane_dict = plane_upr
    if plane_dict is None:
        # Conservative default: Z-up plane
        plane_dict = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0}

    # Canonicalize joint names to improve head-forward detection
    from freemat.single_fm.viz.kin_features import _canonicalize_joint_names as _canon
    names = _canon(pose.joint_names or [])

    # Validity and forward vector checks for windowing
    try:
        C_used = None if bool(params.ignore_confidence_for_validity) else getattr(pose, "C", None)
        valid_frac = frame_valid_fraction(X, C_used, conf_thresh=float(params.conf_thresh))
    except Exception:
        valid_frac = np.isfinite(X).all(axis=2).mean(axis=1).astype(float)
    try:
        from freemat.single_fm.viz.kin_features import _compute_head_forward_robust as _fwd_robust
        n_hat = np.array([
            float(plane_dict.get('a', 0.0)),
            float(plane_dict.get('b', 0.0)),
            float(plane_dict.get('c', 1.0)),
        ], dtype=float)
        fwd = _fwd_robust(X, names, n_hat)
    except Exception:
        fwd = np.full((X.shape[0], 3), np.nan, dtype=float)
    fwd_ok = np.isfinite(fwd).all(axis=1) & (np.linalg.norm(fwd, axis=1) > 1e-9)
    # Windowing based on pose coverage only; do not require fwd_ok for trimming
    vmask = (valid_frac >= float(params.min_valid_frac))
    # Choose a stable time window: longest contiguous valid run
    if vmask.any():
        try:
            i0_v, i1_v = longest_contiguous_segment(vmask.astype(float), min_valid_frac=0.5)
        except Exception:
            # Fallback: first and last True
            i0_v = int(np.argmax(vmask))
            i1_v = int(len(vmask) - np.argmax(vmask[::-1]))
    else:
        i0_v, i1_v = 0, len(t_pose)
        if params.debug:
            print(f"[KIN][warn] No frames meet min_valid_frac={float(params.min_valid_frac):.2f}; using full pose span for bins")
    t0_valid = float(t_pose[max(i0_v, 0)])
    t1_valid = float(t_pose[max(i1_v - 1, 0)]) if i1_v > 0 else float(t_pose[0])

    # Build target bin centers
    if bins_parquet:
        t_bins = pd.read_parquet(bins_parquet)["t_bin_s"].to_numpy(dtype=float)
        bin_s = float(np.nanmedian(np.diff(t_bins))) if len(t_bins) > 1 else float(params.bin_s)
    else:
        bin_s = float(params.bin_s)
        # Prefer valid window by default to avoid empty stretches
        t0 = t0_valid
        t1 = t1_valid
        edges = np.arange(t0, t1 + 1e-9, bin_s, dtype=float)
        t_bins = (0.5 * (edges[:-1] + edges[1:])) if edges.size >= 2 else np.array([(t0 + t1) / 2.0], dtype=float)

    # Compute features on pose frames then interpolate to t_bins
    compute_params = _ComputeParams(
        bin_s=bin_s,
        smooth_win_s=float(params.smooth_win_s),
        include_speed=bool(params.include_speed),
        include_pos=False,
        include_delta_yaw_rate=False,
        use_Z_in_pos=False,
        min_forward_norm=float(params.min_forward_norm),
    )
    if params.debug:
        frac_overall = float(np.nanmean(valid_frac)) if np.isfinite(valid_frac).any() else 0.0
        print(f"[KIN] session={sid} T={X.shape[0]} J={X.shape[1]} bin_s={bin_s:.3f}")
        print(f"[KIN] valid_frac: mean={frac_overall:.2f} min_valid_frac={float(params.min_valid_frac):.2f} passed={int(vmask.sum())}/{len(vmask)}; min_forward_norm={float(params.min_forward_norm):.3f}")
        try:
            idx0_dbg = int(find_nonempty_frame_index(pose, min_valid_frac=float(params.min_valid_frac)))
            print(f"[KIN] first non-empty frame: idx={idx0_dbg} t={float(t_pose[idx0_dbg]):.3f}s")
        except Exception:
            pass
        fwd_fin = np.isfinite(fwd).all(axis=1)
        fwd_nonzero = (np.linalg.norm(fwd, axis=1) > 1e-9)
        print(f"[KIN] fwd_ok frames: finite={int(fwd_fin.sum())}/{len(fwd_fin)} nonzero={int(fwd_nonzero.sum())}/{len(fwd_nonzero)}")
        print(f"[KIN] time window used: start={t_bins[0]:.3f}s end={t_bins[-1]:.3f}s (pose valid window {t0_valid:.3f}-{t1_valid:.3f}s)")

    frames = compute_kin_features(pose, plane_dict, t_bins, params=compute_params)
    head_df = frames.get("head")
    if head_df is None or head_df.empty:
        raise RuntimeError("Head-frame KIN features were not computed")

    # Select and rename columns for a clean head-only schema
    out = pd.DataFrame({"t_bin_s": head_df["t_bin_s"].to_numpy(dtype=float)})
    out["v_parallel"] = head_df["v_parallel"].to_numpy(dtype=float)
    out["v_perp"] = head_df["v_perp"].to_numpy(dtype=float)
    out["v_z"] = head_df["v_z"].to_numpy(dtype=float)
    # prefer head yaw rate, and delta_yaw (head - torso)
    if "yaw_rate_head" in head_df.columns:
        out["yaw_rate"] = head_df["yaw_rate_head"].to_numpy(dtype=float)
    elif "yaw_rate" in head_df.columns:
        out["yaw_rate"] = head_df["yaw_rate"].to_numpy(dtype=float)
    else:
        out["yaw_rate"] = np.nan
    if "delta_yaw" in head_df.columns:
        out["delta_yaw"] = head_df["delta_yaw"].to_numpy(dtype=float)
    else:
        out["delta_yaw"] = np.nan
    # pitch_rate and roll_rate (HEAD_KIN features)
    if "pitch_rate" in head_df.columns:
        out["pitch_rate"] = head_df["pitch_rate"].to_numpy(dtype=float)
    else:
        out["pitch_rate"] = np.nan
    if "roll_rate" in head_df.columns:
        out["roll_rate"] = head_df["roll_rate"].to_numpy(dtype=float)
    else:
        out["roll_rate"] = np.nan
    # Static orientation features
    if "gaze_elevation" in head_df.columns:
        out["gaze_elevation"] = head_df["gaze_elevation"].to_numpy(dtype=float)
    if "head_roll" in head_df.columns:
        out["head_roll"] = head_df["head_roll"].to_numpy(dtype=float)
    if params.include_speed and "speed" in head_df.columns:
        out["speed"] = head_df["speed"].to_numpy(dtype=float)

    # Acceleration (3D magnitude)
    if params.include_acceleration:
        from freemat_data import compute_center_of_mass
        com = compute_center_of_mass(X)
        fps = pose.fps if pose.fps > 0 else 5.0
        accel = compute_acceleration(com, fps, smooth_sigma=float(params.smooth_win_s))
        # Interpolate to bin centers
        accel_interp = np.interp(out["t_bin_s"].to_numpy(), t_pose, accel)
        out["acceleration"] = accel_interp

    # Position features (allocentric)
    if params.include_pos:
        from freemat_data import compute_center_of_mass
        from freemat.single_fm.kin.spatial import compute_distance_to_center

        com = compute_center_of_mass(X)
        t_out = out["t_bin_s"].to_numpy()

        # Interpolate COM to bin centers
        com_x_interp = np.interp(t_out, t_pose, com[:, 0])
        com_y_interp = np.interp(t_out, t_pose, com[:, 1])
        com_z_interp = np.interp(t_out, t_pose, com[:, 2])
        out["pos_x"] = com_x_interp
        out["pos_y"] = com_y_interp
        out["pos_z"] = com_z_interp

        # Distance to arena center
        d_center = compute_distance_to_center(com)
        out["distance_to_center"] = np.interp(t_out, t_pose, d_center)

    # Gaze direction components (head-forward unit vector)
    if params.include_gaze:
        from freemat.single_fm.viz.kin_features import _compute_head_forward_robust as _fwd_robust
        n_hat = np.array([
            float(plane_dict.get('a', 0.0)),
            float(plane_dict.get('b', 0.0)),
            float(plane_dict.get('c', 1.0)),
        ], dtype=float)
        head_fwd = _fwd_robust(X, names, n_hat)
        # Normalize to unit vectors
        norms = np.linalg.norm(head_fwd, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        head_fwd_unit = head_fwd / norms

        t_out = out["t_bin_s"].to_numpy()
        out["gaze_x"] = np.interp(t_out, t_pose, head_fwd_unit[:, 0])
        out["gaze_y"] = np.interp(t_out, t_pose, head_fwd_unit[:, 1])
        out["gaze_z"] = np.interp(t_out, t_pose, head_fwd_unit[:, 2])

    out.insert(1, "session_id", sid)

    # Optional per-session z-scoring across predictor columns
    if params.zscore_per_session:
        for c in _collect_predictor_columns(out):
            m = float(np.nanmean(out[c])) if np.isfinite(out[c]).any() else 0.0
            s = float(np.nanstd(out[c])) if np.isfinite(out[c]).any() else 1.0
            s = s if s > 1e-12 else 1.0
            out[c] = (out[c] - m) / s

    if params.debug:
        cov = {c: float(np.isfinite(out[c]).mean()) for c in _collect_predictor_columns(out)}
        print(f"[KIN] coverage: " + ", ".join([f"{k}={v:.2f}" for k, v in cov.items()]))
        try:
            fb = float(head_df.attrs.get("com_heading_fallback_frac", np.nan))
            if np.isfinite(fb):
                print(f"[KIN] forward fallback fraction (COM-heading): {fb:.2f}")
        except Exception:
            pass
        for c in _collect_predictor_columns(out):
            arr = out[c].to_numpy(dtype=float)
            fin = arr[np.isfinite(arr)]
            if fin.size:
                mu, sd = float(np.mean(fin)), float(np.std(fin))
                if sd < 1e-6:
                    print(f"[KIN][warn] column '{c}' is near-constant: mean={mu:.3e}, sd={sd:.3e}")
            else:
                print(f"[KIN][warn] column '{c}' is all-NaN in the chosen window")

    qc: Dict[str, object] = {
        "session_id": sid,
        "bin_s": bin_s,
        "smooth_win_s": float(params.smooth_win_s),
        "min_forward_norm": float(params.min_forward_norm),
        "n_bins": int(len(out)),
        "pose_npz": pose_npz_path or pose0.path or "",
        "aligned_to": (os.path.abspath(bins_parquet) if bins_parquet else None),
        "min_valid_frac": float(params.min_valid_frac),
        "conf_thresh": float(params.conf_thresh),
        "t_window": [float(t_bins[0]), float(t_bins[-1])],
        "pose_valid_window": [t0_valid, t1_valid],
        "start_stop_tsv": params.start_stop_tsv,
        "start_stop_applied": start_stop_applied,
    }
    # Attach fallback and yaw coverage if available
    try:
        fb = head_df.attrs.get("com_heading_fallback_frac", None)
        if fb is not None:
            qc["com_heading_fallback_frac"] = float(fb)
    except Exception:
        pass
    # Add simple coverage stats
    cov = {f"cov_{c}": float(np.isfinite(out[c]).mean()) for c in _collect_predictor_columns(out)}
    qc.update(cov)
    return out, qc


def make_lagged_long(df: pd.DataFrame, *, lags_ms: Sequence[int], bin_s: float) -> pd.DataFrame:
    """Create a long-form lagged KIN feature table.

    Returns a DataFrame with columns: `t_bin_s`, `session_id`, `lag_ms`, and
    the predictor columns shifted by each lag. Edge rows are dropped per lag.
    """
    base_cols = _collect_predictor_columns(df)
    lagged = _make_lagged(df[["t_bin_s", *base_cols]], lags_ms, bin_s, drop_edge=True)
    rows = []
    sid = str(df["session_id"].iloc[0]) if "session_id" in df.columns and len(df) > 0 else ""
    for L, d in lagged.items():
        tmp = d.copy()
        tmp.insert(1, "session_id", sid)
        tmp.insert(2, "lag_ms", int(L))
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["t_bin_s", "session_id", "lag_ms", *base_cols])
    return pd.concat(rows, axis=0, ignore_index=True)


# =============================================================================
# Helper Functions for compute_kin_full
# =============================================================================

def compute_egocentric_velocity(
    com: np.ndarray,
    head_forward: np.ndarray,
    fps: float,
    speed_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute egocentric velocity components (v_parallel, v_perp, v_z).

    Args:
        com: Center of mass positions (T, 3) in meters.
        head_forward: Head-forward unit vectors (T, 3).
        fps: Frame rate in Hz.
        speed_threshold: Speed below which v_parallel/v_perp are NaN.

    Returns:
        Tuple of (v_parallel, v_perp, v_z), each (T,) array.
    """
    from freemat_data.kinematics import central_diff

    com = np.asarray(com, dtype=np.float64)
    head_forward = np.asarray(head_forward, dtype=np.float64)
    T = len(com)

    dt = 1.0 / fps
    velocity = central_diff(com, dt)  # (T, 3)

    # Planar velocity (XY only)
    v_xy = velocity[:, :2]
    speed_xy = np.linalg.norm(v_xy, axis=1)

    # Head-forward in XY plane (normalized)
    hf_xy = head_forward[:, :2]
    hf_norm = np.linalg.norm(hf_xy, axis=1, keepdims=True)
    hf_xy_unit = np.where(hf_norm > 1e-9, hf_xy / hf_norm, np.nan)

    # Head-left (perpendicular to head-forward in XY, leftward positive)
    hl_xy = np.column_stack([-hf_xy_unit[:, 1], hf_xy_unit[:, 0]])

    # Project velocity onto head frame
    v_parallel = np.sum(v_xy * hf_xy_unit, axis=1)
    v_perp = np.sum(v_xy * hl_xy, axis=1)
    v_z = velocity[:, 2]

    # Apply speed threshold
    low_speed = speed_xy < speed_threshold
    v_parallel[low_speed] = np.nan
    v_perp[low_speed] = np.nan

    return v_parallel, v_perp, v_z


def compute_head_torso_delta(
    head_yaw: np.ndarray,
    torso_yaw: np.ndarray,
) -> np.ndarray:
    """Compute head-torso yaw difference, wrapped to [-π, π].

    Args:
        head_yaw: Head yaw angles (T,) in radians.
        torso_yaw: Torso yaw angles (T,) in radians.

    Returns:
        Yaw difference (T,) in radians, in [-π, π].
    """
    delta = head_yaw - torso_yaw

    # Wrap to [-π, π]
    delta = np.mod(delta + np.pi, 2 * np.pi) - np.pi

    return delta


def compute_gaze_elevation(
    head_forward_3d: np.ndarray,
) -> np.ndarray:
    """Compute gaze elevation angle (floor-relative pitch).

    Args:
        head_forward_3d: 3D head-forward unit vectors (T, 3).

    Returns:
        Elevation angles (T,) in radians, in [-π/2, π/2].
    """
    head_forward_3d = np.asarray(head_forward_3d, dtype=np.float64)

    # Project to floor plane (XY)
    hf_xy = head_forward_3d[:, :2]
    hf_xy_norm = np.linalg.norm(hf_xy, axis=1)

    # Elevation = angle between forward and its floor projection
    # = atan2(z, xy_norm)
    elevation = np.arctan2(head_forward_3d[:, 2], hf_xy_norm)

    return elevation


def compute_acceleration(
    com: np.ndarray,
    fps: float,
    smooth_sigma: float = 0.2,
) -> np.ndarray:
    """Compute 3D acceleration magnitude with optional smoothing.

    Args:
        com: Center of mass positions (T, 3) in meters.
        fps: Frame rate in Hz.
        smooth_sigma: Gaussian smoothing sigma in seconds (0 for no smoothing).

    Returns:
        Acceleration magnitude (T,) in m/s².
    """
    from freemat_data.kinematics import central_diff

    com = np.asarray(com, dtype=np.float64)
    dt = 1.0 / fps

    # Apply smoothing if requested
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma_frames = smooth_sigma * fps
        com_smooth = np.column_stack([
            gaussian_filter1d(com[:, i], sigma_frames, mode='nearest')
            for i in range(3)
        ])
    else:
        com_smooth = com

    # Compute velocity and acceleration
    velocity = central_diff(com_smooth, dt)
    acceleration = central_diff(velocity, dt)

    # Return magnitude
    return np.linalg.norm(acceleration, axis=1)


def apply_lag(
    features: np.ndarray,
    t_bins: np.ndarray,
    lag_ms: int,
    bin_s: float,
) -> np.ndarray:
    """Apply temporal lag to features.

    Positive lag means: align neural[t] with features[t - lag].
    (Neural activity at time t relates to behavior from lag_ms ago.)

    Args:
        features: Feature array (T,) or (T, D).
        t_bins: Time bins (T,) in seconds.
        lag_ms: Lag in milliseconds (positive = shift features back).
        bin_s: Bin size in seconds.

    Returns:
        Lagged features, same shape as input.
    """
    features = np.asarray(features, dtype=np.float64)
    lag_s = lag_ms / 1000.0
    lag_bins = int(round(lag_s / bin_s))

    if features.ndim == 1:
        result = np.full_like(features, np.nan)
        if lag_bins >= 0:
            # Shift backward: features[i] = features[i - lag_bins]
            if lag_bins < len(features):
                result[lag_bins:] = features[:-lag_bins] if lag_bins > 0 else features
        else:
            # Shift forward: features[i] = features[i + |lag_bins|]
            shift = -lag_bins
            if shift < len(features):
                result[:-shift] = features[shift:]
    else:
        result = np.full_like(features, np.nan)
        if lag_bins >= 0:
            if lag_bins < len(features):
                result[lag_bins:] = features[:-lag_bins] if lag_bins > 0 else features
        else:
            shift = -lag_bins
            if shift < len(features):
                result[:-shift] = features[shift:]

    return result


def zscore_features(
    features: np.ndarray,
) -> np.ndarray:
    """Z-score features to mean=0, std=1.

    Args:
        features: Feature array (T,) or (T, D).

    Returns:
        Z-scored features, same shape.
    """
    features = np.asarray(features, dtype=np.float64)

    if features.ndim == 1:
        valid = np.isfinite(features)
        if valid.any():
            mean = np.mean(features[valid])
            std = np.std(features[valid])
            std = std if std > 1e-12 else 1.0
            result = (features - mean) / std
        else:
            result = features.copy()
    else:
        result = np.empty_like(features)
        for d in range(features.shape[1]):
            result[:, d] = zscore_features(features[:, d])

    return result


def compute_torso_yaw(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]] = None,
    floor_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Compute torso yaw angle from hip→shoulder direction.

    Args:
        X: Pose array (T, J, 3).
        joint_names: Optional joint names.
        floor_normal: Floor plane normal.

    Returns:
        Torso yaw angles (T,) in radians.
    """
    from freemat.single_fm.kin.head_forward import _get_joint_index

    X = np.asarray(X, dtype=np.float64)
    T, J, _ = X.shape

    idx_lhip = _get_joint_index('left_hip', joint_names, J)
    idx_rhip = _get_joint_index('right_hip', joint_names, J)
    idx_lsho = _get_joint_index('left_shoulder', joint_names, J)
    idx_rsho = _get_joint_index('right_shoulder', joint_names, J)

    if None in (idx_lhip, idx_rhip, idx_lsho, idx_rsho):
        return np.full(T, np.nan, dtype=np.float64)

    hip_mid = 0.5 * (X[:, idx_lhip, :] + X[:, idx_rhip, :])
    sho_mid = 0.5 * (X[:, idx_lsho, :] + X[:, idx_rsho, :])

    # Torso forward = hip → shoulder
    torso_fwd = sho_mid - hip_mid

    # Project to floor plane
    floor_normal = np.asarray(floor_normal, dtype=np.float64)
    floor_normal = floor_normal / (np.linalg.norm(floor_normal) + 1e-12)
    dot = (torso_fwd @ floor_normal.reshape(3, 1)).reshape(-1, 1)
    torso_fwd_proj = torso_fwd - dot * floor_normal.reshape(1, 3)

    # Yaw = atan2(y, x)
    return np.arctan2(torso_fwd_proj[:, 1], torso_fwd_proj[:, 0])


__all__ = [
    "KinParams",
    "build_kin_features",
    "make_lagged_long",
    "compute_egocentric_velocity",
    "compute_head_torso_delta",
    "compute_gaze_elevation",
    "compute_acceleration",
    "apply_lag",
    "zscore_features",
    "compute_torso_yaw",
]
