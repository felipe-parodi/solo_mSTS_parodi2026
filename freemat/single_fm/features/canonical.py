# freemat/single_fm/features/canonical.py
"""Canonical feature computation: single source of truth.

This module provides the authoritative feature computation used by both
animation/visualization (QA) and encoding pipelines.

Ground truth sources:
- BODY_KIN: matplotlib animation (feature_animation.py) - uses body-heading
- VIS: viser viewer (gaze_renderer.py) - uses head_pos + gaze_dir
- HEAD_KIN: no animation equivalent (encoding-only)

BODY_KIN Features (body-heading based, matches animation):
    v_parallel: dot(vel_horiz, body_heading) - forward/backward velocity
    v_perp: dot(vel_horiz, perp(body_heading)) - lateral velocity (left +)
    speed: norm(vel) - 3D speed magnitude

    body_heading = (nose - mid_hips), projected to horizontal, normalized.
    This allows v_parallel to be negative when walking backward.

    NOTE: v_z removed for floor-focused analysis (Fig 2B).

VIS Features (matches viser visualization):
    gaze_x, gaze_y, gaze_z: head forward direction (3D)
    d_wall_horiz: horizontal wall distance from HEAD along GAZE direction
    vis_motion: v_parallel_gaze / (d_wall_horiz + eps) - looming proxy

    NOTE: d_wall_horiz uses HEAD position + GAZE direction (not COM + body_heading).
    vis_motion uses velocity projected onto gaze (different from BODY_KIN v_parallel).

HEAD_KIN Features (encoding-only, no animation equivalent):
    yaw_rate: d/dt(head_yaw)
    pitch_rate: d/dt(gaze_elevation)
    roll_rate: d/dt(head_roll)
    delta_yaw: head_yaw - torso_yaw
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from freemat.single_fm.data.arena import ArenaGeometry

from freemat.single_fm.features.utils import (
    _central_diff,
    _get_horiz_indices,
    _get_vertical_axis,
    _nan_moving_average,
    _viser_compute_gaze_direction,
    _viser_compute_head_position,
)


# =============================================================================
# Floor Filtering (matches animation)
# =============================================================================

def compute_floor_mask(
    X: np.ndarray,
    vertical_idx: int,
    threshold: float = 0.5,
    floor_level: Optional[float] = None,
) -> np.ndarray:
    """Compute per-frame on-floor mask using ankles.

    Uses ankle heights relative to the calibrated floor level from floor_plane.json.
    floor_level is REQUIRED - the analysis should not run without proper calibration.

    COCO-17 ankle indices: 15 (left_ankle), 16 (right_ankle)

    Args:
        X: Pose array (T, J, 3) with J >= 17 (COCO format).
        vertical_idx: Vertical axis index (0=X, 1=Y, 2=Z).
        threshold: Max height above floor to be on_floor (default 0.5m).
        floor_level: Calibrated floor height from floor_plane.json (REQUIRED).

    Returns:
        Boolean mask (T,) where True = on floor.

    Raises:
        ValueError: If floor_level is None (floor_plane.json required).
    """
    if floor_level is None:
        raise ValueError(
            "floor_level is required - must be computed from floor_plane.json. "
            "Ensure floor_plane.json exists next to pose3d_smoothed.npz."
        )

    T = X.shape[0]

    if X.shape[1] < 17:
        # Fallback to COM-based if not COCO-17
        com_vert = np.nanmean(X, axis=1)[:, vertical_idx]
        valid = np.isfinite(com_vert)
        if not valid.any():
            return np.ones(T, dtype=bool)  # Can't determine, assume on floor
        return (com_vert - floor_level) < threshold

    # COCO ankle indices = 15, 16
    ankles = X[:, [15, 16], vertical_idx]  # (T, 2)

    valid = np.isfinite(ankles)
    if not valid.any():
        return np.ones(T, dtype=bool)  # Can't determine, assume on floor

    # Per-frame: min ankle height above floor
    # Use nanmin to handle NaN values (if one ankle is missing, use the other)
    min_ankle_height = np.nanmin(ankles, axis=1) - floor_level

    # Handle frames where both ankles are NaN
    both_nan = ~np.any(np.isfinite(ankles), axis=1)
    on_floor = min_ankle_height < threshold
    on_floor[both_nan] = True  # Can't determine, assume on floor

    return on_floor


# =============================================================================
# Body Heading Computation (Animation Ground Truth)
# =============================================================================

def compute_body_heading(X: np.ndarray, vertical_idx: int = 2) -> np.ndarray:
    """Compute body-forward heading from pose (nose-to-hips direction).

    Uses COCO-17 keypoint indices:
    - 0: nose
    - 11, 12: left_hip, right_hip

    This gives the body's facing direction independent of movement direction,
    allowing v_parallel to be negative when walking backward.

    Args:
        X: Pose array (T, J, 3) with J >= 13 for COCO-17.
        vertical_idx: Vertical axis index (0=X, 1=Y, 2=Z).

    Returns:
        Body heading vectors (T, 2) in horizontal plane, normalized.
    """
    T = X.shape[0]
    horiz_idxs = _get_horiz_indices(vertical_idx)

    if X.shape[1] < 13:
        # Not enough keypoints, return default forward
        return np.tile(np.array([1.0, 0.0]), (T, 1))

    nose = X[:, 0, :]  # (T, 3)
    left_hip = X[:, 11, :]
    right_hip = X[:, 12, :]
    mid_hips = (left_hip + right_hip) / 2

    # Body heading = nose - hips (projected to horizontal plane)
    body_dir = np.zeros((T, 2), dtype=float)
    body_dir[:, 0] = nose[:, horiz_idxs[0]] - mid_hips[:, horiz_idxs[0]]
    body_dir[:, 1] = nose[:, horiz_idxs[1]] - mid_hips[:, horiz_idxs[1]]

    # Normalize
    body_norm = np.linalg.norm(body_dir, axis=1, keepdims=True)
    body_norm = np.where(body_norm > 0.01, body_norm, 1.0)
    body_heading = body_dir / body_norm

    # Handle invalid frames
    invalid = ~np.all(np.isfinite(X[:, [0, 11, 12], :]), axis=(1, 2))
    body_heading[invalid] = np.array([1.0, 0.0])

    return body_heading


# =============================================================================
# BODY_KIN Features (Animation Ground Truth)
# =============================================================================

def compute_body_kin_features(
    X: np.ndarray,
    fps: float,
    plane_dict: Dict,
    smooth_win_s: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Canonical BODY_KIN features using body heading.

    Uses the same computation as feature_animation.py (animation ground truth).
    Body heading = nose → mid-hips, projected to horizontal plane.

    NOTE: v_z removed for floor-focused analysis (Fig 2B).

    Args:
        X: Pose array (T, J, 3).
        fps: Pose frame rate.
        plane_dict: Floor plane dict with keys a, b, c, d.
        smooth_win_s: Smoothing window in seconds (0 = no smoothing for animation).

    Returns:
        Dict with per-frame arrays:
            v_parallel, v_perp, speed, body_heading (T, 2)
    """
    from freemat_data import compute_center_of_mass

    # Get axes
    vertical_idx = _get_vertical_axis(plane_dict)
    horiz_idxs = _get_horiz_indices(vertical_idx)

    # Compute COM
    com = compute_center_of_mass(X)  # (T, 3)

    # Optional smoothing
    if smooth_win_s > 0:
        com = _nan_moving_average(com, smooth_win_s, fps)

    # Compute velocity via central difference
    dt = 1.0 / fps
    velocity = _central_diff(com, dt)  # (T, 3)

    # Body heading (from pose, NOT velocity)
    body_heading = compute_body_heading(X, vertical_idx)  # (T, 2)

    # Horizontal velocity
    vel_horiz = np.zeros((X.shape[0], 2), dtype=float)
    vel_horiz[:, 0] = velocity[:, horiz_idxs[0]]
    vel_horiz[:, 1] = velocity[:, horiz_idxs[1]]

    # v_parallel: velocity along body-forward axis (can be NEGATIVE = backward!)
    v_parallel = np.sum(vel_horiz * body_heading, axis=1)

    # v_perp: leftward velocity (90 deg CCW from body heading)
    perp_unit = np.zeros_like(body_heading)
    perp_unit[:, 0] = -body_heading[:, 1]
    perp_unit[:, 1] = body_heading[:, 0]
    v_perp = np.sum(vel_horiz * perp_unit, axis=1)

    # Speed: 3D magnitude
    speed = np.linalg.norm(velocity, axis=1)

    # NOTE: v_z removed for floor-focused analysis (Fig 2B)
    return {
        "v_parallel": v_parallel,
        "v_perp": v_perp,
        "speed": speed,
        "body_heading": body_heading,
    }


# =============================================================================
# VIS Features
# =============================================================================

def compute_head_forward(X: np.ndarray, plane_dict: Dict) -> np.ndarray:
    """Compute 3D head forward direction.

    Uses freemat_data.compute_head_forward with fallback.

    Args:
        X: Pose array (T, J, 3).
        plane_dict: Floor plane dict.

    Returns:
        Head forward vectors (T, 3), normalized.
    """
    from freemat_data import compute_head_forward as _compute_head_forward

    # Get plane normal
    n_hat = np.array([
        float(plane_dict.get('a', 0.0)),
        float(plane_dict.get('b', 0.0)),
        float(plane_dict.get('c', 1.0)),
    ], dtype=float)
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)

    # Try freemat_data computation
    from freemat_data.skeletons import COCO17_NAMES
    head_fwd = _compute_head_forward(X, list(COCO17_NAMES))

    # Normalize
    norms = np.linalg.norm(head_fwd, axis=1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    head_fwd_unit = head_fwd / norms

    return head_fwd_unit


def compute_vis_features(
    X: np.ndarray,
    fps: float,
    plane_dict: Dict,
    arena: "ArenaGeometry",
) -> Dict[str, np.ndarray]:
    """Canonical VIS features matching viser visualization.

    Uses HEAD position + GAZE direction for d_wall_horiz (matches viser viewer).
    This is different from BODY_KIN which uses body_heading.

    NOTE: vis_motion uses velocity projected onto GAZE direction, not body_heading.
    This represents looming in the visual field (what the monkey sees approaching).

    IMPORTANT: Uses viser's exact compute_head_position() and compute_gaze_direction()
    functions to ensure alignment with visualization:
    - head_pos: nose if available, else ear average, else first valid joint
    - gaze_dir: weighted blend (35% eyes + 65% ears) -> nose direction

    Args:
        X: Pose array (T, J, 3).
        fps: Pose frame rate.
        plane_dict: Floor plane dict.
        arena: ArenaGeometry for hex walls (required).

    Returns:
        Dict with per-frame arrays:
            gaze_x, gaze_y, gaze_z, d_wall_horiz, vis_motion

    Raises:
        ValueError: If arena is None.
    """
    if arena is None:
        raise ValueError(
            "ArenaGeometry is required for VIS feature computation. "
            "Circular fallback has been removed — provide a valid ArenaGeometry."
        )
    from freemat_data import compute_center_of_mass
    from freemat_data.skeletons import COCO17_NAMES

    vertical_idx = _get_vertical_axis(plane_dict)
    horiz_idxs = _get_horiz_indices(vertical_idx)
    T = X.shape[0]
    joint_names = list(COCO17_NAMES)

    # Use inlined viser functions for head position and gaze direction
    # (copied from gaze_renderer.py to avoid import chain issues)
    head_pos = np.array([_viser_compute_head_position(X[i], joint_names) for i in range(T)])  # (T, 3)
    gaze_dir = np.array([_viser_compute_gaze_direction(X[i], joint_names) for i in range(T)])  # (T, 3)

    # Project gaze to horizontal plane for d_wall_horiz
    gaze_horiz = np.zeros((T, 3), dtype=float)
    gaze_horiz[:, horiz_idxs[0]] = gaze_dir[:, horiz_idxs[0]]
    gaze_horiz[:, horiz_idxs[1]] = gaze_dir[:, horiz_idxs[1]]
    gaze_horiz_norm = np.linalg.norm(gaze_horiz, axis=1, keepdims=True)
    gaze_horiz_norm = np.where(gaze_horiz_norm > 1e-9, gaze_horiz_norm, 1.0)
    gaze_horiz_unit = gaze_horiz / gaze_horiz_norm

    # Compute d_wall_horiz: distance from HEAD along GAZE direction (hex arena)
    from freemat.single_fm.data.arena import compute_d_wall_horiz_batch
    d_wall_horiz = compute_d_wall_horiz_batch(head_pos, gaze_horiz_unit, arena)

    # Handle non-finite values with arena-derived max distance
    max_dist = arena.hexagon_radius * 2.0
    d_wall_horiz = np.where(np.isfinite(d_wall_horiz), d_wall_horiz, max_dist)

    # Compute velocity for vis_motion
    # NOTE: vis_motion uses velocity projected onto GAZE, not body_heading
    # This represents looming in the visual field
    com = compute_center_of_mass(X)
    dt = 1.0 / fps
    velocity = _central_diff(com, dt)  # (T, 3)

    # v_parallel_gaze: velocity projected onto gaze direction
    # Different from BODY_KIN v_parallel which uses body_heading!
    v_parallel_gaze = np.sum(velocity * gaze_dir, axis=1)

    # vis_motion = v_parallel_gaze / (d_wall_horiz + eps)
    eps = 0.1
    vis_motion = v_parallel_gaze / (d_wall_horiz + eps)

    return {
        "gaze_x": gaze_dir[:, 0],
        "gaze_y": gaze_dir[:, 1],
        "gaze_z": gaze_dir[:, 2],
        "d_wall_horiz": d_wall_horiz,
        "vis_motion": vis_motion,
    }


# =============================================================================
# HEAD_KIN Features (Encoding-only)
# =============================================================================

def compute_head_kin_features(
    X: np.ndarray,
    fps: float,
    plane_dict: Dict,
    smooth_win_s: float = 0.1,
) -> Dict[str, np.ndarray]:
    """HEAD_KIN features (encoding-only, no animation equivalent).

    These features describe head orientation changes and are useful for
    encoding but have no visual animation equivalent for QA.

    Args:
        X: Pose array (T, J, 3).
        fps: Pose frame rate.
        plane_dict: Floor plane dict.
        smooth_win_s: Smoothing window for rate computation.

    Returns:
        Dict with per-frame arrays:
            yaw_rate, pitch_rate, roll_rate, delta_yaw
    """
    # Import from existing kin_features module (HEAD_KIN computation is unchanged)
    from freemat.single_fm.viz.kin_features import (
        compute_kin_features,
        KinParams,
    )

    # Create a minimal pose object with required attributes
    class MinimalPose:
        def __init__(self, X, fps, joint_names=None):
            self.X = X
            self.fps = fps
            from freemat_data.skeletons import COCO17_NAMES
            self.joint_names = joint_names or list(COCO17_NAMES)
            # fids needed for pose_time_seconds fallback
            self.fids = np.arange(X.shape[0])
            self.path = None

    pose = MinimalPose(X, fps)

    # Compute using existing module (HEAD_KIN is computed correctly there)
    t_pose = np.arange(X.shape[0], dtype=float) / fps
    params = KinParams(
        bin_s=1.0 / fps,  # Per-frame
        smooth_win_s=smooth_win_s,
        include_speed=False,
        include_pos=False,
    )

    # Pass t_pose_video explicitly to avoid pose_time_seconds() fallback
    frames = compute_kin_features(pose, plane_dict, t_pose, params=params, t_pose_video=t_pose)
    head_df = frames.get("head")

    if head_df is None or head_df.empty:
        T = X.shape[0]
        return {
            "yaw_rate": np.full(T, np.nan),
            "pitch_rate": np.full(T, np.nan),
            "roll_rate": np.full(T, np.nan),
            "delta_yaw": np.full(T, np.nan),
        }

    # Extract HEAD_KIN features (these are correct in kin_features.py)
    return {
        "yaw_rate": head_df["yaw_rate_head"].to_numpy(dtype=float),
        "pitch_rate": head_df["pitch_rate"].to_numpy(dtype=float),
        "roll_rate": head_df["roll_rate"].to_numpy(dtype=float),
        "delta_yaw": head_df["delta_yaw"].to_numpy(dtype=float),
    }


# =============================================================================
# Binning for Encoding
# =============================================================================

def bin_features(
    features: Dict[str, np.ndarray],
    t_pose: np.ndarray,
    t_bins: np.ndarray,
    method: str = 'interp',
) -> pd.DataFrame:
    """Bin per-frame features to encoding time bins.

    Args:
        features: Dict of feature arrays, each (T,) or (T, D).
        t_pose: Pose time axis (T,) in same units as t_bins.
        t_bins: Target bin centers (N,).
        method: 'interp' for linear interpolation (default), 'mean' for bin averaging.

    Returns:
        DataFrame with columns t_bin_s + feature names, shape (N, ...).
    """
    result = {"t_bin_s": t_bins.astype(float)}

    for name, arr in features.items():
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            # Scalar feature
            if method == 'interp':
                valid = np.isfinite(arr)
                if valid.sum() >= 2:
                    result[name] = np.interp(t_bins, t_pose[valid], arr[valid])
                else:
                    result[name] = np.full(len(t_bins), np.nan)
            else:
                # Mean binning
                binned = np.zeros(len(t_bins), dtype=float)
                for i, t in enumerate(t_bins):
                    # Find frames in bin
                    dt = np.median(np.diff(t_bins)) if len(t_bins) > 1 else 0.5
                    mask = (t_pose >= t - dt/2) & (t_pose < t + dt/2)
                    if mask.any():
                        binned[i] = np.nanmean(arr[mask])
                    else:
                        binned[i] = np.nan
                result[name] = binned

        elif arr.ndim == 2:
            # Vector feature (e.g., body_heading)
            for d in range(arr.shape[1]):
                col_name = f"{name}_{d}"
                col_arr = arr[:, d]
                if method == 'interp':
                    valid = np.isfinite(col_arr)
                    if valid.sum() >= 2:
                        result[col_name] = np.interp(t_bins, t_pose[valid], col_arr[valid])
                    else:
                        result[col_name] = np.full(len(t_bins), np.nan)
                else:
                    binned = np.zeros(len(t_bins), dtype=float)
                    for i, t in enumerate(t_bins):
                        dt = np.median(np.diff(t_bins)) if len(t_bins) > 1 else 0.5
                        mask = (t_pose >= t - dt/2) & (t_pose < t + dt/2)
                        if mask.any():
                            binned[i] = np.nanmean(col_arr[mask])
                        else:
                            binned[i] = np.nan
                    result[col_name] = binned

    return pd.DataFrame(result)


# =============================================================================
# Convenience Function for Full Pipeline
# =============================================================================

def compute_all_features(
    X: np.ndarray,
    fps: float,
    plane_dict: Dict,
    t_bins: np.ndarray,
    t_pose: np.ndarray,
    arena: "ArenaGeometry",
    smooth_win_s: float = 0.0,
) -> pd.DataFrame:
    """Compute all features and bin to encoding time.

    Convenience function that computes BODY_KIN, VIS, and HEAD_KIN features
    and bins them to the specified time axis.

    Args:
        X: Pose array (T, J, 3).
        fps: Pose frame rate.
        plane_dict: Floor plane dict.
        t_bins: Target bin centers for encoding.
        t_pose: Pose time axis (same units as t_bins).
        arena: ArenaGeometry for hex walls (required).
        smooth_win_s: Smoothing for BODY_KIN (0 for animation-compatible).

    Returns:
        DataFrame with all features binned to t_bins.
    """
    # Compute BODY_KIN
    body_kin = compute_body_kin_features(X, fps, plane_dict, smooth_win_s)

    # Compute VIS
    vis = compute_vis_features(X, fps, plane_dict, arena)

    # Compute HEAD_KIN
    head_kin = compute_head_kin_features(X, fps, plane_dict, smooth_win_s=0.1)

    # Merge all features
    # NOTE: v_z removed for floor-focused analysis (Fig 2B)
    # NOTE: d_obstacle removed (redundant with d_wall_horiz)
    all_features = {}
    for name in ["v_parallel", "v_perp", "speed"]:
        all_features[name] = body_kin[name]
    for name in ["gaze_x", "gaze_y", "gaze_z", "d_wall_horiz", "vis_motion"]:
        all_features[name] = vis[name]
    for name in ["yaw_rate", "pitch_rate", "roll_rate", "delta_yaw"]:
        all_features[name] = head_kin[name]

    # Bin to encoding time
    return bin_features(all_features, t_pose, t_bins, method='interp')
