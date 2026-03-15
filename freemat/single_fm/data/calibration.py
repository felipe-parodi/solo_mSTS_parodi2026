"""Calibration loading for single-FM sessions (standalone paper repo).

Loads pre-computed calibration data (R_total, floor_level, arena geometry)
from pre-computed calibration JSON files.

No external triangulation dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from freemat.single_fm.data.arena import ArenaGeometry


def load_calibration_json(calib_json_path: Path, debug: bool = False) -> Dict:
    """Load pre-computed calibration from JSON.

    Args:
        calib_json_path: Path to calibration_<session_id>.json.
        debug: Print debug info.

    Returns:
        Dict with keys: R_total, t_total, floor_level, up_axis, plane_dict, arena.
    """
    with open(calib_json_path) as f:
        data = json.load(f)

    result = {
        "session_id": data["session_id"],
        "video_fps": float(data["video_fps"]),
        "R_total": np.array(data["R_total"]),
        "t_total": np.array(data["t_total"]) if data["t_total"] is not None else None,
        "floor_level": float(data["floor_level"]),
        "up_axis": data["up_axis"],
        "plane_dict": data["plane_dict"],
    }

    if debug:
        print(
            f"[Loader] Loaded calibration JSON: session={result['session_id']}, "
            f"floor={result['floor_level']:.4f}m, up_axis={result['up_axis']}"
        )

    return result


def load_arena_from_json(calib_json_path: Path, debug: bool = False) -> "ArenaGeometry":
    """Load ArenaGeometry from pre-computed calibration JSON.

    Args:
        calib_json_path: Path to calibration_<session_id>.json.
        debug: Print debug info.

    Returns:
        ArenaGeometry instance.
    """
    from freemat.single_fm.data.arena import ArenaGeometry

    arena = ArenaGeometry.from_json(calib_json_path)

    if debug:
        print(
            f"[Loader] ArenaGeometry loaded: {len(arena.walls)} walls, "
            f"radius={arena.hexagon_radius:.2f}m, up_axis={arena.up_axis}, "
            f"floor={arena.floor_height:.2f}m"
        )

    return arena


def apply_rotation_to_pose(
    pose_X: np.ndarray,
    R_total: np.ndarray,
    t_total: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply pre-computed rotation (and optional translation) to pose.

    Args:
        pose_X: Raw pose coordinates (T, J, 3).
        R_total: (3, 3) rotation matrix.
        t_total: Optional (3,) translation vector.

    Returns:
        Rotated pose coordinates (T, J, 3).
    """
    T, J, _ = pose_X.shape
    X_flat = pose_X.reshape(-1, 3)
    X_rotated = (R_total @ X_flat.T).T
    if t_total is not None:
        X_rotated = X_rotated + t_total
    return X_rotated.reshape(T, J, 3)


def compute_horizontal_alignment(
    ref_points: np.ndarray,
    target_points: np.ndarray,
    up_axis: int = 1,
) -> tuple:
    """Compute yaw-only alignment in the horizontal plane + vertical offset.

    Projects camera positions to the horizontal plane (excluding the up
    axis), runs 2D SVD-based Procrustes (rotation + translation), and
    computes vertical offset from mean camera height.

    Args:
        ref_points: (N, 3) reference camera positions.
        target_points: (N, 3) target camera positions (same ordering).
        up_axis: Index of the vertical axis (0=X, 1=Y, 2=Z).

    Returns:
        Tuple of (R_3d, t_3d, rmse).

    Raises:
        ValueError: If point arrays have mismatched shapes.
    """
    if ref_points.shape != target_points.shape:
        raise ValueError(
            f"Shape mismatch: ref={ref_points.shape}, target={target_points.shape}"
        )

    h_indices = [i for i in range(3) if i != up_axis]

    ref_2d = ref_points[:, h_indices]
    tgt_2d = target_points[:, h_indices]

    ref_mean_2d = ref_2d.mean(axis=0)
    tgt_mean_2d = tgt_2d.mean(axis=0)
    ref_c = ref_2d - ref_mean_2d
    tgt_c = tgt_2d - tgt_mean_2d

    H = tgt_c.T @ ref_c
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_2d = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T

    t_2d = ref_mean_2d - R_2d @ tgt_mean_2d

    t_v = float(
        ref_points[:, up_axis].mean() - target_points[:, up_axis].mean()
    )

    R_3d = np.eye(3)
    R_3d[np.ix_(h_indices, h_indices)] = R_2d

    t_3d = np.zeros(3)
    t_3d[h_indices] = t_2d
    t_3d[up_axis] = t_v

    aligned = (R_3d @ target_points.T).T + t_3d
    rmse = float(np.sqrt(np.mean(np.sum((aligned - ref_points) ** 2, axis=1))))

    return R_3d, t_3d, rmse


__all__ = [
    "load_calibration_json",
    "load_arena_from_json",
    "apply_rotation_to_pose",
    "compute_horizontal_alignment",
]
