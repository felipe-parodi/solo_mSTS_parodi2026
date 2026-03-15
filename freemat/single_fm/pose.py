"""Pose utilities for Single-FM analyses.

Thin wrappers composing project-agnostic ``freemat_data`` pose loading,
upright orientation, and a standard processing pipeline with conservative
defaults. Functions here are module-level (not notebook-specific) and
fail loudly when required assets (e.g., floor plane, hex footprint) are
missing.

Single-FM coordinate convention:
- Enforce Z-up in all downstream visualizations and analyses, even when
  source NPZs are delivered in a Y-up upright frame. Helpers below
  standardize to Z-up (pose + floor plane) and keep notebooks minimal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from dataclasses import dataclass

from freemat_data import (
    SessionRef,
    Pose3D,
    PosePipelineParams,
    run_pose_pipeline,
    orient_with_floor,
    orient_pose_upright,
    load_pose3d_from_session,
    pose_time_seconds,
)
from freemat_data import load_hex_uv
from freemat_data import coco17_bone_edges, coco17_indices
from freemat_data.calibration import rotation_from_floor_json
from freemat_data.viz.mpl import (
    draw_skeleton_mpl,
    draw_joint_height_lines_mpl,
    plot_hex_outline_mpl,
    set_axes_equal_3d,
)
from freemat_data.vis import (
    plane_basis_from_dict,
    project_points_to_uv,
    uv_to_xyz,
    foot_of_perpendicular_to_plane,
)
import os
import numpy as np


@dataclass
class PoseProcessed:
    """Processed pose artifact and QC metrics.

    Attributes:
        pose: Processed upright pose (trimmed/interpolated/smoothed)
        qc: Dictionary of QC arrays/metrics from the processing pipeline
    """

    pose: Optional[Pose3D]
    qc: Dict[str, Any]


def _rot_x(theta: float) -> np.ndarray:
    """Rotation about X-axis by angle theta (radians)."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _align_normal_to_z(n: np.ndarray) -> np.ndarray:
    """Minimal rotation that aligns a 3D vector to +Z.

    Args:
        n: 3-vector normal.

    Returns:
        Rotation matrix R (3,3) such that R @ n_hat ≈ [0,0,1].
    """
    n = np.asarray(n, dtype=float).reshape(3)
    n = n / (float(np.linalg.norm(n)) + 1e-12)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    c = float(n @ z)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        # 180-degree flip about X (arbitrary axis orthogonal to Z)
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(n, z)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
    s = float(np.linalg.norm(np.cross(n, z)))
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=float,
    )
    return np.eye(3) + K + K @ K * ((1 - c) / (s**2 + 1e-12))


def load_pose_zup_assets(
    sess: SessionRef,
) -> Tuple[Pose3D, Dict[str, float], Optional[np.ndarray], np.ndarray]:
    """Load upright pose and return Z-up pose, plane, hex, and time axis.

    Steps:
        1) Load pose from session YAML.
        2) Apply upright orientation from floor_plane.json when available.
        3) Convert to Z-up (swap Y/Z when up_axis=='Y') and minimally align
           plane normal to +Z.
        4) Load hex UV footprint if present.

    Args:
        sess: Session reference for this recording.

    Returns:
        (pose_z, plane_z, hex_uv, t_pose) where pose_z is Z-up.

    Raises:
        FileNotFoundError: if pose or floor_plane.json is missing.
    """
    pose_raw = load_pose3d_from_session(sess, animal=0)
    if pose_raw is None:
        raise FileNotFoundError("load_pose_zup_assets: pose_3d not resolved from session")
    # Prefer floor JSON next to the NPZ
    npz_dir = os.path.dirname(os.path.abspath(pose_raw.path)) if pose_raw.path else ""
    floor_path = os.path.join(npz_dir, "floor_plane.json")
    if not os.path.isfile(floor_path):
        raise FileNotFoundError(
            "load_pose_zup_assets: floor_plane.json not found next to pose NPZ"
        )
    # Upright orientation and plane meta
    # Prefer upright plane from NPZ meta (already leveled/upright by viewer)
    plane_meta = None
    up_axis_meta = None
    if isinstance(pose_raw.meta, dict):
        plane_meta = pose_raw.meta.get("upright_plane")
        if isinstance(plane_meta, dict):
            up_axis_meta = plane_meta.get("up_axis") or pose_raw.meta.get("up_axis")

    if isinstance(plane_meta, dict) and all(k in plane_meta for k in ("a","b","c","d")):
        # Pose is already upright (from NPZ); do NOT apply orient_with_floor again
        pose_base = pose_raw
        plane_upr = {k: float(plane_meta[k]) for k in ("a","b","c","d")}
        up_axis = str(up_axis_meta or pose_base.up_axis or "Y").upper()
    else:
        # Fall back to floor_plane.json
        try:
            pose_base, _ = orient_with_floor(pose_raw)
        except Exception:
            pose_base = pose_raw
        _, plane_upr, up_axis = rotation_from_floor_json(floor_path)
        if plane_upr is None:
            raise FileNotFoundError("load_pose_zup_assets: plane missing in floor_plane.json and NPZ meta")

    # Convert to Z-up via proper rotation (handedness preserved)
    ua = (up_axis or pose_base.up_axis or "Y").strip().upper()
    R_y2z = _rot_x(np.pi / 2.0) if ua == "Y" else np.eye(3, dtype=float)
    # Apply rotation to pose
    Xr = (R_y2z @ pose_base.X.reshape(-1, 3).T).T.reshape(pose_base.X.shape)
    pose_z = Pose3D(
        X=Xr.astype(pose_base.X.dtype, copy=False),
        C=pose_base.C,
        fids=pose_base.fids,
        fps=pose_base.fps,
        joint_names=pose_base.joint_names,
        up_axis="Z",
        meta=pose_base.meta,
        path=pose_base.path,
    )
    # Rotate plane normal; d unchanged under pure rotation about origin
    n0 = np.array([
        float(plane_upr.get("a", 0.0)),
        float(plane_upr.get("b", 0.0)),
        float(plane_upr.get("c", 1.0)),
    ], dtype=float)
    n1 = R_y2z @ n0
    # If plane came from NPZ meta (assumed already leveled), skip further leveling to preserve yaw
    use_rfix = not (isinstance(plane_meta, dict) and all(k in plane_meta for k in ("a","b","c","d")))
    if use_rfix:
        R_fix = _align_normal_to_z(n1)
        if not np.allclose(R_fix, np.eye(3), atol=1e-12):
            Xr2 = (R_fix @ pose_z.X.reshape(-1, 3).T).T.reshape(pose_z.X.shape)
            pose_z = Pose3D(
                X=Xr2.astype(pose_z.X.dtype, copy=False),
                C=pose_z.C,
                fids=pose_z.fids,
                fps=pose_z.fps,
                joint_names=pose_z.joint_names,
                up_axis="Z",
                meta=pose_z.meta,
                path=pose_z.path,
            )
            n1 = R_fix @ n1
    # Normalize plane normal and enforce positive Z component
    nn = float(np.linalg.norm(n1))
    if nn > 1e-12:
        n1 = n1 / nn
    if n1[2] < 0:
        n1 = -n1
        d_sign = -1.0
    else:
        d_sign = 1.0
    plane_z = {"a": float(n1[0]), "b": float(n1[1]), "c": float(n1[2]), "d": float(plane_upr.get("d", 0.0)) * d_sign}
    # Hex UV footprint (optional but recommended)
    try:
        hex_uv = load_hex_uv(npz_dir)
    except Exception:
        hex_uv = None
    # Ensure joint_names use canonical COCO-17 tokens when short-named inputs are present
    pose_z = _rename_joints_coco17_if_needed(pose_z)
    t_pose = pose_time_seconds(pose_z)
    return pose_z, plane_z, hex_uv, t_pose


def find_nonempty_frame_index(pose: Pose3D, *, min_valid_frac: float = 0.5) -> int:
    """Pick a frame index with sufficient valid joints.

    Args:
        pose: Pose3D positions.
        min_valid_frac: Minimum fraction of joints with finite XYZ.

    Returns:
        Frame index with the highest valid fraction; falls back to 0.
    """
    X = np.asarray(pose.X, dtype=float)
    if X.size == 0:
        return 0
    valid = np.isfinite(X).all(axis=2).mean(axis=1)
    if valid.size == 0:
        return 0
    # Prefer frames meeting the threshold; otherwise best available
    idx = np.where(valid >= float(min_valid_frac))[0]
    if idx.size:
        # pick the median of the qualifying frames for visual stability
        return int(idx[len(idx) // 2])
    return int(np.nanargmax(valid))


def plot_skeleton_frame_with_floor(
    ax,
    pose: Pose3D,
    frame_i: int,
    *,
    plane: Dict[str, float],
    hex_uv: Optional[np.ndarray] = None,
    min_conf: float = 0.0,
) -> None:
    """Plot a single colored, connected 3D skeleton frame with the floor.

    Args:
        ax: Matplotlib 3D Axes (projection='3d').
        pose: Z-up Pose3D.
        frame_i: Frame index to render.
        plane: Upright floor plane coefficients.
        hex_uv: Optional hex footprint in plane UV coordinates.
        min_conf: Minimum per-joint confidence for drawing.
    """
    J = pose.X.shape[1]
    conf = None if pose.C is None else (
        pose.C[frame_i].reshape(J) if pose.C.ndim == 2 else None
    )
    draw_skeleton_mpl(
        ax,
        pose.X[frame_i],
        joint_names=pose.joint_names,
        confidences=conf,
        min_conf=min_conf,
        linewidth=1.3,
        markersize=6.0,
        with_labels=False,
    )
    draw_joint_height_lines_mpl(ax, pose.X[frame_i], plane=plane, color="#777", alpha=0.6, linewidth=0.9)
    if hex_uv is not None:
        plot_hex_outline_mpl(ax, plane, hex_uv)
    else:
        # Fallback: draw a light rectangular outline on the plane near the skeleton footprint
        P = pose.X[frame_i]
        if np.isfinite(P).any():
            F = foot_of_perpendicular_to_plane(P[np.isfinite(P).all(axis=1)], plane)
            if F.size > 0:
                ctr = np.nanmean(F, axis=0)
                # Size from pose extent
                ext = np.nanmax(np.linalg.norm(P - ctr.reshape(1, 3), axis=1))
                size = float(ext) if np.isfinite(ext) and ext > 0 else 0.5
                # Build rectangle in UV and map to XYZ
                Uc, Vc = project_points_to_uv(ctr.reshape(1, 3), plane)
                Uc, Vc = float(Uc[0]), float(Vc[0])
                half = size
                rect_uv = np.array([
                    [Uc - half, Vc - half],
                    [Uc + half, Vc - half],
                    [Uc + half, Vc + half],
                    [Uc - half, Vc + half],
                    [Uc - half, Vc - half],
                ], dtype=float)
                rect_xyz = uv_to_xyz(rect_uv[:, 0], rect_uv[:, 1], plane)
                ax.plot(rect_xyz[:, 0], rect_xyz[:, 1], rect_xyz[:, 2], color="#BBBBBB", linewidth=0.8, alpha=0.9)
    set_axes_equal_3d(ax, pose.X[frame_i])
    # Clean axes: no grid, faint panes, no ticks
    try:
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            try:
                pane.set_edgecolor('w'); pane.set_alpha(0.0)
            except Exception:
                pass
    except Exception:
        pass


def _rename_joints_coco17_if_needed(pose: Pose3D) -> Pose3D:
    """Rename short-form joints to canonical COCO-17 tokens if necessary.

    This local mapping keeps the project-independent data layer untouched while
    allowing single_fm visuals to use consistent COCO-17 naming for styling.
    No indices change — only labels.
    """
    names = list(pose.joint_names or [])
    if not names:
        return pose
    # Short → canonical map
    alias = {
        'l_shoulder': 'left_shoulder', 'r_shoulder': 'right_shoulder',
        'l_elbow': 'left_elbow', 'r_elbow': 'right_elbow',
        'l_wrist': 'left_wrist', 'r_wrist': 'right_wrist',
        'l_hip': 'left_hip', 'r_hip': 'right_hip',
        'l_knee': 'left_knee', 'r_knee': 'right_knee',
        'l_ankle': 'left_ankle', 'r_ankle': 'right_ankle',
        'lefteye': 'left_eye', 'righteye': 'right_eye',
    }
    changed = False
    out = []
    for n in names:
        s = str(n or '').strip()
        k = s.lower()
        if k in alias:
            out.append(alias[k])
            changed = True
        else:
            out.append(k)
            if k != s:
                changed = True
    if not changed:
        return pose
    return Pose3D(
        X=pose.X, C=pose.C, fids=pose.fids, fps=pose.fps,
        joint_names=out, up_axis=pose.up_axis, meta=pose.meta, path=pose.path,
    )


def load_and_process(
    sess: SessionRef,
    *,
    params: Optional[PosePipelineParams] = None,
) -> PoseProcessed:
    """Load 3D pose, orient upright, and run the processing pipeline.

    This function:
    1) Loads pose via session YAML
    2) Orients pose upright using ``floor_plane.json`` next to the NPZ when present
       (falls back to a Y/Z swap if floor metadata is missing)
    3) Runs the standard processing pipeline (centering, trimming, gap fill, smooth,
       derivatives, CoM/speed, outlier flags) using the upright floor plane and
       hex footprint (both required for centering and heights)

    Args:
        sess: Session reference
        params: Optional pipeline parameter overrides

    Returns:
        PoseProcessed containing the processed pose and QC dict

    Raises:
        FileNotFoundError: If required assets (pose NPZ, floor/hex files) are missing
        RuntimeError: If processing fails due to invalid inputs
    """

    # 1) Load raw pose
    pose_raw = load_pose3d_from_session(sess, animal=0)
    if pose_raw is None:
        raise FileNotFoundError("load_and_process: pose_3d not resolved from session")

    # 2) Upright orientation (floor-aware when available)
    try:
        pose_upright, _ = orient_with_floor(pose_raw)
    except Exception:
        # Fallback to safe axis preference if floor metadata is missing or malformed
        pose_upright = orient_pose_upright(pose_raw, prefer="Z")

    # 3) Resolve plane + hex footprint next to the NPZ (required for centering)
    npz_dir = os.path.dirname(os.path.abspath(pose_raw.path)) if pose_raw.path else ""
    floor_path = os.path.join(npz_dir, "floor_plane.json")
    plane = None
    up_axis = None
    if os.path.isfile(floor_path):
        _, plane_upr, up_axis = rotation_from_floor_json(floor_path)
        plane = plane_upr
    if plane is None:
        raise FileNotFoundError(
            "load_and_process: floor_plane.json missing or without plane; required for centering/heights"
        )
    try:
        hex_uv = load_hex_uv(npz_dir)
    except Exception as e:
        raise FileNotFoundError(
            f"load_and_process: hex footprint not found next to pose NPZ ({npz_dir})"
        ) from e
    # Build COCO-17 bone edges based on available joint names
    try:
        idx = coco17_indices(pose_upright.joint_names)
        edges = coco17_bone_edges(idx)
    except Exception:
        edges = []

    # 3b) Enforce Z-up for downstream plots and metrics. If the floor metadata
    # declares a different up-axis (e.g., 'Y'), swap Y/Z in both pose and plane.
    def _plane_to_z_up(plane_dict: Dict[str, float], up: Optional[str]) -> Dict[str, float]:
        a = float(plane_dict.get("a", 0.0))
        b = float(plane_dict.get("b", 0.0))
        c = float(plane_dict.get("c", 1.0))
        d = float(plane_dict.get("d", 0.0))
        if isinstance(up, str) and up.strip().upper() == "Y":
            # Swap Y and Z coefficients to express the same plane in Z-up coords
            return {"a": a, "b": c, "c": b, "d": d}
        return {"a": a, "b": b, "c": c, "d": d}

    plane_z = _plane_to_z_up(plane, up_axis)
    pose_upright = orient_pose_upright(pose_upright, prefer="Z")

    # 3c) Enforce that the floor normal is exactly +Z by applying a minimal
    # corrective rotation to the pose (and plane). This guards against
    # near-upright JSONs where the plane isn't perfectly horizontal.
    import numpy as _np

    def _align_normal_to_z(n: _np.ndarray) -> _np.ndarray:
        n = _np.asarray(n, dtype=float).reshape(3)
        n = n / (float(_np.linalg.norm(n)) + 1e-12)
        z = _np.array([0.0, 0.0, 1.0], dtype=float)
        c = float(n @ z)
        if c > 0.999999:
            return _np.eye(3)
        if c < -0.999999:
            # 180-degree flip about X (arbitrary)
            return _np.diag([1.0, -1.0, -1.0])
        axis = _np.cross(n, z)
        axis = axis / (float(_np.linalg.norm(axis)) + 1e-12)
        s = float(_np.linalg.norm(_np.cross(n, z)))
        K = _np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
        return _np.eye(3) + K + K @ K * ((1 - c) / (s**2 + 1e-12))

    n0 = _np.array([float(plane_z.get("a", 0.0)), float(plane_z.get("b", 0.0)), float(plane_z.get("c", 1.0))], dtype=float)
    R_fix = _align_normal_to_z(n0)
    # Apply correction to pose coordinates
    Xr = (R_fix @ pose_upright.X.reshape(-1, 3).T).T.reshape(pose_upright.X.shape)
    pose_upright = Pose3D(
        X=Xr.astype(pose_upright.X.dtype, copy=False),
        C=pose_upright.C,
        fids=pose_upright.fids,
        fps=pose_upright.fps,
        joint_names=pose_upright.joint_names,
        up_axis="Z",
        meta=pose_upright.meta,
        path=pose_upright.path,
    )
    # Rotate plane normal too (d unchanged under pure rotation)
    n1 = R_fix @ n0
    plane_z = {"a": float(n1[0]), "b": float(n1[1]), "c": float(n1[2]), "d": float(plane_z.get("d", 0.0))}

    # 4) Run pipeline
    pp = params or PosePipelineParams(max_gap_s=1.0, smooth_win_s=0.5, limb_len_tol=0.25)
    pose_proc, _features, qc = run_pose_pipeline(
        pose_upright,
        plane=plane_z,
        hex_uv=hex_uv,
        params=pp,
        edges=edges,
        fps=None,
    )
    return PoseProcessed(pose=pose_proc, qc=qc)


__all__ = [
    "PoseProcessed",
    "load_and_process",
    "load_pose_zup_assets",
    "find_nonempty_frame_index",
    "plot_skeleton_frame_with_floor",
]
