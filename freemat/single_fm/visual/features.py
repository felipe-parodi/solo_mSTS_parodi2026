from __future__ import annotations

"""Visual features for single_fm (project-specific orchestration).

This module assembles per-bin visual covariates by:
- Loading session + calibration (upright) and upright pose
- Mapping calibration cameras to scene video files (extraction report preferred)
- Projecting a head-anchored ROI into each camera over time
- Computing optic-flow summaries and motion-energy features in the ROI
- Visibility-weighting across cameras and aggregating to a single feature vector

Notebooks should use this module to test steps incrementally and inspect outputs.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from freemat_data import (
    load_session_yaml,
    load_pose3d_from_session,
    load_pose3d,
    load_calibration_from_session,
    apply_upright_from_floor,
    orient_pose_with_floor,
    pose_time_seconds,
    video_time_to_pose_axis,
    camera_forward_world,
    write_timestamp_index_csv,
    iter_frame_pairs,
)
from freemat_data import (
    head_direction,
    compute_center_of_mass,
    compute_head_forward,
    body_orientation,
)
from freemat_data import coco17_indices
from freemat_data import (
    compute_motion_energy_in_roi,
    pca_topk,
    bilinear_sample,
    dense_flow,
)
from freemat_data import (
    frame_valid_fraction,
    longest_contiguous_segment,
)
from freemat_data.calibration import load_extraction_report
from freemat.single_fm.pose import find_nonempty_frame_index


@dataclass
class VisualParams:
    """Parameters controlling visual feature extraction.

    Args:
        bin_s: Target bin size in seconds (default: 0.05).
        me_k: Number of motion-energy PCs to retain.
        flow_algo: Optical-flow backend ('farneback' or 'tvl1' if available).
        lookahead_m: Distance in meters along head-forward for ROI center.
        roi_radius_deg: Angular half-radius in degrees (converted to pixels).
        roi_radius_px: Pixel half-radius (used if roi_radius_deg is None).
    """

    bin_s: float = 0.05
    me_k: int = 12
    flow_algo: str = "farneback"
    lookahead_m: float = 1.0
    roi_radius_deg: Optional[float] = None
    roi_radius_px: int = 60
    roi_radius_m: Optional[float] = None  # metric ROI radius on the plane; overrides px if set
    patch_hw: Tuple[int, int] = (200, 200)  # (H, W) pixels for virtual patch
    mpp: float = 0.005  # meters-per-pixel on plane
    zscore_per_session: bool = True
    kin_smooth_win_s: float = 0.50
    # Performance/selection
    frame_stride: int = 1  # process every Nth frame-pair
    cams_include: Optional[Sequence[str]] = None  # restrict to a subset (e.g., ['Cam_001'])
    # Pose validity
    conf_thresh: float = 0.5  # confidence threshold if C present
    min_valid_frac: float = 0.9  # minimum fraction of valid joints per frame
    # Video seek and performance
    seek_preroll_s: float = 2.0  # seek a bit earlier than start to stabilize decoder
    reindex_ts: bool = False  # force regenerate timestamp CSVs
    flow_ds: int = 1  # downsample factor for optic flow (1 = full, 2 = half, etc.)
    # Coverage/yaw robustness
    min_coverage: float = 0.80  # minimum patch coverage [0,1] on both frames
    min_forward_norm: float = 0.05  # minimum in-plane forward norm for yaw validity


def _invert_extraction_map(mapping: Dict[str, str]) -> Dict[str, Sequence[str]]:
    cam_to_prefix: Dict[str, Sequence[str]] = {}
    for pref, cam in (mapping or {}).items():
        cam_to_prefix.setdefault(cam, []).append(pref)
    return cam_to_prefix


def _discover_camera_files(
    *,
    video_dir: Optional[str],
    extraction_report: Optional[str],
    yaml_camera_map: Optional[Dict[str, Iterable[str]]] = None,
) -> Dict[str, str]:
    """Map calibration camera names to representative video files.

    Preferred: extraction_report mapping device prefixes → Cam_XXX, inverted
    and matched via rglob under `video_dir`. Fallback: explicit YAML globs.
    Returns a dict {camera_name: file_path}.
    """
    cam_files: Dict[str, str] = {}
    root = Path(video_dir) if video_dir else None
    if extraction_report and os.path.isfile(extraction_report):
        mapping = load_extraction_report(extraction_report)  # prefix -> Cam_XXX
        inv = _invert_extraction_map(mapping)
        for cam, prefixes in inv.items():
            matches: list[Path] = []
            if root:
                for pref in prefixes:
                    for m in root.rglob(f"{pref}-*"):
                        if m.is_file() and m.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                            matches.append(m)
            if matches:
                matches = sorted(set(matches), key=lambda p: (p.stat().st_size, str(p)), reverse=True)
                cam_files[cam] = str(matches[0])
    elif yaml_camera_map and root:
        for cam, patt in yaml_camera_map.items():
            pats = [patt] if isinstance(patt, str) else list(patt or [])
            matches: list[Path] = []
            for pat in pats:
                try:
                    for m in root.rglob(pat):
                        if m.is_file():
                            matches.append(m)
                except Exception:
                    pass
            if matches:
                matches = sorted(matches, key=lambda p: (p.stat().st_size, str(p)), reverse=True)
                cam_files[cam] = str(matches[0])
    return cam_files


def _roi_radius_px_from_deg(K: np.ndarray, theta_deg: float) -> float:
    """Convert angular half-radius to pixels using intrinsics K.

    Uses r_px ≈ f * tan(theta), where f is min(fx, fy) from K.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    f = min(fx, fy) if np.isfinite(fx) and np.isfinite(fy) else max(fx, fy)
    theta = float(theta_deg) * np.pi / 180.0
    return float(f * np.tan(theta))


def _prefer_processed_pose(sref) -> Optional[str]:
    """Return processed upright pose path if present.

    Args:
        sref: SessionRef-like object with `session_id` attribute.

    Returns:
        Path to `results/<sid>/pose_processed.npz` if it exists, else None.
    """
    sid = getattr(sref, "session_id", None)
    if not sid:
        return None
    cand = os.path.join("results", "single_fm", str(sid), "pose_processed.npz")
    return cand if os.path.isfile(cand) else None


def _resolve_floor_plane_from_npz_dir(npz_path: Optional[str]) -> Optional[str]:
    """Find `floor_plane.json` next to a pose NPZ, if present.

    Args:
        npz_path: Path to pose NPZ

    Returns:
        Path to floor_plane.json if found, else None.
    """
    if not npz_path:
        return None
    base = os.path.dirname(os.path.abspath(npz_path))
    fp = os.path.join(base, "floor_plane.json")
    return fp if os.path.isfile(fp) else None


def _ray_plane_intersect(P: np.ndarray, v: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Intersect world-space ray `P + s v` with plane `n^T X + d = 0`.

    Returns:
        World point on the plane; NaNs if parallel.
    """
    denom = float(n @ v)
    if abs(denom) < 1e-9:
        return np.full(3, np.nan)
    s = -(n @ P + d) / denom
    return P + s * v


def _tangent_basis_on_plane(f_head: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an orthonormal basis (e1,e2) on plane with normal `n`.

    e1 is the projection of `f_head` onto the plane; e2 = n × e1.
    """
    # Project head-forward onto plane; normalize → e1; e2 = n × e1
    f_proj = f_head - (np.dot(n, f_head) / (np.dot(n, n) + 1e-12)) * n
    if np.linalg.norm(f_proj) < 1e-9:
        t = np.array([1.0, 0.0, 0.0], dtype=float)
        e1 = t - (t @ n) * n / (np.dot(n, n) + 1e-12)
    else:
        e1 = f_proj
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1, e2


def _build_patch_world_grid(Qw: np.ndarray, e1: np.ndarray, e2: np.ndarray, H: int, W: int, mpp: float) -> np.ndarray:
    """Build world coordinates for a planar patch grid.

    Args:
        Qw: Patch center in world (3,)
        e1: Tangent basis vector (forward)
        e2: Tangent basis vector (lateral)
        H, W: Patch dimensions in pixels
        mpp: Meters-per-pixel (scale on plane)

    Returns:
        World coordinates array (3, H, W).
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    J, I = np.meshgrid(np.arange(W), np.arange(H))
    Xw = (
        Qw[:, None, None]
        + ((J - cx) * mpp)[None, :, :] * e1[:, None, None]
        + ((I - cy) * mpp)[None, :, :] * e2[:, None, None]
    )
    return Xw  # (3,H,W)


def _project_world_grid_to_image(Xw: np.ndarray, K: np.ndarray, R_wc: np.ndarray, T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project a world patch grid to image pixel coordinates.

    Args:
        Xw: (3,H,W) world points on a plane
        K: (3,3) intrinsics
        R_wc: (3,3) world-to-camera rotation
        T_wc: (3,) world-to-camera translation

    Returns:
        (u,v) arrays of shape (H,W), with NaNs for points behind the camera.
    """
    Xc = R_wc @ Xw.reshape(3, -1) + T_wc.reshape(3, 1)
    z = Xc[2]
    z[z <= 1e-9] = np.nan
    u = K[0, 0] * Xc[0] / z + K[0, 2]
    v = K[1, 1] * Xc[1] / z + K[1, 2]
    H, W = Xw.shape[1:]
    return u.reshape(H, W), v.reshape(H, W)


def _smooth_1d(x: np.ndarray, win_s: float, fps: float) -> np.ndarray:
    """Simple moving-average smoothing for 1D signals.

    Args:
        x: Array to smooth
        win_s: Window length in seconds
        fps: Sampling rate (Hz)
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(win_s) or win_s <= 0 or not np.isfinite(fps) or fps <= 0:
        return x
    w = max(1, int(round(win_s * fps)))
    if w <= 1:
        return x
    k = np.ones(w, dtype=float)
    k /= k.sum()
    return np.convolve(np.nan_to_num(x, nan=0.0), k, mode="same")


def _head_forward_from_pose(X: np.ndarray, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Phead, Fhead) from pose positions.

    Phead is proxied by the center-of-mass; Fhead uses nose->eye-center unit vector.
    """
    idx = coco17_indices(list(joint_names)) if callable(coco17_indices) else {}
    fwd = head_direction(
        X,
        idx_nose=idx.get("nose"),
        idx_left_eye=idx.get("left_eye"),
        idx_right_eye=idx.get("right_eye"),
    )
    com = compute_center_of_mass(X)
    return com, fwd


def build_visual_features(
    session_yaml: str,
    *,
    params: Optional[VisualParams] = None,
    yaml_video_cfg: Optional[dict] = None,
    workers: int = 0,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Assemble per-bin visual features for a session.

    Args:
        session_yaml: Path to per-session YAML.
        params: VisualParams controlling extraction; defaults are sensible.
        yaml_video_cfg: Optional override providing video_extraction settings
            (camera_name_map, timestamp_index) typically loaded from YAML.

    Returns:
        (df, qc) where df has columns:
          - t_bin_s, flow_mag, flow_div, flow_curl, flow_dir_x, flow_dir_y,
            me_pc1..me_pcK, wsum, cams_used
        qc: dict with summaries (coverage, correlations, per-camera timestamp paths).

    Raises:
        FileNotFoundError: if session YAML missing.
        RuntimeError: on missing critical assets (pose/calibration/video mapping).
    """
    if params is None:
        params = VisualParams()
    sref = load_session_yaml(session_yaml)
    # Prefer processed upright NPZ if available
    pose_npz_path = _prefer_processed_pose(sref)
    if pose_npz_path and os.path.isfile(pose_npz_path):
        pose0 = load_pose3d(pose_npz_path)
    else:
        pose0 = load_pose3d_from_session(sref)
    if pose0 is None:
        raise RuntimeError("Pose not resolved from session YAML or processed NPZ not found")
    # Upright calibration and pose
    calib = load_calibration_from_session(sref)
    floor_json = _resolve_floor_plane_from_npz_dir(pose0.path)
    calib_upr, R_upr, plane_upr, _ = apply_upright_from_floor(calib, floor_json or {})
    pose, _ = orient_pose_with_floor(pose0)

    # Times and head features
    t_pose = pose_time_seconds(pose)
    fps_pose = float(pose.fps)
    X = pose.X
    Phead = compute_center_of_mass(X)
    # Plane normal (for projection if needed)
    if plane_upr and isinstance(plane_upr, dict):
        n_vec = np.array([
            float(plane_upr.get("a", 0.0)),
            float(plane_upr.get("b", 0.0)),
            float(plane_upr.get("c", 1.0)),
        ], dtype=float)
    else:
        n_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    # Primary: head forward
    Fhead = compute_head_forward(X, pose.joint_names)
    forward_source = "head"
    nrm_f = np.linalg.norm(Fhead, axis=1)
    if not (np.isfinite(nrm_f) & (nrm_f > 1e-9)).any():
        # Fallback 1: body orientation (hips→shoulders), projected to plane
        J = X.shape[1]
        names = [str(n).strip().lower() for n in (pose.joint_names or [])]
        lut = {n: i for i, n in enumerate(names)} if names else {}
        def gi(k: str, default: Optional[int]) -> Optional[int]:
            if lut:
                return lut.get(k)
            if J >= 17:
                base = {"left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12}
                return base.get(k, default)
            return default
        body_fwd = body_orientation(
            X,
            idx_lhip=gi("left_hip", None), idx_rhip=gi("right_hip", None),
            idx_lsho=gi("left_shoulder", None), idx_rsho=gi("right_shoulder", None),
            plane_normal=n_vec,
        )
        nrm_b = np.linalg.norm(body_fwd, axis=1)
        if (np.isfinite(nrm_b) & (nrm_b > 1e-9)).any():
            Fhead = body_fwd
            forward_source = "body"
        else:
            # Fallback 2: COM velocity direction projected to plane
            dt_tmp = 1.0 / max(fps_pose, 1e-6)
            Vcom = np.gradient(Phead, dt_tmp, axis=0, edge_order=2)
            n_hat = n_vec / (np.linalg.norm(n_vec) + 1e-12)
            Vp = Vcom - (Vcom @ n_hat.reshape(3))[:, None] * n_hat.reshape(1, 3)
            nrm_v = np.linalg.norm(Vp, axis=1, keepdims=True)
            ok = nrm_v[:, 0] > 1e-9
            Vp[ok] = (Vp[ok].T / nrm_v[ok, 0]).T
            Fhead = np.where(np.isfinite(Vp), Vp, np.nan)
            nrm_v2 = np.linalg.norm(Fhead, axis=1)
            if (np.isfinite(nrm_v2) & (nrm_v2 > 1e-9)).any():
                forward_source = "velocity"
            else:
                raise RuntimeError(
                    "Head/Body/Velocity forward vectors unavailable from pose. Check COCO-17 joint ordering or provide a features parquet."
                )

    # Kinematics (speed, yaw rate) for validation
    dt = 1.0 / max(fps_pose, 1e-6)
    V = np.gradient(Phead, dt, axis=0, edge_order=2)
    speed = np.linalg.norm(V, axis=1)

    # Helpers for robust yaw
    def _yaw_rate_from_vec(vec: np.ndarray, *, thr: float) -> Tuple[np.ndarray, np.ndarray]:
        # Project vector onto plane and compute yaw rate where forward strength is sufficient.
        proj = vec - (vec @ n_vec.reshape(3))[:, None] * n_vec.reshape(1, 3) / (np.dot(n_vec, n_vec) + 1e-12)
        fwd_n = np.linalg.norm(proj, axis=1)
        valid = np.isfinite(proj).all(axis=1) & (fwd_n >= float(thr))
        # Fixed plane basis
        t_ = np.array([1.0, 0.0, 0.0], dtype=float)
        e1p = np.cross(n_vec, t_)
        if np.linalg.norm(e1p) < 1e-9:
            t_ = np.array([0.0, 1.0, 0.0], dtype=float)
            e1p = np.cross(n_vec, t_)
        e1p = e1p / (np.linalg.norm(e1p) + 1e-12)
        e2p = np.cross(n_vec, e1p); e2p = e2p / (np.linalg.norm(e2p) + 1e-12)
        u = (proj @ e1p.reshape(3)); v = (proj @ e2p.reshape(3))
        th = np.arctan2(v, u); th = np.unwrap(th)
        # Temporarily interpolate th for derivative, then remask invalid
        idx = np.arange(th.size, dtype=float)
        if valid.sum() >= 2:
            th_f = np.interp(idx, idx[valid], th[valid])
            yr = np.gradient(th_f, dt, edge_order=2)
            yr[~valid] = np.nan
        else:
            yr = np.full_like(th, np.nan, dtype=float)
        return yr, valid

    # Build candidates: head, body, velocity
    yaw_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    yr_head, v_head = _yaw_rate_from_vec(Fhead, thr=float(params.min_forward_norm))
    yaw_candidates.append(("head", yr_head, v_head))
    # Body-forward over entire sequence
    body_fwd_all = None
    try:
        names = [str(n).strip().lower() for n in (pose.joint_names or [])]
        lut = {n: i for i, n in enumerate(names)} if names else {}
        J = X.shape[1]
        def gi(k: str, default: Optional[int]) -> Optional[int]:
            if lut:
                return lut.get(k)
            if J >= 17:
                base = {"left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12}
                return base.get(k, default)
            return default
        body_fwd_all = body_orientation(
            X,
            idx_lhip=gi("left_hip", None), idx_rhip=gi("right_hip", None),
            idx_lsho=gi("left_shoulder", None), idx_rsho=gi("right_shoulder", None),
            plane_normal=n_vec,
        )
    except Exception:
        body_fwd_all = None
    if body_fwd_all is not None:
        # Align sign to head where both valid to avoid flips
        try:
            f_proj = Fhead - (Fhead @ n_vec.reshape(3))[:, None] * n_vec.reshape(1, 3) / (np.dot(n_vec, n_vec) + 1e-12)
            b_proj = body_fwd_all - (body_fwd_all @ n_vec.reshape(3))[:, None] * n_vec.reshape(1, 3) / (np.dot(n_vec, n_vec) + 1e-12)
            dot = np.sum(f_proj * b_proj, axis=1)
            sign = np.sign(dot); sign[sign == 0] = 1.0
            body_fwd_all = (body_fwd_all.T * sign).T
        except Exception:
            pass
        yr_body, v_body = _yaw_rate_from_vec(body_fwd_all, thr=float(params.min_forward_norm))
        yaw_candidates.append(("body", yr_body, v_body))
    # Velocity direction projected to plane
    n_hat = n_vec / (np.linalg.norm(n_vec) + 1e-12)
    Vp = V - (V @ n_hat.reshape(3))[:, None] * n_hat.reshape(1, 3)
    nrm_v = np.linalg.norm(Vp, axis=1, keepdims=True)
    ok = nrm_v[:, 0] > 1e-9
    Vp_norm = np.where(ok[:, None], Vp / (nrm_v + 1e-12), np.nan)
    yr_vel, v_vel = _yaw_rate_from_vec(Vp_norm, thr=0.0)
    yaw_candidates.append(("velocity", yr_vel, v_vel))

    # Select the most informative yaw source: choose the first with sufficient valid fraction and variance
    yaw_source = "head"
    yaw_rate = yr_head
    best_valid_frac = float(np.mean(np.isfinite(yr_head))) if np.isfinite(yr_head).any() else 0.0
    for name, yr, vmask in yaw_candidates:
        m = np.isfinite(yr)
        n_bins = int(m.sum())
        std = float(np.nanstd(yr[m])) if n_bins >= 3 else 0.0
        frac = float(np.mean(m)) if yr.size > 0 else 0.0
        if n_bins >= 3 and std > 1e-6 and frac >= max(0.05, best_valid_frac):
            yaw_source = name
            yaw_rate = yr
            best_valid_frac = frac
            break
    # Smooth
    if params.kin_smooth_win_s and params.kin_smooth_win_s > 0:
        speed = _smooth_1d(speed, float(params.kin_smooth_win_s), fps_pose)
        yaw_rate = _smooth_1d(yaw_rate, float(params.kin_smooth_win_s), fps_pose)

    # Pose validity mask — only compute VIS where we have usable 3D pose
    # Primary definition: frame_valid_fraction(X,C) ≥ min_valid_frac AND head-forward finite/non-trivial
    C_in = getattr(pose, "C", None)
    valid_frac = frame_valid_fraction(X, C_in, conf_thresh=float(params.conf_thresh))
    # Fallback: if confidences are degenerate (e.g., all ~0), ignore confidences and use finite-XYZ only
    use_xyz_only = False
    if C_in is not None:
        try:
            C_arr = np.asarray(C_in, dtype=float)
            # Consider degenerate if virtually no samples exceed threshold
            frac_over = float(np.mean(C_arr >= float(params.conf_thresh))) if C_arr.size > 0 else 0.0
            if not np.isfinite(frac_over) or frac_over < 0.05:
                valid_frac = np.isfinite(X).all(axis=2).mean(axis=1).astype(float)
                use_xyz_only = True
        except Exception:
            # If any issue with confidences, fall back silently to XYZ validity
            valid_frac = np.isfinite(X).all(axis=2).mean(axis=1).astype(float)
            use_xyz_only = True
    else:
        # No confidences provided → finite-XYZ validity
        use_xyz_only = True
    fwd_ok = np.isfinite(Fhead).all(axis=1) & (np.linalg.norm(Fhead, axis=1) > 1e-9)
    pose_valid_mask = (valid_frac >= float(params.min_valid_frac)) & fwd_ok

    # Camera files mapping
    video_dir = (sref.data_files or {}).get("video_dir") or (sref.metadata or {}).get("video_dir")
    extraction_report = (sref.data_files or {}).get("extraction_report")
    camera_name_map = None
    if yaml_video_cfg is None:
        # lazy load from YAML file contents
        try:
            import yaml  # type: ignore

            with open(session_yaml, "r", encoding="utf-8") as f:
                yobj = yaml.safe_load(f) or {}
            yaml_video_cfg = (yobj.get("video_extraction") or {})
        except Exception:
            yaml_video_cfg = {}
    camera_name_map = dict((yaml_video_cfg or {}).get("camera_name_map") or {})
    _ts_out = dict((yaml_video_cfg or {}).get("timestamp_index") or {})

    cam_files = _discover_camera_files(
        video_dir=video_dir,
        extraction_report=extraction_report,
        yaml_camera_map=camera_name_map,
    )
    if not cam_files:
        raise RuntimeError("No camera files resolved; check video_dir and extraction_report or YAML camera_name_map")
    if debug:
        print(f"[VIS] Cameras mapped: {len(cam_files)} -> {sorted(cam_files.keys())}")
    # Optional camera subset
    if params.cams_include:
        allow = {str(c) for c in params.cams_include}
        cam_files = {k: v for k, v in cam_files.items() if k in allow}
        if not cam_files:
            raise RuntimeError(f"cams_include filtered out all cameras: {sorted(allow)}")

    # Target bins (optionally restrict to time window)
    t0_all, t1_all = float(np.nanmin(t_pose)), float(np.nanmax(t_pose))
    # Compute main valid window from pose validity (longest contiguous run)
    vmask = (valid_frac >= float(params.min_valid_frac)) & fwd_ok
    if vmask.any():
        try:
            # Use boolean-as-float with threshold 0.5 to get longest True-run
            i0_v, i1_v = longest_contiguous_segment(vmask.astype(float), min_valid_frac=0.5)
        except Exception:
            # Fallback: first/last True index
            i0_v = int(np.argmax(vmask))
            i1_v = int(len(vmask) - np.argmax(vmask[::-1]))
    else:
        i0_v, i1_v = 0, len(t_pose)
    t0_valid, t1_valid = float(t_pose[max(i0_v, 0)]), float(t_pose[max(i1_v - 1, 0)]) if i1_v > 0 else float(t_pose[0])
    # Auto-start at first valid pose if duration provided and start not specified
    if start_s is None and duration_s is not None:
        try:
            idx0 = int(find_nonempty_frame_index(pose, min_valid_frac=float(params.min_valid_frac)))
        except Exception:
            idx0 = int(np.argmax(np.isfinite(X).all(axis=(1, 2))))
        start_s = float(t_pose[idx0])
    t0 = float(start_s) if start_s is not None else t0_all
    if duration_s is not None:
        t1 = min(t1_all, t0 + float(duration_s))
    else:
        t1 = float(end_s) if end_s is not None else t1_all
    # If caller didn’t specify any windowing, prefer the valid window by default
    if start_s is None and end_s is None and duration_s is None:
        t0, t1 = t0_valid, t1_valid
    if debug:
        frac_overall = float(np.mean(valid_frac)) if np.isfinite(valid_frac).any() else 0.0
        frac_pass = float(np.mean(valid_frac >= float(params.min_valid_frac))) if np.isfinite(valid_frac).any() else 0.0
        frac_fwd = float(np.mean(fwd_ok)) if fwd_ok.size else 0.0
        frac_both = float(np.mean(pose_valid_mask)) if pose_valid_mask.size else 0.0
        mode = "xyz_only" if use_xyz_only else "conf+xyz"
        print(f"[VIS] Pose valid window: start={t0_valid:.3f}s end={t1_valid:.3f}s (min_valid_frac={float(params.min_valid_frac):.2f}, conf_thresh={float(params.conf_thresh):.2f}, gating={mode})")
        print(f"[VIS] Time window: start={t0:.3f}s end={t1:.3f}s (pose span {t0_all:.3f}-{t1_all:.3f}s; mean valid_frac={frac_overall:.2f}; pass>=thr={frac_pass:.2f}; fwd_ok={frac_fwd:.2f}; both={frac_both:.2f})")
        try:
            idx0_dbg = int(find_nonempty_frame_index(pose, min_valid_frac=float(params.min_valid_frac)))
            print(f"[VIS] First non-empty frame via find_nonempty_frame_index: idx={idx0_dbg} t={float(t_pose[idx0_dbg]):.3f}s")
        except Exception as _e:
            pass
    bin_s = float(params.bin_s)
    t_bins = np.arange(t0, t1, bin_s, dtype=float)
    nb = t_bins.size
    keys = ("flow_mag", "flow_div", "flow_curl", "flow_dir_x", "flow_dir_y")
    acc = {k: np.zeros((nb,), dtype=float) for k in keys}
    wsum = np.zeros((nb,), dtype=float)
    cams_used = np.zeros((nb,), dtype=int)
    me_rows: list[Optional[Tuple[float, np.ndarray]]] = [None] * nb

    # Per-camera processing (optional parallel)
    ts_index_paths: Dict[str, str] = {}

    def _process_one(cam: str, vpath: str):
        if debug:
            print(f"[VIS] Cam {cam}: {vpath}")
        meta = (calib_upr or {}).get(cam) or {}
        K = np.asarray(meta.get("K"), dtype=float).reshape(3, 3) if "K" in meta else None
        R_wc = np.asarray(meta.get("R_wc"), dtype=float).reshape(3, 3) if "R_wc" in meta else None
        C = np.asarray(meta.get("C"), dtype=float).reshape(3) if "C" in meta else None
        if K is None or R_wc is None or C is None:
            raise RuntimeError(f"Calibration incomplete for camera {cam}")
        T_wc = (-R_wc @ C.reshape(3, 1)).reshape(3)
        out_csv = _ts_out.get(cam) or os.path.join("results", "single_fm", str(sref.session_id), "vis_features", "ts_index", f"{cam}.csv")
        if bool(params.reindex_ts) or (not os.path.isfile(out_csv)):
            write_timestamp_index_csv(vpath, out_csv)
        # Seek into the desired window on the video axis (approximate pose time ≈ video time)
        v_start = max(0.0, float(t0) - float(params.seek_preroll_s))
        v_end = float(t1) + 1.0
        if debug:
            print(f"[VIS] Cam {cam}: seek ~ video [{v_start:.3f}, {v_end:.3f}]s for pose window [{t0:.3f}, {t1:.3f}]s")
        pair_iter = iter_frame_pairs(vpath, start_s=v_start, end_s=v_end)
        fcam = camera_forward_world(R_wc)
        fcam = fcam / (np.linalg.norm(fcam) + 1e-12)
        Hpatch, Wpatch = params.patch_hw
        cy, cx = (Hpatch - 1) / 2.0, (Wpatch - 1) / 2.0
        if plane_upr and isinstance(plane_upr, dict):
            n = np.array([float(plane_upr.get("a", 0.0)), float(plane_upr.get("b", 0.0)), float(plane_upr.get("c", 1.0))], dtype=float)
            d = float(plane_upr.get("d", 0.0))
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
            d = 0.0
        yy_grid, xx_grid = np.mgrid[0:Hpatch, 0:Wpatch]
        r_px_eff_cam = int(round((float(params.roi_radius_m) / float(params.mpp)) if (params.roi_radius_m is not None) else float(params.roi_radius_px)))
        roi_mask = (xx_grid - cx) ** 2 + (yy_grid - cy) ** 2 <= float(r_px_eff_cam) ** 2
        acc_cam = {k: np.zeros((nb,), dtype=float) for k in keys}
        wsum_cam = np.zeros((nb,), dtype=float)
        cams_used_cam = np.zeros((nb,), dtype=int)
        me_rows_cam: list[Optional[Tuple[float, np.ndarray]]] = [None] * nb
        frame_idx = 0
        stride_dynamic = None if int(params.frame_stride) <= 0 else int(params.frame_stride)
        dt_pair_est = None
        pairs_total = 0
        pairs_used = 0
        pairs_skipped_pose = 0
        skipped_time_window = 0
        skipped_plane = 0
        skipped_coverage = 0
        skipped_wneg = 0
        skipped_empty_mask = 0
        t_cam_start = time.perf_counter()
        t_sum_sample = 0.0
        t_sum_flow = 0.0
        for (img0, t0s), (img1, t1s) in pair_iter:
            frame_idx += 1
            pairs_total += 1
            if stride_dynamic is None and dt_pair_est is None:
                dt_pair_est = float(max(t1s - t0s, 1e-6))
                stride_dynamic = max(1, int(round(float(params.bin_s) / dt_pair_est)))
                if debug:
                    print(f"[VIS] Cam {cam}: pair_dt~{dt_pair_est:.4f}s stride={stride_dynamic}")
            if int(stride_dynamic) > 1 and (frame_idx % int(stride_dynamic) != 0):
                continue
            t_cur = float(video_time_to_pose_axis(np.array([t1s], dtype=float), pose)[0])
            # Enforce time window
            if t_cur < t0:
                skipped_time_window += 1
                continue
            if t_cur > t1:
                # Past time window; stop processing this camera quickly
                break
            if not (t0 <= t_cur <= t1):
                continue
            N = len(t_pose)
            idxp = int(np.clip(np.searchsorted(t_pose, t_cur), 1, N - 1))
            # Gate on pose validity at this time — skip if pose missing/invalid
            if not bool(pose_valid_mask[idxp]):
                pairs_skipped_pose += 1
                continue
            fh = Fhead[idxp]
            nh = fh / (np.linalg.norm(fh) + 1e-12)
            nh = nh / (np.linalg.norm(nh) + 1e-12)
            # Robust ROI center: orthogonal projection of head point to plane,
            # then optional lookahead along plane-forward (e1).
            # Qbase = Phead - ((n·Phead + d)/(n·n)) n
            denom_n = float(np.dot(n, n)) + 1e-12
            dist_along_n = (float(n @ Phead[idxp]) + float(d)) / denom_n
            Qbase = Phead[idxp] - dist_along_n * n
            e1, e2 = _tangent_basis_on_plane(nh, n)
            Qw = Qbase + (float(params.lookahead_m) * e1 if (params.lookahead_m and float(params.lookahead_m) != 0.0) else 0.0)
            if not np.isfinite(Qw).all():
                skipped_plane += 1
                continue
            Xw = _build_patch_world_grid(Qw, e1, e2, Hpatch, Wpatch, float(params.mpp))
            ugrid, vgrid = _project_world_grid_to_image(Xw, K, R_wc, T_wc)
            g0 = (0.2126 * img0[..., 0] + 0.7152 * img0[..., 1] + 0.0722 * img0[..., 2]) if img0.ndim == 3 else img0
            g1 = (0.2126 * img1[..., 0] + 0.7152 * img1[..., 1] + 0.0722 * img1[..., 2]) if img1.ndim == 3 else img1
            t_sample0 = time.perf_counter()
            patch0 = bilinear_sample(g0, ugrid, vgrid)
            patch1 = bilinear_sample(g1, ugrid, vgrid)
            t_sample1 = time.perf_counter()
            t_sum_sample += (t_sample1 - t_sample0)
            cov0 = float(np.isfinite(patch0).mean()); cov1 = float(np.isfinite(patch1).mean())
            if min(cov0, cov1) < float(params.min_coverage):
                skipped_coverage += 1
                continue
            p0 = np.nan_to_num(patch0, nan=0.0).astype(np.float32)
            p1 = np.nan_to_num(patch1, nan=0.0).astype(np.float32)
            # Optional downsample for faster flow
            ds = max(1, int(params.flow_ds))
            if ds > 1:
                try:
                    import cv2  # type: ignore
                    p0_ds = cv2.resize(p0, (Wpatch // ds, Hpatch // ds), interpolation=cv2.INTER_AREA)
                    p1_ds = cv2.resize(p1, (Wpatch // ds, Hpatch // ds), interpolation=cv2.INTER_AREA)
                    fx_ds, fy_ds = dense_flow(p0_ds, p1_ds, algo=params.flow_algo)
                    fx = cv2.resize(fx_ds, (Wpatch, Hpatch), interpolation=cv2.INTER_LINEAR) * float(ds)
                    fy = cv2.resize(fy_ds, (Wpatch, Hpatch), interpolation=cv2.INTER_LINEAR) * float(ds)
                except Exception:
                    fx, fy = dense_flow(p0, p1, algo=params.flow_algo)
            else:
                fx, fy = dense_flow(p0, p1, algo=params.flow_algo)
            t_flow1 = time.perf_counter()
            t_sum_flow += (t_flow1 - t_sample1)
            try:
                import cv2  # type: ignore
                fx = cv2.GaussianBlur(fx, (3, 3), 0); fy = cv2.GaussianBlur(fy, (3, 3), 0)
            except Exception:
                # Smoothing is optional; continue without it if unavailable
                pass
            mask = roi_mask
            if not mask.any():
                skipped_empty_mask += 1
                continue
            mag = np.hypot(fx, fy)
            du_dx = np.gradient(fx, axis=1); dv_dy = np.gradient(fy, axis=0)
            dv_dx = np.gradient(fy, axis=1); du_dy = np.gradient(fx, axis=0)
            div = (du_dx + dv_dy)[mask].mean(); curl = (dv_dx - du_dy)[mask].mean()
            mag_mean = mag[mask].mean()
            mx = float(fx[mask].mean()); my = float(fy[mask].mean())
            norm = np.hypot(mx, my) + 1e-12
            stats = {"flow_mag": float(mag_mean), "flow_div": float(div), "flow_curl": float(curl), "flow_dir_x": mx / norm, "flow_dir_y": my / norm}
            w = float(max(0.0, np.dot(nh, fcam)))
            if w <= 0:
                skipped_wneg += 1
                continue
            diff = (p1 - p0).astype(np.float32)
            me_vec = compute_motion_energy_in_roi(diff, (cx, cy), float(r_px_eff_cam))
            b = int(np.clip(np.searchsorted(t_bins, t_cur) - 1, 0, nb - 1))
            for k in keys:
                acc_cam[k][b] += w * float(stats[k])
            wsum_cam[b] += w
            cams_used_cam[b] += 1
            pairs_used += 1
            if me_vec is not None:
                if me_rows_cam[b] is None:
                    me_rows_cam[b] = (float(w), me_vec.astype(float))
                else:
                    w_prev, v_prev = me_rows_cam[b]
                    w_tot = float(w_prev + w)
                    v_new = (v_prev * w_prev + me_vec * w) / (w_tot + 1e-12)
                    me_rows_cam[b] = (w_tot, v_new)
        t_cam_end = time.perf_counter()
        if debug:
            t_total = t_cam_end - t_cam_start
            print(
                f"[VIS] Cam {cam} done. pairs_total={pairs_total} used={pairs_used} \n"
                f"       skipped: pose={pairs_skipped_pose} time_window={skipped_time_window} plane={skipped_plane} coverage={skipped_coverage} w<=0={skipped_wneg} empty_mask={skipped_empty_mask} \n"
                f"       timing: total={t_total:.2f}s sample={t_sum_sample:.2f}s flow={t_sum_flow:.2f}s"
            )
        return cam, out_csv, acc_cam, wsum_cam, cams_used_cam, me_rows_cam, {
            "pairs_total": int(pairs_total),
            "pairs_used": int(pairs_used),
            "pairs_skipped_pose": int(pairs_skipped_pose),
            "skipped_time_window": int(skipped_time_window),
            "skipped_plane": int(skipped_plane),
            "skipped_coverage": int(skipped_coverage),
            "skipped_wneg": int(skipped_wneg),
            "skipped_empty_mask": int(skipped_empty_mask),
        }

    results = []
    if int(workers) > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futs = {ex.submit(_process_one, cam, vpath): cam for cam, vpath in cam_files.items()}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for cam, vpath in cam_files.items():
            results.append(_process_one(cam, vpath))

    qc_pairs: Dict[str, Dict[str, int]] = {}
    for cam, out_csv, acc_cam, wsum_cam, cams_used_cam, me_rows_cam, pair_stats in results:
        ts_index_paths[cam] = out_csv
        qc_pairs[cam] = pair_stats
        for k in keys:
            acc[k] += acc_cam[k]
        wsum += wsum_cam
        cams_used += cams_used_cam
        for i in range(nb):
            if me_rows_cam[i] is None:
                continue
            w_cam, v_cam = me_rows_cam[i]
            if me_rows[i] is None:
                me_rows[i] = (float(w_cam), v_cam.astype(float))
            else:
                w_prev, v_prev = me_rows[i]
                w_tot = float(w_prev + w_cam)
                v_new = (v_prev * w_prev + v_cam * w_cam) / (w_tot + 1e-12)
                me_rows[i] = (w_tot, v_new)

    # Normalize by wsum and assemble DataFrame (avoid invalid divide warnings)
    feat: Dict[str, np.ndarray] = {}
    mask_w = wsum > 0
    for k in keys:
        out = np.full_like(wsum, np.nan, dtype=float)
        np.divide(acc[k], wsum, out=out, where=mask_w)
        feat[k] = out
    df = pd.DataFrame({"t_bin_s": t_bins})
    for k in keys:
        df[k] = feat[k]
    df["wsum"] = wsum
    df["cams_used"] = cams_used

    # PCA over ME rows where vectors are available
    if any(r is not None for r in me_rows):
        maxlen = max((len(r[1]) for r in me_rows if r is not None), default=0)
        if maxlen > 0:
            M = np.full((nb, maxlen), np.nan, dtype=float)
            for i, r in enumerate(me_rows):
                if r is not None:
                    M[i, : len(r[1])] = r[1]
            msk = np.isfinite(M).all(axis=1)
            if msk.any():
                scores, comps = pca_topk(M[msk], params.me_k)
                K = scores.shape[1]
                out = np.full((nb, K), np.nan, dtype=float)
                out[msk, :] = scores
                for j in range(K):
                    df[f"me_pc{j+1}"] = out[:, j]

    # Optional z-scoring
    if params.zscore_per_session:
        for col in [*keys] + [c for c in df.columns if c.startswith("me_pc")]:
            if col in df.columns:
                m = float(np.nanmean(df[col])) if np.isfinite(df[col]).any() else 0.0
                s = float(np.nanstd(df[col])) if np.isfinite(df[col]).any() else 1.0
                s = s if s > 1e-12 else 1.0
                df[col] = (df[col] - m) / s

    # QC summary with basic correlations for validation
    # Interpolate kinematics onto bin centers for correlation
    sp_b = np.interp(t_bins, t_pose, speed)
    yr_b = np.interp(t_bins, t_pose, yaw_rate)
    def _nan_corr(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        x = a[m].astype(float)
        y = b[m].astype(float)
        x -= float(np.nanmean(x)); y -= float(np.nanmean(y))
        sx = float(np.sqrt(np.sum(x * x)) + 1e-12)
        sy = float(np.sqrt(np.sum(y * y)) + 1e-12)
        if sx < 1e-9 or sy < 1e-9:
            return float("nan")
        return float(np.dot(x, y) / (sx * sy))
    a_div = df["flow_div"].to_numpy()
    a_curl = df["flow_curl"].to_numpy()
    m_div = np.isfinite(a_div) & np.isfinite(sp_b)
    m_curl = np.isfinite(a_curl) & np.isfinite(yr_b)
    c_div = _nan_corr(a_div, sp_b)
    c_curl = _nan_corr(a_curl, yr_b)
    qc: Dict[str, object] = {
        "bin_s": bin_s,
        "flow_algo": params.flow_algo,
        "roi_radius_px": float(params.roi_radius_px),
        "roi_radius_m": float(params.roi_radius_m) if params.roi_radius_m is not None else None,
        "roi_radius_deg": float(params.roi_radius_deg) if params.roi_radius_deg is not None else None,
        "timestamp_index_csv": ts_index_paths,
        "cams_mapped": list(cam_files.keys()),
        "corr_div_speed": c_div,
        "corr_curl_yaw": c_curl,
        "session_id": str(getattr(sref, "session_id", "")),
        "forward_source": forward_source,
        "yaw_source": yaw_source,
        "yaw_std": float(np.nanstd(yaw_rate)) if np.isfinite(yaw_rate).any() else 0.0,
        "yaw_valid_frac": float(np.mean(np.isfinite(yaw_rate))) if yaw_rate.size else 0.0,
        "corr_bins_div_speed": int(m_div.sum()),
        "corr_bins_curl_yaw": int(m_curl.sum()),
        "pose_valid_frames": int(np.sum(pose_valid_mask)),
        "pose_total_frames": int(len(pose_valid_mask)),
        "per_camera_pair_counts": qc_pairs,
        "validity_mode": ("xyz_only" if use_xyz_only else "conf+xyz"),
        "valid_frac_mean": float(np.nanmean(valid_frac)) if np.isfinite(valid_frac).any() else 0.0,
        "valid_frac_pass": float(np.mean(valid_frac >= float(params.min_valid_frac))) if np.isfinite(valid_frac).any() else 0.0,
        "fwd_ok_frac": float(np.mean(fwd_ok)) if fwd_ok.size else 0.0,
        "pose_valid_frac": float(np.mean(pose_valid_mask)) if pose_valid_mask.size else 0.0,
    }
    return df, qc
