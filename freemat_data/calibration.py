from __future__ import annotations

"""Lightweight calibration loaders for FreeMat data IO.

This module provides project-agnostic loaders for camera intrinsics/extrinsics
used in triangulation visualizations and downstream analyses. It avoids runtime
dependencies on the poyopo triangulation package by:

- Preferring a JSON calibration (combined intrinsics + extrinsics) when given
- Providing a best-effort PKL loader that can unpickle multical workspaces
  without the original modules by installing dummy classes into sys.modules

Returned schema per camera:
    {
      'K': (3,3) float ndarray,
      'R_wc': (3,3) float ndarray,
      'C': (3,) float ndarray,     # camera center in world
      'P': (3,4) float ndarray,    # projection matrix K @ [R|T]
      'image_size': (H, W) or (None, None)
    }
"""

from typing import Dict, Optional, Any
import os
import sys
import json
import types
import numpy as np
import pickle

from .types import SessionRef


def _install_dummy_multical_modules() -> None:
    """Install minimal dummy modules/classes so pickle can load multical workspaces.

    This maps 'multical' and submodules to simple module stubs, and injects
    placeholder classes expected by the workspace pickle. The goal is to allow
    pickle.load to succeed so that we can introspect the resulting object's
    attributes to extract K, R_wc, C.
    """
    try:
        import multical  # noqa: F401
        return  # available; nothing to do
    except Exception:
        pass

    def _ensure_module(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    # Create package and expected submodules
    pkg = _ensure_module('multical')
    ws_mod = _ensure_module('multical.workspace')
    cfg_mod = _ensure_module('multical.config')

    class _Workspace:  # placeholder
        pass

    class _PointTable:  # placeholder
        pass

    class _MCEstimatorType:  # placeholder
        pass

    # Attach placeholder classes to submodules
    ws_mod.Workspace = _Workspace  # type: ignore[attr-defined]
    ws_mod.PointTable = _PointTable  # type: ignore[attr-defined]
    cfg_mod.MCEstimatorType = _MCEstimatorType  # type: ignore[attr-defined]


def _extract_from_workspace(ws: Any) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract per-camera matrices from a loaded multical workspace object.

    Relies on common attribute names observed in multical workspaces:
      - ws.names.camera -> iterable of camera names
      - ws.calibrations[...] provides access to:
          .cameras (list per-cam intrinsics with 'intrinsic', 'dist', 'image_size')
          .camera_poses.pose_table.poses -> (N,4,4) world-to-camera matrices
    """
    cams: Dict[str, Dict[str, np.ndarray]] = {}

    # Names
    try:
        cam_names = list(getattr(getattr(ws, 'names'), 'camera'))
    except Exception as e:
        raise RuntimeError(f"workspace missing names.camera: {e}")

    # Find a calibration entry
    try:
        cals = getattr(ws, 'calibrations')
        cal_obj = next(iter(cals.values()))
    except Exception as e:
        raise RuntimeError(f"workspace missing calibrations: {e}")

    # Intrinsics
    K_list = []
    try:
        cameras = getattr(cal_obj, 'cameras')
        if len(cam_names) != len(cameras):
            # continue anyway; align by index
            pass
        for i, cam_calib in enumerate(cameras):
            try:
                K_raw = getattr(cam_calib, 'intrinsic', None)
                dist_raw = getattr(cam_calib, 'dist', None)
                img_size = getattr(cam_calib, 'image_size', (None, None))
                K = np.array(K_raw, dtype=float) if K_raw is not None else None
                if K is None or K.shape != (3, 3):
                    K_list.append(None)
                else:
                    K_list.append(K)
                name = cam_names[i] if i < len(cam_names) else f"cam_{i}"
                cams[name] = {
                    'K': K if K is not None else np.eye(3, dtype=float),
                    'image_size': tuple(img_size) if isinstance(img_size, (tuple, list)) else (None, None),
                }
            except Exception:
                name = cam_names[i] if i < len(cam_names) else f"cam_{i}"
                cams[name] = {'K': np.eye(3, dtype=float), 'image_size': (None, None)}
                K_list.append(None)
    except Exception as e:
        raise RuntimeError(f"failed to extract intrinsics: {e}")

    # Extrinsics (world-to-camera)
    try:
        poses = getattr(getattr(getattr(cal_obj, 'camera_poses'), 'pose_table'), 'poses')
        M = np.array(poses, dtype=float)  # (N,4,4)
        if M.ndim != 3 or M.shape[1:] != (4, 4):
            raise RuntimeError(f"unexpected pose shapes: {getattr(M, 'shape', None)}")
        for i in range(M.shape[0]):
            name = cam_names[i] if i < len(cam_names) else f"cam_{i}"
            R_wc = M[i, :3, :3]
            T_wc = M[i, :3, 3].reshape(3, 1)
            C = (-R_wc.T @ T_wc).reshape(3)
            K = cams.get(name, {}).get('K', np.eye(3))
            Rt = np.hstack([R_wc, T_wc])
            P = K @ Rt
            cams[name].update({'R_wc': R_wc, 'C': C, 'P': P})
    except Exception as e:
        raise RuntimeError(f"failed to extract extrinsics: {e}")

    # Ensure arrays are np.float64
    for v in cams.values():
        for k in ('K', 'R_wc', 'C', 'P'):
            if k in v:
                v[k] = np.asarray(v[k], dtype=float)
    return cams


def load_calibration_pkl(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load calibration from a multical workspace PKL without external deps."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    _install_dummy_multical_modules()
    with open(path, 'rb') as f:
        ws = pickle.load(f)
    return _extract_from_workspace(ws)


def load_calibration_json(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load calibration from a combined JSON with keys cameras and camera_poses.

    This matches the structure used by calibration/compare_mcal_json_pkl.py.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        data = json.load(f)
    cams: Dict[str, Dict[str, np.ndarray]] = {}
    cams_node = data.get('cameras') or {}
    poses_node = data.get('camera_poses') or {}
    for name, cd in cams_node.items():
        K = np.asarray(cd.get('K'), dtype=float)
        dist = np.asarray(cd.get('dist'), dtype=float) if cd.get('dist') is not None else None
        size = cd.get('image_size') or (None, None)
        cams[name] = {
            'K': K if K.shape == (3, 3) else np.eye(3),
            'image_size': tuple(size),
        }
        if dist is not None:
            cams[name]['dist'] = dist
    for name, ed in poses_node.items():
        R = np.asarray(ed.get('R'), dtype=float)
        T = np.asarray(ed.get('T'), dtype=float).reshape(3, 1)
        if name not in cams:
            cams[name] = {'K': np.eye(3), 'image_size': (None, None)}
        R_wc = R
        C = (-R_wc.T @ T).reshape(3)
        P = cams[name]['K'] @ np.hstack([R_wc, T])
        cams[name].update({'R_wc': R_wc, 'C': C, 'P': P})
    return cams


def load_extraction_report(path: str) -> Dict[str, str]:
    """Parse extraction_report.txt to mapping {video_prefix: Cam_XXX}.

    Lines look like: "Camera: <video-name> (Cam_XXX)". We return a mapping from
    the video prefix (token before first '-') to the camera name.
    """
    import re
    mapping: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return mapping
    pat = re.compile(r"^Camera:\s+([^\s]+)\s+\((Cam_\d{3})\)")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.match(line.strip())
                if not m:
                    continue
                video = m.group(1)
                cam = m.group(2)
                prefix = video.split('-')[0]
                mapping[prefix] = cam
    except Exception:
        return {}
    return mapping


def load_calibration_from_session(session: SessionRef) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Load calibration using paths from SessionRef (pkl -> json fallback)."""
    # Try PKL first
    pkl = session.data_files.get('calib_pkl') if session.data_files else None
    if pkl and os.path.isfile(str(pkl)):
        try:
            return load_calibration_pkl(str(pkl))
        except Exception:
            # Fallback: probe for JSON siblings next to PKL
            base = os.path.dirname(str(pkl))
            for fn in ("calibration.json", "calibration_merged.json", "calib.json"):
                cand = os.path.join(base, fn)
                if os.path.isfile(cand):
                    try:
                        return load_calibration_json(cand)
                    except Exception:
                        continue
    # Try JSON
    for key in ('calib_json', 'calibration_json'):
        path = session.data_files.get(key) if session.data_files else None
        if path and os.path.isfile(str(path)):
            try:
                return load_calibration_json(str(path))
            except Exception:
                pass
    return None


def apply_world_rotation_to_calib(calib: Dict[str, dict], R: np.ndarray) -> Dict[str, dict]:
    """Apply a world-space rotation R (3x3) to all cameras.

    Matches triangulation.transforms.apply_world_rotation_to_calib:
      World' = R @ World
      C' = R @ C
      R_wc' = R_wc @ R^T
      T' = -R_wc' @ C'
      P' = K @ [R_wc' | T']
    """
    import numpy as np
    R = np.asarray(R, dtype=float).reshape(3, 3)
    out: Dict[str, dict] = {}
    for cam, meta in calib.items():
        K = np.asarray(meta.get('K'), dtype=float).reshape(3, 3)
        R_wc = np.asarray(meta.get('R_wc'), dtype=float).reshape(3, 3)
        C = np.asarray(meta.get('C'), dtype=float).reshape(3)
        Rn = R_wc @ R.T
        Cn = (R @ C.reshape(3, 1)).reshape(3)
        Tn = (-Rn @ Cn.reshape(3, 1)).reshape(3, 1)
        Rt = np.hstack([Rn, Tn])
        Pn = K @ Rt
        nm = dict(meta)
        nm['R_wc'] = Rn
        nm['C'] = Cn
        nm['P'] = Pn
        out[cam] = nm
    return out


def apply_center_translation_to_calib(calib: Dict[str, dict], center_shift: np.ndarray) -> Dict[str, dict]:
    """Translate camera centers by a world-space shift and update projection.

    Applies a translation in world coordinates to each camera center C and
    updates T = -R_wc @ C accordingly while keeping R_wc and K unchanged.

    Args:
        calib: Per-camera calibration dict with keys 'K', 'R_wc', 'C'.
        center_shift: (3,) array, world-space translation to subtract from C.

    Returns:
        New calibration dict with updated 'C' and 'P'.
    """
    import numpy as np
    cs = np.asarray(center_shift, dtype=float).reshape(3)
    out: Dict[str, dict] = {}
    for name, meta in calib.items():
        K = np.asarray(meta.get('K'), dtype=float).reshape(3, 3)
        R_wc = np.asarray(meta.get('R_wc'), dtype=float).reshape(3, 3)
        Cw = np.asarray(meta.get('C'), dtype=float).reshape(3)
        Cn = Cw - cs
        Tn = (-R_wc @ Cn.reshape(3, 1)).reshape(3, 1)
        Rt = np.hstack([R_wc, Tn])
        Pn = K @ Rt
        nm = dict(meta)
        nm['C'] = Cn
        nm['P'] = Pn
        out[name] = nm
    return out


def rotation_from_floor_json(floor: Dict[str, Any] | str) -> tuple[Optional[np.ndarray], Optional[Dict[str, float]], Optional[str]]:
    """Extract upright rotation and plane from a floor_plane.json object or path.

    Returns (R_upright, plane_upright_dict, up_axis_str). R_upright combines R_align and R_yaw
    if both present: R_total = R_yaw @ R_align. If only one is present, returns that. If neither,
    returns (None, plane_upright or None, up_axis or None).
    """
    import numpy as np
    import json as _json
    obj: Dict[str, Any]
    if isinstance(floor, str):
        if not os.path.isfile(floor):
            raise FileNotFoundError(floor)
        with open(floor, 'r') as f:
            obj = _json.load(f)
    else:
        obj = floor
    up_axis = None
    try:
        ua = obj.get('up_axis')
        if isinstance(ua, str):
            up_axis = ua.strip().upper()
    except Exception:
        up_axis = None
    upr = obj.get('upright') or {}
    Ra = upr.get('R_align')
    Ry = upr.get('R_yaw')
    R_upr: Optional[np.ndarray] = None
    try:
        Ra_m = np.asarray(Ra, dtype=float).reshape(3, 3) if isinstance(Ra, list) else None
        Ry_m = np.asarray(Ry, dtype=float).reshape(3, 3) if isinstance(Ry, list) else None
        if Ra_m is not None and Ry_m is not None:
            R_upr = Ry_m @ Ra_m
        elif Ra_m is not None:
            R_upr = Ra_m
        elif Ry_m is not None:
            R_upr = Ry_m
    except Exception:
        R_upr = None
    plane_u = None
    try:
        plane_u = upr.get('plane') if isinstance(upr, dict) else None
        if not isinstance(plane_u, dict):
            plane_u = obj.get('plane') if isinstance(obj, dict) else None
    except Exception:
        plane_u = None
    return R_upr, plane_u, up_axis


def apply_upright_from_floor(calib: Dict[str, dict], floor: Dict[str, Any] | str) -> tuple[Dict[str, dict], Optional[np.ndarray], Optional[Dict[str, float]], Optional[str]]:
    """Rotate calibration into upright frame using the floor JSON.

    Returns (calib_upright, R_upright, plane_upright, up_axis)
    """
    R_upr, plane_u, up_axis = rotation_from_floor_json(floor)
    if R_upr is None:
        return calib, None, plane_u, up_axis
    return apply_world_rotation_to_calib(calib, R_upr), R_upr, plane_u, up_axis


def camera_forward_world(R_wc: np.ndarray) -> np.ndarray:
    """Return the camera optical axis (forward) in world coordinates.

    For a world-to-camera rotation `R_wc`, the camera's +Z axis in world is
    `R_wc.T @ [0,0,1]`.

    Args:
        R_wc: (3,3) world-to-camera rotation matrix.

    Returns:
        Unit 3-vector indicating camera forward direction in world.
    """
    R = np.asarray(R_wc, dtype=float).reshape(3, 3)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    f = R.T @ ez
    n = float(np.linalg.norm(f))
    return f / (n + 1e-12)


def project_world_to_pixels(K: np.ndarray, R_wc: np.ndarray, T_wc: np.ndarray, Pw: np.ndarray) -> tuple[float, float]:
    """Project a world-space point into pixel coordinates.

    Args:
        K: (3,3) camera intrinsics.
        R_wc: (3,3) world-to-camera rotation.
        T_wc: (3,) or (3,1) world-to-camera translation.
        Pw: (3,) world point (meters).

    Returns:
        (u, v) pixel coordinates. Returns (nan, nan) if point is behind camera.
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    R = np.asarray(R_wc, dtype=float).reshape(3, 3)
    t = np.asarray(T_wc, dtype=float).reshape(3, 1)
    Xw = np.asarray(Pw, dtype=float).reshape(3, 1)
    Xc = R @ Xw + t
    if float(Xc[2, 0]) <= 1e-6:
        return float("nan"), float("nan")
    uvw = K @ Xc
    u = float(uvw[0, 0] / uvw[2, 0])
    v = float(uvw[1, 0] / uvw[2, 0])
    return u, v
