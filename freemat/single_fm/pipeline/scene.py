"""Scene assembly helpers: upright-aware 3D CoM + calibration bundle.

This module centralizes the logic for building the 3D session overview panel
used in Single‑FM Figure 1. It resolves upright transforms deterministically
and avoids silent fallbacks that can misalign cameras and trajectories.

Typical usage:
    >>> from freemat.single_fm.pipeline.scene import resolve_session_and_assets, build_3d_scene
    >>> sess, assets = resolve_session_and_assets(session_id='single_0526_hedy')
    >>> bundle = build_3d_scene(sess, assets)
    >>> # bundle.com_xyz, bundle.calib_upr_centered, bundle.plane_upright

Design notes:
- Project‑agnostic primitives live in freemat_data (data IO, calibration
  upright transforms, 3D plotting helpers). This module glues those into a
  session‑aware pipeline for Single‑FM notebooks.
- We fail loudly when upright information cannot be resolved instead of
  defaulting to raw calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import json
import os
import numpy as np
import pandas as pd

from freemat_data import (
    SessionRef,
    Registry,
    load_session_yaml,
    load_calibration_from_session,
    apply_world_rotation_to_calib,
    apply_upright_from_floor,
    apply_center_translation_to_calib,
    compute_center_of_mass,
)


@dataclass
class SceneAssets:
    """Paths and manifest info needed for NB02.

    Attributes:
        aligned_parquet: Path to aligned 1.0 s parquet for the session
        pose_features_parquet: Optional path to pose features with CoM columns
        pose_processed_npz: Optional processed pose NPZ path (upright, centered)
        pose_upright_npz: Optional upright NPZ path (not centered)
        qc_json: Optional QC JSON path holding center translation
        manifest_csv: Manifest CSV used to resolve assets
        session_yaml: Session YAML path (authoritative for data_files paths)
    """

    aligned_parquet: str
    pose_features_parquet: Optional[str]
    pose_processed_npz: Optional[str]
    pose_upright_npz: Optional[str]
    qc_json: Optional[str]
    manifest_csv: Optional[str]
    session_yaml: str


@dataclass
class Scene3D:
    """Upright‑aware 3D plotting bundle for Figure 1.

    Attributes:
        com_xyz: (T,3) CoM positions in upright, centered world coords
        calib_upr_centered: Calibration dict with upright rotation and center shift applied
        plane_upright: Upright floor plane dict (a,b,c,d) if available
        sources: Human‑readable provenance notes
    """

    com_xyz: np.ndarray
    calib_upr_centered: Optional[Dict[str, dict]]
    plane_upright: Optional[Dict[str, float]]
    sources: Dict[str, Any]
    center_shift: Optional[np.ndarray]


def _read_manifest_row(manifest_csv: str, session_id: str) -> Optional[pd.Series]:
    if not manifest_csv or not os.path.isfile(manifest_csv):
        return None
    df = pd.read_csv(manifest_csv)
    row = df[df.get("session_id") == session_id]
    if row.empty:
        return None
    return row.tail(1).iloc[0]


def _fix_slashes(path: str) -> str:
    return path.replace("\\\\", "/") if isinstance(path, str) else path


def resolve_session_and_assets(
    session_id: Optional[str] = None,
    *,
    session_yaml: Optional[str] = None,
    sessions_yaml: str = "configs/sessions.yaml",
    manifest_csv: str = "results/single_fm/_manifest.csv",
) -> Tuple[SessionRef, SceneAssets]:
    """Resolve a SessionRef and NB02 asset paths from manifest/YAML.

    Args:
        session_id: Session identifier (required if using sessions_yaml)
        session_yaml: Per‑session YAML (preferred)
        sessions_yaml: Registry YAML path (fallback)
        manifest_csv: Path to results manifest CSV

    Returns:
        (SessionRef, SceneAssets)

    Raises:
        ValueError: If session cannot be resolved or aligned parquet missing
    """

    if session_yaml:
        sess = load_session_yaml(session_yaml)
    else:
        if not session_id:
            raise ValueError("resolve_session_for_nb02: provide session_yaml or (session_id + sessions_yaml)")
        reg = Registry(sessions_yaml)
        sess = reg.get(session_id)
        if sess is None:
            raise ValueError(f"Unknown session in registry: {session_id}")
        session_yaml = sess.source_yaml or ""

    sid = sess.session_id
    # Defaults (non‑manifest)
    aligned_parquet = os.path.join("results", "single_fm", "aligned", f"session_{sid}_bin1p0.parquet")
    pose_features_parquet = os.path.join("results", "single_fm", sid, "features", f"pose_features_{sid}.parquet")
    pose_processed_npz = os.path.join("results", "single_fm", sid, "pose_processed.npz")
    pose_upright_npz = os.path.join("results", "single_fm", sid, "pose_upright.npz")
    qc_json = os.path.join("results", "single_fm", sid, f"qc_{sid}.json")

    row = _read_manifest_row(manifest_csv, sid)
    if row is not None:
        # Use explicit manifest column names when available
        aligned_parquet = _fix_slashes(str(row.get("parquet_bin1p0") or aligned_parquet))
        pose_features_parquet = _fix_slashes(str(row.get("pose_features") or pose_features_parquet))
        pose_processed_npz = _fix_slashes(str(row.get("pose_processed_npz") or pose_processed_npz))
        pose_upright_npz = _fix_slashes(str(row.get("pose_upright_npz") or pose_upright_npz))
        qc_json = _fix_slashes(str(row.get("qc") or qc_json))
        if row.get("yaml"):
            session_yaml = _fix_slashes(str(row.get("yaml")))

    # Sanity existence checks are deferred to builder stage (more actionable errors)
    assets = SceneAssets(
        aligned_parquet=aligned_parquet,
        pose_features_parquet=pose_features_parquet if os.path.isfile(pose_features_parquet) else None,
        pose_processed_npz=pose_processed_npz if os.path.isfile(pose_processed_npz) else None,
        pose_upright_npz=pose_upright_npz if os.path.isfile(pose_upright_npz) else None,
        qc_json=qc_json if os.path.isfile(qc_json) else None,
        manifest_csv=manifest_csv if os.path.isfile(manifest_csv) else None,
        session_yaml=session_yaml or "",
    )
    return sess, assets


def _center_shift_from_qc_or_npz(qc_json_path: Optional[str], pose_npz: Optional[str]) -> Optional[np.ndarray]:
    # Try QC JSON first
    if qc_json_path and os.path.isfile(qc_json_path):
        with open(qc_json_path, "r") as f:
            q = json.load(f)
        v = q.get("center_translation_xyz")
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return np.asarray(v, dtype=float)
    # Try NPZ meta (processed pose stores center)
    if pose_npz and os.path.isfile(pose_npz) and pose_npz.endswith("pose_processed.npz"):
        with np.load(pose_npz, allow_pickle=False) as z:
            if "meta_json" in z.files:
                s = str(z["meta_json"].item()) if hasattr(z["meta_json"], "shape") and z["meta_json"].shape == () else str(z["meta_json"]) 
                try:
                    m = json.loads(s)
                except Exception:
                    m = {}
                v = (m or {}).get("center_translation_xyz")
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    return np.asarray(v, dtype=float)
    return None


def _resolve_floor_json_from_yaml(session_yaml: str) -> Optional[str]:
    if not session_yaml or not os.path.isfile(session_yaml):
        return None
    try:
        sess = load_session_yaml(session_yaml)
    except Exception:
        return None
    pose_path = None
    if sess.data_files and isinstance(sess.data_files.get("pose_3d"), str):
        pose_path = str(sess.data_files["pose_3d"])  # raw triangulation NPZ
    if not pose_path or not os.path.isfile(pose_path):
        return None
    floor = os.path.join(os.path.dirname(pose_path), "floor_plane.json")
    return floor if os.path.isfile(floor) else None


def _com_from_features_or_npz(features_parquet: Optional[str], pose_npz: Optional[str]) -> Tuple[np.ndarray, str]:
    # Prefer features parquet with CoM columns
    if features_parquet and os.path.isfile(features_parquet):
        df = pd.read_parquet(features_parquet)
        cx = [c for c in df.columns if c.lower() in ("com_x", "x", "pos_x", "x_m")]
        cy = [c for c in df.columns if c.lower() in ("com_y", "y", "pos_y", "y_m")]
        cz = [c for c in df.columns if c.lower() in ("com_z", "z", "pos_z", "z_m")]
        if cx and cy and cz:
            x = pd.to_numeric(df[cx[0]], errors="coerce").values
            y = pd.to_numeric(df[cy[0]], errors="coerce").values
            z = pd.to_numeric(df[cz[0]], errors="coerce").values
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if m.any():
                return np.stack([x[m], y[m], z[m]], axis=1), "features"
    # Fallback to NPZ CoM
    if pose_npz and os.path.isfile(pose_npz):
        with np.load(pose_npz, allow_pickle=False) as z:
            X = z["xyz"]  # (T,J,3)
            m = np.isfinite(X).any(axis=(1, 2))
            if not m.any():
                raise ValueError("Pose NPZ has no finite frames for CoM")
            return compute_center_of_mass(X[m]), "npz"
    raise ValueError("No features parquet or pose NPZ available for CoM")


def build_3d_scene(sess: SessionRef, assets: SceneAssets) -> Scene3D:
    """Construct CoM + upright calibration + plane for the 3D panel.

    Steps:
      1) Build CoM from features parquet if available, else from NPZ.
      2) Resolve upright rotation and plane:
         - Prefer floor_plane.json next to raw pose_3d from the session YAML.
         - If missing, raise with actionable message (do not fall back silently).
      3) Load calibration from session and rotate it upright.
      4) Center both CoM (if needed) and camera centers using center_shift.

    Returns:
        UprightBundle with com_xyz, calib_upr_centered, plane_upright, and provenance.

    Raises:
        FileNotFoundError / ValueError with clear messages when inputs are missing.
    """

    # 1) CoM
    pose_npz_choice = assets.pose_processed_npz or assets.pose_upright_npz
    com_xyz, com_src = _com_from_features_or_npz(assets.pose_features_parquet, pose_npz_choice)

    # Centering: processed NPZ is already centered; upright NPZ needs shift
    center_shift = None
    if assets.pose_processed_npz and pose_npz_choice and pose_npz_choice.endswith("pose_processed.npz"):
        center_shift = np.zeros(3, dtype=float)
    else:
        center_shift = _center_shift_from_qc_or_npz(assets.qc_json, assets.pose_processed_npz)
        if center_shift is not None and com_src == "npz":
            com_xyz = com_xyz - center_shift.reshape(1, 3)

    # 2) Upright rotation and plane (from raw pose folder)
    floor_json = _resolve_floor_json_from_yaml(assets.session_yaml)
    if floor_json is None:
        raise FileNotFoundError(
            "floor_plane.json not found next to data_files.pose_3d (check session YAML)."
        )

    # 3) Calibration load and upright
    calib = load_calibration_from_session(sess)
    if calib is None:
        raise FileNotFoundError("Calibration not found in session (calib_pkl/calib_json)")
    calib_upr, R_upr, plane_upr, up_axis = apply_upright_from_floor(calib, floor_json)
    if R_upr is None:
        raise ValueError("Upright rotation missing in floor_plane.json (upright.R_* not found)")

    # 4) Apply center shift to cameras when known
    calib_draw = calib_upr
    if center_shift is not None and np.any(np.asarray(center_shift) != 0.0):
        calib_draw = apply_center_translation_to_calib(calib_upr, center_shift)

    sources = {
        "com_source": com_src,
        "pose_npz_used": pose_npz_choice,
        "features": assets.pose_features_parquet,
        "center_shift_source": "qc_json/meta" if center_shift is not None else "none",
        "floor_plane_json": floor_json,
        "upright_up_axis": up_axis,
        "calibration_source": "session data_files",
        "manifest_csv": assets.manifest_csv,
    }
    return Scene3D(
        com_xyz=com_xyz,
        calib_upr_centered=calib_draw,
        plane_upright=plane_upr,
        sources=sources,
        center_shift=center_shift,
    )


__all__ = [
    "SceneAssets",
    "Scene3D",
    "resolve_session_and_assets",
    "build_3d_scene",
    "upright_sanity",
]


def upright_sanity(bundle: Scene3D) -> Dict[str, float]:
    """Compute simple upright alignment diagnostics.

    Returns metrics to help debug orientation:
      - plane_up_dot: |dot(plane.normal, world_up)| (expect ~1)
      - mean_cam_up_dot: mean |dot(cam_forward, world_up)| (expect ~0)
      - com_z_p50: median CoM height (m)
      - com_z_min/max: height bounds (m)

    world_up is inferred from bundle.sources['upright_up_axis'] (defaults to Z).
    """
    up_axis = str(bundle.sources.get("upright_up_axis") or "Z").upper()
    world_up = np.array([0.0, 0.0, 1.0]) if up_axis == "Z" else np.array([0.0, 1.0, 0.0])
    out: Dict[str, float] = {}
    # Plane normal alignment
    if bundle.plane_upright:
        a = float(bundle.plane_upright.get("a", 0.0))
        b = float(bundle.plane_upright.get("b", 0.0))
        c = float(bundle.plane_upright.get("c", 1.0))
        n = np.array([a, b, c], dtype=float)
        if np.linalg.norm(n) > 0:
            n = n / np.linalg.norm(n)
            out["plane_up_dot"] = float(abs(n @ world_up))
    # Camera forward vs world up
    cam_dots = []
    if bundle.calib_upr_centered:
        for meta in bundle.calib_upr_centered.values():
            R_wc = np.asarray(meta.get("R_wc"), dtype=float).reshape(3, 3)
            # Camera optical axis is +Z in camera coords → world forward = R_wc.T @ [0,0,1]
            fwd_world = R_wc.T @ np.array([0.0, 0.0, 1.0])
            fwd_world /= max(1e-9, np.linalg.norm(fwd_world))
            cam_dots.append(abs(float(fwd_world @ world_up)))
    if cam_dots:
        out["mean_cam_up_dot"] = float(np.mean(cam_dots))
    # CoM height stats
    if bundle.com_xyz is not None and bundle.com_xyz.size:
        z = np.asarray(bundle.com_xyz)[:, 2]
        z = z[np.isfinite(z)]
        if z.size:
            out["com_z_min"] = float(np.min(z))
            out["com_z_p50"] = float(np.median(z))
            out["com_z_max"] = float(np.max(z))
    return out
