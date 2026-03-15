from __future__ import annotations

"""Pose3D loader that understands triangulation NPZ schema (self-contained)."""

from typing import Any, Dict, Optional, List
import os
import numpy as np

from ..types import Pose3D, SessionRef


def _load_npz_minimal(path: str) -> Dict[str, Any]:
    # allow_pickle=True needed for object arrays (joint_names, meta, etc.)
    # SECURITY NOTE: Only load NPZ files from trusted sources. Pickled objects
    # can execute arbitrary code. This is acceptable for internal pipeline outputs
    # but should not be used for user-provided or untrusted files.
    with np.load(path, allow_pickle=True) as z:
        out: Dict[str, Any] = {k: z[k] for k in z.files}
    # Attempt to parse meta_json if present
    if "meta_json" in out:
        try:
            import json
            mj = out["meta_json"]
            # Handle scalar str, 0-D ndarray, or 1-D length-1 ndarray
            if isinstance(mj, np.ndarray):
                try:
                    if mj.shape == ():
                        s = str(mj.item())
                    else:
                        s = str(mj.flat[0])
                except Exception:
                    s = str(mj)
            else:
                s = str(mj)
            out["meta"] = json.loads(s)
        except Exception:
            out["meta"] = None
    return out


def load_pose3d(npz_path: str, *, animal: int = 0) -> Pose3D:
    """Load 3D pose from a triangulation NPZ.

    Accepts NPZ files saved by freemat_vid_inference.triangulation.pose3d_io.save_pose3d_npz.
    If that loader is not importable, falls back to a minimal np.load-based parser.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    d: Dict[str, Any] = _load_npz_minimal(npz_path)

    xyz = d.get("xyz")
    if xyz is None or not isinstance(xyz, np.ndarray) or xyz.ndim != 4 or xyz.shape[-1] != 3:
        raise ValueError("NPZ missing valid 'xyz' with shape (T,N,J,3)")
    if animal < 0 or animal >= xyz.shape[1]:
        raise IndexError(f"animal index {animal} out of range N={xyz.shape[1]}")
    X = xyz[:, int(animal), :, :].astype(np.float32, copy=True)
    C = None
    conf = d.get("conf")
    if isinstance(conf, np.ndarray) and conf.ndim == 3:
        C = conf[:, int(animal), :].astype(np.float32, copy=True)
    fids = d.get("fids")
    if not isinstance(fids, np.ndarray) or fids.ndim != 1 or fids.shape[0] != X.shape[0]:
        fids = np.arange(X.shape[0], dtype=np.int64)
    # Robust fps parsing: accept scalar, 0-D ndarray, or 1-element ndarray
    fps_val = 0.0
    if "fps" in d:
        v = d["fps"]
        try:
            if isinstance(v, np.ndarray):
                fps_val = float(v.item()) if v.shape == () else float(v.flat[0])
            else:
                fps_val = float(v)
        except Exception:
            fps_val = 0.0
    fps = float(fps_val)
    # Joint names: accept numpy array or list
    jn = d.get("joint_names")
    joint_names: Optional[List[str]] = None
    if jn is not None:
        try:
            if isinstance(jn, np.ndarray):
                joint_names = [str(x) for x in jn.tolist()]
            elif isinstance(jn, list):
                joint_names = [str(x) for x in jn]
            else:
                joint_names = [str(jn)]
        except Exception:
            joint_names = None
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else None
    # up-axis preference: explicit meta.up_axis; else try meta.params.up_axis/world_up_axis; else 'Z'
    up_axis = "Z"
    if isinstance(meta, dict):
        if isinstance(meta.get("up_axis"), str):
            up_axis = str(meta["up_axis"]).strip().upper()
        else:
            params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
            cand = params.get("up_axis") or params.get("world_up_axis")
            if isinstance(cand, str):
                up_axis = str(cand).strip().upper()
    # Fallback fps from meta.params if top-level fps was absent/unparseable
    if fps <= 0.0 and isinstance(meta, dict):
        try:
            params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
            v = params.get("fps")
            if v is not None:
                fps = float(v)
        except Exception:
            pass
    return Pose3D(X=X, C=C, fids=fids.astype(np.int64), fps=fps, joint_names=joint_names, up_axis=up_axis, meta=meta, path=npz_path)


def load_pose3d_from_session(session: SessionRef, *, animal: int = 0) -> Optional[Pose3D]:
    """Load pose for a session using configured path if present.

    Looks for keys 'pose_3d' or 'pose3d_npz' in session.data_files.
    Returns None if not configured.
    """
    path = session.data_files.get("pose_3d") or session.data_files.get("pose3d_npz")
    if not path:
        return None
    return load_pose3d(str(path), animal=animal)
