from __future__ import annotations

"""Visualization helpers for upright 3D pose scenes.

This module avoids direct Plotly dependencies by returning geometry arrays.
Notebooks can convert these into Plotly traces (Scatter3d/Mesh3d).

Provided utilities:
- Plane basis helpers (from upright plane a,b,c,d)
- Load hex footprint (floor_hex_uv.npz) and derive regular hex from camera UVs
- Build checkerboard meshes clipped to a hex polygon in plane UV space
- Camera gizmo wireframe segments (small frustum outline)

New geometry helper:
- foot_of_perpendicular_to_plane: orthogonal projection of 3D points onto a plane
"""

from typing import Dict, List, Optional, Tuple
import os
import numpy as np


# -------------------- Plane helpers --------------------

def plane_basis_from_dict(plane: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (n,e1,e2,x0) from upright plane dict with keys a,b,c,d.

    n: unit normal (3,)
    e1,e2: orthonormal in-plane basis (3,)
    x0: a point on the plane (3,) (closest to origin: -d*n)
    """
    a = float(plane.get("a", 0.0))
    b = float(plane.get("b", 0.0))
    c = float(plane.get("c", 1.0))
    d = float(plane.get("d", 0.0))
    n = np.array([a, b, c], dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    # Choose a stable reference to build e1
    k = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, k); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1); e2 /= (np.linalg.norm(e2) + 1e-12)
    x0 = -d * n
    return n, e1, e2, x0


def project_points_to_uv(P: np.ndarray, plane: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points (N,3) onto the plane UV basis from plane dict.

    Returns (U,V) arrays of shape (N,).
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    n, e1, e2, x0 = plane_basis_from_dict(plane)
    Q = P - x0.reshape(1, 3)
    U = Q @ e1
    V = Q @ e2
    return U, V


def uv_to_xyz(U: np.ndarray, V: np.ndarray, plane: Dict[str, float]) -> np.ndarray:
    """Map UV coordinates back to 3D points on the plane (M,3)."""
    U = np.asarray(U, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float).reshape(-1)
    n, e1, e2, x0 = plane_basis_from_dict(plane)
    X = x0.reshape(1, 3) + np.outer(U, e1) + np.outer(V, e2)
    return X


def foot_of_perpendicular_to_plane(P: np.ndarray, plane: Dict[str, float]) -> np.ndarray:
    """Orthogonal projection of points onto a plane ax+by+cz+d=0.

    Args:
        P: (N,3) array of 3D points in world coordinates.
        plane: Dict with keys a,b,c,d representing the upright plane.

    Returns:
        (N,3) array of foot points on the plane along the plane normal.

    Notes:
        The plane normal is n = [a,b,c]. For each point X, the foot is:
            X_foot = X - ((a x + b y + c z + d) / ||n||^2) * n
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    a = float(plane.get("a", 0.0))
    b = float(plane.get("b", 0.0))
    c = float(plane.get("c", 1.0))
    d = float(plane.get("d", 0.0))
    n = np.array([a, b, c], dtype=float)
    nn2 = float(np.dot(n, n)) + 1e-12
    s = (P @ n + d) / nn2  # (N,) signed scalar along n
    F = P - s[:, None] * n[None, :]
    return F


# -------------------- Hex footprint --------------------

def load_hex_uv(npz_or_dir: str) -> Optional[np.ndarray]:
    """Load hex footprint from floor_hex_uv.npz if present; return (K,2) or None."""
    d = npz_or_dir
    if os.path.isfile(d):
        d = os.path.dirname(os.path.abspath(d))
    fpath = os.path.join(d, "floor_hex_uv.npz")
    if not os.path.isfile(fpath):
        return None
    try:
        with np.load(fpath, allow_pickle=False) as z:
            if "hex_uv" in z.files:
                hv = np.asarray(z["hex_uv"], dtype=float)
                if hv.ndim == 2 and hv.shape[1] == 2 and hv.shape[0] >= 6:
                    return hv
    except Exception:
        return None
    return None


def regular_hex_from_uv(uv: np.ndarray) -> Optional[np.ndarray]:
    """Fit a regular hex (6,2) to a set of UV points (N,2).

    Mirrors the approach in triangulation/cli/viser_pose.py:
    - Orientation phi from 6-fold moment of point angles
    - Inradius from median projection onto nearest side normal
    """
    if not isinstance(uv, np.ndarray) or uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] < 6:
        return None
    ctr = uv.mean(axis=0)
    D = uv - ctr.reshape(1, 2)
    ang = np.arctan2(D[:, 1], D[:, 0])
    S6 = np.exp(1j * 6.0 * ang).sum()
    phi = float(np.angle(S6) / 6.0)
    thetas = phi + (np.arange(6) * (np.pi / 3.0))
    N = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    k_idx = np.mod(np.round((ang - phi) / (np.pi / 3.0)).astype(int), 6)
    proj = np.array([float(np.dot(D[i], N[k_idx[i]])) for i in range(D.shape[0])], dtype=float)
    proj = proj[np.isfinite(proj) & (proj > 0)]
    if proj.size < 3:
        return None
    r_in = float(np.median(proj))
    R = float(2.0 * r_in / (np.sqrt(3.0) + 1e-12))
    vdirs = phi + (np.pi / 6.0) + (np.arange(6) * (np.pi / 3.0))
    V = np.stack([np.cos(vdirs), np.sin(vdirs)], axis=1)
    hex_uv = (ctr.reshape(1, 2) + R * V).astype(float)
    return hex_uv


# -------------------- Checkerboard meshes --------------------

def checker_mesh_from_hex(
    plane: Dict[str, float],
    hex_uv: np.ndarray,
    cell: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two triangle meshes (dark/light) covering a checkerboard clipped by hex polygon.

    Returns (Vd,Fd,Vl,Fl):
      - V*: (M,3) vertices (world coords on plane)
      - F*: (K,3) triangle indices
    """
    import matplotlib.path as mpath  # local import to keep this module lightweight

    uv = np.asarray(hex_uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] < 6:
        return (np.zeros((0, 3), dtype=float),) * 4
    umin, umax = float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))
    vmin, vmax = float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))
    uu = np.arange(umin, umax + cell, cell)
    vv = np.arange(vmin, vmax + cell, cell)
    hex_path = mpath.Path(uv)

    Vd: List[np.ndarray] = []
    Vl: List[np.ndarray] = []
    Fd: List[List[int]] = []
    Fl: List[List[int]] = []
    vd = 0
    vl = 0

    for i in range(len(uu) - 1):
        for j in range(len(vv) - 1):
            cx = 0.5 * (uu[i] + uu[i + 1])
            cy = 0.5 * (vv[j] + vv[j + 1])
            if not hex_path.contains_point((cx, cy)):
                continue
            sq = np.array(
                [
                    [uu[i], vv[j]],
                    [uu[i + 1], vv[j]],
                    [uu[i + 1], vv[j + 1]],
                    [uu[i], vv[j + 1]],
                ],
                dtype=float,
            )
            X = uv_to_xyz(sq[:, 0], sq[:, 1], plane)
            # Checker parity from grid indices
            if ((i + j) % 2) == 0:
                Vd.append(X)
                Fd += [[vd + 0, vd + 1, vd + 2], [vd + 0, vd + 2, vd + 3]]
                vd += 4
            else:
                Vl.append(X)
                Fl += [[vl + 0, vl + 1, vl + 2], [vl + 0, vl + 2, vl + 3]]
                vl += 4

    Vd = np.vstack(Vd) if Vd else np.zeros((0, 3), dtype=float)
    Vl = np.vstack(Vl) if Vl else np.zeros((0, 3), dtype=float)
    Fd = np.asarray(Fd, dtype=np.uint32) if Fd else np.zeros((0, 3), dtype=np.uint32)
    Fl = np.asarray(Fl, dtype=np.uint32) if Fl else np.zeros((0, 3), dtype=np.uint32)
    return Vd, Fd, Vl, Fl


# -------------------- Camera gizmos --------------------

def camera_gizmo_segments(
    calib_upr: Dict[str, dict],
    *,
    scale: float,
    ref_camera: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Return wireframe segments for each camera: list of (P0,P1,color).

    - Uses a small pyramidal frustum with a square sensor plane.
    - Colors the reference camera differently if provided.
    """
    segs: List[Tuple[np.ndarray, np.ndarray, str]] = []
    if not isinstance(calib_upr, dict):
        return segs
    for name, meta in calib_upr.items():
        try:
            Cw = np.asarray(meta["C"], dtype=float).reshape(3)
            R_wc = np.asarray(meta["R_wc"], dtype=float).reshape(3, 3)
        except Exception:
            continue
        col = "rgba(0,200,255,0.9)" if (isinstance(ref_camera, str) and name == ref_camera) else "rgba(80,80,80,0.7)"
        # Build a tiny square sensor plane and connect to Cw
        # Depth and size scale with the scene extent so gizmos remain small
        depth = 0.08 * float(scale)
        half = 0.8 * depth
        film = np.array(
            [
                [-half, -half, depth],
                [half, -half, depth],
                [half, half, depth],
                [-half, half, depth],
            ],
            dtype=float,
        )
        R_cw = R_wc.T
        # Map to world
        film_w = (film @ R_cw.T) + Cw.reshape(1, 3)
        # Pyramid edges and film square
        idx = [0, 1, 2, 3]
        for k in range(4):
            a = film_w[idx[k]]
            b = film_w[idx[(k + 1) % 4]]
            segs.append((a, b, col))
            segs.append((Cw, a, col))
    return segs
