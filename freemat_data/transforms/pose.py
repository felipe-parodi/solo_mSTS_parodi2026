from __future__ import annotations

"""Pose transforms: interpolation, validity, centering, orientation, features.

All functions are small and composable, with explicit types and no hidden
defaults. Complex orientation (floor-plane) is handled in `freemat_data.util`
and the plane/hex helpers in `freemat_data.vis`.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import numpy as np

from ..types import Pose3D
from ..util import orient_pose_upright, orient_pose_with_floor
from ..vis import uv_to_xyz
from ..skeletons import COCO17_NAMES


@dataclass
class PosePipelineParams:
    """Parameters controlling pose processing.

    Args:
        conf_thresh: Confidence threshold for validity when confidences are present.
        min_valid_frac: Minimum per-frame valid fraction to consider a frame useful.
        max_gap_s: Maximum gap length (seconds) to interpolate.
        smooth_win_s: Smoothing window (seconds) for optional pose smoothing.
        limb_len_tol: Fractional tolerance for limb-length deviations (e.g., 0.25 → ±25%).
        vel_z_thresh: Z-score threshold for velocity outliers per joint.
        vel_spike_frac: Fraction of joints exceeding `vel_z_thresh` to flag a frame.
    """

    conf_thresh: float = 0.5
    min_valid_frac: float = 0.5
    max_gap_s: float = 1.0
    smooth_win_s: float = 0.5
    limb_len_tol: float = 0.25
    vel_z_thresh: float = 5.0
    vel_spike_frac: float = 0.25


def interpolate_pose_linear(
    X: np.ndarray,
    C: Optional[np.ndarray] = None,
    *,
    conf_thresh: float = 0.5,
    max_gap: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate small gaps per joint and coordinate.

    Args:
        X: Array of shape (T, J, 3) with 3D joint positions (meters).
        C: Optional confidence array (T, J) in [0, 1]; if provided,
           treat samples with C < conf_thresh as missing.
        conf_thresh: Confidence threshold to mark missing samples.
        max_gap: Maximum consecutive missing samples to interpolate.

    Returns:
        (X_filled, mask_valid) where mask_valid is (T, J) boolean of post‑interp validity.
    """
    X = np.asarray(X, dtype=float)
    T, J, D = X.shape
    assert D == 3, "X must be (T,J,3)"
    if C is None:
        valid = np.isfinite(X).all(axis=-1)
    else:
        valid = (np.asarray(C, dtype=float) >= float(conf_thresh)) & np.isfinite(X).all(axis=-1)

    X_out = X.copy()
    for j in range(J):
        for d in range(3):
            v = valid[:, j]
            y = X_out[:, j, d]
            # Identify missing segments
            miss = ~v
            if not miss.any():
                continue
            idx = np.arange(T)
            # Run-length encode gaps
            k = 0
            while k < T:
                if not miss[k]:
                    k += 1
                    continue
                s = k
                while k < T and miss[k]:
                    k += 1
                e = k  # [s,e)
                gap = e - s
                if gap <= max_gap:
                    # Interpolate between neighbors if available
                    left = s - 1
                    right = e
                    y0 = y[left] if left >= 0 and np.isfinite(y[left]) else None
                    y1 = y[right] if right < T and np.isfinite(y[right]) else None
                    if y0 is not None and y1 is not None:
                        frac = (idx[s:e] - left) / float(right - left)
                        y[s:e] = (1.0 - frac) * y0 + frac * y1
                        valid[s:e, j] = True
                # else: leave as missing
    return X_out, valid


def smooth_pose(
    X: np.ndarray,
    *,
    fps: float,
    win_s: float = 0.5,
    poly: int = 3,
) -> np.ndarray:
    """Smooth pose coordinates along time using Savitzky–Golay or moving average.

    Uses Savitzky–Golay if SciPy is available; falls back to a centered moving
    average otherwise. NaN values are linearly filled temporarily for filtering
    and restored afterward.

    Args:
        X: Pose positions (T, J, 3).
        fps: Frames per second.
        win_s: Window size in seconds (converted to odd frame count).
        poly: Polynomial order for Savitzky–Golay.

    Returns:
        Smoothed positions with same shape as input.
    """
    X = np.asarray(X, dtype=float)
    T, J, D = X.shape
    if T <= 2 or fps <= 0:
        return X.copy()
    win = max(3, int(round(float(win_s) * float(fps))))
    if win % 2 == 0:
        win += 1
    # Temporary fill for NaNs using linear interpolation per (j,d)
    Y = X.copy()
    t = np.arange(T)
    for j in range(J):
        for d in range(D):
            y = Y[:, j, d]
            mask = np.isfinite(y)
            if mask.sum() < 2:
                continue
            if not mask.all():
                Y[~mask, j, d] = np.interp(t[~mask], t[mask], y[mask])
    # Apply filter
    try:
        from scipy.signal import savgol_filter  # type: ignore

        Y = savgol_filter(Y, window_length=win, polyorder=min(poly, win - 1), axis=0, mode="interp")
    except Exception:
        # Moving average fallback (explicit per-(j,d) convolution)
        k = win
        pad = k // 2
        kernel = np.ones((k,), dtype=float) / float(k)
        Y_filt = np.empty_like(Y)
        for j in range(J):
            for d in range(D):
                y = Y[:, j, d]
                ypad = np.pad(y, (pad, pad), mode="edge")
                Y_filt[:, j, d] = np.convolve(ypad, kernel, mode="valid")
        Y = Y_filt
    # Restore original NaNs
    out = Y
    out[~np.isfinite(X)] = np.nan
    return out


def frame_valid_fraction(
    X: np.ndarray,
    C: Optional[np.ndarray] = None,
    *,
    conf_thresh: float = 0.5,
) -> np.ndarray:
    """Compute fraction of valid joints per frame.

    With confidences, a joint is valid if C >= conf_thresh and coordinates are
    finite; otherwise validity is based only on finite coordinates.

    Args:
        X: Pose positions (T, J, 3).
        C: Optional confidences (T, J) in [0, 1].
        conf_thresh: Minimum confidence to count a joint as valid.

    Returns:
        valid_frac: (T,) array with fraction in [0, 1].
    """
    X = np.asarray(X, dtype=float)
    valid_xyz = np.isfinite(X).all(axis=-1)
    if C is not None:
        C = np.asarray(C, dtype=float)
        valid = valid_xyz & (C >= float(conf_thresh))
    else:
        valid = valid_xyz
    denom = X.shape[1] if X.shape[1] > 0 else 1
    return valid.sum(axis=1).astype(float) / float(denom)


def longest_contiguous_segment(valid_frac: np.ndarray, *, min_valid_frac: float = 0.5) -> Tuple[int, int]:
    """Find the longest contiguous run where valid fraction ≥ threshold.

    Args:
        valid_frac: (T,) fraction of valid joints per frame.
        min_valid_frac: Threshold for considering a frame valid.

    Returns:
        (i0, i1) indices (inclusive start, exclusive end) of the longest run.

    Raises:
        ValueError: If no frames meet the threshold.
    """
    v = np.asarray(valid_frac, dtype=float) >= float(min_valid_frac)
    if not v.any():
        raise ValueError("No frames meet the minimum valid fraction threshold")
    T = v.size
    best_len = 0
    best = (0, 0)
    k = 0
    while k < T:
        if not v[k]:
            k += 1
            continue
        s = k
        while k < T and v[k]:
            k += 1
        e = k
        if e - s > best_len:
            best_len = e - s
            best = (s, e)
    return best


def center_to_enclosure(
    pose: Pose3D,
    *,
    plane: Dict[str, float],
    hex_uv: np.ndarray,
) -> Pose3D:
    """Translate pose so the hex footprint centroid maps to the origin.

    Args:
        pose: Upright pose (world frame consistent with plane).
        plane: Upright plane dict with keys a,b,c,d.
        hex_uv: (K,2) hex polygon vertices in plane-UV coordinates.

    Returns:
        A new Pose3D translated so the hex centroid is at (0,0,0).
    """
    if hex_uv is None or not isinstance(hex_uv, np.ndarray) or hex_uv.ndim != 2 or hex_uv.shape[1] != 2:
        raise ValueError("hex_uv must be an array of shape (K,2)")
    ctr_uv = np.asarray(hex_uv, dtype=float).mean(axis=0)
    ctr_xyz = uv_to_xyz(np.asarray([ctr_uv[0]]), np.asarray([ctr_uv[1]]), plane).reshape(3)
    X = pose.X - ctr_xyz.reshape(1, 1, 3)
    return Pose3D(
        X=X,
        C=pose.C,
        fids=pose.fids,
        fps=pose.fps,
        joint_names=pose.joint_names,
        up_axis=pose.up_axis,
        meta=pose.meta,
        path=pose.path,
    )


def compute_center_of_mass(
    X: np.ndarray,
    *,
    joints: Optional[Sequence[int]] = None,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute center‑of‑mass of selected joints.

    Args:
        X: Pose positions (T, J, 3).
        joints: Indices of joints to include (default: all).
        weights: Optional per‑joint weights (same length as `joints`).

    Returns:
        Array (T, 3) with CoM coordinates.
    """
    X = np.asarray(X, dtype=float)
    if joints is None:
        sel = np.arange(X.shape[1])
    else:
        sel = np.asarray(joints, dtype=int)
    Y = X[:, sel, :]
    if weights is None:
        w = np.ones((Y.shape[1],), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        assert w.shape[0] == Y.shape[1], "weights length must match selected joints"
    w = w / (w.sum() + 1e-12)
    return (Y * w[None, :, None]).sum(axis=1)


def compute_speed_from_com(com_xyz: np.ndarray, fps: float) -> np.ndarray:
    """Finite‑difference speed (m/s) from CoM positions.

    Args:
        com_xyz: (T, 3) array of CoM positions (meters).
        fps: Frames per second.

    Returns:
        Speed vector (T,) in m/s.
    """
    com_xyz = np.asarray(com_xyz, dtype=float)
    if com_xyz.shape[0] <= 1 or float(fps) <= 0:
        return np.zeros((com_xyz.shape[0],), dtype=float)
    vel = np.gradient(com_xyz, 1.0 / float(fps), axis=0, edge_order=2)
    return np.linalg.norm(vel, axis=-1)


def voxel_occupancy(
    XYZ: np.ndarray,
    *,
    bins: tuple[int, int, int] | int = (80, 80, 40),
    ranges: Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute 3D voxel occupancy from positions.

    Args:
        XYZ: Positions (T, 3) in meters.
        bins: Number of bins per axis or a single int.
        ranges: Optional ((xmin,xmax),(ymin,ymax),(zmin,zmax)) ranges.

    Returns:
        (H, edges) where H is (nx, ny, nz) voxel counts and edges is a list
        [x_edges, y_edges, z_edges].
    """
    XYZ = np.asarray(XYZ, dtype=float).reshape(-1, 3)
    m = np.isfinite(XYZ).all(axis=1)
    if not m.any():
        import numpy as _np
        H = _np.zeros((1, 1, 1), dtype=float)
        edges = [
            _np.linspace(0, 1, 2),
            _np.linspace(0, 1, 2),
            _np.linspace(0, 1, 2),
        ]
        return H, edges
    H, edges = np.histogramdd(XYZ[m, :], bins=bins, range=ranges)
    return H.astype(float), list(edges)


def occupancy_projections(H: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D occupancy to XY, XZ, and YZ planes.

    Args:
        H: 3D occupancy counts (nx, ny, nz).

    Returns:
        (Hxy, Hxz, Hyz) 2D arrays.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 3:
        raise ValueError("H must be 3D: (nx, ny, nz)")
    Hxy = H.sum(axis=2)
    Hxz = H.sum(axis=1)
    Hyz = H.sum(axis=0)
    return Hxy, Hxz, Hyz


def height_tier_distribution(
    z: np.ndarray,
    cuts: tuple[float, float],
) -> dict[str, float]:
    """Compute percent time in height tiers defined by cutpoints.

    Args:
        z: Height values (T,).
        cuts: (low, high) meters separating tiers.

    Returns:
        Dict with keys 'floor', 'mid', 'high' giving percentages in [0,100].
    """
    import numpy as _np
    z = _np.asarray(z, dtype=float).reshape(-1)
    m = _np.isfinite(z)
    if not m.any():
        return {"floor": 0.0, "mid": 0.0, "high": 0.0}
    z = z[m]
    lo, hi = float(cuts[0]), float(cuts[1])
    n = z.size
    p_floor = float((_np.sum(z < lo) / n) * 100.0)
    p_mid = float((_np.sum((z >= lo) & (z < hi)) / n) * 100.0)
    p_high = float((_np.sum(z >= hi) / n) * 100.0)
    return {"floor": p_floor, "mid": p_mid, "high": p_high}


def apply_upright(pose: Pose3D, prefer: str = "Z") -> Pose3D:
    """Standardize pose vertical axis (minimal Y/Z swap when needed)."""
    return orient_pose_upright(pose, prefer=prefer)


def orient_with_floor(pose: Pose3D) -> Tuple[Pose3D, Optional[str]]:
    """Orient pose using floor_plane.json if present next to the NPZ."""
    return orient_pose_with_floor(pose, prefer="upright")


def compute_derivatives(X: np.ndarray, *, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-joint velocity and acceleration via finite differences.

    Args:
        X: Pose positions (T, J, 3).
        fps: Frames per second.

    Returns:
        (V, A) where V and A are (T, J, 3).
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] <= 2 or fps <= 0:
        z = np.zeros_like(X)
        return z, z
    dt = 1.0 / float(fps)
    V = np.gradient(X, dt, axis=0, edge_order=2)
    A = np.gradient(V, dt, axis=0, edge_order=2)
    return V, A


def heights_from_floor(X: np.ndarray, *, plane: Dict[str, float]) -> np.ndarray:
    """Signed heights (meters) of joints from the upright floor plane.

    Plane is given by ax + by + cz + d = 0. Heights are positive in the
    direction of the plane normal.

    Args:
        X: Pose positions (T, J, 3).
        plane: Dict with keys a,b,c,d.

    Returns:
        heights: (T, J) array of signed distances.
    """
    X = np.asarray(X, dtype=float)
    a = float(plane.get("a", 0.0))
    b = float(plane.get("b", 0.0))
    c = float(plane.get("c", 1.0))
    d = float(plane.get("d", 0.0))
    n = np.array([a, b, c], dtype=float)
    nn = float(np.linalg.norm(n)) + 1e-12
    S = X[..., 0] * a + X[..., 1] * b + X[..., 2] * c + d
    return S / nn


def _zscore_nan(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0) + 1e-12
    return (x - mu) / sd


def remove_outlier_frames(
    X: np.ndarray,
    *,
    fps: float,
    edges: Sequence[Tuple[int, int]],
    limb_len_tol: float = 0.25,
    vel_z_thresh: float = 5.0,
    vel_spike_frac: float = 0.25,
) -> np.ndarray:
    """Detect outlier frames using limb-length deviations and velocity spikes.

    Args:
        X: Pose positions (T, J, 3) with NaNs for missing joints.
        fps: Frames per second.
        edges: Iterable of bone edges as (j0, j1) indices.
        limb_len_tol: Fractional tolerance for |len/median_len - 1|.
        vel_z_thresh: Z threshold for per-joint speed spikes.
        vel_spike_frac: Fraction of joints exceeding vel_z_thresh to mark frame outlier.

    Returns:
        mask_outlier: (T,) boolean; True for outlier frames.
    """
    X = np.asarray(X, dtype=float)
    T, J, _ = X.shape
    # Limb-length baseline
    lengths = []
    for (j0, j1) in edges:
        P = X[:, [j0, j1], :]
        valid = np.isfinite(P).all(axis=(1, 2))
        L = np.linalg.norm(P[:, 0, :] - P[:, 1, :], axis=-1)
        med = np.nanmedian(L[valid]) if valid.any() else np.nan
        if not np.isfinite(med) or med <= 1e-9:
            lengths.append(np.zeros((T,), dtype=float))
            continue
        dev = np.abs(L / med - 1.0)
        lengths.append(dev)
    limb_dev = np.vstack(lengths) if lengths else np.zeros((0, T), dtype=float)
    limb_flag = (limb_dev > float(limb_len_tol)).any(axis=0) if limb_dev.size else np.zeros((T,), dtype=bool)

    # Velocity spikes
    dt = 1.0 / float(max(fps, 1e-6))
    V = np.gradient(X, dt, axis=0, edge_order=2)
    speed = np.linalg.norm(V, axis=-1)  # (T, J)
    Z = _zscore_nan(speed)
    spike = np.isfinite(Z) & (Z > float(vel_z_thresh))
    frac = spike.sum(axis=1).astype(float) / float(max(J, 1))
    vel_flag = frac >= float(vel_spike_frac)

    return limb_flag | vel_flag


def head_direction(
    X: np.ndarray,
    *,
    idx_nose: Optional[int],
    idx_left_eye: Optional[int],
    idx_right_eye: Optional[int],
) -> np.ndarray:
    """Head direction vector from head center to nose.

    Falls back to zeros if required joints are missing.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    out = np.zeros((T, 3), dtype=float)
    if idx_nose is None or (idx_left_eye is None and idx_right_eye is None):
        return out
    nose = X[:, idx_nose, :]
    parts = []
    if idx_left_eye is not None:
        parts.append(X[:, idx_left_eye, :])
    if idx_right_eye is not None:
        parts.append(X[:, idx_right_eye, :])
    head = np.nanmean(np.stack(parts, axis=0), axis=0)
    vec = nose - head
    # Normalize per-frame
    n = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
    return vec / n


def compute_head_forward(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute head-forward unit vectors using COCO-17 semantics with fallbacks.

    Strategy (per frame):
      1) Use nose → eye-center when nose and both eyes are present.
      2) Else use nose → ear-center when nose and both ears are present.
      3) Else use cross(up, right-shoulder - left-shoulder) as a coarse proxy.

    If ``joint_names`` are provided and match COCO-17 names, indices are
    resolved by name. If names are unavailable but ``J == 17``, indices are
    assumed to follow standard COCO-17 order.

    Args:
        X: Array of 3D joint positions, shape (T, J, 3), in meters.
        joint_names: Optional list of joint names; COCO-17 names recommended.

    Returns:
        Array of shape (T, 3) with unit head-forward vectors. Frames without
        sufficient landmarks are set to NaN.
    """
    X = np.asarray(X, dtype=float)
    T, J, D = X.shape
    assert D == 3, "X must have shape (T,J,3)"

    # Build name→index LUT when possible
    names = [str(n).strip().lower() for n in (joint_names or [])]
    lut = {n: i for i, n in enumerate(names)} if names else {}

    def gi(k: str) -> Optional[int]:
        return lut.get(k)

    idx_nose = gi("nose") if lut else None
    idx_leye = gi("left_eye") if lut else None
    idx_reye = gi("right_eye") if lut else None
    idx_lear = gi("left_ear") if lut else None
    idx_rear = gi("right_ear") if lut else None
    idx_lsho = gi("left_shoulder") if lut else None
    idx_rsho = gi("right_shoulder") if lut else None

    # If names unavailable but joints follow COCO ordering, assume first 17 match COCO
    # Accept J >= 17 to support supersets (extra joints appended after COCO-17)
    if not lut and J >= 17:
        # Map canonical COCO-17 names to indices 0..16
        idx_nose = 0
        idx_leye = 1; idx_reye = 2
        idx_lear = 3; idx_rear = 4
        idx_lsho = 5; idx_rsho = 6

    def _safe(idx: Optional[int]) -> Optional[np.ndarray]:
        return X[:, int(idx), :] if (idx is not None and 0 <= int(idx) < J) else None

    nose = _safe(idx_nose)
    leye = _safe(idx_leye); reye = _safe(idx_reye)
    lear = _safe(idx_lear); rear = _safe(idx_rear)
    lsho = _safe(idx_lsho); rsho = _safe(idx_rsho)

    fwd = np.full((T, 3), np.nan, dtype=float)

    valid_nose = np.isfinite(nose).all(axis=1) if nose is not None else np.zeros((T,), dtype=bool)
    valid_eyes = (np.isfinite(leye).all(axis=1) & np.isfinite(reye).all(axis=1)) if (leye is not None and reye is not None) else np.zeros((T,), dtype=bool)
    valid_ears = (np.isfinite(lear).all(axis=1) & np.isfinite(rear).all(axis=1)) if (lear is not None and rear is not None) else np.zeros((T,), dtype=bool)
    valid_sho = (np.isfinite(lsho).all(axis=1) & np.isfinite(rsho).all(axis=1)) if (lsho is not None and rsho is not None) else np.zeros((T,), dtype=bool)

    # 1) Prefer nose → eye-center
    m = valid_nose & valid_eyes
    if m.any():
        eye_ctr = 0.5 * (leye[m] + reye[m])
        fwd[m] = nose[m] - eye_ctr

    # 2) Fallback: nose → ear-center
    m2 = (~np.isfinite(fwd).all(axis=1)) & valid_nose & valid_ears
    if m2.any():
        ear_ctr = 0.5 * (lear[m2] + rear[m2])
        fwd[m2] = nose[m2] - ear_ctr

    # 3) Coarse proxy from shoulders and up-axis
    m3 = (~np.isfinite(fwd).all(axis=1)) & valid_sho
    if m3.any():
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        shoulder_vec = rsho[m3] - lsho[m3]
        fwd[m3] = np.cross(up.reshape(1, 3), shoulder_vec)

    # Normalize; keep NaN for invalid frames
    nrm = np.linalg.norm(fwd, axis=1)
    good = np.isfinite(fwd).all(axis=1) & (nrm > 1e-12)
    fwd[good] = (fwd[good].T / nrm[good]).T
    fwd[~good] = np.nan
    return fwd


def body_orientation(
    X: np.ndarray,
    *,
    idx_lhip: Optional[int],
    idx_rhip: Optional[int],
    idx_lsho: Optional[int],
    idx_rsho: Optional[int],
    plane_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Body forward direction projected onto the floor plane.

    Defined as vector from hip-center to shoulder-center, projected to the
    plane orthogonal to `plane_normal` (if provided).
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    out = np.zeros((T, 3), dtype=float)
    if None in (idx_lhip, idx_rhip, idx_lsho, idx_rsho):
        return out
    hip = 0.5 * (X[:, idx_lhip, :] + X[:, idx_rhip, :])
    sho = 0.5 * (X[:, idx_lsho, :] + X[:, idx_rsho, :])
    vec = sho - hip
    if plane_normal is not None:
        n = np.asarray(plane_normal, dtype=float).reshape(1, 3)
        n = n / (np.linalg.norm(n) + 1e-12)
        vec = vec - (vec @ n.T) * n
    nrm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
    return vec / nrm


def symmetry_index(
    X: np.ndarray,
    *,
    pairs: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Symmetry index per frame based on left/right joint magnitudes.

    Computes mean(|L| - |R|) over all pairs; closer to 0 indicates symmetry.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    if not pairs:
        return np.zeros((T,), dtype=float)
    vals = []
    for (l, r) in pairs:
        L = np.linalg.norm(X[:, l, :], axis=-1)
        R = np.linalg.norm(X[:, r, :], axis=-1)
        vals.append(L - R)
    D = np.stack(vals, axis=1)
    return np.nanmean(D, axis=1)


def handedness_sign(
    X: np.ndarray,
    *,
    pairs: Sequence[Tuple[int, int]],
) -> float:
    """Estimate coordinate handedness from left/right x-axis separation.

    Computes the average sign of (x_left - x_right) across frames and pairs.
    Positive value implies left joints tend to have larger x than right joints.
    Returns 0.0 if insufficient data.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0 or not pairs:
        return 0.0
    vals = []
    for (l, r) in pairs:
        xl = X[:, l, 0]
        xr = X[:, r, 0]
        d = xl - xr
        if np.isfinite(d).any():
            vals.append(float(np.nanmean(d)))
    if not vals:
        return 0.0
    m = float(np.mean(vals))
    return float(np.sign(m)) if np.isfinite(m) and m != 0.0 else 0.0


def run_pose_pipeline(
    pose: Pose3D,
    *,
    plane: Dict[str, float],
    hex_uv: np.ndarray,
    params: PosePipelineParams,
    edges: Sequence[Tuple[int, int]],
    fps: Optional[float] = None,
) -> Tuple[Pose3D, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run a basic pose processing pipeline end-to-end.

    Args:
        pose: Upright Pose3D in world coordinates (consistent with plane).
        plane: Upright floor plane parameters with keys a, b, c, d.
        hex_uv: Hex footprint vertices in plane UV space (K,2); required.
        params: Processing parameters controlling gaps, smoothing, thresholds.
        edges: Bone edges as (j0, j1) for limb-length checks.
        fps: Optional override for frame rate (defaults to pose.fps).

    Steps:
        1) Center pose to enclosure using hex centroid.
        2) Compute per-frame valid fraction and select longest contiguous run
           meeting min_valid_frac.
        3) Mark outlier frames (limb-length and velocity) and set to NaN.
        4) Interpolate short gaps (≤ max_gap_s) and smooth (win_s).
        5) Derive features (velocity, acceleration, heights, CoM/speed,
           body-forward, head-direction) and produce QC metrics.

    Returns:
        Tuple of:
          - pose_out: Processed Pose3D (trimmed, interpolated, smoothed).
          - features: Dict of arrays (e.g., 'V', 'A', 'heights', 'com', 'speed',
            'body_forward', 'head_direction', 'valid_frac', 'outliers').
          - qc: Dict with 'segment' [i0,i1] and 'valid_global' (full valid_frac).
    """
    if hex_uv is None or not isinstance(hex_uv, np.ndarray):
        raise ValueError("hex_uv is required for this pipeline")
    fps0 = float(fps if fps is not None else pose.fps)

    # Center to enclosure
    pose_c = center_to_enclosure(pose, plane=plane, hex_uv=hex_uv)

    # Validity and longest segment
    valid_frac = frame_valid_fraction(pose_c.X, pose_c.C, conf_thresh=params.conf_thresh)
    min_thr_req = float(params.min_valid_frac)
    min_thr_used = min_thr_req
    try:
        i0, i1 = longest_contiguous_segment(valid_frac, min_valid_frac=min_thr_used)
    except ValueError:
        # Relax threshold adaptively if no frames meet the criterion.
        vf_mean = float(np.nanmean(valid_frac)) if np.isfinite(valid_frac).any() else 0.0
        min_thr_used = max(0.25, vf_mean * 0.5)
        try:
            i0, i1 = longest_contiguous_segment(valid_frac, min_valid_frac=min_thr_used)
        except ValueError:
            # Degenerate fallback: use the entire range
            i0, i1 = 0, pose_c.T
    Xw = pose_c.X[i0:i1]

    # Outlier frames
    outliers = remove_outlier_frames(
        Xw, fps=fps0, edges=edges, limb_len_tol=params.limb_len_tol,
        vel_z_thresh=params.vel_z_thresh, vel_spike_frac=params.vel_spike_frac,
    )
    Xw[outliers] = np.nan

    # Interpolate short gaps
    max_gap = max(1, int(round(params.max_gap_s * fps0)))
    Xf, _ = interpolate_pose_linear(Xw, None, conf_thresh=params.conf_thresh, max_gap=max_gap)

    # Smooth
    Xs = smooth_pose(Xf, fps=fps0, win_s=params.smooth_win_s)

    # Derivatives and features
    V, A = compute_derivatives(Xs, fps=fps0)
    H = heights_from_floor(Xs, plane=plane)
    com = compute_center_of_mass(Xs)
    speed = compute_speed_from_com(com, fps0)

    # Starter features: body orientation and head direction (COCO-17 by name)
    try:
        from ..skeletons import coco17_indices, coco17_left_right_pairs
    except Exception:
        coco17_indices = None  # type: ignore
        coco17_left_right_pairs = None  # type: ignore
    idx = coco17_indices(pose.joint_names) if coco17_indices is not None else {}
    plane_n = np.array([float(plane.get("a", 0.0)), float(plane.get("b", 0.0)), float(plane.get("c", 1.0))], dtype=float)
    body_fwd = body_orientation(
        Xs,
        idx_lhip=idx.get("left_hip"), idx_rhip=idx.get("right_hip"),
        idx_lsho=idx.get("left_shoulder"), idx_rsho=idx.get("right_shoulder"),
        plane_normal=plane_n,
    )
    head_dir = head_direction(
        Xs,
        idx_nose=idx.get("nose"),
        idx_left_eye=idx.get("left_eye"),
        idx_right_eye=idx.get("right_eye"),
    )
    pairs = coco17_left_right_pairs(idx) if coco17_left_right_pairs is not None else []
    sym = symmetry_index(Xs, pairs=pairs) if pairs else np.zeros((Xs.shape[0],), dtype=float)
    handed = handedness_sign(Xs, pairs=pairs) if pairs else 0.0

    # Build outputs
    pose_out = Pose3D(
        X=Xs,
        C=None if pose.C is None else pose.C[i0:i1],
        fids=pose.fids[i0:i1],
        fps=pose.fps,
        joint_names=pose.joint_names,
        up_axis=pose.up_axis,
        meta=pose.meta,
        path=pose.path,
    )
    features = {
        "V": V,
        "A": A,
        "heights": H,
        "com": com,
        "speed": speed,
        "body_forward": body_fwd,
        "head_direction": head_dir,
        "valid_frac": valid_frac[i0:i1],
        "outliers": outliers,
        "symmetry": sym,
    }
    qc = {
        "segment": np.array([i0, i1], dtype=np.int64),
        "valid_global": valid_frac,
        "min_valid_frac_required": np.array([min_thr_req], dtype=float),
        "min_valid_frac_used": np.array([min_thr_used], dtype=float),
        "handedness_sign": np.array([handed], dtype=float),
    }
    return pose_out, features, qc
