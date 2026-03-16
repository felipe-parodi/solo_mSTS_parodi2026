# kin_features.py
from __future__ import annotations

"""Kinematic features for single_fm (project-specific).

This module builds *head/torso/world* reference-frame kinematic predictors
for encoding/variance analyses, with clean handling of lags and per-session
z-scoring (fit-on-train, apply-on-test to avoid leakage).

Why this exists
---------------
- Cameras are static and the arena is static; only the monkey moves.
- Visual regressors (VIS) are computed from a *virtual head-centric patch* on the
  floor plane; they capture retinal-slip-like flow.
- To test "beyond optic flow", we need compact **KIN** features that reflect
  self-motion geometry, expressed in different reference frames and at multiple
  lags.

Public API
----------
- `KinParams`: knobs for smoothing, inclusion of POS/speed, etc.
- `compute_kin_features(...)` → dict of per-frame DataFrames aligned to `t_bins`.
- `make_lagged(kin_df, lags_ms, bin_s, drop_edge=True)` → {lag_ms: DataFrame}.
- `SessionScaler`: per-session standardization (fit/train → transform/test).

Expected inputs
---------------
- `pose`: object with fields `X` (N, J, 3), `joint_names` (list[str]), `fps` (float).
- `plane`: dict-like with keys `a,b,c,d` for the floor plane in world coords, or
  a tuple `(n, d)` where `n=(a,b,c)`.
- `t_bins`: monotonically increasing 1D array of bin times (seconds) to align to.

Notes
-----
- Uses skeleton-only geometry; no velocities from IMUs required.
- Torso forward is derived from shoulder line + floor normal, sign-aligned to
  head forward to remove 180° flips.
- Yaw is defined around the floor normal (upright), using a stable in-plane basis.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

# Project utilities assumed available in your repo; keep imports light.
try:
    from freemat_data import (
        compute_center_of_mass,
        compute_head_forward,
        pose_time_seconds,
        coco17_indices,
        frame_valid_fraction,
    )
except Exception:  # pragma: no cover - allow unit tests to stub
    compute_center_of_mass = None  # type: ignore
    compute_head_forward = None  # type: ignore
    pose_time_seconds = None  # type: ignore
    coco17_indices = lambda names: {n: i for i, n in enumerate(names)}  # minimal fallback
    frame_valid_fraction = None  # type: ignore


# -----------------------------
# Parameters
# -----------------------------

@dataclass
class KinParams:
    bin_s: float = 0.05
    smooth_win_s: float = 0.050  # light MA on COM and yaw before diff (seconds)
    include_speed: bool = True
    include_pos: bool = True      # POS (allocentric XY or XYZ)
    include_delta_yaw_rate: bool = True
    use_Z_in_pos: bool = False
    min_valid_frac: float = 0.9
    conf_thresh: float = 0.0
    ignore_confidence_for_validity: bool = True
    min_forward_norm: float = 0.05  # require sufficient in-plane forward strength


# -----------------------------
# Small linear-algebra helpers
# -----------------------------

def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project vector(s) v onto plane with normal n (both in world coords)."""
    n = np.asarray(n, dtype=float).reshape(1, 3)
    return v - (v @ n.T) * n / (np.dot(n.flatten(), n.flatten()) + 1e-12)


def _central_diff(X: np.ndarray, dt: float) -> np.ndarray:
    """Central difference with linear endpoints and NaNs propagated."""
    X = np.asarray(X, dtype=float)
    d = np.empty_like(X)
    d[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    d[0] = (X[1] - X[0]) / dt
    d[-1] = (X[-1] - X[-2]) / dt
    return d


def _nan_moving_average(x: np.ndarray, win_s: float, fps: float) -> np.ndarray:
    """NaN-aware moving average with 'same' alignment.

    Preserves NaNs when the window has no valid samples.
    """
    if win_s <= 0 or fps <= 0:
        return x
    w = max(1, int(round(win_s * fps)))
    if w <= 1:
        return x
    k = np.ones(w, dtype=float)
    # handle vector-valued x (N, D)
    if x.ndim == 1:
        v = np.asarray(x, dtype=float)
        m = np.isfinite(v).astype(float)
        v0 = np.nan_to_num(v, nan=0.0)
        num = np.convolve(v0, k, mode="same")
        den = np.convolve(m, k, mode="same")
        out = np.divide(num, den, out=np.full_like(v, np.nan, dtype=float), where=den > 0)
        return out
    else:
        out = np.empty_like(x, dtype=float)
        for j in range(x.shape[1]):
            v = np.asarray(x[:, j], dtype=float)
            m = np.isfinite(v).astype(float)
            v0 = np.nan_to_num(v, nan=0.0)
            num = np.convolve(v0, k, mode="same")
            den = np.convolve(m, k, mode="same")
            out[:, j] = np.divide(num, den, out=np.full_like(v, np.nan, dtype=float), where=den > 0)
        return out


def _unwrap_phase(theta: np.ndarray) -> np.ndarray:
    """Unwrap phase, handling NaN values by unwrapping only finite values.

    np.unwrap propagates NaN through the entire array if any NaN exists.
    This version unwraps only finite values and preserves NaN positions.
    """
    theta = np.asarray(theta, dtype=float)
    m = np.isfinite(theta)
    if m.sum() < 2:
        return np.full_like(theta, np.nan)
    out = np.full_like(theta, np.nan)
    out[m] = np.unwrap(theta[m])
    return out


def _wrap_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# Skeleton-derived directions
# -----------------------------

def _torso_forward_from_skeleton(X: np.ndarray, names: Sequence[str], n: np.ndarray,
                                 f_head_proj: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute torso forward vector from shoulders and floor normal.

    - e_y (left-right) = proj_plane(R_shoulder - L_shoulder)
    - e_z (up)         = n (unit)
    - e_x (forward)    = e_y × e_z (right-handed)
    - Sign is aligned so that e_x · f_head_proj ≥ 0 if f_head_proj is provided.

    Returns: (N,3) unit forward vectors; falls back to f_head_proj if unavailable.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    get = lambda k: name_to_idx.get(k, None)
    li, ri = get("left_shoulder"), get("right_shoulder")
    if li is None or ri is None:
        # Fallback: use head forward projection if provided
        if f_head_proj is not None:
            return _unit(f_head_proj)
        # else arbitrary forward along world-X
        N = X.shape[0]
        return np.tile(np.array([1.0, 0.0, 0.0], float), (N, 1))
    L = X[:, li, :]
    R = X[:, ri, :]
    e_y = _project_to_plane(R - L, n)
    e_y = _unit(e_y)
    e_z = _unit(np.broadcast_to(np.asarray(n, float).reshape(1, 3), e_y.shape))
    e_x = _unit(np.cross(e_y, e_z))
    if f_head_proj is not None:
        sign = np.sign(np.sum(e_x * f_head_proj, axis=1, keepdims=True))
        sign[sign == 0] = 1.0
        e_x = e_x * sign
        e_y = e_y * sign  # keep right-handedness consistent
    return e_x


# -----------------------------
# Frames and yaw
# -----------------------------

def _inplane_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a stable orthonormal basis (e1, e2) spanning the plane ⟂ n."""
    n = _unit(np.asarray(n, float).reshape(1, 3))[0]
    t = np.array([1.0, 0.0, 0.0], float)
    if abs(np.dot(t, n)) > 0.95:  # nearly parallel → choose Y
        t = np.array([0.0, 1.0, 0.0], float)
    e1 = _unit(np.cross(n, t))
    e2 = _unit(np.cross(n, e1))
    return e1, e2


def _frame_axes_from_forward(forward_world: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a world forward vector and plane normal, return (ex, ey, ez) per-time.

    ex = proj_plane(forward) (unit), ez = unit(n), ey = ez × ex
    """
    ez = _unit(np.broadcast_to(np.asarray(n, float).reshape(1, 3), forward_world.shape))
    ex = _unit(_project_to_plane(forward_world, n))
    ey = _unit(np.cross(ez, ex))
    return ex, ey, ez


def _R_F_from_axes(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> np.ndarray:
    """Build per-time rotation matrices R_{W<-F} with columns (ex, ey, ez)."""
    # Stack as (N,3,3)
    return np.stack([ex, ey, ez], axis=-1)  # columns


def _yaw_from_ex(ex: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Yaw angle of ex around normal n, using a fixed in-plane basis."""
    e1, e2 = _inplane_basis_from_normal(n)
    c = np.sum(ex * e1.reshape(1, 3), axis=1)
    s = np.sum(ex * e2.reshape(1, 3), axis=1)
    return np.arctan2(s, c)


# -----------------------------
# Main computation
# -----------------------------

def compute_kin_features(
    pose,
    plane: Mapping[str, float] | Tuple[np.ndarray, float] | np.ndarray,
    t_bins: np.ndarray,
    params: Optional[KinParams] = None,
    t_pose_video: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute kinematic features aligned to `t_bins` for frames: head/torso/world.

    Returns a dict mapping frame → DataFrame with columns:
        ['t_bin_s', 'v_parallel', 'v_perp', 'v_z', 'yaw_rate_head',
         'yaw_rate_torso', 'delta_yaw', 'delta_yaw_rate',
         'pitch_rate', 'roll_rate', 'gaze_elevation', 'head_roll',
         ('speed' if included), ('X','Y','Z' if included)]

    Feature definitions:
    - yaw_rate_head: d(head_yaw)/dt, + left turn (CCW from above)
    - pitch_rate: d(gaze_elevation)/dt, + looking up
    - roll_rate: d(head_roll)/dt, + right ear down
    - delta_yaw: head_yaw - torso_yaw, + head left of torso
    - gaze_elevation: arctan2(head_fwd_z, ||head_fwd_xy||), radians
    - head_roll: head tilt around gaze axis, radians

    Args:
        pose: Pose object with X (N,J,3), joint_names, fps.
        plane: Floor plane as dict {a,b,c,d}, tuple (n_vec, d), or normal array.
        t_bins: Target bin centers in seconds (typically video time).
        params: KinParams configuration.
        t_pose_video: Optional pose time axis in video seconds (fids / video_fps).
            If provided, used for interpolation instead of pose_time_seconds().
            IMPORTANT: Use this to fix time alignment when t_bins are in video time
            but pose_time_seconds() returns relative/stretched time.

    Design choices:
    - COM velocity via central differences (after light MA smoothing).
    - Yaw rate from unwrapped yaw(t) around the floor normal.
    - Torso forward from shoulder line x normal, sign-aligned to head forward.
    - Pitch from head-forward vertical component (gaze elevation).
    - Roll from ear positions (head "up" vector via cross product).
    - Features are computed on pose timestamps then *interpolated* to `t_bins`.
    """
    if params is None:
        params = KinParams()

    # Pose basics
    X = np.asarray(pose.X, float)  # (N,J,3)
    names: List[str] = list(getattr(pose, "joint_names", [str(i) for i in range(X.shape[1])]))
    names = _canonicalize_joint_names(names)
    fps = float(getattr(pose, "fps", 1.0))
    # Use provided video time axis if available (fixes time alignment bug)
    if t_pose_video is not None:
        t_pose = np.asarray(t_pose_video, dtype=float)
    elif pose_time_seconds is not None:
        t_pose = pose_time_seconds(pose)
    else:  # fallback: 0..N-1 / fps
        t_pose = np.arange(X.shape[0], dtype=float) / max(fps, 1e-6)

    # Plane normal & offset
    if isinstance(plane, tuple) and len(plane) == 2:
        n_vec, d = plane
        n_vec = np.asarray(n_vec, float)
    elif isinstance(plane, dict):
        n_vec = np.array([float(plane.get("a", 0.0)), float(plane.get("b", 0.0)), float(plane.get("c", 1.0))])
        d = float(plane.get("d", 0.0))
    else:
        n_vec = np.asarray(plane, float).reshape(3)
        d = 0.0
    n_hat = _unit(n_vec.reshape(1, 3))[0]

    # COM first (for potential fallback) and head forward
    if compute_center_of_mass is None:
        raise RuntimeError("freemat_data.compute_center_of_mass is required.")
    P = compute_center_of_mass(X)  # (N,3)
    # A rough velocity for fallback (no gating yet)
    dt_tmp = 1.0 / max(fps, 1e-6)
    P_tmp = _nan_moving_average(P, params.smooth_win_s, fps)
    V_tmp = np.full_like(P_tmp, np.nan, dtype=float)
    for j in range(3):
        y = P_tmp[:, j]
        m = np.isfinite(y)
        if m.sum() >= 2:
            idx = np.arange(len(y), dtype=float)
            y_f = np.interp(idx, idx[m], y[m])
            V_tmp[:, j] = _central_diff(y_f, dt_tmp)
    # Robust head-forward with skeleton; may be incomplete
    f_head = _compute_head_forward_robust(X, names, n_hat)
    # COM-heading fallback: project COM velocity onto floor and use as forward where needed
    v_proj = _project_to_plane(V_tmp, n_hat)
    v_dir = _unit(v_proj)
    fwd_norm = np.linalg.norm(f_head, axis=1)
    fwd_bad = ~(np.isfinite(f_head).all(axis=1) & (fwd_norm > 1e-9))
    v_good = np.isfinite(v_dir).all(axis=1) & (np.linalg.norm(V_tmp, axis=1) > 1e-9)
    use_fallback = fwd_bad & v_good
    if use_fallback.any():
        f_head[use_fallback] = v_dir[use_fallback]
        # lightweight notice; caller may add more detailed logging
        try:
            n_fb = int(use_fallback.sum())
            n_tot = int(len(use_fallback))
            print(f"[KIN] COM-heading fallback used for {n_fb}/{n_tot} frames ({100.0*n_fb/max(n_tot,1):.1f}%)")
        except Exception:
            pass

    # Torso forward (aligned to head fwd sign)
    f_head_proj = _project_to_plane(f_head, n_hat)
    f_torso = _torso_forward_from_skeleton(X, names, n_hat, f_head_proj=f_head_proj)

    # Frame axes & rotations (per time)
    hx, hy, hz = _frame_axes_from_forward(f_head, n_hat)
    tx, ty, tz = _frame_axes_from_forward(f_torso, n_hat)
    # World frame uses identity

    R_W_from_H = _R_F_from_axes(hx, hy, hz)   # (N,3,3)
    R_W_from_T = _R_F_from_axes(tx, ty, tz)

    # Valid mask for linear velocities: finite COM and forward, plus pose coverage threshold
    valid_lin = np.isfinite(P).all(axis=1) & np.isfinite(f_head).all(axis=1)
    try:
        if frame_valid_fraction is not None and hasattr(pose, "X"):
            vf = frame_valid_fraction(pose.X, None if bool(getattr(params, "ignore_confidence_for_validity", True)) else getattr(pose, "C", None), conf_thresh=float(params.conf_thresh))
            if vf is not None:
                valid_lin = valid_lin & (np.asarray(vf, dtype=float) >= float(params.min_valid_frac))
    except Exception:
        pass
    # Enforce a forward strength threshold in-plane (avoid unstable axes)
    fwd_plane = _project_to_plane(f_head, n_hat)
    fwd_strength = np.linalg.norm(fwd_plane, axis=1)
    try:
        thr = float(getattr(params, 'min_forward_norm', 0.0))
    except Exception:
        thr = 0.0
    valid_lin = valid_lin & (fwd_strength >= thr)

    # COM smoothing before differentiation (vector-wise), NaN-aware
    P_s = _nan_moving_average(P, params.smooth_win_s, fps)
    dt = 1.0 / max(fps, 1e-6)
    # Derivatives with NaN-aware interpolation per coordinate
    V = np.full_like(P, np.nan, dtype=float)
    for j in range(3):
        y = P_s[:, j]
        # interpolate NaNs temporally for derivative, then mask invalid back to NaN
        m = np.isfinite(y)
        if m.sum() >= 2:
            idx = np.arange(len(y), dtype=float)
            y_f = np.interp(idx, idx[m], y[m])
            d = _central_diff(y_f, dt)
            d[~valid_lin] = np.nan
            V[:, j] = d
        else:
            V[:, j] = np.nan
    speed = np.linalg.norm(V, axis=1)

    # Yaw & rates (head & torso)
    yaw_h = _unwrap_phase(_yaw_from_ex(hx, n_hat))
    yaw_t = _unwrap_phase(_yaw_from_ex(tx, n_hat))
    # Pre-derivative smoothing (unwrap domain)
    yaw_h = _nan_moving_average(yaw_h, params.smooth_win_s, fps)
    yaw_t = _nan_moving_average(yaw_t, params.smooth_win_s, fps)
    # Interpolate NaNs for differentiation; remask with yaw-valid mask (decoupled from joint coverage)
    def _diff_nan_safe(y: np.ndarray, mask_valid: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2:
            return np.full_like(y, np.nan, dtype=float)
        idx = np.arange(len(y), dtype=float)
        y_f = np.interp(idx, idx[m], y[m])
        d = _central_diff(y_f, dt)
        d[~mask_valid] = np.nan
        return d

    valid_yaw = np.isfinite(f_head).all(axis=1) & (fwd_strength >= thr)
    yaw_rate_h = _diff_nan_safe(yaw_h, valid_yaw)
    yaw_rate_t = _diff_nan_safe(yaw_t, valid_yaw)
    delta_yaw = _wrap_pi(yaw_h - yaw_t)
    delta_yaw_rate = yaw_rate_h - yaw_rate_t

    # Pitch (gaze elevation) and pitch_rate
    # gaze_elevation = arctan2(f_head_z, ||f_head_xy||) where xy is floor plane
    f_head_xy = _project_to_plane(f_head, n_hat)
    f_head_xy_norm = np.linalg.norm(f_head_xy, axis=1)
    # f_head vertical component = f_head · n_hat
    f_head_z = np.sum(f_head * n_hat.reshape(1, 3), axis=1)
    gaze_elevation = np.arctan2(f_head_z, f_head_xy_norm)
    # Smooth before differentiation
    gaze_elevation_s = _nan_moving_average(gaze_elevation, params.smooth_win_s, fps)
    pitch_rate = _diff_nan_safe(gaze_elevation_s, valid_yaw)

    # Roll and roll_rate (head tilt around gaze axis)
    # Compute head "up" from ear positions: head_up = cross(ear_to_ear, head_forward)
    idx_lear = _find_indices(names, ['left_ear'])
    idx_rear = _find_indices(names, ['right_ear'])
    if idx_lear is not None and idx_rear is not None:
        left_ear = X[:, idx_lear, :]
        right_ear = X[:, idx_rear, :]
        ear_to_ear = right_ear - left_ear  # (N, 3)
        # head_up = ear_to_ear × f_head (right-hand rule: up when looking forward)
        head_up_raw = np.cross(ear_to_ear, f_head)
        head_up = _unit(head_up_raw)
        # Roll = angle between head_up and world_up (n_hat)
        # Project head_up onto plane perpendicular to gaze to isolate roll
        # Simpler: roll = arctan2(head_up · head_left, head_up · world_up)
        # where head_left = n_hat × f_head_xy_unit
        f_head_xy_unit = _unit(f_head_xy)
        head_left = np.cross(n_hat.reshape(1, 3), f_head_xy_unit)
        # Roll: positive = right ear down (clockwise when viewed from behind)
        cos_roll = np.sum(head_up * n_hat.reshape(1, 3), axis=1)
        sin_roll = np.sum(head_up * head_left, axis=1)
        head_roll = np.arctan2(sin_roll, cos_roll)
        # Smooth and differentiate
        head_roll_s = _nan_moving_average(head_roll, params.smooth_win_s, fps)
        # Need to unwrap roll too (can wrap at ±π)
        head_roll_s = _unwrap_phase(head_roll_s)
        roll_rate = _diff_nan_safe(head_roll_s, valid_yaw)
    else:
        # No ear keypoints available
        gaze_elevation_arr = gaze_elevation  # keep for output
        head_roll = np.full(X.shape[0], np.nan, dtype=float)
        roll_rate = np.full(X.shape[0], np.nan, dtype=float)

    # Transform linear velocity into each frame at each time
    # v_F = R_{F<-W} V, where R_{F<-W} = (R_{W<-F})^T
    R_H_from_W = np.transpose(R_W_from_H, (0, 2, 1))
    R_T_from_W = np.transpose(R_W_from_T, (0, 2, 1))

    def _frame_vel(R_F_from_W: np.ndarray, V: np.ndarray) -> np.ndarray:
        # batch matrix-vector multiply: (N,3,3) @ (N,3) -> (N,3)
        return np.einsum("nij,nj->ni", R_F_from_W, V, optimize=True)

    vH = _frame_vel(R_H_from_W, V)
    vT = _frame_vel(R_T_from_W, V)
    vW = V.copy()

    # Pack time-series into dict per-frame
    frames: Dict[str, Dict[str, np.ndarray]] = {}
    frames["head"] = {
        "v_parallel": vH[:, 0],
        "v_perp": vH[:, 1],
        "v_z": vH[:, 2],
        "yaw_rate_head": yaw_rate_h,
        "yaw_rate_torso": yaw_rate_t,
        "delta_yaw": delta_yaw,
        "delta_yaw_rate": delta_yaw_rate,
        "pitch_rate": pitch_rate,
        "roll_rate": roll_rate,
        "gaze_elevation": gaze_elevation,
        "head_roll": head_roll,
    }
    frames["torso"] = {
        "v_parallel": vT[:, 0],
        "v_perp": vT[:, 1],
        "v_z": vT[:, 2],
        "yaw_rate_head": yaw_rate_h,
        "yaw_rate_torso": yaw_rate_t,
        "delta_yaw": delta_yaw,
        "delta_yaw_rate": delta_yaw_rate,
        "pitch_rate": pitch_rate,
        "roll_rate": roll_rate,
        "gaze_elevation": gaze_elevation,
        "head_roll": head_roll,
    }
    frames["world"] = {
        "v_parallel": vW[:, 0],
        "v_perp": vW[:, 1],
        "v_z": vW[:, 2],
        "yaw_rate_head": yaw_rate_h,
        "yaw_rate_torso": yaw_rate_t,
        "delta_yaw": delta_yaw,
        "delta_yaw_rate": delta_yaw_rate,
        "pitch_rate": pitch_rate,
        "roll_rate": roll_rate,
        "gaze_elevation": gaze_elevation,
        "head_roll": head_roll,
    }

    if params.include_speed:
        for f in frames.values():
            f["speed"] = speed
    if params.include_pos:
        if params.use_Z_in_pos:
            pos_cols = {"X": P[:, 0], "Y": P[:, 1], "Z": P[:, 2]}
        else:
            pos_cols = {"X": P[:, 0], "Y": P[:, 1]}
        for f in frames.values():
            f.update(pos_cols)

    # Interpolate all series to the requested bin centers
    def _interp_series(y: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        """Interpolate y(x_src) to x_dst with NaN outside the valid domain."""
        y = np.asarray(y, float)
        m = np.isfinite(y)
        if m.sum() < 2:
            return np.full_like(x_dst, np.nan, float)
        # Use numpy.interp with explicit NaN fill for out-of-bounds
        try:
            return np.interp(x_dst, x_src[m], y[m], left=np.nan, right=np.nan)
        except TypeError:
            # older numpy: no left/right → do two passes and mask outside
            y_dst = np.interp(x_dst, x_src[m], y[m])
            y_dst[x_dst < x_src[m][0]] = np.nan
            y_dst[x_dst > x_src[m][-1]] = np.nan
            return y_dst

    out: Dict[str, pd.DataFrame] = {}
    for frame_name, feats in frames.items():
        data = {k: _interp_series(v, t_pose, t_bins) for k, v in feats.items()}
        df = pd.DataFrame(data)
        df.insert(0, "t_bin_s", t_bins.astype(float))
        # attach coverage and fallback stats for QC consumers
        try:
            cov_pose = {k: float(np.isfinite(v).mean()) for k, v in feats.items()}
            df.attrs["coverage_pose_frame"] = cov_pose
            df.attrs["com_heading_fallback_frac"] = float(use_fallback.mean())
        except Exception:
            pass
        out[frame_name] = df

    return out


# -----------------------------
# Lags and scaling utilities
# -----------------------------

def make_lagged(kin_df: pd.DataFrame, lags_ms: Sequence[int], bin_s: float, *, drop_edge: bool = True) -> Dict[int, pd.DataFrame]:
    """Return dict lag_ms → DataFrame with lagged predictors.

    - Positive lag means *KIN leads spikes* (shift features forward).
    - Implements edge handling by NaNs (or row drop if `drop_edge`).
    - Does not modify the `t_bin_s` column.
    """
    bin_ms = int(round(bin_s * 1000))
    out: Dict[int, pd.DataFrame] = {}
    base_cols = [c for c in kin_df.columns if c != "t_bin_s"]
    for L in lags_ms:
        shift_bins = int(round(L / bin_ms))
        dfL = kin_df.copy()
        for c in base_cols:
            dfL[c] = dfL[c].shift(shift_bins)
        if drop_edge:
            dfL = dfL.iloc[max(0, shift_bins):] if shift_bins >= 0 else dfL.iloc[:shift_bins]
        out[L] = dfL
    return out


class SessionScaler:
    """Per-session standardizer (fit-on-train, apply-on-test), leakage-safe.

    Example
    -------
    scaler = SessionScaler(by="session_id")
    scaler.fit(df_train, cols=predictor_cols)
    X_test = scaler.transform(df_test)
    """

    def __init__(self, by: str = "session_id") -> None:
        self.by = by
        self.mu: Dict[Tuple, Dict[str, float]] = {}
        self.sd: Dict[Tuple, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, cols: Sequence[str]) -> "SessionScaler":
        g = df.groupby(self.by)
        self.mu = {k: {c: float(v[c].mean()) for c in cols} for k, v in g}
        self.sd = {k: {c: float(v[c].std(ddof=0) or 1.0) for c in cols} for k, v in g}
        return self

    def transform(self, df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if not self.mu:
            raise RuntimeError("Call fit() before transform().")
        if cols is None:
            cols = [c for c in df.columns if c not in (self.by, "t_bin_s")]
        out = df.copy()
        for k, sub in out.groupby(self.by).groups.items():
            idx = out.index.isin(sub)
            mu = self.mu.get(k, {})
            sd = self.sd.get(k, {})
            for c in cols:
                m = mu.get(c, float(out.loc[idx, c].mean()))
                s = sd.get(c, float(out.loc[idx, c].std(ddof=0) or 1.0))
                out.loc[idx, c] = (out.loc[idx, c] - m) / (s if s > 1e-12 else 1.0)
        return out
# -----------------------------
# Name canonicalization
# -----------------------------

def _canonicalize_joint_names(names_in: Sequence[str]) -> List[str]:
    """Map common short aliases and kpt_N format to canonical COCO-17 tokens.

    When pose files use generic `kpt_N` names (e.g., `['kpt_0', 'kpt_1', ..., 'kpt_16']`)
    and there are exactly 17 joints, this function converts them to semantic COCO-17
    names by position, enabling proper head-forward vector computation.
    """
    from freemat_data.skeletons import COCO17_NAMES

    # First check if names are kpt_N format and J=17 → convert to COCO-17 by position
    if len(names_in) == 17:
        is_kpt_format = all(
            str(n).strip().lower().startswith(('kpt_', 'kpt', 'joint_', 'joint', 'pt_'))
            or str(n).strip().isdigit()
            for n in names_in
        )
        if is_kpt_format:
            return list(COCO17_NAMES)  # Return canonical COCO-17 names by position

    # Existing alias mapping for semantic name variants
    alias = {
        'l_shoulder': 'left_shoulder', 'r_shoulder': 'right_shoulder',
        'l_elbow': 'left_elbow', 'r_elbow': 'right_elbow',
        'l_wrist': 'left_wrist', 'r_wrist': 'right_wrist',
        'l_hip': 'left_hip', 'r_hip': 'right_hip',
        'l_knee': 'left_knee', 'r_knee': 'right_knee',
        'l_ankle': 'left_ankle', 'r_ankle': 'right_ankle',
        'lefteye': 'left_eye', 'righteye': 'right_eye',
        'l_eye': 'left_eye', 'r_eye': 'right_eye',
        'l_ear': 'left_ear', 'r_ear': 'right_ear',
    }
    out: List[str] = []
    for n in (names_in or []):
        k = str(n).strip().lower()
        out.append(alias.get(k, k))
    return out


def _find_indices(names: List[str], keys: Sequence[str]) -> Optional[int]:
    lut = {n: i for i, n in enumerate(names)}
    for k in keys:
        i = lut.get(k)
        if i is not None:
            return i
    return None


def _compute_head_forward_robust(X: np.ndarray, names: Sequence[str], plane_n: Optional[np.ndarray]) -> np.ndarray:
    """Robust head-forward with fallbacks and plane-aware shoulder proxy.

    Strategy:
      1) Try freemat_data.compute_head_forward with canonicalized names.
      2) If insufficient finite frames, try local construction from
         (nose -> eye-center) or (nose -> ear-center), else shoulders × up.
      3) As a last resort, return NaNs (caller may fall back to COM heading).
    """
    N = X.shape[0]
    names_c = _canonicalize_joint_names(list(names or []))
    # 1) Primary path
    fwd = None
    if compute_head_forward is not None:
        try:
            fwd1 = compute_head_forward(X, names_c)
            if np.isfinite(fwd1).all(axis=1).mean() > 0.05:
                fwd = fwd1
        except Exception:
            fwd = None
    if fwd is not None:
        return fwd
    # 2) Manual reconstruction
    J = X.shape[1]
    def gi(key_list: Sequence[str]) -> Optional[int]:
        return _find_indices(names_c, key_list)
    idx_nose = gi(["nose"]) or gi(["snout", "tip_nose"])  # extras just in case
    idx_leye = gi(["left_eye", "lefteye", "l_eye"]) ; idx_reye = gi(["right_eye", "righteye", "r_eye"]) 
    idx_lear = gi(["left_ear", "l_ear"]) ; idx_rear = gi(["right_ear", "r_ear"]) 
    idx_lsho = gi(["left_shoulder", "l_shoulder"]) ; idx_rsho = gi(["right_shoulder", "r_shoulder"]) 
    f = np.full((N, 3), np.nan, dtype=float)
    def _safe(idx: Optional[int]) -> Optional[np.ndarray]:
        return X[:, int(idx), :] if (idx is not None and 0 <= int(idx) < J) else None
    nose = _safe(idx_nose)
    leye = _safe(idx_leye); reye = _safe(idx_reye)
    lear = _safe(idx_lear); rear = _safe(idx_rear)
    lsho = _safe(idx_lsho); rsho = _safe(idx_rsho)
    # nose -> eye-center
    if nose is not None and leye is not None and reye is not None:
        m = np.isfinite(nose).all(axis=1) & np.isfinite(leye).all(axis=1) & np.isfinite(reye).all(axis=1)
        if m.any():
            ctr = 0.5 * (leye[m] + reye[m])
            f[m] = nose[m] - ctr
    # nose -> ear-center
    if np.isnan(f).any() and nose is not None and lear is not None and rear is not None:
        m = (~np.isfinite(f).all(axis=1)) & np.isfinite(nose).all(axis=1) & np.isfinite(lear).all(axis=1) & np.isfinite(rear).all(axis=1)
        if m.any():
            ctr = 0.5 * (lear[m] + rear[m])
            f[m] = nose[m] - ctr
    # shoulders × up
    if np.isnan(f).any() and lsho is not None and rsho is not None:
        m = (~np.isfinite(f).all(axis=1)) & np.isfinite(lsho).all(axis=1) & np.isfinite(rsho).all(axis=1)
        if m.any():
            if plane_n is not None:
                up = _unit(np.asarray(plane_n, float).reshape(1, 3))[0]
            else:
                up = np.array([0.0, 0.0, 1.0], dtype=float)
            shoulder = rsho[m] - lsho[m]
            f[m] = np.cross(up.reshape(1, 3), shoulder)
    # Normalize
    nrm = np.linalg.norm(f, axis=1)
    good = np.isfinite(f).all(axis=1) & (nrm > 1e-12)
    f[good] = (f[good].T / nrm[good]).T
    f[~good] = np.nan
    return f
