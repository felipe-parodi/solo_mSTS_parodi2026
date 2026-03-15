"""Pose target construction for neural decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from freemat_data.loaders.pose3d import load_pose3d
from freemat_data.transforms.pose import frame_valid_fraction


@dataclass
class PoseTargets:
    """Container for synchronized pose decoding targets.

    Attributes:
        t_centers: Time bin centers in seconds, shape ``(T,)``.
        joints_flat: Root-centered joint positions, shape ``(T, D_pos)``.
        velocities_flat: Joint velocities in world frame, shape ``(T, D_vel)``.
        joint_names: Compact joint names in order (HEAD + 12 body joints).
        root_xyz: Hips-midpoint world coordinates aligned to *t_centers*,
            shape ``(T, 3)``.
    """

    t_centers: np.ndarray  # (T,)
    joints_flat: np.ndarray  # (T, D_pos) positions, root-centered
    velocities_flat: np.ndarray  # (T, D_vel) velocities (world)
    joint_names: List[str]  # compact names in order (HEAD + 12 body joints)
    root_xyz: np.ndarray  # (T, 3) hips-midpoint world coords aligned to t_centers


def _smooth_positions_savgol(X: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter  # type: ignore
        w = int(window)
        if w % 2 == 0:
            w += 1
        Xr = X.reshape(-1, 3)
        out = np.empty_like(Xr)
        for d in range(3):
            out[:, d] = savgol_filter(Xr[:, d], window_length=min(w, max(5, (Xr.shape[0] // 20) * 2 + 1)), polyorder=int(poly), mode="interp")
        return out.reshape(X.shape)
    except (ImportError, ValueError, TypeError):
        # Fallback: moving average
        k = max(3, int(window) | 1)
        Xr = X.reshape(-1, 3)
        out = np.empty_like(Xr)
        for d in range(3):
            v = Xr[:, d]
            ker = np.ones(k) / k
            out[:, d] = np.convolve(v, ker, mode="same")
        return out.reshape(X.shape)


def _root_center(X: np.ndarray, root_idx: int) -> np.ndarray:
    root = X[:, root_idx : root_idx + 1, :]
    return X - root


def _central_diff(X: np.ndarray, dt: float) -> np.ndarray:
    Y = np.empty_like(X)
    Y[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    Y[0] = (X[1] - X[0]) / dt
    Y[-1] = (X[-1] - X[-2]) / dt
    return Y


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def _interp_nan_1d(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate NaN gaps up to max_gap samples (inside only)."""
    try:
        s = pd.Series(y)
        yi = s.interpolate(method="linear", limit=max_gap, limit_area="inside").to_numpy()
        return yi
    except (ValueError, TypeError):
        # Simple manual linear interpolation inside finite spans
        x = np.arange(y.size)
        m = np.isfinite(y)
        if m.sum() < 2:
            return y.copy()
        yi = np.interp(x, x[m], y[m])
        # Respect max_gap: restore NaN in long NaN runs
        if max_gap >= 0:
            isn = ~m
            if isn.any():
                # Find run lengths
                run = 0
                for i in range(len(isn)):
                    if isn[i]:
                        run += 1
                    else:
                        if run > max_gap:
                            yi[i - run : i] = np.nan
                        run = 0
                if run > max_gap:
                    yi[len(isn) - run :] = np.nan
        return yi


def _compute_head_summary(Xs: np.ndarray) -> np.ndarray:
    """Compute a summary head keypoint from available nose/eyes/ears.

    Fallbacks: midpoint of shoulders, then midpoint of hips, else NaN.
    Xs: (T_pose, 17, 3)
    Returns (T_pose, 3) with NaNs when not constructible.
    """
    T_pose = Xs.shape[0]
    head_idx = [0, 1, 2, 3, 4]  # nose, eyes, ears
    L_sh, R_sh = 5, 6
    L_hip, R_hip = 11, 12
    head = np.full((T_pose, 3), np.nan, float)
    valid = np.isfinite(Xs).all(axis=2)
    have_head = valid[:, head_idx].any(axis=1)
    # Use mean of available head components
    if np.any(have_head):
        vals = Xs[:, head_idx, :]
        m = valid[:, head_idx]
        sums = np.nansum(np.where(m[..., None], vals, 0.0), axis=1)
        cnts = m.sum(axis=1).clip(min=1).astype(float)
        head[have_head] = sums[have_head] / cnts[have_head, None]
    # Fallbacks where head is still NaN
    still_nan = ~np.isfinite(head).all(axis=1)
    sh_ok = valid[:, L_sh] & valid[:, R_sh]
    hip_ok = valid[:, L_hip] & valid[:, R_hip]
    use_sh = still_nan & sh_ok
    head[use_sh] = _midpoint(Xs[use_sh, L_sh, :], Xs[use_sh, R_sh, :])
    still_nan = ~np.isfinite(head).all(axis=1)
    use_hip = still_nan & hip_ok
    head[use_hip] = _midpoint(Xs[use_hip, L_hip, :], Xs[use_hip, R_hip, :])
    return head


def _apply_segment_mask(
    m_body: np.ndarray,
    m_head: np.ndarray,
    t_centers: np.ndarray,
    *,
    seg_buffer_s: float,
    min_seg_s: float,
) -> np.ndarray:
    """Build a boolean keep mask over t_centers based on segment rules."""
    m = (m_body & m_head).astype(bool)
    if m.size == 0:
        return m
    dt = float(np.median(np.diff(t_centers))) if t_centers.size > 1 else 1.0
    # Buffer invalids
    b = int(round(max(seg_buffer_s, 0.0) / max(dt, 1e-9)))
    if b > 0:
        inv = (~m).astype(int)
        ker = np.ones(2 * b + 1, dtype=int)
        invb = np.convolve(inv, ker, mode="same") > 0
        m = ~invb
    # Enforce min segment length
    Lmin = int(round(max(min_seg_s, 0.0) / max(dt, 1e-9)))
    if Lmin > 1:
        m2 = m.copy()
        i = 0
        n = m.size
        while i < n:
            if m[i]:
                j = i
                while j < n and m[j]:
                    j += 1
                if (j - i) < Lmin:
                    m2[i:j] = False
                i = j
            else:
                i += 1
        m = m2
    return m


def build_pose_targets(
    npz_path: str,
    t_centers: np.ndarray,
    *,
    k_pca: int,
    interp_head_s: float,
    seg_buffer_s: float,
    min_seg_s: float,
    vel_root_centered: bool,
    segment_mode: str = "head_body_segments",
) -> Tuple[PoseTargets, Dict[str, object]]:
    """Load pose NPZ and produce synchronized targets.

    Loads 3-D pose data, smooths positions, computes velocities, builds a
    composite HEAD joint, and aligns everything to the supplied time-bin
    centers.  Two segmentation strategies are supported.

    Args:
        npz_path: Path to the pose NPZ file (COCO-17 layout).
        t_centers: Time bin centers in seconds, shape ``(T,)``.
        k_pca: Number of PCA components (reserved for downstream use).
        interp_head_s: Maximum gap duration (seconds) for head-keypoint
            NaN interpolation.
        seg_buffer_s: Buffer (seconds) around invalid frames to exclude.
        min_seg_s: Minimum valid-segment duration (seconds) to keep.
        vel_root_centered: If ``True``, subtract root velocity from joint
            velocities.
        segment_mode: Segmentation strategy.  ``"head_body_segments"``
            (default) uses HEAD + 12 body joints with buffered segments.
            ``"longest17"`` uses the longest contiguous segment where all
            17 joints are valid, with no interpolation.

    Returns:
        Tuple of ``(PoseTargets, meta)`` where *meta* is a dict with
        diagnostics such as joint counts, FPS, validity fraction, and
        the keep mask.

    Raises:
        ValueError: If no valid segments remain after masking.
    """
    pose = load_pose3d(npz_path)
    X = np.asarray(pose.X, dtype=float)
    T_pose, J, _ = X.shape
    names = pose.joint_names or [f"J{i}" for i in range(J)]

    # Smooth all joints and derive velocities/root
    Xs = _smooth_positions_savgol(X, window=7, poly=2)
    dt_pose = 1.0 / max(float(pose.fps or 30.0), 1e-6)
    V = _central_diff(Xs, dt_pose)
    hips_ok = np.isfinite(Xs[:, 11, :]).all(axis=1) & np.isfinite(Xs[:, 12, :]).all(axis=1)
    root_pose = np.full((T_pose, 3), np.nan, float)
    root_pose[hips_ok] = _midpoint(Xs[hips_ok, 11, :], Xs[hips_ok, 12, :])
    V_root = _central_diff(root_pose, dt_pose)

    # Map to FR bins by nearest frame index (avoids systematic early bias)
    idx = np.clip(np.round(t_centers / dt_pose).astype(int), 0, T_pose - 1)

    if str(segment_mode) == "longest17":
        valid17 = np.isfinite(Xs).all(axis=(1, 2))
        # Longest True run
        best_len = 0
        best_i = 0
        i = 0
        while i < valid17.size:
            if valid17[i]:
                j = i
                while j < valid17.size and valid17[j]:
                    j += 1
                if (j - i) > best_len:
                    best_len = j - i
                    best_i = i
                i = j
            else:
                i += 1
        if best_len <= 1:
            raise ValueError("No contiguous segment with all 17 joints valid.")
        f0, f1 = best_i, best_i + best_len - 1
        keep_mask = (idx >= f0) & (idx <= f1)
        if not np.any(keep_mask):
            raise ValueError("No FR bins fall within the longest-17 segment.")
        fid = idx[keep_mask]
        X_sel = Xs[fid]
        V_sel = V[fid]
        root_sel = root_pose[fid]
        V_root_sel = V_root[fid]
        # Root-centered positions; velocities optionally root-centered
        X_rc = X_sel - root_sel[:, None, :]
        if vel_root_centered:
            V_sel = V_sel - V_root_sel[:, None, :]
        joints_flat = X_rc.reshape(X_rc.shape[0], -1)
        velocities_flat = V_sel.reshape(V_sel.shape[0], -1)
        compact_names = names
        t_keep = t_centers[keep_mask]
    else:
        # Head summary from smoothed joints
        head_all = _compute_head_summary(Xs)
        max_gap_frames = int(round(float(interp_head_s) / dt_pose))
        for d in range(3):
            head_all[:, d] = _interp_nan_1d(head_all[:, d], max_gap_frames)
        head_all = _smooth_positions_savgol(head_all.reshape(T_pose, 1, 3), window=7, poly=2).reshape(T_pose, 3)
        V_head = _central_diff(head_all, dt_pose)
        BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        compact_names = [
            "HEAD",
            names[5] if len(names) > 5 else "LShoulder",
            names[6] if len(names) > 6 else "RShoulder",
            names[7] if len(names) > 7 else "LElbow",
            names[8] if len(names) > 8 else "RElbow",
            names[9] if len(names) > 9 else "LWrist",
            names[10] if len(names) > 10 else "RWrist",
            names[11] if len(names) > 11 else "LHip",
            names[12] if len(names) > 12 else "RHip",
            names[13] if len(names) > 13 else "LKnee",
            names[14] if len(names) > 14 else "RKnee",
            names[15] if len(names) > 15 else "LAnkle",
            names[16] if len(names) > 16 else "RAnkle",
        ]
        X_body_sel = Xs[idx][:, BODY, :]
        V_body_sel = V[idx][:, BODY, :]
        head_sel = head_all[idx]
        V_head_sel = V_head[idx]
        root_sel = root_pose[idx]
        V_root_sel = V_root[idx]
        m_body = np.isfinite(X_body_sel).all(axis=(1, 2))
        m_head = np.isfinite(head_sel).all(axis=1)
        keep_mask = _apply_segment_mask(m_body, m_head, t_centers, seg_buffer_s=seg_buffer_s, min_seg_s=min_seg_s)
        if not np.any(keep_mask):
            raise ValueError("No valid segments found after applying head+body mask. Consider relaxing seg parameters.")
        X_body_sel = X_body_sel[keep_mask]
        V_body_sel = V_body_sel[keep_mask]
        head_sel = head_sel[keep_mask]
        V_head_sel = V_head_sel[keep_mask]
        root_sel = root_sel[keep_mask]
        V_root_sel = V_root_sel[keep_mask]
        t_keep = t_centers[keep_mask]
        X_comp = np.concatenate([head_sel[:, None, :], X_body_sel], axis=1)
        X_rc = X_comp - root_sel[:, None, :]
        V_comp = np.concatenate([V_head_sel[:, None, :], V_body_sel], axis=1)
        if vel_root_centered:
            V_comp = V_comp - V_root_sel[:, None, :]
        joints_flat = X_rc.reshape(X_rc.shape[0], -1)
        velocities_flat = V_comp.reshape(V_comp.shape[0], -1)

    # Validity fraction (pose-only; for reporting)
    if pose.C is not None and pose.C.shape[:2] == (T_pose, J):
        vf = frame_valid_fraction(X, pose.C, conf_thresh=0.5)
        vf_mean = float(np.mean(vf))
    else:
        vf_mean = float(np.isfinite(X).all(axis=2).mean())

    targets = PoseTargets(
        t_centers=t_keep,
        joints_flat=joints_flat,
        velocities_flat=velocities_flat,
        joint_names=compact_names,
        root_xyz=root_sel,
    )
    meta = {
        "J_all": int(J),
        "joint_names_all": names,
        "compact_joint_names": compact_names,
        "fps": float(pose.fps or 30.0),
        "valid_frac_mean": vf_mean,
        "keep_mask": keep_mask.tolist(),
        "vel_root_centered": bool(vel_root_centered),
        "segment_mode": str(segment_mode),
    }
    return targets, meta
