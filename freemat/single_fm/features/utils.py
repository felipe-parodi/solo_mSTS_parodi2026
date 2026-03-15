# freemat/single_fm/features/utils.py
"""Low-level utility functions for canonical feature computation.

Provides vector math helpers, smoothing, axis helpers, and viser-compatible
gaze computation functions used by canonical.py.

The viser gaze functions are inlined (copied from
freemat_vid_inference/triangulation/viewer/gaze_renderer.py) to keep
canonical.py self-contained and avoid import chain issues.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# Vector / Math Utilities
# =============================================================================

def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize to unit vector, handling zero-length."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project vector(s) v onto plane with normal n."""
    n = np.asarray(n, dtype=float).reshape(1, 3)
    return v - (v @ n.T) * n / (np.dot(n.flatten(), n.flatten()) + 1e-12)


def _central_diff(X: np.ndarray, dt: float) -> np.ndarray:
    """Central difference with linear endpoints."""
    X = np.asarray(X, dtype=float)
    d = np.empty_like(X)
    d[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    d[0] = (X[1] - X[0]) / dt
    d[-1] = (X[-1] - X[-2]) / dt
    return d


def _nan_moving_average(x: np.ndarray, win_s: float, fps: float) -> np.ndarray:
    """NaN-aware moving average."""
    if win_s <= 0 or fps <= 0:
        return x
    w = max(1, int(round(win_s * fps)))
    if w <= 1:
        return x
    k = np.ones(w, dtype=float)
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


# =============================================================================
# Axis Helpers
# =============================================================================

def _get_vertical_axis(plane_dict: Dict) -> int:
    """Get vertical axis index from plane normal."""
    a = abs(float(plane_dict.get("a", 0.0)))
    b = abs(float(plane_dict.get("b", 0.0)))
    c = abs(float(plane_dict.get("c", 1.0)))
    if a >= b and a >= c:
        return 0
    elif b >= a and b >= c:
        return 1
    return 2


def _get_horiz_indices(vertical_idx: int) -> tuple:
    """Get horizontal plane indices."""
    if vertical_idx == 0:
        return (1, 2)
    elif vertical_idx == 1:
        return (0, 2)
    return (0, 1)


# =============================================================================
# Viser Gaze Functions (inlined to avoid import chain issues)
# =============================================================================
# These are copied from freemat_vid_inference/triangulation/viewer/gaze_renderer.py
# to make canonical.py self-contained. They are the source of truth for VIS features.


def _viser_shoulder_fallback(
    pose: np.ndarray,
    left_shoulder_idx: Optional[int],
    right_shoulder_idx: Optional[int],
) -> np.ndarray:
    """Compute forward direction from shoulder perpendicular.

    Args:
        pose: Single frame pose array (J, 3).
        left_shoulder_idx: Index of left shoulder joint.
        right_shoulder_idx: Index of right shoulder joint.

    Returns:
        Unit forward direction vector (3,).
    """
    if left_shoulder_idx is not None and right_shoulder_idx is not None:
        ls = pose[left_shoulder_idx]
        rs = pose[right_shoulder_idx]
        if np.all(np.isfinite(ls)) and np.all(np.isfinite(rs)):
            shoulder_vec = rs - ls
            forward = np.array([-shoulder_vec[1], shoulder_vec[0], 0.0])
            norm = np.linalg.norm(forward)
            if norm > 1e-6:
                return forward / norm
    return np.array([1.0, 0.0, 0.0])


def _viser_compute_head_position(
    pose: np.ndarray,
    joint_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute head position from pose keypoints.

    Uses nose keypoint if available, otherwise average of head/upper body joints.

    Args:
        pose: Single frame pose array (J, 3).
        joint_names: List of joint names.

    Returns:
        Head position (3,).
    """
    J = pose.shape[0]

    # Try to find nose or head joint
    if joint_names:
        lnames = [str(n or "").strip().lower() for n in joint_names]
        for i, nm in enumerate(lnames):
            if nm == "nose" and np.all(np.isfinite(pose[i])):
                return pose[i].copy()

        # Try head center (average of ear keypoints)
        ears = []
        for i, nm in enumerate(lnames):
            if "ear" in nm and np.all(np.isfinite(pose[i])):
                ears.append(pose[i])
        if len(ears) >= 2:
            return np.mean(ears, axis=0)

    # Fallback: use first valid joint (usually nose in COCO-17)
    if J >= 1 and np.all(np.isfinite(pose[0])):
        return pose[0].copy()

    # Last resort: centroid of valid joints
    valid = np.all(np.isfinite(pose), axis=1)
    if valid.any():
        return np.mean(pose[valid], axis=0)

    return np.array([np.nan, np.nan, np.nan])


def _viser_compute_gaze_direction(
    pose: np.ndarray,
    joint_names: Optional[List[str]] = None,
    eye_weight: float = 0.35,
) -> np.ndarray:
    """Compute gaze/head-forward direction from pose keypoints.

    Uses weighted blend of eye_midpoint and ear_midpoint as base,
    with nose as target. Weights can be tuned to adjust pitch.

    Higher eye_weight -> more downward pitch (nose below eyes)
    Lower eye_weight -> more horizontal (ears at similar height to nose)

    Args:
        pose: Single frame pose array (J, 3).
        joint_names: List of joint names.
        eye_weight: Weight for eyes in blend (0-1). Default 0.35 favors ears
            with eye contribution to balance pitch.

    Returns:
        Unit gaze direction vector (3,).
    """
    J = pose.shape[0]

    # Find joint indices
    nose_idx = None
    left_eye_idx = None
    right_eye_idx = None
    left_ear_idx = None
    right_ear_idx = None
    left_shoulder_idx = None
    right_shoulder_idx = None

    if joint_names:
        lnames = [str(n or "").strip().lower() for n in joint_names]
        for i, nm in enumerate(lnames):
            if nm == "nose":
                nose_idx = i
            elif nm == "left_eye":
                left_eye_idx = i
            elif nm == "right_eye":
                right_eye_idx = i
            elif nm == "left_ear":
                left_ear_idx = i
            elif nm == "right_ear":
                right_ear_idx = i
            elif nm == "left_shoulder":
                left_shoulder_idx = i
            elif nm == "right_shoulder":
                right_shoulder_idx = i

    # COCO-17 fallback indices
    if J == 17:
        if nose_idx is None:
            nose_idx = 0
        if left_eye_idx is None:
            left_eye_idx = 1
        if right_eye_idx is None:
            right_eye_idx = 2
        if left_ear_idx is None:
            left_ear_idx = 3
        if right_ear_idx is None:
            right_ear_idx = 4
        if left_shoulder_idx is None:
            left_shoulder_idx = 5
        if right_shoulder_idx is None:
            right_shoulder_idx = 6

    # Get nose position
    nose = None
    if nose_idx is not None and np.all(np.isfinite(pose[nose_idx])):
        nose = pose[nose_idx]

    if nose is None:
        # No nose - fall back to shoulder-perpendicular
        return _viser_shoulder_fallback(pose, left_shoulder_idx, right_shoulder_idx)

    # Get eye midpoint
    eye_midpoint = None
    if left_eye_idx is not None and right_eye_idx is not None:
        left_eye = pose[left_eye_idx]
        right_eye = pose[right_eye_idx]
        if np.all(np.isfinite(left_eye)) and np.all(np.isfinite(right_eye)):
            eye_midpoint = 0.5 * (left_eye + right_eye)

    # Get ear midpoint
    ear_midpoint = None
    if left_ear_idx is not None and right_ear_idx is not None:
        left_ear = pose[left_ear_idx]
        right_ear = pose[right_ear_idx]
        if np.all(np.isfinite(left_ear)) and np.all(np.isfinite(right_ear)):
            ear_midpoint = 0.5 * (left_ear + right_ear)

    # Compute weighted base
    base = None
    if eye_midpoint is not None and ear_midpoint is not None:
        # Weighted blend: ears help "pull up" the gaze slightly
        base = eye_weight * eye_midpoint + (1 - eye_weight) * ear_midpoint
    elif eye_midpoint is not None:
        base = eye_midpoint
    elif ear_midpoint is not None:
        base = ear_midpoint

    # Compute direction from base to nose
    if base is not None:
        direction = nose - base
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            return direction / norm

    # Fallback: shoulder-perpendicular
    return _viser_shoulder_fallback(pose, left_shoulder_idx, right_shoulder_idx)


__all__ = [
    "_unit",
    "_project_to_plane",
    "_central_diff",
    "_nan_moving_average",
    "_get_vertical_axis",
    "_get_horiz_indices",
    "_viser_shoulder_fallback",
    "_viser_compute_head_position",
    "_viser_compute_gaze_direction",
]
