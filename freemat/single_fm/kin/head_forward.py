"""Head-forward vector computation strategies for single-FM.

Provides multiple strategies for computing the head-forward direction vector
from COCO-17 keypoints, with support for confidence gating and fallbacks.

Strategies:
    1. eye_midpoint_to_nose: Primary (eye midpoint → nose)
    2. ear_midpoint_to_nose: Fallback (ear midpoint → nose)
    3. confidence_weighted: Weight eye positions by confidence scores
    4. cascade: Try eyes → ears → NaN

Usage:
    >>> from freemat.single_fm.kin.head_forward import compute_head_forward_strategies
    >>> head_fwd = compute_head_forward_strategies(
    ...     X, joint_names, strategy='cascade', confidence=C
    ... )
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

# COCO-17 joint indices
COCO17_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

HeadForwardStrategy = Literal[
    'eye_midpoint_to_nose',
    'ear_midpoint_to_nose',
    'confidence_weighted',
    'cascade'
]


def _is_semantic_name(name: str) -> bool:
    """Check if a joint name is semantic (not kpt_N or similar)."""
    name = str(name).strip().lower()
    # Check for non-semantic patterns
    if name.startswith(('kpt_', 'kpt', 'joint_', 'joint', 'pt_', 'pt', 'j_', 'j')):
        return False
    try:
        int(name)  # Pure number like "0", "1"
        return False
    except ValueError:
        pass
    return True


def _get_joint_index(
    joint_name: str,
    joint_names: Optional[Sequence[str]],
    n_joints: int,
) -> Optional[int]:
    """Get joint index by name or default COCO-17 order.

    Args:
        joint_name: Name of the joint to find.
        joint_names: Optional list of joint names.
        n_joints: Number of joints in the pose array.

    Returns:
        Joint index or None if not found.
    """
    # Check if joint_names contains semantic names
    if joint_names:
        names_lower = [str(n).strip().lower() for n in joint_names]
        has_semantic_names = any(_is_semantic_name(n) for n in names_lower)

        if has_semantic_names:
            # Try to find by name first
            target = joint_name.lower()
            if target in names_lower:
                return names_lower.index(target)
            # If semantic names exist but don't include this joint,
            # fall back to COCO indices if available
            if n_joints >= 17:
                return COCO17_INDICES.get(target)
            return None
        elif n_joints >= 17:
            # Non-semantic names (kpt_0, etc.) - use COCO-17 indices
            return COCO17_INDICES.get(joint_name.lower())
    elif n_joints >= 17:
        # No names provided - assume COCO-17 order
        return COCO17_INDICES.get(joint_name.lower())
    return None


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length, keeping NaN for invalid frames."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 1e-12, norms, np.nan)
    return vectors / norms


def _eye_midpoint_to_nose(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]],
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Compute head-forward as eye midpoint → nose direction.

    Args:
        X: Pose array (T, J, 3).
        joint_names: Optional joint names.
        confidence: Optional confidence array (T, J).
        conf_threshold: Minimum confidence to use a joint.

    Returns:
        Head-forward vectors (T, 3), unit normalized.
    """
    T, J, _ = X.shape

    idx_nose = _get_joint_index('nose', joint_names, J)
    idx_left_eye = _get_joint_index('left_eye', joint_names, J)
    idx_right_eye = _get_joint_index('right_eye', joint_names, J)

    if idx_nose is None or idx_left_eye is None or idx_right_eye is None:
        return np.full((T, 3), np.nan, dtype=np.float64)

    nose = X[:, idx_nose, :]
    left_eye = X[:, idx_left_eye, :]
    right_eye = X[:, idx_right_eye, :]

    # Check validity
    valid_nose = np.isfinite(nose).all(axis=1)
    valid_left_eye = np.isfinite(left_eye).all(axis=1)
    valid_right_eye = np.isfinite(right_eye).all(axis=1)

    # Apply confidence threshold if provided
    if confidence is not None and conf_threshold > 0:
        valid_nose &= confidence[:, idx_nose] >= conf_threshold
        valid_left_eye &= confidence[:, idx_left_eye] >= conf_threshold
        valid_right_eye &= confidence[:, idx_right_eye] >= conf_threshold

    valid = valid_nose & valid_left_eye & valid_right_eye

    result = np.full((T, 3), np.nan, dtype=np.float64)
    if valid.any():
        eye_midpoint = 0.5 * (left_eye[valid] + right_eye[valid])
        result[valid] = nose[valid] - eye_midpoint

    return _normalize_vectors(result)


def _ear_midpoint_to_nose(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]],
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Compute head-forward as ear midpoint → nose direction.

    Args:
        X: Pose array (T, J, 3).
        joint_names: Optional joint names.
        confidence: Optional confidence array (T, J).
        conf_threshold: Minimum confidence to use a joint.

    Returns:
        Head-forward vectors (T, 3), unit normalized.
    """
    T, J, _ = X.shape

    idx_nose = _get_joint_index('nose', joint_names, J)
    idx_left_ear = _get_joint_index('left_ear', joint_names, J)
    idx_right_ear = _get_joint_index('right_ear', joint_names, J)

    if idx_nose is None or idx_left_ear is None or idx_right_ear is None:
        return np.full((T, 3), np.nan, dtype=np.float64)

    nose = X[:, idx_nose, :]
    left_ear = X[:, idx_left_ear, :]
    right_ear = X[:, idx_right_ear, :]

    # Check validity
    valid_nose = np.isfinite(nose).all(axis=1)
    valid_left_ear = np.isfinite(left_ear).all(axis=1)
    valid_right_ear = np.isfinite(right_ear).all(axis=1)

    # Apply confidence threshold if provided
    if confidence is not None and conf_threshold > 0:
        valid_nose &= confidence[:, idx_nose] >= conf_threshold
        valid_left_ear &= confidence[:, idx_left_ear] >= conf_threshold
        valid_right_ear &= confidence[:, idx_right_ear] >= conf_threshold

    valid = valid_nose & valid_left_ear & valid_right_ear

    result = np.full((T, 3), np.nan, dtype=np.float64)
    if valid.any():
        ear_midpoint = 0.5 * (left_ear[valid] + right_ear[valid])
        result[valid] = nose[valid] - ear_midpoint

    return _normalize_vectors(result)


def _confidence_weighted(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]],
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Compute head-forward with confidence-weighted eye positions.

    Args:
        X: Pose array (T, J, 3).
        joint_names: Optional joint names.
        confidence: Confidence array (T, J). Required for this strategy.
        conf_threshold: Minimum confidence to use a joint.

    Returns:
        Head-forward vectors (T, 3), unit normalized. Frames with zero
        total confidence are marked as NaN.
    """
    T, J, _ = X.shape

    if confidence is None:
        # Fall back to unweighted if no confidence provided
        return _eye_midpoint_to_nose(X, joint_names, confidence, conf_threshold)

    idx_nose = _get_joint_index('nose', joint_names, J)
    idx_left_eye = _get_joint_index('left_eye', joint_names, J)
    idx_right_eye = _get_joint_index('right_eye', joint_names, J)

    if idx_nose is None or idx_left_eye is None or idx_right_eye is None:
        return np.full((T, 3), np.nan, dtype=np.float64)

    nose = X[:, idx_nose, :]
    left_eye = X[:, idx_left_eye, :]
    right_eye = X[:, idx_right_eye, :]

    conf_left = confidence[:, idx_left_eye]
    conf_right = confidence[:, idx_right_eye]

    # Check validity - position must be finite AND confidence above threshold
    valid_nose = np.isfinite(nose).all(axis=1)
    valid_left_eye = np.isfinite(left_eye).all(axis=1) & (conf_left >= conf_threshold)
    valid_right_eye = np.isfinite(right_eye).all(axis=1) & (conf_right >= conf_threshold)

    # Need at least one valid eye and the nose
    valid = valid_nose & (valid_left_eye | valid_right_eye)

    result = np.full((T, 3), np.nan, dtype=np.float64)

    for t in range(T):
        if not valid[t]:
            continue

        # Compute weighted eye midpoint
        weights = []
        positions = []

        if valid_left_eye[t]:
            weights.append(conf_left[t])
            positions.append(left_eye[t])
        if valid_right_eye[t]:
            weights.append(conf_right[t])
            positions.append(right_eye[t])

        if weights:
            weights = np.array(weights)
            weight_sum = weights.sum()
            # Guard against zero weight sum (both confidences are 0)
            if weight_sum < 1e-12:
                # Mark as invalid - result[t] remains NaN
                continue
            weights = weights / weight_sum
            eye_midpoint = np.average(positions, axis=0, weights=weights)
            result[t] = nose[t] - eye_midpoint

    return _normalize_vectors(result)


def _cascade(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]],
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Compute head-forward with cascade: eyes → ears → NaN.

    Args:
        X: Pose array (T, J, 3).
        joint_names: Optional joint names.
        confidence: Optional confidence array (T, J).
        conf_threshold: Minimum confidence to use a joint.

    Returns:
        Head-forward vectors (T, 3), unit normalized.
    """
    T = X.shape[0]

    # Try eyes first
    result = _eye_midpoint_to_nose(X, joint_names, confidence, conf_threshold)

    # Fill NaN frames with ears
    nan_frames = ~np.isfinite(result).all(axis=1)
    if nan_frames.any():
        ear_result = _ear_midpoint_to_nose(X, joint_names, confidence, conf_threshold)
        result[nan_frames] = ear_result[nan_frames]

    return result


def compute_head_forward_strategies(
    X: np.ndarray,
    joint_names: Optional[Sequence[str]] = None,
    strategy: HeadForwardStrategy = 'cascade',
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Compute head-forward vectors using the specified strategy.

    Args:
        X: Pose array (T, J, 3) in meters.
        joint_names: Optional list of joint names (COCO-17 order assumed if None).
        strategy: Head-forward computation strategy.
        confidence: Optional confidence array (T, J).
        conf_threshold: Minimum confidence to use a joint.

    Returns:
        Head-forward unit vectors (T, 3). NaN for invalid frames.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"X must have shape (T, J, 3), got {X.shape}")

    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64)
        if confidence.shape[:2] != X.shape[:2]:
            raise ValueError(
                f"confidence shape {confidence.shape} must match X shape {X.shape[:2]}"
            )

    strategies = {
        'eye_midpoint_to_nose': _eye_midpoint_to_nose,
        'ear_midpoint_to_nose': _ear_midpoint_to_nose,
        'confidence_weighted': _confidence_weighted,
        'cascade': _cascade,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(strategies.keys())}")

    return strategies[strategy](X, joint_names, confidence, conf_threshold)


def project_to_floor(
    vectors: np.ndarray,
    floor_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Project vectors onto the floor plane and normalize.

    Args:
        vectors: Vectors to project (T, 3).
        floor_normal: Normal vector of the floor plane.

    Returns:
        Projected and normalized vectors (T, 3).
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    floor_normal = np.asarray(floor_normal, dtype=np.float64)
    floor_normal = floor_normal / (np.linalg.norm(floor_normal) + 1e-12)

    # Project: v_proj = v - (v · n) * n
    dot = (vectors @ floor_normal.reshape(3, 1)).reshape(-1, 1)
    projected = vectors - dot * floor_normal.reshape(1, 3)

    return _normalize_vectors(projected)


def compute_yaw_from_forward(
    head_forward: np.ndarray,
    floor_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Compute yaw angle (allocentric) from head-forward vectors.

    Args:
        head_forward: Head-forward unit vectors (T, 3).
        floor_normal: Floor plane normal (default Z-up).

    Returns:
        Yaw angles in radians (T,), in range [-π, π].
    """
    # Project to floor plane
    projected = project_to_floor(head_forward, floor_normal)

    # Yaw = atan2(y, x) for Z-up coordinate system
    return np.arctan2(projected[:, 1], projected[:, 0])


__all__ = [
    "HeadForwardStrategy",
    "compute_head_forward_strategies",
    "project_to_floor",
    "compute_yaw_from_forward",
    "COCO17_INDICES",
]
