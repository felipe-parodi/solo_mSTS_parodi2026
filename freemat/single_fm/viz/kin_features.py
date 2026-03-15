# freemat/single_fm/viz/kin_features.py
"""Backward-compatibility shim — canonical location is features/kin_features.py."""

from freemat.single_fm.features.kin_features import (  # noqa: F401
    KinParams,
    SessionScaler,
    compute_kin_features,
    make_lagged,
    _canonicalize_joint_names,
    _compute_head_forward_robust,
    _find_indices,
    _unit,
    _project_to_plane,
    _central_diff,
    _nan_moving_average,
    _unwrap_phase,
    _wrap_pi,
    _torso_forward_from_skeleton,
    _inplane_basis_from_normal,
    _frame_axes_from_forward,
    _R_F_from_axes,
    _yaw_from_ex,
)

__all__ = [
    "KinParams",
    "SessionScaler",
    "compute_kin_features",
    "make_lagged",
    "_canonicalize_joint_names",
    "_compute_head_forward_robust",
    "_find_indices",
]
