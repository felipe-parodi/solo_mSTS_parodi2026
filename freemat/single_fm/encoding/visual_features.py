"""Feature column definitions for Figure 2B encoding analysis.

Defines standard column names for VIS (visual proxy) and KIN (kinematic) features
used in the unique variance analysis. Feature computation is handled by the
canonical module: `freemat.single_fm.features.canonical`.

VIS features (head-derived visual proxies):
- gaze_x, gaze_y, gaze_z: Head-forward unit vector components
- d_wall_horiz: Horizontal wall distance from HEAD along GAZE direction
- vis_motion: Looming proxy = v_parallel_gaze / (d_wall_horiz + eps)

BODY_KIN features (body kinematics):
- v_parallel: Forward/backward velocity along body heading
- v_perp: Lateral velocity perpendicular to body heading
- speed: 3D speed magnitude

HEAD_KIN features (head kinematics):
- yaw_rate: Head yaw angular velocity
- pitch_rate: Head pitch angular velocity
- roll_rate: Head roll angular velocity
- delta_yaw: Head-torso yaw difference

Example:
    >>> from freemat.single_fm.encoding.visual_features import (
    ...     VIS_COLS, KIN_COLS, BODY_KIN_COLS, HEAD_KIN_COLS,
    ... )
    >>> # Use SingleFMSessionLoader for feature computation:
    >>> from freemat.single_fm.data import SingleFMSessionLoader
    >>> loader = SingleFMSessionLoader(session_yaml)
    >>> merged = loader.get_aligned_data(bin_s=0.5)
"""

from __future__ import annotations

from typing import Dict, List

# Column definitions (full 3D)
# d_wall_horiz: horizontal wall distance from HEAD along GAZE (matches viser)
# vis_motion: v_parallel_gaze / (d_wall_horiz + eps) — looming in visual field
#   v_parallel_gaze = velocity projected onto gaze direction (NOT body-heading v_parallel)
# NOTE: VIS features use head_pos + gaze_dir (matches viser), not COM + body_heading
VIS_COLS = ["gaze_x", "gaze_y", "gaze_z", "d_wall_horiz", "vis_motion"]

# KIN columns split into BODY and HEAD groups (Fig 2B breakdown)
# NOTE: v_z removed for floor-focused analysis (Fig 2B)
BODY_KIN_COLS = ["v_parallel", "v_perp", "speed"]
HEAD_KIN_COLS = ["yaw_rate", "pitch_rate", "roll_rate", "delta_yaw"]
KIN_COLS = BODY_KIN_COLS + HEAD_KIN_COLS  # Combined for backward compat

# 3D position features (COM position, floor-relative z)
# POS features are linear and unlagged (no temporal kernel expansion).
POS_COLS = ["com_x", "com_y", "com_z"]

# 2D floor-plane feature definitions (same features, 2D interpretation)
VIS_COLS_2D = ["gaze_x", "gaze_y", "d_wall_horiz", "vis_motion"]
KIN_COLS_2D = BODY_KIN_COLS + HEAD_KIN_COLS


def get_feature_cols(use_2d_only: bool = False) -> tuple:
    """Get the appropriate VIS and KIN column lists.

    Args:
        use_2d_only: If True, return 2D floor-plane columns.

    Returns:
        Tuple of (vis_cols, kin_cols).
    """
    if use_2d_only:
        return VIS_COLS_2D, KIN_COLS_2D
    return VIS_COLS, KIN_COLS


def get_feature_groups(
    use_2d_only: bool = False,
    include_pos: bool = True,
) -> Dict[str, List[str]]:
    """Get feature group mapping for N-way variance decomposition.

    Args:
        use_2d_only: If True, use 2D floor-plane VIS columns.
        include_pos: If True, include 3D position group.

    Returns:
        Dict mapping group name to list of column names.
    """
    groups: Dict[str, List[str]] = {
        "vis": VIS_COLS_2D if use_2d_only else VIS_COLS,
        "body_kin": BODY_KIN_COLS,
        "head_kin": HEAD_KIN_COLS,
    }
    if include_pos:
        groups["pos"] = POS_COLS
    return groups


# Display labels for figures (code name -> display name)
GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "vis": "VIS",
    "body_kin": "BODY_KIN",
    "head_kin": "HEAD_KIN",
    "pos": "3D POS",
}


__all__ = [
    "get_feature_cols",
    "get_feature_groups",
    "VIS_COLS",
    "KIN_COLS",
    "BODY_KIN_COLS",
    "HEAD_KIN_COLS",
    "POS_COLS",
    "VIS_COLS_2D",
    "KIN_COLS_2D",
    "GROUP_DISPLAY_NAMES",
]
