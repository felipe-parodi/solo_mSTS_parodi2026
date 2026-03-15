"""Feature group definitions for unique variance (dropout) analysis.

Defines predictor groups for Musall-style variance partitioning:
- G1 (Translation): Self-locomotion features
- G2 (Orientation): Head/body direction features
- G3 (Drift): Session time nuisance regressor

Usage:
    from analysis.features.groups import FEATURE_GROUPS, get_group_indices

    feature_names = ["speed", "accel", "head_yaw", "session_time"]
    group_indices = get_group_indices(feature_names)
    # Returns: {"G1_translation": [0, 1], "G2_orientation": [2], "G3_drift": [3]}
"""

from __future__ import annotations

from typing import Dict, List


# Feature group definitions (G1-G3)
# Maps group name -> list of feature name patterns
# Order matters: more specific patterns should come first
FEATURE_GROUPS: Dict[str, List[str]] = {
    # G2 checked first - more specific orientation patterns
    "G2_orientation": [
        "head_yaw",
        "head_pitch",
        "head_roll",
        "body_heading",
        "body_yaw",
        "yaw_rate",
        "heading",
        "head_dir",
        "yaw",
        "pitch",
        "roll",
    ],
    # G3 - drift/time patterns
    "G3_drift": [
        "session_time",
        "time_in_session",
        "drift",
    ],
    # G1 - translation/locomotion (checked last due to short patterns)
    "G1_translation": [
        "speed",
        "accel",
        "acceleration",
        "velocity",
        "pos_x",
        "pos_y",
        "pos_z",
        "position",
    ],
}


def get_group_indices(
    feature_names: List[str],
    groups: Dict[str, List[str]] = None,
) -> Dict[str, List[int]]:
    """Map feature names to group indices for dropout analysis.

    Each feature is assigned to at most one group (first match wins).
    Groups are checked in definition order, so put more specific patterns first.

    Args:
        feature_names: List of feature names in the design matrix.
        groups: Optional custom group definitions. Defaults to FEATURE_GROUPS.

    Returns:
        Dict mapping group name to list of column indices.

    Example:
        >>> names = ["speed", "accel", "head_yaw", "session_time"]
        >>> get_group_indices(names)
        {'G1_translation': [0, 1], 'G2_orientation': [2], 'G3_drift': [3]}
    """
    if groups is None:
        groups = FEATURE_GROUPS

    result: Dict[str, List[int]] = {}
    assigned: set = set()  # Track already-assigned indices

    for group_name, patterns in groups.items():
        indices = []
        for i, feat_name in enumerate(feature_names):
            if i in assigned:
                continue  # Skip already-assigned features
            feat_lower = feat_name.lower()
            for pattern in patterns:
                if pattern.lower() in feat_lower:
                    indices.append(i)
                    assigned.add(i)
                    break  # Move to next feature
        if indices:
            result[group_name] = sorted(indices)

    return result


def get_group_mask(
    feature_names: List[str],
    group_name: str,
    groups: Dict[str, List[str]] = None,
) -> List[bool]:
    """Get boolean mask for features in a specific group.

    Args:
        feature_names: List of feature names.
        group_name: Name of the group (e.g., "G1_translation").
        groups: Optional custom group definitions.

    Returns:
        Boolean list where True indicates feature is in the group.
    """
    group_indices = get_group_indices(feature_names, groups)
    indices_set = set(group_indices.get(group_name, []))
    return [i in indices_set for i in range(len(feature_names))]


def validate_groups(
    feature_names: List[str],
    groups: Dict[str, List[str]] = None,
) -> Dict[str, any]:
    """Validate that feature names map correctly to groups.

    Args:
        feature_names: List of feature names.
        groups: Optional custom group definitions.

    Returns:
        Dict with validation info: assigned, unassigned, multi_assigned features.
    """
    if groups is None:
        groups = FEATURE_GROUPS

    group_indices = get_group_indices(feature_names, groups)

    # Find all assigned indices
    all_assigned = set()
    for indices in group_indices.values():
        all_assigned.update(indices)

    # Find unassigned
    unassigned = [
        feature_names[i]
        for i in range(len(feature_names))
        if i not in all_assigned
    ]

    # Find multi-assigned (in multiple groups)
    index_counts: Dict[int, List[str]] = {}
    for group_name, indices in group_indices.items():
        for idx in indices:
            if idx not in index_counts:
                index_counts[idx] = []
            index_counts[idx].append(group_name)

    multi_assigned = {
        feature_names[idx]: groups_list
        for idx, groups_list in index_counts.items()
        if len(groups_list) > 1
    }

    return {
        "group_indices": group_indices,
        "unassigned": unassigned,
        "multi_assigned": multi_assigned,
        "n_features": len(feature_names),
        "n_assigned": len(all_assigned),
    }
