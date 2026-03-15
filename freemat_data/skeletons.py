from __future__ import annotations

"""COCO-17 skeleton helpers and styling.

Provides utilities to resolve joint indices from names, left/right pairs,
and bone edges, plus a consistent color scheme for joints/edges and a
compact metadata payload for persistence across pipelines.

Goals:
- No hard‑coded indices; everything uses ``joint_names``.
- Stable, colorblind‑friendly colors for left/right/head groups.
- A small "skeleton" dict you can embed in Pose3D meta or QC JSON.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


# Canonical COCO-17 order (common variant used in many toolchains)
COCO17_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def coco17_indices(joint_names: Optional[Sequence[str]]) -> Dict[str, int]:
    """Return mapping from COCO joint name to index based on provided joint_names.

    Args:
        joint_names: Sequence of length J with joint labels.

    Returns:
        Dict mapping each expected COCO name to index if found.
    """
    idx: Dict[str, int] = {}
    if not joint_names:
        return idx
    lookup = {str(n).strip().lower(): i for i, n in enumerate(joint_names)}
    for name in COCO17_NAMES:
        key = name.strip().lower()
        if key in lookup:
            idx[name] = lookup[key]
    return idx


def coco17_left_right_pairs(idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """Return left/right index pairs (L,R) for symmetry metrics.

    Missing names are skipped.
    """
    pairs: List[Tuple[int, int]] = []
    def add(l: str, r: str) -> None:
        if l in idx and r in idx:
            pairs.append((idx[l], idx[r]))

    add("left_eye", "right_eye")
    add("left_ear", "right_ear")
    add("left_shoulder", "right_shoulder")
    add("left_elbow", "right_elbow")
    add("left_wrist", "right_wrist")
    add("left_hip", "right_hip")
    add("left_knee", "right_knee")
    add("left_ankle", "right_ankle")
    return pairs


def coco17_bone_edges(idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """Return bone edges as (j0,j1) index pairs for limb-length checks.

    Uses a standard COCO-17 kinematic tree. Missing joints are skipped.
    """
    edges: List[Tuple[int, int]] = []
    def add(a: str, b: str) -> None:
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))

    # Head
    add("nose", "left_eye"); add("nose", "right_eye")
    add("left_eye", "left_ear"); add("right_eye", "right_ear")
    # Torso and arms
    add("left_shoulder", "right_shoulder")
    add("left_shoulder", "left_elbow"); add("left_elbow", "left_wrist")
    add("right_shoulder", "right_elbow"); add("right_elbow", "right_wrist")
    # Hips and legs
    add("left_hip", "right_hip")
    add("left_shoulder", "left_hip"); add("right_shoulder", "right_hip")
    add("left_hip", "left_knee"); add("left_knee", "left_ankle")
    add("right_hip", "right_knee"); add("right_knee", "right_ankle")
    return edges


# -------------------- Grouping and colors --------------------

# Group names used for coloring
GROUP_HEAD = "head"
GROUP_LEFT = "left"
GROUP_RIGHT = "right"
GROUP_OTHER = "other"

# Color palette (hex) — readable on light backgrounds
COLOR_HEAD = "#F0E442"   # yellow
COLOR_LEFT = "#4DBBD5"   # blue-cyan
COLOR_RIGHT = "#E64B35"  # red-orange
COLOR_OTHER = "#7F7F7F"  # neutral gray


def coco17_group_map(joint_names: Optional[Sequence[str]]) -> List[str]:
    """Return per-joint group labels (head/left/right/other) for COCO‑17.

    Args:
        joint_names: Sequence of J names.

    Returns:
        List of length J with group labels.
    """
    if not joint_names:
        return []
    groups: List[str] = []
    for n in joint_names:
        s = str(n or "").strip().lower()
        if s in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}:
            groups.append(GROUP_HEAD)
        elif s.startswith("left_"):
            groups.append(GROUP_LEFT)
        elif s.startswith("right_"):
            groups.append(GROUP_RIGHT)
        else:
            groups.append(GROUP_OTHER)
    return groups


def color_for_group(group: str) -> str:
    g = str(group or "").strip().lower()
    if g == GROUP_LEFT:
        return COLOR_LEFT
    if g == GROUP_RIGHT:
        return COLOR_RIGHT
    if g == GROUP_HEAD:
        return COLOR_HEAD
    return COLOR_OTHER


def joint_colors_from_groups(groups: Sequence[str]) -> List[str]:
    """Map per-joint groups to hex colors."""
    return [color_for_group(g) for g in groups]


def edge_colors_from_groups(edges: Sequence[Tuple[int, int]], groups: Sequence[str]) -> List[str]:
    """Return per-edge colors from joint group labels.

    Convention:
    - L-L edge → left color
    - R-R edge → right color
    - Head‑only edge → head color
    - Mixed → neutral gray
    """
    out: List[str] = []
    for a, b in edges:
        ga = groups[a] if a < len(groups) else GROUP_OTHER
        gb = groups[b] if b < len(groups) else GROUP_OTHER
        if ga == gb == GROUP_LEFT:
            out.append(COLOR_LEFT)
        elif ga == gb == GROUP_RIGHT:
            out.append(COLOR_RIGHT)
        elif ga == gb == GROUP_HEAD:
            out.append(COLOR_HEAD)
        else:
            out.append(COLOR_OTHER)
    return out


def skeleton_style_from_names(joint_names: Optional[Sequence[str]]) -> Dict[str, object]:
    """Build a style dict with edges and colors for given joint_names.

    Returns a dict with keys:
        scheme: 'coco17'
        joint_names: list[str]
        groups: list[str]
        edges: list[tuple[int,int]]
        joint_colors: list[str]
        edge_colors: list[str]
    """
    names = list(joint_names) if joint_names is not None else []
    idx = coco17_indices(names)
    edges = coco17_bone_edges(idx)
    groups = coco17_group_map(names)
    jcols = joint_colors_from_groups(groups) if groups else []
    ecols = edge_colors_from_groups(edges, groups) if edges else []
    return {
        "scheme": "coco17",
        "joint_names": names,
        "groups": groups,
        "edges": edges,
        "joint_colors": jcols,
        "edge_colors": ecols,
    }


def skeleton_meta(joint_names: Optional[Sequence[str]]) -> Dict[str, object]:
    """Return a compact, JSON‑serializable skeleton description.

    Use this in Pose3D meta or QC JSON to persist the skeleton definition
    alongside data so downstream packages can render consistently.
    """
    sty = skeleton_style_from_names(joint_names)
    # Convert edges to name pairs for readability too
    names = sty["joint_names"]
    idx = coco17_indices(names)
    inv = {i: n for n, i in idx.items()}
    edge_names = [(inv.get(a, names[a] if a < len(names) else str(a)), inv.get(b, names[b] if b < len(names) else str(b))) for (a, b) in sty["edges"]]
    return {
        "scheme": sty["scheme"],
        "joint_names": names,
        "groups": sty["groups"],
        "edges": sty["edges"],
        "edge_names": edge_names,
        "palette": {
            GROUP_LEFT: COLOR_LEFT,
            GROUP_RIGHT: COLOR_RIGHT,
            GROUP_HEAD: COLOR_HEAD,
            GROUP_OTHER: COLOR_OTHER,
        },
    }


def resolve_skeleton_style_for_pose(
    joint_names: Optional[Sequence[str]],
    J: int,
) -> Dict[str, object]:
    """Resolve a skeleton style with a pragmatic COCO‑17 fallback.

    This helper first attempts to derive edges/colors from the provided
    ``joint_names`` using the COCO‑17 name mapping. If that yields no edges and
    the joint count ``J`` is exactly 17, it assumes COCO‑17 ordering and
    returns the canonical COCO style (edges/colors) regardless of names.

    Args:
        joint_names: Optional list of joint labels from the source pose.
        J: Number of joints (pose.J).

    Returns:
        Style dictionary with keys compatible with ``skeleton_style_from_names``:
        ``scheme, joint_names, groups, edges, joint_colors, edge_colors``.

    Notes:
        - This function does not attempt to parse per‑dataset meta. It is a
          minimal guard to ensure 17‑joint poses render as connected COCO‑17 by
          default. A richer meta‑driven resolver can be layered on later.
    """
    # First try name-based resolution
    sty = skeleton_style_from_names(joint_names)
    edges = sty.get("edges", [])
    if isinstance(edges, list) and len(edges) > 0:
        return sty
    # Pragmatic fallback: assume COCO‑17 order if J == 17
    if int(J) == 17:
        # Build style using canonical COCO‑17 name order
        sty_co = skeleton_style_from_names(COCO17_NAMES)
        # Return as-is; edges refer to indices [0..16]
        return sty_co
    # Otherwise return the name-based (possibly empty) style
    return sty
