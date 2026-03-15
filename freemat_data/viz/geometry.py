from __future__ import annotations

"""Renderer-agnostic geometry helpers (single namespace entry).

This module re-exports array-producing geometry utilities for 3D scenes
from ``freemat_data.vis`` so users can import them under a single
``freemat_data.viz`` namespace.
"""

from ..vis import (
    plane_basis_from_dict,
    project_points_to_uv,
    uv_to_xyz,
    load_hex_uv,
    regular_hex_from_uv,
    checker_mesh_from_hex,
    camera_gizmo_segments,
)

__all__ = [
    "plane_basis_from_dict",
    "project_points_to_uv",
    "uv_to_xyz",
    "load_hex_uv",
    "regular_hex_from_uv",
    "checker_mesh_from_hex",
    "camera_gizmo_segments",
]

