"""Spatial feature computation for single-FM (wall distance, camera geometry).

Computes distance to walls from camera-derived vertical wall planes.

Usage:
    >>> from freemat.single_fm.kin.spatial import (
    ...     load_camera_positions,
    ...     compute_wall_planes,
    ...     compute_distance_to_wall,
    ... )
    >>> cam_positions = load_camera_positions(extraction_report_path)
    >>> wall_normals, wall_points = compute_wall_planes(cam_positions)
    >>> distances = compute_distance_to_wall(com_positions, wall_normals, wall_points)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_camera_positions(extraction_report_path: str) -> np.ndarray:
    """Load camera positions from multical extraction report.

    Parses the extraction_report.txt file to extract camera extrinsic positions.

    Args:
        extraction_report_path: Path to extraction_report.txt from multical.

    Returns:
        Camera positions array (N_cameras, 3) in meters.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If parsing fails.
    """
    path = Path(extraction_report_path)
    if not path.exists():
        raise FileNotFoundError(f"Extraction report not found: {path}")

    # Parse the extraction report
    # Format varies but typically contains camera names and translation vectors
    # Look for lines with camera positions in format:
    # camera_name: t=[x, y, z] or similar

    positions = []
    camera_names = []

    with open(path, 'r') as f:
        content = f.read()

    # Try multiple patterns to find camera positions
    # Pattern 1: t=[x, y, z]
    pattern1 = r'(\w+):\s*t=\[([-\d.e]+),\s*([-\d.e]+),\s*([-\d.e]+)\]'
    matches = re.findall(pattern1, content)

    if matches:
        for match in matches:
            name, x, y, z = match
            camera_names.append(name)
            positions.append([float(x), float(y), float(z)])
    else:
        # Pattern 2: Camera name followed by translation vector
        pattern2 = r'Camera\s+(\w+).*?translation.*?\[([-\d.e\s,]+)\]'
        matches = re.findall(pattern2, content, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches:
                name = match[0]
                coords = [float(x) for x in re.findall(r'[-\d.e]+', match[1])]
                if len(coords) >= 3:
                    camera_names.append(name)
                    positions.append(coords[:3])

    if not positions:
        raise ValueError(f"Could not parse camera positions from {path}")

    return np.array(positions, dtype=np.float64)


def compute_wall_planes(
    camera_positions: np.ndarray,
    arena_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vertical wall planes from camera positions.

    Creates wall planes perpendicular to the floor, passing through each
    camera position, with normals pointing toward the arena center.

    Args:
        camera_positions: Camera positions (N, 3) in meters.
        arena_center: Center of the arena (3,).

    Returns:
        Tuple of (wall_normals, wall_points):
            wall_normals: Unit normal vectors (N, 3), pointing toward center.
            wall_points: Points on each wall (N, 3), the camera positions.
    """
    camera_positions = np.asarray(camera_positions, dtype=np.float64)
    arena_center = np.asarray(arena_center, dtype=np.float64)

    # For each camera, create a vertical wall plane
    # Normal points from camera toward arena center (XY only, perpendicular to floor)
    N = len(camera_positions)

    wall_normals = np.zeros((N, 3), dtype=np.float64)
    wall_points = camera_positions.copy()

    for i in range(N):
        # Direction from camera to center (XY plane only)
        direction = arena_center[:2] - camera_positions[i, :2]
        norm = np.linalg.norm(direction)

        if norm > 1e-9:
            # Normal is horizontal, pointing toward center
            wall_normals[i, 0] = direction[0] / norm
            wall_normals[i, 1] = direction[1] / norm
            wall_normals[i, 2] = 0.0  # Vertical wall, no Z component in normal
        else:
            # Camera at center - arbitrary normal
            wall_normals[i] = [1.0, 0.0, 0.0]

    return wall_normals, wall_points


def compute_distance_to_wall(
    positions: np.ndarray,
    wall_normals: np.ndarray,
    wall_points: np.ndarray,
) -> np.ndarray:
    """Compute distance to the nearest wall for each position.

    Args:
        positions: Query positions (T, 3) in meters.
        wall_normals: Wall normal vectors (N_walls, 3), pointing inward.
        wall_points: Points on each wall (N_walls, 3).

    Returns:
        Distance to nearest wall (T,) in meters, always >= 0.
    """
    positions = np.asarray(positions, dtype=np.float64)
    wall_normals = np.asarray(wall_normals, dtype=np.float64)
    wall_points = np.asarray(wall_points, dtype=np.float64)

    T = len(positions)
    N_walls = len(wall_normals)

    # Compute signed distance to each wall plane
    # d = (p - p0) · n
    # Positive distance means inside the arena (on the normal side)

    distances_to_walls = np.zeros((T, N_walls), dtype=np.float64)

    for w in range(N_walls):
        # Vector from wall point to each position
        diff = positions - wall_points[w].reshape(1, 3)
        # Signed distance (dot product with normal)
        distances_to_walls[:, w] = diff @ wall_normals[w]

    # Take minimum distance to any wall (clamped to 0)
    # Since normals point inward, positive distance means inside
    min_distances = np.min(distances_to_walls, axis=1)

    # Clamp to non-negative (in case position is outside arena)
    return np.maximum(min_distances, 0.0)


def compute_hexagonal_wall_planes(
    radius: float = 3.0,
    center: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute wall planes for a regular hexagonal arena.

    Creates 6 wall planes at 60-degree intervals.

    Args:
        radius: Distance from center to each wall in meters.
        center: Arena center position.

    Returns:
        Tuple of (wall_normals, wall_points).
    """
    center = np.asarray(center, dtype=np.float64)

    angles = np.arange(0, 2 * np.pi, np.pi / 3)  # 0, 60, 120, 180, 240, 300 degrees
    N = len(angles)

    wall_normals = np.zeros((N, 3), dtype=np.float64)
    wall_points = np.zeros((N, 3), dtype=np.float64)

    for i, angle in enumerate(angles):
        # Normal pointing inward (toward center)
        wall_normals[i, 0] = -np.cos(angle)
        wall_normals[i, 1] = -np.sin(angle)
        wall_normals[i, 2] = 0.0

        # Point on wall at distance radius from center
        wall_points[i, 0] = center[0] + radius * np.cos(angle)
        wall_points[i, 1] = center[1] + radius * np.sin(angle)
        wall_points[i, 2] = center[2]

    return wall_normals, wall_points


def compute_distance_to_center(
    positions: np.ndarray,
    center: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """Compute radial distance from arena center (XY plane only).

    Args:
        positions: Query positions (T, 3) in meters.
        center: Arena center position.

    Returns:
        Radial distance (T,) in meters.
    """
    positions = np.asarray(positions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    # XY distance only
    diff_xy = positions[:, :2] - center[:2].reshape(1, 2)
    return np.linalg.norm(diff_xy, axis=1)


__all__ = [
    "load_camera_positions",
    "compute_wall_planes",
    "compute_distance_to_wall",
    "compute_hexagonal_wall_planes",
    "compute_distance_to_center",
]
