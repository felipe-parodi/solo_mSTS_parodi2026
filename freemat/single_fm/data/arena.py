"""Standalone arena geometry for the paper repo.

Loads pre-computed hexagonal arena geometry from JSON and provides the
distance computation methods needed for VIS feature generation (d_wall_horiz).

No external triangulation dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WallPlane:
    """A single wall plane of the hexagonal arena.

    Plane equation: normal . (p - point_on_plane) = 0
    Or equivalently: a*x + b*y + c*z + d = 0 where (a,b,c) = normal, d = offset.

    Attributes:
        wall_id: Wall identifier ("wall_0", "wall_1", etc.).
        normal: Unit normal vector pointing INWARD (toward arena center).
        offset: Plane offset d in ax + by + cz + d = 0.
        point_on_plane: A point on the wall plane.
        vertices: Two corner points where this wall meets adjacent walls.
    """

    wall_id: str
    normal: np.ndarray
    offset: float
    point_on_plane: np.ndarray
    vertices: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def distance_to_point(self, point: np.ndarray) -> float:
        """Signed distance from point to plane (positive = inside arena)."""
        return float(np.dot(self.normal, point) + self.offset)

    def ray_intersection(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> Optional[float]:
        """Compute ray-plane intersection distance.

        Args:
            origin: Ray origin point (3,).
            direction: Ray direction unit vector (3,).

        Returns:
            Distance t such that origin + t*direction is on the plane,
            or None if ray is parallel to plane or intersection is behind.
        """
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-9:
            return None
        t = -(np.dot(self.normal, origin) + self.offset) / denom
        return t if t > 0 else None


@dataclass
class ArenaGeometry:
    """Hexagonal arena geometry with wall planes and distance methods.

    Attributes:
        walls: Dict mapping wall_id -> WallPlane.
        center: Arena center point (3,).
        up_axis: Vertical axis ('Y' or 'Z').
        hexagon_radius: Circumradius of inscribed hexagon (meters).
        hexagon_vertices: (6, 3) vertices of the hexagon.
        floor_height: Floor plane height (meters).
        ceiling_height: Ceiling height (meters).
    """

    walls: Dict[str, WallPlane]
    center: np.ndarray
    up_axis: str
    hexagon_radius: float
    hexagon_vertices: np.ndarray
    floor_height: float = 0.0
    ceiling_height: float = 2.5

    def compute_d_wall_horiz(
        self, position: np.ndarray, direction: np.ndarray
    ) -> Tuple[np.ndarray, List[Optional[str]]]:
        """Compute horizontal distance to nearest wall along gaze.

        Projects gaze direction onto the horizontal plane and computes
        ray-wall intersection for all 6 walls, returning the nearest.

        Args:
            position: (3,) or (N, 3) positions.
            direction: (3,) or (N, 3) gaze direction unit vectors.

        Returns:
            Tuple of (distances, wall_ids).
            For batch input: distances is (N,), wall_ids is list of N strings.
        """
        if position.ndim == 1:
            d, w = self._compute_d_wall_horiz_single(position, direction)
            return np.array([d]), [w]

        N = position.shape[0]
        distances = np.full(N, np.inf)
        wall_ids: List[Optional[str]] = [None] * N

        for i in range(N):
            if np.any(np.isnan(position[i])) or np.any(np.isnan(direction[i])):
                continue
            distances[i], wall_ids[i] = self._compute_d_wall_horiz_single(
                position[i], direction[i]
            )

        return distances, wall_ids

    def _compute_d_wall_horiz_single(
        self, position: np.ndarray, direction: np.ndarray
    ) -> Tuple[float, Optional[str]]:
        """Compute horizontal d_wall for a single point."""
        up_idx = 2 if self.up_axis.upper() == "Z" else 1

        # Project direction to horizontal plane
        dir_horiz = direction.copy()
        dir_horiz[up_idx] = 0.0
        dir_norm = np.linalg.norm(dir_horiz)

        if dir_norm < 1e-9:
            return self.hexagon_radius, None

        dir_horiz = dir_horiz / dir_norm

        # Project position to floor level
        pos_horiz = position.copy()
        pos_horiz[up_idx] = self.floor_height

        # Check all walls
        min_dist = np.inf
        nearest_wall = None

        for wall_id, wall in self.walls.items():
            t = wall.ray_intersection(pos_horiz, dir_horiz)
            if t is not None and t < min_dist:
                min_dist = t
                nearest_wall = wall_id

        return min_dist, nearest_wall

    @classmethod
    def from_json(cls, path: Path) -> "ArenaGeometry":
        """Load ArenaGeometry from a pre-computed calibration JSON.

        Args:
            path: Path to calibration_<session_id>.json.

        Returns:
            ArenaGeometry instance.
        """
        with open(path) as f:
            data = json.load(f)

        arena_data = data["arena"]

        walls = {}
        for wall_id, wall_dict in arena_data["walls"].items():
            verts = wall_dict.get("vertices")
            if verts is not None and len(verts) == 2:
                vertices = (np.array(verts[0]), np.array(verts[1]))
            else:
                vertices = None
            walls[wall_id] = WallPlane(
                wall_id=wall_id,
                normal=np.array(wall_dict["normal"]),
                offset=float(wall_dict["offset"]),
                point_on_plane=np.array(wall_dict["point_on_plane"]),
                vertices=vertices,
            )

        return cls(
            walls=walls,
            center=np.array(arena_data["center"]),
            up_axis=arena_data["up_axis"],
            hexagon_radius=float(arena_data["hexagon_radius"]),
            hexagon_vertices=np.array(arena_data["hexagon_vertices"]),
            floor_height=float(arena_data["floor_height"]),
            ceiling_height=float(arena_data["ceiling_height"]),
        )


def compute_d_wall_horiz_batch(
    positions: np.ndarray,
    directions: np.ndarray,
    arena: ArenaGeometry,
) -> np.ndarray:
    """Compute d_wall_horiz for a batch of positions and directions.

    Args:
        positions: (T, 3) array of positions.
        directions: (T, 3) array of gaze direction unit vectors.
        arena: ArenaGeometry with wall planes.

    Returns:
        (T,) array of horizontal distances to nearest wall.
    """
    distances, _ = arena.compute_d_wall_horiz(positions, directions)
    return distances
