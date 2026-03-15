"""Standalone tests for the paper repo decoupling.

Tests that the new repo's calibration, arena, and import graph
work correctly without freemat_vid_inference or multical.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. Arena geometry: load from JSON, verify distances
# ---------------------------------------------------------------------------
class TestArenaFromJSON:
    """Verify ArenaGeometry loads from pre-computed JSON and computes distances."""

    CALIB_DIR = REPO_ROOT / "configs"
    SESSIONS = [
        "hedy_0526", "hedy_0605",
        "hooke_0525", "hooke_0609", "hooke_0613",
    ]

    @pytest.fixture(params=SESSIONS)
    def calib_path(self, request):
        p = self.CALIB_DIR / f"calibration_{request.param}.json"
        if not p.exists():
            pytest.skip(f"Calibration JSON missing: {p.name}")
        return p

    def test_arena_loads(self, calib_path):
        from freemat.single_fm.data.arena import ArenaGeometry
        arena = ArenaGeometry.from_json(calib_path)
        assert len(arena.walls) == 6
        assert arena.hexagon_radius > 0.5  # at least 0.5m
        assert arena.center.shape == (3,)
        assert arena.hexagon_vertices.shape == (6, 3)

    def test_d_wall_horiz_from_center(self, calib_path):
        from freemat.single_fm.data.arena import ArenaGeometry, compute_d_wall_horiz_batch
        arena = ArenaGeometry.from_json(calib_path)
        pos = arena.center.reshape(1, 3)
        direction = np.array([[1.0, 0.0, 0.0]])
        d = compute_d_wall_horiz_batch(pos, direction, arena)
        assert d.shape == (1,)
        assert 0.5 < d[0] < 3.0, f"Unreasonable distance from center: {d[0]:.2f}m"

    def test_d_wall_horiz_nan_safe(self, calib_path):
        from freemat.single_fm.data.arena import compute_d_wall_horiz_batch, ArenaGeometry
        arena = ArenaGeometry.from_json(calib_path)
        pos = np.array([[np.nan, np.nan, np.nan], arena.center])
        direction = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        d = compute_d_wall_horiz_batch(pos, direction, arena)
        assert d.shape == (2,)
        assert np.isinf(d[0]), "NaN position should yield inf distance"
        assert np.isfinite(d[1]), "Valid position should yield finite distance"

    def test_wall_planes_have_inward_normals(self, calib_path):
        from freemat.single_fm.data.arena import ArenaGeometry
        arena = ArenaGeometry.from_json(calib_path)
        for wall_id, wall in arena.walls.items():
            # Normal should point inward → positive dot with (center - wall_point)
            to_center = arena.center - wall.point_on_plane
            dot = np.dot(wall.normal, to_center)
            assert dot > 0, f"{wall_id}: normal points outward (dot={dot:.3f})"


# ---------------------------------------------------------------------------
# 2. Calibration JSON: verify structure and values
# ---------------------------------------------------------------------------
class TestCalibrationJSON:
    """Verify calibration JSONs have correct structure and sane values."""

    CALIB_DIR = REPO_ROOT / "configs"
    SESSIONS = [
        "hedy_0526", "hedy_0605",
        "hooke_0525", "hooke_0609", "hooke_0613",
    ]

    @pytest.fixture(params=SESSIONS)
    def calib_data(self, request):
        p = self.CALIB_DIR / f"calibration_{request.param}.json"
        if not p.exists():
            pytest.skip(f"Calibration JSON missing: {p.name}")
        with open(p) as f:
            return json.load(f)

    def test_required_keys(self, calib_data):
        required = {"session_id", "video_fps", "R_total", "floor_level", "up_axis", "plane_dict", "arena"}
        assert required.issubset(calib_data.keys())

    def test_R_total_is_rotation(self, calib_data):
        R = np.array(calib_data["R_total"])
        assert R.shape == (3, 3)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6, f"R_total determinant = {det:.6f}, expected 1.0"
        # R @ R^T should be identity
        eye = R @ R.T
        assert np.allclose(eye, np.eye(3), atol=1e-6), "R_total is not orthogonal"

    def test_floor_level_range(self, calib_data):
        fl = calib_data["floor_level"]
        assert -2.0 < fl < 2.0, f"Floor level {fl:.2f}m seems unreasonable"

    def test_video_fps(self, calib_data):
        fps = calib_data["video_fps"]
        assert fps in (30.0, 40.0), f"Unexpected fps: {fps}"

    def test_up_axis(self, calib_data):
        assert calib_data["up_axis"] in ("X", "Y", "Z")

    def test_arena_has_6_walls(self, calib_data):
        assert calib_data["arena"]["n_walls"] == 6
        assert len(calib_data["arena"]["walls"]) == 6


# ---------------------------------------------------------------------------
# 3. Import graph: verify core modules import without freemat_vid_inference
# ---------------------------------------------------------------------------
class TestImportGraph:
    """Verify that core pipeline modules import without freemat_vid_inference."""

    def test_arena_module(self):
        from freemat.single_fm.data.arena import ArenaGeometry, WallPlane, compute_d_wall_horiz_batch
        assert ArenaGeometry is not None
        assert WallPlane is not None
        assert compute_d_wall_horiz_batch is not None

    def test_calibration_module(self):
        from freemat.single_fm.data.calibration import (
            load_calibration_json,
            load_arena_from_json,
            apply_rotation_to_pose,
            compute_horizontal_alignment,
        )
        assert load_calibration_json is not None
        assert load_arena_from_json is not None
        assert apply_rotation_to_pose is not None
        assert compute_horizontal_alignment is not None

    def test_apply_rotation_to_pose(self):
        from freemat.single_fm.data.calibration import apply_rotation_to_pose
        R = np.eye(3)
        X = np.random.randn(10, 17, 3)
        X_rot = apply_rotation_to_pose(X, R)
        np.testing.assert_allclose(X_rot, X, atol=1e-12)

    def test_apply_rotation_with_translation(self):
        from freemat.single_fm.data.calibration import apply_rotation_to_pose
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        X = np.zeros((5, 17, 3))
        X_rot = apply_rotation_to_pose(X, R, t)
        expected = np.zeros((5, 17, 3)) + t
        np.testing.assert_allclose(X_rot, expected, atol=1e-12)

    def test_horizontal_alignment(self):
        from freemat.single_fm.data.calibration import compute_horizontal_alignment
        pts = np.random.randn(10, 3)
        R, t, rmse = compute_horizontal_alignment(pts, pts, up_axis=1)
        assert rmse < 1e-10, "Identity alignment should have near-zero RMSE"


# ---------------------------------------------------------------------------
# 4. Session YAML: verify updated configs
# ---------------------------------------------------------------------------
class TestSessionYAMLs:
    """Verify session YAMLs have relative paths and no freemat_vid_inference refs."""

    CONFIG_DIR = REPO_ROOT / "configs"

    @pytest.fixture(params=[
        "session_hedy_0526.yaml",
        "session_M2_0605.yaml",
        "session_M1_0525.yaml",
        "session_M1_0609.yaml",
        "session_M1_0613.yaml",
    ])
    def yaml_content(self, request):
        import yaml
        p = self.CONFIG_DIR / request.param
        with open(p) as f:
            return yaml.safe_load(f), p.read_text()

    def test_no_absolute_paths(self, yaml_content):
        _, raw = yaml_content
        assert "C:/" not in raw, "Session YAML contains absolute Windows path"
        assert "A:/" not in raw, "Session YAML contains absolute A: drive path"

    def test_no_triangulation_config(self, yaml_content):
        data, _ = yaml_content
        assert "triangulation_config" not in data, "Session YAML still references triangulation_config"

    def test_no_freemat_vid_inference(self, yaml_content):
        _, raw = yaml_content
        assert "freemat_vid_inference" not in raw

    def test_has_session_id(self, yaml_content):
        data, _ = yaml_content
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_has_data_files(self, yaml_content):
        data, _ = yaml_content
        assert "data_files" in data
        df = data["data_files"]
        assert "pose3d_npz" in df
        assert "syncmap_json" in df
        assert "motifs_dir" in df


# ---------------------------------------------------------------------------
# 5. Cross-session consistency
# ---------------------------------------------------------------------------
class TestCrossSessionConsistency:
    """Verify all 5 calibrations are self-consistent."""

    CALIB_DIR = REPO_ROOT / "configs"
    SESSIONS = [
        "hedy_0526", "hedy_0605",
        "hooke_0525", "hooke_0609", "hooke_0613",
    ]

    def test_all_sessions_same_up_axis(self):
        axes = set()
        for sid in self.SESSIONS:
            p = self.CALIB_DIR / f"calibration_{sid}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            axes.add(data["up_axis"])
        assert len(axes) == 1, f"Inconsistent up_axis across sessions: {axes}"

    def test_arena_radii_similar(self):
        radii = []
        for sid in self.SESSIONS:
            p = self.CALIB_DIR / f"calibration_{sid}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            radii.append(data["arena"]["hexagon_radius"])
        assert max(radii) - min(radii) < 1.0, f"Arena radii vary too much: {radii}"

    def test_floor_levels_similar(self):
        floors = []
        for sid in self.SESSIONS:
            p = self.CALIB_DIR / f"calibration_{sid}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            floors.append(data["floor_level"])
        assert max(floors) - min(floors) < 0.5, f"Floor levels vary too much: {floors}"
