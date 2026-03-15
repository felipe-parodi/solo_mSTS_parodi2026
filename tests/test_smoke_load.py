"""Smoke test: end-to-end session loading for hedy_0526.

Requires data/ to be populated with raw data files.
Tests the full chain: YAML → path resolution → pose/spike loading →
sync → feature computation → aligned merge.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

YAML_PATH = REPO_ROOT / "configs" / "session_hedy_0526.yaml"
DATA_PRESENT = (REPO_ROOT / "data" / "240526-hedy" / "pose3d_smoothed.npz").exists()

pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="Raw data not in data/")


@pytest.fixture(scope="module")
def loader():
    """Shared loader instance for all tests (expensive to init)."""
    from freemat.single_fm.data.session_loader import SingleFMSessionLoader

    return SingleFMSessionLoader(str(YAML_PATH), debug=True)


class TestPathResolution:
    """Verify that relative YAML paths resolve to real files."""

    def test_paths_resolve(self):
        from freemat_data import load_session_yaml

        sess = load_session_yaml(str(YAML_PATH))
        pose_path = sess.data_files.get("pose3d_npz")
        assert pose_path is not None
        assert Path(pose_path).is_absolute(), f"Path not absolute: {pose_path}"
        assert Path(pose_path).exists(), f"Resolved path missing: {pose_path}"

    def test_syncmap_resolves(self):
        from freemat_data import load_session_yaml

        sess = load_session_yaml(str(YAML_PATH))
        sync_path = sess.data_files.get("syncmap_json")
        assert sync_path is not None
        assert Path(sync_path).exists(), f"Sync map missing: {sync_path}"

    def test_antenna_resolves_from_repo_root(self):
        from freemat_data import load_session_yaml

        sess = load_session_yaml(str(YAML_PATH))
        ant_path = sess.data_files.get("antenna_json")
        assert ant_path is not None
        assert Path(ant_path).exists(), f"Antenna JSON missing: {ant_path}"


class TestPoseLoading:
    """Verify pose loading and orientation."""

    def test_get_pose(self, loader):
        pose = loader.get_pose()
        assert pose.X.ndim == 3
        assert pose.X.shape[2] == 3
        assert pose.X.shape[0] > 1000, "Too few frames"

    def test_video_fps(self, loader):
        assert loader.video_fps == 40.0

    def test_get_pose_oriented(self, loader):
        raw = loader.get_pose()
        oriented = loader.get_pose_oriented()
        assert oriented.X.shape == raw.X.shape
        assert not np.allclose(raw.X, oriented.X), "Rotation had no effect"

    def test_floor_level(self, loader):
        _ = loader.get_pose_oriented()
        assert -2.0 < loader.floor_level < 2.0


class TestArena:
    """Verify arena geometry loads from calibration JSON."""

    def test_arena_loads(self, loader):
        arena = loader.arena
        assert len(arena.walls) == 6
        assert arena.hexagon_radius > 0.5


class TestSpikeLoading:
    """Verify spike loading and binning."""

    def test_get_spikes(self, loader):
        df = loader.get_spikes(bin_s=0.5)
        assert len(df) > 0
        assert "fr" in df.columns
        assert "unit_index" in df.columns
        n_units = df["unit_index"].nunique()
        assert n_units > 100, f"Expected 400+ units, got {n_units}"


class TestAlignedData:
    """Verify full spike+feature merge."""

    def test_get_aligned_data(self, loader):
        merged = loader.get_aligned_data(bin_s=0.5)
        assert len(merged) > 0
        assert "fr" in merged.columns
        assert "com_x" in merged.columns
        assert "speed" in merged.columns
