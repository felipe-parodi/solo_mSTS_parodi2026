"""Smoke test: end-to-end session loading for all 5 sessions.

Requires data/ to be populated with raw data files.
Tests the full chain: YAML -> path resolution -> pose/spike loading ->
sync -> feature computation -> aligned merge.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_PRESENT = (REPO_ROOT / "data" / "240526-M2" / "pose3d_smoothed.npz").exists()
RESULTS_PRESENT = (REPO_ROOT / "results" / "production").exists()

pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="Raw data not in data/")

# Session configs: (yaml_filename, expected_units, expected_fps)
SESSIONS = [
    ("session_M2_0526.yaml", "M2_0526", 443, 40.0),
    ("session_M2_0605.yaml", "M2_0605", 400, 40.0),
    ("session_M1_0525.yaml", "M1_0525", 186, 40.0),
    ("session_M1_0609.yaml", "M1_0609", 191, 30.0),
    ("session_M1_0613.yaml", "M1_0613", 184, 40.0),
]


@pytest.fixture(params=SESSIONS, ids=[s[1] for s in SESSIONS])
def session_info(request):
    return request.param


@pytest.fixture
def loader(session_info):
    from freemat.single_fm.data.session_loader import SingleFMSessionLoader

    yaml_name, session_id, _, _ = session_info
    yaml_path = REPO_ROOT / "configs" / yaml_name
    return SingleFMSessionLoader(str(yaml_path), debug=False)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
class TestPathResolution:
    """Verify that relative YAML paths resolve to real files."""

    def test_paths_resolve(self, session_info):
        from freemat_data import load_session_yaml

        yaml_name = session_info[0]
        sess = load_session_yaml(str(REPO_ROOT / "configs" / yaml_name))
        pose_path = sess.data_files.get("pose3d_npz")
        assert pose_path is not None
        assert Path(pose_path).is_absolute(), f"Not absolute: {pose_path}"
        assert Path(pose_path).exists(), f"Missing: {pose_path}"


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------
class TestPoseLoading:
    """Verify pose loading and orientation for all sessions."""

    def test_get_pose(self, loader, session_info):
        pose = loader.get_pose()
        assert pose.X.ndim == 3
        assert pose.X.shape[2] == 3
        assert pose.X.shape[0] > 1000

    def test_video_fps(self, loader, session_info):
        _, _, _, expected_fps = session_info
        assert loader.video_fps == expected_fps

    def test_get_pose_oriented(self, loader):
        oriented = loader.get_pose_oriented()
        assert oriented.X.ndim == 3
        assert -2.0 < loader.floor_level < 2.0


# ---------------------------------------------------------------------------
# Arena + spikes
# ---------------------------------------------------------------------------
class TestArenaAndSpikes:
    """Verify arena and spike loading for all sessions."""

    def test_arena_loads(self, loader):
        arena = loader.arena
        assert len(arena.walls) == 6
        assert arena.hexagon_radius > 0.5

    def test_spike_loading(self, loader, session_info):
        _, _, expected_units, _ = session_info
        df = loader.get_spikes(bin_s=0.5)
        assert len(df) > 0
        n_units = df["unit_index"].nunique()
        assert n_units == expected_units, f"Got {n_units}, expected {expected_units}"

    def test_bank_labels(self, loader):
        _ = loader.get_spikes(bin_s=0.5)
        bundle = loader._spikes_bundle
        assert bundle.unit_bank is not None, "Bank labels not loaded"
        banks = set(bundle.unit_bank)
        assert banks == {"ventral", "dorsal"}, f"Unexpected banks: {banks}"


# ---------------------------------------------------------------------------
# Full merge (one session only — slow)
# ---------------------------------------------------------------------------
class TestAlignedData:
    """Verify full spike+feature merge for one session."""

    def test_get_aligned_data(self):
        from freemat.single_fm.data.session_loader import SingleFMSessionLoader

        yaml_path = REPO_ROOT / "configs" / "session_M2_0526.yaml"
        loader = SingleFMSessionLoader(str(yaml_path), debug=False)
        merged = loader.get_aligned_data(bin_s=0.5)
        assert len(merged) > 0
        assert "fr" in merged.columns
        assert "com_x" in merged.columns
        assert "speed" in merged.columns
        assert "gaze_x" in merged.columns  # VIS features
        assert "yaw_rate" in merged.columns  # HEAD_KIN features


# ---------------------------------------------------------------------------
# Pre-computed results
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RESULTS_PRESENT, reason="results/ not populated")
class TestPrecomputedResults:
    """Verify pre-computed production results are loadable."""

    def test_fig2_summary_loads(self):
        # Try both possible locations
        for subdir in ["M2_bin05", "all4_bin05"]:
            p = REPO_ROOT / "results" / "production" / subdir / "summary" / "fig2b_summary.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                assert len(df) > 0
                assert "full_r2" in df.columns or "r2_full" in df.columns
                return
        pytest.skip("fig2b_summary.parquet not found in expected locations")

    def test_fig3_summary_loads(self):
        p = REPO_ROOT / "results" / "production" / "fig3_bin05" / "summary" / "fig3_summary.parquet"
        if not p.exists():
            pytest.skip("fig3_summary not found")
        df = pd.read_parquet(p)
        assert len(df) > 0

    def test_fig5_results_load(self):
        fig5_dir = REPO_ROOT / "results" / "fig5_cebra"
        # Check at least one session has results
        found = False
        for session_dir in fig5_dir.iterdir():
            if not session_dir.is_dir():
                continue
            results_json = session_dir / "results.json"
            if results_json.exists():
                with open(results_json) as f:
                    data = json.load(f)
                assert isinstance(data, dict)
                found = True
                break
        assert found, "No fig5 results.json found"

    def test_fig5_embeddings_load(self):
        fig5_dir = REPO_ROOT / "results" / "fig5_cebra"
        found = False
        for session_dir in fig5_dir.iterdir():
            if not session_dir.is_dir():
                continue
            emb_path = session_dir / "embeddings.npz"
            if emb_path.exists():
                data = np.load(emb_path, allow_pickle=True)
                assert len(data.files) > 0
                found = True
                break
        assert found, "No fig5 embeddings.npz found"

    def test_unit_kind_lookup_loads(self):
        p = REPO_ROOT / "results" / "production" / "unit_kind_lookup.parquet"
        if not p.exists():
            pytest.skip("unit_kind_lookup.parquet not found")
        df = pd.read_parquet(p)
        assert len(df) > 0
        assert "unit_kind" in df.columns
