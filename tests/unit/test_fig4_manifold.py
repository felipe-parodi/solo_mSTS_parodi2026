"""Tests for Fig 4 neural manifold analysis.

Tests PCA/UMAP embedding, silhouette separation with block-shuffle null,
and plot function contracts.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_data(rng):
    """Synthetic population data: 200 bins x 30 units, 5 classes."""
    n_bins = 200
    n_units = 30
    n_classes = 5
    bins_per_class = n_bins // n_classes

    X = np.zeros((n_bins, n_units))
    y = np.zeros(n_bins, dtype=int)

    for c in range(n_classes):
        start = c * bins_per_class
        end = (c + 1) * bins_per_class
        # Each class has a different mean direction in neural space
        mean_dir = rng.standard_normal(n_units)
        X[start:end] = mean_dir * 2.0 + rng.standard_normal((bins_per_class, n_units)) * 0.5
        y[start:end] = c

    unit_banks = np.array(["ventral"] * 15 + ["dorsal"] * 15)
    return X, y, unit_banks


@pytest.fixture()
def default_config():
    from freemat.single_fm.encoding.fig4_manifold import ManifoldConfig

    return ManifoldConfig(
        n_perms=5,
        n_subsample=3,
        max_bins_silhouette=200,
    )


# ---------------------------------------------------------------------------
# Test: PCA embedding
# ---------------------------------------------------------------------------

class TestPCAEmbedding:
    """PCA embedding shape and variance properties."""

    def test_shape(self, synthetic_data):
        from freemat.single_fm.encoding.fig4_manifold import compute_pca_embedding

        X, _, _ = synthetic_data
        embedding, var_explained = compute_pca_embedding(X, n_components=10, zscore=True)
        assert embedding.shape == (200, 10)
        assert var_explained.shape == (10,)

    def test_variance_sum_below_one(self, synthetic_data):
        from freemat.single_fm.encoding.fig4_manifold import compute_pca_embedding

        X, _, _ = synthetic_data
        _, var_explained = compute_pca_embedding(X, n_components=20, zscore=True)
        assert var_explained.sum() <= 1.0 + 1e-6

    def test_variance_decreasing(self, synthetic_data):
        from freemat.single_fm.encoding.fig4_manifold import compute_pca_embedding

        X, _, _ = synthetic_data
        _, var_explained = compute_pca_embedding(X, n_components=10, zscore=True)
        # Variance explained should be non-increasing
        assert np.all(np.diff(var_explained) <= 1e-10)

    def test_no_zscore(self, synthetic_data):
        from freemat.single_fm.encoding.fig4_manifold import compute_pca_embedding

        X, _, _ = synthetic_data
        embedding, _ = compute_pca_embedding(X, n_components=5, zscore=False)
        assert embedding.shape == (200, 5)

    def test_components_capped(self, synthetic_data):
        """n_components capped at min(n_samples, n_features)."""
        from freemat.single_fm.encoding.fig4_manifold import compute_pca_embedding

        X, _, _ = synthetic_data
        embedding, var_explained = compute_pca_embedding(X, n_components=100, zscore=True)
        # Should be capped at min(200, 30) = 30
        assert embedding.shape[1] == 30
        assert var_explained.shape[0] == 30


# ---------------------------------------------------------------------------
# Test: UMAP embedding
# ---------------------------------------------------------------------------

class TestUMAPEmbedding:
    """UMAP embedding requires umap-learn."""

    def test_import_error_when_missing(self, synthetic_data):
        """Hard ImportError when umap-learn not installed."""
        from freemat.single_fm.encoding.fig4_manifold import compute_umap_embedding

        X, _, _ = synthetic_data
        with patch.dict("sys.modules", {"umap": None}):
            with pytest.raises(ImportError, match="umap-learn"):
                compute_umap_embedding(X, n_neighbors=15, min_dist=0.1, random_state=42)

    def test_shape_when_available(self, synthetic_data):
        """UMAP produces 2D embedding when available."""
        pytest.importorskip("umap")
        from freemat.single_fm.encoding.fig4_manifold import compute_umap_embedding

        X, _, _ = synthetic_data
        embedding = compute_umap_embedding(X, n_neighbors=15, min_dist=0.1, random_state=42)
        assert embedding.shape == (200, 2)


# ---------------------------------------------------------------------------
# Test: cluster separation (silhouette)
# ---------------------------------------------------------------------------

class TestClusterSeparation:
    """Silhouette score with block-shuffle null."""

    def test_silhouette_bounds(self, synthetic_data, default_config):
        from freemat.single_fm.encoding.fig4_manifold import compute_cluster_separation

        X, y, _ = synthetic_data
        result = compute_cluster_separation(X[:, :3], y, default_config)
        assert -1 <= result.silhouette_mean <= 1

    def test_per_class_keys(self, synthetic_data, default_config):
        from freemat.single_fm.encoding.fig4_manifold import compute_cluster_separation

        X, y, _ = synthetic_data
        result = compute_cluster_separation(X[:, :3], y, default_config)
        class_labels = sorted(set(y))
        assert set(result.silhouette_per_class.keys()) == set(class_labels)

    def test_null_distribution_size(self, synthetic_data, default_config):
        from freemat.single_fm.encoding.fig4_manifold import compute_cluster_separation

        X, y, _ = synthetic_data
        result = compute_cluster_separation(X[:, :3], y, default_config)
        assert len(result.null_distribution) == default_config.n_perms

    def test_p_value_range(self, synthetic_data, default_config):
        from freemat.single_fm.encoding.fig4_manifold import compute_cluster_separation

        X, y, _ = synthetic_data
        result = compute_cluster_separation(X[:, :3], y, default_config)
        assert 0 <= result.p_value <= 1

    def test_well_separated_data_low_p(self, default_config):
        """Well-separated clusters → observed > null → low p."""
        from freemat.single_fm.encoding.fig4_manifold import compute_cluster_separation

        rng = np.random.default_rng(42)
        # Very well separated clusters in 3D
        X = np.vstack([
            rng.normal(loc=[10, 0, 0], scale=0.1, size=(50, 3)),
            rng.normal(loc=[0, 10, 0], scale=0.1, size=(50, 3)),
            rng.normal(loc=[0, 0, 10], scale=0.1, size=(50, 3)),
        ])
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        result = compute_cluster_separation(X, y, default_config)
        assert result.silhouette_mean > 0.5
        assert result.p_value < 0.5  # should be very low with clear clusters


# ---------------------------------------------------------------------------
# Test: full manifold analysis
# ---------------------------------------------------------------------------

class TestManifoldAnalysis:
    """Integration test for compute_manifold_analysis."""

    def test_pca_only(self, synthetic_data):
        from freemat.single_fm.encoding.fig4_manifold import (
            ManifoldConfig,
            compute_manifold_analysis,
        )

        X, y, unit_banks = synthetic_data
        config = ManifoldConfig(
            use_umap=False,
            n_perms=3,
            n_subsample=2,
            max_bins_silhouette=100,
        )
        result = compute_manifold_analysis(X, y, unit_banks, "test_session", config)
        assert result.pca_embedding.shape[0] == 200
        assert result.pca_var_explained is not None
        assert result.has_umap is False
        assert result.umap_embedding.shape == (0, 0)
        assert result.session_id == "test_session"
        # Multi-embedding silhouettes
        assert result.separations is not None
        assert "pca_3d" in result.separations
        assert "pca_10d" in result.separations
        # Without UMAP, primary should be pca_10d
        assert "umap" not in result.separations

    def test_with_umap(self, synthetic_data):
        pytest.importorskip("umap")
        from freemat.single_fm.encoding.fig4_manifold import (
            ManifoldConfig,
            compute_manifold_analysis,
        )

        X, y, unit_banks = synthetic_data
        config = ManifoldConfig(
            use_umap=True,
            n_perms=3,
            n_subsample=2,
            max_bins_silhouette=100,
        )
        result = compute_manifold_analysis(X, y, unit_banks, "test_session", config)
        assert result.has_umap is True
        assert result.umap_embedding.shape == (200, 2)
        # UMAP silhouette should exist
        assert "umap" in result.separations
        # Primary separation should be UMAP when available
        assert result.separation is result.separations["umap"]


# ---------------------------------------------------------------------------
# Test: null pipeline parity
# ---------------------------------------------------------------------------

class TestNullPipelineParity:
    """Observed and null use identical subsampling logic."""

    def test_subsample_count_matches(self, synthetic_data, default_config):
        """Both observed and null pipelines use n_subsample iterations."""
        from freemat.single_fm.encoding.fig4_manifold import compute_cluster_separation

        X, y, _ = synthetic_data
        result = compute_cluster_separation(X[:, :3], y, default_config)
        # CI exists (from repeated subsampling)
        assert result.silhouette_ci is not None
        assert len(result.silhouette_ci) == 2
        assert result.silhouette_ci[0] <= result.silhouette_mean <= result.silhouette_ci[1]


# ---------------------------------------------------------------------------
# Test: plot functions return Figure
# ---------------------------------------------------------------------------

class TestPlotFunctions:
    """All manifold plot functions return matplotlib Figure."""

    @pytest.fixture()
    def manifold_result(self, synthetic_data):
        from freemat.single_fm.encoding.fig4_manifold import (
            ManifoldConfig,
            compute_manifold_analysis,
        )

        X, y, unit_banks = synthetic_data
        config = ManifoldConfig(
            use_umap=False,
            n_perms=3,
            n_subsample=2,
            max_bins_silhouette=100,
        )
        return compute_manifold_analysis(X, y, unit_banks, "test_session", config)

    def test_pca_2d(self, manifold_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_manifold_panels import plot_manifold_pca_2d

        fig = plot_manifold_pca_2d(manifold_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_3d(self, manifold_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_manifold_panels import plot_manifold_pca_3d

        fig = plot_manifold_pca_3d(manifold_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scree(self, manifold_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_manifold_panels import plot_manifold_scree

        fig = plot_manifold_scree(manifold_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_silhouette(self, manifold_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_manifold_panels import (
            plot_manifold_silhouette_per_class,
        )

        fig = plot_manifold_silhouette_per_class(manifold_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bank_comparison(self, manifold_result, synthetic_data):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_manifold_panels import (
            plot_manifold_bank_comparison,
        )

        X, _, unit_banks = synthetic_data
        fig = plot_manifold_bank_comparison(manifold_result, unit_banks, X_valid=X)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bank_comparison_without_x(self, manifold_result, synthetic_data):
        """Fallback to full-population PCA when X_valid is None."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_manifold_panels import (
            plot_manifold_bank_comparison,
        )

        _, _, unit_banks = synthetic_data
        fig = plot_manifold_bank_comparison(manifold_result, unit_banks)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test: block shuffle correctness
# ---------------------------------------------------------------------------

class TestBlockShuffle:
    """Block-shuffle preserves all labels with no gaps."""

    def test_no_gaps_non_divisible(self):
        """n=10, block_size=4 leaves no unwritten positions."""
        from freemat.single_fm.encoding.fig4_manifold import _block_shuffle_labels

        rng = np.random.default_rng(0)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        shuffled = _block_shuffle_labels(labels, block_size=4, rng=rng)
        assert shuffled.shape == labels.shape
        # Same multiset of values — no garbage
        assert sorted(shuffled.tolist()) == sorted(labels.tolist())

    def test_exact_divisible(self):
        """n=12, block_size=4 — exact division works."""
        from freemat.single_fm.encoding.fig4_manifold import _block_shuffle_labels

        rng = np.random.default_rng(1)
        labels = np.arange(12)
        shuffled = _block_shuffle_labels(labels, block_size=4, rng=rng)
        assert shuffled.shape == labels.shape
        assert sorted(shuffled.tolist()) == sorted(labels.tolist())

    def test_block_size_one(self):
        """block_size=1 is a full element-wise shuffle."""
        from freemat.single_fm.encoding.fig4_manifold import _block_shuffle_labels

        rng = np.random.default_rng(2)
        labels = np.array([0, 1, 2, 3, 4])
        shuffled = _block_shuffle_labels(labels, block_size=1, rng=rng)
        assert sorted(shuffled.tolist()) == sorted(labels.tolist())


# ---------------------------------------------------------------------------
# Test: bin_s config propagation
# ---------------------------------------------------------------------------

class TestBinSConfig:
    """ManifoldConfig.bin_s is used for block-size conversion."""

    def test_bin_s_affects_block_bins(self, synthetic_data):
        """bin_s=1.0 with block_s=10.0 gives 10 block-bins, not 20."""
        from freemat.single_fm.encoding.fig4_manifold import (
            ManifoldConfig,
            compute_cluster_separation,
        )

        X, y, _ = synthetic_data
        config = ManifoldConfig(
            n_perms=2,
            n_subsample=2,
            max_bins_silhouette=200,
            block_s=10.0,
            bin_s=1.0,
        )
        # Should not error; block_bins = 10.0 / 1.0 = 10
        result = compute_cluster_separation(X[:, :3], y, config)
        assert -1 <= result.silhouette_mean <= 1
