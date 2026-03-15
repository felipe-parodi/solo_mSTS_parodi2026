"""Unit tests for Temporal Generalization Matrix (TGM) analysis."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.syllables.tgm import TGMResult, _apply_embargo, compute_tgm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_bins: int = 500,
    n_units: int = 30,
    n_classes: int = 5,
    bout_len: int = 10,
    signal_strength: float = 2.0,
    seed: int = 42,
):
    """Create synthetic data with block motif structure and class-specific firing.

    Each motif class has a distinct mean firing pattern, so a decoder should
    be able to identify motif identity from neural activity.
    """
    rng = np.random.RandomState(seed)

    # Create block motif labels
    y = np.zeros(n_bins, dtype=int)
    classes = np.arange(n_classes)
    for i in range(0, n_bins, bout_len):
        y[i : i + bout_len] = rng.choice(classes)

    # Create neural activity with class-specific means
    templates = rng.randn(n_classes, n_units) * signal_strength
    X = np.zeros((n_bins, n_units), dtype=np.float32)
    for t in range(n_bins):
        X[t] = templates[y[t]] + rng.randn(n_units) * 1.0

    # Detect transitions
    onset_bins = []
    post_syllables = []
    for t in range(1, n_bins):
        if y[t] != y[t - 1]:
            onset_bins.append(t)
            post_syllables.append(y[t])
    onset_bins = np.array(onset_bins, dtype=int)
    post_syllables = np.array(post_syllables, dtype=int)

    return X, y, onset_bins, post_syllables


# ---------------------------------------------------------------------------
# Tests: _apply_embargo
# ---------------------------------------------------------------------------

class TestApplyEmbargo:
    def test_no_embargo(self):
        onset_bins = np.array([10, 20, 30, 40, 50])
        train_idx = np.array([0, 1, 2])
        test_idx = np.array([3, 4])
        result = _apply_embargo(onset_bins, train_idx, test_idx, embargo_bins=0)
        np.testing.assert_array_equal(result, train_idx)

    def test_removes_nearby_transitions(self):
        onset_bins = np.array([10, 12, 20, 40, 42])
        train_idx = np.array([0, 1, 2])  # onsets at 10, 12, 20
        test_idx = np.array([3, 4])       # onsets at 40, 42
        # embargo=5: exclude train transitions within 5 bins of test (40, 42)
        # 10: dist to 40=30, 42=32 → keep
        # 12: dist to 40=28, 42=30 → keep
        # 20: dist to 40=20, 42=22 → keep
        result = _apply_embargo(onset_bins, train_idx, test_idx, embargo_bins=5)
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_embargo_excludes_close_transitions(self):
        onset_bins = np.array([10, 38, 40, 42, 50])
        train_idx = np.array([0, 1])  # onsets at 10, 38
        test_idx = np.array([2, 3, 4])  # onsets at 40, 42, 50
        # embargo=4: exclude if dist <= 4
        # 10: min dist to {40,42,50} = 30 → keep
        # 38: min dist to {40,42,50} = 2 → exclude (<=4)
        result = _apply_embargo(onset_bins, train_idx, test_idx, embargo_bins=4)
        np.testing.assert_array_equal(result, np.array([0]))


# ---------------------------------------------------------------------------
# Tests: compute_tgm
# ---------------------------------------------------------------------------

class TestComputeTGM:
    def test_output_shape(self):
        X, y, onsets, post = _make_synthetic_data(n_bins=300, n_units=20)
        pre, post_bins = 4, 5
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=pre, post_bins=post_bins,
            n_folds=3, embargo_bins=2, n_pca=10,
        )
        window = pre + post_bins
        assert result.matrix.shape == (window, window)
        assert result.diagonal.shape == (window,)
        assert len(result.tau_sec) == window
        assert result.n_classes > 1

    def test_returns_tgm_result(self):
        X, y, onsets, post = _make_synthetic_data()
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=4, post_bins=5, n_folds=3, n_pca=10,
        )
        assert isinstance(result, TGMResult)
        assert result.n_transitions > 0
        assert result.n_folds > 0
        assert 0.0 < result.chance_level < 1.0

    def test_diagonal_above_chance_with_signal(self):
        """With strong signal, diagonal should exceed chance."""
        X, y, onsets, post = _make_synthetic_data(
            n_bins=1000, n_units=30, signal_strength=3.0,
        )
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=4, post_bins=5, n_folds=5, n_pca=20,
        )
        # At least some diagonal entries should beat chance
        assert np.mean(result.diagonal) > result.chance_level

    def test_shuffled_labels_near_chance(self):
        """With shuffled labels, accuracy should be near chance."""
        X, y, onsets, post = _make_synthetic_data(
            n_bins=500, n_units=20, signal_strength=3.0,
        )
        rng = np.random.RandomState(99)
        shuffled_post = rng.permutation(post)
        result = compute_tgm(
            X, y, onsets, shuffled_post,
            pre_bins=4, post_bins=5, n_folds=3, n_pca=10,
        )
        # Mean should be within 0.15 of chance (some noise expected)
        assert abs(np.mean(result.matrix) - result.chance_level) < 0.15

    def test_no_pca(self):
        """TGM should work without PCA."""
        X, y, onsets, post = _make_synthetic_data(n_bins=300, n_units=10)
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=4, post_bins=5, n_folds=3, n_pca=0,
        )
        assert result.matrix.shape == (9, 9)

    def test_pca_capped_at_n_units(self):
        """n_pca > n_units should not crash."""
        X, y, onsets, post = _make_synthetic_data(n_bins=300, n_units=10)
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=4, post_bins=5, n_folds=3, n_pca=100,
        )
        assert result.matrix.shape == (9, 9)

    def test_too_few_transitions_raises(self):
        """Should raise if not enough transitions for CV."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.ones(100, dtype=int)
        onsets = np.array([50], dtype=int)
        post = np.array([1], dtype=int)
        with pytest.raises(ValueError, match="Too few valid transitions"):
            compute_tgm(X, y, onsets, post, n_folds=5)

    def test_per_fold_matrices_stored(self):
        X, y, onsets, post = _make_synthetic_data(n_bins=300, n_units=20)
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=4, post_bins=5, n_folds=3, n_pca=10,
        )
        assert len(result.per_fold_matrices) == result.n_folds
        for fm in result.per_fold_matrices:
            assert fm.shape == result.matrix.shape

    def test_tau_sec_values(self):
        """tau_sec should be centered on onset at 0."""
        X, y, onsets, post = _make_synthetic_data(n_bins=300, n_units=10)
        result = compute_tgm(
            X, y, onsets, post,
            pre_bins=4, post_bins=5, n_folds=3, n_pca=0,
        )
        expected = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(result.tau_sec, expected)
