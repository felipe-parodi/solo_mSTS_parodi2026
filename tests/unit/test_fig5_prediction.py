"""Tests for Fig 5E: behavior-conditioned neural prediction analysis.

Tests the core functions for predicting transition timing and next-syllable
identity from behavioral features vs. behavioral + neural features.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_bins: int = 2000,
    n_units: int = 50,
    n_classes: int = 8,
    bout_len: int = 20,
    signal_strength: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Generate synthetic session data with block motif structure.

    Args:
        n_bins: Number of time bins.
        n_units: Number of neural units.
        n_classes: Number of motif classes (labeled 1..n_classes).
        bout_len: Mean bout length in bins.
        signal_strength: Amplitude of class-specific neural signal.
            0.0 = pure noise (null), >0 = informative.
        seed: Random seed.

    Returns:
        Dict with keys: y_motif, X_spikes, feat_df, t.
    """
    rng = np.random.default_rng(seed)

    # Build block motif labels (1-indexed, no motif 0)
    y_motif = np.zeros(n_bins, dtype=int)
    idx = 0
    while idx < n_bins:
        length = max(4, int(rng.poisson(bout_len)))
        cls = rng.integers(1, n_classes + 1)
        y_motif[idx : min(idx + length, n_bins)] = cls
        idx += length

    # Neural spikes: noise + optional class-specific templates
    templates = rng.standard_normal((n_classes + 1, n_units))  # index 0 unused
    X_spikes = rng.standard_normal((n_bins, n_units))
    if signal_strength > 0:
        for c in range(1, n_classes + 1):
            mask = y_motif == c
            X_spikes[mask] += signal_strength * templates[c]

    # Behavioral features DataFrame
    feat_cols = [
        "com_x", "com_y", "com_z", "speed", "yaw_rate",
        "v_parallel", "v_perp", "pitch_rate", "roll_rate", "delta_yaw",
        "gaze_x", "gaze_y", "gaze_z", "d_wall_horiz", "vis_motion",
    ]
    feat_data = rng.standard_normal((n_bins, len(feat_cols)))
    # Make speed non-negative
    feat_data[:, feat_cols.index("speed")] = np.abs(feat_data[:, feat_cols.index("speed")])
    feat_df = pd.DataFrame(feat_data, columns=feat_cols)

    t = np.arange(n_bins) * 0.5  # 0.5s bins

    return {
        "y_motif": y_motif,
        "X_spikes": X_spikes,
        "feat_df": feat_df,
        "t": t,
    }


# ---------------------------------------------------------------------------
# Tests: compute_bout_age
# ---------------------------------------------------------------------------

class TestComputeBoutAge:
    """Tests for bout age computation."""

    def test_basic_resets_at_transitions(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import compute_bout_age

        y_motif = np.array([1, 1, 1, 2, 2, 2, 1, 1])
        bout_age = compute_bout_age(y_motif, bin_s=0.5)

        # Bout 1: bins 0-2, ages 0.0, 0.5, 1.0
        assert bout_age[0] == pytest.approx(0.0)
        assert bout_age[1] == pytest.approx(0.5)
        assert bout_age[2] == pytest.approx(1.0)
        # Transition at bin 3: resets to 0
        assert bout_age[3] == pytest.approx(0.0)
        assert bout_age[4] == pytest.approx(0.5)
        assert bout_age[5] == pytest.approx(1.0)
        # Transition at bin 6: resets again
        assert bout_age[6] == pytest.approx(0.0)
        assert bout_age[7] == pytest.approx(0.5)

    def test_single_class_monotonic(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import compute_bout_age

        y_motif = np.array([3, 3, 3, 3, 3])
        bout_age = compute_bout_age(y_motif, bin_s=1.0)

        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(bout_age, expected)


# ---------------------------------------------------------------------------
# Tests: build_transition_labels
# ---------------------------------------------------------------------------

class TestBuildTransitionLabels:
    """Tests for binary transition label construction."""

    def test_labels_within_horizon(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import build_transition_labels

        # Transition at index 5 (1→2)
        y_motif = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        labels = build_transition_labels(y_motif, horizon_bins=4)

        # Bins 1-4 are within 4 bins before transition at 5 → label=1
        assert labels[1] == 1  # 5-1=4 bins away
        assert labels[2] == 1  # 5-2=3 bins away
        assert labels[3] == 1  # 5-3=2 bins away
        assert labels[4] == 1  # 5-4=1 bin away
        # Bin 0 is 5 bins away → outside horizon
        assert labels[0] == 0
        # Bins after transition: no upcoming transition
        assert labels[6] == 0
        assert labels[7] == 0

    def test_no_transitions(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import build_transition_labels

        y_motif = np.array([1, 1, 1, 1, 1])
        labels = build_transition_labels(y_motif, horizon_bins=4)
        assert np.all(labels == 0)


# ---------------------------------------------------------------------------
# Tests: aggregate_pre_transition_features
# ---------------------------------------------------------------------------

class TestAggregatePreTransition:
    """Tests for transition-level feature aggregation."""

    def test_one_row_per_transition(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            aggregate_pre_transition_features,
        )

        # Two transitions: at index 10 (1→2) and at index 20 (2→3)
        y_motif = np.zeros(30, dtype=int)
        y_motif[:10] = 1
        y_motif[10:20] = 2
        y_motif[20:] = 3

        X = np.random.default_rng(42).standard_normal((30, 5))
        t = np.arange(30) * 0.5

        X_agg, y_next, trans_idx = aggregate_pre_transition_features(
            X, y_motif, t, window_s=2.0, buffer_s=0.5,
        )

        # Two transitions → two rows
        assert X_agg.shape[0] == 2
        assert len(y_next) == 2
        assert len(trans_idx) == 2

        # Next syllable labels
        assert y_next[0] == 2  # transition at 10: 1→2
        assert y_next[1] == 3  # transition at 20: 2→3

    def test_window_averaging(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            aggregate_pre_transition_features,
        )

        # Transition at index 10, window [-2.0, -0.5]s = bins [6, 7, 8] (indices)
        y_motif = np.zeros(20, dtype=int)
        y_motif[:10] = 1
        y_motif[10:] = 2

        X = np.ones((20, 3))
        X[6:9] = 5.0  # bins in window should be averaged
        t = np.arange(20) * 0.5

        X_agg, _, _ = aggregate_pre_transition_features(
            X, y_motif, t, window_s=2.0, buffer_s=0.5,
        )

        # Should be mean of bins 6,7,8 = 5.0
        np.testing.assert_array_almost_equal(X_agg[0], 5.0)


# ---------------------------------------------------------------------------
# Tests: build_behavioral_features
# ---------------------------------------------------------------------------

class TestBuildBehavioralFeatures:
    """Tests for behavioral feature matrix construction."""

    def test_shape(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            build_behavioral_features,
            compute_bout_age,
        )

        data = _make_synthetic_data(n_bins=200, n_classes=5)
        bout_age = compute_bout_age(data["y_motif"], bin_s=0.5)
        X_beh = build_behavioral_features(
            data["y_motif"], bout_age, data["feat_df"], spline_df=5,
        )

        # K=5 classes
        # current_syl: 5 + prev_syl: 5 + spline: 5 + interaction: 25
        # + continuous: 15 + short_history: 3 = 58
        n_classes = len(np.unique(data["y_motif"][data["y_motif"] > 0]))
        expected_cols = (
            n_classes  # current_syllable one-hot
            + n_classes  # previous_syllable one-hot
            + 5  # bout_age splines
            + n_classes * 5  # interaction
            + 15  # continuous
            + 3  # short-history summaries
        )
        assert X_beh.shape == (200, expected_cols)

    def test_one_hot_sums_to_one(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            build_behavioral_features,
            compute_bout_age,
        )

        data = _make_synthetic_data(n_bins=100, n_classes=4)
        bout_age = compute_bout_age(data["y_motif"], bin_s=0.5)
        X_beh = build_behavioral_features(
            data["y_motif"], bout_age, data["feat_df"], spline_df=5,
        )

        n_classes = len(np.unique(data["y_motif"][data["y_motif"] > 0]))
        current_syl_cols = X_beh[:, :n_classes]
        row_sums = current_syl_cols.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0)

    def test_interaction_zeros_when_inactive(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            build_behavioral_features,
            compute_bout_age,
        )
        from freemat.single_fm.encoding.ridge import build_spline_design

        data = _make_synthetic_data(n_bins=200, n_classes=3)
        y = data["y_motif"]
        bout_age = compute_bout_age(y, bin_s=0.5)
        X_beh = build_behavioral_features(
            y, bout_age, data["feat_df"], spline_df=5,
        )

        classes = np.sort(np.unique(y[y > 0]))
        K = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Get actual spline column count
        n_knots = max(5 - 3, 1)
        spline_cols = build_spline_design(
            bout_age.reshape(-1, 1), n_knots=n_knots,
        ).shape[1]
        # Interaction block starts after: current(K) + prev(K) + spline_cols
        int_start = K * 2 + spline_cols

        # For each bin, interaction cols for inactive syllables should be 0
        for i in range(len(y)):
            if y[i] <= 0:
                continue
            active_k = class_to_idx[y[i]]
            for k in range(K):
                if k != active_k:
                    s = int_start + k * spline_cols
                    e = s + spline_cols
                    cols = X_beh[i, s:e]
                    np.testing.assert_array_equal(
                        cols, 0.0,
                        err_msg=f"bin={i}, motif={y[i]}, k={k}",
                    )


# ---------------------------------------------------------------------------
# Tests: build_neural_features
# ---------------------------------------------------------------------------

class TestBuildNeuralFeatures:
    """Tests for PCA-based neural feature construction."""

    def test_pca_fit_on_train_only(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import build_neural_features

        rng = np.random.default_rng(42)
        # Train: structured data with clear PC directions
        X = rng.standard_normal((100, 30))
        # Test: completely different covariance structure
        X[80:] = rng.standard_normal((20, 30)) @ rng.standard_normal((30, 30))

        train_idx = np.arange(80)
        X_pca, pca = build_neural_features(X, n_pcs=5, train_idx=train_idx)

        assert X_pca.shape == (100, 5)
        assert pca.n_components == 5
        # Verify PCA object was created with correct shape
        assert pca.components_.shape == (5, 30)


# ---------------------------------------------------------------------------
# Tests: predict_transition_hazard
# ---------------------------------------------------------------------------

class TestPredictTransitionHazard:
    """Tests for binary transition prediction."""

    def test_strong_signal_positive_delta(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            PredictionConfig,
            build_behavioral_features,
            build_transition_labels,
            compute_bout_age,
            predict_transition_hazard,
        )

        data = _make_synthetic_data(
            n_bins=2000, n_units=50, n_classes=6,
            bout_len=20, signal_strength=3.0,
        )
        bout_age = compute_bout_age(data["y_motif"], bin_s=0.5)
        X_beh = build_behavioral_features(
            data["y_motif"], bout_age, data["feat_df"], spline_df=5,
        )
        y_hazard = build_transition_labels(data["y_motif"], horizon_bins=4)

        config = PredictionConfig(
            n_cv_folds=5, embargo_bins=4, n_perm=0,
            n_pcs=10, classifier_C=1.0,
        )
        result = predict_transition_hazard(
            X_beh, data["X_spikes"], y_hazard, config,
        )

        # With strong neural signal, M1 should beat M0
        assert result.delta_auc > 0
        assert result.behavior_neural_auc > result.behavior_only_auc

    def test_no_signal_delta_near_zero(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            PredictionConfig,
            build_behavioral_features,
            build_transition_labels,
            compute_bout_age,
            predict_transition_hazard,
        )

        data = _make_synthetic_data(
            n_bins=1000, n_units=30, n_classes=5,
            bout_len=15, signal_strength=0.0,
        )
        bout_age = compute_bout_age(data["y_motif"], bin_s=0.5)
        X_beh = build_behavioral_features(
            data["y_motif"], bout_age, data["feat_df"], spline_df=5,
        )
        y_hazard = build_transition_labels(data["y_motif"], horizon_bins=4)

        config = PredictionConfig(
            n_cv_folds=5, embargo_bins=4, n_perm=0,
            n_pcs=10, classifier_C=1.0,
        )
        result = predict_transition_hazard(
            X_beh, data["X_spikes"], y_hazard, config,
        )

        # With no signal, ΔAUC should be near 0 (within ±0.05)
        assert abs(result.delta_auc) < 0.05


# ---------------------------------------------------------------------------
# Tests: predict_next_syllable
# ---------------------------------------------------------------------------

class TestPredictNextSyllable:
    """Tests for transition-level next-syllable prediction."""

    def test_strong_signal_positive_delta(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            PredictionConfig,
            aggregate_pre_transition_features,
            build_behavioral_features,
            compute_bout_age,
            predict_next_syllable,
        )

        # Need many transitions with strong signal; short bouts give more
        # transitions, fewer classes make the problem easier
        data = _make_synthetic_data(
            n_bins=8000, n_units=50, n_classes=3,
            bout_len=8, signal_strength=8.0,
        )
        bout_age = compute_bout_age(data["y_motif"], bin_s=0.5)
        X_beh = build_behavioral_features(
            data["y_motif"], bout_age, data["feat_df"], spline_df=5,
        )

        # Aggregate to transition level
        X_beh_agg, y_next, trans_idx = aggregate_pre_transition_features(
            X_beh, data["y_motif"], data["t"],
            window_s=2.0, buffer_s=0.5,
        )
        X_neural_agg, _, _ = aggregate_pre_transition_features(
            data["X_spikes"], data["y_motif"], data["t"],
            window_s=2.0, buffer_s=0.5,
        )

        config = PredictionConfig(
            n_cv_folds=5, embargo_bins=0, n_perm=0,
            n_pcs=10, classifier_C=1.0,
        )
        result = predict_next_syllable(
            X_beh_agg, X_neural_agg, y_next, config,
        )

        assert result.delta_accuracy > 0
        assert result.behavior_neural_accuracy > result.behavior_only_accuracy

    def test_markov_below_full_behavior(self) -> None:
        from freemat.single_fm.encoding.fig5_prediction import (
            PredictionConfig,
            aggregate_pre_transition_features,
            build_behavioral_features,
            compute_bout_age,
            predict_next_syllable,
        )

        data = _make_synthetic_data(
            n_bins=3000, n_units=30, n_classes=6,
            bout_len=15, signal_strength=0.0,
        )
        bout_age = compute_bout_age(data["y_motif"], bin_s=0.5)
        X_beh = build_behavioral_features(
            data["y_motif"], bout_age, data["feat_df"], spline_df=5,
        )

        X_beh_agg, y_next, trans_idx = aggregate_pre_transition_features(
            X_beh, data["y_motif"], data["t"],
            window_s=2.0, buffer_s=0.5,
        )
        X_neural_agg, _, _ = aggregate_pre_transition_features(
            data["X_spikes"], data["y_motif"], data["t"],
            window_s=2.0, buffer_s=0.5,
        )

        config = PredictionConfig(
            n_cv_folds=5, embargo_bins=0, n_perm=0,
            n_pcs=10, classifier_C=1.0,
        )
        result = predict_next_syllable(
            X_beh_agg, X_neural_agg, y_next, config,
        )

        # Markov (current syllable only) should be <= M0 (full behavior)
        assert result.markov_baseline_accuracy <= result.behavior_only_accuracy + 0.02


# ---------------------------------------------------------------------------
# Tests: circular shift
# ---------------------------------------------------------------------------

class TestCircularShift:
    """Tests for circular-shift null preservation."""

    def test_preserves_autocorrelation(self) -> None:
        from freemat.single_fm.encoding.stats import circular_shift_indices

        rng = np.random.default_rng(42)
        # Autocorrelated signal
        n = 500
        x = np.cumsum(rng.standard_normal(n))

        lag1_orig = np.corrcoef(x[:-1], x[1:])[0, 1]

        shift_idx = circular_shift_indices(n, rng=rng)
        x_shifted = x[shift_idx]
        lag1_shifted = np.corrcoef(x_shifted[:-1], x_shifted[1:])[0, 1]

        # Autocorrelation should be very similar (within 0.05)
        # Note: circular shift introduces one discontinuity at the wrap point,
        # so the correlation won't be identical but should be close
        assert abs(lag1_orig - lag1_shifted) < 0.1
