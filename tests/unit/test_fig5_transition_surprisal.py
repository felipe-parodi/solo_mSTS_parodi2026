"""Tests for the Figure 5 transition-level surprisal analysis.

Tests cover:
- Transition extraction with stability and window constraints
- Motif filtering by incoming transition support
- Behavioral feature construction for transition-level model
- Behavior-only model fitting and out-of-fold probabilities
- Surprisal computation and z-scoring
- Neural response extraction (delta_fr)
- Per-unit surprisal regression
- Session-level null (permutation within groups)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freemat.single_fm.encoding.fig5_transition_surprisal import (
    BehaviorModelResult,
    TransitionDataset,
    TransitionSurprisalConfig,
    build_transition_features,
    compute_neural_responses,
    compute_surprisal,
    extract_valid_transitions,
    filter_motifs_by_transition_support,
    fit_behavior_model,
    fit_surprisal_regression,
    run_session_null,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_motif_sequence(
    pattern: list[tuple[int, int]],
    bin_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a motif sequence from (motif_id, n_bins) runs.

    Args:
        pattern: List of (motif_id, duration_in_bins).
        bin_s: Bin size in seconds.

    Returns:
        (y_motif, bin_times, height_tier)
    """
    y_motif_parts: list[np.ndarray] = []
    for motif_id, n_bins in pattern:
        y_motif_parts.append(np.full(n_bins, motif_id, dtype=int))
    y_motif = np.concatenate(y_motif_parts)
    bin_times = np.arange(len(y_motif), dtype=float) * bin_s
    # Simple height tier: deterministic from position
    height_tier = np.zeros(len(y_motif), dtype=int)
    return y_motif, bin_times, height_tier


# ---------------------------------------------------------------------------
# Phase 1: Transition extraction
# ---------------------------------------------------------------------------


class TestExtractValidTransitions:
    """Tests for extract_valid_transitions()."""

    def test_basic_transitions_detected(self) -> None:
        """Adjacent motif changes should be detected as transitions."""
        # 10 bins motif 1, 10 bins motif 2, 10 bins motif 3
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 10), (2, 10), (3, 10)]
        )
        config = TransitionSurprisalConfig(bin_s=0.5)
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        # Two transitions: 1->2 at bin 10, 2->3 at bin 20
        assert isinstance(ds, TransitionDataset)
        assert len(ds.trans_bins) == 2
        np.testing.assert_array_equal(ds.current_motif, [1, 2])
        np.testing.assert_array_equal(ds.next_motif, [2, 3])

    def test_stability_filter_removes_rapid_transitions(self) -> None:
        """Transitions preceded by another transition within pre-window are excluded."""
        # Motif 1 for 2 bins, motif 2 for 2 bins, motif 3 for 10 bins
        # Transition at bin 2 (1->2): pre-window [-2.0s,-0.5s] = bins [-2, 1]
        # overlaps start → excluded by window clipping
        # Transition at bin 4 (2->3): pre-window [-2.0s,-0.5s] = bins [0, 3]
        # motif changes at bin 2 within window → excluded by stability
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 2), (2, 2), (3, 10)]
        )
        config = TransitionSurprisalConfig(bin_s=0.5)
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        # Both transitions should be excluded
        assert len(ds.trans_bins) == 0

    def test_window_clipping_excludes_edge_transitions(self) -> None:
        """Transitions too close to start or end of session are excluded."""
        # Transition at bin 2 — pre window needs bins [-2, 1], but bin -2 doesn't exist
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 2), (2, 10)]
        )
        config = TransitionSurprisalConfig(bin_s=0.5)
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        assert len(ds.trans_bins) == 0

    def test_missing_motifs_excluded(self) -> None:
        """Transitions involving motif 0 (unlabeled) should be excluded."""
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 10), (0, 5), (2, 10)]
        )
        config = TransitionSurprisalConfig(bin_s=0.5)
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        # 1->0 at bin 10: next_motif=0 → excluded
        # 0->2 at bin 15: current_motif=0 → excluded
        assert len(ds.trans_bins) == 0

    def test_previous_motif_extracted(self) -> None:
        """previous_motif should be the motif before the current bout."""
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 10), (2, 10), (3, 10)]
        )
        config = TransitionSurprisalConfig(bin_s=0.5)
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        # First transition (1->2): previous motif is 0 (start of session)
        # Second transition (2->3): previous motif is 1
        assert ds.previous_motif[1] == 1

    def test_bout_age_computed(self) -> None:
        """bout_age should reflect time in current motif before transition."""
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 10), (2, 10), (3, 10)]
        )
        config = TransitionSurprisalConfig(bin_s=0.5)
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        # Second transition at bin 20: motif 2 started at bin 10,
        # bout_age = 10 bins * 0.5s = 5.0s at transition
        assert ds.bout_age[1] == pytest.approx(4.5, abs=0.5)

    def test_index_arrays_have_correct_shape(self) -> None:
        """Window index arrays should have correct number of bins."""
        y_motif, bin_times, height_tier = _make_motif_sequence(
            [(1, 10), (2, 10), (3, 10)]
        )
        config = TransitionSurprisalConfig(
            bin_s=0.5,
            pre_feature_window_s=2.0,
            pre_buffer_s=0.5,
            pre_neural_window_s=1.0,
            post_neural_window_s=1.0,
        )
        ds = extract_valid_transitions(y_motif, bin_times, height_tier, config)

        n_trans = len(ds.trans_bins)
        assert n_trans > 0
        # pre_feature: [-2.0s, -0.5s] at 0.5s bins = 3 bins
        assert ds.pre_feature_idx.shape == (n_trans, 3)
        # pre_neural: [-1.0s, 0.0s] = 2 bins
        assert ds.pre_neural_idx.shape == (n_trans, 2)
        # post_neural: [0.0s, +1.0s] = 2 bins
        assert ds.post_neural_idx.shape == (n_trans, 2)


# ---------------------------------------------------------------------------
# Phase 1: Motif filtering
# ---------------------------------------------------------------------------


class TestFilterMotifsByTransitionSupport:
    """Tests for filter_motifs_by_transition_support()."""

    def test_motifs_with_enough_incoming_retained(self) -> None:
        """Motifs with >= min_incoming incoming transitions should be retained."""
        rng = np.random.default_rng(42)
        n = 100
        next_motif = rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])

        retained = filter_motifs_by_transition_support(
            next_motif, min_incoming=15
        )

        # All motifs should have enough with n=100 and these probabilities
        for m in retained:
            assert np.sum(next_motif == m) >= 15

    def test_rare_motifs_excluded(self) -> None:
        """Motifs with < min_incoming incoming transitions should be excluded."""
        next_motif = np.array([1] * 50 + [2] * 30 + [3] * 5)

        retained = filter_motifs_by_transition_support(
            next_motif, min_incoming=15
        )

        assert 1 in retained
        assert 2 in retained
        assert 3 not in retained


# ---------------------------------------------------------------------------
# Phase 2: Behavior model features
# ---------------------------------------------------------------------------


class TestBuildTransitionFeatures:
    """Tests for build_transition_features()."""

    def test_output_shape(self) -> None:
        """Feature matrix should have one row per transition."""
        n_trans = 20
        rng = np.random.default_rng(42)
        current_motif = rng.choice([1, 2, 3], size=n_trans)
        previous_motif = rng.choice([0, 1, 2, 3], size=n_trans)
        bout_age = rng.uniform(0.5, 10.0, size=n_trans)
        height_tier = rng.choice([0, 1, 2], size=n_trans)
        retained_motifs = [1, 2, 3]

        n_feat_cols = 14  # VIS + BODY_KIN + com_x + com_y
        pre_features = rng.standard_normal((n_trans, n_feat_cols))

        X = build_transition_features(
            current_motif, previous_motif, bout_age, height_tier,
            pre_features, retained_motifs,
        )

        assert X.shape[0] == n_trans
        assert X.shape[1] > 0
        assert np.all(np.isfinite(X))

    def test_one_hot_sums_to_one(self) -> None:
        """Current motif one-hot columns should sum to 1 per row."""
        n_trans = 20
        rng = np.random.default_rng(42)
        current_motif = rng.choice([1, 2, 3], size=n_trans)
        previous_motif = rng.choice([0, 1, 2, 3], size=n_trans)
        bout_age = rng.uniform(0.5, 10.0, size=n_trans)
        height_tier = rng.choice([0, 1, 2], size=n_trans)
        retained_motifs = [1, 2, 3]
        pre_features = rng.standard_normal((n_trans, 14))

        X = build_transition_features(
            current_motif, previous_motif, bout_age, height_tier,
            pre_features, retained_motifs,
        )

        # First K+1 cols are current motif one-hot (K retained + 1 other)
        n_current = len(retained_motifs) + 1
        current_block = X[:, :n_current]
        np.testing.assert_array_equal(current_block.sum(axis=1), np.ones(n_trans))


# ---------------------------------------------------------------------------
# Phase 2: Behavior model
# ---------------------------------------------------------------------------


class TestFitBehaviorModel:
    """Tests for fit_behavior_model()."""

    def test_probability_shape_and_normalization(self) -> None:
        """Out-of-fold probabilities should sum to 1 per transition."""
        rng = np.random.default_rng(42)
        n_trans = 100
        n_features = 20
        X = rng.standard_normal((n_trans, n_features))
        y_next = rng.choice([1, 2, 3], size=n_trans)
        trans_bins = np.arange(n_trans) * 5  # spread out

        config = TransitionSurprisalConfig(n_cv_folds=3, embargo_bins=2)
        result = fit_behavior_model(X, y_next, trans_bins, config)

        assert isinstance(result, BehaviorModelResult)
        assert result.p_hat.shape == (n_trans, 3)
        np.testing.assert_allclose(result.p_hat.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.p_markov.sum(axis=1), 1.0, atol=1e-6)

    def test_nll_is_finite(self) -> None:
        """NLL values should be finite numbers."""
        rng = np.random.default_rng(42)
        n_trans = 100
        X = rng.standard_normal((n_trans, 20))
        y_next = rng.choice([1, 2, 3], size=n_trans)
        trans_bins = np.arange(n_trans) * 5

        config = TransitionSurprisalConfig(n_cv_folds=3, embargo_bins=2)
        result = fit_behavior_model(X, y_next, trans_bins, config)

        assert np.isfinite(result.nll_full)
        assert np.isfinite(result.nll_markov)


# ---------------------------------------------------------------------------
# Phase 2: Surprisal
# ---------------------------------------------------------------------------


class TestComputeSurprisal:
    """Tests for compute_surprisal()."""

    def test_surprisal_positive(self) -> None:
        """Surprisal should be non-negative (probabilities <= 1)."""
        rng = np.random.default_rng(42)
        n = 50
        classes = np.array([1, 2, 3])
        p_hat = rng.dirichlet([2, 2, 2], size=n)
        p_markov = rng.dirichlet([1, 1, 1], size=n)
        y_next = rng.choice(classes, size=n)

        result = compute_surprisal(p_hat, p_markov, y_next, classes)

        assert np.all(result.surprisal >= 0)
        assert np.all(np.isfinite(result.surprisal))

    def test_z_surprisal_standardized(self) -> None:
        """Z-scored surprisal should have mean ~0 and std ~1."""
        rng = np.random.default_rng(42)
        n = 200
        classes = np.array([1, 2, 3])
        p_hat = rng.dirichlet([2, 2, 2], size=n)
        p_markov = rng.dirichlet([1, 1, 1], size=n)
        y_next = rng.choice(classes, size=n)

        result = compute_surprisal(p_hat, p_markov, y_next, classes)

        assert abs(result.z_surprisal.mean()) < 0.1
        assert abs(result.z_surprisal.std() - 1.0) < 0.1

    def test_low_prob_clipped(self) -> None:
        """Near-zero probabilities should not produce infinite surprisal."""
        classes = np.array([1, 2])
        # One transition where the observed class has near-zero probability
        p_hat = np.array([[1e-10, 1 - 1e-10]])
        p_markov = np.array([[0.5, 0.5]])
        y_next = np.array([1])  # class 1 has p=1e-10

        result = compute_surprisal(p_hat, p_markov, y_next, classes)

        assert np.all(np.isfinite(result.surprisal))

    def test_delta_surprisal_computed(self) -> None:
        """delta_surprisal should equal surprisal_full - surprisal_markov."""
        classes = np.array([1, 2, 3])
        p_hat = np.array([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]])
        p_markov = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
        y_next = np.array([1, 2])

        result = compute_surprisal(p_hat, p_markov, y_next, classes)

        expected_full = -np.log(np.array([0.6, 0.8]))
        expected_markov = -np.log(np.array([0.33, 0.33]))
        expected_delta = expected_full - expected_markov
        np.testing.assert_allclose(result.delta_surprisal, expected_delta, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 3: Neural response
# ---------------------------------------------------------------------------


class TestComputeNeuralResponses:
    """Tests for compute_neural_responses()."""

    def test_delta_fr_shape(self) -> None:
        """delta_fr should have shape (n_trans, n_units)."""
        n_bins, n_units = 100, 10
        rng = np.random.default_rng(42)
        X_spikes_z = rng.standard_normal((n_bins, n_units))

        # Fake transitions with 2-bin pre and 2-bin post windows
        pre_neural_idx = np.array([[20, 21], [50, 51]])
        post_neural_idx = np.array([[22, 23], [52, 53]])

        delta_fr = compute_neural_responses(
            X_spikes_z, pre_neural_idx, post_neural_idx
        )

        assert delta_fr.shape == (2, n_units)

    def test_delta_fr_is_post_minus_pre(self) -> None:
        """delta_fr should equal mean(post) - mean(pre) for each unit."""
        X = np.zeros((10, 2))
        X[2, 0] = 1.0  # pre bin, unit 0
        X[3, 0] = 1.0  # pre bin, unit 0
        X[4, 0] = 3.0  # post bin, unit 0
        X[5, 0] = 3.0  # post bin, unit 0

        pre_idx = np.array([[2, 3]])
        post_idx = np.array([[4, 5]])

        delta_fr = compute_neural_responses(X, pre_idx, post_idx)

        # post mean = 3.0, pre mean = 1.0, delta = 2.0
        assert delta_fr[0, 0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Phase 3: Per-unit regression
# ---------------------------------------------------------------------------


class TestFitSurprisalRegression:
    """Tests for fit_surprisal_regression()."""

    def test_output_has_required_columns(self) -> None:
        """Output should contain beta_surprisal, t_stat, p_value per unit."""
        rng = np.random.default_rng(42)
        n_trans, n_units = 80, 5
        delta_fr = rng.standard_normal((n_trans, n_units))
        z_surprisal = rng.standard_normal(n_trans)

        controls_df = pd.DataFrame({
            "current_motif": rng.choice([1, 2, 3], size=n_trans),
            "next_motif": rng.choice([1, 2, 3], size=n_trans),
            "height_tier": rng.choice([0, 1, 2], size=n_trans),
            "bout_age": rng.uniform(0.5, 10.0, size=n_trans),
            "post_speed": rng.uniform(0, 2, size=n_trans),
            "post_abs_yaw": rng.uniform(0, 3, size=n_trans),
            "pre_speed": rng.uniform(0, 2, size=n_trans),
            "pre_abs_yaw": rng.uniform(0, 3, size=n_trans),
        })

        result = fit_surprisal_regression(
            delta_fr, z_surprisal, controls_df,
            unit_ids=np.arange(n_units),
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_units
        assert "beta_surprisal" in result.columns
        assert "t_stat" in result.columns
        assert "p_value" in result.columns
        assert "unit_id" in result.columns

    def test_positive_beta_for_correlated_signal(self) -> None:
        """When delta_fr correlates with surprisal, beta should be positive."""
        rng = np.random.default_rng(42)
        n_trans = 200
        z_surprisal = rng.standard_normal(n_trans)

        # Unit 0: delta_fr = 2 * surprisal + noise
        delta_fr = np.column_stack([
            2.0 * z_surprisal + rng.normal(scale=0.3, size=n_trans),
            rng.standard_normal(n_trans),  # Unit 1: pure noise
        ])

        controls_df = pd.DataFrame({
            "current_motif": rng.choice([1, 2], size=n_trans),
            "next_motif": rng.choice([1, 2], size=n_trans),
            "height_tier": rng.choice([0, 1, 2], size=n_trans),
            "bout_age": rng.uniform(0.5, 10.0, size=n_trans),
            "post_speed": rng.uniform(0, 2, size=n_trans),
            "post_abs_yaw": rng.uniform(0, 3, size=n_trans),
            "pre_speed": rng.uniform(0, 2, size=n_trans),
            "pre_abs_yaw": rng.uniform(0, 3, size=n_trans),
        })

        result = fit_surprisal_regression(
            delta_fr, z_surprisal, controls_df,
            unit_ids=np.array([0, 1]),
        )

        # Unit 0 should have positive beta
        assert result.loc[result["unit_id"] == 0, "beta_surprisal"].values[0] > 0.5


# ---------------------------------------------------------------------------
# Phase 4: Session-level null
# ---------------------------------------------------------------------------


class TestRunSessionNull:
    """Tests for run_session_null()."""

    def test_null_distribution_length(self) -> None:
        """Null distribution should have n_perm values."""
        rng = np.random.default_rng(42)
        n_trans, n_units = 50, 5
        delta_fr = rng.standard_normal((n_trans, n_units))
        z_surprisal = rng.standard_normal(n_trans)
        current_motif = rng.choice([1, 2, 3], size=n_trans)
        next_motif = rng.choice([1, 2, 3], size=n_trans)

        controls_df = pd.DataFrame({
            "current_motif": current_motif,
            "next_motif": next_motif,
            "height_tier": rng.choice([0, 1, 2], size=n_trans),
            "bout_age": rng.uniform(0.5, 10.0, size=n_trans),
            "post_speed": rng.uniform(0, 2, size=n_trans),
            "post_abs_yaw": rng.uniform(0, 3, size=n_trans),
            "pre_speed": rng.uniform(0, 2, size=n_trans),
            "pre_abs_yaw": rng.uniform(0, 3, size=n_trans),
        })

        result = run_session_null(
            delta_fr, z_surprisal, controls_df,
            current_motif, next_motif,
            n_perm=10,
            min_pair_for_pair_shuffle=5,
        )

        assert len(result.null_medians) == 10
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_bounds(self) -> None:
        """P-value should be between 1/(n_perm+1) and 1.0."""
        rng = np.random.default_rng(42)
        n_trans, n_units = 50, 3
        delta_fr = rng.standard_normal((n_trans, n_units))
        z_surprisal = rng.standard_normal(n_trans)
        current_motif = rng.choice([1, 2], size=n_trans)
        next_motif = rng.choice([1, 2], size=n_trans)

        controls_df = pd.DataFrame({
            "current_motif": current_motif,
            "next_motif": next_motif,
            "height_tier": np.zeros(n_trans, dtype=int),
            "bout_age": rng.uniform(0.5, 10.0, size=n_trans),
            "post_speed": rng.uniform(0, 2, size=n_trans),
            "post_abs_yaw": rng.uniform(0, 3, size=n_trans),
            "pre_speed": rng.uniform(0, 2, size=n_trans),
            "pre_abs_yaw": rng.uniform(0, 3, size=n_trans),
        })

        n_perm = 20
        result = run_session_null(
            delta_fr, z_surprisal, controls_df,
            current_motif, next_motif,
            n_perm=n_perm,
            min_pair_for_pair_shuffle=5,
        )

        assert result.p_value >= 1.0 / (n_perm + 1)
        assert result.p_value <= 1.0
