"""Tests for the Figure 5 coarsened affordance analysis.

Tests stay-vs-switch future labeling, Dirichlet-smoothed future distributions,
and coarse affordance alignment (Spearman rho between neural distance and
future JS divergence).
"""
from __future__ import annotations

import numpy as np

from freemat.single_fm.encoding.fig5_coarse_affordance import (
    CoarseAffordanceConfig,
    compute_coarse_affordance_alignment,
    compute_coarse_future_labels,
    compute_stay_switch_distribution,
)


# ── Coarse Future Labels ───────────────────────────────────────────────────


class TestCoarseFutureLabels:
    """Tests for compute_coarse_future_labels()."""

    def test_switch_detected_within_window(self) -> None:
        """Bins near a motif change should be labeled 'switch'."""
        # [1,1,1, 2,2,2] — bins 0-2 are motif 1, bins 3-5 are motif 2
        y_motif = np.array([1, 1, 1, 2, 2, 2])
        result = compute_coarse_future_labels(y_motif, window_bins=2)
        stay_switch = result["stay_vs_switch"]

        # Bin 0: motif changes at bin 3, which is 3 bins away (> window=2) → stay
        assert stay_switch[0] == 0
        # Bin 1: motif changes at bin 3, 2 bins away (== window) → switch
        assert stay_switch[1] == 1
        # Bin 2: motif changes at bin 3, 1 bin away (< window) → switch
        assert stay_switch[2] == 1
        # Bin 3: motif changes at bin 3 itself (motif=2 here). Next change
        # would be end of array. Within window_bins=2, bins 4,5 are still
        # motif 2 → stay
        assert stay_switch[3] == 0

    def test_no_switch_all_same_motif(self) -> None:
        """All bins with same motif should be labeled 'stay'."""
        y_motif = np.array([3, 3, 3, 3, 3])
        result = compute_coarse_future_labels(y_motif, window_bins=4)
        np.testing.assert_array_equal(result["stay_vs_switch"], np.zeros(5))

    def test_end_of_array_bins_are_stay(self) -> None:
        """Bins near the end with insufficient lookahead default to 'stay'."""
        y_motif = np.array([1, 1, 2])
        result = compute_coarse_future_labels(y_motif, window_bins=2)
        # Last bin (index 2): no future bins to check → stay
        assert result["stay_vs_switch"][2] == 0

    def test_motif_zero_ignored_for_switch(self) -> None:
        """Transitions through motif 0 (unlabeled) should still count."""
        y_motif = np.array([1, 1, 0, 0, 2, 2])
        result = compute_coarse_future_labels(y_motif, window_bins=3)
        # Bin 1: within 3 bins, motif changes at bin 2 (1→0) → switch
        assert result["stay_vs_switch"][1] == 1


# ── Stay/Switch Distribution ──────────────────────────────────────────────


class TestStaySwitchDistribution:
    """Tests for compute_stay_switch_distribution()."""

    def test_distribution_sums_to_one(self) -> None:
        """Dirichlet-smoothed distribution should sum to 1.0."""
        y_motif = np.array([1, 1, 1, 2, 2, 1, 1, 1, 2, 2])
        height_tier = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        future = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0])

        dist = compute_stay_switch_distribution(
            future_labels=future,
            y_motif=y_motif,
            height_tier=height_tier,
            motif_id=1,
            tier_id=0,
            alpha=1.0,
        )

        assert dist.shape == (2,)
        np.testing.assert_allclose(dist.sum(), 1.0)

    def test_pure_switch_state(self) -> None:
        """State where all bins are 'switch' should have high switch prob."""
        y_motif = np.array([1, 1, 1])
        height_tier = np.array([0, 0, 0])
        future = np.array([1, 1, 1])  # all switch

        dist = compute_stay_switch_distribution(
            future_labels=future,
            y_motif=y_motif,
            height_tier=height_tier,
            motif_id=1,
            tier_id=0,
            alpha=1.0,
        )

        # With alpha=1 smoothing and 3 switch observations:
        # stay = 1/(1+4) = 0.2, switch = 4/(1+4) = 0.8
        assert dist[1] > dist[0]

    def test_empty_state_returns_uniform(self) -> None:
        """State with no matching bins should return uniform."""
        y_motif = np.array([2, 2, 2])
        height_tier = np.array([0, 0, 0])
        future = np.array([0, 1, 0])

        dist = compute_stay_switch_distribution(
            future_labels=future,
            y_motif=y_motif,
            height_tier=height_tier,
            motif_id=1,  # not present
            tier_id=0,
            alpha=1.0,
        )

        np.testing.assert_allclose(dist, [0.5, 0.5])


# ── Coarse Affordance Alignment ──────────────────────────────────────────


class TestCoarseAffordanceAlignment:
    """Tests for compute_coarse_affordance_alignment()."""

    def test_positive_signal_synthetic(self) -> None:
        """Neural distance should correlate with future divergence."""
        rng = np.random.default_rng(42)

        # Motif 1, tier 0: mostly stay (stable state, cluster at origin)
        # Motif 1, tier 1: mostly switch (unstable state, cluster at +10)
        # → large neural distance, large future JS divergence
        #
        # Motif 2, tier 0: mixed stay/switch (cluster at +20)
        # Motif 2, tier 1: mixed stay/switch (cluster at +20.5)
        # → small neural distance, small future JS divergence

        n = 60
        emb_blocks, y_blocks, tier_blocks, future_blocks = [], [], [], []

        # Motif 1, tier 0: mostly stay
        emb_blocks.append(rng.standard_normal((n, 3)) * 0.3)
        y_blocks.append(np.ones(n, dtype=int))
        tier_blocks.append(np.zeros(n, dtype=int))
        future_blocks.append(np.zeros(n, dtype=int))  # all stay

        # Motif 1, tier 1: mostly switch
        emb_blocks.append(rng.standard_normal((n, 3)) * 0.3 + 10.0)
        y_blocks.append(np.ones(n, dtype=int))
        tier_blocks.append(np.ones(n, dtype=int))
        future_blocks.append(np.ones(n, dtype=int))  # all switch

        # Motif 2, tier 0: half stay / half switch
        emb_blocks.append(rng.standard_normal((n, 3)) * 0.3 + 20.0)
        y_blocks.append(np.full(n, 2, dtype=int))
        tier_blocks.append(np.zeros(n, dtype=int))
        half = n // 2
        future_blocks.append(
            np.concatenate([np.zeros(half), np.ones(n - half)]).astype(int)
        )

        # Motif 2, tier 1: half stay / half switch (same future → small JS)
        emb_blocks.append(rng.standard_normal((n, 3)) * 0.3 + 20.5)
        y_blocks.append(np.full(n, 2, dtype=int))
        tier_blocks.append(np.ones(n, dtype=int))
        future_blocks.append(
            np.concatenate([np.zeros(half), np.ones(n - half)]).astype(int)
        )

        embedding = np.vstack(emb_blocks)
        y_motif = np.concatenate(y_blocks)
        height_tier = np.concatenate(tier_blocks)
        future_labels = np.concatenate(future_blocks)

        config = CoarseAffordanceConfig(min_bins=10, min_state_pairs=2)
        result = compute_coarse_affordance_alignment(
            embedding=embedding,
            y_motif=y_motif,
            height_tier=height_tier,
            future_labels=future_labels,
            config=config,
        )

        assert result is not None
        assert result.n_pairs >= 2
        # Motif 1 pair has large neural dist + large JS; motif 2 has small + small
        # → expect positive correlation
        assert result.rho > 0

    def test_insufficient_pairs_returns_none(self) -> None:
        """With only one motif having multi-tier data, should return None."""
        rng = np.random.default_rng(42)
        n = 30

        embedding = rng.standard_normal((n, 3))
        y_motif = np.ones(n, dtype=int)
        height_tier = np.array([0] * 15 + [1] * 15)
        future_labels = np.zeros(n, dtype=int)

        config = CoarseAffordanceConfig(min_bins=10, min_state_pairs=3)
        result = compute_coarse_affordance_alignment(
            embedding=embedding,
            y_motif=y_motif,
            height_tier=height_tier,
            future_labels=future_labels,
            config=config,
        )

        # Only 1 motif → 1 pair, but min_state_pairs=3 → None
        assert result is None

    def test_two_pairs_returns_none_with_default_config(self) -> None:
        """Two pairs is below the default min_state_pairs=3."""
        rng = np.random.default_rng(99)
        n = 30

        # 2 motifs × 2 tiers = 2 pairs, below default threshold of 3
        embedding = np.vstack([
            rng.standard_normal((n, 3)),
            rng.standard_normal((n, 3)) + 5.0,
            rng.standard_normal((n, 3)) + 10.0,
            rng.standard_normal((n, 3)) + 15.0,
        ])
        y_motif = np.array([1] * n + [1] * n + [2] * n + [2] * n)
        height_tier = np.array([0] * n + [1] * n + [0] * n + [1] * n)
        future_labels = np.zeros(4 * n, dtype=int)

        config = CoarseAffordanceConfig(min_bins=10)  # default min_state_pairs=3
        result = compute_coarse_affordance_alignment(
            embedding=embedding,
            y_motif=y_motif,
            height_tier=height_tier,
            future_labels=future_labels,
            config=config,
        )

        assert result is None

    def test_single_tier_motif_excluded(self) -> None:
        """Motifs present in only one tier should be excluded from pairs.

        With only 1 qualifying motif across 2 tiers, we get 1 pair.
        Spearman on a single pair is undefined (returns NaN), so the
        non-finite guard correctly returns None.
        """
        rng = np.random.default_rng(42)
        n = 30

        # Motif 1: both tiers (qualifies → 1 pair)
        # Motif 2: tier 0 only (excluded)
        embedding = rng.standard_normal((3 * n, 3))
        y_motif = np.array([1] * n + [1] * n + [2] * n)
        height_tier = np.array([0] * n + [1] * n + [0] * n)
        future_labels = np.zeros(3 * n, dtype=int)

        # min_state_pairs=1 lets the single pair through the count guard,
        # but Spearman on 1 observation is non-finite → returns None
        config = CoarseAffordanceConfig(min_bins=10, min_state_pairs=1)
        result = compute_coarse_affordance_alignment(
            embedding=embedding,
            y_motif=y_motif,
            height_tier=height_tier,
            future_labels=future_labels,
            config=config,
        )

        assert result is None
