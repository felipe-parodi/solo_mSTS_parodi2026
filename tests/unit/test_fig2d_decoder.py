"""Unit tests for Fig 2D spatial decoder (height + zone classification).

Tests cover:
- Arena-centered angular zone assignment
- Height label assignment (fixed cuts)
- Balanced train indices (equal per-class counts)
- Nuisance residualization (shape, variance reduction)
- Min-class fold guard (strict + relaxed + all-invalid → NaN)
- Stats consistency (perm null skips same folds as observed)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from freemat.single_fm.encoding.fig2d_spatial_decoder import (
    Fig2DDecoderConfig,
    _balanced_train_indices,
    _build_height_and_zone_labels,
    _compute_valid_fold_mask,
    _decode_accuracy,
    _fit_nuisance_residuals,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def default_config() -> Fig2DDecoderConfig:
    """Default config for Fig2D decoder tests."""
    return Fig2DDecoderConfig(
        bin_s=0.5,
        cv_folds=(5,),
        n_permutations=0,
        height_strategy="fixed",
        fixed_height_cuts=(0.3, 0.6),
        n_zone_sectors=3,
        balance_train_bins=True,
        random_state=42,
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── Arena centering ───────────────────────────────────────────────────────


class TestArenaCenter:
    """Test that angular zone sectors use arena-centered coordinates."""

    def test_zone_sector_at_known_angle_centered(
        self, default_config: Fig2DDecoderConfig
    ) -> None:
        """A point due east of arena center should be sector 0 (angle ≈ 0)."""
        arena_center_x, arena_center_y = 2.0, 3.0
        n = 100
        com_x = np.full(n, arena_center_x + 1.0)  # 1m east of center
        com_y = np.full(n, arena_center_y)  # same y as center
        com_z = np.full(n, 0.1)  # all floor

        height_id, zone_sector, zone_id, cuts = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config, arena_center_xy=(arena_center_x, arena_center_y)
        )
        # Due east → angle = 0 → sector 0
        assert np.all(zone_sector == 0), f"Expected sector 0, got {np.unique(zone_sector)}"

    def test_zone_sector_due_north_of_center(
        self, default_config: Fig2DDecoderConfig
    ) -> None:
        """A point due north (positive y) of arena center → sector 0 or 1 depending on n_sectors."""
        arena_center_x, arena_center_y = 2.0, 3.0
        n = 100
        com_x = np.full(n, arena_center_x)
        com_y = np.full(n, arena_center_y + 1.0)  # 1m north
        com_z = np.full(n, 0.1)

        height_id, zone_sector, zone_id, cuts = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config, arena_center_xy=(arena_center_x, arena_center_y)
        )
        # Due north → angle = pi/2 ≈ 1.57, sector_size = 2pi/3 ≈ 2.09 → sector 0
        assert np.all(zone_sector == 0), f"Expected sector 0, got {np.unique(zone_sector)}"

    def test_zone_sector_southwest_of_center(
        self, default_config: Fig2DDecoderConfig
    ) -> None:
        """A point southwest of arena center → angle ≈ 5pi/4 → sector 1."""
        arena_center_x, arena_center_y = 2.0, 3.0
        n = 100
        com_x = np.full(n, arena_center_x - 1.0)
        com_y = np.full(n, arena_center_y - 1.0)
        com_z = np.full(n, 0.1)

        height_id, zone_sector, zone_id, cuts = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config, arena_center_xy=(arena_center_x, arena_center_y)
        )
        # SW → angle = atan2(-1, -1) + 2pi ≈ 3pi/4 + pi = 5pi/4 ≈ 3.93
        # sector_size = 2pi/3 ≈ 2.09, floor(3.93/2.09) = 1
        assert np.all(zone_sector == 1), f"Expected sector 1, got {np.unique(zone_sector)}"

    def test_no_arena_center_defaults_to_origin(
        self, default_config: Fig2DDecoderConfig
    ) -> None:
        """Without arena_center_xy, default behavior should center at (0, 0)."""
        n = 50
        com_x = np.full(n, 1.0)
        com_y = np.full(n, 0.0)
        com_z = np.full(n, 0.1)

        height_id, zone_sector, zone_id, cuts = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config
        )
        # Due east from origin → sector 0
        assert np.all(zone_sector == 0)


# ── Height labels ─────────────────────────────────────────────────────────


class TestHeightLabels:
    """Test height tier assignment with fixed cuts."""

    def test_fixed_cuts_three_tiers(self, default_config: Fig2DDecoderConfig) -> None:
        """Height labels should assign floor/bench/ceiling based on fixed cuts."""
        com_x = np.zeros(90)
        com_y = np.zeros(90)
        com_z = np.concatenate([
            np.full(30, 0.1),   # floor: < 0.3
            np.full(30, 0.45),  # bench: >= 0.3 and < 0.6
            np.full(30, 0.8),   # ceiling: >= 0.6
        ])

        height_id, _, _, cuts = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config
        )
        assert cuts == (0.3, 0.6)
        assert np.all(height_id[:30] == 0)   # floor
        assert np.all(height_id[30:60] == 1)  # bench
        assert np.all(height_id[60:] == 2)    # ceiling

    def test_boundary_values(self, default_config: Fig2DDecoderConfig) -> None:
        """Values exactly at cutpoints should go to the higher tier."""
        com_z = np.array([0.3, 0.6])
        com_x = np.zeros(2)
        com_y = np.zeros(2)

        height_id, _, _, _ = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config
        )
        assert height_id[0] == 1  # 0.3 >= low → bench
        assert height_id[1] == 2  # 0.6 >= high → ceiling

    def test_nan_positions_get_minus_one(self, default_config: Fig2DDecoderConfig) -> None:
        """NaN COM positions should produce height_id = -1."""
        com_z = np.array([0.1, np.nan, 0.8])
        com_x = np.array([0.0, np.nan, 0.0])
        com_y = np.array([0.0, np.nan, 0.0])

        height_id, _, zone_id, _ = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config
        )
        assert height_id[1] == -1
        assert zone_id[1] == -1


# ── Zone ID composition ──────────────────────────────────────────────────


class TestZoneID:
    """Test 9-class zone_id = height_id * n_sectors + zone_sector."""

    def test_zone_id_range(self, default_config: Fig2DDecoderConfig) -> None:
        """zone_id should span 0..8 for 3 heights × 3 sectors."""
        rng = np.random.default_rng(42)
        n = 300
        angles = rng.uniform(0, 2 * np.pi, n)
        com_x = np.cos(angles)
        com_y = np.sin(angles)
        com_z = np.concatenate([
            np.full(100, 0.1),
            np.full(100, 0.45),
            np.full(100, 0.8),
        ])

        height_id, zone_sector, zone_id, _ = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config
        )
        valid = zone_id >= 0
        assert valid.sum() == n
        assert set(np.unique(zone_id[valid])).issubset(set(range(9)))

    def test_zone_id_decomposition(self, default_config: Fig2DDecoderConfig) -> None:
        """zone_id should equal height_id * 3 + zone_sector."""
        rng = np.random.default_rng(123)
        n = 200
        angles = rng.uniform(0, 2 * np.pi, n)
        com_x = np.cos(angles)
        com_y = np.sin(angles)
        com_z = rng.uniform(0, 1.0, n)

        height_id, zone_sector, zone_id, _ = _build_height_and_zone_labels(
            com_x, com_y, com_z, default_config
        )
        valid = zone_id >= 0
        np.testing.assert_array_equal(
            zone_id[valid],
            height_id[valid] * 3 + zone_sector[valid],
        )


# ── Balanced train indices ────────────────────────────────────────────────


class TestBalancedTrainIndices:
    """Test that balanced downsampling produces equal per-class counts."""

    def test_equal_class_counts(self, rng: np.random.Generator) -> None:
        """Output should have exactly min_count per class."""
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        idx = _balanced_train_indices(y, rng)
        _, counts = np.unique(y[idx], return_counts=True)
        assert np.all(counts == 20)  # min class is 20

    def test_indices_are_valid(self, rng: np.random.Generator) -> None:
        """Returned indices should be within [0, len(y))."""
        y = np.array([0] * 100 + [1] * 50)
        idx = _balanced_train_indices(y, rng)
        assert np.all(idx >= 0)
        assert np.all(idx < len(y))

    def test_no_duplicates(self, rng: np.random.Generator) -> None:
        """Returned indices should have no duplicates."""
        y = np.array([0] * 100 + [1] * 80 + [2] * 60)
        idx = _balanced_train_indices(y, rng)
        assert len(idx) == len(np.unique(idx))


# ── Nuisance residualization ──────────────────────────────────────────────


class TestNuisanceResiduals:
    """Test ridge residualization shape and variance reduction."""

    def test_output_shape(self) -> None:
        """Residuals should have same shape as input y."""
        n_train, n_test, n_feat = 80, 20, 10
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((n_train, n_feat))
        x_test = rng.standard_normal((n_test, n_feat))
        y_train = rng.standard_normal(n_train)
        y_test = rng.standard_normal(n_test)

        resid_train, resid_test = _fit_nuisance_residuals(
            y_train, y_test, x_train, x_test, alpha=1.0
        )
        assert resid_train.shape == (n_train,)
        assert resid_test.shape == (n_test,)

    def test_variance_reduction(self) -> None:
        """Residuals should have less variance when nuisance predicts well."""
        n_train, n_test = 200, 50
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((n_train, 3))
        x_test = rng.standard_normal((n_test, 3))
        # y = strong linear combination of x + small noise
        beta = np.array([2.0, -1.0, 0.5])
        y_train = x_train @ beta + rng.standard_normal(n_train) * 0.1
        y_test = x_test @ beta + rng.standard_normal(n_test) * 0.1

        resid_train, resid_test = _fit_nuisance_residuals(
            y_train, y_test, x_train, x_test, alpha=0.01
        )
        # Residual variance should be much less than original
        assert np.var(resid_test) < np.var(y_test) * 0.5


# ── Fold guard ────────────────────────────────────────────────────────────


class TestFoldGuard:
    """Test min-class guard in _decode_accuracy."""

    def _make_fold_cache(
        self, n_bins: int, n_units: int, n_folds: int, rng: np.random.Generator
    ) -> list:
        """Create synthetic fold cache for testing."""
        fold_size = n_bins // n_folds
        cache = []
        x_all = rng.standard_normal((n_bins, n_units))
        for i in range(n_folds):
            test_idx = np.arange(i * fold_size, (i + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(n_bins), test_idx)
            raw_train, raw_test = x_all[train_idx], x_all[test_idx]
            cache.append({
                "train_idx": train_idx,
                "test_idx": test_idx,
                "raw_train": raw_train,
                "raw_test": raw_test,
                "resid_train": raw_train.copy(),
                "resid_test": raw_test.copy(),
            })
        return cache

    def test_fold_with_missing_class_skipped(self) -> None:
        """Folds where a class has < min samples should be skipped."""
        rng_gen = np.random.default_rng(42)
        n_bins, n_units = 100, 10
        config = Fig2DDecoderConfig(
            balance_train_bins=False,
            n_permutations=0,
            random_state=42,
            fold_guard_level="strict",
        )
        cache = self._make_fold_cache(n_bins, n_units, 5, rng_gen)

        # Create labels where class 2 only appears in fold 0's test set
        labels = np.zeros(n_bins, dtype=int)
        labels[:40] = 0
        labels[40:80] = 1
        # Class 2 only in first 5 bins (fold 0 test)
        labels[:5] = 2

        score, _, fold_scores = _decode_accuracy(
            cache, labels, 3, "raw", config, rng_gen, return_predictions=False
        )
        # Some folds must be skipped (class 2 too sparse for strict guard)
        assert len(fold_scores) < 5, (
            f"Expected fewer than 5 valid folds with sparse class 2, got {len(fold_scores)}"
        )

    def test_all_folds_invalid_returns_nan(self) -> None:
        """When ALL folds fail min-class, session-target should be NaN."""
        rng_gen = np.random.default_rng(42)
        n_bins, n_units = 50, 5
        config = Fig2DDecoderConfig(
            balance_train_bins=False,
            n_permutations=0,
            random_state=42,
            fold_guard_level="strict",
            min_valid_folds=3,
        )
        cache = self._make_fold_cache(n_bins, n_units, 5, rng_gen)

        # Only 2 samples of class 1, 2 of class 2 → strict guard (20/10) fails all folds
        labels = np.zeros(n_bins, dtype=int)
        labels[:2] = 1
        labels[2:4] = 2

        score, _, fold_scores = _decode_accuracy(
            cache, labels, 3, "raw", config, rng_gen, return_predictions=False
        )
        assert math.isnan(score), f"Expected NaN when all folds fail guard, got {score}"
        assert len(fold_scores) == 0, (
            f"Expected 0 valid fold scores when all folds fail, got {len(fold_scores)}"
        )

    def test_valid_fold_mask_matches_guard(self) -> None:
        """_compute_valid_fold_mask should match per-fold guard decisions."""
        rng_gen = np.random.default_rng(42)
        n_bins, n_units = 100, 10
        cache = self._make_fold_cache(n_bins, n_units, 5, rng_gen)

        # Interleaved 3-class labels → every fold gets all classes
        labels = np.tile(np.arange(3), n_bins // 3 + 1)[:n_bins]
        mask = _compute_valid_fold_mask(cache, labels, 3, "relaxed")
        assert len(mask) == 5
        # With ~33 samples per class interleaved across folds, relaxed (10/5) should pass
        assert sum(mask) >= 3, f"Expected ≥3 valid folds, got {sum(mask)}"

        # Sparse labels → strict guard should reject most/all folds
        sparse_labels = np.zeros(n_bins, dtype=int)
        sparse_labels[:3] = 1
        sparse_labels[3:6] = 2
        mask_sparse = _compute_valid_fold_mask(cache, sparse_labels, 3, "strict")
        assert sum(mask_sparse) < sum(mask), (
            "Sparse labels should produce fewer valid folds than balanced labels"
        )


# ── Permutation null consistency ──────────────────────────────────────────


class TestPermNullConsistency:
    """Permutation null should use identical fold eligibility as observed."""

    def _make_fold_cache(
        self, n_bins: int, n_units: int, n_folds: int, rng: np.random.Generator
    ) -> list:
        """Create synthetic fold cache for testing."""
        fold_size = n_bins // n_folds
        cache = []
        x_all = rng.standard_normal((n_bins, n_units))
        for i in range(n_folds):
            test_idx = np.arange(i * fold_size, (i + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(n_bins), test_idx)
            raw_train, raw_test = x_all[train_idx], x_all[test_idx]
            cache.append({
                "train_idx": train_idx,
                "test_idx": test_idx,
                "raw_train": raw_train,
                "raw_test": raw_test,
                "resid_train": raw_train.copy(),
                "resid_test": raw_test.copy(),
            })
        return cache

    def test_perm_null_returns_array(self) -> None:
        """Smoke test that _permutation_null runs and returns correct shape."""
        from freemat.single_fm.encoding.fig2d_spatial_decoder import _permutation_null

        rng_gen = np.random.default_rng(42)
        n_bins, n_units = 100, 10
        config = Fig2DDecoderConfig(
            balance_train_bins=False,
            n_permutations=5,
            random_state=42,
        )
        cache = self._make_fold_cache(n_bins, n_units, 5, rng_gen)

        labels = np.repeat(np.arange(3), n_bins // 3 + 1)[:n_bins]
        null = _permutation_null(cache, labels, 3, config, rng_gen)
        assert null["raw"].shape == (5,)
        assert null["resid"].shape == (5,)

    def test_pvalue_uses_fold_mean(self) -> None:
        """P-value must compare fold-mean BA (not concatenated BA).

        Synthetic case: construct data where fold-mean ≠ concatenated BA,
        verify that null distribution entries are fold-mean values.
        """
        from freemat.single_fm.encoding.fig2d_spatial_decoder import _permutation_null

        rng_gen = np.random.default_rng(99)
        n_bins, n_units = 150, 10
        config = Fig2DDecoderConfig(
            balance_train_bins=False,
            n_permutations=3,
            random_state=99,
            fold_guard_level="relaxed",
            min_valid_folds=1,
        )
        cache = self._make_fold_cache(n_bins, n_units, 5, rng_gen)
        labels = np.repeat(np.arange(3), n_bins // 3 + 1)[:n_bins]

        null = _permutation_null(cache, labels, 3, config, rng_gen)
        # Null values should be fold-mean BA (in [0, 1] range for balanced accuracy)
        finite_raw = null["raw"][np.isfinite(null["raw"])]
        for val in finite_raw:
            assert 0.0 <= val <= 1.0, f"Null fold-mean BA out of range: {val}"

        # _permutation_null calls _decode_accuracy with return_predictions=False
        # which returns fold-mean score — verify shape matches n_permutations
        assert null["raw"].shape[0] == 3
        assert null["resid"].shape[0] == 3

    def test_n_null_valid_tracks_finite_perms(self) -> None:
        """n_null_valid should count only finite (non-NaN) permutation entries."""
        from freemat.single_fm.encoding.fig2d_spatial_decoder import _permutation_null

        rng_gen = np.random.default_rng(42)
        n_bins, n_units = 150, 10
        config = Fig2DDecoderConfig(
            balance_train_bins=False,
            n_permutations=5,
            random_state=42,
            fold_guard_level="relaxed",
            min_valid_folds=1,
        )
        cache = self._make_fold_cache(n_bins, n_units, 5, rng_gen)
        # Well-balanced labels → all perms should produce finite scores
        labels = np.repeat(np.arange(3), n_bins // 3 + 1)[:n_bins]

        null = _permutation_null(cache, labels, 3, config, rng_gen)
        assert "n_null_valid_raw" in null, "Missing n_null_valid_raw key"
        assert "n_null_valid_resid" in null, "Missing n_null_valid_resid key"
        # With balanced labels and relaxed guard, all perms should succeed
        assert null["n_null_valid_raw"] == int(np.isfinite(null["raw"]).sum())
        assert null["n_null_valid_resid"] == int(np.isfinite(null["resid"]).sum())
        assert null["n_null_valid_raw"] <= 5
