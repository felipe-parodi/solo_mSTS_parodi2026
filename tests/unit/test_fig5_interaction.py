"""Tests for the Figure 5 additive-vs-interaction encoding analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from freemat.single_fm.encoding.fig5_interaction import (
    InteractionConfig,
    build_height_tier,
    build_interaction_design,
    fit_unit_interaction_model,
)


def _make_feature_df(
    rows: list[tuple[int, int]],
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a synthetic feature table and motif labels from cell specs.

    Args:
        rows: List of ``(motif_id, tier_id)`` assignments, one per bin.
        seed: RNG seed.

    Returns:
        Tuple of ``(feat_df, y_motif, bin_times)``.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    motif = np.array([m for m, _ in rows], dtype=int)
    tier = np.array([t for _, t in rows], dtype=int)
    bin_times = np.arange(n, dtype=float) * 0.5

    com_z = np.select(
        [tier == 0, tier == 1, tier == 2],
        [0.25, 0.75, 1.5],
        default=0.25,
    ).astype(float)

    feat_df = pd.DataFrame(
        {
            "gaze_x": rng.normal(size=n),
            "gaze_y": rng.normal(size=n),
            "gaze_z": rng.normal(size=n),
            "d_wall_horiz": rng.uniform(0.2, 2.0, size=n),
            "vis_motion": rng.normal(size=n),
            "v_parallel": rng.normal(size=n),
            "v_perp": rng.normal(size=n),
            "speed": np.abs(rng.normal(loc=0.5, scale=0.2, size=n)),
            "yaw_rate": rng.normal(size=n),
            "pitch_rate": rng.normal(size=n),
            "roll_rate": rng.normal(size=n),
            "delta_yaw": rng.normal(size=n),
            "com_x": rng.normal(size=n),
            "com_y": rng.normal(size=n),
            "com_z": com_z,
        },
        index=bin_times,
    )
    return feat_df, motif, bin_times


def _balanced_rows(
    counts: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    """Expand ``(motif, tier) -> count`` into per-bin rows."""
    rows: list[tuple[int, int]] = []
    for (motif_id, tier_id), count in counts.items():
        rows.extend([(motif_id, tier_id)] * count)
    return rows


def test_build_height_tier_matches_fixed_thresholds() -> None:
    com_z = np.array([0.0, 0.49, 0.5, 1.19, 1.2, 2.2], dtype=float)
    observed = build_height_tier(com_z)
    expected = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    np.testing.assert_array_equal(observed, expected)


def test_build_height_tier_nan_stays_invalid() -> None:
    """NaN and out-of-range com_z should remain -1, not be coerced to floor."""
    com_z = np.array([np.nan, -0.5, 0.25, 3.5], dtype=float)
    observed = build_height_tier(com_z)
    assert observed[0] == -1, "NaN should stay -1"
    assert observed[1] == -1, "Negative com_z should stay -1"
    assert observed[2] == 0, "Valid floor value should be tier 0"
    assert observed[3] == -1, "Out-of-range com_z should stay -1"


def test_build_interaction_design_session_qualifies_motifs_and_cells() -> None:
    rows = _balanced_rows(
        {
            (1, 0): 30,
            (1, 1): 30,
            (2, 0): 30,
            (3, 0): 25,
            (3, 1): 25,
            (3, 2): 5,
        }
    )
    feat_df, y_motif, bin_times = _make_feature_df(rows, seed=1)
    config = InteractionConfig(
        lag_s=0.0,
        n_lags=0,
        min_bins_per_cell=20,
        min_tiers_per_motif=2,
    )

    design = build_interaction_design(feat_df, y_motif, bin_times, config)

    assert design.retained_motifs == [1, 3]
    assert design.retained_cells == [(1, 0), (1, 1), (3, 0), (3, 1)]
    assert design.n_rows_final == 110
    assert "com_z" not in design.control_col_names
    assert "com_x" in design.control_col_names
    assert "com_y" in design.control_col_names


def test_interaction_block_is_one_hot_for_retained_cells() -> None:
    rows = _balanced_rows({(1, 0): 12, (1, 1): 12, (2, 0): 12, (2, 1): 12})
    feat_df, y_motif, bin_times = _make_feature_df(rows, seed=2)
    config = InteractionConfig(
        lag_s=0.0,
        n_lags=0,
        min_bins_per_cell=10,
        min_tiers_per_motif=2,
    )
    design = build_interaction_design(feat_df, y_motif, bin_times, config)

    assert design.X_interaction.shape == (48, 4)
    row_sums = design.X_interaction.sum(axis=1)
    np.testing.assert_array_equal(row_sums, np.ones(48))


def test_fit_unit_interaction_model_stays_small_for_additive_signal() -> None:
    rows = _balanced_rows({(1, 0): 40, (1, 1): 40, (2, 0): 40, (2, 1): 40})
    feat_df, y_motif, bin_times = _make_feature_df(rows, seed=3)
    config = InteractionConfig(
        bin_s=0.5,
        cv_folds=3,
        cv_block_s=5.0,
        cv_gap_s=0.5,
        n_perm=8,
        n_perm_null=0,
        lag_s=0.0,
        n_lags=0,
        min_bins_per_cell=20,
        min_bins_per_unit=60,
        r2_threshold=0.0,
        effect_threshold=0.0,
    )
    design = build_interaction_design(feat_df, y_motif, bin_times, config)
    rng = np.random.default_rng(4)

    motif_effect = np.where(design.y_motif == 1, 1.0, -0.6)
    tier_effect = np.where(design.height_tier == 1, 0.8, -0.4)
    controls_effect = 0.5 * design.X_controls[:, 0]
    y = motif_effect + tier_effect + controls_effect + rng.normal(scale=0.15, size=len(design.y_motif))

    result = fit_unit_interaction_model(y, design, config)

    assert result["delta_r2_interaction"] < 0.05
    assert result["unique_motif_x_tier_shuffle"] < 0.05


def test_fit_unit_interaction_model_detects_true_interaction() -> None:
    rows = _balanced_rows({(1, 0): 40, (1, 1): 40, (2, 0): 40, (2, 1): 40})
    feat_df, y_motif, bin_times = _make_feature_df(rows, seed=5)
    config = InteractionConfig(
        bin_s=0.5,
        cv_folds=3,
        cv_block_s=5.0,
        cv_gap_s=0.5,
        n_perm=8,
        n_perm_null=0,
        lag_s=0.0,
        n_lags=0,
        min_bins_per_cell=20,
        min_bins_per_unit=60,
        r2_threshold=0.0,
        effect_threshold=0.0,
    )
    design = build_interaction_design(feat_df, y_motif, bin_times, config)
    rng = np.random.default_rng(6)

    motif_effect = np.where(design.y_motif == 1, 1.0, -0.6)
    tier_effect = np.where(design.height_tier == 1, 0.8, -0.4)
    interaction_effect = np.where(
        (design.y_motif == 1) & (design.height_tier == 1),
        1.8,
        0.0,
    )
    controls_effect = 0.5 * design.X_controls[:, 0]
    y = (
        motif_effect
        + tier_effect
        + interaction_effect
        + controls_effect
        + rng.normal(scale=0.12, size=len(design.y_motif))
    )

    result = fit_unit_interaction_model(y, design, config)

    assert result["delta_r2_interaction"] > 0.05
    assert result["unique_motif_x_tier_shuffle"] > 0.02
