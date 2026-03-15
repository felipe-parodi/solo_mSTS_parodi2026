"""Unit tests for Fig 3 RF-index computation.

Tests cover:
- compute_rf_index with known delta_r2 values
- Boundary cases: equal dr2, extreme values, zero denominator
- Bounded output [-1, +1]
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class TestComputeRFIndex:
    """Test rf_index = (dr2_allo - dr2_body) / (|dr2_allo| + |dr2_body| + eps)."""

    def test_equal_dr2_returns_zero(self) -> None:
        """Equal delta_r2_allo and delta_r2_body → rf_index = 0."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": [0.1, 0.2, 0.05],
            "delta_r2_body": [0.1, 0.2, 0.05],
        })
        result = compute_rf_index(df)
        np.testing.assert_allclose(result["rf_index"].to_numpy(), 0.0, atol=1e-5)

    def test_allo_dominant_near_plus_one(self) -> None:
        """dr2_allo >> dr2_body → rf_index near +1."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": [0.5, 0.3],
            "delta_r2_body": [0.001, 0.0],
        })
        result = compute_rf_index(df)
        assert np.all(result["rf_index"] > 0.9)

    def test_body_dominant_near_minus_one(self) -> None:
        """dr2_body >> dr2_allo → rf_index near -1."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": [0.001, 0.0],
            "delta_r2_body": [0.5, 0.3],
        })
        result = compute_rf_index(df)
        assert np.all(result["rf_index"] < -0.9)

    def test_both_negative_still_bounded(self) -> None:
        """Both negative delta_r2 → rf_index still in [-1, +1]."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": [-0.05, -0.1, -0.01],
            "delta_r2_body": [-0.03, -0.02, -0.08],
        })
        result = compute_rf_index(df)
        assert np.all(result["rf_index"] >= -1.0)
        assert np.all(result["rf_index"] <= 1.0)

    def test_zero_denominator_returns_zero(self) -> None:
        """Both dr2 = 0 → rf_index = 0 (eps guard)."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": [0.0, 0.0],
            "delta_r2_body": [0.0, 0.0],
        })
        result = compute_rf_index(df)
        np.testing.assert_allclose(result["rf_index"].to_numpy(), 0.0, atol=1e-4)

    def test_preserves_other_columns(self) -> None:
        """compute_rf_index should add rf_index without dropping existing columns."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "unit_id": [1, 2, 3],
            "session_id": ["a", "a", "a"],
            "delta_r2_allo": [0.1, 0.2, 0.3],
            "delta_r2_body": [0.05, 0.1, 0.15],
            "winner_frame": ["allo", "allo", "allo"],
        })
        result = compute_rf_index(df)
        assert "rf_index" in result.columns
        assert "unit_id" in result.columns
        assert "winner_frame" in result.columns
        assert len(result) == 3

    def test_nan_dr2_produces_nan_rf_index(self) -> None:
        """NaN delta_r2 → NaN rf_index."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": [0.1, np.nan],
            "delta_r2_body": [0.05, 0.1],
        })
        result = compute_rf_index(df)
        assert np.isfinite(result["rf_index"].iloc[0])
        assert np.isnan(result["rf_index"].iloc[1])


class TestRFIndexZeroUnits:
    """Test that zero-unit sessions don't crash the runner."""

    def test_zero_units_does_not_crash(self) -> None:
        """compute_rf_index on empty DataFrame should return empty without error."""
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": pd.Series(dtype=float),
            "delta_r2_body": pd.Series(dtype=float),
        })
        result = compute_rf_index(df)
        assert "rf_index" in result.columns
        assert len(result) == 0

    def test_zero_units_percentage_guard(self) -> None:
        """Verify the percentage f-string doesn't crash with n_units=0.

        Simulates the runner's logic: after compute_rf_index returns an empty
        DataFrame, the guard should skip the percentage computation.
        """
        from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index

        df = pd.DataFrame({
            "delta_r2_allo": pd.Series(dtype=float),
            "delta_r2_body": pd.Series(dtype=float),
        })
        df_rf = compute_rf_index(df)
        n_units = len(df_rf)
        # This is the exact guard from the runner
        if n_units == 0:
            msg = "0 units after rf_index computation, skipping"
        else:
            n_allo = int((df_rf["rf_index"] > 0).sum())
            msg = f"{n_units} units, allo-dominant={n_allo}/{n_units} ({100*n_allo/n_units:.0f}%)"
        assert "0 units" in msg


class TestRFIndexBankSplit:
    """Test that bank-split violin plots produce separate groups."""

    def test_bank_split_produces_multiple_groups(self) -> None:
        """With 'bank' column present, split_by_bank=True should use 2+ violin groups."""
        from freemat.single_fm.viz.fig3_rf_panels import plot_rf_index_distribution

        df = pd.DataFrame({
            "rf_index": np.concatenate([
                np.random.default_rng(42).uniform(-0.5, 0.5, 30),
                np.random.default_rng(43).uniform(-0.3, 0.8, 30),
            ]),
            "bank": ["ventral"] * 30 + ["dorsal"] * 30,
        })
        fig = plot_rf_index_distribution(df, split_by_bank=True)
        ax = fig.axes[0]
        # With split_by_bank=True and a 'bank' column, x-tick labels should
        # contain the bank names (ventral, dorsal)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(tick_labels) >= 2, (
            f"Expected ≥2 x-tick labels for bank split, got {tick_labels}"
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_bank_column_still_works(self) -> None:
        """Without 'bank' column, split_by_bank=True should not crash."""
        from freemat.single_fm.viz.fig3_rf_panels import plot_rf_index_distribution

        df = pd.DataFrame({
            "rf_index": np.random.default_rng(42).uniform(-1, 1, 20),
        })
        fig = plot_rf_index_distribution(df, split_by_bank=True)
        # Should produce a figure without error (falls back to unsplit)
        assert len(fig.axes) >= 1
        import matplotlib.pyplot as plt
        plt.close(fig)
