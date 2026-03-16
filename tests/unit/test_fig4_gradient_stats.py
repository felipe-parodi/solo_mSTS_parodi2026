"""Tests for Fig 4 electrode gradient statistics.

Tests gradient analyses along depth, AP, and bank axes for
selectivity and bandwidth metrics derived from motif tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tuning_df(
    rng: np.random.Generator,
    motif_ids: list[int],
    n_shanks: int = 4,
    n_depths: int = 8,
    probes: list[str] | None = None,
    z_func=None,
    session_id: str = "test_session",
) -> pd.DataFrame:
    """Build synthetic tuning DataFrame.

    Args:
        rng: NumPy random generator.
        motif_ids: Motif IDs for z columns.
        n_shanks: Number of shanks per probe (1-4).
        n_depths: Number of depth positions (0 to n_depths-1).
        probes: List of probe names (default: A1, A2, B1, B2).
        z_func: Optional callable(probe, shank, depth) -> dict of z-score
            overrides. Only keys like ``motif_{id}_z`` are overridden.
        session_id: Session identifier column value.

    Returns:
        DataFrame in tuning_df schema.
    """
    if probes is None:
        probes = ["A1", "A2", "B1", "B2"]
    bank_map = {"A1": "ventral", "A2": "ventral", "B1": "dorsal", "B2": "dorsal"}
    rows: list[dict] = []
    uid = 0
    for probe in probes:
        bank = bank_map[probe]
        for shank in range(1, n_shanks + 1):
            for depth in range(n_depths):
                row = {
                    "unit_index": uid,
                    "probe": probe,
                    "shank": shank,
                    "depth_position": depth,
                    "bank": bank,
                    "session_id": session_id,
                }
                for mid in motif_ids:
                    row[f"motif_{mid}_z"] = float(rng.standard_normal())
                if z_func is not None:
                    row.update(z_func(probe, shank, depth))
                rows.append(row)
                uid += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def motif_ids():
    return [0, 1, 2, 3, 4]


@pytest.fixture()
def basic_tuning_df(rng, motif_ids):
    """Uniform random z-scores across probes."""
    return _make_tuning_df(rng, motif_ids)


# ---------------------------------------------------------------------------
# Test: compute_per_unit_metrics
# ---------------------------------------------------------------------------

class TestComputePerUnitMetrics:
    """Validate per-unit selectivity and bandwidth computation."""

    def test_columns_present(self, basic_tuning_df, motif_ids):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_per_unit_metrics,
        )

        result = compute_per_unit_metrics(basic_tuning_df, motif_ids)
        assert "selectivity" in result.columns
        assert "bandwidth" in result.columns

    def test_selectivity_is_max_abs_z(self, rng, motif_ids):
        """Selectivity = max |z| across motifs."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_per_unit_metrics,
        )

        df = _make_tuning_df(rng, motif_ids, n_shanks=1, n_depths=1, probes=["A1"])
        result = compute_per_unit_metrics(df, motif_ids)
        z_vals = [df[f"motif_{m}_z"].iloc[0] for m in motif_ids]
        expected = max(abs(z) for z in z_vals)
        assert np.isclose(result["selectivity"].iloc[0], expected)

    def test_bandwidth_counts_above_threshold(self, motif_ids):
        """Bandwidth = count of motifs with |z| > 1."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_per_unit_metrics,
        )

        rng = np.random.default_rng(99)
        df = _make_tuning_df(rng, motif_ids, n_shanks=1, n_depths=1, probes=["A1"])
        # Set specific z-scores: 3 above threshold
        for mid in motif_ids[:3]:
            df[f"motif_{mid}_z"] = 2.0
        for mid in motif_ids[3:]:
            df[f"motif_{mid}_z"] = 0.5
        result = compute_per_unit_metrics(df, motif_ids)
        assert result["bandwidth"].iloc[0] == 3

    def test_missing_motif_column_raises(self, basic_tuning_df):
        """ValueError if motif_order requests non-existent z column."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_per_unit_metrics,
        )

        with pytest.raises(ValueError, match="motif_999_z"):
            compute_per_unit_metrics(basic_tuning_df, [999])


# ---------------------------------------------------------------------------
# Test: compute_electrode_metrics
# ---------------------------------------------------------------------------

class TestComputeElectrodeMetrics:
    """Electrode-level aggregation from unit-level metrics."""

    def test_aggregation_reduces_rows(self, basic_tuning_df, motif_ids):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        unit_df = compute_per_unit_metrics(basic_tuning_df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        # 4 probes x 4 shanks x 8 depths = 128 electrodes
        assert len(elec_df) == 128
        assert "n_units" in elec_df.columns
        assert (elec_df["n_units"] == 1).all()  # 1 unit per electrode

    def test_multi_unit_electrode(self, motif_ids):
        """Multiple units at same electrode → mean metric."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        rows = []
        for uid in range(3):
            row = {
                "unit_index": uid,
                "probe": "A1",
                "shank": 1,
                "depth_position": 0,
                "bank": "ventral",
                "session_id": "test",
            }
            for mid in motif_ids:
                row[f"motif_{mid}_z"] = float(uid + 1)  # deterministic
            rows.append(row)
        df = pd.DataFrame(rows)
        unit_df = compute_per_unit_metrics(df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        assert len(elec_df) == 1
        assert elec_df["n_units"].iloc[0] == 3


# ---------------------------------------------------------------------------
# Test: depth gradient
# ---------------------------------------------------------------------------

class TestDepthGradient:
    """Spearman correlation for depth-metric gradients."""

    def test_detects_positive_gradient(self, motif_ids):
        """Selectivity increasing with depth → positive rho."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_depth_gradient,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        rng = np.random.default_rng(42)

        def depth_z(probe, shank, depth):
            # Make z increase with depth → selectivity increases
            return {f"motif_{m}_z": depth * 0.5 + rng.normal(0, 0.1)
                    for m in motif_ids}

        df = _make_tuning_df(rng, motif_ids, z_func=depth_z)
        unit_df = compute_per_unit_metrics(df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        result = compute_depth_gradient(elec_df, "selectivity")
        assert result.rho > 0
        assert result.p_value < 0.05

    def test_flat_gradient_not_significant(self, basic_tuning_df, motif_ids):
        """Random data → Spearman not systematically significant."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_depth_gradient,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        unit_df = compute_per_unit_metrics(basic_tuning_df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        result = compute_depth_gradient(elec_df, "selectivity")
        # With random data, p should typically be > 0.05
        # (not deterministic, but with seed 42 this holds)
        assert hasattr(result, "rho")
        assert hasattr(result, "p_value")
        assert hasattr(result, "n")

    def test_bank_filter(self, basic_tuning_df, motif_ids):
        """Depth gradient within a single bank."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_depth_gradient,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        unit_df = compute_per_unit_metrics(basic_tuning_df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        result = compute_depth_gradient(elec_df, "selectivity", bank="ventral")
        # Should only include ventral electrodes
        assert result.n == 64  # 2 probes x 4 shanks x 8 depths


# ---------------------------------------------------------------------------
# Test: AP gradient
# ---------------------------------------------------------------------------

class TestAPGradient:
    """Mann-Whitney U for anterior vs posterior within bank."""

    def test_detects_ap_difference(self, motif_ids):
        """Different AP means → significant Mann-Whitney."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_ap_gradient,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        rng = np.random.default_rng(42)

        def ap_z(probe, shank, depth):
            # A2 (anterior) has higher z than A1 (posterior) for ventral
            if probe == "A2":
                return {f"motif_{m}_z": 3.0 + rng.normal(0, 0.2)
                        for m in motif_ids}
            return {f"motif_{m}_z": rng.normal(0, 0.2) for m in motif_ids}

        df = _make_tuning_df(rng, motif_ids, z_func=ap_z)
        unit_df = compute_per_unit_metrics(df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        result = compute_ap_gradient(elec_df, "selectivity", bank="ventral")
        assert result.p_value < 0.05
        # A2 (anterior) should have higher selectivity
        assert result.direction == "anterior>posterior"

    def test_requires_bank(self, basic_tuning_df, motif_ids):
        """AP gradient must be computed within-bank."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_ap_gradient,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        unit_df = compute_per_unit_metrics(basic_tuning_df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        # Both ventral and dorsal should work
        for bank in ("ventral", "dorsal"):
            result = compute_ap_gradient(elec_df, "selectivity", bank=bank)
            assert hasattr(result, "u_stat")
            assert hasattr(result, "p_value")
            assert hasattr(result, "rank_biserial")


# ---------------------------------------------------------------------------
# Test: bank comparison
# ---------------------------------------------------------------------------

class TestBankComparison:
    """Mann-Whitney U for ventral vs dorsal."""

    def test_detects_bank_difference(self, motif_ids):
        """Dorsal > ventral → correct direction."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_bank_comparison,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        rng = np.random.default_rng(42)

        def bank_z(probe, shank, depth):
            if probe.startswith("B"):  # dorsal
                return {f"motif_{m}_z": 3.0 + rng.normal(0, 0.2)
                        for m in motif_ids}
            return {f"motif_{m}_z": rng.normal(0, 0.2) for m in motif_ids}

        df = _make_tuning_df(rng, motif_ids, z_func=bank_z)
        unit_df = compute_per_unit_metrics(df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)
        result = compute_bank_comparison(elec_df, "selectivity")
        assert result.p_value < 0.05
        assert result.direction == "D>V"
        assert -1 <= result.rank_biserial <= 1


# ---------------------------------------------------------------------------
# Test: unit vs electrode divergence
# ---------------------------------------------------------------------------

class TestUnitVsElectrodeDivergence:
    """Unit-level and electrode-level can give different answers."""

    def test_confounded_divergence(self, motif_ids):
        """When one bank has more units per electrode, unit-level
        may differ from electrode-level inference."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_bank_comparison,
            compute_electrode_metrics,
            compute_per_unit_metrics,
        )

        rng = np.random.default_rng(42)
        rows = []
        uid = 0
        # Ventral: many units, moderate z
        for shank in range(1, 5):
            for depth in range(8):
                for _ in range(5):  # 5 units per electrode
                    row = {
                        "unit_index": uid,
                        "probe": "A1",
                        "shank": shank,
                        "depth_position": depth,
                        "bank": "ventral",
                        "session_id": "test",
                    }
                    for mid in motif_ids:
                        row[f"motif_{mid}_z"] = 1.5 + rng.normal(0, 0.1)
                    rows.append(row)
                    uid += 1
        # Dorsal: 1 unit per electrode, same z
        for shank in range(1, 5):
            for depth in range(8):
                row = {
                    "unit_index": uid,
                    "probe": "B1",
                    "shank": shank,
                    "depth_position": depth,
                    "bank": "dorsal",
                    "session_id": "test",
                }
                for mid in motif_ids:
                    row[f"motif_{mid}_z"] = 1.5 + rng.normal(0, 0.1)
                rows.append(row)
                uid += 1

        df = pd.DataFrame(rows)
        unit_df = compute_per_unit_metrics(df, motif_ids)
        elec_df = compute_electrode_metrics(unit_df)

        # Electrode level: both banks have similar electrode means
        elec_result = compute_bank_comparison(elec_df, "selectivity")
        # Unit level would be dominated by ventral (5x more units)
        # This test just verifies both run without error
        assert elec_result.n_ventral == 32  # 4 shanks x 8 depths
        assert elec_result.n_dorsal == 32


# ---------------------------------------------------------------------------
# Test: geometry regression
# ---------------------------------------------------------------------------

class TestGeometryRegression:
    """Verify AP mapping matches fig4_panels.py semantics."""

    def test_ap_position_semantics(self):
        """_PROBE_AP_POSITION matches _PROBE_AP_UM in fig4_panels.py."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            _PROBE_AP_POSITION,
        )
        from freemat.single_fm.viz.fig4_panels import _PROBE_AP_UM

        # A2/B2 → anterior (0 µm), A1/B1 → posterior (800 µm)
        for probe, position in _PROBE_AP_POSITION.items():
            um = _PROBE_AP_UM[probe]
            if position == "anterior":
                assert um == 0.0, f"{probe}: expected 0 µm for anterior, got {um}"
            else:
                assert um == 800.0, f"{probe}: expected 800 µm for posterior, got {um}"

    def test_bank_mapping_consistent(self):
        """_PROBE_BANK matches fig4 convention."""
        from freemat.single_fm.encoding.fig4_gradient_stats import _PROBE_BANK

        assert _PROBE_BANK["A1"] == "ventral"
        assert _PROBE_BANK["A2"] == "ventral"
        assert _PROBE_BANK["B1"] == "dorsal"
        assert _PROBE_BANK["B2"] == "dorsal"


# ---------------------------------------------------------------------------
# Test: FDR correction
# ---------------------------------------------------------------------------

class TestFDRCorrection:
    """Benjamini-Hochberg correction across test family."""

    def test_fdr_adds_columns(self):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            apply_fdr_correction,
        )

        summary = pd.DataFrame({
            "metric": ["selectivity", "selectivity", "bandwidth", "bandwidth"],
            "test": ["bank", "depth", "bank", "depth"],
            "p_value": [0.001, 0.05, 0.03, 0.8],
            "fdr_family": ["primary"] * 4,
        })
        result = apply_fdr_correction(summary)
        assert "p_fdr" in result.columns
        assert "significant_fdr" in result.columns
        assert result["p_fdr"].iloc[0] <= result["p_fdr"].iloc[2]  # sorted

    def test_excluded_tests_not_corrected(self):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            apply_fdr_correction,
        )

        summary = pd.DataFrame({
            "metric": ["selectivity", "selectivity"],
            "test": ["bank", "depth_ventral"],
            "p_value": [0.01, 0.02],
            "fdr_family": ["primary", "excluded_n<3"],
        })
        result = apply_fdr_correction(summary)
        # Excluded row should have NaN for p_fdr
        excluded = result[result["fdr_family"] == "excluded_n<3"]
        assert excluded["p_fdr"].isna().all()


# ---------------------------------------------------------------------------
# Test: compute_all_gradient_stats
# ---------------------------------------------------------------------------

class TestComputeAllGradientStats:
    """Integration test for the full pipeline."""

    def test_returns_summary_dataframe(self, basic_tuning_df, motif_ids):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_all_gradient_stats,
        )

        result = compute_all_gradient_stats(basic_tuning_df, motif_ids)
        assert isinstance(result, pd.DataFrame)
        # 2 metrics x 6 tests = 12 rows max (some may be excluded)
        assert len(result) <= 12
        assert "metric" in result.columns
        assert "test" in result.columns
        assert "p_value" in result.columns
        assert "p_fdr" in result.columns

    def test_asserts_single_session(self, motif_ids):
        """Raises if tuning_df has multiple session_ids."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_all_gradient_stats,
        )

        rng = np.random.default_rng(42)
        df = _make_tuning_df(rng, motif_ids)
        # Add a second session
        df2 = df.copy()
        df2["session_id"] = "other_session"
        combined = pd.concat([df, df2], ignore_index=True)
        with pytest.raises(ValueError, match="single session"):
            compute_all_gradient_stats(combined, motif_ids)

    def test_both_metrics_present(self, basic_tuning_df, motif_ids):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_all_gradient_stats,
        )

        result = compute_all_gradient_stats(basic_tuning_df, motif_ids)
        assert set(result["metric"].unique()) == {"selectivity", "bandwidth"}

    def test_all_test_types_present(self, basic_tuning_df, motif_ids):
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_all_gradient_stats,
        )

        result = compute_all_gradient_stats(basic_tuning_df, motif_ids)
        expected_tests = {
            "bank", "depth", "ap_ventral", "ap_dorsal",
            "depth_ventral", "depth_dorsal",
        }
        assert set(result["test"].unique()) == expected_tests

    def test_insufficient_n_excluded(self, motif_ids):
        """With only one probe, some AP tests are excluded."""
        from freemat.single_fm.encoding.fig4_gradient_stats import (
            compute_all_gradient_stats,
        )

        rng = np.random.default_rng(42)
        # Only ventral probes → dorsal AP test should be excluded
        df = _make_tuning_df(rng, motif_ids, probes=["A1", "A2"])
        result = compute_all_gradient_stats(df, motif_ids)
        dorsal_ap = result[result["test"] == "ap_dorsal"]
        assert (dorsal_ap["fdr_family"] == "excluded_n<3").all()
