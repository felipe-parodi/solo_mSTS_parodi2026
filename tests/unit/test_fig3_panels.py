"""Tests for fig3_panels module (W5.2 refactoring).

Verifies all 7 fig3 panel functions are importable and run without
error on synthetic data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PANEL_NAMES = [
    "fig3_panel_a_winner_bars",
    "fig3_panel_b_boxplots",
    "fig3_panel_c_scatter",
    "fig3_panel_d_cdf",
    "fig3_panel_e_bank_dissociation",
    "fig3_pie_charts",
    "fig3_supp_r2_filtered",
]


def _make_synthetic_fig3(n: int = 200) -> pd.DataFrame:
    """Create synthetic Fig 3-like DataFrame."""
    rng = np.random.default_rng(42)
    frames = ["allo", "body", "head"]
    return pd.DataFrame({
        "session_id": rng.choice(["M2_0526", "M1_0525"], n),
        "subject": rng.choice(["M2", "M1"], n),
        "unit_id": np.arange(n),
        "winner_frame": rng.choice(frames, n),
        "delta_r2_allo": rng.normal(0.01, 0.005, n),
        "delta_r2_body": rng.normal(0.008, 0.005, n),
        "delta_r2_head": rng.normal(0.003, 0.005, n),
        "r2_base": rng.uniform(0, 0.05, n),
        "unit_bank": rng.choice(["ventral", "dorsal"], n),
        "unit_kind": rng.choice(["SUA", "MUA"], n),
    })


class TestImportability:
    """All 7 panel functions importable from fig3_panels."""

    def test_all_importable(self) -> None:
        from freemat.single_fm.viz import fig3_panels

        for name in _PANEL_NAMES:
            assert hasattr(fig3_panels, name), f"Missing: {name}"
            assert callable(getattr(fig3_panels, name))


class TestPanelSmoke:
    """Smoke tests: each panel runs without error on synthetic data."""

    def test_winner_bars(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_panel_a_winner_bars

        df = _make_synthetic_fig3()
        fig3_panel_a_winner_bars(df, tmp_path)
        assert (tmp_path / "fig3A_winner_bars.png").exists()

    def test_boxplots(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_panel_b_boxplots

        df = _make_synthetic_fig3()
        fig3_panel_b_boxplots(df, tmp_path)
        assert (tmp_path / "fig3B_boxplot.png").exists()

    def test_scatter(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_panel_c_scatter

        df = _make_synthetic_fig3()
        fig3_panel_c_scatter(df, tmp_path)
        assert (tmp_path / "fig3C_scatter_allo_body.png").exists()

    def test_cdf(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_panel_d_cdf

        df = _make_synthetic_fig3()
        fig3_panel_d_cdf(df, tmp_path)
        assert (tmp_path / "fig3D_cdf.png").exists()

    def test_bank_dissociation(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_panel_e_bank_dissociation

        df = _make_synthetic_fig3()
        fig3_panel_e_bank_dissociation(df, tmp_path)
        assert (tmp_path / "fig3E_bank_dissociation.png").exists()

    def test_pie_charts(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_pie_charts

        df = _make_synthetic_fig3()
        fig3_pie_charts(df, tmp_path)
        assert (tmp_path / "fig3_pie_charts.png").exists()

    def test_r2_filtered(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_supp_r2_filtered

        df = _make_synthetic_fig3()
        fig3_supp_r2_filtered(df, tmp_path)
        assert (tmp_path / "fig3_supp_r2_filtered.png").exists()


class TestTagSuffix:
    """Verify tag suffix is appended to filenames."""

    def test_tag_in_filename(self, tmp_path: Path) -> None:
        from freemat.single_fm.viz.fig3_panels import fig3_panel_b_boxplots

        df = _make_synthetic_fig3()
        fig3_panel_b_boxplots(df, tmp_path, tag="_M2")
        assert (tmp_path / "fig3B_boxplot_M2.png").exists()
        assert (tmp_path / "fig3B_boxplot_M2.svg").exists()
