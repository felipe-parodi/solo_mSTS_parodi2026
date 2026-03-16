"""Figure 2B visualization: Delta CV-R² distribution for KIN vs VIS.

This module re-exports from the split submodules for backward compatibility.
The actual implementations live in:
- fig2b_panels.py: Core panel plots (histograms, scatter, bars, variance)
- fig2b_testard.py: Testard-style boxplots, scatter with marginals, summary stats
"""

# Core panel plots
from freemat.single_fm.viz.fig2b_panels import (
    plot_fig2b_delta_r2_distribution,
    plot_fig2b_variance_partitioning,
    plot_fig2b_scatter_kin_vis,
    plot_fig2b_summary_bars,
    generate_fig2b_panel,
    plot_fig2b_3way_variance_partitioning,
    plot_fig2b_3way_scatter,
    plot_fig2b_summary_bars_3way,
    plot_fig2b_by_bank,
    generate_fig2b_panel_3way,
    # N-way (dynamic group discovery)
    plot_fig2b_nway_variance_partitioning,
    plot_fig2b_nway_summary_bars,
    generate_fig2b_panel_nway,
)

# Testard-style plots
from freemat.single_fm.viz.fig2b_testard import (
    plot_fig2b_testard_boxplots,
    plot_fig2c_unique_scatter,
    compute_testard_summary,
)

__all__ = [
    "plot_fig2b_delta_r2_distribution",
    "plot_fig2b_variance_partitioning",
    "plot_fig2b_scatter_kin_vis",
    "plot_fig2b_summary_bars",
    "generate_fig2b_panel",
    # N-way partitioning (dynamic group discovery)
    "plot_fig2b_nway_variance_partitioning",
    "plot_fig2b_nway_summary_bars",
    "generate_fig2b_panel_nway",
    # Backward-compatible aliases
    "plot_fig2b_3way_variance_partitioning",
    "plot_fig2b_3way_scatter",
    "plot_fig2b_summary_bars_3way",
    "generate_fig2b_panel_3way",
    # Per-bank analysis
    "plot_fig2b_by_bank",
    # Testard-style
    "plot_fig2b_testard_boxplots",
    "plot_fig2c_unique_scatter",
    "compute_testard_summary",
]
