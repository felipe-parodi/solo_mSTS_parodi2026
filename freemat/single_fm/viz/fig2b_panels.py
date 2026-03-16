"""Figure 2B panel plots: histograms, scatter, bars, variance partitioning.

Contains all core panel plotting functions for Figure 2B visualization.
Supports both 2-way (KIN vs VIS) and 3-way (VIS vs BODY_KIN vs HEAD_KIN) partitioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical group ordering, display names, and colors
_GROUP_ORDER = ["vis", "body_kin", "head_kin", "pos"]
_GROUP_DISPLAY: Dict[str, str] = {
    "vis": "VIS",
    "body_kin": "BODY_KIN",
    "head_kin": "HEAD_KIN",
    "pos": "3D POS",
}
_GROUP_COLORS: Dict[str, str] = {
    "vis": "#457B9D",
    "body_kin": "#E63946",
    "head_kin": "#F4A261",
    "pos": "#7B3294",
    "shared": "#A8DADC",
}


def _discover_groups(results_df: pd.DataFrame) -> List[str]:
    """Discover feature groups from unique_*_shuffle columns.

    Returns:
        Ordered list of group names (e.g., ["vis", "body_kin", "head_kin", "pos"]).
    """
    found = []
    for g in _GROUP_ORDER:
        col = f"unique_{g}_shuffle"
        if col in results_df.columns:
            found.append(g)
    # Also check for non-canonical groups
    for c in results_df.columns:
        if c.startswith("unique_") and c.endswith("_shuffle"):
            g = c.replace("unique_", "").replace("_shuffle", "")
            if g not in found:
                found.append(g)
    return found


def _resolve_unique_col(df: pd.DataFrame, group: str) -> str:
    """Resolve unique variance column name for a feature group.

    Prefers shuffle > dropout > bare naming convention.

    Args:
        df: Results DataFrame.
        group: Feature group name (e.g., "vis", "body_kin", "head_kin", "kin").

    Returns:
        Column name that exists in df.

    Raises:
        KeyError: If no matching column found.
    """
    candidates = [
        f"unique_{group}_shuffle",
        f"unique_{group}_dropout",
        f"unique_{group}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No unique variance column found for '{group}'. Tried: {candidates}")


def _setup_nature_style() -> None:
    """Apply Nature-style matplotlib settings."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "svg.fonttype": "none",
    })


def plot_fig2b_delta_r2_distribution(
    results_df: pd.DataFrame,
    effect_threshold: float = 0.005,
    null_threshold: Optional[float] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Plot Figure 2B: Delta CV-R² distribution.

    Shows histogram of unique_kin (delta R²) values with significance
    coloring based on effect size and permutation p-value.

    Args:
        results_df: DataFrame with columns: unique_kin, sig_kin, p_kin.
        effect_threshold: Minimum delta R² for effect size threshold.
        null_threshold: 95th percentile of null distribution (optional).
        output_path: Path to save figure (both PNG and SVG if provided).
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    delta_r2 = results_df["unique_kin"].values
    sig_mask = results_df["sig_kin"].values if "sig_kin" in results_df.columns else np.zeros(len(delta_r2), dtype=bool)

    # Histogram bins
    bins = np.linspace(
        min(-0.02, np.nanmin(delta_r2) - 0.005),
        max(0.10, np.nanmax(delta_r2) + 0.005),
        40,
    )

    # Plot non-significant neurons
    ax.hist(
        delta_r2[~sig_mask],
        bins=bins,
        alpha=0.6,
        color="#888888",
        edgecolor="white",
        linewidth=0.5,
        label="Not significant",
    )

    # Plot significant neurons
    ax.hist(
        delta_r2[sig_mask],
        bins=bins,
        alpha=0.9,
        color="#E63946",
        edgecolor="white",
        linewidth=0.5,
        label="Significant (KIN)",
    )

    # Effect size threshold line
    ax.axvline(
        effect_threshold,
        color="#2A9D8F",
        linestyle="--",
        linewidth=2,
        label=f"Effect threshold ({effect_threshold*100:.1f}%)",
    )

    # Null threshold line (if provided)
    if null_threshold is not None:
        ax.axvline(
            null_threshold,
            color="#264653",
            linestyle=":",
            linewidth=1.5,
            label=f"Null 95th pct ({null_threshold*100:.1f}%)",
        )

    # Zero reference line
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Labels
    ax.set_xlabel("Unique KIN Variance (\u0394CV-R\u00b2)")
    ax.set_ylabel("Number of Neurons")
    ax.set_title("Figure 2B: Kinematic Encoding Beyond Visual Proxies")

    # Legend
    ax.legend(loc="upper right", frameon=False)

    # Summary stats as text box
    n_total = len(delta_r2)
    n_sig = sig_mask.sum()
    mean_delta = np.nanmean(delta_r2)
    median_delta = np.nanmedian(delta_r2)

    textstr = (
        f"N = {n_total} neurons\n"
        f"Significant: {n_sig} ({100*n_sig/n_total:.1f}%)\n"
        f"Mean \u0394R\u00b2: {mean_delta:.4f}\n"
        f"Median \u0394R\u00b2: {median_delta:.4f}"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#CCCCCC")
    ax.text(
        0.98, 0.72,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plt.tight_layout()

    # Save figure
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        # Also save SVG
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def plot_fig2b_variance_partitioning(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot variance partitioning stacked bar chart.

    Shows unique KIN, unique VIS, and shared variance for each neuron,
    sorted by total R².

    Args:
        results_df: DataFrame with columns: unique_kin, unique_vis, shared, full_r2.
        output_path: Path to save figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by full R²
    df_sorted = results_df.sort_values("full_r2", ascending=False).reset_index(drop=True)

    n_units = len(df_sorted)
    x = np.arange(n_units)

    # Get variance components (clamp negatives to 0 for visualization)
    unique_kin = np.maximum(df_sorted["unique_kin"].values, 0)
    unique_vis = np.maximum(df_sorted["unique_vis"].values, 0)
    shared = np.maximum(df_sorted["shared"].values, 0)

    # Stacked bars
    ax.bar(x, unique_kin, label="Unique KIN", color="#E63946", alpha=0.9)
    ax.bar(x, unique_vis, bottom=unique_kin, label="Unique VIS", color="#457B9D", alpha=0.9)
    ax.bar(x, shared, bottom=unique_kin + unique_vis, label="Shared", color="#A8DADC", alpha=0.9)

    ax.set_xlabel("Neurons (sorted by R\u00b2)")
    ax.set_ylabel("CV-R\u00b2 Variance Explained")
    ax.set_title("Variance Partitioning: KIN vs VIS")
    ax.legend(loc="upper right", frameon=False)

    # Remove x-axis ticks for cleaner look
    ax.set_xticks([])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def plot_fig2b_scatter_kin_vis(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Scatter plot of unique KIN vs unique VIS.

    Args:
        results_df: DataFrame with columns: unique_kin, unique_vis, sig_kin, sig_vis.
        output_path: Path to save figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    unique_kin = results_df["unique_kin"].values
    unique_vis = results_df["unique_vis"].values

    # Color by significance
    sig_kin = results_df["sig_kin"].values if "sig_kin" in results_df.columns else np.zeros(len(unique_kin), dtype=bool)
    sig_vis = results_df["sig_vis"].values if "sig_vis" in results_df.columns else np.zeros(len(unique_vis), dtype=bool)

    # Categories: neither, KIN only, VIS only, both
    neither = ~sig_kin & ~sig_vis
    kin_only = sig_kin & ~sig_vis
    vis_only = ~sig_kin & sig_vis
    both = sig_kin & sig_vis

    # Plot each category
    ax.scatter(unique_vis[neither], unique_kin[neither], alpha=0.4, c="#888888", s=20, label="Neither")
    ax.scatter(unique_vis[kin_only], unique_kin[kin_only], alpha=0.7, c="#E63946", s=30, label="KIN only")
    ax.scatter(unique_vis[vis_only], unique_kin[vis_only], alpha=0.7, c="#457B9D", s=30, label="VIS only")
    ax.scatter(unique_vis[both], unique_kin[both], alpha=0.9, c="#9B2335", s=40, label="Both", marker="d")

    # Diagonal line (equal contribution)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel("Unique VIS Variance (\u0394CV-R\u00b2)")
    ax.set_ylabel("Unique KIN Variance (\u0394CV-R\u00b2)")
    ax.set_title("KIN vs VIS Unique Variance")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def plot_fig2b_summary_bars(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Summary bar plot of mean variance components with error bars.

    Args:
        results_df: DataFrame with columns: unique_kin, unique_vis, shared.
        output_path: Path to save figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Compute means and SEM
    components = ["unique_kin", "unique_vis", "shared"]
    labels = ["Unique KIN", "Unique VIS", "Shared"]
    colors = ["#E63946", "#457B9D", "#A8DADC"]

    means = []
    sems = []
    for comp in components:
        vals = results_df[comp].values
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals) / np.sqrt(np.sum(np.isfinite(vals))))

    x = np.arange(len(components))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.9, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean CV-R\u00b2")
    ax.set_title("Variance Partitioning Summary")

    # Add value labels on bars
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sem + 0.002,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def generate_fig2b_panel(
    results_df: pd.DataFrame,
    output_dir: str,
    effect_threshold: float = 0.005,
    null_threshold: Optional[float] = None,
) -> None:
    """Generate all Figure 2B panels and save to output directory.

    Args:
        results_df: Results DataFrame from run_fig2b_encoding.
        output_dir: Directory to save figures.
        effect_threshold: Effect size threshold for significance.
        null_threshold: 95th percentile of null distribution (optional).
            If None and results_df has 'null_95_kin' column, computes median.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compute null threshold from results if not provided
    if null_threshold is None and "null_95_kin" in results_df.columns:
        valid_null = results_df["null_95_kin"].dropna()
        if len(valid_null) > 0:
            null_threshold = float(valid_null.median())

    # Main panel: Delta R² distribution
    plot_fig2b_delta_r2_distribution(
        results_df,
        effect_threshold=effect_threshold,
        null_threshold=null_threshold,
        output_path=str(Path(output_dir) / "fig2b_delta_r2_distribution.png"),
    )

    # Supplementary: Variance partitioning
    plot_fig2b_variance_partitioning(
        results_df,
        output_path=str(Path(output_dir) / "fig2b_variance_partitioning.png"),
    )

    # Supplementary: KIN vs VIS scatter
    plot_fig2b_scatter_kin_vis(
        results_df,
        output_path=str(Path(output_dir) / "fig2b_scatter_kin_vis.png"),
    )

    # Supplementary: Summary bars
    plot_fig2b_summary_bars(
        results_df,
        output_path=str(Path(output_dir) / "fig2b_summary_bars.png"),
    )

    plt.close("all")


# ============================================================================
# 3-Way Partitioning Visualizations (VIS vs BODY_KIN vs HEAD_KIN)
# ============================================================================


def plot_fig2b_nway_variance_partitioning(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Plot N-way variance partitioning stacked bar chart.

    Discovers groups dynamically from column names. Shows unique variance
    for each group and shared component, for each neuron sorted by full_r2.

    Args:
        results_df: DataFrame with unique_*_shuffle columns and full_r2.
        output_path: Path to save figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Discover groups dynamically
    group_keys = _discover_groups(results_df)
    if not group_keys:
        # Fallback to hardcoded 3-way
        group_keys = ["vis", "body_kin", "head_kin"]

    # Sort by full R²
    df_sorted = results_df.sort_values("full_r2", ascending=False).reset_index(drop=True)

    n_units = len(df_sorted)
    x = np.arange(n_units)

    # Get variance components (clamp negatives to 0 for visualization)
    unique_arrays = {}
    sum_unique = np.zeros(n_units)
    for g in group_keys:
        col = _resolve_unique_col(df_sorted, g)
        vals = np.maximum(df_sorted[col].values, 0)
        unique_arrays[g] = vals
        sum_unique += vals

    # Shared = full_r2 - sum of uniques (clamped >= 0)
    shared_col = "shared_all" if "shared_all" in df_sorted.columns else None
    if shared_col:
        shared = np.maximum(df_sorted[shared_col].values, 0)
    else:
        shared = np.maximum(df_sorted["full_r2"].values - sum_unique, 0)

    # Stacked bars: one per group + shared
    bottom = np.zeros(n_units)
    for g in group_keys:
        label = f"Unique {_GROUP_DISPLAY.get(g, g.upper())}"
        color = _GROUP_COLORS.get(g, "#888888")
        ax.bar(x, unique_arrays[g], bottom=bottom, label=label, color=color, alpha=0.9)
        bottom += unique_arrays[g]
    ax.bar(x, shared, bottom=bottom, label="Shared", color=_GROUP_COLORS["shared"], alpha=0.9)

    n_groups = len(group_keys)
    title_groups = " vs ".join(_GROUP_DISPLAY.get(g, g.upper()) for g in group_keys)
    ax.set_xlabel("Neurons (sorted by R\u00b2)")
    ax.set_ylabel("CV-R\u00b2 Variance Explained")
    ax.set_title(f"{n_groups}-Way Variance Partitioning: {title_groups}")
    ax.legend(loc="upper right", frameon=False)

    # Remove x-axis ticks for cleaner look
    ax.set_xticks([])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


# Backward-compatible alias
plot_fig2b_3way_variance_partitioning = plot_fig2b_nway_variance_partitioning


def plot_fig2b_3way_scatter(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Scatter plot comparing unique variances across 3 feature families.

    Creates 3 panels: VIS vs BODY_KIN, VIS vs HEAD_KIN, BODY_KIN vs HEAD_KIN.

    Args:
        results_df: DataFrame with columns: unique_vis, unique_body_kin, unique_head_kin, sig_*.
        output_path: Path to save figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get values (resolve column names: shuffle > dropout > bare)
    vis_col = _resolve_unique_col(results_df, "vis")
    body_col = _resolve_unique_col(results_df, "body_kin")
    head_col = _resolve_unique_col(results_df, "head_kin")
    unique_vis = results_df[vis_col].values
    unique_body_kin = results_df[body_col].values
    unique_head_kin = results_df[head_col].values

    # Get significance if available
    sig_vis = results_df.get("sig_vis", pd.Series([False] * len(results_df))).values
    sig_body_kin = results_df.get("sig_body_kin", pd.Series([False] * len(results_df))).values
    sig_head_kin = results_df.get("sig_head_kin", pd.Series([False] * len(results_df))).values

    # Panel 1: VIS vs BODY_KIN
    ax = axes[0]
    ax.scatter(unique_vis[~sig_vis & ~sig_body_kin], unique_body_kin[~sig_vis & ~sig_body_kin],
               alpha=0.4, c="#888888", s=20, label="Neither")
    ax.scatter(unique_vis[sig_vis & ~sig_body_kin], unique_body_kin[sig_vis & ~sig_body_kin],
               alpha=0.7, c="#457B9D", s=30, label="VIS only")
    ax.scatter(unique_vis[~sig_vis & sig_body_kin], unique_body_kin[~sig_vis & sig_body_kin],
               alpha=0.7, c="#E63946", s=30, label="BODY_KIN only")
    ax.scatter(unique_vis[sig_vis & sig_body_kin], unique_body_kin[sig_vis & sig_body_kin],
               alpha=0.9, c="#9B2335", s=40, label="Both", marker="d")
    ax.set_xlabel("Unique VIS")
    ax.set_ylabel("Unique BODY_KIN")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel 2: VIS vs HEAD_KIN
    ax = axes[1]
    ax.scatter(unique_vis[~sig_vis & ~sig_head_kin], unique_head_kin[~sig_vis & ~sig_head_kin],
               alpha=0.4, c="#888888", s=20)
    ax.scatter(unique_vis[sig_vis & ~sig_head_kin], unique_head_kin[sig_vis & ~sig_head_kin],
               alpha=0.7, c="#457B9D", s=30)
    ax.scatter(unique_vis[~sig_vis & sig_head_kin], unique_head_kin[~sig_vis & sig_head_kin],
               alpha=0.7, c="#F4A261", s=30)
    ax.scatter(unique_vis[sig_vis & sig_head_kin], unique_head_kin[sig_vis & sig_head_kin],
               alpha=0.9, c="#9B2335", s=40, marker="d")
    ax.set_xlabel("Unique VIS")
    ax.set_ylabel("Unique HEAD_KIN")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel 3: BODY_KIN vs HEAD_KIN
    ax = axes[2]
    ax.scatter(unique_body_kin[~sig_body_kin & ~sig_head_kin], unique_head_kin[~sig_body_kin & ~sig_head_kin],
               alpha=0.4, c="#888888", s=20)
    ax.scatter(unique_body_kin[sig_body_kin & ~sig_head_kin], unique_head_kin[sig_body_kin & ~sig_head_kin],
               alpha=0.7, c="#E63946", s=30)
    ax.scatter(unique_body_kin[~sig_body_kin & sig_head_kin], unique_head_kin[~sig_body_kin & sig_head_kin],
               alpha=0.7, c="#F4A261", s=30)
    ax.scatter(unique_body_kin[sig_body_kin & sig_head_kin], unique_head_kin[sig_body_kin & sig_head_kin],
               alpha=0.9, c="#9B2335", s=40, marker="d")
    ax.set_xlabel("Unique BODY_KIN")
    ax.set_ylabel("Unique HEAD_KIN")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def plot_fig2b_nway_summary_bars(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Summary bar plot of mean variance components with error bars (N-way).

    Discovers groups dynamically from column names.

    Args:
        results_df: DataFrame with unique_*_shuffle columns and optionally shared_all.
        output_path: Path to save figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Discover groups dynamically
    group_keys = _discover_groups(results_df)
    if not group_keys:
        group_keys = ["vis", "body_kin", "head_kin"]

    components = []
    labels = []
    colors = []
    for g in group_keys:
        col = _resolve_unique_col(results_df, g)
        components.append(col)
        labels.append(f"Unique {_GROUP_DISPLAY.get(g, g.upper())}")
        colors.append(_GROUP_COLORS.get(g, "#888888"))

    shared_col = "shared_all" if "shared_all" in results_df.columns else None
    if shared_col:
        components.append(shared_col)
        labels.append("Shared")
        colors.append(_GROUP_COLORS["shared"])

    means = []
    sems = []
    for comp in components:
        vals = results_df[comp].values
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals) / np.sqrt(np.sum(np.isfinite(vals))))

    x = np.arange(len(components))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.9, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean CV-R\u00b2")
    n_groups = len(group_keys)
    ax.set_title(f"{n_groups}-Way Variance Partitioning Summary")

    # Add value labels on bars
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sem + 0.002,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


# Backward-compatible alias
plot_fig2b_summary_bars_3way = plot_fig2b_nway_summary_bars


def plot_fig2b_by_bank(
    results_df: pd.DataFrame,
    output_dir: str,
) -> Optional[plt.Figure]:
    """Generate per-bank comparison figure (ventral vs dorsal).

    Creates a grouped bar chart comparing significance rates for VIS, BODY_KIN,
    and HEAD_KIN across dorsal and ventral electrode banks.

    Args:
        results_df: DataFrame with columns: unit_bank, sig_vis, sig_body_kin, sig_head_kin.
        output_dir: Directory to save figure.

    Returns:
        Matplotlib Figure object or None if no bank info available.
    """
    if "unit_bank" not in results_df.columns:
        return None

    _setup_nature_style()

    # Filter to known banks
    known_banks = results_df[results_df["unit_bank"].isin(["ventral", "dorsal"])]
    if len(known_banks) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Unit counts by bank
    ax = axes[0]
    bank_counts = known_banks["unit_bank"].value_counts()
    colors = {"ventral": "#457B9D", "dorsal": "#E63946"}
    bars = ax.bar(
        bank_counts.index,
        bank_counts.values,
        color=[colors.get(b, "#888888") for b in bank_counts.index],
        alpha=0.9,
        edgecolor="white",
    )
    ax.set_ylabel("Number of SUA Units")
    ax.set_title("Units by Electrode Bank")

    for bar, count in zip(bars, bank_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Panel 2: Significance rates by bank and feature group
    ax = axes[1]

    # Discover significance columns dynamically
    sig_cols = []
    sig_labels = []
    sig_colors = []

    for g in _GROUP_ORDER:
        col = f"sig_{g}"
        if col in results_df.columns:
            sig_cols.append(col)
            sig_labels.append(_GROUP_DISPLAY.get(g, g.upper()))
            sig_colors.append(_GROUP_COLORS.get(g, "#888888"))
    # Legacy fallback for combined KIN column
    if "sig_kin" in results_df.columns and "sig_body_kin" not in results_df.columns:
        sig_cols.append("sig_kin")
        sig_labels.append("KIN")
        sig_colors.append("#E63946")

    if len(sig_cols) == 0:
        plt.close(fig)
        return None

    # Compute significance rates by bank
    banks = ["ventral", "dorsal"]
    x = np.arange(len(banks))
    width = 0.8 / len(sig_cols)

    for i, (col, label, color) in enumerate(zip(sig_cols, sig_labels, sig_colors)):
        rates = []
        for bank in banks:
            bank_df = known_banks[known_banks["unit_bank"] == bank]
            if len(bank_df) > 0:
                rate = 100 * bank_df[col].sum() / len(bank_df)
            else:
                rate = 0
            rates.append(rate)

        offset = (i - len(sig_cols) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=label, color=color, alpha=0.9, edgecolor="white")

        # Add value labels
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(banks)
    ax.set_ylabel("Significant Units (%)")
    ax.set_title("Encoding by Bank and Feature Group")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Add space for labels

    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / "fig2b_by_bank.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def generate_fig2b_panel_nway(
    results_df: pd.DataFrame,
    output_dir: str,
    effect_threshold: float = 0.005,
) -> None:
    """Generate all Figure 2B panels for N-way partitioning.

    Discovers groups dynamically. Works for 3-way or 4-way (with POS).

    Args:
        results_df: Results DataFrame from run_fig2b_encoding.
        output_dir: Directory to save figures.
        effect_threshold: Effect size threshold for significance.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Stacked bar: variance partitioning
    plot_fig2b_nway_variance_partitioning(
        results_df,
        output_path=str(Path(output_dir) / "fig2b_nway_variance_partitioning.png"),
    )

    # 3-panel scatter (legacy, still useful for pairwise view)
    plot_fig2b_3way_scatter(
        results_df,
        output_path=str(Path(output_dir) / "fig2b_3way_scatter.png"),
    )

    # Summary bars
    plot_fig2b_nway_summary_bars(
        results_df,
        output_path=str(Path(output_dir) / "fig2b_nway_summary_bars.png"),
    )

    plt.close("all")


# Backward-compatible alias
generate_fig2b_panel_3way = generate_fig2b_panel_nway


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
]
