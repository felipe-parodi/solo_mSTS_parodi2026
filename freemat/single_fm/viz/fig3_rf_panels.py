"""Figure 3 RF-index visualization panels.

Violin plots of RF-index distribution, optionally split by bank or height tier.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_nature_style() -> None:
    """Apply Nature-style defaults consistent with other single-FM figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "svg.fonttype": "none",
        }
    )


def plot_rf_index_distribution(
    df: pd.DataFrame,
    *,
    split_by_bank: bool = True,
    figsize: tuple[float, float] = (6.0, 4.5),
) -> plt.Figure:
    """Plot RF-index distribution as violin/strip plot.

    Args:
        df: DataFrame with 'rf_index' column and optionally 'bank'.
        split_by_bank: If True, show separate violins for ventral/dorsal.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    finite = df[df["rf_index"].notna()].copy()
    if finite.empty:
        ax.text(0.5, 0.5, "No finite RF-index values", ha="center", va="center")
        ax.axis("off")
        return fig

    if split_by_bank and "bank" in finite.columns:
        banks = sorted(finite["bank"].unique())
        positions = list(range(len(banks)))
        data = [finite[finite["bank"] == b]["rf_index"].to_numpy() for b in banks]
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(banks)
        ax.set_xlabel("Bank")

        # Color violins
        colors = ["#1f77b4", "#d62728"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.6)
    else:
        data = [finite["rf_index"].to_numpy()]
        parts = ax.violinplot(data, positions=[0], showmeans=True, showmedians=True)
        ax.set_xticks([0])
        ax.set_xticklabels(["All units"])

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6, label="Balanced (0)")
    ax.set_ylabel("RF-index")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Reference Frame Index")
    ax.legend(fontsize=9, frameon=False)

    n_units = len(finite)
    median_rf = float(np.nanmedian(finite["rf_index"]))
    ax.text(
        0.98, 0.02,
        f"n={n_units}, med={median_rf:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
    )

    fig.tight_layout()
    return fig


def plot_rf_index_by_height(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    """Plot RF-index violin for floor/bench/ceiling, pooled across sessions.

    Args:
        df: DataFrame with 'rf_index' and 'height_tier' columns.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    tier_order = ["floor", "bench", "ceiling"]
    finite = df[df["rf_index"].notna()].copy()
    if finite.empty or "height_tier" not in finite.columns:
        ax.text(0.5, 0.5, "No height-stratified RF-index data", ha="center", va="center")
        ax.axis("off")
        return fig

    data = []
    labels = []
    for tier in tier_order:
        tier_data = finite[finite["height_tier"] == tier]["rf_index"].to_numpy()
        if len(tier_data) > 0:
            data.append(tier_data)
            labels.append(f"{tier}\n(n={len(tier_data)})")

    if not data:
        ax.text(0.5, 0.5, "No data for any height tier", ha="center", va="center")
        ax.axis("off")
        return fig

    positions = list(range(len(data)))
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel("RF-index")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("RF-index by Height Tier")
    fig.tight_layout()
    return fig


def save_rf_index_panels(
    df: pd.DataFrame,
    output_dir: str,
    *,
    split_by_bank: bool = True,
) -> None:
    """Save RF-index distribution and height-stratified panels.

    Args:
        df: DataFrame with rf_index and optionally bank/height_tier columns.
        output_dir: Directory to save figures.
        split_by_bank: Whether to split distribution by bank.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = plot_rf_index_distribution(df, split_by_bank=split_by_bank)
    fig.savefig(out / "fig3_rf_index_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig3_rf_index_distribution.svg", bbox_inches="tight")
    plt.close(fig)

    if "height_tier" in df.columns:
        fig = plot_rf_index_by_height(df)
        fig.savefig(out / "fig3_rf_index_by_height.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "fig3_rf_index_by_height.svg", bbox_inches="tight")
        plt.close(fig)


def plot_rf_split_half_scatter(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (5.0, 5.0),
) -> plt.Figure:
    """Scatter of RF-index half1 vs half2 with Spearman rho annotation.

    Args:
        df: Wide-format DataFrame with 'rf_index_half1', 'rf_index_half2',
            and optionally 'session_id' for color-coding.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    from scipy.stats import spearmanr

    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = df["rf_index_half1"].to_numpy(dtype=float)
    y = df["rf_index_half2"].to_numpy(dtype=float)

    # Filter to finite pairs
    valid = np.isfinite(x) & np.isfinite(y)
    x_v, y_v = x[valid], y[valid]

    if len(x_v) < 5:
        ax.text(0.5, 0.5, f"Too few paired units ({len(x_v)})", ha="center", va="center")
        ax.axis("off")
        return fig

    # Per-session coloring if available
    if "session_id" in df.columns:
        sessions = df.loc[valid, "session_id"].values
        unique_sess = sorted(set(sessions))
        cmap = plt.cm.tab10
        for i, sess in enumerate(unique_sess):
            mask = sessions == sess
            ax.scatter(x_v[mask], y_v[mask], s=12, alpha=0.6,
                       color=cmap(i % 10), label=sess, zorder=3)
        ax.legend(fontsize=8, frameon=False, markerscale=1.5)
    else:
        ax.scatter(x_v, y_v, s=12, alpha=0.5, color="#1f77b4", zorder=3)

    # Spearman
    rho, p_val = spearmanr(x_v, y_v)
    ax.text(0.05, 0.95, f"Spearman rho = {rho:.3f}\np = {p_val:.1e}\nn = {len(x_v)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10)

    # Identity line
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
              abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim, lim], [-lim, lim], color="gray", ls="--", lw=0.8, alpha=0.6, zorder=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")

    ax.set_xlabel("RF-index (half 1)")
    ax.set_ylabel("RF-index (half 2)")
    ax.set_title("Split-Half RF-index Stability")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_rf_index_distribution",
    "plot_rf_index_by_height",
    "plot_rf_split_half_scatter",
    "save_rf_index_panels",
]
