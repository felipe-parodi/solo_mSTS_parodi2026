"""Figure 2D plotting: raw vs residual spatial decoder performance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_TARGET_LABELS: Dict[str, str] = {
    "height_id": "Height (3-class)",
    "zone_id": "Zone (9-class)",
}


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


def _mean_ci(values: np.ndarray) -> Tuple[float, float]:
    """Return mean and 95% CI half-width."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    if len(finite) == 1:
        return mean, 0.0
    sem = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
    return mean, 1.96 * sem


def _draw_bar_strip_panel(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    chance: float,
    perm_chance: float | None,
    title: str,
    bar_width: float = 0.35,
) -> None:
    """Draw bar + strip + paired-line panel on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        pivot: Pivoted DataFrame with index=session_id, columns=[raw, resid].
        chance: Theoretical chance level.
        perm_chance: Permutation-based chance (or None).
        title: Panel title.
        bar_width: Width of each bar.
    """
    raw_vals = pivot["raw"].to_numpy(dtype=float)
    resid_vals = pivot["resid"].to_numpy(dtype=float)
    n = len(pivot)

    # Bars: mean accuracy
    raw_mean, raw_ci = _mean_ci(raw_vals)
    resid_mean, resid_ci = _mean_ci(resid_vals)
    ax.bar(0, raw_mean, width=bar_width, color="#1f77b4", alpha=0.35, zorder=1)
    ax.bar(1, resid_mean, width=bar_width, color="#d62728", alpha=0.35, zorder=1)

    # Error bars on top of bars
    ax.errorbar(0.0, raw_mean, yerr=raw_ci, fmt="s", color="#1f77b4", markersize=7, capsize=4, linewidth=2, zorder=4)
    ax.errorbar(1.0, resid_mean, yerr=resid_ci, fmt="s", color="#d62728", markersize=7, capsize=4, linewidth=2, zorder=4)

    # Per-session dots + connecting lines
    rng = np.random.default_rng(42)
    for i in range(n):
        jx_raw = rng.uniform(-0.06, 0.06)
        jx_resid = rng.uniform(-0.06, 0.06)
        ax.plot(
            [0 + jx_raw, 1 + jx_resid],
            [raw_vals[i], resid_vals[i]],
            color="#9aa0a6", alpha=0.6, linewidth=0.8, zorder=2,
        )
        ax.scatter(0 + jx_raw, raw_vals[i], s=24, color="#1f77b4", edgecolors="white", linewidths=0.3, zorder=3)
        ax.scatter(1 + jx_resid, resid_vals[i], s=24, color="#d62728", edgecolors="white", linewidths=0.3, zorder=3)

    # Chance levels
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Chance")
    if perm_chance is not None:
        ax.axhline(perm_chance, color="#9467bd", linestyle=":", linewidth=1.2, alpha=0.8, label="Perm. chance")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw FR", "Residual FR"])
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0.02, 0.97, f"n = {n}", transform=ax.transAxes, ha="left", va="top", fontsize=9)


def _draw_paired_dots_panel(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    chance: float,
    perm_chance: float | None,
    title: str,
) -> None:
    """Draw compact paired-dots panel (no bars) for use as inset.

    Args:
        ax: Matplotlib axes to draw on.
        pivot: Pivoted DataFrame with index=session_id, columns=[raw, resid].
        chance: Theoretical chance level.
        perm_chance: Permutation-based chance (or None).
        title: Panel title.
    """
    raw_vals = pivot["raw"].to_numpy(dtype=float)
    resid_vals = pivot["resid"].to_numpy(dtype=float)
    n = len(pivot)

    # Per-session paired dots + connecting lines
    for i in range(n):
        ax.plot(
            [0, 1], [raw_vals[i], resid_vals[i]],
            color="#9aa0a6", alpha=0.7, linewidth=1.0, zorder=2,
        )
        ax.scatter(0, raw_vals[i], s=30, color="#1f77b4", edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(1, resid_vals[i], s=30, color="#d62728", edgecolors="white", linewidths=0.4, zorder=3)

    # Chance level
    ax.axhline(chance, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    if perm_chance is not None:
        ax.axhline(perm_chance, color="#9467bd", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw", "Resid"], fontsize=8)
    ax.set_title(f"{title}  n={n}", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.4, 1.4)


def plot_fig2d_decoder_pairs(
    decoder_df: pd.DataFrame,
    *,
    cv_folds: int = 5,
    figsize: Tuple[float, float] = (7.0, 4.0),
) -> plt.Figure:
    """Plot Height and Zone decoders as two side-by-side bar+strip panels.

    Left panel: Height (3-class) for all sessions.
    Right panel: Position (9-class) for sessions with valid results.
    Each panel has its own y-axis scale and minimal ticks.

    Args:
        decoder_df: Decoder summary table from fig2d_decoder_summary.parquet.
        cv_folds: Which CV-fold setting to visualize.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _setup_nature_style()

    df = decoder_df[decoder_df["cv_folds"] == cv_folds].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No Fig2D decoder rows for cv_folds={cv_folds}", ha="center", va="center")
        ax.axis("off")
        return fig

    # Prepare data for both targets
    panels = [
        ("height_id", "Height (3-class)"),
        ("zone_id", "Position (9-class)"),
    ]
    panel_data = []
    for target, label in panels:
        t_df = df[df["target"] == target].copy()
        if t_df.empty:
            panel_data.append((label, None, None, None))
            continue
        t_pivot = t_df.pivot_table(
            index="session_id", columns="model", values="acc",
            aggfunc="mean", dropna=False,
        )
        if "raw" in t_pivot.columns and "resid" in t_pivot.columns:
            t_pivot = t_pivot.dropna(subset=["raw", "resid"])
        else:
            t_pivot = t_pivot.iloc[0:0]
        if t_pivot.empty:
            panel_data.append((label, None, None, None))
            continue
        t_chance = float(t_df["chance_theory"].iloc[0])
        t_perm = None
        if "chance_perm_mean" in t_df.columns and np.isfinite(t_df["chance_perm_mean"]).any():
            t_perm = float(np.nanmean(t_df["chance_perm_mean"].to_numpy(dtype=float)))
        panel_data.append((label, t_pivot, t_chance, t_perm))

    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"wspace": 0.35},
    )

    for ax, (label, pivot, chance, perm) in zip(axes, panel_data):
        if pivot is None:
            ax.text(0.5, 0.5, f"No {label} data", ha="center", va="center",
                    fontsize=10, color="#888888", transform=ax.transAxes)
            ax.axis("off")
            continue

        _draw_bar_strip_panel(ax, pivot, chance, perm, label)

        # Minimal y-axis: only chance line tick + a few clean round ticks
        y_lo, y_hi = ax.get_ylim()
        # Round ticks at 10% increments from 0 to ceiling
        from math import ceil
        top = ceil(y_hi * 10) / 10  # round up to nearest 10%
        ticks = [t / 10 for t in range(0, int(top * 10) + 1, 1) if t % 2 == 0]
        # Always include chance
        if chance not in ticks:
            ticks.append(chance)
        ticks = sorted(set(ticks))
        # Keep only ticks within visible range
        ticks = [t for t in ticks if y_lo - 0.02 <= t <= y_hi + 0.02]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Only left panel gets y-axis label
    axes[0].set_ylabel("Balanced accuracy")
    if axes[1].get_ylabel():
        axes[1].set_ylabel("")

    # Remove per-panel legends, add a single shared one at top
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=len(handles),
            frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def save_fig2d_decoder_pairs(
    decoder_df: pd.DataFrame,
    output_path: str,
    *,
    cv_folds: int = 5,
) -> plt.Figure:
    """Save paired raw/residual decoder panel as PNG + SVG."""
    fig = plot_fig2d_decoder_pairs(decoder_df, cv_folds=cv_folds)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".svg"), bbox_inches="tight")
    return fig


__all__ = [
    "plot_fig2d_decoder_pairs",
    "save_fig2d_decoder_pairs",
]

