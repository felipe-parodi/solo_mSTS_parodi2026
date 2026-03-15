"""Figure 3 composite: 2×3 panel assembly .

Assembles 6 panels (A-F) into a single publication-ready figure using
nested subgridspecs. Panels A,B are placeholders (schematic + pie charts,
added in Illustrator). Panels C-F are drawn from production data.

Layout::

    Row 0: [A: schematic]   [B: pie charts]   [C: CDF]
    Row 1: [D: scatter]     [E: RF-index]     [F: bank dot]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, spearmanr

from freemat.single_fm.viz.pub_data import (
    FRAME_COLORS,
    FRAME_LABELS,
    FRAMES,
    setup_nature_style,
    winsorize,
)

# Default schematic path (relative to project root)
_SCHEMATIC_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "results" / "single_fm" / "publication_figures"
    / "main" / "fig3" / "fig3_reference_frame_schematic.png"
)


# ── Shared helpers ────────────────────────────────────────────────────────

def _thin_yticks(ax: plt.Axes, nbins: int = 4) -> None:
    """Reduce y-axis ticks to ~nbins for visual clarity."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=False))


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    x_offset: float = -0.01,
    y_pad: float = 0.005,
    force_y: Optional[float] = None,
) -> None:
    """Add bold uppercase panel label at top-left of axes.

    Args:
        ax: Target axes.
        label: Panel letter.
        x_offset: Horizontal offset from axes left edge (fig coords).
        y_pad: Vertical padding above the reference y position.
        force_y: If provided, use this y-reference for row alignment.
    """
    fig = ax.get_figure()
    pos = ax.get_position()
    fig_x = pos.x0 + x_offset
    fig_y = (force_y if force_y is not None else pos.y1) + y_pad
    fig.text(
        fig_x, fig_y, label,
        fontsize=16, fontweight="bold", va="bottom", ha="right",
    )


def _align_row_titles(row_axes: List[plt.Axes], fig: plt.Figure) -> None:
    """Align titles in a row to the highest axes top in figure coords."""
    fig.canvas.draw()
    tops = [ax.get_position().y1 for ax in row_axes]
    max_top = max(tops)
    for ax in row_axes:
        pos = ax.get_position()
        ax_height = pos.y1 - pos.y0
        if ax_height > 0:
            title_y = 1.0 + (max_top - pos.y1) / ax_height
            ax.title.set_position((0.5, title_y))


# ── Panel A,B: Placeholders ──────────────────────────────────────────────

def _draw_placeholder(ax: plt.Axes, text: str) -> None:
    """Draw blank placeholder panel with centered gray label."""
    ax.text(
        0.5, 0.5, text,
        ha="center", va="center", fontsize=11,
        color="gray", fontstyle="italic",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_facecolor("none")


def _draw_panel_a_schematic(
    ax: plt.Axes,
    schematic_path: Optional[Path] = None,
) -> None:
    """Embed reference-frame schematic image in Panel A."""
    path = schematic_path or _SCHEMATIC_PATH
    if path.exists():
        img = mpimg.imread(str(path))
        # Crop transparent margins
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            rows = np.any(alpha > 0.05, axis=1)
            cols = np.any(alpha > 0.05, axis=0)
            if rows.any() and cols.any():
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                img = img[r0:r1 + 1, c0:c1 + 1]
        # Mask white background to transparent
        if img.ndim == 3 and img.shape[2] == 4:
            white = (
                (img[:, :, 0] > 0.95)
                & (img[:, :, 1] > 0.95)
                & (img[:, :, 2] > 0.95)
            )
            img[white, 3] = 0
        ax.imshow(img, aspect="equal")
    else:
        ax.text(
            0.5, 0.5, "Schematic\n(not found)",
            ha="center", va="center", fontsize=11,
            color="gray", fontstyle="italic",
            transform=ax.transAxes,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_facecolor("none")


# ── Panel C: CDF of ΔR² by frame ─────────────────────────────────────────

def _draw_panel_c_cdf(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw cumulative distribution of ΔR² for each reference frame.

    Args:
        ax: Target axes.
        df: Summary DataFrame with ``delta_r2_allo/body/head`` columns.
    """
    for f in FRAMES:
        vals = df[f"delta_r2_{f}"].dropna().values
        vals_sorted = np.sort(vals)
        cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
        ax.plot(
            vals_sorted, cdf, color=FRAME_COLORS[f],
            label=FRAME_LABELS[f], linewidth=2,
        )

    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("$\\Delta R^2$", fontsize=9)
    ax.set_ylabel("Cumulative fraction", fontsize=9)
    ax.set_title("CDF of $\\Delta R^2$", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    _thin_yticks(ax)


# ── Panel D: Allo vs Body scatter ────────────────────────────────────────

def _draw_panel_d_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw allocentric vs body-centric ΔR² scatter.

    Args:
        ax: Target axes.
        df: Summary DataFrame with ``delta_r2_allo`` and ``delta_r2_body``.
    """
    x = winsorize(df["delta_r2_allo"].values)
    y = winsorize(df["delta_r2_body"].values)
    fin = np.isfinite(x) & np.isfinite(y)

    ax.scatter(
        x[fin], y[fin], s=8, alpha=0.3,
        color="#457B9D", rasterized=True,
    )

    lim = max(np.nanmax(x[fin]), np.nanmax(y[fin])) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)

    ax.set_xlabel("$\\Delta R^2$ Allocentric", fontsize=9)
    ax.set_ylabel("$\\Delta R^2$ Body-centric", fontsize=9)

    # Spearman annotation
    r, p = spearmanr(x[fin], y[fin])
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(
        0.05, 0.95, f"$\\rho$ = {r:.2f}\n{p_str}",
        transform=ax.transAxes, fontsize=7, va="top",
    )

    # Win count
    n_allo = int((x[fin] > y[fin]).sum())
    n_total = int(fin.sum())
    ax.text(
        0.95, 0.05,
        f"Allo wins:\n{n_allo}/{n_total} ({100 * n_allo / n_total:.0f}%)",
        transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white",
            edgecolor="grey", alpha=0.8,
        ),
    )

    ax.set_title("Allo vs Body $\\Delta R^2$", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    _thin_yticks(ax)


# ── Panel E: RF-index violins by height tier ─────────────────────────────

_HEIGHT_TIER_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd"]
_HEIGHT_TIER_NAMES = ["Floor", "Bench", "Ceiling"]
_HEIGHT_TIER_ORDER = ["floor", "bench", "ceiling"]


def _draw_panel_e_rf_index(ax: plt.Axes, rf_df: pd.DataFrame) -> None:
    """Draw RF-index violin plots stratified by height tier.

    Args:
        ax: Target axes.
        rf_df: DataFrame with ``rf_index``/``rf_index_tier`` and
            ``height_tier`` columns.
    """
    # Support both column names
    rf_col = "rf_index_tier" if "rf_index_tier" in rf_df.columns else "rf_index"
    finite = rf_df[rf_df[rf_col].notna()].copy()
    finite["_rf"] = finite[rf_col]
    if finite.empty or "height_tier" not in finite.columns:
        ax.text(
            0.5, 0.5, "No RF-index data",
            ha="center", va="center", fontsize=9, color="gray",
            transform=ax.transAxes,
        )
        return

    data = []
    labels = []
    for i, tier in enumerate(_HEIGHT_TIER_ORDER):
        tier_data = finite[finite["height_tier"] == tier]["_rf"].to_numpy()
        if len(tier_data) > 0:
            data.append(tier_data)
            labels.append(_HEIGHT_TIER_NAMES[i])

    if not data:
        return

    positions = list(range(len(data)))
    parts = ax.violinplot(
        data, positions=positions,
        showmeans=False, showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_HEIGHT_TIER_COLORS[i % len(_HEIGHT_TIER_COLORS)])
        pc.set_alpha(0.6)

    # Style median lines
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RF-index", fontsize=9)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("RF-index\nby height", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    _thin_yticks(ax)

    # Annotate medians at their actual y-position (to the right of each violin)
    for i, d in enumerate(data):
        med = float(np.nanmedian(d))
        ax.text(
            i + 0.35, med, f"{med:+.2f}",
            ha="left", va="center", fontsize=6, fontweight="bold",
        )


# ── Panel F: Bank median ΔR² dot plot ────────────────────────────────────

def _draw_panel_f_bank(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw Cleveland dot plot of median ΔR² by bank and frame.

    Args:
        ax: Target axes.
        df: Summary DataFrame with ``unit_bank`` and ``delta_r2_*`` columns.
    """
    if "unit_bank" not in df.columns:
        ax.text(
            0.5, 0.5, "No bank data",
            ha="center", va="center", fontsize=9, color="gray",
            transform=ax.transAxes,
        )
        return

    banks = ["ventral", "dorsal"]
    rng = np.random.default_rng(42)

    for i, f in enumerate(FRAMES):
        for j, bank in enumerate(banks):
            bdf = df[df["unit_bank"] == bank]
            vals = bdf[f"delta_r2_{f}"].dropna().values
            if len(vals) == 0:
                continue
            med = np.median(vals)

            # Bootstrap 95% CI
            boot_meds = [
                np.median(rng.choice(vals, size=len(vals), replace=True))
                for _ in range(1000)
            ]
            ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])

            y_pos = i + (j - 0.5) * 0.25
            marker = "o" if bank == "ventral" else "^"
            ax.errorbar(
                med, y_pos,
                xerr=[[med - ci_lo], [ci_hi - med]],
                fmt=marker, color=FRAME_COLORS[f], markersize=6,
                capsize=3, linewidth=1.2,
            )

        # Mann-Whitney test
        v_vals = df.loc[df["unit_bank"] == "ventral", f"delta_r2_{f}"].dropna()
        d_vals = df.loc[df["unit_bank"] == "dorsal", f"delta_r2_{f}"].dropna()
        if len(v_vals) > 0 and len(d_vals) > 0:
            _, mw_p = mannwhitneyu(v_vals, d_vals, alternative="two-sided")
            sig = ""
            if mw_p < 0.001:
                sig = "***"
            elif mw_p < 0.01:
                sig = "**"
            elif mw_p < 0.05:
                sig = "*"
            if sig:
                max_med = max(np.median(v_vals), np.median(d_vals))
                ax.text(
                    max_med + 0.005, i, sig,
                    fontsize=10, va="center", fontweight="bold",
                )

    ax.set_yticks(range(len(FRAMES)))
    ax.set_yticklabels([FRAME_LABELS[f] for f in FRAMES], fontsize=8)
    ax.set_xlabel("Median $\\Delta R^2$", fontsize=9)
    ax.set_title("Median $\\Delta R^2$\nby bank", fontsize=10, fontweight="bold")

    # Legend
    ax.legend(
        [
            plt.Line2D([0], [0], marker="o", color="grey", linestyle="", markersize=5),
            plt.Line2D([0], [0], marker="^", color="grey", linestyle="", markersize=5),
        ],
        ["Ventral", "Dorsal"],
        fontsize=6, frameon=False, loc="upper right",
    )

    ax.tick_params(labelsize=7)


# ══════════════════════════════════════════════════════════════════════════
# Main composite assembly
# ══════════════════════════════════════════════════════════════════════════

def fig3_composite(
    fig3_df: pd.DataFrame,
    rf_index_df: pd.DataFrame,
    figsize: tuple = (7.1, 4.5),
) -> plt.Figure:
    """Assemble Figure 3 as a 2×3, 6-panel composite.

    Layout::

        Row 0: [A: placeholder] [B: placeholder] [C: CDF]
        Row 1: [D: scatter]     [E: RF-index]    [F: bank dot]

    Args:
        fig3_df: Summary DataFrame from ``load_fig3()``.
        rf_index_df: RF-index DataFrame from ``load_fig3_rf_index()``.
        figsize: Figure size in inches.

    Returns:
        Assembled Figure with 6 panels (A through F).
    """
    setup_nature_style()

    fig = plt.figure(figsize=figsize)

    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 1],
        hspace=0.15,
        left=0.08, right=0.93, top=0.97, bottom=0.05,
    )

    row0 = outer[0].subgridspec(1, 3, wspace=0.55)
    row1 = outer[1].subgridspec(1, 3, wspace=0.55)

    # ═══ Row 0: A (placeholder) + B (placeholder) + C (CDF) ═══

    ax_a = fig.add_subplot(row0[0, 0])
    _draw_panel_a_schematic(ax_a)

    ax_b = fig.add_subplot(row0[0, 1])
    _draw_placeholder(ax_b, "Pie charts")

    ax_c = fig.add_subplot(row0[0, 2])
    ax_c.set_box_aspect(0.85)
    _draw_panel_c_cdf(ax_c, fig3_df)

    # ═══ Row 1: D (scatter) + E (RF-index) + F (bank) ═══

    ax_d = fig.add_subplot(row1[0, 0])
    ax_d.set_box_aspect(1.0)
    _draw_panel_d_scatter(ax_d, fig3_df)

    ax_e = fig.add_subplot(row1[0, 1])
    ax_e.set_box_aspect(1.0)
    _draw_panel_e_rf_index(ax_e, rf_index_df)

    ax_f = fig.add_subplot(row1[0, 2])
    ax_f.set_box_aspect(1.0)
    _draw_panel_f_bank(ax_f, fig3_df)

    # ═══ Panel labels + title alignment ═══
    fig.canvas.draw()

    row0_top = max(ax.get_position().y1 for ax in [ax_a, ax_b, ax_c])
    row1_top = max(ax.get_position().y1 for ax in [ax_d, ax_e, ax_f])

    _add_panel_label(ax_a, "A", x_offset=-0.04, force_y=row0_top)
    _add_panel_label(ax_b, "B", force_y=row0_top)
    _add_panel_label(ax_c, "C", force_y=row0_top)
    _add_panel_label(ax_d, "D", x_offset=-0.04, force_y=row1_top)
    _add_panel_label(ax_e, "E", force_y=row1_top)
    _add_panel_label(ax_f, "F", force_y=row1_top)

    _align_row_titles([ax_c], fig)  # only C has a title in row 0
    _align_row_titles([ax_d, ax_e, ax_f], fig)

    return fig
