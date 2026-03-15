"""Figure 2 composite: 7-panel assembly for Nature Communications.

Assembles panels A-G into a single publication-ready figure using nested
subgridspecs. Panel A is a placeholder (schematic added in Illustrator).
Panels B-G are drawn from production data.

Layout::

    Row 0: [A: placeholder]   [B: Testard boxplots]  [C: scatter + donut + marginals]
    Row 1: [D: height decode]  [E: zone decode]  [F: variance stacked]  [G: spatial map]
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from freemat.single_fm.viz.pub_data import (
    GROUP_COLORS,
    GROUP_LABELS,
    GROUPS,
    setup_nature_style,
)

# Re-use drawing primitives from existing panel modules
from freemat.single_fm.viz.fig2b_testard import (
    _discover_groups as _discover_groups_testard,
    _draw_boxes,
    _get_values,
    _GROUP_DISPLAY as _TESTARD_GROUP_DISPLAY,
)
from freemat.single_fm.viz.fig2f_subpopulation import (
    _CAT_COLORS,
    classify_units,
)
from freemat.single_fm.viz.fig2d_panels import _draw_bar_strip_panel
from freemat.single_fm.viz.fig2b_panels import (
    _discover_groups as _discover_groups_panels,
    _GROUP_COLORS as _PANEL_GROUP_COLORS,
    _resolve_unique_col,
)
from freemat.single_fm.viz.spatial_residuals import _bin_rate_map

# ── Shared helpers ────────────────────────────────────────────────────────


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


# ── Panel B: Testard split boxplots ───────────────────────────────────────


def _draw_panel_b_boxplots(
    gs_cell: GridSpec,
    fig: plt.Figure,
    df: pd.DataFrame,
) -> Tuple[plt.Axes, plt.Axes]:
    """Draw Testard-style split boxplots (full contribution top, unique bottom).

    Args:
        gs_cell: GridSpec cell to subdivide for the two sub-panels.
        fig: Parent figure.
        df: Summary DataFrame (M2 sessions).

    Returns:
        Tuple of (ax_full, ax_unique).
    """
    import matplotlib.colors as mcolors

    sub = gs_cell.subgridspec(2, 1, hspace=0)
    ax_full = fig.add_subplot(sub[0])
    ax_unique = fig.add_subplot(sub[1], sharex=ax_full)

    group_keys = _discover_groups_testard(df)
    if not group_keys:
        group_keys = ["vis", "body_kin", "head_kin", "pos"]

    full_cols = [f"full_contrib_{g}" for g in group_keys if f"full_contrib_{g}" in df.columns]
    unique_cols = [f"unique_{g}_shuffle" for g in group_keys if f"unique_{g}_shuffle" in df.columns]
    groups = [_TESTARD_GROUP_DISPLAY.get(g, g.upper()) for g in group_keys]

    has_bank = "unit_bank" in df.columns
    x_base = np.arange(len(groups))
    green = "#2ca02c"
    green_dark = "#1a7a1a"

    # Top: Full Contribution
    _draw_boxes(
        ax_full, df, full_cols, x_base, has_bank,
        exclude_nonpositive=True, box_color=green,
    )
    ax_full.set_title("Variance decomposition", fontsize=8, fontweight="bold")
    ax_full.set_ylabel("Full\ncontrib.", fontsize=7, labelpad=2)
    ax_full.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax_full.spines["bottom"].set_visible(False)
    ax_full.tick_params(bottom=False, labelbottom=False, labelsize=6)

    # Bottom: Unique Contribution (inverted)
    _draw_boxes(
        ax_unique, df, unique_cols, x_base, has_bank,
        exclude_nonpositive=True, box_color=green_dark,
    )
    ax_unique.set_ylabel("Unique\ncontrib.", fontsize=7, labelpad=2)
    ax_unique.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax_unique.invert_yaxis()
    ax_unique.spines["top"].set_visible(False)

    # X-axis labels
    ax_unique.set_xticks(x_base)
    ax_unique.set_xticklabels(groups, fontweight="bold", fontsize=6, rotation=20, ha="right")

    # Shared y-limit
    all_plotted = []
    for cols_list in [full_cols, unique_cols]:
        vals = _get_values(df, cols_list, exclude_nonpositive=True)
        for arr in vals:
            if len(arr) > 0:
                all_plotted.append(arr)
    if all_plotted:
        combined = np.concatenate(all_plotted)
        p95 = float(np.percentile(combined, 95))
        shared_max = max(np.ceil(p95 / 0.05) * 0.05, 0.05)
    else:
        shared_max = 0.2
    ax_full.set_ylim(0, shared_max)
    ax_unique.set_ylim(shared_max, 0)

    ax_full.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax_unique.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

    # Remove 0 tick from top panel to avoid overlap with bottom panel's 0
    ax_full.set_yticks([t for t in ax_full.get_yticks() if t > 0])

    # Bank legend — bottom-left of unique (bottom) panel
    if has_bank:
        ventral_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=green, ec="black", linewidth=0.8, alpha=0.85,
        )
        dorsal_rgba = mcolors.to_rgba(green, alpha=0.35)
        dorsal_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=dorsal_rgba, ec="black", linewidth=0.8, linestyle="--",
        )
        ax_unique.legend(
            [ventral_patch, dorsal_patch],
            ["Ventral", "Dorsal"],
            loc="lower left", frameon=False, fontsize=5,
        )

    return ax_full, ax_unique


# ── Panel C: Subpopulation scatter with donut + marginals ────────────────


def _draw_panel_c_scatter(
    gs_cell: GridSpec,
    fig: plt.Figure,
    df: pd.DataFrame,
) -> plt.Axes:
    """Draw VIS vs POS scatter with category coloring, donut, and marginals.

    Args:
        gs_cell: GridSpec cell to subdivide for scatter + marginals.
        fig: Parent figure.
        df: Summary DataFrame (all sessions, encoding-positive).

    Returns:
        Main scatter axes.
    """
    sub = gs_cell.subgridspec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.02, wspace=0.02,
    )
    ax = fig.add_subplot(sub[1, 0])
    ax_hist_x = fig.add_subplot(sub[0, 0], sharex=ax)
    ax_hist_y = fig.add_subplot(sub[1, 1], sharey=ax)
    ax_corner = fig.add_subplot(sub[0, 1])
    ax_corner.axis("off")

    # Classify and filter
    vis_col = "unique_vis_shuffle"
    pos_col = "unique_pos_shuffle"
    cats = classify_units(df, sig_vis_col="sig_vis", sig_pos_col="sig_pos")

    sig_mask = cats != "Neither"
    df_sig = df[sig_mask].copy()
    cats_sig = cats[sig_mask]

    # Count categories (sig_vis/sig_pos → matches production CSV: n=871)
    n_both = int((cats_sig == "Both").sum())
    n_pos = int((cats_sig == "POS only").sum())
    n_vis = int((cats_sig == "VIS only").sum())
    n_sig = n_both + n_pos + n_vis
    print(f"  Panel C: n_sig={n_sig}, Both={n_both}, POS={n_pos}, VIS={n_vis}")
    cat_counts = {"Both": n_both, "POS only": n_pos, "VIS only": n_vis}

    x = df_sig[vis_col].values
    y = df_sig[pos_col].values

    # Scatter by category
    for cat in ["Both", "VIS only", "POS only"]:
        mask = cats_sig == cat
        if mask.sum() == 0:
            continue
        ax.scatter(
            x[mask], y[mask],
            c=_CAT_COLORS[cat], alpha=0.4, s=8,
            edgecolors="none",
            label=f"{cat} ({cat_counts[cat]})",
            zorder=2 if cat == "Both" else 3,
            rasterized=True,
        )

    # Diagonal + limits
    xlim = 0.15
    ylim = min(np.nanmax(y) * 1.15, 0.25) if len(y) > 0 else 0.25
    ax.plot([0, max(xlim, ylim)], [0, max(xlim, ylim)], "k--", lw=0.6, alpha=0.4, zorder=1)
    ax.set_xlim(-0.003, xlim)
    ax.set_ylim(-0.003, ylim)

    ax.set_xlabel("Unique VIS $\\Delta R^2$", fontsize=8)
    ax.set_ylabel("Unique POS $\\Delta R^2$", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, fontsize=5, loc="upper right")

    # Marginal histograms
    hist_kw = dict(bins=30, alpha=0.4, edgecolor="none")
    for cat in ["Both", "VIS only", "POS only"]:
        mask = cats_sig == cat
        if mask.sum() == 0:
            continue
        ax_hist_x.hist(x[mask], color=_CAT_COLORS[cat], **hist_kw)
        ax_hist_y.hist(y[mask], orientation="horizontal", color=_CAT_COLORS[cat], **hist_kw)

    ax_hist_x.set_title("Population encoding", fontsize=8, fontweight="bold")
    ax_hist_x.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_hist_y.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    for spine in ax_hist_x.spines.values():
        spine.set_visible(False)
    for spine in ax_hist_y.spines.values():
        spine.set_visible(False)

    # Donut inset
    inset_ax = ax.inset_axes([0.55, -0.02, 0.48, 0.48])
    display_cats = ["Both", "POS only", "VIS only"]
    donut_vals = [cat_counts[c] for c in display_cats]
    donut_colors = [_CAT_COLORS[c] for c in display_cats]
    wedges, texts, autotexts = inset_ax.pie(
        donut_vals,
        colors=donut_colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 0.8},
        textprops={"fontsize": 5},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(5)
        at.set_fontweight("bold")
    inset_ax.text(0, 0, f"n={n_sig}", ha="center", va="center", fontsize=6, fontweight="bold")

    return ax


# ── Panel D & E: Decoder bars ────────────────────────────────────────────


def _draw_decoder_panel(
    ax: plt.Axes,
    decoder_df: pd.DataFrame,
    target: str,
    title: str,
    cv_folds: int = 5,
) -> None:
    """Draw raw vs residual decoder bar+strip panel.

    Args:
        ax: Target axes.
        decoder_df: Decoder summary DataFrame.
        target: Decoder target (``"height_id"`` or ``"zone_id"``).
        title: Panel title.
        cv_folds: CV fold setting to use.
    """
    if decoder_df.empty:
        ax.text(0.5, 0.5, f"No {target} data", ha="center", va="center",
                fontsize=8, color="gray", transform=ax.transAxes)
        ax.axis("off")
        return

    t_df = decoder_df[(decoder_df["cv_folds"] == cv_folds) & (decoder_df["target"] == target)].copy()
    if t_df.empty:
        ax.text(0.5, 0.5, f"No {target} data", ha="center", va="center",
                fontsize=8, color="gray", transform=ax.transAxes)
        ax.axis("off")
        return

    t_pivot = t_df.pivot_table(
        index="session_id", columns="model", values="acc",
        aggfunc="mean", dropna=False,
    )
    if "raw" not in t_pivot.columns or "resid" not in t_pivot.columns:
        ax.text(0.5, 0.5, "Missing raw/resid", ha="center", va="center",
                fontsize=8, color="gray", transform=ax.transAxes)
        return
    t_pivot = t_pivot.dropna(subset=["raw", "resid"])
    if t_pivot.empty:
        return

    chance = float(t_df["chance_theory"].iloc[0])
    perm_chance = None
    if "chance_perm_mean" in t_df.columns and np.isfinite(t_df["chance_perm_mean"]).any():
        perm_chance = float(np.nanmean(t_df["chance_perm_mean"].to_numpy(dtype=float)))

    _draw_bar_strip_panel(ax, t_pivot, chance, perm_chance, title, bar_width=0.3)
    ax.tick_params(labelsize=6)
    ax.set_ylabel("Balanced acc.", fontsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


# ── Panel F: Variance decomposition stacked bars ─────────────────────────


def _draw_panel_f_variance(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw N-way variance partitioning stacked bar chart.

    Args:
        ax: Target axes.
        df: Summary DataFrame with unique_*_shuffle and full_r2 columns.
    """
    group_keys = _discover_groups_panels(df)
    if not group_keys:
        group_keys = GROUPS

    df_sorted = df.sort_values("full_r2", ascending=False).reset_index(drop=True)
    n_units = len(df_sorted)
    x = np.arange(n_units)

    unique_arrays: Dict[str, np.ndarray] = {}
    sum_unique = np.zeros(n_units)
    for g in group_keys:
        col = _resolve_unique_col(df_sorted, g)
        vals = np.maximum(df_sorted[col].values, 0)
        unique_arrays[g] = vals
        sum_unique += vals

    shared = np.maximum(df_sorted["full_r2"].values - sum_unique, 0)

    bottom = np.zeros(n_units)
    for g in group_keys:
        color = GROUP_COLORS.get(g, _PANEL_GROUP_COLORS.get(g, "#888888"))
        ax.bar(x, unique_arrays[g], bottom=bottom, color=color, alpha=0.9, width=1.0)
        bottom += unique_arrays[g]
    ax.bar(x, shared, bottom=bottom, color="#A8DADC", alpha=0.9, width=1.0)

    ax.set_xticks([])
    ax.set_ylabel("CV-R²", fontsize=7)
    ax.set_title("Variance\npartitioning", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)

    # Compact legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GROUP_COLORS.get(g, "#888"), label=GROUP_LABELS.get(g, g.upper()))
               for g in group_keys]
    handles.append(Patch(facecolor="#A8DADC", label="Shared"))
    ax.legend(
        handles=handles, fontsize=8, frameon=False,
        loc="upper right", ncol=1,
    )


# ── Panel G: Spatial modulation map ──────────────────────────────────────


def _draw_panel_g_spatial(
    gs_cell: GridSpec,
    fig: plt.Figure,
    residuals_df: Optional[pd.DataFrame],
    summary_df: Optional[pd.DataFrame],
    n_bins: int = 30,
    min_occupancy: int = 5,
) -> plt.Axes:
    """Draw occupancy + top-unit spatial rate map.

    Args:
        gs_cell: GridSpec cell to subdivide vertically.
        fig: Parent figure.
        residuals_df: Residuals DataFrame for one session.
        summary_df: Summary DataFrame for unit ranking.
        n_bins: Number of spatial bins per dimension.
        min_occupancy: Minimum bin visits.

    Returns:
        Bottom axes (rate map).
    """
    sub = gs_cell.subgridspec(2, 1, hspace=0.55)
    ax_occ = fig.add_subplot(sub[0])
    ax_map = fig.add_subplot(sub[1])

    if residuals_df is None or summary_df is None or residuals_df.empty:
        _draw_placeholder(ax_occ, "Occupancy\n(no data)")
        _draw_placeholder(ax_map, "Rate map\n(no data)")
        return ax_map

    # Shared bin edges
    x_edges = np.linspace(
        residuals_df["com_x"].min() - 0.05,
        residuals_df["com_x"].max() + 0.05,
        n_bins + 1,
    )
    y_edges = np.linspace(
        residuals_df["com_y"].min() - 0.05,
        residuals_df["com_y"].max() + 0.05,
        n_bins + 1,
    )

    # Occupancy
    all_pos = residuals_df.drop_duplicates(subset=["com_x", "com_y", "t_bin_s"])
    x_idx = np.clip(np.digitize(all_pos["com_x"].values, x_edges) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(all_pos["com_y"].values, y_edges) - 1, 0, n_bins - 1)
    occupancy = np.zeros((n_bins, n_bins), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        occupancy[yi, xi] += 1

    dt = 0.5
    if "t_bin_s" in all_pos.columns and len(all_pos) > 1:
        dt = float(np.median(np.diff(np.sort(all_pos["t_bin_s"].unique()))))
    occ_seconds = occupancy * dt
    occ_display = np.ma.masked_where(occupancy == 0, occ_seconds)

    im_occ = ax_occ.pcolormesh(x_edges, y_edges, occ_display, cmap="gray_r", rasterized=True)
    cbar_occ = fig.colorbar(im_occ, ax=ax_occ, fraction=0.046, pad=0.04)
    cbar_occ.ax.tick_params(labelsize=4)
    cbar_occ.set_label("s", fontsize=5)
    ax_occ.set_title("Occupancy (s)", fontsize=7, fontweight="bold")
    ax_occ.set_xticklabels([])
    ax_occ.set_yticklabels([])
    ax_occ.tick_params(length=0)
    ax_occ.set_xlabel("")
    ax_occ.set_ylabel("y (m)", fontsize=6)

    # Top POS unit rate map
    rank_col = "unique_pos_shuffle"
    if rank_col not in summary_df.columns:
        _draw_placeholder(ax_map, "No POS col")
        return ax_map

    # Filter summary to M2_0526 if session_id available
    if "session_id" in summary_df.columns:
        sess_ids = residuals_df["session_id"].unique() if "session_id" in residuals_df.columns else []
        if len(sess_ids) > 0:
            summary_df = summary_df[summary_df["session_id"].isin(sess_ids)]

    ranked = summary_df.dropna(subset=[rank_col]).sort_values(rank_col, ascending=False)
    if ranked.empty:
        _draw_placeholder(ax_map, "No ranked units")
        return ax_map

    top_unit = ranked.iloc[0]
    uid = top_unit["unit_id"]
    unit_data = residuals_df[residuals_df["unit_id"] == uid]

    rate_map, _ = _bin_rate_map(
        unit_data["com_x"].values,
        unit_data["com_y"].values,
        unit_data["y_true"].values,
        x_edges, y_edges,
        min_occupancy=min_occupancy,
    )

    im_map = ax_map.pcolormesh(x_edges, y_edges, rate_map, cmap="hot", rasterized=True)
    cbar_map = fig.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
    cbar_map.ax.tick_params(labelsize=4)
    cbar_map.set_label("sp/s", fontsize=5)
    bank = top_unit.get("unit_bank", "?")
    pos_val = top_unit.get(rank_col, 0)
    # Place unit info as text inside plot to avoid title overlap with top panel
    ax_map.text(
        0.02, 0.97, f"Unit {uid} ({bank})\nPOS={pos_val:.3f}",
        transform=ax_map.transAxes, fontsize=5, va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax_map.set_xticklabels([])
    ax_map.set_yticklabels([])
    ax_map.tick_params(length=0)
    ax_map.set_xlabel("x (m)", fontsize=6)
    ax_map.set_ylabel("y (m)", fontsize=6)

    return ax_map


# ══════════════════════════════════════════════════════════════════════════
# Main composite assembly
# ══════════════════════════════════════════════════════════════════════════


def fig2_composite(
    fig2_all_df: pd.DataFrame,
    decoder_df: pd.DataFrame,
    residuals_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[float, float] = (7.1, 5.5),
) -> plt.Figure:
    """Assemble Figure 2 as a 2-row, 7-panel composite.

    Layout::

        Row 0: [A: placeholder] [B: Testard boxplots] [C: scatter+donut+marginals]
        Row 1: [D: height decode] [E: zone decode] [F: variance stacked] [G: spatial map]

    Args:
        fig2_all_df: Summary DataFrame from ``load_fig2(scope="all4")``.
        decoder_df: Decoder summary from ``load_fig2d_decoders()``.
        residuals_df: Residuals DataFrame for one session (e.g. M2_0526).
        figsize: Figure size in inches.

    Returns:
        Assembled Figure.
    """
    setup_nature_style()

    fig = plt.figure(figsize=figsize)

    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 0.9],
        hspace=0.40,
        left=0.08, right=0.95, top=0.97, bottom=0.06,
    )

    row0 = outer[0].subgridspec(1, 3, wspace=0.45, width_ratios=[1, 1.2, 1.8])
    row1 = outer[1].subgridspec(1, 4, wspace=0.35, width_ratios=[0.8, 0.8, 2.4, 0.9])

    # ═══ Row 0 ═══

    # Panel A: Placeholder
    ax_a = fig.add_subplot(row0[0, 0])
    _draw_placeholder(ax_a, "Schematic\n(add in Illustrator)")

    # Panel B: Testard boxplots (M2 only)
    hedy_sessions = ["M2_0526", "M2_0605"]
    df_hedy = fig2_all_df[fig2_all_df["session_id"].isin(hedy_sessions)].copy()
    ax_b_full, ax_b_unique = _draw_panel_b_boxplots(row0[0, 1], fig, df_hedy)

    # Panel C: Subpopulation scatter (all sessions — classify on full population
    # so FDR significance alone drives the n=871 count, matching production CSVs)
    ax_c = _draw_panel_c_scatter(row0[0, 2], fig, fig2_all_df)

    # ═══ Row 1 ═══

    # Panel D: Height decoding
    ax_d = fig.add_subplot(row1[0, 0])
    _draw_decoder_panel(ax_d, decoder_df, "height_id", "Height\n(3-class)")
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # Move n=X text to bottom-left
    for txt in ax_d.texts:
        if txt.get_text().startswith("n ="):
            txt.set_position((0.02, 0.05))
            txt.set_va("bottom")

    # Panel E: Zone decoding
    ax_e = fig.add_subplot(row1[0, 1])
    _draw_decoder_panel(ax_e, decoder_df, "zone_id", "Position\n(9-class)")
    ax_e.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for txt in ax_e.texts:
        if txt.get_text().startswith("n ="):
            txt.set_position((0.02, 0.05))
            txt.set_va("bottom")

    # Panel F: Variance decomposition (M2)
    ax_f = fig.add_subplot(row1[0, 2])
    _draw_panel_f_variance(ax_f, df_hedy)
    ax_f.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Panel G: Spatial modulation map
    # Pass summary for unit ranking (filter to session in residuals)
    ax_g = _draw_panel_g_spatial(row1[0, 3], fig, residuals_df, fig2_all_df)

    # ═══ Panel labels ═══
    fig.canvas.draw()

    # Shift Panel B content right, keep label at original position
    b_orig_x0 = ax_b_full.get_position().x0
    b_shift = 0.03
    for ax_b in [ax_b_full, ax_b_unique]:
        p = ax_b.get_position()
        ax_b.set_position([p.x0 + b_shift, p.y0, p.width - b_shift, p.height])

    row0_axes = [ax_a, ax_b_full, ax_c]
    row0_top = max(ax.get_position().y1 for ax in row0_axes)

    row1_axes = [ax_d, ax_e, ax_f, ax_g]
    row1_top = max(ax.get_position().y1 for ax in row1_axes)

    _add_panel_label(ax_a, "A", x_offset=-0.04, force_y=row0_top)
    b_label_x_offset = b_orig_x0 - ax_b_full.get_position().x0 - 0.01
    _add_panel_label(ax_b_full, "B", x_offset=b_label_x_offset, force_y=row0_top)
    _add_panel_label(ax_c, "C", force_y=row0_top)
    _add_panel_label(ax_d, "D", x_offset=-0.04, force_y=row1_top)
    _add_panel_label(ax_e, "E", force_y=row1_top)
    _add_panel_label(ax_f, "F", force_y=row1_top)
    _add_panel_label(ax_g, "G", force_y=row1_top)

    return fig
