"""Fig 2F: Subpopulation scatter showing VIS vs POS encoding categories.

Scatter of unique VIS ΔR² vs unique POS ΔR², colored by significance
category (Both, POS-only, VIS-only). Donut inset shows population
composition. Addresses reviewer concern about multiplexing vs segregation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Category colors
_CAT_COLORS = {
    "Both": "#2CA02C",       # vibrant green — multiplexing highlight
    "POS only": "#7B3294",   # POS purple
    "VIS only": "#457B9D",   # VIS blue
    "Neither": "#D9D9D9",    # light grey (not plotted in scatter)
}

_CAT_ORDER = ["Both", "POS only", "VIS only", "Neither"]

# Session colors for per-session scatter
_SESSION_COLORS = {
    "M2_0526": "#2D7D9A",
    "M2_0605": "#5AACCC",
    "M1_0525": "#D4A373",
    "M1_0609": "#B07D4F",
    "M1_0613": "#8B5E3C",
}


def _resolve_sig_columns(
    df: pd.DataFrame,
    sig_vis_col: str,
    sig_pos_col: str,
) -> Tuple[str, str]:
    """Resolve significance column names, falling back if needed.

    Args:
        df: Summary DataFrame.
        sig_vis_col: Preferred VIS significance column.
        sig_pos_col: Preferred POS significance column.

    Returns:
        Tuple of (resolved_vis_col, resolved_pos_col).
    """
    for vis_candidate in [sig_vis_col, "sig_vis_shuffle_fdr", "sig_vis"]:
        if vis_candidate in df.columns:
            resolved_vis = vis_candidate
            break
    else:
        msg = f"No VIS significance column found. Tried: {sig_vis_col}, sig_vis_shuffle_fdr, sig_vis"
        raise KeyError(msg)

    for pos_candidate in [sig_pos_col, "sig_pos_shuffle_fdr", "sig_pos"]:
        if pos_candidate in df.columns:
            resolved_pos = pos_candidate
            break
    else:
        msg = f"No POS significance column found. Tried: {sig_pos_col}, sig_pos_shuffle_fdr, sig_pos"
        raise KeyError(msg)

    return resolved_vis, resolved_pos


def classify_units(
    df: pd.DataFrame,
    sig_vis_col: str = "sig_vis_shuffle_fdr",
    sig_pos_col: str = "sig_pos_shuffle_fdr",
) -> pd.Series:
    """Classify each unit into 4 categories based on significance flags.

    Args:
        df: Summary DataFrame with significance columns.
        sig_vis_col: Column name for VIS significance flag.
        sig_pos_col: Column name for POS significance flag.

    Returns:
        Series with category labels: Both, POS only, VIS only, Neither.
    """
    vis_col, pos_col = _resolve_sig_columns(df, sig_vis_col, sig_pos_col)
    sv = df[vis_col].astype(bool)
    sp = df[pos_col].astype(bool)

    cat = pd.Series("Neither", index=df.index)
    cat[sv & sp] = "Both"
    cat[sp & ~sv] = "POS only"
    cat[sv & ~sp] = "VIS only"
    return cat


def compute_category_counts(
    df: pd.DataFrame,
    sig_vis_col: str = "sig_vis_shuffle_fdr",
    sig_pos_col: str = "sig_pos_shuffle_fdr",
) -> Dict[str, Dict[str, float]]:
    """Compute category counts and percentages.

    Args:
        df: Summary DataFrame.
        sig_vis_col: VIS significance column.
        sig_pos_col: POS significance column.

    Returns:
        Dict with keys: category -> {count, pct_of_total, pct_of_sig}.
    """
    cats = classify_units(df, sig_vis_col, sig_pos_col)
    n_total = len(df)
    n_sig = (cats != "Neither").sum()

    result: Dict[str, Dict[str, float]] = {}
    for cat in _CAT_ORDER:
        count = int((cats == cat).sum())
        result[cat] = {
            "count": count,
            "pct_of_total": 100.0 * count / n_total if n_total > 0 else 0.0,
            "pct_of_sig": 100.0 * count / n_sig if n_sig > 0 and cat != "Neither" else 0.0,
        }
    result["_n_total"] = n_total  # type: ignore[assignment]
    result["_n_sig"] = int(n_sig)  # type: ignore[assignment]
    return result


def plot_fig2f_subpopulation_scatter(
    df: pd.DataFrame,
    sig_vis_col: str = "sig_vis_shuffle_fdr",
    sig_pos_col: str = "sig_pos_shuffle_fdr",
    vis_col: str = "unique_vis_shuffle",
    pos_col: str = "unique_pos_shuffle",
    min_delta_r2: float = 0.01,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, Dict]:
    """Scatter of unique VIS vs POS ΔR², colored by significance category.

    Args:
        df: Fig 2 summary DataFrame.
        sig_vis_col: VIS significance flag column.
        sig_pos_col: POS significance flag column.
        vis_col: Unique VIS ΔR² column.
        pos_col: Unique POS ΔR² column.
        min_delta_r2: Minimum ΔR² threshold — units must have
            max(vis_dr2, pos_dr2) >= this value to be plotted.
        title: Optional figure title.

    Returns:
        Tuple of (Figure, category_counts dict).
    """
    # Resolve columns
    resolved_vis_sig, resolved_pos_sig = _resolve_sig_columns(df, sig_vis_col, sig_pos_col)
    print(f"  [fig2F] Using sig columns: VIS={resolved_vis_sig}, POS={resolved_pos_sig}")

    # Classify all units (for full accounting)
    cats = classify_units(df, resolved_vis_sig, resolved_pos_sig)
    counts = compute_category_counts(df, resolved_vis_sig, resolved_pos_sig)
    n_total = counts["_n_total"]
    n_sig = counts["_n_sig"]

    # Log full composition
    print(f"  [fig2F] Population composition (n={n_total}):")
    for cat in _CAT_ORDER:
        c = counts[cat]
        print(f"    {cat}: {c['count']} ({c['pct_of_total']:.1f}% of total)")
    print(f"  [fig2F] Encoding-positive: {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")

    # Filter to sig units only
    sig_mask = cats != "Neither"
    df_sig = df[sig_mask].copy()
    cats_sig = cats[sig_mask]

    # Filter out negative ΔR² on either axis + apply min ΔR² threshold
    vis_vals = df_sig[vis_col].values
    pos_vals = df_sig[pos_col].values
    nonneg_mask = (vis_vals >= 0) & (pos_vals >= 0)
    if min_delta_r2 > 0:
        nonneg_mask &= np.maximum(vis_vals, pos_vals) >= min_delta_r2
    n_before = len(df_sig)
    df_sig = df_sig[nonneg_mask].copy()
    cats_sig = cats_sig[nonneg_mask]
    n_after = len(df_sig)
    print(f"  [fig2F] Filtering (non-neg + min_delta_r2={min_delta_r2}): "
          f"{n_before} -> {n_after} ({n_before - n_after} removed)")

    # Get data
    x = df_sig[vis_col].values
    y = df_sig[pos_col].values

    # Create figure with GridSpec: scatter + marginal histograms (4:1 ratio)
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.02, wspace=0.02,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax)

    # Plot each category — Both first (behind), specialized on top
    for cat in ["Both", "VIS only", "POS only"]:
        mask = cats_sig == cat
        if mask.sum() == 0:
            continue
        ax.scatter(
            x[mask], y[mask],
            c=_CAT_COLORS[cat],
            alpha=0.45,
            s=20,
            edgecolors="none",
            label=f"{cat} ({counts[cat]['count']})",
            zorder=2 if cat == "Both" else 3,
        )

    # Diagonal + axis limits (cap VIS at 0.15, POS auto-scaled)
    xlim = 0.15
    ylim = min(np.nanmax(y) * 1.15, 0.25) if len(y) > 0 else 0.25
    ax.plot([0, max(xlim, ylim)], [0, max(xlim, ylim)],
            "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.set_xlim(-0.003, xlim)
    ax.set_ylim(-0.003, ylim)

    ax.set_xlabel("Unique VIS $\\Delta R^2$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Unique POS $\\Delta R^2$", fontsize=14, fontweight="bold")

    ax.legend(frameon=False, fontsize=10, loc="upper right")

    # --- Marginal histograms, colored by category ---
    hist_kw = dict(bins=40, alpha=0.4, edgecolor="none")
    for cat in ["Both", "VIS only", "POS only"]:
        mask = cats_sig == cat
        if mask.sum() == 0:
            continue
        ax_hist_x.hist(x[mask], color=_CAT_COLORS[cat], **hist_kw)
        ax_hist_y.hist(y[mask], orientation="horizontal",
                       color=_CAT_COLORS[cat], **hist_kw)

    # Clean up margin axes
    ax_hist_x.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_hist_y.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    for spine in ax_hist_x.spines.values():
        spine.set_visible(False)
    for spine in ax_hist_y.spines.values():
        spine.set_visible(False)

    # Title on the top marginal axis so it doesn't overlap
    title_text = title or f"Population encoding composition (n={n_sig} sig. / {n_total} total)"
    ax_hist_x.set_title(title_text, fontsize=13, fontweight="bold")

    # Hide corner cell
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    # Donut inset — inside the scatter axes (bottom-right)
    inset_ax = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
    display_cats = ["Both", "POS only", "VIS only"]
    donut_vals = [counts[c]["count"] for c in display_cats]
    donut_colors = [_CAT_COLORS[c] for c in display_cats]

    wedges, texts, autotexts = inset_ax.pie(
        donut_vals,
        colors=donut_colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 9},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    # Center text
    inset_ax.text(
        0, 0, f"n={n_sig}",
        ha="center", va="center", fontsize=10, fontweight="bold",
    )

    return fig, counts


def plot_fig2f_by_session(
    df: pd.DataFrame,
    vis_col: str = "unique_vis_shuffle",
    pos_col: str = "unique_pos_shuffle",
    min_delta_r2: float = 0.01,
    title: Optional[str] = None,
) -> plt.Figure:
    """Scatter of unique VIS vs POS ΔR², colored by session.

    Args:
        df: Fig 2 summary DataFrame with session_id column.
        vis_col: Unique VIS ΔR² column.
        pos_col: Unique POS ΔR² column.
        min_delta_r2: Minimum ΔR² threshold for plotting.
        title: Optional figure title.

    Returns:
        Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    sessions = sorted(df["session_id"].unique())
    for sess in sessions:
        sdf = df[df["session_id"] == sess]
        x = sdf[vis_col].values
        y = sdf[pos_col].values

        # Filter negatives + min ΔR² threshold
        keep = (x >= 0) & (y >= 0)
        if min_delta_r2 > 0:
            keep &= np.maximum(x, y) >= min_delta_r2
        x = x[keep]
        y = y[keep]

        color = _SESSION_COLORS.get(sess, "#888888")
        ax.scatter(
            x, y,
            c=color,
            alpha=0.4,
            s=18,
            edgecolors="none",
            label=f"{sess} ({len(x)})",
            zorder=2,
        )

    # Diagonal + axis limits (cap VIS at 0.15, POS auto-scaled)
    xlim = 0.15
    ylim = 0.25
    ax.plot([0, max(xlim, ylim)], [0, max(xlim, ylim)],
            "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.set_xlim(-0.003, xlim)
    ax.set_ylim(-0.003, ylim)

    ax.set_xlabel("Unique VIS $\\Delta R^2$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Unique POS $\\Delta R^2$", fontsize=14, fontweight="bold")
    ax.set_title(title or "VIS vs POS encoding by session", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    return fig
