"""Testard-style boxplots and summary statistics for Figure 2B.

Implements Testard et al. 2024 Fig 3i style visualization:
full contribution (shuffle others) vs unique contribution (shuffle group).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# Display labels for figures (code name -> display name)
_GROUP_DISPLAY: Dict[str, str] = {
    "vis": "VIS",
    "body_kin": "BODY_KIN",
    "head_kin": "HEAD_KIN",
    "pos": "3D POS",
}


def _discover_groups(results_df: pd.DataFrame) -> List[str]:
    """Discover feature groups from full_contrib_* columns.

    Returns:
        Ordered list of group names (e.g., ["vis", "body_kin", "head_kin", "pos"]).
    """
    # Prefer canonical order when possible
    canonical_order = ["vis", "body_kin", "head_kin", "pos"]
    found = []
    for g in canonical_order:
        if f"full_contrib_{g}" in results_df.columns:
            found.append(g)
    # Add any unexpected groups
    for c in results_df.columns:
        if c.startswith("full_contrib_"):
            g = c.replace("full_contrib_", "")
            if g not in found:
                found.append(g)
    return found

__all__ = [
    "plot_fig2b_testard_boxplots",
    "plot_fig2c_unique_scatter",
    "compute_testard_summary",
]


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
        "axes.spines.top": True,
        "axes.spines.right": True,
        "font.family": "sans-serif",
        "svg.fonttype": "none",
    })


def _get_values(
    results_df: pd.DataFrame,
    cols: list,
    bank: Optional[str] = None,
    exclude_nonpositive: bool = False,
) -> list:
    """Get per-unit values for each column (one array per group).

    Args:
        results_df: DataFrame with per-unit results.
        cols: Column names for each group.
        bank: If set, filter to this bank before computing.
        exclude_nonpositive: If True, exclude values <= 0.

    Returns:
        List of arrays, one per column.
    """
    df = results_df if bank is None else results_df[results_df["unit_bank"] == bank]
    out = []
    for col in cols:
        vals = df[col].dropna().values
        if exclude_nonpositive:
            vals = vals[vals > 0]
        out.append(vals)
    return out


def _draw_boxes(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    cols: list,
    x_base: np.ndarray,
    has_bank: bool,
    exclude_nonpositive: bool = False,
    box_color: str = "#2ca02c",
    show_strip: bool = True,
) -> None:
    """Draw boxplots (per-unit values) with optional jittered strip overlay.

    Shows box (Q1-Q3) and median only — no whiskers or caps.
    Dorsal boxes use white hatch on green fill with black border.

    Args:
        ax: Matplotlib axes.
        results_df: DataFrame with per-unit results.
        cols: Column names for each group.
        x_base: X positions for the groups.
        has_bank: Whether to split by ventral/dorsal.
        exclude_nonpositive: If True, exclude values <= 0.
        box_color: Fill color for boxes.
        show_strip: If True, overlay jittered individual data points.
    """
    box_kw = {
        "showfliers": False,
        "patch_artist": True,
        "whis": [25, 75],
        "showcaps": False,
        "medianprops": {"color": "black", "linewidth": 1.5},
        "whiskerprops": {"color": "black", "linewidth": 0},
        "capprops": {"color": "black", "linewidth": 0},
    }

    if has_bank:
        width = 0.35
        for b_idx, bank in enumerate(["ventral", "dorsal"]):
            bank_df = results_df[results_df["unit_bank"] == bank]
            if len(bank_df) == 0:
                continue

            data = _get_values(
                results_df, cols, bank=bank, exclude_nonpositive=exclude_nonpositive,
            )
            offset = (b_idx - 0.5) * width
            x = x_base + offset
            is_dorsal = bank == "dorsal"

            if show_strip:
                rng = np.random.default_rng(42 + b_idx)
                for i, d in enumerate(data):
                    jitter = rng.uniform(-width * 0.15, width * 0.15, len(d))
                    ax.scatter(
                        x[i] + jitter, d, s=1.5, alpha=0.15, color="gray",
                        zorder=1, rasterized=True,
                    )

            bp = ax.boxplot(data, positions=x, widths=width * 0.8, zorder=2, **box_kw)

            for patch in bp["boxes"]:
                if is_dorsal:
                    # Lighter shade for dorsal (SVG hatch doesn't export to Illustrator)
                    rgba = mcolors.to_rgba(box_color, alpha=0.35)
                    patch.set_facecolor(rgba)
                    patch.set_edgecolor("black")
                    patch.set_linestyle("--")
                else:
                    patch.set_facecolor(box_color)
                    patch.set_edgecolor("black")
                    patch.set_alpha(0.85)
                patch.set_linewidth(1.0)
    else:
        data = _get_values(
            results_df, cols, exclude_nonpositive=exclude_nonpositive,
        )

        if show_strip:
            rng = np.random.default_rng(42)
            for i, d in enumerate(data):
                jitter = rng.uniform(-0.15, 0.15, len(d))
                ax.scatter(
                    x_base[i] + jitter, d, s=1.5, alpha=0.15, color="gray",
                    zorder=1, rasterized=True,
                )

        bp = ax.boxplot(data, positions=x_base, widths=0.6, zorder=2, **box_kw)
        for patch in bp["boxes"]:
            patch.set_facecolor(box_color)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)
            patch.set_alpha(0.85)


def plot_fig2b_testard_boxplots(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 8),
    split_by_bank: bool = True,
) -> plt.Figure:
    """Testard et al. 2024 Fig 3i style boxplot.

    Two vertically stacked panels with connected axes (no gap):
    - Top: Full contribution (R² when other groups shuffled, clipped >= 0)
    - Bottom: Unique contribution (delta R², y-axis inverted so 0 at top)

    Boxplots show distribution of per-session medians.
    Ventral = solid green + black border.
    Dorsal = green + white diagonal hatch + black border.
    Both panels share x-axis labels and the same y-axis scale.

    Args:
        results_df: DataFrame with columns:
            - full_contrib_vis, full_contrib_body_kin, full_contrib_head_kin
            - unique_vis_shuffle, unique_body_kin_shuffle, unique_head_kin_shuffle
            - session_id, unit_bank (optional)
        output_path: Path to save figure.
        figsize: Figure size in inches.
        split_by_bank: Whether to split by electrode bank (ventral vs dorsal).

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()
    # Discover feature groups dynamically from column names
    group_keys = _discover_groups(results_df)
    if not group_keys:
        # Fallback for legacy dropout-only results
        full_cols = ["vis_only_r2", "body_kin_only_r2", "head_kin_only_r2"]
        unique_cols = ["unique_vis", "unique_body_kin", "unique_head_kin"]
        groups = ["VIS", "BODY_KIN", "HEAD_KIN"]
    else:
        full_cols = [f"full_contrib_{g}" for g in group_keys]
        unique_cols = [f"unique_{g}_shuffle" for g in group_keys]
        groups = [_GROUP_DISPLAY.get(g, g.upper()) for g in group_keys]

    # Validate columns exist
    full_cols = [c for c in full_cols if c in results_df.columns]
    unique_cols = [c for c in unique_cols if c in results_df.columns]
    if len(full_cols) < 2 or len(unique_cols) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Testard boxplot requires shuffle method results",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        return fig

    # Widen figure for 4+ groups
    if len(groups) > 3 and figsize == (5, 8):
        figsize = (6, 8)

    has_bank = "unit_bank" in results_df.columns and split_by_bank
    x_base = np.arange(len(groups))
    green_light = "#2ca02c"
    green_dark = "#1a7a1a"

    # Connected two-panel layout: hspace=0 so panels touch
    fig, (ax_full, ax_unique) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0},
    )

    # Top: Full Contribution (exclude <= 0)
    _draw_boxes(
        ax_full, results_df, full_cols, x_base, has_bank,
        exclude_nonpositive=True, box_color=green_light,
    )
    ax_full.set_ylabel("Full contribution", fontsize=12, labelpad=12)
    ax_full.axhline(0, color="black", linestyle=":", linewidth=1.5)
    ax_full.spines["bottom"].set_visible(False)
    ax_full.tick_params(bottom=False)

    # Bottom: Unique Contribution — inverted so 0 is at top
    _draw_boxes(
        ax_unique, results_df, unique_cols, x_base, has_bank,
        exclude_nonpositive=True, box_color=green_dark,
    )
    ax_unique.set_ylabel("Unique contribution", fontsize=12, labelpad=12)
    ax_unique.axhline(0, color="black", linestyle=":", linewidth=1.5)
    ax_unique.invert_yaxis()  # 0 at top, values grow downward
    ax_unique.spines["top"].set_visible(False)

    # X-axis labels only on bottom panel
    ax_unique.set_xticks(x_base)
    ax_unique.set_xticklabels(groups, fontweight="bold")

    # Dynamic y-axis: 95th percentile of plotted data, rounded up to nearest 0.05
    all_plotted = []
    for cols_list in [full_cols, unique_cols]:
        vals = _get_values(results_df, cols_list, exclude_nonpositive=True)
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
    ax_unique.set_ylim(shared_max, 0)  # inverted: 0 at top

    # Sparse y-ticks: at most 3 on each panel, same values
    ax_full.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax_unique.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

    # Format 0 as "0" not "0.00"
    def _fmt_tick(val: float, pos: int) -> str:
        return "0" if val == 0 else f"{val:g}"

    ax_full.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_tick))
    ax_unique.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_tick))

    # Shared y-axis label spanning both panels
    fig.text(
        0.01, 0.5, "Explained neural variance (R²)",
        va="center", ha="center", rotation="vertical",
        fontsize=13, fontweight="bold",
    )

    # Legend on top panel
    if has_bank:
        ventral_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=green_light, ec="black", linewidth=1.0,
            alpha=0.85,
        )
        dorsal_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=green_light, ec="black", linewidth=1.0,
            alpha=0.85, hatch="//",
        )
        dorsal_patch._hatch_color = mcolors.to_rgba("white")
        ax_full.legend(
            [ventral_patch, dorsal_patch],
            ["Ventral", "Dorsal"],
            loc="upper right", frameon=False, fontsize=12,
        )

    fig.subplots_adjust(left=0.22)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def plot_fig2c_unique_scatter(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5),
    split_by_bank: bool = True,
    x_group: str = "vis",
    y_group: str = "body_kin",
) -> plt.Figure:
    """Fig 2C: Pairwise unique variance scatter with marginal histograms.

    Scatter of per-unit unique variance contributions with thin attached
    marginal histograms (Option A style). Units with both values <= 0 excluded.

    Args:
        results_df: DataFrame with unique_{group}_shuffle columns.
        output_path: Path to save figure.
        figsize: Figure size in inches.
        split_by_bank: Whether to color by electrode bank (ventral vs dorsal).
        x_group: Group name for x-axis (default: "vis").
        y_group: Group name for y-axis (default: "body_kin").

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    # Resolve column names
    x_col = f"unique_{x_group}_shuffle"
    y_col = f"unique_{y_group}_shuffle"
    # Fallback to dropout columns
    if x_col not in results_df.columns:
        x_col = f"unique_{x_group}"
    if y_col not in results_df.columns:
        y_col = f"unique_{y_group}"

    if x_col not in results_df.columns or y_col not in results_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Missing columns: {x_col}, {y_col}",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        return fig

    x_label = _GROUP_DISPLAY.get(x_group, x_group.upper())
    y_label = _GROUP_DISPLAY.get(y_group, y_group.upper())

    # Filter to units with at least one positive contribution
    mask = (results_df[x_col] > 0) | (results_df[y_col] > 0)
    df = results_df[mask].copy()

    has_bank = "unit_bank" in df.columns and split_by_bank

    # Gridspec: scatter + thin margins (ratio 4:1)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.02, wspace=0.02,
    )
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # Colors
    green = "#2ca02c"
    green_dark = "#1a7a1a"
    gray = "#888888"

    # --- Scatter ---
    if has_bank:
        for bank, color, marker, label in [
            ("ventral", green, "o", "Ventral"),
            ("dorsal", green_dark, "^", "Dorsal"),
        ]:
            bdf = df[df["unit_bank"] == bank]
            if len(bdf) == 0:
                continue
            ax_scatter.scatter(
                bdf[x_col], bdf[y_col],
                c=color, marker=marker, s=12, alpha=0.4,
                edgecolors="none", label=label, rasterized=True,
            )
        ax_scatter.legend(
            loc="upper right", frameon=False, fontsize=10,
            markerscale=1.5,
        )
    else:
        ax_scatter.scatter(
            df[x_col], df[y_col],
            c=green, s=12, alpha=0.4,
            edgecolors="none", rasterized=True,
        )

    # Unity and zero lines
    lim_max = max(df[x_col].quantile(0.99), df[y_col].quantile(0.99)) * 1.1
    ax_scatter.plot([0, lim_max], [0, lim_max], ":", color=gray, linewidth=1, zorder=0)
    ax_scatter.axhline(0, color="black", linewidth=0.5, zorder=0)
    ax_scatter.axvline(0, color="black", linewidth=0.5, zorder=0)

    ax_scatter.set_xlabel(f"Unique {x_label} (R\u00b2)", fontsize=12, fontweight="bold")
    ax_scatter.set_ylabel(f"Unique {y_label} (R\u00b2)", fontsize=12, fontweight="bold")

    # --- Marginal histograms ---
    hist_kw = dict(bins=40, color=gray, alpha=0.3, edgecolor="none")

    if has_bank:
        for bank, color in [("ventral", green), ("dorsal", green_dark)]:
            bdf = df[df["unit_bank"] == bank]
            if len(bdf) == 0:
                continue
            ax_hist_x.hist(bdf[x_col], **{**hist_kw, "color": color})
            ax_hist_y.hist(bdf[y_col], orientation="horizontal",
                           **{**hist_kw, "color": color})
    else:
        ax_hist_x.hist(df[x_col], **hist_kw)
        ax_hist_y.hist(df[y_col], orientation="horizontal", **hist_kw)

    # Clean up margin axes
    ax_hist_x.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_hist_y.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    for spine in ax_hist_x.spines.values():
        spine.set_visible(False)
    for spine in ax_hist_y.spines.values():
        spine.set_visible(False)

    # Hide corner
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    # Correlation annotation with p-value
    from scipy.stats import spearmanr
    r, p = spearmanr(df[x_col], df[y_col])
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax_scatter.text(
        0.05, 0.95, f"\u03c1 = {r:.2f}, {p_str}",
        transform=ax_scatter.transAxes, fontsize=11,
        va="top", ha="left",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def compute_testard_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Testard-style per-session summary statistics.

    Creates a long-form DataFrame with per-session medians for each group
    and metric, suitable for Testard-style reporting. Discovers groups
    dynamically from column names.

    Args:
        results_df: DataFrame with unique variance results.

    Returns:
        DataFrame with columns: session_id, unit_bank, bin_s, group, metric,
        method, value, n_units, frac_negative
    """
    rows = []

    # Discover groups dynamically
    group_keys = _discover_groups(results_df)
    has_shuffle = len(group_keys) > 0
    method = "shuffle" if has_shuffle else "dropout"

    # Build column mappings dynamically
    if has_shuffle:
        full_cols = {
            _GROUP_DISPLAY.get(g, g.upper()): f"full_contrib_{g}"
            for g in group_keys
        }
        unique_cols = {
            _GROUP_DISPLAY.get(g, g.upper()): f"unique_{g}_shuffle"
            for g in group_keys
        }
    else:
        # Legacy dropout fallback
        full_cols = {
            "VIS": "vis_only_r2",
            "BODY_KIN": "body_kin_only_r2",
            "HEAD_KIN": "head_kin_only_r2",
        }
        unique_cols = {
            "VIS": "unique_vis",
            "BODY_KIN": "unique_body_kin",
            "HEAD_KIN": "unique_head_kin",
        }

    # Get bin_s if available
    bin_s = results_df["bin_s"].iloc[0] if "bin_s" in results_df.columns else 0.5

    # Compute per-session, per-bank statistics
    for session_id in results_df["session_id"].unique():
        session_df = results_df[results_df["session_id"] == session_id]

        # Handle bank split
        if "unit_bank" in results_df.columns:
            banks = session_df["unit_bank"].dropna().unique()
        else:
            banks = [None]

        for bank in banks:
            if bank is not None:
                bank_df = session_df[session_df["unit_bank"] == bank]
            else:
                bank_df = session_df

            n_units = len(bank_df)

            for group_label in full_cols:
                # Full contribution
                if full_cols[group_label] in bank_df.columns:
                    col = full_cols[group_label]
                    values = bank_df[col].dropna()
                    if len(values) > 0:
                        rows.append({
                            "session_id": session_id,
                            "unit_bank": bank,
                            "bin_s": bin_s,
                            "group": group_label,
                            "metric": "full",
                            "method": method,
                            "value": float(values.median()),
                            "n_units": n_units,
                            "frac_negative": float((values < 0).mean()),
                        })

                # Unique contribution
                if unique_cols.get(group_label) in bank_df.columns:
                    col = unique_cols[group_label]
                    values = bank_df[col].dropna()
                    if len(values) > 0:
                        rows.append({
                            "session_id": session_id,
                            "unit_bank": bank,
                            "bin_s": bin_s,
                            "group": group_label,
                            "metric": "unique",
                            "method": method,
                            "value": float(values.median()),
                            "n_units": n_units,
                            "frac_negative": float((values < 0).mean()),
                        })

    return pd.DataFrame(rows)
