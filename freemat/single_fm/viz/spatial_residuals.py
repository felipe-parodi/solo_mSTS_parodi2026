"""Spatial firing rate maps for encoding model visualization.

Per-unit rate maps (firing rate binned by 2D position) and 3D COM trajectory
plots colored by firing rate, to visualize spatial tuning of individual neurons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _bin_rate_map(
    com_x: np.ndarray,
    com_y: np.ndarray,
    spike_counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_occupancy: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D spatial firing rate map for one unit.

    Args:
        com_x: X positions per time bin.
        com_y: Y positions per time bin.
        spike_counts: Firing rate per time bin.
        x_edges: Bin edges for X.
        y_edges: Bin edges for Y.
        min_occupancy: Minimum bin visits to include.

    Returns:
        Tuple of (rate_map, occupancy) as masked arrays.
    """
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    x_idx = np.clip(np.digitize(com_x, x_edges) - 1, 0, n_x - 1)
    y_idx = np.clip(np.digitize(com_y, y_edges) - 1, 0, n_y - 1)

    occupancy = np.zeros((n_y, n_x), dtype=int)
    sum_fr = np.zeros((n_y, n_x))

    valid = np.isfinite(spike_counts) & np.isfinite(com_x) & np.isfinite(com_y)
    for xi, yi, fr in zip(x_idx[valid], y_idx[valid], spike_counts[valid]):
        occupancy[yi, xi] += 1
        sum_fr[yi, xi] += fr

    mask = occupancy < min_occupancy
    rate_map = np.ma.masked_where(mask, sum_fr / np.maximum(occupancy, 1))
    occ_masked = np.ma.masked_where(occupancy == 0, occupancy)
    return rate_map, occ_masked


def plot_spatial_firing_rate_maps(
    residuals_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    n_units: int = 9,
    rank_col: str = "unique_pos_shuffle",
    mode: str = "top",
    n_bins: int = 30,
    min_occupancy: int = 5,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    session_label: str = "",
) -> plt.Figure:
    """Plot per-unit spatial firing rate maps for top/bottom POS-encoding units.

    Creates a grid: first panel is occupancy, remaining panels show
    individual unit rate maps ranked by a metric. All unit panels share
    a common color scale (vmin=0, vmax=global max) so spatial tuning
    differences are visually comparable across units and between top/bottom.

    Args:
        residuals_df: DataFrame with columns: unit_id, com_x, com_y, y_true.
            Must be from a SINGLE session.
        summary_df: DataFrame with per-unit summary stats (unit_id, rank_col,
            full_r2, unit_bank).
        n_units: Number of units to display.
        rank_col: Column in summary_df to rank units by.
        mode: "top" for highest-ranked, "bottom" for lowest-ranked.
        n_bins: Number of spatial bins per dimension.
        min_occupancy: Minimum bin visits.
        output_path: Path to save figure (PNG + SVG).
        figsize: Figure size. Auto-computed if None.
        session_label: Session name for title.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    # Rank units — sort descending so head = highest values
    ranked = summary_df.dropna(subset=[rank_col]).sort_values(
        rank_col, ascending=False,
    )
    if mode == "top":
        selected_units = ranked.head(n_units)
    else:
        selected_units = ranked.tail(n_units)

    unit_ids = selected_units["unit_id"].values

    # Compute shared bin edges from ALL data in this session
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

    # Pre-compute ALL rate maps to find shared color scale
    rate_maps = {}
    global_max = 0.0
    for uid in unit_ids:
        unit_data = residuals_df[residuals_df["unit_id"] == uid]
        rm, _ = _bin_rate_map(
            unit_data["com_x"].values,
            unit_data["com_y"].values,
            unit_data["y_true"].values,
            x_edges, y_edges,
            min_occupancy=min_occupancy,
        )
        rate_maps[uid] = rm
        if rm.count() > 0:
            global_max = max(global_max, float(rm.max()))

    # Layout: occupancy + n_units panels
    n_total = 1 + len(unit_ids)
    n_cols = min(6, n_total)
    n_rows = int(np.ceil(n_total / n_cols))
    if figsize is None:
        figsize = (3.2 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    # Panel 0: Occupancy
    all_x = residuals_df.drop_duplicates(subset=["com_x", "com_y", "t_bin_s"])
    occ_x = all_x["com_x"].values
    occ_y = all_x["com_y"].values

    x_idx = np.clip(np.digitize(occ_x, x_edges) - 1, 0, n_bins - 1)
    y_idx = np.clip(np.digitize(occ_y, y_edges) - 1, 0, n_bins - 1)
    occupancy = np.zeros((n_bins, n_bins), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        occupancy[yi, xi] += 1

    if "t_bin_s" in all_x.columns and len(all_x) > 1:
        dt = np.median(np.diff(np.sort(all_x["t_bin_s"].unique())))
    else:
        dt = 0.5
    occ_seconds = occupancy * dt
    occ_display = np.ma.masked_where(occupancy == 0, occ_seconds)

    ax = flat_axes[0]
    im = ax.pcolormesh(
        x_edges, y_edges, occ_display,
        cmap="gray_r", rasterized=True,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Occupancy (s)", fontsize=11, fontweight="bold", color="red")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Per-unit panels — shared color scale
    for i, uid in enumerate(unit_ids):
        ax = flat_axes[i + 1]
        rate_map = rate_maps[uid]

        im = ax.pcolormesh(
            x_edges, y_edges, rate_map,
            cmap="hot", rasterized=True,
            vmin=0, vmax=global_max,
        )
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("FR", fontsize=8)

        row = selected_units[selected_units["unit_id"] == uid].iloc[0]
        bank = row.get("unit_bank", "?")
        pos_val = row.get(rank_col, 0)
        r2_val = row.get("full_r2", 0)
        ax.set_title(
            f"Unit {uid} ({bank})\nPOS={pos_val:.3f}, R$^2$={r2_val:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # Hide unused axes
    for j in range(n_total, len(flat_axes)):
        flat_axes[j].set_visible(False)

    mode_label = "Top" if mode == "top" else "Bottom"
    suptitle = f"Spatial Firing Rate Maps: {mode_label} {len(unit_ids)} POS-encoding Units"
    if session_label:
        suptitle += f" ({session_label})"
    fig.suptitle(suptitle, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


def plot_spatial_firing_rate_3d(
    residuals_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    n_units: int = 6,
    rank_col: str = "unique_pos_shuffle",
    mode: str = "top",
    subsample: int = 2000,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    session_label: str = "",
) -> plt.Figure:
    """Plot 3D COM trajectory colored by firing rate per unit.

    Each subplot shows the animal's 3D center-of-mass trajectory
    (com_x, com_y, com_z) as a scatter plot, with point color
    indicating firing rate. This reveals whether a neuron fires
    preferentially at specific 3D locations.

    Args:
        residuals_df: DataFrame with columns: unit_id, com_x, com_y,
            com_z, y_true. Must be from a SINGLE session.
        summary_df: DataFrame with per-unit summary stats.
        n_units: Number of units to display.
        rank_col: Column in summary_df to rank units by.
        mode: "top" for highest-ranked, "bottom" for lowest-ranked.
        subsample: Max points per unit to avoid overplotting.
        output_path: Path to save figure (PNG + SVG).
        figsize: Figure size.
        session_label: Session name for title.

    Returns:
        Matplotlib Figure object.
    """
    _setup_nature_style()

    # Rank units — sort descending so head = highest values
    ranked = summary_df.dropna(subset=[rank_col]).sort_values(
        rank_col, ascending=False,
    )
    if mode == "top":
        selected_units = ranked.head(n_units)
    else:
        selected_units = ranked.tail(n_units)

    unit_ids = selected_units["unit_id"].values

    n_cols = min(3, len(unit_ids))
    n_rows = int(np.ceil(len(unit_ids) / n_cols))
    if figsize is None:
        figsize = (5.5 * n_cols, 5.0 * n_rows)

    fig = plt.figure(figsize=figsize)

    for i, uid in enumerate(unit_ids):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        unit_data = residuals_df[residuals_df["unit_id"] == uid].copy()

        # Subsample if too many points
        if len(unit_data) > subsample:
            unit_data = unit_data.sample(n=subsample, random_state=42)

        cx = unit_data["com_x"].values
        cy = unit_data["com_y"].values
        cz = unit_data["com_z"].values if "com_z" in unit_data.columns else np.zeros_like(cx)
        fr = unit_data["y_true"].values

        sc = ax.scatter(
            cx, cy, cz,
            c=fr, cmap="hot", s=4, alpha=0.6,
            edgecolors="none", rasterized=True,
        )
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="FR")

        row = selected_units[selected_units["unit_id"] == uid].iloc[0]
        bank = row.get("unit_bank", "?")
        pos_val = row.get(rank_col, 0)
        r2_val = row.get("full_r2", 0)
        ax.set_title(
            f"Unit {uid} ({bank})\nPOS={pos_val:.3f}, R$^2$={r2_val:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    mode_label = "Top" if mode == "top" else "Bottom"
    suptitle = f"3D COM Trajectory by Firing Rate: {mode_label} {len(unit_ids)} POS Units"
    if session_label:
        suptitle += f" ({session_label})"
    fig.suptitle(suptitle, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        svg_path = str(output_path).replace(".png", ".svg")
        fig.savefig(svg_path, bbox_inches="tight")

    return fig


__all__ = [
    "plot_spatial_firing_rate_maps",
    "plot_spatial_firing_rate_3d",
]
