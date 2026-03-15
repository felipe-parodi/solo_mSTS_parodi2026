"""Visualization functions for Figure 4: Behavioral Motif Analysis.

Panel B: Motif tuning heatmap (units x motifs, z-scored mean FR).
Panel C: Motif decoding accuracy with null distribution.
Panel D: Lag sweep (accuracy vs time relative to motif transition).
Supplement: Descriptives, next-syllable prediction, confusion matrix,
    selectivity violin, transition matrix, onset-aligned PETH.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Bank colors — harmonized with Figs 2/3 primary palette
BANK_COLORS = {"ventral": "#457B9D", "dorsal": "#E63946", "pooled": "#555555"}
BANK_LABELS = {"ventral": "Ventral", "dorsal": "Dorsal", "pooled": "Pooled"}


def _setup_nature_style() -> None:
    """Apply Nature-style matplotlib settings matching Figs 2/3."""
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


def plot_fig4b_tuning_heatmap(
    tuning_df: pd.DataFrame,
    motif_order: List[int],
    figsize: tuple = (8, 10),
) -> plt.Figure:
    """Fig 4B: Motif tuning heatmap ordered by electrode depth.

    Args:
        tuning_df: DataFrame from run_fig4_session with motif_{id}_z columns,
            bank, sort_rank columns.
        motif_order: Motif IDs in hierarchical cluster order.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    # Extract z-scored tuning columns in cluster order
    z_cols = [f"motif_{mid}_z" for mid in motif_order]
    missing = [c for c in z_cols if c not in tuning_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Sort units by sort_rank (bank -> probe -> shank -> depth)
    df_sorted = tuning_df.sort_values("sort_rank").reset_index(drop=True)
    Z = df_sorted[z_cols].values

    # Find ventral/dorsal boundary
    banks = df_sorted["bank"].values
    n_ventral = (banks == "ventral").sum()

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1, 0.03, 0.15], figure=fig)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    vmax = 2.0
    im = ax.imshow(
        Z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )

    # Bank separator line
    if 0 < n_ventral < len(banks):
        ax.axhline(n_ventral - 0.5, color="black", linewidth=1.5, linestyle="--")

    # Axis labels
    ax.set_xticks(range(len(motif_order)))
    ax.set_xticklabels([str(m) for m in motif_order], fontsize=11)
    ax.set_xlabel("Motif ID (clustered)")
    ax.set_ylabel("Units (sorted by depth)")

    # Colorbar in dedicated axis
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-scored FR", fontsize=12)

    # Bank labels in the spacer column (gs[2])
    ax_bank = fig.add_subplot(gs[2])
    ax_bank.set_xlim(0, 1)
    ax_bank.set_ylim(ax.get_ylim())
    ax_bank.axis("off")
    y_ventral = n_ventral / 2
    y_dorsal = n_ventral + (len(banks) - n_ventral) / 2
    ax_bank.text(0.3, y_ventral, "Ventral", ha="left", va="center",
                 fontsize=11, fontweight="bold")
    ax_bank.text(0.3, y_dorsal, "Dorsal", ha="left", va="center",
                 fontsize=11, fontweight="bold")

    ax.set_title("Motif Tuning by Electrode Depth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _build_electrode_grid(
    tuning_df: pd.DataFrame,
    motif_order: List[int],
    metric: str = "selectivity",
) -> Dict[str, np.ndarray]:
    """Build 8x4 grids (depth x shank) per probe with selectivity or preferred motif.

    Args:
        tuning_df: DataFrame with probe, shank, depth_position, motif_{id}_z columns.
        motif_order: Motif IDs in cluster order.
        metric: 'selectivity' (max|z| across motifs) or 'preferred' (argmax motif).

    Returns:
        Dict mapping probe name (A1/A2/B1/B2) to 8x4 ndarray. NaN = no data.
    """
    z_cols = [f"motif_{mid}_z" for mid in motif_order]
    # Filter to columns that actually exist (handles cross-session pooling)
    z_cols = [c for c in z_cols if c in tuning_df.columns]
    probes = ["A1", "A2", "B1", "B2"]
    if not z_cols:
        return {p: np.full((8, 4), np.nan) for p in probes}
    grids: Dict[str, np.ndarray] = {}

    for probe in probes:
        grid = np.full((8, 4), np.nan)
        pdf = tuning_df[tuning_df["probe"] == probe]
        if pdf.empty:
            grids[probe] = grid
            continue

        for (shank, depth), group in pdf.groupby(["shank", "depth_position"]):
            z_vals = group[z_cols].values  # (n_units, n_motifs)
            col = int(shank) - 1  # shank 1-4 -> col 0-3
            row = int(depth)      # depth_position 0-7

            # Skip if all z-values are NaN (no valid data at this site)
            if np.all(np.isnan(z_vals)):
                continue

            if metric == "selectivity":
                # max|z| per unit (NaN-safe), then mean across units
                unit_sel = np.nanmax(np.abs(z_vals), axis=1)
                grid[row, col] = float(np.nanmean(unit_sel))
            else:
                # Mean z per motif across units (NaN-safe), then argmax
                mean_z = np.nanmean(z_vals, axis=0)
                if np.all(np.isnan(mean_z)):
                    continue
                grid[row, col] = float(motif_order[int(np.nanargmax(mean_z))])

        grids[probe] = grid

    return grids


def plot_fig4b_electrode_selectivity(
    tuning_df: pd.DataFrame,
    motif_order: List[int],
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Fig 4B alt: Motif selectivity in physical electrode layout.

    2x2 grid: [A1 (Ventral 1), B1 (Dorsal 1); A2 (Ventral 2), B2 (Dorsal 2)].
    Each panel: 8 depths x 4 shanks. Color = mean selectivity (max|z| across motifs).

    Args:
        tuning_df: DataFrame with probe, shank, depth_position, motif_{id}_z columns.
        motif_order: Motif IDs in cluster order.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    grids = _build_electrode_grid(tuning_df, motif_order, metric="selectivity")

    probe_layout = [["A1", "B1"], ["A2", "B2"]]
    probe_titles = {
        "A1": "Ventral Bank 1 (A1)", "A2": "Ventral Bank 2 (A2)",
        "B1": "Dorsal Bank 1 (B1)", "B2": "Dorsal Bank 2 (B2)",
    }

    # Global vmax across all probes
    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    finite = all_vals[np.isfinite(all_vals)]
    vmax = float(np.max(finite)) if len(finite) > 0 else 2.0

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for row_idx, row_probes in enumerate(probe_layout):
        for col_idx, probe in enumerate(row_probes):
            ax = axes[row_idx, col_idx]
            grid = grids[probe]

            # Masked array for NaN (gray background)
            masked = np.ma.masked_invalid(grid)
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color="#DDDDDD")

            im = ax.imshow(
                masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                interpolation="nearest",
            )

            # Annotate cells
            for r in range(8):
                for c in range(4):
                    val = grid[r, c]
                    if np.isfinite(val):
                        color = "white" if val > vmax * 0.6 else "black"
                        ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                                fontsize=8, color=color, fontweight="bold")

            ax.set_xticks(range(4))
            ax.set_xticklabels([f"S{i+1}" for i in range(4)])
            ax.set_yticks(range(8))
            ax.set_yticklabels([f"D{i}" for i in range(8)])
            ax.set_xlabel("Shank")
            ax.set_ylabel("Depth position")
            ax.set_title(probe_titles[probe], fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Selectivity (max |z|)",
                 shrink=0.8, pad=0.08)
    fig.suptitle("Motif Selectivity by Electrode Position", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.92, right=0.82, hspace=0.3, wspace=0.3)
    return fig


def plot_fig4b_electrode_preferred(
    tuning_df: pd.DataFrame,
    motif_order: List[int],
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Fig 4B alt: Preferred motif in physical electrode layout.

    2x2 grid: [A1 (Ventral 1), B1 (Dorsal 1); A2 (Ventral 2), B2 (Dorsal 2)].
    Each panel: 8 depths x 4 shanks. Color = preferred motif ID (categorical).

    Args:
        tuning_df: DataFrame with probe, shank, depth_position, motif_{id}_z columns.
        motif_order: Motif IDs in cluster order.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    grids = _build_electrode_grid(tuning_df, motif_order, metric="preferred")

    probe_layout = [["A1", "B1"], ["A2", "B2"]]
    probe_titles = {
        "A1": "Ventral Bank 1 (A1)", "A2": "Ventral Bank 2 (A2)",
        "B1": "Dorsal Bank 1 (B1)", "B2": "Dorsal Bank 2 (B2)",
    }

    # Build motif -> index mapping for categorical colormap
    motif_to_idx = {mid: i for i, mid in enumerate(motif_order)}
    n_motifs = len(motif_order)
    cmap_cat = plt.cm.tab10

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for row_idx, row_probes in enumerate(probe_layout):
        for col_idx, probe in enumerate(row_probes):
            ax = axes[row_idx, col_idx]
            grid = grids[probe]

            # Convert motif IDs to categorical indices for coloring
            grid_idx = np.full_like(grid, np.nan)
            for r in range(8):
                for c in range(4):
                    val = grid[r, c]
                    if np.isfinite(val):
                        mid = int(val)
                        grid_idx[r, c] = motif_to_idx.get(mid, np.nan)

            masked = np.ma.masked_invalid(grid_idx)
            cmap = plt.cm.tab10.copy()
            cmap.set_bad(color="#DDDDDD")

            ax.imshow(
                masked, aspect="auto", cmap=cmap,
                vmin=-0.5, vmax=n_motifs - 0.5,
                interpolation="nearest",
            )

            # Annotate cells with motif ID
            for r in range(8):
                for c in range(4):
                    val = grid[r, c]
                    if np.isfinite(val):
                        ax.text(c, r, f"{int(val)}", ha="center", va="center",
                                fontsize=9, color="black", fontweight="bold")

            ax.set_xticks(range(4))
            ax.set_xticklabels([f"S{i+1}" for i in range(4)])
            ax.set_yticks(range(8))
            ax.set_yticklabels([f"D{i}" for i in range(8)])
            ax.set_xlabel("Shank")
            ax.set_ylabel("Depth position")
            ax.set_title(probe_titles[probe], fontsize=12, fontweight="bold")

    # Categorical legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cmap_cat(i / n_motifs), label=f"Motif {mid}")
               for i, mid in enumerate(motif_order)]
    fig.legend(handles=handles, loc="center right", fontsize=9,
               title="Preferred Motif", title_fontsize=10,
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Preferred Motif by Electrode Position", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.92, right=0.82, hspace=0.3, wspace=0.3)
    return fig


# ── Voloh-style bubble plot helpers ──────────────────────────────────────

_PROBE_OFFSET: Dict[str, int] = {"A1": 0, "A2": 4, "B1": 10, "B2": 14}

_METRIC_CONFIG: Dict[str, Dict] = {
    "selectivity": {"cmap": "viridis", "label": "Selectivity (max |z|)", "center": None},
    "bandwidth": {"cmap": "cividis", "label": "Bandwidth (#motifs |z|>1)", "center": None},
    "peak_latency": {"cmap": "RdBu_r", "label": "Peak latency (s)", "center": 0.0},
    "response_modulation": {"cmap": "PiYG", "label": "Response modulation (Hz)", "center": 0.0},
    "transition_resp": {"cmap": "magma", "label": "Transition responsiveness", "center": None},
}


def _aggregate_electrode_metrics(
    tuning_df: pd.DataFrame,
    motif_order: list[int],
    peth: np.ndarray | None = None,
    tau_bins: np.ndarray | None = None,
    bin_s: float = 0.5,
) -> pd.DataFrame:
    """Aggregate per-unit metrics to electrode level.

    For each electrode site (probe, shank, depth_position), computes:
      - selectivity: mean of max|z| across motifs per unit
      - bandwidth: mean count of motifs with |z| > 1.0 per unit
      - peak_latency: mean time of peak firing relative to onset (requires PETH)
      - response_modulation: mean post-onset minus pre-onset FR (requires PETH)
      - n_units: number of units at this electrode

    Args:
        tuning_df: DataFrame with probe, shank, depth_position, bank,
            and motif_{id}_z columns.
        motif_order: Motif IDs to use for tuning metrics.
        peth: Optional (n_units, n_time_bins) onset-aligned firing rates.
        tau_bins: Optional relative time bin indices for PETH.
        bin_s: Bin duration in seconds for PETH timing.

    Returns:
        DataFrame with one row per electrode.
    """
    z_cols = [f"motif_{mid}_z" for mid in motif_order]
    # Filter to columns that actually exist (handles cross-session pooling)
    z_cols = [c for c in z_cols if c in tuning_df.columns]
    if not z_cols:
        return pd.DataFrame()

    # Per-unit metrics (NaN-safe for cross-session pooled data)
    z_mat = tuning_df[z_cols].values
    tuning_df = tuning_df.copy()
    tuning_df["_selectivity"] = np.nanmax(np.abs(z_mat), axis=1)
    tuning_df["_bandwidth"] = np.nansum(np.abs(z_mat) > 1.0, axis=1).astype(float)

    agg_cols = {"_selectivity": "mean", "_bandwidth": "mean"}

    if peth is not None and tau_bins is not None:
        t_sec = tau_bins * bin_s
        # Peak latency
        peak_idx = np.argmax(peth, axis=1)
        tuning_df["_peak_latency"] = t_sec[peak_idx]
        agg_cols["_peak_latency"] = "mean"
        # Response modulation: post [0, +1s] minus pre [-2s, -0.5s]
        post_mask = (t_sec >= 0) & (t_sec <= 1.0)
        pre_mask = (t_sec >= -2.0) & (t_sec < -0.5)
        if post_mask.any() and pre_mask.any():
            post_mean = peth[:, post_mask].mean(axis=1)
            pre_mean = peth[:, pre_mask].mean(axis=1)
            tuning_df["_response_modulation"] = post_mean - pre_mean
            agg_cols["_response_modulation"] = "mean"
        # Transition responsiveness: std(onset window) / std(baseline)
        onset_mask = (t_sec >= -0.5) & (t_sec <= 0.5)
        baseline_mask = (t_sec >= -3.0) & (t_sec < -1.5)
        if onset_mask.any() and baseline_mask.any():
            onset_std = peth[:, onset_mask].std(axis=1)
            baseline_std = peth[:, baseline_mask].std(axis=1)
            # Avoid division by zero
            safe_baseline = np.where(baseline_std > 1e-6, baseline_std, 1e-6)
            tuning_df["_transition_resp"] = onset_std / safe_baseline
            agg_cols["_transition_resp"] = "mean"

    group_keys = ["probe", "shank", "depth_position", "bank"]
    agg = tuning_df.groupby(group_keys).agg(
        n_units=("_selectivity", "size"),
        **{k.lstrip("_"): (k, v) for k, v in agg_cols.items()},
    ).reset_index()

    return agg


def _electrode_scatter_coords(
    electrode_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Map electrode positions to scatter plot coordinates.

    Layout: [A1 | A2 | gap | B1 | B2], each probe 4 shank columns.
    Y-axis = depth (0 at top, 7 at bottom).

    Args:
        electrode_df: DataFrame with probe, shank, depth_position columns.

    Returns:
        Tuple of (x_coords, y_coords) arrays.
    """
    x = electrode_df["probe"].map(_PROBE_OFFSET).values + (electrode_df["shank"].values - 1)
    y = electrode_df["depth_position"].values
    return x.astype(float), y.astype(float)


def plot_fig4b_bubble(
    tuning_df: pd.DataFrame,
    motif_order: list[int],
    peth: np.ndarray | None = None,
    tau_bins: np.ndarray | None = None,
    bin_s: float = 0.5,
    metrics: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    gradient_stats: Dict | None = None,
) -> plt.Figure:
    """Fig 4B: Voloh-style bubble plot of electrode-level metrics.

    Each circle represents one electrode site. Size encodes the number
    of units recorded there. Color encodes the metric value.

    Layout: [Ventral (A1 A2) | gap | Dorsal (B1 B2)]

    Args:
        tuning_df: DataFrame with probe, shank, depth_position, bank,
            motif_{id}_z columns.
        motif_order: Motif IDs for tuning computation.
        peth: Optional (n_units, n_time_bins) onset-aligned PETH.
        tau_bins: Optional relative bin indices for PETH.
        bin_s: Bin size in seconds (for PETH timing).
        metrics: Which metrics to plot. Default: selectivity + bandwidth
            (+ peak_latency + response_modulation if PETH provided).
        figsize: Override figure size.
        gradient_stats: Optional dict keyed by metric name, each value a
            dict with 'direction' and 'p_str' for annotation. When
            provided, selectivity/bandwidth use these stats. When None,
            selectivity/bandwidth skip annotation entirely. PETH metrics
            (transition_resp, peak_latency, response_modulation) always
            use the inline t-test regardless of this parameter.

    Returns:
        Matplotlib Figure with one subplot per metric.
    """
    _setup_nature_style()

    agg = _aggregate_electrode_metrics(tuning_df, motif_order, peth, tau_bins, bin_s)
    x, y = _electrode_scatter_coords(agg)

    # Default to selectivity + bandwidth (PETH metrics only if explicitly requested)
    if metrics is None:
        metrics = ["selectivity", "bandwidth"]
    metrics = [m for m in metrics if m in agg.columns]

    n_panels = len(metrics)
    if figsize is None:
        figsize = (6 * n_panels, 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    sizes = agg["n_units"].values * 80

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        cfg = _METRIC_CONFIG[metric]
        values = agg[metric].values

        # Color scale bounds
        if cfg["center"] is not None:
            vmax = np.nanmax(np.abs(values))
            vmin = -vmax
        else:
            vmin = 0
            vmax = np.nanmax(values)

        sc = ax.scatter(
            x, y, s=sizes, c=values,
            cmap=cfg["cmap"], vmin=vmin, vmax=vmax,
            edgecolors="black", linewidths=0.5, alpha=0.85, zorder=3,
        )

        # Bank separator
        ax.axvline(8.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # Probe labels
        for probe, offset in _PROBE_OFFSET.items():
            ax.text(offset + 1.5, -0.8, probe, ha="center", fontsize=9,
                    fontweight="bold")

        # Bank labels (above plot)
        ax.text(3.5, -1.5, "Ventral", ha="center", fontsize=11,
                fontweight="bold", color=BANK_COLORS["ventral"])
        ax.text(13.5, -1.5, "Dorsal", ha="center", fontsize=11,
                fontweight="bold", color=BANK_COLORS["dorsal"])

        # Axes
        ax.set_xlim(-1, 18)
        ax.set_ylim(7.5, -2.0)  # Inverted: D0 at top
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"D{i}" for i in range(8)])
        ax.set_xticks([])
        ax.set_title(cfg["label"], fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Depth position")

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        # Bank-level p-value annotation
        _PETH_METRICS = {"transition_resp", "peak_latency", "response_modulation"}
        _TUNING_METRICS = {"selectivity", "bandwidth"}
        annotation_text = None

        if metric in _TUNING_METRICS and gradient_stats is not None:
            # Use pre-computed gradient stats for tuning metrics
            if metric in gradient_stats:
                gs = gradient_stats[metric]
                annotation_text = f"{gs['direction']}, {gs['p_str']}"
        elif metric in _PETH_METRICS and "bank" in agg.columns:
            # Inline t-test for PETH metrics (gradient module doesn't cover these)
            from scipy.stats import ttest_ind
            ventral_vals = agg.loc[agg["bank"] == "ventral", metric].dropna()
            dorsal_vals = agg.loc[agg["bank"] == "dorsal", metric].dropna()
            if len(ventral_vals) > 1 and len(dorsal_vals) > 1:
                _, p_bank = ttest_ind(ventral_vals, dorsal_vals, equal_var=False)
                p_str = "p<0.001" if p_bank < 0.001 else f"p={p_bank:.3f}"
                v_mean = ventral_vals.mean()
                d_mean = dorsal_vals.mean()
                higher = "D>V" if d_mean > v_mean else "V>D"
                annotation_text = f"{higher}, {p_str}"

        if annotation_text is not None:
            ax.text(
                0.5, 0.97, annotation_text,
                ha="center", va="top", fontsize=10,
                fontstyle="italic", color="#555555",
                transform=ax.transAxes,
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 2,
                      "edgecolor": "none"},
            )

    # Size legend in first panel
    ax0 = axes[0]
    for n_val, lbl in [(1, "1"), (2, "2"), (3, "3+")]:
        ax0.scatter([], [], s=n_val * 80, c="gray", alpha=0.5,
                    edgecolors="black", linewidths=0.5,
                    label=f"n = {lbl}")
    ax0.legend(loc="lower left", fontsize=9, frameon=True, title="Units",
               title_fontsize=9, scatterpoints=1)

    fig.suptitle("Electrode-Level Metrics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# Physical probe geometry (µm) — from configs/banks.yaml
_INTER_SHANK_UM = 500
_INTER_CONTACT_UM = 550
# AP offset between anterior (A2/B2) and posterior (A1/B1) platforms.
# From NeuroNexus spec: platform_spacing_um in configs/banks.yaml.
_AP_OFFSET_UM = 800

# Probe AP positions: A2/B2 anterior, A1/B1 posterior
_PROBE_AP_UM: Dict[str, float] = {
    "A2": 0.0, "A1": _AP_OFFSET_UM,
    "B2": 0.0, "B1": _AP_OFFSET_UM,
}


def _electrode_3d_coords(
    electrode_df: pd.DataFrame,
    ap_offset_um: float = _AP_OFFSET_UM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map electrode positions to 3D physical coordinates (µm).

    X = medial-lateral (shank position within probe).
    Y = anterior-posterior (platform offset).
    Z = depth (superficial to deep, positive downward).

    Args:
        electrode_df: DataFrame with probe, shank, depth_position columns.
        ap_offset_um: AP distance between anterior and posterior platforms.

    Returns:
        Tuple of (x, y, z) coordinate arrays in µm.
    """
    probe_ap = {"A2": 0.0, "A1": ap_offset_um, "B2": 0.0, "B1": ap_offset_um}
    x = (electrode_df["shank"].values - 1) * _INTER_SHANK_UM
    y = electrode_df["probe"].map(probe_ap).values
    z = electrode_df["depth_position"].values * _INTER_CONTACT_UM
    return x.astype(float), y.astype(float), z.astype(float)


def plot_fig4b_bubble_3d(
    tuning_df: pd.DataFrame,
    motif_order: list[int],
    peth: np.ndarray | None = None,
    tau_bins: np.ndarray | None = None,
    bin_s: float = 0.5,
    metric: str = "selectivity",
    ap_offset_um: float = _AP_OFFSET_UM,
    figsize: tuple[float, float] = (14, 5),
    gradient_stats: Dict | None = None,
) -> plt.Figure:
    """Fig 4B: 3D bubble plot of electrode-level metric by bank.

    Two 3D subplots (Ventral, Dorsal) showing physical probe geometry.
    Bubble size = n_units, bubble color = metric.

    Args:
        tuning_df: DataFrame with probe, shank, depth_position, bank,
            motif_{id}_z columns.
        motif_order: Motif IDs for tuning computation.
        peth: Optional onset-aligned PETH array.
        tau_bins: Optional relative bin indices for PETH.
        bin_s: Bin size in seconds.
        metric: Which metric to color by (selectivity, bandwidth,
            peak_latency, response_modulation).
        ap_offset_um: AP distance between platforms in µm.
        figsize: Figure size.
        gradient_stats: Optional dict keyed by metric name for annotation.
            See plot_fig4b_bubble for dispatch logic.

    Returns:
        Matplotlib Figure with two 3D subplots.
    """
    _setup_nature_style()

    agg = _aggregate_electrode_metrics(tuning_df, motif_order, peth, tau_bins, bin_s)
    x, y, z = _electrode_3d_coords(agg, ap_offset_um)

    if metric not in agg.columns:
        available = [m for m in _METRIC_CONFIG if m in agg.columns]
        metric = available[0] if available else "selectivity"

    cfg = _METRIC_CONFIG[metric]
    values = agg[metric].values
    sizes = agg["n_units"].values * 60

    if cfg["center"] is not None:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.nanmax(values))

    fig = plt.figure(figsize=figsize)
    bank_configs = [
        ("Ventral", ["A1", "A2"], 121),
        ("Dorsal", ["B1", "B2"], 122),
    ]

    for bank_label, probes, subplot_pos in bank_configs:
        ax = fig.add_subplot(subplot_pos, projection="3d")
        mask = agg["probe"].isin(probes).values
        if not mask.any():
            ax.set_title(f"{bank_label} (no data)")
            continue

        sc = ax.scatter(
            x[mask], y[mask], z[mask],
            s=sizes[mask], c=values[mask],
            cmap=cfg["cmap"], vmin=vmin, vmax=vmax,
            edgecolors="black", linewidths=0.4, alpha=0.85, depthshade=False,
        )

        # Axis labels with units
        ax.set_xlabel("Shank (µm)", fontsize=10, labelpad=8)
        ax.set_ylabel("A-P (µm)", fontsize=10, labelpad=8)
        ax.set_zlabel("Depth (µm)", fontsize=10, labelpad=8)

        # Invert Z so deeper = lower visually
        ax.invert_zaxis()

        # Shank tick labels
        ax.set_xticks([i * _INTER_SHANK_UM for i in range(4)])
        ax.set_xticklabels([f"S{i+1}" for i in range(4)], fontsize=8)

        # AP tick labels
        ap_vals = sorted(set(y[mask]))
        ap_labels = ["Ant", "Post"] if len(ap_vals) == 2 else [f"{v:.0f}" for v in ap_vals]
        ax.set_yticks(ap_vals)
        ax.set_yticklabels(ap_labels, fontsize=8)

        # Depth tick labels
        ax.set_zticks([i * _INTER_CONTACT_UM for i in range(0, 8, 2)])
        ax.set_zticklabels([f"D{i}" for i in range(0, 8, 2)], fontsize=8)

        ax.set_title(f"{bank_label} Bank", fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=8)

        # Viewing angle
        ax.view_init(elev=25, azim=-50)

    # Manually positioned colorbar to avoid overlap with dorsal data
    fig.subplots_adjust(left=0.02, right=0.78, top=0.85, bottom=0.05, wspace=0.05)
    cax = fig.add_axes([0.82, 0.15, 0.02, 0.55])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(cfg["label"], fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Bank-level p-value annotation
    _PETH_METRICS = {"transition_resp", "peak_latency", "response_modulation"}
    _TUNING_METRICS = {"selectivity", "bandwidth"}
    annotation_text = None

    if metric in _TUNING_METRICS and gradient_stats is not None:
        if metric in gradient_stats:
            gs = gradient_stats[metric]
            annotation_text = f"{gs['direction']}, {gs['p_str']}"
    elif metric in _PETH_METRICS:
        ventral_vals = agg.loc[agg["bank"] == "ventral", metric].dropna()
        dorsal_vals = agg.loc[agg["bank"] == "dorsal", metric].dropna()
        if len(ventral_vals) > 1 and len(dorsal_vals) > 1:
            from scipy.stats import ttest_ind as _ttest
            _, p_bank = _ttest(ventral_vals, dorsal_vals, equal_var=False)
            higher = "D>V" if dorsal_vals.mean() > ventral_vals.mean() else "V>D"
            p_str = "p<0.001" if p_bank < 0.001 else f"p={p_bank:.3f}"
            annotation_text = f"{higher}, {p_str}"

    if annotation_text is not None:
        fig.text(0.50, 0.92, annotation_text, ha="center", va="bottom",
                 fontsize=11, fontstyle="italic",
                 bbox={"facecolor": "white", "alpha": 0.7, "pad": 3,
                       "edgecolor": "none"})

    # Size legend
    for n_val in [1, 3, 5]:
        fig.axes[0].scatter([], [], [], s=n_val * 60, c="gray", alpha=0.5,
                            edgecolors="black", linewidths=0.4,
                            label=f"n = {n_val}")
    fig.axes[0].legend(loc="upper left", fontsize=8, frameon=True,
                       title="Units", title_fontsize=8, scatterpoints=1)

    fig.suptitle(f"3D Electrode Metrics: {cfg['label']}", fontsize=14,
                 fontweight="bold", y=0.98)
    return fig


def plot_fig4c_decoding_accuracy(
    dec_summary: pd.DataFrame,
    nulls: Dict[str, np.ndarray],
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Fig 4C: Decoding accuracy bars with null distribution.

    Args:
        dec_summary: DataFrame with bank, balanced_accuracy columns.
        nulls: Dict mapping bank -> (n_perm,) null accuracy array.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    bank_order = ["ventral", "dorsal", "pooled"]
    banks_present = [b for b in bank_order if b in dec_summary["bank"].values]

    x_pos = np.arange(len(banks_present))
    bar_width = 0.5

    for i, bank in enumerate(banks_present):
        row = dec_summary[dec_summary["bank"] == bank].iloc[0]
        acc = row["balanced_accuracy"]
        color = BANK_COLORS.get(bank, "#999999")

        # Null distribution violin
        if bank in nulls and len(nulls[bank]) > 0:
            null_vals = nulls[bank]
            parts = ax.violinplot(
                null_vals, positions=[x_pos[i]], widths=bar_width * 0.8,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#CCCCCC")
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("#888888")

            # Significance
            p_val = (1 + np.sum(null_vals >= acc)) / (1 + len(null_vals))
            if p_val < 0.001:
                sig_str = "***"
            elif p_val < 0.01:
                sig_str = "**"
            elif p_val < 0.05:
                sig_str = "*"
            else:
                sig_str = "n.s."
            ax.text(x_pos[i], acc + 0.015, sig_str, ha="center", va="bottom",
                    fontsize=9)

            # Pilot indicator: show n_perm when small
            n_perm = len(null_vals)
            if n_perm <= 100:
                ax.text(x_pos[i], -0.02, f"n={n_perm}", ha="center",
                        fontsize=8, color="gray", fontstyle="italic")

        # CV fold error bars (SEM)
        if "fold_accuracies" in dec_summary.columns:
            import json
            fold_accs = json.loads(row["fold_accuracies"])
            if len(fold_accs) > 1:
                sem = np.std(fold_accs) / np.sqrt(len(fold_accs))
                ax.errorbar(x_pos[i], acc, yerr=sem, color="black",
                            capsize=4, capthick=1.5, lw=1.5, zorder=4, fmt="none")

        # Observed accuracy bar
        ax.bar(x_pos[i], acc, width=bar_width, color=color, edgecolor="white",
               alpha=0.9, zorder=3)

    # Chance line (1/n_classes)
    n_classes = dec_summary["n_classes"].iloc[0] if "n_classes" in dec_summary.columns else 10
    chance = 1.0 / n_classes
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=f"Chance (1/{n_classes})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([BANK_LABELS.get(b, b) for b in banks_present])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Motif Decoding from Neural Activity", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def plot_fig4d_lag_sweep(
    lag_df: pd.DataFrame,
    n_classes: int = 10,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Fig 4D: Lag sweep — accuracy vs time relative to motif transition.

    Args:
        lag_df: DataFrame with bank, lag_sec, accuracy columns.
        n_classes: Number of motif classes (for chance line).
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    bank_order = ["ventral", "dorsal"]
    banks_present = [b for b in bank_order if b in lag_df["bank"].unique()]

    for bank in banks_present:
        sub = lag_df[lag_df["bank"] == bank].sort_values("lag_sec")
        color = BANK_COLORS.get(bank, "#999999")
        label = BANK_LABELS.get(bank, bank)

        # Use per-fold mean (CI midpoint) when available so band wraps the line
        has_ci = ("accuracy_ci_low" in sub.columns
                  and "accuracy_ci_high" in sub.columns)
        if has_ci:
            ci_low = sub["accuracy_ci_low"].values
            ci_high = sub["accuracy_ci_high"].values
            valid_ci = ~(np.isnan(ci_low) | np.isnan(ci_high))
            # Per-fold mean = midpoint of CI band
            y_line = np.where(valid_ci, (ci_low + ci_high) / 2, sub["accuracy"].values)
            if np.any(valid_ci):
                ax.fill_between(
                    sub["lag_sec"].values[valid_ci],
                    ci_low[valid_ci],
                    ci_high[valid_ci],
                    color=color, alpha=0.2,
                )
        else:
            y_line = sub["accuracy"].values

        ax.plot(sub["lag_sec"], y_line, "o-", color=color, label=label,
                linewidth=2, markersize=5)

        # Mark optimal lag
        best_idx = int(np.argmax(y_line))
        ax.plot(sub["lag_sec"].values[best_idx], y_line[best_idx], "*", color=color,
                markersize=12, zorder=5)

    # Reference lines
    chance = 1.0 / n_classes
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[1], chance, " Chance", va="bottom", ha="right",
            fontsize=10, color="gray", fontstyle="italic")

    ax.set_xlabel("Lag (seconds)\n\u2190 neural leads | neural lags \u2192")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Decoding Accuracy vs Neural\u2013Behavior Lag", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


# Backward-compatible re-exports (moved to fig4_supplements.py in W5.1)
from freemat.single_fm.viz.fig4_supplements import (  # noqa: F401,E402
    plot_fig4_supp_next_syllable,
    plot_fig4_supp_descriptives,
    plot_fig4_supp_confusion_matrix,
    plot_fig4_supp_selectivity_by_bank,
    plot_fig4_supp_transition_matrix,
    plot_fig4_supp_transition_no_self,
    plot_fig4_supp_onset_peth,
    plot_fig4_supp_onset_peth_smoothed,
)

__all__ = [
    "plot_fig4b_tuning_heatmap",
    "plot_fig4b_electrode_selectivity",
    "plot_fig4b_electrode_preferred",
    "plot_fig4b_bubble",
    "plot_fig4b_bubble_3d",
    "plot_fig4c_decoding_accuracy",
    "plot_fig4d_lag_sweep",
    "plot_fig4_supp_next_syllable",
    "plot_fig4_supp_descriptives",
    "plot_fig4_supp_confusion_matrix",
    "plot_fig4_supp_selectivity_by_bank",
    "plot_fig4_supp_transition_matrix",
    "plot_fig4_supp_transition_no_self",
    "plot_fig4_supp_onset_peth",
    "plot_fig4_supp_onset_peth_smoothed",
]
