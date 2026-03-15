"""Supplement visualization functions for Figure 4: Behavioral Motif Analysis.

Extracted from fig4_panels.py (W5.1 refactoring).

Functions:
    plot_fig4_supp_next_syllable: Next-syllable prediction around transitions.
    plot_fig4_supp_descriptives: Motif frequency and tuning variance.
    plot_fig4_supp_confusion_matrix: Per-bank confusion matrix heatmaps.
    plot_fig4_supp_selectivity_by_bank: Selectivity violin by bank.
    plot_fig4_supp_transition_matrix: Syllable transition probabilities.
    plot_fig4_supp_transition_no_self: Transitions excluding self-transitions.
    plot_fig4_supp_onset_peth: Onset-aligned population PETH heatmap.
    plot_fig4_supp_onset_peth_smoothed: Raw vs Gaussian-smoothed PETH.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Duplicated from fig4_panels to avoid circular import
# (fig4_panels re-exports from this module for backward compat)
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


def plot_fig4_supp_next_syllable(
    next_df: pd.DataFrame,
    bin_s: float = 0.5,
    n_classes: int = 10,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Supplement: Next-syllable prediction accuracy around transitions.

    Args:
        next_df: DataFrame with tau_sec, accuracy columns.
        bin_s: Bin size in seconds.
        n_classes: Number of motif classes.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    tau = next_df["tau_sec"].values
    acc = next_df["accuracy"].values

    ax.plot(tau, acc, "o-", color="#7B3294", linewidth=2, markersize=5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(1.0 / n_classes, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label=f"Chance (1/{n_classes})")

    ax.set_xlabel("Time relative to transition (s)")
    ax.set_ylabel("Next-Syllable Accuracy")
    ax.set_title("Next-Motif Prediction Around Transitions", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def plot_fig4_supp_descriptives(
    tuning_df: pd.DataFrame,
    motif_order: List[int],
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Supplement: Motif descriptive statistics.

    Subpanels:
        A) Frequency bar chart (bin counts per motif).
        B) Tuning variance across units per motif.

    Args:
        tuning_df: Tuning DataFrame with motif_{id}_count and motif_{id}_z columns.
        motif_order: Motif IDs in display order.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: Frequency
    ax = axes[0]
    counts = []
    for mid in motif_order:
        col = f"motif_{mid}_count"
        if col in tuning_df.columns:
            counts.append(int(tuning_df[col].iloc[0]))
        else:
            counts.append(0)
    x_pos = np.arange(len(motif_order))
    ax.bar(x_pos, counts, color="#457B9D", edgecolor="white", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in motif_order], fontsize=11)
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Bin count")
    ax.set_title("Motif Frequency", fontweight="bold")

    # Panel B: Tuning variance
    ax = axes[1]
    z_cols = [f"motif_{mid}_z" for mid in motif_order]
    available = [c for c in z_cols if c in tuning_df.columns]
    if available:
        var_per_motif = tuning_df[available].var(axis=0).values
        ax.bar(np.arange(len(available)), var_per_motif, color="#E63946",
               edgecolor="white", alpha=0.9)
        ax.set_xticks(np.arange(len(available)))
        ax.set_xticklabels([str(motif_order[i]) for i in range(len(available))], fontsize=11)
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Tuning variance (across units)")
    ax.set_title("Motif Selectivity", fontweight="bold")

    fig.tight_layout()
    return fig


def plot_fig4_supp_confusion_matrix(
    confusion_matrices: Dict[str, np.ndarray],
    class_labels: Dict[str, np.ndarray],
    cluster_order: List[int] | None = None,
    normalize: bool = True,
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Supplement: Confusion matrix heatmaps per bank.

    Args:
        confusion_matrices: Dict bank -> (N, N) confusion matrix.
        class_labels: Dict bank -> (N,) motif ID labels.
        cluster_order: If given, reorder columns/rows to match.
        normalize: If True, row-normalize to P(predicted | true).
        figsize: Figure size.

    Returns:
        Matplotlib Figure with one subplot per bank.
    """
    _setup_nature_style()

    bank_order = ["ventral", "dorsal", "pooled"]
    banks_present = [b for b in bank_order if b in confusion_matrices]
    n_banks = len(banks_present)
    if n_banks == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_banks, figsize=figsize,
                              constrained_layout=True)
    if n_banks == 1:
        axes = [axes]

    for ax, bank in zip(axes, banks_present):
        cm = confusion_matrices[bank].astype(float)
        labels = class_labels[bank]

        # Reorder to match cluster order
        if cluster_order is not None:
            label_list = list(labels)
            order_idx = [label_list.index(m) for m in cluster_order
                         if m in label_list]
            cm = cm[np.ix_(order_idx, order_idx)]
            labels = labels[order_idx]

        # Row-normalize
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1.0 if normalize else None,
                        interpolation="nearest")

        # Annotate cells with values >= threshold
        thresh = 0.10 if normalize else cm.max() * 0.1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val >= thresh:
                    text_color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=text_color)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted motif")
        ax.set_ylabel("True motif")
        n_total = int(confusion_matrices[bank].sum())
        title = f"{BANK_LABELS.get(bank, bank)} (n={n_total})"
        ax.set_title(title, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.08,
                 label="P(predicted | true)" if normalize else "Count")
    return fig


def plot_fig4_supp_selectivity_by_bank(
    tuning_df: pd.DataFrame,
    motif_order: List[int],
    figsize: tuple = (5, 5),
) -> plt.Figure:
    """Supplement: Per-unit motif selectivity violin by bank.

    Selectivity = max absolute z-score across motifs for each unit.

    Args:
        tuning_df: Tuning DataFrame with motif_{id}_z and bank columns.
        motif_order: Motif IDs to compute selectivity over.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    z_cols = [f"motif_{mid}_z" for mid in motif_order
              if f"motif_{mid}_z" in tuning_df.columns]

    # Compute selectivity: max |z| across motifs per unit
    z_matrix = tuning_df[z_cols].values
    selectivity = np.nanmax(np.abs(z_matrix), axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    bank_order = ["ventral", "dorsal"]
    data_by_bank = []
    labels = []
    colors = []
    for bank in bank_order:
        mask = tuning_df["bank"].values == bank
        vals = selectivity[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            data_by_bank.append(vals)
            labels.append(f"{BANK_LABELS[bank]}\n(n={len(vals)})")
            colors.append(BANK_COLORS[bank])

    if data_by_bank:
        parts = ax.violinplot(data_by_bank, positions=range(len(data_by_bank)),
                              showmeans=False, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")

        # Overlay strip (jittered dots)
        for i, vals in enumerate(data_by_bank):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(i + jitter, vals, s=3, alpha=0.3, color=colors[i],
                       zorder=2)

        # Mann-Whitney U test
        if len(data_by_bank) == 2:
            from scipy.stats import mannwhitneyu
            stat, p_val = mannwhitneyu(data_by_bank[0], data_by_bank[1],
                                       alternative="two-sided")
            if p_val < 0.001:
                sig_str = f"p={p_val:.1e}"
            else:
                sig_str = f"p={p_val:.3f}"
            ax.text(0.95, 0.95, sig_str, ha="right", va="top",
                    fontsize=9, transform=ax.transAxes)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Selectivity (max |z|)")
    ax.set_title("Motif Selectivity by Bank", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fig4_supp_transition_matrix(
    transition_matrix: np.ndarray,
    labels: np.ndarray,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Supplement: Syllable transition probability heatmap.

    Args:
        transition_matrix: (N, N) row-normalized transition probabilities.
        labels: (N,) motif IDs.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(transition_matrix, cmap="YlOrRd", vmin=0,
                    interpolation="nearest")

    # Annotate cells >= 0.05
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            val = transition_matrix[i, j]
            if val >= 0.05:
                text_color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("To motif")
    ax.set_ylabel("From motif")
    ax.set_title("Motif Transition Probabilities", fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="P(next | current)")
    fig.tight_layout()
    return fig


def plot_fig4_supp_transition_no_self(
    transition_matrix: np.ndarray,
    labels: np.ndarray,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Supplement: Transition probabilities excluding self-transitions.

    Zeros out diagonal (self-transitions), re-normalizes rows,
    and displays the off-diagonal transition structure.

    Args:
        transition_matrix: (N, N) row-normalized transition probabilities.
        labels: (N,) motif IDs.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    matrix = transition_matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0,
                    interpolation="nearest")

    # Annotate cells >= 0.03 (lower threshold since values are smaller)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val >= 0.03:
                text_color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("To motif")
    ax.set_ylabel("From motif")
    ax.set_title("Motif Transition Probabilities (excl. self)",
                  fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="P(next | current)")
    fig.tight_layout()
    return fig


def plot_fig4_supp_onset_peth(
    peth: np.ndarray,
    tau_bins: np.ndarray,
    unit_banks: np.ndarray,
    bin_s: float = 0.5,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Supplement: Onset-aligned population PETH heatmap.

    Shows z-scored mean firing rate relative to syllable transitions,
    sorted by bank then depth.

    Args:
        peth: (n_units, window_size) mean firing aligned to onsets.
        tau_bins: Relative bin indices.
        unit_banks: Bank labels per unit.
        bin_s: Bin duration in seconds.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    # Z-score each unit (row) for display
    row_mean = np.nanmean(peth, axis=1, keepdims=True)
    row_std = np.nanstd(peth, axis=1, keepdims=True)
    row_std[row_std == 0] = 1
    peth_z = (peth - row_mean) / row_std

    # Sort by bank (ventral first)
    bank_order = np.array([0 if b == "ventral" else 1 for b in unit_banks])
    sort_idx = np.argsort(bank_order, kind="stable")
    peth_z = peth_z[sort_idx]
    sorted_banks = unit_banks[sort_idx]
    n_ventral = (sorted_banks == "ventral").sum()

    tau_sec = tau_bins * bin_s

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1, 0.03, 0.15], figure=fig)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    vmax = 2.0
    im = ax.imshow(
        peth_z, aspect="auto", cmap="viridis", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
        extent=[tau_sec[0], tau_sec[-1], peth_z.shape[0], 0],
    )

    # Bank separator
    if 0 < n_ventral < len(sorted_banks):
        ax.axhline(n_ventral, color="white", linewidth=1.5, linestyle="--")

    # Onset line
    ax.axvline(0, color="white", linewidth=1.5, linestyle="-", alpha=0.8)

    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("Units (sorted by bank)")

    # Colorbar in dedicated axis
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-scored FR", fontsize=12)

    # Bank labels in the spacer column
    ax_bank = fig.add_subplot(gs[2])
    ax_bank.set_xlim(0, 1)
    ax_bank.set_ylim(ax.get_ylim())
    ax_bank.axis("off")
    y_ventral = n_ventral / 2
    y_dorsal = n_ventral + (len(sorted_banks) - n_ventral) / 2
    ax_bank.text(0.3, y_ventral, "Ventral", ha="left", va="center",
                 fontsize=11, fontweight="bold")
    ax_bank.text(0.3, y_dorsal, "Dorsal", ha="left", va="center",
                 fontsize=11, fontweight="bold")

    ax.set_title("Population Activity Around Motif Transitions",
                  fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fig4_supp_onset_peth_smoothed(
    peth: np.ndarray,
    tau_bins: np.ndarray,
    unit_banks: np.ndarray,
    bin_s: float = 0.5,
    sigma_bins: int = 2,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Supplement: Raw vs Gaussian-smoothed onset-aligned PETH.

    Two-panel figure: raw (left) and temporally smoothed (right).
    Uses bilinear interpolation and percentile-based vmax for
    publication-quality heatmaps.

    Args:
        peth: (n_units, window_size) mean firing aligned to onsets.
        tau_bins: Relative bin indices.
        unit_banks: Bank labels per unit.
        bin_s: Bin duration in seconds.
        sigma_bins: Gaussian smoothing sigma in bins.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with raw and smoothed panels.
    """
    from scipy.ndimage import gaussian_filter1d

    _setup_nature_style()

    # Z-score each unit (row)
    row_mean = np.nanmean(peth, axis=1, keepdims=True)
    row_std = np.nanstd(peth, axis=1, keepdims=True)
    row_std[row_std == 0] = 1
    peth_z = (peth - row_mean) / row_std

    # Smoothed version
    peth_smooth = gaussian_filter1d(peth_z, sigma=sigma_bins, axis=1)

    # Sort by bank (ventral first)
    bank_order = np.array([0 if b == "ventral" else 1 for b in unit_banks])
    sort_idx = np.argsort(bank_order, kind="stable")
    peth_z = peth_z[sort_idx]
    peth_smooth = peth_smooth[sort_idx]
    sorted_banks = unit_banks[sort_idx]
    n_ventral = (sorted_banks == "ventral").sum()

    tau_sec = tau_bins * bin_s
    # Percentile-based vmax for better dynamic range
    vmax = float(np.percentile(np.abs(peth_z[np.isfinite(peth_z)]), 98))
    vmax = max(vmax, 0.5)  # Floor to avoid degenerate color scale

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.04], figure=fig)
    ax_raw = fig.add_subplot(gs[0])
    ax_sm = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    for ax, data, title in [
        (ax_raw, peth_z, "Raw"),
        (ax_sm, peth_smooth, f"Smoothed (\u03c3={sigma_bins * bin_s:.1f}s)"),
    ]:
        im = ax.imshow(
            data, aspect="auto", cmap="viridis", vmin=-vmax, vmax=vmax,
            interpolation="bilinear",
            extent=[tau_sec[0], tau_sec[-1], data.shape[0], 0],
        )
        if 0 < n_ventral < len(sorted_banks):
            ax.axhline(n_ventral, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(0, color="white", linewidth=1.5, linestyle="-", alpha=0.8)
        ax.set_xlabel("Time relative to onset (s)")
        ax.set_title(title, fontsize=14, fontweight="bold")

    ax_raw.set_ylabel("Units (sorted by bank)")
    ax_sm.set_yticklabels([])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-scored FR", fontsize=12)

    fig.tight_layout()
    return fig


__all__ = [
    "plot_fig4_supp_next_syllable",
    "plot_fig4_supp_descriptives",
    "plot_fig4_supp_confusion_matrix",
    "plot_fig4_supp_selectivity_by_bank",
    "plot_fig4_supp_transition_matrix",
    "plot_fig4_supp_transition_no_self",
    "plot_fig4_supp_onset_peth",
    "plot_fig4_supp_onset_peth_smoothed",
    "plot_fig4_supp_gradient_summary",
]


def plot_fig4_supp_gradient_summary(
    gradient_df: pd.DataFrame,
    session_id: str = "",
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Supplement: Gradient statistics summary table figure.

    Renders the gradient stats DataFrame as a color-coded table showing
    test results for selectivity and bandwidth across bank, depth, and
    AP axes.

    Args:
        gradient_df: Summary DataFrame from compute_all_gradient_stats.
        session_id: Session identifier for title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with table.
    """
    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Prepare display columns
    display_cols = ["metric", "test", "direction", "statistic", "p_value",
                    "effect_size", "n1", "n2", "fdr_family"]
    if "p_fdr" in gradient_df.columns:
        display_cols.append("p_fdr")
    if "significant_fdr" in gradient_df.columns:
        display_cols.append("significant_fdr")

    cols = [c for c in display_cols if c in gradient_df.columns]
    table_data = gradient_df[cols].copy()

    # Format numeric columns
    for col in ["statistic", "effect_size", "p_value", "p_fdr"]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(
                lambda v: f"{v:.4f}" if pd.notna(v) else "—"
            )
    for col in ["n1", "n2"]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(
                lambda v: f"{int(v)}" if pd.notna(v) else "—"
            )

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Color significant rows
    if "significant_fdr" in gradient_df.columns:
        for row_idx in range(len(gradient_df)):
            if gradient_df["significant_fdr"].iloc[row_idx]:
                for col_idx in range(len(cols)):
                    table[row_idx + 1, col_idx].set_facecolor("#E8F5E9")

    title = "Electrode Gradient Statistics"
    if session_id:
        title += f" ({session_id})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig
