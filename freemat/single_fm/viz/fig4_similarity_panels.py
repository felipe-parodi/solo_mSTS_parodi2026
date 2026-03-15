"""Figure 4 similarity vs confusion visualization panels.

Scatter plot of pose similarity vs neural confusion + side-by-side heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _setup_nature_style() -> None:
    """Apply Nature-style defaults."""
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


def plot_similarity_vs_confusion_scatter(
    result: Dict[str, object],
    *,
    figsize: tuple[float, float] = (5.5, 5.0),
) -> plt.Figure:
    """Scatter plot: pose similarity (x) vs neural confusion (y).

    Args:
        result: Dict from correlate_similarity_confusion with keys
            rho, p_one_sided, p_two_sided, sim_vec, conf_vec.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    sim_vec = np.asarray(result["sim_vec"])
    conf_vec = np.asarray(result["conf_vec"])
    rho = result["rho"]
    p_one = result.get("p_one_sided", float("nan"))

    ax.scatter(sim_vec, conf_vec, s=30, alpha=0.7, color="#1f77b4", edgecolors="white", linewidth=0.5)

    # Fit line
    if len(sim_vec) > 2:
        z = np.polyfit(sim_vec, conf_vec, 1)
        x_line = np.linspace(sim_vec.min(), sim_vec.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="#d62728", lw=1.5, ls="--", alpha=0.8)

    ax.set_xlabel("Pose similarity (cosine distance)")
    ax.set_ylabel("Neural confusion (symmetrized)")
    ax.set_title("Syllable Similarity vs Neural Confusion")

    # Annotation
    p_str = f"p₁={p_one:.3f}" if np.isfinite(p_one) else "p₁=N/A"
    ax.text(
        0.02, 0.98,
        f"ρ = {rho:.3f}\n{p_str}\nn = {len(sim_vec)} pairs",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    return fig


def plot_similarity_confusion_heatmaps(
    sim_matrix: np.ndarray,
    conf_matrix: np.ndarray,
    labels: List[int],
    *,
    figsize: tuple[float, float] = (12.0, 5.0),
) -> plt.Figure:
    """Side-by-side heatmaps of similarity and confusion matrices.

    Args:
        sim_matrix: (N, N) similarity distance matrix.
        conf_matrix: (N, N) symmetrized confusion matrix.
        labels: Shared label ordering.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _setup_nature_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    im1 = ax1.imshow(sim_matrix, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_title("Pose Similarity (cosine dist)")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(conf_matrix, cmap="Reds", aspect="auto")
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title("Neural Confusion (symmetrized)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Syllable Similarity vs Neural Confusion", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def save_similarity_panels(
    result: Dict[str, object],
    sim_matrix: np.ndarray,
    conf_matrix: np.ndarray,
    labels: List[int],
    output_dir: str,
) -> None:
    """Save scatter and heatmap panels.

    Args:
        result: Dict from correlate_similarity_confusion.
        sim_matrix: (N, N) similarity distance matrix.
        conf_matrix: (N, N) symmetrized confusion matrix.
        labels: Shared label ordering.
        output_dir: Directory to save figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = plot_similarity_vs_confusion_scatter(result)
    fig.savefig(out / "fig4_similarity_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig4_similarity_scatter.svg", bbox_inches="tight")
    plt.close(fig)

    fig = plot_similarity_confusion_heatmaps(sim_matrix, conf_matrix, labels)
    fig.savefig(out / "fig4_similarity_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig4_similarity_heatmaps.svg", bbox_inches="tight")
    plt.close(fig)


def plot_similarity_session_summary(
    per_session: List[Dict],
    summary: Dict,
    *,
    figsize: tuple[float, float] = (5.5, 4.0),
) -> plt.Figure:
    """Dot plot of per-session Spearman rho with median + sign test.

    Args:
        per_session: List of dicts with keys session_id, rho, p_one_sided.
        summary: Dict with median_rho, sign_test_p, n_sessions.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    _setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    session_ids = [r["session_id"] for r in per_session]
    rhos = [r["rho"] for r in per_session]
    p_vals = [r.get("p_one_sided", float("nan")) for r in per_session]

    y_pos = list(range(len(session_ids)))

    colors = ["#2D7D9A" if p < 0.05 else "#999999" for p in p_vals]
    ax.scatter(rhos, y_pos, s=80, c=colors, zorder=3, edgecolors="white", linewidth=0.5)

    # Median line
    median_rho = summary.get("median_rho", np.nanmedian(rhos))
    ax.axvline(median_rho, color="#d62728", ls="--", lw=1.5, alpha=0.8, label=f"Median = {median_rho:.3f}")
    ax.axvline(0, color="black", ls="-", lw=0.5, alpha=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(session_ids, fontsize=10)
    ax.set_xlabel("Spearman ρ (similarity vs confusion)")
    ax.set_title("Per-Session Similarity–Confusion Correlation")

    sign_p = summary.get("sign_test_p", float("nan"))
    n_neg = summary.get("n_negative", sum(1 for r in rhos if r < 0))
    n_tot = summary.get("n_sessions", len(rhos))
    ax.text(
        0.02, 0.02,
        f"Sign test: {n_neg}/{n_tot} negative, p = {sign_p:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_similarity_vs_confusion_scatter",
    "plot_similarity_confusion_heatmaps",
    "plot_similarity_session_summary",
    "save_similarity_panels",
]
