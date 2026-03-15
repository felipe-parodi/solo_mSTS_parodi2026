"""Visualization panels for Figure 4 neural manifold analysis.

PCA/UMAP scatter plots colored by behavioral motif, scree plot,
per-class silhouette bars, and ventral-vs-dorsal bank comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from freemat.single_fm.encoding.fig4_manifold import ManifoldResult
from freemat.single_fm.viz.fig4_panels import BANK_COLORS

# Motif color palette — categorical, colorblind-friendly (tab10)
_MOTIF_CMAP = plt.cm.tab10


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


def plot_manifold_pca_2d(
    result: ManifoldResult,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """PC1 vs PC2 scatter colored by motif label.

    Args:
        result: ManifoldResult from compute_manifold_analysis.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    n_components = result.pca_embedding.shape[1]
    if n_components < 2:
        ax.set_title("Insufficient PCA components for 2D plot")
        fig.tight_layout()
        return fig

    classes = sorted(set(result.motif_labels))
    for c in classes:
        mask = result.motif_labels == c
        ax.scatter(
            result.pca_embedding[mask, 0],
            result.pca_embedding[mask, 1],
            s=8, alpha=0.5, label=f"Motif {c}",
            color=_MOTIF_CMAP(c % 10),
        )

    var = result.pca_var_explained
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("Neural Manifold (PCA)", fontweight="bold")
    ax.legend(markerscale=3, fontsize=9, frameon=True, ncol=2,
              loc="best", title="Motif", title_fontsize=9)
    fig.tight_layout()
    return fig


def plot_manifold_pca_3d(
    result: ManifoldResult,
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """PC1/PC2/PC3 3D scatter colored by motif.

    Args:
        result: ManifoldResult from compute_manifold_analysis.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with 3D axes.
    """
    _setup_nature_style()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    classes = sorted(set(result.motif_labels))
    n_components = result.pca_embedding.shape[1]
    if n_components < 3:
        ax.set_title("Insufficient PCA components for 3D")
        return fig

    for c in classes:
        mask = result.motif_labels == c
        ax.scatter(
            result.pca_embedding[mask, 0],
            result.pca_embedding[mask, 1],
            result.pca_embedding[mask, 2],
            s=8, alpha=0.4, label=f"Motif {c}",
            color=_MOTIF_CMAP(c % 10),
        )

    var = result.pca_var_explained
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var[2]:.1%})")
    ax.set_title("Neural Manifold (PCA 3D)", fontweight="bold")
    ax.legend(markerscale=3, fontsize=8, frameon=True, ncol=2, loc="best")
    fig.tight_layout()
    return fig


def plot_manifold_umap(
    result: ManifoldResult,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """UMAP 2D scatter colored by motif.

    Args:
        result: ManifoldResult with has_umap=True.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    if not result.has_umap or result.umap_embedding.size == 0:
        ax.set_title("UMAP not computed")
        return fig

    classes = sorted(set(result.motif_labels))
    for c in classes:
        mask = result.motif_labels == c
        ax.scatter(
            result.umap_embedding[mask, 0],
            result.umap_embedding[mask, 1],
            s=8, alpha=0.5, label=f"Motif {c}",
            color=_MOTIF_CMAP(c % 10),
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Neural Manifold (UMAP)", fontweight="bold")
    ax.legend(markerscale=3, fontsize=9, frameon=True, ncol=2,
              loc="best", title="Motif", title_fontsize=9)
    fig.tight_layout()
    return fig


def plot_manifold_scree(
    result: ManifoldResult,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Scree plot: variance explained per PC.

    Args:
        result: ManifoldResult.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    n_pcs = len(result.pca_var_explained)
    pcs = np.arange(1, n_pcs + 1)
    ax.bar(pcs, result.pca_var_explained * 100, color="#457B9D", edgecolor="black",
           linewidth=0.5, alpha=0.85)

    cum_var = np.cumsum(result.pca_var_explained) * 100
    ax2 = ax.twinx()
    ax2.plot(pcs, cum_var, "o-", color="#E63946", markersize=4, linewidth=1.5)
    ax2.set_ylabel("Cumulative %", color="#E63946")
    ax2.tick_params(axis="y", labelcolor="#E63946")
    ax2.spines["top"].set_visible(False)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot", fontweight="bold")
    ax.set_xticks(pcs[::2])
    fig.tight_layout()
    return fig


def plot_manifold_silhouette_per_class(
    result: ManifoldResult,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Per-motif silhouette bars with null comparison.

    Args:
        result: ManifoldResult with separation metrics.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    sep = result.separation
    classes = sorted(sep.silhouette_per_class.keys())
    values = [sep.silhouette_per_class[c] for c in classes]
    labels = [f"Motif {c}" for c in classes]
    colors = [_MOTIF_CMAP(c % 10) for c in classes]

    ax.bar(range(len(classes)), values, color=colors,
           edgecolor="black", linewidth=0.5, alpha=0.85)

    # Observed mean line
    ax.axhline(sep.silhouette_mean, color="black", linestyle="-",
               linewidth=1.5, label=f"Observed: {sep.silhouette_mean:.3f}")
    # Null mean line
    ax.axhline(sep.null_mean, color="gray", linestyle="--",
               linewidth=1.5, label=f"Null: {sep.null_mean:.3f}")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Cluster Separation (p={sep.p_value:.3f})", fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    return fig


def plot_manifold_bank_comparison(
    result: ManifoldResult,
    unit_banks: np.ndarray,
    X_valid: np.ndarray | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Side-by-side PCA: ventral-only vs dorsal-only units.

    When ``X_valid`` is provided, PCA is re-computed from each bank's
    unit subset so the two panels show genuinely different embeddings.
    Without it, the full-population PCA is shown in both panels (with a
    note in the title).

    Args:
        result: ManifoldResult.
        unit_banks: Bank labels per unit (n_units,).
        X_valid: Raw firing-rate matrix (n_bins, n_units) for per-bank
            PCA recomputation. If None, falls back to full-population PCA.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with two subplots.
    """
    from freemat.single_fm.encoding.fig4_manifold import compute_pca_embedding

    _setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    classes = sorted(set(result.motif_labels))

    for ax, bank in zip(axes, ["ventral", "dorsal"]):
        bank_mask = unit_banks == bank
        if not bank_mask.any():
            ax.set_title(f"{bank.capitalize()} (no units)")
            continue

        if X_valid is not None:
            bank_cols = np.where(bank_mask)[0]
            X_bank = X_valid[:, bank_cols]
            emb, var = compute_pca_embedding(X_bank, n_components=2, zscore=True)
            pc1_label = f"PC1 ({var[0]:.1%})"
            pc2_label = f"PC2 ({var[1]:.1%})"
        else:
            emb = result.pca_embedding[:, :2]
            pc1_label = "PC1 (full pop.)"
            pc2_label = "PC2 (full pop.)"

        for c in classes:
            c_mask = result.motif_labels == c
            ax.scatter(
                emb[c_mask, 0], emb[c_mask, 1],
                s=8, alpha=0.4, label=f"Motif {c}",
                color=_MOTIF_CMAP(c % 10),
            )

        ax.set_xlabel(pc1_label)
        ax.set_ylabel(pc2_label)
        ax.set_title(f"{bank.capitalize()} Bank",
                      fontweight="bold",
                      color=BANK_COLORS.get(bank, "black"))

    axes[0].legend(markerscale=3, fontsize=8, frameon=True, ncol=2, loc="best")
    fig.suptitle("Bank-Specific Neural Manifold", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
