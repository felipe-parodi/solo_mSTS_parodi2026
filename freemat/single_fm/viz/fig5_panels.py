"""Figure 5 panel plots: CEBRA population manifold analysis.

Panels:
  5A: Spatial-lite + CEBRA-Time embeddings colored by height tier (2D + 3D)
  5B: Behavior CEBRA embedding colored by motif
  5C: Decoding comparison (CEBRA vs PCA vs Raw) with SEM error bars
  5D: Transition velocity ratio profile (normalized, no null CI band)
  5E: Context-dependent coding ratios (with min_qualifying_motifs filter)
  5F: Cross-session replication (M2) — split accuracy vs ratio subpanels
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from freemat.single_fm.viz.pub_data import (
    HEIGHT_COLORS_FIG5,
    HEIGHT_NAMES,
    METHOD_COLORS,
    MOTIF_NAMES_FIG5,
    SUBJECT_COLORS,
    SUBJECT_MAP,
    setup_nature_style,
)

_LEGEND_KW: Dict[str, Any] = dict(
    facecolor="#D9D9D9", edgecolor="#999999", framealpha=0.95,
)


# ── Panel 5A: Embeddings colored by height ──────────────────────────────

def fig5a_embedding_height(
    emb_spatial_3d: np.ndarray,
    emb_time_3d: np.ndarray,
    height_tier: np.ndarray,
    y_motif: Optional[np.ndarray] = None,
    title_prefix: str = "",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Side-by-side 3D scatter: Spatial-lite CEBRA (left) + CEBRA-Time (right).

    Args:
        emb_spatial_3d: (n_bins, 3) Spatial-lite CEBRA embedding.
        emb_time_3d: (n_bins, 3) Self-supervised CEBRA-Time embedding.
        height_tier: (n_bins,) height tier labels (0=floor, 1=bench, 2=ceiling).
        y_motif: (n_bins,) optional motif labels to mask motif-0 bins.
        title_prefix: Optional prefix for subplot titles.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()

    # Mask out motif-0 bins (ethogram gaps) if motif labels provided
    mask = np.ones(len(height_tier), dtype=bool)
    if y_motif is not None:
        mask = y_motif > 0

    fig = plt.figure(figsize=figsize)

    for idx, (emb, subtitle) in enumerate([
        (emb_spatial_3d, "CEBRA-Spatial-lite"),
        (emb_time_3d, "CEBRA-Time (self-supervised)"),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        for tier in sorted(HEIGHT_COLORS_FIG5.keys()):
            tier_mask = mask & (height_tier == tier)
            if tier_mask.sum() == 0:
                continue
            ax.scatter(
                emb[tier_mask, 0], emb[tier_mask, 1], emb[tier_mask, 2],
                c=HEIGHT_COLORS_FIG5[tier], s=1, alpha=0.4,
                label=HEIGHT_NAMES[tier], rasterized=True,
            )
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.set_zlabel("Dim 3", fontsize=10)
        ax.set_title(f"{title_prefix}{subtitle}", fontweight="bold")
        ax.legend(markerscale=8, loc="upper right", fontsize=9, **_LEGEND_KW)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


# ── Panel 5A (v2): 2D embedding projection colored by height ─────────────

def fig5a_embedding_height_2d(
    emb_spatial_3d: np.ndarray,
    emb_time_3d: np.ndarray,
    height_tier: np.ndarray,
    y_motif: Optional[np.ndarray] = None,
    title_prefix: str = "",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Side-by-side 2D scatter (dims 1 vs 2): Spatial-lite (left) + CEBRA-Time (right).

    Includes silhouette score as small illustrative annotation (not a
    main quantitative claim — supervised embeddings can produce
    misleadingly high silhouette).

    Args:
        emb_spatial_3d: (n_bins, 3) Spatial-lite CEBRA embedding.
        emb_time_3d: (n_bins, 3) Self-supervised CEBRA-Time embedding.
        height_tier: (n_bins,) height tier labels (0=floor, 1=bench, 2=ceiling).
        y_motif: (n_bins,) optional motif labels to mask motif-0 bins.
        title_prefix: Optional prefix for subplot titles.

    Returns:
        matplotlib Figure.
    """
    from sklearn.metrics import silhouette_score

    setup_nature_style()

    mask = np.ones(len(height_tier), dtype=bool)
    if y_motif is not None:
        mask = y_motif > 0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (emb, subtitle) in zip(axes, [
        (emb_spatial_3d, "CEBRA-Spatial-lite"),
        (emb_time_3d, "CEBRA-Time (self-supervised)"),
    ]):
        for tier in sorted(HEIGHT_COLORS_FIG5.keys()):
            tier_mask = mask & (height_tier == tier)
            if tier_mask.sum() == 0:
                continue
            ax.scatter(
                emb[tier_mask, 0], emb[tier_mask, 1],
                c=HEIGHT_COLORS_FIG5[tier], s=1, alpha=0.3,
                label=HEIGHT_NAMES[tier], rasterized=True,
            )

        # Silhouette score (illustrative only)
        valid = mask & np.isin(height_tier, list(HEIGHT_COLORS_FIG5.keys()))
        if valid.sum() > 100:
            n_sample = min(5000, valid.sum())
            rng = np.random.default_rng(42)
            idx = rng.choice(np.where(valid)[0], n_sample, replace=False)
            sil = silhouette_score(emb[idx, :2], height_tier[idx])
            ax.text(
                0.03, 0.03, f"sil = {sil:.2f}",
                transform=ax.transAxes, fontsize=8, color="gray",
                fontstyle="italic",
            )

        ax.set_xlabel("Dim 1", fontsize=11)
        ax.set_ylabel("Dim 2", fontsize=11)
        ax.set_title(f"{title_prefix}{subtitle}", fontweight="bold")
        ax.legend(markerscale=8, loc="upper right", fontsize=9, **_LEGEND_KW)

    fig.tight_layout()
    return fig


# ── Panel 5B1: Height embedding 3D (Spatial-lite only) ─────────────────


def fig5b1_embedding_height_3d(
    emb_spatial_3d: np.ndarray,
    height_tier: np.ndarray,
    y_motif: Optional[np.ndarray] = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """Single 3D scatter of Spatial-lite CEBRA embedding colored by height.

    Args:
        emb_spatial_3d: (n_bins, 3) Spatial-lite CEBRA embedding.
        height_tier: (n_bins,) height tier labels (0=floor, 1=bench, 2=ceiling).
        y_motif: Optional motif labels to mask motif-0 bins.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()

    mask = np.ones(len(height_tier), dtype=bool)
    if y_motif is not None:
        mask = y_motif > 0

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for tier in sorted(HEIGHT_COLORS_FIG5.keys()):
        tier_mask = mask & (height_tier == tier)
        if tier_mask.sum() == 0:
            continue
        ax.scatter(
            emb_spatial_3d[tier_mask, 0],
            emb_spatial_3d[tier_mask, 1],
            emb_spatial_3d[tier_mask, 2],
            c=HEIGHT_COLORS_FIG5[tier], s=1, alpha=0.4,
            label=HEIGHT_NAMES[tier], rasterized=True,
        )

    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.set_zlabel("Dim 3", fontsize=10)
    ax.set_title("CEBRA-Spatial-lite", fontweight="bold")
    ax.legend(markerscale=8, loc="upper right", fontsize=9, **_LEGEND_KW)
    ax.tick_params(labelsize=8)

    # Training-regime annotation
    ax.text2D(
        0.02, 0.95, "Trained without\nheight labels",
        transform=ax.transAxes, fontsize=9, fontstyle="italic",
        va="top", color="#333333",
    )

    fig.tight_layout()
    return fig


# ── Panel 5B2: Motif embedding 3D ─────────────────────────────────────

def fig5b_embedding_motif(
    emb_behavior_3d: np.ndarray,
    y_motif: np.ndarray,
    motif_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """3D scatter of Behavior CEBRA embedding colored by motif class.

    Args:
        emb_behavior_3d: (n_bins, 3) Behavior CEBRA embedding.
        y_motif: (n_bins,) motif labels.
        motif_names: Optional mapping motif_id -> name.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()
    if motif_names is None:
        motif_names = MOTIF_NAMES_FIG5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    unique_motifs = sorted(set(y_motif[y_motif > 0]))
    cmap = plt.colormaps["tab10"]

    for i, m in enumerate(unique_motifs):
        m_mask = y_motif == m
        label = motif_names.get(m, f"M{m}")
        ax.scatter(
            emb_behavior_3d[m_mask, 0], emb_behavior_3d[m_mask, 1],
            emb_behavior_3d[m_mask, 2],
            c=[cmap(i % 10)], s=1, alpha=0.4, label=label, rasterized=True,
        )

    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.set_zlabel("Dim 3", fontsize=10)
    ax.set_title("CEBRA-Behavior", fontweight="bold")
    if len(unique_motifs) <= 12:
        ax.legend(markerscale=8, loc="upper right", fontsize=8, ncol=2,
                  **_LEGEND_KW)
    ax.tick_params(labelsize=8)

    # Training-regime annotation
    ax.text2D(
        0.02, 0.95, "Trained with full\nbehavioral labels",
        transform=ax.transAxes, fontsize=9, fontstyle="italic",
        va="top", color="#333333",
    )

    fig.tight_layout()
    return fig


# ── Panel 5C: Decoding comparison ───────────────────────────────────────

def fig5c_decoding_comparison(
    results_all: Dict[str, Any],
    sessions: Optional[List[str]] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Grouped bar chart: CEBRA vs PCA vs Raw for height + motif decoding.

    Args:
        results_all: Dict mapping session_key -> results dict (from results.json).
        sessions: Session keys to include (default: all in results_all).

    Returns:
        matplotlib Figure with two sub-panels (height left, motif right).
    """
    setup_nature_style()
    if sessions is None:
        sessions = list(results_all.keys())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    method_keys_height = [
        ("CEBRA-SpatialLite-8D kNN", "CEBRA"),
        ("PCA-8D kNN", "PCA-8D"),
        ("Raw logistic", "Raw"),
    ]
    method_keys_motif = [
        ("CEBRA-Behavior-8D kNN", "CEBRA"),
        ("PCA-8D kNN", "PCA-8D"),
        ("Raw logistic", "Raw"),
    ]

    for ax, (panel_title, metric_key, method_keys, chance_default) in zip(
        axes,
        [
            ("Height (3-class)", "height_3class", method_keys_height, 1 / 3),
            ("Motif", "motif", method_keys_motif, 0.1),
        ],
    ):
        n_sessions = len(sessions)
        n_methods = len(method_keys)
        width = 0.8 / n_methods
        x = np.arange(n_sessions)

        for j, (json_key, display_name) in enumerate(method_keys):
            accs = []
            sems = []
            n_folds = 5
            for sess in sessions:
                r = results_all[sess]
                decode = r.get("decode_results", {}).get(metric_key, {})
                method_result = decode.get(json_key, {})
                # Use knn_ba for CEBRA/PCA, logistic_ba for Raw
                if "knn_ba" in method_result:
                    accs.append(method_result["knn_ba"])
                    std = method_result.get("knn_std", 0)
                elif "logistic_ba" in method_result:
                    accs.append(method_result["logistic_ba"])
                    std = method_result.get("logistic_std", 0)
                else:
                    accs.append(np.nan)
                    std = 0
                sems.append(std / np.sqrt(n_folds))

            color = METHOD_COLORS.get(display_name, "#999999")
            ax.bar(
                x + j * width - (n_methods - 1) * width / 2,
                accs, width, label=display_name, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5,
                yerr=sems, capsize=2, error_kw={"linewidth": 1},
            )

        # Chance line
        chance_vals = []
        for sess in sessions:
            r = results_all[sess]
            decode = r.get("decode_results", {}).get(metric_key, {})
            first_method = list(decode.values())[0] if decode else {}
            chance_vals.append(first_method.get("chance", chance_default))
        mean_chance = np.nanmean(chance_vals)
        ax.axhline(mean_chance, color="gray", linestyle="--", linewidth=1,
                    label=f"Chance ({mean_chance:.1%})", zorder=0)

        # Labels
        short_labels = []
        for s in sessions:
            subject = SUBJECT_MAP.get(s, s)
            date = s.split("_")[1]
            short_labels.append(f"{subject}\n{date}")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(panel_title, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", **_LEGEND_KW)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# ── Panel 5C1 / 5C2: Split decode panels ──────────────────────────────

def _decode_single_panel(
    ax: plt.Axes,
    results_all: Dict[str, Any],
    sessions: List[str],
    panel_title: str,
    metric_key: str,
    method_keys: List[tuple],
    chance_default: float,
) -> None:
    """Draw a single grouped-bar decoding panel on provided axes."""
    n_sessions = len(sessions)
    n_methods = len(method_keys)
    width = 0.8 / n_methods
    x = np.arange(n_sessions)

    for j, (json_key, display_name) in enumerate(method_keys):
        accs = []
        sems = []
        for sess in sessions:
            r = results_all[sess]
            decode = r.get("decode_results", {}).get(metric_key, {})
            method_result = decode.get(json_key, {})
            if "knn_ba" in method_result:
                accs.append(method_result["knn_ba"])
                std = method_result.get("knn_std", 0)
            elif "logistic_ba" in method_result:
                accs.append(method_result["logistic_ba"])
                std = method_result.get("logistic_std", 0)
            else:
                accs.append(np.nan)
                std = 0
            sems.append(std / np.sqrt(5))

        color = METHOD_COLORS.get(display_name, "#999999")
        ax.bar(
            x + j * width - (n_methods - 1) * width / 2,
            accs, width, label=display_name, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
            yerr=sems, capsize=2, error_kw={"linewidth": 1},
        )

    chance_vals = []
    for sess in sessions:
        r = results_all[sess]
        decode = r.get("decode_results", {}).get(metric_key, {})
        first_method = list(decode.values())[0] if decode else {}
        chance_vals.append(first_method.get("chance", chance_default))
    mean_chance = np.nanmean(chance_vals)
    ax.axhline(mean_chance, color="gray", linestyle="--", linewidth=1,
               label=f"Chance ({mean_chance:.1%})", zorder=0)

    short_labels = [f"{SUBJECT_MAP.get(s, s)}\n{s.split('_')[1]}" for s in sessions]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(panel_title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", **_LEGEND_KW)
    ax.set_ylim(0, 1)


def fig5c1_decode_height(
    results_all: Dict[str, Any],
    sessions: Optional[List[str]] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Standalone height (3-class) decoding bar chart.

    Args:
        results_all: Dict mapping session_key -> results dict.
        sessions: Session keys to include.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()
    if sessions is None:
        sessions = list(results_all.keys())
    fig, ax = plt.subplots(figsize=figsize)
    _decode_single_panel(
        ax, results_all, sessions,
        panel_title="Height (3-class)",
        metric_key="height_3class",
        method_keys=[
            ("CEBRA-SpatialLite-8D kNN", "CEBRA"),
            ("PCA-8D kNN", "PCA-8D"),
            ("Raw logistic", "Raw"),
        ],
        chance_default=1 / 3,
    )
    fig.tight_layout()
    return fig


def fig5c2_decode_syllable(
    results_all: Dict[str, Any],
    sessions: Optional[List[str]] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Standalone syllable decoding bar chart.

    Args:
        results_all: Dict mapping session_key -> results dict.
        sessions: Session keys to include.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()
    if sessions is None:
        sessions = list(results_all.keys())
    fig, ax = plt.subplots(figsize=figsize)
    _decode_single_panel(
        ax, results_all, sessions,
        panel_title="Syllable",
        metric_key="motif",
        method_keys=[
            ("CEBRA-Behavior-8D kNN", "CEBRA"),
            ("PCA-8D kNN", "PCA-8D"),
            ("Raw logistic", "Raw"),
        ],
        chance_default=0.1,
    )
    fig.tight_layout()
    return fig


# ── Panel 5D: Transition velocity ───────────────────────────────────────

def fig5d_transition_velocity(
    dynamics_cebra: Dict[str, Any],
    dynamics_pca: Optional[Dict[str, Any]] = None,
    null_timecourse: Optional[np.ndarray] = None,
    figsize: tuple = (7, 4.5),
) -> plt.Figure:
    """Velocity profile around transitions with null CI band.

    Args:
        dynamics_cebra: CEBRA dynamics result dict (from results.json or
            recomputed wide-window dynamics).
        dynamics_pca: PCA dynamics result dict (optional overlay).
        null_timecourse: (n_shifts, n_time_bins) array of null velocity
            timecourses for CI shading.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()

    t_axis = np.array(dynamics_cebra["time_axis"])
    mean_vel = np.array(dynamics_cebra["mean_velocity"])
    sem_vel = np.array(dynamics_cebra["sem_velocity"])
    baseline = dynamics_cebra["baseline_velocity"]
    vel_ratio = dynamics_cebra.get("velocity_ratio", 0)
    p_val = dynamics_cebra.get("velocity_null_p", 1.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Null CI band
    if null_timecourse is not None and null_timecourse.shape[0] > 0:
        null_lo = np.percentile(null_timecourse, 2.5, axis=0)
        null_hi = np.percentile(null_timecourse, 97.5, axis=0)
        ax.fill_between(t_axis[:len(null_lo)], null_lo, null_hi,
                        alpha=0.2, color="gray", label="95% null CI")

    # CEBRA velocity
    ax.fill_between(t_axis, mean_vel - sem_vel, mean_vel + sem_vel,
                    alpha=0.3, color=METHOD_COLORS["CEBRA"])
    ax.plot(t_axis, mean_vel, color=METHOD_COLORS["CEBRA"], linewidth=2,
            label="CEBRA-Behavior")

    # PCA overlay
    if dynamics_pca and dynamics_pca.get("status") == "ok":
        pca_mean = np.array(dynamics_pca["mean_velocity"])
        pca_sem = np.array(dynamics_pca["sem_velocity"])
        pca_t = np.array(dynamics_pca["time_axis"])
        pca_baseline = dynamics_pca["baseline_velocity"]

        ax2 = ax.twinx()
        ax2.fill_between(pca_t, pca_mean - pca_sem, pca_mean + pca_sem,
                         alpha=0.2, color=METHOD_COLORS["PCA-8D"])
        ax2.plot(pca_t, pca_mean, color=METHOD_COLORS["PCA-8D"],
                 linewidth=2, linestyle="--", label="PCA-8D")
        ax2.axhline(pca_baseline, color=METHOD_COLORS["PCA-8D"],
                     linestyle=":", linewidth=1, alpha=0.5)
        ax2.set_ylabel("PCA-8D velocity (a.u.)", color=METHOD_COLORS["PCA-8D"])
        ax2.tick_params(axis="y", labelcolor=METHOD_COLORS["PCA-8D"])
        ax2.spines["right"].set_visible(True)

    # Reference lines
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.3f})")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label="Transition onset")

    # Annotation
    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(
        0.02, 0.95,
        f"Vel. ratio = {vel_ratio:.2f}\n{p_str}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="none"),
    )

    ax.set_xlabel("Time relative to transition (s)")
    ax.set_ylabel("CEBRA latent velocity (a.u.)")
    ax.set_title("Neural trajectory speed at behavioral transitions",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", **_LEGEND_KW)

    fig.tight_layout()
    return fig


# ── Panel 5D (v2): Transition velocity RATIO plot ────────────────────────

def fig5d_transition_velocity_ratio(
    dynamics_cebra: Dict[str, Any],
    dynamics_pca: Optional[Dict[str, Any]] = None,
    session_label: str = "",
    figsize: tuple = (7, 4.5),
) -> plt.Figure:
    """Velocity ratio profile around transitions (CEBRA + PCA on same axis).

    Plots mean_velocity / baseline (ratio) instead of raw velocity.
    No null CI band, no twinx — both methods normalized to same scale.
    Uses ±2s window from stored results.json.

    Args:
        dynamics_cebra: CEBRA dynamics result dict (from results.json).
        dynamics_pca: PCA dynamics result dict (optional overlay).
        session_label: Session label for title annotation.

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()

    t_axis = np.array(dynamics_cebra["time_axis"])
    mean_vel = np.array(dynamics_cebra["mean_velocity"])
    sem_vel = np.array(dynamics_cebra["sem_velocity"])
    baseline = dynamics_cebra["baseline_velocity"]
    vel_ratio_scalar = dynamics_cebra.get("velocity_ratio", 0)
    p_val = dynamics_cebra.get("velocity_null_p", 1.0)
    n_trans = dynamics_cebra.get("n_transitions", 0)

    fig, ax = plt.subplots(figsize=figsize)

    # CEBRA velocity ratio
    ratio_mean = mean_vel / baseline if baseline > 0 else mean_vel
    ratio_sem = sem_vel / baseline if baseline > 0 else sem_vel

    ax.fill_between(t_axis, ratio_mean - ratio_sem, ratio_mean + ratio_sem,
                    alpha=0.25, color=METHOD_COLORS["CEBRA"])
    ax.plot(t_axis, ratio_mean, color=METHOD_COLORS["CEBRA"], linewidth=2,
            label="CEBRA-Behavior")

    # PCA velocity ratio (same axis)
    if dynamics_pca and dynamics_pca.get("status") == "ok":
        pca_mean = np.array(dynamics_pca["mean_velocity"])
        pca_sem = np.array(dynamics_pca["sem_velocity"])
        pca_t = np.array(dynamics_pca["time_axis"])
        pca_baseline = dynamics_pca["baseline_velocity"]

        if pca_baseline > 0:
            pca_ratio_mean = pca_mean / pca_baseline
            pca_ratio_sem = pca_sem / pca_baseline
        else:
            pca_ratio_mean = pca_mean
            pca_ratio_sem = pca_sem

        ax.fill_between(pca_t, pca_ratio_mean - pca_ratio_sem,
                        pca_ratio_mean + pca_ratio_sem,
                        alpha=0.15, color=METHOD_COLORS["PCA-8D"])
        ax.plot(pca_t, pca_ratio_mean, color=METHOD_COLORS["PCA-8D"],
                linewidth=2, linestyle="--", label="PCA-8D")

    # Baseline reference line at ratio=1.0
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="Baseline (ratio = 1)")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label="Transition onset")

    # Annotation
    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    label_text = f"Vel. ratio = {vel_ratio_scalar:.2f}\n{p_str}"
    if n_trans > 0:
        label_text += f"\nn = {n_trans} transitions"
    ax.text(
        0.02, 0.95, label_text,
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="none"),
    )

    ax.set_xlabel("Time relative to transition (s)")
    ax.set_ylabel("Velocity ratio (vs. non-transition baseline)")
    title = "Neural trajectory speed at behavioral transitions"
    if session_label:
        title += f" — {session_label}"
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", **_LEGEND_KW)

    fig.tight_layout()
    return fig


# ── Panel 5E: Context-dependent coding ──────────────────────────────────

def fig5e_context_coding(
    results_all: Dict[str, Any],
    sessions: Optional[List[str]] = None,
    min_qualifying_motifs: int = 1,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Bar chart of context ratio per session (CEBRA + PCA backup).

    Args:
        results_all: Dict mapping session_key -> results dict.
        sessions: Session keys to include.
        min_qualifying_motifs: Minimum number of qualifying motifs to include
            a session. Sessions with fewer are excluded (default: 1 = no filter).

    Returns:
        matplotlib Figure.
    """
    setup_nature_style()
    if sessions is None:
        sessions = list(results_all.keys())

    # Count qualifying motifs per session and filter
    filtered_sessions = []
    excluded_sessions = []
    for sess in sessions:
        r = results_all[sess]
        cc = r.get("context_cebra", {})
        motif_set = set()
        for sr in cc.get("syllable_results", []):
            motif_set.add(sr.get("motif", 0))
        if len(motif_set) >= min_qualifying_motifs:
            filtered_sessions.append(sess)
        else:
            excluded_sessions.append((sess, len(motif_set)))

    if excluded_sessions:
        for sess, k in excluded_sessions:
            print(f"  [fig5e] Excluding {sess} (k={k} < min={min_qualifying_motifs})")

    sessions = filtered_sessions

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(sessions))
    width = 0.35

    cebra_ratios = []
    pca_ratios = []
    n_motifs_list = []

    for sess in sessions:
        r = results_all[sess]
        cc = r.get("context_cebra", {})
        cp = r.get("context_pca", {})
        cebra_ratios.append(cc.get("context_ratio", 0))
        pca_ratios.append(cp.get("context_ratio", 0))
        motif_set = set()
        for sr in cc.get("syllable_results", []):
            motif_set.add(sr.get("motif", 0))
        n_motifs_list.append(len(motif_set))

    ax.bar(x - width / 2, cebra_ratios, width, label="CEBRA-Spatial-lite",
           color=METHOD_COLORS["CEBRA"], alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, pca_ratios, width, label="PCA-8D",
           color=METHOD_COLORS["PCA-8D"], alpha=0.85, edgecolor="white")

    # Significance stars
    for i, sess in enumerate(sessions):
        r = results_all[sess]
        p_cebra = r.get("context_cebra", {}).get("p_value", 1.0)
        star = "***" if p_cebra < 0.001 else ("**" if p_cebra < 0.01 else
               ("*" if p_cebra < 0.05 else "n.s."))
        ypos = max(cebra_ratios[i], pca_ratios[i]) + 0.1
        ax.text(i, ypos, star, ha="center", fontsize=10, fontweight="bold")

    # Annotate n_motifs
    for i, n in enumerate(n_motifs_list):
        ax.text(i, -0.15, f"k={n}", ha="center", fontsize=9, color="gray",
                transform=ax.get_xaxis_transform())

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="No context dependence")

    short_labels = []
    for s in sessions:
        subject = SUBJECT_MAP.get(s, s)
        date = s.split("_")[1]
        short_labels.append(f"{subject}\n{date}")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Context ratio\n(between / within distance)")
    ax.set_title("Context-dependent coding (behavior-in-place)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", **_LEGEND_KW)
    max_val = max(max(cebra_ratios), max(pca_ratios))
    ax.set_ylim(0, max_val + 0.6)

    fig.tight_layout()
    return fig


# ── Panel 5F: Cross-session replication ─────────────────────────────────

def fig5f_cross_session(
    results_all: Dict[str, Any],
    session_a: str = "M2_0526",
    session_b: str = "M2_0605",
    figsize: tuple = (10, 4.5),
) -> plt.Figure:
    """M2 cross-session replication: split into accuracy vs ratio subpanels.

    Left subpanel: decode accuracy metrics (motif, next-syl prediction).
    Right subpanel: structural metrics (context ratio, velocity ratio).
    Height decoding omitted (that claim belongs to Fig 2/3).

    Args:
        results_all: Dict mapping session_key -> results dict.
        session_a: First session key (M2 session 1).
        session_b: Second session key (M2 session 2).

    Returns:
        matplotlib Figure with two subpanels.
    """
    setup_nature_style()

    def _extract_acc(r: Dict[str, Any]) -> Dict[str, float]:
        decode = r.get("decode_results", {})
        mt = decode.get("motif", {})
        dc = r.get("dynamics_cebra", {})
        return {
            "Motif\ndecode": mt.get("CEBRA-Behavior-8D kNN", {}).get("knn_ba", 0),
            "Next-syl\nprediction": dc.get("next_syllable_prediction", {}).get("knn_ba", 0),
        }

    def _extract_ratios(r: Dict[str, Any]) -> Dict[str, float]:
        dc = r.get("dynamics_cebra", {})
        cc = r.get("context_cebra", {})
        return {
            "Context\nratio": cc.get("context_ratio", 0),
            "Velocity\nratio": dc.get("velocity_ratio", 0),
        }

    ra = results_all.get(session_a, {})
    rb = results_all.get(session_b, {})

    subject_a = SUBJECT_MAP.get(session_a, session_a)
    subject_b = SUBJECT_MAP.get(session_b, session_b)
    date_a = session_a.split("_")[1]
    date_b = session_b.split("_")[1]
    color_a = SUBJECT_COLORS.get(subject_a, "#2D7D9A")
    color_b = SUBJECT_COLORS.get(subject_b, "#D4A373")

    fig, (ax_acc, ax_ratio) = plt.subplots(1, 2, figsize=figsize)

    # Left: accuracy metrics
    acc_a = _extract_acc(ra)
    acc_b = _extract_acc(rb)
    labels_acc = list(acc_a.keys())
    vals_acc_a = list(acc_a.values())
    vals_acc_b = list(acc_b.values())
    x_acc = np.arange(len(labels_acc))

    ax_acc.scatter(x_acc, vals_acc_a, s=80, color=color_a, zorder=3,
                   label=f"{subject_a} {date_a}", edgecolors="black", linewidth=0.5)
    ax_acc.scatter(x_acc, vals_acc_b, s=80, color=color_b, zorder=3,
                   marker="D", label=f"{subject_b} {date_b}",
                   edgecolors="black", linewidth=0.5)
    for i in range(len(labels_acc)):
        ax_acc.plot([x_acc[i], x_acc[i]], [vals_acc_a[i], vals_acc_b[i]],
                    color="gray", linewidth=1, zorder=1)

    ax_acc.set_xticks(x_acc)
    ax_acc.set_xticklabels(labels_acc, fontsize=10)
    ax_acc.set_ylabel("Balanced accuracy")
    ax_acc.set_title("Decode accuracy", fontweight="bold")
    ax_acc.legend(fontsize=9, loc="upper right")
    ax_acc.set_ylim(0, max(max(vals_acc_a), max(vals_acc_b)) * 1.3)

    # Right: ratio metrics
    ratio_a = _extract_ratios(ra)
    ratio_b = _extract_ratios(rb)
    labels_ratio = list(ratio_a.keys())
    vals_ratio_a = list(ratio_a.values())
    vals_ratio_b = list(ratio_b.values())
    x_ratio = np.arange(len(labels_ratio))

    ax_ratio.scatter(x_ratio, vals_ratio_a, s=80, color=color_a, zorder=3,
                     edgecolors="black", linewidth=0.5)
    ax_ratio.scatter(x_ratio, vals_ratio_b, s=80, color=color_b, zorder=3,
                     marker="D", edgecolors="black", linewidth=0.5)
    for i in range(len(labels_ratio)):
        ax_ratio.plot([x_ratio[i], x_ratio[i]], [vals_ratio_a[i], vals_ratio_b[i]],
                      color="gray", linewidth=1, zorder=1)

    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                     alpha=0.5, label="Baseline")
    ax_ratio.set_xticks(x_ratio)
    ax_ratio.set_xticklabels(labels_ratio, fontsize=10)
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_title("Structural metrics", fontweight="bold")
    ax_ratio.legend(fontsize=9, loc="upper right")
    ax_ratio.set_ylim(0, max(max(vals_ratio_a), max(vals_ratio_b)) * 1.3)

    fig.suptitle(f"Cross-session replication ({subject_a})",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


# ── Utility: Recompute transition dynamics with wider window ────────────

def recompute_transition_dynamics_wide(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    bin_s: float = 0.5,
    window_s: float = 4.0,
    n_jitter_shifts: int = 500,
) -> Dict[str, Any]:
    """Re-compute transition dynamics from saved embeddings with wider window.

    Args:
        embedding: (n_bins, n_dim) latent embedding.
        y_motif: (n_bins,) motif labels.
        bin_s: Bin size in seconds.
        window_s: Window size in seconds (each side).
        n_jitter_shifts: Number of circular-shift null tests.

    Returns:
        Dict with velocity profiles, null timecourse array, etc.
    """
    window_bins = int(window_s / bin_s)

    # Find syllable transitions (exclude motif 0)
    transitions = []
    for i in range(1, len(y_motif)):
        if y_motif[i] != y_motif[i - 1] and y_motif[i] > 0 and y_motif[i - 1] > 0:
            if i >= window_bins and i < len(y_motif) - window_bins:
                transitions.append(i)

    n_trans = len(transitions)
    if n_trans < 10:
        return {"n_transitions": n_trans, "status": "too_few_transitions"}

    # Extract velocities around transitions
    total_window = 2 * window_bins + 1
    velocities = np.zeros((n_trans, total_window - 1))

    for j, idx in enumerate(transitions):
        start = idx - window_bins
        end = idx + window_bins + 1
        traj = embedding[start:end]
        diffs = np.diff(traj, axis=0)
        velocities[j] = np.linalg.norm(diffs, axis=1)

    mean_velocity = np.mean(velocities, axis=0)
    sem_velocity = np.std(velocities, axis=0) / np.sqrt(n_trans)
    time_axis = np.arange(-window_bins, window_bins) * bin_s

    # Non-transition baseline
    full_vel = np.linalg.norm(np.diff(embedding, axis=0), axis=1)
    non_trans_mask = np.ones(len(full_vel), dtype=bool)
    for idx in transitions:
        lo = max(0, idx - window_bins - 1)
        hi = min(len(full_vel), idx + window_bins)
        non_trans_mask[lo:hi] = False
    baseline = float(np.mean(full_vel[non_trans_mask])) if non_trans_mask.any() else float(np.mean(full_vel))

    center_idx = window_bins
    peak_vel = float(mean_velocity[center_idx - 1])
    vel_ratio = peak_vel / baseline if baseline > 0 else 0.0

    # Circular-shift null: full timecourse
    rng = np.random.default_rng(42)
    null_timecourses = np.zeros((n_jitter_shifts, total_window - 1))
    null_peak_vels = np.zeros(n_jitter_shifts)
    n_total = len(y_motif)

    for shift_i in range(n_jitter_shifts):
        shift = rng.integers(1, n_total)
        shifted_motif = np.roll(y_motif, shift)
        null_trans = []
        for i in range(1, len(shifted_motif)):
            if (shifted_motif[i] != shifted_motif[i - 1]
                    and shifted_motif[i] > 0 and shifted_motif[i - 1] > 0):
                if window_bins <= i < len(shifted_motif) - window_bins:
                    null_trans.append(i)
        if len(null_trans) >= 10:
            null_vels = []
            for idx in null_trans[:n_trans]:
                start = idx - window_bins
                end = idx + window_bins + 1
                traj = embedding[start:end]
                diffs = np.diff(traj, axis=0)
                null_vels.append(np.linalg.norm(diffs, axis=1))
            null_mean = np.mean(null_vels, axis=0)
            null_timecourses[shift_i] = null_mean
            null_peak_vels[shift_i] = null_mean[center_idx - 1]
        else:
            null_timecourses[shift_i] = np.nan
            null_peak_vels[shift_i] = np.nan

    # Remove NaN rows
    valid_null = ~np.isnan(null_peak_vels)
    null_peak_valid = null_peak_vels[valid_null]
    p_value = float(np.mean(null_peak_valid >= peak_vel)) if len(null_peak_valid) > 0 else 1.0
    null_tc_valid = null_timecourses[valid_null]

    return {
        "n_transitions": n_trans,
        "status": "ok",
        "mean_velocity": mean_velocity.tolist(),
        "sem_velocity": sem_velocity.tolist(),
        "time_axis": time_axis.tolist(),
        "baseline_velocity": baseline,
        "peak_velocity_at_transition": peak_vel,
        "velocity_ratio": vel_ratio,
        "velocity_null_p": p_value,
        "velocity_null_n_shifts": n_jitter_shifts,
        "null_timecourse": null_tc_valid,
        "window_s": window_s,
    }
