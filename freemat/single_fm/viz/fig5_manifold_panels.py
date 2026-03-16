"""Figure 5 manifold analysis panels: conjunctive coding, anticipatory drift, velocity prediction.

New panels for reframed Fig 5 based on post-hoc manifold analyses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from freemat.single_fm.viz.pub_data import (
    SUBJECT_COLORS,
    SUBJECT_MAP,
    save_fig,
    setup_nature_style,
)

# ── Colors ──────────────────────────────────────────────────────────────
POOLED_COLOR = "#2D7D9A"


# ══════════════════════════════════════════════════════════════════════════
# Panel: Factored vs Conjunctive coding
# ══════════════════════════════════════════════════════════════════════════

def plot_conjunctive_coding(
    all_results: List[Dict[str, Any]],
    figsize: tuple = (10, 4.5),
) -> plt.Figure:
    """Heatmap of eta-squared per dimension (motif vs height) + session summary.

    Left: Example session heatmap (8 dims x 2 variables).
    Right: Session dot plot showing conjunctive fraction.

    Args:
        all_results: List of per-session manifold analysis dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    setup_nature_style()

    fig, (ax_heat, ax_dots) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # ── Left: Example heatmap (first session with data) ──
    example = all_results[0]
    fc = example["factored_conjunctive"]
    eta_motif = np.array(fc["eta_motif"])
    eta_height = np.array(fc["eta_height"])
    n_dims = len(eta_motif)

    # Stack as (2 x n_dims) matrix
    eta_matrix = np.array([eta_motif, eta_height])

    im = ax_heat.imshow(
        eta_matrix, aspect="auto", cmap="YlOrRd", vmin=0,
        vmax=max(eta_matrix.max(), 0.1),
    )
    ax_heat.set_yticks([0, 1])
    ax_heat.set_yticklabels(["Motif", "Height"], fontweight="bold")
    ax_heat.set_xticks(range(n_dims))
    ax_heat.set_xticklabels([f"D{i+1}" for i in range(n_dims)])
    ax_heat.set_xlabel("CEBRA dimension")

    # Annotate values
    for row in range(2):
        for col in range(n_dims):
            val = eta_matrix[row, col]
            color = "white" if val > 0.3 else "black"
            ax_heat.text(
                col, row, f"{val:.2f}",
                ha="center", va="center", fontsize=8, color=color,
            )

    sess_id = example["session_id"]
    subject = SUBJECT_MAP.get(sess_id, sess_id)
    ax_heat.set_title(
        f"Variance explained per dimension ({subject})",
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label=r"$\eta^2$", pad=0.02)

    # ── Right: Conjunctive fraction across sessions ──
    sessions = []
    conj_fracs = []
    colors = []
    for r in all_results:
        sess = r["session_id"]
        sessions.append(sess)
        conj_fracs.append(r["factored_conjunctive"]["conjunctive_fraction"] * 100)
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    x = np.arange(len(sessions))
    ax_dots.bar(x, conj_fracs, color=colors, alpha=0.8, edgecolor="white", width=0.6)

    ax_dots.set_ylim(0, 110)
    ax_dots.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_dots.set_ylabel("Conjunctive dimensions (%)")
    ax_dots.set_title("All dimensions encode\nboth motif + height", fontweight="bold")

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")
    ax_dots.set_xticks(x)
    ax_dots.set_xticklabels(tick_labels, fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel: Anticipatory drift
# ══════════════════════════════════════════════════════════════════════════

def plot_anticipatory_drift(
    all_results: List[Dict[str, Any]],
    figsize: tuple = (10, 4.5),
) -> plt.Figure:
    """Time-course of cosine similarity to next-motif vs current-motif centroid.

    Left: Mean curves (all sessions overlaid) with transition at t=0.
    Right: Pre-transition ramp magnitude per session with p-values.

    Args:
        all_results: List of per-session manifold analysis dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    setup_nature_style()

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # ── Left: Time-course curves ──
    all_next = []
    all_current = []
    time_axis = None

    for r in all_results:
        ad = r["anticipatory_drift"]
        if not ad["time_axis"]:
            continue
        t = np.array(ad["time_axis"])
        sim_next = np.array(ad["mean_sim_next"])
        sim_current = np.array(ad["mean_sim_current"])

        if time_axis is None:
            time_axis = t

        ax_curve.plot(t, sim_next, color="#E63946", alpha=0.25, linewidth=0.8)
        ax_curve.plot(t, sim_current, color="#457B9D", alpha=0.25, linewidth=0.8)

        all_next.append(sim_next)
        all_current.append(sim_current)

    # Grand mean
    if all_next and time_axis is not None:
        mean_next = np.mean(all_next, axis=0)
        mean_current = np.mean(all_current, axis=0)

        ax_curve.plot(
            time_axis, mean_next, color="#E63946", linewidth=2.5,
            label="Next motif",
        )
        ax_curve.plot(
            time_axis, mean_current, color="#457B9D", linewidth=2.5,
            label="Current motif",
        )

    ax_curve.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_curve.set_xlabel("Time from transition (s)")
    ax_curve.set_ylabel("Cosine similarity to centroid")
    ax_curve.set_title("Anticipatory drift", fontweight="bold")
    ax_curve.legend(loc="center left", fontsize=9)

    # ── Right: Pre-transition ramp per session ──
    sessions = []
    ramps = []
    p_vals = []
    colors = []

    for r in all_results:
        ad = r["anticipatory_drift"]
        sess = r["session_id"]
        sessions.append(sess)
        ramps.append(ad["pre_ramp"])
        p_vals.append(ad["p_value"])
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    x = np.arange(len(sessions))
    ax_bar.bar(x, ramps, color=colors, alpha=0.8, edgecolor="white", width=0.6)

    # Significance markers
    for i, (ramp, p) in enumerate(zip(ramps, p_vals)):
        marker = _p_to_stars(p)
        if marker:
            ax_bar.text(x[i], ramp + 0.003, marker, ha="center", va="bottom", fontsize=10)

    ax_bar.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_bar.set_ylabel("Pre-transition ramp\n(late - early similarity)")
    ax_bar.set_title("Drift magnitude", fontweight="bold")

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tick_labels, fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel: Velocity → transition prediction
# ══════════════════════════════════════════════════════════════════════════

def plot_velocity_transition_prediction(
    all_results: List[Dict[str, Any]],
    figsize: tuple = (6, 4.5),
) -> plt.Figure:
    """AUC-ROC for neural velocity predicting upcoming transitions.

    Bar chart of AUC per session with chance line and significance markers.

    Args:
        all_results: List of per-session manifold analysis dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    sessions = []
    aucs = []
    p_vals = []
    colors = []

    for r in all_results:
        vp = r["velocity_prediction"]
        sess = r["session_id"]
        sessions.append(sess)
        aucs.append(vp["auc_roc"])
        p_vals.append(vp["p_value"])
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    x = np.arange(len(sessions))
    ax.bar(x, aucs, color=colors, alpha=0.8, edgecolor="white", width=0.6)

    # Chance line
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(len(sessions) - 0.5, 0.505, "Chance", ha="right", va="bottom",
            fontsize=9, color="gray")

    # Significance markers
    for i, (auc, p) in enumerate(zip(aucs, p_vals)):
        marker = _p_to_stars(p)
        if marker:
            ax.text(x[i], auc + 0.008, marker, ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0.4, max(aucs) + 0.06)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Neural velocity predicts\nupcoming transitions", fontweight="bold")

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Supplement: Timescale separation
# ══════════════════════════════════════════════════════════════════════════

def plot_timescale_separation(
    all_results: List[Dict[str, Any]],
    figsize: tuple = (10, 4.5),
) -> plt.Figure:
    """Autocorrelation decay curves: neural vs behavioral.

    Left: Decay curves per session. Right: Tau ratio summary.

    Args:
        all_results: List of per-session manifold analysis dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    setup_nature_style()

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # ── Left: Autocorrelation curves ──
    for r in all_results:
        ts = r["timescale"]
        lag = np.array(ts["lag_axis_s"])
        neural = np.array(ts["neural_autocorr"])
        behav = np.array(ts["behavioral_autocorr"])

        sess = r["session_id"]
        subject = SUBJECT_MAP.get(sess, sess)
        color = SUBJECT_COLORS.get(subject, "#999999")

        ax_curve.plot(lag, neural, color=color, linewidth=1.5, alpha=0.7)
        ax_curve.plot(lag, behav, color=color, linewidth=1.5, alpha=0.7, linestyle="--")

    # Legend proxy
    ax_curve.plot([], [], color="black", linewidth=1.5, label="Neural")
    ax_curve.plot([], [], color="black", linewidth=1.5, linestyle="--", label="Behavioral")
    ax_curve.legend(loc="upper right", fontsize=9)

    ax_curve.set_xlabel("Lag (s)")
    ax_curve.set_ylabel("Autocorrelation")
    ax_curve.set_title("Temporal autocorrelation", fontweight="bold")
    ax_curve.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    # ── Right: Tau ratio ──
    sessions = []
    ratios = []
    colors = []

    for r in all_results:
        ts = r["timescale"]
        sess = r["session_id"]
        sessions.append(sess)
        ratios.append(ts["tau_ratio"])
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    x = np.arange(len(sessions))
    ax_bar.bar(x, ratios, color=colors, alpha=0.8, edgecolor="white", width=0.6)
    ax_bar.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax_bar.text(len(sessions) - 0.5, 1.02, "Neural = Behavioral",
                ha="right", va="bottom", fontsize=8, color="gray")

    ax_bar.set_ylabel(r"$\tau_{neural}$ / $\tau_{behavioral}$")
    ax_bar.set_title("Timescale ratio", fontweight="bold")

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tick_labels, fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Supplement: Embedding RSA
# ══════════════════════════════════════════════════════════════════════════

def plot_embedding_rsa(
    all_results: List[Dict[str, Any]],
    figsize: tuple = (6, 4.5),
) -> plt.Figure:
    """Mantel rho for embedding centroid distances vs kinematic similarity.

    Args:
        all_results: List of per-session manifold analysis dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    setup_nature_style()

    fig, ax = plt.subplots(figsize=figsize)

    sessions = []
    rhos = []
    p_vals = []
    colors = []

    for r in all_results:
        rsa = r["embedding_rsa"]
        if rsa.get("rho") is None:
            continue
        sess = r["session_id"]
        sessions.append(sess)
        rhos.append(rsa["rho"])
        p_vals.append(rsa["p_permutation"])
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    if not sessions:
        ax.text(0.5, 0.5, "No RSA data", transform=ax.transAxes, ha="center")
        return fig

    x = np.arange(len(sessions))
    ax.bar(x, rhos, color=colors, alpha=0.8, edgecolor="white", width=0.6)

    for i, (rho, p) in enumerate(zip(rhos, p_vals)):
        marker = _p_to_stars(p)
        if marker:
            y_pos = rho + 0.02 if rho > 0 else rho - 0.03
            va = "bottom" if rho > 0 else "top"
            ax.text(x[i], y_pos, marker, ha="center", va=va, fontsize=10)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel(r"Mantel $\rho$ (centroid dist. ~ kin. similarity)")
    ax.set_title("Embedding geometry tracks\nbehavioral similarity", fontweight="bold")

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Data loading helper
# ══════════════════════════════════════════════════════════════════════════

def load_manifold_results(
    results_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load manifold analysis results from all session subdirs.

    Args:
        results_dir: Directory containing per-session subdirs.

    Returns:
        List of per-session result dicts, sorted by session_id.
    """
    if results_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        results_dir = _root / "results" / "single_fm" / "fig5_manifold_analyses"

    if not results_dir.exists():
        return []

    results: List[Dict[str, Any]] = []
    for sess_dir in sorted(results_dir.iterdir()):
        rpath = sess_dir / "manifold_analyses.json"
        if rpath.exists():
            with open(rpath) as f:
                results.append(json.load(f))

    return results


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════

def generate_fig5_manifold_panels(
    out_dir: Path,
    results_dir: Optional[Path] = None,
) -> None:
    """Generate all manifold analysis panels and save.

    Args:
        out_dir: Output directory for saved figures.
        results_dir: Directory with manifold analysis results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_manifold_results(results_dir)
    if not all_results:
        print("  [SKIP] No manifold analysis results found")
        return

    print(f"  Loaded manifold results for {len(all_results)} sessions")

    # Main panels
    print("  Conjunctive coding heatmap...")
    fig = plot_conjunctive_coding(all_results)
    save_fig(fig, "fig5_conjunctive_coding", out_dir)

    print("  Anticipatory drift curves...")
    fig = plot_anticipatory_drift(all_results)
    save_fig(fig, "fig5_anticipatory_drift", out_dir)

    print("  Velocity → transition prediction...")
    fig = plot_velocity_transition_prediction(all_results)
    save_fig(fig, "fig5_velocity_prediction", out_dir)

    # Supplement panels
    print("  Timescale separation (supplement)...")
    fig = plot_timescale_separation(all_results)
    save_fig(fig, "fig5_supp_timescale_separation", out_dir)

    print("  Embedding RSA (supplement)...")
    fig = plot_embedding_rsa(all_results)
    save_fig(fig, "fig5_supp_embedding_rsa", out_dir)


def _p_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


__all__ = [
    "plot_conjunctive_coding",
    "plot_anticipatory_drift",
    "plot_velocity_transition_prediction",
    "plot_timescale_separation",
    "plot_embedding_rsa",
    "load_manifold_results",
    "generate_fig5_manifold_panels",
]
