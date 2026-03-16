"""Figure 4 pooled panels: reframed behavioral-state coding (no bank split).

Panels:
  4A: Pooled onset-aligned PETHs + lagged decoding curve
  4B: Residualized vs raw motif decoding comparison
  4C: Unique motif variance (5-way: VIS, BODY_KIN, HEAD_KIN, POS, MOTIF)
  4D: Pooled RSA — neural confusion vs transition adjacency
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freemat.single_fm.viz.pub_data import (
    SUBJECT_COLORS,
    SUBJECT_MAP,
    save_fig,
)

# ── Colors ──────────────────────────────────────────────────────────────
POOLED_COLOR = "#2D7D9A"
SESSION_ALPHA = 0.35

# Motif encoding group colors (extend existing palette with MOTIF)
MOTIF_GROUP_COLORS: Dict[str, str] = {
    "vis": "#457B9D",
    "body_kin": "#E63946",
    "head_kin": "#F4A261",
    "pos": "#7B3294",
    "motif": "#2CA02C",
}
MOTIF_GROUP_LABELS: Dict[str, str] = {
    "vis": "VIS",
    "body_kin": "BODY\nKIN",
    "head_kin": "HEAD\nKIN",
    "pos": "3D\nPOS",
    "motif": "MOTIF",
}

# Syllable ID → descriptive name mapping
SYLLABLE_NAMES: Dict[int, str] = {
    1: "Rest",
    2: "Stand",
    3: "Pace",
    4: "Inspect",
    5: "Pivot",
    6: "Travel",
    7: "Scan",
    8: "Walk",
    9: "Perch",
    10: "Reach",
    11: "Sit",
    12: "Slow walk",
}


def _setup_nature_style() -> None:
    """Apply Nature-style matplotlib settings."""
    plt.rcParams.update({
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
    })


# ══════════════════════════════════════════════════════════════════════════
# Panel 4A: Pooled PETH heatmap + lag sweep curve
# ══════════════════════════════════════════════════════════════════════════

def plot_fig4a_pooled_peth_and_lag(
    peth_data: Dict[str, Any],
    lag_sweep_df: pd.DataFrame,
    sessions: Optional[List[str]] = None,
    figsize: tuple = (14, 5.5),
) -> plt.Figure:
    """Pooled PETH heatmap (all units, all sessions) + lag sweep curve.

    Args:
        peth_data: Dict from load_fig4()["peth"], keyed by session.
        lag_sweep_df: DataFrame with session_id, bank, lag_sec, accuracy.
        sessions: Sessions to include (default: all with fine PETH).
        figsize: Figure size.

    Returns:
        Matplotlib Figure with 2 subplots: heatmap (left), lag curve (right).
    """
    _setup_nature_style()

    if sessions is None:
        sessions = sorted(
            s for s in peth_data if peth_data[s].get("peth_fine") is not None
        )

    fig, (ax_peth, ax_lag) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # ── Left: Pooled PETH heatmap ──
    all_peth_z = []
    for sess_key in sessions:
        sess = peth_data.get(sess_key, {})
        peth = sess.get("peth_fine")
        if peth is None:
            continue
        # Z-score each unit row
        row_mean = peth.mean(axis=1, keepdims=True)
        row_std = peth.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        peth_z = (peth - row_mean) / row_std
        all_peth_z.append(peth_z)

    if all_peth_z:
        pooled_peth = np.vstack(all_peth_z)
        n_units = pooled_peth.shape[0]

        # Sort by peak latency
        peak_idx = np.argmax(pooled_peth, axis=1)
        sort_order = np.argsort(peak_idx)
        peth_sorted = pooled_peth[sort_order]

        # Get time axis from first available session
        ref_sess = peth_data[sessions[0]]
        tau_bins = ref_sess["tau_bins_fine"]
        bin_s = ref_sess.get("bin_s_fine", 0.05)
        time_axis = tau_bins * bin_s

        im = ax_peth.imshow(
            peth_sorted, aspect="auto", cmap="RdBu_r",
            vmin=-2, vmax=2,
            extent=[time_axis[0], time_axis[-1], n_units - 0.5, -0.5],
            interpolation="bilinear",
        )
        ax_peth.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
        ax_peth.set_xlabel("Time from transition (s)")
        ax_peth.set_ylabel(f"Units (n={n_units}, sorted by peak)")
        ax_peth.set_title("Pooled motif-transition PETH", fontweight="bold")
        fig.colorbar(im, ax=ax_peth, label="Z-scored FR", shrink=0.7, pad=0.02)
    else:
        ax_peth.text(0.5, 0.5, "No PETH data", transform=ax_peth.transAxes, ha="center")

    # ── Right: Lag sweep curve ──
    if not lag_sweep_df.empty:
        # Pool across banks: average ventral + dorsal per session per lag
        pooled_lag = lag_sweep_df.groupby(
            ["session_id", "lag_sec"], as_index=False
        )["accuracy"].mean()

        # Per-session thin lines
        for sess_key in sessions:
            sess_lag = pooled_lag[pooled_lag["session_id"] == sess_key]
            if sess_lag.empty:
                continue
            subject = SUBJECT_MAP.get(sess_key, sess_key)
            color = SUBJECT_COLORS.get(subject, "#999999")
            ax_lag.plot(
                sess_lag["lag_sec"], sess_lag["accuracy"] * 100,
                color=color, alpha=SESSION_ALPHA, linewidth=1.0,
            )

        # Grand mean + SEM
        mean_lag = pooled_lag.groupby("lag_sec")["accuracy"].agg(["mean", "sem"])
        ax_lag.plot(
            mean_lag.index, mean_lag["mean"] * 100,
            color=POOLED_COLOR, linewidth=2.5, label="Mean",
        )
        ax_lag.fill_between(
            mean_lag.index,
            (mean_lag["mean"] - mean_lag["sem"]) * 100,
            (mean_lag["mean"] + mean_lag["sem"]) * 100,
            color=POOLED_COLOR, alpha=0.2,
        )

        # Chance line (from decoding: 1/n_classes, use median across sessions)
        if "n_classes" in lag_sweep_df.columns:
            chance = 100.0 / lag_sweep_df["n_classes"].median()
        else:
            chance = 10.0  # default for ~10 motifs
        ax_lag.axhline(chance, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax_lag.text(
            0.98, chance + 0.5, "Chance",
            transform=ax_lag.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color="gray",
        )

        ax_lag.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_lag.set_xlabel("Lag (s, negative = neural leads)")
        ax_lag.set_ylabel("Balanced accuracy (%)")
        ax_lag.set_title("Lagged motif decoding", fontweight="bold")
        ax_lag.legend(loc="upper right", fontsize=9)
    else:
        ax_lag.text(0.5, 0.5, "No lag sweep data", transform=ax_lag.transAxes, ha="center")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4B: Residualized vs raw motif decoding
# ══════════════════════════════════════════════════════════════════════════

def plot_fig4b_residualized_decoding(
    resid_results: List[Dict[str, Any]],
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Grouped bar chart: raw vs residualized motif decoding per session.

    Args:
        resid_results: List of per-session result dicts from
            run_fig4_residualized_decoding.py (keys: session_id, raw,
            residualized, chance, raw_p_value, resid_p_value).
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    if not resid_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No residualized decoding data",
                transform=ax.transAxes, ha="center")
        return fig

    n_sess = len(resid_results)
    x = np.arange(n_sess)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    raw_accs = [r["raw"]["balanced_accuracy"] * 100 for r in resid_results]
    resid_accs = [r["residualized"]["balanced_accuracy"] * 100 for r in resid_results]
    chances = [r["chance"] * 100 for r in resid_results]
    labels = []
    for r in resid_results:
        sess = r["session_id"]
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        labels.append(f"{subject}\n{date}")

    # Bars
    ax.bar(
        x - bar_width / 2, raw_accs, bar_width,
        label="Raw", color="#457B9D", alpha=0.85, edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2, resid_accs, bar_width,
        label="Residualized", color="#E63946", alpha=0.85, edgecolor="white",
    )

    # Per-fold error bars (if available)
    for i, r in enumerate(resid_results):
        raw_folds = r["raw"].get("per_fold_accuracy", [])
        resid_folds = r["residualized"].get("per_fold_accuracy", [])
        if raw_folds:
            ax.errorbar(
                x[i] - bar_width / 2, raw_accs[i],
                yerr=np.std(raw_folds) * 100,
                fmt="none", color="black", capsize=3, linewidth=1,
            )
        if resid_folds:
            ax.errorbar(
                x[i] + bar_width / 2, resid_accs[i],
                yerr=np.std(resid_folds) * 100,
                fmt="none", color="black", capsize=3, linewidth=1,
            )

    # Chance line (use mean across sessions)
    mean_chance = np.mean(chances)
    ax.axhline(mean_chance, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        n_sess - 0.5, mean_chance + 0.3, f"Chance ({mean_chance:.0f}%)",
        ha="right", va="bottom", fontsize=9, color="gray",
    )

    # P-value annotations
    for i, r in enumerate(resid_results):
        p_resid = r.get("resid_p_value", 1.0)
        marker = _p_to_stars(p_resid)
        if marker:
            y_top = max(raw_accs[i], resid_accs[i]) + 2
            ax.text(x[i], y_top, marker, ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Motif decoding: raw vs residualized", fontweight="bold")
    ax.legend(loc="upper right")

    # Y-axis starts at 0
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def _p_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ══════════════════════════════════════════════════════════════════════════
# Panel 4C: Unique motif variance (5-way boxplot)
# ══════════════════════════════════════════════════════════════════════════

def plot_fig4c_unique_motif_variance(
    motif_encoding_df: pd.DataFrame,
    groups: Optional[List[str]] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Boxplot of unique delta-R2 across 5 feature groups, with session dots.

    Args:
        motif_encoding_df: Summary DataFrame from 5-way encoding run with
            columns like unique_{group}_shuffle for each group.
        groups: Feature groups to plot (default: all 5).
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    if groups is None:
        groups = ["vis", "body_kin", "head_kin", "pos", "motif"]

    # Filter to groups that exist in data
    available = [g for g in groups if f"unique_{g}_shuffle" in motif_encoding_df.columns]

    if not available:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No motif encoding data\n(run with --include-motif)",
                transform=ax.transAxes, ha="center", fontsize=12)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    data_arrays = []
    colors = []
    labels = []
    for g in available:
        col = f"unique_{g}_shuffle"
        vals = motif_encoding_df[col].dropna().values
        data_arrays.append(vals)
        colors.append(MOTIF_GROUP_COLORS.get(g, "#888888"))
        labels.append(MOTIF_GROUP_LABELS.get(g, g.upper()))

    positions = np.arange(1, len(available) + 1)

    # Boxplots
    bp = ax.boxplot(
        data_arrays, positions=positions, widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Session median dots
    if "session_id" in motif_encoding_df.columns:
        sess_ids = motif_encoding_df["session_id"].unique()
        for sess in sess_ids:
            subject = SUBJECT_MAP.get(sess, sess)
            sess_color = SUBJECT_COLORS.get(subject, "#555555")
            sess_df = motif_encoding_df[motif_encoding_df["session_id"] == sess]
            for j, g in enumerate(available):
                col = f"unique_{g}_shuffle"
                if col in sess_df.columns:
                    median_val = sess_df[col].median()
                    ax.scatter(
                        positions[j], median_val,
                        color=sess_color, s=30, zorder=5,
                        edgecolors="white", linewidth=0.5,
                    )

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"Unique $\Delta R^2$")
    ax.set_title("Unique variance by feature group (5-way)", fontweight="bold")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4D: Pooled RSA — neural confusion vs transition adjacency
# ══════════════════════════════════════════════════════════════════════════

def plot_fig4d_pooled_rsa(
    similarity_per_session: List[Dict[str, Any]],
    bank_rsa_results: Optional[List[Dict[str, Any]]] = None,
    confusion_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    class_labels: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    example_session: str = "M2_0526",
    figsize: tuple = (13, 5),
) -> plt.Figure:
    """Pooled RSA: confusion heatmap (left) + Mantel rho scatter (right).

    Args:
        similarity_per_session: List of dicts with session_id, rho, p_one_sided.
        bank_rsa_results: Optional list of bank RSA dicts with
            ventral_trans_rho, dorsal_trans_rho (for transition adjacency).
        confusion_data: Dict[session][bank] -> confusion matrix ndarray.
        class_labels: Dict[session][bank] -> label array.
        example_session: Session to use for example confusion heatmap.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with 2 subplots.
    """
    _setup_nature_style()

    has_confusion = (
        confusion_data is not None
        and example_session in confusion_data
    )
    has_rsa = bool(similarity_per_session)

    n_panels = int(has_confusion) + int(has_rsa)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No RSA data", transform=ax.transAxes, ha="center")
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    panel_idx = 0

    # ── Left: Example confusion matrix heatmap ──
    if has_confusion:
        ax = axes[panel_idx]
        panel_idx += 1

        # Use pooled confusion if available, else ventral
        bank_key = "pooled" if "pooled" in confusion_data[example_session] else "ventral"
        cm = confusion_data[example_session][bank_key]
        labels = class_labels[example_session][bank_key] if class_labels else None

        # Normalize rows
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0)
        if labels is not None:
            tick_labels = [str(int(lbl)) for lbl in labels]
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
            ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Predicted motif")
        ax.set_ylabel("True motif")
        subject = SUBJECT_MAP.get(example_session, example_session)
        ax.set_title(f"Confusion matrix ({subject})", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="P(predicted | true)")

    # ── Right: Mantel rho per session ──
    if has_rsa:
        ax = axes[panel_idx]

        # Use transition adjacency RSA if bank data available, else pose similarity
        if bank_rsa_results:
            _plot_transition_rsa_dots(ax, bank_rsa_results)
        else:
            _plot_pose_rsa_dots(ax, similarity_per_session)

    fig.tight_layout()
    return fig


def _plot_transition_rsa_dots(
    ax: plt.Axes,
    bank_rsa_results: List[Dict[str, Any]],
) -> None:
    """Dot plot of transition adjacency Mantel rho (pooled V+D mean).

    Args:
        ax: Axes to plot on.
        bank_rsa_results: List of per-session dicts with ventral_trans_rho,
            dorsal_trans_rho, ventral_trans_p, dorsal_trans_p.
    """
    sessions = []
    rhos = []
    p_vals = []

    for r in bank_rsa_results:
        sess = r["session_id"]
        v_rho = r.get("ventral_trans_rho", np.nan)
        d_rho = r.get("dorsal_trans_rho", np.nan)
        v_p = r.get("ventral_trans_p", 1.0)
        d_p = r.get("dorsal_trans_p", 1.0)

        # Pool: average V and D
        pooled_rho = np.nanmean([v_rho, d_rho])
        pooled_p = min(v_p, d_p)  # conservative: use smaller p
        sessions.append(sess)
        rhos.append(pooled_rho)
        p_vals.append(pooled_p)

    x = np.arange(len(sessions))
    colors = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    ax.bar(x, rhos, color=colors, alpha=0.7, edgecolor="white", width=0.6)

    # Significance markers
    for i, (rho, p) in enumerate(zip(rhos, p_vals)):
        marker = _p_to_stars(p)
        if marker:
            y_pos = rho + 0.02 if rho > 0 else rho - 0.04
            ax.text(x[i], y_pos, marker, ha="center", va="bottom", fontsize=10)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Mantel rho")
    ax.set_title("Neural confusion ~ transition adjacency", fontweight="bold")


def _plot_pose_rsa_dots(
    ax: plt.Axes,
    similarity_per_session: List[Dict[str, Any]],
) -> None:
    """Dot plot of pose similarity Mantel rho per session.

    Args:
        ax: Axes to plot on.
        similarity_per_session: List of dicts with session_id, rho, p_one_sided.
    """
    # Normalize session IDs
    _norm = {
        "M1_0525": "M1_0525", "M1_0609": "M1_0609", "M1_0613": "M1_0613",
        "M2_0605": "M2_0605",
    }

    sessions = []
    rhos = []
    p_vals = []

    for r in similarity_per_session:
        sess_raw = r["session_id"]
        sess = _norm.get(sess_raw, sess_raw)
        sessions.append(sess)
        rhos.append(r["rho"])
        p_vals.append(r.get("p_two_sided", r.get("p_one_sided", 1.0)))

    x = np.arange(len(sessions))
    colors = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    ax.bar(x, rhos, color=colors, alpha=0.7, edgecolor="white", width=0.6)

    for i, (rho, p) in enumerate(zip(rhos, p_vals)):
        marker = _p_to_stars(p)
        if marker:
            y_pos = rho - 0.03 if rho < 0 else rho + 0.02
            va = "top" if rho < 0 else "bottom"
            ax.text(x[i], y_pos, marker, ha="center", va=va, fontsize=10)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    tick_labels = []
    for sess in sessions:
        subject = SUBJECT_MAP.get(sess, sess)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{subject}\n{date}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Mantel rho (pose sim. ~ confusion)")
    ax.set_title("Neural confusion ~ pose similarity", fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════
# Panel 4B (new): Temporal Generalization Matrix
# ══════════════════════════════════════════════════════════════════════════

# Session order mapping for TGM data directory names
_TGM_SESSION_DIRS: Dict[str, str] = {
    "M1_0525": "single_0525_M1",
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


def load_tgm_data(
    tgm_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load TGM results from all sessions.

    Args:
        tgm_dir: Directory containing per-session TGM subdirs.

    Returns:
        Dict keyed by canonical session_id (e.g. 'M2_0526') with
        'matrix', 'tau_sec', 'diagonal', 'chance_level', etc.
    """
    if tgm_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        tgm_dir = _root / "results" / "single_fm" / "fig4_tgm"

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for sess_key, subdir_name in _TGM_SESSION_DIRS.items():
        npz_path = tgm_dir / subdir_name / "fig4_tgm.npz"
        if npz_path.exists():
            d = dict(np.load(npz_path, allow_pickle=True))
            data[sess_key] = d
    return data


def load_participation_data(
    tgm_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load participation results from all sessions.

    Args:
        tgm_dir: Directory containing per-session participation subdirs.

    Returns:
        Dict keyed by canonical session_id with 'binary_matrix',
        'participation_frac', 'bandwidth', 'motif_ids'.
    """
    if tgm_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        tgm_dir = _root / "results" / "single_fm" / "fig4_tgm"

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for sess_key, subdir_name in _TGM_SESSION_DIRS.items():
        npz_path = tgm_dir / subdir_name / "fig4_participation.npz"
        if npz_path.exists():
            d = dict(np.load(npz_path, allow_pickle=True))
            data[sess_key] = d
    return data


def plot_fig4b_tgm(
    tgm_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    tgm_dir: Optional[Path] = None,
    exemplar_session: str = "M2_0605",
    figsize: tuple = (13, 5),
) -> plt.Figure:
    """TGM heatmap (exemplar session) + diagonal accuracy trace with all sessions.

    Left: Single best-session TGM heatmap for visual clarity.
    Right: All 5 sessions' diagonal traces (thin lines) + bold mean + SEM.

    Args:
        tgm_data: Pre-loaded TGM data (from load_tgm_data). If None, loads
            from tgm_dir.
        tgm_dir: Directory with TGM results (used if tgm_data is None).
        exemplar_session: Session key for the heatmap (default: M2_0605).
        figsize: Figure size.

    Returns:
        Matplotlib Figure with 2 subplots: heatmap (left), diagonal (right).
    """
    _setup_nature_style()

    # M1/M2 labels for paper (not real names)
    _SUBJ_LABEL = {"M2": "M2", "M1": "M1"}

    if tgm_data is None:
        tgm_data = load_tgm_data(tgm_dir)

    if not tgm_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No TGM data", transform=ax.transAxes, ha="center")
        return fig

    fig, (ax_mat, ax_diag) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.8]}
    )

    # ── Collect data across sessions ──
    chance_vals = []
    tau_sec = None

    for sess_key in sorted(tgm_data.keys()):
        d = tgm_data[sess_key]
        chance_vals.append(float(d["chance_level"]))
        if tau_sec is None:
            tau_sec = d["tau_sec"]

    mean_chance = np.mean(chance_vals)

    # ── Left: Exemplar session TGM heatmap ──
    if exemplar_session not in tgm_data:
        exemplar_session = sorted(tgm_data.keys())[0]

    exemplar = tgm_data[exemplar_session]
    exemplar_matrix = exemplar["matrix"]
    exemplar_chance = float(exemplar["chance_level"])
    exemplar_subject = SUBJECT_MAP.get(exemplar_session, exemplar_session)
    exemplar_label = _SUBJ_LABEL.get(exemplar_subject, exemplar_subject)

    extent = [tau_sec[0], tau_sec[-1], tau_sec[-1], tau_sec[0]]

    im = ax_mat.imshow(
        exemplar_matrix,
        aspect="equal",
        cmap="RdYlBu_r",
        vmin=exemplar_chance - 0.02,
        vmax=np.max(exemplar_matrix) + 0.01,
        extent=extent,
        interpolation="nearest",
        origin="upper",
    )

    # Onset crosshair
    ax_mat.axhline(0, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_mat.axvline(0, color="white", linewidth=1.0, linestyle="--", alpha=0.7)

    # Diagonal reference line
    ax_mat.plot(
        [tau_sec[0], tau_sec[-1]], [tau_sec[0], tau_sec[-1]],
        color="white", linewidth=0.8, linestyle=":", alpha=0.5,
    )

    ax_mat.set_xlabel("Testing time (s)")
    ax_mat.set_ylabel("Training time (s)")
    ax_mat.set_title(f"Temporal generalization ({exemplar_label})", fontweight="bold")

    cb = fig.colorbar(im, ax=ax_mat, shrink=0.8, pad=0.02)
    cb.set_label("Balanced accuracy", fontsize=11)

    # ── Right: Diagonal trace with session lines ──
    for sess_key in sorted(tgm_data.keys()):
        d = tgm_data[sess_key]
        subject = SUBJECT_MAP.get(sess_key, sess_key)
        color = SUBJECT_COLORS.get(subject, "#999999")
        ax_diag.plot(
            tau_sec, d["diagonal"] * 100,
            color=color, alpha=SESSION_ALPHA, linewidth=1.0,
        )

    # Grand mean + SEM
    diag_stack = np.array([d["diagonal"] for d in tgm_data.values()])
    mean_diag = diag_stack.mean(axis=0)
    sem_diag = diag_stack.std(axis=0) / np.sqrt(len(diag_stack))

    ax_diag.plot(
        tau_sec, mean_diag * 100,
        color=POOLED_COLOR, linewidth=2.5, label="Mean",
    )
    ax_diag.fill_between(
        tau_sec,
        (mean_diag - sem_diag) * 100,
        (mean_diag + sem_diag) * 100,
        color=POOLED_COLOR, alpha=0.2,
    )

    # Chance line
    ax_diag.axhline(
        mean_chance * 100, color="gray", linestyle=":", linewidth=1, alpha=0.6,
    )
    ax_diag.text(
        0.02, mean_chance * 100 + 0.5,
        "Chance", transform=ax_diag.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=9, color="gray",
    )

    # Onset line
    ax_diag.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax_diag.set_xlabel("Time from transition (s)")
    ax_diag.set_ylabel("Balanced accuracy (%)")
    ax_diag.set_title("Diagonal (same-time decoding)", fontweight="bold")

    # Legend for subjects with M1/M2 labels
    from matplotlib.lines import Line2D
    handles = []
    for subj, label in [("M2", "M2"), ("M1", "M1")]:
        handles.append(Line2D(
            [0], [0], color=SUBJECT_COLORS[subj], linewidth=1.5,
            alpha=0.6, label=label,
        ))
    handles.append(Line2D(
        [0], [0], color=POOLED_COLOR, linewidth=2.5, label="Mean",
    ))
    ax_diag.legend(handles=handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4D (new): Population Participation — Jaccard Overlap
# ══════════════════════════════════════════════════════════════════════════

def plot_fig4d_participation(
    tuning_dir: Optional[Path] = None,
    sessions: Optional[List[str]] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Motif tuning similarity: correlation of tuning profiles (left) + selectivity (right).

    Left: Motif-by-motif Pearson correlation of z-scored tuning vectors
    across all units, pooled across sessions. Shows which motifs recruit
    similar vs distinct neural populations (continuous, no threshold).
    Right: Per-unit selectivity (max z-score) distribution by subject.

    Args:
        tuning_dir: Directory with per-session fig4_tuning.parquet files.
        sessions: Session subdirectory names to load. If None, uses defaults.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with 2 subplots.
    """
    _setup_nature_style()

    if tuning_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        tuning_dir = _root / "results" / "single_fm" / "production_20260220_2019" / "fig4_all4"

    if sessions is None:
        sessions = list(_TGM_SESSION_DIRS.values())

    # ── Load tuning z-scores from parquet files ──
    # Use union of all syllable IDs; average each cell over sessions that have both
    per_session_data = []
    all_selectivity = []
    all_selectivity_subj = []

    for sess_name in sessions:
        tuning_path = tuning_dir / sess_name / "fig4_tuning.parquet"
        if not tuning_path.exists():
            continue
        df = pd.read_parquet(tuning_path)
        z_cols = sorted([c for c in df.columns if c.endswith("_z") and c.startswith("motif_")])
        if z_cols:
            motif_ids = [int(c.split("_")[1]) for c in z_cols]
            per_session_data.append((sess_name, motif_ids, df))

    if not per_session_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No tuning data found", transform=ax.transAxes, ha="center")
        return fig

    # Union of all syllable IDs across sessions, sorted alphabetically by name
    all_motif_ids_raw = set().union(*(set(m) for _, m, _ in per_session_data))
    all_motif_ids = sorted(all_motif_ids_raw, key=lambda m: SYLLABLE_NAMES.get(m, str(m)))

    n_motifs = len(all_motif_ids)
    motif_to_idx = {m: i for i, m in enumerate(all_motif_ids)}

    # Accumulate correlation values and counts for averaging
    corr_sum = np.zeros((n_motifs, n_motifs), dtype=np.float64)
    corr_count = np.zeros((n_motifs, n_motifs), dtype=np.int32)

    for sess_name, motif_ids, df in per_session_data:
        z_cols = [f"motif_{m}_z" for m in motif_ids]
        z_matrix = df[z_cols].to_numpy()  # (n_units, n_session_motifs)

        # Correlation across units between tuning profiles
        corr = np.corrcoef(z_matrix.T)  # (n_session_motifs, n_session_motifs)

        # Map into the full union matrix
        for i_local, mi in enumerate(motif_ids):
            for j_local, mj in enumerate(motif_ids):
                gi, gj = motif_to_idx[mi], motif_to_idx[mj]
                corr_sum[gi, gj] += corr[i_local, j_local]
                corr_count[gi, gj] += 1

        # Per-unit selectivity = max |z| across syllables
        selectivity = np.max(np.abs(z_matrix), axis=1)

        # Determine subject from session name
        sess_key = None
        for k, v in _TGM_SESSION_DIRS.items():
            if v == sess_name:
                sess_key = k
                break
        subject = SUBJECT_MAP.get(sess_key, "Unknown") if sess_key else "Unknown"

        all_selectivity.extend(selectivity.tolist())
        all_selectivity_subj.extend([subject] * len(selectivity))

    # Average, masking cells with no data
    corr_count[corr_count == 0] = 1  # avoid division by zero
    mean_corr = corr_sum / corr_count

    if not all_selectivity:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No tuning data found", transform=ax.transAxes, ha="center")
        return fig

    fig, (ax_corr, ax_sel) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.8]}
    )

    # Use a diverging colormap centered on 0
    vmax = max(abs(np.min(mean_corr[~np.eye(n_motifs, dtype=bool)])),
               abs(np.max(mean_corr[~np.eye(n_motifs, dtype=bool)])))
    vmax = max(vmax, 0.1)  # minimum range

    im = ax_corr.imshow(
        mean_corr, cmap="RdBu_r", aspect="equal",
        vmin=-vmax, vmax=vmax,
    )

    syl_labels = [SYLLABLE_NAMES.get(m, str(m)) for m in all_motif_ids]
    ax_corr.set_xticks(range(n_motifs))
    ax_corr.set_yticks(range(n_motifs))
    ax_corr.set_xticklabels(syl_labels, fontsize=9, rotation=45, ha="right")
    ax_corr.set_yticklabels(syl_labels, fontsize=9)
    ax_corr.set_xlabel("Syllable")
    ax_corr.set_ylabel("Syllable")
    ax_corr.set_title("Syllable tuning similarity", fontweight="bold")
    fig.colorbar(im, ax=ax_corr, shrink=0.8, label="Pearson r")

    # ── Right: Selectivity distribution by subject ──
    selectivity_arr = np.array(all_selectivity)
    subjects_arr = np.array(all_selectivity_subj)

    bins = np.linspace(0, min(selectivity_arr.max(), 8), 30)

    _SUBJ_LABEL = {"M2": "M2", "M1": "M1"}
    for subj in ["M2", "M1"]:
        mask = subjects_arr == subj
        if mask.any():
            ax_sel.hist(
                selectivity_arr[mask], bins=bins, alpha=0.5,
                color=SUBJECT_COLORS[subj], label=_SUBJ_LABEL[subj],
                density=True, edgecolor="white",
            )

    # Pooled outline
    ax_sel.hist(
        selectivity_arr, bins=bins, alpha=0.0, edgecolor=POOLED_COLOR,
        linewidth=2.0, density=True, histtype="step", label="All",
    )

    ax_sel.set_xlabel("Max |z-score| across syllables")
    ax_sel.set_ylabel("Density")
    ax_sel.set_title("Syllable selectivity", fontweight="bold")
    ax_sel.legend(fontsize=9)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4D supplement: Per-session tuning similarity matrices
# ══════════════════════════════════════════════════════════════════════════

# Session display labels (M1/M2 for paper)
_SESSION_LABELS: Dict[str, str] = {
    "single_0525_M1": "M1 (0525)",
    "single_0526_M2": "M2 (0526)",
    "single_0605_M2": "M2 (0605)",
    "single_0609_M1": "M1 (0609)",
    "single_0613_M1": "M1 (0613)",
}


def plot_fig4d_participation_per_session(
    tuning_dir: Optional[Path] = None,
    sessions: Optional[List[str]] = None,
    figsize: tuple = (20, 4),
) -> plt.Figure:
    """Per-session syllable tuning correlation matrices.

    One heatmap per session showing Pearson correlation of z-scored tuning
    profiles across units. Each session shows all its syllables.

    Args:
        tuning_dir: Directory with per-session fig4_tuning.parquet files.
        sessions: Session subdirectory names to load. If None, uses defaults.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with 5 subplots (one per session).
    """
    _setup_nature_style()

    if tuning_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        tuning_dir = _root / "results" / "single_fm" / "production_20260220_2019" / "fig4_all4"

    if sessions is None:
        sessions = list(_TGM_SESSION_DIRS.values())

    # Load per-session data
    session_corrs: List[tuple] = []  # (label, motif_ids, corr_matrix)
    for sess_name in sessions:
        tuning_path = tuning_dir / sess_name / "fig4_tuning.parquet"
        if not tuning_path.exists():
            continue
        df = pd.read_parquet(tuning_path)
        z_cols = sorted([c for c in df.columns if c.endswith("_z") and c.startswith("motif_")])
        if not z_cols:
            continue

        motif_ids = [int(c.split("_")[1]) for c in z_cols]
        z_matrix = df[z_cols].to_numpy()
        corr = np.corrcoef(z_matrix.T)

        # Sort alphabetically by syllable name
        names = [SYLLABLE_NAMES.get(m, str(m)) for m in motif_ids]
        alpha_order = np.argsort(names)
        motif_ids = [motif_ids[i] for i in alpha_order]
        corr = corr[np.ix_(alpha_order, alpha_order)]

        label = _SESSION_LABELS.get(sess_name, sess_name)
        session_corrs.append((label, motif_ids, corr))

    n_sess = len(session_corrs)
    if n_sess == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No tuning data found", transform=ax.transAxes, ha="center")
        return fig

    fig, axes = plt.subplots(1, n_sess, figsize=figsize)
    if n_sess == 1:
        axes = [axes]

    # Shared color scale across all sessions
    all_off_diag = []
    for _, mids, corr in session_corrs:
        n = corr.shape[0]
        all_off_diag.extend(corr[~np.eye(n, dtype=bool)].tolist())
    vmax = max(abs(min(all_off_diag)), abs(max(all_off_diag)), 0.1)

    for ax, (label, motif_ids, corr) in zip(axes, session_corrs):
        n = corr.shape[0]
        im = ax.imshow(corr, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)

        syl_labels = [SYLLABLE_NAMES.get(m, str(m)) for m in motif_ids]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(syl_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(syl_labels, fontsize=7)
        ax.set_title(label, fontweight="bold", fontsize=11)

    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Pearson r", pad=0.02)
    fig.suptitle("Syllable tuning similarity (per session)", fontsize=14, fontweight="bold")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════

def load_residualized_decoding_results(
    results_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load residualized decoding results from all session subdirs.

    Args:
        results_dir: Directory containing per-session subdirs with results.json.

    Returns:
        List of per-session result dicts, sorted by session_id.
    """
    if results_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        results_dir = _root / "results" / "single_fm" / "fig4_residualized_decoding"

    if not results_dir.exists():
        return []

    results: List[Dict[str, Any]] = []
    for sess_dir in sorted(results_dir.iterdir()):
        rpath = sess_dir / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                results.append(json.load(f))

    return results


def load_motif_encoding_summary(
    motif_encoding_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load 5-way encoding summary with motif unique variance.

    Args:
        motif_encoding_dir: Directory with summary/fig2b_summary.parquet.

    Returns:
        Summary DataFrame, or empty DataFrame if not found.
    """
    if motif_encoding_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        motif_encoding_dir = _root / "results" / "single_fm" / "fig4_motif_encoding"

    summary_path = motif_encoding_dir / "summary" / "fig2b_summary.parquet"
    if not summary_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(summary_path)
    if "session_id" in df.columns:
        df["subject"] = df["session_id"].map(SUBJECT_MAP)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Orchestration wrappers (for generate_publication_figures.py)
# ══════════════════════════════════════════════════════════════════════════

def generate_fig4_pooled_panels(
    fig4_data: Dict[str, Any],
    out_dir: Path,
    similarity_per_session: Optional[List[Dict[str, Any]]] = None,
    bank_rsa_results: Optional[List[Dict[str, Any]]] = None,
    resid_results_dir: Optional[Path] = None,
    motif_encoding_dir: Optional[Path] = None,
) -> None:
    """Generate all 4 pooled Fig 4 panels and save.

    Args:
        fig4_data: Dict from load_fig4() with peth, lag_sweep, confusion, etc.
        out_dir: Output directory for saved figures.
        similarity_per_session: Pose similarity RSA results (for 4D fallback).
        bank_rsa_results: Bank-level transition RSA results (for 4D preferred).
        resid_results_dir: Directory with residualized decoding results (4B).
        motif_encoding_dir: Directory with 5-way motif encoding results (4C).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(fig4_data.get("cluster_orders", {}).keys())

    # ── Panel 4A: Pooled PETH + Lag sweep ──
    print("  4A: Pooled PETH + lag sweep")
    fig = plot_fig4a_pooled_peth_and_lag(
        fig4_data["peth"], fig4_data["lag_sweep"], sessions=sessions,
    )
    save_fig(fig, "fig4A_pooled_peth_lag", out_dir)

    # ── Panel 4B: Residualized vs raw decoding ──
    print("  4B: Residualized vs raw decoding")
    resid_results = load_residualized_decoding_results(resid_results_dir)
    if resid_results:
        fig = plot_fig4b_residualized_decoding(resid_results)
        save_fig(fig, "fig4B_residualized_decoding", out_dir)
    else:
        print("    [SKIP] Run run_fig4_residualized_decoding.py first")

    # ── Panel 4C: Unique motif variance ──
    print("  4C: Unique motif variance (5-way)")
    motif_df = load_motif_encoding_summary(motif_encoding_dir)
    if not motif_df.empty and "unique_motif_shuffle" in motif_df.columns:
        fig = plot_fig4c_unique_motif_variance(motif_df)
        save_fig(fig, "fig4C_unique_motif_variance", out_dir)
    else:
        print("    [SKIP] Run run_fig2b_all_sessions.py --include-motif first")

    # ── Panel 4D: Pooled RSA ──
    print("  4D: Pooled RSA")
    fig = plot_fig4d_pooled_rsa(
        similarity_per_session=similarity_per_session or [],
        bank_rsa_results=bank_rsa_results,
        confusion_data=fig4_data.get("confusion"),
        class_labels=fig4_data.get("class_labels"),
    )
    save_fig(fig, "fig4D_pooled_rsa", out_dir)


__all__ = [
    "plot_fig4a_pooled_peth_and_lag",
    "plot_fig4b_tgm",
    "plot_fig4b_residualized_decoding",
    "plot_fig4c_unique_motif_variance",
    "plot_fig4d_pooled_rsa",
    "plot_fig4d_participation",
    "plot_fig4d_participation_per_session",
    "load_tgm_data",
    "load_participation_data",
    "load_residualized_decoding_results",
    "load_motif_encoding_summary",
    "generate_fig4_pooled_panels",
]
