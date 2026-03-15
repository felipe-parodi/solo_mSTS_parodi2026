"""Figure 4 composite: 6-panel assembly for Nature Communications.

Assembles 6 panels (A-F) into a single publication-ready figure
using nested subgridspecs (outer 2x1 for rows, inner 1x3 per row).

Layout::

    Row 0: A (PETH ±2s)     B (TGM heatmap)      C (TGM diagonal)
    Row 1: D (Mantel scatter) E (tuning sim.)      F (selectivity)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats

from freemat.single_fm.viz.pub_data import (
    ALL_SESSIONS,
    SUBJECT_COLORS,
    SUBJECT_MAP,
    setup_nature_style,
)

# ── Shared constants ──────────────────────────────────────────────────────
POOLED_COLOR = "#2D7D9A"
SESSION_ALPHA = 0.35

_SUBJ_LABEL: Dict[str, str] = {"M2": "M2", "M1": "M1"}

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


# ── Helpers ───────────────────────────────────────────────────────────────

def _thin_yticks(ax: plt.Axes, nbins: int = 4) -> None:
    """Reduce y-axis ticks to approximately *nbins* for visual clarity."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=False))


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    x_offset: float = -0.01,
    y_pad: float = 0.005,
) -> None:
    """Add bold uppercase panel label (A, B, ...) at top-left of axes.

    Uses the actual axes position (after ``fig.canvas.draw()``) so that
    labels track the constrained box, not the gridspec cell.
    """
    fig = ax.get_figure()
    pos = ax.get_position()
    fig_x = pos.x0 + x_offset
    fig_y = pos.y1 + y_pad
    fig.text(
        fig_x, fig_y, label,
        fontsize=16, fontweight="bold", va="bottom", ha="right",
    )


def _align_row_titles(row_axes: List[plt.Axes], fig: plt.Figure) -> None:
    """Align titles in a row to the highest axes top in figure coords.

    After ``set_box_aspect`` constrains non-square panels, their tops sit
    lower than square panels in the same row.  This helper adjusts each
    title's y-position so all titles in the row land at the same height.

    Args:
        row_axes: Axes objects belonging to one gridspec row.
        fig: Parent figure (used for coordinate computation).
    """
    fig.canvas.draw()
    tops = [ax.get_position().y1 for ax in row_axes]
    max_top = max(tops)
    for ax in row_axes:
        pos = ax.get_position()
        ax_height = pos.y1 - pos.y0
        if ax_height > 0:
            title_y = 1.0 + (max_top - pos.y1) / ax_height
            ax.title.set_position((0.5, title_y))


def _p_to_stars(p: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p: P-value.

    Returns:
        Asterisk string (``***``, ``**``, ``*``, or empty).
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ══════════════════════════════════════════════════════════════════════════
# Panel A: PETH heatmap
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_a_peth(
    ax: plt.Axes,
    peth_data: Dict[str, Any],
    sessions: List[str],
    fig: plt.Figure,
) -> None:
    """Draw pooled onset-aligned PETH heatmap (all units, all sessions).

    Units are z-scored per row and sorted by peak latency. The heatmap
    uses bilinear interpolation with an RdBu_r colormap clipped at
    +/-2 z-scores.

    Args:
        ax: Target axes.
        peth_data: Dict keyed by session_key, each containing
            ``peth_fine`` (n_units x n_bins ndarray), ``tau_bins_fine``,
            and ``bin_s_fine``.
        sessions: Ordered list of session keys to include.
        fig: Parent figure (for colorbar attachment).
    """
    all_peth_z: List[np.ndarray] = []
    for sess_key in sessions:
        sess = peth_data.get(sess_key, {})
        peth = sess.get("peth_fine")
        if peth is None:
            continue
        row_mean = peth.mean(axis=1, keepdims=True)
        row_std = peth.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        peth_z = (peth - row_mean) / row_std
        all_peth_z.append(peth_z)

    if not all_peth_z:
        ax.text(
            0.5, 0.5, "No PETH data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    pooled_peth = np.vstack(all_peth_z)

    ref_sess = peth_data[sessions[0]]
    tau_bins = ref_sess["tau_bins_fine"]
    bin_s = ref_sess.get("bin_s_fine", 0.05)
    time_axis = tau_bins * bin_s

    # Restrict to ±2s window
    win_mask = np.abs(time_axis) <= 2.0
    pooled_peth = pooled_peth[:, win_mask]
    time_axis = time_axis[win_mask]

    n_units = pooled_peth.shape[0]

    peak_idx = np.argmax(pooled_peth, axis=1)
    sort_order = np.argsort(peak_idx)
    peth_sorted = pooled_peth[sort_order]

    im = ax.imshow(
        peth_sorted, aspect="auto", cmap="RdBu_r",
        vmin=-2, vmax=2,
        extent=[time_axis[0], time_axis[-1], n_units - 0.5, -0.5],
        interpolation="bilinear",
    )
    ax.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"Units (n={n_units})", fontsize=8)
    ax.set_title("Pooled PETH", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    cb.set_label("Z-scored FR", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════
# Panel B: Lagged decoding curve
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_b_lag(
    ax: plt.Axes,
    lag_sweep_df: pd.DataFrame,
    sessions: List[str],
) -> None:
    """Draw lagged motif decoding curve with per-session traces.

    Args:
        ax: Target axes.
        lag_sweep_df: DataFrame with columns ``session_id``, ``bank``,
            ``lag_sec``, ``accuracy``, and optionally ``n_classes``.
        sessions: Ordered list of session keys.
    """
    if lag_sweep_df.empty:
        ax.text(
            0.5, 0.5, "No lag sweep data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    pooled_lag = lag_sweep_df.groupby(
        ["session_id", "lag_sec"], as_index=False,
    )["accuracy"].mean()

    for sess_key in sessions:
        sess_lag = pooled_lag[pooled_lag["session_id"] == sess_key]
        if sess_lag.empty:
            continue
        subject = SUBJECT_MAP.get(sess_key, sess_key)
        color = SUBJECT_COLORS.get(subject, "#999999")
        ax.plot(
            sess_lag["lag_sec"], sess_lag["accuracy"] * 100,
            color=color, alpha=SESSION_ALPHA, linewidth=1.0,
        )

    mean_lag = pooled_lag.groupby("lag_sec")["accuracy"].agg(["mean", "sem"])
    ax.plot(
        mean_lag.index, mean_lag["mean"] * 100,
        color=POOLED_COLOR, linewidth=2.5, label="Mean",
    )
    ax.fill_between(
        mean_lag.index,
        (mean_lag["mean"] - mean_lag["sem"]) * 100,
        (mean_lag["mean"] + mean_lag["sem"]) * 100,
        color=POOLED_COLOR, alpha=0.2,
    )

    if "n_classes" in lag_sweep_df.columns:
        chance = 100.0 / lag_sweep_df["n_classes"].median()
    else:
        chance = 10.0
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98, chance + 0.5, "Chance",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=6, color="gray",
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Lag (s)", fontsize=8)
    ax.set_ylabel("Balanced acc. (%)", fontsize=8)
    ax.set_title("Lagged decoding", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper right")
    _thin_yticks(ax)


# ══════════════════════════════════════════════════════════════════════════
# Panel C: TGM heatmap (exemplar session)
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_c_tgm(
    ax: plt.Axes,
    tgm_data: Dict[str, Dict[str, np.ndarray]],
    exemplar_session: str,
    fig: plt.Figure,
) -> None:
    """Draw temporal generalization matrix heatmap for one exemplar session.

    Args:
        ax: Target axes.
        tgm_data: Dict keyed by session_key, each containing ``matrix``
            (2D ndarray), ``chance_level`` (float), and ``tau_sec``
            (1D ndarray of time points).
        exemplar_session: Session key for heatmap display.
        fig: Parent figure (for colorbar).
    """
    if not tgm_data:
        ax.text(
            0.5, 0.5, "No TGM data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    if exemplar_session not in tgm_data:
        exemplar_session = sorted(tgm_data.keys())[0]

    exemplar = tgm_data[exemplar_session]
    matrix = exemplar["matrix"]
    chance = float(exemplar["chance_level"])
    tau_sec = exemplar["tau_sec"]

    subject = SUBJECT_MAP.get(exemplar_session, exemplar_session)
    subj_label = _SUBJ_LABEL.get(subject, subject)

    extent = [tau_sec[0], tau_sec[-1], tau_sec[-1], tau_sec[0]]

    im = ax.imshow(
        matrix, aspect="equal", cmap="RdYlBu_r",
        vmin=chance - 0.02, vmax=np.max(matrix) + 0.01,
        extent=extent, interpolation="nearest", origin="upper",
    )

    ax.axhline(0, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(0, color="white", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.plot(
        [tau_sec[0], tau_sec[-1]], [tau_sec[0], tau_sec[-1]],
        color="white", linewidth=0.8, linestyle=":", alpha=0.5,
    )

    ax.set_xlabel("Test time (s)", fontsize=8)
    ax.set_ylabel("Train time (s)", fontsize=8)
    ax.set_title(f"TGM ({subj_label})", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    cb.set_label("Bal. acc.", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════
# Panel D: TGM diagonal traces (all sessions)
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_d_diagonal(
    ax: plt.Axes,
    tgm_data: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Draw TGM diagonal accuracy traces for all sessions.

    Args:
        ax: Target axes.
        tgm_data: Dict keyed by session_key, each containing
            ``diagonal`` (1D ndarray), ``chance_level`` (float), and
            ``tau_sec`` (1D ndarray).
    """
    if not tgm_data:
        ax.text(
            0.5, 0.5, "No TGM data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    chance_vals: List[float] = []
    tau_sec = None

    for sess_key in sorted(tgm_data.keys()):
        d = tgm_data[sess_key]
        chance_vals.append(float(d["chance_level"]))
        if tau_sec is None:
            tau_sec = d["tau_sec"]

    mean_chance = np.mean(chance_vals)

    for sess_key in sorted(tgm_data.keys()):
        d = tgm_data[sess_key]
        subject = SUBJECT_MAP.get(sess_key, sess_key)
        color = SUBJECT_COLORS.get(subject, "#999999")
        ax.plot(
            tau_sec, d["diagonal"] * 100,
            color=color, alpha=SESSION_ALPHA, linewidth=1.0,
        )

    diag_stack = np.array([d["diagonal"] for d in tgm_data.values()])
    mean_diag = diag_stack.mean(axis=0)
    sem_diag = diag_stack.std(axis=0) / np.sqrt(len(diag_stack))

    ax.plot(
        tau_sec, mean_diag * 100,
        color=POOLED_COLOR, linewidth=2.5, label="Mean",
    )
    ax.fill_between(
        tau_sec,
        (mean_diag - sem_diag) * 100,
        (mean_diag + sem_diag) * 100,
        color=POOLED_COLOR, alpha=0.2,
    )

    ax.axhline(
        mean_chance * 100, color="gray", linestyle=":", linewidth=1, alpha=0.6,
    )
    ax.text(
        0.02, mean_chance * 100 + 0.5, "Chance",
        transform=ax.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=6, color="gray",
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Balanced acc. (%)", fontsize=8)
    ax.set_title("Diagonal decoding", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

    handles = []
    for subj, label in [("M2", "M2"), ("M1", "M1")]:
        handles.append(Line2D(
            [0], [0], color=SUBJECT_COLORS[subj], linewidth=1.5,
            alpha=0.6, label=label,
        ))
    handles.append(Line2D(
        [0], [0], color=POOLED_COLOR, linewidth=2.5, label="Mean",
    ))
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    _thin_yticks(ax)


# ══════════════════════════════════════════════════════════════════════════
# Panel E: Behavioral transition probability matrix
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_e_transitions(
    ax: plt.Axes,
    transition_data: Dict[str, Dict[str, np.ndarray]],
    fig: plt.Figure,
) -> None:
    """Draw session-averaged behavioral transition probability heatmap.

    Transition matrices are aligned on shared syllable labels across
    sessions, averaged, and symmetrized for display.

    Args:
        ax: Target axes.
        transition_data: Dict keyed by session_key, each containing
            ``matrix`` (N x N ndarray) and ``labels`` (N-element array).
        fig: Parent figure (for colorbar).
    """
    if not transition_data:
        ax.text(
            0.5, 0.5, "No transition data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    # Find union of all labels across sessions
    all_labels_set: set = set()
    for d in transition_data.values():
        all_labels_set.update(int(lbl) for lbl in d["labels"])
    all_labels = sorted(all_labels_set)
    n = len(all_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    # Accumulate and average
    mat_sum = np.zeros((n, n), dtype=np.float64)
    mat_count = np.zeros((n, n), dtype=np.int32)

    for d in transition_data.values():
        sess_labels = [int(lbl) for lbl in d["labels"]]
        mat = d["matrix"]
        for i_local, li in enumerate(sess_labels):
            for j_local, lj in enumerate(sess_labels):
                gi, gj = label_to_idx[li], label_to_idx[lj]
                mat_sum[gi, gj] += mat[i_local, j_local]
                mat_count[gi, gj] += 1

    mat_count[mat_count == 0] = 1
    avg_trans = mat_sum / mat_count

    # Symmetrize for display
    avg_sym = 0.5 * (avg_trans + avg_trans.T)

    im = ax.imshow(avg_sym, cmap="viridis", aspect="equal", vmin=0)

    syl_labels = [SYLLABLE_NAMES.get(m, str(m)) for m in all_labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(syl_labels, fontsize=5, rotation=45, ha="right")
    ax.set_yticklabels(syl_labels, fontsize=5)
    ax.set_title("Transition prob.", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    cb.set_label("P(transition)", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════
# Panel F: Mantel scatter (transition prob vs neural confusion)
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_f_mantel_scatter(
    ax: plt.Axes,
    transition_data: Dict[str, Dict[str, np.ndarray]],
    confusion_data: Dict[str, Dict[str, np.ndarray]],
    class_labels: Dict[str, Dict[str, np.ndarray]],
    bank_rsa_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Scatter of transition prob vs neural confusion at syllable-pair level.

    Each dot is one off-diagonal syllable pair (i < j), colored by
    session. A pooled regression line and Mantel rho range annotation
    are shown.

    Args:
        ax: Target axes.
        transition_data: Dict keyed by session_key with ``matrix`` and
            ``labels`` arrays.
        confusion_data: Dict[session][bank] -> confusion matrix ndarray.
        class_labels: Dict[session][bank] -> label array.
        bank_rsa_results: Optional list of bank-level RSA dicts for rho
            annotation.
    """
    if not transition_data or not confusion_data:
        ax.text(
            0.5, 0.5, "No data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    all_trans_vals: List[float] = []
    all_conf_vals: List[float] = []
    all_colors: List[str] = []
    per_session_rhos: List[float] = []

    for sess_key in sorted(transition_data.keys()):
        if sess_key not in confusion_data:
            continue

        t_data = transition_data[sess_key]
        t_mat = t_data["matrix"].astype(float)
        t_labels = [int(lbl) for lbl in t_data["labels"]]

        # Get pooled confusion matrix
        bank_key = (
            "pooled" if "pooled" in confusion_data[sess_key]
            else "ventral"
        )
        c_mat = confusion_data[sess_key][bank_key].astype(float)
        c_labels = [int(lbl) for lbl in class_labels[sess_key][bank_key]]

        # Find shared labels
        shared = sorted(set(t_labels) & set(c_labels))
        if len(shared) < 3:
            continue

        # Align both matrices to shared labels
        t_idx = [t_labels.index(lbl) for lbl in shared]
        c_idx = [c_labels.index(lbl) for lbl in shared]

        t_aligned = t_mat[np.ix_(t_idx, t_idx)]
        c_aligned = c_mat[np.ix_(c_idx, c_idx)]

        # Symmetrize
        t_sym = 0.5 * (t_aligned + t_aligned.T)
        c_sym = 0.5 * (c_aligned + c_aligned.T)

        # Row-normalize confusion
        row_sums = c_sym.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        c_norm = c_sym / row_sums

        # Extract upper triangle (excluding diagonal)
        n_shared = len(shared)
        tri_idx = np.triu_indices(n_shared, k=1)
        t_vec = t_sym[tri_idx]
        c_vec = c_norm[tri_idx]

        # Filter out pairs with zero transition probability (uninformative)
        nonzero_mask = t_vec > 0
        t_vec = t_vec[nonzero_mask]
        c_vec = c_vec[nonzero_mask]

        subject = SUBJECT_MAP.get(sess_key, sess_key)
        color = SUBJECT_COLORS.get(subject, "#999999")

        all_trans_vals.extend(t_vec.tolist())
        all_conf_vals.extend(c_vec.tolist())
        all_colors.extend([color] * len(t_vec))

        # Per-session Spearman rho
        if len(t_vec) > 2:
            rho, _ = stats.spearmanr(t_vec, c_vec)
            per_session_rhos.append(rho)

    if not all_trans_vals:
        ax.text(
            0.5, 0.5, "No paired data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    trans_arr = np.array(all_trans_vals)
    conf_arr = np.array(all_conf_vals)

    ax.scatter(
        trans_arr, conf_arr, s=15, alpha=0.5,
        c=all_colors, edgecolors="white", linewidth=0.3,
    )

    # Pooled regression line
    if len(trans_arr) > 2:
        z = np.polyfit(trans_arr, conf_arr, 1)
        x_line = np.linspace(trans_arr.min(), trans_arr.max(), 100)
        ax.plot(
            x_line, np.polyval(z, x_line),
            color="#333333", linewidth=1.5, linestyle="--", alpha=0.8,
        )

    # Annotate with rho range
    if bank_rsa_results:
        rho_vals = []
        for r in bank_rsa_results:
            v_rho = r.get("ventral_trans_rho", np.nan)
            d_rho = r.get("dorsal_trans_rho", np.nan)
            rho_vals.append(float(np.nanmean([v_rho, d_rho])))
        rho_min = min(rho_vals)
        rho_max = max(rho_vals)
        rho_str = f"rho = {rho_min:.2f}-{rho_max:.2f}"
    elif per_session_rhos:
        rho_str = f"rho = {min(per_session_rhos):.2f}-{max(per_session_rhos):.2f}"
    else:
        rho_str = ""

    if rho_str:
        ax.text(
            0.03, 0.97, rho_str,
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                  "alpha": 0.8, "edgecolor": "gray", "linewidth": 0.5},
        )

    # Subject legend
    handles = []
    for subj, label in [("M2", "M2"), ("M1", "M1")]:
        handles.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=SUBJECT_COLORS[subj],
            markersize=5, label=label, linewidth=0,
        ))
    ax.legend(handles=handles, loc="lower right", fontsize=6)

    ax.set_xlabel("Transition prob.", fontsize=8)
    ax.set_ylabel("Neural confusion", fontsize=8)
    ax.set_title("Confusion ~ transitions", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)
    _thin_yticks(ax)


# ══════════════════════════════════════════════════════════════════════════
# Panel G: Tuning correlation matrix
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_g_tuning(
    ax: plt.Axes,
    tuning_corr: np.ndarray,
    tuning_motif_ids: List[int],
    fig: plt.Figure,
) -> None:
    """Draw motif-by-motif mean tuning correlation matrix.

    Args:
        ax: Target axes.
        tuning_corr: Pre-computed mean correlation matrix.
        tuning_motif_ids: Ordered motif IDs for rows/columns.
        fig: Parent figure (for colorbar).
    """
    n_motifs = tuning_corr.shape[0]

    off_diag = tuning_corr[~np.eye(n_motifs, dtype=bool)]
    vmax = max(abs(np.min(off_diag)), abs(np.max(off_diag)), 0.1)

    im = ax.imshow(
        tuning_corr, cmap="RdBu_r", aspect="equal",
        vmin=-vmax, vmax=vmax,
    )

    syl_labels = [
        SYLLABLE_NAMES.get(m, str(m)) for m in tuning_motif_ids
    ]
    ax.set_xticks(range(n_motifs))
    ax.set_yticks(range(n_motifs))
    ax.set_xticklabels(syl_labels, fontsize=5, rotation=45, ha="right")
    ax.set_yticklabels(syl_labels, fontsize=5)
    ax.set_title("Tuning similarity", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    cb.set_label("Pearson r", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════
# Panel H: Selectivity distribution
# ══════════════════════════════════════════════════════════════════════════

def _draw_panel_h_selectivity(
    ax: plt.Axes,
    selectivity_arr: np.ndarray,
    selectivity_subjects: np.ndarray,
) -> None:
    """Draw per-subject selectivity (max |z-score|) histograms.

    Args:
        ax: Target axes.
        selectivity_arr: 1D array of per-unit max |z-score| values.
        selectivity_subjects: 1D array of subject names.
    """
    if selectivity_arr is None or len(selectivity_arr) == 0:
        ax.text(
            0.5, 0.5, "No selectivity data",
            transform=ax.transAxes, ha="center", color="gray",
        )
        return

    bins = np.linspace(0, min(float(selectivity_arr.max()), 8), 30)

    for subj in ["M2", "M1"]:
        mask = selectivity_subjects == subj
        if mask.any():
            ax.hist(
                selectivity_arr[mask], bins=bins, alpha=0.5,
                color=SUBJECT_COLORS[subj],
                label=_SUBJ_LABEL[subj],
                density=True, edgecolor="white",
            )

    ax.hist(
        selectivity_arr, bins=bins, alpha=0.0, edgecolor=POOLED_COLOR,
        linewidth=2.0, density=True, histtype="step", label="All",
    )

    ax.set_xlabel("Max |z-score|", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("Selectivity", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)
    _thin_yticks(ax)


# ══════════════════════════════════════════════════════════════════════════
# Main composite assembly
# ══════════════════════════════════════════════════════════════════════════

def fig4_composite(
    peth_data: Dict[str, Any],
    tgm_data: Dict[str, Dict[str, np.ndarray]],
    transition_data: Dict[str, Dict[str, np.ndarray]],
    confusion_data: Dict[str, Dict[str, np.ndarray]],
    class_labels: Dict[str, Dict[str, np.ndarray]],
    bank_rsa_results: Optional[List[Dict[str, Any]]] = None,
    tuning_corr: Optional[np.ndarray] = None,
    tuning_motif_ids: Optional[List[int]] = None,
    selectivity_arr: Optional[np.ndarray] = None,
    selectivity_subjects: Optional[np.ndarray] = None,
    sessions: Optional[List[str]] = None,
    exemplar_tgm_session: str = "M2_0605",
    figsize: tuple = (7.1, 5.5),
) -> plt.Figure:
    """Assemble Figure 4 as a 6-panel (A-F) composite in 2x3 layout.

    Layout::

        Row 0: A (PETH ±2s)      B (TGM heatmap)      C (TGM diagonal)
        Row 1: D (Mantel scatter) E (tuning sim.)      F (selectivity)

    Args:
        peth_data: Dict keyed by session_key with ``peth_fine``,
            ``tau_bins_fine``, ``bin_s_fine`` arrays.
        tgm_data: Dict keyed by session_key with ``matrix``,
            ``chance_level``, ``tau_sec``, ``diagonal`` arrays.
        transition_data: Dict keyed by session_key with ``matrix``
            (N x N ndarray) and ``labels`` (N-element array).
        confusion_data: Dict[session][bank] -> confusion matrix ndarray.
        class_labels: Dict[session][bank] -> label array.
        bank_rsa_results: Optional list of bank-level transition RSA
            dicts for Mantel rho annotation.
        tuning_corr: Pre-computed mean correlation matrix.
        tuning_motif_ids: Ordered motif IDs for tuning matrix.
        selectivity_arr: 1D array of per-unit max |z-score| values.
        selectivity_subjects: 1D subject name array.
        sessions: Session keys to include.
        exemplar_tgm_session: Session key for TGM heatmap exemplar.
        figsize: Figure size in inches.

    Returns:
        Assembled Figure with 6 panels (A through F).
    """
    setup_nature_style()

    if sessions is None:
        sessions = [s for s in ALL_SESSIONS if s in peth_data]

    fig = plt.figure(figsize=figsize)

    # Outer grid: 2 rows
    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 1],
        hspace=0.02,
        left=0.08, right=0.93, top=0.97, bottom=0.05,
    )

    # ═══ Row 0: A (PETH ±2s), B (TGM heatmap), C (TGM diagonal) ═══
    row0 = outer[0].subgridspec(1, 3, wspace=0.55)

    ax_a = fig.add_subplot(row0[0, 0])
    ax_a.set_box_aspect(0.85)
    _draw_panel_a_peth(ax_a, peth_data, sessions, fig)

    ax_b = fig.add_subplot(row0[0, 1])
    ax_b.set_box_aspect(0.85)
    _draw_panel_c_tgm(ax_b, tgm_data, exemplar_tgm_session, fig)

    ax_c = fig.add_subplot(row0[0, 2])
    ax_c.set_box_aspect(0.75)
    _draw_panel_d_diagonal(ax_c, tgm_data)

    # ═══ Row 1: D (Mantel scatter), E (tuning sim.), F (selectivity) ═══
    row1 = outer[1].subgridspec(1, 3, wspace=0.55)

    ax_d = fig.add_subplot(row1[0, 0])
    ax_d.set_box_aspect(0.75)
    _draw_panel_f_mantel_scatter(
        ax_d, transition_data, confusion_data, class_labels, bank_rsa_results,
    )

    ax_e = fig.add_subplot(row1[0, 1])
    ax_e.set_box_aspect(0.85)
    if tuning_corr is not None and tuning_motif_ids is not None:
        _draw_panel_g_tuning(ax_e, tuning_corr, tuning_motif_ids, fig)
    else:
        ax_e.text(
            0.5, 0.5, "No tuning data",
            transform=ax_e.transAxes, ha="center", color="gray",
        )
        ax_e.set_title("Tuning similarity", fontsize=9, fontweight="bold")

    ax_f = fig.add_subplot(row1[0, 2])
    ax_f.set_box_aspect(0.75)
    if selectivity_arr is not None and selectivity_subjects is not None:
        _draw_panel_h_selectivity(ax_f, selectivity_arr, selectivity_subjects)
    else:
        ax_f.text(
            0.5, 0.5, "No selectivity data",
            transform=ax_f.transAxes, ha="center", color="gray",
        )
        ax_f.set_title("Selectivity", fontsize=9, fontweight="bold")

    # Thin ticks on dense heatmaps (E = tuning similarity, 12 syllable labels)
    fig.canvas.draw()
    for i, label in enumerate(ax_e.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)
    for i, label in enumerate(ax_e.get_yticklabels()):
        if i % 2 != 0:
            label.set_visible(False)

    # Finalize constrained positions, then place labels + align titles
    fig.canvas.draw()

    _add_panel_label(ax_a, "A", x_offset=-0.04)
    _add_panel_label(ax_b, "B")
    _add_panel_label(ax_c, "C")
    _add_panel_label(ax_d, "D", x_offset=-0.04)
    _add_panel_label(ax_e, "E")
    _add_panel_label(ax_f, "F")

    _align_row_titles([ax_a, ax_b, ax_c], fig)
    _align_row_titles([ax_d, ax_e, ax_f], fig)

    return fig


__all__ = [
    "fig4_composite",
    "SYLLABLE_NAMES",
]
