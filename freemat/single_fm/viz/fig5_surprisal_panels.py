"""Figure 5 follow-up: Tier 1 surprisal dashboard panels.

Ten diagnostic panels for the transition-level surprisal analysis:
A — Transition count heatmap
D — Markov vs full-model NLL bars
F — Surprisal histogram/density
G — Surprisal by motif pair heatmap
L — Time-resolved surprisal beta
M — Pre vs post beta comparison
N — Unit-level beta histogram
U — Null histogram + observed line
T — Matched high vs low surprisal
V1 — Beta vs motif selectivity scatter
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freemat.single_fm.viz.pub_data import (
    MOTIF_NAMES_FIG5,
    SUBJECT_COLORS,
    SUBJECT_MAP,
    save_fig,
)

# ── Style ─────────────────────────────────────────────────────────────────


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


def _motif_name(m: int) -> str:
    """Short motif label for axis ticks."""
    return MOTIF_NAMES_FIG5.get(m, f"M{m}")


# ══════════════════════════════════════════════════════════════════════════
# Panel A: Transition count heatmap
# ══════════════════════════════════════════════════════════════════════════


def plot_transition_count_heatmap(
    current_motif: np.ndarray,
    next_motif: np.ndarray,
    retained_motifs: List[int],
    session_id: str,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Heatmap of transition counts (current x next).

    Args:
        current_motif: Current motif per transition.
        next_motif: Next motif per transition.
        retained_motifs: Motif IDs kept in analysis.
        session_id: Session identifier for title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    motifs = sorted(retained_motifs)
    K = len(motifs)
    m2i = {m: i for i, m in enumerate(motifs)}

    counts = np.zeros((K, K), dtype=int)
    for cur, nxt in zip(current_motif, next_motif):
        ci, ni = m2i.get(int(cur)), m2i.get(int(nxt))
        if ci is not None and ni is not None:
            counts[ci, ni] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(counts, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels([_motif_name(m) for m in motifs], rotation=45, ha="right")
    ax.set_yticks(range(K))
    ax.set_yticklabels([_motif_name(m) for m in motifs])
    ax.set_xlabel("Next motif")
    ax.set_ylabel("Current motif")
    ax.set_title(f"Transition counts — {session_id}")

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = counts[i, j]
            if val > 0:
                color = "white" if val > counts.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Count", shrink=0.8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel D: Markov vs full-model NLL bars
# ══════════════════════════════════════════════════════════════════════════


def plot_model_nll_comparison(
    session_summaries: List[Dict[str, Any]],
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Bar chart comparing Markov vs full-model NLL per session.

    Args:
        session_summaries: List of session summary dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    completed = [s for s in session_summaries if s.get("status") == "complete"]
    n = len(completed)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No completed sessions", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    labels = [s["session_id"] for s in completed]
    nll_markov = [s["nll_markov"] for s in completed]
    nll_full = [s["nll_full"] for s in completed]

    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, nll_markov, w, label="Markov", color="#D4A373", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, nll_full, w, label="Full model", color="#2D7D9A", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean NLL")
    ax.set_title("Behavior model vs Markov baseline")
    ax.legend(frameon=False)

    # Mark sessions where full beats Markov
    for i, s in enumerate(completed):
        if s["model_beats_markov"]:
            ax.text(i, max(nll_markov[i], nll_full[i]) + 0.02, "*",
                    ha="center", fontsize=14, fontweight="bold", color="#2D7D9A")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel F: Surprisal histogram/density
# ══════════════════════════════════════════════════════════════════════════


def plot_surprisal_histogram(
    surprisal: np.ndarray,
    z_surprisal: np.ndarray,
    session_id: str,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Side-by-side histogram of raw and z-scored surprisal.

    Args:
        surprisal: Raw surprisal values.
        z_surprisal: Z-scored surprisal.
        session_id: Session identifier.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(surprisal, bins=30, color="#2D7D9A", edgecolor="white", alpha=0.85)
    ax1.axvline(np.median(surprisal), color="red", ls="--", lw=1.5, label=f"median={np.median(surprisal):.2f}")
    ax1.set_xlabel("Surprisal (-log p)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Raw surprisal — {session_id}")
    ax1.legend(frameon=False)

    ax2.hist(z_surprisal, bins=30, color="#7B2D8E", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Z-scored surprisal")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Z-scored surprisal — {session_id}")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel G: Surprisal by motif pair heatmap
# ══════════════════════════════════════════════════════════════════════════


def plot_surprisal_by_pair_heatmap(
    z_surprisal: np.ndarray,
    current_motif: np.ndarray,
    next_motif: np.ndarray,
    retained_motifs: List[int],
    session_id: str,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Heatmap of mean z-scored surprisal per (current, next) pair.

    Args:
        z_surprisal: Z-scored surprisal per transition.
        current_motif: Current motif per transition.
        next_motif: Next motif per transition.
        retained_motifs: Motif IDs kept.
        session_id: Session identifier.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    motifs = sorted(retained_motifs)
    K = len(motifs)
    mean_surp = np.full((K, K), np.nan)
    for i_m, m_cur in enumerate(motifs):
        for j_m, m_nxt in enumerate(motifs):
            mask = (current_motif == m_cur) & (next_motif == m_nxt)
            if mask.sum() > 0:
                mean_surp[i_m, j_m] = np.mean(z_surprisal[mask])

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.nanmax(np.abs(mean_surp[np.isfinite(mean_surp)])) if np.any(np.isfinite(mean_surp)) else 1.0
    im = ax.imshow(mean_surp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(K))
    ax.set_xticklabels([_motif_name(m) for m in motifs], rotation=45, ha="right")
    ax.set_yticks(range(K))
    ax.set_yticklabels([_motif_name(m) for m in motifs])
    ax.set_xlabel("Next motif")
    ax.set_ylabel("Current motif")
    ax.set_title(f"Mean z-surprisal by pair — {session_id}")

    # Annotate
    for i in range(K):
        for j in range(K):
            val = mean_surp[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < vmax * 0.5 else "white")

    fig.colorbar(im, ax=ax, label="Mean z-surprisal", shrink=0.8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel L: Time-resolved surprisal beta
# ══════════════════════════════════════════════════════════════════════════


def plot_time_resolved_beta(
    X_spikes_z: np.ndarray,
    trans_bins: np.ndarray,
    z_surprisal: np.ndarray,
    controls_df: pd.DataFrame,
    unit_ids: np.ndarray,
    bin_s: float,
    session_id: str,
    window_bins: int = 8,
    figsize: Tuple[float, float] = (8, 4.5),
) -> plt.Figure:
    """Time-resolved beta(surprisal) curve across peri-transition windows.

    For each time lag relative to transition onset, compute the median
    beta_surprisal across units. If the effect is prediction-error-like,
    the beta should peak post-transition.

    Args:
        X_spikes_z: Z-scored spike matrix (n_bins, n_units).
        trans_bins: Transition onset bin indices.
        z_surprisal: Z-scored surprisal per transition.
        controls_df: Control variables DataFrame.
        unit_ids: Unit identifiers.
        bin_s: Bin size in seconds.
        session_id: Session identifier.
        window_bins: Half-window in bins around transition.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    from freemat.single_fm.encoding.fig5_transition_surprisal import (
        fit_surprisal_regression,
    )

    _setup_nature_style()

    n_bins_total = X_spikes_z.shape[0]
    lags = np.arange(-window_bins, window_bins + 1)
    lag_times = lags * bin_s
    median_betas = np.zeros(len(lags))
    q25_betas = np.zeros(len(lags))
    q75_betas = np.zeros(len(lags))

    for li, lag in enumerate(lags):
        # For each lag, compute single-bin neural response
        target_bins = trans_bins + lag
        valid = (target_bins >= 0) & (target_bins < n_bins_total)
        if valid.sum() < 10:
            median_betas[li] = np.nan
            q25_betas[li] = np.nan
            q75_betas[li] = np.nan
            continue

        fr_at_lag = X_spikes_z[target_bins[valid]]  # (n_valid_trans, n_units)

        # Use the same controls for valid transitions
        controls_valid = controls_df.iloc[np.where(valid)[0]].reset_index(drop=True)
        z_valid = z_surprisal[valid]

        result_df = fit_surprisal_regression(
            fr_at_lag, z_valid, controls_valid, unit_ids,
        )
        betas = result_df["beta_surprisal"].dropna()
        median_betas[li] = betas.median()
        q25_betas[li] = betas.quantile(0.25)
        q75_betas[li] = betas.quantile(0.75)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(lag_times, q25_betas, q75_betas, alpha=0.2, color="#2D7D9A")
    ax.plot(lag_times, median_betas, color="#2D7D9A", lw=2, marker="o", markersize=4)
    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7, label="Transition onset")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Time from transition (s)")
    ax.set_ylabel("Median beta(surprisal)")
    ax.set_title(f"Time-resolved surprisal beta — {session_id}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel M: Pre vs post beta comparison
# ══════════════════════════════════════════════════════════════════════════


def plot_pre_vs_post_beta(
    session_summaries: List[Dict[str, Any]],
    figsize: Tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Paired dot plot: pre- vs post-transition median beta per session.

    Args:
        session_summaries: List of session summary dicts.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    completed = [s for s in session_summaries if s.get("status") == "complete"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, s in enumerate(completed):
        sid = s["session_id"]
        subject = SUBJECT_MAP.get(sid, sid)
        color = SUBJECT_COLORS.get(subject, "#888888")
        pre = s["median_beta_surprisal_pre"]
        post = s["median_beta_surprisal_post"]

        ax.plot([0, 1], [pre, post], color=color, alpha=0.6, lw=1.5)
        ax.scatter([0], [pre], color=color, s=80, zorder=5, edgecolors="black", linewidths=0.5)
        ax.scatter([1], [post], color=color, s=80, zorder=5, edgecolors="black", linewidths=0.5,
                   label=sid)

    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-transition", "Post-transition"])
    ax.set_ylabel("Median beta(surprisal)")
    ax.set_title("Pre vs post timing control")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel N: Unit-level beta histogram
# ══════════════════════════════════════════════════════════════════════════


def plot_unit_beta_histogram(
    unit_results_df: pd.DataFrame,
    session_id: str,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Histogram of per-unit beta_surprisal with significance marking.

    Args:
        unit_results_df: DataFrame with beta_surprisal and p_value columns.
        session_id: Session identifier.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    betas = unit_results_df["beta_surprisal"].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(betas, bins=30, color="#2D7D9A", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvline(betas.median(), color="red", ls="--", lw=1.5,
               label=f"median={betas.median():.4f}")

    n_pos = (betas > 0).sum()
    n_tot = len(betas)
    frac_pos = n_pos / n_tot if n_tot > 0 else 0

    # Mark FDR-significant units
    if "p_value_fdr" in unit_results_df.columns:
        sig = unit_results_df.loc[
            unit_results_df["p_value_fdr"].lt(0.05) & unit_results_df["beta_surprisal"].notna()
        ]
        if len(sig) > 0:
            ax.scatter(sig["beta_surprisal"], np.zeros(len(sig)) + ax.get_ylim()[1] * 0.05,
                       marker="v", color="red", s=30, zorder=5, label=f"FDR sig (n={len(sig)})")

    ax.set_xlabel("Beta(surprisal)")
    ax.set_ylabel("Number of units")
    ax.set_title(f"Unit-level beta distribution — {session_id}\n"
                 f"({n_pos}/{n_tot} positive, {frac_pos:.0%})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel U: Null histogram + observed line
# ══════════════════════════════════════════════════════════════════════════


def plot_null_histogram(
    null_medians: np.ndarray,
    observed_median: float,
    perm_p: float,
    session_id: str,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Histogram of null-distribution median betas with observed value.

    Args:
        null_medians: Null distribution of session-level median betas.
        observed_median: Observed median beta.
        perm_p: Permutation p-value.
        session_id: Session identifier.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(null_medians, bins=30, color="#CCCCCC", edgecolor="white", alpha=0.85,
            label="Null distribution")
    ax.axvline(observed_median, color="red", lw=2, ls="--",
               label=f"Observed = {observed_median:.4f}")
    ax.axvline(0, color="gray", ls=":", lw=0.8)

    ax.set_xlabel("Session median beta(surprisal)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation null — {session_id} (p = {perm_p:.3f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel T: Matched high vs low surprisal
# ══════════════════════════════════════════════════════════════════════════


def plot_matched_surprisal(
    delta_fr_mean: np.ndarray,
    z_surprisal: np.ndarray,
    current_motif: np.ndarray,
    next_motif: np.ndarray,
    post_speed: np.ndarray,
    session_id: str,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Matched comparison of neural response for high vs low surprisal.

    Within each (current_motif, next_motif) pair and speed tercile,
    splits transitions into low vs high surprisal (median split) and
    compares mean delta_fr.

    Args:
        delta_fr_mean: Mean delta_fr across units per transition (n_trans,).
        z_surprisal: Z-scored surprisal per transition.
        current_motif: Current motif per transition.
        next_motif: Next motif per transition.
        post_speed: Post-transition speed per transition.
        session_id: Session identifier.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()

    # Build strata: (current, next, speed_tercile)
    speed_terciles = pd.qcut(post_speed, 3, labels=[0, 1, 2], duplicates="drop").codes

    low_vals, high_vals = [], []
    pairs = list(zip(current_motif, next_motif))
    unique_pairs = set(pairs)

    for cur, nxt in unique_pairs:
        for tercile in range(3):
            mask = (
                (current_motif == cur) &
                (next_motif == nxt) &
                (speed_terciles == tercile)
            )
            if mask.sum() < 4:
                continue
            surp_stratum = z_surprisal[mask]
            dfr_stratum = delta_fr_mean[mask]
            med = np.median(surp_stratum)
            low_mask = surp_stratum <= med
            high_mask = surp_stratum > med

            if low_mask.sum() > 0 and high_mask.sum() > 0:
                low_vals.append(np.mean(dfr_stratum[low_mask]))
                high_vals.append(np.mean(dfr_stratum[high_mask]))

    fig, ax = plt.subplots(figsize=figsize)

    if len(low_vals) > 0:
        low_arr = np.array(low_vals)
        high_arr = np.array(high_vals)

        # Paired dots
        for lo, hi in zip(low_arr, high_arr):
            ax.plot([0, 1], [lo, hi], color="#AAAAAA", alpha=0.4, lw=0.8)

        ax.scatter(np.zeros(len(low_arr)), low_arr, color="#2D7D9A", s=40,
                   alpha=0.7, edgecolors="black", linewidths=0.3, label="Low surprisal")
        ax.scatter(np.ones(len(high_arr)), high_arr, color="#D4483B", s=40,
                   alpha=0.7, edgecolors="black", linewidths=0.3, label="High surprisal")

        # Means
        ax.scatter([0], [low_arr.mean()], color="#2D7D9A", s=150, marker="D",
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.scatter([1], [high_arr.mean()], color="#D4483B", s=150, marker="D",
                   edgecolors="black", linewidths=1.5, zorder=5)

        diff = high_arr.mean() - low_arr.mean()
        ax.set_title(f"Matched surprisal comparison — {session_id}\n"
                     f"{len(low_arr)} strata, diff = {diff:.4f}")
    else:
        ax.text(0.5, 0.5, "Insufficient strata", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"Matched surprisal — {session_id}")

    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low surprisal", "High surprisal"])
    ax.set_ylabel("Mean delta FR (z-scored)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel V1: Beta vs motif selectivity scatter
# ══════════════════════════════════════════════════════════════════════════


def plot_beta_vs_selectivity(
    unit_results_df: pd.DataFrame,
    session_id: str,
    selectivity_col: str = "r2",
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Scatter plot of beta_surprisal vs encoding R-squared.

    Args:
        unit_results_df: DataFrame with beta_surprisal and selectivity metric.
        session_id: Session identifier.
        selectivity_col: Column name for selectivity metric.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    _setup_nature_style()
    df = unit_results_df.dropna(subset=["beta_surprisal", selectivity_col])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[selectivity_col], df["beta_surprisal"],
               s=20, alpha=0.5, color="#2D7D9A", edgecolors="none")
    ax.axhline(0, color="gray", ls=":", lw=0.8)

    # Correlation
    if len(df) > 5:
        from scipy import stats as sp_stats
        rho, p_val = sp_stats.spearmanr(df[selectivity_col], df["beta_surprisal"])
        ax.set_title(f"Beta vs R-squared — {session_id}\n"
                     f"Spearman rho = {rho:.3f}, p = {p_val:.3f}")
    else:
        ax.set_title(f"Beta vs R-squared — {session_id}")

    ax.set_xlabel("Regression R-squared")
    ax.set_ylabel("Beta(surprisal)")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Dashboard generator: wire all panels together
# ══════════════════════════════════════════════════════════════════════════


def generate_tier1_dashboard(
    session_key: str,
    results_dir: Path,
    all_summaries: List[Dict[str, Any]],
    *,
    X_spikes_z: Optional[np.ndarray] = None,
    trans_bins: Optional[np.ndarray] = None,
    z_surprisal: Optional[np.ndarray] = None,
    controls_df: Optional[pd.DataFrame] = None,
    unit_ids: Optional[np.ndarray] = None,
    bin_s: float = 0.5,
) -> None:
    """Generate all Tier 1 panels for one session.

    Reads saved CSV/NPY files from the results directory, plus optionally
    takes live data for panels that need the full spike matrix (Panel L).

    Args:
        session_key: Session identifier.
        results_dir: Root results directory (contains per-session subdirs).
        all_summaries: List of all session summaries.
        X_spikes_z: Z-scored spike matrix (needed for Panel L).
        trans_bins: Transition bins (needed for Panel L).
        z_surprisal: Z-scored surprisal (needed for Panel L).
        controls_df: Controls DataFrame (needed for Panel L).
        unit_ids: Unit IDs (needed for Panel L).
        bin_s: Bin size in seconds.
    """
    session_dir = results_dir / session_key
    fig_dir = session_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load saved data
    surp_df = pd.read_csv(session_dir / "surprisal_values.csv")
    post_df = pd.read_csv(session_dir / "unit_results_post.csv")
    null_medians = np.load(session_dir / "null_medians.npy")

    import json
    with open(session_dir / "summary.json") as f:
        summary = json.load(f)

    current_motif = surp_df["current_motif"].values
    next_motif = surp_df["next_motif"].values
    retained_motifs = summary["retained_motifs"]
    surprisal = surp_df["surprisal"].values
    z_surp = surp_df["z_surprisal"].values
    post_speed = surp_df["post_speed"].values

    # Panel A: Transition count heatmap
    print("  Panel A: Transition count heatmap")
    fig = plot_transition_count_heatmap(current_motif, next_motif, retained_motifs, session_key)
    save_fig(fig, "panel_A_transition_counts", fig_dir)

    # Panel D: NLL comparison (cross-session)
    print("  Panel D: NLL comparison")
    fig = plot_model_nll_comparison(all_summaries)
    save_fig(fig, "panel_D_nll_comparison", fig_dir)

    # Panel F: Surprisal histogram
    print("  Panel F: Surprisal histogram")
    fig = plot_surprisal_histogram(surprisal, z_surp, session_key)
    save_fig(fig, "panel_F_surprisal_histogram", fig_dir)

    # Panel G: Surprisal by pair
    print("  Panel G: Surprisal by pair")
    fig = plot_surprisal_by_pair_heatmap(z_surp, current_motif, next_motif, retained_motifs, session_key)
    save_fig(fig, "panel_G_surprisal_by_pair", fig_dir)

    # Panel L: Time-resolved beta (needs live data)
    if X_spikes_z is not None and trans_bins is not None and z_surprisal is not None:
        print("  Panel L: Time-resolved beta")
        fig = plot_time_resolved_beta(
            X_spikes_z, trans_bins, z_surprisal, controls_df,
            unit_ids, bin_s, session_key,
        )
        save_fig(fig, "panel_L_time_resolved_beta", fig_dir)
    else:
        print("  Panel L: SKIPPED (needs live spike data)")

    # Panel M: Pre vs post beta (cross-session)
    print("  Panel M: Pre vs post beta")
    fig = plot_pre_vs_post_beta(all_summaries)
    save_fig(fig, "panel_M_pre_vs_post_beta", fig_dir)

    # Panel N: Unit beta histogram
    print("  Panel N: Unit beta histogram")
    fig = plot_unit_beta_histogram(post_df, session_key)
    save_fig(fig, "panel_N_unit_beta_histogram", fig_dir)

    # Panel U: Null histogram
    print("  Panel U: Null histogram")
    fig = plot_null_histogram(
        null_medians, summary["observed_median_beta"],
        summary["permutation_p"], session_key,
    )
    save_fig(fig, "panel_U_null_histogram", fig_dir)

    # Panel T: Matched surprisal (needs live spike data for population delta_fr)
    print("  Panel T: Matched surprisal")
    if X_spikes_z is not None and trans_bins is not None:
        from freemat.single_fm.encoding.fig5_transition_surprisal import (
            compute_neural_responses,
        )
        # Reconstruct post_neural_idx from trans_bins
        n_post = int(round(1.0 / bin_s))
        post_neural_idx = np.column_stack([
            trans_bins + j for j in range(n_post)
        ])
        # Clip to valid range
        valid = np.all(post_neural_idx < X_spikes_z.shape[0], axis=1) & np.all(post_neural_idx >= 0, axis=1)
        if valid.sum() > 10:
            pre_neural_idx = np.column_stack([
                trans_bins - n_post + j for j in range(n_post)
            ])
            delta_fr = compute_neural_responses(X_spikes_z, pre_neural_idx[valid], post_neural_idx[valid])
            pop_mean_dfr = delta_fr.mean(axis=1)
            fig = plot_matched_surprisal(
                pop_mean_dfr, z_surp[valid],
                current_motif[valid], next_motif[valid],
                post_speed[valid], session_key,
            )
            save_fig(fig, "panel_T_matched_surprisal", fig_dir)
        else:
            print("  Panel T: SKIPPED (insufficient valid bins)")
    else:
        print("  Panel T: SKIPPED (needs live spike data)")

    # Panel V1: Beta vs selectivity
    print("  Panel V1: Beta vs selectivity")
    fig = plot_beta_vs_selectivity(post_df, session_key)
    save_fig(fig, "panel_V1_beta_vs_selectivity", fig_dir)

    print(f"  Dashboard complete: {fig_dir}")


__all__ = [
    "plot_transition_count_heatmap",
    "plot_model_nll_comparison",
    "plot_surprisal_histogram",
    "plot_surprisal_by_pair_heatmap",
    "plot_time_resolved_beta",
    "plot_pre_vs_post_beta",
    "plot_unit_beta_histogram",
    "plot_null_histogram",
    "plot_matched_surprisal",
    "plot_beta_vs_selectivity",
    "generate_tier1_dashboard",
]
