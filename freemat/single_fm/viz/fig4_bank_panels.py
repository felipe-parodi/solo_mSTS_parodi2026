"""Figure 4 bank-split panels: dorsal vs ventral coding comparisons.

Panels:
  4A: Bank-split PETH heatmaps (peak-latency ordering)
  4B: Bank-specific lag curves + anticipatory index
  4C: Pooled bankwise statistics (selectivity / bandwidth / transition_resp)
  4D: Bank-specific RSA (neural confusion vs behavioral similarity)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from freemat.single_fm.viz.pub_data import SUBJECT_MAP

# ── Bank colors ─────────────────────────────────────────────────────────
BANK_COLORS: Dict[str, str] = {
    "ventral": "#457B9D",
    "dorsal": "#E63946",
    "pooled": "#555555",
}


def _setup_nature_style() -> None:
    """Apply Nature journal style to matplotlib."""
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
    })


# ══════════════════════════════════════════════════════════════════════════
# Panel 4A: Bank-split PETH heatmaps (peak-latency ordering)
# ══════════════════════════════════════════════════════════════════════════

def fig4a_bank_peth(
    peth_data: Dict[str, Any],
    session_key: str = "M2_0526",
) -> plt.Figure:
    """Side-by-side PETH heatmaps sorted by peak latency within each bank.

    Uses fine (50ms) PETH data. Units within each bank are sorted by the
    time of their maximum response, creating a visible temporal wave pattern.

    Args:
        peth_data: Dict from load_fig4()["peth"], keyed by session.
        session_key: Session to plot.

    Returns:
        matplotlib Figure with two heatmap panels (ventral left, dorsal right).
    """
    _setup_nature_style()

    sess = peth_data.get(session_key, {})
    peth = sess.get("peth_fine")
    tau_bins = sess.get("tau_bins_fine")
    bin_s = sess.get("bin_s_fine", 0.05)
    unit_banks = sess.get("unit_banks_fine")

    if peth is None or unit_banks is None:
        print(f"  [SKIP] No fine PETH for {session_key}")
        return plt.figure()

    time_axis = tau_bins * bin_s  # convert bin indices to seconds

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, bank_name in zip(axes, ["ventral", "dorsal"]):
        bank_mask = np.array([b == bank_name for b in unit_banks])
        bank_peth = peth[bank_mask]
        n_units = bank_peth.shape[0]

        if n_units == 0:
            ax.set_title(f"{bank_name.capitalize()} (n=0)")
            continue

        # Z-score each unit (row) for consistent colormap
        row_mean = bank_peth.mean(axis=1, keepdims=True)
        row_std = bank_peth.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        peth_z = (bank_peth - row_mean) / row_std

        # Sort by peak latency (time of max response)
        peak_idx = np.argmax(peth_z, axis=1)
        sort_order = np.argsort(peak_idx)
        peth_sorted = peth_z[sort_order]

        im = ax.imshow(
            peth_sorted, aspect="auto", cmap="RdBu_r",
            vmin=-2, vmax=2,
            extent=[time_axis[0], time_axis[-1], n_units - 0.5, -0.5],
            interpolation="bilinear",
        )
        ax.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.set_xlabel("Time from transition (s)")
        ax.set_ylabel("Units (sorted by peak latency)")
        ax.set_title(
            f"{bank_name.capitalize()} (n={n_units})",
            fontweight="bold", color=BANK_COLORS[bank_name],
        )

    fig.colorbar(im, ax=axes, label="Z-scored firing rate", shrink=0.8)
    subject = SUBJECT_MAP.get(session_key, session_key)
    fig.suptitle(
        f"Motif-transition PETHs — {subject} ({session_key})",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    return fig


def fig4a_bank_peth_all_sessions(
    peth_data: Dict[str, Any],
    sessions: Optional[List[str]] = None,
) -> plt.Figure:
    """Multi-session bank-split PETH: 2 rows (ventral/dorsal) x N sessions.

    Args:
        peth_data: Dict from load_fig4()["peth"], keyed by session.
        sessions: Sessions to include (default: all with fine PETH).

    Returns:
        matplotlib Figure.
    """
    _setup_nature_style()

    if sessions is None:
        sessions = [s for s in peth_data if "peth_fine" in peth_data[s]]

    n_sess = len(sessions)
    if n_sess == 0:
        return plt.figure()

    fig, axes = plt.subplots(2, n_sess, figsize=(4 * n_sess, 8), squeeze=False)

    for col, sess_key in enumerate(sessions):
        sess = peth_data[sess_key]
        peth = sess["peth_fine"]
        tau_bins = sess["tau_bins_fine"]
        bin_s = sess.get("bin_s_fine", 0.05)
        unit_banks = sess["unit_banks_fine"]
        time_axis = tau_bins * bin_s

        for row, bank_name in enumerate(["ventral", "dorsal"]):
            ax = axes[row, col]
            bank_mask = np.array([b == bank_name for b in unit_banks])
            bank_peth = peth[bank_mask]
            n_units = bank_peth.shape[0]

            if n_units == 0:
                ax.text(0.5, 0.5, "No units", transform=ax.transAxes, ha="center")
                continue

            row_mean = bank_peth.mean(axis=1, keepdims=True)
            row_std = bank_peth.std(axis=1, keepdims=True)
            row_std[row_std == 0] = 1.0
            peth_z = (bank_peth - row_mean) / row_std

            peak_idx = np.argmax(peth_z, axis=1)
            sort_order = np.argsort(peak_idx)
            peth_sorted = peth_z[sort_order]

            ax.imshow(
                peth_sorted, aspect="auto", cmap="RdBu_r",
                vmin=-2, vmax=2,
                extent=[time_axis[0], time_axis[-1], n_units - 0.5, -0.5],
                interpolation="bilinear",
            )
            ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.8)

            if row == 0:
                _pub = {"M2": "M2", "M1": "M1"}
                subject = SUBJECT_MAP.get(sess_key, sess_key)
                pub_subj = _pub.get(subject, subject)
                ax.set_title(f"{pub_subj}\n{sess_key.split('_')[1]}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{bank_name.capitalize()}\n(n={n_units})",
                              color=BANK_COLORS[bank_name], fontweight="bold")
            else:
                ax.set_ylabel(f"n={n_units}", fontsize=9)
            if row == 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle("Bank-split syllable-transition PETHs (peak-latency sorted)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4B: Bank-specific lag curves + anticipatory index
# ══════════════════════════════════════════════════════════════════════════

def _compute_anticipatory_index(lag_df: pd.DataFrame) -> float:
    """Compute anticipatory index from lag sweep data for one bank/session.

    AI = mean(acc[lag in {-2.0, -1.5, -1.0, -0.5}])
       - mean(acc[lag in {+0.5, +1.0, +1.5, +2.0}])

    Positive AI means neural activity leads behavior.

    Args:
        lag_df: Rows for one bank/session from fig4_lag_sweep.parquet.

    Returns:
        Anticipatory index (float).
    """
    pre_lags = [-2.0, -1.5, -1.0, -0.5]
    post_lags = [0.5, 1.0, 1.5, 2.0]
    pre = lag_df[lag_df["lag_sec"].isin(pre_lags)]["accuracy"].mean()
    post = lag_df[lag_df["lag_sec"].isin(post_lags)]["accuracy"].mean()
    return float(pre - post)


def fig4b_bank_lag_curves(
    lag_sweep_df: pd.DataFrame,
    sessions: Optional[List[str]] = None,
) -> plt.Figure:
    """Bank-specific lag curves: ventral vs dorsal across sessions.

    Thin lines per session, bold session-mean. Annotates optimal lag per bank
    and computes anticipatory index.

    Args:
        lag_sweep_df: Concatenated lag_sweep.parquet from load_fig4()["lag_sweep"].
        sessions: Sessions to include.

    Returns:
        matplotlib Figure with lag curves and AI annotation.
    """
    _setup_nature_style()

    if sessions is None:
        sessions = lag_sweep_df["session_id"].unique().tolist()

    fig, (ax_main, ax_ai) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]},
    )

    ai_data: Dict[str, List[float]] = {"ventral": [], "dorsal": []}

    for bank in ["ventral", "dorsal"]:
        color = BANK_COLORS[bank]
        all_accs = []

        for sess in sessions:
            subset = lag_sweep_df[
                (lag_sweep_df["session_id"] == sess)
                & (lag_sweep_df["bank"] == bank)
            ].sort_values("lag_sec")

            if subset.empty:
                continue

            lags = subset["lag_sec"].values
            accs = subset["accuracy"].values
            ax_main.plot(lags, accs, color=color, alpha=0.25, linewidth=1)
            all_accs.append(accs)

            ai = _compute_anticipatory_index(subset)
            ai_data[bank].append(ai)

        if all_accs:
            mean_acc = np.mean(all_accs, axis=0)
            ax_main.plot(lags, mean_acc, color=color, linewidth=2.5,
                         label=f"{bank.capitalize()} (mean)")

            # Annotate optimal lag
            opt_idx = np.argmax(mean_acc)
            opt_lag = lags[opt_idx]
            opt_acc = mean_acc[opt_idx]
            ax_main.annotate(
                f"{opt_lag:+.1f}s", xy=(opt_lag, opt_acc),
                xytext=(opt_lag + 0.3, opt_acc + 0.02),
                fontsize=9, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    ax_main.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_main.set_xlabel("Lag (s) — neural relative to behavior")
    ax_main.set_ylabel("Balanced accuracy")
    ax_main.set_title("Bank-specific syllable decoding by lag", fontweight="bold")
    ax_main.legend(fontsize=10)

    # Anticipatory index panel
    for bank in ["ventral", "dorsal"]:
        vals = ai_data[bank]
        color = BANK_COLORS[bank]
        x_jitter = 0 if bank == "ventral" else 1
        ax_ai.scatter(
            [x_jitter] * len(vals), vals, color=color, s=60,
            edgecolors="black", linewidth=0.5, zorder=3,
        )
        ax_ai.bar(x_jitter, np.mean(vals), width=0.6, color=color, alpha=0.3)

    # Wilcoxon test (paired by session)
    if len(ai_data["ventral"]) >= 3 and len(ai_data["dorsal"]) >= 3:
        n_paired = min(len(ai_data["ventral"]), len(ai_data["dorsal"]))
        stat, p_val = stats.wilcoxon(
            ai_data["ventral"][:n_paired],
            ai_data["dorsal"][:n_paired],
        )
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        ax_ai.text(0.5, 0.95, f"Wilcoxon\n{p_str}", transform=ax_ai.transAxes,
                   ha="center", va="top", fontsize=9)

    ax_ai.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_ai.set_xticks([0, 1])
    ax_ai.set_xticklabels(["Ventral", "Dorsal"], fontsize=11)
    ax_ai.set_ylabel("Anticipatory index (AI)")
    ax_ai.set_title("AI\n(pre − post)", fontweight="bold")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4C: Pooled bankwise statistics
# ══════════════════════════════════════════════════════════════════════════

def _compute_unit_metrics(tuning_df: pd.DataFrame) -> pd.DataFrame:
    """Derive selectivity, bandwidth, transition_resp from tuning parquet.

    Args:
        tuning_df: Concatenated fig4_tuning.parquet.

    Returns:
        DataFrame with added selectivity, bandwidth columns per unit.
    """
    z_cols = [c for c in tuning_df.columns if c.endswith("_z")]

    if not z_cols:
        print("  [WARN] No z-score columns found in tuning data")
        return tuning_df

    z_matrix = tuning_df[z_cols].values

    # Selectivity: max |z| across motifs
    tuning_df = tuning_df.copy()
    tuning_df["selectivity"] = np.nanmax(np.abs(z_matrix), axis=1)

    # Bandwidth: count of motifs with |z| > 1.0
    tuning_df["bandwidth"] = np.nansum(np.abs(z_matrix) > 1.0, axis=1)

    # Response modulation: range of z-scores (max - min)
    tuning_df["response_modulation"] = (
        np.nanmax(z_matrix, axis=1) - np.nanmin(z_matrix, axis=1)
    )

    return tuning_df


def _compute_transition_resp(
    peth_data: Dict[str, Any],
    sessions: List[str],
) -> pd.DataFrame:
    """Compute per-unit transition responsiveness from fine PETH.

    Transition responsiveness = peak z-scored PETH in ±1s window around t=0
    divided by mean absolute PETH in baseline (−4 to −2s).

    Args:
        peth_data: Dict from load_fig4()["peth"], keyed by session.
        sessions: Sessions to process.

    Returns:
        DataFrame with session_id, unit_index, bank, transition_resp.
    """
    rows = []
    for sess_key in sessions:
        sess = peth_data.get(sess_key, {})
        peth = sess.get("peth_fine")
        tau_bins = sess.get("tau_bins_fine")
        bin_s = sess.get("bin_s_fine", 0.05)
        unit_banks = sess.get("unit_banks_fine")

        if peth is None or unit_banks is None:
            continue

        time_s = tau_bins * bin_s

        # Baseline: -4 to -2s
        baseline_mask = (time_s >= -4.0) & (time_s < -2.0)
        # Transition window: -1 to +1s
        trans_mask = (time_s >= -1.0) & (time_s <= 1.0)

        for i in range(peth.shape[0]):
            baseline_val = np.mean(np.abs(peth[i, baseline_mask]))
            if baseline_val == 0:
                baseline_val = 1e-6
            trans_peak = np.max(np.abs(peth[i, trans_mask]))
            rows.append({
                "session_id": sess_key,
                "unit_index": i,
                "bank": unit_banks[i],
                "transition_resp": trans_peak / baseline_val,
            })

    return pd.DataFrame(rows)


def fig4c_bankwise_stats(
    tuning_df: pd.DataFrame,
    peth_data: Optional[Dict[str, Any]] = None,
    sessions: Optional[List[str]] = None,
) -> plt.Figure:
    """Three paired-comparison panels: selectivity, bandwidth, transition_resp.

    Session is the replication unit for inferential stats. Unit-level
    distributions shown as violins, session medians as overlaid dots.
    P-values from Wilcoxon signed-rank on per-session medians (n=5).

    Args:
        tuning_df: Concatenated fig4_tuning.parquet (from load_fig4()["tuning"]).
        peth_data: Dict from load_fig4()["peth"] for transition_resp (optional).
        sessions: Sessions to include.

    Returns:
        matplotlib Figure with 3 subpanels.
    """
    _setup_nature_style()

    df = _compute_unit_metrics(tuning_df)

    if sessions is None:
        sessions = sorted(df["session_id"].unique().tolist())

    metrics = ["selectivity", "bandwidth", "response_modulation"]
    metric_labels = ["Selectivity\n(max |z|)", "Bandwidth\n(# motifs |z|>1)",
                     "Response modulation\n(z range)"]

    # Add transition responsiveness if PETH available
    if peth_data is not None:
        trans_df = _compute_transition_resp(peth_data, sessions)
        if not trans_df.empty:
            metrics.append("transition_resp")
            metric_labels.append("Transition resp.\n(peak / baseline)")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        if metric == "transition_resp":
            plot_df = trans_df
        else:
            plot_df = df

        if metric not in plot_df.columns:
            ax.text(0.5, 0.5, f"No {metric}", transform=ax.transAxes, ha="center")
            continue

        # Violin per bank
        for i, bank in enumerate(["ventral", "dorsal"]):
            bank_data = plot_df[plot_df["bank"] == bank][metric].dropna()
            if len(bank_data) == 0:
                continue

            parts = ax.violinplot(
                bank_data, positions=[i], showextrema=False, widths=0.7,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(BANK_COLORS[bank])
                pc.set_alpha(0.3)

            # Box plot inside
            bp = ax.boxplot(
                bank_data, positions=[i], widths=0.3,
                patch_artist=True, showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(BANK_COLORS[bank])
                patch.set_alpha(0.6)
            for element in ["whiskers", "caps", "medians"]:
                plt.setp(bp[element], color="black", linewidth=1)

        # Session medians as overlaid dots (connected)
        sess_medians_v = []
        sess_medians_d = []
        for sess in sessions:
            if metric == "transition_resp":
                sess_data = trans_df[trans_df["session_id"] == sess]
            else:
                sess_data = df[df["session_id"] == sess]

            v_med = sess_data[sess_data["bank"] == "ventral"][metric].median()
            d_med = sess_data[sess_data["bank"] == "dorsal"][metric].median()

            if not np.isnan(v_med) and not np.isnan(d_med):
                sess_medians_v.append(v_med)
                sess_medians_d.append(d_med)
                ax.plot([0, 1], [v_med, d_med], color="gray",
                        linewidth=0.8, alpha=0.5, zorder=2)

            ax.scatter(0, v_med, color=BANK_COLORS["ventral"], s=40,
                       edgecolors="black", linewidth=0.5, zorder=3)
            ax.scatter(1, d_med, color=BANK_COLORS["dorsal"], s=40,
                       edgecolors="black", linewidth=0.5, zorder=3)

        # Wilcoxon signed-rank on session medians
        if len(sess_medians_v) >= 3:
            stat, p_val = stats.wilcoxon(sess_medians_v, sess_medians_d)
            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            ax.text(0.5, 0.97, p_str, transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, fontstyle="italic")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ventral", "Dorsal"], fontsize=11)
        ax.set_ylabel(label)

    fig.suptitle("Bank-specific coding properties (all sessions pooled)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Panel 4D: Bank-specific RSA
# ══════════════════════════════════════════════════════════════════════════

def fig4d_bank_rsa(
    rsa_results: List[Dict[str, Any]],
) -> plt.Figure:
    """Bank-specific RSA: neural confusion vs pose similarity & transition adjacency.

    Grouped bar or paired-dot: rho by (bank x reference_matrix) with session dots.

    Args:
        rsa_results: List of per-session dicts with keys:
            session_id, ventral_pose_rho, dorsal_pose_rho,
            ventral_trans_rho, dorsal_trans_rho, and p-values.

    Returns:
        matplotlib Figure.
    """
    _setup_nature_style()

    if not rsa_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No RSA data available", transform=ax.transAxes,
                ha="center")
        return fig

    fig, (ax_pose, ax_trans) = plt.subplots(1, 2, figsize=(10, 5))

    sessions = [r["session_id"] for r in rsa_results]
    n = len(sessions)

    # Pose similarity RSA
    v_pose = [r.get("ventral_pose_rho", np.nan) for r in rsa_results]
    d_pose = [r.get("dorsal_pose_rho", np.nan) for r in rsa_results]

    for i in range(n):
        ax_pose.plot([0, 1], [v_pose[i], d_pose[i]], color="gray",
                     linewidth=0.8, alpha=0.5, zorder=1)
    ax_pose.scatter([0] * n, v_pose, color=BANK_COLORS["ventral"], s=60,
                    edgecolors="black", linewidth=0.5, zorder=3,
                    label="Ventral")
    ax_pose.scatter([1] * n, d_pose, color=BANK_COLORS["dorsal"], s=60,
                    edgecolors="black", linewidth=0.5, zorder=3,
                    label="Dorsal")

    # Bar for means
    ax_pose.bar(0, np.nanmean(v_pose), width=0.5,
                color=BANK_COLORS["ventral"], alpha=0.2)
    ax_pose.bar(1, np.nanmean(d_pose), width=0.5,
                color=BANK_COLORS["dorsal"], alpha=0.2)

    if n >= 3:
        valid = ~(np.isnan(v_pose) | np.isnan(d_pose))
        if valid.sum() >= 3:
            stat, p = stats.wilcoxon(
                np.array(v_pose)[valid], np.array(d_pose)[valid],
            )
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax_pose.text(0.5, 0.95, p_str, transform=ax_pose.transAxes,
                         ha="center", va="top", fontsize=9, fontstyle="italic")

    ax_pose.set_xticks([0, 1])
    ax_pose.set_xticklabels(["Ventral", "Dorsal"])
    ax_pose.set_ylabel("Mantel ρ (neural ↔ pose similarity)")
    ax_pose.set_title("Pose similarity RSA", fontweight="bold")
    ax_pose.legend(fontsize=9)

    # Transition adjacency RSA
    v_trans = [r.get("ventral_trans_rho", np.nan) for r in rsa_results]
    d_trans = [r.get("dorsal_trans_rho", np.nan) for r in rsa_results]

    for i in range(n):
        ax_trans.plot([0, 1], [v_trans[i], d_trans[i]], color="gray",
                      linewidth=0.8, alpha=0.5, zorder=1)
    ax_trans.scatter([0] * n, v_trans, color=BANK_COLORS["ventral"], s=60,
                     edgecolors="black", linewidth=0.5, zorder=3)
    ax_trans.scatter([1] * n, d_trans, color=BANK_COLORS["dorsal"], s=60,
                     edgecolors="black", linewidth=0.5, zorder=3)

    ax_trans.bar(0, np.nanmean(v_trans), width=0.5,
                 color=BANK_COLORS["ventral"], alpha=0.2)
    ax_trans.bar(1, np.nanmean(d_trans), width=0.5,
                 color=BANK_COLORS["dorsal"], alpha=0.2)

    if n >= 3:
        valid = ~(np.isnan(v_trans) | np.isnan(d_trans))
        if valid.sum() >= 3:
            stat, p = stats.wilcoxon(
                np.array(v_trans)[valid], np.array(d_trans)[valid],
            )
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax_trans.text(0.5, 0.95, p_str, transform=ax_trans.transAxes,
                          ha="center", va="top", fontsize=9, fontstyle="italic")

    ax_trans.set_xticks([0, 1])
    ax_trans.set_xticklabels(["Ventral", "Dorsal"])
    ax_trans.set_ylabel("Mantel ρ (neural ↔ transition adjacency)")
    ax_trans.set_title("Transition adjacency RSA", fontweight="bold")

    fig.suptitle("Bank-specific representational similarity",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
