#!/usr/bin/env python
"""Temporal Split-Half Stability Control for Figure 2.

Splits each session chronologically into early/late halves and re-runs
the full Fig 2 encoding pipeline on each half independently.  Compares
unique variance estimates across halves to test temporal stability of
encoding.

Usage:
    # Smoke test (1 session, 20 perms)
    python scripts/single_fm/run_fig2b_temporal_split_control.py \
        --sessions M2_0526 --n-perm 20 \
        --output-dir results/single_fm/temporal_split_smoke \
        --use-gpu --debug

    # Full production (all 5 sessions, 100 perms)
    python scripts/single_fm/run_fig2b_temporal_split_control.py \
        --sessions M2_0526 M2_0605 M1_0525 M1_0609 M1_0613 \
        --n-perm 100 --n-perm-null 0 \
        --baseline-dir results/single_fm/production_20260209_2301/all4_bin05 \
        --output-dir results/single_fm/temporal_split_control \
        --use-gpu
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Canonical inference time TSV (explicit path, not default)
_CANONICAL_TSV = str(_PROJECT_ROOT / "docs" / "inf_start_stop_times.tsv")

# Session configuration mapping (matches run_rssi_control.py)
SESSIONS: Dict[str, str] = {
    "M2_0526": "configs/single_fm/session_M2_0526.yaml",
    "M2_0605": "configs/single_fm/session_M2_0605.yaml",
    "M1_0525": "configs/single_fm/session_M1_0525.yaml",
    "M1_0609": "configs/single_fm/session_M1_0609.yaml",
    "M1_0613": "configs/single_fm/session_M1_0613.yaml",
}

SESSION_OUTPUT_DIRS: Dict[str, str] = {
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0525": "single_0525_M1",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}

# Map session keys to TSV date keys (YYMMDD)
SESSION_DATES: Dict[str, str] = {
    "M2_0526": "240526",
    "M2_0605": "240605",
    "M1_0525": "240525",
    "M1_0609": "240609",
    "M1_0613": "240613",
}

# Feature groups to compare
GROUPS = ["vis", "body_kin", "head_kin", "pos"]

# Publication style
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
})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Temporal Split-Half Stability Control for Figure 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=list(SESSIONS.keys()),
        default=list(SESSIONS.keys()),
        help="Sessions to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for temporal split results",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Baseline production dir for comparison (e.g., .../all4_bin05)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=100,
        help="Number of shuffle iterations (default: 100)",
    )
    parser.add_argument(
        "--n-perm-null",
        type=int,
        default=0,
        help="Number of null permutations for p-values (default: 0, matches production)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="Parallel jobs for CPU mode (default: 6; forced to 1 when --use-gpu)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration (forces n_jobs=1)",
    )
    parser.add_argument(
        "--split-frac",
        type=float,
        default=0.5,
        help="Fraction of session for early half (default: 0.5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    return parser.parse_args()


def generate_split_tsvs(
    sess_key: str,
    output_dir: str,
    split_frac: float = 0.5,
) -> Dict[str, str]:
    """Generate early/late TSV files for a session.

    Reads the canonical inference time window, computes the midpoint,
    and writes two TSV files with the early and late windows.

    Args:
        sess_key: Session key (e.g., ``"M2_0526"``).
        output_dir: Directory to write TSV files.
        split_frac: Fraction of session for early half.

    Returns:
        Dict with ``"early"`` and ``"late"`` keys mapping to TSV paths.
    """
    from inference_time_filter import InferenceTimeFilter, seconds_to_time_str

    time_filter = InferenceTimeFilter.from_tsv(_CANONICAL_TSV)
    session_date = SESSION_DATES[sess_key]
    time_range = time_filter.get_range(session_date)

    if time_range is None:
        raise ValueError(
            f"No time range found for session {sess_key} "
            f"(date={session_date}) in {_CANONICAL_TSV}"
        )

    start_s = time_range.start_seconds
    end_s = time_range.end_seconds
    mid_s = start_s + (end_s - start_s) * split_frac

    start_str = seconds_to_time_str(start_s)
    mid_str = seconds_to_time_str(mid_s)
    end_str = seconds_to_time_str(end_s)

    os.makedirs(output_dir, exist_ok=True)
    tsv_paths: Dict[str, str] = {}

    for half_name, t_start, t_end in [
        ("early", start_str, mid_str),
        ("late", mid_str, end_str),
    ]:
        tsv_path = os.path.join(output_dir, f"{half_name}_window.tsv")
        with open(tsv_path, "w") as f:
            f.write("Session, Inf Start:Stop\n")
            f.write(f"{session_date}, {t_start}-{t_end}\n")
        tsv_paths[half_name] = tsv_path

    print(f"  Split windows for {sess_key}:")
    print(f"    Full:  {start_str} - {end_str} ({end_s - start_s:.0f}s)")
    print(f"    Early: {start_str} - {mid_str} ({mid_s - start_s:.0f}s)")
    print(f"    Late:  {mid_str} - {end_str} ({end_s - mid_s:.0f}s)")

    return tsv_paths


def run_split_sanity_checks(
    early_df: pd.DataFrame,
    late_df: pd.DataFrame,
    sess_key: str,
    split_frac: float,
    baseline_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Validate split results using n_bins column.

    Args:
        early_df: Results from early half.
        late_df: Results from late half.
        sess_key: Session key for logging.
        split_frac: Expected fraction for early half.
        baseline_df: Optional full-session baseline results.

    Returns:
        Dict of sanity check metrics.
    """
    metrics: Dict[str, float] = {}

    # Paired merge on unit_id
    merged = early_df[["unit_id", "n_bins"]].merge(
        late_df[["unit_id", "n_bins"]],
        on="unit_id",
        suffixes=("_early", "_late"),
    )

    n_early_only = len(set(early_df["unit_id"]) - set(late_df["unit_id"]))
    n_late_only = len(set(late_df["unit_id"]) - set(early_df["unit_id"]))
    n_both = len(merged)

    metrics["n_units_early"] = len(early_df)
    metrics["n_units_late"] = len(late_df)
    metrics["n_units_both"] = n_both
    metrics["n_units_early_only"] = n_early_only
    metrics["n_units_late_only"] = n_late_only

    print(f"  Sanity checks for {sess_key}:")
    print(f"    Units: early={len(early_df)}, late={len(late_df)}, "
          f"both={n_both}, early-only={n_early_only}, late-only={n_late_only}")

    # Bin count ratio
    if n_both > 0:
        total_bins = merged["n_bins_early"] + merged["n_bins_late"]
        ratio = merged["n_bins_early"] / total_bins
        median_ratio = float(ratio.median())
        metrics["median_bin_ratio"] = median_ratio

        # Check ±10% of expected split_frac
        in_range = ((ratio >= split_frac - 0.1) & (ratio <= split_frac + 0.1)).mean()
        metrics["pct_bins_in_range"] = float(in_range * 100)

        print(f"    Median bin ratio (early/total): {median_ratio:.3f} "
              f"(expected {split_frac:.2f})")
        print(f"    Units within ±10%: {in_range * 100:.1f}%")

    # Baseline retention
    if baseline_df is not None and n_both > 0:
        base_merge = merged.merge(
            baseline_df[["unit_id", "n_bins"]].rename(
                columns={"n_bins": "n_bins_full"}
            ),
            on="unit_id",
            how="inner",
        )
        if len(base_merge) > 0:
            total = base_merge["n_bins_early"] + base_merge["n_bins_late"]
            diff = (total - base_merge["n_bins_full"]).abs()
            median_diff = float(diff.median())
            metrics["median_bin_diff_vs_full"] = median_diff
            n_baseline_units = len(baseline_df)
            n_retained = len(base_merge)
            metrics["baseline_unit_retention_pct"] = float(
                n_retained / n_baseline_units * 100
            )
            print(f"    Baseline retention: {n_retained}/{n_baseline_units} units "
                  f"({n_retained / n_baseline_units * 100:.1f}%)")
            print(f"    Median |n_bins_sum - n_bins_full|: {median_diff:.1f} bins "
                  f"(boundary quantization tolerance: ±5)")

    return metrics


def compute_paired_metrics(
    merged: pd.DataFrame,
    sess_label: str,
) -> Dict[str, float]:
    """Compute paired early/late metrics for all feature groups.

    Args:
        merged: DataFrame with ``_early`` / ``_late`` suffixed columns,
            joined on ``(session_id, unit_id)``.
        sess_label: Label for logging (session key or ``"pooled"``).

    Returns:
        Dict of metric name to value.
    """
    metrics: Dict[str, float] = {"label": sess_label, "n_paired": len(merged)}

    for g in GROUPS:
        col_e = f"unique_{g}_shuffle_early"
        col_l = f"unique_{g}_shuffle_late"

        if col_e not in merged.columns or col_l not in merged.columns:
            continue

        vals_e = merged[col_e].to_numpy(dtype=np.float64)
        vals_l = merged[col_l].to_numpy(dtype=np.float64)
        valid = np.isfinite(vals_e) & np.isfinite(vals_l)
        n_valid = int(valid.sum())

        if n_valid < 2:
            continue

        ve = vals_e[valid]
        vl = vals_l[valid]

        # Medians
        metrics[f"{g}_median_early"] = float(np.median(ve))
        metrics[f"{g}_median_late"] = float(np.median(vl))

        # % change (guard: NA when |median_early| < 1e-4)
        med_e = float(np.median(ve))
        med_l = float(np.median(vl))
        if abs(med_e) >= 1e-4:
            metrics[f"{g}_pct_change"] = float(
                abs(med_l - med_e) / abs(med_e) * 100
            )
        else:
            metrics[f"{g}_pct_change"] = float("nan")

        # Spearman correlation
        if n_valid >= 5:
            rho, p_spearman = spearmanr(ve, vl)
            metrics[f"{g}_spearman_rho"] = float(rho)
            metrics[f"{g}_spearman_p"] = float(p_spearman)

        # Sign agreement: sign(early) == sign(late)
        sign_agree = np.mean(np.sign(ve) == np.sign(vl))
        metrics[f"{g}_sign_agreement"] = float(sign_agree)

        # Wilcoxon signed-rank (guard: n>=10, not all diffs zero)
        diffs = ve - vl
        nonzero_diffs = diffs[diffs != 0.0]
        if len(nonzero_diffs) >= 10:
            stat, p_wilcox = wilcoxon(nonzero_diffs)
            metrics[f"{g}_wilcoxon_stat"] = float(stat)
            metrics[f"{g}_wilcoxon_p"] = float(p_wilcox)
        else:
            metrics[f"{g}_wilcoxon_stat"] = float("nan")
            metrics[f"{g}_wilcoxon_p"] = float("nan")

    return metrics


def check_acceptance_thresholds(
    metrics: Dict[str, float],
) -> List[str]:
    """Check preregistered acceptance thresholds.

    Args:
        metrics: Paired metrics dict (from :func:`compute_paired_metrics`).

    Returns:
        List of failure messages (empty if all pass).
    """
    failures: List[str] = []

    for g in GROUPS:
        med_e_key = f"{g}_median_early"
        med_l_key = f"{g}_median_late"
        rho_key = f"{g}_spearman_rho"
        pct_key = f"{g}_pct_change"

        if med_e_key not in metrics:
            continue

        med_e = metrics[med_e_key]
        med_l = metrics[med_l_key]

        # Threshold 1: late within 50-150% of early
        pct = metrics.get(pct_key, float("nan"))
        if not np.isnan(pct) and pct > 50.0:
            failures.append(
                f"{g.upper()}: |% change| = {pct:.1f}% > 50% threshold"
            )

        # Threshold 2: positive medians in both halves (skip HEAD_KIN, often ~0)
        if g != "head_kin":
            if med_e <= 0:
                failures.append(f"{g.upper()}: early median = {med_e:.4f} <= 0")
            if med_l <= 0:
                failures.append(f"{g.upper()}: late median = {med_l:.4f} <= 0")

        # Threshold 3: Spearman > 0.3
        rho = metrics.get(rho_key, float("nan"))
        if not np.isnan(rho) and rho < 0.3:
            failures.append(
                f"{g.upper()}: Spearman rho = {rho:.3f} < 0.3 threshold"
            )

    return failures


def plot_diagnostics(
    all_merged: pd.DataFrame,
    all_early: pd.DataFrame,
    all_late: pd.DataFrame,
    pooled_metrics: Dict[str, float],
    output_dir: str,
) -> None:
    """Generate 4-panel diagnostic figure.

    Args:
        all_merged: Paired merged DataFrame (early+late).
        all_early: Concatenated early results.
        all_late: Concatenated late results.
        pooled_metrics: Pooled metrics dict.
        output_dir: Directory for output plots.
    """
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Bin counts per half per session
    ax = axes[0, 0]
    sessions_in_data = sorted(all_merged["session_id"].unique())
    x = np.arange(len(sessions_in_data))
    width = 0.35

    early_bins = []
    late_bins = []
    for sid in sessions_in_data:
        mask = all_merged["session_id"] == sid
        early_bins.append(float(all_merged.loc[mask, "n_bins_early"].median()))
        late_bins.append(float(all_merged.loc[mask, "n_bins_late"].median()))

    ax.bar(x - width / 2, early_bins, width, label="Early", color="steelblue")
    ax.bar(x + width / 2, late_bins, width, label="Late", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions_in_data, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median bins per unit", fontweight="bold")
    ax.set_title("A. Bin Counts per Half", fontweight="bold")
    ax.legend()

    # Panel B: Boxplots early/late per group (pooled)
    ax = axes[0, 1]
    box_data = []
    box_labels = []
    box_colors = []
    for g in GROUPS:
        col_e = f"unique_{g}_shuffle_early"
        col_l = f"unique_{g}_shuffle_late"
        if col_e in all_merged.columns and col_l in all_merged.columns:
            box_data.append(all_merged[col_e].dropna().values)
            box_data.append(all_merged[col_l].dropna().values)
            box_labels.extend([f"{g.upper()}\nEarly", f"{g.upper()}\nLate"])
            box_colors.extend(["steelblue", "coral"])

    if box_data:
        bp = ax.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Unique \u0394R\u00b2", fontweight="bold")
        ax.set_title("B. Early vs Late Unique Variance", fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    # Panel C: Scatter early vs late POS
    ax = axes[1, 0]
    col_e = "unique_pos_shuffle_early"
    col_l = "unique_pos_shuffle_late"
    if col_e in all_merged.columns and col_l in all_merged.columns:
        ve = all_merged[col_e].to_numpy(dtype=np.float64)
        vl = all_merged[col_l].to_numpy(dtype=np.float64)
        valid = np.isfinite(ve) & np.isfinite(vl)
        ve_v, vl_v = ve[valid], vl[valid]

        ax.scatter(ve_v, vl_v, alpha=0.3, s=8, edgecolors="none", color="steelblue")
        lims = [
            min(ve_v.min(), vl_v.min()),
            max(ve_v.max(), vl_v.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=0.8, label="Identity")
        ax.set_xlabel("POS \u0394R\u00b2 (Early)", fontweight="bold")
        ax.set_ylabel("POS \u0394R\u00b2 (Late)", fontweight="bold")
        ax.set_title("C. POS Stability: Early vs Late", fontweight="bold")

        # Annotate Spearman + Wilcoxon
        rho = pooled_metrics.get("pos_spearman_rho", float("nan"))
        p_w = pooled_metrics.get("pos_wilcoxon_p", float("nan"))
        ann_parts = [f"n={int(valid.sum())}"]
        if not np.isnan(rho):
            ann_parts.append(f"Spearman \u03c1={rho:.3f}")
        if not np.isnan(p_w):
            ann_parts.append(f"Wilcoxon p={p_w:.4f}")
        ax.text(
            0.05, 0.95,
            "\n".join(ann_parts),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.legend(loc="lower right")

    # Panel D: Per-session % change per group
    ax = axes[1, 1]
    # Collect per-session metrics (stored alongside pooled)
    per_sess_path = os.path.join(output_dir, "summary", "temporal_split_summary.csv")
    if os.path.isfile(per_sess_path):
        summary = pd.read_csv(per_sess_path)
        sess_rows = summary[summary["label"] != "pooled"]
        if len(sess_rows) > 0:
            x_sess = np.arange(len(GROUPS))
            n_sess = len(sess_rows)
            bar_width = 0.8 / max(n_sess, 1)
            for i, (_, row) in enumerate(sess_rows.iterrows()):
                pct_vals = [row.get(f"{g}_pct_change", float("nan")) for g in GROUPS]
                offset = (i - n_sess / 2 + 0.5) * bar_width
                ax.bar(
                    x_sess + offset,
                    pct_vals,
                    bar_width,
                    label=row["label"],
                    alpha=0.7,
                )
            ax.axhline(50, color="red", linestyle="--", linewidth=0.8, label="50% threshold")
            ax.set_xticks(x_sess)
            ax.set_xticklabels([g.upper() for g in GROUPS])
            ax.set_ylabel("|% change| (early \u2192 late)", fontweight="bold")
            ax.set_title("D. Per-Session Stability", fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Temporal Split-Half Stability Control", fontweight="bold", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("png", "svg"):
        fig.savefig(os.path.join(diag_dir, f"temporal_split_diagnostics.{ext}"))
    plt.close(fig)
    print(f"  Saved diagnostic plots to {diag_dir}")


def _infer_bank_column(merged: pd.DataFrame) -> pd.Series:
    """Infer a unified bank label per row from merged early/late columns."""
    if "unit_bank_early" in merged.columns and "unit_bank_late" in merged.columns:
        return merged["unit_bank_early"].fillna(merged["unit_bank_late"]).astype(str)
    if "unit_bank_early" in merged.columns:
        return merged["unit_bank_early"].astype(str)
    if "unit_bank_late" in merged.columns:
        return merged["unit_bank_late"].astype(str)
    return pd.Series(np.repeat("pooled", len(merged)), index=merged.index, dtype=object)


def compute_pos_stability_by_session_bank(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """Compute POS split-half stability metrics per session and bank.

    Args:
        merged: Paired early/late table with unique_pos_shuffle columns.

    Returns:
        DataFrame with per-session per-bank correlations.
    """
    col_e = "unique_pos_shuffle_early"
    col_l = "unique_pos_shuffle_late"
    if col_e not in merged.columns or col_l not in merged.columns:
        return pd.DataFrame()

    work = merged.copy()
    work["unit_bank"] = _infer_bank_column(work)

    rows: List[Dict[str, float | str | int]] = []
    grouped = work.groupby(["session_id", "unit_bank"], dropna=False)
    for (session_id, bank), gdf in grouped:
        ve = gdf[col_e].to_numpy(dtype=np.float64)
        vl = gdf[col_l].to_numpy(dtype=np.float64)
        valid = np.isfinite(ve) & np.isfinite(vl)
        n = int(valid.sum())
        rho = float("nan")
        p = float("nan")
        if n >= 3:
            rho_val, p_val = spearmanr(ve[valid], vl[valid])
            rho = float(rho_val)
            p = float(p_val)
        rows.append(
            {
                "session_id": str(session_id),
                "unit_bank": str(bank),
                "n_units": n,
                "rho_spearman": rho,
                "p_spearman": p,
                "median_early": float(np.nanmedian(ve)),
                "median_late": float(np.nanmedian(vl)),
            }
        )
    return pd.DataFrame(rows)


def plot_pos_scatter_by_session_bank(
    merged: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save per-session POS early-vs-late scatter plots split by bank.

    Args:
        merged: Paired early/late table with POS unique variance columns.
        output_dir: Base output directory.
    """
    col_e = "unique_pos_shuffle_early"
    col_l = "unique_pos_shuffle_late"
    if col_e not in merged.columns or col_l not in merged.columns:
        return

    work = merged.copy()
    work["unit_bank"] = _infer_bank_column(work)
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    sessions = sorted(work["session_id"].astype(str).unique())
    for session_id in sessions:
        sdf = work[work["session_id"].astype(str) == session_id]
        banks = sorted(sdf["unit_bank"].astype(str).unique())
        n_cols = len(banks)

        # Compute global range across all banks for consistent shared axes
        all_ve = sdf[col_e].to_numpy(dtype=np.float64)
        all_vl = sdf[col_l].to_numpy(dtype=np.float64)
        all_valid = np.isfinite(all_ve) & np.isfinite(all_vl)
        if all_valid.any():
            global_lo = float(min(np.min(all_ve[all_valid]), np.min(all_vl[all_valid])))
            global_hi = float(max(np.max(all_ve[all_valid]), np.max(all_vl[all_valid])))
        else:
            global_lo, global_hi = 0.0, 1.0

        fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.5), sharex=True, sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, bank in zip(axes, banks):
            bdf = sdf[sdf["unit_bank"].astype(str) == bank]
            ve = bdf[col_e].to_numpy(dtype=np.float64)
            vl = bdf[col_l].to_numpy(dtype=np.float64)
            valid = np.isfinite(ve) & np.isfinite(vl)
            ve_v = ve[valid]
            vl_v = vl[valid]

            ax.scatter(ve_v, vl_v, alpha=0.35, s=10, edgecolors="none")
            ax.plot([global_lo, global_hi], [global_lo, global_hi], "k--", linewidth=0.9, alpha=0.8)
            ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.6)
            ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.6)

            rho = float("nan")
            p = float("nan")
            if len(ve_v) >= 3:
                rho_val, p_val = spearmanr(ve_v, vl_v)
                rho = float(rho_val)
                p = float(p_val)

            p_txt = "NA" if np.isnan(p) else f"{p:.3g}"
            r_txt = "NA" if np.isnan(rho) else f"{rho:.3f}"
            ax.text(
                0.03,
                0.97,
                f"n={len(ve_v)}\nρ={r_txt}\np={p_txt}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#bbbbbb"),
            )
            ax.set_title(f"{bank.capitalize()}", fontweight="bold")
            ax.set_xlabel("POS ΔR² (Early)")

        axes[0].set_xlim(global_lo, global_hi)
        axes[0].set_ylim(global_lo, global_hi)
        axes[0].set_ylabel("POS ΔR² (Late)")
        fig.suptitle(f"POS Split-Half Stability — {session_id}", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        png = os.path.join(diag_dir, f"temporal_split_pos_scatter_{session_id}.png")
        svg = os.path.join(diag_dir, f"temporal_split_pos_scatter_{session_id}.svg")
        fig.savefig(png)
        fig.savefig(svg)
        plt.close(fig)


def main() -> None:
    """Run temporal split-half stability control."""
    args = parse_args()
    sessions_to_run = args.sessions

    print("=" * 60)
    print("Temporal Split-Half Stability Control for Figure 2")
    print("=" * 60)
    print(f"Sessions: {sessions_to_run}")
    print(f"Output: {args.output_dir}")
    print(f"Perms: {args.n_perm} (null: {args.n_perm_null})")
    print(f"Split fraction: {args.split_frac}")
    print(f"GPU: {args.use_gpu}")
    n_jobs = args.n_jobs
    print(f"n_jobs: {n_jobs}")
    if args.baseline_dir:
        print(f"Baseline: {args.baseline_dir}")
    print()

    from freemat.single_fm.encoding.config import Fig2BConfig
    from freemat.single_fm.encoding.run_fig2b import run_fig2b_encoding

    # Production-parity config (matches run_production_all.sh)
    config = Fig2BConfig(
        bin_s=0.5,
        lag_s=1.0,
        include_time_drift=True,
        floor_filter=False,
        use_shuffle_method=True,
        n_perm=args.n_perm,
        n_perm_null=args.n_perm_null,
        n_jobs=n_jobs,
        use_gpu=args.use_gpu,
        include_pos=True,
        save_residuals=False,
        validate_coordinates=False,  # sessions already validated in production
        debug=args.debug,
    )

    # ── Step 1: Generate split TSVs and run encoding ──────────────────

    all_early: List[pd.DataFrame] = []
    all_late: List[pd.DataFrame] = []
    sanity_metrics: List[Dict] = []

    n_sessions = len(sessions_to_run)
    for idx, sess_key in enumerate(sessions_to_run, 1):
        yaml_path = str(_PROJECT_ROOT / SESSIONS[sess_key])
        out_name = SESSION_OUTPUT_DIRS[sess_key]
        session_out = os.path.join(args.output_dir, out_name)

        print(f"\n{'=' * 50}")
        print(f"[{idx}/{n_sessions}] {sess_key}")
        print(f"{'=' * 50}")

        # Generate early/late TSVs
        tsv_paths = generate_split_tsvs(
            sess_key, session_out, split_frac=args.split_frac
        )

        # Load baseline for sanity checks (if provided)
        baseline_df = None
        if args.baseline_dir:
            base_path = os.path.join(
                args.baseline_dir, out_name, "encoding",
                "fig2b_unique_variance.parquet",
            )
            if os.path.isfile(base_path):
                baseline_df = pd.read_parquet(base_path)

        half_results: Dict[str, pd.DataFrame] = {}

        for half_name in ("early", "late"):
            print(f"\n  --- {half_name} half ---")
            t0 = time.time()

            result_df = run_fig2b_encoding(
                session_yaml=yaml_path,
                output_dir=os.path.join(session_out, half_name),
                config=config,
                start_stop_tsv=tsv_paths[half_name],
            )

            elapsed = time.time() - t0
            n_units = len(result_df) if result_df is not None else 0
            print(f"  Done: {n_units} units in {elapsed:.1f}s")

            if result_df is not None and len(result_df) > 0:
                result_df = result_df.copy()
                result_df["half"] = half_name
                half_results[half_name] = result_df

        # Sanity checks
        if "early" in half_results and "late" in half_results:
            sm = run_split_sanity_checks(
                half_results["early"],
                half_results["late"],
                sess_key,
                args.split_frac,
                baseline_df=baseline_df,
            )
            sm["session"] = sess_key
            sanity_metrics.append(sm)

            all_early.append(half_results["early"])
            all_late.append(half_results["late"])

    # ── Step 2: Aggregate and compute paired metrics ──────────────────

    if not all_early or not all_late:
        print("\nNo results to aggregate. Exiting.")
        return

    early_cat = pd.concat(all_early, ignore_index=True)
    late_cat = pd.concat(all_late, ignore_index=True)

    # Paired merge
    merge_cols = ["session_id", "unit_id"]
    value_cols = (
        [f"unique_{g}_shuffle" for g in GROUPS]
        + ["full_r2", "n_bins", "alpha_selected", "unit_bank"]
    )
    # Keep only columns that exist
    early_keep = merge_cols + [c for c in value_cols if c in early_cat.columns]
    late_keep = merge_cols + [c for c in value_cols if c in late_cat.columns]

    all_merged = early_cat[early_keep].merge(
        late_cat[late_keep],
        on=merge_cols,
        suffixes=("_early", "_late"),
    )
    print(f"\nPaired units: {len(all_merged)} "
          f"(from {len(early_cat)} early + {len(late_cat)} late)")

    # Per-session metrics
    # session_id in parquet is the raw value from YAML (e.g., "M2_0526").
    # Build exact lookup: session_key -> set of session_id values in that key's data.
    sess_id_unique = set(all_merged["session_id"].unique())
    print(f"  session_ids in merged data: {sorted(sess_id_unique)}")

    sess_key_to_ids: Dict[str, List[str]] = {}
    for sk in sessions_to_run:
        # Prefer exact match; fall back to fuzzy only if needed
        matched = [sid for sid in sess_id_unique if str(sid) == sk]
        if not matched:
            date = SESSION_DATES[sk]
            name_prefix = sk.split("_")[0]
            matched = [
                sid for sid in sess_id_unique
                if date in str(sid) and name_prefix in str(sid)
            ]
        print(f"  {sk} -> matched session_ids: {matched}")
        if not matched:
            print(f"  WARNING: No session_id match for {sk}")
        sess_key_to_ids[sk] = matched

    summary_rows: List[Dict] = []
    for sess_key in sessions_to_run:
        ids = sess_key_to_ids[sess_key]
        if not ids:
            continue
        sess_merged = all_merged[all_merged["session_id"].isin(ids)]
        if len(sess_merged) > 0:
            m = compute_paired_metrics(sess_merged, sess_key)
            summary_rows.append(m)

    # Per-subject metrics (M2 = hedy_*, M1 = hooke_*)
    SUBJECT_SESSIONS: Dict[str, List[str]] = {
        "M2": [k for k in sessions_to_run if k.startswith("M2")],
        "M1": [k for k in sessions_to_run if k.startswith("M1")],
    }
    for subj, subj_sessions in SUBJECT_SESSIONS.items():
        if not subj_sessions:
            continue
        subj_ids = []
        for sk in subj_sessions:
            subj_ids.extend(sess_key_to_ids.get(sk, []))
        subj_mask = all_merged["session_id"].isin(subj_ids)
        subj_merged = all_merged[subj_mask]
        if len(subj_merged) > 0:
            m = compute_paired_metrics(subj_merged, subj)
            summary_rows.append(m)

    # Pooled metrics
    pooled_metrics = compute_paired_metrics(all_merged, "pooled")
    summary_rows.append(pooled_metrics)

    # ── Duplicate guard ─────────────────────────────────────────────────
    # Fail loud if two different session labels produce identical metrics
    # (prevents silently publishing duplicated summary artifacts).
    per_session_rows = [r for r in summary_rows if r["label"] not in ("pooled", "M2", "M1")]
    for i, r1 in enumerate(per_session_rows):
        for r2 in per_session_rows[i + 1:]:
            if r1["label"] != r2["label"] and r1["n_paired"] == r2["n_paired"]:
                # Check if metrics are identical (byte-for-byte)
                keys = [k for k in r1 if k != "label"]
                if all(r1[k] == r2[k] for k in keys if k in r2):
                    raise RuntimeError(
                        f"DUPLICATE SUMMARY BUG: sessions '{r1['label']}' and "
                        f"'{r2['label']}' have identical metrics (n_paired={r1['n_paired']}). "
                        f"Matched session_ids: {sess_key_to_ids}"
                    )

    # ── Step 3: Summary output ────────────────────────────────────────

    summary_dir = os.path.join(args.output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(summary_dir, "temporal_split_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")

    # Paired parquet
    parquet_path = os.path.join(summary_dir, "temporal_split_paired.parquet")
    all_merged.to_parquet(parquet_path, index=False)
    print(f"Saved paired table: {parquet_path}")

    pos_bank_df = compute_pos_stability_by_session_bank(all_merged)
    if len(pos_bank_df) > 0:
        pos_bank_path = os.path.join(summary_dir, "temporal_split_pos_stability_by_bank.csv")
        pos_bank_df.to_csv(pos_bank_path, index=False)
        print(f"Saved POS per-session/bank stability: {pos_bank_path}")

    # Sanity check CSV
    if sanity_metrics:
        sanity_df = pd.DataFrame(sanity_metrics)
        sanity_path = os.path.join(summary_dir, "temporal_split_sanity.csv")
        sanity_df.to_csv(sanity_path, index=False)

    # ── Step 4: Print results ─────────────────────────────────────────

    def _print_metrics_block(metrics: Dict[str, float], label: str) -> None:
        n = metrics.get("n_paired", 0)
        print(f"\n{label} ({n} paired units):")
        for g in GROUPS:
            med_e = metrics.get(f"{g}_median_early", float("nan"))
            med_l = metrics.get(f"{g}_median_late", float("nan"))
            pct = metrics.get(f"{g}_pct_change", float("nan"))
            rho = metrics.get(f"{g}_spearman_rho", float("nan"))
            sign_ag = metrics.get(f"{g}_sign_agreement", float("nan"))
            p_w = metrics.get(f"{g}_wilcoxon_p", float("nan"))

            pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "NA"
            rho_str = f"{rho:.3f}" if not np.isnan(rho) else "NA"
            sign_str = f"{sign_ag * 100:.1f}%" if not np.isnan(sign_ag) else "NA"
            p_str = f"{p_w:.4f}" if not np.isnan(p_w) else "NA"

            print(f"  {g.upper():10s}  early={med_e:.4f}  late={med_l:.4f}  "
                  f"|%chg|={pct_str}  rho={rho_str}  sign={sign_str}  "
                  f"Wilcoxon p={p_str}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Per-session
    for row in summary_rows:
        label = row.get("label", "?")
        if label not in ("pooled", "M2", "M1"):
            _print_metrics_block(row, label)

    # Per-subject
    for row in summary_rows:
        label = row.get("label", "?")
        if label in ("M2", "M1"):
            _print_metrics_block(row, label)

    # Pooled
    _print_metrics_block(pooled_metrics, "Pooled")

    # ── Step 5: Acceptance thresholds ─────────────────────────────────

    failures = check_acceptance_thresholds(pooled_metrics)
    print("\nAcceptance thresholds:")
    if not failures:
        print("  ALL PASSED")
    else:
        for f in failures:
            print(f"  FAIL: {f}")
        print("\n  ** Nonstationarity evidence detected. **")

    # ── Step 6: Diagnostic plots ──────────────────────────────────────

    print("\nGenerating diagnostic plots...")
    plot_diagnostics(
        all_merged, early_cat, late_cat, pooled_metrics, args.output_dir
    )
    plot_pos_scatter_by_session_bank(all_merged, args.output_dir)

    # ── Step 7: Baseline comparison (if provided) ─────────────────────

    if args.baseline_dir:
        print("\nComparing to full-session baseline...")
        for g in GROUPS:
            col = f"unique_{g}_shuffle"
            if col not in early_cat.columns:
                continue

            # Pool half-session medians
            med_early_pooled = float(early_cat[col].median())
            med_late_pooled = float(late_cat[col].median())

            # Load baseline
            base_vals = []
            for sess_key in sessions_to_run:
                out_name = SESSION_OUTPUT_DIRS[sess_key]
                base_path = os.path.join(
                    args.baseline_dir, out_name, "encoding",
                    "fig2b_unique_variance.parquet",
                )
                if os.path.isfile(base_path):
                    bdf = pd.read_parquet(base_path)
                    if col in bdf.columns:
                        base_vals.extend(bdf[col].dropna().tolist())

            if base_vals:
                med_full = float(np.median(base_vals))
                if abs(med_full) >= 1e-4:
                    ratio_e = med_early_pooled / med_full * 100
                    ratio_l = med_late_pooled / med_full * 100
                    print(f"  {g.upper():10s}  full={med_full:.4f}  "
                          f"early={ratio_e:.1f}%  late={ratio_l:.1f}%  of full")
                else:
                    print(f"  {g.upper():10s}  full={med_full:.4f} (near zero, skip ratio)")

    print("\nDone!")


if __name__ == "__main__":
    main()
