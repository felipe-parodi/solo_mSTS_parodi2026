#!/usr/bin/env python
"""Batch runner for Figure 3: Reference Frame Encoding across sessions.

Compares allocentric vs body-centric vs head-centric encoding of COM velocity
and displacement-to-center features.

Usage:
    python scripts/single_fm/run_fig3_all_sessions.py \
        --output-dir results/single_fm/fig3_ref_frames \
        --sessions hedy_0526 M2_0605

    python scripts/single_fm/run_fig3_all_sessions.py \
        --output-dir results/single_fm/fig3_ref_frames \
        --bin-s 0.25 --cv-folds 5
"""

from __future__ import annotations

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Session configuration (same as Fig 2B)
SESSIONS: Dict[str, str] = {
    "M1_0525": "configs/single_fm/session_M1_0525.yaml",
    "hedy_0526": "configs/single_fm/session_hedy_0526.yaml",
    "M2_0605": "configs/single_fm/session_M2_0605.yaml",
    "M1_0609": "configs/single_fm/session_M1_0609.yaml",
    "M1_0613": "configs/single_fm/session_M1_0613.yaml",
}

SESSION_OUTPUT_DIRS: Dict[str, str] = {
    "M1_0525": "single_0525_M1",
    "hedy_0526": "single_0526_hedy",
    "M2_0605": "single_0605_M2",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch runner for Figure 3: Reference Frame Encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--sessions", nargs="+", choices=list(SESSIONS.keys()), default=None,
        help="Sessions to run (default: all)",
    )
    parser.add_argument("--bin-s", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--no-time-drift", action="store_true")
    parser.add_argument(
        "--floor-filter", action="store_true", default=True,
        help="Filter to time bins where monkey is on the floor (default: True)",
    )
    parser.add_argument(
        "--no-floor-filter", action="store_true",
        help="Disable floor filtering (include all time bins)",
    )
    parser.add_argument(
        "--floor-height-threshold", type=float, default=0.5,
        help="Max height above floor (m) for floor filter (default: 0.5)",
    )
    parser.add_argument(
        "--floor-filter-before-features", action="store_true",
        help="NaN-mask off-floor frames BEFORE feature computation",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--alpha-mode", choices=["per_frame", "shared_allo"],
        default="per_frame",
        help="Alpha selection mode: per_frame (independent nested CV per frame, "
             "default) or shared_allo (allocentric-selected alpha reused for all)",
    )
    parser.add_argument(
        "--canonical-alignment", type=str, default=None,
        help="Path to canonical_alignment.json for cross-session arena alignment",
    )

    return parser.parse_args()


def main() -> None:
    """Run Fig 3 batch analysis."""
    args = parse_args()

    from freemat.single_fm.encoding.run_fig3 import (
        Fig3Config,
        run_fig3_encoding,
        aggregate_fig3_results,
    )

    # Determine sessions
    session_keys = args.sessions or list(SESSIONS.keys())

    # Resolve floor filter flag conflicts
    floor_filter = args.floor_filter and not args.no_floor_filter

    # Build config
    config = Fig3Config(
        bin_s=args.bin_s,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        include_time_drift=not args.no_time_drift,
        floor_filter=floor_filter,
        floor_height_threshold=args.floor_height_threshold,
        floor_filter_before_features=args.floor_filter_before_features,
        debug=args.debug,
        alpha_mode=args.alpha_mode,
    )

    print("\n[Fig3-Batch] Reference Frame Encoding")
    print(f"[Fig3-Batch] Sessions: {', '.join(session_keys)}")
    print(f"[Fig3-Batch] Config: bin_s={config.bin_s}, cv_folds={config.cv_folds}, "
          f"drift={'ON' if config.include_time_drift else 'OFF'}, "
          f"floor_filter={'ON' if config.floor_filter else 'OFF'}, "
          f"alpha_mode={config.alpha_mode}")
    print()

    # Run each session
    session_dfs: list = []
    total_start = time.time()

    for skey in session_keys:
        yaml_path = SESSIONS[skey]
        out_subdir = SESSION_OUTPUT_DIRS[skey]
        session_output = str(Path(args.output_dir) / out_subdir)

        print(f"{'=' * 60}")
        print(f"[Fig3-Batch] Processing: {skey}")
        print(f"[Fig3-Batch] YAML: {yaml_path}")
        print(f"[Fig3-Batch] Output: {session_output}")
        print(f"{'=' * 60}")

        t0 = time.time()
        df = run_fig3_encoding(
            yaml_path, session_output,
            config=config,
            canonical_alignment_path=args.canonical_alignment,
        )
        elapsed = time.time() - t0

        if len(df) > 0:
            session_dfs.append(df)
            print(f"[Fig3-Batch] {skey}: {len(df)} units in {elapsed:.1f}s")

            # Quick summary
            r2_med = df["r2_base"].median()
            winner_counts = df["winner_frame"].value_counts().to_dict()
            print(f"  Median R2 base: {r2_med:.4f}")
            print(f"  Median delta_r2: allo={df['delta_r2_allo'].median():.4f}, "
                  f"body={df['delta_r2_body'].median():.4f}, "
                  f"head={df['delta_r2_head'].median():.4f}")
            print(f"  Winners: {winner_counts}")
        else:
            print(f"[Fig3-Batch] {skey}: FAILED or no valid units")

        print()

    total_elapsed = time.time() - total_start

    # Aggregate
    if session_dfs:
        combined = aggregate_fig3_results(session_dfs)
        summary_dir = Path(args.output_dir) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        combined.to_parquet(summary_dir / "fig3_summary.parquet", index=False)

        # Generate figures
        fig_dir = summary_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        _generate_summary_figures(combined, fig_dir)

        # Print summary
        print(f"{'=' * 60}")
        print("[Fig3-Batch] SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Total units: {len(combined)}")
        print(f"Sessions: {combined['session_id'].nunique()}")
        print()

        # R2 summary
        for frame in ["allo", "body", "head"]:
            col = f"delta_r2_{frame}"
            med = combined[col].median()
            pos_frac = (combined[col] > 0).mean() * 100
            print(f"  delta_r2_{frame}: median={med:.4f}, >0: {pos_frac:.1f}%")

        # Winner summary
        winner_counts = combined["winner_frame"].value_counts()
        total = len(combined)
        print()
        for frame, count in winner_counts.items():
            print(f"  Winner {frame}: {count} ({100*count/total:.1f}%)")

        # Bank breakdown
        if "unit_bank" in combined.columns:
            print()
            for bank, bdf in combined.groupby("unit_bank"):
                winners = bdf["winner_frame"].value_counts()
                print(f"  {bank} ({len(bdf)} units):", end="")
                for frame, count in winners.items():
                    print(f" {frame}={count}", end="")
                print()

        print(f"\n[Fig3-Batch] Saved: {summary_dir / 'fig3_summary.parquet'}")
        print(f"[Fig3-Batch] Figures: {fig_dir}")

    print("\n[Fig3-Batch] Done.")


def _generate_summary_figures(
    df: "pd.DataFrame",  # noqa: F821
    fig_dir: Path,
) -> None:
    """Generate summary figures for Fig 3."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    frame_colors = {"allo": "#457B9D", "body": "#E63946", "head": "#F4A261"}
    bank_colors = {"dorsal": "#2D3142", "ventral": "#BFC0C0"}
    frames_ordered = ["allo", "body", "head"]
    frame_labels = {"allo": "Allo", "body": "Body", "head": "Head"}
    has_banks = "unit_bank" in df.columns and df["unit_bank"].nunique() >= 2

    def _save(fig: plt.Figure, name: str) -> None:
        fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / f"{name}.svg", bbox_inches='tight')
        plt.close(fig)

    # ================================================================
    # Panel 1: Scatter colored by bank
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [("allo", "body"), ("allo", "head")]
    for ax, (xf, yf) in zip(axes, pairs):
        x = df[f"delta_r2_{xf}"].values
        y = df[f"delta_r2_{yf}"].values
        fin = np.isfinite(x) & np.isfinite(y)

        if has_banks:
            for bank, color in bank_colors.items():
                mask = (df["unit_bank"] == bank).values & fin
                ax.scatter(x[mask], y[mask], s=10, alpha=0.35, color=color,
                           label=bank.capitalize(), rasterized=True, zorder=2)
        else:
            ax.scatter(x[fin], y[fin], s=8, alpha=0.4, color="#555555",
                       rasterized=True, zorder=2)

        lim = max(abs(np.nanmin([x[fin], y[fin]])),
                  abs(np.nanmax([x[fin], y[fin]]))) * 1.1
        lim = max(lim, 0.01)
        ax.set_xlim(-lim * 0.05, lim)
        ax.set_ylim(-lim * 0.05, lim)
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, lw=1, zorder=1)
        ax.axhline(0, color='grey', lw=0.5, alpha=0.3, zorder=0)
        ax.axvline(0, color='grey', lw=0.5, alpha=0.3, zorder=0)
        ax.set_xlabel(f"$\\Delta R^2$ {frame_labels[xf]}")
        ax.set_ylabel(f"$\\Delta R^2$ {frame_labels[yf]}")

        from scipy.stats import spearmanr
        r, p = spearmanr(x[fin], y[fin])
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"\u03c1 = {r:.2f}, {p_str}",
                transform=ax.transAxes, fontsize=11, va="top")

        # Count above/below diagonal — make prominent
        n_above = int((y[fin] > x[fin]).sum())
        n_below = int((y[fin] <= x[fin]).sum())
        ax.text(0.95, 0.05,
                f"{frame_labels[xf]} wins: {n_below} ({100*n_below/fin.sum():.0f}%)\n"
                f"{frame_labels[yf]} wins: {n_above} ({100*n_above/fin.sum():.0f}%)",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='grey', alpha=0.8))

    if has_banks:
        axes[1].legend(loc="upper left", fontsize=10, markerscale=2,
                       framealpha=0.8)
    fig.suptitle("Reference Frame Comparison", fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, "fig3b_scatter")

    # ================================================================
    # Panel 2: Winner bars — overall + by bank (side-by-side)
    # ================================================================
    if has_banks:
        from scipy.stats import chi2_contingency

        banks = sorted(df["unit_bank"].unique())
        n_banks = len(banks)
        fig, axes = plt.subplots(1, n_banks + 1,
                                 figsize=(4 * (n_banks + 1), 4.5),
                                 sharey=True)

        # Overall
        ax = axes[0]
        total = len(df)
        counts = [int((df["winner_frame"] == f).sum()) for f in frames_ordered]
        colors = [frame_colors[f] for f in frames_ordered]
        bars = ax.bar([frame_labels[f] for f in frames_ordered], counts,
                      color=colors, edgecolor="black", linewidth=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{c}\n({100*c/total:.0f}%)", ha='center', va='bottom',
                    fontsize=9)
        ax.set_ylabel("Number of neurons")
        ax.set_title(f"All (n={total})", fontweight='bold')

        # Per bank
        for i, bank in enumerate(banks):
            ax = axes[i + 1]
            bdf = df[df["unit_bank"] == bank]
            n_bank = len(bdf)
            counts_b = [int((bdf["winner_frame"] == f).sum())
                        for f in frames_ordered]
            bars = ax.bar([frame_labels[f] for f in frames_ordered], counts_b,
                          color=colors, edgecolor="black", linewidth=0.5)
            for bar, c in zip(bars, counts_b):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{c}\n({100*c/n_bank:.0f}%)", ha='center',
                        va='bottom', fontsize=9)
            ax.set_title(f"{bank.capitalize()} (n={n_bank})", fontweight='bold')

        # Chi-squared test
        contingency = np.array([
            [(df.loc[df["unit_bank"] == bank, "winner_frame"] == f).sum()
             for f in frames_ordered]
            for bank in banks
        ])
        if contingency.min() > 0:
            chi2, pval, dof, _ = chi2_contingency(contingency)
            p_str = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
            fig.suptitle(
                f"Best-encoding frame by bank "
                f"(\u03c7\u00b2({dof}) = {chi2:.1f}, {p_str})",
                fontweight='bold', y=1.04)
        else:
            fig.suptitle("Best-encoding frame by bank",
                         fontweight='bold', y=1.04)

        fig.tight_layout()
        _save(fig, "fig3_winner_bars_by_bank")
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        total = len(df)
        counts = [int((df["winner_frame"] == f).sum()) for f in frames_ordered]
        colors = [frame_colors[f] for f in frames_ordered]
        bars = ax.bar([frame_labels[f] for f in frames_ordered], counts,
                      color=colors, edgecolor="black", linewidth=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{c}\n({100*c/total:.0f}%)", ha='center', va='bottom',
                    fontsize=10)
        ax.set_ylabel("Number of neurons")
        ax.set_title("Best-encoding reference frame", fontweight='bold')
        fig.tight_layout()
        _save(fig, "fig3_winner_bars")

    # ================================================================
    # Panel 3: delta_r2 violins by bank (with clearer labels)
    # ================================================================
    if has_banks:
        from scipy.stats import mannwhitneyu

        banks = sorted(df["unit_bank"].unique())
        fig, axes = plt.subplots(1, len(frames_ordered), figsize=(12, 4.5),
                                 sharey=True)

        for ax, frame in zip(axes, frames_ordered):
            col = f"delta_r2_{frame}"
            data_by_bank = []
            for bank in banks:
                vals = df.loc[df["unit_bank"] == bank, col].dropna().values
                data_by_bank.append(vals)

            vp = ax.violinplot(data_by_bank, positions=range(len(banks)),
                               showmedians=True, showextrema=False)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(frame_colors[frame])
                body.set_alpha(0.5 + 0.2 * i)
                body.set_edgecolor("black")
                body.set_linewidth(0.5)
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(1.5)

            ax.set_xticks(range(len(banks)))
            ax.set_xticklabels([b.capitalize() for b in banks])
            ax.axhline(0, color='grey', lw=0.5, ls='--')
            ax.set_title(frame_labels[frame], fontweight='bold',
                         color=frame_colors[frame])

            # Mann-Whitney U between banks (spelled out)
            if len(data_by_bank) == 2 and len(data_by_bank[0]) > 5:
                u_stat, u_pval = mannwhitneyu(
                    data_by_bank[0], data_by_bank[1],
                    alternative='two-sided')
                p_str = ("p < 0.001" if u_pval < 0.001
                         else f"p = {u_pval:.3f}")
                ax.text(0.5, 0.02, f"Mann-Whitney {p_str}",
                        transform=ax.transAxes, fontsize=9, ha="center",
                        va="bottom", style="italic", color="#444444")

            # Median annotations — offset above the median line
            for i, vals in enumerate(data_by_bank):
                med = np.median(vals)
                ax.annotate(f"{med:.3f}",
                            xy=(i, med), xytext=(0, 8),
                            textcoords="offset points",
                            fontsize=9, ha="center", fontweight="bold")

        axes[0].set_ylabel("$\\Delta R^2$ (frame \u2212 base)")
        fig.suptitle("Unique variance by reference frame and bank",
                     fontweight='bold', y=1.02)
        fig.tight_layout()
        _save(fig, "fig3_delta_r2_violin_by_bank")

    # ================================================================
    # Panel 4: Boxplots by bank (requested in addition to violins)
    # ================================================================
    if has_banks:
        banks = sorted(df["unit_bank"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = []
        box_data = []
        box_colors = []
        tick_labels = []
        group_centers = []

        for gi, frame in enumerate(frames_ordered):
            col = f"delta_r2_{frame}"
            base = gi * (len(banks) + 1)
            group_centers.append(base + (len(banks) - 1) / 2)
            for bi, bank in enumerate(banks):
                vals = df.loc[df["unit_bank"] == bank, col].dropna().values
                box_data.append(vals)
                positions.append(base + bi)
                box_colors.append(bank_colors[bank])
                tick_labels.append(f"{bank[:1].upper()}")

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)

        ax.set_xticks(group_centers)
        ax.set_xticklabels([frame_labels[f] for f in frames_ordered],
                           fontsize=12)
        ax.axhline(0, color='grey', lw=0.5, ls='--')
        ax.set_ylabel("$\\Delta R^2$ (frame \u2212 base)")
        ax.set_title("Unique variance by frame and bank", fontweight='bold')

        # Manual legend
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=bank_colors[b], edgecolor='black',
                                label=b.capitalize()) for b in banks]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=11)

        fig.tight_layout()
        _save(fig, "fig3_delta_r2_boxplot_by_bank")

    # ================================================================
    # Panel 5: Overall delta_r2 boxplots
    # ================================================================
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [frame_labels[f] for f in frames_ordered]
    data = [df[f"delta_r2_{f}"].dropna().values for f in frames_ordered]
    colors = [frame_colors[f] for f in frames_ordered]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_ylabel("$\\Delta R^2$ (frame \u2212 base)")
    ax.set_title("Unique variance by reference frame", fontweight='bold')
    fig.tight_layout()
    _save(fig, "fig3_delta_r2_boxplot")

    # ================================================================
    # Panel 6: Allocentric advantage — paired difference by neuron
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, other in zip(axes, ["body", "head"]):
        advantage = (df["delta_r2_allo"] - df[f"delta_r2_{other}"]).dropna()
        advantage_sorted = advantage.sort_values(ascending=False).values

        n = len(advantage_sorted)
        cols = np.where(advantage_sorted > 0, frame_colors["allo"],
                        frame_colors[other])
        ax.bar(range(n), advantage_sorted, width=1.0, color=cols,
               edgecolor="none", alpha=0.7)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel("Neurons (sorted)")
        ax.set_ylabel(
            f"$\\Delta R^2_{{allo}}$ \u2212 $\\Delta R^2_{{{other}}}$")
        n_allo_wins = int((advantage_sorted > 0).sum())
        ax.set_title(
            f"Allo vs {frame_labels[other]}: "
            f"{n_allo_wins}/{n} ({100*n_allo_wins/n:.0f}%) allo wins",
            fontweight='bold')

        # Add bank coloring if available
        if has_banks:
            idx_sorted = (df["delta_r2_allo"] - df[f"delta_r2_{other}"]
                          ).dropna().sort_values(ascending=False).index
            bank_vals = df.loc[idx_sorted, "unit_bank"].values
            for i, bk in enumerate(bank_vals):
                if bk == "ventral":
                    ax.plot(i, advantage_sorted[i], '.', color='white',
                            ms=1, zorder=3)

    fig.suptitle("Allocentric advantage per neuron", fontweight='bold',
                 y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_allo_advantage")

    # ================================================================
    # Panel 8: Normalized stacked bar — proportion per bank
    # ================================================================
    if has_banks:
        banks = sorted(df["unit_bank"].unique())
        fig, ax = plt.subplots(figsize=(5, 4.5))

        x_pos = np.arange(len(banks))
        bottom = np.zeros(len(banks))

        for frame in frames_ordered:
            proportions = []
            for bank in banks:
                bdf = df[df["unit_bank"] == bank]
                n_bank = len(bdf)
                n_frame = int((bdf["winner_frame"] == frame).sum())
                proportions.append(100 * n_frame / n_bank if n_bank > 0 else 0)
            bars = ax.bar(x_pos, proportions, bottom=bottom, width=0.55,
                          color=frame_colors[frame], edgecolor="black",
                          linewidth=0.5, label=frame_labels[frame])
            # Label each segment if > 3%
            for i, (p, b) in enumerate(zip(proportions, bottom)):
                if p > 3:
                    ax.text(x_pos[i], b + p / 2, f"{p:.0f}%",
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='white')
                elif p > 0:
                    # Small segments: label outside
                    ax.annotate(f"{p:.0f}%", xy=(x_pos[i], b + p),
                                xytext=(x_pos[i] + 0.35, b + p - 1),
                                fontsize=8, ha='left', fontweight='bold',
                                arrowprops=dict(arrowstyle='-', color='grey',
                                                lw=0.5))
            bottom += proportions

        ax.set_xticks(x_pos)
        ax.set_xticklabels([b.capitalize() for b in banks], fontsize=12)
        ax.set_ylabel("Proportion of neurons (%)")
        ax.set_ylim(0, 108)
        # Place legend outside the bars
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=10, framealpha=0.9)
        ax.set_title("Winner frame proportion by bank", fontweight='bold')
        fig.tight_layout()
        _save(fig, "fig3_stacked_proportion_by_bank")

    # ================================================================
    # Panel 9: Cleveland dot plot — bank effect per frame
    # ================================================================
    if has_banks:
        from scipy.stats import mannwhitneyu

        banks = sorted(df["unit_bank"].unique())
        fig, ax = plt.subplots(figsize=(6, 4))

        y_positions = np.arange(len(frames_ordered))
        for bi, bank in enumerate(banks):
            medians = []
            ci_lo = []
            ci_hi = []
            for frame in frames_ordered:
                vals = df.loc[df["unit_bank"] == bank,
                              f"delta_r2_{frame}"].dropna().values
                med = np.median(vals)
                # Bootstrap 95% CI for median
                rng = np.random.default_rng(42)
                boot_meds = [np.median(rng.choice(vals, size=len(vals)))
                             for _ in range(1000)]
                lo, hi = np.percentile(boot_meds, [2.5, 97.5])
                medians.append(med)
                ci_lo.append(med - lo)
                ci_hi.append(hi - med)

            offset = -0.12 + bi * 0.24
            ax.errorbar(medians, y_positions + offset,
                        xerr=[ci_lo, ci_hi],
                        fmt='o', markersize=8, capsize=4,
                        color=bank_colors[bank],
                        label=bank.capitalize(), zorder=3)

        # Connect pairs with lines
        for fi in range(len(frames_ordered)):
            frame = frames_ordered[fi]
            vals_d = df.loc[df["unit_bank"] == banks[0],
                            f"delta_r2_{frame}"].dropna().values
            vals_v = df.loc[df["unit_bank"] == banks[1],
                            f"delta_r2_{frame}"].dropna().values
            med_d = np.median(vals_d)
            med_v = np.median(vals_v)
            ax.plot([med_d, med_v],
                    [y_positions[fi] - 0.12, y_positions[fi] + 0.12],
                    'k-', lw=0.8, alpha=0.4)

            # Significance star
            u_stat, u_pval = mannwhitneyu(vals_d, vals_v,
                                          alternative='two-sided')
            if u_pval < 0.001:
                star = "***"
            elif u_pval < 0.01:
                star = "**"
            elif u_pval < 0.05:
                star = "*"
            else:
                star = "n.s."
            ax.text(max(med_d, med_v) + 0.005, y_positions[fi],
                    star, fontsize=11, va="center", fontweight="bold")

        ax.set_yticks(y_positions)
        ax.set_yticklabels([frame_labels[f] for f in frames_ordered],
                           fontsize=12)
        ax.set_xlabel("Median $\\Delta R^2$ (with 95% bootstrap CI)")
        ax.set_title("Bank effect on reference frame encoding",
                     fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.invert_yaxis()
        fig.tight_layout()
        _save(fig, "fig3_bank_effect_dotplot")

    # ================================================================
    # Panel 10: Strip plot — head frame focus by bank
    # ================================================================
    if has_banks:
        banks = sorted(df["unit_bank"].unique())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

        for ax, frame in zip(axes, frames_ordered):
            col = f"delta_r2_{frame}"
            for bi, bank in enumerate(banks):
                vals = df.loc[df["unit_bank"] == bank, col].dropna().values
                # Jitter
                rng = np.random.default_rng(42 + bi)
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(bi + jitter, vals, s=6, alpha=0.25,
                           color=bank_colors[bank], rasterized=True,
                           zorder=2)
                # Median marker
                med = np.median(vals)
                ax.plot([bi - 0.25, bi + 0.25], [med, med],
                        color=frame_colors[frame], lw=2.5, zorder=4)
                ax.text(bi, med + 0.008, f"{med:.3f}", ha='center',
                        fontsize=9, fontweight='bold',
                        color=frame_colors[frame],
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))

            ax.set_xticks(range(len(banks)))
            ax.set_xticklabels([b.capitalize() for b in banks])
            ax.axhline(0, color='grey', lw=0.5, ls='--')
            ax.set_title(frame_labels[frame], fontweight='bold',
                         color=frame_colors[frame], fontsize=13)

        axes[0].set_ylabel("$\\Delta R^2$ (frame \u2212 base)")
        fig.suptitle("Individual neuron encoding by bank",
                     fontweight='bold', y=1.02)
        fig.tight_layout()
        _save(fig, "fig3_strip_by_bank")

    # ================================================================
    # Panel 7: CDF of delta_r2 by frame
    # ================================================================
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for frame in frames_ordered:
        vals = df[f"delta_r2_{frame}"].dropna().values
        vals_sorted = np.sort(vals)
        cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
        ax.plot(vals_sorted, cdf, color=frame_colors[frame], lw=2,
                label=frame_labels[frame])
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel("$\\Delta R^2$ (frame \u2212 base)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("CDF of unique variance by frame", fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_delta_r2_cdf")

    print("[Fig3-Batch] Generated summary figures")


if __name__ == "__main__":
    main()
