"""Fig 3 reference-frame panel plotting functions.

Extracted from ``scripts/single_fm/generate_publication_figures.py`` (W5.2).
Each function creates a self-contained matplotlib Figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr

from freemat.single_fm.viz.pub_data import (
    FRAME_COLORS,
    FRAME_LABELS,
    FRAMES,
    save_fig,
    winsorize,
)

__all__: List[str] = [
    "fig3_panel_a_winner_bars",
    "fig3_panel_b_boxplots",
    "fig3_panel_c_scatter",
    "fig3_panel_d_cdf",
    "fig3_panel_e_bank_dissociation",
    "fig3_pie_charts",
    "fig3_supp_r2_filtered",
]


def fig3_panel_a_winner_bars(
    df: pd.DataFrame, out_dir: Path, tag: str = ""
) -> None:
    """Fig 3A: Winner frame bar chart (all + by subject).

    Args:
        df: Summary DataFrame with ``winner_frame`` and ``subject`` columns.
        out_dir: Output directory.
        tag: Filename suffix.
    """
    subjects = ["M2", "M1"]
    subsets = [("All", df)] + [(s, df[df["subject"] == s]) for s in subjects]

    fig, axes = plt.subplots(1, len(subsets), figsize=(4.5 * len(subsets), 4.5),
                             sharey=True)
    for ax, (label, sdf) in zip(axes, subsets):
        n = len(sdf)
        if n == 0:
            continue
        counts = [int((sdf["winner_frame"] == f).sum()) for f in FRAMES]
        bars = ax.bar(
            [FRAME_LABELS[f] for f in FRAMES], counts,
            color=[FRAME_COLORS[f] for f in FRAMES],
            edgecolor="black", linewidth=0.5,
        )
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{c}\n({100*c/n:.0f}%)", ha="center", va="bottom",
                    fontsize=9)
        ax.set_title(f"{label} (n={n})", fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Number of neurons")

    fig.suptitle("Winner Reference Frame", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"fig3A_winner_bars{tag}", out_dir)


def fig3_panel_b_boxplots(
    df: pd.DataFrame, out_dir: Path, tag: str = ""
) -> None:
    """Fig 3B: Delta R2 boxplots by frame.

    Args:
        df: Summary DataFrame with ``delta_r2_*`` columns.
        out_dir: Output directory.
        tag: Filename suffix.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [winsorize(df[f"delta_r2_{f}"].dropna().values) for f in FRAMES]
    bp = ax.boxplot(data, labels=[FRAME_LABELS[f] for f in FRAMES],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, f in zip(bp["boxes"], FRAMES):
        patch.set_facecolor(FRAME_COLORS[f])
        patch.set_alpha(0.7)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("$\\Delta R^2$ (frame $-$ base)")
    ax.set_title(f"Reference Frame Encoding (n={len(df)})", fontweight="bold")

    for i, f in enumerate(FRAMES):
        med = np.median(df[f"delta_r2_{f}"].dropna().values)
        ax.text(i + 1, med + 0.003, f"{med:.3f}", ha="center",
                fontsize=9, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, f"fig3B_boxplot{tag}", out_dir)


def fig3_panel_c_scatter(
    df: pd.DataFrame, out_dir: Path, tag: str = ""
) -> None:
    """Fig 3C: Allo vs Body scatter.

    Args:
        df: Summary DataFrame with ``delta_r2_allo`` and ``delta_r2_body``.
        out_dir: Output directory.
        tag: Filename suffix.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    x = winsorize(df["delta_r2_allo"].values)
    y = winsorize(df["delta_r2_body"].values)
    fin = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[fin], y[fin], s=8, alpha=0.3, color="#457B9D", rasterized=True)
    lim = max(np.nanmax(x[fin]), np.nanmax(y[fin])) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel("$\\Delta R^2$ Allocentric")
    ax.set_ylabel("$\\Delta R^2$ Body-centric")
    r, p = spearmanr(df["delta_r2_allo"].values[fin], df["delta_r2_body"].values[fin])
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.05, 0.95, f"$\\rho$ = {r:.2f}, {p_str}",
            transform=ax.transAxes, fontsize=11, va="top")
    n_allo = int((x[fin] > y[fin]).sum())
    ax.text(0.95, 0.05,
            f"Allo wins: {n_allo}/{fin.sum()} ({100*n_allo/fin.sum():.0f}%)",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))
    ax.set_title(f"Allo vs Body (n={fin.sum()})", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"fig3C_scatter_allo_body{tag}", out_dir)


def fig3_panel_d_cdf(
    df: pd.DataFrame, out_dir: Path, tag: str = ""
) -> None:
    """Fig 3D: CDF of delta R2 by frame.

    Args:
        df: Summary DataFrame with ``delta_r2_*`` columns.
        out_dir: Output directory.
        tag: Filename suffix.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for f in FRAMES:
        vals = df[f"delta_r2_{f}"].dropna().values
        vals_sorted = np.sort(vals)
        cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
        ax.plot(vals_sorted, cdf, color=FRAME_COLORS[f],
                label=FRAME_LABELS[f], linewidth=2)
    ax.set_xlabel("$\\Delta R^2$ (frame $-$ base)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of Reference Frame Encoding", fontweight="bold")
    ax.legend(frameon=False)
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    save_fig(fig, f"fig3D_cdf{tag}", out_dir)


def fig3_panel_e_bank_dissociation(
    df: pd.DataFrame, out_dir: Path, tag: str = ""
) -> None:
    """Fig 3E: Bank dissociation panels (stacked proportion + dot plot).

    Args:
        df: Summary DataFrame with ``unit_bank``, ``winner_frame``, ``delta_r2_*``.
        out_dir: Output directory.
        tag: Filename suffix.
    """
    if "unit_bank" not in df.columns:
        return

    banks = ["ventral", "dorsal"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: Stacked proportion bar
    ax = axes[0]
    for i, bank in enumerate(banks):
        bdf = df[df["unit_bank"] == bank]
        n = len(bdf)
        if n == 0:
            continue
        bottom = 0
        for f in FRAMES:
            pct = (bdf["winner_frame"] == f).sum() / n * 100
            ax.bar(i, pct, bottom=bottom, color=FRAME_COLORS[f],
                   edgecolor="white", linewidth=0.5)
            if pct > 3:
                ax.text(i, bottom + pct / 2, f"{pct:.1f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if f == "allo" else "black")
            bottom += pct

    ax.set_xticks(range(len(banks)))
    ax.set_xticklabels([f"{b.title()}\n(n={len(df[df.unit_bank == b])})" for b in banks])
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Winner Proportions by Bank", fontweight="bold")
    legend_patches = [Patch(facecolor=FRAME_COLORS[f], label=FRAME_LABELS[f])
                      for f in FRAMES]
    ax.legend(handles=legend_patches, frameon=False, loc="upper right")

    # Panel 2: Median delta R2 by bank (Cleveland dot plot)
    ax = axes[1]
    for i, f in enumerate(FRAMES):
        for j, bank in enumerate(banks):
            bdf = df[df["unit_bank"] == bank]
            vals = bdf[f"delta_r2_{f}"].dropna().values
            med = np.median(vals)
            n_boot = 1000
            rng = np.random.default_rng(42)
            boot_meds = [np.median(rng.choice(vals, size=len(vals), replace=True))
                         for _ in range(n_boot)]
            ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])
            y_pos = i + (j - 0.5) * 0.25
            marker = "o" if bank == "ventral" else "^"
            ax.errorbar(med, y_pos, xerr=[[med - ci_lo], [ci_hi - med]],
                        fmt=marker, color=FRAME_COLORS[f], markersize=8,
                        capsize=4, linewidth=1.5)

        # Mann-Whitney test
        v_vals = df.loc[df["unit_bank"] == "ventral", f"delta_r2_{f}"].dropna()
        d_vals = df.loc[df["unit_bank"] == "dorsal", f"delta_r2_{f}"].dropna()
        _, mw_p = mannwhitneyu(v_vals, d_vals, alternative="two-sided")
        sig = "*" if mw_p < 0.05 else ""
        sig += "*" if mw_p < 0.01 else ""
        sig += "*" if mw_p < 0.001 else ""
        if sig:
            ax.text(max(np.median(v_vals), np.median(d_vals)) + 0.005,
                    i, sig, fontsize=14, va="center", fontweight="bold")

    ax.set_yticks(range(len(FRAMES)))
    ax.set_yticklabels([FRAME_LABELS[f] for f in FRAMES])
    ax.set_xlabel("Median $\\Delta R^2$")
    ax.set_title("Bank Comparison (95% CI)", fontweight="bold")
    ax.legend(
        [plt.Line2D([0], [0], marker="o", color="grey", linestyle=""),
         plt.Line2D([0], [0], marker="^", color="grey", linestyle="")],
        ["Ventral", "Dorsal"],
        frameon=False, loc="lower right",
    )

    fig.suptitle("Bank Dissociation", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"fig3E_bank_dissociation{tag}", out_dir)


def fig3_pie_charts(
    df: pd.DataFrame, out_dir: Path, tag: str = ""
) -> None:
    """Fig 3: Pie charts showing winner-frame proportions (pooled + by subject).

    Layout: left = All (large), right top = M1, right bottom = M2.

    Args:
        df: Summary DataFrame with ``winner_frame`` and ``subject`` columns.
        out_dir: Output directory.
        tag: Filename suffix.
    """
    from matplotlib.gridspec import GridSpec

    pie_colors = [FRAME_COLORS[f] for f in FRAMES]
    pie_labels = [FRAME_LABELS[f] for f in FRAMES]

    subsets = {
        "All": df,
        "M1": df[df["subject"] == "M1"],
        "M2": df[df["subject"] == "M2"],
    }
    counts = {}
    for key, sdf in subsets.items():
        counts[key] = [int((sdf["winner_frame"] == f).sum()) for f in FRAMES]

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Left: All (spans both rows)
    ax_all = fig.add_subplot(gs[:, 0])
    n_all = sum(counts["All"])
    wedges, texts, autotexts = ax_all.pie(
        counts["All"],
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax_all.set_title(f"All (n={n_all:,})", fontweight="bold", fontsize=14, pad=12)

    # Right top: M1
    ax_m1 = fig.add_subplot(gs[0, 1])
    n_m1 = sum(counts["M1"])
    w1, t1, a1 = ax_m1.pie(
        counts["M1"],
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    for at in a1:
        at.set_fontsize(10)
    ax_m1.set_title(f"M1 (n={n_m1:,})", fontweight="bold", fontsize=12, pad=8)

    # Right bottom: M2
    ax_m2 = fig.add_subplot(gs[1, 1])
    n_m2 = sum(counts["M2"])
    w2, t2, a2 = ax_m2.pie(
        counts["M2"],
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    for at in a2:
        at.set_fontsize(10)
    ax_m2.set_title(f"M2 (n={n_m2:,})", fontweight="bold", fontsize=12, pad=8)

    fig.suptitle("Winner Reference Frame", fontweight="bold", fontsize=14, y=1.02)
    save_fig(fig, f"fig3_pie_charts{tag}", out_dir)


def fig3_supp_r2_filtered(
    df: pd.DataFrame, out_dir: Path
) -> None:
    """Supplement: Fig 3 winner bars at different R2 thresholds.

    Args:
        df: Summary DataFrame with ``winner_frame`` and ``r2_base`` columns.
        out_dir: Output directory.
    """
    thresholds = [0.0, 0.005, 0.01, 0.02]
    fig, axes = plt.subplots(1, len(thresholds),
                             figsize=(4 * len(thresholds), 4.5), sharey=True)
    for ax, thr in zip(axes, thresholds):
        if "r2_base" in df.columns:
            sdf = df[df["r2_base"] >= thr]
        else:
            sdf = df
        n = len(sdf)
        if n == 0:
            ax.set_title(f"$R^2 \\geq$ {thr} (n=0)")
            continue
        counts = [int((sdf["winner_frame"] == f).sum()) for f in FRAMES]
        bars = ax.bar([FRAME_LABELS[f] for f in FRAMES], counts,
                      color=[FRAME_COLORS[f] for f in FRAMES],
                      edgecolor="black", linewidth=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{c}\n({100*c/n:.0f}%)", ha="center", va="bottom",
                    fontsize=9)
        ax.set_title(f"$R^2_{{base}} \\geq$ {thr}\n(n={n})", fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Number of neurons")

    fig.suptitle("Fig 3: Winner frame vs R$^2$ threshold",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    save_fig(fig, "fig3_supp_r2_filtered", out_dir)
