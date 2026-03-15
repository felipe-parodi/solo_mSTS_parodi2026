"""Population and per-bank summary plots for Step-A results.

Generates distribution plots, significant-count bars with bootstrap CIs,
per-bank ECDFs, and scatter summaries from the saved metrics parquet.

Usage:
    python -m freemat.single_fm.plots_summary \
        --metrics results/stepA_metrics.parquet \
        --outdir results/stepA_summary \
        --thresh-look 0.01 --thresh-move 0.0 --thresh-align 0.0 \
        [--match-units]

Notes:
    - "Significant" is operationalized as metric > threshold; for robust
      inference prefer permutation/null modeling.
    - When --match-units is set, per-bank comparisons are downsampled to the
      minimum available units across banks (random, fixed seed).
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from freemat.single_fm.io.save import save_png_svg


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _bootstrap_ci(x: np.ndarray, n: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap median and its percentile CI.

    Uses bootstrap of median (not mean) to ensure the point estimate lies
    between the CI bounds; this avoids negative yerr in plotting when using the
    median as the summary statistic.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    boots = np.array([np.nanmedian(rng.choice(x, size=x.size, replace=True)) for _ in range(n)], dtype=float)
    med = float(np.nanmedian(boots))
    lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return med, lo, hi


def _match_units_per_bank(df: pd.DataFrame, bank_col: str = "bank", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = df[bank_col].value_counts(dropna=False)
    if sizes.empty:
        return df.copy()
    n = int(sizes.min())
    parts = []
    for b, g in df.groupby(bank_col, sort=False):
        if len(g) <= n:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
            parts.append(g.loc[idx].copy())
    out = pd.concat(parts, axis=0, ignore_index=False).sort_index()
    return out


def plot_metric_distributions(metrics: pd.DataFrame, metric: str, out_path: str, *, match_units: bool = False) -> None:
    if metric not in metrics.columns:
        return
    df = metrics[["bank", metric]].dropna().copy()
    if df.empty:
        return
    if match_units:
        df = _match_units_per_bank(df)
    banks = list(df["bank"].dropna().unique())
    data = [df.loc[df["bank"] == b, metric].to_numpy(float) for b in banks]

    _ensure_dir(out_path)
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#6baed6")
        pc.set_edgecolor("#3182bd")
        pc.set_alpha(0.5)
    # Add boxplot overlay (quartiles)
    ax.boxplot(data, widths=0.15, showfliers=False, medianprops={"color": "black"})
    # Add bootstrapped medians + CI as errorbars
    for i, arr in enumerate(data, start=1):
        med, lo, hi = _bootstrap_ci(arr)
        # Ensure non-negative error bars
        yerr = np.array([[max(0.0, med - lo)], [max(0.0, hi - med)]], dtype=float)
        ax.errorbar(i, med, yerr=yerr, fmt="o", color="black", capsize=3)
    ax.set_xticks(range(1, len(banks) + 1))
    ax.set_xticklabels(banks)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: per-bank distributions")
    fig.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def _binomial_ci_wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for binomial proportion.

    Returns (lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    from math import sqrt
    z = 1.959963984540054  # ~95%
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def plot_sig_counts(metrics: pd.DataFrame, metric: str, thresh: float, out_path: str, *, match_units: bool = False) -> None:
    if metric not in metrics.columns:
        return
    df = metrics[["bank", metric]].dropna().copy()
    if df.empty:
        return
    if match_units:
        df = _match_units_per_bank(df)
    banks = list(df["bank"].dropna().unique())
    props = []
    ci_l = []
    ci_h = []
    use_perm = False
    for b in banks:
        db = df.loc[df["bank"] == b]
        # Prefer permutation-based significance if available
        sig_col = None
        if metric == "r2_look_still" and "sig_look_still" in metrics.columns:
            sig_col = "sig_look_still"
        elif metric == "move_unique" and "sig_move_unique" in metrics.columns:
            sig_col = "sig_move_unique"
        elif metric == "r2_align" and "sig_align" in metrics.columns:
            sig_col = "sig_align"
        if sig_col is not None:
            use_perm = True
            xb = metrics.loc[db.index, sig_col].astype(bool).to_numpy().astype(int)
        else:
            xb = (db[metric].to_numpy(float) > float(thresh)).astype(int)
        n = int(xb.size)
        k = int(xb.sum())
        p = (k / n) if n > 0 else float("nan")
        lo, hi = _binomial_ci_wilson(k, n, alpha=0.05)
        props.append(p)
        ci_l.append(max(0.0, p - lo))
        ci_h.append(max(0.0, hi - p))
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    ypos = np.arange(len(banks))
    ax.bar(ypos, props, color="#74c476", alpha=0.8)
    ax.errorbar(ypos, props, yerr=[ci_l, ci_h], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(ypos)
    ax.set_xticklabels(banks)
    ax.set_ylim(0, 1)
    if use_perm:
        ax.set_ylabel("% units permutation-significant")
        ax.set_title(f"Significant counts (permutation): {metric}")
    else:
        ax.set_ylabel(f"% units with {metric} > {thresh}")
        ax.set_title(f"Significant counts: {metric}")
    fig.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def plot_ecdf(metrics: pd.DataFrame, metric: str, out_path: str, *, match_units: bool = False) -> None:
    if metric not in metrics.columns:
        return
    df = metrics[["bank", metric]].dropna().copy()
    if df.empty:
        return
    if match_units:
        df = _match_units_per_bank(df)
    banks = list(df["bank"].dropna().unique())
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    for b in banks:
        x = np.sort(df.loc[df["bank"] == b, metric].to_numpy(float))
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=b)
    ax.set_xlabel(metric)
    ax.set_ylabel("ECDF")
    ax.set_title(f"ECDF by bank: {metric}")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def plot_scatter_move_vs_look(metrics: pd.DataFrame, out_path: str, *, match_units: bool = False) -> None:
    cols = ["move_unique", "r2_look_still", "bank"]
    if not set(cols).issubset(metrics.columns):
        return
    df = metrics[cols].dropna().copy()
    if df.empty:
        return
    if match_units:
        df = _match_units_per_bank(df)
    banks = list(df["bank"].dropna().unique())
    colors = {b: c for b, c in zip(banks, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]) }
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    for b in banks:
        g = df[df["bank"] == b]
        ax.scatter(g["r2_look_still"], g["move_unique"], s=10, alpha=0.6, color=colors[b], label=b)
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("r2_look_still")
    ax.set_ylabel("move_unique")
    ax.set_title("Unit metrics: move vs look")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def plot_unit_heatmap(metrics: pd.DataFrame, out_path: str, *, per_bank: bool = True, max_units: int | None = 300) -> None:
    cols = ["unit_id", "bank", "r2_look_still", "move_unique", "r2_align"]
    if not set(cols).issubset(metrics.columns):
        return
    df = metrics[cols].dropna().copy()
    if df.empty:
        return
    # Sort units by bank then by move_unique desc
    df = df.sort_values(["bank", "move_unique"], ascending=[True, False])
    if isinstance(max_units, int) and max_units > 0 and len(df) > max_units:
        df = df.head(max_units)
    M = df[["r2_look_still", "move_unique", "r2_align"]].to_numpy(float)
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(6, max(3, 0.1 * len(df))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["r2_look_still", "move_unique", "r2_align"], rotation=45, ha="right")
    ax.set_title("Per-unit metrics (sorted by bank, move_unique)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("metric value")
    fig.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def plot_top_units_bar(metrics: pd.DataFrame, out_path: str, *, metric: str = "move_unique", top_n: int = 10) -> None:
    if not set(["unit_id", "bank", metric]).issubset(metrics.columns):
        return
    df = metrics[["unit_id", "bank", metric]].dropna().copy()
    if df.empty:
        return
    g = df.sort_values(["bank", metric], ascending=[True, False]).groupby("bank", sort=False).head(top_n)
    _ensure_dir(out_path)
    # build labels as bank:unit
    g["label"] = g["bank"].astype(str) + ":" + g["unit_id"].astype(str)
    fig = plt.figure(figsize=(min(12, 0.3 * len(g)), 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(len(g)), g[metric].to_numpy(float), color="#9ecae1")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["label"].tolist(), rotation=90)
    ax.set_ylabel(metric)
    ax.set_title(f"Top {top_n} units per bank by {metric}")
    fig.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Step-A population summary plots")
    ap.add_argument("--metrics", required=True, help="Path to stepA_metrics.parquet")
    ap.add_argument("--outdir", required=True, help="Output directory for summary plots")
    ap.add_argument("--thresh-look", type=float, default=0.01, help="Threshold for r2_look_still significant counts")
    ap.add_argument("--thresh-move", type=float, default=0.0, help="Threshold for move_unique significant counts")
    ap.add_argument("--thresh-align", type=float, default=0.0, help="Threshold for r2_align significant counts")
    ap.add_argument("--match-units", action="store_true", help="Downsample banks to equal number of units for comparisons")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    M = pd.read_parquet(args.metrics)
    # Distributions
    plot_metric_distributions(M, "r2_look_still", os.path.join(args.outdir, "dist_r2_look_still.png"), match_units=args.match_units)
    plot_metric_distributions(M, "move_unique", os.path.join(args.outdir, "dist_move_unique.png"), match_units=args.match_units)
    plot_metric_distributions(M, "r2_align", os.path.join(args.outdir, "dist_r2_align.png"), match_units=args.match_units)
    # Significant counts with CIs
    plot_sig_counts(M, "r2_look_still", float(args.thresh_look), os.path.join(args.outdir, "sig_r2_look_still.png"), match_units=args.match_units)
    plot_sig_counts(M, "move_unique", float(args.thresh_move), os.path.join(args.outdir, "sig_move_unique.png"), match_units=args.match_units)
    plot_sig_counts(M, "r2_align", float(args.thresh_align), os.path.join(args.outdir, "sig_r2_align.png"), match_units=args.match_units)
    # ECDFs
    plot_ecdf(M, "r2_look_still", os.path.join(args.outdir, "ecdf_r2_look_still.png"), match_units=args.match_units)
    plot_ecdf(M, "move_unique", os.path.join(args.outdir, "ecdf_move_unique.png"), match_units=args.match_units)
    plot_ecdf(M, "r2_align", os.path.join(args.outdir, "ecdf_r2_align.png"), match_units=args.match_units)
    # Scatter summary
    plot_scatter_move_vs_look(M, os.path.join(args.outdir, "scatter_move_vs_look.png"), match_units=args.match_units)
    # Heatmap and top units
    plot_unit_heatmap(M, os.path.join(args.outdir, "heatmap_units.png"))
    plot_top_units_bar(M, os.path.join(args.outdir, "top_units_move_unique.png"), metric="move_unique", top_n=10)


if __name__ == "__main__":
    main()
