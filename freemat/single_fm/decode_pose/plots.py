"""Visualisation helpers for pose decoding (Single-FM)."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    """Create directory tree if it doesn't exist."""
    os.makedirs(p, exist_ok=True)


def plot_r2_bars(
    r2_dim: np.ndarray,
    J_minus1: int,
    out_path: Optional[str],
    title: Optional[str] = None,
) -> None:
    """Bar chart of per-dimension R² scores."""
    if r2_dim.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(r2_dim.size), r2_dim, color="#4C78A8")
    ax.set_ylabel("R²")
    ax.set_xlabel("Joint-dim index")
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(0, color="#666", linewidth=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lag_curve(
    r2_by_lag: Dict[str, np.ndarray],
    alphas: np.ndarray,
    chosen_lag: int,
    out_path: Optional[str],
    title: Optional[str] = None,
) -> None:
    """Lag sweep curve (max R² over alpha per lag)."""
    if not r2_by_lag:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    lags = np.array([int(k) for k in r2_by_lag.keys()])
    lags_sorted = np.sort(lags)
    means = []
    for L in lags_sorted:
        vals = r2_by_lag[str(L)]
        means.append(np.nanmax(vals) if vals.size else np.nan)
    ax.plot(lags_sorted, means, marker="o", color="#72B7B2")
    ax.axvline(chosen_lag, color="#C44E52", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Lag (bins)")
    ax.set_ylabel("Val R² (max over α)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_null_violins(
    real_r2: float,
    null_r2: np.ndarray,
    out_path: Optional[str],
    title: Optional[str] = None,
) -> None:
    """Violin plot of null R² distribution with real value overlay."""
    fig, ax = plt.subplots(figsize=(4, 3))
    parts = ax.violinplot(null_r2[~np.isnan(null_r2)], showmeans=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#9ecae1")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    ax.scatter([1.1], [real_r2], color="#C44E52", zorder=3, label="real")
    ax.set_ylabel("Mean R²")
    ax.set_xticks([])
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
