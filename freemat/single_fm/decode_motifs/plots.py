"""Visualization helpers for motif decoding."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def _save_fig(fig: plt.Figure, path_no_ext: str) -> None:
    fig.tight_layout()
    fig.savefig(path_no_ext + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_motif_bars(metrics_by_bank: Dict[str, Dict[str, float]], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    banks = list(metrics_by_bank.keys())
    f1s = [metrics_by_bank[b].get("macro_f1", np.nan) for b in banks]
    bacc = [metrics_by_bank[b].get("balanced_acc", np.nan) for b in banks]
    x = np.arange(len(banks))
    w = 0.35
    ax.bar(x - w / 2, f1s, width=w, label="Macro-F1")
    ax.bar(x + w / 2, bacc, width=w, label="Balanced Acc")
    ax.set_xticks(x)
    ax.set_xticklabels([b.title() for b in banks])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    _save_fig(fig, os.path.join(out_dir, "motif_bars"))


def plot_confusion(C: np.ndarray, classes: np.ndarray, out_path: str, title: str = "") -> None:
    C = np.asarray(C, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(Cn, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([str(int(c)) for c in classes], rotation=90)
    ax.set_yticklabels([str(int(c)) for c in classes])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(pred|true)")
    _save_fig(fig, out_path)


def plot_weights_heatmap(W_u_h: np.ndarray, out_path: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(W_u_h, aspect="auto", cmap="magma")
    ax.set_xlabel("Lag (bins, recent→past)")
    ax.set_ylabel("Unit")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|weight|")
    _save_fig(fig, out_path)


def plot_pr_curve(y_true: np.ndarray, p: np.ndarray, out_path: str, title: str = "") -> None:
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(rec, prec, color="#1f77b4", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ttl = title or "PR curve"
    ax.set_title(f"{ttl} (AP={ap:.3f})")
    _save_fig(fig, out_path)
