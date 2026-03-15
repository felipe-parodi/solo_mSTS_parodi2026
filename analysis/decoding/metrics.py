"""Evaluation metrics for neural decoding.

Provides balanced accuracy, F1 scores, and confusion matrix computation.

Core Metrics:

    Balanced Accuracy:
        BA = (1/C) × Σ_c Recall_c

        Where:
            - C = number of classes
            - Recall_c = TP_c / (TP_c + FN_c)

        Equivalent to macro-averaged recall. Handles class imbalance by
        weighting each class equally regardless of sample count.

    Macro F1 Score:
        F1_macro = (1/C) × Σ_c F1_c

        Where:
            F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
            Precision_c = TP_c / (TP_c + FP_c)
            Recall_c = TP_c / (TP_c + FN_c)

    Weighted F1 Score:
        F1_weighted = Σ_c (n_c / n) × F1_c

        Weights each class F1 by its sample proportion.

    Chance Level:
        Uniform: chance = 1/C
        Majority: chance = max(n_c) / n

Usage:
    from analysis.decoding.metrics import compute_metrics, DecodingMetrics

    metrics = compute_metrics(y_true, y_pred, class_names=['rest', 'locomotion'])
    print(f"Balanced accuracy: {metrics.balanced_accuracy:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


@dataclass
class DecodingMetrics:
    """Container for decoding evaluation metrics."""

    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_accuracy: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    class_names: List[str]
    n_samples: int
    n_classes: int


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> DecodingMetrics:
    """Compute comprehensive decoding metrics.

    Mathematical Definitions:
        Per-class accuracy (from confusion matrix):
            Acc_c = CM[c,c] / Σ_j CM[c,j] = TP_c / (TP_c + FN_c)

        See module docstring for balanced accuracy and F1 formulas.

    Args:
        y_true: Ground truth labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        class_names: Optional list of class names (inferred if not provided).

    Returns:
        DecodingMetrics with balanced_accuracy, f1_macro, f1_weighted,
        per_class_accuracy, per_class_f1, confusion_matrix, and metadata.
    """
    # Infer class names if not provided
    if class_names is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(c) for c in unique_labels]
    else:
        unique_labels = class_names

    n_classes = len(unique_labels)
    n_samples = len(y_true)

    # Balanced accuracy (macro-averaged recall)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # F1 scores
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wgt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=unique_labels, zero_division=0
    )
    per_class_f1 = {str(c): float(f) for c, f in zip(unique_labels, f1_per_class)}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Per-class accuracy from confusion matrix
    per_class_acc = {}
    for i, c in enumerate(unique_labels):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            per_class_acc[str(c)] = float(cm[i, i]) / float(row_sum)
        else:
            per_class_acc[str(c)] = np.nan

    return DecodingMetrics(
        balanced_accuracy=float(bal_acc),
        f1_macro=float(f1_mac),
        f1_weighted=float(f1_wgt),
        per_class_accuracy=per_class_acc,
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        class_names=[str(c) for c in unique_labels],
        n_samples=n_samples,
        n_classes=n_classes,
    )


def aggregate_fold_metrics(
    fold_metrics: List[DecodingMetrics],
) -> Dict[str, float]:
    """Aggregate metrics across CV folds.

    Args:
        fold_metrics: List of DecodingMetrics from each fold.

    Returns:
        Dict with mean and std of key metrics.
    """
    if not fold_metrics:
        return {
            "balanced_accuracy_mean": np.nan,
            "balanced_accuracy_std": np.nan,
            "f1_macro_mean": np.nan,
            "f1_macro_std": np.nan,
        }

    bal_accs = [m.balanced_accuracy for m in fold_metrics]
    f1_macs = [m.f1_macro for m in fold_metrics]

    return {
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "f1_macro_mean": float(np.mean(f1_macs)),
        "f1_macro_std": float(np.std(f1_macs)),
        "n_folds": len(fold_metrics),
    }


def compute_chance_level(
    y: np.ndarray,
    method: str = "uniform",
) -> float:
    """Compute chance-level accuracy for comparison.

    Mathematical Definition:
        Uniform (random guessing):
            chance = 1 / C

            Where C = number of unique classes.

        Majority (always predict most frequent):
            chance = max_c(n_c) / n

            Where n_c = count of class c, n = total samples.

    Args:
        y: Labels (to compute class distribution), shape (n_samples,).
        method: 'uniform' (1/n_classes) or 'majority' (most frequent class).

    Returns:
        Chance-level accuracy ∈ (0, 1].
    """
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)

    if method == "uniform":
        return 1.0 / n_classes
    elif method == "majority":
        return float(counts.max()) / float(counts.sum())
    else:
        raise ValueError(f"Unknown method: {method}")


def is_above_chance(
    accuracy: float,
    y: np.ndarray,
    alpha: float = 0.05,
    n_permutations: int = 1000,
) -> bool:
    """Simple check if accuracy is above chance (heuristic).

    Mathematical Definition:
        Binomial standard error approximation:
            SE = √(p̂(1-p̂) / n)

        Where p̂ = chance-level probability (1/C for uniform).

        One-sided z-test threshold:
            threshold = chance + z_(1-α) × SE

        Decision: accuracy > threshold → above chance.

    Note:
        This is a quick heuristic. For proper statistical testing,
        use permutation tests which make no distributional assumptions.

    Args:
        accuracy: Observed accuracy ∈ [0, 1].
        y: Labels (to compute chance level and sample size).
        alpha: Significance level (default 0.05).
        n_permutations: Not used (placeholder for future permutation test).

    Returns:
        True if accuracy is likely above chance at significance level alpha.
    """
    chance = compute_chance_level(y, method="uniform")
    n_samples = len(y)

    # Simple binomial standard error approximation
    se = np.sqrt(chance * (1 - chance) / n_samples)

    # Z-score threshold for one-sided test
    from scipy.stats import norm

    z_threshold = norm.ppf(1 - alpha)

    return accuracy > (chance + z_threshold * se)
