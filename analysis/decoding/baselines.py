"""Baseline methods for neural decoding validation.

Provides null models to validate that decoding performance is meaningful:
- Persistence baseline: predicts y[t+h] = y[t] (behavioral inertia)
- Segment shuffle: shuffles labels in blocks to preserve autocorrelation

Usage:
    from analysis.decoding.baselines import persistence_baseline, segment_shuffle_baseline

    # For future prediction at horizon h
    persist_acc = persistence_baseline(y_current, y_future)

    # For null distribution
    null_accs = segment_shuffle_baseline(X, y, classifier, n_shuffles=100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score


@dataclass
class BaselineResult:
    """Container for baseline analysis results."""

    persistence_accuracy: float
    shuffle_accuracy_mean: float
    shuffle_accuracy_std: float
    shuffle_accuracies: np.ndarray
    n_shuffles: int
    actual_accuracy: Optional[float] = None
    p_value: Optional[float] = None


def persistence_baseline(
    y_current: np.ndarray,
    y_future: np.ndarray,
) -> float:
    """Compute accuracy of persistence prediction.

    Predicts that y_future = y_current (state doesn't change).
    This is the "behavioral inertia" baseline for future prediction.

    Args:
        y_current: Current state labels at time t.
        y_future: Future state labels at time t+h.

    Returns:
        Balanced accuracy of persistence prediction.

    Example:
        >>> y_curr = np.array([0, 0, 1, 1, 2])
        >>> y_fut = np.array([0, 1, 1, 2, 2])
        >>> acc = persistence_baseline(y_curr, y_fut)
        >>> acc  # 3/5 correct (indices 0, 2, 4)
        0.6
    """
    if len(y_current) != len(y_future):
        raise ValueError("y_current and y_future must have same length")

    if len(y_current) == 0:
        return np.nan

    # Persistence prediction: future = current
    y_pred = y_current
    return balanced_accuracy_score(y_future, y_pred)


def segment_shuffle(
    y: np.ndarray,
    segment_size_bins: int = 50,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Shuffle labels in contiguous segments.

    Preserves autocorrelation structure by shuffling blocks of labels
    rather than individual samples. This creates a null distribution
    that maintains local temporal structure.

    Args:
        y: Original labels, shape (n_samples,).
        segment_size_bins: Size of segments to shuffle.
        random_state: Random seed for reproducibility.

    Returns:
        Shuffled labels with same shape.

    Example:
        >>> y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        >>> y_shuf = segment_shuffle(y, segment_size_bins=3, random_state=42)
        >>> # Segments [0,0,0], [1,1,1], [2,2,2] are shuffled as blocks
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y)

    if n_samples == 0:
        return y.copy()

    if segment_size_bins >= n_samples:
        # Single segment, just return copy
        return y.copy()

    # Split into segments
    n_segments = int(np.ceil(n_samples / segment_size_bins))
    segments = []
    for i in range(n_segments):
        start = i * segment_size_bins
        end = min((i + 1) * segment_size_bins, n_samples)
        segments.append(y[start:end].copy())

    # Shuffle segment order
    rng.shuffle(segments)

    # Concatenate back
    y_shuffled = np.concatenate(segments)

    # Trim to original length (in case of uneven split)
    return y_shuffled[:n_samples]


def segment_shuffle_baseline(
    y: np.ndarray,
    n_shuffles: int = 100,
    segment_size_bins: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Generate null distribution via segment shuffling.

    Creates multiple segment-shuffled versions of labels.
    Used to compute null accuracy distribution.

    Args:
        y: Original labels.
        n_shuffles: Number of shuffle iterations.
        segment_size_bins: Segment size for shuffling.
        random_state: Base random seed.

    Returns:
        Array of shuffled label arrays, shape (n_shuffles, n_samples).
    """
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31, size=n_shuffles)

    shuffled = []
    for seed in seeds:
        y_shuf = segment_shuffle(y, segment_size_bins, random_state=int(seed))
        shuffled.append(y_shuf)

    return np.array(shuffled)


def compute_baseline_comparison(
    actual_accuracy: float,
    null_accuracies: np.ndarray,
) -> Tuple[float, float]:
    """Compute p-value and z-score against null distribution.

    Args:
        actual_accuracy: Observed decoding accuracy.
        null_accuracies: Array of null accuracies from shuffles.

    Returns:
        Tuple of (p_value, z_score).
    """
    if len(null_accuracies) == 0:
        return np.nan, np.nan

    # One-sided p-value: P(null >= actual)
    n_greater = np.sum(null_accuracies >= actual_accuracy)
    p_value = (n_greater + 1) / (len(null_accuracies) + 1)

    # Z-score
    null_mean = np.mean(null_accuracies)
    null_std = np.std(null_accuracies)
    if null_std > 0:
        z_score = (actual_accuracy - null_mean) / null_std
    else:
        z_score = np.inf if actual_accuracy > null_mean else 0.0

    return float(p_value), float(z_score)


def run_full_baseline_analysis(
    y_current: np.ndarray,
    y_future: np.ndarray,
    actual_accuracy: float,
    n_shuffles: int = 100,
    segment_size_bins: int = 50,
    random_state: int = 42,
) -> BaselineResult:
    """Run complete baseline analysis for future prediction.

    Computes persistence baseline and segment-shuffle null distribution.

    Args:
        y_current: Current state labels.
        y_future: Future state labels.
        actual_accuracy: Observed decoding accuracy.
        n_shuffles: Number of segment shuffles.
        segment_size_bins: Segment size.
        random_state: Random seed.

    Returns:
        BaselineResult with all baseline metrics.
    """
    # Persistence baseline
    persist_acc = persistence_baseline(y_current, y_future)

    # Segment shuffle baseline (shuffle y_future, keeping y_current aligned)
    y_future_shuffled = segment_shuffle_baseline(
        y_future, n_shuffles, segment_size_bins, random_state
    )

    # Compute accuracy for each shuffle (as if predicting shuffled labels)
    # This measures how much structure the shuffle destroys
    shuffle_accs = []
    for y_shuf in y_future_shuffled:
        # Accuracy of persistence against shuffled future
        acc = persistence_baseline(y_current, y_shuf)
        shuffle_accs.append(acc)
    shuffle_accs = np.array(shuffle_accs)

    # P-value against shuffle null
    p_value, _ = compute_baseline_comparison(actual_accuracy, shuffle_accs)

    return BaselineResult(
        persistence_accuracy=persist_acc,
        shuffle_accuracy_mean=float(np.mean(shuffle_accs)),
        shuffle_accuracy_std=float(np.std(shuffle_accs)),
        shuffle_accuracies=shuffle_accs,
        n_shuffles=n_shuffles,
        actual_accuracy=actual_accuracy,
        p_value=p_value,
    )
