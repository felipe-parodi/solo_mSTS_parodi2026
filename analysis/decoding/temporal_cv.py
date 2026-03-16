"""Temporal cross-validation utilities for neural decoding.

Provides blocked temporal CV with embargo to prevent autocorrelation leakage.
Reuses patterns from existing analysis/decoding.py but in a modular form.

Usage:
    from analysis.decoding.temporal_cv import make_temporal_folds, shift_labels_for_horizon

    folds = make_temporal_folds(n_samples=1000, n_folds=5, embargo_bins=20)
    for train_idx, test_idx in folds:
        # train/test on fold

    # Multi-strategy CV (replaces legacy _make_cv_splits)
    from analysis.decoding.temporal_cv import make_cv_splits

    splits = make_cv_splits(n_samples=1000, n_folds=5, strategy="temporal_kfold", embargo_bins=20)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


@dataclass
class FoldInfo:
    """Information about a single CV fold."""

    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_train: int
    n_test: int


def make_temporal_folds(
    n_samples: int,
    n_folds: int = 5,
    embargo_bins: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create contiguous temporal folds with embargo.

    Splits timeline into n_folds contiguous blocks. For each fold,
    the block is the test set and remaining samples (minus embargo
    buffer) form the training set.

    Args:
        n_samples: Total number of time bins.
        n_folds: Number of CV folds.
        embargo_bins: Number of bins to exclude around test block boundaries.

    Returns:
        List of (train_idx, test_idx) tuples.

    Example:
        >>> folds = make_temporal_folds(100, n_folds=5, embargo_bins=2)
        >>> len(folds)
        5
        >>> train_idx, test_idx = folds[0]
        >>> len(test_idx)  # ~20 samples per fold
        20
    """
    if n_samples == 0:
        return []
    if n_folds <= 1 or n_samples < n_folds:
        # Degenerate case: return all as test
        return [(np.array([], dtype=int), np.arange(n_samples))]

    indices = np.arange(n_samples)
    blocks = np.array_split(indices, n_folds)
    splits = []

    for block in blocks:
        if block.size == 0:
            continue
        start = int(block[0])
        end = int(block[-1])

        # Apply embargo around test block
        left = max(0, start - embargo_bins)
        right = min(n_samples - 1, end + embargo_bins)

        # Training mask: everything except embargo region
        mask = np.ones(n_samples, dtype=bool)
        mask[left : right + 1] = False
        train_idx = indices[mask]

        splits.append((train_idx, block))

    return splits


def make_temporal_folds_with_info(
    n_samples: int,
    n_folds: int = 5,
    embargo_bins: int = 0,
) -> List[FoldInfo]:
    """Create temporal folds with detailed info.

    Same as make_temporal_folds but returns FoldInfo objects.
    """
    folds = make_temporal_folds(n_samples, n_folds, embargo_bins)
    return [
        FoldInfo(
            fold_idx=i,
            train_idx=train_idx,
            test_idx=test_idx,
            n_train=len(train_idx),
            n_test=len(test_idx),
        )
        for i, (train_idx, test_idx) in enumerate(folds)
    ]


def shift_labels_for_horizon(
    y: np.ndarray,
    horizon_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift labels for future-state prediction.

    For predicting y[t+h] from features at time t, we shift labels
    forward by h bins.

    Args:
        y: Original labels, shape (n_samples,).
        horizon_bins: Number of bins to shift forward (positive = future).

    Returns:
        Tuple of:
            - y_shifted: Shifted labels, shape (n_valid,)
            - valid_mask: Boolean mask for valid samples, shape (n_samples,)

    Example:
        >>> y = np.array([0, 1, 1, 2, 2])
        >>> y_shifted, valid = shift_labels_for_horizon(y, horizon_bins=2)
        >>> y_shifted  # y[2:] aligned with X[:3]
        array([1, 2, 2])
        >>> valid
        array([ True,  True,  True, False, False])
    """
    n_samples = len(y)
    if horizon_bins <= 0:
        return y.copy(), np.ones(n_samples, dtype=bool)

    if horizon_bins >= n_samples:
        return np.array([], dtype=y.dtype), np.zeros(n_samples, dtype=bool)

    # Valid indices: can predict y[t+h] from X[t]
    n_valid = n_samples - horizon_bins
    valid_mask = np.zeros(n_samples, dtype=bool)
    valid_mask[:n_valid] = True

    # Shifted labels: y[h:] corresponds to X[:n_valid]
    y_shifted = y[horizon_bins:]

    return y_shifted, valid_mask


def get_change_only_mask(
    y_current: np.ndarray,
    y_future: np.ndarray,
) -> np.ndarray:
    """Get mask for samples where state changes.

    For prediction tasks, focus on bins where the future state
    differs from the current state (more informative than
    predicting persistence).

    Args:
        y_current: Current state labels at time t.
        y_future: Future state labels at time t+h.

    Returns:
        Boolean mask where y_future != y_current.

    Example:
        >>> y_curr = np.array([0, 0, 1, 1, 2])
        >>> y_fut = np.array([0, 1, 1, 2, 2])
        >>> mask = get_change_only_mask(y_curr, y_fut)
        >>> mask
        array([False,  True, False,  True, False])
    """
    return y_current != y_future


def apply_horizon_to_folds(
    folds: List[Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    horizon_bins: int,
    change_only: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Apply horizon shift to pre-computed folds.

    For future-state prediction, adjusts fold indices to account
    for the horizon shift. Optionally filters to change-only samples.

    Args:
        folds: List of (train_idx, test_idx) from make_temporal_folds.
        y: Original labels.
        horizon_bins: Prediction horizon.
        change_only: If True, filter to samples where state changes.

    Returns:
        List of (train_idx, test_idx, y_shifted) tuples.
        Indices are adjusted for valid samples only.
    """
    y_shifted, valid_mask = shift_labels_for_horizon(y, horizon_bins)
    valid_indices = np.where(valid_mask)[0]

    results = []
    for train_idx, test_idx in folds:
        # Filter to valid samples
        train_valid = train_idx[train_idx < len(y_shifted)]
        test_valid = test_idx[test_idx < len(y_shifted)]

        if len(train_valid) == 0 or len(test_valid) == 0:
            continue

        if change_only:
            # Get change mask for test samples
            y_curr_test = y[test_valid]
            y_fut_test = y_shifted[test_valid]
            change_mask = get_change_only_mask(y_curr_test, y_fut_test)
            test_valid = test_valid[change_mask]

            if len(test_valid) == 0:
                continue

        results.append((train_valid, test_valid, y_shifted))

    return results


def make_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    strategy: Literal["temporal_kfold", "stratified_kfold", "kfold"] = "temporal_kfold",
    y: Optional[np.ndarray] = None,
    embargo_bins: int = 0,
    random_state: Optional[int] = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create CV splits with multiple strategy options.

    Unified interface for cross-validation split generation, supporting
    temporal, stratified, and standard k-fold strategies.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of CV folds.
        strategy: CV strategy to use:
            - "temporal_kfold": Contiguous blocks with embargo (default)
            - "stratified_kfold": Stratified by labels (requires y)
            - "kfold": Standard random k-fold
        y: Labels array, required for stratified_kfold.
        embargo_bins: Bins to exclude around test blocks (temporal only).
        random_state: Random seed for non-temporal strategies.

    Returns:
        List of (train_idx, test_idx) tuples.

    Example:
        >>> splits = make_cv_splits(100, n_folds=5, strategy="temporal_kfold")
        >>> len(splits)
        5

        >>> y = np.array([0, 0, 1, 1] * 25)
        >>> splits = make_cv_splits(100, n_folds=5, strategy="stratified_kfold", y=y)
        >>> len(splits)
        5
    """
    if n_samples == 0:
        return []
    if n_folds <= 1 or n_samples < n_folds:
        indices = np.arange(n_samples)
        return [(np.array([], dtype=int), indices)]

    indices = np.arange(n_samples)

    if strategy == "temporal_kfold":
        return make_temporal_folds(n_samples, n_folds, embargo_bins)

    elif strategy == "stratified_kfold":
        if y is None:
            raise ValueError("y is required for stratified_kfold strategy")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return [(indices[tr], indices[te]) for tr, te in skf.split(indices, y)]

    elif strategy == "kfold":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return [(indices[tr], indices[te]) for tr, te in kf.split(indices)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'temporal_kfold', 'stratified_kfold', or 'kfold'")
