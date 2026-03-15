"""Cross-validation utilities for temporal encoding models.

This module provides temporal cross-validation strategies that preserve
the temporal structure of neural data and avoid autocorrelation leakage.

Key Features:
    - Contiguous temporal K-fold splits (no shuffling)
    - Behavior-boundary-aware fold creation
    - Embargo zones to prevent information leakage
    - Segment-based shuffling for null models
"""

from typing import List, Optional, Tuple

import numpy as np

# Handle sklearn import
try:
    from sklearn.model_selection import KFold as KFold_cpu
    HAS_SKLEARN = True
except (ImportError, ValueError):
    HAS_SKLEARN = False

    class KFold_cpu:
        """Fallback KFold implementation when sklearn unavailable."""

        def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, X, y=None):
            n = len(X)
            fold_size = n // self.n_splits
            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else n
                test_idx = list(range(start, end))
                train_idx = list(range(0, start)) + list(range(end, n))
                yield train_idx, test_idx


def temporal_kfold_indices(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create contiguous, non-overlapping temporal K-fold splits.

    Mathematical Definition:
        Partition n samples into K contiguous folds:
            fold_size_base = ⌊n / K⌋
            fold_sizes = [fold_size_base + 1] × (n mod K) + [fold_size_base] × (K - n mod K)

        For fold k:
            test_idx  = [start_k, start_k + fold_size_k)
            train_idx = [0, start_k) ∪ [start_k + fold_size_k, n)

        This ensures:
            - No temporal overlap between train and test
            - All samples used exactly once as test
            - Avoids autocorrelation leakage from random splits

    Args:
        n_samples: Number of time samples.
        n_splits: Number of folds (>=2).

    Returns:
        List of (train_idx, test_idx) arrays for each fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for cross-validation")
    fold_sizes = [n_samples // n_splits] * n_splits
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1
    indices = np.arange(n_samples)
    splits = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        test_idx = indices[start:stop]
        train_idx = (
            np.concatenate([indices[:start], indices[stop:]])
            if start > 0 or stop < n_samples
            else np.array([], dtype=int)
        )
        splits.append((train_idx, test_idx))
        start = stop
    return splits


def temporal_kfold_indices_with_boundaries(
    n_samples: int,
    n_splits: int,
    allowed_cut_indices: Optional[np.ndarray],
    embargo_bins: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create contiguous temporal folds whose cut points align to behavior boundaries.

    This prevents splitting a behavioral bout across train/test, which could
    cause information leakage due to autocorrelation within bouts.

    Args:
        n_samples: Number of rows (effective bins).
        n_splits: Number of outer folds.
        allowed_cut_indices: Sorted unique indices in [1, n_samples-1] where a fold may cut.
        embargo_bins: Number of bins to exclude around fold boundaries (default 0).

    Returns:
        List of (train_idx, test_idx) for each fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for cross-validation")
    if allowed_cut_indices is None or len(allowed_cut_indices) == 0:
        # Fallback to plain temporal kfold
        return temporal_kfold_indices(n_samples, n_splits)

    cuts = []
    target = n_samples / n_splits
    start = 0
    # choose n_splits-1 cut points
    for i in range(1, n_splits):
        desired = int(round(i * target))
        # find first allowed cut >= desired
        ahead = allowed_cut_indices[allowed_cut_indices >= desired]
        if ahead.size > 0:
            cut = int(ahead[0])
        else:
            # fallback: nearest allowed behind desired
            behind = allowed_cut_indices[allowed_cut_indices > start]
            if behind.size > 0:
                cut = int(behind[-1])
            else:
                cut = desired
        # guard: ensure progress and not overflow
        cut = max(start + 1, min(cut, n_samples - (n_splits - i)))
        cuts.append(cut)
        start = cut
    # build folds
    indices = np.arange(n_samples)
    prev = 0
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for cut in cuts:
        test_idx = indices[prev:cut]
        if embargo_bins > 0:
            left = max(0, prev - embargo_bins)
            right = min(n_samples, cut + embargo_bins)
            mask = np.ones(n_samples, dtype=bool)
            mask[left:right] = False
            train_idx = indices[mask]
        else:
            train_idx = (
                np.concatenate([indices[:prev], indices[cut:]])
                if prev > 0 or cut < n_samples
                else np.array([], dtype=int)
            )
        splits.append((train_idx, test_idx))
        prev = cut
    # last fold
    test_idx = indices[prev:n_samples]
    if embargo_bins > 0:
        left = max(0, prev - embargo_bins)
        right = n_samples
        mask = np.ones(n_samples, dtype=bool)
        mask[left:right] = False
        train_idx = indices[mask]
    else:
        train_idx = indices[:prev] if prev > 0 else np.array([], dtype=int)
    splits.append((train_idx, test_idx))
    return splits


def compute_behavior_change_indices(
    binned_self: Optional[np.ndarray],
    binned_partner: Optional[np.ndarray],
    y_slice: slice,
) -> np.ndarray:
    """Compute indices where behavior state changes (allowed cut points for folds).

    Change is detected when the concatenated behavior indicator vectors change
    from one bin to the next within the y_slice range.

    Args:
        binned_self: Binary indicator matrix for self behaviors, shape (n_bins, n_behaviors).
        binned_partner: Binary indicator matrix for partner behaviors (optional).
        y_slice: Slice defining the valid time range.

    Returns:
        Array of indices where behavior state changes.
    """
    start = y_slice.start or 0
    stop = y_slice.stop if y_slice.stop is not None else 0
    n = max(0, stop - start)
    if n <= 1:
        return np.array([], dtype=int)
    # assemble matrix (n, features)
    parts = []
    if binned_self is not None and binned_self.size > 0:
        parts.append(binned_self[start:stop, :])
    if binned_partner is not None and binned_partner.size > 0:
        parts.append(binned_partner[start:stop, :])
    if not parts:
        return np.array([], dtype=int)
    mat = np.concatenate(parts, axis=1).astype(np.int8, copy=False)
    # changes where any column differs from previous row
    diffs = np.any(mat[1:] != mat[:-1], axis=1)
    change_idx = np.nonzero(diffs)[0] + 1  # offset by 1 to get row index
    # valid cut indices are within [1, n-1]
    change_idx = change_idx[(change_idx >= 1) & (change_idx <= n - 1)]
    # unique sorted
    return np.unique(change_idx)


def segment_shuffle(
    y: np.ndarray, segment_bins: int, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle 1D array by permuting contiguous segments of fixed length.

    Mathematical Definition:
        Given array y of length n and segment size S:
            1. Partition y into segments: [y[0:S], y[S:2S], ..., y[kS:n]]
            2. Randomly permute segment order: π(segments)
            3. Concatenate: y_shuffled = concat(π(segments))

        Properties:
            - Preserves within-segment temporal autocorrelation
            - Destroys global alignment with predictors
            - More conservative than full permutation (accounts for local structure)

        This is preferred over full permutation when data has temporal dependencies.

    Args:
        y: Array to shuffle.
        segment_bins: Size of each segment in bins.
        rng: NumPy random generator for reproducibility.

    Returns:
        Segment-shuffled array.
    """
    y_np = np.asarray(y).copy()
    n = y_np.shape[0]
    if segment_bins <= 0 or segment_bins >= n:
        # Fallback to full permutation if invalid segment length
        return rng.permutation(y_np)
    # Build segment start indices
    starts = list(range(0, n, segment_bins))
    segments = [y_np[s:e] for s, e in zip(starts, starts[1:] + [n])]
    rng.shuffle(segments)
    return np.concatenate(segments)


# Legacy aliases for backward compatibility
_temporal_kfold_indices = temporal_kfold_indices
_temporal_kfold_indices_with_boundaries = temporal_kfold_indices_with_boundaries
_compute_behavior_change_indices = compute_behavior_change_indices
_segment_shuffle = segment_shuffle


__all__ = [
    "temporal_kfold_indices",
    "temporal_kfold_indices_with_boundaries",
    "compute_behavior_change_indices",
    "segment_shuffle",
    "KFold_cpu",
    "HAS_SKLEARN",
    # Legacy aliases
    "_temporal_kfold_indices",
    "_temporal_kfold_indices_with_boundaries",
    "_compute_behavior_change_indices",
    "_segment_shuffle",
]
