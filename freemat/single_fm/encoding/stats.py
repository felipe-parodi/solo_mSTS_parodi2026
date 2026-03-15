"""Permutation testing utilities for unique variance analysis.

Provides block permutation and permutation-based p-values for 3-way
unique variance (VIS vs BODY_KIN vs HEAD_KIN) using shuffle-style nulls.

Example:
    >>> from freemat.single_fm.encoding.stats import permutation_test_unique_variance_3way
    >>> p_vis, p_body, p_head, null_vis, null_body, null_head = (
    ...     permutation_test_unique_variance_3way(
    ...         X_vis, X_body, X_head, y, folds, bin_s=0.5,
    ...         obs_unique_vis=0.02, obs_unique_body_kin=0.01,
    ...         obs_unique_head_kin=0.005, n_perm=1000,
    ...     )
    ... )
"""

from __future__ import annotations

from typing import List, Dict, Optional, Sequence, Tuple

import numpy as np

from freemat.single_fm.encoding.ridge import (
    DEFAULT_ALPHAS,
    HAS_GPU,
    HAS_NUMBA,
    compute_cv_r2,
    compute_cv_r2_fixed_alpha,
    compute_cv_r2_fixed_alpha_batch,
)

# Optional Numba support
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _build_permuted_indices(
    T: int,
    block_len: int,
    block_order: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated index building from pre-shuffled block order."""
    n_blocks = len(block_order)
    remainder = T - n_blocks * block_len

    indices = np.empty(T, dtype=np.int64)
    idx = 0

    # Build indices from shuffled blocks
    for b in range(n_blocks):
        block_idx = block_order[b]
        start = block_idx * block_len
        for k in range(block_len):
            indices[idx] = start + k
            idx += 1

    # Handle remainder (not shuffled)
    if remainder > 0:
        start = n_blocks * block_len
        for k in range(remainder):
            indices[idx] = start + k
            idx += 1

    return indices


def block_permute_indices(
    T: int,
    block_len: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate block-permuted indices preserving local autocorrelation.

    Args:
        T: Total number of time points.
        block_len: Block length in bins.
        rng: Random number generator.

    Returns:
        Permuted indices array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = T // block_len

    # Shuffle block order using numpy RNG
    block_order = rng.permutation(n_blocks).astype(np.int64)

    # Use Numba-accelerated index building if available
    if HAS_NUMBA:
        return _build_permuted_indices(T, block_len, block_order)

    # Fallback to pure Python
    remainder = T % block_len
    indices = []
    for block_idx in block_order:
        start = block_idx * block_len
        end = start + block_len
        indices.extend(range(start, end))

    if remainder > 0:
        start = n_blocks * block_len
        indices.extend(range(start, T))

    return np.array(indices, dtype=np.int64)


def circular_shift_indices(
    T: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate circular-shifted indices (fully preserves autocorrelation).

    Shifts the entire timeseries by a random offset and wraps around.
    Unlike block permutation, this preserves the full autocorrelation
    structure of the signal (Testard-style).

    Args:
        T: Total number of time points.
        rng: Random number generator.

    Returns:
        Circularly shifted indices array of length T.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random shift between 1 and T-1 (avoid identity shift)
    shift = rng.integers(1, T)
    return np.roll(np.arange(T, dtype=np.int64), -shift)


def permutation_test_unique_variance_3way(
    X_vis: np.ndarray,
    X_body_kin: np.ndarray,
    X_head_kin: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    bin_s: float,
    obs_unique_vis: float,
    obs_unique_body_kin: float,
    obs_unique_head_kin: float,
    n_perm: int = 1000,
    block_s: float = 10.0,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
    fixed_alpha: Optional[float] = None,
    method: str = "block",
    r2_full: Optional[float] = None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute permutation p-values for 3-way unique variance (shuffle-style).

    Uses shuffle-style null: null_X[i] = r2_full - r2(X permuted).
    This matches the observed statistic formula (r2_full - r2_with_shuffled_group),
    ensuring the null and observed use the same formula.

    Supports two null construction methods:
    - "block": Block permutation (10s blocks shuffled). Partially preserves
      local autocorrelation.
    - "circular": Circular shift (wrap around). Fully preserves
      autocorrelation structure (Testard-style).

    Args:
        X_vis: VIS design matrix.
        X_body_kin: BODY_KIN design matrix.
        X_head_kin: HEAD_KIN design matrix.
        y: Target vector.
        folds: CV folds.
        bin_s: Bin size in seconds.
        obs_unique_vis: Observed unique VIS variance.
        obs_unique_body_kin: Observed unique BODY_KIN variance.
        obs_unique_head_kin: Observed unique HEAD_KIN variance.
        n_perm: Number of permutations.
        block_s: Block length for permutation in seconds (block method only).
        rng: Random number generator.
        use_gpu: Whether to use GPU acceleration.
        fixed_alpha: If provided, use fixed alpha instead of nested CV.
        method: Null construction method ("block" or "circular").
        r2_full: Pre-computed full-model R² (computed if not provided).

    Returns:
        Tuple of (p_vis, p_body_kin, p_head_kin, null_dist_vis, null_dist_body_kin, null_dist_head_kin).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = X_vis.shape[0]
    block_len = max(1, int(round(block_s / bin_s)))

    def _permute(T: int) -> np.ndarray:
        if method == "circular":
            return circular_shift_indices(T, rng)
        return block_permute_indices(T, block_len, rng)

    null_vis = np.zeros(n_perm)
    null_body_kin = np.zeros(n_perm)
    null_head_kin = np.zeros(n_perm)

    # Compute r2_full once if not provided (shuffle-style null needs it)
    X_full = np.hstack([X_vis, X_body_kin, X_head_kin])

    if fixed_alpha is not None:
        # Fast path: fixed alpha, no inner CV
        if r2_full is None:
            r2_full = compute_cv_r2_fixed_alpha(X_full, y, folds, fixed_alpha, use_gpu=use_gpu)

        D_total = X_full.shape[1]
        _use_batch = use_gpu and HAS_GPU and n_perm >= 4

        if _use_batch:
            # --- Batch GPU path: all permutations in one GPU call per group ---
            # VIS null
            X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
            for i in range(n_perm):
                X_batch[i] = np.hstack([X_vis[_permute(T)], X_body_kin, X_head_kin])
            r2_batch = compute_cv_r2_fixed_alpha_batch(X_batch, y, folds, fixed_alpha)
            null_vis[:] = r2_full - r2_batch
            del X_batch

            # BODY_KIN null
            X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
            for i in range(n_perm):
                X_batch[i] = np.hstack([X_vis, X_body_kin[_permute(T)], X_head_kin])
            r2_batch = compute_cv_r2_fixed_alpha_batch(X_batch, y, folds, fixed_alpha)
            null_body_kin[:] = r2_full - r2_batch
            del X_batch

            # HEAD_KIN null
            X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
            for i in range(n_perm):
                X_batch[i] = np.hstack([X_vis, X_body_kin, X_head_kin[_permute(T)]])
            r2_batch = compute_cv_r2_fixed_alpha_batch(X_batch, y, folds, fixed_alpha)
            null_head_kin[:] = r2_full - r2_batch
            del X_batch
        else:
            # --- Sequential path ---
            for i in range(n_perm):
                X_vis_perm = X_vis[_permute(T)]
                X_full_perm = np.hstack([X_vis_perm, X_body_kin, X_head_kin])
                r2_full_perm = compute_cv_r2_fixed_alpha(X_full_perm, y, folds, fixed_alpha, use_gpu=use_gpu)
                null_vis[i] = r2_full - r2_full_perm

                X_body_kin_perm = X_body_kin[_permute(T)]
                X_full_perm = np.hstack([X_vis, X_body_kin_perm, X_head_kin])
                r2_full_perm = compute_cv_r2_fixed_alpha(X_full_perm, y, folds, fixed_alpha, use_gpu=use_gpu)
                null_body_kin[i] = r2_full - r2_full_perm

                X_head_kin_perm = X_head_kin[_permute(T)]
                X_full_perm = np.hstack([X_vis, X_body_kin, X_head_kin_perm])
                r2_full_perm = compute_cv_r2_fixed_alpha(X_full_perm, y, folds, fixed_alpha, use_gpu=use_gpu)
                null_head_kin[i] = r2_full - r2_full_perm
    else:
        # Slower path: nested CV for alpha selection
        alphas = DEFAULT_ALPHAS
        if r2_full is None:
            r2_full = compute_cv_r2(X_full, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)

        for i in range(n_perm):
            # Permute VIS features
            X_vis_perm = X_vis[_permute(T)]
            X_full_perm = np.hstack([X_vis_perm, X_body_kin, X_head_kin])
            r2_full_perm = compute_cv_r2(X_full_perm, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
            null_vis[i] = r2_full - r2_full_perm

            # Permute BODY_KIN features
            X_body_kin_perm = X_body_kin[_permute(T)]
            X_full_perm = np.hstack([X_vis, X_body_kin_perm, X_head_kin])
            r2_full_perm = compute_cv_r2(X_full_perm, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
            null_body_kin[i] = r2_full - r2_full_perm

            # Permute HEAD_KIN features
            X_head_kin_perm = X_head_kin[_permute(T)]
            X_full_perm = np.hstack([X_vis, X_body_kin, X_head_kin_perm])
            r2_full_perm = compute_cv_r2(X_full_perm, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
            null_head_kin[i] = r2_full - r2_full_perm

    # Compute p-values (one-sided: proportion of null >= observed)
    p_vis = (np.sum(null_vis >= obs_unique_vis) + 1) / (n_perm + 1)
    p_body_kin = (np.sum(null_body_kin >= obs_unique_body_kin) + 1) / (n_perm + 1)
    p_head_kin = (np.sum(null_head_kin >= obs_unique_head_kin) + 1) / (n_perm + 1)

    return float(p_vis), float(p_body_kin), float(p_head_kin), null_vis, null_body_kin, null_head_kin


def permutation_test_unique_variance(
    X_vis: np.ndarray,
    X_kin: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    bin_s: float,
    obs_unique_kin: float,
    obs_unique_vis: float,
    n_perm: int = 1000,
    block_s: float = 10.0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
    fixed_alpha: Optional[float] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute permutation p-values for 2-way unique variance (KIN vs VIS).

    Uses block permutation to preserve autocorrelation structure.

    Args:
        X_vis: VIS design matrix.
        X_kin: KIN design matrix.
        y: Target vector.
        folds: CV folds.
        bin_s: Bin size in seconds.
        obs_unique_kin: Observed unique KIN variance.
        obs_unique_vis: Observed unique VIS variance.
        n_perm: Number of permutations.
        block_s: Block length for permutation in seconds.
        alphas: Ridge alpha grid (ignored if fixed_alpha provided).
        rng: Random number generator.
        use_gpu: Whether to use GPU acceleration.
        fixed_alpha: If provided, use fixed alpha instead of nested CV (~10x faster).

    Returns:
        Tuple of (p_kin, p_vis, null_dist_kin, null_dist_vis).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = X_vis.shape[0]
    block_len = max(1, int(round(block_s / bin_s)))

    null_kin = np.zeros(n_perm)
    null_vis = np.zeros(n_perm)

    if fixed_alpha is not None:
        r2_vis_only = compute_cv_r2_fixed_alpha(X_vis, y, folds, fixed_alpha, use_gpu=use_gpu)
        r2_kin_only = compute_cv_r2_fixed_alpha(X_kin, y, folds, fixed_alpha, use_gpu=use_gpu)

        for i in range(n_perm):
            perm_idx = block_permute_indices(T, block_len, rng)
            X_kin_perm = X_kin[perm_idx]
            X_full_perm = np.hstack([X_vis, X_kin_perm])
            r2_full_perm = compute_cv_r2_fixed_alpha(X_full_perm, y, folds, fixed_alpha, use_gpu=use_gpu)
            null_kin[i] = r2_full_perm - r2_vis_only

            perm_idx = block_permute_indices(T, block_len, rng)
            X_vis_perm = X_vis[perm_idx]
            X_full_perm = np.hstack([X_vis_perm, X_kin])
            r2_full_perm = compute_cv_r2_fixed_alpha(X_full_perm, y, folds, fixed_alpha, use_gpu=use_gpu)
            null_vis[i] = r2_full_perm - r2_kin_only
    else:
        r2_vis_only = compute_cv_r2(X_vis, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
        r2_kin_only = compute_cv_r2(X_kin, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)

        for i in range(n_perm):
            perm_idx = block_permute_indices(T, block_len, rng)
            X_kin_perm = X_kin[perm_idx]
            X_full_perm = np.hstack([X_vis, X_kin_perm])
            r2_full_perm = compute_cv_r2(X_full_perm, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
            null_kin[i] = r2_full_perm - r2_vis_only

            perm_idx = block_permute_indices(T, block_len, rng)
            X_vis_perm = X_vis[perm_idx]
            X_full_perm = np.hstack([X_vis_perm, X_kin])
            r2_full_perm = compute_cv_r2(X_full_perm, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
            null_vis[i] = r2_full_perm - r2_kin_only

    p_kin = (np.sum(null_kin >= obs_unique_kin) + 1) / (n_perm + 1)
    p_vis = (np.sum(null_vis >= obs_unique_vis) + 1) / (n_perm + 1)

    return float(p_kin), float(p_vis), null_kin, null_vis


__all__ = [
    "block_permute_indices",
    "circular_shift_indices",
    "permutation_test_unique_variance",
    "permutation_test_unique_variance_3way",
]
