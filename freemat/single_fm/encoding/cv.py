"""Cross-validation utilities for encoding models.

Provides blocked temporal CV with embargo gaps, designed for autocorrelated
neural time series data.

Example:
    >>> from freemat.single_fm.encoding.cv import CVConfig, blocked_cv_folds
    >>> cfg = CVConfig(folds=5, block_s=30.0, gap_s=2.0)
    >>> folds = blocked_cv_folds(T=5000, bin_s=0.5, cfg=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CVConfig:
    """Cross-validation configuration for encoding models.

    Args:
        folds: Number of CV folds.
        block_s: Block length in seconds for temporal CV.
        gap_s: Embargo gap in seconds between train/test.
    """

    folds: int = 5
    block_s: float = 30.0
    gap_s: float = 2.0


@lru_cache(maxsize=64)
def _cached_blocked_cv_folds(
    T: int,
    bin_s: float,
    folds: int,
    block_s: float,
    gap_s: float,
) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Cached fold computation returning hashable tuples.

    The result is cached to avoid recomputation when called with same args.
    """
    B = max(1, int(round(block_s / bin_s)))
    G = max(0, int(round(gap_s / bin_s)))
    n_blocks = T // B
    idx_blocks = np.arange(n_blocks)
    assign = idx_blocks % folds

    folds_out = []
    for f in range(folds):
        test_blocks = idx_blocks[assign == f]
        test_idx = []
        for b in test_blocks:
            s = b * B
            e = min((b + 1) * B, T)
            test_idx.append(np.arange(s, e, dtype=int))
        test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)

        mask = np.ones(T, dtype=bool)
        for b in test_blocks:
            s = max(0, b * B - G)
            e = min(T, (b + 1) * B + G)
            mask[s:e] = False
        train_idx = np.where(mask)[0]

        # Convert to tuples for hashability
        folds_out.append((tuple(train_idx.tolist()), tuple(test_idx.tolist())))

    return tuple(folds_out)


def blocked_cv_folds(
    T: int,
    bin_s: float,
    cfg: CVConfig,
) -> List[Dict[str, np.ndarray]]:
    """Build contiguous CV folds with exclusion gaps.

    Uses LRU caching for repeated calls with same parameters.

    Args:
        T: Number of time points.
        bin_s: Bin width in seconds.
        cfg: CV configuration.

    Returns:
        List of dicts with 'train' and 'test' index arrays.
    """
    # Get cached result (as tuples)
    cached = _cached_blocked_cv_folds(T, bin_s, cfg.folds, cfg.block_s, cfg.gap_s)

    # Convert back to expected format
    return [
        {"train": np.array(tr, dtype=int), "test": np.array(te, dtype=int)}
        for tr, te in cached
    ]


__all__ = [
    "CVConfig",
    "blocked_cv_folds",
]
