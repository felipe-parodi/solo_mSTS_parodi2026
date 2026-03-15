"""Unique variance computation for KIN vs VIS encoding analysis.

This module re-exports from the split submodules for backward compatibility.
The actual implementations live in:
- cv.py: Cross-validation utilities (CVConfig, blocked_cv_folds)
- ridge.py: Ridge regression and R² (ridge_fit_predict, compute_cv_r2, etc.)
- stats.py: Permutation testing (block_permute_indices, permutation_test_*)
- variance.py: Variance partitioning (UniqueVariance3Way*, compute_unique_variance_*)
"""

# CV utilities
from freemat.single_fm.encoding.cv import (
    CVConfig,
    blocked_cv_folds,
)

# Ridge regression and scoring
from freemat.single_fm.encoding.ridge import (
    DEFAULT_ALPHAS,
    HAS_GPU,
    HAS_NUMBA,
    ridge_fit_predict,
    ridge_fit_predict_gpu,
    r2_score,
    build_spline_design,
    inner_cv_select_alpha,
    compute_cv_r2,
    compute_cv_r2_fixed_alpha,
)

# Permutation testing
from freemat.single_fm.encoding.stats import (
    block_permute_indices,
    permutation_test_unique_variance,
    permutation_test_unique_variance_3way,
)

# Variance partitioning
from freemat.single_fm.encoding.variance import (
    UniqueVarianceResult,
    UniqueVariance3WayResult,
    UniqueVariance3WayShuffleResult,
    compute_unique_variance_kin_vis,
    compute_unique_variance_3way,
    compute_unique_variance_3way_shuffle,
)

__all__ = [
    "CVConfig",
    "UniqueVarianceResult",
    "UniqueVariance3WayResult",
    "UniqueVariance3WayShuffleResult",
    "DEFAULT_ALPHAS",
    "blocked_cv_folds",
    "compute_cv_r2",
    "compute_cv_r2_fixed_alpha",
    "ridge_fit_predict",
    "ridge_fit_predict_gpu",
    "r2_score",
    "build_spline_design",
    "block_permute_indices",
    "permutation_test_unique_variance",
    "permutation_test_unique_variance_3way",
    "compute_unique_variance_kin_vis",
    "compute_unique_variance_3way",
    "compute_unique_variance_3way_shuffle",
    "HAS_GPU",
    "HAS_NUMBA",
]
