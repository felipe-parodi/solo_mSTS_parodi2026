"""Unique variance (dropout) analysis for neural encoding.

Implements Musall et al. 2019 / Churchland-style variance partitioning:
- Fit full model with all predictor groups
- For each group, refit without that group
- Unique variance = R²_full - R²_without_group

Usage:
    from analysis.unique_variance import compute_unique_variance
    from analysis.features.groups import get_group_indices

    group_indices = get_group_indices(feature_names)
    uv_result = compute_unique_variance(X, y, group_indices)
    print(uv_result["unique_variance"])  # {"G1_translation": 0.05, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.models.ridge import (
    RidgeResult,
    compute_r2,
    fit_ridge_cv,
)


@dataclass
class UniqueVarianceResult:
    """Result container for unique variance analysis."""

    full_r2: float
    unique_variance: Dict[str, float]
    dropout_r2: Dict[str, float]
    unique_percent: Dict[str, float]
    unexplained_percent: float
    n_features: int
    group_sizes: Dict[str, int]


def compute_unique_variance(
    X: np.ndarray,
    y: np.ndarray,
    group_indices: Dict[str, List[int]],
    n_folds: int = 5,
    lambda_grid: Optional[np.ndarray] = None,
    embargo_bins: int = 0,
    standardize: bool = True,
) -> UniqueVarianceResult:
    """Compute unique variance explained by each predictor group.

    For each group g:
        UV_g = R²_full - R²_without_g

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        group_indices: Dict mapping group name to feature indices.
        n_folds: Number of CV folds.
        lambda_grid: Regularization grid for ridge CV.
        embargo_bins: Temporal embargo around test folds.
        standardize: Whether to z-score features.

    Returns:
        UniqueVarianceResult with unique variance per group.
    """
    n_samples, n_features = X.shape

    # Fit full model
    full_result = fit_ridge_cv(
        X, y,
        n_folds=n_folds,
        lambda_grid=lambda_grid,
        embargo_bins=embargo_bins,
        standardize=standardize,
    )
    full_r2 = full_result.cv_r2

    # Compute unique variance for each group
    unique_variance: Dict[str, float] = {}
    dropout_r2: Dict[str, float] = {}
    group_sizes: Dict[str, int] = {}

    for group_name, indices in group_indices.items():
        # Create design matrix without this group
        keep_mask = np.ones(n_features, dtype=bool)
        keep_mask[indices] = False
        X_dropout = X[:, keep_mask]

        group_sizes[group_name] = len(indices)

        # Fit model without this group
        if X_dropout.shape[1] > 0:
            dropout_result = fit_ridge_cv(
                X_dropout, y,
                n_folds=n_folds,
                lambda_grid=lambda_grid,
                embargo_bins=embargo_bins,
                standardize=standardize,
            )
            r2_without = dropout_result.cv_r2
        else:
            # No features left - R² is effectively 0
            r2_without = 0.0

        dropout_r2[group_name] = r2_without
        unique_variance[group_name] = max(0.0, full_r2 - r2_without)

    # Compute percentages
    unique_percent: Dict[str, float] = {}
    total_unique = 0.0

    for group_name, uv in unique_variance.items():
        if full_r2 > 0:
            pct = (uv / full_r2) * 100
        else:
            pct = 0.0
        unique_percent[group_name] = pct
        total_unique += pct

    unexplained_percent = max(0.0, 100.0 - total_unique)

    return UniqueVarianceResult(
        full_r2=full_r2,
        unique_variance=unique_variance,
        dropout_r2=dropout_r2,
        unique_percent=unique_percent,
        unexplained_percent=unexplained_percent,
        n_features=n_features,
        group_sizes=group_sizes,
    )


def compute_unique_variance_per_unit(
    X: np.ndarray,
    Y: np.ndarray,
    group_indices: Dict[str, List[int]],
    n_folds: int = 5,
    lambda_grid: Optional[np.ndarray] = None,
    embargo_bins: int = 0,
    standardize: bool = True,
) -> List[UniqueVarianceResult]:
    """Compute unique variance for each unit (column) in Y.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        Y: Neural activity matrix, shape (n_samples, n_units).
        group_indices: Dict mapping group name to feature indices.
        n_folds: Number of CV folds.
        lambda_grid: Regularization grid.
        embargo_bins: Temporal embargo.
        standardize: Whether to z-score.

    Returns:
        List of UniqueVarianceResult, one per unit.
    """
    n_units = Y.shape[1]
    results = []

    for unit_idx in range(n_units):
        y_unit = Y[:, unit_idx]
        result = compute_unique_variance(
            X, y_unit,
            group_indices=group_indices,
            n_folds=n_folds,
            lambda_grid=lambda_grid,
            embargo_bins=embargo_bins,
            standardize=standardize,
        )
        results.append(result)

    return results


def aggregate_unique_variance(
    results: List[UniqueVarianceResult],
    bank_labels: Optional[List[str]] = None,
) -> Dict:
    """Aggregate unique variance results across units.

    Args:
        results: List of per-unit UniqueVarianceResult.
        bank_labels: Optional list of bank labels ("dorsal"/"ventral") per unit.

    Returns:
        Dict with aggregated statistics.
    """
    n_units = len(results)

    # Extract per-unit values
    full_r2_values = np.array([r.full_r2 for r in results])

    # Get group names from first result
    group_names = list(results[0].unique_variance.keys())

    # Aggregate UV per group
    uv_per_group: Dict[str, np.ndarray] = {}
    for group_name in group_names:
        uv_per_group[group_name] = np.array([
            r.unique_variance[group_name] for r in results
        ])

    summary = {
        "n_units": n_units,
        "full_r2": {
            "mean": float(np.mean(full_r2_values)),
            "std": float(np.std(full_r2_values)),
            "median": float(np.median(full_r2_values)),
            "values": full_r2_values,
        },
        "unique_variance": {},
    }

    for group_name, uv_values in uv_per_group.items():
        summary["unique_variance"][group_name] = {
            "mean": float(np.mean(uv_values)),
            "std": float(np.std(uv_values)),
            "median": float(np.median(uv_values)),
            "values": uv_values,
        }

    # Split by bank if labels provided
    if bank_labels is not None:
        bank_labels = np.array(bank_labels)
        unique_banks = np.unique(bank_labels)

        summary["by_bank"] = {}
        for bank in unique_banks:
            bank_mask = bank_labels == bank
            bank_results = [r for r, m in zip(results, bank_mask) if m]

            if bank_results:
                summary["by_bank"][bank] = aggregate_unique_variance(bank_results)

    return summary
