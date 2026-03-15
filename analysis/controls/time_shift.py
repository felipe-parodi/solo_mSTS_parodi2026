"""Time-shift control for neural encoding validation.

Circularly shifts predictors (X) relative to neural activity (y) to break
temporal alignment while preserving autocorrelation structure.

If encoding is genuine, time-shifted controls should collapse to ~0 R².

Usage:
    from analysis.controls.time_shift import run_time_shift_control

    control_result = run_time_shift_control(
        X, y,
        group_indices=group_indices,
        n_shifts=100,
        shift_range_s=(30, 120),
        bin_width_s=0.1,
    )
    print(f"p-value: {control_result['p_value']:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.models.ridge import fit_ridge_cv, compute_r2
from analysis.unique_variance import compute_unique_variance, UniqueVarianceResult


@dataclass
class TimeShiftControlResult:
    """Result container for time-shift control analysis."""

    actual_r2: float
    null_r2_values: np.ndarray
    null_r2_mean: float
    null_r2_std: float
    p_value: float
    actual_uv: Optional[Dict[str, float]]
    null_uv_values: Optional[Dict[str, np.ndarray]]
    n_shifts: int
    shift_range_bins: Tuple[int, int]


def time_shift_predictors(
    X: np.ndarray,
    shift_bins: int,
) -> np.ndarray:
    """Circularly shift predictor matrix by a fixed number of bins.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        shift_bins: Number of bins to shift (positive = forward).

    Returns:
        Shifted design matrix.
    """
    return np.roll(X, shift=shift_bins, axis=0)


def run_time_shift_control(
    X: np.ndarray,
    y: np.ndarray,
    n_shifts: int = 100,
    shift_range_s: Tuple[float, float] = (30.0, 120.0),
    bin_width_s: float = 0.1,
    group_indices: Optional[Dict[str, List[int]]] = None,
    n_folds: int = 5,
    lambda_grid: Optional[np.ndarray] = None,
    embargo_bins: int = 0,
    random_seed: int = 42,
) -> TimeShiftControlResult:
    """Run time-shift control analysis.

    Repeatedly shifts predictors by random offsets and refits model.
    Null distribution should center near 0 if encoding is genuine.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        n_shifts: Number of random shifts to perform.
        shift_range_s: Range of shift magnitudes in seconds.
        bin_width_s: Bin width in seconds (to convert shifts to bins).
        group_indices: Optional group indices for UV analysis.
        n_folds: Number of CV folds.
        lambda_grid: Regularization grid.
        embargo_bins: Temporal embargo.
        random_seed: Random seed for reproducibility.

    Returns:
        TimeShiftControlResult with null distribution and p-value.
    """
    rng = np.random.RandomState(random_seed)

    # Convert shift range to bins
    min_shift_bins = int(shift_range_s[0] / bin_width_s)
    max_shift_bins = int(shift_range_s[1] / bin_width_s)

    # Fit actual model
    actual_result = fit_ridge_cv(
        X, y,
        n_folds=n_folds,
        lambda_grid=lambda_grid,
        embargo_bins=embargo_bins,
    )
    actual_r2 = actual_result.cv_r2

    # Compute actual UV if group indices provided
    actual_uv = None
    if group_indices is not None:
        uv_result = compute_unique_variance(
            X, y,
            group_indices=group_indices,
            n_folds=n_folds,
            lambda_grid=lambda_grid,
            embargo_bins=embargo_bins,
        )
        actual_uv = uv_result.unique_variance

    # Generate null distribution
    null_r2_values = []
    null_uv_values: Optional[Dict[str, List[float]]] = None

    if group_indices is not None:
        null_uv_values = {g: [] for g in group_indices.keys()}

    for _ in range(n_shifts):
        # Random shift (positive or negative)
        shift_magnitude = rng.randint(min_shift_bins, max_shift_bins + 1)
        shift_direction = rng.choice([-1, 1])
        shift_bins = shift_magnitude * shift_direction

        # Shift predictors
        X_shifted = time_shift_predictors(X, shift_bins)

        # Fit model on shifted data
        shifted_result = fit_ridge_cv(
            X_shifted, y,
            n_folds=n_folds,
            lambda_grid=lambda_grid,
            embargo_bins=embargo_bins,
        )
        null_r2_values.append(shifted_result.cv_r2)

        # Compute UV on shifted data
        if group_indices is not None and null_uv_values is not None:
            shifted_uv = compute_unique_variance(
                X_shifted, y,
                group_indices=group_indices,
                n_folds=n_folds,
                lambda_grid=lambda_grid,
                embargo_bins=embargo_bins,
            )
            for g, uv in shifted_uv.unique_variance.items():
                null_uv_values[g].append(uv)

    null_r2_values = np.array(null_r2_values)

    # Compute p-value (one-sided: actual > null)
    n_greater = np.sum(null_r2_values >= actual_r2)
    p_value = (n_greater + 1) / (n_shifts + 1)

    # Convert null UV lists to arrays
    null_uv_arrays = None
    if null_uv_values is not None:
        null_uv_arrays = {g: np.array(v) for g, v in null_uv_values.items()}

    return TimeShiftControlResult(
        actual_r2=actual_r2,
        null_r2_values=null_r2_values,
        null_r2_mean=float(np.mean(null_r2_values)),
        null_r2_std=float(np.std(null_r2_values)),
        p_value=p_value,
        actual_uv=actual_uv,
        null_uv_values=null_uv_arrays,
        n_shifts=n_shifts,
        shift_range_bins=(min_shift_bins, max_shift_bins),
    )


def run_time_shift_control_per_unit(
    X: np.ndarray,
    Y: np.ndarray,
    n_shifts: int = 100,
    shift_range_s: Tuple[float, float] = (30.0, 120.0),
    bin_width_s: float = 0.1,
    group_indices: Optional[Dict[str, List[int]]] = None,
    n_folds: int = 5,
    lambda_grid: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> List[TimeShiftControlResult]:
    """Run time-shift control for each unit.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        Y: Neural activity matrix, shape (n_samples, n_units).
        n_shifts: Number of random shifts.
        shift_range_s: Range of shift magnitudes.
        bin_width_s: Bin width.
        group_indices: Optional group indices for UV.
        n_folds: Number of CV folds.
        lambda_grid: Regularization grid.
        random_seed: Random seed.

    Returns:
        List of TimeShiftControlResult, one per unit.
    """
    n_units = Y.shape[1]
    results = []

    for unit_idx in range(n_units):
        y_unit = Y[:, unit_idx]
        result = run_time_shift_control(
            X, y_unit,
            n_shifts=n_shifts,
            shift_range_s=shift_range_s,
            bin_width_s=bin_width_s,
            group_indices=group_indices,
            n_folds=n_folds,
            lambda_grid=lambda_grid,
            random_seed=random_seed + unit_idx,  # Different seed per unit
        )
        results.append(result)

    return results
