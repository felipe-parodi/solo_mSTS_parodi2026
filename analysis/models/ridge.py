"""Ridge regression for neural encoding with temporal cross-validation.

Implements Musall/Churchland-style ridge regression for predicting neural
activity from kinematic features, with proper temporal CV to avoid leakage.

Key functions:
- fit_ridge(): Fit ridge regression with fixed lambda
- fit_ridge_cv(): Fit with nested CV for lambda selection
- compute_r2(): Compute coefficient of determination

Usage:
    from analysis.models.ridge import fit_ridge_cv, compute_r2

    result = fit_ridge_cv(X, y, n_folds=5, lambda_grid=np.logspace(-4, 4, 20))
    print(f"CV R²: {result['cv_r2']:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeResult:
    """Result container for ridge regression fit."""

    coefficients: np.ndarray
    intercept: float
    lambda_opt: float
    cv_r2: float
    train_r2: float
    fold_results: List[Dict]


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination).

    Args:
        y_true: True target values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        R² score. Can be negative if predictions are worse than mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, float]:
    """Fit ridge regression with fixed regularization.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        alpha: Regularization strength (lambda).
        fit_intercept: Whether to fit intercept term.

    Returns:
        Tuple of (coefficients, intercept).
    """
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    return model.coef_, model.intercept_


def predict_ridge(
    X: np.ndarray,
    coefficients: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Predict using ridge regression coefficients.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        coefficients: Ridge coefficients, shape (n_features,).
        intercept: Intercept term.

    Returns:
        Predictions, shape (n_samples,).
    """
    return X @ coefficients + intercept


def _temporal_kfold_indices(
    n_samples: int,
    n_folds: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create contiguous temporal folds for cross-validation.

    Splits data into n_folds contiguous blocks to preserve temporal structure
    and avoid autocorrelation leakage.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    fold_size = n_samples // n_folds
    folds = []

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples

        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n_samples),
        ])

        folds.append((train_idx, test_idx))

    return folds


def _apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo_bins: int,
) -> np.ndarray:
    """Remove training samples within embargo distance of test set.

    Args:
        train_idx: Training indices.
        test_idx: Test indices.
        embargo_bins: Number of bins to exclude around test boundaries.

    Returns:
        Filtered training indices.
    """
    if embargo_bins <= 0:
        return train_idx

    test_min, test_max = test_idx.min(), test_idx.max()
    embargo_start = test_min - embargo_bins
    embargo_end = test_max + embargo_bins

    mask = (train_idx < embargo_start) | (train_idx > embargo_end)
    return train_idx[mask]


def fit_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    lambda_grid: Optional[np.ndarray] = None,
    inner_folds: int = 4,
    embargo_bins: int = 0,
    standardize: bool = True,
) -> RidgeResult:
    """Fit ridge regression with nested temporal cross-validation.

    Outer loop: Evaluate model performance with temporal CV.
    Inner loop: Select optimal lambda via CV on training data.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        n_folds: Number of outer CV folds.
        lambda_grid: Grid of regularization values to search.
            Defaults to logspace(-6, 6, 20).
        inner_folds: Number of inner CV folds for lambda selection.
        embargo_bins: Number of bins to exclude around test boundaries.
        standardize: Whether to z-score features within each fold.

    Returns:
        RidgeResult with coefficients, optimal lambda, and CV metrics.
    """
    if lambda_grid is None:
        lambda_grid = np.logspace(-6, 6, 20)

    n_samples, n_features = X.shape
    outer_folds = _temporal_kfold_indices(n_samples, n_folds)

    fold_results = []
    test_predictions = np.zeros(n_samples)
    test_mask = np.zeros(n_samples, dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        # Apply embargo
        train_idx_clean = _apply_embargo(train_idx, test_idx, embargo_bins)

        X_train, X_test = X[train_idx_clean], X[test_idx]
        y_train, y_test = y[train_idx_clean], y[test_idx]

        # Standardize within fold (fit on train only)
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Inner CV to select lambda
        lambda_opt = _select_lambda_inner_cv(
            X_train, y_train, lambda_grid, inner_folds
        )

        # Fit final model on full training set
        coef, intercept = fit_ridge(X_train, y_train, alpha=lambda_opt)

        # Evaluate on test set
        y_pred = predict_ridge(X_test, coef, intercept)
        test_r2 = compute_r2(y_test, y_pred)

        # Store predictions for full CV R² computation
        test_predictions[test_idx] = y_pred
        test_mask[test_idx] = True

        fold_results.append({
            "fold": fold_idx,
            "lambda_opt": lambda_opt,
            "test_r2": test_r2,
            "n_train": len(train_idx_clean),
            "n_test": len(test_idx),
        })

    # Compute overall CV R² from held-out predictions
    cv_r2 = compute_r2(y[test_mask], test_predictions[test_mask])

    # Fit final model on all data with median optimal lambda
    lambda_final = np.median([f["lambda_opt"] for f in fold_results])

    if standardize:
        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
    else:
        X_scaled = X

    coef_final, intercept_final = fit_ridge(X_scaled, y, alpha=lambda_final)
    train_r2 = compute_r2(y, predict_ridge(X_scaled, coef_final, intercept_final))

    return RidgeResult(
        coefficients=coef_final,
        intercept=intercept_final,
        lambda_opt=lambda_final,
        cv_r2=cv_r2,
        train_r2=train_r2,
        fold_results=fold_results,
    )


def _select_lambda_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    lambda_grid: np.ndarray,
    n_folds: int,
) -> float:
    """Select optimal lambda via inner cross-validation.

    Args:
        X: Training design matrix (already scaled).
        y: Training target vector.
        lambda_grid: Grid of lambda values to try.
        n_folds: Number of CV folds.

    Returns:
        Optimal lambda value.
    """
    n_samples = X.shape[0]
    inner_folds = _temporal_kfold_indices(n_samples, n_folds)

    best_lambda = lambda_grid[0]
    best_score = -np.inf

    for alpha in lambda_grid:
        scores = []

        for train_idx, val_idx in inner_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            coef, intercept = fit_ridge(X_train, y_train, alpha=alpha)
            y_pred = predict_ridge(X_val, coef, intercept)
            r2 = compute_r2(y_val, y_pred)

            if not np.isnan(r2):
                scores.append(r2)

        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_lambda = alpha

    return best_lambda


def fit_ridge_per_unit(
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    lambda_grid: Optional[np.ndarray] = None,
    embargo_bins: int = 0,
    standardize: bool = True,
) -> List[RidgeResult]:
    """Fit ridge regression for each unit (column) in Y.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        Y: Neural activity matrix, shape (n_samples, n_units).
        n_folds: Number of outer CV folds.
        lambda_grid: Grid of regularization values.
        embargo_bins: Temporal embargo around test folds.
        standardize: Whether to z-score features.

    Returns:
        List of RidgeResult, one per unit.
    """
    n_units = Y.shape[1]
    results = []

    for unit_idx in range(n_units):
        y_unit = Y[:, unit_idx]
        result = fit_ridge_cv(
            X, y_unit,
            n_folds=n_folds,
            lambda_grid=lambda_grid,
            embargo_bins=embargo_bins,
            standardize=standardize,
        )
        results.append(result)

    return results
