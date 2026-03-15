"""Poisson GLM fitting with cross-validation for neural encoding.

This module provides functions for fitting Generalized Linear Models with
Poisson distribution and log link function, appropriate for neural spike
count data.

Key Features:
    - L2 (ridge) regularization
    - Temporal cross-validation strategies
    - Alpha grid search for hyperparameter tuning
    - Embargo zones to prevent information leakage
    - Nested CV for per-fold alpha selection
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cv import (
    temporal_kfold_indices,
    temporal_kfold_indices_with_boundaries,
    KFold_cpu,
)
from .poisson import poisson_loglik, poisson_deviance

logger = logging.getLogger(__name__)

# Handle sklearn import
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
    HAS_SKLEARN = True
except (ImportError, ValueError):
    HAS_SKLEARN = False
    logger.warning("sklearn not available, using fallback implementations")

    class StandardScaler:
        """Fallback StandardScaler when sklearn unavailable."""
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    class PoissonRegressor:
        """Fallback Poisson regressor (OLS approximation)."""
        def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-3):
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, X, y):
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            # Simple ridge regression as fallback
            n_features = X.shape[1]
            A = X.T @ X + self.alpha * np.eye(n_features)
            b = X.T @ np.log(np.clip(y, 1e-10, None) + 1)
            params = np.linalg.solve(A, b)
            if self.fit_intercept:
                self.intercept_ = params[0]
                self.coef_ = params[1:]
            else:
                self.coef_ = params
            return self

        def predict(self, X):
            linear = X @ self.coef_ + self.intercept_
            return np.exp(linear)

        def score(self, X, y):
            mu = self.predict(X)
            dev_model = poisson_deviance(y, mu)
            dev_null = poisson_deviance(y, np.full_like(y, np.mean(y)))
            return 1 - dev_model / dev_null if dev_null > 1e-12 else 0.0


def fit_poisson_glm_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    n_splits_cv: int,
    alpha_glm: float,
    scaler: "StandardScaler" = None,
    random_state_cv: Optional[int] = None,
    alpha_grid: Optional[List[float]] = None,
    cv_strategy: str = "temporal_kfold",
    behavior_change_indices: Optional[np.ndarray] = None,
    cv_score_metric: str = "d2",
    embargo_bins: int = 0,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """Fit Poisson GLM with L2 regularization using cross-validation.

    Fits a Generalized Linear Model with Poisson distribution and log link
    function, appropriate for count/rate data. Uses k-fold cross-validation
    to evaluate model performance.

    Mathematical Model:
        y ~ Poisson(λ)
        log(λ) = Xβ + b
        Loss = -ℓ(μ; y) + α||β||²

    Args:
        X: Design matrix of shape (n_samples, n_features).
        y: Target neural activity, shape (n_samples,). Should be non-negative.
        n_splits_cv: Number of cross-validation folds.
        alpha_glm: L2 regularization strength. Higher values increase
            regularization. Use 0 for unregularized model.
        scaler: Optional StandardScaler for feature normalization.
        random_state_cv: Random seed for reproducible CV splits.
        alpha_grid: Optional list of alpha values for grid search.
        cv_strategy: CV strategy - 'temporal_kfold', 'temporal_kfold_behavior_boundaries',
            or 'random'.
        behavior_change_indices: Indices where behavior changes (for boundary-aware CV).
        cv_score_metric: Metric for alpha selection - 'd2', 'loglik', or 'deviance'.
        embargo_bins: Number of bins to exclude around fold boundaries.

    Returns:
        Tuple containing:
            - mean_d2_score: Mean D² (deviance explained) across CV folds
            - coefficients: Fitted model coefficients, shape (n_features,)
            - intercept: Fitted model intercept
            - y_pred: Predictions on full dataset from final model

    Note:
        D² score is analogous to R² for Poisson regression, ranging from
        -∞ to 1, where 1 indicates perfect prediction and 0 indicates
        performance equivalent to predicting the mean.

    Example:
        >>> d2, coef, intercept, y_pred = fit_poisson_glm_elastic_net(
        >>>     X_design, spike_counts, n_splits_cv=5, alpha_glm=0.1
        >>> )
        >>> print(f"Cross-validated D²: {d2:.3f}")
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)

    if scaler is None:
        scaler = StandardScaler()

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X_np.shape[0] < n_splits_cv:
        logger.warning(
            f"Number of samples ({X_np.shape[0]}) is less than n_splits_cv ({n_splits_cv}). "
            "Returning NaN score."
        )
        num_coeffs = X_np.shape[1] if X_np.ndim == 2 else 0
        return np.nan, np.zeros(num_coeffs), np.nan, np.array([])

    # Build CV splits
    if cv_strategy == "temporal_kfold_behavior_boundaries":
        splits = temporal_kfold_indices_with_boundaries(
            X_np.shape[0], n_splits_cv, behavior_change_indices
        )
    elif cv_strategy == "temporal_kfold":
        splits = temporal_kfold_indices(X_np.shape[0], n_splits_cv)
    else:
        kf = KFold_cpu(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)
        splits = list(kf.split(X_np))

    def _apply_embargo(train_idx, test_idx):
        if embargo_bins <= 0:
            return train_idx
        if len(test_idx) == 0:
            return train_idx
        tmin = int(np.min(test_idx)) - embargo_bins
        tmax = int(np.max(test_idx)) + embargo_bins
        mask = (train_idx < tmin) | (train_idx > tmax)
        return train_idx[mask]

    def _cv_score_for_alpha(alpha_value: float, metric: str = "d2") -> float:
        scores_local: List[float] = []
        for train_idx, test_idx in splits:
            if cv_strategy.startswith("temporal_kfold"):
                train_idx = _apply_embargo(train_idx, test_idx)
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = PoissonRegressor(alpha=alpha_value, fit_intercept=True, max_iter=1000, tol=1e-3)
            model.fit(X_train_scaled, y_train)

            if metric == "d2":
                score = model.score(X_test_scaled, y_test)
                scores_local.append(score)
            elif metric == "loglik":
                mu = model.predict(X_test_scaled)
                scores_local.append(poisson_loglik(y_test, mu))
            elif metric == "deviance":
                mu = model.predict(X_test_scaled)
                scores_local.append(-poisson_deviance(y_test, mu))
            else:
                score = model.score(X_test_scaled, y_test)
                scores_local.append(score)

        return float(np.nanmean(np.array(scores_local))) if scores_local else np.nan

    # Alpha selection via grid search
    if alpha_grid and len(alpha_grid) > 0:
        mean_scores = [(a, _cv_score_for_alpha(a, cv_score_metric)) for a in alpha_grid]
        alpha_scores = [s for (_, s) in mean_scores]
        if all(np.isnan(alpha_scores)):
            mean_d2_score = np.nan
            best_alpha = alpha_glm
        else:
            best_alpha, mean_d2_score = max(
                mean_scores, key=lambda x: (-np.inf if np.isnan(x[1]) else x[1])
            )
    else:
        best_alpha = alpha_glm
        mean_d2_score = _cv_score_for_alpha(best_alpha, cv_score_metric)

    # Final fit on all data
    X_scaled_full = scaler.fit_transform(X_np)
    final_model = PoissonRegressor(alpha=best_alpha, fit_intercept=True, max_iter=1000, tol=1e-3)
    final_model.fit(X_scaled_full, y_np)
    coeffs = final_model.coef_
    intercept = final_model.intercept_
    y_pred = final_model.predict(X_scaled_full)

    logger.info(
        f"Fit Poisson GLM. Mean {cv_score_metric}: {mean_d2_score:.4f}. "
        f"Best alpha: {best_alpha}. Num coeffs: {len(coeffs) if hasattr(coeffs, '__len__') else 0}"
    )
    return mean_d2_score, np.array(coeffs), intercept, np.array(y_pred)


def fit_poisson_glm_cv_detailed(
    X: np.ndarray,
    y: np.ndarray,
    n_splits_cv: int,
    alpha_init: float,
    scaler: "StandardScaler" = None,
    random_state_cv: Optional[int] = None,
    alpha_grid: Optional[List[float]] = None,
    cv_strategy: str = "temporal_kfold",
    per_fold_alpha: bool = False,
    inner_cv_n_splits: Optional[int] = None,
    behavior_change_indices_full: Optional[np.ndarray] = None,
    cv_score_metric: str = "d2",
    embargo_bins: int = 0,
) -> Tuple[Any, np.ndarray, float, np.ndarray, Dict[str, Any]]:
    """Fit Poisson GLM with cross-validation; return detailed CV info.

    If `per_fold_alpha` is True and `alpha_grid` provided, performs nested CV:
    for each outer fold, selects best alpha via inner temporal CV.

    Args:
        X: Design matrix of shape (n_samples, n_features).
        y: Target neural activity, shape (n_samples,).
        n_splits_cv: Number of outer cross-validation folds.
        alpha_init: Initial/default regularization strength.
        scaler: Optional StandardScaler for feature normalization.
        random_state_cv: Random seed for reproducible CV splits.
        alpha_grid: Optional list of alpha values for grid search.
        cv_strategy: CV strategy.
        per_fold_alpha: If True, select alpha per outer fold via nested CV.
        inner_cv_n_splits: Number of inner CV folds for nested alpha selection.
        behavior_change_indices_full: Indices where behavior changes.
        cv_score_metric: Metric for alpha selection.
        embargo_bins: Number of bins to exclude around fold boundaries.

    Returns:
        Tuple containing:
            - final_model: Fitted model on full data
            - coefficients: Model coefficients
            - intercept: Model intercept
            - y_pred: Predictions on full data
            - cv_info: Dictionary with detailed CV information including
              per-fold scores, selected alphas, etc.
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)

    if scaler is None:
        scaler = StandardScaler()

    if inner_cv_n_splits is None:
        inner_cv_n_splits = max(2, n_splits_cv - 1)

    # Build outer CV splits
    if cv_strategy == "temporal_kfold_behavior_boundaries":
        splits = temporal_kfold_indices_with_boundaries(
            X_np.shape[0], n_splits_cv, behavior_change_indices_full
        )
    elif cv_strategy == "temporal_kfold":
        splits = temporal_kfold_indices(X_np.shape[0], n_splits_cv)
    else:
        kf = KFold_cpu(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)
        splits = list(kf.split(X_np))

    def _apply_embargo(train_idx, test_idx):
        if embargo_bins <= 0:
            return train_idx
        if len(test_idx) == 0:
            return train_idx
        tmin = int(np.min(test_idx)) - embargo_bins
        tmax = int(np.max(test_idx)) + embargo_bins
        mask = (train_idx < tmin) | (train_idx > tmax)
        return train_idx[mask]

    d2_per_fold = []
    alpha_per_fold = []

    for train_idx, test_idx in splits:
        if cv_strategy.startswith("temporal_kfold"):
            train_idx = _apply_embargo(train_idx, test_idx)

        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Alpha selection for this fold
        if per_fold_alpha and alpha_grid:
            best_inner_alpha = alpha_init
            best_inner_score = -np.inf

            inner_splits = temporal_kfold_indices(len(X_train_scaled), inner_cv_n_splits)
            for alpha_candidate in alpha_grid:
                inner_scores = []
                for inner_train, inner_test in inner_splits:
                    X_inner_train = X_train_scaled[inner_train]
                    X_inner_test = X_train_scaled[inner_test]
                    y_inner_train = y_train[inner_train]
                    y_inner_test = y_train[inner_test]

                    model = PoissonRegressor(alpha=alpha_candidate, fit_intercept=True, max_iter=1000)
                    model.fit(X_inner_train, y_inner_train)
                    inner_scores.append(model.score(X_inner_test, y_inner_test))

                mean_inner = np.nanmean(inner_scores)
                if mean_inner > best_inner_score:
                    best_inner_score = mean_inner
                    best_inner_alpha = alpha_candidate

            fold_alpha = best_inner_alpha
        else:
            fold_alpha = alpha_init

        alpha_per_fold.append(fold_alpha)

        # Fit and evaluate on outer fold
        model = PoissonRegressor(alpha=fold_alpha, fit_intercept=True, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        d2_per_fold.append(model.score(X_test_scaled, y_test))

    # Final fit on all data
    best_alpha_global = np.median(alpha_per_fold) if alpha_per_fold else alpha_init
    X_scaled_full = scaler.fit_transform(X_np)
    final_model = PoissonRegressor(alpha=best_alpha_global, fit_intercept=True, max_iter=1000)
    final_model.fit(X_scaled_full, y_np)

    cv_info = {
        "d2_cv_mean": float(np.nanmean(d2_per_fold)),
        "d2_cv_per_fold": d2_per_fold,
        "alpha_per_fold": alpha_per_fold,
        "alpha_selected_global": best_alpha_global,
    }

    return (
        final_model,
        np.array(final_model.coef_),
        final_model.intercept_,
        final_model.predict(X_scaled_full),
        cv_info,
    )


__all__ = [
    "fit_poisson_glm_elastic_net",
    "fit_poisson_glm_cv_detailed",
    "StandardScaler",
    "PoissonRegressor",
]
