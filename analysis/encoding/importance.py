"""Feature importance analysis for neural encoding models.

This module provides functions for computing feature importance through
leave-one-out analysis, computing per-behavior and per-lag contributions
to model performance.

Key Metrics:
    - Delta D² per behavior: Importance via leave-one-behavior-out
    - Delta D² per lag: Importance of each temporal lag
    - Per-behavior D²: Model fit during specific behavioral states
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .glm import fit_poisson_glm_elastic_net

logger = logging.getLogger(__name__)


def compute_delta_d2_per_behavior(
    X_design: np.ndarray,
    y: np.ndarray,
    feature_idx_to_behavior: Dict[int, str],
    full_model_d2: float,
    n_splits_cv: int,
    alpha_glm: float,
    random_state_cv: Optional[int] = None,
    alpha_grid: Optional[List[float]] = None,
    cv_strategy: str = "temporal_kfold",
    cv_score_metric: str = "d2",
    embargo_bins: int = 0,
) -> Dict[str, float]:
    """Compute Delta D² for each behavior via leave-one-behavior-out.

    For each behavior, refits the model excluding all features associated
    with that behavior and computes the drop in D². Larger drops indicate
    more important behaviors.

    Mathematical Definition:
        ΔD²(behavior) = D²_full - D²_without_behavior

        Where D²_full uses all features and D²_without_behavior excludes
        all columns associated with that behavior across all lags.

    Args:
        X_design: Full design matrix, shape (n_samples, n_features).
        y: Target neural activity, shape (n_samples,).
        feature_idx_to_behavior: Mapping from column index to behavior label.
        full_model_d2: D² score from full model (pre-computed).
        n_splits_cv: Number of CV folds.
        alpha_glm: Regularization strength.
        random_state_cv: Random seed for CV.
        alpha_grid: Optional alpha grid for selection.
        cv_strategy: CV strategy.
        cv_score_metric: Metric for model evaluation.
        embargo_bins: Bins to exclude around fold boundaries.

    Returns:
        Dictionary mapping behavior labels to their ΔD² values.
        Positive values indicate the behavior improves model fit.

    Example:
        >>> delta_d2 = compute_delta_d2_per_behavior(
        >>>     X, y, feature_map, d2_full, n_splits_cv=5, alpha_glm=0.1
        >>> )
        >>> for beh, importance in sorted(delta_d2.items(), key=lambda x: -x[1]):
        >>>     print(f"{beh}: ΔD² = {importance:.4f}")
    """
    if X_design.shape[1] <= 1:
        return {}

    unique_behaviors = sorted(list(set(feature_idx_to_behavior.values())))
    delta_d2_scores = {}

    for behavior_to_remove in unique_behaviors:
        cols_to_keep = [i for i, beh in feature_idx_to_behavior.items() if beh != behavior_to_remove]

        if not cols_to_keep:
            d2_reduced = 0.0
        else:
            X_reduced = X_design[:, cols_to_keep]
            d2_reduced, _, _, _ = fit_poisson_glm_elastic_net(
                X_reduced,
                y,
                n_splits_cv,
                alpha_glm,
                random_state_cv=random_state_cv,
                alpha_grid=alpha_grid,
                cv_strategy=cv_strategy,
                behavior_change_indices=None,
                cv_score_metric=cv_score_metric,
                embargo_bins=embargo_bins,
            )

        if not np.isnan(full_model_d2) and not np.isnan(d2_reduced):
            delta_d2_scores[behavior_to_remove] = full_model_d2 - d2_reduced

    return delta_d2_scores


def compute_delta_d2_per_lag(
    X_design: np.ndarray,
    y: np.ndarray,
    feature_idx_to_lag: Dict[int, int],
    full_model_d2: float,
    n_splits_cv: int,
    alpha_glm: float,
    random_state_cv: Optional[int] = None,
    alpha_grid: Optional[List[float]] = None,
    cv_strategy: str = "temporal_kfold",
    cv_score_metric: str = "d2",
    embargo_bins: int = 0,
) -> Dict[str, float]:
    """Compute Delta D² for each temporal lag via leave-one-lag-out.

    For each lag value, refits the model excluding all features at that
    lag and computes the drop in D². Helps identify which temporal
    offsets are most predictive.

    Args:
        X_design: Full design matrix, shape (n_samples, n_features).
        y: Target neural activity, shape (n_samples,).
        feature_idx_to_lag: Mapping from column index to lag value.
        full_model_d2: D² score from full model.
        n_splits_cv: Number of CV folds.
        alpha_glm: Regularization strength.
        random_state_cv: Random seed for CV.
        alpha_grid: Optional alpha grid.
        cv_strategy: CV strategy.
        cv_score_metric: Metric for evaluation.
        embargo_bins: Bins to exclude around boundaries.

    Returns:
        Dictionary mapping lag values (as strings) to their ΔD² values.

    Example:
        >>> delta_d2_lag = compute_delta_d2_per_lag(
        >>>     X, y, lag_map, d2_full, n_splits_cv=5, alpha_glm=0.1
        >>> )
        >>> # Find most important lag
        >>> best_lag = max(delta_d2_lag, key=delta_d2_lag.get)
        >>> print(f"Most important lag: {best_lag}")
    """
    if X_design.shape[1] <= 1 or len(set(feature_idx_to_lag.values())) <= 1:
        return {}

    unique_lags = sorted(list(set(feature_idx_to_lag.values())))
    delta_d2_per_lag = {}

    for lag_to_remove in unique_lags:
        cols_to_keep = [i for i, lag in feature_idx_to_lag.items() if lag != lag_to_remove]

        if not cols_to_keep:
            d2_reduced = 0.0
        else:
            X_reduced = X_design[:, cols_to_keep]
            d2_reduced, _, _, _ = fit_poisson_glm_elastic_net(
                X_reduced,
                y,
                n_splits_cv,
                alpha_glm,
                random_state_cv=random_state_cv,
                alpha_grid=alpha_grid,
                cv_strategy=cv_strategy,
                behavior_change_indices=None,
                cv_score_metric=cv_score_metric,
                embargo_bins=embargo_bins,
            )

        if not np.isnan(full_model_d2) and not np.isnan(d2_reduced):
            delta_d2_per_lag[str(lag_to_remove)] = full_model_d2 - d2_reduced

    return delta_d2_per_lag


def compute_per_behavior_d2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    binned_regressors: np.ndarray,
    behavior_labels: List[str],
    prefix: str = "",
) -> Dict[str, float]:
    """Compute D² score separately for each behavioral state.

    Evaluates model performance during specific behavioral states by
    computing D² only on time bins where that behavior is active.

    Args:
        y_true: True neural activity, shape (n_samples,).
        y_pred: Predicted neural activity, shape (n_samples,).
        binned_regressors: Binary behavior indicators, shape (n_samples, n_behaviors).
        behavior_labels: Names of behaviors in column order.
        prefix: Prefix for behavior names (e.g., 'self_', 'partner_').

    Returns:
        Dictionary mapping prefixed behavior names to D² scores.
        NaN values indicate insufficient samples for that behavior.

    Example:
        >>> per_beh_d2 = compute_per_behavior_d2(
        >>>     y_true, y_pred, self_regressors, ['rest', 'groom'],
        >>>     prefix='self_'
        >>> )
        >>> for beh, d2 in per_beh_d2.items():
        >>>     print(f"{beh}: D² = {d2:.3f}")
    """
    per_behavior_scores = {}
    overall_mean = np.mean(y_true)

    for i, behavior in enumerate(behavior_labels):
        if i >= binned_regressors.shape[1]:
            continue

        behavior_mask = binned_regressors[:, i] == 1
        y_true_bout = y_true[behavior_mask]
        y_pred_bout = y_pred[behavior_mask]

        if y_true_bout.size == 0:
            per_behavior_scores[f"{prefix}{behavior}"] = np.nan
            continue

        if y_true_bout.size == 1:
            per_behavior_scores[f"{prefix}{behavior}"] = np.nan
            continue

        ss_res = np.sum((y_true_bout - y_pred_bout) ** 2)
        ss_tot = np.sum((y_true_bout - overall_mean) ** 2)

        if ss_tot > 1e-9:
            d2_bout = 1 - (ss_res / ss_tot)
            d2_bout = np.clip(d2_bout, -1.0, 1.0)
        else:
            d2_bout = 0.0 if ss_res < 1e-9 else -1.0

        per_behavior_scores[f"{prefix}{behavior}"] = d2_bout

    return per_behavior_scores


def build_feature_to_behavior_map(
    regressor_labels_self: List[str],
    regressor_labels_partner: List[str],
    lags_in_bins: List[int],
) -> Dict[int, str]:
    """Build mapping from feature column index to behavior label.

    Args:
        regressor_labels_self: Self behavior labels.
        regressor_labels_partner: Partner behavior labels.
        lags_in_bins: Lag values used in design matrix.

    Returns:
        Dictionary mapping column indices to behavior labels.
    """
    feature_idx_to_behavior = {}
    col_idx = 0
    for lag in lags_in_bins:
        for r_label in regressor_labels_self:
            feature_idx_to_behavior[col_idx] = r_label
            col_idx += 1
        for r_label_p in regressor_labels_partner:
            feature_idx_to_behavior[col_idx] = r_label_p
            col_idx += 1
    return feature_idx_to_behavior


def build_feature_to_lag_map(
    regressor_labels_self: List[str],
    regressor_labels_partner: List[str],
    lags_in_bins: List[int],
) -> Dict[int, int]:
    """Build mapping from feature column index to lag value.

    Args:
        regressor_labels_self: Self behavior labels.
        regressor_labels_partner: Partner behavior labels.
        lags_in_bins: Lag values used in design matrix.

    Returns:
        Dictionary mapping column indices to lag values.
    """
    feature_idx_to_lag = {}
    col_idx = 0
    num_behaviors_per_lag = len(regressor_labels_self) + len(regressor_labels_partner)
    for lag in lags_in_bins:
        for _ in range(num_behaviors_per_lag):
            feature_idx_to_lag[col_idx] = lag
            col_idx += 1
    return feature_idx_to_lag


def zscore_coefficients(
    coefficients: np.ndarray,
    n_behavior_features: int,
    zscore_behavior_only: bool = True,
) -> Dict[str, Any]:
    """Z-score coefficients within a neuron.

    Normalizes coefficients to remove firing rate heterogeneity effects,
    essential for comparing coefficients across neurons.

    Args:
        coefficients: Model coefficients, shape (n_features,).
        n_behavior_features: Number of behavioral features (before controls).
        zscore_behavior_only: If True, compute mean/std from behavior features only.

    Returns:
        Dictionary with 'z_scored', 'mean', 'std', and 'zscore_behavior_only'.
    """
    if len(coefficients) == 0:
        return {
            "z_scored": np.array([]),
            "mean": 0.0,
            "std": 0.0,
            "zscore_behavior_only": zscore_behavior_only,
        }

    coef_np = np.asarray(coefficients)

    if zscore_behavior_only and n_behavior_features >= 2:
        subset = coef_np[:n_behavior_features]
    else:
        subset = coef_np

    coef_mean = float(np.mean(subset)) if subset.size > 0 else 0.0
    coef_std = float(np.std(subset)) if subset.size > 0 else 0.0

    if coef_std > 0:
        z_scored = (coef_np - coef_mean) / coef_std
    else:
        z_scored = np.zeros_like(coef_np)

    return {
        "z_scored": z_scored,
        "mean": coef_mean,
        "std": coef_std,
        "zscore_behavior_only": zscore_behavior_only,
    }


__all__ = [
    "compute_delta_d2_per_behavior",
    "compute_delta_d2_per_lag",
    "compute_per_behavior_d2",
    "build_feature_to_behavior_map",
    "build_feature_to_lag_map",
    "zscore_coefficients",
]
