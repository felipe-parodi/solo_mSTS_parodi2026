"""Generalized Additive Models (GAM) for neural encoding.

This module provides GAM fitting functions that combine linear terms for
categorical behavioral predictors with smooth spline terms for continuous
control variables. Supports both Poisson (count data) and Gaussian
(continuous data) response distributions.

Key Features:
    - B-spline smoothers for continuous variables
    - Linear terms for categorical behavioral predictors
    - Cross-validated model fitting
    - Support for Poisson (spike counts) and Gaussian (continuous) families
    - Automatic degree of freedom selection

Mathematical Framework:
    Poisson GAM:
        y ~ Poisson(λ)
        log(λ) = Σ β_j X_j + Σ f_k(Z_k)

    Gaussian GAM:
        y ~ Normal(μ, σ²)
        μ = Σ β_j X_j + Σ f_k(Z_k)

    Where:
        - X_j are linear (behavioral) predictors
        - Z_k are smooth (continuous) predictors
        - f_k are B-spline basis functions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cv import temporal_kfold_indices, temporal_kfold_indices_with_boundaries, KFold_cpu
from .poisson import poisson_deviance

logger = logging.getLogger(__name__)

# Check for statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.gam.api import GLMGam, BSplines
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not available, GAM fitting will be limited")


def fit_poisson_gam_cv(
    X_linear: np.ndarray,
    Z_smooth: Optional[np.ndarray],
    y: np.ndarray,
    n_splits_cv: int,
    cv_strategy: str = "temporal_kfold",
    behavior_change_indices: Optional[np.ndarray] = None,
    gam_df: int = 6,
    gam_degree: int = 3,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """Fit Poisson GAM with linear behavioral terms and smooth control terms.

    Combines linear terms for categorical behavioral predictors with B-spline
    smoothers for continuous control variables (velocity, audio, etc.).

    Mathematical Model:
        y ~ Poisson(λ)
        log(λ) = X_linear · β + Σ_k f_k(Z_k)

        Where f_k are B-spline basis functions with specified degrees of freedom.

    Args:
        X_linear: Linear (behavioral) predictors, shape (n_samples, n_behaviors).
        Z_smooth: Continuous predictors for smoothing, shape (n_samples, n_controls).
            If None or empty, falls back to standard GLM.
        y: Target spike counts, shape (n_samples,).
        n_splits_cv: Number of cross-validation folds.
        cv_strategy: CV strategy - 'temporal_kfold' or 'temporal_kfold_behavior_boundaries'.
        behavior_change_indices: Indices where behavior changes (for boundary-aware CV).
        gam_df: Degrees of freedom for each smooth term (default 6).
        gam_degree: B-spline degree (default 3 = cubic).

    Returns:
        Tuple containing:
            - mean_de: Mean deviance explained across CV folds
            - coef_linear: Coefficients for linear terms
            - intercept: Model intercept
            - y_pred: Predictions on full data

    Example:
        >>> # Fit GAM with behavioral and control features
        >>> de, coef, intercept, y_pred = fit_poisson_gam_cv(
        >>>     X_behavior, X_controls, spike_counts, n_splits_cv=5
        >>> )
        >>> print(f"Deviance explained: {de:.3f}")
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for GAM fitting. Install with: pip install statsmodels")

    X_np = np.asarray(X_linear)
    y_np = np.asarray(y)
    Z_np = None if Z_smooth is None else np.asarray(Z_smooth)

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X_linear and y must have same number of samples")
    if X_np.shape[0] < n_splits_cv:
        return np.nan, np.zeros(X_np.shape[1]), np.nan, np.array([])

    # Build CV splits
    if cv_strategy == "temporal_kfold_behavior_boundaries":
        splits = temporal_kfold_indices_with_boundaries(X_np.shape[0], n_splits_cv, behavior_change_indices)
    elif cv_strategy == "temporal_kfold":
        splits = temporal_kfold_indices(X_np.shape[0], n_splits_cv)
    else:
        kf = KFold_cpu(n_splits=n_splits_cv, shuffle=True, random_state=42)
        splits = list(kf.split(X_np))

    # Prepare smoother for full fit
    if Z_np is not None and Z_np.size > 0:
        df_list = [int(gam_df)] * Z_np.shape[1]
        bs_full = BSplines(Z_np, df=df_list, degree=[gam_degree] * Z_np.shape[1])
    else:
        bs_full = None

    scores: List[float] = []
    for train_idx, test_idx in splits:
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        if Z_np is not None and Z_np.size > 0:
            Z_tr, Z_te = Z_np[train_idx], Z_np[test_idx]
            bs = BSplines(Z_tr, df=[int(gam_df)] * Z_tr.shape[1], degree=[gam_degree] * Z_tr.shape[1])
            model = GLMGam(y_tr, X=X_tr, smoother=bs, family=sm.families.Poisson())
            res = model.fit()
            mu_te = res.predict(exog=X_te, exog_smooth=Z_te)
        else:
            model = sm.GLM(y_tr, X_tr, family=sm.families.Poisson())
            res = model.fit()
            mu_te = res.predict(X_te)

        # Deviance explained on test
        dev_model = poisson_deviance(y_te, mu_te)
        mu_null = np.full_like(y_te, np.clip(np.mean(y_tr), 1e-12, None), dtype=float)
        dev_null = poisson_deviance(y_te, mu_null)
        de = 1.0 - (dev_model / dev_null) if dev_null > 1e-12 else np.nan
        scores.append(de)

    mean_de = float(np.nanmean(np.array(scores))) if scores else np.nan

    # Final fit on all data
    if Z_np is not None and Z_np.size > 0:
        model_final = GLMGam(y_np, X=X_np, smoother=bs_full, family=sm.families.Poisson())
        res_final = model_final.fit()
        mu_full = res_final.predict(exog=X_np, exog_smooth=Z_np)
        coef_linear = np.array(res_final.params[:X_np.shape[1]])
        intercept = float(res_final.params[0]) if coef_linear.size > 0 else 0.0
    else:
        model_final = sm.GLM(y_np, X_np, family=sm.families.Poisson())
        res_final = model_final.fit()
        mu_full = res_final.predict(X_np)
        coef_linear = np.array(res_final.params)
        intercept = float(res_final.params[0]) if coef_linear.size > 0 else 0.0

    return mean_de, coef_linear, intercept, np.array(mu_full)


def fit_gaussian_gam_cv(
    X_linear: np.ndarray,
    Z_smooth: Optional[np.ndarray],
    y: np.ndarray,
    n_splits_cv: int,
    cv_strategy: str = "temporal_kfold",
    behavior_change_indices: Optional[np.ndarray] = None,
    gam_df: int = 6,
    gam_degree: int = 3,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """Fit Gaussian GAM for continuous response variables.

    Similar to fit_poisson_gam_cv but uses Gaussian (Normal) distribution
    for continuous response variables like velocity, position, or LFP.

    Mathematical Model:
        y ~ Normal(μ, σ²)
        μ = X_linear · β + Σ_k f_k(Z_k)

    Args:
        X_linear: Linear (behavioral) predictors, shape (n_samples, n_behaviors).
        Z_smooth: Continuous predictors for smoothing, shape (n_samples, n_controls).
        y: Target continuous variable, shape (n_samples,).
        n_splits_cv: Number of cross-validation folds.
        cv_strategy: CV strategy.
        behavior_change_indices: Indices where behavior changes.
        gam_df: Degrees of freedom for each smooth term.
        gam_degree: B-spline degree.

    Returns:
        Tuple containing:
            - r2: Mean R² (variance explained) across CV folds
            - coef_linear: Coefficients for linear terms
            - intercept: Model intercept
            - y_pred: Predictions on full data

    Example:
        >>> # Fit GAM to predict velocity from neural activity
        >>> r2, coef, intercept, y_pred = fit_gaussian_gam_cv(
        >>>     neural_features, control_features, velocity, n_splits_cv=5
        >>> )
        >>> print(f"Variance explained: {r2:.3f}")
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for GAM fitting")

    X_np = np.asarray(X_linear)
    y_np = np.asarray(y).astype(float)
    Z_np = None if Z_smooth is None else np.asarray(Z_smooth)

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X_linear and y must have same number of samples")
    if X_np.shape[0] < n_splits_cv:
        return np.nan, np.zeros(X_np.shape[1]), np.nan, np.array([])

    # Build CV splits
    if cv_strategy == "temporal_kfold_behavior_boundaries":
        splits = temporal_kfold_indices_with_boundaries(X_np.shape[0], n_splits_cv, behavior_change_indices)
    elif cv_strategy == "temporal_kfold":
        splits = temporal_kfold_indices(X_np.shape[0], n_splits_cv)
    else:
        kf = KFold_cpu(n_splits=n_splits_cv, shuffle=True, random_state=42)
        splits = list(kf.split(X_np))

    # Prepare smoother
    if Z_np is not None and Z_np.size > 0:
        df_list = [int(gam_df)] * Z_np.shape[1]
        bs_full = BSplines(Z_np, df=df_list, degree=[gam_degree] * Z_np.shape[1])
    else:
        bs_full = None

    scores: List[float] = []
    for train_idx, test_idx in splits:
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        if Z_np is not None and Z_np.size > 0:
            Z_tr, Z_te = Z_np[train_idx], Z_np[test_idx]
            bs = BSplines(Z_tr, df=[int(gam_df)] * Z_tr.shape[1], degree=[gam_degree] * Z_tr.shape[1])
            model = GLMGam(y_tr, X=X_tr, smoother=bs, family=sm.families.Gaussian())
            res = model.fit()
            mu_te = res.predict(exog=X_te, exog_smooth=Z_te)
        else:
            model = sm.GLM(y_tr, X_tr, family=sm.families.Gaussian())
            res = model.fit()
            mu_te = res.predict(X_te)

        # R² on test
        ss_res = np.sum((y_te - mu_te) ** 2)
        ss_tot = np.sum((y_te - np.mean(y_tr)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
        scores.append(r2)

    mean_r2 = float(np.nanmean(np.array(scores))) if scores else np.nan

    # Final fit on all data
    if Z_np is not None and Z_np.size > 0:
        model_final = GLMGam(y_np, X=X_np, smoother=bs_full, family=sm.families.Gaussian())
        res_final = model_final.fit()
        mu_full = res_final.predict(exog=X_np, exog_smooth=Z_np)
        coef_linear = np.array(res_final.params[:X_np.shape[1]])
        intercept = float(res_final.params[0]) if coef_linear.size > 0 else 0.0
    else:
        model_final = sm.GLM(y_np, X_np, family=sm.families.Gaussian())
        res_final = model_final.fit()
        mu_full = res_final.predict(X_np)
        coef_linear = np.array(res_final.params)
        intercept = float(res_final.params[0]) if coef_linear.size > 0 else 0.0

    return mean_r2, coef_linear, intercept, np.array(mu_full)


def fit_gam_with_continuous(
    X_categorical: np.ndarray,
    X_continuous: np.ndarray,
    y: np.ndarray,
    n_splits_cv: int = 5,
    family: str = "poisson",
    cv_strategy: str = "temporal_kfold",
    gam_df: int = 6,
    gam_degree: int = 3,
    smooth_continuous: bool = True,
) -> Dict[str, Any]:
    """Unified GAM fitting for mixed categorical and continuous predictors.

    This is the main entry point for GAM models that need to handle both
    categorical (behavioral state) and continuous (kinematic, audio) features.

    Args:
        X_categorical: Categorical/binary predictors (behavioral states),
            shape (n_samples, n_categorical).
        X_continuous: Continuous predictors (velocity, audio, etc.),
            shape (n_samples, n_continuous).
        y: Target variable, shape (n_samples,).
        n_splits_cv: Number of cross-validation folds.
        family: Response distribution - 'poisson' or 'gaussian'.
        cv_strategy: CV strategy.
        gam_df: Degrees of freedom for smooth terms.
        gam_degree: B-spline degree for smooth terms.
        smooth_continuous: If True, use B-splines for continuous variables.
            If False, use linear terms for all predictors.

    Returns:
        Dictionary containing:
            - 'score': Cross-validated D² (Poisson) or R² (Gaussian)
            - 'coef_categorical': Coefficients for categorical predictors
            - 'coef_continuous': Coefficients for continuous predictors (linear only)
            - 'intercept': Model intercept
            - 'y_pred': Predictions on full data
            - 'family': Distribution family used
            - 'smooth_terms': Info about smooth terms (if applicable)

    Example:
        >>> # Mixed categorical and continuous predictors
        >>> results = fit_gam_with_continuous(
        >>>     X_behavior,  # Binary behavioral indicators
        >>>     X_kinematics,  # Continuous velocity, position
        >>>     spike_counts,
        >>>     family='poisson',
        >>>     smooth_continuous=True
        >>> )
        >>> print(f"D² = {results['score']:.3f}")
    """
    X_cat = np.asarray(X_categorical)
    X_cont = np.asarray(X_continuous)
    y_arr = np.asarray(y)

    n_cat = X_cat.shape[1] if X_cat.ndim > 1 else 1
    n_cont = X_cont.shape[1] if X_cont.ndim > 1 else 1

    if smooth_continuous and HAS_STATSMODELS:
        # Use GAM with smooth terms for continuous
        if family == "poisson":
            score, coef, intercept, y_pred = fit_poisson_gam_cv(
                X_cat, X_cont, y_arr, n_splits_cv,
                cv_strategy=cv_strategy, gam_df=gam_df, gam_degree=gam_degree
            )
        else:
            score, coef, intercept, y_pred = fit_gaussian_gam_cv(
                X_cat, X_cont, y_arr, n_splits_cv,
                cv_strategy=cv_strategy, gam_df=gam_df, gam_degree=gam_degree
            )

        return {
            "score": score,
            "coef_categorical": coef[:n_cat] if len(coef) >= n_cat else coef,
            "coef_continuous": None,  # Absorbed into smooth terms
            "intercept": intercept,
            "y_pred": y_pred,
            "family": family,
            "smooth_terms": {
                "n_continuous": n_cont,
                "gam_df": gam_df,
                "gam_degree": gam_degree,
            },
        }
    else:
        # Fall back to linear model for all terms
        X_combined = np.hstack([X_cat, X_cont])
        if family == "poisson":
            score, coef, intercept, y_pred = fit_poisson_gam_cv(
                X_combined, None, y_arr, n_splits_cv, cv_strategy=cv_strategy
            )
        else:
            score, coef, intercept, y_pred = fit_gaussian_gam_cv(
                X_combined, None, y_arr, n_splits_cv, cv_strategy=cv_strategy
            )

        return {
            "score": score,
            "coef_categorical": coef[:n_cat] if len(coef) >= n_cat else coef,
            "coef_continuous": coef[n_cat:] if len(coef) > n_cat else np.array([]),
            "intercept": intercept,
            "y_pred": y_pred,
            "family": family,
            "smooth_terms": None,
        }


# Legacy alias
_fit_gam_cv = fit_poisson_gam_cv


__all__ = [
    "fit_poisson_gam_cv",
    "fit_gaussian_gam_cv",
    "fit_gam_with_continuous",
    "HAS_STATSMODELS",
    # Legacy
    "_fit_gam_cv",
]
