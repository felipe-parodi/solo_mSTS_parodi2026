"""Non-redundancy analysis for interaction summary variables.

Tests whether ledger, alignment, and surprise are related but non-redundant
(each captures unique variance about the interaction).

Analysis:
    1. Correlation matrix: raw correlations among the three variables
    2. Partial correlation: correlations after residualizing on controls

Goal: Show they're related (correlated) but not identical (retain unique variance).

Usage:
    from analysis.interaction_summaries.non_redundancy import (
        compute_non_redundancy_matrix,
    )

    result = compute_non_redundancy_matrix(
        ledger=grooming_imbalance,
        alignment=alignment_series,
        surprise=surprisal_series,
        controls=control_covariates,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NonRedundancyResult:
    """Result of non-redundancy analysis.

    Attributes:
        correlation_matrix: Raw correlation matrix (3x3)
        partial_correlation_matrix: Partial correlations after controlling
        variable_names: Names of the three variables
        n_samples: Number of valid samples used
        p_values_raw: P-values for raw correlations
        p_values_partial: P-values for partial correlations
    """

    correlation_matrix: np.ndarray
    partial_correlation_matrix: np.ndarray
    variable_names: List[str]
    n_samples: int
    p_values_raw: Optional[np.ndarray]
    p_values_partial: Optional[np.ndarray]

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of pairwise relationships."""
        names = self.variable_names
        summary = {}

        for i in range(3):
            for j in range(i + 1, 3):
                pair_name = f"{names[i]}_vs_{names[j]}"
                summary[pair_name] = {
                    "raw_correlation": float(self.correlation_matrix[i, j]),
                    "partial_correlation": float(self.partial_correlation_matrix[i, j]),
                }
                if self.p_values_raw is not None:
                    summary[pair_name]["p_raw"] = float(self.p_values_raw[i, j])
                if self.p_values_partial is not None:
                    summary[pair_name]["p_partial"] = float(self.p_values_partial[i, j])

        return summary


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation and approximate p-value.

    Returns:
        Tuple of (correlation, p_value)
    """
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 3:
        return np.nan, np.nan

    x_valid = x[valid]
    y_valid = y[valid]
    n = len(x_valid)

    # Correlation
    x_z = (x_valid - np.mean(x_valid)) / (np.std(x_valid) + 1e-12)
    y_z = (y_valid - np.mean(y_valid)) / (np.std(y_valid) + 1e-12)
    r = np.mean(x_z * y_z)

    # Approximate p-value using t-distribution
    if abs(r) >= 1.0 - 1e-12:
        p = 0.0
    else:
        t = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-12))
        # Two-tailed p-value approximation
        p = 2 * (1 - _approx_t_cdf(abs(t), n - 2))

    return r, p


def _approx_t_cdf(t: float, df: int) -> float:
    """Approximate t-distribution CDF using normal approximation for large df."""
    if df <= 0:
        return 0.5

    # For df > 30, use normal approximation
    if df > 30:
        from math import erf, sqrt
        return 0.5 * (1 + erf(t / sqrt(2)))

    # Simple approximation for smaller df
    # This is rough but avoids scipy dependency
    x = df / (df + t ** 2)
    return 1 - 0.5 * (x ** (df / 2))


def _residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residualize X on controls C using OLS.

    Args:
        X: Data matrix, shape (n_samples, n_vars)
        C: Control matrix, shape (n_samples, n_controls)

    Returns:
        Residuals, shape (n_samples, n_vars)
    """
    # Filter valid samples (no NaN in controls)
    valid_rows = ~np.any(np.isnan(C), axis=1)

    residuals = np.full_like(X, np.nan)

    C_valid = C[valid_rows]
    X_valid = X[valid_rows]

    # Add intercept
    C_aug = np.hstack([np.ones((C_valid.shape[0], 1)), C_valid])

    # OLS for each variable in X
    for j in range(X.shape[1]):
        y = X_valid[:, j]
        y_valid_mask = ~np.isnan(y)

        if y_valid_mask.sum() < C_aug.shape[1] + 1:
            continue

        C_j = C_aug[y_valid_mask]
        y_j = y[y_valid_mask]

        # Solve normal equations
        CTC = C_j.T @ C_j
        beta = np.linalg.solve(CTC + 1e-6 * np.eye(CTC.shape[0]), C_j.T @ y_j)
        y_pred = C_aug @ beta

        # Store residuals in original positions
        full_resid = np.full(len(X), np.nan)
        full_resid[valid_rows] = X_valid[:, j] - y_pred
        residuals[:, j] = full_resid

    return residuals


def _compute_correlation_matrix(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute correlation matrix and p-values.

    Args:
        data: Data matrix, shape (n_samples, n_vars)

    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    n_vars = data.shape[1]
    corr_matrix = np.eye(n_vars)
    p_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            r, p = _pearson_correlation(data[:, i], data[:, j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    return corr_matrix, p_matrix


def compute_non_redundancy_matrix(
    ledger: np.ndarray,
    alignment: np.ndarray,
    surprise: np.ndarray,
    controls: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
) -> NonRedundancyResult:
    """Compute correlation and partial correlation matrix for interaction summaries.

    Args:
        ledger: Grooming imbalance time series
        alignment: Inter-brain alignment time series
        surprise: Surprisal time series
        controls: Control covariate matrix (optional, for partial correlations)
        variable_names: Names for the three variables

    Returns:
        NonRedundancyResult with correlation matrices
    """
    if variable_names is None:
        variable_names = ["ledger", "alignment", "surprise"]

    # Stack variables
    n = len(ledger)
    data = np.column_stack([ledger, alignment, surprise])

    # Raw correlation matrix
    raw_corr, raw_p = _compute_correlation_matrix(data)

    # Partial correlations (after residualizing on controls)
    if controls is not None:
        data_resid = _residualize(data, controls)
        partial_corr, partial_p = _compute_correlation_matrix(data_resid)
    else:
        partial_corr = raw_corr.copy()
        partial_p = raw_p.copy()

    # Count valid samples
    valid_mask = ~np.any(np.isnan(data), axis=1)
    n_samples = int(valid_mask.sum())

    return NonRedundancyResult(
        correlation_matrix=raw_corr,
        partial_correlation_matrix=partial_corr,
        variable_names=variable_names,
        n_samples=n_samples,
        p_values_raw=raw_p,
        p_values_partial=partial_p,
    )


def compute_unique_variance_decomposition(
    ledger: np.ndarray,
    alignment: np.ndarray,
    surprise: np.ndarray,
    neural_rates: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Decompose neural variance explained by each summary variable.

    For each unit, computes:
    - R² from ledger alone
    - R² from alignment alone
    - R² from surprise alone
    - R² from all three
    - Unique variance for each (variance explained that no other explains)

    Args:
        ledger: Grooming imbalance time series
        alignment: Inter-brain alignment time series
        surprise: Surprisal time series
        neural_rates: Neural rates, shape (n_samples, n_units)
        controls: Control covariates (regressed out first)

    Returns:
        Dict with variance decomposition statistics
    """
    n_samples, n_units = neural_rates.shape

    # Stack predictors
    X_all = np.column_stack([ledger, alignment, surprise])

    # Residualize neural data on controls if provided
    if controls is not None:
        Y = _residualize(neural_rates, controls)
    else:
        Y = neural_rates.copy()

    # Filter valid samples
    valid = ~np.any(np.isnan(X_all), axis=1)

    results = {
        "r2_ledger_only": [],
        "r2_alignment_only": [],
        "r2_surprise_only": [],
        "r2_all": [],
        "unique_ledger": [],
        "unique_alignment": [],
        "unique_surprise": [],
    }

    for ui in range(n_units):
        y = Y[:, ui]
        y_valid = ~np.isnan(y)
        mask = valid & y_valid

        if mask.sum() < 5:
            for k in results:
                results[k].append(np.nan)
            continue

        y_m = y[mask]
        X_m = X_all[mask]

        # Compute R² for different predictor sets
        ss_tot = np.sum((y_m - np.mean(y_m)) ** 2) + 1e-12

        def _r2(X_pred: np.ndarray) -> float:
            if X_pred.shape[1] == 0:
                return 0.0
            X_aug = np.hstack([np.ones((len(y_m), 1)), X_pred])
            beta = np.linalg.lstsq(X_aug, y_m, rcond=None)[0]
            y_hat = X_aug @ beta
            ss_res = np.sum((y_m - y_hat) ** 2)
            return max(0, 1 - ss_res / ss_tot)

        r2_ledger = _r2(X_m[:, 0:1])
        r2_align = _r2(X_m[:, 1:2])
        r2_surp = _r2(X_m[:, 2:3])
        r2_all = _r2(X_m)

        # Unique variance
        r2_no_ledger = _r2(X_m[:, 1:3])
        r2_no_align = _r2(np.column_stack([X_m[:, 0], X_m[:, 2]]))
        r2_no_surp = _r2(X_m[:, 0:2])

        results["r2_ledger_only"].append(r2_ledger)
        results["r2_alignment_only"].append(r2_align)
        results["r2_surprise_only"].append(r2_surp)
        results["r2_all"].append(r2_all)
        results["unique_ledger"].append(r2_all - r2_no_ledger)
        results["unique_alignment"].append(r2_all - r2_no_align)
        results["unique_surprise"].append(r2_all - r2_no_surp)

    # Summarize across units
    summary = {}
    for k, v in results.items():
        v_arr = np.array(v)
        summary[k] = {
            "mean": float(np.nanmean(v_arr)),
            "std": float(np.nanstd(v_arr)),
            "median": float(np.nanmedian(v_arr)),
        }

    return summary
