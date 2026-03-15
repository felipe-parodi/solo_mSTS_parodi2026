"""Poisson GLM mathematical functions for neural encoding.

This module provides core mathematical functions for Poisson Generalized
Linear Models used in neural encoding analysis.

Core Metrics:
    - Log-likelihood: ℓ(μ; y) = Σ_t [y_t · log(μ_t) - μ_t]
    - Deviance: D = 2 · Σ_t [y_t · log(y_t / μ_t) - (y_t - μ_t)]
    - Deviance Explained (D²): D² = 1 - D_model / D_null

These metrics are analogous to R² for Gaussian models but appropriate
for count data following a Poisson distribution.
"""

import numpy as np


def poisson_loglik(y: np.ndarray, mu: np.ndarray, ignore_const: bool = True) -> float:
    """Compute Poisson log-likelihood up to constant term.

    Mathematical Definition:
        Full Poisson log-likelihood:
            ℓ(μ; y) = Σ_t [y_t · log(μ_t) - μ_t - log(y_t!)]

        Ignoring constant (default, since y! is fixed w.r.t. μ):
            ℓ(μ; y) ≈ Σ_t [y_t · log(μ_t) - μ_t]

        With Stirling approximation for log(y!):
            log(y!) ≈ y · log(y) - y  (for y > 0)

    Note:
        μ is clipped to [1e-12, ∞) to avoid log(0).

    Args:
        y: Observed spike counts, shape (n_samples,).
        mu: Predicted rates, shape (n_samples,).
        ignore_const: If True, omit the -log(y!) term.

    Returns:
        Log-likelihood value (higher is better).
    """
    y_np = np.asarray(y).astype(float)
    mu_np = np.asarray(mu).astype(float)
    mu_np = np.clip(mu_np, 1e-12, None)
    if ignore_const:
        return float(np.sum(y_np * np.log(mu_np) - mu_np))
    else:
        # Includes log(y!) using Stirling approximation for speed
        stirling = y_np * np.log(np.clip(y_np, 1e-12, None)) - y_np
        return float(np.sum(y_np * np.log(mu_np) - mu_np - stirling))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute Poisson deviance (lower is better).

    Mathematical Definition:
        Poisson deviance:
            D = 2 · Σ_t [y_t · log(y_t / μ_t) - (y_t - μ_t)]

        For y_t = 0:
            term_t = -(0 - μ_t) = μ_t

        For y_t > 0:
            term_t = y_t · log(y_t / μ_t) - (y_t - μ_t)

        Deviance Explained (D²):
            D² = 1 - D_model / D_null

        Where D_null is deviance with μ = mean(y) for all t.
        D² ∈ (-∞, 1], with 1 being perfect prediction.

    Args:
        y: Observed spike counts, shape (n_samples,).
        mu: Predicted rates, shape (n_samples,).

    Returns:
        Deviance value (lower is better, 0 = perfect fit).
    """
    y_np = np.asarray(y).astype(float)
    mu_np = np.asarray(mu).astype(float)
    mu_np = np.clip(mu_np, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_np > 0, y_np / mu_np, 1.0)
        term = np.where(
            y_np > 0, y_np * np.log(ratio) - (y_np - mu_np), -(y_np - mu_np)
        )
    dev = 2.0 * np.sum(term)
    return float(dev)


def deviance_explained(y: np.ndarray, mu: np.ndarray, mu_null: np.ndarray = None) -> float:
    """Compute deviance explained (pseudo-R² for Poisson models).

    Mathematical Definition:
        D² = 1 - D_model / D_null

        Where:
            D_model = deviance with predicted rates μ
            D_null = deviance with null model (constant rate = mean(y))

        D² ∈ (-∞, 1]:
            - D² = 1: Perfect prediction
            - D² = 0: No better than null model
            - D² < 0: Worse than null model

    Args:
        y: Observed spike counts, shape (n_samples,).
        mu: Predicted rates from model, shape (n_samples,).
        mu_null: Null model predictions (default: constant mean rate).

    Returns:
        Deviance explained value.
    """
    y_np = np.asarray(y).astype(float)
    mu_np = np.asarray(mu).astype(float)

    if mu_null is None:
        mean_rate = np.clip(np.mean(y_np), 1e-12, None)
        mu_null = np.full_like(y_np, mean_rate)
    else:
        mu_null = np.asarray(mu_null).astype(float)

    dev_model = poisson_deviance(y_np, mu_np)
    dev_null = poisson_deviance(y_np, mu_null)

    if dev_null < 1e-12:
        return np.nan

    return float(1.0 - dev_model / dev_null)


def overdispersion_test(y: np.ndarray, mu: np.ndarray, n_params: int) -> dict:
    """Test for overdispersion in Poisson model.

    Computes the Pearson chi-square statistic and dispersion parameter.
    Overdispersion (phi > 1) suggests variance exceeds mean, violating
    Poisson assumption. Consider Negative Binomial if phi > 1.5.

    Mathematical Definition:
        Pearson χ² = Σ_t [(y_t - μ_t)² / μ_t]
        φ = χ² / df
        df = n_samples - n_params

    Args:
        y: Observed spike counts.
        mu: Predicted rates.
        n_params: Number of model parameters (coefficients + intercept).

    Returns:
        Dictionary with 'phi' (dispersion parameter), 'overdispersed' flag,
        and 'pearson_chi2' statistic.
    """
    y_np = np.asarray(y).astype(float)
    mu_np = np.asarray(mu).astype(float)
    mu_np = np.clip(mu_np, 1e-12, None)

    df = max(int(y_np.shape[0] - n_params), 1)

    if len(mu_np) > 0 and np.all(np.isfinite(mu_np)):
        pearson = np.sum(((y_np - mu_np) ** 2) / mu_np)
        phi = float(pearson / df)
        return {
            "phi": phi,
            "overdispersed": bool(phi > 1.5),
            "pearson_chi2": float(pearson),
            "df": df,
        }
    else:
        return {
            "phi": np.nan,
            "overdispersed": False,
            "pearson_chi2": np.nan,
            "df": df,
        }


def calculate_p_value_from_shuffles(
    actual_score: float, shuffled_scores: np.ndarray
) -> float:
    """Calculate permutation test p-value from shuffle distribution.

    Computes one-sided p-value as the proportion of shuffled scores that
    equal or exceed the actual score, with continuity correction.

    Mathematical Definition:
        p = (1 + Σ_i 𝟙[shuffle_i ≥ actual]) / (1 + n_shuffles)

        The +1 in numerator and denominator provides:
            - Continuity correction
            - Conservative estimate
            - Avoids p = 0

    Args:
        actual_score: Observed test statistic (e.g., D² score).
        shuffled_scores: Array of scores from null distribution.

    Returns:
        One-sided p-value in (0, 1].
    """
    shuffled = np.asarray(shuffled_scores)
    valid = shuffled[np.isfinite(shuffled)]
    if valid.size == 0:
        return np.nan
    # Count shuffles >= actual (one-sided, greater)
    n_ge = np.sum(valid >= actual_score)
    p_val = (1 + n_ge) / (1 + valid.size)
    return float(p_val)


# Legacy aliases for backward compatibility
_poisson_loglik = poisson_loglik
_poisson_deviance = poisson_deviance


__all__ = [
    "poisson_loglik",
    "poisson_deviance",
    "deviance_explained",
    "overdispersion_test",
    "calculate_p_value_from_shuffles",
    # Legacy aliases
    "_poisson_loglik",
    "_poisson_deviance",
]
