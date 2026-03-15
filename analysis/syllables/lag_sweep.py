"""Lead/lag analysis for predictive vs reactive coding.

This module provides analyses for Panel D of Figure 3:
- Shifted-neural lag sweep: decode syllable[t] from X[t+tau]
- Lead time extraction: find tau* that maximizes accuracy

The key question: Does the neural code lead transitions (predictive)
or lag them (reactive)? If tau* < 0, neural activity leads behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LagSweepResult:
    """Results from lag sweep analysis.

    Attributes:
        lags_bins: Lag values in bins (negative = neural leads).
        lags_sec: Lag values in seconds.
        accuracy_curve: Accuracy at each lag.
        accuracy_ci_low: Lower CI (if computed).
        accuracy_ci_high: Upper CI (if computed).
        optimal_lag_bins: Lag with highest accuracy.
        optimal_lag_sec: Optimal lag in seconds.
        optimal_accuracy: Accuracy at optimal lag.
        lead_time_bins: If optimal_lag < 0, this is the lead time.
        lead_time_sec: Lead time in seconds.
        n_samples_per_lag: Valid samples at each lag.
    """

    lags_bins: np.ndarray
    lags_sec: np.ndarray
    accuracy_curve: np.ndarray
    accuracy_ci_low: Optional[np.ndarray] = None
    accuracy_ci_high: Optional[np.ndarray] = None
    optimal_lag_bins: int = 0
    optimal_lag_sec: float = 0.0
    optimal_accuracy: float = 0.0
    lead_time_bins: int = 0
    lead_time_sec: float = 0.0
    n_samples_per_lag: Optional[np.ndarray] = None


def lag_sweep_decoding(
    X: np.ndarray,
    y: np.ndarray,
    lags_bins: Optional[np.ndarray] = None,
    bin_size_sec: float = 0.1,
    n_folds: int = 5,
    embargo_bins: int = 20,
    random_state: int = 42,
) -> LagSweepResult:
    """Decode syllable[t] from neural features shifted by lag.

    For each lag tau:
        X_shifted[t] = X[t + tau]
        Decode y[t] from X_shifted[t]

    Positive lag means neural features are from the future (neural lags behavior).
    Negative lag means neural features are from the past (neural leads behavior).

    Args:
        X: Neural features (n_bins, n_features).
        y: Syllable labels (n_bins,).
        lags_bins: Array of lags to test in bins. Default: [-10, ..., 10].
        bin_size_sec: Bin size in seconds for converting to time.
        n_folds: Number of CV folds.
        embargo_bins: Embargo around fold boundaries.
        random_state: Random seed.

    Returns:
        LagSweepResult with accuracy curve and optimal lag.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    n_bins = X.shape[0]

    if lags_bins is None:
        lags_bins = np.arange(-10, 11)  # -10 to +10 bins

    lags_bins = np.asarray(lags_bins)
    lags_sec = lags_bins * bin_size_sec

    accuracy_curve = np.full(len(lags_bins), np.nan)
    n_samples_per_lag = np.zeros(len(lags_bins), dtype=int)
    # Per-fold accuracy for CI computation
    fold_accuracies: List[List[float]] = [[] for _ in range(len(lags_bins))]

    # Create temporal folds once
    from analysis.decoding import make_temporal_folds

    folds = make_temporal_folds(n_bins, n_folds, embargo_bins)

    for i, lag in enumerate(lags_bins):
        # Shift neural features
        if lag >= 0:
            # Neural from future: X_shifted[t] = X[t + lag]
            # Valid range: t in [0, n_bins - lag)
            valid_start = 0
            valid_end = n_bins - lag
            X_shifted = X[lag:, :]
            y_aligned = y[:valid_end]
        else:
            # Neural from past: X_shifted[t] = X[t + lag] = X[t - |lag|]
            # Valid range: t in [|lag|, n_bins)
            valid_start = -lag
            valid_end = n_bins
            X_shifted = X[:n_bins + lag, :]
            y_aligned = y[valid_start:]

        n_valid = len(y_aligned)
        if n_valid < 50:  # Need minimum samples
            continue

        n_samples_per_lag[i] = n_valid

        # Cross-validated decoding
        predictions = np.full(n_valid, -1, dtype=int)

        # Map fold indices to valid range
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Adjust indices for shifted data
            if lag >= 0:
                train_valid = train_idx[train_idx < valid_end]
                test_valid = test_idx[test_idx < valid_end]
            else:
                train_valid = train_idx[(train_idx >= valid_start) & (train_idx < valid_end)] - valid_start
                test_valid = test_idx[(test_idx >= valid_start) & (test_idx < valid_end)] - valid_start

            if len(train_valid) == 0 or len(test_valid) == 0:
                continue

            # Scale and fit
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_shifted[train_valid])
            X_test = scaler.transform(X_shifted[test_valid])

            y_train = y_aligned[train_valid]
            y_test = y_aligned[test_valid]

            # Skip if too few classes
            if len(np.unique(y_train)) < 2:
                continue

            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=random_state,
                solver="lbfgs",
            )
            clf.fit(X_train, y_train)

            predictions[test_valid] = clf.predict(X_test)

            # Per-fold accuracy
            fold_preds = clf.predict(X_test)
            fold_acc = balanced_accuracy_score(y_test, fold_preds)
            fold_accuracies[i].append(float(fold_acc))

        # Compute accuracy on test predictions
        test_mask = predictions != -1
        if np.sum(test_mask) > 0:
            accuracy_curve[i] = balanced_accuracy_score(
                y_aligned[test_mask], predictions[test_mask]
            )

    # Compute CI from per-fold accuracies (mean +/- 1.96*SEM)
    accuracy_ci_low = np.full(len(lags_bins), np.nan)
    accuracy_ci_high = np.full(len(lags_bins), np.nan)
    for i in range(len(lags_bins)):
        accs = fold_accuracies[i]
        if len(accs) >= 2:
            mean_acc = np.mean(accs)
            sem = np.std(accs) / np.sqrt(len(accs))
            accuracy_ci_low[i] = mean_acc - 1.96 * sem
            accuracy_ci_high[i] = mean_acc + 1.96 * sem

    # Find optimal lag
    valid_mask = ~np.isnan(accuracy_curve)
    if np.any(valid_mask):
        best_idx = np.nanargmax(accuracy_curve)
        optimal_lag_bins = int(lags_bins[best_idx])
        optimal_lag_sec = float(lags_sec[best_idx])
        optimal_accuracy = float(accuracy_curve[best_idx])
    else:
        optimal_lag_bins = 0
        optimal_lag_sec = 0.0
        optimal_accuracy = np.nan

    # Lead time: positive if neural leads (optimal_lag < 0)
    lead_time_bins = -optimal_lag_bins if optimal_lag_bins < 0 else 0
    lead_time_sec = -optimal_lag_sec if optimal_lag_sec < 0 else 0.0

    return LagSweepResult(
        lags_bins=lags_bins,
        lags_sec=lags_sec,
        accuracy_curve=accuracy_curve,
        accuracy_ci_low=accuracy_ci_low,
        accuracy_ci_high=accuracy_ci_high,
        optimal_lag_bins=optimal_lag_bins,
        optimal_lag_sec=optimal_lag_sec,
        optimal_accuracy=optimal_accuracy,
        lead_time_bins=lead_time_bins,
        lead_time_sec=lead_time_sec,
        n_samples_per_lag=n_samples_per_lag,
    )


def lag_sweep_by_target(
    X: np.ndarray,
    y_self: np.ndarray,
    y_partner: np.ndarray,
    lags_bins: Optional[np.ndarray] = None,
    bin_size_sec: float = 0.1,
    **kwargs,
) -> Dict[str, LagSweepResult]:
    """Run lag sweep for both self and partner syllables.

    Args:
        X: Neural features (n_bins, n_features).
        y_self: Self syllable labels.
        y_partner: Partner syllable labels.
        lags_bins: Lags to test.
        bin_size_sec: Bin size.
        **kwargs: Additional args for lag_sweep_decoding.

    Returns:
        Dict with 'self' and 'partner' LagSweepResult.
    """
    results = {}

    results["self"] = lag_sweep_decoding(
        X, y_self, lags_bins, bin_size_sec, **kwargs
    )
    results["partner"] = lag_sweep_decoding(
        X, y_partner, lags_bins, bin_size_sec, **kwargs
    )

    return results


def lag_sweep_by_bank(
    X: np.ndarray,
    y: np.ndarray,
    unit_bank_labels: np.ndarray,
    lags_bins: Optional[np.ndarray] = None,
    bin_size_sec: float = 0.1,
    **kwargs,
) -> Dict[str, LagSweepResult]:
    """Run lag sweep separately for each neural bank.

    Args:
        X: Neural features (n_bins, n_units).
        y: Syllable labels.
        unit_bank_labels: Bank label per unit ('ventral' or 'dorsal').
        lags_bins: Lags to test.
        bin_size_sec: Bin size.
        **kwargs: Additional args.

    Returns:
        Dict with 'ventral', 'dorsal', 'pooled' results.
    """
    results = {}

    for bank in ["ventral", "dorsal"]:
        bank_mask = unit_bank_labels == bank
        if np.sum(bank_mask) == 0:
            continue
        X_bank = X[:, bank_mask]
        results[bank] = lag_sweep_decoding(
            X_bank, y, lags_bins, bin_size_sec, **kwargs
        )

    # Pooled
    results["pooled"] = lag_sweep_decoding(X, y, lags_bins, bin_size_sec, **kwargs)

    return results


def compute_lead_time_comparison(
    result_self: LagSweepResult,
    result_partner: LagSweepResult,
) -> Dict[str, float]:
    """Compare lead times between self and partner decoding.

    Args:
        result_self: Lag sweep result for self syllables.
        result_partner: Lag sweep result for partner syllables.

    Returns:
        Dict with comparison metrics.
    """
    return {
        "lead_time_self_sec": result_self.lead_time_sec,
        "lead_time_partner_sec": result_partner.lead_time_sec,
        "lead_time_diff_sec": result_partner.lead_time_sec - result_self.lead_time_sec,
        "optimal_accuracy_self": result_self.optimal_accuracy,
        "optimal_accuracy_partner": result_partner.optimal_accuracy,
    }


def compute_pre_onset_accuracy(
    result: LagSweepResult,
    pre_range_sec: Tuple[float, float] = (-0.3, 0.0),
) -> float:
    """Extract mean accuracy in a pre-onset window.

    Useful for comparing predictive coding across conditions.

    Args:
        result: LagSweepResult from lag sweep.
        pre_range_sec: Time range before onset (negative values).

    Returns:
        Mean accuracy in the specified range.
    """
    mask = (result.lags_sec >= pre_range_sec[0]) & (result.lags_sec < pre_range_sec[1])

    if not np.any(mask):
        return np.nan

    pre_acc = result.accuracy_curve[mask]
    return float(np.nanmean(pre_acc))
