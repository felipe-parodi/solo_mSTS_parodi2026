"""Transition-aligned temporal dynamics analysis.

This module provides analyses for Panel C of Figure 3:
- Peri-event time histograms (PETHs) aligned to syllable onsets
- Onset-aligned decoding accuracy curves
- Peak timing and width extraction

The key question: How sharply does the neural code update at behavioral
boundaries? What is the latency and precision of boundary-locked coding?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .alignment import detect_bin_onsets, get_onset_aligned_indices


@dataclass
class OnsetAlignedResult:
    """Results from onset-aligned analysis.

    Attributes:
        tau_bins: Relative time bins (negative = before onset).
        tau_sec: Relative times in seconds.
        accuracy_curve: Accuracy at each tau.
        accuracy_ci_low: Lower confidence interval.
        accuracy_ci_high: Upper confidence interval.
        peak_tau_sec: Time of peak accuracy (tau*).
        peak_accuracy: Accuracy at peak.
        width_sec: Width of peak (FWHM or similar).
        n_onsets: Number of onsets used.
        n_samples_per_tau: Samples contributing at each tau.
    """

    tau_bins: np.ndarray
    tau_sec: np.ndarray
    accuracy_curve: np.ndarray
    accuracy_ci_low: Optional[np.ndarray] = None
    accuracy_ci_high: Optional[np.ndarray] = None
    peak_tau_sec: float = 0.0
    peak_accuracy: float = 0.0
    width_sec: float = 0.0
    n_onsets: int = 0
    n_samples_per_tau: Optional[np.ndarray] = None


def compute_onset_peth(
    X: np.ndarray,
    onset_bins: np.ndarray,
    pre_bins: int = 20,
    post_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute peri-event time histogram aligned to onsets.

    Args:
        X: Neural data (n_bins, n_units).
        onset_bins: Bin indices where onsets occur.
        pre_bins: Number of bins before onset.
        post_bins: Number of bins after onset (including onset).

    Returns:
        Tuple of:
            peth: (n_units, window_size) mean firing aligned to onsets.
            tau_bins: Relative bin indices.
    """
    n_bins, n_units = X.shape
    window_size = pre_bins + post_bins

    # Get aligned indices
    indices, valid = get_onset_aligned_indices(onset_bins, n_bins, pre_bins, post_bins)

    if len(onset_bins) == 0:
        return np.full((n_units, window_size), np.nan), np.arange(-pre_bins, post_bins)

    # Extract windows and compute mean
    # Shape: (n_onsets, window_size, n_units)
    windows = []
    for i, onset in enumerate(onset_bins):
        window_data = np.full((window_size, n_units), np.nan)
        for j, (idx, is_valid) in enumerate(zip(indices[i], valid[i])):
            if is_valid:
                window_data[j] = X[idx]
        windows.append(window_data)

    windows = np.array(windows)  # (n_onsets, window_size, n_units)

    # Mean across onsets (ignoring NaN)
    peth = np.nanmean(windows, axis=0).T  # (n_units, window_size)

    tau_bins = np.arange(-pre_bins, post_bins)

    return peth, tau_bins


def compute_onset_aligned_accuracy(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    onset_bins: np.ndarray,
    post_onset_labels: np.ndarray,
    pre_bins: int = 20,
    post_bins: int = 20,
    target: str = "post_onset",
) -> OnsetAlignedResult:
    """Compute accuracy as a function of time relative to syllable onsets.

    For each onset and each relative time tau, evaluate whether the prediction
    at that time matches the target label.

    Args:
        predictions: Predicted syllables per bin (n_bins,).
            -1 indicates bins not used for testing.
        true_labels: True syllable labels per bin (n_bins,).
        onset_bins: Bin indices where onsets occur.
        post_onset_labels: Syllable IDs after each onset.
        pre_bins: Number of bins before onset to analyze.
        post_bins: Number of bins after onset to analyze.
        target: What to compare against:
            - "post_onset": Syllable after the transition.
            - "current": True label at each time point.

    Returns:
        OnsetAlignedResult with accuracy curve and metrics.
    """
    n_bins = len(predictions)
    window_size = pre_bins + post_bins
    tau_bins = np.arange(-pre_bins, post_bins)

    if len(onset_bins) == 0:
        return OnsetAlignedResult(
            tau_bins=tau_bins,
            tau_sec=tau_bins.astype(float),  # Will be converted with bin_size
            accuracy_curve=np.full(window_size, np.nan),
            n_onsets=0,
        )

    # Get aligned indices
    indices, valid = get_onset_aligned_indices(onset_bins, n_bins, pre_bins, post_bins)

    # Compute accuracy at each tau
    correct_counts = np.zeros(window_size)
    total_counts = np.zeros(window_size)

    for i, onset in enumerate(onset_bins):
        target_label = post_onset_labels[i]

        for j in range(window_size):
            idx = indices[i, j]
            if not valid[i, j]:
                continue
            if predictions[idx] == -1:  # Not a test bin
                continue

            pred = predictions[idx]

            if target == "post_onset":
                # Compare to the post-onset syllable
                is_correct = pred == target_label
            else:  # "current"
                # Compare to the true label at that time
                is_correct = pred == true_labels[idx]

            correct_counts[j] += int(is_correct)
            total_counts[j] += 1

    # Compute accuracy
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_curve = np.where(
            total_counts > 0, correct_counts / total_counts, np.nan
        )

    # Find peak
    valid_acc = accuracy_curve[~np.isnan(accuracy_curve)]
    if len(valid_acc) > 0:
        peak_idx = np.nanargmax(accuracy_curve)
        peak_tau = tau_bins[peak_idx]
        peak_accuracy = accuracy_curve[peak_idx]
    else:
        peak_tau = 0
        peak_accuracy = np.nan

    # Compute width (FWHM-like)
    width = _compute_peak_width(accuracy_curve, tau_bins)

    return OnsetAlignedResult(
        tau_bins=tau_bins,
        tau_sec=tau_bins.astype(float),  # Caller should multiply by bin_size
        accuracy_curve=accuracy_curve,
        peak_tau_sec=float(peak_tau),
        peak_accuracy=float(peak_accuracy),
        width_sec=float(width),
        n_onsets=len(onset_bins),
        n_samples_per_tau=total_counts.astype(int),
    )


def _compute_peak_width(
    curve: np.ndarray,
    tau_bins: np.ndarray,
    baseline_method: str = "chance",
    n_classes: int = 10,
) -> float:
    """Compute width of peak in accuracy curve.

    Uses width at half-max above baseline (chance level).

    Args:
        curve: Accuracy curve.
        tau_bins: Corresponding tau values.
        baseline_method: How to determine baseline:
            - "chance": Use 1/n_classes.
            - "min": Use minimum of curve.
        n_classes: Number of classes for chance baseline.

    Returns:
        Width in tau units.
    """
    valid_mask = ~np.isnan(curve)
    if np.sum(valid_mask) < 3:
        return 0.0

    curve_valid = curve[valid_mask]
    tau_valid = tau_bins[valid_mask]

    if baseline_method == "chance":
        baseline = 1.0 / n_classes
    else:
        baseline = np.min(curve_valid)

    peak = np.max(curve_valid)
    if peak <= baseline:
        return 0.0

    # Half-height threshold
    threshold = baseline + 0.5 * (peak - baseline)

    # Find indices above threshold
    above = curve_valid >= threshold
    if not np.any(above):
        return 0.0

    above_indices = np.where(above)[0]
    width = tau_valid[above_indices[-1]] - tau_valid[above_indices[0]]

    return float(width)


def compute_onset_jitter_control(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    onset_bins: np.ndarray,
    post_onset_labels: np.ndarray,
    pre_bins: int = 20,
    post_bins: int = 20,
    jitter_range_bins: int = 20,
    n_iterations: int = 100,
    random_state: int = 42,
) -> OnsetAlignedResult:
    """Compute onset-aligned accuracy with jittered onset times.

    This is a control analysis: if temporal alignment is real, jittering
    onset times should flatten the accuracy peak.

    Args:
        predictions: Predicted syllables per bin.
        true_labels: True syllable labels per bin.
        onset_bins: Original onset bin indices.
        post_onset_labels: Syllable IDs after each onset.
        pre_bins: Bins before onset.
        post_bins: Bins after onset.
        jitter_range_bins: Maximum jitter in bins (±jitter_range_bins/2).
        n_iterations: Number of jitter iterations to average.
        random_state: Random seed.

    Returns:
        OnsetAlignedResult with jittered accuracy curve.
    """
    rng = np.random.RandomState(random_state)
    n_bins = len(predictions)
    window_size = pre_bins + post_bins

    all_curves = []

    for _ in range(n_iterations):
        # Jitter onset times
        jitter = rng.randint(-jitter_range_bins // 2, jitter_range_bins // 2 + 1, size=len(onset_bins))
        jittered_onsets = np.clip(onset_bins + jitter, 0, n_bins - 1)

        # Compute accuracy with jittered onsets
        result = compute_onset_aligned_accuracy(
            predictions,
            true_labels,
            jittered_onsets,
            post_onset_labels,
            pre_bins,
            post_bins,
            target="post_onset",
        )
        all_curves.append(result.accuracy_curve)

    # Average across iterations
    all_curves = np.array(all_curves)
    mean_curve = np.nanmean(all_curves, axis=0)

    tau_bins = np.arange(-pre_bins, post_bins)

    return OnsetAlignedResult(
        tau_bins=tau_bins,
        tau_sec=tau_bins.astype(float),
        accuracy_curve=mean_curve,
        n_onsets=len(onset_bins),
    )


def compute_next_syllable_accuracy(
    X: np.ndarray,
    syllables: np.ndarray,
    onset_bins: np.ndarray,
    post_onset_labels: np.ndarray,
    pre_bins: int = 20,
    post_bins: int = 20,
    config: Optional[Dict] = None,
) -> OnsetAlignedResult:
    """Decode the NEXT syllable at each tau relative to onset.

    This directly measures predictive coding: can we decode the upcoming
    syllable from neural activity before the transition?

    Args:
        X: Neural features (n_bins, n_features).
        syllables: Current syllable labels (n_bins,).
        onset_bins: Onset bin indices.
        post_onset_labels: Syllable IDs after each onset.
        pre_bins: Bins before onset.
        post_bins: Bins after onset.
        config: Optional classifier config.

    Returns:
        OnsetAlignedResult with next-syllable accuracy curve.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    n_bins = X.shape[0]
    window_size = pre_bins + post_bins
    tau_bins = np.arange(-pre_bins, post_bins)

    if len(onset_bins) == 0:
        return OnsetAlignedResult(
            tau_bins=tau_bins,
            tau_sec=tau_bins.astype(float),
            accuracy_curve=np.full(window_size, np.nan),
            n_onsets=0,
        )

    # Get aligned indices
    indices, valid = get_onset_aligned_indices(onset_bins, n_bins, pre_bins, post_bins)

    accuracy_curve = np.full(window_size, np.nan)
    n_samples = np.zeros(window_size, dtype=int)

    # For each tau, collect X and y (next syllable)
    for tau_idx in range(window_size):
        X_tau = []
        y_tau = []

        for i, onset in enumerate(onset_bins):
            idx = indices[i, tau_idx]
            if not valid[i, tau_idx]:
                continue

            X_tau.append(X[idx])
            y_tau.append(post_onset_labels[i])  # The next syllable

        if len(X_tau) < 10:  # Need minimum samples
            continue

        X_tau = np.array(X_tau)
        y_tau = np.array(y_tau)
        n_samples[tau_idx] = len(y_tau)

        # Quick decode: cross-validated
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tau)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs",
        )

        # Use cross_val_predict for efficiency
        n_cv = min(5, len(np.unique(y_tau)))
        if n_cv < 2:
            continue

        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)

        y_pred = cross_val_predict(clf, X_scaled, y_tau, cv=cv)

        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        accuracy_curve[tau_idx] = balanced_accuracy_score(y_tau, y_pred)

    # Find peak
    valid_acc = accuracy_curve[~np.isnan(accuracy_curve)]
    if len(valid_acc) > 0:
        peak_idx = np.nanargmax(accuracy_curve)
        peak_tau = tau_bins[peak_idx]
        peak_accuracy = accuracy_curve[peak_idx]
    else:
        peak_tau = 0
        peak_accuracy = np.nan

    return OnsetAlignedResult(
        tau_bins=tau_bins,
        tau_sec=tau_bins.astype(float),
        accuracy_curve=accuracy_curve,
        peak_tau_sec=float(peak_tau),
        peak_accuracy=float(peak_accuracy),
        n_onsets=len(onset_bins),
        n_samples_per_tau=n_samples,
    )
