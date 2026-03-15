"""Neural surprisal: prediction error from brain's predictive model.

Neural surprisal captures "how unexpected was the observed next action given
the neural state right now" - requires calibrated probabilistic prediction.

Model:
    p_φ(s_{t+1} | X_t, [s_t]) trained on neural data

Surprisal:
    Surp_neural(t) = -log p_φ(s_{t+1} = observed | X_t, [s_t])

Key distinction from behavioral surprisal:
- Behavioral: property of interaction statistics
- Neural: property of the brain's instantaneous predictive model

If mid-STS is genuinely predictive:
- Neural surprisal peaks earlier/cleaner around ruptures than behavioral
- Neural surprisal shows bank/OT modulation when behavior statistics are matched

Usage:
    from analysis.interaction_summaries.surprise.neural import (
        compute_neural_surprisal,
    )

    result = compute_neural_surprisal(
        neural_rates=X,
        next_states=y_next,
        current_states=y_current,  # optional conditioning
        n_folds=5,
        embargo_bins=20,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class NeuralSurprisalResult:
    """Result of neural surprisal computation.

    Attributes:
        surprisal: Surprisal time series, shape (n_samples,)
        probabilities: Predicted probability of true class, shape (n_samples,)
        next_states: Next states observed
        valid_mask: Boolean mask for valid samples
        mean_surprisal: Mean surprisal over valid samples
        std_surprisal: Std of surprisal
        accuracy: Classification accuracy (for reference)
        classes: Class labels
    """

    surprisal: np.ndarray
    probabilities: np.ndarray
    next_states: np.ndarray
    valid_mask: np.ndarray
    mean_surprisal: float
    std_surprisal: float
    accuracy: float
    classes: np.ndarray

    def surprisal_by_state(self) -> Dict[str, Dict[str, float]]:
        """Get surprisal statistics grouped by next state."""
        stats: Dict[str, List[float]] = {}

        for i in range(len(self.surprisal)):
            if not self.valid_mask[i]:
                continue
            state = str(self.next_states[i])
            if state not in stats:
                stats[state] = []
            stats[state].append(self.surprisal[i])

        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "count": len(v),
            }
            for k, v in stats.items()
        }


def _make_temporal_cv_splits(
    n_samples: int,
    n_folds: int,
    embargo_bins: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create temporal CV splits with embargo.

    Args:
        n_samples: Total number of samples
        n_folds: Number of CV folds
        embargo_bins: Number of bins to exclude around fold boundaries

    Returns:
        List of (train_indices, test_indices) tuples
    """
    fold_size = n_samples // n_folds
    splits = []

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples

        # Apply embargo
        train_end_before = max(0, test_start - embargo_bins)
        train_start_after = min(n_samples, test_end + embargo_bins)

        train_idx = np.concatenate([
            np.arange(0, train_end_before),
            np.arange(train_start_after, n_samples)
        ])
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def compute_neural_surprisal(
    neural_rates: np.ndarray,
    next_states: np.ndarray,
    current_states: Optional[np.ndarray] = None,
    n_folds: int = 5,
    embargo_bins: int = 20,
    calibration_method: str = "isotonic",
    prob_floor: float = 1e-12,
    random_state: int = 42,
) -> NeuralSurprisalResult:
    """Compute neural surprisal using calibrated classifier.

    Trains a calibrated multi-class classifier to predict next state from
    neural activity (optionally conditioned on current state), then computes
    surprisal as negative log probability of the observed next state.

    Args:
        neural_rates: Neural firing rates, shape (n_samples, n_units)
        next_states: Next behavioral state labels, shape (n_samples,)
        current_states: Current state labels for conditioning (optional)
        n_folds: Number of CV folds
        embargo_bins: Number of bins to exclude around fold boundaries
        calibration_method: Calibration method ('isotonic' or 'sigmoid')
        prob_floor: Minimum probability to prevent -inf surprisal
        random_state: Random seed for reproducibility

    Returns:
        NeuralSurprisalResult with surprisal time series
    """
    n_samples, n_units = neural_rates.shape

    if len(next_states) != n_samples:
        raise ValueError("next_states must have same length as neural_rates")

    # Filter valid samples (non-empty next states)
    valid_mask = np.array([s is not None and str(s) != "" for s in next_states])

    # Encode labels
    le = LabelEncoder()
    y_encoded = np.full(n_samples, -1, dtype=int)
    y_encoded[valid_mask] = le.fit_transform([str(s) for s, v in zip(next_states, valid_mask) if v])
    classes = le.classes_
    n_classes = len(classes)

    # Build features: neural + optional current state one-hot
    X = neural_rates.copy()

    if current_states is not None:
        # One-hot encode current states
        unique_current = sorted(set(str(s) for s in current_states if s))
        if len(unique_current) > 1:
            state_to_idx = {s: i for i, s in enumerate(unique_current)}
            state_onehot = np.zeros((n_samples, len(unique_current)), dtype=float)
            for t, s in enumerate(current_states):
                if s and str(s) in state_to_idx:
                    state_onehot[t, state_to_idx[str(s)]] = 1.0
            X = np.hstack([X, state_onehot])

    # Create CV splits
    splits = _make_temporal_cv_splits(n_samples, n_folds, embargo_bins)

    # Initialize output arrays
    surprisal = np.full(n_samples, np.nan)
    probabilities = np.full(n_samples, np.nan)
    predictions = np.full(n_samples, -1, dtype=int)

    # CV loop
    for train_idx, test_idx in splits:
        # Filter to valid samples
        train_valid = train_idx[valid_mask[train_idx]]
        test_valid = test_idx[valid_mask[test_idx]]

        if len(train_valid) < n_classes * 2 or len(test_valid) == 0:
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_valid])
        X_test = scaler.transform(X[test_valid])
        y_train = y_encoded[train_valid]
        y_test = y_encoded[test_valid]

        # Check class distribution
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            continue

        # Fit calibrated classifier
        base_clf = LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )

        # Inner CV for calibration (at least 2 folds)
        inner_cv = max(2, min(3, len(unique_train)))

        try:
            clf = CalibratedClassifierCV(
                estimator=base_clf,
                method=calibration_method,
                cv=inner_cv,
            )
            clf.fit(X_train, y_train)
        except Exception:
            # Fall back to sigmoid calibration
            clf = CalibratedClassifierCV(
                estimator=base_clf,
                method="sigmoid",
                cv=inner_cv,
            )
            clf.fit(X_train, y_train)

        # Predict probabilities for test fold
        prob_matrix = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        # Get probability of true class for each test sample
        clf_classes = list(clf.classes_)
        for j, idx in enumerate(test_valid):
            true_class = y_encoded[idx]
            if true_class in clf_classes:
                class_col = clf_classes.index(true_class)
                p = float(prob_matrix[j, class_col])
            else:
                p = prob_floor  # Unseen class

            p = max(p, prob_floor)  # Floor to prevent -inf
            probabilities[idx] = p
            surprisal[idx] = -np.log(p)
            predictions[idx] = pred[j]

    # Compute accuracy on valid samples with predictions
    pred_mask = predictions >= 0
    accuracy = float(np.mean(predictions[pred_mask] == y_encoded[pred_mask])) if pred_mask.any() else np.nan

    # Summary statistics
    valid_surp = surprisal[~np.isnan(surprisal)]
    mean_surp = float(np.mean(valid_surp)) if len(valid_surp) > 0 else np.nan
    std_surp = float(np.std(valid_surp)) if len(valid_surp) > 1 else 0.0

    return NeuralSurprisalResult(
        surprisal=surprisal,
        probabilities=probabilities,
        next_states=next_states,
        valid_mask=~np.isnan(surprisal),
        mean_surprisal=mean_surp,
        std_surprisal=std_surp,
        accuracy=accuracy,
        classes=classes,
    )


def compute_neural_surprisal_by_bank(
    neural_rates: np.ndarray,
    next_states: np.ndarray,
    unit_bank_labels: List[str],
    current_states: Optional[np.ndarray] = None,
    n_folds: int = 5,
    embargo_bins: int = 20,
    random_state: int = 42,
) -> Dict[str, NeuralSurprisalResult]:
    """Compute neural surprisal separately for dorsal and ventral banks.

    Args:
        neural_rates: Neural firing rates, shape (n_samples, n_units)
        next_states: Next behavioral state labels
        unit_bank_labels: Bank label for each unit ('dorsal' or 'ventral')
        current_states: Current state labels for conditioning (optional)
        n_folds: Number of CV folds
        embargo_bins: Embargo size
        random_state: Random seed

    Returns:
        Dict mapping bank name to NeuralSurprisalResult
    """
    results = {}
    banks = ["dorsal", "ventral"]

    for bank in banks:
        # Select units for this bank
        bank_mask = np.array([b.lower() == bank for b in unit_bank_labels])

        if not bank_mask.any():
            continue

        X_bank = neural_rates[:, bank_mask]

        results[bank] = compute_neural_surprisal(
            neural_rates=X_bank,
            next_states=next_states,
            current_states=current_states,
            n_folds=n_folds,
            embargo_bins=embargo_bins,
            random_state=random_state,
        )

    return results
