"""Multi-class syllable decoding from neural activity.

This module provides syllable classification using the existing decoding
infrastructure, adapted for multi-class syllable labels (~15-20 classes).

Core Methods:

    Multi-Class Classification:
        Model: Logistic Regression or LinearSVC with balanced class weights.

        Class weight adjustment:
            w_c = n_samples / (n_classes × n_c)

        This upweights rare classes proportionally to their inverse frequency.

    Evaluation Metrics:
        Balanced Accuracy: BA = (1/C) × Σ_c Recall_c
        Macro F1: F1 = (1/C) × Σ_c F1_c

        See analysis.decoding.metrics for full definitions.

    Permutation Null:
        For i = 1..n_permutations:
            y_shuffled = random_permutation(y)
            null_acc[i] = decode(X, y_shuffled)

        p-value = (1 + Σ_i 𝟙[null_acc_i ≥ observed_acc]) / (1 + n_perm)

    Partner vs Self Weight Index:
        WI = (acc_partner - acc_self) / (acc_partner + acc_self)

        WI ∈ [-1, +1]: negative = self-dominated, positive = partner-dominated.

Key analyses:
- Self syllable decoding: Decode M1's syllable from M1's spikes
- Partner syllable decoding: Decode M2's syllable from M1's spikes
- Model family comparison: spikes-only vs controls-only vs combined
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from analysis.decoding import (
    StateDecoder,
    StateDecoderConfig,
    DecodingResult,
    compute_metrics,
    make_temporal_folds,
)


@dataclass
class SyllableDecodingConfig:
    """Configuration for syllable decoding.

    Attributes:
        n_folds: Number of temporal CV folds.
        embargo_bins: Number of bins to exclude around fold boundaries.
        n_permutations: Number of label shuffles for null distribution.
        classifier_type: "logistic" or "svm".
        top_n_syllables: Number of top syllables to keep (rest → OTHER).
        other_id: ID for non-top-N syllables.
        random_state: Random seed for reproducibility.
    """

    n_folds: int = 5
    embargo_bins: int = 20  # 2s at 100ms bins
    n_permutations: int = 1000
    classifier_type: str = "logistic"
    top_n_syllables: int = 10
    other_id: int = -1
    random_state: int = 42


@dataclass
class SyllableDecodingResult:
    """Result from syllable decoding.

    Attributes:
        balanced_accuracy: Balanced accuracy across classes.
        macro_f1: Macro-averaged F1 score.
        confusion_matrix: (n_classes, n_classes) confusion matrix.
        class_labels: List of class labels corresponding to confusion matrix.
        per_fold_accuracy: Accuracy per CV fold.
        predictions: Per-bin predicted syllables (test bins only, -1 for train).
        probabilities: Per-bin class probabilities if available.
        n_samples: Total number of samples.
        n_train_samples: Number of training samples.
        n_test_samples: Number of test samples.
    """

    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    class_labels: List[int]
    per_fold_accuracy: List[float]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    n_samples: int = 0
    n_train_samples: int = 0
    n_test_samples: int = 0


def decode_syllables(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[SyllableDecodingConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> SyllableDecodingResult:
    """Decode syllable identity from neural features.

    Uses temporal block CV with embargo to prevent leakage.

    Args:
        X: Feature matrix (n_samples, n_features). Typically spike rates.
        y: Syllable labels (n_samples,). Integer IDs.
        config: Decoding configuration. Uses defaults if None.
        sample_weight: Optional per-sample weights.

    Returns:
        SyllableDecodingResult with metrics and predictions.
    """
    if config is None:
        config = SyllableDecodingConfig()

    n_samples = X.shape[0]
    if n_samples != len(y):
        raise ValueError(f"X has {n_samples} samples but y has {len(y)}")

    # Get unique classes
    unique_classes = sorted(list(set(y)))
    n_classes = len(unique_classes)

    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    # Map labels to contiguous indices for sklearn
    label_to_idx = {label: i for i, label in enumerate(unique_classes)}
    y_mapped = np.array([label_to_idx[label] for label in y])

    # Create temporal folds
    folds = make_temporal_folds(n_samples, config.n_folds, config.embargo_bins)

    # Storage for predictions
    predictions = np.full(n_samples, -1, dtype=int)
    all_fold_accuracies = []

    # Confusion matrix accumulator
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Import sklearn here to allow module to load without it
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, f1_score

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) == 0:
            continue

        # Scale features within fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        y_train = y_mapped[train_idx]
        y_test = y_mapped[test_idx]

        # Fit classifier
        if config.classifier_type == "logistic":
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.random_state,
                solver="lbfgs",
            )
        else:
            clf = LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                random_state=config.random_state,
            )

        sw = sample_weight[train_idx] if sample_weight is not None else None
        clf.fit(X_train, y_train, sample_weight=sw)

        # Predict
        y_pred = clf.predict(X_test)
        predictions[test_idx] = y_pred

        # Per-fold accuracy
        fold_acc = balanced_accuracy_score(y_test, y_pred)
        all_fold_accuracies.append(fold_acc)

        # Update confusion matrix
        for true_label, pred_label in zip(y_test, y_pred):
            conf_matrix[true_label, pred_label] += 1

    # Compute overall metrics on test predictions
    test_mask = predictions != -1
    y_true_test = y_mapped[test_mask]
    y_pred_test = predictions[test_mask]

    balanced_acc = balanced_accuracy_score(y_true_test, y_pred_test)
    macro_f1 = f1_score(y_true_test, y_pred_test, average="macro")

    # Map predictions back to original labels
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    predictions_orig = np.array(
        [idx_to_label.get(p, -1) if p != -1 else -1 for p in predictions]
    )

    return SyllableDecodingResult(
        balanced_accuracy=balanced_acc,
        macro_f1=macro_f1,
        confusion_matrix=conf_matrix,
        class_labels=unique_classes,
        per_fold_accuracy=all_fold_accuracies,
        predictions=predictions_orig,
        n_samples=n_samples,
        n_train_samples=int(np.sum(~test_mask)),
        n_test_samples=int(np.sum(test_mask)),
    )


def decode_self_and_partner_syllables(
    X_spk: np.ndarray,
    y_self: np.ndarray,
    y_partner: np.ndarray,
    config: Optional[SyllableDecodingConfig] = None,
) -> Dict[str, SyllableDecodingResult]:
    """Decode both self and partner syllables from neural activity.

    Args:
        X_spk: Neural features (n_samples, n_units).
        y_self: Self syllable labels (n_samples,).
        y_partner: Partner syllable labels (n_samples,).
        config: Decoding configuration.

    Returns:
        Dict with 'self' and 'partner' SyllableDecodingResult.
    """
    results = {}

    # Decode self syllables
    results["self"] = decode_syllables(X_spk, y_self, config)

    # Decode partner syllables
    results["partner"] = decode_syllables(X_spk, y_partner, config)

    return results


def decode_syllables_by_bank(
    X_spk: np.ndarray,
    y: np.ndarray,
    unit_bank_labels: np.ndarray,
    config: Optional[SyllableDecodingConfig] = None,
) -> Dict[str, SyllableDecodingResult]:
    """Decode syllables separately for each neural bank.

    Args:
        X_spk: Neural features (n_samples, n_units).
        y: Syllable labels (n_samples,).
        unit_bank_labels: Bank label for each unit ('ventral' or 'dorsal').
        config: Decoding configuration.

    Returns:
        Dict with keys 'ventral', 'dorsal', 'pooled'.
    """
    results = {}
    banks = ["ventral", "dorsal"]

    for bank in banks:
        bank_mask = unit_bank_labels == bank
        if np.sum(bank_mask) == 0:
            continue
        X_bank = X_spk[:, bank_mask]
        results[bank] = decode_syllables(X_bank, y, config)

    # Also decode with all units
    results["pooled"] = decode_syllables(X_spk, y, config)

    return results


def decode_model_families(
    X_spk: np.ndarray,
    X_ctl: np.ndarray,
    y: np.ndarray,
    config: Optional[SyllableDecodingConfig] = None,
) -> Dict[str, SyllableDecodingResult]:
    """Decode syllables with different model families.

    Model families:
        M0: controls only (kinematics + geometry)
        M1: spikes only
        M2: spikes + controls

    Args:
        X_spk: Neural features (n_samples, n_units).
        X_ctl: Control features (n_samples, n_control_features).
        y: Syllable labels (n_samples,).
        config: Decoding configuration.

    Returns:
        Dict with keys 'M0_controls', 'M1_spikes', 'M2_combined'.
    """
    results = {}

    # M0: controls only
    results["M0_controls"] = decode_syllables(X_ctl, y, config)

    # M1: spikes only
    results["M1_spikes"] = decode_syllables(X_spk, y, config)

    # M2: combined
    X_combined = np.hstack([X_spk, X_ctl])
    results["M2_combined"] = decode_syllables(X_combined, y, config)

    return results


def compute_permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[SyllableDecodingConfig] = None,
    n_permutations: Optional[int] = None,
) -> np.ndarray:
    """Compute null distribution via label shuffling.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Syllable labels (n_samples,).
        config: Decoding configuration.
        n_permutations: Override config.n_permutations if provided.

    Returns:
        Array of balanced accuracies from shuffled labels.
    """
    if config is None:
        config = SyllableDecodingConfig()

    n_perm = n_permutations if n_permutations is not None else config.n_permutations
    rng = np.random.RandomState(config.random_state)

    null_accuracies = []
    for _ in range(n_perm):
        y_shuffled = rng.permutation(y)
        result = decode_syllables(X, y_shuffled, config)
        null_accuracies.append(result.balanced_accuracy)

    return np.array(null_accuracies)


def compute_weight_index(
    acc_partner: float,
    acc_self: float,
) -> float:
    """Compute partner vs self weighting index.

    Mathematical Definition:
        WI = (acc_partner - acc_self) / (acc_partner + acc_self)

        Interpretation:
            WI = -1: Only self information (acc_partner = 0)
            WI =  0: Equal self and partner information
            WI = +1: Only partner information (acc_self = 0)

        Useful for comparing ventral (expected self-dominated) vs
        dorsal (expected partner-dominated) banks.

    Args:
        acc_partner: Partner syllable decoding accuracy ∈ [0, 1].
        acc_self: Self syllable decoding accuracy ∈ [0, 1].

    Returns:
        Weight index in [-1, 1], or 0.0 if both accuracies are 0.
    """
    denom = acc_partner + acc_self
    if denom <= 0:
        return 0.0
    return (acc_partner - acc_self) / denom
