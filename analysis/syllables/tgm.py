"""Temporal Generalization Matrix (TGM) for motif state coding.

Trains a motif decoder at each time bin relative to transition onset,
then tests at all other time bins. A broad above-chance block (not just
a thin diagonal) demonstrates persistent state coding rather than
transient event detection.

Reference: King & Dehaene (2014) "Characterizing the dynamics of mental
representations: the temporal generalization method."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TGMResult:
    """Temporal Generalization Matrix result.

    Attributes:
        matrix: Accuracy matrix (n_tau, n_tau). Rows = train time,
            columns = test time.
        tau_sec: Time axis in seconds relative to transition onset.
        n_transitions: Number of transitions used.
        n_folds: Number of CV folds.
        chance_level: Expected chance accuracy (1 / n_classes).
        n_classes: Number of motif classes.
        diagonal: Diagonal of the matrix (same-time decoding).
        per_fold_matrices: Per-fold TGM matrices for statistics.
    """

    matrix: np.ndarray
    tau_sec: np.ndarray
    n_transitions: int
    n_folds: int
    chance_level: float
    n_classes: int
    diagonal: np.ndarray
    per_fold_matrices: List[np.ndarray] = field(default_factory=list)


def _apply_embargo(
    onset_bins: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo_bins: int,
) -> np.ndarray:
    """Remove training transitions within embargo distance of test transitions.

    Args:
        onset_bins: Bin indices of all transitions.
        train_idx: Indices into onset_bins for training set.
        test_idx: Indices into onset_bins for test set.
        embargo_bins: Exclusion radius in bins.

    Returns:
        Filtered train_idx with embargoed transitions removed.
    """
    if embargo_bins <= 0:
        return train_idx

    test_onsets = onset_bins[test_idx]
    train_onsets = onset_bins[train_idx]

    # For each train transition, check distance to nearest test transition
    # Shape: (n_train, n_test)
    dists = np.abs(train_onsets[:, np.newaxis] - test_onsets[np.newaxis, :])
    min_dist = dists.min(axis=1)

    keep = min_dist > embargo_bins
    return train_idx[keep]


def compute_tgm(
    X: np.ndarray,
    y: np.ndarray,
    onset_bins: np.ndarray,
    post_syllables: np.ndarray,
    pre_bins: int = 8,
    post_bins: int = 9,
    n_folds: int = 5,
    embargo_bins: int = 4,
    n_pca: int = 50,
    random_state: int = 42,
) -> TGMResult:
    """Compute Temporal Generalization Matrix for motif decoding.

    For each pair (tau_train, tau_test), trains a classifier on neural
    activity at tau_train relative to transition onset and tests at
    tau_test. Uses transition-based cross-validation with embargo.

    Args:
        X: Neural activity matrix (n_bins, n_units).
        y: Binned syllable labels (n_bins,). Used only for class count.
        onset_bins: Bin indices of syllable transitions.
        post_syllables: Syllable IDs after each transition.
        pre_bins: Number of bins before onset in window.
        post_bins: Number of bins after onset (including onset bin).
        n_folds: Number of CV folds (stratified by post_syllable).
        embargo_bins: Exclude training transitions within this many bins
            of any test transition.
        n_pca: Number of PCA components for dimensionality reduction.
            Set to 0 to skip PCA.
        random_state: Random seed for reproducibility.

    Returns:
        TGMResult with the generalization matrix and metadata.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_bins, n_units = X.shape
    window_size = pre_bins + post_bins
    tau_sec = (np.arange(window_size) - pre_bins) * 0.5  # assumes 0.5s bins

    # --- Extract onset-aligned windows ---
    # Filter transitions with fully valid windows
    valid_transitions = []
    for i, ob in enumerate(onset_bins):
        start = ob - pre_bins
        end = ob + post_bins
        if start >= 0 and end <= n_bins:
            valid_transitions.append(i)
    valid_transitions = np.array(valid_transitions, dtype=int)

    if len(valid_transitions) < n_folds * 2:
        raise ValueError(
            f"Too few valid transitions ({len(valid_transitions)}) for "
            f"{n_folds}-fold CV. Need at least {n_folds * 2}."
        )

    valid_onsets = onset_bins[valid_transitions]
    valid_labels = post_syllables[valid_transitions]
    n_trans = len(valid_transitions)

    # Build 3D tensor: (n_transitions, window_size, n_units)
    X_windows = np.zeros((n_trans, window_size, n_units), dtype=np.float32)
    for i, ob in enumerate(valid_onsets):
        X_windows[i] = X[ob - pre_bins : ob + post_bins]

    # --- Cross-validation ---
    classes = np.unique(valid_labels)
    n_classes = len(classes)
    chance = 1.0 / n_classes

    # Cap n_pca at n_units
    actual_pca = min(n_pca, n_units) if n_pca > 0 else 0

    tgm_sum = np.zeros((window_size, window_size), dtype=np.float64)
    fold_matrices = []

    rng = np.random.RandomState(random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(2**31))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_trans), valid_labels)):
        # Apply embargo
        train_idx = _apply_embargo(valid_onsets, train_idx, test_idx, embargo_bins)
        if len(train_idx) < n_classes:
            continue

        train_labels = valid_labels[train_idx]
        test_labels = valid_labels[test_idx]

        fold_tgm = np.full((window_size, window_size), np.nan, dtype=np.float64)

        for t_train in range(window_size):
            # Standardize + PCA on train data at this time point
            X_train_t = X_windows[train_idx, t_train, :]
            X_test_all = X_windows[test_idx, :, :]  # (n_test, window, n_units)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_t)

            if actual_pca > 0:
                fold_pca = min(actual_pca, len(train_idx) - 1, n_units)
                if fold_pca < 1:
                    fold_pca = min(len(train_idx), n_units)
                pca = PCA(n_components=fold_pca, random_state=random_state)
                X_train_pca = pca.fit_transform(X_train_scaled)
            else:
                X_train_pca = X_train_scaled

            # Fit classifier
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs",
            )
            clf.fit(X_train_pca, train_labels)

            # Test at all time points
            for t_test in range(window_size):
                X_test_t = X_test_all[:, t_test, :]
                X_test_scaled = scaler.transform(X_test_t)
                if actual_pca > 0:
                    X_test_pca = pca.transform(X_test_scaled)
                else:
                    X_test_pca = X_test_scaled

                y_pred = clf.predict(X_test_pca)
                fold_tgm[t_train, t_test] = balanced_accuracy_score(
                    test_labels, y_pred
                )

        fold_matrices.append(fold_tgm)
        tgm_sum += np.nan_to_num(fold_tgm)

    n_valid_folds = len(fold_matrices)
    if n_valid_folds == 0:
        raise ValueError("No valid CV folds (all had too few training samples after embargo).")

    tgm_mean = tgm_sum / n_valid_folds

    return TGMResult(
        matrix=tgm_mean,
        tau_sec=tau_sec,
        n_transitions=n_trans,
        n_folds=n_valid_folds,
        chance_level=chance,
        n_classes=n_classes,
        diagonal=np.diag(tgm_mean),
        per_fold_matrices=fold_matrices,
    )


def compute_tgm_shuffle_null(
    X: np.ndarray,
    y: np.ndarray,
    onset_bins: np.ndarray,
    post_syllables: np.ndarray,
    n_perm: int = 50,
    **tgm_kwargs,
) -> np.ndarray:
    """Compute null distribution for TGM via label permutation.

    Shuffles post_syllables labels and recomputes TGM for each permutation.

    Args:
        X: Neural activity matrix (n_bins, n_units).
        y: Binned syllable labels (n_bins,).
        onset_bins: Bin indices of syllable transitions.
        post_syllables: Syllable IDs after each transition.
        n_perm: Number of permutations.
        **tgm_kwargs: Additional kwargs passed to compute_tgm().

    Returns:
        Null TGM matrices (n_perm, window_size, window_size).
    """
    rng = np.random.RandomState(tgm_kwargs.get("random_state", 42))
    null_matrices = []

    for i in range(n_perm):
        shuffled = rng.permutation(post_syllables)
        result = compute_tgm(
            X, y, onset_bins, shuffled,
            **{**tgm_kwargs, "random_state": rng.randint(2**31)},
        )
        null_matrices.append(result.matrix)

    return np.array(null_matrices)
