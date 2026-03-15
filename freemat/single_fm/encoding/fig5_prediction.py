"""Fig 5E: Behavior-conditioned neural prediction analysis.

Tests whether neural activity adds predictive information beyond behavioral
features for (1) transition-within-2s classification and (2) next-syllable
identity prediction at the transition level.

The key comparison is Beh-only vs Beh+Neural.
If Beh+Neural > Beh-only, neural activity carries prospective information
beyond what the behavioral stream alone provides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from analysis.decoding.temporal_cv import make_temporal_folds
from freemat.single_fm.encoding.ridge import build_spline_design
from freemat.single_fm.encoding.stats import circular_shift_indices


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PredictionConfig:
    """Configuration for behavior-conditioned prediction analysis."""

    bin_s: float = 0.5
    n_cv_folds: int = 5
    embargo_bins: int = 4  # 2s at 0.5s bins
    n_perm: int = 500
    n_pcs: int = 20
    hazard_horizon_bins: int = 4  # look-ahead for transition (2s)
    pre_window_s: float = 2.0  # pre-transition window for next-syllable
    pre_buffer_s: float = 0.5  # buffer before transition onset
    spline_df: int = 5  # df for bout age splines (n_knots = df - 3)
    classifier_C: float = 1.0
    random_state: int = 42


@dataclass
class HazardResult:
    """Results for transition-within-2s classification."""

    behavior_only_auc: float
    behavior_neural_auc: float
    delta_auc: float
    delta_auc_p: float
    behavior_only_ap: float
    behavior_neural_ap: float
    delta_ap: float
    delta_ap_p: float
    n_bins: int
    transition_prevalence: float
    n_pcs_used: int
    per_fold_auc_behavior: List[float] = field(default_factory=list)
    per_fold_auc_neural: List[float] = field(default_factory=list)
    per_fold_ap_behavior: List[float] = field(default_factory=list)
    per_fold_ap_neural: List[float] = field(default_factory=list)


@dataclass
class NextSyllableResult:
    """Results for transition-level next-syllable prediction."""

    behavior_only_accuracy: float
    behavior_neural_accuracy: float
    delta_accuracy: float
    delta_accuracy_p: float
    behavior_only_loglik: float
    behavior_neural_loglik: float
    delta_loglik: float
    delta_loglik_p: float
    markov_baseline_accuracy: float
    n_transitions: int
    n_classes: int
    chance: float
    per_fold_acc_behavior: List[float] = field(default_factory=list)
    per_fold_acc_neural: List[float] = field(default_factory=list)


@dataclass
class PredictionResult:
    """Combined results for both prediction tasks."""

    session_id: str
    transition_hazard: HazardResult
    next_syllable: NextSyllableResult
    config: PredictionConfig


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

# Continuous behavioral feature columns (must match feature_builder output)
CONTINUOUS_COLS = [
    "com_x", "com_y", "com_z", "speed", "yaw_rate",
    "v_parallel", "v_perp", "pitch_rate", "roll_rate", "delta_yaw",
    "gaze_x", "gaze_y", "gaze_z", "d_wall_horiz", "vis_motion",
]


def compute_bout_age(y_motif: np.ndarray, bin_s: float) -> np.ndarray:
    """Compute time since last transition for each bin.

    Args:
        y_motif: Motif labels per bin (1-indexed, 0 = invalid).
        bin_s: Bin size in seconds.

    Returns:
        Array of bout age in seconds, same length as y_motif.
        Resets to 0.0 at each transition.
    """
    n = len(y_motif)
    bout_age = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if y_motif[i] != y_motif[i - 1]:
            bout_age[i] = 0.0
        else:
            bout_age[i] = bout_age[i - 1] + bin_s
    return bout_age


def build_transition_labels(
    y_motif: np.ndarray,
    horizon_bins: int,
) -> np.ndarray:
    """Build binary labels: will a transition occur within horizon_bins?

    Args:
        y_motif: Motif labels per bin.
        horizon_bins: Number of forward bins to check.

    Returns:
        Binary array (0/1), same length as y_motif.
    """
    n = len(y_motif)
    labels = np.zeros(n, dtype=int)

    # Find all transition indices
    transitions = set()
    for i in range(1, n):
        if y_motif[i] != y_motif[i - 1] and y_motif[i] > 0 and y_motif[i - 1] > 0:
            transitions.add(i)

    # For each bin, check if any transition is within horizon
    for t_idx in transitions:
        start = max(0, t_idx - horizon_bins)
        for j in range(start, t_idx):
            labels[j] = 1

    return labels


def aggregate_pre_transition_features(
    X: np.ndarray,
    y_motif: np.ndarray,
    t: np.ndarray,
    window_s: float = 2.0,
    buffer_s: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate features into one sample per transition.

    For each transition, averages features over the window
    [-window_s, -buffer_s] before the transition.

    Args:
        X: Feature matrix (n_bins, n_features).
        y_motif: Motif labels per bin.
        t: Time array per bin.
        window_s: Start of pre-transition window (seconds before transition).
        buffer_s: End of pre-transition window (seconds before transition).

    Returns:
        Tuple of (X_aggregated, y_next_syllable, transition_indices).
        Each has one row per valid transition.
    """
    bin_s = float(t[1] - t[0]) if len(t) > 1 else 0.5
    window_bins = int(round(window_s / bin_s))
    buffer_bins = int(round(buffer_s / bin_s))

    X_agg_list = []
    y_next_list = []
    trans_idx_list = []

    for i in range(1, len(y_motif)):
        if y_motif[i] != y_motif[i - 1] and y_motif[i] > 0 and y_motif[i - 1] > 0:
            # Valid transition at index i
            # Window: [i - window_bins, i - buffer_bins)
            win_start = i - window_bins
            win_end = i - buffer_bins

            if win_start < 0 or win_end <= win_start:
                continue

            # Check that all bins in window have the same (pre-transition) motif
            window_motifs = y_motif[win_start:win_end]
            if not np.all(window_motifs == y_motif[i - 1]):
                continue
            if not np.all(window_motifs > 0):
                continue

            X_window = X[win_start:win_end]
            X_agg_list.append(X_window.mean(axis=0))
            y_next_list.append(y_motif[i])
            trans_idx_list.append(i)

    if len(X_agg_list) == 0:
        return (
            np.zeros((0, X.shape[1])),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    return (
        np.stack(X_agg_list),
        np.array(y_next_list, dtype=int),
        np.array(trans_idx_list, dtype=int),
    )


def build_short_history_features(
    feat_df: pd.DataFrame,
    window_bins: int = 4,
) -> np.ndarray:
    """Build short-timescale behavioral summaries.

    Args:
        feat_df: DataFrame with behavioral features.
        window_bins: Number of bins for rolling window (default 4 = 2s).

    Returns:
        Array of shape (n_bins, 3): [recent_mean_speed, recent_mean_yaw,
        recent_delta_speed].
    """
    n = len(feat_df)
    result = np.zeros((n, 3), dtype=np.float64)

    speed = feat_df["speed"].values
    yaw = feat_df["yaw_rate"].values

    for i in range(n):
        start = max(0, i - window_bins + 1)
        result[i, 0] = np.mean(speed[start : i + 1])
        result[i, 1] = np.mean(yaw[start : i + 1])
        lag_idx = max(0, i - window_bins)
        result[i, 2] = speed[i] - speed[lag_idx]

    return result


def build_behavioral_features(
    y_motif: np.ndarray,
    bout_age: np.ndarray,
    feat_df: pd.DataFrame,
    spline_df: int = 5,
) -> np.ndarray:
    """Build the full behavioral feature matrix (Beh-only basis).

    Includes: current_syllable one-hot, previous_syllable one-hot,
    bout_age spline basis, syllable x bout_age interactions,
    continuous behavioral features, and short-history summaries.

    Args:
        y_motif: Motif labels per bin (1-indexed).
        bout_age: Bout age in seconds per bin.
        feat_df: DataFrame with continuous behavioral features.
        spline_df: Degrees of freedom for bout age splines.

    Returns:
        Feature matrix of shape (n_bins, n_features).
    """
    n = len(y_motif)
    classes = np.sort(np.unique(y_motif[y_motif > 0]))
    K = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # 1. Current syllable one-hot (K cols)
    current_oh = np.zeros((n, K), dtype=np.float64)
    for i in range(n):
        if y_motif[i] in class_to_idx:
            current_oh[i, class_to_idx[y_motif[i]]] = 1.0

    # 2. Previous syllable one-hot (K cols)
    prev_oh = np.zeros((n, K), dtype=np.float64)
    prev_motif = y_motif[0]
    for i in range(n):
        if i > 0 and y_motif[i] != y_motif[i - 1] and y_motif[i - 1] > 0:
            prev_motif = y_motif[i - 1]
        if prev_motif in class_to_idx:
            prev_oh[i, class_to_idx[prev_motif]] = 1.0

    # 3. Bout age spline basis (spline_df cols)
    n_knots = max(spline_df - 3, 1)
    bout_age_splines = build_spline_design(
        bout_age.reshape(-1, 1), n_knots=n_knots,
    )
    # Ensure correct number of columns
    actual_spline_cols = bout_age_splines.shape[1]

    # 4. Syllable x bout_age_spline interaction (K * actual_spline_cols cols)
    interactions = np.zeros((n, K * actual_spline_cols), dtype=np.float64)
    for k in range(K):
        mask = current_oh[:, k] == 1.0
        start_col = k * actual_spline_cols
        end_col = start_col + actual_spline_cols
        interactions[mask, start_col:end_col] = bout_age_splines[mask]

    # 5. Continuous behavioral features
    cont_cols = [c for c in CONTINUOUS_COLS if c in feat_df.columns]
    continuous = feat_df[cont_cols].values.astype(np.float64)

    # 6. Short-history summaries (3 cols)
    short_hist = build_short_history_features(feat_df, window_bins=4)

    return np.concatenate(
        [current_oh, prev_oh, bout_age_splines, interactions,
         continuous, short_hist],
        axis=1,
    )


def build_neural_features(
    X_spikes: np.ndarray,
    n_pcs: int,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, PCA]:
    """Build PCA-based neural features (fit on training data only).

    Args:
        X_spikes: Raw firing rate matrix (n_bins, n_units).
        n_pcs: Number of PCs to retain.
        train_idx: Indices of training bins (PCA fit on these only).

    Returns:
        Tuple of (X_pca, pca_model). X_pca has shape (n_bins, n_pcs).
    """
    n_pcs_actual = min(n_pcs, X_spikes.shape[1])

    scaler = StandardScaler()
    scaler.fit(X_spikes[train_idx])
    X_scaled = scaler.transform(X_spikes)

    pca = PCA(n_components=n_pcs_actual)
    pca.fit(X_scaled[train_idx])
    X_pca = pca.transform(X_scaled)

    return X_pca, pca


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------


def _fit_predict_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
    all_classes: np.ndarray | None = None,
) -> np.ndarray:
    """Fit logistic regression and return predicted probabilities.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        C: Regularization strength.
        random_state: Random seed.
        all_classes: Full set of class labels. If provided and the
            classifier only sees a subset in training, the output
            probability matrix is padded to cover all classes.

    Returns:
        Predicted probabilities (n_test, n_classes).
    """
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    # Pad to full class set if some classes are missing from this fold
    if all_classes is not None and len(clf.classes_) < len(all_classes):
        proba_full = np.zeros((len(X_test), len(all_classes)), dtype=proba.dtype)
        for i, c in enumerate(clf.classes_):
            idx = np.searchsorted(all_classes, c)
            proba_full[:, idx] = proba[:, i]
        # Re-normalize rows so they sum to 1
        row_sums = proba_full.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba_full /= row_sums
        return proba_full

    return proba


def predict_transition_hazard(
    X_beh: np.ndarray,
    X_spikes: np.ndarray,
    y_hazard: np.ndarray,
    config: PredictionConfig,
) -> HazardResult:
    """Predict transition timing: Beh-only vs Beh+Neural.

    Args:
        X_beh: Behavioral feature matrix (n_bins, n_beh_features).
        X_spikes: Raw spike matrix (n_bins, n_units) for PCA.
        y_hazard: Binary transition labels (n_bins,).
        config: Prediction configuration.

    Returns:
        HazardResult with AUC and AP for both models.
    """
    n_bins = len(y_hazard)
    folds = make_temporal_folds(
        n_bins, n_folds=config.n_cv_folds, embargo_bins=config.embargo_bins,
    )

    per_fold_auc_beh = []
    per_fold_auc_neu = []
    per_fold_ap_beh = []
    per_fold_ap_neu = []

    for train_idx, test_idx in folds:
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        # Check that both classes are present
        if len(np.unique(y_hazard[train_idx])) < 2:
            continue
        if len(np.unique(y_hazard[test_idx])) < 2:
            continue

        # Z-score behavioral features
        scaler = StandardScaler()
        X_beh_train = scaler.fit_transform(X_beh[train_idx])
        X_beh_test = scaler.transform(X_beh[test_idx])

        # Build neural PCs (fit on train)
        X_pca, _ = build_neural_features(
            X_spikes, config.n_pcs, train_idx,
        )
        X_pca_train = X_pca[train_idx]
        X_pca_test = X_pca[test_idx]

        # Beh-only model
        proba_m0 = _fit_predict_logistic(
            X_beh_train, y_hazard[train_idx], X_beh_test,
            C=config.classifier_C, random_state=config.random_state,
        )

        # Beh+Neural model
        X_comb_train = np.hstack([X_beh_train, X_pca_train])
        X_comb_test = np.hstack([X_beh_test, X_pca_test])
        proba_m1 = _fit_predict_logistic(
            X_comb_train, y_hazard[train_idx], X_comb_test,
            C=config.classifier_C, random_state=config.random_state,
        )

        y_test = y_hazard[test_idx]
        # AUC and AP on positive class probability
        auc_m0 = roc_auc_score(y_test, proba_m0[:, 1])
        auc_m1 = roc_auc_score(y_test, proba_m1[:, 1])
        ap_m0 = average_precision_score(y_test, proba_m0[:, 1])
        ap_m1 = average_precision_score(y_test, proba_m1[:, 1])

        per_fold_auc_beh.append(auc_m0)
        per_fold_auc_neu.append(auc_m1)
        per_fold_ap_beh.append(ap_m0)
        per_fold_ap_neu.append(ap_m1)

    mean_auc_beh = float(np.mean(per_fold_auc_beh))
    mean_auc_neu = float(np.mean(per_fold_auc_neu))
    mean_ap_beh = float(np.mean(per_fold_ap_beh))
    mean_ap_neu = float(np.mean(per_fold_ap_neu))

    n_pcs_used = min(config.n_pcs, X_spikes.shape[1])

    return HazardResult(
        behavior_only_auc=mean_auc_beh,
        behavior_neural_auc=mean_auc_neu,
        delta_auc=mean_auc_neu - mean_auc_beh,
        delta_auc_p=-1.0,  # filled by permutation test
        behavior_only_ap=mean_ap_beh,
        behavior_neural_ap=mean_ap_neu,
        delta_ap=mean_ap_neu - mean_ap_beh,
        delta_ap_p=-1.0,
        n_bins=n_bins,
        transition_prevalence=float(np.mean(y_hazard)),
        n_pcs_used=n_pcs_used,
        per_fold_auc_behavior=per_fold_auc_beh,
        per_fold_auc_neural=per_fold_auc_neu,
        per_fold_ap_behavior=per_fold_ap_beh,
        per_fold_ap_neural=per_fold_ap_neu,
    )


def predict_next_syllable(
    X_beh_agg: np.ndarray,
    X_neural_agg: np.ndarray,
    y_next: np.ndarray,
    config: PredictionConfig,
) -> NextSyllableResult:
    """Predict next syllable at transition level.

    Args:
        X_beh_agg: Aggregated behavioral features (n_trans, n_beh_feat).
        X_neural_agg: Aggregated neural features (n_trans, n_units).
        y_next: Next syllable labels (n_trans,).
        config: Prediction configuration.

    Returns:
        NextSyllableResult with accuracy and log-likelihood for all models.
    """
    n_trans = len(y_next)
    classes = np.unique(y_next)
    n_classes = len(classes)
    chance = 1.0 / n_classes

    # Temporal CV on transition indices (ordered by time)
    folds = make_temporal_folds(
        n_trans, n_folds=config.n_cv_folds, embargo_bins=config.embargo_bins,
    )

    per_fold_acc_beh = []
    per_fold_acc_neu = []
    per_fold_loglik_beh = []
    per_fold_loglik_neu = []
    per_fold_acc_markov = []

    for train_idx, test_idx in folds:
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        y_train = y_next[train_idx]
        y_test = y_next[test_idx]

        # Skip if not enough classes
        if len(np.unique(y_train)) < 2:
            continue

        # Z-score behavioral features
        scaler_beh = StandardScaler()
        X_beh_tr = scaler_beh.fit_transform(X_beh_agg[train_idx])
        X_beh_te = scaler_beh.transform(X_beh_agg[test_idx])

        # Neural PCs (fit on train)
        X_pca, _ = build_neural_features(
            X_neural_agg, config.n_pcs, train_idx,
        )
        X_pca_tr = X_pca[train_idx]
        X_pca_te = X_pca[test_idx]

        # Beh-only model
        proba_m0 = _fit_predict_logistic(
            X_beh_tr, y_train, X_beh_te,
            C=config.classifier_C,
            random_state=config.random_state,
            all_classes=classes,
        )
        pred_m0 = classes[np.argmax(proba_m0, axis=1)]
        acc_m0 = balanced_accuracy_score(y_test, pred_m0)
        loglik_m0 = -log_loss(y_test, proba_m0, labels=classes)

        # Beh+Neural model
        X_comb_tr = np.hstack([X_beh_tr, X_pca_tr])
        X_comb_te = np.hstack([X_beh_te, X_pca_te])
        proba_m1 = _fit_predict_logistic(
            X_comb_tr, y_train, X_comb_te,
            C=config.classifier_C,
            random_state=config.random_state,
            all_classes=classes,
        )
        pred_m1 = classes[np.argmax(proba_m1, axis=1)]
        acc_m1 = balanced_accuracy_score(y_test, pred_m1)
        loglik_m1 = -log_loss(y_test, proba_m1, labels=classes)

        # Markov baseline: current syllable only (first K cols of X_beh)
        n_beh_classes = len(np.unique(y_next[y_next > 0]))
        X_markov_tr = X_beh_tr[:, :n_beh_classes]
        X_markov_te = X_beh_te[:, :n_beh_classes]
        proba_mk = _fit_predict_logistic(
            X_markov_tr, y_train, X_markov_te,
            C=config.classifier_C,
            random_state=config.random_state,
            all_classes=classes,
        )
        pred_mk = classes[np.argmax(proba_mk, axis=1)]
        acc_mk = balanced_accuracy_score(y_test, pred_mk)

        per_fold_acc_beh.append(acc_m0)
        per_fold_acc_neu.append(acc_m1)
        per_fold_loglik_beh.append(loglik_m0)
        per_fold_loglik_neu.append(loglik_m1)
        per_fold_acc_markov.append(acc_mk)

    mean_acc_beh = float(np.mean(per_fold_acc_beh))
    mean_acc_neu = float(np.mean(per_fold_acc_neu))
    mean_loglik_beh = float(np.mean(per_fold_loglik_beh))
    mean_loglik_neu = float(np.mean(per_fold_loglik_neu))

    return NextSyllableResult(
        behavior_only_accuracy=mean_acc_beh,
        behavior_neural_accuracy=mean_acc_neu,
        delta_accuracy=mean_acc_neu - mean_acc_beh,
        delta_accuracy_p=-1.0,  # filled by permutation test
        behavior_only_loglik=mean_loglik_beh,
        behavior_neural_loglik=mean_loglik_neu,
        delta_loglik=mean_loglik_neu - mean_loglik_beh,
        delta_loglik_p=-1.0,
        markov_baseline_accuracy=float(np.mean(per_fold_acc_markov)),
        n_transitions=n_trans,
        n_classes=n_classes,
        chance=chance,
        per_fold_acc_behavior=per_fold_acc_beh,
        per_fold_acc_neural=per_fold_acc_neu,
    )


# ---------------------------------------------------------------------------
# Permutation testing
# ---------------------------------------------------------------------------


def run_permutation_test_hazard(
    X_beh: np.ndarray,
    X_spikes: np.ndarray,
    y_hazard: np.ndarray,
    observed_delta_auc: float,
    observed_delta_ap: float,
    config: PredictionConfig,
) -> Tuple[float, float, List[float], List[float]]:
    """Circular-shift permutation test for transition hazard ΔAUC and ΔAP.

    Args:
        X_beh: Behavioral features.
        X_spikes: Raw spike matrix.
        y_hazard: Binary transition labels.
        observed_delta_auc: Observed ΔAUC from real data.
        observed_delta_ap: Observed ΔAP from real data.
        config: Configuration.

    Returns:
        Tuple of (p_auc, p_ap, null_delta_aucs, null_delta_aps).
    """
    rng = np.random.default_rng(config.random_state)
    null_delta_aucs = []
    null_delta_aps = []

    for _ in range(config.n_perm):
        shift_idx = circular_shift_indices(len(X_spikes), rng=rng)
        X_spikes_shifted = X_spikes[shift_idx]

        result = predict_transition_hazard(
            X_beh, X_spikes_shifted, y_hazard,
            PredictionConfig(
                n_cv_folds=config.n_cv_folds,
                embargo_bins=config.embargo_bins,
                n_perm=0,
                n_pcs=config.n_pcs,
                classifier_C=config.classifier_C,
                random_state=config.random_state,
            ),
        )
        null_delta_aucs.append(result.delta_auc)
        null_delta_aps.append(result.delta_ap)

    null_aucs = np.array(null_delta_aucs)
    null_aps = np.array(null_delta_aps)
    p_auc = float((1 + np.sum(null_aucs >= observed_delta_auc)) / (1 + config.n_perm))
    p_ap = float((1 + np.sum(null_aps >= observed_delta_ap)) / (1 + config.n_perm))

    return p_auc, p_ap, null_delta_aucs, null_delta_aps


def run_permutation_test_next_syllable(
    X_beh_agg: np.ndarray,
    X_neural_agg: np.ndarray,
    y_next: np.ndarray,
    observed_delta_acc: float,
    observed_delta_loglik: float,
    config: PredictionConfig,
) -> Tuple[float, float, List[float], List[float]]:
    """Circular-shift permutation test for next-syllable prediction.

    Shifts the neural features relative to the transition-level samples.

    Args:
        X_beh_agg: Aggregated behavioral features.
        X_neural_agg: Aggregated neural features.
        y_next: Next syllable labels.
        observed_delta_acc: Observed Δaccuracy.
        observed_delta_loglik: Observed Δlog-likelihood.
        config: Configuration.

    Returns:
        Tuple of (p_acc, p_loglik, null_delta_accs, null_delta_logliks).
    """
    rng = np.random.default_rng(config.random_state + 1)
    null_delta_accs = []
    null_delta_logliks = []

    for _ in range(config.n_perm):
        shift_idx = circular_shift_indices(len(X_neural_agg), rng=rng)
        X_neural_shifted = X_neural_agg[shift_idx]

        result = predict_next_syllable(
            X_beh_agg, X_neural_shifted, y_next,
            PredictionConfig(
                n_cv_folds=config.n_cv_folds,
                embargo_bins=config.embargo_bins,
                n_perm=0,
                n_pcs=config.n_pcs,
                classifier_C=config.classifier_C,
                random_state=config.random_state,
            ),
        )
        null_delta_accs.append(result.delta_accuracy)
        null_delta_logliks.append(result.delta_loglik)

    null_accs = np.array(null_delta_accs)
    null_llks = np.array(null_delta_logliks)
    p_acc = float((1 + np.sum(null_accs >= observed_delta_acc)) / (1 + config.n_perm))
    p_llk = float((1 + np.sum(null_llks >= observed_delta_loglik)) / (1 + config.n_perm))

    return p_acc, p_llk, null_delta_accs, null_delta_logliks


__all__ = [
    "PredictionConfig",
    "HazardResult",
    "NextSyllableResult",
    "PredictionResult",
    "compute_bout_age",
    "build_transition_labels",
    "aggregate_pre_transition_features",
    "build_behavioral_features",
    "build_short_history_features",
    "build_neural_features",
    "predict_transition_hazard",
    "predict_next_syllable",
    "run_permutation_test_hazard",
    "run_permutation_test_next_syllable",
]
