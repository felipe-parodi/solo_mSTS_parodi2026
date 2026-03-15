"""Figure 5 follow-up: transition-level surprisal analysis.

Tests whether mSTS carries a surprise-related response when the observed
next motif is unlikely under a behavior-only transition model.

Core question: does neural activity register unexpected state changes?
NOT: does neural activity predict the future beyond behavior.

Unit of analysis: one sample = one valid motif transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from freemat.single_fm.encoding.fig5_prediction import compute_bout_age


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TransitionSurprisalConfig:
    """Configuration for transition-level surprisal analysis.

    Attributes:
        bin_s: Bin size in seconds.
        pre_feature_window_s: Pre-transition feature window start (seconds before).
        pre_buffer_s: Buffer before transition onset (seconds).
        pre_neural_window_s: Pre-transition neural baseline window (seconds).
        post_neural_window_s: Post-transition neural response window (seconds).
        min_incoming_transitions: Minimum incoming transitions to retain a motif.
        n_cv_folds: Number of temporal CV folds.
        embargo_bins: Embargo gap in bins between train/test folds.
        C_logistic: Regularization parameter for multinomial logistic.
        laplace_alpha: Laplace smoothing for Markov baseline.
        n_perm: Number of permutations for session-level null.
        min_pair_for_pair_shuffle: Min A->B instances for within-pair shuffle.
        p_threshold: Significance threshold.
        prob_floor: Minimum probability to avoid infinite surprisal.
    """

    bin_s: float = 0.5
    pre_feature_window_s: float = 2.0
    pre_buffer_s: float = 0.5
    pre_neural_window_s: float = 1.0
    post_neural_window_s: float = 1.0
    min_incoming_transitions: int = 15
    n_cv_folds: int = 5
    embargo_bins: int = 4
    C_logistic: float = 1.0
    laplace_alpha: float = 1.0
    n_perm: int = 500
    min_pair_for_pair_shuffle: int = 5
    p_threshold: float = 0.05
    prob_floor: float = 1e-6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TransitionDataset:
    """Valid transitions extracted from a session.

    Attributes:
        trans_bins: Bin indices of transition onsets.
        current_motif: Motif before transition (y[t-1]).
        next_motif: Motif after transition (y[t]).
        previous_motif: Motif before the current bout.
        bout_age: Time in current motif at transition (seconds).
        height_tier: Height tier at transition (0/1/2).
        pre_feature_idx: Indices for pre-feature window per transition.
        pre_neural_idx: Indices for neural baseline window per transition.
        post_neural_idx: Indices for neural response window per transition.
    """

    trans_bins: np.ndarray
    current_motif: np.ndarray
    next_motif: np.ndarray
    previous_motif: np.ndarray
    bout_age: np.ndarray
    height_tier: np.ndarray
    pre_feature_idx: np.ndarray
    pre_neural_idx: np.ndarray
    post_neural_idx: np.ndarray


@dataclass
class BehaviorModelResult:
    """Results from the behavior-only expectation model.

    Attributes:
        p_hat: Out-of-fold predicted probabilities (n_trans, K).
        p_markov: Markov baseline probabilities (n_trans, K).
        nll_full: Mean NLL of the full behavior model.
        nll_markov: Mean NLL of the Markov baseline.
        fold_indices: Train/test splits used.
        classes: Unique class labels.
    """

    p_hat: np.ndarray
    p_markov: np.ndarray
    nll_full: float
    nll_markov: float
    fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    classes: np.ndarray


@dataclass
class SurprisalResult:
    """Per-transition surprisal values.

    Attributes:
        surprisal: Raw surprisal = -log(p_hat of observed class).
        z_surprisal: Z-scored surprisal within session.
        surprisal_markov: Markov-based surprisal.
        delta_surprisal: surprisal - surprisal_markov.
    """

    surprisal: np.ndarray
    z_surprisal: np.ndarray
    surprisal_markov: np.ndarray
    delta_surprisal: np.ndarray


@dataclass
class NullResult:
    """Session-level permutation null result.

    Attributes:
        observed_median: Observed session-level median beta_surprisal.
        null_medians: Null distribution of median betas.
        p_value: One-sided permutation p-value.
    """

    observed_median: float
    null_medians: np.ndarray
    p_value: float


# ---------------------------------------------------------------------------
# Step 1: Transition extraction
# ---------------------------------------------------------------------------


def extract_valid_transitions(
    y_motif: np.ndarray,
    bin_times: np.ndarray,
    height_tier: np.ndarray,
    config: TransitionSurprisalConfig,
) -> TransitionDataset:
    """Extract valid transitions with stability and window constraints.

    A transition at bin t is valid if:
    - y[t-1] != y[t] (motif changes)
    - Both y[t-1] and y[t] are > 0 (non-missing)
    - Current motif is stable in pre-feature window (no other transition)
    - Full pre and post windows fit within session bounds

    Args:
        y_motif: Motif labels per bin (1-indexed, 0 = invalid).
        bin_times: Time of each bin center in seconds.
        height_tier: Height tier per bin (0/1/2 or -1 for invalid).
        config: Analysis configuration.

    Returns:
        TransitionDataset with valid transitions.
    """
    n_bins = len(y_motif)
    bin_s = config.bin_s

    # Window sizes in bins
    n_pre_feature = int(round(
        (config.pre_feature_window_s - config.pre_buffer_s) / bin_s
    ))
    n_pre_buffer = int(round(config.pre_buffer_s / bin_s))
    n_pre_neural = int(round(config.pre_neural_window_s / bin_s))
    n_post_neural = int(round(config.post_neural_window_s / bin_s))

    # Total bins needed before transition onset
    n_before = n_pre_feature + n_pre_buffer
    # Total bins needed after transition onset (inclusive of onset bin)
    n_after = n_post_neural

    # Compute bout age for the full session
    bout_age_full = compute_bout_age(y_motif, bin_s)

    # Find all transition bins
    transition_mask = np.zeros(n_bins, dtype=bool)
    for i in range(1, n_bins):
        if y_motif[i] != y_motif[i - 1] and y_motif[i] > 0 and y_motif[i - 1] > 0:
            transition_mask[i] = True

    candidate_bins = np.where(transition_mask)[0]

    # Filter candidates
    valid_trans: List[int] = []
    valid_current: List[int] = []
    valid_next: List[int] = []
    valid_previous: List[int] = []
    valid_bout_age: List[float] = []
    valid_height: List[int] = []
    valid_pre_feat: List[np.ndarray] = []
    valid_pre_neural: List[np.ndarray] = []
    valid_post_neural: List[np.ndarray] = []

    for t in candidate_bins:
        # Window clipping check
        if t - n_before < 0 or t + n_after > n_bins:
            continue

        # Pre-feature window: [t - n_before, t - n_pre_buffer)
        pre_start = t - n_before
        pre_end = t - n_pre_buffer
        pre_feat_bins = np.arange(pre_start, pre_end)

        # Stability check: current motif must be constant in pre-feature window
        pre_motifs = y_motif[pre_feat_bins]
        current = y_motif[t - 1]
        if not np.all(pre_motifs == current):
            continue

        # Also check the buffer zone is stable
        buffer_bins = np.arange(pre_end, t)
        if len(buffer_bins) > 0 and not np.all(y_motif[buffer_bins] == current):
            continue

        # Find previous motif (motif before the current bout)
        prev_motif = 0  # default: start of session
        for j in range(pre_start - 1, -1, -1):
            if y_motif[j] != current:
                prev_motif = int(y_motif[j])
                break

        # Pre-neural window: [t - n_pre_neural, t)
        pre_neural_bins = np.arange(t - n_pre_neural, t)

        # Post-neural window: [t, t + n_post_neural)
        post_neural_bins = np.arange(t, t + n_post_neural)

        valid_trans.append(t)
        valid_current.append(int(current))
        valid_next.append(int(y_motif[t]))
        valid_previous.append(prev_motif)
        valid_bout_age.append(float(bout_age_full[t - 1]))
        valid_height.append(int(height_tier[t - 1]))
        valid_pre_feat.append(pre_feat_bins)
        valid_pre_neural.append(pre_neural_bins)
        valid_post_neural.append(post_neural_bins)

    if not valid_trans:
        return TransitionDataset(
            trans_bins=np.array([], dtype=int),
            current_motif=np.array([], dtype=int),
            next_motif=np.array([], dtype=int),
            previous_motif=np.array([], dtype=int),
            bout_age=np.array([], dtype=float),
            height_tier=np.array([], dtype=int),
            pre_feature_idx=np.empty((0, n_pre_feature), dtype=int),
            pre_neural_idx=np.empty((0, n_pre_neural), dtype=int),
            post_neural_idx=np.empty((0, n_post_neural), dtype=int),
        )

    return TransitionDataset(
        trans_bins=np.array(valid_trans, dtype=int),
        current_motif=np.array(valid_current, dtype=int),
        next_motif=np.array(valid_next, dtype=int),
        previous_motif=np.array(valid_previous, dtype=int),
        bout_age=np.array(valid_bout_age, dtype=float),
        height_tier=np.array(valid_height, dtype=int),
        pre_feature_idx=np.array(valid_pre_feat, dtype=int),
        pre_neural_idx=np.array(valid_pre_neural, dtype=int),
        post_neural_idx=np.array(valid_post_neural, dtype=int),
    )


# ---------------------------------------------------------------------------
# Step 2: Motif filtering
# ---------------------------------------------------------------------------


def filter_motifs_by_transition_support(
    next_motif: np.ndarray,
    min_incoming: int = 15,
) -> List[int]:
    """Retain motifs with sufficient incoming transition support.

    Args:
        next_motif: Array of next-motif labels at each transition.
        min_incoming: Minimum number of transitions into a motif.

    Returns:
        Sorted list of retained motif IDs.
    """
    unique, counts = np.unique(next_motif[next_motif > 0], return_counts=True)
    retained = [int(m) for m, c in zip(unique, counts) if c >= min_incoming]
    return sorted(retained)


# ---------------------------------------------------------------------------
# Step 3: Behavior model features
# ---------------------------------------------------------------------------


def build_transition_features(
    current_motif: np.ndarray,
    previous_motif: np.ndarray,
    bout_age: np.ndarray,
    height_tier: np.ndarray,
    pre_features: np.ndarray,
    retained_motifs: List[int],
) -> np.ndarray:
    """Build the feature matrix for the behavior-only transition model.

    Features: current_motif (one-hot), previous_motif (one-hot),
    bout_age (scalar), height_tier (one-hot), pre-window feature means.

    Args:
        current_motif: Current motif per transition.
        previous_motif: Previous motif per transition.
        bout_age: Bout age per transition (seconds).
        height_tier: Height tier per transition (0/1/2).
        pre_features: Pre-window aggregated continuous features (n_trans, n_feat).
        retained_motifs: List of retained motif IDs.

    Returns:
        Feature matrix (n_trans, n_cols).
    """
    all_motifs = sorted(retained_motifs) + [0]  # 0 = "other"

    def _one_hot(labels: np.ndarray, classes: list) -> np.ndarray:
        out = np.zeros((len(labels), len(classes)), dtype=np.float64)
        for j, c in enumerate(classes):
            if c == 0:
                # "other": anything not in retained_motifs
                out[:, j] = (~np.isin(labels, retained_motifs)).astype(float)
            else:
                out[:, j] = (labels == c).astype(float)
        return out

    blocks = [
        _one_hot(current_motif, all_motifs),
        _one_hot(previous_motif, all_motifs),
        bout_age.reshape(-1, 1).astype(np.float64),
        _one_hot(height_tier, [0, 1, 2]),
        np.asarray(pre_features, dtype=np.float64),
    ]

    return np.hstack(blocks)


def aggregate_pre_features(
    feat_df: pd.DataFrame,
    pre_feature_idx: np.ndarray,
    feature_cols: List[str],
) -> np.ndarray:
    """Aggregate continuous features over pre-transition windows.

    Args:
        feat_df: Full session feature DataFrame (one row per bin).
        pre_feature_idx: (n_trans, n_pre_bins) bin indices.
        feature_cols: Column names to aggregate.

    Returns:
        (n_trans, len(feature_cols)) array of pre-window means.
    """
    values = feat_df[feature_cols].values.astype(np.float64)
    n_trans = pre_feature_idx.shape[0]
    out = np.zeros((n_trans, len(feature_cols)), dtype=np.float64)
    for i in range(n_trans):
        out[i] = np.nanmean(values[pre_feature_idx[i]], axis=0)
    return out


# ---------------------------------------------------------------------------
# Step 3: Behavior-only model
# ---------------------------------------------------------------------------


def _make_transition_folds(
    n_trans: int,
    trans_bins: np.ndarray,
    n_folds: int,
    embargo_bins: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create temporal CV folds over transitions ordered by time.

    Args:
        n_trans: Number of transitions.
        trans_bins: Bin indices of transitions (used for ordering).
        n_folds: Number of folds.
        embargo_bins: Embargo gap in original bin indices.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    order = np.argsort(trans_bins)
    fold_size = n_trans // n_folds
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n_trans
        test_ordered = order[start:end]

        # Embargo: exclude transitions within embargo_bins of any test transition
        train_mask = np.ones(n_trans, dtype=bool)
        train_mask[test_ordered] = False

        for ti in test_ordered:
            t_bin = trans_bins[ti]
            for tj in range(n_trans):
                if train_mask[tj]:
                    if abs(trans_bins[tj] - t_bin) <= embargo_bins:
                        train_mask[tj] = False

        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_ordered))

    return folds


def _markov_probabilities(
    current_motif_train: np.ndarray,
    next_motif_train: np.ndarray,
    current_motif_test: np.ndarray,
    classes: np.ndarray,
    laplace_alpha: float = 1.0,
) -> np.ndarray:
    """Compute Markov P(next | current) with Laplace smoothing.

    Args:
        current_motif_train: Current motifs in training set.
        next_motif_train: Next motifs in training set.
        current_motif_test: Current motifs in test set.
        classes: All class labels.
        laplace_alpha: Smoothing parameter.

    Returns:
        (n_test, K) probability matrix.
    """
    K = len(classes)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}

    # Count transitions per (current, next)
    counts = np.full((K, K), laplace_alpha, dtype=np.float64)
    for cur, nxt in zip(current_motif_train, next_motif_train):
        ci = class_to_idx.get(int(cur))
        ni = class_to_idx.get(int(nxt))
        if ci is not None and ni is not None:
            counts[ci, ni] += 1.0

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums

    # Look up test transitions
    n_test = len(current_motif_test)
    p_markov = np.zeros((n_test, K), dtype=np.float64)
    for i, cur in enumerate(current_motif_test):
        ci = class_to_idx.get(int(cur))
        if ci is not None:
            p_markov[i] = probs[ci]
        else:
            p_markov[i] = 1.0 / K  # uniform for unknown current

    return p_markov


def fit_behavior_model(
    X_beh: np.ndarray,
    y_next: np.ndarray,
    trans_bins: np.ndarray,
    config: TransitionSurprisalConfig,
    *,
    current_motif: Optional[np.ndarray] = None,
) -> BehaviorModelResult:
    """Fit behavior-only multinomial model and compute out-of-fold probabilities.

    Args:
        X_beh: Feature matrix (n_trans, n_features).
        y_next: Next motif labels (n_trans,).
        trans_bins: Bin indices of transitions (for temporal ordering).
        config: Analysis configuration.
        current_motif: Current motif per transition (for Markov baseline).

    Returns:
        BehaviorModelResult with out-of-fold probabilities and NLL.
    """
    classes = np.sort(np.unique(y_next))
    K = len(classes)
    n_trans = len(y_next)

    folds = _make_transition_folds(
        n_trans, trans_bins, config.n_cv_folds, config.embargo_bins
    )

    p_hat = np.zeros((n_trans, K), dtype=np.float64)
    p_markov = np.zeros((n_trans, K), dtype=np.float64)

    for train_idx, test_idx in folds:
        # Scale features per fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_beh[train_idx])
        X_test = scaler.transform(X_beh[test_idx])

        # Fit multinomial logistic
        clf = LogisticRegression(
            solver="lbfgs",
            C=config.C_logistic,
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_train, y_next[train_idx])
        p_hat[test_idx] = clf.predict_proba(X_test)

        # Markov baseline
        if current_motif is not None:
            p_markov[test_idx] = _markov_probabilities(
                current_motif[train_idx],
                y_next[train_idx],
                current_motif[test_idx],
                classes,
                laplace_alpha=config.laplace_alpha,
            )
        else:
            p_markov[test_idx] = 1.0 / K

    # Clip probabilities
    p_hat = np.clip(p_hat, config.prob_floor, None)
    p_hat = p_hat / p_hat.sum(axis=1, keepdims=True)

    p_markov = np.clip(p_markov, config.prob_floor, None)
    p_markov = p_markov / p_markov.sum(axis=1, keepdims=True)

    # Compute NLL
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    nll_full_vals = np.zeros(n_trans)
    nll_markov_vals = np.zeros(n_trans)
    for i in range(n_trans):
        ci = class_to_idx[int(y_next[i])]
        nll_full_vals[i] = -np.log(p_hat[i, ci])
        nll_markov_vals[i] = -np.log(p_markov[i, ci])

    return BehaviorModelResult(
        p_hat=p_hat,
        p_markov=p_markov,
        nll_full=float(np.mean(nll_full_vals)),
        nll_markov=float(np.mean(nll_markov_vals)),
        fold_indices=folds,
        classes=classes,
    )


# ---------------------------------------------------------------------------
# Step 4: Surprisal
# ---------------------------------------------------------------------------


def compute_surprisal(
    p_hat: np.ndarray,
    p_markov: np.ndarray,
    y_next: np.ndarray,
    classes: np.ndarray,
    prob_floor: float = 1e-6,
) -> SurprisalResult:
    """Compute per-transition surprisal and z-scored surprisal.

    Args:
        p_hat: Out-of-fold probabilities (n_trans, K).
        p_markov: Markov baseline probabilities (n_trans, K).
        y_next: Observed next motif labels.
        classes: Class labels.
        prob_floor: Minimum probability to prevent infinite surprisal.

    Returns:
        SurprisalResult with raw, z-scored, and delta surprisal.
    """
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    n = len(y_next)

    p_hat_clipped = np.clip(p_hat, prob_floor, None)
    p_markov_clipped = np.clip(p_markov, prob_floor, None)

    surprisal = np.zeros(n, dtype=np.float64)
    surprisal_markov = np.zeros(n, dtype=np.float64)

    for i in range(n):
        ci = class_to_idx[int(y_next[i])]
        surprisal[i] = -np.log(p_hat_clipped[i, ci])
        surprisal_markov[i] = -np.log(p_markov_clipped[i, ci])

    delta_surprisal = surprisal - surprisal_markov

    # Z-score within session
    mu = np.mean(surprisal)
    sd = np.std(surprisal)
    if sd < 1e-12:
        sd = 1.0
    z_surprisal = (surprisal - mu) / sd

    return SurprisalResult(
        surprisal=surprisal,
        z_surprisal=z_surprisal,
        surprisal_markov=surprisal_markov,
        delta_surprisal=delta_surprisal,
    )


# ---------------------------------------------------------------------------
# Step 5: Neural response
# ---------------------------------------------------------------------------


def compute_neural_responses(
    X_spikes_z: np.ndarray,
    pre_neural_idx: np.ndarray,
    post_neural_idx: np.ndarray,
) -> np.ndarray:
    """Compute delta firing rate (post - pre) per unit per transition.

    Args:
        X_spikes_z: Z-scored spike matrix (n_bins, n_units).
        pre_neural_idx: (n_trans, n_pre_bins) bin indices for baseline.
        post_neural_idx: (n_trans, n_post_bins) bin indices for response.

    Returns:
        delta_fr array of shape (n_trans, n_units).
    """
    n_trans = pre_neural_idx.shape[0]
    n_units = X_spikes_z.shape[1]
    delta_fr = np.zeros((n_trans, n_units), dtype=np.float64)

    for i in range(n_trans):
        pre_mean = np.mean(X_spikes_z[pre_neural_idx[i]], axis=0)
        post_mean = np.mean(X_spikes_z[post_neural_idx[i]], axis=0)
        delta_fr[i] = post_mean - pre_mean

    return delta_fr


# ---------------------------------------------------------------------------
# Step 6: Per-unit regression
# ---------------------------------------------------------------------------


def _build_regression_design(
    z_surprisal: np.ndarray,
    controls_df: pd.DataFrame,
) -> np.ndarray:
    """Build the OLS design matrix with surprisal + controls.

    Categorical variables (current_motif, next_motif, height_tier) are
    one-hot encoded with the first category dropped.

    Args:
        z_surprisal: Z-scored surprisal (n_trans,).
        controls_df: Control variables DataFrame.

    Returns:
        Design matrix (n_trans, n_cols) including intercept.
    """
    n = len(z_surprisal)
    blocks: List[np.ndarray] = [
        np.ones((n, 1)),  # intercept
        z_surprisal.reshape(-1, 1),  # surprisal
    ]

    # Categorical one-hot (drop first)
    for cat_col in ["current_motif", "next_motif", "height_tier"]:
        if cat_col in controls_df.columns:
            vals = controls_df[cat_col].values
            unique_vals = np.sort(np.unique(vals))
            if len(unique_vals) > 1:
                for v in unique_vals[1:]:  # drop first
                    blocks.append((vals == v).astype(float).reshape(-1, 1))

    # Continuous controls
    for cont_col in ["bout_age", "post_speed", "post_abs_yaw",
                      "pre_speed", "pre_abs_yaw"]:
        if cont_col in controls_df.columns:
            blocks.append(
                controls_df[cont_col].values.astype(float).reshape(-1, 1)
            )

    return np.hstack(blocks)


def fit_surprisal_regression(
    delta_fr: np.ndarray,
    z_surprisal: np.ndarray,
    controls_df: pd.DataFrame,
    unit_ids: np.ndarray,
) -> pd.DataFrame:
    """Fit per-unit OLS regression of delta_fr on surprisal + controls.

    Args:
        delta_fr: (n_trans, n_units) firing rate change.
        z_surprisal: (n_trans,) z-scored surprisal.
        controls_df: DataFrame with control columns.
        unit_ids: Unit identifiers.

    Returns:
        DataFrame with one row per unit: unit_id, beta_surprisal,
        t_stat, p_value, r2, n_transitions.
    """
    from scipy import stats as sp_stats

    X = _build_regression_design(z_surprisal, controls_df)
    n, p = X.shape
    n_units = delta_fr.shape[1]

    records: List[Dict[str, Any]] = []
    for u in range(n_units):
        y = delta_fr[:, u]
        finite_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_u = X[finite_mask]
        y_u = y[finite_mask]
        n_u = len(y_u)

        if n_u <= p + 1:
            records.append({
                "unit_id": int(unit_ids[u]),
                "beta_surprisal": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "r2": np.nan,
                "n_transitions": n_u,
            })
            continue

        # OLS via normal equations
        beta, residuals, rank, sv = np.linalg.lstsq(X_u, y_u, rcond=None)

        y_pred = X_u @ beta
        ss_res = np.sum((y_u - y_pred) ** 2)
        ss_tot = np.sum((y_u - y_u.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Standard errors
        dof = n_u - p
        if dof > 0 and ss_res > 0:
            mse = ss_res / dof
            try:
                cov = mse * np.linalg.inv(X_u.T @ X_u)
                se = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                se = np.full(p, np.nan)
        else:
            se = np.full(p, np.nan)

        # beta_surprisal is index 1 (after intercept)
        beta_surp = float(beta[1])
        se_surp = float(se[1]) if np.isfinite(se[1]) else np.nan
        if np.isfinite(se_surp) and se_surp > 1e-12:
            t_stat = beta_surp / se_surp
            p_val = float(2.0 * sp_stats.t.sf(abs(t_stat), dof))
        else:
            t_stat = np.nan
            p_val = np.nan

        records.append({
            "unit_id": int(unit_ids[u]),
            "beta_surprisal": beta_surp,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "r2": float(r2),
            "n_transitions": n_u,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 7: Session-level null
# ---------------------------------------------------------------------------


def _shuffle_within_groups(
    z_surprisal: np.ndarray,
    current_motif: np.ndarray,
    next_motif: np.ndarray,
    min_pair_for_pair_shuffle: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle surprisal within transition identity groups.

    For A->B pairs with >= min_pair instances, shuffle within pair.
    Otherwise, fall back to shuffling within current motif A.

    Args:
        z_surprisal: Original z-scored surprisal.
        current_motif: Current motif per transition.
        next_motif: Next motif per transition.
        min_pair_for_pair_shuffle: Min pair count for within-pair shuffle.
        rng: Random number generator.

    Returns:
        Shuffled z_surprisal array.
    """
    shuffled = z_surprisal.copy()
    n = len(z_surprisal)
    assigned = np.zeros(n, dtype=bool)

    # Try within-pair shuffle first
    pairs = {}
    for i in range(n):
        key = (int(current_motif[i]), int(next_motif[i]))
        pairs.setdefault(key, []).append(i)

    for key, indices in pairs.items():
        if len(indices) >= min_pair_for_pair_shuffle:
            idx = np.array(indices)
            shuffled[idx] = rng.permutation(shuffled[idx])
            assigned[idx] = True

    # Fall back: within current motif for unassigned
    if not np.all(assigned):
        motif_groups: Dict[int, List[int]] = {}
        for i in range(n):
            if not assigned[i]:
                m = int(current_motif[i])
                motif_groups.setdefault(m, []).append(i)

        for indices in motif_groups.values():
            idx = np.array(indices)
            shuffled[idx] = rng.permutation(shuffled[idx])

    return shuffled


def run_session_null(
    delta_fr: np.ndarray,
    z_surprisal: np.ndarray,
    controls_df: pd.DataFrame,
    current_motif: np.ndarray,
    next_motif: np.ndarray,
    n_perm: int = 500,
    min_pair_for_pair_shuffle: int = 5,
    seed: int = 42,
) -> NullResult:
    """Run session-level permutation test for surprisal effect.

    Shuffles z_surprisal within transition identity groups, refits
    all unit regressions, and computes null distribution of session-level
    median beta_surprisal.

    Args:
        delta_fr: (n_trans, n_units) firing rate change.
        z_surprisal: (n_trans,) z-scored surprisal.
        controls_df: Control variables DataFrame.
        current_motif: Current motif per transition.
        next_motif: Next motif per transition.
        n_perm: Number of permutations.
        min_pair_for_pair_shuffle: Min pair count for within-pair shuffle.
        seed: Random seed.

    Returns:
        NullResult with observed median, null distribution, and p-value.
    """
    unit_ids = np.arange(delta_fr.shape[1])

    # Observed
    obs_df = fit_surprisal_regression(
        delta_fr, z_surprisal, controls_df, unit_ids
    )
    observed_median = float(obs_df["beta_surprisal"].median())

    # Null distribution
    rng = np.random.default_rng(seed)
    null_medians = np.zeros(n_perm, dtype=np.float64)

    for p in range(n_perm):
        shuffled = _shuffle_within_groups(
            z_surprisal, current_motif, next_motif,
            min_pair_for_pair_shuffle, rng,
        )
        null_df = fit_surprisal_regression(
            delta_fr, shuffled, controls_df, unit_ids
        )
        null_medians[p] = float(null_df["beta_surprisal"].median())

    # One-sided p-value
    p_value = float(
        (np.sum(null_medians >= observed_median) + 1) / (n_perm + 1)
    )

    return NullResult(
        observed_median=observed_median,
        null_medians=null_medians,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "TransitionSurprisalConfig",
    "TransitionDataset",
    "BehaviorModelResult",
    "SurprisalResult",
    "NullResult",
    "extract_valid_transitions",
    "filter_motifs_by_transition_support",
    "build_transition_features",
    "aggregate_pre_features",
    "fit_behavior_model",
    "compute_surprisal",
    "compute_neural_responses",
    "fit_surprisal_regression",
    "run_session_null",
]
