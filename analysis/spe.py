"""Utilities for Social Prediction Error (SPE) analysis.

This module builds calibrated next-state probabilistic predictors for
transition-aware vs. state-only models to compute a per-bin Social
Prediction Error time series:

    SPE(t) = log P(s_{t+1} | s_t, X_t) - log P(s_{t+1} | X_t)

Where X_t is the subject's neural activity at time t; s_t and s_{t+1} are the
target's current and next behavioral states (bin-center labels).

It mirrors the decoding utilities' temporal CV, applies fold-wise scaling,
and performs probability calibration via isotonic regression (with sigmoid
fallback when needed). Outputs include calibrated probabilities per fold,
time series of SPE and "social surprise" -SPE, and helper functions to build
transition bigram heatmaps.

Example:
    >>> from data_utils.spe_utils import build_spe_dataset, compute_spe_timeseries
    >>> X_trans, X_state, y_next, info = build_spe_dataset(
    ...     neural_units_by_bins, behavior_df, time_col, beh_col,
    ...     bin_size_sec=1.0, smoothing_sigma_sec=0.1,
    ...     exclude_states=['unknown','other','transition']
    ... )
    >>> spe_res = compute_spe_timeseries(
    ...     X_trans, X_state, y_next,
    ...     n_splits=5, embargo_bins=2, prob_floor=1e-12,
    ...     method='isotonic', random_state=42
    ... )
    >>> spe = spe_res['spe']  # length == n_samples (NaN for excluded bins)

Note:
    - This module does not perform global z-scoring of rates; scaling occurs
      fold-wise via StandardScaler to avoid leakage.
    - Labels are assigned using bin centers and filtered according to
      provided exclude list. Bins where next state is undefined are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from analysis.decoding import make_cv_splits


@dataclass
class SPEDataset:
    X_trans: np.ndarray  # (n_samples, n_features + n_states_onehot)
    X_state: np.ndarray  # (n_samples, n_features)
    y_next: np.ndarray   # encoded labels (n_samples,)
    classes_: np.ndarray # label names (strings)
    sample_bin_idx: np.ndarray  # original bin indices in full timeline
    current_state_labels: np.ndarray  # text labels for s_t
    next_state_labels: np.ndarray     # text labels for s_{t+1}


def _assign_bin_center_labels(
    n_bins: int,
    bin_size_sec: float,
    behavior_df: pd.DataFrame,
    seconds_col_name: str,
    behavior_col_name: str,
) -> List[Optional[str]]:
    """Assign behavior label to each bin using bin centers.

    Returns a list of length n_bins with string labels or None when unlabeled.
    """
    labels = [None] * int(n_bins)
    # Compute bin centers
    centers = (np.arange(n_bins) + 0.5) * float(bin_size_sec)
    # Iterate behavioral events
    if behavior_df is None or behavior_df.empty:
        return labels
    for i, row in behavior_df.iterrows():
        raw = row.get(behavior_col_name, None)
        start = row.get(seconds_col_name, None)
        if raw is None or start is None or pd.isna(start):
            continue
        # Determine end
        if i < len(behavior_df) - 1:
            end = behavior_df[seconds_col_name].iloc[i + 1]
            if pd.isna(end):
                end = float(n_bins) * float(bin_size_sec)
        else:
            end = float(n_bins) * float(bin_size_sec)
        # Validate start and end are numeric before converting
        if not isinstance(start, (int, float, np.integer, np.floating)):
            continue
        if not isinstance(end, (int, float, np.integer, np.floating)):
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        beh = str(raw).strip().lower()
        mask = (centers >= start_f) & (centers < end_f)
        idxs = np.where(mask)[0]
        for idx in idxs:
            if labels[idx] is None:
                labels[idx] = beh
    return labels


def _shift_next(labels: List[Optional[str]]) -> List[Optional[str]]:
    """Compute next-state labels: y_next[t] = labels[t+1]; last bin -> None."""
    n = len(labels)
    if n == 0:
        return []
    out = [None] * n
    for t in range(n - 1):
        out[t] = labels[t + 1]
    out[-1] = None
    return out


def _one_hot_from_labels(
    labels: np.ndarray,
    classes: List[str],
) -> np.ndarray:
    """One-hot encode string labels into columns ordered by classes."""
    idx_map = {c: i for i, c in enumerate(classes)}
    n = labels.shape[0]
    k = len(classes)
    mat = np.zeros((n, k), dtype=np.float32)
    for i, s in enumerate(labels):
        j = idx_map.get(str(s), None)
        if j is not None:
            mat[i, j] = 1.0
    return mat


def build_spe_dataset(
    neural_units_by_bins: np.ndarray,
    behavior_df: pd.DataFrame,
    seconds_col_name: str,
    behavior_col_name: str,
    bin_size_sec: float,
    smoothing_sigma_sec: float = 0.1,
    exclude_states: Optional[List[str]] = None,
) -> SPEDataset:
    """Construct inputs for SPE models from neural and behavior data.

    - Smooth neural rates with Gaussian sigma (seconds) independent of bin size.
    - Assign s_t and s_{t+1} via bin centers; filter excluded labels.
    - Return design matrices for transition and state-only models aligned
      to the same sample set.
    """
    if neural_units_by_bins is None or neural_units_by_bins.size == 0:
        raise ValueError("Empty neural array")
    # Ensure shape (units, bins)
    if neural_units_by_bins.shape[0] < neural_units_by_bins.shape[1]:
        units_by_bins = neural_units_by_bins
    else:
        # Heuristic: transpose if rows >> cols (likely bins x units)
        u, b = neural_units_by_bins.shape
        units_by_bins = neural_units_by_bins.T if u > b else neural_units_by_bins

    n_units, n_bins = int(units_by_bins.shape[0]), int(units_by_bins.shape[1])
    # Smooth across time
    sigma_bins = float(smoothing_sigma_sec) / float(bin_size_sec) if smoothing_sigma_sec else 0.0
    X_full = units_by_bins if sigma_bins <= 0 else gaussian_filter1d(units_by_bins, sigma=sigma_bins, axis=1)
    # Features as (samples, features)
    X_samples = X_full.T  # (n_bins, n_units)

    # Labels at bin centers
    s_t = _assign_bin_center_labels(n_bins, bin_size_sec, behavior_df, seconds_col_name, behavior_col_name)
    s_tp1 = _shift_next(s_t)

    # Exclusion: specific labels not meaningful
    excl = set()
    if exclude_states:
        excl.update([str(x).strip().lower() for x in exclude_states])
    # Always exclude obvious non-behavior markers
    excl.update(["undefined", "miscellaneous", "misc", "enters", "experimenter enters", "experimenter leaves", "start"])

    s_t_arr = np.array([None if (x is None or str(x) in excl) else str(x) for x in s_t], dtype=object)
    s_tp1_arr = np.array([None if (x is None or str(x) in excl) else str(x) for x in s_tp1], dtype=object)

    # Valid samples: both s_t and s_{t+1} present
    valid_mask = np.array([(a is not None) and (b is not None) for a, b in zip(s_t_arr, s_tp1_arr)], dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("No valid samples after applying exclusions and next-state mask")

    # Build label encoder on next-state labels
    y_next_str = s_tp1_arr[valid_mask].astype(str)
    le = LabelEncoder()
    y_next = le.fit_transform(y_next_str)
    classes_ = le.classes_

    # Build current-state one-hot using union of classes across current/next
    states_union = sorted(list(set(s_t_arr[valid_mask].astype(str)).union(set(y_next_str.tolist()))))
    onehot_curr = _one_hot_from_labels(s_t_arr[valid_mask].astype(str), states_union)

    # Return matrices aligned to valid_mask
    X_state = X_samples[valid_mask, :].astype(np.float32, copy=False)
    X_trans = np.hstack([X_state, onehot_curr]).astype(np.float32, copy=False)
    sample_idx = np.nonzero(valid_mask)[0]

    return SPEDataset(
        X_trans=X_trans,
        X_state=X_state,
        y_next=y_next,
        classes_=classes_,
        sample_bin_idx=sample_idx,
        current_state_labels=s_t_arr[valid_mask].astype(str),
        next_state_labels=y_next_str,
    )


def _fit_calibrated_linear_svm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    method: str = "isotonic",
    random_state: int = 42,
    inner_cv_folds: int = 5,
) -> CalibratedClassifierCV:
    """Fit a LinearSVC with calibration on training data only.

    Uses CalibratedClassifierCV with the specified method ('isotonic' or 'sigmoid').
    """
    base = LinearSVC(class_weight="balanced", max_iter=5000, random_state=random_state)
    inner_cv = max(2, min(inner_cv_folds, len(np.unique(y_tr))))
    clf = CalibratedClassifierCV(base_estimator=base, method=method, cv=inner_cv)
    clf.fit(X_tr, y_tr)
    return clf


def compute_spe_timeseries(
    X_trans: np.ndarray,
    X_state: np.ndarray,
    y_next: np.ndarray,
    n_splits: int = 5,
    embargo_bins: int = 0,
    cv_strategy: str = "temporal_kfold",
    prob_floor: float = 1e-12,
    method: str = "isotonic",
    random_state: int = 42,
    return_fold_objects: bool = False,
) -> Dict[str, Any]:
    """Compute calibrated probabilities and SPE using temporal CV.

    Returns a dict with keys:
      - 'p_trans': ndarray (n_samples,), probability of true next class under transition model
      - 'p_state': ndarray (n_samples,), probability under state-only model
      - 'spe': ndarray (n_samples,), log p_trans - log p_state (nan where undefined)
      - 'fold_indices': list of (train_idx, test_idx) in sample space
    """
    if X_trans.shape[0] != X_state.shape[0] or X_trans.shape[0] != y_next.shape[0]:
        raise ValueError("Mismatched sample sizes for SPE inputs")

    n = X_trans.shape[0]
    # Build CV splits in sample space
    # For temporal_kfold, y is not used; pass None
    splits = make_cv_splits(n_samples=n, n_folds=n_splits, strategy=cv_strategy, y=None, embargo_bins=embargo_bins, random_state=random_state)

    # Pre-allocate
    p_trans = np.full(n, np.nan, dtype=np.float64)
    p_state = np.full(n, np.nan, dtype=np.float64)

    fold_models: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        if te_idx.size == 0:
            continue
        # Scale within fold
        s1, s2 = StandardScaler(), StandardScaler()
        Xtr_trans = s1.fit_transform(X_trans[tr_idx])
        Xte_trans = s1.transform(X_trans[te_idx])
        Xtr_state = s2.fit_transform(X_state[tr_idx])
        Xte_state = s2.transform(X_state[te_idx])
        ytr = y_next[tr_idx]
        # Fit calibrated models with robust fallback
        # TODO: Replace try-except with input validation per CLAUDE.md
        # Catches calibration failures from sklearn's CalibratedClassifierCV
        try:
            clf_trans = _fit_calibrated_linear_svm(Xtr_trans, ytr, method=method, random_state=random_state)
        except (ValueError, RuntimeWarning):
            clf_trans = _fit_calibrated_linear_svm(Xtr_trans, ytr, method="sigmoid", random_state=random_state)
        try:
            clf_state = _fit_calibrated_linear_svm(Xtr_state, ytr, method=method, random_state=random_state)
        except (ValueError, RuntimeWarning):
            clf_state = _fit_calibrated_linear_svm(Xtr_state, ytr, method="sigmoid", random_state=random_state)

        # Predict calibrated probabilities for test fold
        prob_trans = clf_trans.predict_proba(Xte_trans)
        prob_state = clf_state.predict_proba(Xte_state)
        # Align to y labels; both calibrated models share the same set of classes (ytr)
        # However, to be safe, compute probability of true class index per row using the models' classes_
        cls_t = list(clf_trans.classes_)
        cls_s = list(clf_state.classes_)
        for j, idx in enumerate(te_idx):
            y_true = int(y_next[idx])
            # Find column index in each prob matrix; if absent, use NaN -> replaced by floor later
            try:
                col_t = cls_t.index(y_true)
                p_t = float(prob_trans[j, col_t])
            except ValueError:
                p_t = np.nan
            try:
                col_s = cls_s.index(y_true)
                p_s = float(prob_state[j, col_s])
            except ValueError:
                p_s = np.nan
            p_trans[idx] = p_t
            p_state[idx] = p_s

        # Optionally collect fold objects for serialization by caller
        if return_fold_objects:
            fold_models.append({
                "fold": int(fold_id),
                "scaler_transition": s1,
                "calibrated_transition": clf_trans,
                "scaler_stateonly": s2,
                "calibrated_stateonly": clf_state,
                "classes_transition": list(getattr(clf_trans, "classes_", [])),
                "classes_stateonly": list(getattr(clf_state, "classes_", [])),
                "train_indices": np.asarray(tr_idx, dtype=int),
                "test_indices": np.asarray(te_idx, dtype=int),
            })

    # Probability floor before log
    p_trans_safe = np.clip(p_trans, prob_floor, 1.0)
    p_state_safe = np.clip(p_state, prob_floor, 1.0)
    spe = np.log(p_trans_safe) - np.log(p_state_safe)

    out = {
        "p_trans": p_trans,
        "p_state": p_state,
        "spe": spe,
        "fold_indices": splits,
    }
    if return_fold_objects:
        out["fold_models"] = fold_models
    return out


def discretize_speed_to_quintiles(
    speed_series: np.ndarray,
    smooth_sigma_sec: float,
    bin_size_sec: float,
) -> np.ndarray:
    """Smooth speed then discretize into 5 quantile states per session.

    Returns integer states in {0,1,2,3,4} aligned to speed length.
    """
    if speed_series is None or len(speed_series) == 0:
        raise ValueError("Empty speed series")
    sigma_bins = float(smooth_sigma_sec) / float(bin_size_sec) if smooth_sigma_sec else 0.0
    v = speed_series.astype(float)
    if sigma_bins > 0:
        v = gaussian_filter1d(v, sigma=max(1e-9, sigma_bins))
    # Compute quintile thresholds excluding NaNs
    v_valid = v[np.isfinite(v)]
    if v_valid.size < 5:
        # Degenerate: return all zeros
        return np.zeros_like(v, dtype=int)
    qs = np.quantile(v_valid, [0.2, 0.4, 0.6, 0.8])
    states = np.zeros_like(v, dtype=int)
    states[v > qs[0]] = 1
    states[v > qs[1]] = 2
    states[v > qs[2]] = 3
    states[v > qs[3]] = 4
    return states


def compute_bigram_spe(
    current_states: np.ndarray,
    next_states: np.ndarray,
    spe_values: np.ndarray,
    min_count: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate mean SPE per transition bigram and counts.

    Returns (mean_df, count_df) with index and columns as sorted states.
    Cells with counts < min_count are NaN in mean_df.
    """
    s_cur = pd.Series(current_states.astype(str))
    s_nxt = pd.Series(next_states.astype(str))
    spe = pd.Series(spe_values)
    valid = spe.notna()
    df = pd.DataFrame({"cur": s_cur[valid].values, "nxt": s_nxt[valid].values, "spe": spe[valid].values})
    # Pivot mean and count
    mean_mat = df.pivot_table(index="cur", columns="nxt", values="spe", aggfunc="mean")
    cnt_mat = df.pivot_table(index="cur", columns="nxt", values="spe", aggfunc="count").fillna(0).astype(int)
    mean_masked = mean_mat.where(cnt_mat >= int(min_count))
    return mean_masked, cnt_mat
