"""Cross-validation, feature engineering, and metrics for motif decoding."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def chronological_folds(T: int, k: int, purge: int) -> List[Dict[str, np.ndarray]]:
    """Generate chronological folds with purge windows around boundaries.

    Args:
        T: Number of timepoints
        k: Number of folds
        purge: Purge window in bins to remove around test segment in training

    Returns:
        List of dicts with 'train' and 'test' index arrays
    """
    k = max(1, int(k))
    fold_sizes = [T // k + (1 if i < (T % k) else 0) for i in range(k)]
    idx = np.arange(T)
    folds: List[Dict[str, np.ndarray]] = []
    start = 0
    for fs in fold_sizes:
        test = idx[start : start + fs]
        start += fs
        # Purge around test segment
        left = max(test[0] - purge, 0)
        right = min(test[-1] + purge, T - 1)
        mask = np.ones(T, dtype=bool)
        mask[left : right + 1] = False
        train = idx[mask]
        folds.append({"train": train, "test": test})
    return folds


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------


def build_lagged_design(fr_tu: np.ndarray, history_bins: int) -> np.ndarray:
    """Build causal history design (T_eff, U*H) with most-recent-first order."""
    from numpy.lib.stride_tricks import as_strided

    T, U = fr_tu.shape
    H = int(history_bins)
    if H <= 0 or H > T:
        raise ValueError("history_bins must be in [1, T]")
    s0, s1 = fr_tu.strides
    Z = as_strided(fr_tu, shape=(T - H + 1, H, U), strides=(s0, s0, s1))
    Zr = Z[:, ::-1, :]
    return Zr.reshape(T - H + 1, U * H)


def apply_lag_vec(y: np.ndarray, lag_bins: int) -> Tuple[np.ndarray, slice, slice]:
    """Apply integer lag to align labels with X design length.

    Positive lag means labels shifted forward (predicting future).
    Returns (y_aligned, x_slice, y_slice) to index X and original y.
    """
    if lag_bins == 0:
        return y.copy(), slice(None), slice(None)
    if lag_bins > 0:
        return y[lag_bins:], slice(None, -lag_bins), slice(None)
    # lag_bins < 0
    k = -int(lag_bins)
    return y[: -k], slice(k, None), slice(None)


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def _standardize_train_test(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s, scaler.mean_.copy(), scaler.scale_.copy()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _macro_auroc_ovr(y_true: np.ndarray, P: np.ndarray, classes: np.ndarray) -> float:
    try:
        y_bin = np.zeros((y_true.size, classes.size), dtype=int)
        for i, c in enumerate(classes):
            y_bin[:, i] = (y_true == c).astype(int)
        aucs = []
        for i in range(classes.size):
            if np.unique(y_bin[:, i]).size < 2:
                continue
            aucs.append(roc_auc_score(y_bin[:, i], P[:, i]))
        return float(np.mean(aucs)) if aucs else float("nan")
    except (ValueError, IndexError):
        return float("nan")


def _pr_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, p))


def _event_from_prob_trace(p: np.ndarray, thr: float) -> np.ndarray:
    """Return predicted onset indices where prob crosses threshold from below."""
    above = p >= float(thr)
    rising = np.where((~above[:-1]) & (above[1:]))[0] + 1
    return rising.astype(int)


def _match_events(true_idx: np.ndarray, pred_idx: np.ndarray, *, tol_bins: int) -> Tuple[int, int, int, np.ndarray]:
    """Greedy event matching within ±tol_bins; returns TP, FP, FN, latencies."""
    true_idx = np.asarray(true_idx, int)
    pred_idx = np.asarray(pred_idx, int)
    used_pred = np.zeros(pred_idx.size, dtype=bool)
    latencies: List[int] = []
    TP = 0
    for t in true_idx:
        # Find unmatched predictions within window; pick nearest
        m = np.where((~used_pred) & (np.abs(pred_idx - t) <= tol_bins))[0]
        if m.size:
            j = m[np.argmin(np.abs(pred_idx[m] - t))]
            used_pred[j] = True
            TP += 1
            latencies.append(int(pred_idx[j] - t))
    FP = int((~used_pred).sum())
    FN = int(true_idx.size - TP)
    return TP, FP, FN, np.array(latencies, int)


# ---------------------------------------------------------------------------
# Grid and bank helpers
# ---------------------------------------------------------------------------


def _grid_history() -> List[float]:
    return [0.5, 1.0, 1.5]


def _grid_lags_ms(lags_ms: np.ndarray, *, quick: bool) -> np.ndarray:
    xs = np.asarray(lags_ms, float)
    if quick:
        # downsample by 2 for speed
        return xs[::2] if xs.size > 1 else xs
    return xs


def _subsample_units(fr: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    U = fr.shape[1]
    if n >= U:
        return np.arange(U, dtype=int)
    return rng.choice(U, size=int(n), replace=False)


def _prepare_X_bank(fr_d: Optional[np.ndarray], fr_v: Optional[np.ndarray], bank: str) -> Tuple[np.ndarray, List[int], List[int]]:
    """Return FR[T,U], unit_indices_d, unit_indices_v used for features for a bank."""
    if bank == "dorsal":
        if fr_d is None:
            raise ValueError("No dorsal spikes present")
        return fr_d, list(range(fr_d.shape[1])), []
    if bank == "ventral":
        if fr_v is None:
            raise ValueError("No ventral spikes present")
        return fr_v, [], list(range(fr_v.shape[1]))
    # both
    if fr_d is None and fr_v is None:
        raise ValueError("No spikes present for either bank")
    if fr_d is None:
        return fr_v, [], list(range(fr_v.shape[1]))
    if fr_v is None:
        return fr_d, list(range(fr_d.shape[1])), []
    return np.hstack([fr_d, fr_v]), list(range(fr_d.shape[1])), list(range(fr_v.shape[1]))


def _weights_to_unit_lag(W: np.ndarray, U: int, H: int) -> np.ndarray:
    """Map flat coef vector (U*H,) to (U,H) with recent-first lag order."""
    if W.ndim > 1:
        w = W.reshape(-1)
    else:
        w = W
    return np.abs(w.reshape(H, U).T)
