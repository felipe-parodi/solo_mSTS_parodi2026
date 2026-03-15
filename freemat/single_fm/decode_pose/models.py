"""Decoding models, metrics, and cross-validation for pose decoding.

Contains the lagged design matrix builder, chronological cross-validation,
Ridge/PLS model training, nested hyperparameter selection, and evaluation
metrics.  All functions preserve exact numeric behaviour from the original
monolithic ``decode_pose.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Run-level configuration replacing module-level globals.

    Attributes:
        k_pca: Number of PCA components to retain for POSE_PCS.
        null_perm: Number of null-model permutations (0 to skip).
        seed: Random seed for reproducibility.
        bin_s: Bin width in seconds for spike binning.
        dtype: NumPy dtype used for X/Y arrays.
        model: Model type, ``"ridge"`` or ``"pls"``.
        pls_grid: Grid of PLS component counts to search over.
    """

    k_pca: int = 10
    null_perm: int = 0
    seed: int = 0
    bin_s: float = 0.025
    dtype: type = field(default_factory=lambda: np.float64)
    model: str = "ridge"
    pls_grid: np.ndarray = field(
        default_factory=lambda: np.array([2, 4, 8, 12], dtype=int),
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Container for cross-validated model results.

    Attributes:
        y_true: Concatenated ground-truth values across folds.
        y_pred: Concatenated predictions across folds.
        r2_per_dim: Per-dimension R² scores.
        overall_r2: Scalar R² across all dimensions.
        mpjpe_val: Mean per-joint position error (optional).
        r_per_dim: Per-dimension Pearson correlations (optional).
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    r2_per_dim: np.ndarray
    overall_r2: float
    mpjpe_val: Optional[float] = None
    r_per_dim: Optional[np.ndarray] = None


EPS = 1e-8


# ---------------------------------------------------------------------------
# Z-scoring helpers
# ---------------------------------------------------------------------------


def _zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for z-scoring (NaN-safe).

    Args:
        X: Input array of shape ``(T, D)``.

    Returns:
        ``(mu, sd)`` where zero-variance dimensions get ``sd = 1.0``.
    """
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < EPS] = 1.0
    return mu, sd


def _zscore_apply(
    X: np.ndarray, mu: np.ndarray, sd: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed z-score parameters.

    Args:
        X: Input array of shape ``(T, D)``.
        mu: Mean array of shape ``(D,)``.
        sd: Std array of shape ``(D,)``.

    Returns:
        Z-scored array with same shape as *X*.
    """
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------


def build_lagged_design(fr_tu: np.ndarray, history_bins: int) -> np.ndarray:
    """Build causal history design ``(T_eff, U*H)`` using stride tricks.

    Args:
        fr_tu: Firing-rate array of shape ``(T, U)``.
        history_bins: Number of causal bins *H* to include.

    Returns:
        Design matrix of shape ``(T - H + 1, U * H)``, most recent first.

    Raises:
        ValueError: If *history_bins* is not in ``[1, T]``.
    """
    T, U = fr_tu.shape
    H = int(history_bins)
    if H <= 0 or H > T:
        raise ValueError("history_bins must be in [1, T]")
    s0, s1 = fr_tu.strides
    Z = as_strided(fr_tu, shape=(T - H + 1, H, U), strides=(s0, s0, s1))
    Zr = Z[:, ::-1, :]  # recent first
    return Zr.reshape(T - H + 1, U * H)


def compact_edges_head_body13() -> List[Tuple[int, int]]:
    """Edge list for compact 13-joint skeleton.

    Joint order: HEAD, LSh, RSh, LElb, RElb, LWri, RWri,
    LHip, RHip, LKnee, RKnee, LAnk, RAnk.
    """
    H, LSh, RSh, LElb, RElb, LWri, RWri = range(7)
    LHip, RHip, LKnee, RKnee, LAnk, RAnk = range(7, 13)
    return [
        (H, LSh), (H, RSh),
        (LSh, RSh),
        (LSh, LElb), (LElb, LWri),
        (RSh, RElb), (RElb, RWri),
        (LSh, LHip), (RSh, RHip),
        (LHip, RHip),
        (LHip, LKnee), (LKnee, LAnk),
        (RHip, RKnee), (RKnee, RAnk),
    ]


def apply_lag(
    y: np.ndarray, lag_bins: int,
) -> Tuple[np.ndarray, slice, slice]:
    """Shift targets by lag, keeping only overlapping time.

    Positive lag means pose follows neural (neural leads).

    Args:
        y: Target array of shape ``(T, ...)``.
        lag_bins: Integer lag in bins.

    Returns:
        ``(y_lagged, x_slice, y_slice)`` to align with X from
        :func:`build_lagged_design`.
    """
    L = int(lag_bins)
    if L == 0:
        return y, slice(None), slice(None)
    if L > 0:
        # y shifted forward (drop first L)
        return y[L:], slice(None, -L), slice(None)
    # L < 0: y shifted backward (drop last -L)
    return y[:L], slice(-L, None), slice(None)


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------


def chronological_folds(
    T: int, k: int, purge: int,
) -> List[Dict[str, np.ndarray]]:
    """Generate chronological folds with purge windows.

    Args:
        T: Total number of time points.
        k: Number of folds.
        purge: Purge window in bins around test boundaries.

    Returns:
        List of dicts with ``'train'`` and ``'test'`` index arrays.
    """
    k = int(k)
    fold_sizes = [T // k + (1 if i < (T % k) else 0) for i in range(k)]
    idx = np.arange(T)
    folds: List[Dict[str, np.ndarray]] = []
    start = 0
    for fs in fold_sizes:
        test = idx[start : start + fs]
        start += fs
        left = max(test[0] - purge, 0)
        right = min(test[-1] + purge, T - 1)
        mask = np.ones(T, dtype=bool)
        mask[left : right + 1] = False
        train = idx[mask]
        folds.append({"train": train, "test": test})
    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall R² score (NaN-safe, multi-output)."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ss_res = np.nansum((yt - yp) ** 2)
    mu = np.nanmean(yt, axis=0)
    ss_tot = np.nansum((yt - mu) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return max(1.0 - ss_res / ss_tot, -np.inf)


def per_dim_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-dimension R² scores (NaN-safe)."""
    mu = np.nanmean(y_true, axis=0)
    ss_res = np.nansum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.nansum((y_true - mu) ** 2, axis=0)
    out = np.full(y_true.shape[1], np.nan, float)
    m = ss_tot > 0
    out[m] = 1.0 - ss_res[m] / ss_tot[m]
    return out


def mpjpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-joint position error in 3-D."""
    T = y_true.shape[0]
    if T == 0:
        return float("nan")
    d = y_true - y_pred
    Jm1 = y_true.shape[1] // 3
    d3 = d.reshape(T, Jm1, 3)
    return float(np.nanmean(np.linalg.norm(d3, axis=2)))


def pearson_r_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-dimension Pearson correlation (NaN-safe, vectorised)."""
    yt = y_true - np.nanmean(y_true, axis=0, keepdims=True)
    yp = y_pred - np.nanmean(y_pred, axis=0, keepdims=True)
    num = np.nansum(yt * yp, axis=0)
    den = np.sqrt(np.nansum(yt ** 2, axis=0) * np.nansum(yp ** 2, axis=0))
    r = np.full(yt.shape[1], np.nan, float)
    m = den > 0
    r[m] = num[m] / den[m]
    return r


# ---------------------------------------------------------------------------
# Nested cross-validation: Ridge alpha + lag
# ---------------------------------------------------------------------------


def nested_select_alpha_lag(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray,
    lags_bins: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
) -> Tuple[float, int, Dict[str, np.ndarray]]:
    """Select global Ridge alpha and lag via nested CV with Gram caching.

    Args:
        X: Design matrix of shape ``(T, P)``.
        Y: Target matrix of shape ``(T, D)``.
        alphas: Ridge regularisation strengths to search.
        lags_bins: Integer lags (in bins) to search.
        folds: Fold dicts with ``'train'`` and ``'test'`` index arrays.

    Returns:
        ``(best_alpha, best_lag, results_by_lag)`` where *results_by_lag*
        maps lag string keys to arrays of mean R² per alpha.
    """
    results_by_lag: Dict[str, np.ndarray] = {}
    best_r2 = -np.inf
    best_alpha = float(alphas[0])
    best_lag = int(lags_bins[0])
    I_cache = None

    for L in lags_bins.astype(int):
        YL, x_sl, _ = apply_lag(Y, L)
        XH = X[x_sl]
        T_eff = XH.shape[0]

        # Per-fold caches
        fold_caches = []
        for f in folds:
            tr = f["train"]
            te = f["test"]
            tr = tr[(tr >= 0) & (tr < T_eff)]
            te = te[(te >= 0) & (te < T_eff)]
            if tr.size == 0 or te.size == 0:
                continue
            mu_x, sd_x = _zscore_fit(XH[tr])
            Xtr = _zscore_apply(XH[tr], mu_x, sd_x)
            Xte = _zscore_apply(XH[te], mu_x, sd_x)
            mu_y, sd_y = _zscore_fit(YL[tr])
            Ytr = _zscore_apply(YL[tr], mu_y, sd_y)
            Yte = _zscore_apply(YL[te], mu_y, sd_y)
            XtX = Xtr.T @ Xtr
            XtY = Xtr.T @ Ytr
            fold_caches.append({
                "XtX": XtX, "XtY": XtY, "Xte": Xte, "Yte": Yte,
            })

        lag_r2: List[float] = []
        for a in alphas:
            r2_vals = []
            for cache in fold_caches:
                XtX = cache["XtX"]
                XtY = cache["XtY"]
                Xte = cache["Xte"]
                Yte = cache["Yte"]
                if I_cache is None or I_cache.shape[0] != XtX.shape[0]:
                    I_cache = np.eye(XtX.shape[0], dtype=XtX.dtype)
                A = XtX + float(a) * I_cache
                W = np.linalg.solve(A, XtY)
                Yph = Xte @ W
                r2_vals.append(r2_score(Yte, Yph))
            r2_arr = np.array(r2_vals, float)
            mr2 = float(np.nanmean(r2_arr)) if r2_arr.size else -np.inf
            lag_r2.append(mr2)
            if mr2 > best_r2:
                best_r2 = mr2
                best_alpha = float(a)
                best_lag = int(L)
        results_by_lag[str(int(L))] = np.array(lag_r2, float)

    return best_alpha, best_lag, results_by_lag


# ---------------------------------------------------------------------------
# Nested cross-validation: PLS ncomp + lag
# ---------------------------------------------------------------------------


def nested_select_lag_pls(
    X: np.ndarray,
    Y: np.ndarray,
    ncomp_grid: np.ndarray,
    lags_bins: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
) -> Tuple[int, int, Dict[str, np.ndarray]]:
    """Select global PLS n_components and lag via nested CV.

    Args:
        X: Design matrix of shape ``(T, P)``.
        Y: Target matrix of shape ``(T, D)``.
        ncomp_grid: PLS component counts to search.
        lags_bins: Integer lags (in bins) to search.
        folds: Fold dicts with ``'train'`` and ``'test'`` index arrays.

    Returns:
        ``(best_k, best_lag, results_by_lag)``.
    """
    results_by_lag: Dict[str, np.ndarray] = {}
    best_r2 = -np.inf
    best_k = int(ncomp_grid[0])
    best_lag = int(lags_bins[0])

    for L in lags_bins.astype(int):
        YL, x_sl, _ = apply_lag(Y, L)
        XH = X[x_sl]
        T_eff = XH.shape[0]
        lag_r2: List[float] = []
        for k in ncomp_grid.astype(int):
            r2_vals = []
            for f in folds:
                tr = f["train"]
                te = f["test"]
                tr = tr[(tr >= 0) & (tr < T_eff)]
                te = te[(te >= 0) & (te < T_eff)]
                if tr.size == 0 or te.size == 0:
                    continue
                mu_x, sd_x = _zscore_fit(XH[tr])
                Xtr = _zscore_apply(XH[tr], mu_x, sd_x)
                Xte = _zscore_apply(XH[te], mu_x, sd_x)
                mu_y, sd_y = _zscore_fit(YL[tr])
                Ytr = _zscore_apply(YL[tr], mu_y, sd_y)
                Yte = _zscore_apply(YL[te], mu_y, sd_y)
                pls = PLSRegression(n_components=int(k), scale=False)
                pls.fit(Xtr, Ytr)
                Yph = pls.predict(Xte)
                r2_vals.append(r2_score(Yte, Yph))
            r2_arr = np.array(r2_vals, float)
            mr2 = float(np.nanmean(r2_arr)) if r2_arr.size else -np.inf
            lag_r2.append(mr2)
            if mr2 > best_r2:
                best_r2 = mr2
                best_k = int(k)
                best_lag = int(L)
        results_by_lag[str(int(L))] = np.array(lag_r2, float)

    return best_k, best_lag, results_by_lag


# ---------------------------------------------------------------------------
# Train / evaluate (Ridge)
# ---------------------------------------------------------------------------


def train_evaluate(
    X: np.ndarray,
    Y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    alpha: float,
    lag_bins: int,
) -> TrainResult:
    """Fit Ridge model across CV folds and return concatenated results.

    Args:
        X: Design matrix of shape ``(T, P)``.
        Y: Target matrix of shape ``(T, D)``.
        folds: Fold dicts with ``'train'`` and ``'test'`` index arrays.
        alpha: Ridge regularisation strength.
        lag_bins: Integer lag applied to the target.

    Returns:
        :class:`TrainResult` with aggregated predictions and metrics.
    """
    YL, x_sl, _ = apply_lag(Y, int(lag_bins))
    XH = X[x_sl]
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    for f in folds:
        tr = f["train"]
        te = f["test"]
        tr = tr[(tr >= 0) & (tr < XH.shape[0])]
        te = te[(te >= 0) & (te < XH.shape[0])]
        if tr.size == 0 or te.size == 0:
            continue
        scaler_x = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler_x.fit_transform(XH[tr])
        Xte = scaler_x.transform(XH[te])
        scaler_y = StandardScaler(with_mean=True, with_std=True)
        Ytr = scaler_y.fit_transform(YL[tr])
        Yte = scaler_y.transform(YL[te])
        mdl = Ridge(alpha=float(alpha), fit_intercept=False)
        mdl.fit(Xtr, Ytr)
        Yph = mdl.predict(Xte)
        y_true_all.append(Yte)
        y_pred_all.append(Yph)
    y_true = np.vstack(y_true_all) if y_true_all else np.empty((0, Y.shape[1]))
    y_pred = np.vstack(y_pred_all) if y_pred_all else np.empty((0, Y.shape[1]))
    r2_dim = per_dim_r2(y_true, y_pred) if y_true.size else np.array([])
    return TrainResult(
        y_true=y_true,
        y_pred=y_pred,
        r2_per_dim=r2_dim,
        overall_r2=r2_score(y_true, y_pred),
    )


# ---------------------------------------------------------------------------
# Train / evaluate (PLS)
# ---------------------------------------------------------------------------


def train_evaluate_pls(
    X: np.ndarray,
    Y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    n_components: int,
    lag_bins: int,
) -> TrainResult:
    """Fit PLS model across CV folds and return concatenated results.

    Args:
        X: Design matrix of shape ``(T, P)``.
        Y: Target matrix of shape ``(T, D)``.
        folds: Fold dicts with ``'train'`` and ``'test'`` index arrays.
        n_components: Number of PLS components.
        lag_bins: Integer lag applied to the target.

    Returns:
        :class:`TrainResult` with aggregated predictions and metrics.
    """
    YL, x_sl, _ = apply_lag(Y, int(lag_bins))
    XH = X[x_sl]
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    for f in folds:
        tr = f["train"]
        te = f["test"]
        tr = tr[(tr >= 0) & (tr < XH.shape[0])]
        te = te[(te >= 0) & (te < XH.shape[0])]
        if tr.size == 0 or te.size == 0:
            continue
        mu_x, sd_x = _zscore_fit(XH[tr])
        Xtr = _zscore_apply(XH[tr], mu_x, sd_x)
        Xte = _zscore_apply(XH[te], mu_x, sd_x)
        mu_y, sd_y = _zscore_fit(YL[tr])
        Ytr = _zscore_apply(YL[tr], mu_y, sd_y)
        Yte = _zscore_apply(YL[te], mu_y, sd_y)
        pls = PLSRegression(n_components=int(n_components), scale=False)
        pls.fit(Xtr, Ytr)
        Yph = pls.predict(Xte)
        y_true_all.append(Yte)
        y_pred_all.append(Yph)
    y_true = np.vstack(y_true_all) if y_true_all else np.empty((0, Y.shape[1]))
    y_pred = np.vstack(y_pred_all) if y_pred_all else np.empty((0, Y.shape[1]))
    r2_dim = per_dim_r2(y_true, y_pred) if y_true.size else np.array([])
    return TrainResult(
        y_true=y_true,
        y_pred=y_pred,
        r2_per_dim=r2_dim,
        overall_r2=r2_score(y_true, y_pred),
    )
