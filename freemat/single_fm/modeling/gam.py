from __future__ import annotations

"""GAM modeling utilities for Single-FM Figure 2.

Implements a compact Gaussian GAM with cubic regression splines per predictor,
blocked cross-validation with gaps, a lag scan, and basic nulls for continuous
predictors via block permutation or within-block scrambling.

Designed for use from the NB04 notebook. Keeps dependencies light; requires
`patsy` for spline design.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import warnings


@dataclass
class CVConfig:
    folds: int = 5
    block_s: float = 60.0
    gap_s: float = 10.0


def build_spline_design_patsy(X: np.ndarray, df: int) -> np.ndarray:
    try:
        import patsy  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("patsy is required for spline design; pip install patsy") from e
    X = np.asarray(X, dtype=float)
    T, D = X.shape
    cols: List[np.ndarray] = []
    for j in range(D):
        d = {"x": X[:, j]}
        B = patsy.dmatrix(f"cr(x, df={int(df)}) - 1", d, return_type="dataframe").values
        cols.append(B)
    return np.concatenate(cols, axis=1) if cols else np.zeros((T, 0), float)


def build_spline_design(X: np.ndarray, df: int) -> np.ndarray:
    """Alias to `build_spline_design_patsy` for notebook parity.

    Args:
        X: Predictor matrix (T, D)
        df: Degrees of freedom per covariate (cubic regression spline)

    Returns:
        Design matrix with concatenated spline bases.
    """
    return build_spline_design_patsy(X, df)


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float).reshape(-1)
    Xte = np.asarray(Xte, float)
    if Xtr.size == 0:
        return np.zeros((Xte.shape[0],), float)
    XtX = Xtr.T @ Xtr
    aI = float(alpha) * np.eye(XtX.shape[0])
    w = np.linalg.solve(XtX + aI, Xtr.T @ ytr)
    return Xte @ w


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    mu = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mu) ** 2)) or 1.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def apply_lag(X: np.ndarray, lag_bins: int) -> np.ndarray:
    if lag_bins == 0:
        return X
    T, D = X.shape
    Z = np.zeros((T, D), float)
    if lag_bins > 0:
        Z[lag_bins:, :] = X[: T - lag_bins, :]
    else:
        k = -lag_bins
        Z[: T - k, :] = X[k:, :]
    return Z


def lag_bins_list(max_ms: int, bin_s: float) -> List[int]:
    step = int(round(max(bin_s * 1000.0, 1.0)))
    L = int(round(max_ms / step))
    return [int(k) for k in range(-L, L + 1)]


def inner_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    alphas: Iterable[float],
) -> float:
    best_a = None
    best_r = -np.inf
    for a in alphas:
        r_sum = 0.0
        n = 0
        for fd in folds:
            tr, te = fd["train"], fd["test"]
            if tr.size == 0 or te.size == 0:
                continue
            yhat = ridge_fit_predict(X[tr], y[tr], X[te], float(a))
            r_sum += r2_score(y[te], yhat)
            n += 1
        r = r_sum / max(n, 1)
        if r > best_r:
            best_r = r
            best_a = float(a)
    return float(best_a if best_a is not None else list(alphas)[0])


def outer_cv_best_lag(
    X_base: np.ndarray,
    y: np.ndarray,
    folds_outer: List[Dict[str, np.ndarray]],
    folds_inner: List[Dict[str, np.ndarray]],
    lag_list: List[int],
    alphas: Iterable[float],
) -> Tuple[float, int]:
    # For each lag, pick alpha on inner CV using training folds; evaluate on test
    r2_by_lag: Dict[int, List[float]] = {lag: [] for lag in lag_list}
    # Cache lagged designs once (independent of folds)
    X_lag_cache: Dict[int, np.ndarray] = {lag: apply_lag(X_base, lag) for lag in lag_list}
    for fd in folds_outer:
        tr, te = fd["train"], fd["test"]
        if tr.size == 0 or te.size == 0:
            continue
        for lag in lag_list:
            X_lag = X_lag_cache[lag]
            a_star = inner_alpha_cv(X_lag[tr], y[tr], folds_inner, alphas)
            yhat = ridge_fit_predict(X_lag[tr], y[tr], X_lag[te], a_star)
            r2_by_lag[lag].append(r2_score(y[te], yhat))
    # Aggregate and select best
    means = {lag: (np.mean(v) if len(v) else -np.inf) for lag, v in r2_by_lag.items()}
    best_lag = max(means.keys(), key=lambda k: means[k])
    return float(means[best_lag]), int(best_lag)


def evaluate_family(
    X_family: np.ndarray,
    y: np.ndarray,
    bin_s: float,
    cv_cfg: CVConfig,
    lag_ms: int,
    alphas: Iterable[float],
) -> Tuple[float, int]:
    folds_outer = make_block_folds(y.shape[0], bin_s=bin_s, cfg=cv_cfg)
    # Use a coarser inner fold (3-fold) but same block scheme
    inner_cfg = CVConfig(folds=3, block_s=cv_cfg.block_s, gap_s=cv_cfg.gap_s)
    folds_inner = make_block_folds(y.shape[0], bin_s=bin_s, cfg=inner_cfg)
    lags = lag_bins_list(lag_ms, bin_s)
    return outer_cv_best_lag(X_family, y, folds_outer, folds_inner, lags, alphas)


def make_block_folds(
    T: int,
    *,
    bin_s: float,
    cfg: Optional[CVConfig] = None,
    folds: Optional[int] = None,
    block_s: Optional[float] = None,
    gap_s: Optional[float] = None,
) -> List[Dict[str, np.ndarray]]:
    """Convenience wrapper to build contiguous CV folds with exclusion gaps.

    Accepts either a `CVConfig` or explicit `folds/block_s/gap_s` (for
    compatibility with notebook-style calls).

    Args:
        T: Number of time points
        bin_s: Bin width in seconds
        cfg: Optional CVConfig
        folds: Number of folds (overrides cfg if provided)
        block_s: Block length in seconds (overrides cfg if provided)
        gap_s: Gap length around test blocks in seconds (overrides cfg if provided)

    Returns:
        List of dicts with 'train' and 'test' index arrays
    """
    base = cfg or CVConfig()
    if folds is not None:
        base.folds = int(folds)
    if block_s is not None:
        base.block_s = float(block_s)
    if gap_s is not None:
        base.gap_s = float(gap_s)
    return _make_block_folds_impl(T, bin_s=bin_s, cfg=base)


def _make_block_folds_impl(T: int, *, bin_s: float, cfg: CVConfig) -> List[Dict[str, np.ndarray]]:
    B = max(1, int(round(cfg.block_s / bin_s)))
    G = max(0, int(round(cfg.gap_s / bin_s)))
    n_blocks = T // B
    idx_blocks = np.arange(n_blocks)
    assign = idx_blocks % cfg.folds
    folds_out: List[Dict[str, np.ndarray]] = []
    for f in range(cfg.folds):
        test_blocks = idx_blocks[assign == f]
        test_idx = []
        for b in test_blocks:
            s = b * B
            e = min((b + 1) * B, T)
            test_idx.append(np.arange(s, e, dtype=int))
        test_idx = np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=int)
        mask = np.ones(T, dtype=bool)
        for b in test_blocks:
            s = max(0, b * B - G)
            e = min(T, (b + 1) * B + G)
            mask[s:e] = False
        train_idx = np.where(mask)[0]
        folds_out.append({"train": train_idx, "test": test_idx})
    return folds_out


def family_null_r2(
    X_family: np.ndarray,
    y: np.ndarray,
    bin_s: float,
    cv_cfg: CVConfig,
    lag_ms: int,
    alphas: Iterable[float],
    method: str,
    block_s: float,
    n_null: int = 100,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    from freemat_data import (
        block_permutation_indices,
        within_block_scramble_indices,
        apply_permutation_to_columns,
    )

    if rng is None:
        rng = np.random.RandomState(0)
    T = X_family.shape[0]
    B = max(1, int(round(block_s / bin_s)))
    # Warn if null block length in bins is shorter than max tested lag
    lag_bins = lag_bins_list(lag_ms, bin_s)
    max_abs_lag = max(abs(k) for k in lag_bins) if lag_bins else 0
    if B < max_abs_lag:
        warnings.warn(
            f"family_null_r2: block_len (bins) {B} < max tested lag {max_abs_lag}; nulls may leak temporal structure",
            RuntimeWarning,
        )
    r2 = np.zeros((n_null,), float)
    for i in range(n_null):
        if method == "block_permute":
            idx = block_permutation_indices(T, block_len=B, rng=rng)
        elif method == "within_block_scramble":
            idx = within_block_scramble_indices(T, block_len=B, rng=rng)
        else:
            raise ValueError(f"Unknown null method: {method}")
        Xp = apply_permutation_to_columns(X_family, idx)
        r, _ = evaluate_family(Xp, y, bin_s, cv_cfg, lag_ms, alphas)
        r2[i] = r
    return r2


def empirical_p(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, float)
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    """Column-wise standardization (mean 0, std 1; NaN-safe)."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd <= 0, 1.0, sd)
    return (X - mu) / sd


def design_from_df(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    df_spline: int = 6,
    standardize: bool = True,
) -> np.ndarray:
    """Build a spline design matrix from DataFrame columns.

    Args:
        df: Input DataFrame with predictor columns
        cols: Column names to include (order preserved)
        df_spline: Degrees of freedom per covariate for spline basis
        standardize: If True, z-score each covariate before spline expansion

    Returns:
        Design matrix (T, sum_j basis_j)
    """
    if not cols:
        return np.zeros((len(df), 0), float)
    X = df.loc[:, list(cols)].to_numpy(dtype=float)
    if standardize:
        X = _standardize_columns(X)
    return build_spline_design_patsy(X, int(df_spline))


def evaluate_families_from_df(
    df: pd.DataFrame,
    *,
    y_col: str,
    kin_cols: Sequence[str] = (),
    vis_cols: Sequence[str] = (),
    bin_s: float,
    cv_cfg: Optional[CVConfig] = None,
    lag_ms: int = 1000,
    alphas: Iterable[float] = (0.1, 1.0, 10.0),
    df_spline: int = 6,
    standardize: bool = True,
) -> Dict[str, Tuple[float, int]]:
    """Evaluate KIN, VIS, and FULL families for a single response vector.

    Drops rows with NaNs across selected columns and returns best R² and lag
    per family. Suitable for per-unit loops in notebooks.

    Args:
        df: DataFrame with response and predictor columns
        y_col: Response column name (e.g., 'fr_z')
        kin_cols: KIN predictor columns
        vis_cols: VIS predictor columns
        bin_s: Bin width in seconds
        cv_cfg: CV configuration (defaults to CVConfig())
        lag_ms: Max absolute lag to scan (ms)
        alphas: Ridge alpha grid for inner CV
        df_spline: Degrees of freedom per covariate
        standardize: Standardize predictors before spline expansion

    Returns:
        Dict family -> (r2, best_lag)
    """
    cols_all = list(kin_cols) + list(vis_cols)
    used = [c for c in [y_col, *cols_all] if c in df.columns]
    if not used or y_col not in df.columns:
        return {"KIN": (np.nan, 0), "VIS": (np.nan, 0), "FULL": (np.nan, 0)}
    dd = df.loc[:, used].copy()
    dd = dd.dropna(axis=0, how="any")
    if dd.empty:
        return {"KIN": (np.nan, 0), "VIS": (np.nan, 0), "FULL": (np.nan, 0)}
    y = dd[y_col].to_numpy(dtype=float).reshape(-1)
    cfg = cv_cfg or CVConfig()
    out: Dict[str, Tuple[float, int]] = {}
    Xk = design_from_df(dd, kin_cols, df_spline=df_spline, standardize=standardize) if kin_cols else None
    Xv = design_from_df(dd, vis_cols, df_spline=df_spline, standardize=standardize) if vis_cols else None
    if kin_cols:
        out["KIN"] = evaluate_family(Xk, y, bin_s=bin_s, cv_cfg=cfg, lag_ms=lag_ms, alphas=alphas)  # type: ignore[arg-type]
    else:
        out["KIN"] = (np.nan, 0)
    if vis_cols:
        out["VIS"] = evaluate_family(Xv, y, bin_s=bin_s, cv_cfg=cfg, lag_ms=lag_ms, alphas=alphas)  # type: ignore[arg-type]
    else:
        out["VIS"] = (np.nan, 0)
    if kin_cols or vis_cols:
        parts = []
        if Xk is not None:
            parts.append(Xk)
        if Xv is not None:
            parts.append(Xv)
        Xf = np.hstack(parts) if parts else np.zeros((len(dd), 0))
        out["FULL"] = evaluate_family(Xf, y, bin_s=bin_s, cv_cfg=cfg, lag_ms=lag_ms, alphas=alphas)
    else:
        out["FULL"] = (np.nan, 0)
    return out


__all__ = [
    "CVConfig",
    "make_block_folds",
    "build_spline_design_patsy",
    "build_spline_design",
    "ridge_fit_predict",
    "r2_score",
    "apply_lag",
    "evaluate_family",
    "family_null_r2",
    "empirical_p",
    "design_from_df",
    "evaluate_families_from_df",
]
