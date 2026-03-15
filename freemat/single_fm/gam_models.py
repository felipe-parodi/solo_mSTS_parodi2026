"""GAM modeling for Step-A (looking vs movement).

Prefers pyGAM if available; falls back to statsmodels OLS with B-splines for
nonlinear terms. Applies leakage-safe per-session scaling within folds and
optionally winsorizes smooth terms based on train percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from freemat_data.cv import SessionScaler, blocked_folds

try:  # optional
    from pygam import LinearGAM, s
    _HAVE_PYGAM = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYGAM = False

import statsmodels.api as sm
from patsy import dmatrix


def _cv_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nan-safe cross-validated R^2 with variance floor.

    Returns NaN when target variance is effectively zero to avoid
    uninformative, arbitrarily large negative R^2 values.
    """
    var = float(np.nanvar(y_true))
    # Use a slightly stricter floor than 1e-12 to avoid pathological ratios
    if not np.isfinite(var) or var < 1e-8:
        return float("nan")
    mse = float(np.nanmean((y_true - y_pred) ** 2))
    return 1.0 - mse / (var + 1e-12)


def _winsorize_pair(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str], lo: float = 0.01, hi: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cols:
        return train, test
    qs = train[cols].quantile([lo, hi])
    loq = qs.loc[lo].to_numpy(dtype=float)
    hiq = qs.loc[hi].to_numpy(dtype=float)
    tr = train.copy()
    te = test.copy()
    Xtr = tr[cols].to_numpy(float)
    Xte = te[cols].to_numpy(float)
    Xtr = np.clip(Xtr, loq, hiq)
    Xte = np.clip(Xte, loq, hiq)
    tr.loc[:, cols] = Xtr
    te.loc[:, cols] = Xte
    return tr, te


def _build_formula(colnames: Sequence[str], smooth_cols: Sequence[int]) -> str:
    terms = []
    for i, c in enumerate(colnames):
        if i in set(smooth_cols):
            terms.append(f"bs({c}, df=7, degree=3)")
        else:
            terms.append(c)
    return " + ".join(terms)


def fit_cv_linear_gam(
    X: np.ndarray,
    y: np.ndarray,
    sess: np.ndarray,
    folds: np.ndarray,
    smooth_cols: Sequence[int],
    *,
    max_train_rows: int | None = None,
    solver: str = "auto",
) -> tuple[np.ndarray, object | None]:
    """Cross-validated linear-GAM fitting with leakage-safe scaling.

    Args:
        X: Predictor matrix (N, D).
        y: Target vector (N,).
        sess: Session IDs aligned to rows of X.
        folds: Fold IDs in [0, K) aligned to rows of X.
        smooth_cols: Indices of columns in X to model with smooth functions;
            other columns are linear.

    Returns:
        (yhat, gam_obj) where `yhat` are CV predictions and `gam_obj` is a
        lightweight fitted model object (last fold) or None.
    """
    N, D = X.shape
    yhat = np.full(N, np.nan, dtype=float)
    kfolds = int(np.max(folds) + 1) if len(folds) else 0
    colnames = [f"x{i}" for i in range(D)]
    gam_obj: object | None = None

    # DataFrame wrapper to use SessionScaler and patsy design
    DF = pd.DataFrame({c: X[:, j] for j, c in enumerate(colnames)})
    DF["session_id"] = sess
    DF["y"] = y

    for k in range(kfolds):
        tr = folds != k
        te = ~tr
        if te.sum() == 0 or tr.sum() == 0:
            continue
        DFtr = DF.loc[tr, :].copy()
        DFte = DF.loc[te, :].copy()

        # Leakage-safe scaling per session: fit on train, apply to train/test
        ss = SessionScaler(by="session_id").fit(DFtr, cols=colnames)
        DFtr = ss.transform(DFtr)
        DFte = ss.transform(DFte)

        # Winsorize smooth terms based on train percentiles
        smooth_names = [colnames[i] for i in smooth_cols]
        DFtr, DFte = _winsorize_pair(DFtr, DFte, smooth_names, lo=0.01, hi=0.99)

        # Prefer pyGAM unless caller overrides solver
        if solver == "auto" and _HAVE_PYGAM:
            # pyGAM design: build separate matrices for linear/smooth
            from pygam import LinearGAM, s

            # Build a concatenated X for GAM; pyGAM uses indices directly
            Xtr = DFtr[colnames].to_numpy(float)
            Xte = DFte[colnames].to_numpy(float)
            terms = None
            for i in range(D):
                if i in set(smooth_cols):
                    terms = (s(i) if terms is None else terms + s(i))
            gam = LinearGAM(terms).fit(Xtr, DFtr["y"].to_numpy(float))
            yhat[te] = gam.predict(Xte)
            # Keep last fold model for plots
            gam_obj = {
                "backend": "pyGAM",
                "model": gam,
                "colnames": colnames,
                "smooth_cols": list(smooth_cols),
            }
        else:
            # statsmodels OLS with B-splines via patsy
            formula = _build_formula(colnames, smooth_cols)
            # Optional subsample of train rows for speed
            if isinstance(max_train_rows, int) and max_train_rows > 0 and len(DFtr) > max_train_rows:
                # stratified by session to preserve per-session distribution
                parts = []
                per_sess = max(1, max_train_rows // max(1, DFtr["session_id"].nunique()))
                rng = np.random.default_rng(0)
                for sid, gg in DFtr.groupby("session_id", sort=False):
                    if len(gg) <= per_sess:
                        parts.append(gg)
                    else:
                        idx = rng.choice(gg.index.to_numpy(), size=per_sess, replace=False)
                        parts.append(DFtr.loc[idx])
                DFtr_sub = pd.concat(parts, axis=0).sort_index()
            else:
                DFtr_sub = DFtr

            Xtr = dmatrix(formula, DFtr_sub, return_type="dataframe")
            Xte = dmatrix(formula, DFte, return_type="dataframe")
            ytr = DFtr_sub["y"].to_numpy(float)

            if solver == "lstsq":
                # Fast least-squares solve without statsmodels overhead
                beta, *_ = np.linalg.lstsq(Xtr.to_numpy(float), ytr, rcond=None)
                yhat[te] = (Xte.to_numpy(float) @ beta)
                gam_obj = {
                    "backend": "lstsq",
                    "model": None,
                    "formula": formula,
                    "colnames": colnames,
                    "smooth_cols": list(smooth_cols),
                }
            else:
                mod = sm.OLS(ytr, Xtr).fit()
                yhat[te] = mod.predict(Xte)
                gam_obj = {
                    "backend": "statsmodels",
                    "model": mod,
                    "formula": formula,
                    "colnames": colnames,
                    "smooth_cols": list(smooth_cols),
                }

    return yhat, gam_obj

def _prepare_folds(T: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    tdf = T[["session_id", "t_bin_s"]].copy()
    return blocked_folds(tdf, n_folds=int(n_folds)).to_numpy(int)


class BatchDesignCache:
    """Reusable per-fold design for fast OLS predictions across many targets.

    Builds patsy design matrices once per fold (with leakage-safe per-session
    scaling and winsorization on smooth terms) and caches X_tr, X_te and the
    Cholesky factor of X_tr^T X_tr for efficient predictions on different y.
    """

    def __init__(self, folds: np.ndarray, colnames: Sequence[str], smooth_cols: Sequence[int]) -> None:
        self.folds = folds.astype(int)
        self.colnames = list(colnames)
        self.smooth_cols = list(smooth_cols)
        self.formula = _build_formula(self.colnames, self.smooth_cols)
        # Per-fold caches
        self.idx_tr: list[np.ndarray] = []
        self.idx_te: list[np.ndarray] = []
        self.Xtr: list[np.ndarray] = []
        self.Xte: list[np.ndarray] = []
        self.chol: list[np.ndarray] = []

    def add_fold(self, idx_tr: np.ndarray, idx_te: np.ndarray, Xtr: np.ndarray, Xte: np.ndarray) -> None:
        self.idx_tr.append(idx_tr.astype(int))
        self.idx_te.append(idx_te.astype(int))
        self.Xtr.append(np.asarray(Xtr))
        self.Xte.append(np.asarray(Xte))
        XtX = (self.Xtr[-1].T @ self.Xtr[-1]).astype(np.float64)
        eps = 1e-8 * np.eye(XtX.shape[0], dtype=np.float64)
        self.chol.append(np.linalg.cholesky(XtX + eps))

    def predict(self, y: np.ndarray, N: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        yhat = np.full(N, np.nan, dtype=float)
        for k in range(len(self.idx_tr)):
            ytr = y[self.idx_tr[k]]
            rhs = self.Xtr[k].T @ ytr
            v = np.linalg.solve(self.chol[k].T, np.linalg.solve(self.chol[k], rhs))
            yhat[self.idx_te[k]] = self.Xte[k] @ v
        return yhat


def build_batch_cache(T: pd.DataFrame, cols: Sequence[str], smooth_cols: Sequence[int], n_folds: int = 5, *, dtype32: bool = True) -> BatchDesignCache:
    """Construct a reusable per-fold design cache for batch OLS predictions.

    Args:
        T: Table with predictors, 'session_id', and 't_bin_s'.
        cols: Predictor column names in T in the training order.
        smooth_cols: Indices within `cols` to treat as smooth (B-splines).
        n_folds: Number of blocked CV folds.
        dtype32: If True, store design matrices as float32 to save memory.
    """
    folds = _prepare_folds(T, n_folds=int(n_folds))
    colnames = [f"x{i}" for i in range(len(cols))]
    DF = pd.DataFrame({c: T[k].to_numpy(float) for c, k in zip(colnames, cols)})
    DF["session_id"] = T["session_id"].to_numpy()
    cache = BatchDesignCache(folds=folds, colnames=colnames, smooth_cols=smooth_cols)
    for k in range(int(n_folds)):
        tr = folds != k
        te = ~tr
        DFtr = DF.loc[tr, :].copy()
        DFte = DF.loc[te, :].copy()
        ss = SessionScaler(by="session_id").fit(DFtr, cols=colnames)
        DFtr = ss.transform(DFtr)
        DFte = ss.transform(DFte)
        s_names = [colnames[i] for i in smooth_cols]
        DFtr, DFte = _winsorize_pair(DFtr, DFte, s_names, lo=0.01, hi=0.99)
        Xtr = dmatrix(cache.formula, DFtr, return_type="dataframe").to_numpy(np.float32 if dtype32 else np.float64)
        Xte = dmatrix(cache.formula, DFte, return_type="dataframe").to_numpy(np.float32 if dtype32 else np.float64)
        cache.add_fold(np.where(tr)[0], np.where(te)[0], Xtr, Xte)
    return cache


def run_M1_look_still(T: pd.DataFrame, s0_quantile: float = 0.15, *, max_train_rows: int | None = None, solver: str = "auto", n_folds: int = 5, batch_cache: BatchDesignCache | None = None) -> dict:
    """M1: Look tuning while still (speed below s0 per session).

    Predictors: Fourier(theta_look) + smooth(yaw_rate).
    """
    keep_mask = np.zeros(len(T), dtype=bool)
    for sid, g in T.groupby("session_id", sort=False):
        sp = g["speed"].to_numpy(float)
        sp_fin = sp[np.isfinite(sp)]
        if sp_fin.size == 0:
            continue
        s0 = float(np.quantile(sp_fin, s0_quantile))
        m = (T.index.isin(g.index)) & (T["speed"].to_numpy(float) < s0)
        keep_mask |= m
    G = T.loc[keep_mask].copy()
    # Per-model NaN mask: only the predictors used by M1 (Fourier + yaw_rate).
    cols_m1 = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "yaw_rate"]
    if not G.empty:
        m_fin = np.ones(len(G), dtype=bool)
        for c in cols_m1:
            if c in G.columns:
                m_fin &= np.isfinite(G[c].to_numpy(float))
        G = G.loc[m_fin].copy()
    if len(G) < 10:
        return {"metrics": {"r2_look_still": float("nan")}, "per_fold": {}}

    folds = _prepare_folds(G, n_folds=int(n_folds))
    y = G["fr"].to_numpy(float)
    sess = G["session_id"].to_numpy()
    cols_four = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"]
    X = G[cols_four + ["yaw_rate"]].to_numpy(float)
    smooth_cols = [X.shape[1] - 1]  # yaw_rate only
    if solver == "batch" and batch_cache is not None:
        yhat = batch_cache.predict(y, N=len(G))
        gam_obj = {"backend": "batch"}
    else:
        yhat, gam_obj = fit_cv_linear_gam(X, y, sess, folds, smooth_cols, max_train_rows=max_train_rows, solver=solver)
    r2 = _cv_r2(y, yhat)
    return {
        "metrics": {"r2_look_still": float(r2)},
        "per_fold": {"folds": folds, "yhat": yhat, "idx": G.index.to_numpy()},
        "model": gam_obj,
    }


def run_M2_move_unique(T: pd.DataFrame, *, max_train_rows: int | None = None, solver: str = "auto", n_folds: int = 5, batch_cache_look: BatchDesignCache | None = None, batch_cache_full: BatchDesignCache | None = None) -> dict:
    """M2: Movement unique variance controlling for look.

    LOOK baseline: Fourier(theta_look) + smooth(yaw_rate)
    MOVE+LOOK:     + smooth(v_parallel, v_perp, v_z)
    """
    # Per-model NaN mask: use the intersection rows where FULL model predictors are finite.
    cols_four = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"]
    cols_full = cols_four + ["yaw_rate", "v_parallel", "v_perp", "v_z"]
    if not T.empty:
        m_fin = np.ones(len(T), dtype=bool)
        for c in cols_full:
            if c in T.columns:
                m_fin &= np.isfinite(T[c].to_numpy(float))
        T = T.loc[m_fin].copy()
    folds = _prepare_folds(T, n_folds=int(n_folds))
    y = T["fr"].to_numpy(float)
    sess = T["session_id"].to_numpy()
    Xlook = T[cols_four + ["yaw_rate"]].to_numpy(float)
    smooth_cols_look = [Xlook.shape[1] - 1]
    if solver == "batch" and batch_cache_look is not None:
        yhatL = batch_cache_look.predict(y, N=len(T))
    else:
        yhatL, _ = fit_cv_linear_gam(Xlook, y, sess, folds, smooth_cols_look, max_train_rows=max_train_rows, solver=solver)
    r2_look_all = _cv_r2(y, yhatL)

    Xfull = T[cols_full].to_numpy(float)
    smooth_cols_full = list(range(len(cols_four), Xfull.shape[1]))  # smooth kinetic + yaw_rate
    if solver == "batch" and batch_cache_full is not None:
        yhatF = batch_cache_full.predict(y, N=len(T))
        gam_obj = {"backend": "batch"}
    else:
        yhatF, gam_obj = fit_cv_linear_gam(Xfull, y, sess, folds, smooth_cols_full, max_train_rows=max_train_rows, solver=solver)
    r2_full = _cv_r2(y, yhatF)

    move_unique = float(r2_full - r2_look_all) if np.isfinite(r2_full) and np.isfinite(r2_look_all) else float("nan")
    return {
        "metrics": {
            "r2_move_plus_look": float(r2_full),
            "r2_look_all": float(r2_look_all),
            "move_unique": move_unique,
        },
        "per_fold": {"folds": folds, "yhat_full": yhatF, "yhat_look": yhatL, "idx": T.index.to_numpy()},
        "model": gam_obj,
    }


def run_M3_alignment(T: pd.DataFrame, *, max_train_rows: int | None = None, solver: str = "auto", n_folds: int = 5, batch_cache: BatchDesignCache | None = None) -> dict:
    """M3: Alignment model with smooth(speed, align_cos, yaw_rate)."""
    # Per-model NaN mask
    cols_m3 = ["speed", "align_cos", "yaw_rate"]
    if not T.empty:
        m_fin = np.ones(len(T), dtype=bool)
        for c in cols_m3:
            if c in T.columns:
                m_fin &= np.isfinite(T[c].to_numpy(float))
        T = T.loc[m_fin].copy()
    folds = _prepare_folds(T, n_folds=int(n_folds))
    y = T["fr"].to_numpy(float)
    sess = T["session_id"].to_numpy()
    X = T[cols_m3].to_numpy(float)
    smooth_cols = [0, 1, 2]
    if solver == "batch" and batch_cache is not None:
        yhat = batch_cache.predict(y, N=len(T))
        gam_obj = {"backend": "batch"}
    else:
        yhat, gam_obj = fit_cv_linear_gam(X, y, sess, folds, smooth_cols, max_train_rows=max_train_rows, solver=solver)
    r2 = _cv_r2(y, yhat)
    return {
        "metrics": {"r2_align": float(r2)},
        "per_fold": {"folds": folds, "yhat": yhat, "idx": T.index.to_numpy()},
        "model": gam_obj,
    }


__all__ = [
    "fit_cv_linear_gam",
    "BatchDesignCache",
    "build_batch_cache",
    "run_M1_look_still",
    "run_M2_move_unique",
    "run_M3_alignment",
]
