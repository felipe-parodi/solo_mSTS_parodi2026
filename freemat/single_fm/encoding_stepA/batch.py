"""Batch-mode scoring and permutation tests for Step-A encoding.

Extracts the batch OLS cache building, global prediction, per-unit scoring,
and permutation test loop from the monolithic ``encoding_stepA.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from freemat.single_fm.gam_models import build_batch_cache
from freemat.single_fm.gam_models import _cv_r2 as _safe_r2



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    x: np.ndarray, n: int = 1000, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the median.

    Args:
        x: 1D array of values (NaN/Inf filtered internally).
        n: Number of bootstrap resamples.
        alpha: Significance level for CI (default 0.05 → 95% CI).

    Returns:
        Tuple of (median, lower_bound, upper_bound).
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    med = float(np.nanmedian(x))
    rng = np.random.default_rng(0)
    boots = np.array(
        [np.nanmedian(rng.choice(x, size=x.size, replace=True)) for _ in range(n)]
    )
    lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return med, lo, hi


def _console_summary(results: pd.DataFrame) -> None:
    """Print population-level summary of Step-A metrics to console.

    Args:
        results: DataFrame with per-unit metrics (``r2_look_still``,
            ``move_unique``, ``r2_align``, ``bank``, optionally
            ``sig_look_still``).
    """
    # Prefer permutation-based significance if present
    if "sig_look_still" in results.columns:
        tuned = results["sig_look_still"].astype(float)
        by_bank = results.groupby("bank")["sig_look_still"].mean().astype(float)
    else:
        tuned = (results["r2_look_still"] > 0.01).astype(float)
        by_bank = results.groupby("bank")["r2_look_still"].apply(
            lambda s: float(np.nanmean(s > 0.01))
        )
    overall = float(np.nanmean(tuned))
    r2L = _bootstrap_ci(results["r2_look_still"].to_numpy(float))
    muv = _bootstrap_ci(results["move_unique"].to_numpy(float))
    r2A = _bootstrap_ci(results["r2_align"].to_numpy(float))
    print(f"Look-tuned (still) % units: {overall:.3f}")
    if not by_bank.empty:
        print("By bank:")
        for b, p in by_bank.items():
            print(f"  {b}: {p:.3f}")
    print(
        f"Median [95% CI] r2_look_still: {r2L[0]:.3f} [{r2L[1]:.3f}, {r2L[2]:.3f}]"
    )
    print(f"Median [95% CI] move_unique: {muv[0]:.3f} [{muv[1]:.3f}, {muv[2]:.3f}]")
    print(f"Median [95% CI] r2_align: {r2A[0]:.3f} [{r2A[1]:.3f}, {r2A[2]:.3f}]")


# ---------------------------------------------------------------------------
# Batch cache data structures
# ---------------------------------------------------------------------------


@dataclass
class BatchCaches:
    """Container for precomputed batch OLS caches across model types."""

    M1: Optional[Any] = None
    L: Optional[Any] = None
    F: Optional[Any] = None
    M3: Optional[Any] = None
    Gstill: pd.DataFrame = field(default_factory=pd.DataFrame)
    T_LF: pd.DataFrame = field(default_factory=pd.DataFrame)
    T_M3: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Batch cache building
# ---------------------------------------------------------------------------


def build_stillness_subset(
    T: pd.DataFrame, s0q: float,
) -> pd.DataFrame:
    """Build the stillness-gated subset for M1 scoring.

    Args:
        T: Merged neural+feature table.
        s0q: Stillness threshold quantile per session.

    Returns:
        Filtered DataFrame containing only still frames.
    """
    keep_mask = np.zeros(len(T), dtype=bool)
    for _sid, g in T.groupby("session_id", sort=False):
        sp = g["speed"].to_numpy(float)
        sp_fin = sp[np.isfinite(sp)]
        if sp_fin.size == 0:
            continue
        s0 = float(np.quantile(sp_fin, float(s0q)))
        m = (T.index.isin(g.index)) & (T["speed"].to_numpy(float) < s0)
        keep_mask |= m
    return T.loc[keep_mask].copy()


def build_all_batch_caches(
    T: pd.DataFrame,
    *,
    s0q: float,
    n_folds: int,
) -> BatchCaches:
    """Build batch OLS caches for M1, M2 (L+F), and M3 models.

    Args:
        T: Merged neural+feature table.
        s0q: Stillness quantile for M1.
        n_folds: Number of CV folds.

    Returns:
        BatchCaches with all precomputed cache objects.
    """
    caches = BatchCaches()

    # M1: stillness subset
    Gstill = build_stillness_subset(T, s0q)
    cols_M1 = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "yaw_rate"]
    if not Gstill.empty:
        m1_fin = np.ones(len(Gstill), dtype=bool)
        for c in cols_M1:
            m1_fin &= np.isfinite(Gstill[c].to_numpy(float))
        Gstill = Gstill.loc[m1_fin].copy()
        if not Gstill.empty:
            caches.M1 = build_batch_cache(
                Gstill, cols_M1,
                smooth_cols=[len(cols_M1) - 1],
                n_folds=n_folds, dtype32=True,
            )
    caches.Gstill = Gstill

    # M2: look-only (L) and full (F) models
    cols_four = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"]
    cols_L = cols_four + ["yaw_rate"]
    cols_F = cols_four + ["yaw_rate", "v_parallel", "v_perp", "v_z"]
    m2_fin = np.ones(len(T), dtype=bool)
    for c in cols_F:
        m2_fin &= np.isfinite(T[c].to_numpy(float))
    T_LF = T.loc[m2_fin].copy()
    if not T_LF.empty:
        caches.L = build_batch_cache(
            T_LF, cols_L,
            smooth_cols=[len(cols_L) - 1],
            n_folds=n_folds, dtype32=True,
        )
        caches.F = build_batch_cache(
            T_LF, cols_F,
            smooth_cols=list(range(len(cols_four), len(cols_F))),
            n_folds=n_folds, dtype32=True,
        )
    caches.T_LF = T_LF

    # M3: alignment model
    cols_M3 = ["speed", "align_cos", "yaw_rate"]
    m3_fin = np.ones(len(T), dtype=bool)
    for c in cols_M3:
        m3_fin &= np.isfinite(T[c].to_numpy(float))
    T_M3 = T.loc[m3_fin].copy()
    if not T_M3.empty:
        caches.M3 = build_batch_cache(
            T_M3, cols_M3,
            smooth_cols=[0, 1, 2],
            n_folds=n_folds, dtype32=True,
        )
    caches.T_M3 = T_M3

    return caches


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def _compute_global_predictions(
    T: pd.DataFrame,
    caches: BatchCaches,
    *,
    s0q: float,
    n_folds: int,
) -> dict[str, Optional[np.ndarray]]:
    """Compute global cross-validated predictions for all model types.

    Args:
        T: Merged neural+feature table.
        caches: Precomputed batch caches.
        s0q: Stillness quantile for M1.
        n_folds: Number of CV folds.

    Returns:
        Dict with keys ``yhat_M1``, ``yhat_L``, ``yhat_F``, ``yhat_M3``
        (each an ndarray or None).
    """
    preds: dict[str, Optional[np.ndarray]] = {
        "yhat_M1": None, "yhat_L": None, "yhat_F": None, "yhat_M3": None,
    }

    # M1
    Gstill = caches.Gstill
    if not Gstill.empty:
        cols_M1 = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "yaw_rate"]
        y_still = Gstill["fr"].to_numpy(float)
        cache_M1 = build_batch_cache(
            Gstill, cols_M1,
            smooth_cols=[len(cols_M1) - 1],
            n_folds=n_folds, dtype32=True,
        )
        preds["yhat_M1"] = cache_M1.predict(y_still, N=len(Gstill))

    # M2
    T_LF = caches.T_LF
    if not T_LF.empty:
        cols_four = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"]
        cols_L = cols_four + ["yaw_rate"]
        cols_F = cols_four + ["yaw_rate", "v_parallel", "v_perp", "v_z"]
        y_all_LF = T_LF["fr"].to_numpy(float)
        cache_L = build_batch_cache(
            T_LF, cols_L,
            smooth_cols=[len(cols_L) - 1],
            n_folds=n_folds, dtype32=True,
        )
        cache_F = build_batch_cache(
            T_LF, cols_F,
            smooth_cols=list(range(len(cols_four), len(cols_F))),
            n_folds=n_folds, dtype32=True,
        )
        preds["yhat_L"] = cache_L.predict(y_all_LF, N=len(T_LF))
        preds["yhat_F"] = cache_F.predict(y_all_LF, N=len(T_LF))

    # M3
    T_M3 = caches.T_M3
    if not T_M3.empty:
        cols_M3 = ["speed", "align_cos", "yaw_rate"]
        y_all_M3 = T_M3["fr"].to_numpy(float)
        cache_M3 = build_batch_cache(
            T_M3, cols_M3,
            smooth_cols=[0, 1, 2],
            n_folds=n_folds, dtype32=True,
        )
        preds["yhat_M3"] = cache_M3.predict(y_all_M3, N=len(T_M3))

    return preds


def score_units_batch(
    T: pd.DataFrame,
    caches: BatchCaches,
    preds: dict[str, Optional[np.ndarray]],
) -> list[dict]:
    """Score each unit using precomputed global predictions.

    Args:
        T: Merged neural+feature table.
        caches: Batch caches (for Gstill, T_LF, T_M3 tables).
        preds: Global predictions from ``_compute_global_predictions``.

    Returns:
        List of per-unit metric dicts.
    """
    Gstill = caches.Gstill
    T_LF = caches.T_LF
    T_M3 = caches.T_M3
    yhat_M1 = preds["yhat_M1"]
    yhat_L = preds["yhat_L"]
    yhat_F = preds["yhat_F"]
    yhat_M3 = preds["yhat_M3"]

    y_still = Gstill["fr"].to_numpy(float) if not Gstill.empty else np.array([], dtype=float)

    records: list[dict] = []
    for uid, G in T.groupby("unit_id", sort=False):
        try:
            rec: dict = {"unit_id": uid, "bank": G["bank"].iloc[0]}
            # M1
            if not Gstill.empty and yhat_M1 is not None:
                m_u_still = Gstill["unit_id"].to_numpy() == uid
                if int(m_u_still.sum()) >= 10:
                    rec["r2_look_still"] = float(
                        _safe_r2(y_still[m_u_still], yhat_M1[m_u_still])
                    )
                else:
                    rec["r2_look_still"] = float("nan")
            else:
                rec["r2_look_still"] = float("nan")
            # M2
            if yhat_L is not None and yhat_F is not None and not T_LF.empty:
                m_u = T_LF["unit_id"].to_numpy() == uid
                if int(m_u.sum()) >= 10:
                    y_u = T_LF["fr"].to_numpy(float)[m_u]
                    r2L = float(_safe_r2(y_u, yhat_L[m_u]))
                    r2F = float(_safe_r2(y_u, yhat_F[m_u]))
                    rec["r2_look_all"] = r2L
                    rec["r2_move_plus_look"] = r2F
                    rec["move_unique"] = (
                        (r2F - r2L)
                        if np.isfinite(r2F) and np.isfinite(r2L)
                        else float("nan")
                    )
                else:
                    rec["r2_look_all"] = float("nan")
                    rec["r2_move_plus_look"] = float("nan")
                    rec["move_unique"] = float("nan")
            else:
                rec["r2_look_all"] = float("nan")
                rec["r2_move_plus_look"] = float("nan")
                rec["move_unique"] = float("nan")
            # M3
            if yhat_M3 is not None and not T_M3.empty:
                m_u3 = T_M3["unit_id"].to_numpy() == uid
                if int(m_u3.sum()) >= 10:
                    y_u3 = T_M3["fr"].to_numpy(float)[m_u3]
                    rec["r2_align"] = float(_safe_r2(y_u3, yhat_M3[m_u3]))
                else:
                    rec["r2_align"] = float("nan")
            else:
                rec["r2_align"] = float("nan")
            # Sanity checks
            if np.isfinite(rec.get("r2_move_plus_look", np.nan)) and np.isfinite(
                rec.get("r2_look_all", np.nan)
            ):
                if rec["r2_move_plus_look"] + 1e-9 < rec["r2_look_all"]:
                    logging.warning(
                        "Unit %s: r2_move_plus_look < r2_look_all (%.4f < %.4f)",
                        str(uid),
                        rec["r2_move_plus_look"],
                        rec["r2_look_all"],
                    )
            for k in ("r2_look_still", "r2_look_all", "r2_move_plus_look", "r2_align"):
                v = rec.get(k, np.nan)
                if np.isfinite(v) and abs(v) > 10.0:
                    logging.warning(
                        "Unit %s: extreme %s = %.4f (check variance and predictions)",
                        str(uid), k, v,
                    )
            records.append(rec)
        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            logging.exception("Unit %s failed (batch mode): %s", str(uid), e)
    return records


# ---------------------------------------------------------------------------
# Permutation loop
# ---------------------------------------------------------------------------


def _circular_shift_session(
    y: np.ndarray,
    idx_by_sess: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply per-session circular shift to a target vector.

    Args:
        y: Original target vector.
        idx_by_sess: Dict mapping session_id to index arrays.
        rng: Random generator.

    Returns:
        Permuted copy of y.
    """
    yperm = np.empty_like(y)
    for _sid, idxs in idx_by_sess.items():
        if idxs.size <= 1:
            yperm[idxs] = y[idxs]
        else:
            k = int(rng.integers(1, idxs.size))
            yperm[idxs] = y[idxs][np.roll(np.arange(idxs.size), k)]
    return yperm


def run_permutations_batch(
    T: pd.DataFrame,
    caches: BatchCaches,
    records: list[dict],
    *,
    perm_n: int,
    n_folds: int,
) -> dict[str, dict]:
    """Run permutation tests for M1, M2 (move_unique), and M3.

    Args:
        T: Merged neural+feature table.
        caches: Batch caches.
        records: Observed per-unit metric records from ``score_units_batch``.
        perm_n: Number of permutations.
        n_folds: Number of CV folds.

    Returns:
        Dict with keys ``'look_still'``, ``'move_unique'``, ``'align'``,
        each mapping unit_id to a p-value float.
    """
    rng = np.random.default_rng(0)
    Gstill = caches.Gstill
    T_LF = caches.T_LF
    T_M3 = caches.T_M3

    uid_list = sorted(T["unit_id"].unique().tolist())

    # Index helpers
    idx_S_by_uid: dict = {}
    if not Gstill.empty:
        uid_s = Gstill["unit_id"].to_numpy()
        for uid in uid_list:
            idx_S_by_uid[uid] = np.where(uid_s == uid)[0]

    # Observed metrics
    obs_L = {r["unit_id"]: r.get("r2_look_all", np.nan) for r in records}
    obs_F = {r["unit_id"]: r.get("r2_move_plus_look", np.nan) for r in records}
    obs_A = {r["unit_id"]: r.get("r2_align", np.nan) for r in records}
    obs_M1 = {r["unit_id"]: r.get("r2_look_still", np.nan) for r in records}

    # Counters
    cnt_M1 = {uid: 0 for uid in uid_list}
    cnt_L = {uid: 0 for uid in uid_list}
    cnt_F = {uid: 0 for uid in uid_list}
    cnt_A = {uid: 0 for uid in uid_list}
    cnt_MU = {uid: 0 for uid in uid_list}

    # Session grouping for circular shift
    idx_by_sess_LF: dict = {}
    if not T_LF.empty:
        sess_vals = T_LF["session_id"].to_numpy()
        idx_by_sess_LF = {
            sid: np.where(sess_vals == sid)[0] for sid in pd.unique(sess_vals)
        }
    idx_by_sess_M3: dict = {}
    if not T_M3.empty:
        sess_vals = T_M3["session_id"].to_numpy()
        idx_by_sess_M3 = {
            sid: np.where(sess_vals == sid)[0] for sid in pd.unique(sess_vals)
        }
    idx_by_sess_S: dict = {}
    if not Gstill.empty:
        sess_vals = Gstill["session_id"].to_numpy()
        idx_by_sess_S = {
            sid: np.where(sess_vals == sid)[0] for sid in pd.unique(sess_vals)
        }

    batch_M1 = caches.M1
    batch_L = caches.L
    batch_F = caches.F
    batch_M3 = caches.M3

    for b in range(perm_n):
        # Progress log every ~10%
        try:
            step = max(1, perm_n // 10)
            if (b % step) == 0:
                logging.info("[perm] %d/%d", b, perm_n)
        except (TypeError, ValueError):
            pass

        # M2 permutations on T_LF
        yperm_LF = yhatL_p = yhatF_p = None
        if not T_LF.empty and batch_L is not None and batch_F is not None:
            y_all_LF = T_LF["fr"].to_numpy(float)
            if idx_by_sess_LF:
                yperm_LF = _circular_shift_session(y_all_LF, idx_by_sess_LF, rng)
            else:
                yperm_LF = rng.permutation(y_all_LF)
            yhatL_p = batch_L.predict(yperm_LF, N=len(T_LF))
            yhatF_p = batch_F.predict(yperm_LF, N=len(T_LF))

        # M3 permutations on T_M3
        yperm_M3 = yhat3_p = None
        if not T_M3.empty and batch_M3 is not None:
            y_all_M3 = T_M3["fr"].to_numpy(float)
            if idx_by_sess_M3:
                yperm_M3 = _circular_shift_session(y_all_M3, idx_by_sess_M3, rng)
            else:
                yperm_M3 = rng.permutation(y_all_M3)
            yhat3_p = batch_M3.predict(yperm_M3, N=len(T_M3))

        # M1 permutations on Gstill
        yperm_S = yhat1_p = None
        if not Gstill.empty and batch_M1 is not None:
            y_still = Gstill["fr"].to_numpy(float)
            if idx_by_sess_S:
                yperm_S = _circular_shift_session(y_still, idx_by_sess_S, rng)
            else:
                yperm_S = rng.permutation(y_still)
            yhat1_p = batch_M1.predict(yperm_S, N=len(Gstill))

        # Score per uid
        for uid in uid_list:
            # M2
            if yperm_LF is not None and yhatL_p is not None and yhatF_p is not None and not T_LF.empty:
                idx_u = np.where(T_LF["unit_id"].to_numpy() == uid)[0]
                if idx_u.size >= 10:
                    r2L_p = _safe_r2(yperm_LF[idx_u], yhatL_p[idx_u])
                    r2F_p = _safe_r2(yperm_LF[idx_u], yhatF_p[idx_u])
                    if np.isfinite(obs_L.get(uid, np.nan)) and np.isfinite(r2L_p) and (r2L_p >= float(obs_L[uid])):
                        cnt_L[uid] += 1
                    if np.isfinite(obs_F.get(uid, np.nan)) and np.isfinite(r2F_p) and (r2F_p >= float(obs_F[uid])):
                        cnt_F[uid] += 1
                    if (
                        np.isfinite(obs_F.get(uid, np.nan))
                        and np.isfinite(obs_L.get(uid, np.nan))
                        and np.isfinite(r2F_p)
                        and np.isfinite(r2L_p)
                    ):
                        mu_p = r2F_p - r2L_p
                        mu_obs = float(obs_F[uid]) - float(obs_L[uid])
                        if mu_p >= mu_obs:
                            cnt_MU[uid] += 1
            # M3
            if yperm_M3 is not None and yhat3_p is not None and not T_M3.empty:
                idx_u3 = np.where(T_M3["unit_id"].to_numpy() == uid)[0]
                if idx_u3.size >= 10:
                    r2A_p = _safe_r2(yperm_M3[idx_u3], yhat3_p[idx_u3])
                    if np.isfinite(obs_A.get(uid, np.nan)) and np.isfinite(r2A_p) and (r2A_p >= float(obs_A[uid])):
                        cnt_A[uid] += 1
            # M1
            if yperm_S is not None and yhat1_p is not None and uid in idx_S_by_uid:
                idx_s = idx_S_by_uid[uid]
                if idx_s.size >= 10:
                    r2M1_p = _safe_r2(yperm_S[idx_s], yhat1_p[idx_s])
                    if np.isfinite(obs_M1.get(uid, np.nan)) and np.isfinite(r2M1_p) and (r2M1_p >= float(obs_M1[uid])):
                        cnt_M1[uid] += 1

    # Compute p-values
    denom = float(perm_n + 1)
    perm_pvals: dict[str, dict] = {"look_still": {}, "move_unique": {}, "align": {}}
    for uid in uid_list:
        perm_pvals["look_still"][uid] = (
            (1.0 + float(cnt_M1.get(uid, 0))) / denom
            if not Gstill.empty
            else float("nan")
        )
        perm_pvals["move_unique"][uid] = (1.0 + float(cnt_MU.get(uid, 0))) / denom
        perm_pvals["align"][uid] = (1.0 + float(cnt_A.get(uid, 0))) / denom
    return perm_pvals
