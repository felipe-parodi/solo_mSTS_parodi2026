"""Project-agnostic cross-validation helpers.

Provides contiguous, per-session blocked folds and a leakage-safe per-session
scaler for predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def blocked_folds(times_df: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    """Assign contiguous fold IDs per session using `t_bin_s` ordering.

    Args:
        times_df: DataFrame with columns `session_id` and `t_bin_s`.
        n_folds: Number of folds.

    Returns:
        A `pd.Series` of length `len(times_df)` with integer fold IDs in [0, n_folds).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if not {"session_id", "t_bin_s"}.issubset(times_df.columns):
        raise ValueError("times_df must contain 'session_id' and 't_bin_s'")
    df = times_df.reset_index(drop=True).copy()
    df = df.sort_values(["session_id", "t_bin_s"], kind="mergesort")
    folds = np.full(len(df), -1, dtype=int)
    for _, g in df.groupby("session_id", sort=False):
        idx = g.index.to_numpy()
        splits = np.array_split(idx, n_folds)
        for k, ids in enumerate(splits):
            folds[ids] = k
    return pd.Series(folds, index=df.index).sort_index()


@dataclass
class _Stats:
    mu: np.ndarray
    sd: np.ndarray


class SessionScaler:
    """Per-session z-scoring fitted on train only (leakage-safe).

    Example:
        >>> from freemat_data.cv import SessionScaler
        >>> ss = SessionScaler().fit(train_df, cols=['speed','yaw_rate'])
        >>> Xz_test = ss.transform(test_df)
    """

    def __init__(self, by: str = "session_id") -> None:
        self.by = str(by)
        self._stats: Dict[str, _Stats] = {}
        self._cols: list[str] = []

    def fit(self, df: pd.DataFrame, cols: Sequence[str]) -> "SessionScaler":
        """Fit per-session mean and std for given columns.

        Args:
            df: Training DataFrame containing `by` and `cols`.
            cols: Columns to scale.

        Returns:
            Self.

        Raises:
            KeyError: If required columns are missing.
        """
        if self.by not in df.columns:
            raise KeyError(f"missing grouping column '{self.by}'")
        cols = list(cols)
        for c in cols:
            if c not in df.columns:
                raise KeyError(f"missing column '{c}' for scaling")
        self._cols = cols
        self._stats.clear()
        for sid, g in df.groupby(self.by, sort=False):
            X = g[cols].to_numpy(dtype=float)
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            sd[sd < 1e-12] = 1.0
            self._stats[str(sid)] = _Stats(mu=mu, sd=sd)
        return self

    def transform(self, df: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.DataFrame:
        """Apply fitted per-session scaling to provided DataFrame.

        Args:
            df: DataFrame with a `by` column, to be transformed in-place on `cols`.
            cols: Optional subset of columns; defaults to the columns seen at fit.

        Returns:
            A new DataFrame with scaled columns.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if not self._stats:
            raise RuntimeError("SessionScaler.transform called before fit")
        cols = list(cols) if cols is not None else list(self._cols)
        out = df.copy()
        for sid, g in out.groupby(self.by, sort=False):
            stats = self._stats.get(str(sid))
            if stats is None:
                # For unseen sessions at transform-time, leave values as-is to avoid leakage.
                continue
            X = g[cols].to_numpy(dtype=float)
            Xz = (X - stats.mu) / stats.sd
            out.loc[g.index, cols] = Xz
        return out


__all__ = ["blocked_folds", "SessionScaler"]

