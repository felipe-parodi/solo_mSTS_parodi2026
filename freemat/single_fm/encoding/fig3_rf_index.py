"""Fig 3 RF-index: continuous reference-frame preference metric.

RF-index = (delta_r2_allo - delta_r2_body) / (|delta_r2_allo| + |delta_r2_body| + eps)

Bounded [-1, +1]:
  +1 = purely allocentric
  -1 = purely body-centric
   0 = equal contribution

Two modes:
1. Post-hoc (fast): compute rf_index from existing fig3_reference_frames.parquet
2. Height-stratified (slower): re-fit ridge per unit, score on height-restricted
   test subsets to get per-height RF-index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_EPS: float = 1e-6


def compute_rf_index(
    df: pd.DataFrame,
    *,
    eps: float = _DEFAULT_EPS,
) -> pd.DataFrame:
    """Add rf_index column to a DataFrame with delta_r2_allo and delta_r2_body.

    Args:
        df: DataFrame with columns 'delta_r2_allo' and 'delta_r2_body'.
        eps: Small constant to prevent division by zero.

    Returns:
        Copy of df with 'rf_index' column added.
    """
    dr2_allo = df["delta_r2_allo"].to_numpy(dtype=float)
    dr2_body = df["delta_r2_body"].to_numpy(dtype=float)

    numerator = dr2_allo - dr2_body
    denominator = np.abs(dr2_allo) + np.abs(dr2_body) + eps

    rf_index = numerator / denominator

    result = df.copy()
    result["rf_index"] = rf_index
    return result


def compute_height_stratified_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    height_tiers: np.ndarray,
    *,
    min_bins_per_tier: int = 20,
    min_var_eps: float = 1e-6,
) -> dict[int, float]:
    """Compute R² per height tier on test predictions.

    Args:
        y_true: True firing rates for test bins.
        y_pred: Predicted firing rates for test bins.
        height_tiers: Height tier labels (0=floor, 1=bench, 2=ceiling) for test bins.
        min_bins_per_tier: Minimum number of test bins per tier.
        min_var_eps: Minimum variance threshold for y_true in a tier.

    Returns:
        Dict mapping tier_id → R² (NaN if insufficient data or zero variance).
    """
    r2_per_tier: dict[int, float] = {}
    for tier in sorted(np.unique(height_tiers)):
        mask = height_tiers == tier
        n_bins = int(mask.sum())
        if n_bins < min_bins_per_tier:
            r2_per_tier[int(tier)] = float("nan")
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        if np.std(y_t) < min_var_eps:
            r2_per_tier[int(tier)] = float("nan")
            continue

        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_per_tier[int(tier)] = float(r2)

    return r2_per_tier


__all__ = [
    "compute_rf_index",
    "compute_height_stratified_r2",
]
