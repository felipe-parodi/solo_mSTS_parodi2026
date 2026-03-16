"""Electrode gradient statistics for Figure 4 motif tuning metrics.

Computes depth, anterior-posterior, and bank gradient tests for
selectivity and bandwidth at the electrode level. Includes FDR
correction across the test family (2 metrics x 4 primary tests = 8).

Inference is per-session only (no cross-session pooling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

# Probe geometry — defined locally to avoid circular imports from viz.
# Semantics match _PROBE_AP_UM in freemat/single_fm/viz/fig4_panels.py:
#   A2/B2 → anterior (0 µm), A1/B1 → posterior (800 µm).
_PROBE_AP_POSITION: dict[str, str] = {
    "A1": "posterior",
    "A2": "anterior",
    "B1": "posterior",
    "B2": "anterior",
}

_PROBE_BANK: dict[str, str] = {
    "A1": "ventral",
    "A2": "ventral",
    "B1": "dorsal",
    "B2": "dorsal",
}

# Minimum sample size for each test arm
_MIN_N = 3


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DepthResult:
    """Spearman correlation result for depth gradient."""

    rho: float
    p_value: float
    n: int
    metric: str
    bank: Optional[str]


@dataclass
class APResult:
    """Mann-Whitney U result for anterior vs posterior."""

    u_stat: float
    p_value: float
    rank_biserial: float
    direction: str
    n_anterior: int
    n_posterior: int
    median_anterior: float
    median_posterior: float
    metric: str
    bank: str


@dataclass
class BankResult:
    """Mann-Whitney U result for ventral vs dorsal."""

    u_stat: float
    p_value: float
    rank_biserial: float
    direction: str
    n_ventral: int
    n_dorsal: int
    median_ventral: float
    median_dorsal: float
    metric: str


# ---------------------------------------------------------------------------
# Per-unit metrics
# ---------------------------------------------------------------------------

def compute_per_unit_metrics(
    tuning_df: pd.DataFrame,
    motif_order: list[int],
) -> pd.DataFrame:
    """Compute selectivity and bandwidth per unit from tuning z-scores.

    Args:
        tuning_df: DataFrame with motif_{id}_z columns, plus probe,
            shank, depth_position, bank columns.
        motif_order: Motif IDs whose z-score columns must exist.

    Returns:
        Copy of tuning_df with added selectivity and bandwidth columns.

    Raises:
        ValueError: If any motif_{id}_z column is missing.
    """
    z_cols = []
    for mid in motif_order:
        col = f"motif_{mid}_z"
        if col not in tuning_df.columns:
            msg = f"Missing column {col} in tuning_df"
            raise ValueError(msg)
        z_cols.append(col)

    result = tuning_df.copy()
    z_mat = result[z_cols].values
    result["selectivity"] = np.nanmax(np.abs(z_mat), axis=1)
    result["bandwidth"] = np.nansum(np.abs(z_mat) > 1.0, axis=1).astype(float)
    return result


# ---------------------------------------------------------------------------
# Electrode-level aggregation
# ---------------------------------------------------------------------------

def compute_electrode_metrics(unit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-unit metrics to electrode level (mean per site).

    Args:
        unit_df: DataFrame from compute_per_unit_metrics with selectivity,
            bandwidth, probe, shank, depth_position, bank columns.

    Returns:
        DataFrame with one row per electrode site, columns:
        probe, shank, depth_position, bank, selectivity, bandwidth, n_units.
    """
    group_keys = ["probe", "shank", "depth_position", "bank"]
    agg = unit_df.groupby(group_keys).agg(
        selectivity=("selectivity", "mean"),
        bandwidth=("bandwidth", "mean"),
        n_units=("selectivity", "size"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# Rank-biserial effect size
# ---------------------------------------------------------------------------

def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U.

    Args:
        u_stat: Mann-Whitney U statistic.
        n1: Size of first group.
        n2: Size of second group.

    Returns:
        Rank-biserial r in [-1, 1].
    """
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


# ---------------------------------------------------------------------------
# Gradient test functions
# ---------------------------------------------------------------------------

def compute_depth_gradient(
    df: pd.DataFrame,
    metric: str,
    bank: Optional[str] = None,
) -> DepthResult:
    """Spearman correlation between metric and depth position.

    Args:
        df: Electrode-level DataFrame with depth_position and metric columns.
        metric: Column name to test (selectivity or bandwidth).
        bank: Optional bank filter (ventral or dorsal). None = pooled.

    Returns:
        DepthResult with Spearman rho, p-value, and n.
    """
    subset = df if bank is None else df[df["bank"] == bank]
    values = subset[metric].dropna()
    depths = subset.loc[values.index, "depth_position"]

    if len(values) < _MIN_N:
        return DepthResult(rho=np.nan, p_value=np.nan, n=len(values),
                           metric=metric, bank=bank)

    rho, p = spearmanr(depths.values, values.values)
    return DepthResult(rho=float(rho), p_value=float(p), n=len(values),
                       metric=metric, bank=bank)


def compute_ap_gradient(
    df: pd.DataFrame,
    metric: str,
    bank: str,
) -> APResult:
    """Mann-Whitney U for anterior vs posterior within a bank.

    Args:
        df: Electrode-level DataFrame with probe, bank, and metric columns.
        metric: Column name to test.
        bank: Which bank to test within (ventral or dorsal).

    Returns:
        APResult with U statistic, p-value, rank-biserial, and direction.
    """
    bank_df = df[df["bank"] == bank].copy()

    # Map probes to AP position
    bank_df["ap_position"] = bank_df["probe"].map(_PROBE_AP_POSITION)

    anterior = bank_df.loc[bank_df["ap_position"] == "anterior", metric].dropna()
    posterior = bank_df.loc[bank_df["ap_position"] == "posterior", metric].dropna()

    if len(anterior) < _MIN_N or len(posterior) < _MIN_N:
        return APResult(
            u_stat=np.nan, p_value=np.nan, rank_biserial=np.nan,
            direction="insufficient_n",
            n_anterior=len(anterior), n_posterior=len(posterior),
            median_anterior=np.nan, median_posterior=np.nan,
            metric=metric, bank=bank,
        )

    u_stat, p = mannwhitneyu(anterior.values, posterior.values, alternative="two-sided")
    r = _rank_biserial(u_stat, len(anterior), len(posterior))
    med_a = float(np.median(anterior))
    med_p = float(np.median(posterior))
    direction = "anterior>posterior" if med_a > med_p else "posterior>anterior"

    return APResult(
        u_stat=float(u_stat), p_value=float(p), rank_biserial=float(r),
        direction=direction,
        n_anterior=len(anterior), n_posterior=len(posterior),
        median_anterior=med_a, median_posterior=med_p,
        metric=metric, bank=bank,
    )


def compute_bank_comparison(
    df: pd.DataFrame,
    metric: str,
) -> BankResult:
    """Mann-Whitney U for ventral vs dorsal banks.

    Args:
        df: Electrode-level DataFrame with bank and metric columns.
        metric: Column name to test.

    Returns:
        BankResult with U statistic, p-value, rank-biserial, and direction.
    """
    ventral = df.loc[df["bank"] == "ventral", metric].dropna()
    dorsal = df.loc[df["bank"] == "dorsal", metric].dropna()

    if len(ventral) < _MIN_N or len(dorsal) < _MIN_N:
        return BankResult(
            u_stat=np.nan, p_value=np.nan, rank_biserial=np.nan,
            direction="insufficient_n",
            n_ventral=len(ventral), n_dorsal=len(dorsal),
            median_ventral=np.nan, median_dorsal=np.nan,
            metric=metric,
        )

    u_stat, p = mannwhitneyu(ventral.values, dorsal.values, alternative="two-sided")
    r = _rank_biserial(u_stat, len(ventral), len(dorsal))
    med_v = float(np.median(ventral))
    med_d = float(np.median(dorsal))
    direction = "V>D" if med_v > med_d else "D>V"

    return BankResult(
        u_stat=float(u_stat), p_value=float(p), rank_biserial=float(r),
        direction=direction,
        n_ventral=len(ventral), n_dorsal=len(dorsal),
        median_ventral=med_v, median_dorsal=med_d,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------

def apply_fdr_correction(
    summary_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to summary table.

    Only rows with fdr_family == "primary" are included in the
    correction. Excluded rows get NaN for p_fdr.

    Args:
        summary_df: DataFrame with p_value and fdr_family columns.
        alpha: FDR threshold.

    Returns:
        Copy of summary_df with added p_fdr and significant_fdr columns.
    """
    result = summary_df.copy()
    result["p_fdr"] = np.nan
    result["significant_fdr"] = False

    primary_mask = result["fdr_family"] == "primary"
    if not primary_mask.any():
        return result

    p_vals = result.loc[primary_mask, "p_value"].values
    n_tests = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, n_tests + 1)

    # BH adjusted p-values
    p_adj = p_vals * n_tests / ranks
    # Enforce monotonicity (step-up)
    p_adj_sorted = p_adj[sorted_idx]
    for i in range(n_tests - 2, -1, -1):
        p_adj_sorted[i + 1] = min(p_adj_sorted[i + 1], 1.0)
        p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i + 1])
    p_adj_sorted[0] = min(p_adj_sorted[0], 1.0)
    p_adj_final = np.empty_like(p_adj)
    p_adj_final[sorted_idx] = p_adj_sorted

    result.loc[primary_mask, "p_fdr"] = p_adj_final
    result.loc[primary_mask, "significant_fdr"] = p_adj_final < alpha
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def compute_all_gradient_stats(
    tuning_df: pd.DataFrame,
    motif_order: list[int],
) -> pd.DataFrame:
    """Run all gradient tests for a single session.

    Tests selectivity and bandwidth along bank, depth, and AP axes.
    Returns a summary DataFrame with FDR correction applied.

    Args:
        tuning_df: Per-unit tuning DataFrame with session_id column.
        motif_order: Motif IDs for z-score columns.

    Returns:
        Summary DataFrame with one row per test, including p_value,
        p_fdr, effect_size, direction, and sample sizes.

    Raises:
        ValueError: If tuning_df contains multiple session_ids.
    """
    if "session_id" in tuning_df.columns:
        n_sessions = tuning_df["session_id"].nunique()
        if n_sessions > 1:
            msg = f"compute_all_gradient_stats requires a single session, got {n_sessions}"
            raise ValueError(msg)

    unit_df = compute_per_unit_metrics(tuning_df, motif_order)
    elec_df = compute_electrode_metrics(unit_df)

    rows: list[dict] = []
    metrics = ["selectivity", "bandwidth"]

    for metric in metrics:
        # 1. Bank comparison (primary FDR)
        bank_res = compute_bank_comparison(elec_df, metric)
        rows.append({
            "metric": metric,
            "test": "bank",
            "statistic": bank_res.u_stat,
            "p_value": bank_res.p_value,
            "effect_size": bank_res.rank_biserial,
            "direction": bank_res.direction,
            "n1": bank_res.n_ventral,
            "n2": bank_res.n_dorsal,
            "fdr_family": "primary" if not np.isnan(bank_res.p_value) else "excluded_n<3",
        })

        # 2. Depth pooled (primary FDR)
        depth_res = compute_depth_gradient(elec_df, metric)
        rows.append({
            "metric": metric,
            "test": "depth",
            "statistic": depth_res.rho,
            "p_value": depth_res.p_value,
            "effect_size": depth_res.rho,
            "direction": "positive" if depth_res.rho > 0 else "negative",
            "n1": depth_res.n,
            "n2": np.nan,
            "fdr_family": "primary" if not np.isnan(depth_res.p_value) else "excluded_n<3",
        })

        # 3. AP ventral (primary FDR)
        ap_v = compute_ap_gradient(elec_df, metric, bank="ventral")
        rows.append({
            "metric": metric,
            "test": "ap_ventral",
            "statistic": ap_v.u_stat,
            "p_value": ap_v.p_value,
            "effect_size": ap_v.rank_biserial,
            "direction": ap_v.direction,
            "n1": ap_v.n_anterior,
            "n2": ap_v.n_posterior,
            "fdr_family": "primary" if not np.isnan(ap_v.p_value) else "excluded_n<3",
        })

        # 4. AP dorsal (primary FDR)
        ap_d = compute_ap_gradient(elec_df, metric, bank="dorsal")
        rows.append({
            "metric": metric,
            "test": "ap_dorsal",
            "statistic": ap_d.u_stat,
            "p_value": ap_d.p_value,
            "effect_size": ap_d.rank_biserial,
            "direction": ap_d.direction,
            "n1": ap_d.n_anterior,
            "n2": ap_d.n_posterior,
            "fdr_family": "primary" if not np.isnan(ap_d.p_value) else "excluded_n<3",
        })

        # 5. Depth ventral (exploratory)
        depth_v = compute_depth_gradient(elec_df, metric, bank="ventral")
        rows.append({
            "metric": metric,
            "test": "depth_ventral",
            "statistic": depth_v.rho,
            "p_value": depth_v.p_value,
            "effect_size": depth_v.rho,
            "direction": "positive" if depth_v.rho > 0 else "negative",
            "n1": depth_v.n,
            "n2": np.nan,
            "fdr_family": "exploratory",
        })

        # 6. Depth dorsal (exploratory)
        depth_d = compute_depth_gradient(elec_df, metric, bank="dorsal")
        rows.append({
            "metric": metric,
            "test": "depth_dorsal",
            "statistic": depth_d.rho,
            "p_value": depth_d.p_value,
            "effect_size": depth_d.rho,
            "direction": "positive" if depth_d.rho > 0 else "negative",
            "n1": depth_d.n,
            "n2": np.nan,
            "fdr_family": "exploratory",
        })

    summary = pd.DataFrame(rows)
    summary = apply_fdr_correction(summary)
    return summary
