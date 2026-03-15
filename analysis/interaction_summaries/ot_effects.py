"""OT (oxytocin) effect analysis for interaction summaries.

Provides both simple group comparisons and linear mixed models (LMM)
for testing drug × bank interactions.

Treatment structure:
- OT sessions: One monkey dosed with oxytocin
- Saline sessions: Both monkeys receive saline
- Treatment labels per subject: 'OT', 'saline', 'undosed_partner', 'none'

Two approaches:
1. Simple: Between-group comparison with Cohen's d
2. LMM: metric ~ drug * bank + (1|session)

Usage:
    from analysis.interaction_summaries.ot_effects import (
        compute_ot_effect_simple,
        compute_ot_effect_lmm,
    )

    # Simple comparison
    result = compute_ot_effect_simple(
        ot_values=[0.65, 0.70, 0.68],
        saline_values=[0.55, 0.58, 0.52],
        metric_name='alignment',
    )

    # LMM (requires session-level data)
    result = compute_ot_effect_lmm(
        session_df=session_data,
        metric_col='alignment',
        drug_col='drug',
        bank_col='bank',
        session_col='session_id',
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OTEffectResult:
    """Result of OT effect analysis.

    Attributes:
        metric_name: Name of the metric being compared
        ot_mean: Mean value for OT condition
        ot_std: Std for OT condition
        ot_n: Number of OT samples
        saline_mean: Mean for saline condition
        saline_std: Std for saline condition
        saline_n: Number of saline samples
        difference: OT - saline difference
        cohens_d: Effect size (Cohen's d)
        p_value: P-value if available
        lmm_results: Optional LMM fit results
    """

    metric_name: str
    ot_mean: float
    ot_std: float
    ot_n: int
    saline_mean: float
    saline_std: float
    saline_n: int
    difference: float
    cohens_d: float
    p_value: Optional[float] = None
    lmm_results: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict[str, Any]:
        """Get summary as dict."""
        return {
            "metric": self.metric_name,
            "ot": {"mean": self.ot_mean, "std": self.ot_std, "n": self.ot_n},
            "saline": {"mean": self.saline_mean, "std": self.saline_std, "n": self.saline_n},
            "difference": self.difference,
            "cohens_d": self.cohens_d,
            "p_value": self.p_value,
        }


def compute_ot_effect_simple(
    ot_values: List[float],
    saline_values: List[float],
    metric_name: str = "metric",
) -> OTEffectResult:
    """Compute OT vs saline effect with simple statistics.

    Args:
        ot_values: List of metric values for OT sessions/units
        saline_values: List of metric values for saline sessions/units
        metric_name: Name of the metric

    Returns:
        OTEffectResult with effect statistics
    """
    # Filter NaN
    ot_arr = np.array([v for v in ot_values if not np.isnan(v)])
    sal_arr = np.array([v for v in saline_values if not np.isnan(v)])

    ot_mean = float(np.mean(ot_arr)) if len(ot_arr) > 0 else np.nan
    ot_std = float(np.std(ot_arr, ddof=1)) if len(ot_arr) > 1 else 0.0
    sal_mean = float(np.mean(sal_arr)) if len(sal_arr) > 0 else np.nan
    sal_std = float(np.std(sal_arr, ddof=1)) if len(sal_arr) > 1 else 0.0

    diff = ot_mean - sal_mean

    # Cohen's d
    pooled_std = np.sqrt((ot_std ** 2 + sal_std ** 2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else np.nan

    # Approximate p-value using Welch's t-test
    p_value = None
    if len(ot_arr) > 1 and len(sal_arr) > 1:
        p_value = _welch_t_test(ot_arr, sal_arr)

    return OTEffectResult(
        metric_name=metric_name,
        ot_mean=ot_mean,
        ot_std=ot_std,
        ot_n=len(ot_arr),
        saline_mean=sal_mean,
        saline_std=sal_std,
        saline_n=len(sal_arr),
        difference=diff,
        cohens_d=cohens_d,
        p_value=p_value,
    )


def _welch_t_test(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Welch's t-test p-value (two-tailed).

    Args:
        x: First sample
        y: Second sample

    Returns:
        Two-tailed p-value
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan

    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)

    # Welch's t-statistic
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-12:
        return np.nan

    t = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value
    p = 2 * _approx_t_sf(abs(t), df)

    return float(p)


def _approx_t_sf(t: float, df: float) -> float:
    """Approximate t-distribution survival function (1 - CDF)."""
    if df <= 0:
        return 0.5

    # For large df, use normal approximation
    if df > 100:
        from math import erf, sqrt
        return 0.5 * (1 - erf(t / sqrt(2)))

    # Simple approximation using beta function relation
    x = df / (df + t * t)
    # Incomplete beta approximation
    # For t > 0: p = 0.5 * I_x(df/2, 0.5)
    # Use rough approximation
    p = 0.5 * (x ** (df / 2))

    return min(max(p, 0.0), 1.0)


def compute_ot_effect_lmm(
    values: np.ndarray,
    drug_labels: np.ndarray,
    bank_labels: np.ndarray,
    session_ids: np.ndarray,
    metric_name: str = "metric",
) -> OTEffectResult:
    """Compute OT effect using linear mixed model.

    Model: metric ~ drug * bank + (1|session)

    Args:
        values: Metric values (one per unit or sample)
        drug_labels: Drug condition labels ('OT', 'saline')
        bank_labels: Bank labels ('dorsal', 'ventral')
        session_ids: Session identifiers for random effect
        metric_name: Name of the metric

    Returns:
        OTEffectResult with LMM statistics

    Note:
        Requires statsmodels for LMM fitting. Falls back to simple
        comparison if statsmodels is not available.
    """
    # Filter valid samples
    valid = ~np.isnan(values)
    valid = valid & np.isin(drug_labels, ['OT', 'saline', 'oxytocin'])
    valid = valid & np.isin(bank_labels, ['dorsal', 'ventral'])

    values_v = values[valid]
    drug_v = drug_labels[valid]
    bank_v = bank_labels[valid]
    session_v = session_ids[valid]

    # Normalize drug labels
    drug_v = np.array(['OT' if d in ('OT', 'oxytocin') else 'saline' for d in drug_v])

    # Simple comparison first
    ot_mask = drug_v == 'OT'
    sal_mask = drug_v == 'saline'

    simple_result = compute_ot_effect_simple(
        ot_values=list(values_v[ot_mask]),
        saline_values=list(values_v[sal_mask]),
        metric_name=metric_name,
    )

    # Try LMM with statsmodels
    lmm_results = None

    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame({
            'metric': values_v,
            'drug': drug_v,
            'bank': bank_v,
            'session': session_v,
        })

        # Fit LMM: metric ~ drug * bank + (1|session)
        model = smf.mixedlm(
            "metric ~ drug * bank",
            df,
            groups=df['session'],
        )
        fit = model.fit(method='lbfgs', maxiter=500)

        lmm_results = {
            'converged': bool(getattr(fit, 'converged', True)),
            'params': {k: float(v) for k, v in fit.params.items()},
            'pvalues': {k: float(v) for k, v in fit.pvalues.items()},
            'conf_int': {
                k: [float(fit.conf_int().loc[k, 0]), float(fit.conf_int().loc[k, 1])]
                for k in fit.params.index
            },
            'aic': float(fit.aic),
            'bic': float(fit.bic),
            'llf': float(fit.llf),
        }

        # Extract drug effect p-value
        drug_key = [k for k in fit.pvalues.index if 'drug' in k.lower() and 'bank' not in k.lower()]
        if drug_key:
            simple_result.p_value = float(fit.pvalues[drug_key[0]])

    except ImportError:
        pass  # statsmodels not available
    except Exception:
        pass  # LMM fitting failed

    simple_result.lmm_results = lmm_results

    return simple_result


def compute_ot_effect_by_bank(
    values: np.ndarray,
    drug_labels: np.ndarray,
    bank_labels: np.ndarray,
    metric_name: str = "metric",
) -> Dict[str, OTEffectResult]:
    """Compute OT effect separately for each bank.

    Args:
        values: Metric values
        drug_labels: Drug condition labels
        bank_labels: Bank labels
        metric_name: Metric name

    Returns:
        Dict mapping bank name to OTEffectResult
    """
    results = {}

    for bank in ['dorsal', 'ventral']:
        bank_mask = np.array([b.lower() == bank for b in bank_labels])

        if not bank_mask.any():
            continue

        v_bank = values[bank_mask]
        d_bank = drug_labels[bank_mask]

        # Normalize drug labels
        d_norm = np.array(['OT' if d in ('OT', 'oxytocin') else 'saline'
                          for d in d_bank])

        ot_vals = v_bank[d_norm == 'OT']
        sal_vals = v_bank[d_norm == 'saline']

        results[bank] = compute_ot_effect_simple(
            ot_values=list(ot_vals),
            saline_values=list(sal_vals),
            metric_name=f"{metric_name}_{bank}",
        )

    return results


def aggregate_ot_effects_across_sessions(
    session_results: List[Dict[str, Any]],
    metric_key: str,
) -> OTEffectResult:
    """Aggregate OT effects across multiple sessions.

    Args:
        session_results: List of per-session result dicts
        metric_key: Key to extract metric value from each result

    Returns:
        OTEffectResult aggregated across sessions
    """
    ot_values = []
    sal_values = []

    for result in session_results:
        value = result.get(metric_key)
        drug = result.get('drug', result.get('drug_condition', ''))

        if value is None or np.isnan(value):
            continue

        if drug in ('OT', 'oxytocin'):
            ot_values.append(value)
        elif drug in ('saline',):
            sal_values.append(value)

    return compute_ot_effect_simple(
        ot_values=ot_values,
        saline_values=sal_values,
        metric_name=metric_key,
    )
