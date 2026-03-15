"""Comparison utilities for decoding results.

Provides functions to compare decoding results across conditions,
compute effect sizes, and aggregate at the session level.

Usage:
    from analysis.decoding.comparison import (
        compute_weight_index,
        compute_incremental_value,
        compare_conditions,
        aggregate_sessions,
    )

    # Partner-vs-self weighting (OT modulation metric)
    weight = compute_weight_index(perf_partner=0.65, perf_self=0.55)

    # Incremental value of neural data
    delta = compute_incremental_value(perf_m2=0.70, perf_m0=0.55)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state_decoder import DecodingResult


@dataclass
class ComparisonResult:
    """Result of comparing two conditions."""

    condition_a: str
    condition_b: str
    accuracy_a: float
    accuracy_b: float
    difference: float  # a - b
    relative_difference: float  # (a - b) / mean
    effect_size: Optional[float] = None  # Cohen's d if available


@dataclass
class SessionSummary:
    """Summary of decoding results for a single session."""

    session_id: str
    drug_condition: str
    bank: Optional[str]
    balanced_accuracy: float
    f1_macro: float
    n_samples: int
    n_classes: int
    weight_index: Optional[float] = None
    incremental_value: Optional[float] = None


@dataclass
class DrugEffectResult:
    """Result of OT vs saline comparison."""

    metric_name: str
    ot_mean: float
    ot_std: float
    saline_mean: float
    saline_std: float
    difference: float  # OT - saline
    effect_size: float  # Cohen's d
    n_ot: int
    n_saline: int


def compute_weight_index(
    perf_partner: float,
    perf_self: float,
) -> float:
    """Compute partner-vs-self weighting index.

    WeightIndex = (Perf_partner - Perf_self) / (Perf_partner + Perf_self)

    Positive values indicate stronger partner encoding relative to self.
    This is the primary OT modulation metric (Option 1 in Figure 2 spec).

    Args:
        perf_partner: Balanced accuracy for partner state decoding.
        perf_self: Balanced accuracy for self state decoding.

    Returns:
        Weight index in range [-1, 1].

    Example:
        >>> weight = compute_weight_index(0.65, 0.55)
        >>> weight  # Partner slightly stronger
        0.083
    """
    if np.isnan(perf_partner) or np.isnan(perf_self):
        return np.nan

    denominator = perf_partner + perf_self
    if denominator == 0:
        return 0.0

    return (perf_partner - perf_self) / denominator


def compute_incremental_value(
    perf_combined: float,
    perf_controls: float,
) -> float:
    """Compute incremental value of neural data.

    Delta = Perf(M2) - Perf(M0)

    Positive values indicate neural data adds information beyond controls.
    This is OT modulation metric Option 2 in Figure 2 spec.

    Args:
        perf_combined: Performance with spikes + controls (M2).
        perf_controls: Performance with controls only (M0).

    Returns:
        Incremental value (difference in accuracy).

    Example:
        >>> delta = compute_incremental_value(0.70, 0.55)
        >>> delta
        0.15
    """
    if np.isnan(perf_combined) or np.isnan(perf_controls):
        return np.nan

    return perf_combined - perf_controls


def compare_conditions(
    result_a: DecodingResult,
    result_b: DecodingResult,
    name_a: str = "A",
    name_b: str = "B",
) -> ComparisonResult:
    """Compare decoding results between two conditions.

    Args:
        result_a: DecodingResult for condition A.
        result_b: DecodingResult for condition B.
        name_a: Label for condition A.
        name_b: Label for condition B.

    Returns:
        ComparisonResult with difference metrics.
    """
    acc_a = result_a.balanced_accuracy
    acc_b = result_b.balanced_accuracy

    diff = acc_a - acc_b
    mean_acc = (acc_a + acc_b) / 2
    rel_diff = diff / mean_acc if mean_acc > 0 else np.nan

    # Effect size (Cohen's d approximation using pooled std)
    effect_size = None
    if result_a.balanced_accuracy_std > 0 and result_b.balanced_accuracy_std > 0:
        pooled_std = np.sqrt(
            (result_a.balanced_accuracy_std ** 2 + result_b.balanced_accuracy_std ** 2) / 2
        )
        if pooled_std > 0:
            effect_size = diff / pooled_std

    return ComparisonResult(
        condition_a=name_a,
        condition_b=name_b,
        accuracy_a=acc_a,
        accuracy_b=acc_b,
        difference=diff,
        relative_difference=rel_diff,
        effect_size=effect_size,
    )


def compute_drug_effect(
    ot_values: List[float],
    saline_values: List[float],
    metric_name: str = "accuracy",
) -> DrugEffectResult:
    """Compute OT vs saline effect statistics.

    Args:
        ot_values: List of metric values for OT sessions.
        saline_values: List of metric values for saline sessions.
        metric_name: Name of the metric being compared.

    Returns:
        DrugEffectResult with effect statistics.

    Example:
        >>> ot = [0.65, 0.70, 0.68]
        >>> sal = [0.55, 0.58, 0.52]
        >>> effect = compute_drug_effect(ot, sal, 'weight_index')
    """
    ot_arr = np.array([v for v in ot_values if not np.isnan(v)])
    sal_arr = np.array([v for v in saline_values if not np.isnan(v)])

    ot_mean = float(np.mean(ot_arr)) if len(ot_arr) > 0 else np.nan
    ot_std = float(np.std(ot_arr)) if len(ot_arr) > 1 else 0.0
    sal_mean = float(np.mean(sal_arr)) if len(sal_arr) > 0 else np.nan
    sal_std = float(np.std(sal_arr)) if len(sal_arr) > 1 else 0.0

    diff = ot_mean - sal_mean

    # Cohen's d
    pooled_std = np.sqrt((ot_std ** 2 + sal_std ** 2) / 2)
    effect_size = diff / pooled_std if pooled_std > 0 else np.nan

    return DrugEffectResult(
        metric_name=metric_name,
        ot_mean=ot_mean,
        ot_std=ot_std,
        saline_mean=sal_mean,
        saline_std=sal_std,
        difference=diff,
        effect_size=effect_size,
        n_ot=len(ot_arr),
        n_saline=len(sal_arr),
    )


def create_session_summary(
    session_id: str,
    drug_condition: str,
    result_self: Optional[DecodingResult],
    result_partner: Optional[DecodingResult],
    result_m0: Optional[DecodingResult] = None,
    result_m2: Optional[DecodingResult] = None,
    bank: Optional[str] = None,
) -> SessionSummary:
    """Create summary for a single session's decoding results.

    Args:
        session_id: Session identifier.
        drug_condition: Drug condition ('OT', 'saline', etc.).
        result_self: Result for self state decoding.
        result_partner: Result for partner state decoding.
        result_m0: Result for controls-only model (for incremental value).
        result_m2: Result for combined model (for incremental value).
        bank: Optional bank label ('dorsal', 'ventral').

    Returns:
        SessionSummary with key metrics.
    """
    # Use partner result as primary
    primary = result_partner if result_partner else result_self

    if primary is None:
        return SessionSummary(
            session_id=session_id,
            drug_condition=drug_condition,
            bank=bank,
            balanced_accuracy=np.nan,
            f1_macro=np.nan,
            n_samples=0,
            n_classes=0,
        )

    # Compute weight index if both self and partner available
    weight_index = None
    if result_self and result_partner:
        weight_index = compute_weight_index(
            result_partner.balanced_accuracy,
            result_self.balanced_accuracy,
        )

    # Compute incremental value if M0 and M2 available
    incremental = None
    if result_m0 and result_m2:
        incremental = compute_incremental_value(
            result_m2.balanced_accuracy,
            result_m0.balanced_accuracy,
        )

    return SessionSummary(
        session_id=session_id,
        drug_condition=drug_condition,
        bank=bank,
        balanced_accuracy=primary.balanced_accuracy,
        f1_macro=primary.f1_macro,
        n_samples=primary.n_samples,
        n_classes=primary.n_classes,
        weight_index=weight_index,
        incremental_value=incremental,
    )


def aggregate_sessions(
    summaries: List[SessionSummary],
    by: str = "drug_condition",
) -> Dict[str, Dict[str, float]]:
    """Aggregate session summaries by condition.

    Args:
        summaries: List of SessionSummary objects.
        by: Variable to group by ('drug_condition', 'bank', or both).

    Returns:
        Dict mapping group -> {metric: mean/std}.

    Example:
        >>> agg = aggregate_sessions(summaries, by='drug_condition')
        >>> agg['OT']['balanced_accuracy_mean']
        0.65
    """
    groups: Dict[str, List[SessionSummary]] = {}

    for s in summaries:
        if by == "drug_condition":
            key = s.drug_condition
        elif by == "bank":
            key = s.bank or "all"
        elif by == "drug_bank":
            key = f"{s.drug_condition}_{s.bank or 'all'}"
        else:
            key = "all"

        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    result = {}
    for key, sessions in groups.items():
        accs = [s.balanced_accuracy for s in sessions if not np.isnan(s.balanced_accuracy)]
        weights = [s.weight_index for s in sessions if s.weight_index and not np.isnan(s.weight_index)]
        increments = [s.incremental_value for s in sessions if s.incremental_value and not np.isnan(s.incremental_value)]

        result[key] = {
            "balanced_accuracy_mean": float(np.mean(accs)) if accs else np.nan,
            "balanced_accuracy_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
            "weight_index_mean": float(np.mean(weights)) if weights else np.nan,
            "weight_index_std": float(np.std(weights)) if len(weights) > 1 else 0.0,
            "incremental_value_mean": float(np.mean(increments)) if increments else np.nan,
            "incremental_value_std": float(np.std(increments)) if len(increments) > 1 else 0.0,
            "n_sessions": len(sessions),
        }

    return result


def compute_bank_difference(
    dorsal_result: DecodingResult,
    ventral_result: DecodingResult,
) -> Dict[str, float]:
    """Compute difference between dorsal and ventral banks.

    Args:
        dorsal_result: Decoding result for dorsal bank.
        ventral_result: Decoding result for ventral bank.

    Returns:
        Dict with comparison metrics.
    """
    comparison = compare_conditions(
        dorsal_result, ventral_result,
        name_a="dorsal", name_b="ventral"
    )

    return {
        "dorsal_accuracy": comparison.accuracy_a,
        "ventral_accuracy": comparison.accuracy_b,
        "dorsal_minus_ventral": comparison.difference,
        "relative_difference": comparison.relative_difference,
        "effect_size": comparison.effect_size,
    }
