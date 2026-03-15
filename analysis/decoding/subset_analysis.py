"""Subset analysis utilities for conditional decoding.

Provides functions to run decoding on data subsets:
- By condition (e.g., decode partner when self_state is fixed)
- By feature subset (e.g., dorsal vs ventral units)

These are composition patterns that use StateDecoder on filtered data.

Usage:
    from analysis.decoding.subset_analysis import (
        decode_by_condition,
        decode_by_feature_groups,
        decode_same_self_state,
    )

    # Same-self-state partner decoding (confound control)
    results = decode_same_self_state(
        decoder, X, y_self, y_partner,
        self_states=['rest', 'locomotion'],
    )

    # Bank-specific decoding
    results = decode_by_feature_groups(
        decoder, X, y,
        feature_groups={'dorsal': dorsal_idx, 'ventral': ventral_idx},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .state_decoder import DecodingResult, StateDecoder, StateDecoderConfig


@dataclass
class SubsetResult:
    """Result for a single subset analysis."""

    subset_name: str
    decoding_result: DecodingResult
    n_samples: int
    subset_mask: np.ndarray  # Boolean mask used


@dataclass
class ConditionResults:
    """Results for decode_by_condition."""

    results_by_condition: Dict[str, SubsetResult]
    condition_variable: str
    aggregate_accuracy: float  # Weighted average
    n_conditions: int


@dataclass
class FeatureGroupResults:
    """Results for decode_by_feature_groups."""

    results_by_group: Dict[str, DecodingResult]
    group_sizes: Dict[str, int]
    n_groups: int


def decode_by_condition(
    decoder: StateDecoder,
    X: np.ndarray,
    y_target: np.ndarray,
    condition: np.ndarray,
    condition_values: Optional[List[Any]] = None,
    min_samples: int = 50,
) -> ConditionResults:
    """Run decoding separately for each condition value.

    Useful for stratified analysis, e.g., decoding by drug condition
    or time period.

    Args:
        decoder: Configured StateDecoder instance.
        X: Feature matrix, shape (n_samples, n_features).
        y_target: Target labels to decode, shape (n_samples,).
        condition: Condition variable, shape (n_samples,).
        condition_values: Specific values to analyze. If None, uses all unique.
        min_samples: Minimum samples required per condition.

    Returns:
        ConditionResults with per-condition decoding results.

    Example:
        >>> # Decode by drug condition
        >>> results = decode_by_condition(decoder, X, y, drug_labels, ['OT', 'saline'])
    """
    if condition_values is None:
        condition_values = list(np.unique(condition))

    results_by_condition = {}
    weighted_sum = 0.0
    total_samples = 0

    for cond_val in condition_values:
        mask = condition == cond_val
        n_samples = np.sum(mask)

        if n_samples < min_samples:
            continue

        X_subset = X[mask]
        y_subset = y_target[mask]

        result = decoder.decode_current_state(X_subset, y_subset)

        results_by_condition[str(cond_val)] = SubsetResult(
            subset_name=str(cond_val),
            decoding_result=result,
            n_samples=n_samples,
            subset_mask=mask,
        )

        if not np.isnan(result.balanced_accuracy):
            weighted_sum += result.balanced_accuracy * n_samples
            total_samples += n_samples

    aggregate_acc = weighted_sum / total_samples if total_samples > 0 else np.nan

    return ConditionResults(
        results_by_condition=results_by_condition,
        condition_variable="condition",
        aggregate_accuracy=aggregate_acc,
        n_conditions=len(results_by_condition),
    )


def decode_by_feature_groups(
    decoder: StateDecoder,
    X: np.ndarray,
    y: np.ndarray,
    feature_groups: Dict[str, List[int]],
    equalize_features: bool = False,
    n_subsamples: int = 50,
    random_state: int = 42,
) -> FeatureGroupResults:
    """Run decoding with different feature subsets.

    Useful for bank comparison (dorsal vs ventral) where each bank
    has different units.

    Args:
        decoder: Configured StateDecoder instance.
        X: Full feature matrix, shape (n_samples, n_features).
        y: Target labels, shape (n_samples,).
        feature_groups: Dict mapping group name to feature indices.
        equalize_features: If True, subsample larger groups to match smallest.
        n_subsamples: Number of subsampling iterations if equalizing.
        random_state: Random seed for subsampling.

    Returns:
        FeatureGroupResults with per-group decoding results.

    Example:
        >>> # Bank-specific decoding
        >>> groups = {'dorsal': [0,1,2,3], 'ventral': [4,5,6,7]}
        >>> results = decode_by_feature_groups(decoder, X, y, groups)
    """
    group_sizes = {name: len(indices) for name, indices in feature_groups.items()}

    if equalize_features:
        min_size = min(group_sizes.values())
        return _decode_equalized(
            decoder, X, y, feature_groups, min_size, n_subsamples, random_state
        )

    results_by_group = {}
    for group_name, indices in feature_groups.items():
        X_subset = X[:, indices]
        result = decoder.decode_current_state(X_subset, y)
        results_by_group[group_name] = result

    return FeatureGroupResults(
        results_by_group=results_by_group,
        group_sizes=group_sizes,
        n_groups=len(feature_groups),
    )


def _decode_equalized(
    decoder: StateDecoder,
    X: np.ndarray,
    y: np.ndarray,
    feature_groups: Dict[str, List[int]],
    target_size: int,
    n_subsamples: int,
    random_state: int,
) -> FeatureGroupResults:
    """Decode with equalized feature counts via subsampling."""
    rng = np.random.RandomState(random_state)

    results_by_group = {}
    group_sizes = {}

    for group_name, indices in feature_groups.items():
        indices = np.array(indices)
        n_features = len(indices)
        group_sizes[group_name] = n_features

        if n_features <= target_size:
            # Use all features
            X_subset = X[:, indices]
            result = decoder.decode_current_state(X_subset, y)
            results_by_group[group_name] = result
        else:
            # Subsample and average
            accs = []
            for _ in range(n_subsamples):
                sub_idx = rng.choice(indices, size=target_size, replace=False)
                X_subset = X[:, sub_idx]
                result = decoder.decode_current_state(X_subset, y)
                if not np.isnan(result.balanced_accuracy):
                    accs.append(result.balanced_accuracy)

            # Create aggregate result (use last result as template)
            if accs:
                result.balanced_accuracy = float(np.mean(accs))
                result.balanced_accuracy_std = float(np.std(accs))
            results_by_group[group_name] = result

    return FeatureGroupResults(
        results_by_group=results_by_group,
        group_sizes=group_sizes,
        n_groups=len(feature_groups),
    )


def decode_same_self_state(
    decoder: StateDecoder,
    X: np.ndarray,
    y_self: np.ndarray,
    y_partner: np.ndarray,
    self_states: Optional[List[str]] = None,
    top_k: int = 3,
    min_samples: int = 50,
) -> Dict[str, SubsetResult]:
    """Decode partner state with self state held constant.

    This is the primary confound control for Figure 2: shows that
    partner decoding is not just because self_state predicts partner_state.

    Args:
        decoder: Configured StateDecoder instance.
        X: Feature matrix (spikes or controls).
        y_self: Self behavioral state labels.
        y_partner: Partner behavioral state labels.
        self_states: Specific self states to analyze. If None, uses top_k.
        top_k: If self_states is None, use the K most frequent self states.
        min_samples: Minimum samples required per self state.

    Returns:
        Dict mapping self_state -> SubsetResult for partner decoding.

    Example:
        >>> # Does partner decoding persist when self is at rest?
        >>> results = decode_same_self_state(decoder, X_spk, y_self, y_partner)
        >>> for state, res in results.items():
        ...     print(f"Self={state}: partner acc = {res.decoding_result.balanced_accuracy:.3f}")
    """
    # Determine which self states to analyze
    if self_states is None:
        unique, counts = np.unique(y_self, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        self_states = [unique[i] for i in sorted_idx[:top_k]]

    results = {}
    for self_state in self_states:
        mask = y_self == self_state
        n_samples = np.sum(mask)

        if n_samples < min_samples:
            continue

        X_subset = X[mask]
        y_partner_subset = y_partner[mask]

        # Check we have enough partner state diversity
        unique_partner = np.unique(y_partner_subset)
        if len(unique_partner) < 2:
            continue

        result = decoder.decode_current_state(X_subset, y_partner_subset)

        results[str(self_state)] = SubsetResult(
            subset_name=str(self_state),
            decoding_result=result,
            n_samples=n_samples,
            subset_mask=mask,
        )

    return results


def decode_cross_validated_subsets(
    decoder: StateDecoder,
    X: np.ndarray,
    y: np.ndarray,
    subset_masks: Dict[str, np.ndarray],
    use_shared_folds: bool = True,
) -> Dict[str, DecodingResult]:
    """Decode on multiple subsets with optional shared CV folds.

    Useful when comparing across subsets where you want consistent
    train/test splits.

    Args:
        decoder: Configured StateDecoder instance.
        X: Feature matrix.
        y: Target labels.
        subset_masks: Dict mapping subset name to boolean mask.
        use_shared_folds: If True, create folds on intersection and apply.

    Returns:
        Dict mapping subset name to DecodingResult.
    """
    results = {}

    for subset_name, mask in subset_masks.items():
        X_subset = X[mask]
        y_subset = y[mask]

        if len(y_subset) < decoder.config.n_folds * 2:
            continue

        result = decoder.decode_current_state(X_subset, y_subset)
        results[subset_name] = result

    return results
