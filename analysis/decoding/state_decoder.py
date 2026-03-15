"""High-level state decoder for behavioral state classification.

Provides unified interface for current-state and future-state decoding,
supporting multiple model families (controls-only, spikes-only, combined).

Usage:
    from analysis.decoding.state_decoder import StateDecoder, StateDecoderConfig

    config = StateDecoderConfig(n_folds=5, embargo_bins=20)
    decoder = StateDecoder(config)

    # Current state decoding
    result = decoder.decode_current_state(X_spk, y)

    # Future state prediction
    result = decoder.decode_future_state(X_spk, y, horizon_bins=4, change_only=True)

    # Model family comparison (M0/M1/M2)
    results = decoder.decode_model_families(X_spk, X_ctl, y)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .baselines import (
    BaselineResult,
    persistence_baseline,
    run_full_baseline_analysis,
    segment_shuffle,
)
from .classifiers import (
    ClassifierResult,
    LogisticClassifier,
    SVMClassifier,
    create_classifier,
)
from .metrics import (
    DecodingMetrics,
    aggregate_fold_metrics,
    compute_chance_level,
    compute_metrics,
)
from .temporal_cv import (
    get_change_only_mask,
    make_temporal_folds,
    shift_labels_for_horizon,
)


@dataclass
class StateDecoderConfig:
    """Configuration for state decoder."""

    classifier_type: str = "logistic"  # 'logistic' or 'svm'
    n_folds: int = 5
    embargo_bins: int = 20
    C: float = 1.0
    class_weight: str = "balanced"
    standardize: bool = True
    random_state: int = 42


@dataclass
class DecodingResult:
    """Result container for state decoding."""

    balanced_accuracy: float
    balanced_accuracy_std: float
    f1_macro: float
    f1_macro_std: float
    fold_metrics: List[DecodingMetrics]
    aggregate: Dict[str, float]
    config: StateDecoderConfig
    n_samples: int
    n_classes: int
    class_names: List[str]
    chance_level: float
    baselines: Optional[BaselineResult] = None
    horizon_bins: int = 0
    change_only: bool = False


@dataclass
class ModelFamilyResults:
    """Results for M0/M1/M2 model comparison."""

    M0_controls: Optional[DecodingResult]  # Controls only
    M1_spikes: Optional[DecodingResult]  # Spikes only
    M2_combined: Optional[DecodingResult]  # Spikes + Controls
    incremental_value: Optional[float] = None  # M2 - M0


class StateDecoder:
    """Flexible decoder for current and future state prediction.

    Supports:
    - Current-state decoding (y[t] from X[t])
    - Future-state decoding (y[t+h] from X[t])
    - Change-only prediction (filter to state transitions)
    - Model family comparison (M0, M1, M2)
    - Baseline computation (persistence, segment-shuffle)
    """

    def __init__(self, config: Optional[StateDecoderConfig] = None):
        """Initialize state decoder.

        Args:
            config: Decoder configuration. Uses defaults if not provided.
        """
        self.config = config or StateDecoderConfig()

    def decode_current_state(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> DecodingResult:
        """Decode current state y[t] from features X[t].

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: State labels, shape (n_samples,).

        Returns:
            DecodingResult with CV metrics.
        """
        return self._run_decoding(X, y, horizon_bins=0, change_only=False)

    def decode_future_state(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon_bins: int,
        change_only: bool = True,
        compute_baselines: bool = True,
    ) -> DecodingResult:
        """Decode future state y[t+h] from features X[t].

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: State labels, shape (n_samples,).
            horizon_bins: Prediction horizon in bins.
            change_only: If True, only predict on bins where state changes.
            compute_baselines: If True, compute persistence/shuffle baselines.

        Returns:
            DecodingResult with CV metrics and baselines.
        """
        return self._run_decoding(
            X, y,
            horizon_bins=horizon_bins,
            change_only=change_only,
            compute_baselines=compute_baselines,
        )

    def decode_model_families(
        self,
        X_spk: np.ndarray,
        X_ctl: np.ndarray,
        y: np.ndarray,
        horizon_bins: int = 0,
        change_only: bool = False,
    ) -> ModelFamilyResults:
        """Run decoding for three model families.

        M0: Controls only (X_ctl)
        M1: Spikes only (X_spk)
        M2: Combined (X_spk | X_ctl)

        Args:
            X_spk: Spike features, shape (n_samples, n_units).
            X_ctl: Control features, shape (n_samples, n_control_features).
            y: State labels, shape (n_samples,).
            horizon_bins: Prediction horizon (0 for current state).
            change_only: If True, filter to state changes.

        Returns:
            ModelFamilyResults with M0, M1, M2 results.
        """
        # M0: Controls only
        M0 = self._run_decoding(X_ctl, y, horizon_bins, change_only)

        # M1: Spikes only
        M1 = self._run_decoding(X_spk, y, horizon_bins, change_only)

        # M2: Combined
        X_combined = np.column_stack([X_spk, X_ctl])
        M2 = self._run_decoding(X_combined, y, horizon_bins, change_only)

        # Incremental value: how much do spikes add beyond controls?
        incremental = None
        if M0 is not None and M2 is not None:
            incremental = M2.balanced_accuracy - M0.balanced_accuracy

        return ModelFamilyResults(
            M0_controls=M0,
            M1_spikes=M1,
            M2_combined=M2,
            incremental_value=incremental,
        )

    def decode_horizon_sweep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizons: List[int],
        change_only: bool = True,
    ) -> Dict[int, DecodingResult]:
        """Run decoding across multiple prediction horizons.

        Args:
            X: Feature matrix.
            y: State labels.
            horizons: List of horizon values in bins.
            change_only: If True, filter to state changes.

        Returns:
            Dict mapping horizon -> DecodingResult.
        """
        results = {}
        for h in horizons:
            results[h] = self.decode_future_state(
                X, y,
                horizon_bins=h,
                change_only=change_only,
                compute_baselines=(h > 0),  # Only for future prediction
            )
        return results

    def _run_decoding(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon_bins: int = 0,
        change_only: bool = False,
        compute_baselines: bool = False,
    ) -> DecodingResult:
        """Core decoding routine with CV.

        Args:
            X: Feature matrix.
            y: State labels.
            horizon_bins: Prediction horizon.
            change_only: Filter to state changes.
            compute_baselines: Compute baseline comparisons.

        Returns:
            DecodingResult with all metrics.
        """
        n_samples_orig = len(y)

        # Apply horizon shift if needed
        if horizon_bins > 0:
            y_shifted, valid_mask = shift_labels_for_horizon(y, horizon_bins)
            X_valid = X[valid_mask]
            y_current = y[valid_mask]
            y_target = y_shifted

            # Apply change-only filter
            if change_only:
                change_mask = get_change_only_mask(y_current, y_target)
                X_valid = X_valid[change_mask]
                y_current = y_current[change_mask]
                y_target = y_target[change_mask]
        else:
            X_valid = X
            y_target = y
            y_current = y

        n_samples = len(y_target)
        if n_samples < self.config.n_folds * 2:
            # Not enough samples for CV
            return self._empty_result(horizon_bins, change_only)

        # Get class info
        class_names = [str(c) for c in np.unique(y_target)]
        n_classes = len(class_names)
        chance = compute_chance_level(y_target, method="uniform")

        # Create CV folds
        folds = make_temporal_folds(
            n_samples,
            n_folds=self.config.n_folds,
            embargo_bins=self.config.embargo_bins,
        )

        # Run CV
        fold_metrics = []
        for train_idx, test_idx in folds:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y_target[train_idx], y_target[test_idx]

            # Create and fit classifier
            clf = create_classifier(
                self.config.classifier_type,
                C=self.config.C,
                class_weight=self.config.class_weight,
                standardize=self.config.standardize,
                random_state=self.config.random_state,
            )
            clf.fit(X_train, y_train)
            result = clf.predict(X_test)

            # Compute fold metrics
            metrics = compute_metrics(y_test, result.y_pred, class_names)
            fold_metrics.append(metrics)

        if not fold_metrics:
            return self._empty_result(horizon_bins, change_only)

        # Aggregate across folds
        agg = aggregate_fold_metrics(fold_metrics)

        # Compute baselines for future prediction
        baselines = None
        if compute_baselines and horizon_bins > 0:
            baselines = run_full_baseline_analysis(
                y_current=y_current,
                y_future=y_target,
                actual_accuracy=agg["balanced_accuracy_mean"],
                n_shuffles=100,
                segment_size_bins=50,
                random_state=self.config.random_state,
            )

        return DecodingResult(
            balanced_accuracy=agg["balanced_accuracy_mean"],
            balanced_accuracy_std=agg["balanced_accuracy_std"],
            f1_macro=agg["f1_macro_mean"],
            f1_macro_std=agg["f1_macro_std"],
            fold_metrics=fold_metrics,
            aggregate=agg,
            config=self.config,
            n_samples=n_samples,
            n_classes=n_classes,
            class_names=class_names,
            chance_level=chance,
            baselines=baselines,
            horizon_bins=horizon_bins,
            change_only=change_only,
        )

    def _empty_result(
        self,
        horizon_bins: int,
        change_only: bool,
    ) -> DecodingResult:
        """Create empty result for edge cases."""
        return DecodingResult(
            balanced_accuracy=np.nan,
            balanced_accuracy_std=np.nan,
            f1_macro=np.nan,
            f1_macro_std=np.nan,
            fold_metrics=[],
            aggregate={},
            config=self.config,
            n_samples=0,
            n_classes=0,
            class_names=[],
            chance_level=np.nan,
            baselines=None,
            horizon_bins=horizon_bins,
            change_only=change_only,
        )
