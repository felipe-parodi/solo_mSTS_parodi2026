"""Behavioral surprisal: prediction error from interaction dynamics.

Behavioral surprisal captures "how surprising was the partner's next action
under the empirical interaction dynamics" - computed entirely from behavior
labels without neural data.

Model:
    p_θ(s_{t+1} | s_t, context) estimated from transition counts

Surprisal:
    Surp_beh(t) = -log p_θ(s_{t+1} = observed | s_t, context)

Context options (configurable):
    - Minimal: just previous state s_t
    - With regime: s_t + dyadic_regime (interactive/parallel/async)
    - With distance: s_t + distance_bin (quantile)
    - Full: s_t + regime + distance_bin

Usage:
    from analysis.interaction_summaries.surprise.behavioral import (
        compute_behavioral_surprisal,
        fit_transition_model,
    )

    # Fit transition model on training data
    model = fit_transition_model(
        current_states=train_states,
        next_states=train_next_states,
        context_labels=train_regime,  # optional
    )

    # Compute surprisal on test data
    result = compute_behavioral_surprisal(
        current_states=test_states,
        next_states=test_next_states,
        context_labels=test_regime,
        transition_model=model,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TransitionModel:
    """Transition probability model for behavioral surprisal.

    Stores count-based transition probabilities with Laplace smoothing.

    Attributes:
        transition_counts: Dict mapping (context, current_state) -> {next_state: count}
        state_vocab: List of unique state labels
        context_vocab: List of unique context labels (empty string for no context)
        laplace_alpha: Smoothing parameter
    """

    transition_counts: Dict[Tuple[str, str], Dict[str, int]]
    state_vocab: List[str]
    context_vocab: List[str]
    laplace_alpha: float = 1.0

    def get_probability(
        self,
        current_state: str,
        next_state: str,
        context: str = "",
    ) -> float:
        """Get transition probability p(next_state | current_state, context).

        Args:
            current_state: Current behavioral state
            next_state: Next behavioral state
            context: Context label (regime, distance bin, etc.)

        Returns:
            Probability in (0, 1]
        """
        key = (context, current_state)

        # Get counts for this (context, current) pair
        if key in self.transition_counts:
            counts = self.transition_counts[key]
        else:
            # Fall back to marginal if context not seen
            counts = {}
            for (ctx, cur), c in self.transition_counts.items():
                if cur == current_state:
                    for ns, cnt in c.items():
                        counts[ns] = counts.get(ns, 0) + cnt

        # Laplace smoothing
        n_states = len(self.state_vocab)
        total = sum(counts.values()) + self.laplace_alpha * n_states
        count = counts.get(next_state, 0) + self.laplace_alpha

        return count / total

    def get_surprisal(
        self,
        current_state: str,
        next_state: str,
        context: str = "",
    ) -> float:
        """Get surprisal (negative log probability).

        Args:
            current_state: Current behavioral state
            next_state: Next behavioral state
            context: Context label

        Returns:
            Surprisal in nats (natural log)
        """
        p = self.get_probability(current_state, next_state, context)
        return -np.log(p)


@dataclass
class BehavioralSurprisalResult:
    """Result of behavioral surprisal computation.

    Attributes:
        surprisal: Surprisal time series, shape (n_samples,)
        current_states: Current states used, shape (n_samples,)
        next_states: Next states observed, shape (n_samples,)
        context_labels: Context labels used, shape (n_samples,)
        valid_mask: Boolean mask for valid samples
        transition_model: TransitionModel used
        mean_surprisal: Mean surprisal over valid samples
        std_surprisal: Std of surprisal over valid samples
    """

    surprisal: np.ndarray
    current_states: np.ndarray
    next_states: np.ndarray
    context_labels: np.ndarray
    valid_mask: np.ndarray
    transition_model: TransitionModel
    mean_surprisal: float
    std_surprisal: float

    def surprisal_by_transition(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Get surprisal statistics grouped by transition type."""
        stats: Dict[Tuple[str, str], List[float]] = {}

        for i in range(len(self.surprisal)):
            if not self.valid_mask[i]:
                continue
            key = (str(self.current_states[i]), str(self.next_states[i]))
            if key not in stats:
                stats[key] = []
            stats[key].append(self.surprisal[i])

        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "count": len(v),
            }
            for k, v in stats.items()
        }

    def surprisal_by_context(self) -> Dict[str, Dict[str, float]]:
        """Get surprisal statistics grouped by context."""
        stats: Dict[str, List[float]] = {}

        for i in range(len(self.surprisal)):
            if not self.valid_mask[i]:
                continue
            ctx = str(self.context_labels[i]) if self.context_labels[i] else "none"
            if ctx not in stats:
                stats[ctx] = []
            stats[ctx].append(self.surprisal[i])

        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "count": len(v),
            }
            for k, v in stats.items()
        }


def fit_transition_model(
    current_states: np.ndarray,
    next_states: np.ndarray,
    context_labels: Optional[np.ndarray] = None,
    laplace_alpha: float = 1.0,
) -> TransitionModel:
    """Fit a transition probability model from state sequences.

    Args:
        current_states: Array of current state labels
        next_states: Array of next state labels
        context_labels: Optional array of context labels (regime, distance bin)
        laplace_alpha: Laplace smoothing parameter (default 1.0)

    Returns:
        TransitionModel with estimated probabilities
    """
    n = len(current_states)
    if len(next_states) != n:
        raise ValueError("current_states and next_states must have same length")

    if context_labels is None:
        context_labels = np.array([""] * n)
    elif len(context_labels) != n:
        raise ValueError("context_labels must have same length as states")

    # Collect vocabulary
    state_vocab = sorted(set(str(s) for s in current_states if s) |
                         set(str(s) for s in next_states if s))
    context_vocab = sorted(set(str(c) for c in context_labels))

    # Count transitions
    counts: Dict[Tuple[str, str], Dict[str, int]] = {}

    for i in range(n):
        cur = str(current_states[i]) if current_states[i] else ""
        nxt = str(next_states[i]) if next_states[i] else ""
        ctx = str(context_labels[i]) if context_labels[i] else ""

        if not cur or not nxt:
            continue

        key = (ctx, cur)
        if key not in counts:
            counts[key] = {}
        counts[key][nxt] = counts[key].get(nxt, 0) + 1

    return TransitionModel(
        transition_counts=counts,
        state_vocab=state_vocab,
        context_vocab=context_vocab,
        laplace_alpha=laplace_alpha,
    )


def compute_behavioral_surprisal(
    current_states: np.ndarray,
    next_states: np.ndarray,
    context_labels: Optional[np.ndarray] = None,
    transition_model: Optional[TransitionModel] = None,
    train_mask: Optional[np.ndarray] = None,
    laplace_alpha: float = 1.0,
) -> BehavioralSurprisalResult:
    """Compute behavioral surprisal time series.

    If transition_model is not provided, fits one on train_mask or all data.

    Args:
        current_states: Array of current state labels (length n)
        next_states: Array of next state labels (length n)
        context_labels: Optional context labels (regime, distance bin)
        transition_model: Pre-fitted TransitionModel (optional)
        train_mask: Boolean mask for training (fit model on True, compute for all)
        laplace_alpha: Smoothing parameter if fitting new model

    Returns:
        BehavioralSurprisalResult with surprisal time series
    """
    n = len(current_states)
    if len(next_states) != n:
        raise ValueError("current_states and next_states must have same length")

    if context_labels is None:
        context_labels = np.array([""] * n)
    elif len(context_labels) != n:
        raise ValueError("context_labels must have same length")

    # Fit transition model if not provided
    if transition_model is None:
        if train_mask is None:
            train_mask = np.ones(n, dtype=bool)

        transition_model = fit_transition_model(
            current_states=current_states[train_mask],
            next_states=next_states[train_mask],
            context_labels=context_labels[train_mask],
            laplace_alpha=laplace_alpha,
        )

    # Compute surprisal for all samples
    surprisal = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        cur = str(current_states[i]) if current_states[i] else ""
        nxt = str(next_states[i]) if next_states[i] else ""
        ctx = str(context_labels[i]) if context_labels[i] else ""

        if cur and nxt:
            surprisal[i] = transition_model.get_surprisal(cur, nxt, ctx)
            valid_mask[i] = True

    # Summary statistics
    valid_surp = surprisal[valid_mask]
    mean_surp = float(np.mean(valid_surp)) if len(valid_surp) > 0 else np.nan
    std_surp = float(np.std(valid_surp)) if len(valid_surp) > 1 else 0.0

    return BehavioralSurprisalResult(
        surprisal=surprisal,
        current_states=current_states,
        next_states=next_states,
        context_labels=context_labels,
        valid_mask=valid_mask,
        transition_model=transition_model,
        mean_surprisal=mean_surp,
        std_surprisal=std_surp,
    )


def compute_behavioral_surprisal_cv(
    current_states: np.ndarray,
    next_states: np.ndarray,
    context_labels: Optional[np.ndarray] = None,
    n_folds: int = 5,
    laplace_alpha: float = 1.0,
) -> BehavioralSurprisalResult:
    """Compute behavioral surprisal with cross-validation to prevent leakage.

    Fits transition model on training folds, computes surprisal on test folds.

    Args:
        current_states: Array of current state labels
        next_states: Array of next state labels
        context_labels: Optional context labels
        n_folds: Number of CV folds
        laplace_alpha: Smoothing parameter

    Returns:
        BehavioralSurprisalResult with CV-computed surprisal
    """
    n = len(current_states)

    if context_labels is None:
        context_labels = np.array([""] * n)

    # Create temporal folds
    fold_size = n // n_folds
    surprisal = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n)
        ])

        # Fit on training fold
        model = fit_transition_model(
            current_states=current_states[train_idx],
            next_states=next_states[train_idx],
            context_labels=context_labels[train_idx],
            laplace_alpha=laplace_alpha,
        )

        # Compute surprisal on test fold
        for i in test_idx:
            cur = str(current_states[i]) if current_states[i] else ""
            nxt = str(next_states[i]) if next_states[i] else ""
            ctx = str(context_labels[i]) if context_labels[i] else ""

            if cur and nxt:
                surprisal[i] = model.get_surprisal(cur, nxt, ctx)
                valid_mask[i] = True

    # Build final result with last fold's model (for reference only)
    final_model = fit_transition_model(
        current_states=current_states,
        next_states=next_states,
        context_labels=context_labels,
        laplace_alpha=laplace_alpha,
    )

    valid_surp = surprisal[valid_mask]
    mean_surp = float(np.mean(valid_surp)) if len(valid_surp) > 0 else np.nan
    std_surp = float(np.std(valid_surp)) if len(valid_surp) > 1 else 0.0

    return BehavioralSurprisalResult(
        surprisal=surprisal,
        current_states=current_states,
        next_states=next_states,
        context_labels=context_labels,
        valid_mask=valid_mask,
        transition_model=final_model,
        mean_surprisal=mean_surp,
        std_surprisal=std_surp,
    )
