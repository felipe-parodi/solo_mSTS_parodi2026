"""Figure 5 follow-up: coarsened affordance alignment analysis.

Tests whether context-dependent neural separation tracks coarse near-future
behavioral differences. Primary taxonomy: stay-vs-switch (binary: does the
animal change motif within a forward window?).

This is a state-level analysis: for each motif × tier state, we compute
the distribution over coarse future outcomes, then test whether neural
centroid distance between context-specific states (same motif, different
tiers) correlates with the JS divergence of their future distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


@dataclass(frozen=True)
class CoarseAffordanceConfig:
    """Configuration for coarsened affordance alignment.

    Attributes:
        window_bins: Forward window in bins for stay-vs-switch labeling.
        min_bins: Minimum bins per motif × tier cell to include.
        min_state_pairs: Minimum cross-tier pairs for Spearman test.
        alpha: Dirichlet smoothing parameter for future distributions.
    """

    window_bins: int = 4
    min_bins: int = 20
    min_state_pairs: int = 3
    alpha: float = 1.0


@dataclass
class CoarseAffordanceResult:
    """Result of coarsened affordance alignment analysis.

    Attributes:
        rho: Spearman correlation between neural distance and future JS.
        p_value: P-value for the Spearman test.
        n_pairs: Number of cross-tier motif pairs used.
        per_pair_data: Details per motif × tier pair.
    """

    rho: float
    p_value: float
    n_pairs: int
    per_pair_data: List[Dict[str, Any]] = field(default_factory=list)


# ── Coarse Future Labels ────────────────────────────────────────────────


def compute_coarse_future_labels(
    y_motif: np.ndarray,
    window_bins: int = 4,
) -> Dict[str, np.ndarray]:
    """Compute coarse future outcome labels for each bin.

    For each bin, looks forward up to ``window_bins`` bins to determine
    whether the motif identity changes (switch) or stays the same (stay).

    Args:
        y_motif: Integer motif labels of shape ``(n_bins,)``.
        window_bins: Number of bins to look ahead.

    Returns:
        Dict with ``'stay_vs_switch'``: 0 = stay, 1 = switch.
    """
    n = len(y_motif)
    stay_switch = np.zeros(n, dtype=int)

    for i in range(n):
        current = y_motif[i]
        for j in range(1, window_bins + 1):
            if i + j >= n:
                break
            if y_motif[i + j] != current:
                stay_switch[i] = 1
                break

    return {"stay_vs_switch": stay_switch}


# ── Future Distribution ─────────────────────────────────────────────────


def compute_stay_switch_distribution(
    future_labels: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    motif_id: int,
    tier_id: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute Dirichlet-smoothed stay/switch distribution for a state.

    A state is defined as (motif_id, tier_id). Counts all bins matching
    that state and tallies their future labels (0=stay, 1=switch).

    Args:
        future_labels: ``(n_bins,)`` binary stay(0)/switch(1) labels.
        y_motif: ``(n_bins,)`` motif labels.
        height_tier: ``(n_bins,)`` height tier labels.
        motif_id: Current motif identity.
        tier_id: Current height tier.
        alpha: Dirichlet smoothing parameter.

    Returns:
        Probability distribution of shape ``(2,)`` summing to 1.
    """
    counts = np.full(2, alpha, dtype=float)

    mask = (y_motif == motif_id) & (height_tier == tier_id)
    if not mask.any():
        return counts / counts.sum()

    labels_in_state = future_labels[mask]
    counts[0] += np.sum(labels_in_state == 0)
    counts[1] += np.sum(labels_in_state == 1)

    return counts / counts.sum()


# ── Coarse Affordance Alignment ─────────────────────────────────────────


def compute_coarse_affordance_alignment(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    future_labels: np.ndarray,
    config: CoarseAffordanceConfig,
) -> Optional[CoarseAffordanceResult]:
    """Test if neural distance correlates with coarse future divergence.

    For each pair of states (same motif, different tiers):
    1. Neural centroid distance in embedding space.
    2. JS divergence of stay/switch distributions.
    3. Spearman correlation across all qualifying pairs.

    Args:
        embedding: ``(n_bins, n_dim)`` latent embedding.
        y_motif: ``(n_bins,)`` motif labels.
        height_tier: ``(n_bins,)`` height tier.
        future_labels: ``(n_bins,)`` binary stay(0)/switch(1) labels.
        config: Analysis configuration.

    Returns:
        CoarseAffordanceResult, or None if insufficient data.
    """
    motif_ids = np.unique(y_motif[y_motif > 0])
    pair_data: List[Dict[str, Any]] = []

    for m in motif_ids:
        # Find tiers with enough data
        tiers_with_data: List[int] = []
        for tier in range(3):
            mask = (y_motif == m) & (height_tier == tier)
            if mask.sum() >= config.min_bins:
                tiers_with_data.append(tier)

        if len(tiers_with_data) < 2:
            continue

        # Compute centroids and future distributions per tier
        centroids: Dict[int, np.ndarray] = {}
        distributions: Dict[int, np.ndarray] = {}

        for tier in tiers_with_data:
            mask = (y_motif == m) & (height_tier == tier)
            centroids[tier] = np.mean(embedding[mask], axis=0)
            distributions[tier] = compute_stay_switch_distribution(
                future_labels, y_motif, height_tier,
                int(m), tier, alpha=config.alpha,
            )

        # Generate all tier pairs
        for i, t1 in enumerate(tiers_with_data):
            for t2 in tiers_with_data[i + 1:]:
                neural_dist = float(np.linalg.norm(
                    centroids[t1] - centroids[t2]
                ))
                # scipy jensenshannon returns sqrt(JSD)
                future_js = float(
                    jensenshannon(distributions[t1], distributions[t2]) ** 2
                )

                pair_data.append({
                    "motif": int(m),
                    "tier_a": t1,
                    "tier_b": t2,
                    "neural_dist": neural_dist,
                    "future_js": future_js,
                    "dist_a": distributions[t1].tolist(),
                    "dist_b": distributions[t2].tolist(),
                })

    if len(pair_data) < config.min_state_pairs:
        return None

    neural_dists = np.array([p["neural_dist"] for p in pair_data])
    future_js_vals = np.array([p["future_js"] for p in pair_data])

    rho, p_val = spearmanr(neural_dists, future_js_vals)
    if not np.isfinite(rho):
        return None

    return CoarseAffordanceResult(
        rho=float(rho),
        p_value=float(p_val),
        n_pairs=len(pair_data),
        per_pair_data=pair_data,
    )


__all__ = [
    "CoarseAffordanceConfig",
    "CoarseAffordanceResult",
    "compute_coarse_affordance_alignment",
    "compute_coarse_future_labels",
    "compute_stay_switch_distribution",
]
