"""Matched-kinematics context control and affordance alignment for Fig 5D.

Hardens the context-dependent coding claim by:
1. Bout-level nearest-neighbor matching across height tiers on 14 covariates
   (excluding com_z which defines the context variable).
2. Affordance alignment: tests whether neural centroid distance between
   context-specific states correlates with JS divergence of future
   behavioral distributions.

Also contains the refactored ``compute_context_coding()`` shared by the
original CEBRA runner and this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

# 14 matching features (com_z excluded — it defines the context variable)
MATCHING_COLS: List[str] = [
    "com_x", "com_y",
    "speed", "yaw_rate", "v_parallel", "v_perp",
    "pitch_rate", "roll_rate", "delta_yaw",
    "gaze_x", "gaze_y", "gaze_z", "d_wall_horiz", "vis_motion",
]


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatchedContextConfig:
    """Configuration for matched-kinematics analysis.

    Attributes:
        min_bouts_per_tier: Minimum bouts per motif×tier to include.
        n_perms: Number of permutations for significance testing.
        caliper: Maximum standardized Euclidean distance for a match.
        min_transitions: Minimum transitions per state for affordance.
    """

    min_bouts_per_tier: int = 5
    n_perms: int = 500
    caliper: float = 1.0
    min_transitions: int = 8


@dataclass
class MatchedContextResult:
    """Result of matched-kinematics context ratio analysis.

    Attributes:
        matched_ratio: Between-context / within-context distance ratio
            computed on matched bout subsets.
        p_value: Permutation p-value (tier shuffled within matched set).
        n_matched_bouts: Total number of matched bout pairs.
        matched_fraction: Fraction of minority bouts that found a match.
        smd_table: Covariate balance table (SMD per feature).
        per_motif_results: Per-motif matching details.
    """

    matched_ratio: float
    p_value: float
    n_matched_bouts: int
    matched_fraction: float
    smd_table: pd.DataFrame
    per_motif_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AffordanceResult:
    """Result of affordance alignment analysis.

    Attributes:
        per_session_rho: Spearman rho (neural_dist vs future_JS).
        per_session_p: P-value for Spearman rho.
        n_pairs: Number of state pairs used.
        per_pair_data: Details per motif×tier pair.
    """

    per_session_rho: float
    per_session_p: float
    n_pairs: int
    per_pair_data: List[Dict[str, Any]] = field(default_factory=list)


# ── Bout Segmentation ────────────────────────────────────────────────────

def segment_bouts(y_motif: np.ndarray) -> List[Tuple[int, int, int]]:
    """Segment motif label array into contiguous bouts.

    Args:
        y_motif: Integer motif labels of shape ``(n_bins,)``.

    Returns:
        List of ``(start_idx, end_idx, motif_id)`` tuples.
        ``end_idx`` is exclusive (Python slice convention).
    """
    if len(y_motif) == 0:
        return []

    bouts: List[Tuple[int, int, int]] = []
    start = 0
    current = int(y_motif[0])

    for i in range(1, len(y_motif)):
        if int(y_motif[i]) != current:
            bouts.append((start, i, current))
            start = i
            current = int(y_motif[i])
    bouts.append((start, len(y_motif), current))
    return bouts


# ── Bout Representative Features ─────────────────────────────────────────

def compute_bout_features(
    features: np.ndarray,
    bouts: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Compute mid-bout averaged features for each bout.

    Uses the middle 50% of each bout to avoid transition edges.
    For short bouts (< 4 bins), uses all bins.

    Args:
        features: Feature array of shape ``(n_bins, n_features)``.
        bouts: List of ``(start, end, motif_id)`` from ``segment_bouts()``.

    Returns:
        Array of shape ``(n_bouts, n_features)`` with one row per bout.
    """
    result = np.empty((len(bouts), features.shape[1]), dtype=float)

    for i, (start, end, _) in enumerate(bouts):
        bout_len = end - start
        if bout_len < 4:
            result[i] = features[start:end].mean(axis=0)
        else:
            q1 = start + bout_len // 4
            q3 = start + 3 * bout_len // 4
            result[i] = features[q1:q3].mean(axis=0)

    return result


# ── Bout Matching ─────────────────────────────────────────────────────────

def match_bouts_within_motif(
    bout_features: np.ndarray,
    bout_tiers: np.ndarray,
    caliper: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Match bouts across height tiers using nearest-neighbor with caliper.

    For each bout in the minority tier, finds the nearest unmatched bout
    in the majority tier (standardized Euclidean distance). Matches beyond
    the caliper threshold are rejected.

    The caliper is specified as per-dimension standardized distance and
    scaled internally by sqrt(n_features) to convert to Euclidean distance
    in the full feature space.

    Args:
        bout_features: ``(n_bouts, n_features)`` standardized features.
        bout_tiers: ``(n_bouts,)`` tier labels.
        caliper: Maximum per-dimension standardized distance for a valid
            match. Internally scaled by sqrt(n_features).

    Returns:
        Tuple of ``(matched_majority_idx, matched_minority_idx, fraction)``
        where fraction is the proportion of minority bouts successfully
        matched.
    """
    unique_tiers = np.unique(bout_tiers)
    if len(unique_tiers) < 2:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0

    # Scale caliper by sqrt(n_features) for Euclidean distance
    n_features = bout_features.shape[1] if bout_features.ndim == 2 else 1
    euclidean_caliper = caliper * np.sqrt(n_features)

    # Identify minority and majority tiers
    counts = {int(t): int((bout_tiers == t).sum()) for t in unique_tiers}
    minority_tier = min(counts, key=counts.get)
    majority_tier = max(counts, key=counts.get)
    if counts[minority_tier] == counts[majority_tier]:
        minority_tier = int(unique_tiers[0])
        majority_tier = int(unique_tiers[1])

    minority_idx = np.where(bout_tiers == minority_tier)[0]
    majority_idx = np.where(bout_tiers == majority_tier)[0]

    matched_majority = []
    matched_minority = []
    used_majority = set()

    for m_idx in minority_idx:
        best_dist = np.inf
        best_maj = -1

        for maj_idx in majority_idx:
            if int(maj_idx) in used_majority:
                continue
            dist = np.linalg.norm(
                bout_features[m_idx] - bout_features[maj_idx]
            )
            if dist < best_dist:
                best_dist = dist
                best_maj = int(maj_idx)

        if best_maj >= 0 and best_dist <= euclidean_caliper:
            matched_majority.append(best_maj)
            matched_minority.append(int(m_idx))
            used_majority.add(best_maj)

    n_minority = len(minority_idx)
    fraction = len(matched_minority) / n_minority if n_minority > 0 else 0.0

    return (
        np.array(matched_majority, dtype=int),
        np.array(matched_minority, dtype=int),
        fraction,
    )


# ── Covariate Balance ────────────────────────────────────────────────────

def compute_covariate_balance(
    features_a: np.ndarray,
    features_b: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute standardized mean difference (SMD) per feature.

    SMD = (mean_a - mean_b) / sqrt((var_a + var_b) / 2).

    Args:
        features_a: ``(n_a, n_features)`` array for group A.
        features_b: ``(n_b, n_features)`` array for group B.
        feature_names: Column names for the output.

    Returns:
        DataFrame with columns ``['feature', 'mean_a', 'mean_b', 'smd']``.
    """
    mean_a = features_a.mean(axis=0)
    mean_b = features_b.mean(axis=0)
    var_a = features_a.var(axis=0, ddof=1)
    var_b = features_b.var(axis=0, ddof=1)
    pooled_std = np.sqrt((var_a + var_b) / 2.0)
    pooled_std = np.where(pooled_std < 1e-12, 1.0, pooled_std)
    smd = (mean_a - mean_b) / pooled_std

    return pd.DataFrame({
        "feature": feature_names,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "smd": smd,
    })


# ── Context Coding (refactored from run_fig5_cebra.py) ───────────────────

def permute_tiers_within_pairs(
    height_tier: np.ndarray,
    pair_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly swap tier labels within matched bout pairs.

    Args:
        height_tier: Tier label per bin.
        pair_ids: Matched-pair identifier per bin.
        rng: Random number generator.

    Returns:
        Tier labels with each matched pair either left unchanged or swapped.

    Raises:
        ValueError: If ``height_tier`` and ``pair_ids`` are misaligned, or if a
            pair does not contain exactly two unique tiers.
    """
    if len(height_tier) != len(pair_ids):
        raise ValueError("height_tier and pair_ids must have the same length.")

    shuffled_tier = height_tier.copy()
    for pair_id in np.unique(pair_ids):
        pair_mask = pair_ids == pair_id
        pair_tiers = np.unique(height_tier[pair_mask])
        if len(pair_tiers) != 2:
            raise ValueError(
                f"Matched pair {pair_id} must contain exactly two tiers; "
                f"found {len(pair_tiers)}."
            )
        if rng.integers(2) == 1:
            tier_a, tier_b = int(pair_tiers[0]), int(pair_tiers[1])
            shuffled_tier[pair_mask & (height_tier == tier_a)] = tier_b
            shuffled_tier[pair_mask & (height_tier == tier_b)] = tier_a

    return shuffled_tier

def compute_context_coding(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    min_bins: int = 20,
    n_perms: int = 500,
    seed: int = 42,
    min_cells: int = 4,
    pair_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Test if same syllable is coded differently across spatial contexts.

    Computes within-syllable between-context vs within-context distances
    in latent space. Refactored from ``run_fig5_cebra.py:731``.

    Args:
        embedding: ``(n_bins, n_dim)`` latent embedding.
        y_motif: ``(n_bins,)`` motif labels.
        height_tier: ``(n_bins,)`` height tier (0=floor, 1=bench, 2=ceiling).
        min_bins: Minimum bins per syllable×tier cell.
        n_perms: Number of permutations for significance.
        seed: Random seed.
        min_cells: Minimum number of valid syllable×tier cells required.
            Default 4 for full-dataset analysis; use 2 for matched subsets
            (1 motif × 2 tiers is valid).
        pair_ids: Optional matched-pair identifier per bin. When provided,
            permutation nulls swap tier labels within each pair instead of
            shuffling bins within a motif.

    Returns:
        Dict with distance metrics, context_ratio, p_value, etc.
    """
    if pair_ids is not None and len(pair_ids) != len(height_tier):
        raise ValueError("pair_ids must align one-to-one with height_tier.")

    motif_ids = np.unique(y_motif[y_motif > 0])

    # Build syllable × tier occupancy
    occupancy: Dict[Tuple[int, int], int] = {}
    for m in motif_ids:
        for tier in range(3):
            mask = (y_motif == m) & (height_tier == tier)
            count = int(mask.sum())
            if count >= min_bins:
                occupancy[(int(m), tier)] = count

    if len(occupancy) < min_cells:
        return {"status": "insufficient_data", "n_cells": len(occupancy)}

    # Compute within- and between-context distances
    within_dists: List[float] = []
    between_dists: List[float] = []
    syllable_results: List[Dict[str, Any]] = []

    for m in motif_ids:
        tiers_with_data = [
            tier for tier in range(3) if (int(m), tier) in occupancy
        ]
        if len(tiers_with_data) < 2:
            continue

        centroids: Dict[int, np.ndarray] = {}
        for tier in tiers_with_data:
            mask = (y_motif == m) & (height_tier == tier)
            centroids[tier] = np.mean(embedding[mask], axis=0)

        for tier in tiers_with_data:
            mask = (y_motif == m) & (height_tier == tier)
            points = embedding[mask]
            dists = np.linalg.norm(points - centroids[tier], axis=1)
            within_dists.extend(dists.tolist())

        for i, t1 in enumerate(tiers_with_data):
            for t2 in tiers_with_data[i + 1:]:
                d = float(np.linalg.norm(centroids[t1] - centroids[t2]))
                between_dists.append(d)
                syllable_results.append({
                    "motif": int(m),
                    "tier1": t1,
                    "tier2": t2,
                    "centroid_distance": d,
                    "n_tier1": occupancy[(int(m), t1)],
                    "n_tier2": occupancy[(int(m), t2)],
                })

    mean_within = float(np.mean(within_dists)) if within_dists else 0.0
    mean_between = float(np.mean(between_dists)) if between_dists else 0.0
    ratio = mean_between / mean_within if mean_within > 0 else 0.0

    # Permutation test: shuffle tier within each syllable
    rng = np.random.default_rng(seed)
    null_ratios = []
    for _ in range(n_perms):
        if pair_ids is None:
            shuffled_tier = height_tier.copy()
            for m in motif_ids:
                mask = y_motif == m
                idx = np.where(mask)[0]
                shuffled_tier[idx] = rng.permutation(shuffled_tier[idx])
        else:
            shuffled_tier = permute_tiers_within_pairs(
                height_tier=height_tier,
                pair_ids=pair_ids,
                rng=rng,
            )

        null_within: List[float] = []
        null_between: List[float] = []
        for m in motif_ids:
            tiers_ok = [
                tier for tier in range(3)
                if np.sum((y_motif == m) & (shuffled_tier == tier)) >= min_bins
            ]
            if len(tiers_ok) < 2:
                continue
            centroids_null: Dict[int, np.ndarray] = {}
            for tier in tiers_ok:
                mask_t = (y_motif == m) & (shuffled_tier == tier)
                centroids_null[tier] = np.mean(embedding[mask_t], axis=0)
            for tier in tiers_ok:
                mask_t = (y_motif == m) & (shuffled_tier == tier)
                null_within.extend(
                    np.linalg.norm(
                        embedding[mask_t] - centroids_null[tier], axis=1
                    ).tolist()
                )
            for i, t1 in enumerate(tiers_ok):
                for t2 in tiers_ok[i + 1:]:
                    null_between.append(
                        float(np.linalg.norm(
                            centroids_null[t1] - centroids_null[t2]
                        ))
                    )

        nw = np.mean(null_within) if null_within else 1e-6
        nb = np.mean(null_between) if null_between else 0.0
        null_ratios.append(nb / nw if nw > 0 else 0.0)

    p_value = float(np.mean(np.array(null_ratios) >= ratio))

    return {
        "status": "ok",
        "n_cells": len(occupancy),
        "mean_within_distance": mean_within,
        "mean_between_distance": mean_between,
        "context_ratio": ratio,
        "p_value": p_value,
        "n_perms": n_perms,
        "syllable_results": syllable_results,
        "occupancy": {f"{m}_{t}": c for (m, t), c in occupancy.items()},
    }


# ── Matched Context Ratio ────────────────────────────────────────────────

def compute_matched_context_ratio(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    features: np.ndarray,
    config: MatchedContextConfig,
) -> Optional[MatchedContextResult]:
    """Compute context ratio on matched bout subsets.

    For each motif with ≥2 height tiers:
    1. Segment into bouts
    2. Compute mid-bout representative features
    3. Standardize features within motif
    4. Match bouts across tiers (NN, caliper, no replacement)
    5. Recompute context ratio on matched bin indices

    Args:
        embedding: ``(n_bins, n_dim)`` latent embedding.
        y_motif: ``(n_bins,)`` motif labels.
        height_tier: ``(n_bins,)`` height tier.
        features: ``(n_bins, 14)`` matching features (no com_z).
        config: Analysis configuration.

    Returns:
        MatchedContextResult, or None if insufficient data.
    """
    all_bouts = segment_bouts(y_motif)

    # Group bouts by motif (skip motif 0)
    motif_bouts: Dict[int, List[Tuple[int, int, int]]] = {}
    for bout in all_bouts:
        mid = bout[2]
        if mid > 0:
            motif_bouts.setdefault(mid, []).append(bout)

    all_matched_bins: List[int] = []
    all_matched_tiers: List[int] = []
    all_pair_ids: List[int] = []
    all_smd_rows: List[pd.DataFrame] = []
    per_motif_results: List[Dict[str, Any]] = []
    total_minority = 0
    total_matched = 0
    next_pair_id = 0

    for motif_id, bouts in motif_bouts.items():
        # Get tier for each bout
        bout_tiers = np.array([
            int(np.median(height_tier[b[0]:b[1]])) for b in bouts
        ])
        unique_tiers = np.unique(bout_tiers)

        # Keep only tiers with enough bouts
        tier_counts = {int(t): int((bout_tiers == t).sum()) for t in unique_tiers}
        eligible_tiers = [t for t, c in tier_counts.items()
                          if c >= config.min_bouts_per_tier]
        if len(eligible_tiers) < 2:
            continue

        # Filter bouts to eligible tiers only
        keep_mask = np.isin(bout_tiers, eligible_tiers)
        bouts = [b for b, keep in zip(bouts, keep_mask) if keep]
        bout_tiers = bout_tiers[keep_mask]
        tier_counts = {t: c for t, c in tier_counts.items() if t in eligible_tiers}

        # Compute bout-level features
        bout_feats = compute_bout_features(features, bouts)

        # Standardize within motif (z-score across all tiers)
        means = bout_feats.mean(axis=0)
        stds = bout_feats.std(axis=0)
        stds = np.where(stds < 1e-12, 1.0, stds)
        bout_feats_z = (bout_feats - means) / stds

        # Match bouts
        matched_maj, matched_min, frac = match_bouts_within_motif(
            bout_feats_z, bout_tiers, caliper=config.caliper,
        )

        if len(matched_min) < 2:
            continue

        minority_count = min(tier_counts.values())
        total_minority += minority_count
        total_matched += len(matched_min)

        # Covariate balance on matched bouts (raw features)
        feat_a = bout_feats[matched_maj]
        feat_b = bout_feats[matched_min]
        balance = compute_covariate_balance(feat_a, feat_b, MATCHING_COLS)
        all_smd_rows.append(balance)

        # Collect matched bouts and preserve their pair identity for the null.
        for maj_idx, min_idx in zip(matched_maj, matched_min):
            pair_id = next_pair_id
            next_pair_id += 1

            maj_bout = bouts[maj_idx]
            maj_len = maj_bout[1] - maj_bout[0]
            all_matched_bins.extend(range(maj_bout[0], maj_bout[1]))
            all_matched_tiers.extend([int(bout_tiers[maj_idx])] * maj_len)
            all_pair_ids.extend([pair_id] * maj_len)

            min_bout = bouts[min_idx]
            min_len = min_bout[1] - min_bout[0]
            all_matched_bins.extend(range(min_bout[0], min_bout[1]))
            all_matched_tiers.extend([int(bout_tiers[min_idx])] * min_len)
            all_pair_ids.extend([pair_id] * min_len)

        per_motif_results.append({
            "motif_id": motif_id,
            "n_matched": len(matched_min),
            "matched_fraction": frac,
            "tier_counts": tier_counts,
            "max_smd": float(balance["smd"].abs().max()),
        })

    if not all_matched_bins:
        return None

    # Combine matched bins and recompute context ratio
    matched_bins = np.array(all_matched_bins, dtype=int)
    matched_motifs = y_motif[matched_bins]
    matched_tiers = np.array(all_matched_tiers, dtype=int)
    matched_pair_ids = np.array(all_pair_ids, dtype=int)
    matched_emb = embedding[matched_bins]

    # Compute context ratio on matched subset (min_cells=2: 1 motif × 2 tiers ok)
    result_dict = compute_context_coding(
        matched_emb, matched_motifs, matched_tiers,
        min_bins=3, n_perms=config.n_perms, min_cells=2,
        pair_ids=matched_pair_ids,
    )

    if result_dict["status"] != "ok":
        return None

    # Aggregate SMD table
    smd_table = pd.concat(all_smd_rows, ignore_index=True)
    smd_agg = smd_table.groupby("feature")["smd"].apply(
        lambda x: float(np.mean(np.abs(x)))
    ).reset_index()
    smd_agg.columns = ["feature", "smd"]

    matched_fraction = total_matched / total_minority if total_minority > 0 else 0.0

    return MatchedContextResult(
        matched_ratio=result_dict["context_ratio"],
        p_value=result_dict["p_value"],
        n_matched_bouts=total_matched,
        matched_fraction=matched_fraction,
        smd_table=smd_agg,
        per_motif_results=per_motif_results,
    )


# ── Next-Syllable Distribution ───────────────────────────────────────────

def compute_next_syllable_distribution(
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    motif_id: int,
    tier_id: int,
    n_classes: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute Dirichlet-smoothed next-syllable distribution for a state.

    A state is defined as (motif_id, tier_id). Transitions are detected
    where the motif label changes from motif_id to a different value.

    Args:
        y_motif: ``(n_bins,)`` motif labels.
        height_tier: ``(n_bins,)`` height tier labels.
        motif_id: Current motif identity.
        tier_id: Current height tier.
        n_classes: Total number of motif classes (distribution size).
        alpha: Dirichlet smoothing parameter (add-alpha).

    Returns:
        Probability distribution of shape ``(n_classes,)`` summing to 1.
    """
    counts = np.full(n_classes, alpha, dtype=float)

    # Find bins where this state transitions to something else
    mask = (y_motif == motif_id) & (height_tier == tier_id)
    state_idx = np.where(mask)[0]

    for idx in state_idx:
        if idx + 1 < len(y_motif) and y_motif[idx + 1] != motif_id:
            next_label = int(y_motif[idx + 1])
            if 0 <= next_label < n_classes:
                counts[next_label] += 1.0

    total = counts.sum()
    if total > 0:
        return counts / total
    return np.full(n_classes, 1.0 / n_classes)


# ── Affordance Alignment ─────────────────────────────────────────────────

def compute_affordance_alignment(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    features: np.ndarray,
    config: MatchedContextConfig,
) -> Optional[AffordanceResult]:
    """Test if neural distance correlates with future behavioral divergence.

    For each pair of states (same motif, different tier):
    1. Neural centroid distance in embedding space
    2. JS divergence of next-syllable distributions
    3. Kinematic centroid distance (control)
    4. Spearman correlation between neural_dist and future_JS

    Args:
        embedding: ``(n_bins, n_dim)`` latent embedding.
        y_motif: ``(n_bins,)`` motif labels.
        height_tier: ``(n_bins,)`` height tier.
        features: ``(n_bins, 14)`` matching features.
        config: Analysis configuration.

    Returns:
        AffordanceResult, or None if insufficient data.
    """
    motif_ids = np.unique(y_motif[y_motif > 0])
    n_classes = int(y_motif.max()) + 1
    pair_data: List[Dict[str, Any]] = []

    for m in motif_ids:
        # Find tiers with enough data
        tiers_with_data = []
        for tier in range(3):
            mask = (y_motif == m) & (height_tier == tier)
            if mask.sum() >= 10:  # minimum bins
                tiers_with_data.append(tier)

        if len(tiers_with_data) < 2:
            continue

        # Compute centroids and distributions per tier
        centroids: Dict[int, np.ndarray] = {}
        kin_centroids: Dict[int, np.ndarray] = {}
        distributions: Dict[int, np.ndarray] = {}
        n_transitions: Dict[int, int] = {}

        for tier in tiers_with_data:
            mask = (y_motif == m) & (height_tier == tier)
            centroids[tier] = np.mean(embedding[mask], axis=0)
            kin_centroids[tier] = np.mean(features[mask], axis=0)

            dist = compute_next_syllable_distribution(
                y_motif, height_tier, int(m), tier, n_classes, alpha=1.0,
            )
            distributions[tier] = dist

            # Count actual transitions (not smoothed)
            state_idx = np.where(mask)[0]
            n_trans = sum(
                1 for idx in state_idx
                if idx + 1 < len(y_motif) and y_motif[idx + 1] != m
            )
            n_transitions[tier] = n_trans

        # Generate all tier pairs
        for i, t1 in enumerate(tiers_with_data):
            for t2 in tiers_with_data[i + 1:]:
                # Check transition counts
                if (n_transitions[t1] < config.min_transitions
                        or n_transitions[t2] < config.min_transitions):
                    continue

                neural_dist = float(np.linalg.norm(
                    centroids[t1] - centroids[t2]
                ))
                kin_dist = float(np.linalg.norm(
                    kin_centroids[t1] - kin_centroids[t2]
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
                    "kin_dist": kin_dist,
                    "n_trans_a": n_transitions[t1],
                    "n_trans_b": n_transitions[t2],
                })

    if len(pair_data) < 2:
        return None

    neural_dists = np.array([p["neural_dist"] for p in pair_data])
    future_js_vals = np.array([p["future_js"] for p in pair_data])

    rho, p_val = spearmanr(neural_dists, future_js_vals)

    return AffordanceResult(
        per_session_rho=float(rho),
        per_session_p=float(p_val),
        n_pairs=len(pair_data),
        per_pair_data=pair_data,
    )
