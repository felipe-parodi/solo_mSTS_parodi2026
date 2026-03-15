"""Neuro-ethogram: per-unit motif modulation via Cohen's d.

Computes one-vs-rest Cohen's d for each unit x motif combination,
using a globally consistent motif vocabulary (raw llac values from
shared ethogram clustering).

Core Algorithms:

    Global Motif Vocabulary:
        Pool raw llac frame counts across all sessions, exclude motif 0,
        filter to motifs present in >= min_sessions, keep top-N by
        global frequency.

    Cohen's d (one-vs-rest):
        d = (mean_motif - mean_rest) / pooled_sd
        pooled_sd = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
        NaN if motif or rest group has < min_bins samples.

    Sorting:
        Motifs: by median |d| across units (weakest to strongest).
        Units: by max |d| within bank groups (ventral top, dorsal bottom).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from analysis.syllables.segmentation import remap_to_top_n


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GlobalMotifVocabulary:
    """Globally ranked motif vocabulary across sessions.

    Attributes:
        top_n_ids: Globally ranked top-N motif IDs (raw llac values).
        global_counts: Raw llac -> total frame count across all sessions.
        per_session_counts: {session_id: {llac: count}}.
        session_presence: {llac: number of sessions containing it}.
    """

    top_n_ids: List[int]
    global_counts: Dict[int, int]
    per_session_counts: Dict[str, Dict[int, int]]
    session_presence: Dict[int, int]


@dataclass
class NeuroEthogramResult:
    """Per-session neuro-ethogram result.

    Attributes:
        cohens_d: (n_units, n_motifs) raw Cohen's d values (NaN = insufficient data).
        motif_ids: Motif IDs in column order (global vocabulary).
        motif_sort_order: Motif column indices sorted by median |d| (weakest to strongest).
        unit_sort_order: Unit row indices sorted by bank then max |d|.
        unit_ids: Unit IDs in row order.
        unit_banks: Bank labels per unit ('ventral' or 'dorsal').
        summary_per_motif: DataFrame with median|d|, CI, frac>=0.5, frac>=0.8.
        session_id: Session identifier.
    """

    cohens_d: np.ndarray
    motif_ids: List[int]
    motif_sort_order: np.ndarray
    unit_sort_order: np.ndarray
    unit_ids: np.ndarray
    unit_banks: np.ndarray
    summary_per_motif: pd.DataFrame
    session_id: str


# ---------------------------------------------------------------------------
# Global motif vocabulary
# ---------------------------------------------------------------------------

def build_global_motif_vocabulary(
    session_motif_ids: Dict[str, np.ndarray],
    top_n: int = 10,
    exclude_ids: Tuple[int, ...] = (0,),
    min_sessions: int = 2,
) -> GlobalMotifVocabulary:
    """Build globally ranked motif vocabulary from all sessions.

    Args:
        session_motif_ids: {session_id: 1D array of raw llac frame labels}.
        top_n: Number of top motifs to keep.
        exclude_ids: Motif IDs to exclude (e.g., 0 = noise/undefined).
        min_sessions: Motif must appear in >= this many sessions.

    Returns:
        GlobalMotifVocabulary with ranked top-N motif IDs.
    """
    global_counts: Dict[int, int] = {}
    per_session_counts: Dict[str, Dict[int, int]] = {}
    session_presence: Dict[int, int] = {}

    for sid, motif_ids in session_motif_ids.items():
        counts: Dict[int, int] = {}
        for mid in motif_ids:
            mid_int = int(mid)
            if mid_int in exclude_ids:
                continue
            counts[mid_int] = counts.get(mid_int, 0) + 1
            global_counts[mid_int] = global_counts.get(mid_int, 0) + 1

        per_session_counts[sid] = counts

        # Track session presence
        for mid_int in counts:
            session_presence[mid_int] = session_presence.get(mid_int, 0) + 1

    # Filter to motifs present in >= min_sessions
    eligible = {
        mid: count for mid, count in global_counts.items()
        if session_presence.get(mid, 0) >= min_sessions
    }

    # Select top-N by global frequency
    sorted_motifs = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
    top_n_ids = [mid for mid, _ in sorted_motifs[:top_n]]

    print("\n=== Global Motif Vocabulary ===")
    print(f"Total unique motifs (excl {exclude_ids}): {len(global_counts)}")
    print(f"Eligible (>= {min_sessions} sessions): {len(eligible)}")
    print(f"Top-{top_n} selected: {top_n_ids}")
    for mid in top_n_ids:
        print(
            f"  Motif {mid}: {global_counts[mid]} frames, "
            f"{session_presence[mid]} sessions"
        )
    print()

    return GlobalMotifVocabulary(
        top_n_ids=top_n_ids,
        global_counts=global_counts,
        per_session_counts=per_session_counts,
        session_presence=session_presence,
    )


def remap_global_top_n(
    bin_labels: np.ndarray,
    global_vocab: GlobalMotifVocabulary,
    other_id: int = -1,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Remap bin labels using globally ranked vocabulary.

    Reuses existing remap_to_top_n() with pre-computed global counts.

    Args:
        bin_labels: 1D array of raw motif IDs per bin.
        global_vocab: Global vocabulary with frequency counts.
        other_id: ID for non-top-N motifs.

    Returns:
        Tuple of (remapped_labels, mapping_dict).
    """
    return remap_to_top_n(
        bin_labels,
        n=len(global_vocab.top_n_ids),
        other_id=other_id,
        frequency_counts=global_vocab.global_counts,
    )


# ---------------------------------------------------------------------------
# Cohen's d computation
# ---------------------------------------------------------------------------

def compute_cohens_d_one_vs_rest(
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    motif_ids: List[int],
    min_bins: int = 30,
) -> np.ndarray:
    """Compute one-vs-rest Cohen's d for each unit x motif.

    Args:
        X: Firing rate matrix (n_bins, n_units).
        y: Motif labels per bin (n_bins,).
        valid_mask: Boolean mask for bins to include (n_bins,).
        motif_ids: List of motif IDs to evaluate.
        min_bins: Minimum bins in motif or rest group; NaN if below.

    Returns:
        (n_units, n_motifs) array of raw Cohen's d values.
    """
    n_units = X.shape[1]
    n_motifs = len(motif_ids)
    d_matrix = np.full((n_units, n_motifs), np.nan)

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    for j, mid in enumerate(motif_ids):
        motif_mask = y_valid == mid
        rest_mask = ~motif_mask

        n_motif = motif_mask.sum()
        n_rest = rest_mask.sum()

        if n_motif < min_bins or n_rest < min_bins:
            continue  # Leave as NaN

        X_motif = X_valid[motif_mask]
        X_rest = X_valid[rest_mask]

        # Use nan-aware stats (residuals may have NaN for some unit x bin)
        mean_motif = np.nanmean(X_motif, axis=0)
        mean_rest = np.nanmean(X_rest, axis=0)
        n_motif_per_unit = np.sum(~np.isnan(X_motif), axis=0)
        n_rest_per_unit = np.sum(~np.isnan(X_rest), axis=0)
        var_motif = np.nanvar(X_motif, axis=0, ddof=1)
        var_rest = np.nanvar(X_rest, axis=0, ddof=1)

        # Pooled standard deviation (per-unit effective sample sizes)
        pooled_var = (
            (n_motif_per_unit - 1) * var_motif
            + (n_rest_per_unit - 1) * var_rest
        ) / (n_motif_per_unit + n_rest_per_unit - 2)
        pooled_sd = np.sqrt(pooled_var)

        # Avoid division by zero; also require min_bins non-NaN per unit
        nonzero_sd = (
            (pooled_sd > 1e-12)
            & (n_motif_per_unit >= min_bins)
            & (n_rest_per_unit >= min_bins)
        )
        d_matrix[nonzero_sd, j] = (
            (mean_motif[nonzero_sd] - mean_rest[nonzero_sd])
            / pooled_sd[nonzero_sd]
        )

    return d_matrix


# ---------------------------------------------------------------------------
# Full neuro-ethogram computation
# ---------------------------------------------------------------------------

def compute_neuroethogram(
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    motif_ids: List[int],
    unit_ids: np.ndarray,
    unit_banks: np.ndarray,
    session_id: str = "",
    min_bins: int = 30,
) -> NeuroEthogramResult:
    """Compute full neuro-ethogram for one session.

    Args:
        X: Firing rate matrix (n_bins, n_units).
        y: Motif labels per bin (n_bins,).
        valid_mask: Boolean mask for valid bins.
        motif_ids: Global motif IDs to evaluate.
        unit_ids: Unit identifiers (n_units,).
        unit_banks: Bank labels per unit ('ventral'/'dorsal').
        session_id: Session identifier.
        min_bins: Minimum bins for Cohen's d.

    Returns:
        NeuroEthogramResult with sorted Cohen's d matrix and summaries.
    """
    d_matrix = compute_cohens_d_one_vs_rest(
        X, y, valid_mask, motif_ids, min_bins=min_bins,
    )

    # Sort motifs by median |d| (weakest to strongest)
    median_abs_d = np.nanmedian(np.abs(d_matrix), axis=0)
    motif_sort_order = np.argsort(median_abs_d)  # ascending = weakest first

    # Sort units: ventral on top, dorsal on bottom; within each bank by max|d| descending
    max_abs_d = np.nanmax(np.abs(d_matrix), axis=1)
    max_abs_d = np.where(np.isnan(max_abs_d), 0.0, max_abs_d)

    # Bank order: ventral=0, dorsal=1
    bank_order = np.array([
        0 if b == "ventral" else 1 for b in unit_banks
    ])
    # Sort key: bank_order ascending, then max|d| descending within bank
    sort_key = bank_order * 1e6 - max_abs_d  # ventral first, highest |d| first
    unit_sort_order = np.argsort(sort_key)

    # Summary stats per motif
    summary_rows = []
    for j, mid in enumerate(motif_ids):
        col = d_matrix[:, j]
        valid_d = col[~np.isnan(col)]

        if len(valid_d) == 0:
            summary_rows.append({
                "motif_id": mid,
                "median_abs_d": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "frac_d_ge_05": np.nan,
                "frac_d_ge_08": np.nan,
                "n_valid_units": 0,
            })
            continue

        abs_d = np.abs(valid_d)
        median_val = float(np.median(abs_d))

        # Bootstrap 95% CI for median |d|
        n_boot = 1000
        boot_rng = np.random.default_rng(42)
        boot_medians = np.array([
            np.median(boot_rng.choice(abs_d, size=len(abs_d), replace=True))
            for _ in range(n_boot)
        ])
        ci_low = float(np.percentile(boot_medians, 2.5))
        ci_high = float(np.percentile(boot_medians, 97.5))

        summary_rows.append({
            "motif_id": mid,
            "median_abs_d": median_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "frac_d_ge_05": float(np.mean(abs_d >= 0.5)),
            "frac_d_ge_08": float(np.mean(abs_d >= 0.8)),
            "n_valid_units": len(valid_d),
        })

    summary_df = pd.DataFrame(summary_rows)

    return NeuroEthogramResult(
        cohens_d=d_matrix,
        motif_ids=list(motif_ids),
        motif_sort_order=motif_sort_order,
        unit_sort_order=unit_sort_order,
        unit_ids=unit_ids,
        unit_banks=unit_banks,
        summary_per_motif=summary_df,
        session_id=session_id,
    )


__all__ = [
    "GlobalMotifVocabulary",
    "NeuroEthogramResult",
    "build_global_motif_vocabulary",
    "remap_global_top_n",
    "compute_cohens_d_one_vs_rest",
    "compute_neuroethogram",
]
