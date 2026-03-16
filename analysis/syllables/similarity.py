"""Syllable similarity vs neural confusion analysis.

Correlates pose-based syllable similarity (cosine distance) with neural
decoding confusion (symmetrized confusion matrix) using Mantel-style
permutation test.

Hypothesis: behaviorally similar syllables (low cosine distance) should be
more confused by neural decoders → negative Spearman rho.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def load_syllable_similarity(
    motifs_dir: str,
    labels_subset: Optional[List[int]] = None,
) -> tuple[np.ndarray, List[int]]:
    """Load syllable similarity distance matrix from motifs output.

    Args:
        motifs_dir: Path to motifs output directory containing
            similarity_distances.npz.
        labels_subset: If provided, subset to these labels only.

    Returns:
        Tuple of (distance_matrix, labels).
    """
    npz_path = Path(motifs_dir) / "similarity_distances.npz"
    data = np.load(npz_path)
    D = data["D"].astype(float)
    labels = data["labels"].tolist()

    if labels_subset is not None:
        idx = [labels.index(lb) for lb in labels_subset if lb in labels]
        matched_labels = [labels[i] for i in idx]
        D = D[np.ix_(idx, idx)]
        return D, matched_labels

    return D, labels


def symmetrize_confusion(conf_matrix: np.ndarray) -> np.ndarray:
    """Symmetrize a confusion matrix: conf_sym[i,j] = 0.5 * (conf[i,j] + conf[j,i]).

    Args:
        conf_matrix: (N, N) confusion matrix (may be asymmetric).

    Returns:
        Symmetric (N, N) confusion matrix.
    """
    return 0.5 * (conf_matrix + conf_matrix.T)


def _upper_triangle_vec(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle off-diagonal elements as a flat vector."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def correlate_similarity_confusion(
    sim_matrix: np.ndarray,
    conf_matrix: np.ndarray,
    labels: List[int],
    n_perm: int = 1000,
    *,
    rng_seed: int = 42,
) -> Dict[str, object]:
    """Correlate syllable similarity with neural confusion (Mantel-style).

    Args:
        sim_matrix: (N, N) symmetric distance matrix (higher = more dissimilar).
        conf_matrix: (N, N) symmetric confusion matrix (higher = more confused).
        labels: Label ordering for both matrices.
        n_perm: Number of Mantel-style permutations (0 = skip).
        rng_seed: Random seed for permutations.

    Returns:
        Dict with keys: rho, p_one_sided, p_two_sided, null_dist, sim_vec, conf_vec.
    """
    sim_vec = _upper_triangle_vec(sim_matrix)
    conf_vec = _upper_triangle_vec(conf_matrix)

    rho_obs, _ = stats.spearmanr(sim_vec, conf_vec)

    if n_perm <= 0:
        return {
            "rho": float(rho_obs),
            "p_one_sided": float("nan"),
            "p_two_sided": float("nan"),
            "null_dist": np.array([], dtype=float),
            "sim_vec": sim_vec,
            "conf_vec": conf_vec,
        }

    # Mantel-style permutation: permute label identities (rows+cols together)
    rng = np.random.default_rng(rng_seed)
    n = sim_matrix.shape[0]
    null_rhos = np.zeros(n_perm, dtype=float)

    for i in range(n_perm):
        perm = rng.permutation(n)
        sim_perm = sim_matrix[np.ix_(perm, perm)]
        sim_vec_perm = _upper_triangle_vec(sim_perm)
        null_rhos[i], _ = stats.spearmanr(sim_vec_perm, conf_vec)

    # One-sided: proportion of null rhos ≤ observed (testing negative rho)
    p_one_sided = float((np.sum(null_rhos <= rho_obs) + 1) / (n_perm + 1))
    # Two-sided: proportion of null |rhos| ≥ |observed|
    p_two_sided = float((np.sum(np.abs(null_rhos) >= np.abs(rho_obs)) + 1) / (n_perm + 1))

    return {
        "rho": float(rho_obs),
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "null_dist": null_rhos,
        "sim_vec": sim_vec,
        "conf_vec": conf_vec,
    }


def summarize_across_sessions(
    per_session_results: List[Dict[str, object]],
) -> Dict[str, object]:
    """Summarize per-session Spearman rho with median + sign test.

    Args:
        per_session_results: List of dicts, each with 'rho' and 'session_id'.

    Returns:
        Dict with median_rho, n_sessions, n_negative, sign_test_p.
    """
    rhos = np.array([r["rho"] for r in per_session_results], dtype=float)
    n = len(rhos)
    n_negative = int(np.sum(rhos < 0))

    # Sign test: binomial test for proportion of negative rhos
    sign_test_result = stats.binomtest(n_negative, n, p=0.5, alternative="greater")

    return {
        "median_rho": float(np.nanmedian(rhos)),
        "mean_rho": float(np.nanmean(rhos)),
        "n_sessions": n,
        "n_negative": n_negative,
        "sign_test_p": float(sign_test_result.pvalue),
        "per_session_rhos": rhos.tolist(),
    }


__all__ = [
    "load_syllable_similarity",
    "symmetrize_confusion",
    "correlate_similarity_confusion",
    "summarize_across_sessions",
]
