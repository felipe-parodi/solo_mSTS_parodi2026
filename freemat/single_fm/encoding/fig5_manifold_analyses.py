"""Fig 5 manifold analyses: post-hoc computations on existing CEBRA embeddings.

Analyses:
  1. Embedding RSA — centroid distances in CEBRA space vs kinematic similarity
  4. Anticipatory drift — cosine similarity to next-motif centroid before transition
  6. Factored vs conjunctive coding — per-dimension variance explained by space vs motif
  7. Velocity → transition probability — does neural speed predict upcoming transitions
  8. Timescale separation — autocorrelation decay of neural trajectory vs behavior
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

try:
    from freemat.single_fm.viz.pub_data import (
        ALL_SESSIONS,
        DEFAULT_FIG5_DIR,
        load_fig5_embeddings,
    )
except ImportError:
    ALL_SESSIONS = []
    DEFAULT_FIG5_DIR = None
    load_fig5_embeddings = None


# ══════════════════════════════════════════════════════════════════════════
# Analysis 1: Embedding RSA
# ══════════════════════════════════════════════════════════════════════════

def compute_embedding_rsa(
    emb: np.ndarray,
    y_motif: np.ndarray,
    kin_sim_vec: Optional[np.ndarray] = None,
    n_perm: int = 1000,
) -> Dict[str, Any]:
    """Correlate pairwise motif centroid distances in embedding space with kinematic similarity.

    Args:
        emb: Embedding array, shape (n_bins, n_dims).
        y_motif: Motif labels, shape (n_bins,). 0 = invalid.
        kin_sim_vec: Kinematic similarity vector (n_pairs,) from fig4_similarity.
            If None, only centroid distances are returned.
        n_perm: Number of permutations for null distribution.

    Returns:
        Dict with centroid_distances (condensed), rho, p_value, null_dist,
        motif_labels (sorted unique), centroid_distance_matrix.
    """
    valid = y_motif > 0
    emb_valid = emb[valid]
    labels_valid = y_motif[valid]

    unique_motifs = np.sort(np.unique(labels_valid))
    n_classes = len(unique_motifs)

    # Compute centroids
    centroids = np.zeros((n_classes, emb.shape[1]))
    for i, mid in enumerate(unique_motifs):
        centroids[i] = emb_valid[labels_valid == mid].mean(axis=0)

    # Pairwise Euclidean distances between centroids (condensed form)
    centroid_dist_vec = pdist(centroids, metric="euclidean")
    centroid_dist_matrix = squareform(centroid_dist_vec)

    result: Dict[str, Any] = {
        "motif_labels": unique_motifs.tolist(),
        "n_classes": n_classes,
        "centroid_distance_vec": centroid_dist_vec.tolist(),
        "centroid_distance_matrix": centroid_dist_matrix.tolist(),
    }

    # Handle motif count mismatch: if kin_sim_vec has fewer pairs, subset
    # centroids to the N most frequent motifs matching the expected count.
    rsa_centroids = centroids
    rsa_n = n_classes
    if (kin_sim_vec is not None
            and len(kin_sim_vec) != len(centroid_dist_vec)
            and len(kin_sim_vec) > 0):
        n_expected = int(round((1 + (1 + 8 * len(kin_sim_vec)) ** 0.5) / 2))
        if n_expected < n_classes:
            # Keep the n_expected most frequent motifs
            counts = [(mid, int(np.sum(labels_valid == mid))) for mid in unique_motifs]
            counts.sort(key=lambda x: -x[1])
            keep_motifs = np.sort([m for m, _ in counts[:n_expected]])
            rsa_centroids = np.array([
                emb_valid[labels_valid == mid].mean(axis=0) for mid in keep_motifs
            ])
            rsa_n = n_expected
            centroid_dist_vec = pdist(rsa_centroids, metric="euclidean")
            result["motif_labels_rsa"] = keep_motifs.tolist()
            result["n_classes_rsa"] = n_expected
            print(f"    [RSA] Subsetted {n_classes} → {n_expected} motifs "
                  f"(dropped {set(unique_motifs) - set(keep_motifs)})")

    if kin_sim_vec is not None and len(kin_sim_vec) == len(centroid_dist_vec):
        # Correlate: higher kinematic similarity should mean smaller neural distance
        # (or vice versa depending on sign convention)
        rho, p_param = stats.spearmanr(kin_sim_vec, centroid_dist_vec)

        # Permutation null: shuffle centroid labels
        null_rhos = np.zeros(n_perm)
        for p in range(n_perm):
            perm_idx = np.random.permutation(rsa_n)
            perm_centroids = rsa_centroids[perm_idx]
            perm_dist = pdist(perm_centroids, metric="euclidean")
            null_rhos[p], _ = stats.spearmanr(kin_sim_vec, perm_dist)

        p_perm = float((1 + np.sum(np.abs(null_rhos) >= np.abs(rho))) / (1 + n_perm))

        result.update({
            "rho": float(rho),
            "p_parametric": float(p_param),
            "p_permutation": p_perm,
            "null_rhos_mean": float(np.mean(null_rhos)),
            "null_rhos_std": float(np.std(null_rhos)),
        })
    else:
        result["rho"] = None
        result["p_permutation"] = None

    return result


# ══════════════════════════════════════════════════════════════════════════
# Analysis 4: Anticipatory drift
# ══════════════════════════════════════════════════════════════════════════

def compute_anticipatory_drift(
    emb: np.ndarray,
    y_motif: np.ndarray,
    window_bins: int = 4,
    n_perm: int = 500,
) -> Dict[str, Any]:
    """Measure drift toward next-motif centroid before behavioral transitions.

    For each transition, compute cosine similarity between neural state
    and the next-motif centroid over a time window. Compare pre-transition
    drift toward next-motif vs toward current-motif centroid.

    Args:
        emb: Embedding array, shape (n_bins, n_dims).
        y_motif: Motif labels, shape (n_bins,). 0 = invalid.
        window_bins: Half-window size in bins (e.g., 4 = +/- 2s at 0.5s bins).
        n_perm: Number of circular-shift permutations for null.

    Returns:
        Dict with time_axis, mean_sim_next, mean_sim_current, drift_index,
        n_transitions, p_value.
    """
    valid = y_motif > 0
    unique_motifs = np.sort(np.unique(y_motif[valid]))

    # Compute centroids
    centroids: Dict[int, np.ndarray] = {}
    for mid in unique_motifs:
        mask = y_motif == mid
        centroids[int(mid)] = emb[mask].mean(axis=0)

    # Find transitions
    transitions = []
    for i in range(1, len(y_motif)):
        if (y_motif[i] != y_motif[i - 1]
                and y_motif[i] > 0
                and y_motif[i - 1] > 0
                and i >= window_bins
                and i < len(y_motif) - window_bins):
            transitions.append(i)

    n_trans = len(transitions)
    if n_trans < 10:
        return {
            "n_transitions": n_trans,
            "drift_index": float("nan"),
            "p_value": float("nan"),
            "time_axis": [],
            "mean_sim_next": [],
            "mean_sim_current": [],
        }

    n_timepoints = 2 * window_bins
    sim_next_all = np.zeros((n_trans, n_timepoints))
    sim_current_all = np.zeros((n_trans, n_timepoints))

    for t_idx, trans_bin in enumerate(transitions):
        current_motif = int(y_motif[trans_bin - 1])
        next_motif = int(y_motif[trans_bin])
        c_current = centroids[current_motif]
        c_next = centroids[next_motif]

        for j, offset in enumerate(range(-window_bins, window_bins)):
            bin_idx = trans_bin + offset
            state = emb[bin_idx]

            # Cosine similarity
            sim_next_all[t_idx, j] = _cosine_sim(state, c_next)
            sim_current_all[t_idx, j] = _cosine_sim(state, c_current)

    mean_sim_next = np.mean(sim_next_all, axis=0)
    mean_sim_current = np.mean(sim_current_all, axis=0)
    time_axis = np.arange(-window_bins, window_bins) * 0.5  # assuming 0.5s bins

    # Drift index: increase in next-motif similarity from pre to post transition
    # Compare last pre-transition bin to first post-transition bin
    pre_sim_next = np.mean(sim_next_all[:, :window_bins], axis=1)  # pre-transition avg
    post_sim_next = np.mean(sim_next_all[:, window_bins:], axis=1)  # post-transition avg
    drift_index = float(np.mean(post_sim_next - pre_sim_next))

    # More informative: is there a ramp BEFORE transition?
    # Compare first half of pre-transition to second half of pre-transition
    early_pre = np.mean(sim_next_all[:, :window_bins // 2], axis=1)
    late_pre = np.mean(sim_next_all[:, window_bins // 2:window_bins], axis=1)
    pre_ramp = float(np.mean(late_pre - early_pre))

    # Null: circular shift of motif labels
    null_ramps = np.zeros(n_perm)
    n_bins = len(y_motif)
    for p in range(n_perm):
        shift = np.random.randint(window_bins, n_bins - window_bins)
        y_shifted = np.roll(y_motif, shift)

        # Recompute centroids
        perm_centroids: Dict[int, np.ndarray] = {}
        for mid in unique_motifs:
            mask = y_shifted == mid
            if mask.sum() > 0:
                perm_centroids[int(mid)] = emb[mask].mean(axis=0)

        perm_ramps = []
        for trans_bin in transitions:
            nm = int(y_shifted[trans_bin]) if y_shifted[trans_bin] > 0 else 0
            if nm == 0 or nm not in perm_centroids:
                continue
            c_n = perm_centroids[nm]

            early_vals = []
            late_vals = []
            for j in range(window_bins // 2):
                early_vals.append(_cosine_sim(emb[trans_bin - window_bins + j], c_n))
            for j in range(window_bins // 2):
                late_vals.append(_cosine_sim(emb[trans_bin - window_bins // 2 + j], c_n))
            if early_vals and late_vals:
                perm_ramps.append(np.mean(late_vals) - np.mean(early_vals))

        null_ramps[p] = np.mean(perm_ramps) if perm_ramps else 0.0

    p_value = float((1 + np.sum(null_ramps >= pre_ramp)) / (1 + n_perm))

    return {
        "n_transitions": n_trans,
        "time_axis": time_axis.tolist(),
        "mean_sim_next": mean_sim_next.tolist(),
        "mean_sim_current": mean_sim_current.tolist(),
        "drift_index": drift_index,
        "pre_ramp": pre_ramp,
        "p_value": p_value,
        "null_mean": float(np.mean(null_ramps)),
        "null_std": float(np.std(null_ramps)),
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ══════════════════════════════════════════════════════════════════════════
# Analysis 6: Factored vs conjunctive coding
# ══════════════════════════════════════════════════════════════════════════

def compute_factored_vs_conjunctive(
    emb: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
) -> Dict[str, Any]:
    """Test whether embedding dimensions separate space vs motif (factored) or mix them.

    For each dimension, compute eta-squared (variance explained) by motif
    and by height. If dimensions specialize, the code is factored.
    If most dimensions encode both, the code is conjunctive.

    Args:
        emb: Embedding array, shape (n_bins, n_dims).
        y_motif: Motif labels, shape (n_bins,). 0 = invalid.
        height_tier: Height class labels, shape (n_bins,). 0/1/2.

    Returns:
        Dict with per-dimension eta-squared for motif and height,
        conjunctive index (fraction of dims encoding both), and
        factored index.
    """
    n_dims = emb.shape[1]
    valid = y_motif > 0

    emb_valid = emb[valid]
    motif_valid = y_motif[valid]
    height_valid = height_tier[valid]

    eta_motif = np.zeros(n_dims)
    eta_height = np.zeros(n_dims)

    for d in range(n_dims):
        x = emb_valid[:, d]
        eta_motif[d] = _eta_squared(x, motif_valid)
        eta_height[d] = _eta_squared(x, height_valid)

    # Classify dimensions
    threshold = 0.02  # minimum eta-squared to count as "encoding"
    encodes_motif = eta_motif > threshold
    encodes_height = eta_height > threshold
    encodes_both = encodes_motif & encodes_height
    encodes_either = encodes_motif | encodes_height

    n_encoding = int(encodes_either.sum())
    conjunctive_frac = float(encodes_both.sum() / n_encoding) if n_encoding > 0 else 0.0
    factored_frac = 1.0 - conjunctive_frac

    # Dominance index per dimension: (eta_motif - eta_height) / (eta_motif + eta_height)
    total_eta = eta_motif + eta_height
    dominance = np.where(total_eta > 0, (eta_motif - eta_height) / total_eta, 0.0)

    return {
        "eta_motif": eta_motif.tolist(),
        "eta_height": eta_height.tolist(),
        "dominance_index": dominance.tolist(),
        "conjunctive_fraction": conjunctive_frac,
        "factored_fraction": factored_frac,
        "n_dims_motif_only": int((encodes_motif & ~encodes_height).sum()),
        "n_dims_height_only": int((encodes_height & ~encodes_motif).sum()),
        "n_dims_both": int(encodes_both.sum()),
        "n_dims_neither": int((~encodes_either).sum()),
        "threshold": threshold,
    }


def _eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """Compute eta-squared (one-way ANOVA effect size).

    Args:
        x: Continuous variable, shape (n,).
        groups: Group labels, shape (n,).

    Returns:
        Eta-squared value (0 to 1).
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0

    grand_mean = x.mean()
    ss_between = 0.0
    ss_total = np.sum((x - grand_mean) ** 2)

    if ss_total == 0:
        return 0.0

    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g > 0:
            ss_between += n_g * (x[mask].mean() - grand_mean) ** 2

    return float(ss_between / ss_total)


# ══════════════════════════════════════════════════════════════════════════
# Analysis 7: Velocity → transition probability
# ══════════════════════════════════════════════════════════════════════════

def compute_velocity_transition_prediction(
    emb: np.ndarray,
    y_motif: np.ndarray,
    horizon_bins: int = 4,
    n_perm: int = 500,
) -> Dict[str, Any]:
    """Test whether neural velocity predicts upcoming behavioral transitions.

    Compute neural velocity at each time bin, then correlate with whether
    a transition occurs within the next `horizon_bins` bins.

    Args:
        emb: Embedding array, shape (n_bins, n_dims).
        y_motif: Motif labels, shape (n_bins,). 0 = invalid.
        horizon_bins: Future window to check for transitions.
        n_perm: Circular-shift permutations for null.

    Returns:
        Dict with AUC-ROC, mean velocity at transition vs non-transition,
        p_value.
    """
    # Compute velocity (Euclidean distance between consecutive states)
    velocity = np.linalg.norm(np.diff(emb, axis=0), axis=1)  # shape (n_bins-1,)

    # Build transition label: does a transition occur in the next horizon_bins?
    n_bins = len(y_motif)
    transition_upcoming = np.zeros(n_bins - 1, dtype=bool)

    for i in range(n_bins - 1):
        if y_motif[i] <= 0:
            continue
        for j in range(1, min(horizon_bins + 1, n_bins - i)):
            if y_motif[i + j] > 0 and y_motif[i + j] != y_motif[i]:
                transition_upcoming[i] = True
                break

    # Filter to valid bins only
    valid = np.array([y_motif[i] > 0 for i in range(n_bins - 1)])
    vel_valid = velocity[valid]
    trans_valid = transition_upcoming[valid]

    if trans_valid.sum() < 10 or (~trans_valid).sum() < 10:
        return {
            "auc_roc": float("nan"),
            "p_value": float("nan"),
            "n_transition_bins": int(trans_valid.sum()),
            "n_non_transition_bins": int((~trans_valid).sum()),
        }

    # Compute AUC-ROC
    auc = _compute_auc(vel_valid, trans_valid)

    # Mean velocities
    vel_at_trans = float(np.mean(vel_valid[trans_valid]))
    vel_at_non_trans = float(np.mean(vel_valid[~trans_valid]))

    # Null distribution: circular shift of motif labels
    null_aucs = np.zeros(n_perm)
    for p in range(n_perm):
        shift = np.random.randint(horizon_bins, n_bins - horizon_bins)
        y_shifted = np.roll(y_motif, shift)

        trans_shifted = np.zeros(n_bins - 1, dtype=bool)
        for i in range(n_bins - 1):
            if y_shifted[i] <= 0:
                continue
            for j in range(1, min(horizon_bins + 1, n_bins - i)):
                if y_shifted[i + j] > 0 and y_shifted[i + j] != y_shifted[i]:
                    trans_shifted[i] = True
                    break

        trans_shifted_valid = trans_shifted[valid]
        if trans_shifted_valid.sum() >= 5:
            null_aucs[p] = _compute_auc(vel_valid, trans_shifted_valid)
        else:
            null_aucs[p] = 0.5

    p_value = float((1 + np.sum(null_aucs >= auc)) / (1 + n_perm))

    return {
        "auc_roc": float(auc),
        "mean_velocity_pre_transition": vel_at_trans,
        "mean_velocity_non_transition": vel_at_non_trans,
        "velocity_ratio": vel_at_trans / vel_at_non_trans if vel_at_non_trans > 0 else float("nan"),
        "n_transition_bins": int(trans_valid.sum()),
        "n_non_transition_bins": int((~trans_valid).sum()),
        "p_value": p_value,
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "horizon_s": horizon_bins * 0.5,
    }


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC-ROC from scores and binary labels (Mann-Whitney U statistic)."""
    pos = scores[labels]
    neg = scores[~labels]
    # Mann-Whitney U
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u_stat / (len(pos) * len(neg)))


# ══════════════════════════════════════════════════════════════════════════
# Analysis 8: Timescale separation
# ══════════════════════════════════════════════════════════════════════════

def compute_timescale_separation(
    emb: np.ndarray,
    y_motif: np.ndarray,
    max_lag_bins: int = 20,
) -> Dict[str, Any]:
    """Compare autocorrelation timescales of neural trajectory vs behavioral motif labels.

    Args:
        emb: Embedding array, shape (n_bins, n_dims).
        y_motif: Motif labels, shape (n_bins,). 0 = invalid.
        max_lag_bins: Maximum lag to compute (in bins).

    Returns:
        Dict with neural_autocorr, behavioral_autocorr, lag_axis,
        neural_tau (half-life), behavioral_tau.
    """
    valid = y_motif > 0

    # Neural autocorrelation: use embedding distance from t to t+lag
    # Normalized: autocorr(lag) = 1 - mean_dist(lag) / mean_dist(inf)
    emb_valid = emb[valid]
    n_valid = len(emb_valid)

    # Random baseline distance (shuffle)
    n_sample = min(5000, n_valid)
    idx1 = np.random.randint(0, n_valid, n_sample)
    idx2 = np.random.randint(0, n_valid, n_sample)
    baseline_dist = np.mean(np.linalg.norm(emb_valid[idx1] - emb_valid[idx2], axis=1))

    neural_autocorr = np.zeros(max_lag_bins)
    for lag in range(max_lag_bins):
        dists = np.linalg.norm(emb_valid[:n_valid - lag - 1] - emb_valid[lag + 1:], axis=1)
        neural_autocorr[lag] = 1.0 - np.mean(dists) / baseline_dist

    # Behavioral autocorrelation: fraction of bin pairs at lag with same motif
    motif_valid = y_motif[valid]
    behavioral_autocorr = np.zeros(max_lag_bins)
    for lag in range(max_lag_bins):
        matches = motif_valid[:n_valid - lag - 1] == motif_valid[lag + 1:]
        behavioral_autocorr[lag] = float(np.mean(matches))

    lag_axis = (np.arange(max_lag_bins) + 1) * 0.5  # in seconds

    # Fit exponential decay: A * exp(-t/tau) + C
    neural_fit = _fit_exponential_decay(neural_autocorr, lag_axis)
    behavioral_fit = _fit_exponential_decay(behavioral_autocorr, lag_axis)

    n_hl = neural_fit["halflife"]
    b_hl = behavioral_fit["halflife"]

    return {
        "lag_axis_s": lag_axis.tolist(),
        "neural_autocorr": neural_autocorr.tolist(),
        "behavioral_autocorr": behavioral_autocorr.tolist(),
        "neural_tau_s": neural_fit["tau"],
        "behavioral_tau_s": behavioral_fit["tau"],
        "tau_ratio": neural_fit["tau"] / behavioral_fit["tau"]
        if behavioral_fit["tau"] > 0 else float("nan"),
        "neural_halflife_s": n_hl,
        "behavioral_halflife_s": b_hl,
        "halflife_ratio": n_hl / b_hl if np.isfinite(b_hl) and b_hl > 0 else 0.0,
        "neural_floor": neural_fit["C"],
        "behavioral_floor": behavioral_fit["C"],
        "fit_method": "exponential_A_exp_neg_t_tau_plus_C",
    }


def _fit_exponential_decay(
    autocorr: np.ndarray,
    lag_axis: np.ndarray,
    max_fit_s: float = 5.0,
) -> Dict[str, float]:
    """Fit autocorr(t) = A * exp(-t / tau) + C and derive timescale metrics.

    Fits the early portion of the autocorrelation curve (up to *max_fit_s*)
    to avoid contamination from periodic rebounds at longer lags.

    Returns dict with:
        tau: exponential time constant (seconds)
        A: amplitude of decaying component
        C: asymptotic floor
        halflife: time for full autocorr to reach 50% of initial value
            (inf if floor C is above 50% of initial, meaning the signal
            never fully decorrelates — common for persistent behavioral states)

    Args:
        autocorr: Autocorrelation values, shape (n_lags,).
        lag_axis: Lag values in seconds, shape (n_lags,).
        max_fit_s: Maximum lag to include in fit (seconds).

    Returns:
        Dict with fit parameters and derived metrics.
    """
    from scipy.optimize import curve_fit as _curve_fit

    nan_result = {"tau": float("nan"), "A": float("nan"),
                  "C": float("nan"), "halflife": float("nan")}

    mask = lag_axis <= max_fit_s
    t = lag_axis[mask]
    y = autocorr[mask]
    if len(t) < 3 or y[0] <= 0:
        return nan_result

    def _exp_decay(t: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
        return a * np.exp(-t / tau) + c

    try:
        p0 = [y[0] - y[-1], 3.0, y[-1]]
        bounds = ([0, 0.1, 0], [2.0, 100.0, 1.0])
        popt, _ = _curve_fit(_exp_decay, t, y, p0=p0, bounds=bounds, maxfev=5000)
        a_fit, tau_fit, c_fit = popt

        # Extrapolated half-life: solve A*exp(-t/tau) + C = 0.5 * (A + C)
        # => A*exp(-t/tau) = 0.5*A - 0.5*C = 0.5*(A - C)
        # => t = tau * ln(2*A / (A - C))  [requires A > C]
        if a_fit > c_fit and a_fit > 0:
            halflife = float(tau_fit * np.log(2.0 * a_fit / (a_fit - c_fit)))
        else:
            halflife = float("inf")

        return {
            "tau": float(tau_fit),
            "A": float(a_fit),
            "C": float(c_fit),
            "halflife": halflife,
        }
    except (RuntimeError, ValueError):
        return nan_result


# ══════════════════════════════════════════════════════════════════════════
# Runner: compute all analyses for one session
# ══════════════════════════════════════════════════════════════════════════

def run_all_manifold_analyses(
    session_key: str,
    fig5_dir: Optional[Path] = None,
    kin_sim_vec: Optional[np.ndarray] = None,
    n_perm: int = 500,
    embedding_key: str = "emb_behavior_viz_8d",
) -> Dict[str, Any]:
    """Run all 5 manifold analyses for a single session.

    Args:
        session_key: Session key (e.g., 'M2_0526').
        fig5_dir: Directory with CEBRA results.
        kin_sim_vec: Kinematic similarity vector for embedding RSA.
        n_perm: Permutations for null distributions.
        embedding_key: Which embedding to use.

    Returns:
        Dict with results for each analysis.
    """
    data = load_fig5_embeddings(session_key, fig5_dir=fig5_dir)
    emb = data[embedding_key]
    y_motif = data["y_motif"]
    height_tier = data["height_tier"]

    print(f"  Embedding: {embedding_key}, shape={emb.shape}")
    print(f"  Valid motif bins: {(y_motif > 0).sum()}/{len(y_motif)}")

    results: Dict[str, Any] = {"session_id": session_key}

    # 1. Embedding RSA
    print("  [1] Embedding RSA...")
    results["embedding_rsa"] = compute_embedding_rsa(
        emb, y_motif, kin_sim_vec=kin_sim_vec, n_perm=n_perm,
    )
    rho = results["embedding_rsa"].get("rho")
    p = results["embedding_rsa"].get("p_permutation")
    print(f"      rho={rho}, p={p}")

    # 4. Anticipatory drift
    print("  [4] Anticipatory drift...")
    results["anticipatory_drift"] = compute_anticipatory_drift(
        emb, y_motif, window_bins=4, n_perm=n_perm,
    )
    ramp = results["anticipatory_drift"]["pre_ramp"]
    p_drift = results["anticipatory_drift"]["p_value"]
    n_trans = results["anticipatory_drift"]["n_transitions"]
    print(f"      pre_ramp={ramp:.4f}, p={p_drift:.4f}, n_transitions={n_trans}")

    # 6. Factored vs conjunctive
    print("  [6] Factored vs conjunctive coding...")
    results["factored_conjunctive"] = compute_factored_vs_conjunctive(
        emb, y_motif, height_tier,
    )
    conj = results["factored_conjunctive"]["conjunctive_fraction"]
    print(f"      conjunctive={conj:.1%}, "
          f"both={results['factored_conjunctive']['n_dims_both']}, "
          f"motif_only={results['factored_conjunctive']['n_dims_motif_only']}, "
          f"height_only={results['factored_conjunctive']['n_dims_height_only']}")

    # 7. Velocity → transition prediction
    print("  [7] Velocity → transition prediction...")
    results["velocity_prediction"] = compute_velocity_transition_prediction(
        emb, y_motif, horizon_bins=4, n_perm=n_perm,
    )
    auc = results["velocity_prediction"]["auc_roc"]
    p_vel = results["velocity_prediction"]["p_value"]
    print(f"      AUC={auc:.3f}, p={p_vel:.4f}")

    # 8. Timescale separation
    print("  [8] Timescale separation...")
    results["timescale"] = compute_timescale_separation(
        emb, y_motif, max_lag_bins=20,
    )
    n_tau = results["timescale"]["neural_tau_s"]
    b_tau = results["timescale"]["behavioral_tau_s"]
    ratio = results["timescale"]["tau_ratio"]
    print(f"      neural_tau={n_tau:.1f}s, behavioral_tau={b_tau:.1f}s, ratio={ratio:.2f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# CLI runner: all sessions
# ══════════════════════════════════════════════════════════════════════════

def run_all_sessions(
    sessions: Optional[List[str]] = None,
    fig5_dir: Optional[Path] = None,
    similarity_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_perm: int = 500,
) -> List[Dict[str, Any]]:
    """Run manifold analyses for all sessions.

    Args:
        sessions: Session keys (default: all 5).
        fig5_dir: CEBRA results directory.
        similarity_dir: Fig 4 similarity results directory.
        output_dir: Where to save results JSON.
        n_perm: Permutations for nulls.

    Returns:
        List of per-session result dicts.
    """
    _root = Path(__file__).resolve().parent.parent.parent.parent
    fig5_dir = fig5_dir or DEFAULT_FIG5_DIR
    similarity_dir = similarity_dir or (_root / "results" / "single_fm" / "fig4_similarity")
    output_dir = output_dir or (_root / "results" / "single_fm" / "fig5_manifold_analyses")
    output_dir.mkdir(parents=True, exist_ok=True)

    if sessions is None:
        sessions = [s for s in ALL_SESSIONS if (fig5_dir / s / "embeddings.npz").exists()]

    # Session dir mapping for similarity data
    _sim_dir_map = {
        "M2_0526": "single_0526_M2",
        "M2_0605": "single_0605_M2",
        "M1_0525": "single_0525_M1",
        "M1_0609": "single_0609_M1",
        "M1_0613": "single_0613_M1",
    }

    all_results: List[Dict[str, Any]] = []

    for sess_key in sessions:
        print(f"\n{'=' * 60}")
        print(f"Session: {sess_key}")
        print(f"{'=' * 60}")

        # Load kinematic similarity vector if available
        kin_sim_vec = None
        sim_subdir = _sim_dir_map.get(sess_key, sess_key)
        sim_path = similarity_dir / sim_subdir / "fig4_similarity_result.json"
        if sim_path.exists():
            with open(sim_path) as f:
                sim_data = json.load(f)
            kin_sim_vec = np.array(sim_data.get("sim_vec", []))
            print(f"  Loaded kinematic similarity: {len(kin_sim_vec)} pairs")
        else:
            print(f"  [WARN] No kinematic similarity for {sess_key}")

        sess_results = run_all_manifold_analyses(
            sess_key, fig5_dir=fig5_dir,
            kin_sim_vec=kin_sim_vec, n_perm=n_perm,
        )
        all_results.append(sess_results)

        # Save per-session
        sess_out = output_dir / sess_key
        sess_out.mkdir(parents=True, exist_ok=True)
        with open(sess_out / "manifold_analyses.json", "w") as f:
            json.dump(sess_results, f, indent=2, default=str)

    # Print summary
    _print_summary(all_results)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")

    return all_results


def _print_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print summary table across sessions."""
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n{'Session':<15} {'RSA rho':>8} {'RSA p':>7} "
          f"{'Drift':>7} {'Dr p':>6} "
          f"{'Conj%':>6} "
          f"{'AUC':>6} {'AUC p':>7} "
          f"{'N tau':>6} {'B tau':>6}")
    print("-" * 80)

    for r in all_results:
        rsa = r.get("embedding_rsa", {})
        drift = r.get("anticipatory_drift", {})
        fc = r.get("factored_conjunctive", {})
        vel = r.get("velocity_prediction", {})
        ts = r.get("timescale", {})

        rho_str = f"{rsa.get('rho', float('nan')):.3f}" if rsa.get("rho") is not None else "N/A"
        rsa_p = f"{rsa.get('p_permutation', float('nan')):.3f}" if rsa.get("p_permutation") is not None else "N/A"

        print(f"{r['session_id']:<15} "
              f"{rho_str:>8} {rsa_p:>7} "
              f"{drift.get('pre_ramp', float('nan')):>7.4f} {drift.get('p_value', float('nan')):>6.3f} "
              f"{fc.get('conjunctive_fraction', float('nan'))*100:>5.0f}% "
              f"{vel.get('auc_roc', float('nan')):>6.3f} {vel.get('p_value', float('nan')):>7.3f} "
              f"{ts.get('neural_tau_s', float('nan')):>5.1f}s {ts.get('behavioral_tau_s', float('nan')):>5.1f}s")


__all__ = [
    "compute_embedding_rsa",
    "compute_anticipatory_drift",
    "compute_factored_vs_conjunctive",
    "compute_velocity_transition_prediction",
    "compute_timescale_separation",
    "run_all_manifold_analyses",
    "run_all_sessions",
]
