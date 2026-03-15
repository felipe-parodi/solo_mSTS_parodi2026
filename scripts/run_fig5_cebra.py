#!/usr/bin/env python
"""Figure 5: CEBRA population manifold analysis for Nature Communications revision.

Runs the full analysis pipeline:
  1. Session coverage audit (all 5 sessions)
  2. CEBRA supervised + self-supervised embeddings (per session)
  3. Cross-session stability (within-M2)
  4. Nonlinear decoding from latents
  5. Transition dynamics at syllable onsets
  6. Context-dependent coding (syllable × height)

Usage:
    conda activate mucal3
    python scripts/single_fm/run_fig5_cebra.py \
        --output-dir results/fig5_cebra \
        --max-iterations 10000 \
        --n-perms 100

Author: Felipe Parodi + Claude
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import cebra  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402, F401 — used in run_single_session
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

# Project imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.decoding.temporal_cv import make_temporal_folds  # noqa: E402, F401
from freemat.single_fm.data.session_loader import SingleFMSessionLoader  # noqa: E402
from freemat.single_fm.encoding.stats import circular_shift_indices  # noqa: E402, F401
from freemat_data import load_motifs_from_session  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")

# ── Configuration ─────────────────────────────────────────────────────────

SESSION_CONFIGS = {
    "M2_0526": "configs/session_M2_0526.yaml",
    "M2_0605": "configs/session_M2_0605.yaml",
    "M1_0525": "configs/session_M1_0525.yaml",
    "M1_0609": "configs/session_M1_0609.yaml",
    "M1_0613": "configs/session_M1_0613.yaml",
}

# Two CEBRA model families (FLAW 1 fix: no com_z leakage for height claims)
LABELS_SPATIAL_LITE = ["com_x", "com_y", "speed", "yaw_rate"]       # 4 labels, no com_z
LABELS_BEHAVIOR = ["com_x", "com_y", "com_z", "speed", "yaw_rate"]  # 5 labels, full
# Keep original for backward-compat in load_session_data
CONTINUOUS_LABELS = LABELS_BEHAVIOR

# Height tier thresholds (meters above floor)
HEIGHT_THRESHOLDS = {
    "floor": (0.0, 0.5),
    "bench": (0.5, 1.2),
    "ceiling": (1.2, 3.0),
}

MOTIF_NAMES = {
    1: "Rest", 2: "Stand", 3: "Pace", 4: "Inspect", 5: "Pivot",
    6: "Travel", 7: "Scan", 8: "Walk", 9: "Perch", 10: "Reach",
    11: "Sit", 12: "Walk(rare)",
}


@dataclass
class CebraConfig:
    """Configuration for CEBRA analysis."""
    output_dimension_viz: int = 3
    output_dimension_decode: int = 8
    max_iterations: int = 10000
    batch_size: int = 512
    model_architecture: str = "offset10-model"
    bin_s: float = 0.5
    n_knn: int = 5
    n_cv_folds: int = 5
    n_perms: int = 100
    transition_window_s: float = 2.0
    min_motif_bins: int = 20  # min bins per motif class
    embargo_s: float = 2.0  # embargo for temporal CV
    n_jitter_shifts: int = 500  # circular-shift null for transitions


# ── Data Loading ──────────────────────────────────────────────────────────

def load_session_data(
    session_key: str,
    config: CebraConfig,
    floor_filter: bool = False,
) -> Dict[str, Any]:
    """Load aligned spikes + features + motifs for one session.

    Args:
        session_key: Key into SESSION_CONFIGS.
        config: CebraConfig.
        floor_filter: If True, restrict to floor bins only.

    Returns:
        Dict with keys: X, y_cont, y_motif, t, height_tier, session_id,
                         unit_banks, n_units, n_bins, motif_classes
    """
    yaml_path = str(REPO_ROOT / SESSION_CONFIGS[session_key])
    loader = SingleFMSessionLoader(yaml_path)

    print(f"\n{'─'*60}")
    print(f"Loading session: {session_key}")
    print(f"{'─'*60}")

    # ── Aligned data (spikes + features) ──
    merged = loader.get_aligned_data(
        bin_s=config.bin_s,
        floor_filter=floor_filter,
        floor_height_threshold=3.0,  # no floor filter by default
    )
    if merged.empty:
        raise ValueError(f"No merged data for {session_key}")

    # ── Build spike matrix X (n_bins, n_units) ──
    unit_ids = sorted(merged["unit_index"].unique())
    n_units = len(unit_ids)
    bin_times = sorted(merged["t_bin_s"].unique())
    n_bins = len(bin_times)

    # Pivot to wide: rows=bins, cols=units
    spike_wide = merged.pivot_table(
        index="t_bin_s", columns="unit_index", values="fr_z",
        aggfunc="first",
    ).reindex(index=bin_times, columns=unit_ids)
    spike_wide = spike_wide.fillna(0.0)
    X = spike_wide.values.astype(np.float32)
    t = np.array(bin_times, dtype=np.float64)

    # ── Continuous labels (two families) ──
    feat_df = merged.drop_duplicates(subset=["t_bin_s"]).set_index("t_bin_s").reindex(bin_times)

    # Behavior model: full 5 labels
    y_cont_behavior_cols = [c for c in LABELS_BEHAVIOR if c in feat_df.columns]
    y_cont_behavior = feat_df[y_cont_behavior_cols].values.astype(np.float32)

    # Spatial-lite model: 4 labels (no com_z) — for height/space claims
    y_cont_spatial_cols = [c for c in LABELS_SPATIAL_LITE if c in feat_df.columns]
    y_cont_spatial = feat_df[y_cont_spatial_cols].values.astype(np.float32)

    # Fill NaNs in continuous labels with column means
    for arr in [y_cont_behavior, y_cont_spatial]:
        for i in range(arr.shape[1]):
            col = arr[:, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = np.nanmean(col)
                arr[:, i] = col

    # Keep backward-compat alias
    y_cont_cols = y_cont_behavior_cols
    y_cont = y_cont_behavior

    # ── Height tier ──
    com_z_idx = y_cont_cols.index("com_z") if "com_z" in y_cont_cols else None
    if com_z_idx is not None:
        com_z = y_cont[:, com_z_idx]
        height_tier = np.full(n_bins, -1, dtype=int)
        for tier_idx, (name, (lo, hi)) in enumerate(HEIGHT_THRESHOLDS.items()):
            mask = (com_z >= lo) & (com_z < hi)
            height_tier[mask] = tier_idx
        # Clamp outliers
        height_tier[height_tier < 0] = 0
    else:
        height_tier = np.zeros(n_bins, dtype=int)

    # ── Motif labels ──
    motifs = load_motifs_from_session(loader._session)
    if motifs is not None and motifs.t is not None:
        # Align motif labels to bins using nearest-neighbor
        y_motif = _align_motifs_to_bins(motifs, t, config.bin_s)
    else:
        warnings.warn(f"No motifs for {session_key}, using zeros")
        y_motif = np.zeros(n_bins, dtype=int)

    # ── Unit banks + metadata ──
    unit_meta = merged.drop_duplicates(subset=["unit_index"]).set_index("unit_index")
    if "unit_bank" in unit_meta.columns:
        unit_banks = unit_meta.reindex(unit_ids)["unit_bank"].values
    else:
        unit_banks = np.array(["unknown"] * n_units)

    # Unit kinds (SUA/MUA) for robustness controls
    if "unit_kind" in unit_meta.columns:
        unit_kinds = unit_meta.reindex(unit_ids)["unit_kind"].values
    else:
        unit_kinds = np.array(["unknown"] * n_units)

    # Raw unit IDs for one-per-channel filtering (unit_id // 10 = channel)
    unit_ids_raw = np.array(unit_ids, dtype=int)

    # ── Summary ──
    motif_classes = np.unique(y_motif[y_motif > 0])
    n_sua = np.sum(unit_kinds == "SUA")
    n_mua = np.sum(unit_kinds == "MUA")
    print(f"  X shape: {X.shape} ({n_bins} bins × {n_units} units, {n_sua} SUA + {n_mua} MUA)")
    print(f"  Behavior labels: {y_cont_behavior_cols} → shape {y_cont_behavior.shape}")
    print(f"  Spatial-lite labels: {y_cont_spatial_cols} → shape {y_cont_spatial.shape}")
    print(f"  Motif classes: {len(motif_classes)} ({motif_classes})")
    print(f"  Height tier counts: floor={np.sum(height_tier==0)}, "
          f"bench={np.sum(height_tier==1)}, ceiling={np.sum(height_tier==2)}")
    print(f"  Banks: V={np.sum(unit_banks=='ventral')}, "
          f"D={np.sum(unit_banks=='dorsal')}, other={np.sum((unit_banks!='ventral')&(unit_banks!='dorsal'))}")
    print(f"  Time range: [{t[0]:.1f}, {t[-1]:.1f}]s ({(t[-1]-t[0])/60:.1f} min)")

    return {
        "X": X,
        "y_cont": y_cont,
        "y_cont_cols": y_cont_cols,
        "y_cont_behavior": y_cont_behavior,
        "y_cont_behavior_cols": y_cont_behavior_cols,
        "y_cont_spatial": y_cont_spatial,
        "y_cont_spatial_cols": y_cont_spatial_cols,
        "y_motif": y_motif,
        "t": t,
        "height_tier": height_tier,
        "session_id": session_key,
        "unit_banks": unit_banks,
        "unit_kinds": unit_kinds,
        "unit_ids_raw": unit_ids_raw,
        "n_units": n_units,
        "n_bins": n_bins,
        "motif_classes": motif_classes,
    }


def _align_motifs_to_bins(motifs, bin_times: np.ndarray, bin_s: float) -> np.ndarray:
    """Align motif labels to bin centers using majority vote within bin."""
    motif_t = motifs.t
    motif_id = motifs.motif_id
    n_bins = len(bin_times)
    y = np.zeros(n_bins, dtype=int)

    for i, t_center in enumerate(bin_times):
        t_lo = t_center - bin_s / 2
        t_hi = t_center + bin_s / 2
        mask = (motif_t >= t_lo) & (motif_t < t_hi)
        if mask.any():
            labels_in_bin = motif_id[mask]
            # Majority vote (exclude 0 if possible)
            nonzero = labels_in_bin[labels_in_bin > 0]
            if len(nonzero) > 0:
                y[i] = np.bincount(nonzero).argmax()
            else:
                y[i] = 0
        else:
            y[i] = 0
    return y


# ── Session Coverage Audit ────────────────────────────────────────────────

def audit_session_coverage(data: Dict[str, Any]) -> Dict[str, Any]:
    """Produce coverage summary for one session."""
    y_motif = data["y_motif"]
    height_tier = data["height_tier"]
    n_bins = data["n_bins"]

    # Motif counts
    unique_motifs, motif_counts = np.unique(y_motif[y_motif > 0], return_counts=True)
    motif_dist = {int(m): int(c) for m, c in zip(unique_motifs, motif_counts)}

    # Height tier counts
    tier_names = list(HEIGHT_THRESHOLDS.keys())
    height_dist = {tier_names[i]: int(np.sum(height_tier == i)) for i in range(3)}

    # Transition counts
    transitions = np.sum(np.diff(y_motif) != 0)

    # 9-zone classification (3 height × 3 angular sectors — placeholder)
    n_valid_motif_bins = int(np.sum(y_motif > 0))

    summary = {
        "session_id": data["session_id"],
        "n_bins": n_bins,
        "n_units": data["n_units"],
        "n_valid_motif_bins": n_valid_motif_bins,
        "motif_coverage": n_valid_motif_bins / n_bins,
        "n_motif_classes": len(unique_motifs),
        "motif_distribution": motif_dist,
        "height_distribution": height_dist,
        "n_transitions": int(transitions),
        "transitions_per_min": float(transitions) / (data["n_bins"] * 0.5 / 60),
        "duration_min": float(data["t"][-1] - data["t"][0]) / 60,
    }
    return summary


# ── CEBRA Fitting ─────────────────────────────────────────────────────────

def fit_cebra_supervised(
    X: np.ndarray,
    y_cont: np.ndarray,
    config: CebraConfig,
    output_dim: int = 3,
) -> Tuple[Any, np.ndarray]:
    """Fit CEBRA with supervised continuous labels.

    Args:
        X: (n_bins, n_units) firing rates.
        y_cont: (n_bins, n_labels) continuous behavioral labels.
        config: CebraConfig.
        output_dim: Embedding dimension.

    Returns:
        (model, embedding) where embedding is (n_bins, output_dim).
    """
    model = cebra.CEBRA(
        model_architecture=config.model_architecture,
        batch_size=config.batch_size,
        output_dimension=output_dim,
        max_iterations=config.max_iterations,
        verbose=True,
    )
    model.fit(X, y_cont)
    embedding = model.transform(X)
    return model, embedding


def fit_cebra_time(
    X: np.ndarray,
    config: CebraConfig,
    output_dim: int = 3,
) -> Tuple[Any, np.ndarray]:
    """Fit CEBRA-Time (self-supervised, no labels).

    Args:
        X: (n_bins, n_units) firing rates.
        config: CebraConfig.
        output_dim: Embedding dimension.

    Returns:
        (model, embedding).
    """
    model = cebra.CEBRA(
        model_architecture=config.model_architecture,
        batch_size=config.batch_size,
        output_dimension=output_dim,
        max_iterations=config.max_iterations,
        conditional="time_delta",
        verbose=True,
    )
    model.fit(X)
    embedding = model.transform(X)
    return model, embedding


# ── Nonlinear Decoding ────────────────────────────────────────────────────

def decode_from_latents(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_knn: int = 5,
    n_folds: int = 5,
    label_name: str = "unknown",
    exclude_label: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode labels from CEBRA latents using kNN and logistic regression.

    Args:
        embedding: (n_bins, n_dim) latent embedding.
        labels: (n_bins,) integer labels.
        n_knn: k for kNN.
        n_folds: CV folds.
        label_name: Name for reporting.
        exclude_label: Label value to exclude (e.g., 0 for motifs where 0=missing).
            If None, include all labels.

    Returns:
        Dict with accuracies, chance level, etc.
    """
    # Remove missing/excluded labels
    if exclude_label is not None:
        valid = labels != exclude_label
    else:
        valid = np.ones(len(labels), dtype=bool)
    emb_v = embedding[valid]
    lab_v = labels[valid]

    if len(np.unique(lab_v)) < 2:
        return {"label": label_name, "knn_ba": 0.0, "logistic_ba": 0.0,
                "chance": 1.0, "n_samples": len(lab_v), "n_classes": len(np.unique(lab_v))}

    n_classes = len(np.unique(lab_v))
    chance = 1.0 / n_classes

    # Temporal CV with embargo (FLAW 2 fix)
    embargo_bins = 4  # 2s at 0.5s bins
    folds = make_temporal_folds(len(emb_v), n_folds=n_folds, embargo_bins=embargo_bins)

    knn_accs = []
    log_accs = []

    for train_idx, test_idx in folds:
        X_train, X_test = emb_v[train_idx], emb_v[test_idx]
        y_train, y_test = lab_v[train_idx], lab_v[test_idx]

        # kNN
        knn = KNeighborsClassifier(n_neighbors=n_knn)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accs.append(balanced_accuracy_score(y_test, knn_pred))

        # Logistic
        log = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        log.fit(X_train, y_train)
        log_pred = log.predict(X_test)
        log_accs.append(balanced_accuracy_score(y_test, log_pred))

    return {
        "label": label_name,
        "knn_ba": float(np.mean(knn_accs)),
        "knn_std": float(np.std(knn_accs)),
        "logistic_ba": float(np.mean(log_accs)),
        "logistic_std": float(np.std(log_accs)),
        "chance": chance,
        "n_samples": len(lab_v),
        "n_classes": n_classes,
    }


def decode_from_raw_spikes(
    X: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    label_name: str = "unknown",
    exclude_label: Optional[int] = None,
) -> Dict[str, Any]:
    """Baseline decoding from raw spike rates (logistic regression)."""
    if exclude_label is not None:
        valid = labels != exclude_label
    else:
        valid = np.ones(len(labels), dtype=bool)
    X_v = X[valid]
    lab_v = labels[valid]

    if len(np.unique(lab_v)) < 2:
        return {"label": label_name, "logistic_ba": 0.0, "chance": 1.0}

    n_classes = len(np.unique(lab_v))
    chance = 1.0 / n_classes

    # Temporal CV with embargo (FLAW 2 fix)
    embargo_bins = 4  # 2s at 0.5s bins
    folds = make_temporal_folds(len(X_v), n_folds=n_folds, embargo_bins=embargo_bins)
    accs = []
    for train_idx, test_idx in folds:
        log = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        log.fit(X_v[train_idx], lab_v[train_idx])
        pred = log.predict(X_v[test_idx])
        accs.append(balanced_accuracy_score(lab_v[test_idx], pred))

    return {
        "label": label_name,
        "logistic_ba": float(np.mean(accs)),
        "logistic_std": float(np.std(accs)),
        "chance": chance,
        "n_samples": len(lab_v),
        "n_classes": n_classes,
    }


# ── CV-Aware CEBRA Decoding (FLAW 3 fix) ─────────────────────────────────

def decode_cv_aware(
    X: np.ndarray,
    y_cont: np.ndarray,
    y_discrete: np.ndarray,
    config: CebraConfig,
    n_dims: int = 8,
    label_name: str = "unknown",
    exclude_label: Optional[int] = None,
) -> Dict[str, Any]:
    """Leave-one-fold-out CEBRA fitting + decoding (FLAW 3 fix).

    For each temporal CV fold: fit CEBRA on train folds only,
    transform train + test, train kNN on train embeddings,
    evaluate on held-out test embeddings.

    Args:
        X: (n_bins, n_units) firing rates.
        y_cont: (n_bins, n_labels) continuous labels for CEBRA training.
        y_discrete: (n_bins,) discrete labels to decode.
        config: CebraConfig.
        n_dims: CEBRA embedding dimension.
        label_name: Name for reporting.
        exclude_label: Label value to exclude.

    Returns:
        Dict with knn_ba, logistic_ba, chance, per-fold accuracies.
    """
    if exclude_label is not None:
        valid = y_discrete != exclude_label
    else:
        valid = np.ones(len(y_discrete), dtype=bool)

    X_v = X[valid]
    y_cont_v = y_cont[valid]
    lab_v = y_discrete[valid]

    if len(np.unique(lab_v)) < 2:
        return {"label": label_name, "knn_ba": 0.0, "logistic_ba": 0.0,
                "chance": 1.0, "n_samples": len(lab_v), "n_classes": len(np.unique(lab_v))}

    n_classes = len(np.unique(lab_v))
    chance = 1.0 / n_classes

    embargo_bins = int(config.embargo_s / config.bin_s)
    folds = make_temporal_folds(len(X_v), n_folds=config.n_cv_folds,
                                embargo_bins=embargo_bins)

    knn_accs = []
    log_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(f"    CV fold {fold_i + 1}/{len(folds)}: "
              f"train={len(train_idx)}, test={len(test_idx)}")

        # Fit CEBRA on TRAIN data only
        model = cebra.CEBRA(
            model_architecture=config.model_architecture,
            batch_size=config.batch_size,
            output_dimension=n_dims,
            max_iterations=config.max_iterations,
            verbose=False,
        )
        model.fit(X_v[train_idx], y_cont_v[train_idx])

        # Transform train + test through this fold's model
        emb_train = model.transform(X_v[train_idx])
        emb_test = model.transform(X_v[test_idx])

        y_train = lab_v[train_idx]
        y_test = lab_v[test_idx]

        # kNN
        knn = KNeighborsClassifier(n_neighbors=config.n_knn)
        knn.fit(emb_train, y_train)
        knn_pred = knn.predict(emb_test)
        knn_accs.append(balanced_accuracy_score(y_test, knn_pred))

        # Logistic
        log_clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=42)
        log_clf.fit(emb_train, y_train)
        log_pred = log_clf.predict(emb_test)
        log_accs.append(balanced_accuracy_score(y_test, log_pred))

    return {
        "label": label_name,
        "knn_ba": float(np.mean(knn_accs)),
        "knn_std": float(np.std(knn_accs)),
        "logistic_ba": float(np.mean(log_accs)),
        "logistic_std": float(np.std(log_accs)),
        "chance": chance,
        "n_samples": len(lab_v),
        "n_classes": n_classes,
        "per_fold_knn": [float(a) for a in knn_accs],
        "per_fold_logistic": [float(a) for a in log_accs],
        "cv_method": "temporal_cv_aware_cebra",
    }


# ── Transition Dynamics ──────────────────────────────────────────────────

def compute_transition_dynamics(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    t: np.ndarray,
    config: CebraConfig,
) -> Dict[str, Any]:
    """Compute latent velocity around syllable transitions.

    Args:
        embedding: (n_bins, n_dim) latent embedding.
        y_motif: (n_bins,) motif labels.
        t: (n_bins,) time in seconds.
        config: CebraConfig.

    Returns:
        Dict with velocity profiles, prediction correlations, etc.
    """
    bin_s = config.bin_s
    window_bins = int(config.transition_window_s / bin_s)

    # Find syllable transitions (motif changes, excluding to/from 0)
    transitions = []
    for i in range(1, len(y_motif)):
        if y_motif[i] != y_motif[i-1] and y_motif[i] > 0 and y_motif[i-1] > 0:
            if i >= window_bins and i < len(y_motif) - window_bins:
                transitions.append(i)

    n_trans = len(transitions)
    print(f"  Found {n_trans} valid transitions (±{config.transition_window_s}s window)")

    if n_trans < 10:
        return {"n_transitions": n_trans, "status": "too_few_transitions"}

    # Extract trajectories around transitions
    total_window = 2 * window_bins + 1
    velocities = np.zeros((n_trans, total_window - 1))
    trajectories = np.zeros((n_trans, total_window, embedding.shape[1]))

    for j, idx in enumerate(transitions):
        start = idx - window_bins
        end = idx + window_bins + 1
        traj = embedding[start:end]
        trajectories[j] = traj
        # Latent velocity = Euclidean distance between consecutive states
        diffs = np.diff(traj, axis=0)
        velocities[j] = np.linalg.norm(diffs, axis=1)

    mean_velocity = np.mean(velocities, axis=0)
    sem_velocity = np.std(velocities, axis=0) / np.sqrt(n_trans)

    # FLAW 4 fix: Baseline velocity excludes transition neighborhoods
    full_vel = np.linalg.norm(np.diff(embedding, axis=0), axis=1)
    non_trans_mask = np.ones(len(full_vel), dtype=bool)
    for idx in transitions:
        lo = max(0, idx - window_bins - 1)
        hi = min(len(full_vel), idx + window_bins)
        non_trans_mask[lo:hi] = False
    baseline_velocity = float(np.mean(full_vel[non_trans_mask])) if non_trans_mask.any() else float(np.mean(full_vel))

    # Peak velocity at transition relative to baseline
    center_idx = window_bins  # center of ±window
    peak_vel = mean_velocity[center_idx - 1]  # just before transition
    vel_ratio = peak_vel / baseline_velocity if baseline_velocity > 0 else 0.0

    # FLAW 5 fix: Circular-shift null for velocity peak
    n_shifts = config.n_jitter_shifts
    rng = np.random.default_rng(42)
    null_peak_vels = np.zeros(n_shifts)
    n_total = len(y_motif)
    for shift_i in range(n_shifts):
        shifted_idx = circular_shift_indices(n_total, rng)
        shifted_motif = y_motif[shifted_idx]
        # Find transitions in shifted labels
        null_trans = []
        for i in range(1, len(shifted_motif)):
            if (shifted_motif[i] != shifted_motif[i - 1]
                    and shifted_motif[i] > 0 and shifted_motif[i - 1] > 0):
                if window_bins <= i < len(shifted_motif) - window_bins:
                    null_trans.append(i)
        if len(null_trans) >= 10:
            null_vels = []
            for idx in null_trans:
                traj = embedding[idx - window_bins:idx + window_bins + 1]
                diffs = np.diff(traj, axis=0)
                v = np.linalg.norm(diffs, axis=1)
                null_vels.append(v[center_idx - 1])
            null_peak_vels[shift_i] = float(np.mean(null_vels))
        else:
            null_peak_vels[shift_i] = baseline_velocity

    vel_null_p = float(np.mean(null_peak_vels >= peak_vel))
    print(f"  Velocity peak={peak_vel:.4f}, baseline={baseline_velocity:.4f}, "
          f"ratio={vel_ratio:.2f}, null_p={vel_null_p:.4f}")

    # Pre-transition: does distance to next-state centroid decrease?
    class_centroids = {}
    for m in np.unique(y_motif[y_motif > 0]):
        mask = y_motif == m
        if mask.sum() > 0:
            class_centroids[m] = np.mean(embedding[mask], axis=0)

    pre_transition_distances = []
    for j, idx in enumerate(transitions):
        next_motif = y_motif[idx]
        if next_motif in class_centroids:
            centroid = class_centroids[next_motif]
            start = idx - window_bins
            dists = [np.linalg.norm(embedding[k] - centroid) for k in range(start, idx + 1)]
            pre_transition_distances.append(dists)

    if pre_transition_distances:
        mean_pre_dist = np.mean(pre_transition_distances, axis=0)
        dist_slope = np.polyfit(np.arange(len(mean_pre_dist)), mean_pre_dist, 1)[0]
    else:
        mean_pre_dist = None
        dist_slope = None

    # Next-syllable prediction from pre-transition latent state
    pre_states = embedding[[idx - 1 for idx in transitions]]
    next_motifs = y_motif[[idx for idx in transitions]]
    pred_result = decode_from_latents(
        pre_states, next_motifs, n_knn=5, n_folds=min(5, n_trans // 10 + 1),
        label_name="next_syllable_from_pre",
    )

    time_axis = np.arange(-window_bins, window_bins) * bin_s

    return {
        "n_transitions": n_trans,
        "status": "ok",
        "mean_velocity": mean_velocity.tolist(),
        "sem_velocity": sem_velocity.tolist(),
        "time_axis": time_axis.tolist(),
        "baseline_velocity": float(baseline_velocity),
        "baseline_method": "non_transition_bins",
        "peak_velocity_at_transition": float(peak_vel),
        "velocity_ratio": float(vel_ratio),
        "velocity_null_p": vel_null_p,
        "velocity_null_n_shifts": n_shifts,
        "pre_transition_distance_slope": float(dist_slope) if dist_slope is not None else None,
        "next_syllable_prediction": pred_result,
    }


# ── Context-Dependent Coding ─────────────────────────────────────────────

def compute_context_coding(
    embedding: np.ndarray,
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    min_bins: int = 20,
) -> Dict[str, Any]:
    """Test if same syllable is coded differently across spatial contexts.

    Computes within-syllable between-context vs within-context distances
    in latent space.

    Args:
        embedding: (n_bins, n_dim).
        y_motif: (n_bins,) motif labels.
        height_tier: (n_bins,) height tier (0=floor, 1=bench, 2=ceiling).
        min_bins: Minimum bins per syllable×tier cell.

    Returns:
        Dict with distance metrics, significance, etc.
    """
    tier_names = list(HEIGHT_THRESHOLDS.keys())
    motif_ids = np.unique(y_motif[y_motif > 0])

    # Build syllable × tier occupancy
    occupancy = {}
    for m in motif_ids:
        for tier in range(3):
            mask = (y_motif == m) & (height_tier == tier)
            count = mask.sum()
            if count >= min_bins:
                occupancy[(int(m), tier)] = count

    print(f"  Context coding: {len(occupancy)} valid syllable×tier cells "
          f"(min_bins={min_bins})")

    if len(occupancy) < 4:
        return {"status": "insufficient_data", "n_cells": len(occupancy)}

    # For each syllable with ≥2 tier contexts, compute:
    # 1. Within-context variance (avg pairwise distance within same syllable+tier)
    # 2. Between-context variance (avg pairwise distance same syllable, different tier)
    within_dists = []
    between_dists = []
    syllable_results = []

    for m in motif_ids:
        tiers_with_data = [tier for tier in range(3) if (int(m), tier) in occupancy]
        if len(tiers_with_data) < 2:
            continue

        # Centroids per tier
        centroids = {}
        for tier in tiers_with_data:
            mask = (y_motif == m) & (height_tier == tier)
            centroids[tier] = np.mean(embedding[mask], axis=0)

        # Within-context: avg centroid-to-point distance
        for tier in tiers_with_data:
            mask = (y_motif == m) & (height_tier == tier)
            points = embedding[mask]
            dists = np.linalg.norm(points - centroids[tier], axis=1)
            within_dists.extend(dists)

        # Between-context: centroid-to-centroid distances
        for i, t1 in enumerate(tiers_with_data):
            for t2 in tiers_with_data[i+1:]:
                d = np.linalg.norm(centroids[t1] - centroids[t2])
                between_dists.append(d)

                syllable_results.append({
                    "motif": int(m),
                    "motif_name": MOTIF_NAMES.get(int(m), f"M{m}"),
                    "tier1": tier_names[t1],
                    "tier2": tier_names[t2],
                    "centroid_distance": float(d),
                    "n_tier1": occupancy[(int(m), t1)],
                    "n_tier2": occupancy[(int(m), t2)],
                })

    mean_within = float(np.mean(within_dists)) if within_dists else 0.0
    mean_between = float(np.mean(between_dists)) if between_dists else 0.0

    # Ratio: between/within > 1 means context differentiates representation
    ratio = mean_between / mean_within if mean_within > 0 else 0.0

    # Permutation test: shuffle tier assignments within each syllable
    n_perms = 500
    null_ratios = []
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        shuffled_tier = height_tier.copy()
        for m in motif_ids:
            mask = y_motif == m
            idx = np.where(mask)[0]
            shuffled_tier[idx] = rng.permutation(shuffled_tier[idx])

        null_within = []
        null_between = []
        for m in motif_ids:
            tiers_with_data = [tier for tier in range(3)
                               if np.sum((y_motif == m) & (shuffled_tier == tier)) >= min_bins]
            if len(tiers_with_data) < 2:
                continue
            centroids = {}
            for tier in tiers_with_data:
                mask_t = (y_motif == m) & (shuffled_tier == tier)
                centroids[tier] = np.mean(embedding[mask_t], axis=0)
            for tier in tiers_with_data:
                mask_t = (y_motif == m) & (shuffled_tier == tier)
                null_within.extend(np.linalg.norm(embedding[mask_t] - centroids[tier], axis=1))
            for i, t1 in enumerate(tiers_with_data):
                for t2 in tiers_with_data[i+1:]:
                    null_between.append(np.linalg.norm(centroids[t1] - centroids[t2]))

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
        "occupancy": {f"{m}_{tier_names[t]}": c for (m, t), c in occupancy.items()},
    }


# ── Cross-Session Stability ──────────────────────────────────────────────

def compute_cross_session_stability(
    data1: Dict, data2: Dict,
    config: CebraConfig,
) -> Dict[str, Any]:
    """Compute cross-session stability using CEBRA multi-session and decoder transfer.

    Args:
        data1, data2: Session data dicts.
        config: CebraConfig.

    Returns:
        Dict with consistency metrics.
    """
    print(f"\n{'='*60}")
    print(f"Cross-session stability: {data1['session_id']} vs {data2['session_id']}")
    print(f"{'='*60}")

    # Method 1: Train decoder on session 1, test on session 2 (and vice versa)
    # Using height tier as a shared label
    results = {}

    for label_name, label_key in [("height_tier", "height_tier"), ("motif", "y_motif")]:
        # Session 1 → Session 2
        s1_emb = data1.get("embedding_8d")
        s2_emb = data2.get("embedding_8d")
        if s1_emb is None or s2_emb is None:
            continue

        y1 = data1[label_key]
        y2 = data2[label_key]

        # For motif: only use shared classes
        if label_key == "y_motif":
            shared = np.intersect1d(
                np.unique(y1[y1 > 0]),
                np.unique(y2[y2 > 0]),
            )
            if len(shared) < 2:
                results[f"{label_name}_transfer"] = "insufficient_shared_classes"
                continue
            mask1 = np.isin(y1, shared)
            mask2 = np.isin(y2, shared)
        else:
            mask1 = np.ones(len(y1), dtype=bool)
            mask2 = np.ones(len(y2), dtype=bool)

        # Train on S1, test on S2
        knn = KNeighborsClassifier(n_neighbors=config.n_knn)
        knn.fit(s1_emb[mask1], y1[mask1])
        pred_2 = knn.predict(s2_emb[mask2])
        ba_1to2 = balanced_accuracy_score(y2[mask2], pred_2)

        # Train on S2, test on S1
        knn2 = KNeighborsClassifier(n_neighbors=config.n_knn)
        knn2.fit(s2_emb[mask2], y2[mask2])
        pred_1 = knn2.predict(s1_emb[mask1])
        ba_2to1 = balanced_accuracy_score(y1[mask1], pred_1)

        n_classes = len(np.unique(np.concatenate([y1[mask1], y2[mask2]])))
        chance = 1.0 / n_classes

        results[f"{label_name}_s1_to_s2"] = float(ba_1to2)
        results[f"{label_name}_s2_to_s1"] = float(ba_2to1)
        results[f"{label_name}_mean_transfer"] = float((ba_1to2 + ba_2to1) / 2)
        results[f"{label_name}_chance"] = float(chance)

        print(f"  {label_name}: S1→S2={ba_1to2:.3f}, S2→S1={ba_2to1:.3f}, "
              f"mean={((ba_1to2+ba_2to1)/2):.3f}, chance={chance:.3f}")

    return results


# ── Visualization ─────────────────────────────────────────────────────────

def setup_nature_style():
    """Set matplotlib params for Nature journal style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_embedding_3d(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    label_names: Optional[Dict[int, str]] = None,
    cmap: str = "viridis",
    save_path: Optional[Path] = None,
    is_continuous: bool = False,
    colorbar_label: str = "",
):
    """Plot 3D CEBRA embedding colored by labels."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if is_continuous:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c=labels, cmap=cmap, s=1, alpha=0.5)
        plt.colorbar(sc, ax=ax, label=colorbar_label, shrink=0.6)
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("tab20", len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                      c=[colors(i)], s=1, alpha=0.4, label=name)
        if len(unique_labels) <= 12:
            ax.legend(markerscale=5, loc='upper right', fontsize=8)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("CEBRA 1")
    ax.set_ylabel("CEBRA 2")
    ax.set_zlabel("CEBRA 3")

    if save_path:
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_transition_velocity(
    dynamics_result: Dict,
    save_path: Optional[Path] = None,
):
    """Plot latent velocity profile around transitions."""
    if dynamics_result.get("status") != "ok":
        print(f"  Skipping transition plot: {dynamics_result.get('status')}")
        return

    t_axis = np.array(dynamics_result["time_axis"])
    mean_vel = np.array(dynamics_result["mean_velocity"])
    sem_vel = np.array(dynamics_result["sem_velocity"])
    baseline = dynamics_result["baseline_velocity"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(t_axis, mean_vel - sem_vel, mean_vel + sem_vel,
                    alpha=0.3, color="steelblue")
    ax.plot(t_axis, mean_vel, color="steelblue", linewidth=2, label="Mean latent velocity")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Baseline")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Transition")

    ax.set_xlabel("Time relative to transition (s)")
    ax.set_ylabel("Latent velocity (a.u.)")
    ax.set_title("Neural trajectory speed at behavioral transitions", fontweight='bold')
    ax.legend()

    if save_path:
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches='tight')
    plt.close(fig)


def plot_decoding_comparison(
    decode_results: Dict[str, Dict],
    save_path: Optional[Path] = None,
):
    """Bar chart comparing decoding accuracies across methods."""
    fig, axes = plt.subplots(1, len(decode_results), figsize=(5 * len(decode_results), 4))
    if len(decode_results) == 1:
        axes = [axes]

    for ax, (label_name, results) in zip(axes, decode_results.items()):
        methods = []
        accs = []
        errs = []

        for method_name, res in results.items():
            if not isinstance(res, dict):
                continue
            methods.append(method_name)
            # Use kNN_ba for CEBRA methods, logistic_ba for raw
            if "knn_ba" in res and "kNN" in method_name:
                accs.append(res["knn_ba"])
                errs.append(res.get("knn_std", 0))
            elif "logistic_ba" in res:
                accs.append(res["logistic_ba"])
                errs.append(res.get("logistic_std", 0))
            else:
                continue

        x = np.arange(len(methods))
        ax.bar(x, accs, yerr=errs, capsize=5, color=["steelblue", "coral", "seagreen"][:len(methods)])
        # Add chance line
        if results and isinstance(list(results.values())[0], dict):
            chance = list(results.values())[0].get("chance", 0)
            ax.axhline(chance, color="gray", linestyle="--", linewidth=1)
            ax.text(len(methods) - 0.5, chance + 0.01, "chance", fontsize=9, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(label_name, fontweight='bold')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches='tight')
    plt.close(fig)


def plot_context_coding(
    context_result: Dict,
    save_path: Optional[Path] = None,
):
    """Plot context-dependent coding results."""
    if context_result.get("status") != "ok":
        print(f"  Skipping context plot: {context_result.get('status')}")
        return

    syl_results = context_result["syllable_results"]
    if not syl_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [f"{r['motif_name']}\n({r['tier1']}-{r['tier2']})" for r in syl_results]
    dists = [r['centroid_distance'] for r in syl_results]

    x = np.arange(len(names))
    ax.bar(x, dists, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Centroid distance in latent space")
    ax.set_title(f"Context-dependent coding\n"
                 f"(ratio={context_result['context_ratio']:.2f}, "
                 f"p={context_result['p_value']:.3f})",
                 fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches='tight')
    plt.close(fig)


# ── Main Pipeline ─────────────────────────────────────────────────────────

def _one_per_channel_mask(unit_ids_raw: np.ndarray) -> np.ndarray:
    """Select one unit per channel (unit_id // 10 = channel). Returns bool mask."""
    channels = unit_ids_raw // 10
    seen = set()
    mask = np.zeros(len(unit_ids_raw), dtype=bool)
    for i, ch in enumerate(channels):
        if ch not in seen:
            seen.add(ch)
            mask[i] = True
    return mask


def run_single_session(
    session_key: str,
    config: CebraConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run full CEBRA analysis for one session.

    Two model families:
      - Spatial-lite (com_x, com_y, speed, yaw_rate): for height/space claims
      - Behavior (com_x, com_y, com_z, speed, yaw_rate): for motif/transition claims
    """
    session_dir = output_dir / session_key
    session_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X = None  # will hold spike matrix

    # ── Load data ──
    data = load_session_data(session_key, config)
    X = data["X"]

    # ── Coverage audit ──
    coverage = audit_session_coverage(data)
    with open(session_dir / "coverage_audit.json", "w") as f:
        json.dump(coverage, f, indent=2)
    print("\n  Coverage audit saved.")

    # ══════════════════════════════════════════════════════════════════════
    # A. VISUALIZATION EMBEDDINGS (all-data, illustrative only)
    # ══════════════════════════════════════════════════════════════════════

    # A1. Spatial-lite CEBRA (3D) → Fig 5A (colored by height)
    print("\n  [A1] Fitting Spatial-lite CEBRA (3D, viz)...")
    model_spatial_3d, emb_spatial_viz_3d = fit_cebra_supervised(
        X, data["y_cont_spatial"], config, output_dim=3,
    )
    print(f"  Spatial-lite 3D shape: {emb_spatial_viz_3d.shape}")

    # A2. Behavior CEBRA (3D) → Fig 5B (colored by motif), transition dynamics
    print("\n  [A2] Fitting Behavior CEBRA (3D, viz)...")
    model_behavior_3d, emb_behavior_viz_3d = fit_cebra_supervised(
        X, data["y_cont_behavior"], config, output_dim=3,
    )

    # A3. Behavior CEBRA (8D) → cross-session storage
    print("\n  [A3] Fitting Behavior CEBRA (8D, viz)...")
    model_behavior_8d, emb_behavior_viz_8d = fit_cebra_supervised(
        X, data["y_cont_behavior"], config, output_dim=8,
    )
    data["embedding_8d"] = emb_behavior_viz_8d

    # A4. CEBRA-Time (3D, self-supervised) → supplement
    print("\n  [A4] Fitting CEBRA-Time (self-supervised, 3D)...")
    model_time_3d, emb_time_3d = fit_cebra_time(X, config, output_dim=3)

    # A5. PCA-8D baseline (for backup comparisons)
    print("\n  [A5] Fitting PCA-8D baseline...")
    pca_model = PCA(n_components=8)
    emb_pca_8d = pca_model.fit_transform(X)
    print(f"  PCA-8D variance explained: {pca_model.explained_variance_ratio_.sum():.3f}")

    # ── Visualization plots ──
    print("\n  Generating embedding plots...")

    # Spatial-lite colored by height (Fig 5A)
    plot_embedding_3d(emb_spatial_viz_3d, data["height_tier"],
                      f"{session_key} — Spatial-lite CEBRA (height tier)",
                      label_names={0: "Floor", 1: "Bench", 2: "Ceiling"},
                      save_path=session_dir / "emb_spatial_lite_height")

    # Behavior colored by motif (Fig 5B)
    plot_embedding_3d(emb_behavior_viz_3d, data["y_motif"],
                      f"{session_key} — Behavior CEBRA (motif)",
                      label_names=MOTIF_NAMES,
                      save_path=session_dir / "emb_behavior_motif")

    # Self-supervised colored by height (supplement)
    plot_embedding_3d(emb_time_3d, data["height_tier"],
                      f"{session_key} — CEBRA-Time (height tier)",
                      label_names={0: "Floor", 1: "Bench", 2: "Ceiling"},
                      save_path=session_dir / "emb_selfsup_height")

    # Continuous height coloring on spatial-lite
    com_z_idx = (data["y_cont_behavior_cols"].index("com_z")
                 if "com_z" in data["y_cont_behavior_cols"] else None)
    if com_z_idx is not None:
        plot_embedding_3d(emb_spatial_viz_3d, data["y_cont_behavior"][:, com_z_idx],
                          f"{session_key} — Spatial-lite (height, continuous)",
                          cmap="viridis", is_continuous=True,
                          colorbar_label="Height (m)",
                          save_path=session_dir / "emb_spatial_lite_height_continuous")

    # ══════════════════════════════════════════════════════════════════════
    # B. QUANTITATIVE DECODING (CV-aware, reported numbers)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ── B. Quantitative decoding (CV-aware CEBRA) ──")
    decode_results = {}

    # B1. Height decoding: Spatial-lite CEBRA (no com_z leakage)
    print("\n  [B1] Height decoding (Spatial-lite CV, 8D)...")
    height_cebra = decode_cv_aware(
        X, data["y_cont_spatial"], data["height_tier"], config,
        n_dims=8, label_name="CEBRA-SpatialLite-8D_height_3class",
        exclude_label=None,
    )
    height_raw = decode_from_raw_spikes(
        X, data["height_tier"], n_folds=config.n_cv_folds,
        label_name="raw_height_3class", exclude_label=None,
    )
    height_pca = decode_from_latents(
        emb_pca_8d, data["height_tier"], n_knn=config.n_knn,
        n_folds=config.n_cv_folds, label_name="PCA-8D_height_3class",
        exclude_label=None,
    )
    decode_results["height_3class"] = {
        "CEBRA-SpatialLite-8D kNN": height_cebra,
        "PCA-8D kNN": height_pca,
        "Raw logistic": height_raw,
    }
    print(f"  Height: CEBRA-SL={height_cebra['knn_ba']:.3f}, "
          f"PCA={height_pca['knn_ba']:.3f}, "
          f"raw={height_raw['logistic_ba']:.3f}, "
          f"chance={height_cebra['chance']:.3f}")

    # B2. Motif decoding: Behavior CEBRA (com_z OK for motif)
    print("\n  [B2] Motif decoding (Behavior CV, 8D)...")
    motif_cebra = decode_cv_aware(
        X, data["y_cont_behavior"], data["y_motif"], config,
        n_dims=8, label_name="CEBRA-Behavior-8D_motif",
        exclude_label=0,
    )
    motif_raw = decode_from_raw_spikes(
        X, data["y_motif"], n_folds=config.n_cv_folds,
        label_name="raw_motif", exclude_label=0,
    )
    motif_pca = decode_from_latents(
        emb_pca_8d, data["y_motif"], n_knn=config.n_knn,
        n_folds=config.n_cv_folds, label_name="PCA-8D_motif",
        exclude_label=0,
    )
    decode_results["motif"] = {
        "CEBRA-Behavior-8D kNN": motif_cebra,
        "PCA-8D kNN": motif_pca,
        "Raw logistic": motif_raw,
    }
    print(f"  Motif: CEBRA-B={motif_cebra['knn_ba']:.3f}, "
          f"PCA={motif_pca['knn_ba']:.3f}, "
          f"raw={motif_raw['logistic_ba']:.3f}, "
          f"chance={motif_cebra['chance']:.3f}")

    plot_decoding_comparison(decode_results, save_path=session_dir / "decoding_comparison")

    # ══════════════════════════════════════════════════════════════════════
    # C. CONTEXT-DEPENDENT CODING (anchor claim — Spatial-lite, NOT tainted)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ── C. Context-dependent coding (Spatial-lite viz) ──")
    context_cebra = compute_context_coding(
        emb_spatial_viz_3d, data["y_motif"], data["height_tier"],
        min_bins=config.min_motif_bins,
    )
    plot_context_coding(context_cebra, save_path=session_dir / "context_coding_spatial_lite")

    # Near-mandatory backup: PCA-8D centroid distances
    print("  [C backup] Context coding from PCA-8D...")
    context_pca = compute_context_coding(
        emb_pca_8d, data["y_motif"], data["height_tier"],
        min_bins=config.min_motif_bins,
    )
    plot_context_coding(context_pca, save_path=session_dir / "context_coding_pca8d")

    # ══════════════════════════════════════════════════════════════════════
    # D. TRANSITION DYNAMICS (anchor claim — Behavior viz, com_z OK)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ── D. Transition dynamics (Behavior viz) ──")
    dynamics_cebra = compute_transition_dynamics(
        emb_behavior_viz_3d, data["y_motif"], data["t"], config,
    )
    plot_transition_velocity(dynamics_cebra, save_path=session_dir / "transition_velocity_behavior")

    # Near-mandatory backup: PCA-8D velocity + next-syllable
    print("  [D backup] Transition dynamics from PCA-8D...")
    dynamics_pca = compute_transition_dynamics(
        emb_pca_8d, data["y_motif"], data["t"], config,
    )
    plot_transition_velocity(dynamics_pca, save_path=session_dir / "transition_velocity_pca8d")

    # ══════════════════════════════════════════════════════════════════════
    # E. ROBUSTNESS CONTROLS (supplement)
    # ══════════════════════════════════════════════════════════════════════
    robustness = {}

    # E1. SUA-only decoding
    sua_mask = data["unit_kinds"] == "SUA"
    n_sua = int(sua_mask.sum())
    if n_sua >= 10:
        print(f"\n  [E1] SUA-only decoding ({n_sua} units)...")
        X_sua = X[:, sua_mask]
        sua_height = decode_from_raw_spikes(
            X_sua, data["height_tier"], n_folds=config.n_cv_folds,
            label_name="SUA_raw_height", exclude_label=None,
        )
        sua_motif = decode_from_raw_spikes(
            X_sua, data["y_motif"], n_folds=config.n_cv_folds,
            label_name="SUA_raw_motif", exclude_label=0,
        )
        robustness["sua_only"] = {
            "n_units": n_sua,
            "height": sua_height,
            "motif": sua_motif,
        }
        print(f"  SUA height={sua_height['logistic_ba']:.3f}, "
              f"motif={sua_motif['logistic_ba']:.3f}")

    # E2. One-per-channel decoding
    opc_mask = _one_per_channel_mask(data["unit_ids_raw"])
    n_opc = int(opc_mask.sum())
    if n_opc >= 10:
        print(f"\n  [E2] One-per-channel decoding ({n_opc} units)...")
        X_opc = X[:, opc_mask]
        opc_height = decode_from_raw_spikes(
            X_opc, data["height_tier"], n_folds=config.n_cv_folds,
            label_name="OPC_raw_height", exclude_label=None,
        )
        opc_motif = decode_from_raw_spikes(
            X_opc, data["y_motif"], n_folds=config.n_cv_folds,
            label_name="OPC_raw_motif", exclude_label=0,
        )
        robustness["one_per_channel"] = {
            "n_units": n_opc,
            "height": opc_height,
            "motif": opc_motif,
        }
        print(f"  OPC height={opc_height['logistic_ba']:.3f}, "
              f"motif={opc_motif['logistic_ba']:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # F. CLAIM-TO-EVIDENCE TABLE
    # ══════════════════════════════════════════════════════════════════════
    claim_evidence = []

    def _add_claim(metric: str, value: float, model: str, protocol: str,
                   control: str = "primary"):
        claim_evidence.append({
            "session": session_key,
            "metric": metric,
            "value": round(value, 4),
            "model_family": model,
            "train_test_protocol": protocol,
            "control_status": control,
        })

    _add_claim("height_knn_ba", height_cebra["knn_ba"],
               "Spatial-lite", "temporal_cv_aware_cebra")
    _add_claim("height_raw_logistic_ba", height_raw["logistic_ba"],
               "raw_spikes", "temporal_cv")
    _add_claim("height_pca_knn_ba", height_pca["knn_ba"],
               "PCA-8D", "temporal_cv")
    _add_claim("motif_knn_ba", motif_cebra["knn_ba"],
               "Behavior", "temporal_cv_aware_cebra")
    _add_claim("motif_raw_logistic_ba", motif_raw["logistic_ba"],
               "raw_spikes", "temporal_cv")
    _add_claim("motif_pca_knn_ba", motif_pca["knn_ba"],
               "PCA-8D", "temporal_cv")

    if context_cebra.get("status") == "ok":
        _add_claim("context_ratio_cebra", context_cebra["context_ratio"],
                    "Spatial-lite", "all_data_viz", "anchor")
        _add_claim("context_p_cebra", context_cebra["p_value"],
                    "Spatial-lite", "permutation_500")
    if context_pca.get("status") == "ok":
        _add_claim("context_ratio_pca", context_pca["context_ratio"],
                    "PCA-8D", "all_data", "backup")

    if dynamics_cebra.get("status") == "ok":
        _add_claim("vel_ratio_cebra", dynamics_cebra["velocity_ratio"],
                    "Behavior", "all_data_viz", "anchor")
        _add_claim("vel_null_p_cebra", dynamics_cebra["velocity_null_p"],
                    "Behavior", "circular_shift_500")
        ns_pred = dynamics_cebra.get("next_syllable_prediction", {})
        if ns_pred:
            _add_claim("next_syl_knn_cebra", ns_pred.get("knn_ba", 0.0),
                        "Behavior", "temporal_cv")
    if dynamics_pca.get("status") == "ok":
        _add_claim("vel_ratio_pca", dynamics_pca["velocity_ratio"],
                    "PCA-8D", "all_data", "backup")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    session_results = {
        "session_id": session_key,
        "config": asdict(config),
        "coverage": coverage,
        "decode_results": decode_results,
        "dynamics_cebra": dynamics_cebra,
        "dynamics_pca": dynamics_pca,
        "context_cebra": context_cebra,
        "context_pca": context_pca,
        "robustness": robustness,
        "claim_evidence": claim_evidence,
        "elapsed_s": elapsed,
        "model_families": {
            "spatial_lite_labels": LABELS_SPATIAL_LITE,
            "behavior_labels": LABELS_BEHAVIOR,
        },
    }

    with open(session_dir / "results.json", "w") as f:
        json.dump(session_results, f, indent=2, default=str)

    # Save embeddings
    np.savez_compressed(
        session_dir / "embeddings.npz",
        emb_spatial_viz_3d=emb_spatial_viz_3d,
        emb_behavior_viz_3d=emb_behavior_viz_3d,
        emb_behavior_viz_8d=emb_behavior_viz_8d,
        emb_time_3d=emb_time_3d,
        emb_pca_8d=emb_pca_8d,
        y_motif=data["y_motif"],
        height_tier=data["height_tier"],
        t=data["t"],
    )

    # Save CEBRA models
    with open(session_dir / "models.pkl", "wb") as f:
        pickle.dump({
            "model_spatial_3d": model_spatial_3d,
            "model_behavior_3d": model_behavior_3d,
            "model_behavior_8d": model_behavior_8d,
            "model_time_3d": model_time_3d,
        }, f)

    print(f"\n  Session {session_key} complete in {elapsed:.1f}s")
    return data, session_results


def run_full_pipeline(args):
    """Run the full Figure 5 analysis pipeline."""
    setup_nature_style()

    config = CebraConfig(
        max_iterations=args.max_iterations,
        n_perms=args.n_perms,
        bin_s=args.bin_ms / 1000.0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = args.sessions if args.sessions else list(SESSION_CONFIGS.keys())

    print(f"\n{'='*60}")
    print("Figure 5 CEBRA Analysis Pipeline")
    print(f"{'='*60}")
    print(f"Sessions: {sessions}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Bin size: {config.bin_s*1000:.0f} ms")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}")

    # ── Per-session analyses ──
    all_data = {}
    all_results = {}

    for session_key in sessions:
        data, results = run_single_session(session_key, config, output_dir)
        all_data[session_key] = data
        all_results[session_key] = results

    # ── Cross-session stability ──
    # M2: M2_0526 vs M2_0605
    if "M2_0526" in all_data and "M2_0605" in all_data:
        cross_m2 = compute_cross_session_stability(
            all_data["M2_0526"], all_data["M2_0605"], config,
        )
        all_results["cross_session_m2"] = cross_m2
        with open(output_dir / "cross_session_m2.json", "w") as f:
            json.dump(cross_m2, f, indent=2)

    # M1: hooke sessions (if available)
    hooke_keys = [k for k in sessions if "M1" in k]
    if len(hooke_keys) >= 2:
        cross_m1 = compute_cross_session_stability(
            all_data[hooke_keys[0]], all_data[hooke_keys[1]], config,
        )
        all_results["cross_session_m1"] = cross_m1
        with open(output_dir / "cross_session_m1.json", "w") as f:
            json.dump(cross_m1, f, indent=2)

    # ── Aggregate claim-to-evidence table ──
    all_claims = []
    for session_key in sessions:
        all_claims.extend(all_results[session_key].get("claim_evidence", []))
    if all_claims:
        import csv
        csv_path = output_dir / "claim_evidence_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_claims[0].keys())
            writer.writeheader()
            writer.writerows(all_claims)
        print(f"\n  Claim-to-evidence table: {csv_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary = {"sessions": {}, "cross_session": {}}

    for session_key in sessions:
        r = all_results[session_key]
        s = {
            "n_units": r["coverage"]["n_units"],
            "n_bins": r["coverage"]["n_bins"],
            "n_motif_classes": r["coverage"]["n_motif_classes"],
            "n_transitions": r["coverage"]["n_transitions"],
        }
        # Add decoding results
        for label_name in r["decode_results"]:
            for method_name, decode_r in r["decode_results"][label_name].items():
                if isinstance(decode_r, dict):
                    key = f"{label_name}_{method_name}"
                    s[key] = decode_r.get("knn_ba", decode_r.get("logistic_ba", 0))

        # Add dynamics (CEBRA)
        dyn = r.get("dynamics_cebra", {})
        if dyn.get("status") == "ok":
            s["velocity_ratio"] = dyn["velocity_ratio"]
            s["velocity_null_p"] = dyn.get("velocity_null_p")
            s["n_transitions_dynamics"] = dyn["n_transitions"]

        # Add context (CEBRA — Spatial-lite)
        ctx = r.get("context_cebra", {})
        if ctx.get("status") == "ok":
            s["context_ratio"] = ctx["context_ratio"]
            s["context_p"] = ctx["p_value"]

        # Add PCA backups
        dyn_pca = r.get("dynamics_pca", {})
        if dyn_pca.get("status") == "ok":
            s["velocity_ratio_pca"] = dyn_pca["velocity_ratio"]
        ctx_pca = r.get("context_pca", {})
        if ctx_pca.get("status") == "ok":
            s["context_ratio_pca"] = ctx_pca["context_ratio"]

        summary["sessions"][session_key] = s
        print(f"\n{session_key}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    if "cross_session_m2" in all_results:
        summary["cross_session"]["m2"] = all_results["cross_session_m2"]
        print("\nCross-session M2:")
        for k, v in all_results["cross_session_m2"].items():
            print(f"  {k}: {v}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Figure 5 CEBRA analysis")
    parser.add_argument("--output-dir", type=str,
                        default="results/fig5_cebra")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Sessions to run (default: all 5)")
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--bin-ms", type=int, default=500,
                        help="Bin size in milliseconds (default: 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(args)
