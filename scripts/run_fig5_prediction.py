#!/usr/bin/env python
"""Figure 5E: Behavior-conditioned neural prediction analysis.

Tests whether neural activity adds predictive information beyond the
behavioral stream for (1) transition-within-2s and (2) next-syllable
identity.  Reports ΔAUC, ΔAP, Δaccuracy, Δlog-likelihood with
circular-shift permutation p-values.

Usage:
    conda activate mucal3
    python scripts/single_fm/run_fig5_prediction.py \
        --sessions M2_0526 \
        --output-dir results/fig5_prediction \
        --n-perm 10

    # Production (all 5 sessions, 500 perms):
    python scripts/single_fm/run_fig5_prediction.py --n-perm 500

Author: Felipe Parodi + Claude
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Project imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from freemat.single_fm.data.session_loader import SingleFMSessionLoader  # noqa: E402
from freemat.single_fm.encoding.fig5_prediction import (  # noqa: E402
    CONTINUOUS_COLS,
    PredictionConfig,
    PredictionResult,
    aggregate_pre_transition_features,
    build_behavioral_features,
    build_transition_labels,
    compute_bout_age,
    predict_next_syllable,
    predict_transition_hazard,
    run_permutation_test_hazard,
    run_permutation_test_next_syllable,
)
from freemat_data import load_motifs_from_session  # noqa: E402

# ── Session configs ──────────────────────────────────────────────────────

SESSION_CONFIGS = {
    "M2_0526": "configs/session_M2_0526.yaml",
    "M2_0605": "configs/session_M2_0605.yaml",
    "M1_0525": "configs/session_M1_0525.yaml",
    "M1_0609": "configs/session_M1_0609.yaml",
    "M1_0613": "configs/session_M1_0613.yaml",
}

ALL_SESSIONS = list(SESSION_CONFIGS.keys())


# ── Data loading (reuses CEBRA runner pattern) ───────────────────────────

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
            nonzero = labels_in_bin[labels_in_bin > 0]
            if len(nonzero) > 0:
                y[i] = np.bincount(nonzero).argmax()
            else:
                y[i] = 0
        else:
            y[i] = 0
    return y


def load_session_data(session_key: str, bin_s: float = 0.5) -> dict:
    """Load aligned spikes + features + motifs for one session.

    Args:
        session_key: Key into SESSION_CONFIGS.
        bin_s: Bin width in seconds.

    Returns:
        Dict with keys: X_spikes, feat_df, y_motif, t, session_id,
        n_units, n_bins.
    """
    yaml_path = str(REPO_ROOT / SESSION_CONFIGS[session_key])
    loader = SingleFMSessionLoader(yaml_path)

    print(f"\n{'─' * 60}")
    print(f"Loading session: {session_key}")
    print(f"{'─' * 60}")

    merged = loader.get_aligned_data(
        bin_s=bin_s,
        floor_filter=False,
        floor_height_threshold=3.0,
    )
    if merged.empty:
        raise ValueError(f"No merged data for {session_key}")

    # Spike matrix (n_bins × n_units)
    unit_ids = sorted(merged["unit_index"].unique())
    n_units = len(unit_ids)
    bin_times = sorted(merged["t_bin_s"].unique())
    n_bins = len(bin_times)

    spike_wide = merged.pivot_table(
        index="t_bin_s", columns="unit_index", values="fr_z",
        aggfunc="first",
    ).reindex(index=bin_times, columns=unit_ids)
    spike_wide = spike_wide.fillna(0.0)
    X_spikes = spike_wide.values.astype(np.float32)
    t = np.array(bin_times, dtype=np.float64)

    # Feature DataFrame (one row per bin)
    feat_df = (
        merged.drop_duplicates(subset=["t_bin_s"])
        .set_index("t_bin_s")
        .reindex(bin_times)
    )
    # Fill NaNs in continuous columns with column means
    for col in CONTINUOUS_COLS:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna(feat_df[col].mean())

    # Motif labels
    motifs = load_motifs_from_session(loader._session)
    if motifs is not None and motifs.t is not None:
        y_motif = _align_motifs_to_bins(motifs, t, bin_s)
    else:
        raise ValueError(f"No motifs for {session_key}")

    motif_classes = np.unique(y_motif[y_motif > 0])
    print(f"  X shape: {X_spikes.shape} ({n_bins} bins × {n_units} units)")
    print(f"  Motif classes: {len(motif_classes)} ({motif_classes})")
    print(f"  Time range: [{t[0]:.1f}, {t[-1]:.1f}]s ({(t[-1] - t[0]) / 60:.1f} min)")

    return {
        "X_spikes": X_spikes,
        "feat_df": feat_df,
        "y_motif": y_motif,
        "t": t,
        "session_id": session_key,
        "n_units": n_units,
        "n_bins": n_bins,
    }


# ── Single session runner ────────────────────────────────────────────────

def run_single_session(
    session_key: str,
    config: PredictionConfig,
) -> PredictionResult:
    """Run the full prediction analysis for one session.

    Args:
        session_key: Session identifier.
        config: Prediction configuration.

    Returns:
        PredictionResult with hazard and next-syllable results.
    """
    t0 = time.time()
    data = load_session_data(session_key, bin_s=config.bin_s)

    X_spikes = data["X_spikes"]
    feat_df = data["feat_df"]
    y_motif = data["y_motif"]
    t = data["t"]

    # ── Feature construction ──
    print("\n  Building features...")
    bout_age = compute_bout_age(y_motif, config.bin_s)
    X_beh = build_behavioral_features(
        y_motif, bout_age, feat_df, spline_df=config.spline_df,
    )
    y_hazard = build_transition_labels(y_motif, config.hazard_horizon_bins)

    print(f"    Behavioral features: {X_beh.shape}")
    print(f"    Transition prevalence: {np.mean(y_hazard):.3f}")

    # ── Task 1: Transition hazard (bin-level) ──
    print("\n  Task 1: Transition hazard prediction...")
    hazard_result = predict_transition_hazard(
        X_beh, X_spikes, y_hazard, config,
    )
    print(f"    Beh AUC: {hazard_result.behavior_only_auc:.3f}  "
          f"Beh+Neur AUC: {hazard_result.behavior_neural_auc:.3f}  "
          f"ΔAUC: {hazard_result.delta_auc:+.4f}")
    print(f"    Beh AP:  {hazard_result.behavior_only_ap:.3f}  "
          f"Beh+Neur AP:  {hazard_result.behavior_neural_ap:.3f}  "
          f"ΔAP:  {hazard_result.delta_ap:+.4f}")

    # Permutation test for hazard
    if config.n_perm > 0:
        print(f"    Running {config.n_perm} permutations (hazard)...")
        p_auc, p_ap, null_aucs, null_aps = run_permutation_test_hazard(
            X_beh, X_spikes, y_hazard,
            hazard_result.delta_auc, hazard_result.delta_ap,
            config,
        )
        hazard_result.delta_auc_p = p_auc
        hazard_result.delta_ap_p = p_ap
        print(f"    p(ΔAUC): {p_auc:.4f}  p(ΔAP): {p_ap:.4f}")

    # ── Task 2: Next-syllable prediction (transition-level) ──
    print("\n  Task 2: Next-syllable prediction...")

    # Aggregate behavioral features over pre-transition window
    X_beh_agg, y_next, trans_idx = aggregate_pre_transition_features(
        X_beh, y_motif, t,
        window_s=config.pre_window_s, buffer_s=config.pre_buffer_s,
    )
    # Aggregate neural features (raw spikes, PCA done inside predict_)
    X_neural_agg, _, _ = aggregate_pre_transition_features(
        X_spikes, y_motif, t,
        window_s=config.pre_window_s, buffer_s=config.pre_buffer_s,
    )

    n_trans = len(y_next)
    n_classes = len(np.unique(y_next))
    print(f"    Transitions: {n_trans}, classes: {n_classes}, "
          f"chance: {1.0 / max(n_classes, 1):.3f}")

    if n_trans < 20:
        print(f"    WARNING: Only {n_trans} transitions — results unreliable")

    next_syl_result = predict_next_syllable(
        X_beh_agg, X_neural_agg, y_next, config,
    )
    print(f"    Markov:     {next_syl_result.markov_baseline_accuracy:.3f}")
    print(f"    Beh acc:    {next_syl_result.behavior_only_accuracy:.3f}  "
          f"Beh+Neur acc: {next_syl_result.behavior_neural_accuracy:.3f}  "
          f"Δacc: {next_syl_result.delta_accuracy:+.4f}")
    print(f"    Beh LL:     {next_syl_result.behavior_only_loglik:.3f}  "
          f"Beh+Neur LL:  {next_syl_result.behavior_neural_loglik:.3f}  "
          f"ΔLL:  {next_syl_result.delta_loglik:+.4f}")

    # Permutation test for next-syllable
    if config.n_perm > 0:
        print(f"    Running {config.n_perm} permutations (next-syllable)...")
        p_acc, p_llk, null_accs, null_llks = run_permutation_test_next_syllable(
            X_beh_agg, X_neural_agg, y_next,
            next_syl_result.delta_accuracy, next_syl_result.delta_loglik,
            config,
        )
        next_syl_result.delta_accuracy_p = p_acc
        next_syl_result.delta_loglik_p = p_llk
        print(f"    p(Δacc): {p_acc:.4f}  p(ΔLL): {p_llk:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Session {session_key} complete in {elapsed:.1f}s")

    return PredictionResult(
        session_id=session_key,
        transition_hazard=hazard_result,
        next_syllable=next_syl_result,
        config=config,
    )


# ── Save results ─────────────────────────────────────────────────────────

def save_result(result: PredictionResult, output_dir: Path) -> Path:
    """Save prediction results as JSON.

    Args:
        result: PredictionResult to save.
        output_dir: Base output directory.

    Returns:
        Path to saved JSON file.
    """
    session_dir = output_dir / result.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / "prediction_results.json"

    # Convert dataclass to dict (handles nested dataclasses)
    d = {
        "session_id": result.session_id,
        "transition_hazard": asdict(result.transition_hazard),
        "next_syllable": asdict(result.next_syllable),
        "config": asdict(result.config),
    }

    with open(out_path, "w") as f:
        json.dump(d, f, indent=2, default=_json_default)

    print(f"  Saved: {out_path}")
    return out_path


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_summary(results: list[PredictionResult], output_dir: Path) -> Path:
    """Save cross-session summary as JSON.

    Args:
        results: List of PredictionResult for all sessions.
        output_dir: Base output directory.

    Returns:
        Path to saved summary JSON.
    """
    summary = {
        "n_sessions": len(results),
        "sessions": [],
    }

    for r in results:
        summary["sessions"].append({
            "session_id": r.session_id,
            "hazard": {
                "beh_auc": r.transition_hazard.behavior_only_auc,
                "beh_neural_auc": r.transition_hazard.behavior_neural_auc,
                "delta_auc": r.transition_hazard.delta_auc,
                "p_auc": r.transition_hazard.delta_auc_p,
                "beh_ap": r.transition_hazard.behavior_only_ap,
                "beh_neural_ap": r.transition_hazard.behavior_neural_ap,
                "delta_ap": r.transition_hazard.delta_ap,
                "p_ap": r.transition_hazard.delta_ap_p,
                "n_bins": r.transition_hazard.n_bins,
                "prevalence": r.transition_hazard.transition_prevalence,
            },
            "next_syllable": {
                "markov_acc": r.next_syllable.markov_baseline_accuracy,
                "beh_acc": r.next_syllable.behavior_only_accuracy,
                "beh_neural_acc": r.next_syllable.behavior_neural_accuracy,
                "delta_acc": r.next_syllable.delta_accuracy,
                "p_acc": r.next_syllable.delta_accuracy_p,
                "beh_loglik": r.next_syllable.behavior_only_loglik,
                "beh_neural_loglik": r.next_syllable.behavior_neural_loglik,
                "delta_loglik": r.next_syllable.delta_loglik,
                "p_loglik": r.next_syllable.delta_loglik_p,
                "n_transitions": r.next_syllable.n_transitions,
                "n_classes": r.next_syllable.n_classes,
                "chance": r.next_syllable.chance,
            },
        })

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"\nSummary saved: {out_path}")
    return out_path


# ── Print summary table ──────────────────────────────────────────────────

def print_summary_table(results: list[PredictionResult]) -> None:
    """Print a formatted summary table to stdout.

    Args:
        results: List of PredictionResult for all sessions.
    """
    print("\n" + "=" * 80)
    print("TRANSITION HAZARD (bin-level, within 2s)")
    print("=" * 80)
    print(f"{'Session':<14} {'Beh AUC':>8} {'B+N AUC':>8} {'ΔAUC':>8} "
          f"{'p':>7} {'Beh AP':>8} {'B+N AP':>8} {'ΔAP':>8} {'p':>7} "
          f"{'Prev':>6}")
    print("-" * 80)
    for r in results:
        h = r.transition_hazard
        print(f"{r.session_id:<14} {h.behavior_only_auc:>8.3f} "
              f"{h.behavior_neural_auc:>8.3f} {h.delta_auc:>+8.4f} "
              f"{h.delta_auc_p:>7.4f} {h.behavior_only_ap:>8.3f} "
              f"{h.behavior_neural_ap:>8.3f} {h.delta_ap:>+8.4f} "
              f"{h.delta_ap_p:>7.4f} {h.transition_prevalence:>6.3f}")

    print("\n" + "=" * 80)
    print("NEXT-SYLLABLE PREDICTION (transition-level)")
    print("=" * 80)
    print(f"{'Session':<14} {'Markov':>8} {'Beh acc':>8} {'B+N acc':>8} "
          f"{'Δacc':>8} {'p':>7} {'Beh LL':>8} {'B+N LL':>8} "
          f"{'ΔLL':>8} {'p':>7} {'nTrans':>6} {'K':>3}")
    print("-" * 80)
    for r in results:
        ns = r.next_syllable
        print(f"{r.session_id:<14} {ns.markov_baseline_accuracy:>8.3f} "
              f"{ns.behavior_only_accuracy:>8.3f} "
              f"{ns.behavior_neural_accuracy:>8.3f} "
              f"{ns.delta_accuracy:>+8.4f} {ns.delta_accuracy_p:>7.4f} "
              f"{ns.behavior_only_loglik:>8.3f} "
              f"{ns.behavior_neural_loglik:>8.3f} "
              f"{ns.delta_loglik:>+8.4f} {ns.delta_loglik_p:>7.4f} "
              f"{ns.n_transitions:>6} {ns.n_classes:>3}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fig 5E: Behavior-conditioned neural prediction",
    )
    parser.add_argument(
        "--sessions", nargs="*", default=None,
        help="Sessions to run (default: all 5)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/fig5_prediction",
    )
    parser.add_argument(
        "--n-perm", type=int, default=500,
        help="Number of circular-shift permutations (default: 500)",
    )
    parser.add_argument(
        "--n-pcs", type=int, default=20,
        help="Number of PCs for neural features (default: 20)",
    )
    parser.add_argument(
        "--bin-ms", type=int, default=500,
        help="Bin size in milliseconds (default: 500)",
    )
    return parser.parse_args()


def main() -> None:
    """Run prediction analysis for specified sessions."""
    args = parse_args()

    sessions = args.sessions if args.sessions else ALL_SESSIONS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PredictionConfig(
        bin_s=args.bin_ms / 1000.0,
        n_perm=args.n_perm,
        n_pcs=args.n_pcs,
    )

    print(f"Sessions: {sessions}")
    print(f"Output:   {output_dir}")
    print(f"Config:   bin_s={config.bin_s}, n_perm={config.n_perm}, "
          f"n_pcs={config.n_pcs}, C={config.classifier_C}")

    results: list[PredictionResult] = []
    t_start = time.time()

    for session_key in sessions:
        if session_key not in SESSION_CONFIGS:
            print(f"WARNING: Unknown session '{session_key}', skipping")
            continue
        result = run_single_session(session_key, config)
        save_result(result, output_dir)
        results.append(result)

    if len(results) > 0:
        save_summary(results, output_dir)
        print_summary_table(results)

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
