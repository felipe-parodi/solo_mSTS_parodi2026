#!/usr/bin/env python
"""Figure 5 follow-up: transition-level surprisal analysis.

Tests whether mSTS carries a surprise-related response when the observed
next motif is unlikely under a behavior-only transition model.

Usage:
    conda activate mucal3
    python scripts/single_fm/run_fig5_transition_surprisal.py \\
        --output-dir results/fig5_transition_surprisal \\
        --sessions M2_0526 \\
        --n-perm 10

Author: Felipe Parodi + Claude
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Project imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from freemat.single_fm.data.session_loader import SingleFMSessionLoader  # noqa: E402
from freemat.single_fm.encoding.config import apply_fdr_correction  # noqa: E402
from freemat.single_fm.encoding.fig5_interaction import build_height_tier  # noqa: E402
from freemat.single_fm.encoding.fig5_transition_surprisal import (  # noqa: E402
    TransitionSurprisalConfig,
    aggregate_pre_features,
    build_transition_features,
    compute_neural_responses,
    compute_surprisal,
    extract_valid_transitions,
    filter_motifs_by_transition_support,
    fit_behavior_model,
    fit_surprisal_regression,
    run_session_null,
)
from freemat.single_fm.encoding.visual_features import (  # noqa: E402
    BODY_KIN_COLS,
    VIS_COLS,
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

# Continuous feature columns to aggregate in pre-window
PRE_FEATURE_COLS = (
    VIS_COLS
    + [c for c in BODY_KIN_COLS if c != "speed"]
    + ["speed", "com_x", "com_y"]
)


# ── Data loading ─────────────────────────────────────────────────────────


def load_session_data(
    session_key: str,
    bin_s: float = 0.5,
) -> Dict[str, Any]:
    """Load aligned spikes + features + motifs for one session.

    Args:
        session_key: Key into SESSION_CONFIGS.
        bin_s: Bin size in seconds.

    Returns:
        Dict with X_spikes, X_spikes_z, feat_df, y_motif, t, height_tier,
        unit_ids, unit_banks, session_id.
    """
    yaml_path = str(REPO_ROOT / SESSION_CONFIGS[session_key])
    loader = SingleFMSessionLoader(yaml_path)

    merged = loader.get_aligned_data(
        bin_s=bin_s,
        floor_filter=False,
        floor_height_threshold=3.0,
    )
    if merged.empty:
        raise ValueError(f"No merged data for {session_key}")

    unit_ids = sorted(merged["unit_index"].unique())
    bin_times = sorted(merged["t_bin_s"].unique())
    n_bins = len(bin_times)

    # Spike matrix (n_bins, n_units) — raw and z-scored
    spike_raw = merged.pivot_table(
        index="t_bin_s", columns="unit_index", values="fr",
        aggfunc="first",
    ).reindex(index=bin_times, columns=unit_ids).fillna(0.0)

    spike_z = merged.pivot_table(
        index="t_bin_s", columns="unit_index", values="fr_z",
        aggfunc="first",
    ).reindex(index=bin_times, columns=unit_ids).fillna(0.0)

    X_spikes = spike_raw.values.astype(np.float64)
    X_spikes_z = spike_z.values.astype(np.float64)
    t = np.array(bin_times, dtype=np.float64)

    # Feature DataFrame (one row per bin)
    feat_df = (
        merged.drop_duplicates(subset=["t_bin_s"])
        .set_index("t_bin_s")
        .reindex(bin_times)
    )

    # Height tier
    com_z = feat_df["com_z"].values.astype(np.float64) if "com_z" in feat_df.columns else np.zeros(n_bins)
    height_tier = build_height_tier(com_z)

    # Motifs
    motifs = load_motifs_from_session(loader._session)
    if motifs is not None and motifs.t is not None:
        y_motif = _align_motifs_to_bins(motifs, t, bin_s)
    else:
        y_motif = np.zeros(n_bins, dtype=int)

    # Unit banks
    unit_meta = merged.drop_duplicates(subset=["unit_index"]).set_index("unit_index")
    unit_banks = (
        unit_meta.reindex(unit_ids)["unit_bank"].values
        if "unit_bank" in unit_meta.columns
        else np.array(["unknown"] * len(unit_ids))
    )

    return {
        "X_spikes": X_spikes,
        "X_spikes_z": X_spikes_z,
        "feat_df": feat_df,
        "y_motif": y_motif,
        "t": t,
        "height_tier": height_tier,
        "unit_ids": np.array(unit_ids),
        "unit_banks": unit_banks,
        "session_id": session_key,
    }


def _align_motifs_to_bins(
    motifs: Any,
    bin_times: np.ndarray,
    bin_s: float,
) -> np.ndarray:
    """Align motif labels to bin centers using majority vote."""
    motif_t = motifs.t
    motif_id = motifs.motif_id
    n_bins = len(bin_times)
    y = np.zeros(n_bins, dtype=int)

    for i, t_center in enumerate(bin_times):
        t_lo = t_center - bin_s / 2
        t_hi = t_center + bin_s / 2
        mask = (motif_t >= t_lo) & (motif_t < t_hi)
        if not mask.any():
            continue
        labels_in_bin = motif_id[mask]
        nonzero = labels_in_bin[labels_in_bin > 0]
        if len(nonzero) > 0:
            y[i] = int(np.bincount(nonzero).argmax())

    return y


# ── Single session pipeline ─────────────────────────────────────────────


def run_single_session(
    session_key: str,
    config: TransitionSurprisalConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the full surprisal analysis for one session.

    Args:
        session_key: Session identifier.
        config: Analysis configuration.
        output_dir: Directory for results.

    Returns:
        Session summary dict.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Session: {session_key}")
    print(f"{'='*60}")

    # 1. Load data
    data = load_session_data(session_key, bin_s=config.bin_s)
    X_spikes_z = data["X_spikes_z"]
    feat_df = data["feat_df"]
    y_motif = data["y_motif"]
    t = data["t"]
    height_tier = data["height_tier"]
    unit_ids = data["unit_ids"]

    print(f"  Loaded: {X_spikes_z.shape[1]} units, {len(t)} bins")

    # 2. Extract valid transitions
    ds = extract_valid_transitions(y_motif, t, height_tier, config)
    print(f"  Valid transitions: {len(ds.trans_bins)}")

    if len(ds.trans_bins) < 30:
        print(f"  SKIPPING: too few transitions ({len(ds.trans_bins)})")
        return {"session_id": session_key, "status": "skipped", "n_transitions": len(ds.trans_bins)}

    # 3. Filter motifs by transition support
    retained_motifs = filter_motifs_by_transition_support(
        ds.next_motif, min_incoming=config.min_incoming_transitions
    )
    print(f"  Retained motifs: {retained_motifs} (K={len(retained_motifs)})")

    # Keep only transitions where next_motif is retained
    keep_mask = np.isin(ds.next_motif, retained_motifs)
    if keep_mask.sum() < 30:
        print(f"  SKIPPING: too few retained transitions ({keep_mask.sum()})")
        return {"session_id": session_key, "status": "skipped", "n_transitions": int(keep_mask.sum())}

    # Apply mask to all transition arrays
    trans_bins = ds.trans_bins[keep_mask]
    current_motif = ds.current_motif[keep_mask]
    next_motif = ds.next_motif[keep_mask]
    previous_motif = ds.previous_motif[keep_mask]
    bout_age = ds.bout_age[keep_mask]
    trans_height = ds.height_tier[keep_mask]
    pre_feature_idx = ds.pre_feature_idx[keep_mask]
    pre_neural_idx = ds.pre_neural_idx[keep_mask]
    post_neural_idx = ds.post_neural_idx[keep_mask]

    n_trans = len(trans_bins)
    print(f"  Transitions after motif filter: {n_trans}")

    # 4. Build behavior model features
    available_cols = [c for c in PRE_FEATURE_COLS if c in feat_df.columns]
    pre_features = aggregate_pre_features(feat_df, pre_feature_idx, available_cols)

    X_beh = build_transition_features(
        current_motif, previous_motif, bout_age, trans_height,
        pre_features, retained_motifs,
    )
    print(f"  Feature matrix: {X_beh.shape}")

    # 5. Fit behavior model
    beh_result = fit_behavior_model(
        X_beh, next_motif, trans_bins, config,
        current_motif=current_motif,
    )
    print(f"  NLL full={beh_result.nll_full:.3f}, Markov={beh_result.nll_markov:.3f}")
    print(f"  Model {'BEATS' if beh_result.nll_full < beh_result.nll_markov else 'LOSES TO'} Markov")

    # 6. Compute surprisal
    surp = compute_surprisal(
        beh_result.p_hat, beh_result.p_markov, next_motif,
        beh_result.classes, prob_floor=config.prob_floor,
    )
    print(f"  Surprisal: mean={surp.surprisal.mean():.3f}, std={surp.surprisal.std():.3f}")

    # 7. Neural responses
    delta_fr = compute_neural_responses(X_spikes_z, pre_neural_idx, post_neural_idx)
    print(f"  delta_fr shape: {delta_fr.shape}")

    # 8. Build controls for regression
    # Post-window movement features
    post_speed = np.zeros(n_trans)
    post_abs_yaw = np.zeros(n_trans)
    pre_speed = np.zeros(n_trans)
    pre_abs_yaw = np.zeros(n_trans)

    if "speed" in feat_df.columns:
        speed_vals = feat_df["speed"].values.astype(np.float64)
        for i in range(n_trans):
            post_speed[i] = np.nanmean(speed_vals[post_neural_idx[i]])
            pre_speed[i] = np.nanmean(speed_vals[pre_neural_idx[i]])

    if "yaw_rate" in feat_df.columns:
        yaw_vals = np.abs(feat_df["yaw_rate"].values.astype(np.float64))
        for i in range(n_trans):
            post_abs_yaw[i] = np.nanmean(yaw_vals[post_neural_idx[i]])
            pre_abs_yaw[i] = np.nanmean(yaw_vals[pre_neural_idx[i]])

    controls_df = pd.DataFrame({
        "current_motif": current_motif,
        "next_motif": next_motif,
        "height_tier": trans_height,
        "bout_age": bout_age,
        "post_speed": post_speed,
        "post_abs_yaw": post_abs_yaw,
        "pre_speed": pre_speed,
        "pre_abs_yaw": pre_abs_yaw,
    })

    # 9. Per-unit regression (post-transition)
    post_results = fit_surprisal_regression(
        delta_fr, surp.z_surprisal, controls_df, unit_ids,
    )
    post_results = apply_fdr_correction(post_results, p_col="p_value", alpha=config.p_threshold)
    print(f"  Median beta_surprisal: {post_results['beta_surprisal'].median():.4f}")
    print(f"  Fraction positive: {(post_results['beta_surprisal'] > 0).mean():.2%}")

    # 10. Timing control: pre-transition regression
    # Use [-2.0s, -1.0s] vs [-1.0s, 0.0s] for pre delta_fr
    n_pre_early = int(round(1.0 / config.bin_s))
    pre_early_idx = np.zeros_like(pre_neural_idx)
    for i in range(n_trans):
        start = pre_neural_idx[i, 0] - n_pre_early
        if start >= 0:
            pre_early_idx[i] = np.arange(start, start + n_pre_early)
        else:
            pre_early_idx[i] = pre_neural_idx[i]

    delta_fr_pre = compute_neural_responses(X_spikes_z, pre_early_idx, pre_neural_idx)
    pre_results = fit_surprisal_regression(
        delta_fr_pre, surp.z_surprisal, controls_df, unit_ids,
    )
    print(f"  Pre-transition median beta: {pre_results['beta_surprisal'].median():.4f}")

    # 11. Session-level null
    print(f"  Running {config.n_perm} permutations...")
    null_result = run_session_null(
        delta_fr, surp.z_surprisal, controls_df,
        current_motif, next_motif,
        n_perm=config.n_perm,
        min_pair_for_pair_shuffle=config.min_pair_for_pair_shuffle,
    )
    print(f"  Null p-value: {null_result.p_value:.4f}")

    # 12. Save results
    session_dir = output_dir / session_key
    session_dir.mkdir(parents=True, exist_ok=True)

    post_results.to_csv(session_dir / "unit_results_post.csv", index=False)
    pre_results.to_csv(session_dir / "unit_results_pre.csv", index=False)

    n_sig = int(post_results["p_value_fdr"].lt(config.p_threshold).sum()) if "p_value_fdr" in post_results.columns else 0

    summary = {
        "session_id": session_key,
        "status": "complete",
        "n_valid_transitions": n_trans,
        "n_retained_motifs": len(retained_motifs),
        "retained_motifs": retained_motifs,
        "nll_full": float(beh_result.nll_full),
        "nll_markov": float(beh_result.nll_markov),
        "model_beats_markov": bool(beh_result.nll_full < beh_result.nll_markov),
        "surprisal_mean": float(surp.surprisal.mean()),
        "surprisal_std": float(surp.surprisal.std()),
        "n_units": int(len(unit_ids)),
        "median_beta_surprisal_post": float(post_results["beta_surprisal"].median()),
        "fraction_positive_beta_post": float((post_results["beta_surprisal"] > 0).mean()),
        "fraction_sig_fdr_post": float(n_sig / len(unit_ids)) if len(unit_ids) > 0 else 0.0,
        "median_beta_surprisal_pre": float(pre_results["beta_surprisal"].median()),
        "fraction_positive_beta_pre": float((pre_results["beta_surprisal"] > 0).mean()),
        "post_gt_pre": bool(
            post_results["beta_surprisal"].median() > pre_results["beta_surprisal"].median()
        ),
        "permutation_p": float(null_result.p_value),
        "observed_median_beta": float(null_result.observed_median),
        "runtime_s": round(time.time() - t0, 1),
    }

    with open(session_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save null distribution for plotting
    np.save(session_dir / "null_medians.npy", null_result.null_medians)

    # Save surprisal values for plotting
    surp_df = pd.DataFrame({
        "trans_bin": trans_bins,
        "current_motif": current_motif,
        "next_motif": next_motif,
        "height_tier": trans_height,
        "bout_age": bout_age,
        "surprisal": surp.surprisal,
        "z_surprisal": surp.z_surprisal,
        "surprisal_markov": surp.surprisal_markov,
        "delta_surprisal": surp.delta_surprisal,
        "post_speed": post_speed,
        "post_abs_yaw": post_abs_yaw,
    })
    surp_df.to_csv(session_dir / "surprisal_values.csv", index=False)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Summary: beta={summary['median_beta_surprisal_post']:.4f}, "
          f"frac_pos={summary['fraction_positive_beta_post']:.2%}, "
          f"perm_p={summary['permutation_p']:.4f}, "
          f"post>pre={summary['post_gt_pre']}")

    return summary


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Run transition surprisal analysis."""
    parser = argparse.ArgumentParser(description="Fig 5 transition surprisal analysis")
    parser.add_argument("--output-dir", type=str,
                        default="results/fig5_transition_surprisal")
    parser.add_argument("--sessions", nargs="+",
                        default=list(SESSION_CONFIGS.keys()))
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--min-incoming", type=int, default=15)
    parser.add_argument("--C-logistic", type=float, default=1.0)
    parser.add_argument("--dashboard", action="store_true",
                        help="Generate Tier 1 visualization dashboard")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TransitionSurprisalConfig(
        n_perm=args.n_perm,
        n_cv_folds=args.n_cv_folds,
        min_incoming_transitions=args.min_incoming,
        C_logistic=args.C_logistic,
    )

    all_summaries: List[Dict[str, Any]] = []

    for session_key in args.sessions:
        if session_key not in SESSION_CONFIGS:
            print(f"WARNING: Unknown session {session_key}, skipping")
            continue
        summary = run_single_session(session_key, config, output_dir)
        all_summaries.append(summary)

    # Cross-session summary
    completed = [s for s in all_summaries if s.get("status") == "complete"]
    if completed:
        cross_df = pd.DataFrame(completed)
        cross_df.to_csv(output_dir / "cross_session_summary.csv", index=False)

        print(f"\n{'='*60}")
        print("CROSS-SESSION SUMMARY")
        print(f"{'='*60}")
        for s in completed:
            print(f"  {s['session_id']}: beta={s['median_beta_surprisal_post']:.4f}, "
                  f"frac_pos={s['fraction_positive_beta_post']:.2%}, "
                  f"perm_p={s['permutation_p']:.4f}")

        betas = [s["median_beta_surprisal_post"] for s in completed]
        n_pos = sum(1 for b in betas if b > 0)
        print(f"\n  Sign consistency: {n_pos}/{len(betas)} positive")
        print(f"  Model beats Markov: {sum(s['model_beats_markov'] for s in completed)}/{len(completed)}")
        print(f"  Post > Pre: {sum(s['post_gt_pre'] for s in completed)}/{len(completed)}")

    with open(output_dir / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Dashboard generation
    if args.dashboard and completed:
        from freemat.single_fm.viz.fig5_surprisal_panels import (
            generate_tier1_dashboard,
        )

        print(f"\n{'='*60}")
        print("GENERATING TIER 1 DASHBOARD")
        print(f"{'='*60}")

        for s in completed:
            session_key = s["session_id"]
            print(f"\nDashboard for {session_key}:")

            # Reload session data for panels that need live spike data
            data = load_session_data(session_key, bin_s=config.bin_s)

            # Re-extract transitions to get trans_bins and controls
            ds = extract_valid_transitions(
                data["y_motif"], data["t"], data["height_tier"], config,
            )
            retained = filter_motifs_by_transition_support(
                ds.next_motif, min_incoming=config.min_incoming_transitions,
            )
            keep = np.isin(ds.next_motif, retained)

            if keep.sum() < 30:
                print("  SKIPPED dashboard (too few transitions)")
                continue

            t_bins = ds.trans_bins[keep]

            # Rebuild controls_df
            cur = ds.current_motif[keep]
            nxt = ds.next_motif[keep]
            h_tier = ds.height_tier[keep]
            b_age = ds.bout_age[keep]
            n_post = int(round(1.0 / config.bin_s))
            n_trans = len(t_bins)

            post_spd = np.zeros(n_trans)
            post_yaw = np.zeros(n_trans)
            pre_spd = np.zeros(n_trans)
            pre_yaw = np.zeros(n_trans)
            feat_df = data["feat_df"]

            if "speed" in feat_df.columns:
                speed_vals = feat_df["speed"].values.astype(np.float64)
                for i in range(n_trans):
                    post_idx = np.arange(t_bins[i], min(t_bins[i] + n_post, len(speed_vals)))
                    pre_idx = np.arange(max(t_bins[i] - n_post, 0), t_bins[i])
                    if len(post_idx) > 0:
                        post_spd[i] = np.nanmean(speed_vals[post_idx])
                    if len(pre_idx) > 0:
                        pre_spd[i] = np.nanmean(speed_vals[pre_idx])

            if "yaw_rate" in feat_df.columns:
                yaw_vals = np.abs(feat_df["yaw_rate"].values.astype(np.float64))
                for i in range(n_trans):
                    post_idx = np.arange(t_bins[i], min(t_bins[i] + n_post, len(yaw_vals)))
                    pre_idx = np.arange(max(t_bins[i] - n_post, 0), t_bins[i])
                    if len(post_idx) > 0:
                        post_yaw[i] = np.nanmean(yaw_vals[post_idx])
                    if len(pre_idx) > 0:
                        pre_yaw[i] = np.nanmean(yaw_vals[pre_idx])

            ctrl_df = pd.DataFrame({
                "current_motif": cur,
                "next_motif": nxt,
                "height_tier": h_tier,
                "bout_age": b_age,
                "post_speed": post_spd,
                "post_abs_yaw": post_yaw,
                "pre_speed": pre_spd,
                "pre_abs_yaw": pre_yaw,
            })

            # Read z_surprisal from saved CSV
            surp_csv = pd.read_csv(output_dir / session_key / "surprisal_values.csv")
            z_surp = surp_csv["z_surprisal"].values

            generate_tier1_dashboard(
                session_key=session_key,
                results_dir=output_dir,
                all_summaries=all_summaries,
                X_spikes_z=data["X_spikes_z"],
                trans_bins=t_bins,
                z_surprisal=z_surp,
                controls_df=ctrl_df,
                unit_ids=data["unit_ids"],
                bin_s=config.bin_s,
            )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
