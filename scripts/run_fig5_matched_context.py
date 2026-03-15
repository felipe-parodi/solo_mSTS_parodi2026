#!/usr/bin/env python
"""Fig 5D matched-kinematics context control + affordance alignment.

Hardens context-dependent coding claim:
1. Bout-level nearest-neighbor matching across height tiers on 14
   kinematic/visual covariates (excluding com_z).
2. Affordance alignment: tests whether neural centroid distance between
   context-specific states correlates with JS divergence of future
   behavioral distributions.

Usage:
    conda activate mucal3
    python scripts/single_fm/run_fig5_matched_context.py \
        --output-dir results/fig5_matched_context \
        --n-perm 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from freemat.single_fm.data.session_loader import SingleFMSessionLoader  # noqa: E402
from freemat.single_fm.encoding.fig5_matched_context import (  # noqa: E402
    MATCHING_COLS,
    MatchedContextConfig,
    compute_affordance_alignment,
    compute_matched_context_ratio,
)
from freemat.single_fm.viz.pub_data import load_fig5_embeddings  # noqa: E402

# -- Session configs ------------------------------------------------------

SESSION_CONFIGS: Dict[str, str] = {
    "M2_0526": "configs/session_M2_0526.yaml",
    "M2_0605": "configs/session_M2_0605.yaml",
    "M1_0525": "configs/session_M1_0525.yaml",
    "M1_0609": "configs/session_M1_0609.yaml",
    "M1_0613": "configs/session_M1_0613.yaml",
}

ALL_SESSIONS: List[str] = list(SESSION_CONFIGS.keys())


# -- Data loading ---------------------------------------------------------

def _align_motifs_to_bins(
    motifs: Any,
    bin_times: np.ndarray,
    bin_s: float,
) -> np.ndarray:
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
    return y


def load_session_data(
    session_key: str,
    bin_s: float = 0.5,
    fig5_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load embeddings, features, and motifs for one session.

    Args:
        session_key: Key into SESSION_CONFIGS.
        bin_s: Bin width in seconds.
        fig5_dir: Directory with CEBRA v2 results.

    Returns:
        Dict with embeddings, feature array, motif labels, height tiers.
    """
    yaml_path = str(REPO_ROOT / SESSION_CONFIGS[session_key])
    loader = SingleFMSessionLoader(yaml_path)

    print(f"\n{'-' * 60}")
    print(f"Loading session: {session_key}")
    print(f"{'-' * 60}")

    # Load embeddings from CEBRA v2 results
    emb_data = load_fig5_embeddings(session_key, fig5_dir=fig5_dir)
    y_motif = emb_data["y_motif"]
    height_tier = emb_data["height_tier"]
    t = emb_data["t"]
    n_bins = len(t)

    # Load features from session loader
    merged = loader.get_aligned_data(
        bin_s=bin_s,
        floor_filter=False,
        floor_height_threshold=3.0,
    )
    if merged.empty:
        raise ValueError(f"No merged data for {session_key}")

    # Feature DataFrame (one row per bin)
    bin_times = sorted(merged["t_bin_s"].unique())
    feat_df = (
        merged.drop_duplicates(subset=["t_bin_s"])
        .set_index("t_bin_s")
        .reindex(bin_times)
    )

    # Align features to embedding bins
    # The embedding bins may not perfectly match feat_df bins — interpolate
    feat_arr = np.full((n_bins, len(MATCHING_COLS)), np.nan)
    feat_t = np.array(bin_times, dtype=np.float64)

    for j, col in enumerate(MATCHING_COLS):
        if col in feat_df.columns:
            vals = feat_df[col].values.astype(float)
            # Nearest-neighbor lookup for each embedding bin
            for i, t_emb in enumerate(t):
                idx = np.argmin(np.abs(feat_t - t_emb))
                if abs(feat_t[idx] - t_emb) < bin_s:
                    feat_arr[i, j] = vals[idx]

    # Fill remaining NaNs with column means
    for j in range(feat_arr.shape[1]):
        col_mean = np.nanmean(feat_arr[:, j])
        nan_mask = np.isnan(feat_arr[:, j])
        if nan_mask.any():
            feat_arr[nan_mask, j] = col_mean if np.isfinite(col_mean) else 0.0

    motif_classes = np.unique(y_motif[y_motif > 0])
    print(f"  Bins: {n_bins}, Motif classes: {len(motif_classes)}")
    print(f"  Height tier distribution: {dict(zip(*np.unique(height_tier, return_counts=True)))}")
    print(f"  Feature array shape: {feat_arr.shape}")

    return {
        "emb_spatial_viz_3d": emb_data.get("emb_spatial_viz_3d"),
        "emb_pca_8d": emb_data.get("emb_pca_8d"),
        "y_motif": y_motif,
        "height_tier": height_tier,
        "t": t,
        "features": feat_arr,
        "session_id": session_key,
        "n_bins": n_bins,
    }


# -- Single session runner ------------------------------------------------

def run_single_session(
    session_key: str,
    config: MatchedContextConfig,
    fig5_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run matched-context + affordance for one session.

    Args:
        session_key: Session key.
        config: Analysis configuration.
        fig5_dir: CEBRA v2 results directory.

    Returns:
        Results dict with matched_context and affordance sub-dicts.
    """
    data = load_session_data(session_key, fig5_dir=fig5_dir)
    results: Dict[str, Any] = {"session_id": session_key}

    # Run on both embeddings: CEBRA-Spatial-lite 3D and PCA-8D
    for emb_name, emb_key in [
        ("cebra_spatial", "emb_spatial_viz_3d"),
        ("pca_8d", "emb_pca_8d"),
    ]:
        embedding = data[emb_key]
        if embedding is None:
            print(f"  SKIP {emb_name}: no embedding")
            continue

        print(f"\n  -- {emb_name} --")

        # Matched context ratio
        mc_result = compute_matched_context_ratio(
            embedding=embedding,
            y_motif=data["y_motif"],
            height_tier=data["height_tier"],
            features=data["features"],
            config=config,
        )

        mc_dict: Dict[str, Any]
        if mc_result is not None:
            print(f"    Matched ratio: {mc_result.matched_ratio:.3f} "
                  f"(p={mc_result.p_value:.4f})")
            print(f"    Matched bouts: {mc_result.n_matched_bouts}, "
                  f"fraction: {mc_result.matched_fraction:.2f}")
            max_smd = mc_result.smd_table["smd"].abs().max()
            print(f"    Max SMD: {max_smd:.3f}")

            mc_dict = {
                "matched_ratio": mc_result.matched_ratio,
                "p_value": mc_result.p_value,
                "n_matched_bouts": mc_result.n_matched_bouts,
                "matched_fraction": mc_result.matched_fraction,
                "max_smd": float(max_smd),
                "smd_table": mc_result.smd_table.to_dict(orient="records"),
                "per_motif": mc_result.per_motif_results,
            }
        else:
            print("    Matched context: insufficient data")
            mc_dict = {"status": "insufficient_data"}

        results[f"matched_context_{emb_name}"] = mc_dict

        # Affordance alignment
        aff_result = compute_affordance_alignment(
            embedding=embedding,
            y_motif=data["y_motif"],
            height_tier=data["height_tier"],
            features=data["features"],
            config=config,
        )

        aff_dict: Dict[str, Any]
        if aff_result is not None:
            print(f"    Affordance rho: {aff_result.per_session_rho:.3f} "
                  f"(p={aff_result.per_session_p:.4f}, "
                  f"n_pairs={aff_result.n_pairs})")
            aff_dict = {
                "rho": aff_result.per_session_rho,
                "p_value": aff_result.per_session_p,
                "n_pairs": aff_result.n_pairs,
                "per_pair": aff_result.per_pair_data,
            }
        else:
            print("    Affordance: insufficient data")
            aff_dict = {"status": "insufficient_data"}

        results[f"affordance_{emb_name}"] = aff_dict

    results["config"] = {
        "min_bouts_per_tier": config.min_bouts_per_tier,
        "n_perms": config.n_perms,
        "caliper": config.caliper,
        "min_transitions": config.min_transitions,
    }

    return results


# -- Saving / Printing ----------------------------------------------------

def save_result(result: Dict[str, Any], output_dir: Path) -> None:
    """Save per-session result as JSON."""
    session_dir = output_dir / result["session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")


def print_summary_table(all_results: List[Dict[str, Any]]) -> None:
    """Print summary tables for matched context and affordance."""
    print(f"\n{'=' * 80}")
    print("MATCHED-KINEMATICS CONTEXT CONTROL SUMMARY")
    print(f"{'=' * 80}")

    header = f"{'Session':<14} {'Emb':<16} {'Ratio':>7} {'p':>8} {'Matched':>8} {'Frac':>6} {'MaxSMD':>7}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        for emb_name in ["cebra_spatial", "pca_8d"]:
            key = f"matched_context_{emb_name}"
            if key not in r:
                continue
            mc = r[key]
            if "status" in mc:
                print(f"{r['session_id']:<14} {emb_name:<16} {'N/A':>7}")
                continue
            print(
                f"{r['session_id']:<14} {emb_name:<16} "
                f"{mc['matched_ratio']:7.3f} {mc['p_value']:8.4f} "
                f"{mc['n_matched_bouts']:8d} {mc['matched_fraction']:6.2f} "
                f"{mc['max_smd']:7.3f}"
            )

    print(f"\n{'=' * 80}")
    print("AFFORDANCE ALIGNMENT SUMMARY")
    print(f"{'=' * 80}")

    header2 = f"{'Session':<14} {'Emb':<16} {'rho':>7} {'p':>8} {'nPairs':>7}"
    print(header2)
    print("-" * len(header2))

    for r in all_results:
        for emb_name in ["cebra_spatial", "pca_8d"]:
            key = f"affordance_{emb_name}"
            if key not in r:
                continue
            aff = r[key]
            if "status" in aff:
                print(f"{r['session_id']:<14} {emb_name:<16} {'N/A':>7}")
                continue
            print(
                f"{r['session_id']:<14} {emb_name:<16} "
                f"{aff['rho']:7.3f} {aff['p_value']:8.4f} "
                f"{aff['n_pairs']:7d}"
            )


# -- Main -----------------------------------------------------------------

def main() -> None:
    """Run matched-context + affordance analysis on all sessions."""
    parser = argparse.ArgumentParser(
        description="Fig 5D matched-kinematics context control + affordance alignment",
    )
    parser.add_argument(
        "--sessions", nargs="+", default=ALL_SESSIONS,
        help="Sessions to process (default: all 5)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/fig5_matched_context",
    )
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--caliper", type=float, default=1.0)
    parser.add_argument("--min-bouts", type=int, default=5)
    parser.add_argument("--min-transitions", type=int, default=8)
    parser.add_argument(
        "--fig5-dir", type=str, default=None,
        help="CEBRA v2 results dir (default: auto-detect)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig5_dir = Path(args.fig5_dir) if args.fig5_dir else None

    config = MatchedContextConfig(
        min_bouts_per_tier=args.min_bouts,
        n_perms=args.n_perm,
        caliper=args.caliper,
        min_transitions=args.min_transitions,
    )

    print(f"Config: {config}")
    print(f"Sessions: {args.sessions}")
    print(f"Output: {output_dir}")

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for session_key in args.sessions:
        try:
            result = run_single_session(session_key, config, fig5_dir=fig5_dir)
            save_result(result, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {session_key}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    if all_results:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
