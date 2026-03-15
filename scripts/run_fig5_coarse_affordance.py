#!/usr/bin/env python
"""Figure 5 follow-up: coarsened affordance alignment analysis.

Loads pre-computed CEBRA v2 embeddings and tests whether neural centroid
distance between context-specific states (same motif, different height tiers)
correlates with the JS divergence of their stay-vs-switch future distributions.

Usage:
    python scripts/single_fm/run_fig5_coarse_affordance.py \
        --sessions M2_0526 M2_0605 \
        --output-dir results/fig5_coarse_affordance

Author: Felipe Parodi + Claude
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freemat.single_fm.encoding.fig5_coarse_affordance import (  # noqa: E402
    CoarseAffordanceConfig,
    compute_coarse_affordance_alignment,
    compute_coarse_future_labels,
)

# ── Configuration ─────────────────────────────────────────────────────────

ALL_SESSIONS: List[str] = [
    "M2_0526", "M2_0605", "M1_0525", "M1_0609", "M1_0613",
]

# Embedding keys to test (from embeddings.npz)
EMBEDDING_KEYS: Dict[str, str] = {
    "CEBRA-SL-3D": "emb_spatial_viz_3d",
    "CEBRA-B-8D": "emb_behavior_viz_8d",
    "PCA-8D": "emb_pca_8d",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Figure 5 follow-up: coarsened affordance alignment",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=ALL_SESSIONS,
        choices=ALL_SESSIONS,
        help="Session keys to run (default: all 5 sessions).",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="results/fig5_cebra_v2",
        help="Directory containing per-session embeddings.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fig5_coarse_affordance",
        help="Directory for per-session and aggregate outputs.",
    )
    parser.add_argument("--window-bins", type=int, default=4,
                        help="Forward window in bins for stay-vs-switch (default: 4 = 2s at 0.5s bins).")
    parser.add_argument("--min-bins", type=int, default=20,
                        help="Minimum bins per motif x tier cell.")
    parser.add_argument("--min-state-pairs", type=int, default=2,
                        help="Minimum cross-tier pairs for Spearman test.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet smoothing for future distributions.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def run_single_session(
    session_key: str,
    embeddings_dir: Path,
    output_dir: Path,
    config: CoarseAffordanceConfig,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run coarse affordance analysis on one session."""
    npz_path = embeddings_dir / session_key / "embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    y_motif = data["y_motif"]
    height_tier = data["height_tier"]

    # Compute coarse future labels
    future_dict = compute_coarse_future_labels(y_motif, window_bins=config.window_bins)
    future_labels = future_dict["stay_vs_switch"]

    if debug:
        n_switch = int(future_labels.sum())
        print(f"[{session_key}] bins={len(y_motif)}, switch={n_switch} "
              f"({100 * n_switch / len(y_motif):.1f}%)")

    # Run affordance alignment for each embedding type
    session_results: Dict[str, Any] = {
        "session_id": session_key,
        "n_bins": int(len(y_motif)),
        "n_switch": int(future_labels.sum()),
        "switch_fraction": float(future_labels.mean()),
    }

    for emb_name, emb_key in EMBEDDING_KEYS.items():
        if emb_key not in data:
            if debug:
                print(f"  [{session_key}] Skipping {emb_name}: key '{emb_key}' not in npz")
            session_results[emb_name] = {"status": "missing_embedding"}
            continue

        embedding = data[emb_key]

        result = compute_coarse_affordance_alignment(
            embedding=embedding,
            y_motif=y_motif,
            height_tier=height_tier,
            future_labels=future_labels,
            config=config,
        )

        if result is None:
            session_results[emb_name] = {"status": "insufficient_data"}
            print(f"  [{session_key}] {emb_name}: insufficient data")
        else:
            session_results[emb_name] = {
                "status": "ok",
                "rho": result.rho,
                "p_value": result.p_value,
                "n_pairs": result.n_pairs,
                "per_pair_data": result.per_pair_data,
            }
            print(f"  [{session_key}] {emb_name}: rho={result.rho:.3f}, "
                  f"p={result.p_value:.4f}, n_pairs={result.n_pairs}")

    # Save per-session results
    session_dir = output_dir / session_key
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / "coarse_affordance.json").open("w", encoding="utf-8") as f:
        json.dump(session_results, f, indent=2)

    return session_results


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CoarseAffordanceConfig(
        window_bins=args.window_bins,
        min_bins=args.min_bins,
        min_state_pairs=args.min_state_pairs,
        alpha=args.alpha,
    )

    all_results: List[Dict[str, Any]] = []

    for session_key in args.sessions:
        print(f"\n{'-' * 60}")
        print(f"Session: {session_key}")
        print(f"{'-' * 60}")

        result = run_single_session(
            session_key, embeddings_dir, output_dir, config, debug=args.debug,
        )
        all_results.append(result)

    # Save aggregate results
    with (output_dir / "coarse_affordance_all_sessions.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("Summary: Coarse Affordance Alignment (stay-vs-switch)")
    print(f"{'=' * 60}")
    print(f"{'Session':<15} {'Embedding':<15} {'rho':>8} {'p':>10} {'pairs':>6}")
    print("-" * 60)
    for r in all_results:
        for emb_name in EMBEDDING_KEYS:
            emb_r = r.get(emb_name, {})
            if emb_r.get("status") == "ok":
                print(f"{r['session_id']:<15} {emb_name:<15} "
                      f"{emb_r['rho']:>8.3f} {emb_r['p_value']:>10.4f} "
                      f"{emb_r['n_pairs']:>6d}")
            else:
                status = emb_r.get("status", "unknown")
                print(f"{r['session_id']:<15} {emb_name:<15} {'—':>8} {'—':>10} {status:>6}")

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print("\n[fig5_coarse_affordance] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
