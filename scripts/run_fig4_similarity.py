#!/usr/bin/env python
"""Runner for Fig 4 syllable similarity vs neural confusion analysis.

Correlates pose-based syllable similarity with neural decoding confusion
using Mantel-style permutation test. Computes per-session and cross-session
summary statistics.

Example:
    python scripts/single_fm/run_fig4_similarity.py \
        --fig4-dir results/single_fm/fig4_production \
        --motifs-dir "C:/path/to/_motifs_output_40hz_optimal_260209_2141" \
        --output-dir results/single_fm/fig4_similarity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from analysis.syllables.similarity import (  # noqa: E402
    correlate_similarity_confusion,
    load_syllable_similarity,
    summarize_across_sessions,
    symmetrize_confusion,
)
from freemat.single_fm.viz.fig4_similarity_panels import save_similarity_panels  # noqa: E402


SESSION_OUTPUT_DIRS: Dict[str, str] = {
    "M1_0525": "single_0525_M1",
    "hedy_0526": "single_0526_hedy",
    "M2_0605": "single_0605_M2",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Correlate syllable pose similarity with neural confusion."
    )
    parser.add_argument(
        "--fig4-dir", type=str, required=True,
        help="Directory with fig4 production results (confusion matrices).",
    )
    parser.add_argument(
        "--motifs-dir", type=str, required=True,
        help="Path to motifs output dir containing similarity_distances.npz.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--n-perm", type=int, default=1000, help="Mantel permutations.")
    parser.add_argument(
        "--sessions", nargs="+",
        choices=list(SESSION_OUTPUT_DIRS.keys()),
        default=list(SESSION_OUTPUT_DIRS.keys()),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_session_confusion(
    fig4_dir: Path,
    session_subdir: str,
) -> tuple[np.ndarray, List[int]] | None:
    """Load confusion matrix from session fig4 output.

    Returns:
        Tuple of (confusion_matrix, labels) or None if not found.
    """
    npz_path = fig4_dir / session_subdir / "fig4_confusion_matrices.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    conf = data["pooled_confusion"].astype(float)
    labels = data["pooled_labels"].tolist()
    return conf, labels


def main() -> None:
    """Run similarity vs confusion analysis per session + cross-session summary."""
    args = parse_args()
    fig4_dir = Path(args.fig4_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "fig4_dir": str(fig4_dir),
        "motifs_dir": args.motifs_dir,
        "n_perm": args.n_perm,
        "sessions": args.sessions,
    }
    with open(output_root / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Load similarity matrix (shared across sessions — same pose vocabulary)
    D_full, labels_full = load_syllable_similarity(args.motifs_dir)
    print(f"[Similarity] Loaded {D_full.shape[0]}×{D_full.shape[0]} similarity matrix, "
          f"labels={labels_full}")

    per_session_results: List[Dict] = []

    for session_key in args.sessions:
        session_subdir = SESSION_OUTPUT_DIRS[session_key]
        conf_data = _load_session_confusion(fig4_dir, session_subdir)
        if conf_data is None:
            print(f"[Similarity] {session_key}: no confusion matrix found, skipping")
            continue

        conf_raw, conf_labels = conf_data
        print(f"[Similarity] {session_key}: confusion {conf_raw.shape}, labels={conf_labels}")

        # Subset similarity to confusion labels (top-10 intersection)
        common_labels = [lb for lb in conf_labels if lb in labels_full]
        if len(common_labels) < 3:
            print(f"[Similarity] {session_key}: too few common labels ({len(common_labels)}), skipping")
            continue

        D_sub, labels_matched = load_syllable_similarity(args.motifs_dir, labels_subset=common_labels)

        # Reorder confusion to match similarity label order
        conf_idx = [conf_labels.index(lb) for lb in labels_matched]
        conf_sub = conf_raw[np.ix_(conf_idx, conf_idx)]

        # Row-normalize then symmetrize
        row_sums = conf_sub.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        conf_norm = conf_sub / row_sums
        conf_sym = symmetrize_confusion(conf_norm)

        # Correlate
        result = correlate_similarity_confusion(
            D_sub, conf_sym, labels_matched, n_perm=args.n_perm,
        )
        result["session_id"] = session_key

        print(f"[Similarity] {session_key}: rho={result['rho']:.3f}, "
              f"p_one={result['p_one_sided']:.4f}, p_two={result['p_two_sided']:.4f}")

        # Save per-session
        session_out = output_root / session_subdir
        session_out.mkdir(parents=True, exist_ok=True)

        session_result = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in result.items()
        }
        with open(session_out / "fig4_similarity_result.json", "w") as f:
            json.dump(session_result, f, indent=2)

        save_similarity_panels(result, D_sub, conf_sym, labels_matched, str(session_out))

        # Save aligned matrices for publication use (Fix 7)
        np.savez(
            session_out / f"fig4_similarity_matrices_{session_key}.npz",
            sim_sub=D_sub,
            conf_sym=conf_sym,
            labels=np.array(labels_matched),
        )

        per_session_results.append(result)

    if not per_session_results:
        print("[Similarity] No sessions produced results.")
        return

    # Cross-session summary (headline statistic)
    summary = summarize_across_sessions(per_session_results)
    print("\n[Similarity] Cross-session summary:")
    print(f"  Median rho = {summary['median_rho']:.3f}")
    print(f"  N sessions = {summary['n_sessions']}")
    print(f"  N negative = {summary['n_negative']}")
    print(f"  Sign test p = {summary['sign_test_p']:.4f}")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "fig4_similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-session results summary (include vectors for scatter plots)
    per_session_summary = [
        {
            "session_id": r["session_id"],
            "rho": r["rho"],
            "p_one_sided": r["p_one_sided"],
            "p_two_sided": r["p_two_sided"],
            "sim_vec": r["sim_vec"].tolist() if isinstance(r.get("sim_vec"), np.ndarray) else r.get("sim_vec", []),
            "conf_vec": r["conf_vec"].tolist() if isinstance(r.get("conf_vec"), np.ndarray) else r.get("conf_vec", []),
        }
        for r in per_session_results
    ]
    with open(summary_dir / "fig4_similarity_per_session.json", "w") as f:
        json.dump(per_session_summary, f, indent=2)

    print(f"\n[Similarity] Results saved to {output_root}")


if __name__ == "__main__":
    main()
