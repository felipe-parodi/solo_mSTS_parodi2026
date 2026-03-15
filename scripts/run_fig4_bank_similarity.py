#!/usr/bin/env python
"""Run per-bank RSA: neural confusion vs pose similarity AND transition adjacency.

For each session and bank (ventral/dorsal), correlates the bank-specific
neural confusion matrix with two behavioral reference matrices:
  1. Pose similarity (cosine distance between syllable centroids)
  2. Transition adjacency (empirical transition probability matrix)

This establishes whether banks differ in what drives neural confusion:
  - Dorsal → stronger pose similarity correlation (categorical discriminability)
  - Ventral → stronger transition adjacency correlation (sequential structure)

Output: results/single_fm/fig4_bank_similarity/

Usage:
    python scripts/single_fm/run_fig4_bank_similarity.py
    python scripts/single_fm/run_fig4_bank_similarity.py --n-perm 500
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.syllables.similarity import (  # noqa: E402
    correlate_similarity_confusion,
    load_syllable_similarity,
    symmetrize_confusion,
)

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_FIG4_DIR = (
    _PROJECT_ROOT / "results" / "single_fm"
    / "production_20260220_2019" / "fig4_all4"
)
DEFAULT_MOTIFS_DIR = (
    "C:/Users/USER Parodi/Documents/freemat_jan_26_asocial"
    "/_motifs_output_40hz_optimal_260209_2141"
)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "single_fm" / "fig4_bank_similarity"

SESSION_DIRS: Dict[str, str] = {
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0525": "single_0525_M1",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


def _load_transition_matrix(
    fig4_dir: Path, session_dir: str,
) -> tuple[np.ndarray, List[int]]:
    """Load and return transition probability matrix for a session.

    Args:
        fig4_dir: Root fig4 directory.
        session_dir: Session subdirectory name.

    Returns:
        Tuple of (matrix, labels).
    """
    path = fig4_dir / session_dir / "fig4_transition_matrix.npz"
    data = np.load(path)
    return data["matrix"].astype(float), data["labels"].tolist()


def _align_matrices(
    ref_matrix: np.ndarray,
    ref_labels: List[int],
    conf_matrix: np.ndarray,
    conf_labels: List[int],
) -> tuple[np.ndarray, np.ndarray, List[int]]:
    """Align reference and confusion matrices to shared label set.

    Args:
        ref_matrix: Reference matrix (pose similarity or transition).
        ref_labels: Labels for reference matrix.
        conf_matrix: Neural confusion matrix.
        conf_labels: Labels for confusion matrix.

    Returns:
        Tuple of (aligned_ref, aligned_conf, shared_labels).
    """
    shared = sorted(set(ref_labels) & set(conf_labels))
    if len(shared) < 3:
        return np.array([]), np.array([]), shared

    ref_idx = [ref_labels.index(lb) for lb in shared]
    conf_idx = [conf_labels.index(lb) for lb in shared]

    ref_aligned = ref_matrix[np.ix_(ref_idx, ref_idx)]
    conf_aligned = conf_matrix[np.ix_(conf_idx, conf_idx)]

    return ref_aligned, conf_aligned, shared


def run_bank_rsa(
    fig4_dir: Path,
    motifs_dir: str,
    output_dir: Path,
    n_perm: int = 1000,
    sessions: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Run per-bank RSA for all sessions.

    Args:
        fig4_dir: Root fig4 results directory.
        motifs_dir: Directory with similarity_distances.npz.
        output_dir: Output directory for results.
        n_perm: Number of Mantel permutations.
        sessions: Sessions to process (default: all).

    Returns:
        List of per-session result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if sessions is None:
        sessions = [s for s in SESSION_DIRS if (fig4_dir / SESSION_DIRS[s]).exists()]

    # Load full pose similarity matrix
    sim_full, sim_labels = load_syllable_similarity(motifs_dir)
    print(f"Pose similarity: {len(sim_labels)} classes, labels={sim_labels}")

    all_results: List[Dict[str, Any]] = []

    for sess_key in sessions:
        sess_dir = SESSION_DIRS[sess_key]
        print(f"\n{'='*60}")
        print(f"Session: {sess_key} ({sess_dir})")
        print(f"{'='*60}")

        # Load confusion matrices
        cm_path = fig4_dir / sess_dir / "fig4_confusion_matrices.npz"
        if not cm_path.exists():
            print(f"  [SKIP] No confusion matrices: {cm_path}")
            continue
        cm_data = np.load(cm_path)

        # Load transition matrix
        trans_path = fig4_dir / sess_dir / "fig4_transition_matrix.npz"
        if not trans_path.exists():
            print(f"  [SKIP] No transition matrix: {trans_path}")
            continue
        trans_matrix, trans_labels = _load_transition_matrix(fig4_dir, sess_dir)

        result: Dict[str, Any] = {"session_id": sess_key}

        for bank in ["ventral", "dorsal"]:
            cm_key = f"{bank}_confusion"
            lbl_key = f"{bank}_labels"

            if cm_key not in cm_data.files:
                print(f"  [SKIP] No {bank} confusion matrix")
                continue

            conf_raw = cm_data[cm_key].astype(float)
            conf_labels = cm_data[lbl_key].tolist()
            conf_sym = symmetrize_confusion(conf_raw)

            print(f"\n  {bank.upper()} bank: {conf_raw.shape[0]} classes, "
                  f"labels={conf_labels}")

            # --- Pose similarity RSA ---
            sim_aligned, conf_aligned, shared = _align_matrices(
                sim_full, sim_labels, conf_sym, conf_labels,
            )
            if len(shared) < 3:
                print("    Pose RSA: <3 shared labels, skipping")
                result[f"{bank}_pose_rho"] = float("nan")
                result[f"{bank}_pose_p"] = float("nan")
            else:
                rsa_pose = correlate_similarity_confusion(
                    sim_aligned, conf_aligned, shared, n_perm=n_perm,
                )
                result[f"{bank}_pose_rho"] = rsa_pose["rho"]
                result[f"{bank}_pose_p"] = rsa_pose["p_two_sided"]
                print(f"    Pose RSA: rho={rsa_pose['rho']:.3f}, "
                      f"p={rsa_pose['p_two_sided']:.3f} "
                      f"({len(shared)} shared classes)")

            # --- Transition adjacency RSA ---
            # Symmetrize transition matrix for Mantel test
            trans_sym = symmetrize_confusion(trans_matrix)
            trans_aligned, conf_aligned_t, shared_t = _align_matrices(
                trans_sym, trans_labels, conf_sym, conf_labels,
            )
            if len(shared_t) < 3:
                print("    Trans RSA: <3 shared labels, skipping")
                result[f"{bank}_trans_rho"] = float("nan")
                result[f"{bank}_trans_p"] = float("nan")
            else:
                rsa_trans = correlate_similarity_confusion(
                    trans_aligned, conf_aligned_t, shared_t, n_perm=n_perm,
                )
                result[f"{bank}_trans_rho"] = rsa_trans["rho"]
                result[f"{bank}_trans_p"] = rsa_trans["p_two_sided"]
                print(f"    Trans RSA: rho={rsa_trans['rho']:.3f}, "
                      f"p={rsa_trans['p_two_sided']:.3f} "
                      f"({len(shared_t)} shared classes)")

        all_results.append(result)

    # Save results
    out_path = output_dir / "fig4_bank_rsa_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Save config
    cfg_path = output_dir / "run_config.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "fig4_dir": str(fig4_dir),
            "motifs_dir": motifs_dir,
            "n_perm": n_perm,
            "sessions": sessions,
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Session':<15} {'V_pose':>8} {'D_pose':>8} {'V_trans':>8} {'D_trans':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['session_id']:<15} "
              f"{r.get('ventral_pose_rho', float('nan')):>8.3f} "
              f"{r.get('dorsal_pose_rho', float('nan')):>8.3f} "
              f"{r.get('ventral_trans_rho', float('nan')):>8.3f} "
              f"{r.get('dorsal_trans_rho', float('nan')):>8.3f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-bank RSA analysis")
    parser.add_argument("--fig4-dir", type=str, default=str(DEFAULT_FIG4_DIR))
    parser.add_argument("--motifs-dir", type=str, default=DEFAULT_MOTIFS_DIR)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    run_bank_rsa(
        fig4_dir=Path(args.fig4_dir),
        motifs_dir=args.motifs_dir,
        output_dir=Path(args.output_dir),
        n_perm=args.n_perm,
    )
