#!/usr/bin/env python
"""Fig 4B: Residualized motif decoding — does motif signal survive covariate removal?

For each session, decodes motif identity from:
  1. Raw firing rates (y_true from encoding model)
  2. Residualized firing rates (y_true - y_pred, after regressing out POS/VIS/BODY_KIN/HEAD_KIN/drift)

If residualized accuracy >> chance, the motif signal is not reducible to continuous covariates.

Usage:
    python scripts/single_fm/run_fig4_residualized_decoding.py
    python scripts/single_fm/run_fig4_residualized_decoding.py --sessions M2_0526 --n-perm 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.syllables.alignment import align_syllables_to_bins  # noqa: E402
from analysis.syllables.decoding import (  # noqa: E402
    SyllableDecodingConfig,
    SyllableDecodingResult,
    compute_permutation_null,
    decode_syllables,
)
from freemat.single_fm.data import SingleFMSessionLoader  # noqa: E402
from freemat_data import load_motifs_from_session  # noqa: E402

# ── Session configuration ────────────────────────────────────────────────
SESSION_CONFIGS = {
    "M2_0526": "configs/session_M2_0526.yaml",
    "M2_0605": "configs/session_M2_0605.yaml",
    "M1_0525": "configs/session_M1_0525.yaml",
    "M1_0609": "configs/session_M1_0609.yaml",
    "M1_0613": "configs/session_M1_0613.yaml",
}

SESSION_DIRS = {
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0525": "single_0525_M1",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}

DEFAULT_RESIDUALS_DIR = (
    _PROJECT_ROOT / "results" / "single_fm"
    / "production_20260220_2019" / "all4_bin05"
)
DEFAULT_OUTPUT_DIR = (
    _PROJECT_ROOT / "results" / "single_fm" / "fig4_residualized_decoding"
)


def _load_residual_matrices(
    resid_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load residuals and pivot to (n_bins, n_units) matrices.

    Args:
        resid_path: Path to fig2b_residuals.parquet.

    Returns:
        Tuple of (X_raw, X_resid, bin_times) where X_raw uses y_true
        and X_resid uses the residual column.
    """
    df = pd.read_parquet(resid_path)

    # Pivot to (bins × units)
    raw_pivot = df.pivot(
        index="t_bin_s", columns="unit_id", values="y_true"
    ).sort_index()
    resid_pivot = df.pivot(
        index="t_bin_s", columns="unit_id", values="residual"
    ).sort_index()

    X_raw = raw_pivot.to_numpy()
    X_resid = resid_pivot.to_numpy()
    bin_times = raw_pivot.index.to_numpy()

    # Drop bins (rows) where any unit has NaN (off-floor or filtered bins)
    valid_rows = ~np.isnan(X_raw).any(axis=1) & ~np.isnan(X_resid).any(axis=1)
    n_dropped = (~valid_rows).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{len(valid_rows)} bins with NaN values")
        X_raw = X_raw[valid_rows]
        X_resid = X_resid[valid_rows]
        bin_times = bin_times[valid_rows]

    return X_raw, X_resid, bin_times


def _result_to_dict(result: SyllableDecodingResult) -> Dict[str, Any]:
    """Convert SyllableDecodingResult to JSON-serializable dict."""
    return {
        "balanced_accuracy": float(result.balanced_accuracy),
        "macro_f1": float(result.macro_f1),
        "per_fold_accuracy": [float(x) for x in result.per_fold_accuracy],
        "n_samples": int(result.n_samples),
        "n_classes": len(result.class_labels),
        "class_labels": [int(x) for x in result.class_labels],
    }


def run_residualized_decoding(
    sessions: Optional[List[str]] = None,
    residuals_dir: Path = DEFAULT_RESIDUALS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_perm: int = 100,
    bin_s: float = 0.5,
    top_n_motifs: int = 10,
) -> List[Dict[str, Any]]:
    """Run residualized vs raw motif decoding for all sessions.

    Args:
        sessions: Session keys to process (default: all 5).
        residuals_dir: Root dir with residuals parquets.
        output_dir: Output directory for results.
        n_perm: Number of permutations for null distribution.
        bin_s: Bin size in seconds (must match residuals).
        top_n_motifs: Number of top motifs to keep for decoding.

    Returns:
        List of per-session result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if sessions is None:
        sessions = list(SESSION_CONFIGS.keys())

    decode_config = SyllableDecodingConfig(
        n_folds=5,
        embargo_bins=4,  # 4 bins × 0.5s = 2s embargo
        n_permutations=n_perm,
        classifier_type="logistic",
        top_n_syllables=top_n_motifs,
        random_state=42,
    )

    all_results: List[Dict[str, Any]] = []

    for sess_key in sessions:
        print(f"\n{'='*60}")
        print(f"Session: {sess_key}")
        print(f"{'='*60}")

        sess_dir = SESSION_DIRS[sess_key]
        resid_path = residuals_dir / sess_dir / "encoding" / "fig2b_residuals.parquet"
        if not resid_path.exists():
            print(f"  [SKIP] No residuals: {resid_path}")
            continue

        # Load residual matrices
        print("  Loading residuals...")
        X_raw, X_resid, bin_times = _load_residual_matrices(resid_path)
        print(f"  Matrix shape: {X_raw.shape} (bins × units)")

        # Load and align motif labels
        print("  Loading motifs...")
        yaml_path = SESSION_CONFIGS[sess_key]
        loader = SingleFMSessionLoader(yaml_path)
        motifs = load_motifs_from_session(loader._session)
        if motifs is None or motifs.t is None:
            print(f"  [SKIP] No motifs for {sess_key}")
            continue

        # Build bin edges matching the residuals bin times
        t_start, t_end = loader.time_window
        bin_edges = np.arange(
            float(t_start), float(t_end) + float(bin_s), float(bin_s), dtype=float
        )

        motif_labels = align_syllables_to_bins(
            motifs.motif_id, motifs.t, bin_edges,
            method="center", missing_value=-999,
        )

        # Trim to match residuals length (bin_edges may produce 1 extra bin)
        n_resid_bins = X_raw.shape[0]
        if len(motif_labels) > n_resid_bins:
            motif_labels = motif_labels[:n_resid_bins]
        elif len(motif_labels) < n_resid_bins:
            print(f"  [WARN] motif_labels ({len(motif_labels)}) < residual bins ({n_resid_bins})")
            # Trim residuals to match
            X_raw = X_raw[:len(motif_labels)]
            X_resid = X_resid[:len(motif_labels)]
            bin_times = bin_times[:len(motif_labels)]

        # Filter to valid motif bins (exclude missing=-999 and motif_id=0)
        valid_mask = motif_labels > 0
        y_motif = motif_labels[valid_mask]

        # Remap to top-N motifs
        unique_motifs, counts = np.unique(y_motif, return_counts=True)
        n_classes = min(top_n_motifs, len(unique_motifs))
        top_motifs = unique_motifs[np.argsort(-counts)[:n_classes]]
        keep_mask = np.isin(y_motif, top_motifs)

        X_raw_valid = X_raw[valid_mask][keep_mask]
        X_resid_valid = X_resid[valid_mask][keep_mask]
        y_valid = y_motif[keep_mask]

        chance = 1.0 / len(np.unique(y_valid))
        print(f"  Valid bins: {len(y_valid)}/{X_raw.shape[0]}, "
              f"classes: {len(np.unique(y_valid))}, chance: {chance:.1%}")

        # Decode from raw firing rates
        print("  Decoding from raw FRs...")
        result_raw = decode_syllables(X_raw_valid, y_valid, decode_config)
        print(f"    Raw: BA={result_raw.balanced_accuracy:.1%}, "
              f"F1={result_raw.macro_f1:.3f}")

        # Decode from residualized firing rates
        print("  Decoding from residualized FRs...")
        result_resid = decode_syllables(X_resid_valid, y_valid, decode_config)
        print(f"    Resid: BA={result_resid.balanced_accuracy:.1%}, "
              f"F1={result_resid.macro_f1:.3f}")

        # Permutation null (on residualized — the harder test)
        print(f"  Computing permutation null ({n_perm} perms)...")
        null_accs = compute_permutation_null(
            X_resid_valid, y_valid, decode_config, n_permutations=n_perm,
        )
        p_resid = float(
            (1 + np.sum(null_accs >= result_resid.balanced_accuracy)) / (1 + n_perm)
        )
        print(f"    Resid p-value: {p_resid:.4f}")

        # Also compute null for raw
        null_accs_raw = compute_permutation_null(
            X_raw_valid, y_valid, decode_config, n_permutations=n_perm,
        )
        p_raw = float(
            (1 + np.sum(null_accs_raw >= result_raw.balanced_accuracy)) / (1 + n_perm)
        )
        print(f"    Raw p-value: {p_raw:.4f}")

        # Save session results
        sess_result = {
            "session_id": sess_key,
            "n_units": int(X_raw.shape[1]),
            "n_bins_total": int(X_raw.shape[0]),
            "n_bins_valid": int(len(y_valid)),
            "n_classes": int(len(np.unique(y_valid))),
            "chance": float(chance),
            "raw": _result_to_dict(result_raw),
            "residualized": _result_to_dict(result_resid),
            "raw_p_value": p_raw,
            "resid_p_value": p_resid,
            "null_raw_mean": float(np.mean(null_accs_raw)),
            "null_resid_mean": float(np.mean(null_accs)),
        }

        sess_out = output_dir / sess_key
        sess_out.mkdir(parents=True, exist_ok=True)
        with open(sess_out / "results.json", "w") as f:
            json.dump(sess_result, f, indent=2)

        # Save confusion matrices
        np.savez(
            sess_out / "confusion_matrices.npz",
            raw_confusion=result_raw.confusion_matrix,
            resid_confusion=result_resid.confusion_matrix,
            class_labels=np.array(result_raw.class_labels),
        )

        all_results.append(sess_result)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Session':<15} {'Raw BA':>8} {'Resid BA':>10} {'Chance':>8} "
          f"{'Raw p':>7} {'Resid p':>9}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['session_id']:<15} "
              f"{r['raw']['balanced_accuracy']:>8.1%} "
              f"{r['residualized']['balanced_accuracy']:>10.1%} "
              f"{r['chance']:>8.1%} "
              f"{r['raw_p_value']:>7.3f} "
              f"{r['resid_p_value']:>9.3f}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fig 4B: Residualized motif decoding"
    )
    parser.add_argument(
        "--sessions", nargs="*", default=None,
        help="Sessions to run (default: all 5)",
    )
    parser.add_argument(
        "--residuals-dir", type=str, default=str(DEFAULT_RESIDUALS_DIR),
        help="Directory with fig2b_residuals.parquet files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--top-n-motifs", type=int, default=10)
    args = parser.parse_args()

    run_residualized_decoding(
        sessions=args.sessions,
        residuals_dir=Path(args.residuals_dir),
        output_dir=Path(args.output_dir),
        n_perm=args.n_perm,
        top_n_motifs=args.top_n_motifs,
    )
