#!/usr/bin/env python
"""Batch runner for Figure 4: Behavioral Motif Analysis across sessions.

Runs motif tuning, decoding, and lag sweep on single-FM sessions,
aggregates results, and saves per-session + summary outputs.

Sessions (5 with ethograms):
- M1_0525: configs/session_M1_0525.yaml (hooke)
- M2_0526: configs/session_M2_0526.yaml
- M2_0605: configs/session_M2_0605.yaml (hedy)
- M1_0609: configs/session_M1_0609.yaml (hooke)
- M1_0613: configs/session_M1_0613.yaml (hooke)

Usage:
    # Quick pilot on one session (10 perms)
    python scripts/single_fm/run_fig4_all_sessions.py \
        --output-dir results/fig4_pilot \
        --sessions M2_0526 --n-perm 10 --debug

    # M2-only production run
    python scripts/single_fm/run_fig4_all_sessions.py \
        --output-dir results/fig4_hedy \
        --sessions M2_0526 M2_0605 --n-perm 1000

    # All 4 sessions production run
    python scripts/single_fm/run_fig4_all_sessions.py \
        --output-dir results/fig4_all4 --n-perm 1000
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


SESSIONS: Dict[str, str] = {
    "M1_0525": "configs/session_M1_0525.yaml",
    "M2_0526": "configs/session_M2_0526.yaml",
    "M2_0605": "configs/session_M2_0605.yaml",
    "M1_0609": "configs/session_M1_0609.yaml",
    "M1_0613": "configs/session_M1_0613.yaml",
}

SESSION_OUTPUT_DIRS: Dict[str, str] = {
    "M1_0525": "single_0525_M1",
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Fig 4 batch runner")
    p.add_argument("--output-dir", required=True, help="Base output directory")
    p.add_argument("--sessions", nargs="+", default=list(SESSIONS.keys()),
                   choices=list(SESSIONS.keys()), help="Sessions to run")
    p.add_argument("--bin-s", type=float, default=0.5, help="Bin width (s)")
    p.add_argument("--top-n", type=int, default=10, help="Top-N motifs")
    p.add_argument("--n-perm", type=int, default=1000, help="Permutations for null")
    p.add_argument("--canonical-alignment", type=str, default=None,
                   help="Path to canonical_alignment.json for cross-session arena alignment")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run Fig 4 pipeline on selected sessions."""
    args = parse_args()

    import pandas as pd
    from freemat.single_fm.encoding.run_fig4 import Fig4Config, run_fig4_session

    config = Fig4Config(
        bin_s=args.bin_s,
        top_n_motifs=args.top_n,
        n_permutations=args.n_perm,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_tuning = []
    all_decoding = []
    all_lag = []
    all_next = []

    print(f"\n{'='*60}")
    print("Fig 4 Batch Runner")
    print(f"  Sessions: {args.sessions}")
    print(f"  Config: bin_s={config.bin_s}, top_n={config.top_n_motifs}, "
          f"n_perm={config.n_permutations}")
    print(f"  Output: {out_root}")
    print(f"{'='*60}\n")

    for sess_key in args.sessions:
        yaml_path = SESSIONS[sess_key]
        sess_out = out_root / SESSION_OUTPUT_DIRS[sess_key]

        t0 = time.time()
        print(f"\n--- {sess_key} ---")

        results = run_fig4_session(
            session_yaml=yaml_path,
            output_dir=str(sess_out),
            config=config,
            debug=args.debug,
            canonical_alignment_path=args.canonical_alignment,
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        all_tuning.append(results["tuning"])
        all_decoding.append(results["decoding"])
        all_lag.append(results["lag_sweep"])
        all_next.append(results["next_syllable"])

    # Aggregate
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if all_tuning:
        pd.concat(all_tuning, ignore_index=True).to_parquet(
            summary_dir / "fig4_tuning_all.parquet", index=False
        )
    if all_decoding:
        pd.concat(all_decoding, ignore_index=True).to_parquet(
            summary_dir / "fig4_decoding_all.parquet", index=False
        )
    if all_lag:
        pd.concat(all_lag, ignore_index=True).to_parquet(
            summary_dir / "fig4_lag_sweep_all.parquet", index=False
        )
    if all_next:
        pd.concat(all_next, ignore_index=True).to_parquet(
            summary_dir / "fig4_next_syllable_all.parquet", index=False
        )

    print(f"\n{'='*60}")
    print(f"Summary saved to {summary_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
