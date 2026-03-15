#!/usr/bin/env python
"""Fig 5: Post-hoc manifold analyses on existing CEBRA embeddings.

Analyses:
  1. Embedding RSA (centroid distances ~ kinematic similarity)
  4. Anticipatory drift (neural state drifts toward next motif before transition)
  6. Factored vs conjunctive coding (do dimensions specialize?)
  7. Velocity → transition prediction (neural speed predicts upcoming transitions)
  8. Timescale separation (neural vs behavioral autocorrelation)

Usage:
    python scripts/single_fm/run_fig5_manifold_analyses.py
    python scripts/single_fm/run_fig5_manifold_analyses.py --sessions hedy_0526 --n-perm 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from freemat.single_fm.encoding.fig5_manifold_analyses import run_all_sessions  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fig 5: Post-hoc manifold analyses on CEBRA embeddings"
    )
    parser.add_argument(
        "--sessions", nargs="*", default=None,
        help="Sessions to run (default: all 5)",
    )
    parser.add_argument(
        "--fig5-dir", type=str, default=None,
        help="CEBRA results directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument("--n-perm", type=int, default=500)
    args = parser.parse_args()

    run_all_sessions(
        sessions=args.sessions,
        fig5_dir=Path(args.fig5_dir) if args.fig5_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        n_perm=args.n_perm,
    )
