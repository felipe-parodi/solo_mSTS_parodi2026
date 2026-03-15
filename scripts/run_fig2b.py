#!/usr/bin/env python
"""CLI entry point for Figure 2B: Unique Variance Analysis (KIN vs VIS).

Computes unique variance explained by kinematic features beyond visual proxies
for mid-STS neural activity.

Usage:
    python scripts/single_fm/run_fig2b.py \
        --session-yaml configs/session_M1_0525.yaml \
        --output-dir results/240525 \
        --n-perm 1000

    # Quick test with fewer permutations
    python scripts/single_fm/run_fig2b.py \
        --session-yaml configs/session_M1_0525.yaml \
        --output-dir results/240525 \
        --n-perm 100 \
        --debug

    # Process multiple sessions
    python scripts/single_fm/run_fig2b.py \
        --session-yaml configs/session_M1_0525.yaml \
        --session-yaml configs/session_M1_0609.yaml \
        --output-dir results/summary \
        --aggregate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # scripts/single_fm -> freemat_os
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Figure 2B: Unique Variance Analysis (KIN vs VIS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--session-yaml",
        type=str,
        action="append",
        required=True,
        help="Path to session YAML (can specify multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results and figures",
    )
    parser.add_argument(
        "--bin-s",
        type=float,
        default=0.5,
        help="Bin size in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=200,
        help="Number of permutations for significance (default: 200, 0 to skip)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.005,
        help="Minimum delta R² for significance (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="P-value threshold (default: 0.05)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results across sessions",
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        default=True,
        help="Generate Figure 2B plots (default: True)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--generate-diagnostics",
        action="store_true",
        help="Generate KIN/VIS feature diagnostic plots (correlation, timeseries, etc.)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all CPUs, 1 = sequential, default: -1)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for ridge regression (requires CuPy)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Import here to avoid slow startup
    import pandas as pd

    from freemat.single_fm.encoding.run_fig2b import (
        Fig2BConfig,
        run_fig2b_encoding,
        aggregate_fig2b_results,
    )
    from freemat.single_fm.viz.fig2b import generate_fig2b_panel

    # Configuration
    config = Fig2BConfig(
        bin_s=args.bin_s,
        cv_folds=args.cv_folds,
        n_perm=args.n_perm,
        effect_threshold=args.effect_threshold,
        p_threshold=args.p_threshold,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        generate_diagnostics=args.generate_diagnostics,
        debug=args.debug,
    )

    if args.debug:
        print(f"[CLI] Configuration:")
        print(f"      bin_s: {config.bin_s}")
        print(f"      cv_folds: {config.cv_folds}")
        print(f"      n_perm: {config.n_perm}")
        print(f"      effect_threshold: {config.effect_threshold}")
        print(f"      p_threshold: {config.p_threshold}")
        print(f"      n_jobs: {config.n_jobs}")
        print(f"      use_gpu: {config.use_gpu}")

    # Process each session
    all_results: List[pd.DataFrame] = []

    for session_yaml in args.session_yaml:
        print(f"\n[CLI] Processing: {session_yaml}")

        # Determine output directory
        if len(args.session_yaml) == 1 or args.aggregate:
            out_dir = args.output_dir
        else:
            # Per-session output
            session_name = Path(session_yaml).stem.replace("session_", "")
            out_dir = str(Path(args.output_dir) / session_name)

        # Run analysis
        results_df = run_fig2b_encoding(
            session_yaml,
            out_dir,
            config=config,
            save_results=True,
        )

        if len(results_df) > 0:
            all_results.append(results_df)

            # Generate figures for this session
            if args.generate_figures and not args.no_figures:
                figures_dir = str(Path(out_dir) / "figures")
                generate_fig2b_panel(
                    results_df,
                    figures_dir,
                    effect_threshold=args.effect_threshold,
                )
                print(f"[CLI] Figures saved to: {figures_dir}")
        else:
            print(f"[CLI] Warning: No results for {session_yaml}")

    # Aggregate if multiple sessions
    if args.aggregate and len(all_results) > 1:
        print(f"\n[CLI] Aggregating {len(all_results)} sessions...")

        combined_df = aggregate_fig2b_results(
            all_results,
            output_path=str(Path(args.output_dir) / "summary" / "fig2b_summary.parquet"),
        )

        # Generate aggregated figures
        if args.generate_figures and not args.no_figures:
            figures_dir = str(Path(args.output_dir) / "summary" / "figures")
            generate_fig2b_panel(
                combined_df,
                figures_dir,
                effect_threshold=args.effect_threshold,
            )
            print(f"[CLI] Aggregated figures saved to: {figures_dir}")

        # Print summary statistics
        n_total = len(combined_df)
        n_sig_kin = combined_df["sig_kin"].sum()
        n_sig_vis = combined_df["sig_vis"].sum()

        print(f"\n[CLI] === Summary ===")
        print(f"      Total neurons: {n_total}")
        print(f"      Significant KIN: {n_sig_kin} ({100*n_sig_kin/n_total:.1f}%)")
        print(f"      Significant VIS: {n_sig_vis} ({100*n_sig_vis/n_total:.1f}%)")
        print(f"      Mean unique KIN: {combined_df['unique_kin'].mean():.4f}")
        print(f"      Mean unique VIS: {combined_df['unique_vis'].mean():.4f}")

    print("\n[CLI] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
