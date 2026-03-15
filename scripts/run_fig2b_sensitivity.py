#!/usr/bin/env python
"""Sensitivity sweep for Figure 2B encoding analysis.

Runs multiple sensitivity configurations (bin sizes, model types, unit filters)
and produces a cross-configuration comparison report.

Configurations:
- primary:   bin_s=0.5, lag_s=1.0, Ridge, all units (baseline)
- bin_0.3s:  bin_s=0.3, lag_s=1.0, Ridge, all units (finer temporal resolution)
- bin_1.0s:  bin_s=1.0, lag_s=1.0, Ridge, all units (coarser, more stable)
- gam:       bin_s=0.5, lag_s=1.0, GAM,   all units (nonlinear tuning curves)
- sua_only:  bin_s=0.5, lag_s=1.0, Ridge, SUA only
- mua_only:  bin_s=0.5, lag_s=1.0, Ridge, MUA only

Usage:
    # Run all 6 configs
    python scripts/single_fm/run_fig2b_sensitivity.py \
        --output-dir results/single_fm \
        --n-perm 100 --n-perm-null 50

    # Run subset of configs, skip existing results
    python scripts/single_fm/run_fig2b_sensitivity.py \
        --output-dir results/single_fm \
        --configs primary gam \
        --skip-existing

Output Structure:
    results/single_fm_sensitivity/
    +-- primary/
    |   +-- single_0525_M1/encoding/fig2b_unique_variance.parquet
    |   +-- .../
    |   +-- summary/fig2b_summary.parquet
    +-- bin_0.3s/
    |   +-- ... (same layout)
    +-- .../
    +-- comparison/
        +-- sensitivity_comparison.csv
        +-- figures/
            +-- sensitivity_comparison.{png,svg}
            +-- sensitivity_significance_rates.{png,svg}
            +-- sensitivity_session_consistency.{png,svg}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # scripts/single_fm -> freemat_os
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Session configuration (mirrored from run_fig2b_all_sessions.py)
SESSIONS: Dict[str, str] = {
    "M1_0525": "configs/single_fm/session_M1_0525.yaml",
    "hedy_0526": "configs/single_fm/session_hedy_0526.yaml",
    "M2_0605": "configs/single_fm/session_M2_0605.yaml",
    "M1_0609": "configs/single_fm/session_M1_0609.yaml",
    "M1_0613": "configs/single_fm/session_M1_0613.yaml",
}

SESSION_OUTPUT_DIRS: Dict[str, str] = {
    "M1_0525": "single_0525_M1",
    "hedy_0526": "single_0526_hedy",
    "M2_0605": "single_0605_M2",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}

# lag_s: temporal kernel half-window in seconds; n_lags computed automatically
# lag_s=1.0 gives ±1.0s regardless of bin_s
SENSITIVITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "bin_s": 0.5,
        "lag_s": 1.0,
        "use_gam": False,
        "sua_only": False,
        "mua_only": False,
    },
    "bin_0.3s": {
        "bin_s": 0.3,
        "lag_s": 1.0,
        "use_gam": False,
        "sua_only": False,
        "mua_only": False,
    },
    "bin_1.0s": {
        "bin_s": 1.0,
        "lag_s": 1.0,
        "use_gam": False,
        "sua_only": False,
        "mua_only": False,
    },
    "gam": {
        "bin_s": 0.5,
        "lag_s": 1.0,
        "use_gam": True,
        "sua_only": False,
        "mua_only": False,
    },
    "sua_only": {
        "bin_s": 0.5,
        "lag_s": 1.0,
        "use_gam": False,
        "sua_only": True,
        "mua_only": False,
    },
    "mua_only": {
        "bin_s": 0.5,
        "lag_s": 1.0,
        "use_gam": False,
        "sua_only": False,
        "mua_only": True,
    },
}

ALL_CONFIG_NAMES = list(SENSITIVITY_CONFIGS.keys())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sensitivity sweep for Figure 2B encoding analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory (e.g., results/single_fm). "
        "Sensitivity results go into {output_dir}_sensitivity/",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=ALL_CONFIG_NAMES,
        default=ALL_CONFIG_NAMES,
        help="Configs to run (default: all). "
        f"Choices: {', '.join(ALL_CONFIG_NAMES)}",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=list(SESSIONS.keys()),
        default=None,
        help="Specific sessions to run (default: all)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=100,
        help="Shuffle iterations for Testard method averaging (default: 100)",
    )
    parser.add_argument(
        "--n-perm-null",
        type=int,
        default=50,
        help="Permutations for null distributions / p-values (default: 50, 0 to skip)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of CV folds (default: 10)",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.005,
        help="Minimum delta R^2 for significance (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.01,
        help="Minimum R^2 for unit inclusion (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--min-bins",
        type=int,
        default=150,
        help="Minimum bins per unit (default: 150)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs/sessions that already have results",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip per-config figure generation (comparison figures still generated)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs per session (-1 = all CPUs, default: -1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    return parser.parse_args()


def run_sensitivity_config(
    config_name: str,
    config_kwargs: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    sessions_to_run: List[str],
    config_output_dir: str,
    skip_existing: bool = False,
    generate_figures: bool = True,
) -> Optional[List["pd.DataFrame"]]:  # noqa: F821
    """Run a single sensitivity configuration across all sessions.

    Args:
        config_name: Name of the sensitivity configuration.
        config_kwargs: Config-specific kwargs (bin_s, lag_s, use_gam, etc.).
        base_kwargs: Shared kwargs (n_perm, cv_folds, etc.).
        sessions_to_run: List of session keys to process.
        config_output_dir: Output directory for this config.
        skip_existing: Whether to skip sessions with existing results.
        generate_figures: Whether to generate per-session figures.

    Returns:
        List of per-session DataFrames, or None if all sessions failed.
    """
    import pandas as pd

    from freemat.single_fm.encoding.config import Fig2BConfig
    from freemat.single_fm.encoding.run_fig2b import run_fig2b_encoding
    from freemat.single_fm.viz.fig2b import generate_fig2b_panel_3way

    # Merge base kwargs with config-specific overrides
    merged = {**base_kwargs, **config_kwargs}
    config = Fig2BConfig(**merged)

    print(f"\n{'#'*70}")
    print(f"# Sensitivity config: {config_name}")
    print(f"#   bin_s={config.bin_s}, lag_s={config.lag_s} (n_lags={config.n_lags}), "
          f"GAM={config.use_gam}, SUA={config.sua_only}, MUA={config.mua_only}")
    print(f"#   Output: {config_output_dir}")
    print(f"{'#'*70}")

    all_results: List[pd.DataFrame] = []

    for session_key in sessions_to_run:
        session_yaml = SESSIONS[session_key]
        session_out_dir = SESSION_OUTPUT_DIRS.get(session_key, session_key)
        out_dir = str(Path(config_output_dir) / session_out_dir)

        # Check for existing results
        existing_parquet = Path(out_dir) / "encoding" / "fig2b_unique_variance.parquet"
        if skip_existing and existing_parquet.exists():
            print(f"\n[Sensitivity] Skipping {session_key}/{config_name} (results exist)")
            existing_df = pd.read_parquet(existing_parquet)
            all_results.append(existing_df)
            continue

        print(f"\n[Sensitivity] Running {session_key} with config={config_name}")
        start_time = time.time()

        try:
            results_df = run_fig2b_encoding(
                session_yaml,
                out_dir,
                config=config,
                save_results=True,
            )
        except FileNotFoundError as e:
            print(f"[Sensitivity] ERROR: Data not found for {session_key}: {e}")
            continue
        except Exception as e:
            print(f"[Sensitivity] ERROR: Failed for {session_key}: {e}")
            if config.debug:
                import traceback
                traceback.print_exc()
            continue

        if len(results_df) == 0:
            print(f"[Sensitivity] Warning: No results for {session_key}")
            continue

        elapsed = time.time() - start_time
        print(f"[Sensitivity] Completed {session_key}/{config_name} "
              f"in {elapsed:.1f}s ({len(results_df)} units)")

        # Per-session figures
        if generate_figures:
            figures_dir = str(Path(out_dir) / "figures")
            has_3way = (
                "unique_body_kin_dropout" in results_df.columns
                or "unique_body_kin" in results_df.columns
                or "unique_body_kin_shuffle" in results_df.columns
            )
            if has_3way:
                generate_fig2b_panel_3way(
                    results_df,
                    figures_dir,
                    effect_threshold=config.effect_threshold,
                )

        all_results.append(results_df)

    if len(all_results) == 0:
        print(f"[Sensitivity] No results for config={config_name}")
        return None

    return all_results


def main() -> int:
    """Main entry point for sensitivity sweep."""
    args = parse_args()

    # Lazy imports
    import pandas as pd

    from freemat.single_fm.encoding.run_fig2b import aggregate_fig2b_results
    from freemat.single_fm.viz.fig2b_comparison import (
        build_sensitivity_table,
        plot_sensitivity_comparison,
    )

    sessions_to_run = args.sessions if args.sessions else list(SESSIONS.keys())
    configs_to_run = args.configs

    # Sensitivity output root
    sensitivity_dir = Path(f"{args.output_dir}_sensitivity")
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Sensitivity] Figure 2B Sensitivity Analysis")
    print(f"[Sensitivity] Configs: {', '.join(configs_to_run)}")
    print(f"[Sensitivity] Sessions: {', '.join(sessions_to_run)}")
    print(f"[Sensitivity] Output root: {sensitivity_dir}")

    # Shared kwargs across all configs
    base_kwargs: Dict[str, Any] = {
        "cv_folds": args.cv_folds,
        "n_perm": args.n_perm,
        "n_perm_null": args.n_perm_null,
        "effect_threshold": args.effect_threshold,
        "r2_threshold": args.r2_threshold,
        "min_bins_per_unit": args.min_bins,
        "n_jobs": args.n_jobs,
        "debug": args.debug,
    }

    # Run each config and aggregate
    config_aggregated: Dict[str, pd.DataFrame] = {}
    total_start = time.time()

    for config_name in configs_to_run:
        config_kwargs = SENSITIVITY_CONFIGS[config_name]
        config_output_dir = str(sensitivity_dir / config_name)

        # Check if aggregated summary already exists
        summary_path = Path(config_output_dir) / "summary" / "fig2b_summary.parquet"
        if args.skip_existing and summary_path.exists():
            print(f"\n[Sensitivity] Loading existing summary for {config_name}")
            config_aggregated[config_name] = pd.read_parquet(summary_path)
            continue

        session_results = run_sensitivity_config(
            config_name=config_name,
            config_kwargs=config_kwargs,
            base_kwargs=base_kwargs,
            sessions_to_run=sessions_to_run,
            config_output_dir=config_output_dir,
            skip_existing=args.skip_existing,
            generate_figures=not args.no_figures,
        )

        if session_results is None:
            print(f"[Sensitivity] Skipping aggregation for {config_name} (no results)")
            continue

        # Aggregate across sessions for this config
        summary_parquet = str(summary_path)
        aggregated_df = aggregate_fig2b_results(
            session_results,
            output_path=summary_parquet,
        )
        config_aggregated[config_name] = aggregated_df

        print(f"[Sensitivity] {config_name}: {len(aggregated_df)} units aggregated")
        print(f"[Sensitivity] Summary saved: {summary_parquet}")

    total_elapsed = time.time() - total_start

    # --- Cross-config comparison ---
    if len(config_aggregated) < 2:
        print(f"\n[Sensitivity] Need at least 2 configs for comparison, got {len(config_aggregated)}")
        if len(config_aggregated) == 1:
            print("[Sensitivity] Run more configs to generate comparison report.")
        return 0

    print(f"\n{'='*70}")
    print(f"[Sensitivity] Cross-configuration comparison ({len(config_aggregated)} configs)")
    print(f"{'='*70}")

    comparison_dir = sensitivity_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    comparison_df = build_sensitivity_table(config_aggregated)

    # Save comparison CSV
    csv_path = str(comparison_dir / "sensitivity_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"[Sensitivity] Comparison table saved: {csv_path}")

    # Print summary table
    print("\n[Sensitivity] Summary Table:")
    print(f"{'Config':<12} {'Units':>6} {'R2_med':>7} "
          f"{'%VIS':>6} {'%BODY':>6} {'%HEAD':>6} "
          f"{'medVIS':>8} {'medBODY':>8} {'medHEAD':>8}")
    print("-" * 85)
    for _, row in comparison_df.iterrows():
        print(f"{row['config']:<12} {row['n_units']:>6} {row['median_full_r2']:>7.4f} "
              f"{row['pct_sig_vis']:>5.1f}% {row['pct_sig_body_kin']:>5.1f}% "
              f"{row['pct_sig_head_kin']:>5.1f}% "
              f"{row['median_unique_vis']:>8.5f} {row['median_unique_body_kin']:>8.5f} "
              f"{row['median_unique_head_kin']:>8.5f}")

    # Generate comparison figures
    figures_dir = str(comparison_dir / "figures")
    plot_sensitivity_comparison(comparison_df, figures_dir)

    # Identify strongest/most consistent config
    print("\n[Sensitivity] Best configurations:")
    for group in ["vis", "body_kin", "head_kin"]:
        best_pct = comparison_df.loc[comparison_df[f"pct_sig_{group}"].idxmax()]
        best_med = comparison_df.loc[comparison_df[f"median_unique_{group}"].idxmax()]
        best_sess = comparison_df.loc[comparison_df[f"n_sessions_with_sig_{group}"].idxmax()]
        print(f"  {group.upper()}:")
        print(f"    Highest significance rate: {best_pct['config']} ({best_pct[f'pct_sig_{group}']:.1f}%)")
        print(f"    Highest median unique var: {best_med['config']} ({best_med[f'median_unique_{group}']:.5f})")
        print(f"    Most consistent sessions:  {best_sess['config']} ({best_sess[f'n_sessions_with_sig_{group}']} sessions)")

    print(f"\n[Sensitivity] Total time: {total_elapsed:.1f}s")
    print("[Sensitivity] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
