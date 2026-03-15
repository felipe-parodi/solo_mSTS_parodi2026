#!/usr/bin/env python
"""Batch runner for Figure 2B: Unique Variance Analysis across all sessions.

Runs KIN vs VIS unique variance encoding analysis on all 5 single-FM sessions,
aggregates results, and generates pooled summary figures.

Sessions:
- M1_0525: configs/single_fm/session_M1_0525.yaml
- M2_0526: configs/single_fm/session_M2_0526.yaml
- M2_0605: configs/single_fm/session_M2_0605.yaml
- M1_0609: configs/single_fm/session_M1_0609.yaml
- M1_0613: configs/single_fm/session_M1_0613.yaml

Usage:
    # Run all sessions with full permutations
    python scripts/single_fm/run_fig2b_all_sessions.py \
        --output-dir results/single_fm \
        --n-perm 1000

    # Quick test with fewer permutations and diagnostics
    python scripts/single_fm/run_fig2b_all_sessions.py \
        --output-dir results/single_fm \
        --n-perm 100 \
        --generate-diagnostics \
        --debug

    # Run specific sessions only
    python scripts/single_fm/run_fig2b_all_sessions.py \
        --output-dir results/single_fm \
        --sessions M1_0525 M2_0526 \
        --n-perm 100

Output Structure:
    results/single_fm/
    ├── single_0525_M1/
    │   ├── encoding/fig2b_unique_variance.parquet
    │   ├── figures/fig2b_*.{png,svg}
    │   └── diagnostics/ (if --generate-diagnostics)
    ├── single_0526_M2/
    │   └── ... (same structure)
    ├── ... (other sessions)
    └── summary/
        ├── fig2b_summary.parquet (pooled)
        └── figures/fig2b_*.{png,svg} (pooled)
"""

from __future__ import annotations

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # scripts/single_fm -> freemat_os
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Session configuration mapping
SESSIONS: Dict[str, str] = {
    "M1_0525": "configs/single_fm/session_M1_0525.yaml",
    "M2_0526": "configs/single_fm/session_M2_0526.yaml",
    "M2_0605": "configs/single_fm/session_M2_0605.yaml",
    "M1_0609": "configs/single_fm/session_M1_0609.yaml",
    "M1_0613": "configs/single_fm/session_M1_0613.yaml",
}

# Session ID to output dir mapping (matches expected session_id in YAMLs)
SESSION_OUTPUT_DIRS: Dict[str, str] = {
    "M1_0525": "single_0525_M1",
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch runner for Figure 2B across all sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory (e.g., results/single_fm)",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=list(SESSIONS.keys()),
        default=None,
        help="Specific sessions to run (default: all). Choices: " + ", ".join(SESSIONS.keys()),
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
        default=100,
        help="Number of shuffle iterations for Testard method averaging (default: 100)",
    )
    parser.add_argument(
        "--n-perm-null",
        type=int,
        default=500,
        help="Number of permutations for null distributions / p-values (default: 500, 0 to skip)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of CV folds (default: 10 for Testard alignment)",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.005,
        help="Minimum delta R^2 for significance (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="P-value threshold (default: 0.05)",
    )
    parser.add_argument(
        "--generate-diagnostics",
        action="store_true",
        help="Generate KIN/VIS feature diagnostic plots per session",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (only save parquet results)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions that already have results",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs per session (-1 = all CPUs, 1 = sequential, default: -1)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for ridge regression (requires CuPy)",
    )
    parser.add_argument(
        "--use-2d-only",
        action="store_true",
        help="Use 2D floor-plane features only (no v_z, gaze_z, v_forward)",
    )
    parser.add_argument(
        "--floor-filter",
        action="store_true",
        default=True,
        help="Filter to time bins where monkey is on the floor (default: True)",
    )
    parser.add_argument(
        "--no-floor-filter",
        action="store_true",
        help="Disable floor filtering",
    )
    parser.add_argument(
        "--floor-height-threshold",
        type=float,
        default=0.5,
        help="Max height above floor (meters) for floor filter (default: 0.5)",
    )
    parser.add_argument(
        "--floor-filter-before-features",
        action="store_true",
        help="NaN-mask off-floor frames BEFORE feature computation (default: False = compute-then-filter)",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.01,
        help="Minimum R² for unit inclusion (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--min-bins",
        type=int,
        default=150,
        help="Minimum bins per unit (default: 150, skip units with fewer)",
    )
    # GAM and time drift options
    parser.add_argument(
        "--gam",
        action="store_true",
        help="Use GAM with splines on VIS features instead of pure Ridge",
    )
    parser.add_argument(
        "--gam-n-knots",
        type=int,
        default=5,
        help="Number of knots for GAM splines (default: 5)",
    )
    parser.add_argument(
        "--time-drift",
        action="store_true",
        default=True,
        help="Include linear time drift as control covariate (default: True)",
    )
    parser.add_argument(
        "--no-time-drift",
        action="store_true",
        help="Disable time drift covariate",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all 4 configurations: Ridge/GAM × with/without time drift",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    # Unit filtering options
    parser.add_argument(
        "--sua-only",
        action="store_true",
        help="Include only SUA (single-unit) neurons, exclude MUA",
    )
    parser.add_argument(
        "--mua-only",
        action="store_true",
        help="Include only MUA (multi-unit) neurons, exclude SUA",
    )
    # Testard-style options
    parser.add_argument(
        "--use-shuffle-method",
        action="store_true",
        default=True,
        help="Use Testard-style regressor-shuffle method (default: True)",
    )
    parser.add_argument(
        "--no-shuffle-method",
        action="store_true",
        help="Use dropout method instead of regressor-shuffle",
    )
    parser.add_argument(
        "--lag-s",
        type=float,
        default=1.0,
        help="Temporal kernel half-window in seconds (default: 1.0 = ±1.0s; 0 to disable)",
    )
    parser.add_argument(
        "--save-residuals",
        action="store_true",
        help="Save per-bin residuals + COM position for spatial analysis (Fig 2D)",
    )
    parser.add_argument(
        "--no-pos",
        action="store_true",
        help="Exclude 3D POS (com_x, com_y, com_z) from feature groups (3-way only)",
    )
    parser.add_argument(
        "--canonical-alignment",
        type=str,
        default=None,
        help="Path to canonical_alignment.json for cross-session arena alignment",
    )
    parser.add_argument(
        "--include-motif",
        action="store_true",
        help="Add motif identity (one-hot) as a 5th feature group (Fig 4C analysis)",
    )

    return parser.parse_args()


def run_session(
    session_key: str,
    session_yaml: str,
    output_dir: str,
    config: "Fig2BConfig",  # noqa: F821
    generate_figures: bool = True,
    canonical_alignment_path: Optional[str] = None,
) -> Optional["pd.DataFrame"]:  # noqa: F821
    """Run Figure 2B analysis for a single session.

    Args:
        session_key: Session identifier key (e.g., "M1_0525").
        session_yaml: Path to session YAML.
        output_dir: Output directory for this session.
        config: Analysis configuration.
        generate_figures: Whether to generate figures.
        canonical_alignment_path: Path to canonical alignment JSON.

    Returns:
        Results DataFrame or None if failed.
    """

    from freemat.single_fm.encoding.run_fig2b import run_fig2b_encoding
    from freemat.single_fm.viz.fig2b import generate_fig2b_panel_nway

    print(f"\n{'='*60}")
    print(f"[Batch] Processing session: {session_key}")
    print(f"[Batch] YAML: {session_yaml}")
    print(f"[Batch] Output: {output_dir}")
    print(f"[Batch] Config: GAM={config.use_gam}, drift={config.include_time_drift}, POS={config.include_pos}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run encoding analysis
        results_df = run_fig2b_encoding(
            session_yaml,
            output_dir,
            config=config,
            save_results=True,
            canonical_alignment_path=canonical_alignment_path,
        )

        if len(results_df) == 0:
            print(f"[Batch] Warning: No results for {session_key}")
            return None

        # Generate figures
        if generate_figures:
            figures_dir = str(Path(output_dir) / "figures")
            # Detect N-way columns dynamically
            _has_nway = (
                "unique_body_kin_dropout" in results_df.columns
                or "unique_body_kin" in results_df.columns
                or "unique_body_kin_shuffle" in results_df.columns
            )
            if _has_nway:
                generate_fig2b_panel_nway(
                    results_df,
                    figures_dir,
                    effect_threshold=config.effect_threshold,
                )
                print(f"[Batch] Figures saved to: {figures_dir}")

        elapsed = time.time() - start_time
        print(f"[Batch] Completed {session_key} in {elapsed:.1f}s")
        print(f"[Batch] Results: {len(results_df)} units")

        return results_df

    except FileNotFoundError as e:
        print(f"[Batch] ERROR: Data not found for {session_key}: {e}")
        return None
    except Exception as e:
        print(f"[Batch] ERROR: Failed for {session_key}: {e}")
        if config.debug:
            import traceback
            traceback.print_exc()
        return None


def get_config_suffix(use_gam: bool, include_time_drift: bool) -> str:
    """Generate suffix for config-specific output directories.

    Args:
        use_gam: Whether GAM is enabled.
        include_time_drift: Whether time drift is enabled.

    Returns:
        Suffix string like "_ridge_no_drift" or "_gam_with_drift".
    """
    model = "gam" if use_gam else "ridge"
    drift = "with_drift" if include_time_drift else "no_drift"
    return f"_{model}_{drift}"


def run_all_configs(
    args: argparse.Namespace,
    sessions_to_run: List[str],
    base_config_kwargs: Dict,
) -> int:
    """Run all 4 configurations: Ridge/GAM × with/without time drift.

    Args:
        args: Parsed command-line arguments.
        sessions_to_run: List of session keys to process.
        base_config_kwargs: Base configuration kwargs (shared across all configs).

    Returns:
        Exit code (0 = success, 1 = any failures).
    """
    import pandas as pd

    from freemat.single_fm.encoding.run_fig2b import Fig2BConfig, aggregate_fig2b_results
    from freemat.single_fm.viz.fig2b import generate_fig2b_panel_nway

    configs = [
        (False, False),  # Ridge, no drift
        (False, True),   # Ridge, with drift
        (True, False),   # GAM, no drift
        (True, True),    # GAM, with drift
    ]

    any_failures = False

    for use_gam, include_time_drift in configs:
        suffix = get_config_suffix(use_gam, include_time_drift)
        config_label = f"{'GAM' if use_gam else 'Ridge'}, {'with' if include_time_drift else 'no'} drift"

        print(f"\n{'#'*70}")
        print(f"# Configuration: {config_label}")
        print(f"{'#'*70}")

        config = Fig2BConfig(
            use_gam=use_gam,
            gam_n_knots=args.gam_n_knots,
            include_time_drift=include_time_drift,
            **base_config_kwargs,
        )

        all_results: List[pd.DataFrame] = []
        successful_sessions: List[str] = []
        failed_sessions: List[str] = []

        for session_key in sessions_to_run:
            session_yaml = SESSIONS[session_key]
            session_out_dir = SESSION_OUTPUT_DIRS.get(session_key, session_key)
            # Add config suffix to output directory
            out_dir = str(Path(args.output_dir) / f"{session_out_dir}{suffix}")

            existing_parquet = Path(out_dir) / "encoding" / "fig2b_unique_variance.parquet"
            if args.skip_existing and existing_parquet.exists():
                print(f"\n[Batch] Skipping {session_key} (results exist for {config_label})")
                try:
                    existing_df = pd.read_parquet(existing_parquet)
                    all_results.append(existing_df)
                    successful_sessions.append(session_key)
                except Exception as e:
                    print(f"[Batch] Warning: Could not load existing results: {e}")
                continue

            results_df = run_session(
                session_key=session_key,
                session_yaml=session_yaml,
                output_dir=out_dir,
                config=config,
                generate_figures=not args.no_figures,
                canonical_alignment_path=args.canonical_alignment,
            )

            if results_df is not None and len(results_df) > 0:
                all_results.append(results_df)
                successful_sessions.append(session_key)
            else:
                failed_sessions.append(session_key)

        # Aggregate for this configuration
        if len(all_results) > 0:
            summary_dir = Path(args.output_dir) / f"summary{suffix}"
            summary_parquet = str(summary_dir / "fig2b_summary.parquet")
            combined_df = aggregate_fig2b_results(all_results, output_path=summary_parquet)

            print(f"\n[Batch] {config_label} - {len(combined_df)} total neurons")
            print(f"[Batch] Summary saved to: {summary_parquet}")

            if not args.no_figures:
                figures_dir = str(summary_dir / "figures")
                has_nway = (
                    "unique_body_kin_dropout" in combined_df.columns
                    or "unique_body_kin" in combined_df.columns
                    or "unique_body_kin_shuffle" in combined_df.columns
                )
                if has_nway:
                    generate_fig2b_panel_nway(
                        combined_df,
                        figures_dir,
                        effect_threshold=args.effect_threshold,
                    )

        if failed_sessions:
            any_failures = True
            print(f"[Batch] Failed sessions for {config_label}: {', '.join(failed_sessions)}")

    return 1 if any_failures else 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Import here to avoid slow startup
    import pandas as pd

    from freemat.single_fm.encoding.run_fig2b import Fig2BConfig, aggregate_fig2b_results
    from freemat.single_fm.viz.fig2b import (
        generate_fig2b_panel,
        generate_fig2b_panel_nway,
        plot_fig2b_testard_boxplots,
        plot_fig2c_unique_scatter,
        compute_testard_summary,
    )

    # Resolve flag conflicts
    floor_filter = args.floor_filter and not args.no_floor_filter
    use_shuffle_method = args.use_shuffle_method and not args.no_shuffle_method
    include_time_drift = args.time_drift and not args.no_time_drift

    # Always use 3-way partitioning (VIS vs BODY_KIN vs HEAD_KIN)
    use_3way = True

    # Determine which sessions to run
    sessions_to_run = args.sessions if args.sessions else list(SESSIONS.keys())

    # Resolve unit filtering (sua_only vs mua_only)
    sua_only = args.sua_only
    mua_only = args.mua_only
    if sua_only and mua_only:
        print("[Batch] ERROR: Cannot specify both --sua-only and --mua-only")
        return 1

    # Base configuration kwargs (shared across all configs when using --all-configs)
    include_pos = not args.no_pos
    base_config_kwargs = dict(
        bin_s=args.bin_s,
        cv_folds=args.cv_folds,
        n_perm=args.n_perm,
        n_perm_null=args.n_perm_null,
        effect_threshold=args.effect_threshold,
        p_threshold=args.p_threshold,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        use_2d_only=args.use_2d_only,
        floor_filter=floor_filter,
        floor_height_threshold=args.floor_height_threshold,
        floor_filter_before_features=args.floor_filter_before_features,
        use_shuffle_method=use_shuffle_method,
        lag_s=args.lag_s,
        r2_threshold=args.r2_threshold,
        min_bins_per_unit=args.min_bins,
        generate_diagnostics=args.generate_diagnostics,
        save_residuals=args.save_residuals,
        include_pos=include_pos,
        sua_only=sua_only,
        mua_only=mua_only,
        include_motif=args.include_motif,
        debug=args.debug,
    )

    # Handle --all-configs mode
    if args.all_configs:
        print("\n[Batch] Running ALL 4 configurations (Ridge/GAM × with/without drift)")
        print(f"[Batch] Sessions: {', '.join(sessions_to_run)}")
        return run_all_configs(args, sessions_to_run, base_config_kwargs)

    # Single configuration mode
    config = Fig2BConfig(
        include_time_drift=include_time_drift,
        use_gam=args.gam,
        gam_n_knots=args.gam_n_knots,
        **base_config_kwargs,
    )

    print("\n[Batch] Figure 2B: Unique Variance Analysis (Testard et al. 2024 alignment)")
    print(f"[Batch] Sessions: {', '.join(sessions_to_run)}")
    print("[Batch] Configuration:")
    print(f"        bin_s: {config.bin_s}s")
    print(f"        n_perm: {config.n_perm} (shuffle iterations)")
    print(f"        n_perm_null: {config.n_perm_null} (null distribution)")
    print(f"        cv_folds: {config.cv_folds}")
    print(f"        effect_threshold: {config.effect_threshold}")
    print(f"        r2_threshold: {config.r2_threshold}")
    print(f"        min_bins_per_unit: {config.min_bins_per_unit}")
    print(f"        n_jobs: {config.n_jobs}")
    print(f"        use_gpu: {config.use_gpu}")
    print(f"        use_2d_only: {config.use_2d_only}")
    print(f"        floor_filter: {config.floor_filter}")
    if config.floor_filter:
        print(f"        floor_height_threshold: {config.floor_height_threshold}m")
        print(f"        floor_filter_before_features: {config.floor_filter_before_features}")
    print(f"        use_shuffle_method: {config.use_shuffle_method}")
    print(f"        lag_s: {config.lag_s}s -> n_lags={config.n_lags} (+/-{config.n_lags * config.bin_s}s temporal kernel)")
    print(f"        use_gam: {config.use_gam}")
    if config.use_gam:
        print(f"        gam_n_knots: {config.gam_n_knots}")
    print(f"        include_time_drift: {config.include_time_drift}")
    print(f"        generate_diagnostics: {config.generate_diagnostics}")
    print(f"        save_residuals: {config.save_residuals}")
    print(f"        include_pos: {config.include_pos}")

    # Process each session
    all_results: List[pd.DataFrame] = []
    successful_sessions: List[str] = []
    failed_sessions: List[str] = []

    total_start = time.time()

    for session_key in sessions_to_run:
        session_yaml = SESSIONS[session_key]
        session_out_dir = SESSION_OUTPUT_DIRS.get(session_key, session_key)
        out_dir = str(Path(args.output_dir) / session_out_dir)

        # Check if results exist
        existing_parquet = Path(out_dir) / "encoding" / "fig2b_unique_variance.parquet"
        if args.skip_existing and existing_parquet.exists():
            print(f"\n[Batch] Skipping {session_key} (results exist)")
            try:
                existing_df = pd.read_parquet(existing_parquet)
                all_results.append(existing_df)
                successful_sessions.append(session_key)
            except Exception as e:
                print(f"[Batch] Warning: Could not load existing results: {e}")
            continue

        # Run analysis
        results_df = run_session(
            session_key=session_key,
            session_yaml=session_yaml,
            output_dir=out_dir,
            config=config,
            generate_figures=not args.no_figures,
        )

        if results_df is not None and len(results_df) > 0:
            all_results.append(results_df)
            successful_sessions.append(session_key)
        else:
            failed_sessions.append(session_key)

    total_elapsed = time.time() - total_start

    # Aggregate results
    print(f"\n{'='*60}")
    print("[Batch] SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Successful sessions: {len(successful_sessions)}/{len(sessions_to_run)}")

    if failed_sessions:
        print(f"Failed sessions: {', '.join(failed_sessions)}")

    if len(all_results) > 0:
        # Aggregate
        summary_dir = Path(args.output_dir) / "summary"
        summary_parquet = str(summary_dir / "fig2b_summary.parquet")

        combined_df = aggregate_fig2b_results(all_results, output_path=summary_parquet)

        print("\n[Batch] Aggregated results:")
        print(f"        Total neurons: {len(combined_df)}")
        print(f"        Sessions: {combined_df['session_id'].nunique()}")

        # Detect 3-way columns (shuffle or dropout)
        has_3way = (
            "unique_body_kin_dropout" in combined_df.columns
            or "unique_body_kin" in combined_df.columns
            or "unique_body_kin_shuffle" in combined_df.columns
        )

        # Discover groups dynamically for summary reporting
        _group_order = ["vis", "body_kin", "head_kin", "pos"]
        _group_display = {"vis": "VIS", "body_kin": "BODY_KIN", "head_kin": "HEAD_KIN", "pos": "3D POS"}
        discovered_groups = [g for g in _group_order if f"sig_{g}" in combined_df.columns]

        # Per-session breakdown
        print("\n[Batch] Per-session breakdown:")
        for sid in combined_df["session_id"].unique():
            sid_df = combined_df[combined_df["session_id"] == sid]
            n = len(sid_df)
            if discovered_groups:
                sig_parts = [f"{sid_df[f'sig_{g}'].sum()} {_group_display.get(g, g)} sig" for g in discovered_groups]
                print(f"        {sid}: {n} units, {', '.join(sig_parts)}")
            elif "sig_kin" in combined_df.columns:
                n_sig = sid_df["sig_kin"].sum()
                print(f"        {sid}: {n} units, {n_sig} significant ({100*n_sig/n:.1f}%)")

        # Overall statistics (Testard-style reporting)
        print("\n[Batch] Overall statistics:")
        has_shuffle = "unique_vis_shuffle" in combined_df.columns
        has_r2_gate = "r2_threshold_passed" in combined_df.columns
        r2_passed = combined_df["r2_threshold_passed"] if has_r2_gate else pd.Series([True] * len(combined_df))
        n_pass_r2 = r2_passed.sum()

        if discovered_groups:
            # N-way stats with Testard-style reporting
            print(f"        R² > {config.r2_threshold}: {n_pass_r2} ({100*n_pass_r2/len(combined_df):.1f}%)")

            # Significance counts (dynamic)
            for g in discovered_groups:
                n_sig_g = combined_df[f"sig_{g}"].sum()
                print(f"        Significant {_group_display.get(g, g)}: {n_sig_g} ({100*n_sig_g/len(combined_df):.1f}%)")

            # FDR-corrected significance counts
            fdr_parts = []
            for g in discovered_groups:
                fdr_col = f"p_{g}_fdr"
                if fdr_col in combined_df.columns:
                    n_fdr = (combined_df[fdr_col] < 0.05).sum()
                    fdr_parts.append(f"{_group_display.get(g, g)}={n_fdr}")
            if fdr_parts:
                print(f"        FDR-significant (p<0.05): {', '.join(fdr_parts)}")

            # Shuffle-sample p-value summary (primary)
            for g in discovered_groups:
                p_col = f"p_{g}_shuffle"
                if p_col in combined_df.columns:
                    ps = combined_df[p_col].dropna()
                    if len(ps) > 0:
                        n_raw_sig = (ps < 0.05).sum()
                        print(f"        {g}: {n_raw_sig} raw p<0.05 ({100*n_raw_sig/len(ps):.1f}%), median p={ps.median():.4f}")

            # Legacy block vs circular comparison (diagnostic only)
            if "p_vis_block" in combined_df.columns and "p_vis_circular" in combined_df.columns:
                has_pvals = combined_df["p_vis_block"].notna().any()
                if has_pvals:
                    for g in discovered_groups:
                        pb_col = f"p_{g}_block"
                        pc_col = f"p_{g}_circular"
                        if pb_col in combined_df.columns and pc_col in combined_df.columns:
                            pb = combined_df[pb_col].dropna()
                            pc = combined_df[pc_col].dropna()
                            if len(pb) > 0:
                                block_tighter = (pb > pc).mean()
                                print(f"        {g}: block tighter than circular in {100*block_tighter:.0f}% of units")

            # Determine unique variance columns dynamically
            unique_cols = {}
            for g in discovered_groups:
                if has_shuffle:
                    unique_cols[g] = f"unique_{g}_shuffle"
                elif f"unique_{g}_dropout" in combined_df.columns:
                    unique_cols[g] = f"unique_{g}_dropout"
                else:
                    unique_cols[g] = f"unique_{g}"

            # Per-session medians (Testard-style: use median, not mean)
            print("\n[Batch] Per-session median unique variance:")
            for g in discovered_groups:
                col = unique_cols[g]
                if col in combined_df.columns:
                    session_medians = combined_df.groupby("session_id")[col].median()
                    label = _group_display.get(g, g)
                    print(f"        {label}: median={session_medians.median():.4f}, range=[{session_medians.min():.4f}, {session_medians.max():.4f}]")

            # Fraction negative (diagnostic for overfitting)
            print("\n[Batch] Fraction negative (overfitting diagnostic):")
            for g in discovered_groups:
                col = unique_cols[g]
                if col in combined_df.columns:
                    frac_neg = (combined_df[col] < 0).mean()
                    print(f"        {_group_display.get(g, g)}: {100*frac_neg:.1f}%")

            # Full contribution (if shuffle method)
            if has_shuffle:
                print("\n[Batch] Per-session median full contribution (shuffle method):")
                for g in discovered_groups:
                    fc_col = f"full_contrib_{g}"
                    if fc_col in combined_df.columns:
                        session_full = combined_df.groupby("session_id")[fc_col].median()
                        print(f"        {_group_display.get(g, g)}: median={session_full.median():.4f}")

        else:
            # 2-way stats
            n_sig_kin = combined_df["sig_kin"].sum()
            n_sig_vis = combined_df["sig_vis"].sum()
            mean_unique_kin = combined_df["unique_kin"].mean()
            mean_unique_vis = combined_df["unique_vis"].mean()
            print(f"        Significant KIN: {n_sig_kin} ({100*n_sig_kin/len(combined_df):.1f}%)")
            print(f"        Significant VIS: {n_sig_vis} ({100*n_sig_vis/len(combined_df):.1f}%)")
            print(f"        Mean unique KIN: {mean_unique_kin:.4f}")
            print(f"        Mean unique VIS: {mean_unique_vis:.4f}")

        # Per-bank breakdown (if bank info available)
        if "unit_bank" in combined_df.columns and combined_df["unit_bank"].notna().any():
            print("\n[Batch] Per-bank breakdown:")
            for bank in sorted(combined_df["unit_bank"].dropna().unique()):
                bank_df = combined_df[combined_df["unit_bank"] == bank]
                n_bank = len(bank_df)
                if discovered_groups:
                    print(f"        {bank}: {n_bank} units")
                    for g in discovered_groups:
                        label = _group_display.get(g, g)
                        n_sig_g = bank_df[f"sig_{g}"].sum() if f"sig_{g}" in bank_df.columns else 0
                        col = unique_cols.get(g, f"unique_{g}_shuffle")
                        mean_g = bank_df[col].mean() if col in bank_df.columns else 0.0
                        print(f"            {label}: {n_sig_g} sig ({100*n_sig_g/n_bank:.1f}%), mean={mean_g:.4f}")
                else:
                    n_sig_kin = bank_df["sig_kin"].sum()
                    n_sig_vis = bank_df["sig_vis"].sum()
                    print(f"        {bank}: {n_bank} units, {n_sig_kin} KIN sig, {n_sig_vis} VIS sig")

        # Generate aggregated figures
        if not args.no_figures:
            figures_dir = str(summary_dir / "figures")
            if use_3way and has_3way:
                generate_fig2b_panel_nway(
                    combined_df,
                    figures_dir,
                    effect_threshold=args.effect_threshold,
                )
                # Generate Testard-style boxplots
                plot_fig2b_testard_boxplots(
                    combined_df,
                    output_path=str(Path(figures_dir) / "fig2b_testard_boxplots.png"),
                    split_by_bank="unit_bank" in combined_df.columns,
                )
                print("[Batch] Generated Testard-style boxplots")
                # Generate unique scatter plots (Fig 2C)
                plot_fig2c_unique_scatter(
                    combined_df,
                    output_path=str(Path(figures_dir) / "fig2c_scatter_vis_vs_body_kin.png"),
                    split_by_bank="unit_bank" in combined_df.columns,
                    x_group="vis", y_group="body_kin",
                )
                if include_pos and "unique_pos_shuffle" in combined_df.columns:
                    plot_fig2c_unique_scatter(
                        combined_df,
                        output_path=str(Path(figures_dir) / "fig2c_scatter_vis_vs_pos.png"),
                        split_by_bank="unit_bank" in combined_df.columns,
                        x_group="vis", y_group="pos",
                    )
                print("[Batch] Generated unique scatter plots (Fig 2C)")
            else:
                generate_fig2b_panel(
                    combined_df,
                    figures_dir,
                    effect_threshold=args.effect_threshold,
                )
            print(f"\n[Batch] Aggregated figures saved to: {figures_dir}")

        # Save Testard-style summary parquet
        if use_3way:
            testard_summary = compute_testard_summary(combined_df)
            testard_path = str(summary_dir / "fig2b_testard_summary.parquet")
            testard_summary.to_parquet(testard_path, index=False)
            print(f"[Batch] Testard summary saved to: {testard_path}")

        print(f"\n[Batch] Summary saved to: {summary_parquet}")

    print("\n[Batch] Done.")
    return 0 if not failed_sessions else 1


if __name__ == "__main__":
    sys.exit(main())
