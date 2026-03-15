#!/usr/bin/env python
"""Batch runner for Fig 3 RF-index computation.

Two modes:
1. Post-hoc (default): loads existing fig3_reference_frames.parquet per session,
   computes rf_index column. Fast, no model re-fitting.
2. Height-stratified (--height-stratify): fits shared model, scores on
   height-restricted test subsets. Slower, produces per-height RF-index.

Example (post-hoc):
    python scripts/single_fm/run_fig3_rf_index.py \
        --prod-dir results/production_20260220_2019 \
        --output-dir results/fig3_rf \
        --sessions M2_0526 --debug

Example (height-stratified):
    python scripts/single_fm/run_fig3_rf_index.py \
        --prod-dir results/production_20260220_2019 \
        --output-dir results/fig3_rf_height \
        --sessions M2_0526 \
        --height-stratify \
        --height-cuts -0.265 0.752 \
        --canonical-alignment configs/canonical_alignment.json \
        --debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index  # noqa: E402

try:
    from freemat.single_fm.viz.fig3_rf_panels import save_rf_index_panels  # noqa: E402
except ImportError:
    save_rf_index_panels = None


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
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Compute Fig 3 RF-index from existing results.")
    parser.add_argument(
        "--prod-dir", type=str, required=True,
        help="Production directory with fig3_reference_frames.parquet per session.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--sessions", nargs="+", choices=list(SESSIONS.keys()),
        default=list(SESSIONS.keys()),
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="RF-index epsilon.")
    parser.add_argument("--height-stratify", action="store_true", help="Run height-stratified analysis.")
    parser.add_argument(
        "--height-cuts", nargs=2, type=float, default=None,
        help="com_z tertile cuts (2 values). If not set, loaded from --fig2d-config.",
    )
    parser.add_argument(
        "--fig2d-config", type=str, default=None,
        help="Path to Fig2D run_config.json to load height_cuts_used.",
    )
    parser.add_argument(
        "--canonical-alignment", type=str, default=None,
        help="Canonical alignment JSON for cross-session arena alignment.",
    )
    parser.add_argument(
        "--quantile-cuts", action="store_true",
        help="Also run per-session tertile cuts (sensitivity analysis).",
    )
    parser.add_argument(
        "--split-half", action="store_true",
        help="Run split-half stability control.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_height_cuts(args: argparse.Namespace) -> List[float]:
    """Resolve height cuts from CLI args or Fig2D config.

    Args:
        args: Parsed CLI args with height_cuts and fig2d_config fields.

    Returns:
        List of 2 floats [low_cut, high_cut] for com_z tertile boundaries.

    Raises:
        SystemExit: If neither --height-cuts nor --fig2d-config provided.
    """
    if args.height_cuts is not None:
        return list(args.height_cuts)

    if args.fig2d_config is not None:
        fig2d_path = Path(args.fig2d_config)
        if fig2d_path.exists():
            with open(fig2d_path) as f:
                fig2d_cfg = json.load(f)
            cuts = fig2d_cfg.get("height_cuts_used")
            if cuts is not None and len(cuts) == 2:
                print(f"[RF-index] Loaded height cuts from Fig2D config: {cuts}")
                return list(cuts)
            print("[RF-index] WARNING: Fig2D config missing 'height_cuts_used', "
                  "falling back to data quantiles")
        else:
            print(f"[RF-index] WARNING: Fig2D config not found at {fig2d_path}")

    return []  # empty means compute from data quantiles


def _assign_height_tiers(
    com_z: "np.ndarray",
    cuts: List[float],
) -> "np.ndarray":
    """Assign height tier labels (0=floor, 1=bench, 2=ceiling) from com_z.

    Args:
        com_z: Center-of-mass vertical position per bin.
        cuts: Two-element list [low_cut, high_cut].

    Returns:
        Integer array of tier labels (0, 1, 2).
    """
    import numpy as np

    tiers = np.full(len(com_z), 1, dtype=int)  # default = bench
    tiers[com_z <= cuts[0]] = 0  # floor
    tiers[com_z >= cuts[1]] = 2  # ceiling
    return tiers


def _run_height_stratified_single_mode(
    args: argparse.Namespace,
    output_root: Path,
    cut_mode: str,
    fixed_cuts: List[float],
) -> None:
    """Run height-stratified RF-index for one cut mode (fixed or quantile).

    Per-tier re-fit: each tier gets its own alpha selection and model fitting,
    avoiding the shared-model artifact where floor-dominated models produce
    negative R² for bench/ceiling tiers.

    Args:
        args: Parsed CLI args.
        output_root: Output directory (mode-specific subdirectory).
        cut_mode: "fixed" or "quantile".
        fixed_cuts: Height cuts for "fixed" mode (ignored for "quantile").
    """
    import numpy as np
    from joblib import Parallel, delayed

    from freemat.single_fm.encoding.cv import CVConfig
    from freemat.single_fm.encoding.run_fig3 import (
        Fig3Config,
        _prepare_session_for_fig3,
        _process_single_unit_fig3_per_tier_refit,
    )

    config = Fig3Config(
        cv_folds=3,
        min_bins_per_unit=50,
        floor_filter=False,  # keep all heights
        debug=args.debug,
    )
    cv_cfg = CVConfig(
        folds=config.cv_folds,
        block_s=config.cv_block_s,
        gap_s=config.cv_gap_s,
    )

    all_stratified: List[pd.DataFrame] = []

    for session_key in args.sessions:
        session_yaml = str(_PROJECT_ROOT / SESSIONS[session_key])
        session_subdir = SESSION_OUTPUT_DIRS[session_key]

        if not Path(session_yaml).exists():
            print(f"[RF-height/{cut_mode}] {session_key}: YAML not found, skipping")
            continue

        print(f"[RF-height/{cut_mode}] {session_key}: loading session (floor_filter=False)...")
        merged_df, frame_feature_cols, base_cols, session_id = _prepare_session_for_fig3(
            session_yaml, config,
            canonical_alignment_path=args.canonical_alignment,
            floor_filter=False,
        )

        if len(merged_df) == 0:
            print(f"[RF-height/{cut_mode}] {session_key}: no merged data, skipping")
            continue

        if "com_z" not in merged_df.columns:
            print(f"[RF-height/{cut_mode}] {session_key}: no com_z column, skipping")
            continue

        # Resolve height cuts per mode
        if cut_mode == "fixed":
            cuts = fixed_cuts
        else:
            z_vals = merged_df.groupby("t_bin_s")["com_z"].first().dropna().values
            cuts = list(np.quantile(z_vals, [1 / 3, 2 / 3]))
            print(f"[RF-height/{cut_mode}] {session_key}: computed quantile cuts: {cuts}")

        print(f"[RF-height/{cut_mode}] {session_key}: height cuts = {cuts}")

        merged_df["_height_tier"] = _assign_height_tiers(
            merged_df["com_z"].values, cuts,
        )

        tier_counts = merged_df.groupby("_height_tier").size()
        if args.debug:
            print(f"[RF-height/{cut_mode}] {session_key}: tier counts = {dict(tier_counts)}")

        all_unit_ids = merged_df["unit_id"].unique()
        print(f"[RF-height/{cut_mode}] {session_key}: processing {len(all_unit_ids)} units...")

        # Pre-extract per-unit data + height tiers
        unit_data_dict = {}
        unit_tier_dict = {}
        for uid in all_unit_ids:
            mask = merged_df["unit_id"] == uid
            unit_data_dict[uid] = merged_df.loc[mask].copy()
            unit_tier_dict[uid] = merged_df.loc[mask, "_height_tier"].values

        # Bank mapping
        unit_bank_map: Dict[int, str] = {}
        if "unit_bank" in merged_df.columns:
            unit_bank_map = merged_df.groupby("unit_id")["unit_bank"].first().to_dict()

        rng_master = np.random.default_rng(42)
        unit_seeds = {uid: int(rng_master.integers(0, 2**31)) for uid in all_unit_ids}

        results = Parallel(n_jobs=config.n_jobs, verbose=10 if config.debug else 0)(
            delayed(_process_single_unit_fig3_per_tier_refit)(
                uid,
                unit_data_dict[uid],
                frame_feature_cols,
                base_cols,
                cv_cfg,
                config,
                unit_tier_dict[uid],
                session_id,
                unit_seeds[uid],
                unit_bank_map.get(uid),
            )
            for uid in all_unit_ids
        )

        # Flatten: each result is a list of dicts (one per tier) or None
        rows = [row for unit_rows in results if unit_rows is not None for row in unit_rows]

        if not rows:
            print(f"[RF-height/{cut_mode}] {session_key}: no valid units, skipping")
            continue

        session_df = pd.DataFrame(rows)
        session_df["cut_mode"] = cut_mode

        n_units = session_df["unit_id"].nunique()
        n_interp = session_df[session_df["qc_interpretable"]]["unit_id"].nunique()

        for tier_name in ["floor", "bench", "ceiling"]:
            tier_slice = session_df[
                (session_df["height_tier"] == tier_name)
                & session_df["qc_interpretable"]
            ]
            if len(tier_slice) > 0:
                med = float(tier_slice["rf_index"].median())
                print(f"[RF-height/{cut_mode}] {session_key}/{tier_name}: "
                      f"n_interp={len(tier_slice)}, median rf_index={med:.3f}")

        print(f"[RF-height/{cut_mode}] {session_key}: "
              f"{n_units} units total, {n_interp} interpretable")

        # Save per-session
        session_out = output_root / session_subdir / "encoding"
        session_out.mkdir(parents=True, exist_ok=True)
        parquet_name = f"fig3_rf_height_{cut_mode}.parquet"
        session_df.to_parquet(session_out / parquet_name, index=False)

        # Save metadata
        meta = {
            "session_id": session_key,
            "height_cuts": cuts,
            "cut_mode": cut_mode,
            "n_units": n_units,
            "n_interpretable": n_interp,
            "bin_s": config.bin_s,
            "cv_folds": config.cv_folds,
            "min_bins_per_unit": config.min_bins_per_unit,
            "floor_filter": False,
            "method": "per_tier_refit",
        }
        with open(session_out / f"fig3_rf_height_{cut_mode}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        all_stratified.append(session_df)

    if not all_stratified:
        print(f"[RF-height/{cut_mode}] No sessions produced results.")
        return

    # Aggregate
    summary_df = pd.concat(all_stratified, ignore_index=True)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    parquet_name = f"fig3_rf_height_{cut_mode}.parquet"
    summary_df.to_parquet(summary_dir / parquet_name, index=False)

    n_total = summary_df.groupby(["session_id", "unit_id"]).ngroups
    n_interp_all = summary_df[summary_df["qc_interpretable"]].groupby(
        ["session_id", "unit_id"]
    ).ngroups
    print(f"[RF-height/{cut_mode}] Summary: {n_total} units ({n_interp_all} interpretable), "
          f"{len(all_stratified)} sessions")

    # Height-stratified violin figure (if viz available)
    try:
        from freemat.single_fm.viz.fig3_rf_panels import plot_rf_index_by_height
        fig = plot_rf_index_by_height(summary_df)
        figures_dir = summary_dir / f"figures/{cut_mode}"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / "fig3_rf_height_stratified.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures_dir / "fig3_rf_height_stratified.svg", bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"[RF-height/{cut_mode}] Figures saved to {figures_dir}")
    except ImportError:
        pass


def _run_height_stratified(
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    """Run height-stratified RF-index with per-tier re-fit.

    Dispatches to _run_height_stratified_single_mode for each cut mode.

    Args:
        args: Parsed CLI args.
        output_root: Output directory.
    """
    height_cuts = _load_height_cuts(args)

    # Fixed cuts mode (requires --height-cuts or --fig2d-config)
    if height_cuts:
        _run_height_stratified_single_mode(
            args, output_root, cut_mode="fixed", fixed_cuts=height_cuts,
        )

    # Quantile cuts mode (--quantile-cuts flag)
    if args.quantile_cuts:
        _run_height_stratified_single_mode(
            args, output_root, cut_mode="quantile", fixed_cuts=[],
        )


def _run_split_half(
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    """Run split-half RF-index stability control.

    Splits each session temporally, fits independent models on each half,
    computes RF-index per half, then checks Spearman correlation.

    Args:
        args: Parsed CLI args.
        output_root: Output directory.
    """
    import numpy as np
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr

    from freemat.single_fm.encoding.cv import CVConfig
    from freemat.single_fm.encoding.fig3_rf_index import compute_rf_index
    from freemat.single_fm.encoding.run_fig3 import (
        Fig3Config,
        _prepare_session_for_fig3,
        _process_single_unit_fig3,
    )

    config = Fig3Config(
        cv_folds=3,  # reduced from 5: half the data
        min_bins_per_unit=50,  # lower threshold for half-session
        debug=args.debug,
    )
    cv_cfg = CVConfig(
        folds=config.cv_folds,
        block_s=config.cv_block_s,
        gap_s=config.cv_gap_s,
    )

    all_wide: List[pd.DataFrame] = []
    per_session_stats: List[Dict] = []

    for session_key in args.sessions:
        session_yaml = str(_PROJECT_ROOT / SESSIONS[session_key])
        session_subdir = SESSION_OUTPUT_DIRS[session_key]

        if not Path(session_yaml).exists():
            print(f"[RF-split] {session_key}: YAML not found, skipping")
            continue

        print(f"[RF-split] {session_key}: loading session...")
        merged_df, frame_feature_cols, base_cols, session_id = _prepare_session_for_fig3(
            session_yaml, config,
            canonical_alignment_path=args.canonical_alignment,
        )

        if len(merged_df) == 0:
            print(f"[RF-split] {session_key}: no merged data, skipping")
            continue

        # Split at median time
        t_bins = merged_df["t_bin_s"].unique()
        t_median = float(np.median(t_bins))
        half1_df = merged_df[merged_df["t_bin_s"] <= t_median].copy()
        half2_df = merged_df[merged_df["t_bin_s"] > t_median].copy()

        print(f"[RF-split] {session_key}: half1={len(half1_df)} rows, half2={len(half2_df)} rows, "
              f"t_split={t_median:.1f}s")

        # Process each half independently
        half_results = {}
        for half_name, half_df in [("half1", half1_df), ("half2", half2_df)]:
            all_unit_ids = half_df["unit_id"].unique()
            unit_data_dict = {
                uid: half_df[half_df["unit_id"] == uid].copy()
                for uid in all_unit_ids
            }
            unit_bank_map: Dict[int, str] = {}
            if "unit_bank" in half_df.columns:
                unit_bank_map = half_df.groupby("unit_id")["unit_bank"].first().to_dict()

            rng_master = np.random.default_rng(42)
            unit_seeds = {uid: int(rng_master.integers(0, 2**31)) for uid in all_unit_ids}

            results = Parallel(n_jobs=config.n_jobs, verbose=10 if config.debug else 0)(
                delayed(_process_single_unit_fig3)(
                    uid,
                    unit_data_dict[uid],
                    frame_feature_cols,
                    base_cols,
                    cv_cfg,
                    config,
                    session_id,
                    unit_seeds[uid],
                    unit_bank_map.get(uid),
                )
                for uid in all_unit_ids
            )
            valid = [r for r in results if r is not None]
            if valid:
                hdf = pd.DataFrame(valid)
                hdf = compute_rf_index(hdf, eps=args.eps)
                half_results[half_name] = hdf

        if "half1" not in half_results or "half2" not in half_results:
            print(f"[RF-split] {session_key}: one or both halves produced no results, skipping")
            continue

        # Wide-format join
        h1 = half_results["half1"][["unit_id", "session_id", "rf_index"]].rename(
            columns={"rf_index": "rf_index_half1"},
        )
        h2 = half_results["half2"][["unit_id", "rf_index"]].rename(
            columns={"rf_index": "rf_index_half2"},
        )
        wide = h1.merge(h2, on="unit_id", how="inner")

        # Add bank from half1 if available
        if "unit_bank" in half_results["half1"].columns:
            bank_map = half_results["half1"].set_index("unit_id")["unit_bank"].to_dict()
            wide["bank"] = wide["unit_id"].map(bank_map)

        n_paired = len(wide)
        n_dropped = len(half_results["half1"]) + len(half_results["half2"]) - 2 * n_paired
        print(f"[RF-split] {session_key}: {n_paired} paired units, {n_dropped} dropped")

        # Spearman correlation
        x = wide["rf_index_half1"].to_numpy(dtype=float)
        y = wide["rf_index_half2"].to_numpy(dtype=float)
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x_v, y_v = x[valid_mask], y[valid_mask]

        if len(x_v) >= 5:
            rho, p_val = spearmanr(x_v, y_v)
            print(f"[RF-split] {session_key}: Spearman rho={rho:.3f}, p={p_val:.1e}, "
                  f"n_finite={len(x_v)}")
        else:
            rho, p_val = float("nan"), float("nan")
            print(f"[RF-split] {session_key}: too few finite pairs ({len(x_v)}), rho=NaN")

        per_session_stats.append({
            "session_id": session_key,
            "n_paired": n_paired,
            "n_finite": int(valid_mask.sum()),
            "n_dropped": n_dropped,
            "spearman_rho": rho,
            "spearman_p": p_val,
        })

        # Save per-session
        session_out = output_root / session_subdir / "encoding"
        session_out.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(session_out / "fig3_rf_split_half.parquet", index=False)

        all_wide.append(wide)

    if not all_wide:
        print("[RF-split] No sessions produced results.")
        return

    # Aggregate
    summary_df = pd.concat(all_wide, ignore_index=True)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_dir / "fig3_rf_split_half.parquet", index=False)

    # Save stats
    with open(summary_dir / "fig3_rf_split_half_stats.json", "w") as f:
        json.dump(per_session_stats, f, indent=2)

    print(f"[RF-split] Summary: {len(summary_df)} paired units, "
          f"{len(all_wide)} sessions")

    # Scatter figure (if viz available)
    try:
        from freemat.single_fm.viz.fig3_rf_panels import plot_rf_split_half_scatter
        fig = plot_rf_split_half_scatter(summary_df)
        figures_dir = summary_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / "fig3_rf_split_half_scatter.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures_dir / "fig3_rf_split_half_scatter.svg", bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        pass
    print(f"[RF-split] Figures saved to {figures_dir}")


def main() -> None:
    """Load existing fig3 parquets, compute RF-index, save results and figures."""
    args = parse_args()
    prod_dir = Path(args.prod_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "prod_dir": str(prod_dir),
        "sessions": args.sessions,
        "eps": args.eps,
        "height_stratify": args.height_stratify,
        "split_half": args.split_half,
    }
    with open(output_root / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    all_dfs: List[pd.DataFrame] = []

    for session_key in args.sessions:
        session_subdir = SESSION_OUTPUT_DIRS[session_key]
        # Auto-detect prod-dir layout: direct or fig3_bin05/ subdirectory
        fig3_path = prod_dir / session_subdir / "encoding" / "fig3_reference_frames.parquet"
        layout_used = "direct"
        if not fig3_path.exists():
            fig3_path_alt = prod_dir / "fig3_bin05" / session_subdir / "encoding" / "fig3_reference_frames.parquet"
            if fig3_path_alt.exists():
                fig3_path = fig3_path_alt
                layout_used = "fig3_bin05"
            else:
                print(f"[RF-index] {session_key}: fig3 parquet not found at {fig3_path} "
                      f"or {fig3_path_alt}, skipping")
                continue
        if args.debug:
            print(f"[RF-index] {session_key}: loaded from {layout_used} layout: {fig3_path}")

        df = pd.read_parquet(fig3_path)
        if "delta_r2_allo" not in df.columns or "delta_r2_body" not in df.columns:
            print(f"[RF-index] {session_key}: missing delta_r2 columns, skipping")
            continue

        # Rename unit_bank → bank for downstream consumers (Fix 4)
        if "unit_bank" in df.columns and "bank" not in df.columns:
            df = df.rename(columns={"unit_bank": "bank"})

        df_rf = compute_rf_index(df, eps=args.eps)

        if "session_id" not in df_rf.columns:
            df_rf["session_id"] = session_key

        n_units = len(df_rf)
        if n_units == 0:
            print(f"[RF-index] {session_key}: 0 units after rf_index computation, skipping")
            continue
        median_rf = float(df_rf["rf_index"].median())
        n_allo = int((df_rf["rf_index"] > 0).sum())
        print(f"[RF-index] {session_key}: {n_units} units, median rf_index={median_rf:.3f}, "
              f"allo-dominant={n_allo}/{n_units} ({100*n_allo/n_units:.0f}%)")

        # Save per-session
        session_out = output_root / session_subdir / "encoding"
        session_out.mkdir(parents=True, exist_ok=True)
        df_rf.to_parquet(session_out / "fig3_rf_index.parquet", index=False)

        all_dfs.append(df_rf)

    if not all_dfs:
        print("[RF-index] No sessions produced results.")
        return

    # Aggregate
    summary_df = pd.concat(all_dfs, ignore_index=True)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_dir / "fig3_rf_index_all.parquet", index=False)
    print(f"[RF-index] Summary: {len(summary_df)} units across {len(all_dfs)} sessions")

    # Figures (if viz modules available)
    if save_rf_index_panels is not None:
        figures_dir = summary_dir / "figures"
        save_rf_index_panels(summary_df, str(figures_dir), split_by_bank="bank" in summary_df.columns)
        print(f"[RF-index] Figures saved to {figures_dir}")
    else:
        print("[RF-index] Skipping figures (viz modules not available)")

    if args.height_stratify:
        print("\n[RF-index] Running height-stratified analysis...")
        _run_height_stratified(args, output_root)

    if args.split_half:
        print("\n[RF-index] Running split-half stability control...")
        _run_split_half(args, output_root)


if __name__ == "__main__":
    main()
