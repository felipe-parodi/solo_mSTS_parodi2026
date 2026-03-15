#!/usr/bin/env python
"""Batch runner for Figure 2D spatial decoders (height + 9-zone).

Runs within-session raw-vs-residual population decoders and aggregates
session-level outputs for publication plotting.

Supports pooled height cuts: a two-pass approach where the first pass loads
post-filtered COM bins from all sessions to compute consistent height
cutpoints, then the second pass runs decoding with those fixed cuts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freemat.single_fm.data import SingleFMSessionLoader  # noqa: E402
from freemat.single_fm.encoding.fig2d_spatial_decoder import (  # noqa: E402
    Fig2DDecoderConfig,
    run_fig2d_spatial_decoding,
)
from freemat.single_fm.viz.fig2d_panels import save_fig2d_decoder_pairs  # noqa: E402


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


def _collect_post_filtered_com_z(
    session_keys: List[str],
    bin_s: float,
    lag_taps: int,
    *,
    canonical_alignment_path: Optional[str] = None,
    debug: bool = False,
) -> np.ndarray:
    """Load post-filtered/post-lag COM z-values from all sessions.

    Uses the same pipeline as decoding (lag trim + NaN filter) to ensure
    height cuts are computed on the exact bins used for classification.

    Args:
        session_keys: Session identifiers to load.
        bin_s: Bin size in seconds.
        lag_taps: Total lag taps (must be odd).
        canonical_alignment_path: Optional canonical alignment JSON.
        debug: Print diagnostics.

    Returns:
        Concatenated finite COM z-values from all sessions.
    """
    from freemat.single_fm.encoding.temporal import build_lagged_features
    from freemat.single_fm.encoding.visual_features import (
        BODY_KIN_COLS,
        HEAD_KIN_COLS,
        VIS_COLS,
    )

    n_lags = (lag_taps - 1) // 2
    all_com_z: List[np.ndarray] = []

    for session_key in session_keys:
        session_yaml = SESSIONS[session_key]
        loader = SingleFMSessionLoader(
            session_yaml,
            canonical_alignment_path=canonical_alignment_path,
            debug=debug,
        )
        merged_df = loader.get_aligned_data(
            bin_s=bin_s,
            floor_filter=False,
            floor_filter_before_features=False,
            binning_method="mean",
        )
        if merged_df.empty:
            print(f"  [pooled] {session_key}: empty data, skipping")
            continue

        # Get unique time bins with COM
        feature_bins = merged_df.drop_duplicates(subset=["t_bin_s"]).set_index("t_bin_s").sort_index()
        com_z = feature_bins["com_z"].to_numpy(dtype=float)

        # Apply same lag trim as decoding
        nuisance_cols = VIS_COLS + BODY_KIN_COLS + HEAD_KIN_COLS
        missing_cols = [c for c in nuisance_cols if c not in feature_bins.columns]
        if missing_cols:
            print(f"  [pooled] {session_key}: missing nuisance cols {missing_cols}, skipping")
            continue

        x_nuis = feature_bins[nuisance_cols].to_numpy(dtype=float)
        if n_lags > 0:
            x_nuis = build_lagged_features(x_nuis, n_lags)
            # Trim to match: lagged features may pad, so trim both ends
            com_z = com_z[n_lags:-n_lags]
            x_nuis = x_nuis[n_lags:-n_lags]

        # Filter same as decoding: finite COM and nuisance
        valid = np.isfinite(com_z) & np.all(np.isfinite(x_nuis), axis=1)
        com_z_valid = com_z[valid]
        all_com_z.append(com_z_valid)
        if debug and len(com_z_valid) > 0:
            print(f"  [pooled] {session_key}: {valid.sum()} valid bins, "
                  f"com_z range [{com_z_valid.min():.3f}, {com_z_valid.max():.3f}]")
        elif debug:
            print(f"  [pooled] {session_key}: 0 valid bins")

    if not all_com_z:
        raise ValueError("No valid COM data from any session for pooled height cuts.")

    return np.concatenate(all_com_z)


def _save_zone_diagnostic(
    session_key: str,
    session_yaml: str,
    output_dir: Path,
    config: Fig2DDecoderConfig,
    *,
    canonical_alignment_path: Optional[str] = None,
) -> None:
    """Save zone/height diagnostic plots for visual sanity check.

    Generates:
    - Top-down view: COM scatter colored by zone_id with sector boundaries
    - Side view: COM scatter colored by height_id with cut lines

    Args:
        session_key: Session identifier.
        session_yaml: Path to session YAML.
        output_dir: Directory for diagnostic outputs.
        config: Decoder configuration (contains height cuts).
        canonical_alignment_path: Optional canonical alignment JSON.
    """
    from freemat.single_fm.encoding.fig2d_spatial_decoder import (
        _build_height_and_zone_labels,
    )

    loader = SingleFMSessionLoader(
        session_yaml,
        canonical_alignment_path=canonical_alignment_path,
        debug=False,
    )
    merged_df = loader.get_aligned_data(
        bin_s=config.bin_s,
        floor_filter=False,
        floor_filter_before_features=False,
        binning_method="mean",
    )
    if merged_df.empty:
        return

    feature_bins = merged_df.drop_duplicates(subset=["t_bin_s"]).set_index("t_bin_s").sort_index()
    com_x = feature_bins["com_x"].to_numpy(dtype=float)
    com_y = feature_bins["com_y"].to_numpy(dtype=float)
    com_z = feature_bins["com_z"].to_numpy(dtype=float)

    # Use vertical_axis → horiz_idxs mapping matching feature_builder.py
    vert_axis = loader._get_vertical_axis()
    horiz_idxs = (0, 2) if vert_axis == 1 else (1, 2) if vert_axis == 0 else (0, 1)
    arena_center_xy: Optional[Tuple[float, float]] = None
    if hasattr(loader, "arena") and loader.arena is not None:
        center = loader.arena.center
        arena_center_xy = (float(center[horiz_idxs[0]]), float(center[horiz_idxs[1]]))

    height_id, zone_sector, zone_id, cuts = _build_height_and_zone_labels(
        com_x, com_y, com_z, config, arena_center_xy=arena_center_xy,
    )
    valid = zone_id >= 0

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: top-down COM scatter colored by zone_id
    ax = axes[0]
    cmap_zone = plt.cm.tab10
    sc = ax.scatter(
        com_x[valid], com_y[valid], c=zone_id[valid], cmap=cmap_zone,
        s=3, alpha=0.3, vmin=0, vmax=8, rasterized=True,
    )
    if arena_center_xy is not None:
        ax.axvline(arena_center_xy[0], color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(arena_center_xy[1], color="gray", ls="--", lw=0.8, alpha=0.5)
        cx, cy = arena_center_xy
        # Draw sector boundaries
        for s in range(config.n_zone_sectors):
            angle = s * 2.0 * np.pi / config.n_zone_sectors
            r = 1.5  # meters, visual extent
            ax.plot(
                [cx, cx + r * np.cos(angle)],
                [cy, cy + r * np.sin(angle)],
                color="black", ls="-", lw=1.0, alpha=0.6,
            )
    ax.set_xlabel("com_x (m)")
    ax.set_ylabel("com_y (m)")
    ax.set_title(f"{session_key}: top-down (9 zones)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="zone_id", ticks=range(9))

    # Right: side view COM scatter colored by height_id
    ax = axes[1]
    cmap_height = plt.cm.viridis
    sc2 = ax.scatter(
        com_x[valid], com_z[valid], c=height_id[valid], cmap=cmap_height,
        s=3, alpha=0.3, vmin=0, vmax=2, rasterized=True,
    )
    ax.axhline(cuts[0], color="red", ls="--", lw=1.2, label=f"low={cuts[0]:.3f}")
    ax.axhline(cuts[1], color="blue", ls="--", lw=1.2, label=f"high={cuts[1]:.3f}")
    ax.set_xlabel("com_x (m)")
    ax.set_ylabel("com_z (m)")
    ax.set_title(f"{session_key}: side view (3 heights)")
    ax.legend(fontsize=9, frameon=False)
    fig.colorbar(sc2, ax=ax, label="height_id", ticks=[0, 1, 2])

    fig.suptitle(f"Zone diagnostic: {session_key}", fontweight="bold", y=1.02)
    fig.tight_layout()

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(diag_dir / f"zone_diagnostic_{session_key}.png", dpi=200, bbox_inches="tight")
    fig.savefig(diag_dir / f"zone_diagnostic_{session_key}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [diag] Zone diagnostic saved: {diag_dir / f'zone_diagnostic_{session_key}.png'}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run Fig 2D spatial decoders across sessions.")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory.")
    parser.add_argument(
        "--fig2b-root",
        type=str,
        default=None,
        help="Base directory containing fig2b_unique_variance.parquet per session (defaults to output-dir).",
    )
    parser.add_argument("--sessions", nargs="+", choices=list(SESSIONS.keys()), default=list(SESSIONS.keys()))
    parser.add_argument("--bin-s", type=float, default=0.5)
    parser.add_argument("--cv-folds", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--perm-block-s", type=float, default=10.0)
    parser.add_argument("--lag-taps", type=int, default=5)
    parser.add_argument(
        "--height-strategy",
        choices=["quantile", "fixed", "pooled"],
        default="pooled",
        help="'pooled' (default): compute 33/66 quantiles from all sessions. "
             "'fixed': use --height-cuts. 'quantile': per-session quantiles.",
    )
    parser.add_argument(
        "--height-cuts",
        nargs=2,
        type=float,
        default=[0.30, 0.60],
        help="Only used when --height-strategy fixed.",
    )
    parser.add_argument("--no-balance-train", action="store_true")
    parser.add_argument(
        "--fold-guard-level",
        choices=["strict", "relaxed"],
        default="strict",
        help="Fold guard threshold level (strict: train>=20, test>=10; relaxed: train>=10, test>=5).",
    )
    parser.add_argument(
        "--canonical-alignment",
        type=str,
        default=None,
        help="Optional canonical alignment JSON.",
    )
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip zone diagnostic plots.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all configured sessions and aggregate Fig2D decoder summaries."""
    args = parse_args()
    output_root = Path(args.output_dir)
    fig2b_root = Path(args.fig2b_root) if args.fig2b_root else output_root

    # --- Pooled height cut computation (pass 1) ---
    height_strategy = args.height_strategy
    height_cuts: Tuple[float, float] = (float(args.height_cuts[0]), float(args.height_cuts[1]))

    if height_strategy == "pooled":
        print("[Fig2D] Computing pooled height cuts from all sessions...")
        pooled_com_z = _collect_post_filtered_com_z(
            args.sessions,
            bin_s=args.bin_s,
            lag_taps=args.lag_taps,
            canonical_alignment_path=args.canonical_alignment,
            debug=args.debug,
        )
        low, high = np.nanquantile(pooled_com_z, [1.0 / 3.0, 2.0 / 3.0])
        height_cuts = (float(low), float(high))
        height_strategy = "fixed"  # Use fixed cuts from pooled computation
        print(f"[Fig2D] Pooled height cuts: low={height_cuts[0]:.4f}, high={height_cuts[1]:.4f} "
              f"(from {len(pooled_com_z)} bins across {len(args.sessions)} sessions)")

    config = Fig2DDecoderConfig(
        bin_s=args.bin_s,
        cv_folds=tuple(args.cv_folds),
        n_permutations=args.n_perm,
        perm_block_s=args.perm_block_s,
        lag_taps=args.lag_taps,
        height_strategy=height_strategy,
        fixed_height_cuts=height_cuts,
        balance_train_bins=not args.no_balance_train,
        fold_guard_level=args.fold_guard_level,
        debug=args.debug,
    )

    # --- Save run config ---
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "sessions": args.sessions,
        "bin_s": args.bin_s,
        "cv_folds": args.cv_folds,
        "n_permutations": args.n_perm,
        "perm_block_s": args.perm_block_s,
        "lag_taps": args.lag_taps,
        "height_strategy": args.height_strategy,
        "height_cuts_used": list(height_cuts),
        "fold_guard_level": args.fold_guard_level,
        "balance_train_bins": not args.no_balance_train,
        "canonical_alignment": args.canonical_alignment,
    }
    with open(output_root / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # --- Pass 2: run decoding per session ---
    session_tables: List[pd.DataFrame] = []
    for session_key in args.sessions:
        session_yaml = SESSIONS[session_key]
        session_dir = output_root / SESSION_OUTPUT_DIRS[session_key]
        fig2b_path = fig2b_root / SESSION_OUTPUT_DIRS[session_key] / "encoding" / "fig2b_unique_variance.parquet"
        encoding_path = str(fig2b_path) if fig2b_path.exists() else None

        print(f"[Fig2D] Session {session_key}")
        print(f"         YAML: {session_yaml}")
        print(f"         Output: {session_dir}")
        if encoding_path is None:
            print("         Fig2B alpha map: not found (fallback alpha in use)")
        else:
            print(f"         Fig2B alpha map: {encoding_path}")

        # Zone diagnostic (before decoding, for visual sanity check)
        if not args.skip_diagnostics:
            _save_zone_diagnostic(
                session_key,
                session_yaml,
                output_root,
                config,
                canonical_alignment_path=args.canonical_alignment,
            )

        df = run_fig2d_spatial_decoding(
            session_yaml=session_yaml,
            output_dir=str(session_dir),
            config=config,
            encoding_results_path=encoding_path,
            canonical_alignment_path=args.canonical_alignment,
        )

        # Log class counts from metadata JSON
        meta_path = session_dir / "encoding" / "fig2d_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"         Height counts: {meta.get('height_class_counts', {})}")
            print(f"         Zone counts: {meta.get('zone_class_counts', {})}")
        for _, row in df.iterrows():
            n_valid = row.get("n_valid_folds", "?")
            n_null = row.get("n_null_valid", "?")
            print(f"         {row['target']} / {row['model']} / k={row['cv_folds']}: "
                  f"acc={row['acc']:.3f}, acc_concat={row.get('acc_concat', '?')}, "
                  f"valid_folds={n_valid}, null_valid={n_null}")

        session_tables.append(df)

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.concat(session_tables, ignore_index=True)
    summary_path = summary_dir / "fig2d_decoder_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)
    print(f"[Fig2D] Summary saved: {summary_path}")

    figures_dir = summary_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for cv_fold in config.cv_folds:
        fig_name = figures_dir / f"fig2D_spatial_decoders_k{cv_fold}.png"
        fig = save_fig2d_decoder_pairs(summary_df, str(fig_name), cv_folds=cv_fold)
        fig.clf()
        print(f"[Fig2D] Figure saved: {fig_name}")


if __name__ == "__main__":
    main()
