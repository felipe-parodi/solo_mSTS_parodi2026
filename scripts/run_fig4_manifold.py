#!/usr/bin/env python
"""Run neural manifold analysis for Figure 4 sessions.

Computes PCA and UMAP embeddings of population firing rates
colored by behavioral motif, with silhouette cluster separation
and block-shuffle null distribution.

Usage:
    # Quick pilot on one session
    python scripts/single_fm/run_fig4_manifold.py \
        --output-dir results/single_fm/fig4_manifold --sessions M2_0526 --quick

    # PCA-only (no umap-learn required)
    python scripts/single_fm/run_fig4_manifold.py \
        --output-dir results/single_fm/fig4_manifold --sessions M2_0526 --no-umap

    # Full run on all sessions
    python scripts/single_fm/run_fig4_manifold.py \
        --output-dir results/single_fm/fig4_manifold
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Session mapping — copied from run_fig4_all_sessions.py (cross-script
# imports are fragile for batch scripts).
SESSIONS: Dict[str, str] = {
    "M1_0525": "configs/single_fm/session_M1_0525.yaml",
    "M2_0526": "configs/single_fm/session_M2_0526.yaml",
    "M2_0605": "configs/single_fm/session_M2_0605.yaml",
    "M1_0609": "configs/single_fm/session_M1_0609.yaml",
    "M1_0613": "configs/single_fm/session_M1_0613.yaml",
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
    p = argparse.ArgumentParser(description="Fig 4 neural manifold analysis")
    p.add_argument("--output-dir", required=True, help="Base output directory")
    p.add_argument("--sessions", nargs="+", default=list(SESSIONS.keys()),
                   choices=list(SESSIONS.keys()), help="Sessions to run")
    p.add_argument("--bin-s", type=float, default=0.5, help="Bin width (s)")
    p.add_argument("--top-n", type=int, default=10, help="Top-N motifs")
    p.add_argument("--no-umap", action="store_true", help="Skip UMAP (PCA only)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer perms/subsamples")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def run_session_manifold(
    session_key: str,
    config_path: str,
    output_dir: Path,
    bin_s: float,
    top_n: int,
    use_umap: bool,
    quick: bool,
) -> None:
    """Run manifold analysis for a single session.

    Args:
        session_key: Session identifier (e.g. 'M2_0526').
        config_path: Path to session YAML config.
        output_dir: Output directory for this session.
        bin_s: Bin width in seconds.
        top_n: Top-N motifs.
        use_umap: Whether to compute UMAP.
        quick: Quick mode with reduced permutations.
    """
    from freemat.single_fm.encoding.fig4_manifold import (
        ManifoldConfig,
        compute_manifold_analysis,
    )
    from freemat.single_fm.encoding.run_fig4 import Fig4Config, prepare_fig4_data
    from freemat.single_fm.viz.fig4_manifold_panels import (
        plot_manifold_pca_2d,
        plot_manifold_pca_3d,
        plot_manifold_scree,
        plot_manifold_silhouette_per_class,
        plot_manifold_umap,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print(f"  Loading data from {config_path}...")
    fig4_config = Fig4Config(bin_s=bin_s, top_n_motifs=top_n, n_permutations=0)
    data = prepare_fig4_data(config_path, fig4_config)
    X, y = data["X"], data["y"]
    unit_banks = data["unit_banks"]
    valid_mask = data["valid_mask"]
    session_id = data["session_id"]

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    print(f"  Data: {X_valid.shape[0]} bins x {X_valid.shape[1]} units, "
          f"{len(set(y_valid))} motif classes")

    # Configure manifold
    if quick:
        config = ManifoldConfig(
            use_umap=use_umap,
            bin_s=bin_s,
            n_perms=10,
            n_subsample=5,
            max_bins_silhouette=2000,
        )
    else:
        config = ManifoldConfig(use_umap=use_umap, bin_s=bin_s)

    # Run analysis
    print(f"  Computing manifold (UMAP={'yes' if use_umap else 'no'}, "
          f"perms={config.n_perms})...")
    t0 = time.time()
    result = compute_manifold_analysis(X_valid, y_valid, unit_banks, session_id, config)
    elapsed = time.time() - t0
    print(f"  Manifold analysis completed in {elapsed:.1f}s")

    # Save embeddings
    npz_path = output_dir / "fig4_manifold.npz"
    np.savez_compressed(
        npz_path,
        pca_embedding=result.pca_embedding,
        pca_var_explained=result.pca_var_explained,
        umap_embedding=result.umap_embedding,
        motif_labels=result.motif_labels,
        unit_banks=result.unit_banks,
        has_umap=np.array([result.has_umap]),
    )
    print(f"  Saved embeddings to {npz_path}")

    # Save summary JSON with all silhouette embeddings
    sep = result.separation
    summary: dict = {
        "session_id": session_id,
        "n_bins": int(X_valid.shape[0]),
        "n_units": int(X_valid.shape[1]),
        "n_classes": len(set(y_valid)),
        "quick_mode": quick,
        "primary_embedding": next(
            (k for k in ("umap", "pca_10d", "pca_3d")
             if result.separations and k in result.separations),
            "unknown",
        ),
        "primary_silhouette": sep.silhouette_mean,
        "primary_p_value": sep.p_value,
    }

    # Per-embedding silhouette results
    if result.separations:
        summary["silhouettes"] = {}
        for emb_name, sep_result in result.separations.items():
            summary["silhouettes"][emb_name] = {
                "mean": sep_result.silhouette_mean,
                "ci": list(sep_result.silhouette_ci),
                "null_mean": sep_result.null_mean,
                "null_std": sep_result.null_std,
                "p_value": sep_result.p_value,
                "per_class": {str(k): v for k, v in sep_result.silhouette_per_class.items()},
            }
            print(f"  Silhouette [{emb_name}]: {sep_result.silhouette_mean:.3f} "
                  f"(null: {sep_result.null_mean:.3f} +/- {sep_result.null_std:.3f}, "
                  f"p={sep_result.p_value:.3f})")

    json_path = output_dir / "fig4_manifold_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {json_path}")

    # Generate plots
    plots = [
        ("fig4_manifold_pca_2d", plot_manifold_pca_2d(result)),
        ("fig4_manifold_pca_3d", plot_manifold_pca_3d(result)),
        ("fig4_manifold_scree", plot_manifold_scree(result)),
        ("fig4_manifold_silhouette", plot_manifold_silhouette_per_class(result)),
    ]
    if result.has_umap:
        plots.append(("fig4_manifold_umap", plot_manifold_umap(result)))

    for name, fig in plots:
        for ext in ("png", "svg"):
            path = output_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}")


def main() -> None:
    """Run manifold analysis on selected sessions."""
    args = parse_args()

    out_root = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("Fig 4 Neural Manifold Analysis")
    print(f"Sessions: {args.sessions}")
    print(f"UMAP: {'no' if args.no_umap else 'yes'}")
    print(f"Quick: {'yes' if args.quick else 'no'}")
    print(f"Output: {out_root}")
    print(f"{'='*60}\n")

    for session_key in args.sessions:
        config_path = SESSIONS[session_key]
        session_out = out_root / SESSION_OUTPUT_DIRS[session_key]
        print(f"\n--- {session_key} ---")
        t0 = time.time()
        run_session_manifold(
            session_key=session_key,
            config_path=config_path,
            output_dir=session_out,
            bin_s=args.bin_s,
            top_n=args.top_n,
            use_umap=not args.no_umap,
            quick=args.quick,
        )
        print(f"  Total time: {time.time() - t0:.1f}s")

    print(f"\nAll sessions complete. Output: {out_root}")


if __name__ == "__main__":
    main()
