#!/usr/bin/env python
"""Run TGM + panel D candidate analyses for Fig 4.

Lightweight runner for:
  - Temporal Generalization Matrix (TGM) — Fig 4B
  - Bout-duration prediction — panel D candidate
  - Population participation — panel D candidate (from existing tuning data)

Usage:
    # All sessions
    python scripts/single_fm/run_fig4_tgm.py --sessions all

    # Single session pilot
    python scripts/single_fm/run_fig4_tgm.py --sessions hedy_0526

    # Skip bout-duration (fastest)
    python scripts/single_fm/run_fig4_tgm.py --sessions all --skip-duration
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

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

# Default tuning data location (from production Fig 4 runs)
DEFAULT_TUNING_DIR = "results/single_fm/fig4_pilot"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run TGM + panel D analyses for Fig 4"
    )
    parser.add_argument(
        "--sessions", nargs="+", default=["all"],
        help="Session keys (e.g. hedy_0526 M1_0525) or 'all'",
    )
    parser.add_argument(
        "--output-dir", default="results/single_fm/fig4_tgm",
        help="Output directory for results",
    )
    parser.add_argument(
        "--tuning-dir", default=DEFAULT_TUNING_DIR,
        help="Directory with existing fig4_tuning.parquet files",
    )
    parser.add_argument(
        "--n-pca", type=int, default=50,
        help="PCA components for TGM (0 to skip PCA)",
    )
    parser.add_argument(
        "--n-perm-duration", type=int, default=1000,
        help="Permutations for bout-duration null",
    )
    parser.add_argument(
        "--skip-duration", action="store_true",
        help="Skip bout-duration prediction",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if "all" in args.sessions:
        session_keys = list(SESSIONS.keys())
    else:
        session_keys = args.sessions

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    from freemat.single_fm.encoding.run_fig4 import (
        Fig4Config,
        prepare_fig4_data,
        run_bout_duration_prediction,
        run_population_participation,
        run_tgm,
    )

    config = Fig4Config()
    summary_rows = []

    for sess_key in session_keys:
        if sess_key not in SESSIONS:
            print(f"Unknown session: {sess_key}, skipping")
            continue

        session_yaml = SESSIONS[sess_key]
        out_name = SESSION_OUTPUT_DIRS[sess_key]
        out_dir = out_base / out_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Session: {sess_key}")
        print(f"{'='*60}")

        t0 = time.time()

        # ── Load data ──
        print("  Loading data...")
        data = prepare_fig4_data(session_yaml, config, debug=args.debug)
        n_units = data["X"].shape[1]
        n_bins = data["X"].shape[0]
        print(f"  {n_units} units, {n_bins} bins")

        # ── TGM ──
        print("  Computing TGM...")
        t_tgm = time.time()
        tgm_out = run_tgm(
            data["X"], data["y"], config,
            pre_bins=8, post_bins=9, n_pca=args.n_pca,
        )
        tgm_result = tgm_out["result"]
        dt_tgm = time.time() - t_tgm
        print(f"  TGM done in {dt_tgm:.1f}s: {tgm_result.n_transitions} transitions, "
              f"{tgm_result.n_classes} classes, "
              f"diag mean={np.mean(tgm_result.diagonal):.3f}, "
              f"chance={tgm_result.chance_level:.3f}")

        np.savez(
            out_dir / "fig4_tgm.npz",
            matrix=tgm_result.matrix,
            tau_sec=tgm_result.tau_sec,
            diagonal=tgm_result.diagonal,
            chance_level=np.array(tgm_result.chance_level),
            n_transitions=np.array(tgm_result.n_transitions),
            n_classes=np.array(tgm_result.n_classes),
            n_folds=np.array(tgm_result.n_folds),
        )

        # ── Bout-duration prediction ──
        dur_r, dur_p, dur_n = 0.0, 1.0, 0
        if not args.skip_duration:
            print("  Computing bout-duration prediction...")
            t_dur = time.time()
            dur_out = run_bout_duration_prediction(
                data["X"], data["y"], data["valid_mask"],
                data["bin_times"], config, session_yaml,
                n_perm=args.n_perm_duration,
            )
            dur_r = dur_out["r"]
            dur_p = dur_out["p"]
            dur_n = dur_out["n_bouts"]
            dt_dur = time.time() - t_dur
            print(f"  Duration done in {dt_dur:.1f}s: r={dur_r:.3f}, p={dur_p:.3f}, "
                  f"n_bouts={dur_n}")

            np.savez(
                out_dir / "fig4_bout_duration.npz",
                r=np.array(dur_r),
                p=np.array(dur_p),
                n_bouts=np.array(dur_n),
                durations=dur_out.get("durations", np.array([])),
                log_durations=dur_out.get("log_durations", np.array([])),
                predictions=dur_out.get("predictions", np.array([])),
                r_null=dur_out.get("r_null", np.array([])),
            )

        # ── Population participation ──
        tuning_path = Path(args.tuning_dir) / out_name / "fig4_tuning.parquet"
        if tuning_path.exists():
            print("  Computing population participation...")
            # Get motif IDs from cluster order if available
            cluster_path = Path(args.tuning_dir) / out_name / "fig4_cluster_order.json"
            if cluster_path.exists():
                with open(cluster_path) as f:
                    motif_order = json.load(f)["motif_order"]
            else:
                valid_labels = data["y"][data["valid_mask"]]
                motif_order = sorted(set(valid_labels) - {config.missing_value, config.other_id})

            part_out = run_population_participation(
                str(tuning_path), motif_order,
            )
            print(f"  Participation: {part_out['n_units']} units, "
                  f"{len(part_out['motif_ids'])} motifs, "
                  f"mean frac={np.mean(part_out['participation_frac']):.3f}")

            np.savez(
                out_dir / "fig4_participation.npz",
                binary_matrix=part_out["binary_matrix"],
                participation_frac=part_out["participation_frac"],
                bandwidth=part_out["bandwidth"],
                motif_ids=np.array(part_out["motif_ids"]),
            )
        else:
            print(f"  Skipping participation (no tuning data at {tuning_path})")

        dt_total = time.time() - t0
        print(f"  Total: {dt_total:.1f}s")

        summary_rows.append({
            "session": sess_key,
            "n_units": n_units,
            "n_transitions": tgm_result.n_transitions,
            "n_classes": tgm_result.n_classes,
            "tgm_diag_mean": float(np.mean(tgm_result.diagonal)),
            "tgm_chance": float(tgm_result.chance_level),
            "tgm_diag_above_chance": float(np.mean(tgm_result.diagonal) - tgm_result.chance_level),
            "dur_r": dur_r,
            "dur_p": dur_p,
            "dur_n_bouts": dur_n,
            "time_s": dt_total,
        })

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Session':<12} {'Units':>6} {'Trans':>6} {'TGM diag':>10} {'Chance':>8} "
          f"{'Delta':>6} {'Dur r':>7} {'Dur p':>7}")
    for row in summary_rows:
        print(f"{row['session']:<12} {row['n_units']:>6} {row['n_transitions']:>6} "
              f"{row['tgm_diag_mean']:>10.3f} {row['tgm_chance']:>8.3f} "
              f"{row['tgm_diag_above_chance']:>6.3f} "
              f"{row['dur_r']:>7.3f} {row['dur_p']:>7.3f}")

    with open(out_base / "tgm_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nResults saved to {out_base}")


if __name__ == "__main__":
    main()
