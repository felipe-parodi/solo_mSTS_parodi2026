#!/usr/bin/env python
"""Figure 5 follow-up: additive vs interaction encoding analysis.

Runs a per-unit encoding analysis testing whether explicit motif x height-tier
interaction terms improve held-out prediction beyond additive tuning for
motif + tier + continuous controls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freemat.single_fm.data.session_loader import SingleFMSessionLoader  # noqa: E402
from freemat.single_fm.encoding.fig5_interaction import (  # noqa: E402
    InteractionConfig,
    align_motifs_to_bins,
    build_interaction_design,
    finalize_results,
    fit_unit_interaction_model,
    summarize_interaction_results,
)
from freemat_data import load_motifs_from_session  # noqa: E402


SESSION_CONFIGS: Dict[str, str] = {
    "M2_0526": "configs/session_M2_0526.yaml",
    "M2_0605": "configs/session_M2_0605.yaml",
    "M1_0525": "configs/session_M1_0525.yaml",
    "M1_0609": "configs/session_M1_0609.yaml",
    "M1_0613": "configs/session_M1_0613.yaml",
}

ALL_SESSIONS: List[str] = list(SESSION_CONFIGS.keys())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Figure 5 follow-up: additive vs interaction encoding",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=ALL_SESSIONS,
        choices=ALL_SESSIONS,
        help="Session keys to run (default: all 5 sessions).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fig5_interaction",
        help="Directory for per-session and aggregate outputs.",
    )
    parser.add_argument("--bin-s", type=float, default=0.5)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--n-perm-null", type=int, default=500)
    parser.add_argument("--min-bins-per-cell", type=int, default=20)
    parser.add_argument("--min-bins-per-unit", type=int, default=150)
    parser.add_argument("--lag-s", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--sua-only", action="store_true")
    parser.add_argument("--include-time-drift", action="store_true")
    parser.add_argument("--include-rssi", action="store_true")
    parser.add_argument("--include-antenna-distance", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_session_matrices(
    session_key: str,
    config: InteractionConfig,
) -> Dict[str, Any]:
    """Load aligned firing rates, features, and motif labels for one session."""
    yaml_path = str(REPO_ROOT / SESSION_CONFIGS[session_key])
    loader = SingleFMSessionLoader(yaml_path, debug=config.debug)

    merged_df = loader.get_aligned_data(
        bin_s=config.bin_s,
        floor_filter=False,
        floor_height_threshold=3.0,
        floor_filter_before_features=False,
        binning_method=config.binning_method,
    )
    if merged_df.empty:
        raise ValueError(f"No aligned data for session {session_key}.")

    if config.include_rssi:
        from freemat.single_fm.encoding.temporal import load_rssi_for_session

        unique_bins = np.sort(merged_df["t_bin_s"].unique())
        rssi_session = load_rssi_for_session(yaml_path, unique_bins)
        rssi_map = dict(zip(unique_bins, rssi_session))
        merged_df["rssi"] = merged_df["t_bin_s"].map(rssi_map)

    if config.include_antenna_distance:
        from freemat.single_fm.encoding.temporal import load_antenna_distance_for_session

        unique_bins = np.sort(merged_df["t_bin_s"].unique())
        d_antenna_session = load_antenna_distance_for_session(
            yaml_path,
            unique_bins,
            merged_df,
            pipeline_floor_level=loader.floor_level,
        )
        d_antenna_map = dict(zip(unique_bins, d_antenna_session))
        merged_df["d_antenna"] = merged_df["t_bin_s"].map(d_antenna_map)

    if config.sua_only and "unit_kind" in merged_df.columns:
        merged_df = merged_df[merged_df["unit_kind"] == "SUA"].copy()

    merged_df = merged_df.rename(columns={"fr": "firing_rate", "unit_index": "unit_id"})

    unit_ids = sorted(merged_df["unit_id"].unique())
    if len(unit_ids) == 0:
        raise ValueError(f"No units available for session {session_key}.")

    bin_times = np.array(sorted(merged_df["t_bin_s"].unique()), dtype=np.float64)
    spike_wide = (
        merged_df.pivot_table(index="t_bin_s", columns="unit_id", values="firing_rate", aggfunc="first")
        .reindex(index=bin_times, columns=unit_ids)
        .fillna(0.0)
    )

    feat_df = (
        merged_df.drop_duplicates(subset=["t_bin_s"])
        .set_index("t_bin_s")
        .reindex(bin_times)
    )

    motifs = load_motifs_from_session(loader._session)
    if motifs is None or motifs.t is None:
        raise ValueError(f"No motif labels found for session {session_key}.")
    y_motif = align_motifs_to_bins(motifs, bin_times, config.bin_s)

    design = build_interaction_design(
        feat_df=feat_df,
        y_motif=y_motif,
        bin_times=bin_times,
        config=config,
    )

    spike_wide = spike_wide.loc[design.bin_times]

    unit_meta = (
        merged_df.groupby("unit_id")[["unit_kind", "unit_bank"]]
        .first()
        if {"unit_kind", "unit_bank"}.issubset(merged_df.columns)
        else pd.DataFrame(index=unit_ids)
    )

    return {
        "spike_wide": spike_wide,
        "design": design,
        "unit_meta": unit_meta,
        "unit_ids": unit_ids,
        "session_id": session_key,
    }


def _fit_single_unit(
    unit_id: int,
    y: pd.Series,
    design,
    config: InteractionConfig,
    session_id: str,
    unit_meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Fit one unit with metadata lookup."""
    unit_kind = None
    unit_bank = None
    if unit_id in unit_meta.index:
        if "unit_kind" in unit_meta.columns:
            unit_kind = unit_meta.loc[unit_id, "unit_kind"]
        if "unit_bank" in unit_meta.columns:
            unit_bank = unit_meta.loc[unit_id, "unit_bank"]

    return fit_unit_interaction_model(
        y.values.astype(float),
        design,
        config,
        unit_id=int(unit_id),
        unit_kind=unit_kind,
        unit_bank=unit_bank,
        session_id=session_id,
    )


def run_single_session(
    session_key: str,
    output_dir: Path,
    config: InteractionConfig,
) -> Dict[str, Any]:
    """Run one session and save its outputs."""
    t0 = time.time()
    data = load_session_matrices(session_key, config)
    spike_wide = data["spike_wide"]
    design = data["design"]
    unit_meta = data["unit_meta"]

    if config.debug:
        print(
            f"[{session_key}] bins={design.n_rows_final}, motifs={design.retained_motifs}, "
            f"cells={design.retained_cells}"
        )

    unit_ids = list(spike_wide.columns)
    if config.n_jobs == 1:
        rows = [
            _fit_single_unit(
                unit_id=unit_id,
                y=spike_wide[unit_id],
                design=design,
                config=config,
                session_id=session_key,
                unit_meta=unit_meta,
            )
            for unit_id in unit_ids
        ]
    else:
        rows = Parallel(n_jobs=config.n_jobs, verbose=10 if config.debug else 0)(
            delayed(_fit_single_unit)(
                unit_id=unit_id,
                y=spike_wide[unit_id],
                design=design,
                config=config,
                session_id=session_key,
                unit_meta=unit_meta,
            )
            for unit_id in unit_ids
        )

    results_df = pd.DataFrame(rows)
    results_df = finalize_results(results_df, config)
    summary = summarize_interaction_results(results_df, design, session_key)
    summary["elapsed_s"] = round(time.time() - t0, 3)
    summary["config"] = asdict(config)

    session_dir = output_dir / session_key
    session_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(session_dir / "interaction_results.parquet", index=False)
    with (session_dir / "interaction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[{session_key}] units={len(results_df)} "
        f"median delta_R2={summary.get('median_delta_r2_interaction', float('nan')):.4f} "
        f"frac sig={summary.get('fraction_sig_interaction', float('nan')):.3f}"
    )
    return {
        "results_df": results_df,
        "summary": summary,
    }


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = InteractionConfig(
        bin_s=args.bin_s,
        n_perm=args.n_perm,
        n_perm_null=args.n_perm_null,
        lag_s=args.lag_s,
        min_bins_per_cell=args.min_bins_per_cell,
        min_bins_per_unit=args.min_bins_per_unit,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        sua_only=args.sua_only,
        include_time_drift=args.include_time_drift,
        include_rssi=args.include_rssi,
        include_antenna_distance=args.include_antenna_distance,
        debug=args.debug,
    )

    all_dfs: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, Any]] = []

    for session_key in args.sessions:
        result = run_single_session(session_key, output_dir, config)
        all_dfs.append(result["results_df"])
        all_summaries.append(result["summary"])

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_parquet(output_dir / "interaction_results_all_sessions.parquet", index=False)
        with (output_dir / "interaction_summary_all_sessions.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(all_summaries, f, indent=2)

    print("[fig5_interaction] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
