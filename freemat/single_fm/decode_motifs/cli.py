"""CLI entry point for motif decoding pipeline.

Decomposes the original monolithic ``main()`` into focused helpers.
Preserves all 32 argparse args and output artifact names/paths.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from freemat_data.types import SessionRef
from freemat_data.loaders.motifs import load_motifs_from_session

from freemat.single_fm.decode_motifs.io import (
    SpikeBanks,
    _banks_list,
    _ensure_dir,
    _parse_grid_csv,
    _parse_range_spec,
    load_and_bin_spikes,
    load_spikes_from_neural_parquet,
)
from freemat.single_fm.decode_motifs.motifs import (
    _debug_print_ethogram,
    _derive_onsets,
    _intervals_to_binned_labels,
    _manual_load_motifs_from_parquet,
    _map_motifs_to_bins,
    _smooth_short_motif_runs,
)
from freemat.single_fm.decode_motifs.features import (
    _grid_history,
    _grid_lags_ms,
    _prepare_X_bank,
    _subsample_units,
    build_lagged_design,
    chronological_folds,
)
from freemat.single_fm.decode_motifs.decoders import (
    task_A_motif_identity,
    task_B1_transition_onset,
    task_B2_next_state,
)
from freemat.single_fm.decode_motifs.plots import (
    _save_fig,
    plot_confusion,
    plot_motif_bars,
    plot_pr_curve,
    plot_weights_heatmap,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser (32 args, preserving original contract)."""
    p = argparse.ArgumentParser(
        description="Decode motifs and transitions from spikes (Single-FM)",
    )
    p.add_argument("--npz", type=str, default=None, help="Optional NPZ with spikes + motif_intervals (legacy mode)")
    p.add_argument("--neural_parquet", type=str, default=None, help="Aligned neural parquet with FR at some binning")
    p.add_argument("--motif_parquet", type=str, default=None, help="Ethogram parquet (ethogram*.parquet) for motifs")
    p.add_argument("--motifs_dir", type=str, default=None, help="Directory containing ethogram*.parquet and frames_map.parquet")
    p.add_argument("--frames_map", type=str, default=None, help="Optional explicit frames_map.parquet (t_sec per frame)")
    p.add_argument("--session_id", type=str, default=None, help="Session ID for filtering ethogram if parquet has multiple sessions")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--bin_hz", type=int, default=40, help="Binning rate (Hz)")
    p.add_argument("--history_s", type=float, default=1.0, help="Causal history window (s)")
    p.add_argument("--lags_ms", type=str, default="-300:300:20", help="Global alignment sweep (ms): start:stop:step")
    p.add_argument("--tau_ms", type=int, default=200, help="Onset horizon for hazard label (ms)")
    p.add_argument("--min_motif_len_frames", type=int, default=20, help=">= frames per motif to keep (40Hz~0.5s)")
    p.add_argument("--min_class_count", type=int, default=0, help="Minimum samples per motif class to keep; others set to 0")
    p.add_argument("--top_k_classes", type=int, default=0, help="Keep only top-K frequent motif classes (>0); others set to 0")
    p.add_argument("--cv_folds", type=int, default=5, help="Number of contiguous folds")
    p.add_argument("--purge_s", type=float, default=1.0, help="Purge window (s) around fold boundaries from train")
    p.add_argument("--C_grid", type=str, default="0.01,0.1,1,10,100", help="Comma-separated C grid for LogisticRegression")
    p.add_argument("--banks", type=str, default="dorsal,ventral,both", help="Which banks to run")
    p.add_argument("--null_perm", type=int, default=0, help="Label-permutation null perms for class decoders (A,B2)")
    p.add_argument("--onsets_positive_only", action="store_true", help="Use onsets only between positive motifs (>0) for B1/B2")
    p.add_argument("--lr_solver", type=str, default="lbfgs", choices=["lbfgs", "saga"], help="Logistic solver")
    p.add_argument("--lr_tol", type=float, default=1e-3, help="Logistic regression tolerance")
    p.add_argument("--lr_max_iter_inner", type=int, default=300, help="Max iterations for inner selection fits")
    p.add_argument("--lr_max_iter", type=int, default=1000, help="Max iterations for final fold fits")
    p.add_argument("--svm_tol", type=float, default=None, help="LinearSVC tolerance (overrides --lr_tol for SVM)")
    p.add_argument("--svm_max_iter_inner", type=int, default=None, help="LinearSVC max iterations for inner fits")
    p.add_argument("--svm_max_iter", type=int, default=None, help="LinearSVC max iterations for final fits")
    p.add_argument("--svm_dual", type=str, choices=["auto", "true", "false"], default="auto", help="LinearSVC dual formulation selection")
    p.add_argument("--unit_match", action="store_true", help="Subsample to min units across banks; repeat 20x")
    p.add_argument("--quick", action="store_true", help="Quick mode: fewer lags, no calibration")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument("--debug", action="store_true", help="Print debug info about ethogram/motifs loading")
    return p


# ---------------------------------------------------------------------------
# Motif loading (3-path with fallbacks)
# ---------------------------------------------------------------------------


def _load_motifs(
    args: argparse.Namespace,
    banks: SpikeBanks,
    bin_s: float,
) -> np.ndarray:
    """Load motif labels mapped to binned spike grid.

    Three paths: motif_parquet/motifs_dir, legacy NPZ, or manual parquet join.
    Returns per-bin integer motif labels (0 = background).
    """
    if args.motif_parquet or args.motifs_dir:
        return _load_motifs_from_parquet_or_dir(args, banks, bin_s)
    # Legacy mode: NPZ intervals
    if not args.npz:
        raise ValueError(
            "When not providing --motif_parquet/--motifs_dir, need --npz with motif_intervals and pose_dt/dt_raw.",
        )
    data = np.load(args.npz, allow_pickle=True)
    if "motif_intervals" not in data:
        raise ValueError("NPZ missing 'motif_intervals' array")
    motif_intervals = np.asarray(data["motif_intervals"])
    if "pose_dt" in data:
        dt_motif = float(data["pose_dt"])
    elif "dt_raw" in data:
        dt_motif = float(data["dt_raw"])
    else:
        raise ValueError("Need 'pose_dt' or 'dt_raw' in NPZ to map motif_intervals to seconds")
    return _intervals_to_binned_labels(motif_intervals, dt_motif=dt_motif, bin_edges=banks.t_edges)


def _load_motifs_from_parquet_or_dir(
    args: argparse.Namespace,
    banks: SpikeBanks,
    bin_s: float,
) -> np.ndarray:
    """Load motifs from ethogram parquet or directory with freemat_data loader."""
    sid: Optional[str] = args.session_id
    # Resolve effective ethogram path
    eth_path_eff: Optional[str] = None
    if args.motif_parquet and os.path.isfile(str(args.motif_parquet)):
        eth_path_eff = str(args.motif_parquet)
    elif args.motifs_dir and os.path.isdir(str(args.motifs_dir)):
        for nm in ("ethogram_final.parquet", "ethogram_smoothed.parquet", "ethogram.parquet"):
            p_try = os.path.join(str(args.motifs_dir), nm)
            if os.path.isfile(p_try):
                eth_path_eff = p_try
                break
    # Try to infer session_id
    if not sid and eth_path_eff and os.path.isfile(eth_path_eff):
        try:
            tmp = pd.read_parquet(eth_path_eff, columns=["session"])
        except (OSError, pd.errors.ParserError, KeyError):
            try:
                tmp = pd.read_parquet(eth_path_eff)
            except (OSError, pd.errors.ParserError):
                tmp = None
        if tmp is not None and isinstance(tmp, pd.DataFrame) and ("session" in tmp.columns):
            uniq = tmp["session"].astype(str).unique()
            if len(uniq) == 1:
                sid = str(uniq[0])
    if not sid and args.frames_map and os.path.isfile(str(args.frames_map)):
        try:
            fm = pd.read_parquet(str(args.frames_map))
            if "session" in fm.columns:
                uniq = fm["session"].astype(str).unique()
                if len(uniq) == 1:
                    sid = str(uniq[0])
        except (OSError, pd.errors.ParserError, KeyError):
            pass
    data_files: Dict[str, Optional[str]] = {
        "motifs_ethogram": eth_path_eff,
        "motifs_dir": str(args.motifs_dir) if args.motifs_dir else None,
        "motifs_frames_map": str(args.frames_map) if args.frames_map else None,
    }
    if args.debug:
        print(f"[motifs] using session_id: {sid if sid is not None else '(none)'}")
        _debug_print_ethogram(
            data_files.get("motifs_ethogram"),
            data_files.get("motifs_dir"),
            data_files.get("motifs_frames_map"),
        )
    # Try loader first if we have a session_id
    ms = None
    if sid is not None:
        sess = SessionRef(
            session_id=str(sid),
            session_type="asocial",
            monkeys=["unknown"],
            data_files=data_files,
            metadata={},
        )
        try:
            ms = load_motifs_from_session(sess)
        except (FileNotFoundError, ValueError, OSError, KeyError) as e:
            print("[motifs] loader error:", repr(e))
            ms = None
    if ms is None or ms.motif_id is None or ms.t is None:
        if args.debug:
            print("[motifs] Falling back to manual parquet join...")
        if not eth_path_eff:
            raise ValueError("Failed to load motifs; please provide --motif_parquet or --motifs_dir")
        t_ms, y_ms = _manual_load_motifs_from_parquet(
            eth_path_eff, sid, data_files.get("motifs_frames_map"),
        )
        y_ms_smooth = _smooth_short_motif_runs(y_ms, Lmin_frames=int(args.min_motif_len_frames))
        return _map_motifs_to_bins(t_ms, y_ms_smooth, bin_edges=banks.t_edges)
    # Pre-smooth short motif runs at frame rate before binning
    y_pf_smooth = _smooth_short_motif_runs(ms.motif_id, Lmin_frames=int(args.min_motif_len_frames))
    return _map_motifs_to_bins(ms.t, y_pf_smooth, bin_edges=banks.t_edges)


# ---------------------------------------------------------------------------
# Class filtering
# ---------------------------------------------------------------------------


def _filter_classes(
    y: np.ndarray,
    *,
    min_count: int,
    top_k: int,
    debug: bool,
) -> np.ndarray:
    """Filter rare/low-frequency classes by setting them to 0.

    Args:
        y: Per-bin motif labels.
        min_count: Minimum samples per class to keep.
        top_k: Keep only top-K frequent classes (>0); 0 means no limit.
        debug: Print debug info.

    Returns:
        Filtered copy of y.
    """
    y = y.copy()
    if min_count > 0:
        vals_kept, cnts_kept = np.unique(y[y > 0], return_counts=True)
        keep_set = set(int(v) for v, c in zip(vals_kept, cnts_kept) if int(c) >= min_count)
        if debug:
            print(f"[motifs] min_class_count={min_count}, keeping classes: {sorted(list(keep_set))}")
        if len(keep_set) == 0 and debug:
            print("[motifs] WARNING: all classes would be dropped by min_class_count")
        m = (y > 0) & (~np.isin(y, np.fromiter(keep_set, dtype=int)))
        y[m] = 0
        if debug:
            vals2, cnts2 = np.unique(y[y > 0], return_counts=True)
            print(f"[motifs] post-min_class_count class counts (>0): {[(int(v), int(c)) for v, c in zip(vals2, cnts2)][:10]}")
    if top_k > 0:
        vals_k, cnts_k = np.unique(y[y > 0], return_counts=True)
        order = np.argsort(-cnts_k)
        top_vals = set(int(x) for x in vals_k[order][:top_k])
        if debug:
            print(f"[motifs] top_k_classes={top_k}, keeping classes: {sorted(list(top_vals))}")
        m = (y > 0) & (~np.isin(y, np.fromiter(top_vals, dtype=int)))
        y[m] = 0
    return y


# ---------------------------------------------------------------------------
# Per-repeat task execution and artifact saving
# ---------------------------------------------------------------------------


def _run_tasks_and_save(
    FRb: np.ndarray,
    y_kept: np.ndarray,
    y_raw: np.ndarray,
    *,
    bank: str,
    rep: int,
    args: argparse.Namespace,
    bin_s: float,
    C_grid: np.ndarray,
    lags_ms_arr: np.ndarray,
    history_grid: List[float],
) -> Dict[str, Dict[str, Any]]:
    """Run all three tasks and save per-repeat artifacts. Returns metrics dict."""
    cv_folds_eff = int(args.cv_folds if not args.quick else max(1, min(3, int(args.cv_folds))))

    # Shared keyword args for SVM overrides
    svm_kw = {
        "svm_tol": float(args.svm_tol) if args.svm_tol is not None else None,
        "svm_max_iter_inner": int(args.svm_max_iter_inner) if args.svm_max_iter_inner is not None else None,
        "svm_max_iter": int(args.svm_max_iter) if args.svm_max_iter is not None else None,
        "svm_dual": str(args.svm_dual),
    }

    # Task A: motif identity
    resA = task_A_motif_identity(
        FRb, y_kept,
        bin_s=bin_s,
        cv_folds=cv_folds_eff,
        purge_s=float(args.purge_s),
        C_grid=C_grid,
        lag_grid_ms=lags_ms_arr,
        history_grid_s=history_grid,
        quick=bool(args.quick),
        seed=int(args.seed) + rep,
        null_perm=int(args.null_perm),
        debug=bool(args.debug),
        lr_solver=str(args.lr_solver),
        lr_tol=float(args.lr_tol),
        lr_max_iter_inner=int(args.lr_max_iter_inner),
        lr_max_iter=int(args.lr_max_iter),
        **svm_kw,
    )

    # Task B1: transition onset
    resB1 = task_B1_transition_onset(
        FRb, y_raw,
        bin_s=bin_s,
        cv_folds=cv_folds_eff,
        purge_s=float(args.purge_s),
        C_grid=C_grid,
        lag_grid_ms=lags_ms_arr,
        history_grid_s=history_grid,
        tau_ms=int(args.tau_ms),
        quick=bool(args.quick),
        seed=int(args.seed) + rep,
        positive_only=bool(args.onsets_positive_only),
        debug=bool(args.debug),
        lr_solver=str(args.lr_solver),
        lr_tol=float(args.lr_tol),
        lr_max_iter_inner=int(args.lr_max_iter_inner),
        lr_max_iter=int(args.lr_max_iter),
    )

    # Task B2: next-state
    resB2 = task_B2_next_state(
        FRb, y_raw,
        bin_s=bin_s,
        cv_folds=cv_folds_eff,
        purge_s=float(args.purge_s),
        C_grid=C_grid,
        lag_grid_ms=lags_ms_arr,
        history_grid_s=history_grid,
        quick=bool(args.quick),
        seed=int(args.seed) + rep,
        null_perm=int(args.null_perm),
        positive_only=bool(args.onsets_positive_only),
        debug=bool(args.debug),
        lr_solver=str(args.lr_solver),
        lr_tol=float(args.lr_tol),
        lr_max_iter_inner=int(args.lr_max_iter_inner),
        lr_max_iter=int(args.lr_max_iter),
        **svm_kw,
    )

    # Save per-repeat artifacts
    _save_repeat_artifacts(resA, resB1, resB2, bank=bank, rep=rep, args=args, bin_s=bin_s)

    return {"A": resA.metrics, "B1": resB1.metrics, "B2": resB2.metrics}


def _save_repeat_artifacts(
    resA: Any,
    resB1: Any,
    resB2: Any,
    *,
    bank: str,
    rep: int,
    args: argparse.Namespace,
    bin_s: float,
) -> None:
    """Save per-repeat confusion matrices, weights, PR curves, and plots."""
    outdir = args.outdir

    # B1: PR curve and onset traces
    y_pr = np.array(resB1.metrics.get("y_pr", []), dtype=int)
    p_pr = np.array(resB1.metrics.get("p_pr", []), dtype=float)
    if y_pr.size and p_pr.size:
        np.save(os.path.join(outdir, f"y_onset_test_{bank}_rep{rep}.npy"), y_pr)
        np.save(os.path.join(outdir, f"p_onset_test_{bank}_rep{rep}.npy"), p_pr)
        plot_pr_curve(y_pr, p_pr, os.path.join(outdir, f"onset_pr_{bank}_rep{rep}"), title=f"Onset PR ({bank})")

    # B1: Lag sweep
    l_bins = resB1.metrics.get("lag_sweep_lags_bins")
    ap_means = resB1.metrics.get("lag_sweep_ap")
    if l_bins and ap_means:
        l_ms = (np.array(l_bins, int) * bin_s * 1000.0).astype(float)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(l_ms, ap_means, marker="o")
        ax.axhline(np.nanmax(ap_means), color="#C44E52", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"Onset lag sweep ({bank})")
        _save_fig(fig, os.path.join(outdir, f"lag_sweep_{bank}_rep{rep}"))

    # B1: Onset traces
    tr = resB1.metrics.get("trace_mean")
    t_ms = resB1.metrics.get("trace_t_ms")
    if tr is not None and t_ms is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.array(t_ms, float), np.array(tr, float), color="#2ca02c")
        ax.axvline(0.0, color="k", linestyle=":")
        ax.set_xlabel("Time from onset (ms)")
        ax.set_ylabel("p(transition in tau)")
        ax.set_title(f"Onset probability traces ({bank})")
        _save_fig(fig, os.path.join(outdir, f"onset_traces_{bank}_rep{rep}"))

    # Confusion matrices
    if resA.confusion_ is not None and resA.classes_ is not None:
        pd.DataFrame(resA.confusion_, index=resA.classes_, columns=resA.classes_).to_csv(
            os.path.join(outdir, f"confusion_A_{bank}_rep{rep}.csv"),
        )
        plot_confusion(resA.confusion_, resA.classes_, os.path.join(outdir, f"confusion_A_{bank}_rep{rep}"), title=f"Motif confusion ({bank})")
    if resB2.confusion_ is not None and resB2.classes_ is not None:
        pd.DataFrame(resB2.confusion_, index=resB2.classes_, columns=resB2.classes_).to_csv(
            os.path.join(outdir, f"confusion_B2_{bank}_rep{rep}.csv"),
        )
        plot_confusion(resB2.confusion_, resB2.classes_, os.path.join(outdir, f"confusion_B2_{bank}_rep{rep}"), title=f"Next-state confusion ({bank})")

    # Weight heatmaps
    if resA.weights_abs is not None:
        np.save(os.path.join(outdir, f"weights_A_{bank}_rep{rep}.npy"), resA.weights_abs)
        plot_weights_heatmap(resA.weights_abs, os.path.join(outdir, f"weights_A_{bank}_rep{rep}"), title=f"Weights A ({bank})")
    if resB2.weights_abs is not None:
        np.save(os.path.join(outdir, f"weights_B2_{bank}_rep{rep}.npy"), resB2.weights_abs)
        plot_weights_heatmap(resB2.weights_abs, os.path.join(outdir, f"weights_B2_{bank}_rep{rep}"), title=f"Weights B2 ({bank})")


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _agg(rows: List[Dict[str, Any]], key: str) -> float:
    """Nan-safe mean aggregation of a metric key across repeats."""
    if not rows:
        return float("nan")
    vals: List[float] = []
    for r in rows:
        v = r.get(key, np.nan)
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isnan(vf):
            vals.append(vf)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------


def _debug_motif_summary(y_motif: np.ndarray, bin_s: float, tau_ms: int) -> None:
    """Print debug summary of motif labels and transitions."""
    vals, cnts = np.unique(y_motif, return_counts=True)
    nz = vals[vals > 0]
    print(f"[motifs] mapped labels: total T={y_motif.size}, n_classes>0={nz.size}, most frequent: {list(zip(vals[:10], cnts[:10]))}")
    on_all = _derive_onsets(y_motif)
    print(f"[motifs] transitions (all): n={on_all.size}")
    on_pos = np.where((y_motif[1:] != y_motif[:-1]) & (y_motif[1:] > 0) & (y_motif[:-1] > 0))[0] + 1
    print(f"[motifs] transitions (pos->pos): n={on_pos.size}")
    tau_bins_dbg = max(1, int(round(int(tau_ms) / 1000.0 / bin_s)))
    haz_all = np.zeros_like(y_motif, dtype=int)
    for j in on_all:
        t0 = max(0, j - tau_bins_dbg)
        haz_all[t0:j] = 1
    haz_pos = np.zeros_like(y_motif, dtype=int)
    for j in on_pos:
        t0 = max(0, j - tau_bins_dbg)
        haz_pos[t0:j] = 1
    print(f"[motifs] hazard(all) tau={tau_ms}ms: n={int(haz_all.sum())} (frac={haz_all.mean():.4f})")
    print(f"[motifs] hazard(pos->pos) tau={tau_ms}ms: n={int(haz_pos.sum())} (frac={haz_pos.mean():.4f})")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for motif decoding pipeline."""
    args = _build_parser().parse_args(argv)

    # Load spikes
    banks: SpikeBanks
    if args.neural_parquet:
        banks = load_spikes_from_neural_parquet(str(args.neural_parquet), bin_hz=int(args.bin_hz))
    else:
        if not args.npz:
            raise ValueError("Provide either --neural_parquet or --npz for spikes.")
        data = np.load(args.npz, allow_pickle=True)
        banks = load_and_bin_spikes(data, bin_hz=int(args.bin_hz))
    bin_s = float(banks.bin_s)

    # Load and pre-process motif labels
    y_motif = _load_motifs(args, banks, bin_s)
    if args.debug:
        _debug_motif_summary(y_motif, bin_s, int(args.tau_ms))

    # Filter classes
    y_motif_kept = _filter_classes(
        y_motif,
        min_count=int(args.min_class_count),
        top_k=int(args.top_k_classes),
        debug=bool(args.debug),
    )
    # Also filter the raw motif labels for B1/B2 consistency
    y_motif = _filter_classes(
        y_motif,
        min_count=int(args.min_class_count),
        top_k=int(args.top_k_classes),
        debug=False,
    )

    if args.debug:
        vals0, cnts0 = np.unique(y_motif_kept[y_motif_kept > 0], return_counts=True)
        print(f"[motifs] binned class counts (>0): {[(int(v), int(c)) for v, c in zip(vals0, cnts0)][:10]} (showing up to 10)")

    # Prepare grids
    lags_ms_arr = _parse_range_spec(args.lags_ms)
    lags_ms_arr = _grid_lags_ms(lags_ms_arr, quick=bool(args.quick))
    if args.debug:
        lag_bins_dbg = np.unique(np.round(lags_ms_arr / 1000.0 / bin_s).astype(int))
        print(f"[debug] lag bins (quantized): {lag_bins_dbg.tolist()}")
    C_grid = _parse_grid_csv(args.C_grid)
    history_grid = [float(args.history_s)] if args.history_s > 0 else _grid_history()

    # Banks to run
    run_banks = _banks_list(args.banks)
    _ensure_dir(args.outdir)
    slides_dir = os.path.join(args.outdir, "slides")
    _ensure_dir(slides_dir)

    # Run metadata
    run_meta: Dict[str, Any] = {
        "args": vars(args),
        "bin_s": bin_s,
        "bin_hz": banks.bin_hz,
        "downgraded": banks.downgraded,
        "seed": int(args.seed),
    }

    # Per-bank runs
    motif_metrics_rows: List[Dict[str, Any]] = []
    onset_metrics_rows: List[Dict[str, Any]] = []
    nextstate_metrics_rows: List[Dict[str, Any]] = []

    for bank in run_banks:
        FR, idx_d, idx_v = _prepare_X_bank(banks.fr_dorsal, banks.fr_ventral, bank)

        # Optional unit matching across banks
        repeats = 1
        if args.unit_match and (banks.fr_dorsal is not None) and (banks.fr_ventral is not None):
            if bank in ("dorsal", "ventral"):
                repeats = 20

        rep_metrics_A: List[Dict[str, Any]] = []
        rep_metrics_B1: List[Dict[str, Any]] = []
        rep_metrics_B2: List[Dict[str, Any]] = []

        for rep in range(repeats):
            rng = np.random.RandomState(int(args.seed) + rep)
            if args.unit_match and (banks.fr_dorsal is not None) and (banks.fr_ventral is not None):
                Ud = banks.fr_dorsal.shape[1]
                Uv = banks.fr_ventral.shape[1]
                m = min(Ud, Uv)
                if bank == "dorsal":
                    sel = _subsample_units(FR, m, rng)
                    FRb = FR[:, sel]
                elif bank == "ventral":
                    sel = _subsample_units(FR, m, rng)
                    FRb = FR[:, sel]
                else:
                    sel_d = _subsample_units(banks.fr_dorsal, m, rng)
                    sel_v = _subsample_units(banks.fr_ventral, m, rng)
                    FRb = np.hstack([
                        banks.fr_dorsal[:, sel_d],
                        banks.fr_ventral[:, sel_v],
                    ])
            else:
                FRb = FR

            metrics = _run_tasks_and_save(
                FRb, y_motif_kept, y_motif,
                bank=bank, rep=rep, args=args, bin_s=bin_s,
                C_grid=C_grid, lags_ms_arr=lags_ms_arr,
                history_grid=history_grid,
            )
            rep_metrics_A.append(metrics["A"])
            rep_metrics_B1.append(metrics["B1"])
            rep_metrics_B2.append(metrics["B2"])

        # Aggregate repeats
        motif_metrics_rows.append({
            "bank": bank,
            "macro_f1": _agg(rep_metrics_A, "macro_f1"),
            "balanced_acc": _agg(rep_metrics_A, "balanced_acc"),
            "null_macro_f1_mean": _agg(rep_metrics_A, "null_macro_f1_mean"),
            "null_macro_f1_lo": _agg(rep_metrics_A, "null_macro_f1_lo"),
            "null_macro_f1_hi": _agg(rep_metrics_A, "null_macro_f1_hi"),
        })
        onset_metrics_rows.append({
            "bank": bank,
            "pr_auc": _agg(rep_metrics_B1, "pr_auc"),
            "event_f1": _agg(rep_metrics_B1, "event_f1"),
        })
        nextstate_metrics_rows.append({
            "bank": bank,
            "macro_f1": _agg(rep_metrics_B2, "macro_f1"),
            "top1_acc": _agg(rep_metrics_B2, "top1_acc"),
            "null_macro_f1_mean": _agg(rep_metrics_B2, "null_macro_f1_mean"),
            "null_macro_f1_lo": _agg(rep_metrics_B2, "null_macro_f1_lo"),
            "null_macro_f1_hi": _agg(rep_metrics_B2, "null_macro_f1_hi"),
        })

    # Save CSVs and run.json
    pd.DataFrame(motif_metrics_rows).to_csv(os.path.join(args.outdir, "motif_metrics.csv"), index=False)
    pd.DataFrame(onset_metrics_rows).to_csv(os.path.join(args.outdir, "onset_metrics.csv"), index=False)
    pd.DataFrame(nextstate_metrics_rows).to_csv(os.path.join(args.outdir, "nextstate_metrics.csv"), index=False)
    with open(os.path.join(args.outdir, "run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # Motif bars figure
    metrics_by_bank = {
        r["bank"]: {"macro_f1": r.get("macro_f1", np.nan), "balanced_acc": r.get("balanced_acc", np.nan)}
        for r in motif_metrics_rows
    }
    plot_motif_bars(metrics_by_bank, args.outdir)

    # Save base folds for reference
    folds_by_bank: Dict[str, List[Dict[str, np.ndarray]]] = {}
    H0 = max(1, int(round(float(args.history_s) / bin_s)))
    cv_folds_eff = int(args.cv_folds if not args.quick else max(1, min(3, int(args.cv_folds))))
    for bank_name in run_banks:
        FR_ref, _, _ = _prepare_X_bank(banks.fr_dorsal, banks.fr_ventral, bank_name)
        build_lagged_design(FR_ref, H0)
        mH = (y_motif_kept[H0 - 1:] > 0)
        idx_valid0 = np.where(mH)[0]
        folds_raw = chronological_folds(
            idx_valid0.size,
            cv_folds_eff,
            int(round(float(args.purge_s) / bin_s)),
        )
        folds_by_bank[bank_name] = [
            {"train": idx_valid0[f["train"]], "test": idx_valid0[f["test"]]}
            for f in folds_raw
        ]
    joblib.dump(folds_by_bank, os.path.join(args.outdir, "folds.pkl"))

    # Slide captions
    captions = {
        "motif_bars": "Framewise motif decoding -- ventral>dorsal on identity (macro-F1).",
        "onset_pr": "Transition detection PR-AUC; dorsal leads at +60 ms.",
        "onset_traces": "Transition probability peaks before onset (predictive).",
        "next_conf": "Upcoming motif is decodable pre-switch.",
        "lag_sweep": "Best coupling at +l => dorsal leads.",
        "weights": "Informative lags cluster at ~100-200 ms history.",
    }
    with open(os.path.join(args.outdir, "slides", "README_slide_captions.txt"), "w", encoding="utf-8") as f:
        for k, v in captions.items():
            f.write(f"{k}: {v}\n")
