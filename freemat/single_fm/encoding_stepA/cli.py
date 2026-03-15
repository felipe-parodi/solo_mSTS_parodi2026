"""CLI entry point and orchestration for Step-A: Looking vs Movement.

Usage::

    python -m freemat.single_fm.encoding_stepA --neural neural.parquet --pose pose.parquet
    python -m freemat.single_fm.encoding_stepA --neural neural.parquet --pose-npz pose.npz
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

import numpy as np
import pandas as pd

from freemat.single_fm.encoding_stepA.data import (
    PRED_COLS,
    _apply_qc_filter,
    _longest_run_mask,
    _normalize_neural_df,
    _pose_df_from_npz_for_session,
    _sanity_predictor_coverage,
    _variance_ok_per_unit,
)
from freemat.single_fm.encoding_stepA.batch import (
    _console_summary,
    _compute_global_predictions,
    build_all_batch_caches,
    run_permutations_batch,
    score_units_batch,
)
from freemat.single_fm.kin.lite import compute_kin_lite, compute_kin_lite_dense
from freemat.single_fm.gam_models import (
    run_M1_look_still,
    run_M2_move_unique,
    run_M3_alignment,
)
from freemat.single_fm.plots_stepA import (
    plot_partial,
    plot_theta_tuning,
    scatter_yawrate_fr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_outdirs(root: str) -> tuple[str, str]:
    """Create output directories and return paths.

    Args:
        root: Output directory root.

    Returns:
        Tuple of (metrics_parquet_path, figs_directory_path).
    """
    metrics_path = os.path.join(root, "stepA_metrics.parquet")
    figs_dir = os.path.join(root, "stepA_figs")
    os.makedirs(figs_dir, exist_ok=True)
    return metrics_path, figs_dir


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for Step-A CLI."""
    ap = argparse.ArgumentParser(description="Step-A: Looking vs Movement (GAM)")
    ap.add_argument("--neural", required=True, help="Path to neural.parquet")
    ap.add_argument(
        "--pose", required=False,
        help="Path to pose.parquet with raw geometry (P_*, fhead_*, n_*)",
    )
    ap.add_argument(
        "--pose-npz", dest="pose_npz", default=None,
        help="Path to upright/leveled pose NPZ; derive pose geometry on the fly",
    )
    ap.add_argument("--outdir", default="results", help="Output directory root (default: results)")
    ap.add_argument("--s0q", type=float, default=0.15, help="Stillness threshold quantile per session (default: 0.15)")
    ap.add_argument("--smooth_win_s", type=float, default=0.05, help="COM smoothing window in seconds before differencing (default: 0.05s)")
    ap.add_argument("--max_fill_s", type=float, default=0.25, help="Max gap (s) to fill in head-direction before computing yaw_rate (default: 0.25s)")
    ap.add_argument("--min_forward_norm", type=float, default=0.05, help="Minimum in-plane forward norm to accept a frame (default: 0.05)")
    ap.add_argument("--com_max_gap_s", type=float, default=2.0, help="Max COM gap (s) to linearly interpolate before differencing (default: 2.0s)")
    ap.add_argument("--plots", action="store_true", help="If set, also generate population summary plots in the output directory")
    ap.add_argument("--longest", action="store_true", help="Restrict to longest contiguous segment with fully valid pose (all keypoints)")
    ap.add_argument("--perm_n", type=int, default=0, help="Number of permutations per unit for significance (batch mode recommended)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for permutation-based significance (default: 0.05)")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    ap.add_argument("--fast", action="store_true", help="Use faster solver (lstsq) and optional subsampling for CV fits")
    ap.add_argument("--batch", action="store_true", help="Use batch OLS with cached per-fold designs (fast, CPU)")
    ap.add_argument("--max_train_rows", type=int, default=None, help="Max train rows per fold (stratified by session) for fast fits")
    ap.add_argument("--dense-kin", dest="dense_kin", action="store_true", help="Compute kinematics at native pose rate and downsample to neural bins (requires --pose-npz)")
    return ap


def _score_units_nonbatch(
    T: pd.DataFrame,
    *,
    s0q: float,
    solver: str,
    n_folds: int,
    max_train_rows: int | None,
) -> tuple[list[dict], dict[int, dict], dict[int, dict], dict[int, dict]]:
    """Score units using per-unit GAM fits (non-batch mode).

    Args:
        T: Merged neural+feature table.
        s0q: Stillness quantile for M1.
        solver: Solver choice ('auto' or 'lstsq').
        n_folds: Number of CV folds.
        max_train_rows: Optional max train rows per fold.

    Returns:
        Tuple of (records, models_M1, models_M2, models_M3).
    """
    records: list[dict] = []
    models_M1: dict[int, dict] = {}
    models_M2: dict[int, dict] = {}
    models_M3: dict[int, dict] = {}
    for uid, G in T.groupby("unit_id", sort=False):
        try:
            m1 = run_M1_look_still(G, s0_quantile=float(s0q), max_train_rows=max_train_rows, solver=solver, n_folds=n_folds)
            m2 = run_M2_move_unique(G, max_train_rows=max_train_rows, solver=solver, n_folds=n_folds)
            m3 = run_M3_alignment(G, max_train_rows=max_train_rows, solver=solver, n_folds=n_folds)
            rec = {
                "unit_id": uid,
                "bank": G["bank"].iloc[0],
                **m1["metrics"],
                **m2["metrics"],
                **m3["metrics"],
            }
            if np.isfinite(rec.get("r2_move_plus_look", np.nan)) and np.isfinite(rec.get("r2_look_all", np.nan)):
                if rec["r2_move_plus_look"] + 1e-9 < rec["r2_look_all"]:
                    logging.warning(
                        "Unit %s: r2_move_plus_look < r2_look_all (%.4f < %.4f)",
                        str(uid), rec["r2_move_plus_look"], rec["r2_look_all"],
                    )
            for k in ("r2_look_still", "r2_look_all", "r2_move_plus_look", "r2_align"):
                v = rec.get(k, np.nan)
                if np.isfinite(v) and abs(v) > 10.0:
                    logging.warning("Unit %s: extreme %s = %.4f (check variance and predictions)", str(uid), k, v)
            records.append(rec)
            models_M1[uid] = m1.get("model") if solver != "lstsq" else None
            models_M2[uid] = m2.get("model") if solver != "lstsq" else None
            models_M3[uid] = m3.get("model") if solver != "lstsq" else None
        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            logging.exception("Unit %s failed: %s", str(uid), e)
    return records, models_M1, models_M2, models_M3


def _plot_exemplars(
    R: pd.DataFrame,
    T: pd.DataFrame,
    figs_dir: str,
    *,
    solver: str,
    s0q: float,
    models_M1: dict,
    models_M2: dict,
    models_M3: dict,
) -> None:
    """Generate exemplar unit plots for top-6 by move_unique.

    Args:
        R: Results DataFrame with per-unit metrics.
        T: Merged neural+feature table.
        figs_dir: Directory to save figures.
        solver: Solver used in main scoring.
        s0q: Stillness quantile.
        models_M1: Model dicts from non-batch scoring (or empty).
        models_M2: Model dicts from non-batch scoring (or empty).
        models_M3: Model dicts from non-batch scoring (or empty).
    """
    R_sorted = R.sort_values("move_unique", ascending=False)
    exemplars = R_sorted.head(6)["unit_id"].tolist()
    for uid in exemplars:
        G = T[T["unit_id"] == uid].copy()
        try:
            cov = {
                k: float(np.isfinite(G[k].to_numpy(float)).mean())
                for k in ["speed", "v_parallel", "v_perp", "v_z", "theta_look", "yaw_rate", "align_cos"]
                if k in G.columns
            }
            logging.info("[exemplar_cov] uid=%s n=%d %s", str(uid), len(G), {k: round(v, 3) for k, v in cov.items()})
        except (KeyError, TypeError):
            pass
        # If in fast/batch mode, refit slow models for plotting only
        if solver in ("lstsq", "batch"):
            try:
                m1 = run_M1_look_still(G, s0_quantile=float(s0q), max_train_rows=None, solver="auto")
                m2 = run_M2_move_unique(G, max_train_rows=None, solver="auto")
                m3 = run_M3_alignment(G, max_train_rows=None, solver="auto")
            except (ValueError, np.linalg.LinAlgError) as e:
                logging.warning("Refit for exemplar %s failed: %s", str(uid), e)
                m1 = {"model": None}
                m2 = {"model": None}
                m3 = {"model": None}
        else:
            m1 = {"model": models_M1.get(uid), "backend": "statsmodels"}
            m2 = {"model": models_M2.get(uid), "backend": "statsmodels"}
            m3 = {"model": models_M3.get(uid), "backend": "statsmodels"}
        plot_theta_tuning(
            m1.get("model") if isinstance(m1, dict) else m1,
            G, os.path.join(figs_dir, f"u{uid}_theta_tuning.png"),
        )
        plot_partial(
            m2.get("model") if isinstance(m2, dict) else m2,
            G, "v_parallel", os.path.join(figs_dir, f"u{uid}_partial_v_parallel.png"),
        )
        plot_partial(
            m3.get("model") if isinstance(m3, dict) else m3,
            G, "align_cos", os.path.join(figs_dir, f"u{uid}_partial_align_cos.png"),
        )
        scatter_yawrate_fr(G, os.path.join(figs_dir, f"u{uid}_yawrate_scatter.png"))


def _run_summary(R: pd.DataFrame, outdir: str) -> None:
    """Generate optional population-level summary plots.

    Args:
        R: Results DataFrame with per-unit metrics.
        outdir: Output directory root.
    """
    logging.info("[summary] Generating summary plots …")
    try:
        from freemat.single_fm.plots_summary import (
            plot_metric_distributions,
            plot_sig_counts,
            plot_ecdf,
            plot_scatter_move_vs_look,
            plot_unit_heatmap,
            plot_top_units_bar,
        )
        sumdir = os.path.join(outdir, "stepA_summary")
        os.makedirs(sumdir, exist_ok=True)
        plot_metric_distributions(R, "r2_look_still", os.path.join(sumdir, "dist_r2_look_still.png"), match_units=False)
        plot_metric_distributions(R, "move_unique", os.path.join(sumdir, "dist_move_unique.png"), match_units=False)
        plot_metric_distributions(R, "r2_align", os.path.join(sumdir, "dist_r2_align.png"), match_units=False)
        plot_sig_counts(R, "r2_look_still", 0.01, os.path.join(sumdir, "sig_r2_look_still.png"), match_units=False)
        plot_sig_counts(R, "move_unique", 0.0, os.path.join(sumdir, "sig_move_unique.png"), match_units=False)
        plot_sig_counts(R, "r2_align", 0.0, os.path.join(sumdir, "sig_r2_align.png"), match_units=False)
        plot_ecdf(R, "r2_look_still", os.path.join(sumdir, "ecdf_r2_look_still.png"), match_units=False)
        plot_ecdf(R, "move_unique", os.path.join(sumdir, "ecdf_move_unique.png"), match_units=False)
        plot_ecdf(R, "r2_align", os.path.join(sumdir, "ecdf_r2_align.png"), match_units=False)
        plot_scatter_move_vs_look(R, os.path.join(sumdir, "scatter_move_vs_look.png"), match_units=False)
        plot_unit_heatmap(R, os.path.join(sumdir, "heatmap_units.png"))
        plot_top_units_bar(R, os.path.join(sumdir, "top_units_move_unique.png"), metric="move_unique", top_n=10)
        logging.info("[summary] Plots saved to %s", sumdir)
    except (ImportError, ValueError, TypeError, OSError) as e:
        logging.warning("Summary plotting failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Step-A CLI entry point: Looking vs Movement encoding."""
    args = _build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # --- Load and normalize neural data ---
    neural = pd.read_parquet(args.neural)
    neural = _normalize_neural_df(neural)
    neural = _apply_qc_filter(neural)
    try:
        nsess = int(pd.unique(neural["session_id"]).size)
        nunits = int(neural["unit_id"].nunique()) if "unit_id" in neural.columns else 0
        banks = neural["bank"].value_counts(dropna=False).to_dict() if "bank" in neural.columns else {}
        logging.info("Neural normalized: sessions=%d units=%d banks=%s", nsess, nunits, banks)
    except (KeyError, TypeError):
        pass

    # --- Load or derive pose ---
    pose: pd.DataFrame | None = None
    if args.pose:
        try:
            pose = pd.read_parquet(args.pose)
        except (FileNotFoundError, OSError, ValueError) as e:
            logging.warning("Failed to read --pose parquet '%s': %s", args.pose, e)
            pose = None

    required_pose_cols = {
        "session_id", "t_bin_s",
        "P_x", "P_y", "P_z",
        "fhead_x", "fhead_y", "fhead_z",
        "n_x", "n_y", "n_z",
    }
    have_valid_pose = bool(pose is not None and required_pose_cols.issubset(set(pose.columns)))
    if not have_valid_pose:
        if args.pose_npz:
            logging.info("Deriving pose geometry from NPZ using neural bins")
            pose = _pose_df_from_npz_for_session(args.pose_npz, neural)
            try:
                sid = str(pose["session_id"].iloc[0]) if "session_id" in pose.columns and len(pose) else "session"
                os.makedirs(args.outdir, exist_ok=True)
                pose_out = os.path.join(args.outdir, f"pose_from_npz_{sid}.parquet")
                pose.to_parquet(pose_out, index=False)
                logging.info("Derived pose written: %s", pose_out)
            except (OSError, PermissionError) as e:
                logging.warning("Failed to save derived pose parquet: %s", e)
        else:
            missing = set() if pose is None else required_pose_cols.difference(set(pose.columns))
            raise KeyError(
                "Pose input is missing required geometry columns and --pose-npz was not provided. "
                f"Missing: {sorted(missing)}"
            )

    # --- Compute features ---
    if bool(args.dense_kin):
        if not args.pose_npz:
            raise ValueError("--dense-kin requires --pose-npz to compute dense kinematics")
        feats = compute_kin_lite_dense(
            args.pose_npz, neural,
            smooth_win_s=float(args.smooth_win_s),
            max_fill_s=float(args.max_fill_s),
            min_forward_norm=float(args.min_forward_norm),
            com_max_gap_s=float(args.com_max_gap_s),
        )
    else:
        feats = compute_kin_lite(
            pose,
            smooth_win_s=float(args.smooth_win_s),
            max_fill_s=float(args.max_fill_s),
            min_forward_norm=float(args.min_forward_norm),
            com_max_gap_s=float(args.com_max_gap_s),
        )
    _sanity_predictor_coverage(feats)

    # --- Join neural + features ---
    base_cols = ["session_id", "t_bin_s"]
    try:
        if not feats.empty:
            dup_mask = feats.duplicated(subset=base_cols, keep=False)
            if bool(dup_mask.any()):
                n_dup = int(dup_mask.sum())
                logging.warning(
                    "[dense_kin] features contain %d duplicate keys on %s; dropping duplicates by first occurrence",
                    n_dup, str(tuple(base_cols)),
                )
                feats = feats.sort_values(base_cols, kind="mergesort").drop_duplicates(subset=base_cols, keep="first")
    except (KeyError, TypeError):
        pass

    T = neural.merge(feats, on=base_cols, how="inner", suffixes=("_neu", "_pose"))
    for c in PRED_COLS:
        if c in T.columns:
            continue
        if f"{c}_pose" in T.columns:
            T = T.rename(columns={f"{c}_pose": c})
        elif f"{c}_neu" in T.columns:
            T = T.rename(columns={f"{c}_neu": c})

    for col in ("unit_id", "fr"):
        if col not in T.columns:
            raise KeyError(f"Neural table missing required column '{col}' after normalization")
    if "bank" not in T.columns:
        T["bank"] = "unknown"
    present_preds = [c for c in PRED_COLS if c in T.columns]
    extra_cols = [c for c in ("pose_full_valid",) if c in T.columns]
    T = T[["session_id", "unit_id", "bank", "t_bin_s", "fr", *present_preds, *extra_cols]].copy()
    T = T.replace([np.inf, -np.inf], np.nan)
    logging.info("Rows after join: %d", len(T))

    # --- Longest segment filter ---
    if bool(args.longest):
        if "pose_full_valid" not in T.columns:
            logging.warning("--longest requested but pose_full_valid not available; skipping")
        else:
            keep_mask = pd.Series(False, index=T.index)
            for _sid, g in T.groupby("session_id", sort=False):
                m_s = _longest_run_mask(g)
                keep_mask.loc[m_s.index] = m_s.values
            before = len(T)
            T = T.loc[keep_mask.values].copy()
            logging.info("[longest] kept %d/%d rows (%.1f%%)", len(T), before, 100.0 * len(T) / max(1, before))

    # --- Variance filter ---
    logging.info("Units before variance filter: %d", int(T["unit_id"].nunique()) if not T.empty else 0)
    T = _variance_ok_per_unit(T)
    logging.info("Units after variance filter: %d", int(T["unit_id"].nunique()) if not T.empty else 0)
    if T.empty:
        logging.error("No units with valid data after filtering")
        return

    metrics_path, figs_dir = _ensure_outdirs(args.outdir)

    # --- Solver dispatch ---
    solver = "batch" if bool(args.batch) else ("lstsq" if bool(args.fast) else "auto")
    max_train_rows = int(args.max_train_rows) if args.max_train_rows is not None else None
    n_folds = int(args.folds)

    records: list[dict] = []
    models_M1: dict = {}
    models_M2: dict = {}
    models_M3: dict = {}
    perm_pvals: dict[str, dict] = {"look_still": {}, "move_unique": {}, "align": {}}

    if solver == "batch":
        caches = build_all_batch_caches(T, s0q=float(args.s0q), n_folds=n_folds)
        preds = _compute_global_predictions(T, caches, s0q=float(args.s0q), n_folds=n_folds)
        records = score_units_batch(T, caches, preds)
        if isinstance(args.perm_n, int) and args.perm_n > 0:
            perm_pvals = run_permutations_batch(
                T, caches, records, perm_n=int(args.perm_n), n_folds=n_folds,
            )
    else:
        records, models_M1, models_M2, models_M3 = _score_units_nonbatch(
            T, s0q=float(args.s0q), solver=solver,
            n_folds=n_folds, max_train_rows=max_train_rows,
        )

    if not records:
        logging.error("No unit metrics computed")
        return

    # --- Build results DataFrame ---
    R = pd.DataFrame.from_records(records)
    if any(len(d) > 0 for d in perm_pvals.values()):
        try:
            R["p_look_still"] = R["unit_id"].map(perm_pvals["look_still"]).astype(float)
            R["p_move_unique"] = R["unit_id"].map(perm_pvals["move_unique"]).astype(float)
            R["p_align"] = R["unit_id"].map(perm_pvals["align"]).astype(float)
            alpha = float(args.alpha)
            R["sig_look_still"] = R["p_look_still"] < alpha
            R["sig_move_unique"] = R["p_move_unique"] < alpha
            R["sig_align"] = R["p_align"] < alpha
        except (KeyError, TypeError) as e:
            logging.warning("Failed to attach permutation p-values: %s", e)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    R.to_parquet(metrics_path, index=False)

    # --- Exemplar plots ---
    _plot_exemplars(
        R, T, figs_dir,
        solver=solver, s0q=float(args.s0q),
        models_M1=models_M1, models_M2=models_M2, models_M3=models_M3,
    )

    _console_summary(R)

    if args.plots:
        _run_summary(R, args.outdir)
