"""Figure 2B: Unique Variance Analysis (KIN vs VIS).

Main script for running per-unit unique variance encoding analysis.
Computes delta CV-R² for KIN vs VIS predictor families.

Usage:
    >>> from freemat.single_fm.encoding.run_fig2b import run_fig2b_encoding
    >>> results_df = run_fig2b_encoding(
    ...     session_yaml="configs/single_fm/session_M1_0525.yaml",
    ...     output_dir="results/single_fm/240525",
    ...     bin_s=0.5,
    ...     n_perm=1000,
    ... )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from freemat.single_fm.encoding.cv import CVConfig
from freemat.single_fm.encoding.config import Fig2BConfig, apply_fdr_correction
from freemat.single_fm.encoding.temporal import (
    build_lagged_features,
    _add_time_drift_covariate,
    run_subsample_sensitivity,
)
from freemat.single_fm.encoding.variance import (
    compute_unique_variance_3way,
    compute_unique_variance_nway_shuffle,
)
from freemat.single_fm.encoding.ridge import build_spline_design
from freemat.single_fm.encoding.visual_features import (
    BODY_KIN_COLS,
    HEAD_KIN_COLS,
    POS_COLS,
    get_feature_cols,
)
from freemat.single_fm.data import SingleFMSessionLoader


def _process_single_unit(
    unit_id: int,
    unit_data: pd.DataFrame,
    vis_cols: List[str],
    kin_cols: List[str],
    body_kin_cols: List[str],
    head_kin_cols: List[str],
    cv_cfg: CVConfig,
    config: Fig2BConfig,
    session_id: str,
    rng_seed: int,
    unit_bank: Optional[str] = None,
    pos_cols: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single unit for unique variance analysis.

    This function is designed to be called in parallel via joblib.

    Args:
        unit_id: Unit identifier.
        unit_data: Pre-filtered DataFrame for this unit.
        vis_cols: Available VIS column names.
        kin_cols: Available KIN column names (all combined).
        body_kin_cols: BODY_KIN column names (for 3-way).
        head_kin_cols: HEAD_KIN column names (for 3-way).
        cv_cfg: Cross-validation configuration.
        config: Figure 2B configuration.
        session_id: Session identifier.
        rng_seed: Random seed for reproducibility.
        unit_bank: Bank assignment for this unit.
        pos_cols: POS column names (for 4-way with 3D position).

    Returns:
        Dict with results or None if unit should be skipped.
    """
    # Fill NaN in HEAD_KIN features with 0 (rate features: 0 = no rotation)
    nan_cols_to_fill = [
        col for col in head_kin_cols
        if col in unit_data.columns and unit_data[col].isna().all()
    ]
    if nan_cols_to_fill:
        unit_data = unit_data.copy()  # Avoid SettingWithCopyWarning
        for col in nan_cols_to_fill:
            unit_data[col] = 0.0

    # Drop rows with NaN in features (after filling HEAD_KIN)
    feature_cols = vis_cols + kin_cols + ["firing_rate"]
    if "t_bin_s" in unit_data.columns:
        feature_cols.append("t_bin_s")
    unit_data = unit_data.dropna(subset=[c for c in feature_cols if c in unit_data.columns])

    if len(unit_data) < config.min_bins_per_unit:
        return None

    # Extract arrays
    y = unit_data["firing_rate"].values.astype(np.float64)
    X_vis = unit_data[vis_cols].values.astype(np.float64)

    # Use unique RNG per unit for reproducibility in parallel
    rng = np.random.default_rng(rng_seed)

    # N-way partitioning: VIS vs BODY_KIN vs HEAD_KIN (+ optional POS)
    X_body_kin = unit_data[body_kin_cols].values.astype(np.float64)
    X_head_kin = unit_data[head_kin_cols].values.astype(np.float64)

    # Extract and z-score POS features inline (raw meters in DataFrame)
    has_pos = pos_cols is not None and len(pos_cols) > 0
    if has_pos:
        X_pos = unit_data[pos_cols].values.astype(np.float64)
        for j in range(X_pos.shape[1]):
            col_vals = X_pos[:, j]
            finite = np.isfinite(col_vals)
            if finite.any():
                m, s = col_vals[finite].mean(), col_vals[finite].std()
                s = s if s > 1e-12 else 1.0
                X_pos[:, j] = (col_vals - m) / s

    # Apply temporal kernel expansion (lagged features) if configured
    # NOTE: POS is unlagged (position, not dynamics) per design
    if config.n_lags > 0:
        X_vis = build_lagged_features(X_vis, config.n_lags)
        X_body_kin = build_lagged_features(X_body_kin, config.n_lags)
        X_head_kin = build_lagged_features(X_head_kin, config.n_lags)

        # Drop edge rows with NaNs from lagging
        trim = config.n_lags
        X_vis = X_vis[trim:-trim]
        X_body_kin = X_body_kin[trim:-trim]
        X_head_kin = X_head_kin[trim:-trim]
        y = y[trim:-trim]
        if has_pos:
            X_pos = X_pos[trim:-trim]

        # Re-check minimum bins after trimming
        if len(y) < config.min_bins_per_unit:
            return None

    # Apply GAM spline expansion (NOT applied to POS — linear only)
    if config.use_gam:
        X_vis = build_spline_design(X_vis, n_knots=config.gam_n_knots)
        X_body_kin = build_spline_design(X_body_kin, n_knots=config.gam_n_knots)
        X_head_kin = build_spline_design(X_head_kin, n_knots=config.gam_n_knots)

    # Add time drift as a global control covariate (unlagged, linear) to ALL
    # groups.  Appended AFTER lag expansion and AFTER GAM spline expansion so
    # drift itself is neither lagged nor spline-transformed.
    #
    # NOTE: drift is duplicated inside each group matrix, so when a group is
    # shuffled (Testard method) its drift copy is also shuffled.  The bias is
    # small (1 col out of many, slow trend) but for a principled nuisance
    # regressor, a separate shared control block would be cleaner (future).
    if config.include_time_drift and "t_bin_s" in unit_data.columns:
        t_raw = unit_data["t_bin_s"].values.astype(np.float64)
        # Trim to match lagged arrays
        if config.n_lags > 0:
            t_raw = t_raw[config.n_lags:-config.n_lags]
        X_vis = _add_time_drift_covariate(t_raw, X_vis)
        X_body_kin = _add_time_drift_covariate(t_raw, X_body_kin)
        X_head_kin = _add_time_drift_covariate(t_raw, X_head_kin)
        if has_pos:
            X_pos = _add_time_drift_covariate(t_raw, X_pos)

    # Add RSSI as a control covariate (unlagged, z-scored) to ALL groups.
    # Same duplication strategy as time_drift: each group gets a copy.
    if config.include_rssi:
        if "rssi" not in unit_data.columns:
            raise RuntimeError(
                f"include_rssi=True but 'rssi' column missing for unit {unit_id}. "
                "Ensure RSSI was loaded in run_fig2b_encoding()."
            )
        from freemat.single_fm.encoding.temporal import _add_rssi_covariate

        rssi_raw = unit_data["rssi"].values.astype(np.float64)
        if config.n_lags > 0:
            rssi_raw = rssi_raw[config.n_lags:-config.n_lags]
        X_vis = _add_rssi_covariate(rssi_raw, X_vis)
        X_body_kin = _add_rssi_covariate(rssi_raw, X_body_kin)
        X_head_kin = _add_rssi_covariate(rssi_raw, X_head_kin)
        if has_pos:
            X_pos = _add_rssi_covariate(rssi_raw, X_pos)

    # Add antenna distance as a control covariate (unlagged, z-scored) to ALL groups.
    if config.include_antenna_distance:
        if "d_antenna" not in unit_data.columns:
            raise RuntimeError(
                f"include_antenna_distance=True but 'd_antenna' column missing for unit {unit_id}. "
                "Ensure antenna distance was loaded in run_fig2b_encoding()."
            )
        from freemat.single_fm.encoding.temporal import _add_antenna_distance_covariate

        d_ant_raw = unit_data["d_antenna"].values.astype(np.float64)
        if config.n_lags > 0:
            d_ant_raw = d_ant_raw[config.n_lags:-config.n_lags]
        X_vis = _add_antenna_distance_covariate(d_ant_raw, X_vis)
        X_body_kin = _add_antenna_distance_covariate(d_ant_raw, X_body_kin)
        X_head_kin = _add_antenna_distance_covariate(d_ant_raw, X_head_kin)
        if has_pos:
            X_pos = _add_antenna_distance_covariate(d_ant_raw, X_pos)

    # Build feature groups dict for N-way decomposition
    feature_groups: Dict[str, np.ndarray] = {
        "vis": X_vis,
        "body_kin": X_body_kin,
        "head_kin": X_head_kin,
    }
    if has_pos:
        feature_groups["pos"] = X_pos

    # Add motif one-hot encoding if motif_label column is present (Fig 4C)
    if "motif_label" in unit_data.columns:
        motif_vals = unit_data["motif_label"].values
        # After lag trimming, trim motif_vals to match y
        if config.n_lags > 0:
            trim = config.n_lags
            motif_vals = motif_vals[trim:-trim]
        unique_motifs = sorted(set(int(m) for m in motif_vals if m > 0))
        if len(unique_motifs) > 1:
            n_t = len(y)
            X_motif = np.zeros((n_t, len(unique_motifs)), dtype=np.float64)
            for i, mid in enumerate(unique_motifs):
                X_motif[:, i] = (motif_vals[:n_t] == mid).astype(np.float64)
            feature_groups["motif"] = X_motif

    group_names = list(feature_groups.keys())

    # Choose between shuffle method (Testard) and dropout method
    if config.use_shuffle_method:
        # N-way Testard-style regressor-shuffle method
        uv = compute_unique_variance_nway_shuffle(
            feature_groups, y,
            cv_cfg=cv_cfg,
            bin_s=config.bin_s,
            n_perm=config.n_perm,
            n_perm_null=config.n_perm_null,
            block_size_s=config.perm_block_s,
            alphas=config.alphas,
            rng=rng,
            use_gpu=config.use_gpu,
        )

        # Build result dict from N-way to_dict() (flat columns)
        r2_threshold_passed = uv.full_r2 > config.r2_threshold
        result = uv.to_dict()

        # Add significance flags per group
        for g in group_names:
            effect_ok = result.get(f"unique_{g}_shuffle", 0) >= config.effect_threshold
            result[f"sig_{g}"] = effect_ok and r2_threshold_passed

        # Add metadata
        result.update({
            "unit_id": unit_id,
            "session_id": session_id,
            "n_bins": len(y),
            "model_type": "gam" if config.use_gam else "ridge",
            "include_time_drift": config.include_time_drift,
            "include_rssi": config.include_rssi,
            "include_antenna_distance": config.include_antenna_distance,
            "r2_threshold_passed": r2_threshold_passed,
            "unit_bank": unit_bank,
            "bin_s": config.bin_s,
            "n_lags": config.n_lags,
            "floor_filter_before_features": config.floor_filter_before_features,
            "method": "shuffle",
        })

        # Save per-bin residuals for spatial analysis (Fig 2D)
        if config.save_residuals and uv.full_model_predictions is not None:
            preds = uv.full_model_predictions
            T = len(y)
            # Reconstruct aligned residuals from CV test folds
            y_pred_aligned = np.full(T, np.nan)
            y_true_aligned = np.full(T, np.nan)
            y_pred_aligned[preds["test_idx"]] = preds["y_pred"]
            y_true_aligned[preds["test_idx"]] = preds["y_true"]

            # Trim metadata arrays to match lag-trimmed data
            trim = config.n_lags if config.n_lags > 0 else 0
            t_bins_trimmed = unit_data["t_bin_s"].values
            if trim > 0:
                t_bins_trimmed = t_bins_trimmed[trim:-trim]

            resid_dict: Dict[str, Any] = {
                "unit_id": np.full(T, unit_id, dtype=int),
                "t_bin_s": t_bins_trimmed,
                "y_true": y_true_aligned,
                "y_pred": y_pred_aligned,
                "residual": y_true_aligned - y_pred_aligned,
            }

            # Add COM position if available
            for col in ["com_x", "com_y", "com_z"]:
                if col in unit_data.columns:
                    vals = unit_data[col].values
                    if trim > 0:
                        vals = vals[trim:-trim]
                    resid_dict[col] = vals

            # Add on_floor if available
            if "on_floor" in unit_data.columns:
                on_floor_vals = unit_data["on_floor"].values
                if trim > 0:
                    on_floor_vals = on_floor_vals[trim:-trim]
                resid_dict["on_floor"] = on_floor_vals

            result["_residuals"] = resid_dict

        return result
    else:
        # Original dropout method
        uv = compute_unique_variance_3way(
            X_vis, X_body_kin, X_head_kin, y,
            cv_cfg=cv_cfg,
            bin_s=config.bin_s,
            n_perm=config.n_perm,
            perm_block_s=config.perm_block_s,
            alphas=config.alphas,
            rng=rng,
            use_gpu=config.use_gpu,
        )

        r2_threshold_passed = uv.full_r2 > config.r2_threshold

        sig_vis = (
            (uv.p_vis is not None and uv.p_vis < config.p_threshold)
            and (uv.unique_vis >= config.effect_threshold)
        )
        sig_body_kin = (
            (uv.p_body_kin is not None and uv.p_body_kin < config.p_threshold)
            and (uv.unique_body_kin >= config.effect_threshold)
        )
        sig_head_kin = (
            (uv.p_head_kin is not None and uv.p_head_kin < config.p_threshold)
            and (uv.unique_head_kin >= config.effect_threshold)
        )

        null_95_vis = None
        null_95_body_kin = None
        null_95_head_kin = None
        if uv.null_dist_vis is not None and len(uv.null_dist_vis) > 0:
            null_95_vis = float(np.percentile(uv.null_dist_vis, 95))
        if uv.null_dist_body_kin is not None and len(uv.null_dist_body_kin) > 0:
            null_95_body_kin = float(np.percentile(uv.null_dist_body_kin, 95))
        if uv.null_dist_head_kin is not None and len(uv.null_dist_head_kin) > 0:
            null_95_head_kin = float(np.percentile(uv.null_dist_head_kin, 95))

        return {
            "unit_id": unit_id,
            "session_id": session_id,
            "n_bins": len(y),
            "full_r2": uv.full_r2,
            "vis_only_r2": uv.vis_only_r2,
            "body_kin_only_r2": uv.body_kin_only_r2,
            "head_kin_only_r2": uv.head_kin_only_r2,
            "no_vis_r2": uv.no_vis_r2,
            "no_body_kin_r2": uv.no_body_kin_r2,
            "no_head_kin_r2": uv.no_head_kin_r2,
            "unique_vis": uv.unique_vis,
            "unique_body_kin": uv.unique_body_kin,
            "unique_head_kin": uv.unique_head_kin,
            "shared_all": uv.shared_all,
            "p_vis": uv.p_vis,
            "p_body_kin": uv.p_body_kin,
            "p_head_kin": uv.p_head_kin,
            "sig_vis": sig_vis,
            "sig_body_kin": sig_body_kin,
            "sig_head_kin": sig_head_kin,
            "null_95_vis": null_95_vis,
            "null_95_body_kin": null_95_body_kin,
            "null_95_head_kin": null_95_head_kin,
            "model_type": "gam" if config.use_gam else "ridge",
            "include_time_drift": config.include_time_drift,
            "include_rssi": config.include_rssi,
            "include_antenna_distance": config.include_antenna_distance,
            "r2_threshold_passed": r2_threshold_passed,
            "unit_bank": unit_bank,
            "bin_s": config.bin_s,
            "n_lags": config.n_lags,
            "floor_filter_before_features": config.floor_filter_before_features,
            "method": "dropout",
        }


def run_fig2b_encoding(
    session_yaml: str,
    output_dir: str,
    *,
    config: Optional[Fig2BConfig] = None,
    start_stop_tsv: Optional[str] = None,
    unit_ids: Optional[Sequence[int]] = None,
    save_results: bool = True,
    canonical_alignment_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run Figure 2B encoding analysis for one session.

    Args:
        session_yaml: Path to session YAML configuration.
        output_dir: Directory for output files.
        config: Analysis configuration.
        start_stop_tsv: Path to inference time filter TSV. If None,
            uses default ``docs/inf_start_stop_times.tsv``.
        unit_ids: Optional subset of unit IDs to analyze.
        save_results: Whether to save results to parquet.

    Returns:
        DataFrame with per-unit unique variance results.
    """
    if config is None:
        config = Fig2BConfig()

    from freemat_data import load_session_yaml

    sref = load_session_yaml(session_yaml)
    session_id = str(getattr(sref, "session_id", "session"))

    if config.debug:
        print(f"[Fig2B] Session: {session_id}")
        print(f"[Fig2B] Bin size: {config.bin_s}s, n_perm: {config.n_perm}")

    # Load aligned data using SingleFMSessionLoader
    loader = SingleFMSessionLoader(
        session_yaml,
        start_stop_tsv=start_stop_tsv,
        canonical_alignment_path=canonical_alignment_path,
        debug=config.debug,
    )

    if config.debug:
        loader.print_diagnostics()

    # Coordinate validation gate (FAIL-STOP)
    if config.validate_coordinates:
        from freemat.single_fm.encoding.coord_validation import (
            run_validation_gate,
            print_validation_report,
        )
        validation_output = os.path.join(output_dir, session_id) if output_dir else None
        report = run_validation_gate(loader, output_dir=validation_output)
        if config.debug:
            print_validation_report(report)
        if not report.passed:
            failures = ", ".join(
                f"{f.name} (r={f.correlation:.4f})" for f in report.failures
            )
            raise RuntimeError(
                f"[Fig2B] Coordinate validation FAILED for {session_id}. "
                f"Features below r>{report.threshold}: {failures}. "
                f"Encoding aborted. Fix coordinate issues before proceeding."
            )
        elif config.debug:
            print(f"[Fig2B] Coordinate validation PASSED for {session_id}")

    # Get merged spikes + features (all in video time, filtered to valid window)
    merged_df = loader.get_aligned_data(
        bin_s=config.bin_s,
        floor_filter=config.floor_filter,
        floor_height_threshold=config.floor_height_threshold,
        floor_filter_before_features=config.floor_filter_before_features,
        binning_method=config.binning_method,
    )

    if config.debug:
        print(f"[Fig2B] Merged: {len(merged_df)} rows")

    if len(merged_df) == 0:
        print("[Fig2B] WARNING: No merged data! Check time alignment.")
        return pd.DataFrame()

    # Load RSSI for session (once, shared across all units)
    if config.include_rssi:
        from freemat.single_fm.encoding.temporal import load_rssi_for_session

        unique_bins = np.sort(merged_df["t_bin_s"].unique())
        rssi_session = load_rssi_for_session(session_yaml, unique_bins)
        rssi_map = dict(zip(unique_bins, rssi_session))
        merged_df["rssi"] = merged_df["t_bin_s"].map(rssi_map)
        if config.debug:
            print(
                f"[Fig2B] RSSI loaded: {len(unique_bins)} bins, "
                f"range [{rssi_session.min():.1f}, {rssi_session.max():.1f}] "
                f"(z-scored)"
            )

    # Load antenna distance for session (once, shared across all units)
    if config.include_antenna_distance:
        from freemat.single_fm.encoding.temporal import load_antenna_distance_for_session

        unique_bins = np.sort(merged_df["t_bin_s"].unique())
        d_antenna_session = load_antenna_distance_for_session(
            session_yaml, unique_bins, merged_df,
            pipeline_floor_level=loader.floor_level,
        )
        d_antenna_map = dict(zip(unique_bins, d_antenna_session))
        merged_df["d_antenna"] = merged_df["t_bin_s"].map(d_antenna_map)
        n_nan_ant = int(merged_df["d_antenna"].isna().sum())
        if n_nan_ant > 0:
            pct_nan = 100 * n_nan_ant / len(merged_df)
            print(
                f"  d_antenna: {n_nan_ant}/{len(merged_df)} mapped values "
                f"({pct_nan:.1f}%) are NaN (from missing COM bins) "
                f"-- handled per-unit via dropna"
            )
        if config.debug:
            print(
                f"[Fig2B] Antenna distance loaded: {len(unique_bins)} bins, "
                f"range [{d_antenna_session.min():.1f}, {d_antenna_session.max():.1f}] "
                f"(z-scored)"
            )

    # Load and align motif labels if include_motif is True (Fig 4C)
    if config.include_motif:
        from analysis.syllables.alignment import align_syllables_to_bins
        from freemat_data import load_motifs_from_session

        motifs = load_motifs_from_session(loader._session)
        if motifs is None or motifs.t is None:
            raise RuntimeError(
                f"include_motif=True but motifs not found for {session_id}. "
                "Check motifs_dir in session YAML."
            )
        t_start, t_end = loader.time_window
        bin_edges = np.arange(
            float(t_start), float(t_end) + float(config.bin_s),
            float(config.bin_s), dtype=float,
        )
        motif_bin_labels = align_syllables_to_bins(
            motifs.motif_id, motifs.t, bin_edges,
            method="center", missing_value=-999,
        )
        # Map bin times to motif labels
        unique_bins = np.sort(merged_df["t_bin_s"].unique())
        bin_to_motif = {}
        for i, t_center in enumerate(
            (bin_edges[:-1] + bin_edges[1:]) / 2
        ):
            if i < len(motif_bin_labels):
                # Find closest unique bin
                idx = np.argmin(np.abs(unique_bins - t_center))
                if np.abs(unique_bins[idx] - t_center) < config.bin_s:
                    bin_to_motif[unique_bins[idx]] = int(motif_bin_labels[i])
        merged_df["motif_label"] = merged_df["t_bin_s"].map(bin_to_motif).fillna(-999).astype(int)
        n_valid_motif = (merged_df["motif_label"] > 0).sum()
        n_total_rows = len(merged_df)
        if config.debug:
            print(
                f"[Fig2B] Motif labels loaded: {n_valid_motif}/{n_total_rows} "
                f"valid rows ({100*n_valid_motif/n_total_rows:.1f}%)"
            )

    # Filter to SUA or MUA only if requested
    if config.sua_only and 'unit_kind' in merged_df.columns:
        n_units_before = merged_df['unit_index'].nunique()
        merged_df = merged_df[merged_df['unit_kind'] == 'SUA'].copy()
        n_units_after = merged_df['unit_index'].nunique()
        if config.debug:
            print(f"[Fig2B] SUA filter: {n_units_before} -> {n_units_after} units (excluded MUA)")
    elif config.mua_only and 'unit_kind' in merged_df.columns:
        n_units_before = merged_df['unit_index'].nunique()
        merged_df = merged_df[merged_df['unit_kind'] == 'MUA'].copy()
        n_units_after = merged_df['unit_index'].nunique()
        if config.debug:
            print(f"[Fig2B] MUA filter: {n_units_before} -> {n_units_after} units (excluded SUA)")

    # Rename columns to match expected schema
    merged_df = merged_df.rename(columns={
        'fr': 'firing_rate',
        'unit_index': 'unit_id',
    })

    # Get appropriate column lists based on 2D setting
    vis_cols_ref, kin_cols_ref = get_feature_cols(config.use_2d_only)

    if config.debug:
        n_units = merged_df['unit_id'].nunique() if 'unit_id' in merged_df.columns else '?'
        print(f"[Fig2B] Units: {n_units}, VIS cols: {len(vis_cols_ref)}, KIN cols: {len(kin_cols_ref)}")
        if config.use_2d_only:
            print("[Fig2B] Using 2D floor-plane features (excluding gaze_z)")

    # Get available feature columns (use 2D or full column lists)
    vis_cols_available = [c for c in vis_cols_ref if c in merged_df.columns]
    kin_cols_available = [c for c in kin_cols_ref if c in merged_df.columns]

    # For N-way partitioning, get BODY_KIN, HEAD_KIN, and optional POS columns
    body_kin_cols_available = [c for c in BODY_KIN_COLS if c in merged_df.columns]
    head_kin_cols_available = [c for c in HEAD_KIN_COLS if c in merged_df.columns]
    pos_cols_available = (
        [c for c in POS_COLS if c in merged_df.columns]
        if config.include_pos else []
    )

    if config.debug:
        print(f"[Fig2B] VIS cols: {vis_cols_available}")
        print(f"[Fig2B] BODY_KIN cols: {body_kin_cols_available}")
        print(f"[Fig2B] HEAD_KIN cols: {head_kin_cols_available}")
        if pos_cols_available:
            print(f"[Fig2B] POS cols: {pos_cols_available}")

    # Get unique units
    all_unit_ids = merged_df["unit_id"].unique()
    if unit_ids is not None:
        all_unit_ids = [u for u in all_unit_ids if u in unit_ids]

    if config.debug:
        print(f"[Fig2B] Processing {len(all_unit_ids)} units...")

    # CV configuration
    cv_cfg = CVConfig(
        folds=config.cv_folds,
        block_s=config.cv_block_s,
        gap_s=config.cv_gap_s,
    )

    # Pre-extract unit data to avoid repeated DataFrame filtering in parallel
    unit_data_dict = {
        uid: merged_df[merged_df["unit_id"] == uid].copy()
        for uid in all_unit_ids
    }

    # Create unit_id -> unit_bank mapping if bank info available
    unit_bank_map: Dict[int, Optional[str]] = {}
    if "unit_bank" in merged_df.columns:
        unit_bank_map = merged_df.groupby("unit_id")["unit_bank"].first().to_dict()

    # Determine verbosity for joblib
    verbose = 10 if config.debug else 0

    if config.n_jobs == 1:
        if config.debug:
            print("[Fig2B] Running in sequential mode (n_jobs=1)")
        results = []
        for i, unit_id in enumerate(all_unit_ids):
            if config.debug and i % 10 == 0:
                print(f"[Fig2B] Unit {i+1}/{len(all_unit_ids)}: {unit_id}")
            result = _process_single_unit(
                unit_id=unit_id,
                unit_data=unit_data_dict[unit_id],
                vis_cols=vis_cols_available,
                kin_cols=kin_cols_available,
                body_kin_cols=body_kin_cols_available,
                head_kin_cols=head_kin_cols_available,
                cv_cfg=cv_cfg,
                config=config,
                session_id=session_id,
                rng_seed=42 + i,
                unit_bank=unit_bank_map.get(unit_id),
                pos_cols=pos_cols_available if pos_cols_available else None,
            )
            if result is not None:
                results.append(result)
    else:
        if config.debug:
            print(f"[Fig2B] Running in parallel mode (n_jobs={config.n_jobs})")

        results = Parallel(n_jobs=config.n_jobs, prefer="processes", verbose=verbose)(
            delayed(_process_single_unit)(
                unit_id=unit_id,
                unit_data=unit_data_dict[unit_id],
                vis_cols=vis_cols_available,
                kin_cols=kin_cols_available,
                body_kin_cols=body_kin_cols_available,
                head_kin_cols=head_kin_cols_available,
                cv_cfg=cv_cfg,
                config=config,
                session_id=session_id,
                rng_seed=42 + i,
                unit_bank=unit_bank_map.get(unit_id),
                pos_cols=pos_cols_available if pos_cols_available else None,
            )
            for i, unit_id in enumerate(all_unit_ids)
        )
        results = [r for r in results if r is not None]

    # Extract per-bin residuals before building summary DataFrame
    residuals_list = []
    for r in results:
        resid = r.pop("_residuals", None)
        if resid is not None:
            residuals_list.append(pd.DataFrame(resid))

    results_df = pd.DataFrame(results)

    # Summary stats (dynamic group discovery)
    if config.debug and len(results_df) > 0:
        print(f"[Fig2B] Results: {len(results_df)} units")
        n_pass_threshold = results_df["r2_threshold_passed"].sum()
        n_total = len(results_df)
        print(f"[Fig2B] R2 > {config.r2_threshold}: {n_pass_threshold} ({100*n_pass_threshold/n_total:.1f}%)")

        # Discover groups from sig_* columns
        sig_cols = [c for c in results_df.columns if c.startswith("sig_") and c not in ("sig_kin", "sig_kin_fdr")]
        for sc in sorted(sig_cols):
            g_name = sc.replace("sig_", "").upper()
            n_sig = results_df[sc].sum()
            print(f"[Fig2B] Significant {g_name}: {n_sig} ({100*n_sig/n_total:.1f}%)")

        # Mean unique variance per group
        unique_cols = [c for c in results_df.columns if c.startswith("unique_") and c.endswith("_shuffle")]
        for uc in sorted(unique_cols):
            g_name = uc.replace("unique_", "").replace("_shuffle", "").upper()
            print(f"[Fig2B] Mean unique {g_name}: {results_df[uc].mean():.4f}")

    # Save results
    if save_results and len(results_df) > 0:
        os.makedirs(output_dir, exist_ok=True)
        encoding_dir = os.path.join(output_dir, "encoding")
        os.makedirs(encoding_dir, exist_ok=True)

        out_path = os.path.join(encoding_dir, "fig2b_unique_variance.parquet")
        results_df.to_parquet(out_path, index=False)

        # Save per-bin residuals for spatial analysis (Fig 2D)
        if residuals_list:
            residuals_df = pd.concat(residuals_list, ignore_index=True)
            resid_path = os.path.join(encoding_dir, "fig2b_residuals.parquet")
            residuals_df.to_parquet(resid_path, index=False)
            print(f"[Fig2B] Saved residuals: {resid_path} ({len(residuals_df)} rows, {residuals_df['unit_id'].nunique()} units)")

        if config.debug:
            print(f"[Fig2B] Saved: {out_path}")

        # Generate figures
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        from freemat.single_fm.viz.fig2b import (
            generate_fig2b_panel_3way,
            plot_fig2b_by_bank,
        )

        generate_fig2b_panel_3way(results_df, figures_dir)

        # Generate per-bank comparison if bank info available
        if "unit_bank" in results_df.columns:
            plot_fig2b_by_bank(results_df, figures_dir)

        if config.debug:
            print(f"[Fig2B] Saved figures to: {figures_dir}")

    return results_df


def _infer_subject(session_id: str) -> str:
    """Infer subject from session_id (hedy = M2, hooke = M1)."""
    if "M2" in session_id.lower():
        return "M2"
    return "M1"


def _apply_fdr_to_subset(
    df: pd.DataFrame,
    groups: List[str],
    fdr_alpha: float,
    effect_threshold: float,
    r2_threshold: float,
) -> pd.DataFrame:
    """Apply BH-FDR correction to a DataFrame subset."""
    for group in groups:
        df = apply_fdr_correction(
            df, p_col=f"p_{group}_shuffle", alpha=fdr_alpha
        )
        df[f"p_{group}_fdr"] = df[f"p_{group}_shuffle_fdr"]

    # Also apply FDR to legacy block/circular if present (diagnostic)
    for group in groups:
        if f"p_{group}_block" in df.columns:
            df = apply_fdr_correction(
                df, p_col=f"p_{group}_block", alpha=fdr_alpha
            )
        if f"p_{group}_circular" in df.columns:
            df = apply_fdr_correction(
                df, p_col=f"p_{group}_circular", alpha=fdr_alpha
            )

    # Update final significance flags: effect + p_fdr + r2_gate
    for group in groups:
        fdr_col = f"p_{group}_fdr"
        shuffle_col = f"unique_{group}_shuffle"
        if fdr_col in df.columns and shuffle_col in df.columns:
            p_ok = df[fdr_col] < fdr_alpha
            effect_ok = df[shuffle_col] >= effect_threshold
            r2_ok = df["full_r2"] > r2_threshold
            df[f"sig_{group}"] = p_ok & effect_ok & r2_ok

    return df


def aggregate_fig2b_results(
    session_results: List[pd.DataFrame],
    output_path: Optional[str] = None,
    apply_fdr: bool = True,
    fdr_alpha: float = 0.05,
    effect_threshold: float = 0.005,
    r2_threshold: float = 0.01,
) -> pd.DataFrame:
    """Aggregate Figure 2B results across sessions with optional FDR correction.

    Dynamically discovers feature groups from column names (works with any
    number of groups: 3-way, 4-way, or N-way). Applies BH-FDR correction
    per-subject to avoid penalizing one subject for another's weak encoding.

    Args:
        session_results: List of per-session result DataFrames.
        output_path: Optional path to save aggregated results.
        apply_fdr: Whether to apply BH-FDR correction (default: True).
        fdr_alpha: Significance level for FDR correction.
        effect_threshold: Minimum delta R2 for significance.
        r2_threshold: Minimum full R2 for unit inclusion.

    Returns:
        Aggregated DataFrame with FDR-corrected p-values if apply_fdr=True.
    """
    combined = pd.concat(session_results, ignore_index=True)

    if apply_fdr and len(combined) > 0:
        # Discover groups dynamically from p_{group}_shuffle columns
        shuffle_p_cols = [
            c for c in combined.columns
            if c.startswith("p_") and c.endswith("_shuffle") and "_fdr" not in c
        ]
        groups = [c.replace("p_", "").replace("_shuffle", "") for c in shuffle_p_cols]

        if groups:
            # Apply FDR per-subject to avoid cross-subject dilution
            combined["_subject"] = combined["session_id"].map(_infer_subject)
            subjects = combined["_subject"].unique()

            parts = []
            for subj in subjects:
                subj_df = combined[combined["_subject"] == subj].copy()
                subj_df = _apply_fdr_to_subset(
                    subj_df, groups, fdr_alpha, effect_threshold, r2_threshold
                )
                parts.append(subj_df)
            combined = pd.concat(parts, ignore_index=True)
            combined.drop(columns=["_subject"], inplace=True)

        elif "p_kin" in combined.columns or "p_vis" in combined.columns:
            # Legacy 2-way columns
            combined = apply_fdr_correction(combined, p_col="p_kin", alpha=fdr_alpha)
            combined = apply_fdr_correction(combined, p_col="p_vis", alpha=fdr_alpha)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_parquet(output_path, index=False)

    return combined


__all__ = [
    "Fig2BConfig",
    "build_lagged_features",
    "apply_fdr_correction",
    "run_fig2b_encoding",
    "aggregate_fig2b_results",
    "run_subsample_sensitivity",
]
