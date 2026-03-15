"""Main encoding pipeline runner for single-unit GLM analysis.

This module provides the main entry point for running the complete
GLM encoding pipeline on a single neural unit, orchestrating all
the preprocessing, fitting, and analysis steps.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .features import (
    get_cleaned_behavioral_events,
    determine_regressor_behavior_labels,
    create_binned_behavioral_regressors,
    generate_lagged_design_matrix,
    build_feature_names,
)
from .controls import extract_control_features
from .glm import fit_poisson_glm_elastic_net, fit_poisson_glm_cv_detailed
from .gam import fit_poisson_gam_cv
from .cv import compute_behavior_change_indices, segment_shuffle
from .poisson import overdispersion_test, calculate_p_value_from_shuffles
from .importance import (
    compute_delta_d2_per_behavior,
    compute_delta_d2_per_lag,
    compute_per_behavior_d2,
    build_feature_to_behavior_map,
    build_feature_to_lag_map,
    zscore_coefficients,
)

logger = logging.getLogger(__name__)

# Handle pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def run_encoding_for_single_unit(
    single_unit_binned_activity: np.ndarray,
    behavior_events_self_df: "pd.DataFrame",
    behavior_key_col_self: str,
    seconds_col_self: str,
    neural_bin_duration_sec: float,
    lags_in_bins: List[int],
    glm_params: Dict[str, Any],
    cv_n_splits: int,
    excluded_behaviors_for_regressors: List[str],
    min_total_duration_per_behavior_sec: float,
    behavior_events_partner_df: Optional["pd.DataFrame"] = None,
    behavior_key_col_partner: Optional[str] = None,
    seconds_col_partner: Optional[str] = None,
    shuffle_target_y: bool = False,
    random_state_shuffle: Optional[int] = None,
    random_state_cv: Optional[int] = None,
    preprocessed_behavior_data: Optional[Dict[str, Any]] = None,
    include_controls: bool = False,
    mock_data: bool = False,
    control_features: Optional[Dict[str, bool]] = None,
    session_metadata: Optional[Dict[str, Any]] = None,
    z_score_within_neuron: bool = False,
    zscore_behavior_only: bool = False,
    real_control_matrix: Optional[np.ndarray] = None,
    real_control_feature_names: Optional[List[str]] = None,
    reference_behavior: Optional[str] = None,
    append_time_drift_with_real: bool = True,
    include_spike_history: bool = False,
    spike_history_lags: Optional[List[int]] = None,
    n_segment_shuffles: int = 0,
    segment_shuffle_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Run complete GLM encoding pipeline for a single neural unit.

    Orchestrates the full encoding workflow: behavioral preprocessing,
    design matrix construction, model fitting, and importance analysis.
    Supports both self-only and self+partner behavioral encoding, with
    optional control variables for removing nuisance effects.

    Args:
        single_unit_binned_activity: Binned spike counts for one unit,
            shape (n_bins,). Values should be non-negative integers.
        behavior_events_self_df: DataFrame with self behavioral events.
        behavior_key_col_self: Column name for self behavior labels.
        seconds_col_self: Column name for self timestamps (seconds).
        neural_bin_duration_sec: Duration of neural bins in seconds.
        lags_in_bins: List of temporal lags to include. E.g., [-2,-1,0,1,2]
            for ±2 bin window around current time.
        glm_params: Dictionary with GLM parameters. Must contain 'alpha'
            key for regularization strength.
        cv_n_splits: Number of cross-validation folds (minimum 2).
        excluded_behaviors_for_regressors: Behaviors to exclude from model.
        min_total_duration_per_behavior_sec: Minimum duration threshold.
        behavior_events_partner_df: Optional partner behavioral events.
        behavior_key_col_partner: Column for partner behavior labels.
        seconds_col_partner: Column for partner timestamps.
        shuffle_target_y: If True, shuffle neural data for null model.
        random_state_shuffle: Random seed for shuffling.
        random_state_cv: Random seed for cross-validation.
        preprocessed_behavior_data: Optional preprocessed behavioral data
            to avoid redundant computation across units.
        include_controls: If True, includes control variables as nuisance regressors.
        mock_data: If True, uses mock control features for testing.
        control_features: Optional dictionary specifying which control features
            to include.
        session_metadata: Optional metadata about the recording session.
        z_score_within_neuron: If True, z-scores the GLM coefficients.
        zscore_behavior_only: If True, z-score using only behavior features.
        real_control_matrix: Pre-computed control features.
        real_control_feature_names: Names for real control features.
        reference_behavior: Reference behavior for coding scheme.
        append_time_drift_with_real: Append time drift to real controls.
        include_spike_history: If True, include spike history terms.
        spike_history_lags: Lags for spike history (positive integers).
        n_segment_shuffles: Number of segment shuffles for null model.
        segment_shuffle_bins: Segment size for shuffling.

    Returns:
        Dictionary containing model results, coefficients, predictions,
        and importance metrics.
    """
    # Initialize results
    results = {
        "d2_score": np.nan,
        "coefficients": np.array([]),
        "intercept": np.nan,
        "num_features": 0,
        "num_effective_samples": 0,
        "y_pred": np.array([]),
        "y_true": np.array([]),
        "per_behavior_d2": {},
        "delta_d2_per_behavior": {},
        "delta_d2_per_lag": {},
    }

    if single_unit_binned_activity is None or single_unit_binned_activity.size == 0:
        logger.warning("Single unit activity is None or empty.")
        return results

    total_neural_bins = single_unit_binned_activity.shape[0]
    total_neural_duration_sec = total_neural_bins * neural_bin_duration_sec

    # --- 1. Process behavioral data ---
    if preprocessed_behavior_data and all(
        k in preprocessed_behavior_data
        for k in ["cleaned_events_self", "regressor_labels_self", "binned_regressors_self"]
    ):
        cleaned_events_self = preprocessed_behavior_data["cleaned_events_self"]
        regressor_labels_self = preprocessed_behavior_data["regressor_labels_self"]
        binned_regressors_self = preprocessed_behavior_data["binned_regressors_self"]
    else:
        cleaned_events_self = get_cleaned_behavioral_events(
            behavior_events_self_df,
            seconds_col_self,
            behavior_key_col_self,
            excluded_behaviors_for_regressors,
        )
        regressor_labels_self = determine_regressor_behavior_labels(
            cleaned_events_self,
            neural_bin_duration_sec,
            total_neural_duration_sec,
            min_total_duration_per_behavior_sec,
        )
        coding_type = str(glm_params.get("coding", "reference")) if isinstance(glm_params, dict) else "reference"
        binned_regressors_self = create_binned_behavioral_regressors(
            cleaned_events_self,
            regressor_labels_self,
            neural_bin_duration_sec,
            total_neural_bins,
            reference_behavior=reference_behavior,
            coding=coding_type,
        )

    if binned_regressors_self.shape[1] == 0:
        logger.warning("SELF regressors are empty. Cannot proceed.")
        return results

    # Process partner behavior if provided
    binned_regressors_partner = None
    regressor_labels_partner = []
    if behavior_events_partner_df is not None and behavior_key_col_partner and seconds_col_partner:
        if preprocessed_behavior_data and all(
            k in preprocessed_behavior_data
            for k in ["cleaned_events_partner", "regressor_labels_partner", "binned_regressors_partner"]
        ):
            regressor_labels_partner = preprocessed_behavior_data["regressor_labels_partner"]
            binned_regressors_partner = preprocessed_behavior_data["binned_regressors_partner"]
        else:
            cleaned_events_partner = get_cleaned_behavioral_events(
                behavior_events_partner_df,
                seconds_col_partner,
                behavior_key_col_partner,
                excluded_behaviors_for_regressors,
            )
            regressor_labels_partner = determine_regressor_behavior_labels(
                cleaned_events_partner,
                neural_bin_duration_sec,
                total_neural_duration_sec,
                min_total_duration_per_behavior_sec,
            )
            if reference_behavior and reference_behavior in regressor_labels_partner:
                regressor_labels_partner = [b for b in regressor_labels_partner if b != reference_behavior]
            binned_regressors_partner = create_binned_behavioral_regressors(
                cleaned_events_partner,
                regressor_labels_partner,
                neural_bin_duration_sec,
                total_neural_bins,
                reference_behavior=reference_behavior if reference_behavior in regressor_labels_partner else None,
                coding=coding_type,
            )

    # --- 2. Extract control features if requested ---
    control_matrix = None
    control_feature_names = []
    if include_controls:
        if control_features is None:
            control_features = {"velocity": True, "audio": True, "head_direction": True, "time_drift": True}
        control_matrix, control_feature_names, control_metadata = extract_control_features(
            n_bins=total_neural_bins,
            bin_duration_sec=neural_bin_duration_sec,
            behavioral_states=binned_regressors_self,
            behavioral_labels=regressor_labels_self,
            mock_data=mock_data,
            include_velocity=control_features.get("velocity", True),
            include_audio=control_features.get("audio", True),
            include_head_direction=control_features.get("head_direction", True),
            include_time_drift=control_features.get("time_drift", True),
            session_metadata=session_metadata,
            real_control_matrix=real_control_matrix,
            real_control_feature_names=real_control_feature_names,
            append_time_drift_with_real=append_time_drift_with_real,
        )
        results["control_metadata"] = control_metadata
        results["n_control_features"] = len(control_feature_names)
        results["control_feature_names"] = control_feature_names

    # --- 3. Generate lagged design matrix ---
    X_design, y_slice = generate_lagged_design_matrix(
        binned_regressors_self,
        binned_regressors_partner,
        lags_in_bins,
    )

    if X_design.size == 0:
        logger.warning("Lagged design matrix is empty.")
        return results

    # Add control features to design matrix
    if include_controls and control_matrix is not None:
        control_matrix_aligned = control_matrix[y_slice]
        X_design = np.hstack([X_design, control_matrix_aligned])

    # Spike history terms
    effective_start = y_slice.start or 0
    effective_stop = y_slice.stop if y_slice.stop is not None else X_design.shape[0]
    history_names = []
    if include_spike_history:
        lags_list = [int(lag) for lag in (spike_history_lags or [1]) if int(lag) > 0]
        if lags_list:
            max_h = max(lags_list)
            if X_design.shape[0] > max_h:
                X_design = X_design[max_h:, :]
                effective_start = (y_slice.start or 0) + max_h
                history_cols = []
                y_full = single_unit_binned_activity
                for h in lags_list:
                    col = y_full[effective_start - h : effective_stop - h]
                    history_cols.append(col.reshape(-1, 1))
                    history_names.append(f"spike_history_lag{h}")
                if history_cols:
                    hist_mat = np.hstack(history_cols)
                    X_design = np.hstack([X_design, hist_mat])

    # --- 4. Prepare target variable ---
    y_target = single_unit_binned_activity[slice(effective_start, y_slice.stop)]

    if y_target.size == 0 or X_design.shape[0] != y_target.shape[0]:
        logger.warning(f"Target y mismatched with X. y: {y_target.shape}, X: {X_design.shape}")
        return results

    results["num_features"] = X_design.shape[1]
    results["num_effective_samples"] = X_design.shape[0]
    results["y_true"] = y_target

    # Shuffle if needed
    if shuffle_target_y:
        rng = np.random.default_rng(seed=random_state_shuffle)
        y_for_fit = rng.permutation(y_target)
    else:
        y_for_fit = y_target

    # --- 5. Fit model ---
    alpha_grid = glm_params.get("alpha_grid") if isinstance(glm_params, dict) else None
    cv_strategy = glm_params.get("cv_strategy", "temporal_kfold") if isinstance(glm_params, dict) else "temporal_kfold"
    cv_score_metric = glm_params.get("cv_score_metric", "d2") if isinstance(glm_params, dict) else "d2"
    model_type = glm_params.get("model_type", "poisson_glm") if isinstance(glm_params, dict) else "poisson_glm"
    embargo_bins = int(glm_params.get("embargo_bins", 0)) if isinstance(glm_params, dict) else 0

    behavior_change_idx = None
    if cv_strategy == "temporal_kfold_behavior_boundaries":
        behavior_change_idx = compute_behavior_change_indices(
            binned_regressors_self, binned_regressors_partner, slice(effective_start, y_slice.stop)
        )

    if model_type == "poisson_gam":
        Z_controls = control_matrix[y_slice] if include_controls and control_matrix is not None else None
        gam_df = int(glm_params.get("gam_df", 6))
        gam_degree = int(glm_params.get("gam_degree", 3))
        d2, coef, intercept, y_pred = fit_poisson_gam_cv(
            X_design, Z_controls, y_for_fit, cv_n_splits,
            cv_strategy=cv_strategy, behavior_change_indices=behavior_change_idx,
            gam_df=gam_df, gam_degree=gam_degree
        )
        results["model_type"] = "poisson_gam"
    else:
        d2, coef, intercept, y_pred = fit_poisson_glm_elastic_net(
            X_design, y_for_fit, cv_n_splits, glm_params["alpha"],
            random_state_cv=random_state_cv,
            alpha_grid=alpha_grid,
            cv_strategy=cv_strategy,
            behavior_change_indices=behavior_change_idx,
            cv_score_metric=cv_score_metric,
            embargo_bins=embargo_bins,
        )
        results["model_type"] = "poisson_glm"

    results["d2_score"] = d2
    results["coefficients"] = coef
    results["intercept"] = intercept
    results["y_pred"] = y_pred
    results["cv_strategy"] = cv_strategy

    # Overdispersion test
    n_params = X_design.shape[1] + 1
    od_results = overdispersion_test(y_for_fit, y_pred, n_params)
    results["overdispersion_phi"] = od_results["phi"]
    results["overdispersed"] = od_results["overdispersed"]

    # Feature names
    feature_names = build_feature_names(
        regressor_labels_self, regressor_labels_partner, lags_in_bins,
        control_feature_names if include_controls else None,
        history_names if include_spike_history else None
    )
    results["feature_names"] = feature_names

    # Z-score coefficients
    if z_score_within_neuron and len(coef) > 0:
        n_self = len(regressor_labels_self)
        n_partner = len(regressor_labels_partner)
        n_lags = len(lags_in_bins)
        n_behavior_features = (n_self + n_partner) * n_lags
        zscore_result = zscore_coefficients(coef, n_behavior_features, zscore_behavior_only)
        results["z_scored_coefficients"] = zscore_result["z_scored"]
        results["coef_mean"] = zscore_result["mean"]
        results["coef_std"] = zscore_result["std"]
        results["zscore_behavior_only"] = zscore_result["zscore_behavior_only"]

    # --- 6. Importance analysis ---
    if not shuffle_target_y and X_design.shape[1] > 1:
        feature_idx_to_behavior = build_feature_to_behavior_map(
            regressor_labels_self, regressor_labels_partner, lags_in_bins
        )
        results["delta_d2_per_behavior"] = compute_delta_d2_per_behavior(
            X_design, y_for_fit, feature_idx_to_behavior, d2,
            cv_n_splits, glm_params["alpha"], random_state_cv, alpha_grid,
            cv_strategy, cv_score_metric, embargo_bins
        )

        if len(lags_in_bins) > 1:
            feature_idx_to_lag = build_feature_to_lag_map(
                regressor_labels_self, regressor_labels_partner, lags_in_bins
            )
            results["delta_d2_per_lag"] = compute_delta_d2_per_lag(
                X_design, y_for_fit, feature_idx_to_lag, d2,
                cv_n_splits, glm_params["alpha"], random_state_cv, alpha_grid,
                cv_strategy, cv_score_metric, embargo_bins
            )

    # Per-behavior D²
    if not shuffle_target_y and y_target.size > 0 and y_pred.size > 0:
        y_target_np = np.asarray(y_target)
        y_pred_np = np.asarray(y_pred)
        binned_self_sliced = binned_regressors_self[slice(effective_start, y_slice.stop)]
        results["per_behavior_d2"] = compute_per_behavior_d2(
            y_target_np, y_pred_np, binned_self_sliced, regressor_labels_self, "self_"
        )
        if binned_regressors_partner is not None and len(regressor_labels_partner) > 0:
            binned_partner_sliced = binned_regressors_partner[slice(effective_start, y_slice.stop)]
            partner_d2 = compute_per_behavior_d2(
                y_target_np, y_pred_np, binned_partner_sliced, regressor_labels_partner, "partner_"
            )
            results["per_behavior_d2"].update(partner_d2)

    # --- 7. Segment-shuffle null model ---
    if not shuffle_target_y and n_segment_shuffles > 0:
        rng = np.random.default_rng(seed=random_state_shuffle)
        if segment_shuffle_bins is None:
            if neural_bin_duration_sec > 0:
                segment_len_bins = max(1, int(round(10.0 / neural_bin_duration_sec)))
            else:
                segment_len_bins = 10
        else:
            segment_len_bins = max(1, int(segment_shuffle_bins))

        null_scores = []
        for _ in range(n_segment_shuffles):
            y_shuf = segment_shuffle(y_target, segment_len_bins, rng)
            d2_null, _, _, _ = fit_poisson_glm_elastic_net(
                X_design, y_shuf, cv_n_splits, glm_params["alpha"],
                random_state_cv=random_state_cv, alpha_grid=alpha_grid, cv_strategy=cv_strategy
            )
            null_scores.append(d2_null)

        results["null_d2_segment_shuffle"] = null_scores
        null_arr = np.array(null_scores)
        d2_val = d2 if np.isfinite(d2) else -np.inf
        if len(null_arr) > 0 and not np.all(np.isnan(null_arr)):
            p_val = calculate_p_value_from_shuffles(d2_val, null_arr)
        else:
            p_val = np.nan
        results["p_value_segment_shuffle"] = p_val

    logger.info(f"Encoding complete. D²: {d2:.4f}. Shuffled: {shuffle_target_y}")
    return results


__all__ = ["run_encoding_for_single_unit"]
