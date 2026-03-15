"""Behavioral feature preprocessing and design matrix generation.

This module provides functions for converting continuous behavioral events
into discrete binary matrices aligned with neural data bins, and for
creating temporally lagged design matrices for GLM encoding.

Key Functions:
    - get_cleaned_behavioral_events: Clean and standardize behavioral data
    - determine_regressor_behavior_labels: Filter behaviors by duration
    - create_binned_behavioral_regressors: Convert events to binary matrix
    - generate_lagged_design_matrix: Create lagged feature matrix
    - preprocess_behavioral_data_for_encoding: Full preprocessing pipeline
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Handle pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except (ImportError, ValueError):
    HAS_PANDAS = False
    pd = None

# Import behavior cleaning utility
try:
    from ..behavior_utils import clean_behavior
except ImportError:
    def clean_behavior(x):
        """Fallback when behavior_utils unavailable."""
        return [str(x).strip().lower()]


def get_cleaned_behavioral_events(
    behavior_df: "pd.DataFrame",
    seconds_col: str,
    behavior_col: str,
    excluded_behaviors: List[str],
) -> "pd.DataFrame":
    """Clean raw behavioral event DataFrame.

    Performs the following:
    - Selects relevant columns
    - Filters out nulls
    - Applies behavior cleaning/standardization
    - Filters out excluded behaviors
    - Renames columns to 'time_sec' and 'cleaned_behavior'
    - Sorts by time

    Args:
        behavior_df: DataFrame with raw behavioral events.
        seconds_col: Column name for timestamps in seconds.
        behavior_col: Column name for behavior labels.
        excluded_behaviors: List of behaviors to exclude.

    Returns:
        Cleaned DataFrame with standardized 'time_sec' and 'cleaned_behavior' columns.
    """
    if behavior_df is None or behavior_df.empty:
        logger.warning("Input behavior_df is None or empty.")
        return pd.DataFrame({"time_sec": [], "cleaned_behavior": []})

    if seconds_col not in behavior_df.columns or behavior_col not in behavior_df.columns:
        logger.error(f"Required columns '{seconds_col}' or '{behavior_col}' not in DataFrame.")
        raise ValueError(f"Missing columns in behavior_df: {seconds_col} or {behavior_col}")

    # Select relevant columns and filter initial nulls
    processed_df = behavior_df[[seconds_col, behavior_col]].copy()
    processed_df.columns = ["time_sec", "raw_behavior"]
    processed_df = processed_df.dropna(subset=["time_sec", "raw_behavior"])

    if processed_df.empty:
        logger.info("Behavior DataFrame is empty after filtering initial nulls.")
        return pd.DataFrame({"time_sec": [], "cleaned_behavior": []})

    # Apply clean_behavior and handle its list output (taking the first element)
    cleaned_behaviors_series = processed_df["raw_behavior"].apply(
        lambda x: clean_behavior(str(x))[0] if clean_behavior(str(x)) else None
    )

    processed_df["cleaned_behavior"] = cleaned_behaviors_series

    # Filter out None post-cleaning and excluded behaviors
    processed_df = processed_df.dropna(subset=["cleaned_behavior"])
    processed_df = processed_df[~processed_df["cleaned_behavior"].isin(excluded_behaviors)]

    final_df = (
        processed_df[["time_sec", "cleaned_behavior"]]
        .sort_values("time_sec")
        .reset_index(drop=True)
    )

    logger.info(
        f"Processed behavioral events. Original: {len(behavior_df)}, Cleaned & Filtered: {len(final_df)}"
    )
    return final_df


def determine_regressor_behavior_labels(
    cleaned_behavior_events_df: "pd.DataFrame",
    neural_bin_duration_sec: float,
    total_neural_duration_sec: float,
    min_total_duration_per_behavior_sec: float,
) -> List[str]:
    """Determine which behaviors meet minimum duration criteria for inclusion.

    Filters behavioral events to include only those with sufficient total
    duration across the recording session. This prevents overfitting to
    rarely occurring behaviors.

    Args:
        cleaned_behavior_events_df: DataFrame with cleaned behavioral events,
            containing 'time_sec' and 'cleaned_behavior' columns.
        neural_bin_duration_sec: Duration of each neural bin in seconds.
        total_neural_duration_sec: Total duration covered by neural data.
        min_total_duration_per_behavior_sec: Minimum total duration (seconds)
            a behavior must be present to be included as regressor.

    Returns:
        Sorted list of behavior labels meeting the duration criterion.

    Note:
        Duration is calculated as time until next event or end of recording.
    """
    if cleaned_behavior_events_df.empty:
        return []

    behavior_durations = {}

    for i, event in cleaned_behavior_events_df.iterrows():
        behavior = event["cleaned_behavior"]
        start_time = event["time_sec"]

        if i + 1 < len(cleaned_behavior_events_df):
            end_time = cleaned_behavior_events_df["time_sec"].iloc[i + 1]
        else:
            end_time = total_neural_duration_sec

        duration = max(0, end_time - start_time)

        if duration > 1e-6:
            behavior_durations[behavior] = behavior_durations.get(behavior, 0) + duration

    valid_behaviors = [
        b
        for b, total_dur in behavior_durations.items()
        if total_dur >= min_total_duration_per_behavior_sec
    ]

    logger.info(
        f"Determined regressor labels. Found {len(valid_behaviors)} behaviors "
        f"meeting min duration {min_total_duration_per_behavior_sec}s."
    )
    return sorted(list(set(valid_behaviors)))


def create_binned_behavioral_regressors(
    cleaned_behavior_events_df: "pd.DataFrame",
    list_of_regressor_behavior_labels: List[str],
    neural_bin_duration_sec: float,
    total_neural_bins: int,
    reference_behavior: Optional[str] = None,
    coding: str = "reference",
) -> np.ndarray:
    """Create binary matrix indicating behavior presence in each time bin.

    Converts continuous behavioral events into a discrete binary matrix
    aligned with neural data bins. Each column represents one behavior,
    each row represents one time bin.

    Args:
        cleaned_behavior_events_df: DataFrame with 'time_sec' and
            'cleaned_behavior' columns.
        list_of_regressor_behavior_labels: List of behaviors to include
            as columns in the output matrix.
        neural_bin_duration_sec: Duration of each time bin in seconds.
        total_neural_bins: Total number of time bins to create.
        reference_behavior: Behavior to use as reference (excluded from matrix).
        coding: Coding scheme - 'reference' (default) or 'effect'.

    Returns:
        Binary matrix of shape (total_neural_bins, num_behaviors) where
        element [i,j] = 1 if behavior j is active in bin i, else 0.
        Data type is int8 for memory efficiency.

    Example:
        >>> regressors = create_binned_behavioral_regressors(
        >>>     behavior_df, ['rest', 'groom', 'feed'],
        >>>     bin_duration_sec=1.0, total_neural_bins=7200
        >>> )
        >>> print(f"Active behaviors per bin: {regressors.sum(axis=1)}")
    """
    # Normalize inputs
    labels_all = list(dict.fromkeys(list_of_regressor_behavior_labels))
    if coding not in ("reference", "effect"):
        coding = "reference"

    # Determine output columns depending on coding
    if coding == "reference" and reference_behavior and reference_behavior in labels_all:
        labels = [b for b in labels_all if b != reference_behavior]
    elif coding == "effect":
        if reference_behavior and reference_behavior in labels_all:
            labels = [b for b in labels_all if b != reference_behavior]
        else:
            labels = labels_all[:-1] if len(labels_all) > 0 else []
    else:
        labels = labels_all

    if not labels:
        logger.info("No regressor behavior labels provided. Returning empty array.")
        return np.zeros((total_neural_bins, 0), dtype=np.int8)
    if total_neural_bins == 0:
        logger.info("Total_neural_bins is 0. Returning empty array for regressors.")
        return np.zeros((0, len(labels)), dtype=np.int8)

    binned_regressors = np.zeros((total_neural_bins, len(labels)), dtype=np.int8)
    behavior_to_idx_map = {behavior: i for i, behavior in enumerate(labels)}

    total_recording_duration_sec = total_neural_bins * neural_bin_duration_sec

    for i, event in cleaned_behavior_events_df.iterrows():
        behavior = event["cleaned_behavior"]
        start_sec = event["time_sec"]

        # Determine end time
        if i + 1 < len(cleaned_behavior_events_df):
            end_sec = cleaned_behavior_events_df["time_sec"].iloc[i + 1]
        else:
            end_sec = total_recording_duration_sec

        # Clip event times to the total recording duration
        start_sec_clipped = max(0, min(start_sec, total_recording_duration_sec))
        end_sec_clipped = max(0, min(end_sec, total_recording_duration_sec))

        if end_sec_clipped <= start_sec_clipped:
            continue

        start_bin_idx = int(np.floor(start_sec_clipped / neural_bin_duration_sec))
        end_bin_idx = int(np.ceil(end_sec_clipped / neural_bin_duration_sec))

        # Ensure indices are within bounds
        start_bin_idx = max(0, min(start_bin_idx, total_neural_bins - 1))
        end_bin_idx = max(0, min(end_bin_idx, total_neural_bins))

        if start_bin_idx < end_bin_idx:
            if coding == "reference":
                if behavior in behavior_to_idx_map:
                    col = behavior_to_idx_map[behavior]
                    binned_regressors[start_bin_idx:end_bin_idx, col] = 1
            else:  # effect coding
                if reference_behavior and behavior == reference_behavior:
                    binned_regressors[start_bin_idx:end_bin_idx, :] = -1
                elif behavior in behavior_to_idx_map:
                    col = behavior_to_idx_map[behavior]
                    binned_regressors[start_bin_idx:end_bin_idx, col] = 1

    logger.info(f"Created binned behavioral regressors. Shape: {binned_regressors.shape}; coding={coding}")
    return binned_regressors


def generate_lagged_design_matrix(
    binned_behavioral_regressors_self: np.ndarray,
    binned_behavioral_regressors_partner: Optional[np.ndarray],
    lags_in_bins: List[int],
) -> Tuple[np.ndarray, slice]:
    """Generate design matrix with temporally lagged behavioral features.

    Creates a feature matrix where each time point includes behavioral
    states from multiple time lags, enabling causal and anticausal
    modeling of neural responses.

    Args:
        binned_behavioral_regressors_self: Binary matrix of self behaviors,
            shape (n_bins, n_self_behaviors).
        binned_behavioral_regressors_partner: Optional binary matrix of
            partner behaviors, shape (n_bins, n_partner_behaviors).
        lags_in_bins: List of lag values in bins. Negative = past,
            positive = future. E.g., [-2, -1, 0, 1] includes 2 bins
            in past, current bin, and 1 bin in future.

    Returns:
        Tuple containing:
            - Design matrix X of shape (n_valid_bins, n_features) where
              n_features = n_behaviors * n_lags
            - Slice object indicating valid indices for target variable y
              after accounting for lag constraints

    Note:
        The returned slice ensures all lagged features are within bounds.
        For lags [-2, 0, 2] with 100 bins, valid y range is [2:98].

    Example:
        >>> X, y_slice = generate_lagged_design_matrix(
        >>>     self_behaviors, partner_behaviors, [-1, 0, 1]
        >>> )
        >>> y_valid = neural_activity[y_slice]  # Aligned target
    """
    if binned_behavioral_regressors_self.size == 0:
        logger.warning("Self behavioral regressors are empty. Cannot generate design matrix.")
        return np.array([]).reshape(0, 0), slice(0, 0)

    num_total_bins, num_behaviors_self = binned_behavioral_regressors_self.shape
    num_behaviors_partner = 0
    if binned_behavioral_regressors_partner is not None and binned_behavioral_regressors_partner.size > 0:
        if binned_behavioral_regressors_partner.shape[0] != num_total_bins:
            raise ValueError("Partner regressors must have same number of bins as self regressors.")
        num_behaviors_partner = binned_behavioral_regressors_partner.shape[1]

    min_lag = min(lags_in_bins)
    max_lag = max(lags_in_bins)

    # Determine the range of y_bins for which all lags are available
    start_y_idx = -min_lag if min_lag < 0 else 0
    end_y_idx = num_total_bins - max_lag if max_lag > 0 else num_total_bins

    num_effective_y_bins = end_y_idx - start_y_idx

    if num_effective_y_bins <= 0:
        logger.warning(
            f"Not enough bins ({num_total_bins}) to create lagged features with lags {lags_in_bins}. "
            f"Effective y bins: {num_effective_y_bins}"
        )
        return np.array([]).reshape(0, 0), slice(start_y_idx, end_y_idx)

    num_features_per_lag_set = num_behaviors_self + num_behaviors_partner
    total_features = num_features_per_lag_set * len(lags_in_bins)

    X_design_matrix = np.zeros((num_effective_y_bins, total_features), dtype=np.int8)

    current_feature_col_idx = 0
    for lag in lags_in_bins:
        source_start_idx = start_y_idx + lag
        source_end_idx = end_y_idx + lag

        # Features from self
        X_design_matrix[:, current_feature_col_idx:current_feature_col_idx + num_behaviors_self] = (
            binned_behavioral_regressors_self[source_start_idx:source_end_idx, :]
        )
        current_feature_col_idx += num_behaviors_self

        # Features from partner
        if binned_behavioral_regressors_partner is not None and num_behaviors_partner > 0:
            X_design_matrix[:, current_feature_col_idx:current_feature_col_idx + num_behaviors_partner] = (
                binned_behavioral_regressors_partner[source_start_idx:source_end_idx, :]
            )
            current_feature_col_idx += num_behaviors_partner

    valid_bin_indices_for_y = slice(start_y_idx, end_y_idx)
    logger.info(
        f"Generated lagged design matrix. Shape: {X_design_matrix.shape}. Valid y slice: {valid_bin_indices_for_y}"
    )
    return X_design_matrix, valid_bin_indices_for_y


def preprocess_behavioral_data_for_encoding(
    behavior_events_df: "pd.DataFrame",
    self_beh_key_col: str,
    self_sec_col: str,
    partner_beh_key_col: str,
    partner_sec_col: str,
    excluded_behaviors: List[str],
    neural_bin_duration_sec: float,
    total_neural_duration_sec: float,
    total_neural_bins: int,
    min_total_duration_per_behavior_sec: float,
    reference_behavior: Optional[str] = None,
) -> Dict[str, Any]:
    """Preprocess behavioral data for GLM encoding analysis.

    Performs complete preprocessing pipeline for behavioral data, including
    cleaning, filtering, and binning for both self and partner behaviors.
    Results can be cached to avoid redundant processing across units.

    Args:
        behavior_events_df: DataFrame with behavioral events and timestamps.
        self_beh_key_col: Column name for self behavior labels.
        self_sec_col: Column name for self behavior timestamps (seconds).
        partner_beh_key_col: Column name for partner behavior labels.
        partner_sec_col: Column name for partner behavior timestamps (seconds).
        excluded_behaviors: List of behavior labels to exclude from analysis.
        neural_bin_duration_sec: Duration of each neural bin in seconds.
        total_neural_duration_sec: Total recording duration in seconds.
        total_neural_bins: Total number of neural time bins.
        min_total_duration_per_behavior_sec: Minimum total duration for a
            behavior to be included as a regressor.
        reference_behavior: Optional reference behavior for coding scheme.

    Returns:
        Dictionary containing:
            - 'cleaned_events_self': Cleaned self behavioral events DataFrame
            - 'regressor_labels_self': List of self behavior labels meeting criteria
            - 'binned_regressors_self': Binary matrix of self behaviors (n_bins, n_behaviors)
            - 'cleaned_events_partner': Cleaned partner behavioral events DataFrame
            - 'regressor_labels_partner': List of partner behavior labels meeting criteria
            - 'binned_regressors_partner': Binary matrix of partner behaviors

    Example:
        >>> preprocessed = preprocess_behavioral_data_for_encoding(
        >>>     behavior_df, 'Hke_Behavior', 'Hke_Time',
        >>>     'Hdy_Behavior', 'Hdy_Time', ['undefined'],
        >>>     1.0, 7200.0, 7200, 10.0
        >>> )
        >>> print(f"Found {len(preprocessed['regressor_labels_self'])} self behaviors")
    """
    preprocessed = {}

    # Process self behavior
    cleaned_events_self = get_cleaned_behavioral_events(
        behavior_events_df,
        self_sec_col,
        self_beh_key_col,
        excluded_behaviors,
    )
    regressor_labels_self = determine_regressor_behavior_labels(
        cleaned_events_self,
        neural_bin_duration_sec,
        total_neural_duration_sec,
        min_total_duration_per_behavior_sec,
    )
    if reference_behavior and reference_behavior in regressor_labels_self:
        regressor_labels_self = [b for b in regressor_labels_self if b != reference_behavior]
    binned_regressors_self = create_binned_behavioral_regressors(
        cleaned_events_self,
        regressor_labels_self,
        neural_bin_duration_sec,
        total_neural_bins,
        reference_behavior=reference_behavior,
        coding="reference",
    )

    preprocessed["cleaned_events_self"] = cleaned_events_self
    preprocessed["regressor_labels_self"] = regressor_labels_self
    preprocessed["binned_regressors_self"] = binned_regressors_self

    # Process partner behavior
    cleaned_events_partner = get_cleaned_behavioral_events(
        behavior_events_df,
        partner_sec_col,
        partner_beh_key_col,
        excluded_behaviors,
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
        reference_behavior=None,
        coding="reference",
    )

    preprocessed["cleaned_events_partner"] = cleaned_events_partner
    preprocessed["regressor_labels_partner"] = regressor_labels_partner
    preprocessed["binned_regressors_partner"] = binned_regressors_partner

    logger.info(
        f"Preprocessed behavioral data: {len(regressor_labels_self)} self behaviors, "
        f"{len(regressor_labels_partner)} partner behaviors"
    )

    return preprocessed


def build_feature_names(
    regressor_labels_self: List[str],
    regressor_labels_partner: List[str],
    lags_in_bins: List[int],
    control_feature_names: Optional[List[str]] = None,
    history_names: Optional[List[str]] = None,
) -> List[str]:
    """Build list of feature names for design matrix interpretation.

    Args:
        regressor_labels_self: Self behavior labels.
        regressor_labels_partner: Partner behavior labels.
        lags_in_bins: Lag values used in design matrix.
        control_feature_names: Optional control feature names.
        history_names: Optional spike history feature names.

    Returns:
        List of feature names in order matching design matrix columns.
    """
    feature_names = []
    for lag in lags_in_bins:
        for r_label in regressor_labels_self:
            feature_names.append(f"self_{r_label}_lag{lag}")
        if regressor_labels_partner:
            for r_label_p in regressor_labels_partner:
                feature_names.append(f"partner_{r_label_p}_lag{lag}")
    if control_feature_names:
        feature_names.extend(control_feature_names)
    if history_names:
        feature_names.extend(history_names)
    return feature_names


# Legacy aliases for backward compatibility
_determine_regressor_behavior_labels = determine_regressor_behavior_labels


__all__ = [
    "get_cleaned_behavioral_events",
    "determine_regressor_behavior_labels",
    "create_binned_behavioral_regressors",
    "generate_lagged_design_matrix",
    "preprocess_behavioral_data_for_encoding",
    "build_feature_names",
    # Legacy aliases
    "_determine_regressor_behavior_labels",
]
