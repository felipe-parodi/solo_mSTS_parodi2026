"""Control feature extraction for GLM encoding models.

This module provides functions for extracting or generating control variables
(nuisance regressors) such as velocity, audio features, head direction, and
time drift. These help remove confounds from neural encoding analysis.

Control Variables:
    - velocity_3d: Center-of-mass velocity (behavior-modulated)
    - audio_rms: Audio root-mean-square energy
    - audio_spectral_centroid: Frequency center of mass
    - audio_spectral_rolloff: High-frequency cutoff
    - head_direction_sin/cos: Head direction (circular encoding)
    - time_drift: Linear time trend for drift correction
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_control_features(
    n_bins: int,
    bin_duration_sec: float,
    behavioral_states: Optional[np.ndarray] = None,
    behavioral_labels: Optional[List[str]] = None,
    mock_data: bool = False,
    include_velocity: bool = True,
    include_audio: bool = True,
    include_head_direction: bool = True,
    include_time_drift: bool = True,
    session_metadata: Optional[Dict[str, Any]] = None,
    real_control_matrix: Optional[np.ndarray] = None,
    real_control_feature_names: Optional[List[str]] = None,
    append_time_drift_with_real: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Extract or generate control features for GLM encoding analysis.

    Creates control variables (velocity, audio features, head direction, time drift)
    that serve as nuisance regressors in the GLM. When mock_data=True, generates
    synthetic but realistic control features to enable pipeline testing.

    Args:
        n_bins: Number of time bins in the neural data.
        bin_duration_sec: Duration of each bin in seconds.
        behavioral_states: Optional binary matrix of behavioral states
            (n_bins, n_behaviors) used to modulate mock features.
        behavioral_labels: Optional list of behavior names corresponding to columns
            in behavioral_states.
        mock_data: If True, generates mock control features. If False, requires
            real_control_matrix to be provided.
        include_velocity: If True, includes 3D center-of-mass velocity.
        include_audio: If True, includes audio features (RMS, spectral centroid,
            spectral rolloff).
        include_head_direction: If True, includes head direction angle.
        include_time_drift: If True, includes linear time trend for drift.
        session_metadata: Optional metadata dictionary for the session.
        real_control_matrix: Pre-computed control features matrix.
        real_control_feature_names: Names for columns in real_control_matrix.
        append_time_drift_with_real: If True, appends time drift to real controls.

    Returns:
        Tuple containing:
            - control_features: Array of shape (n_bins, n_features) with control
              variables. Features are continuous, not binary like behavioral states.
            - feature_names: List of feature names for each column.
            - metadata: Dictionary with information about data sources
              (e.g., {'velocity': 'mock', 'audio': 'mock'}).

    Example:
        >>> # Generate mock control features for testing
        >>> controls, names, meta = extract_control_features(
        >>>     n_bins=7200, bin_duration_sec=1.0, mock_data=True
        >>> )
        >>> print(f"Generated {len(names)} control features: {names}")

        >>> # With behavioral modulation for more realistic mock data
        >>> controls, names, meta = extract_control_features(
        >>>     n_bins=7200, bin_duration_sec=1.0,
        >>>     behavioral_states=binned_behaviors,
        >>>     behavioral_labels=behavior_names,
        >>>     mock_data=True
        >>> )

    Note:
        When mock_data=True, a warning is logged to ensure users are aware
        that synthetic data is being used. All results cached with mock data
        should include this metadata to prevent accidental mixing with real data.
    """
    # If real controls are provided, validate and optionally append time drift
    if real_control_matrix is not None:
        mat = np.asarray(real_control_matrix)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        if mat.shape[0] != n_bins:
            if mat.shape[0] > n_bins:
                mat = mat[:n_bins, :]
            else:
                pad_len = n_bins - mat.shape[0]
                pad = np.tile(mat[-1, :], (pad_len, 1))
                mat = np.vstack([mat, pad])
        names = real_control_feature_names or [f"control_{i}" for i in range(mat.shape[1])]
        meta = {"controls": "real_precomputed", "n_features": str(mat.shape[1])}
        # Append linear time drift as a real regressor if requested
        if include_time_drift and append_time_drift_with_real:
            time_drift = np.linspace(0, 1, n_bins).reshape(-1, 1)
            mat = np.hstack([mat, time_drift])
            names = list(names) + ["time_drift"]
            meta["time_drift"] = "real_linear_appended"
        return mat, names, meta

    if not mock_data:
        raise NotImplementedError(
            "Real control feature loading requires 'real_control_matrix'. "
            "Pass precomputed controls aligned to neural bins or use mock_data=True."
        )

    # Log warning about using mock data
    if mock_data:
        logger.warning(
            "USING MOCK CONTROL FEATURES. Set mock_data=False and provide "
            "real data paths when available."
        )

    control_features = []
    feature_names = []
    metadata = {}

    # Import scipy for smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        logger.warning("scipy not available, mock features will not be smoothed")

    # Helper to get behavioral modulation
    def get_behavior_modulation(behavior_name: str) -> float:
        """Get activity level for a given behavior to modulate features."""
        if behavioral_states is not None and behavioral_labels is not None:
            if behavior_name in behavioral_labels:
                idx = behavioral_labels.index(behavior_name)
                return behavioral_states[:, idx].mean()
        return 0.5

    # 1. Generate mock 3D velocity (behavior-modulated)
    if include_velocity:
        logger.info("Generating mock 3D velocity features...")

        rng = np.random.default_rng(42)
        base_velocity = rng.standard_normal(n_bins) * 0.5 + 1.0

        # Modulate based on behavioral states if available
        if behavioral_states is not None:
            velocity_modulation = np.ones(n_bins)

            if behavioral_labels is not None:
                for behavior_idx, behavior_name in enumerate(behavioral_labels):
                    if "travel" in behavior_name.lower() or "chase" in behavior_name.lower():
                        velocity_modulation += behavioral_states[:, behavior_idx] * 2.0
                    elif "rest" in behavior_name.lower() or "groom" in behavior_name.lower():
                        velocity_modulation -= behavioral_states[:, behavior_idx] * 0.5

            base_velocity *= np.clip(velocity_modulation, 0.1, 3.0)

        # Smooth the velocity for realism
        if HAS_SCIPY:
            velocity_smooth = gaussian_filter1d(base_velocity, sigma=5)
        else:
            velocity_smooth = base_velocity
        velocity_smooth = np.clip(velocity_smooth, 0, None)

        control_features.append(velocity_smooth.reshape(-1, 1))
        feature_names.append("velocity_3d")
        metadata["velocity"] = "mock_behavior_modulated" if behavioral_states is not None else "mock_random"

    # 2. Generate mock audio features
    if include_audio:
        logger.info("Generating mock audio features...")

        rng = np.random.default_rng(43)

        # Audio RMS energy
        audio_rms = rng.standard_normal(n_bins) * 0.1 + 0.5

        # Spectral centroid
        spectral_centroid = rng.standard_normal(n_bins) * 0.2 + 1.0

        # Spectral rolloff
        spectral_rolloff = rng.standard_normal(n_bins) * 0.15 + 0.8

        # Slight modulation for vocalizations during social behaviors
        if behavioral_states is not None and behavioral_labels is not None:
            for behavior_idx, behavior_name in enumerate(behavioral_labels):
                if any(social in behavior_name.lower() for social in ["groom", "play", "threat"]):
                    audio_rms += behavioral_states[:, behavior_idx] * 0.2

        # Smooth audio features
        if HAS_SCIPY:
            audio_rms = gaussian_filter1d(audio_rms, sigma=3)
            spectral_centroid = gaussian_filter1d(spectral_centroid, sigma=3)
            spectral_rolloff = gaussian_filter1d(spectral_rolloff, sigma=3)

        control_features.extend([
            np.asarray(audio_rms).reshape(-1, 1),
            np.asarray(spectral_centroid).reshape(-1, 1),
            np.asarray(spectral_rolloff).reshape(-1, 1),
        ])
        feature_names.extend(["audio_rms", "audio_spectral_centroid", "audio_spectral_rolloff"])
        metadata["audio"] = "mock_low_variance"

    # 3. Generate mock head direction
    if include_head_direction:
        logger.info("Generating mock head direction features...")

        rng = np.random.default_rng(44)
        angular_velocity = rng.standard_normal(n_bins) * 0.1

        # Modulate turning based on behavior
        if behavioral_states is not None and behavioral_labels is not None:
            for behavior_idx, behavior_name in enumerate(behavioral_labels):
                if "forage" in behavior_name.lower() or "explore" in behavior_name.lower():
                    angular_velocity += behavioral_states[:, behavior_idx] * rng.standard_normal(n_bins) * 0.2

        # Integrate to get head direction
        if HAS_SCIPY:
            angular_velocity_smooth = gaussian_filter1d(angular_velocity, sigma=10)
        else:
            angular_velocity_smooth = angular_velocity
        head_direction = np.cumsum(angular_velocity_smooth) % (2 * np.pi)

        # Convert to sine and cosine for circular continuity
        head_dir_sin = np.sin(head_direction)
        head_dir_cos = np.cos(head_direction)

        control_features.extend([
            head_dir_sin.reshape(-1, 1),
            head_dir_cos.reshape(-1, 1),
        ])
        feature_names.extend(["head_direction_sin", "head_direction_cos"])
        metadata["head_direction"] = "mock_smooth_trajectory"

    # 4. Add time drift (always real, not mocked)
    if include_time_drift:
        logger.info("Adding linear time drift feature...")

        time_drift = np.linspace(0, 1, n_bins)
        control_features.append(time_drift.reshape(-1, 1))
        feature_names.append("time_drift")
        metadata["time_drift"] = "real_linear"

    # Concatenate all features
    if control_features:
        control_matrix = np.hstack(control_features)
    else:
        control_matrix = np.zeros((n_bins, 0))

    logger.info(
        f"Extracted {len(feature_names)} control features: {feature_names}. "
        f"Data sources: {metadata}"
    )

    return control_matrix, feature_names, metadata


__all__ = ["extract_control_features"]
