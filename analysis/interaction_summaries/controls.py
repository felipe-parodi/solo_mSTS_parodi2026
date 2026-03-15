"""Unified control covariates for Figure 4 analyses.

This module provides standardized control variables used across all Figure 4
analyses (ledger, alignment, surprise). Controls are essential to show that
neural effects are not reducible to movement or dyad geometry.

Control Covariate Groups:
    - Self kinematics (per monkey): speed, accel, yaw_rate, heading
    - Dyad geometry: distance, bearing, approach_velocity, relative_orientation

Usage:
    from analysis.interaction_summaries.controls import (
        build_control_covariates,
        residualize_on_controls,
    )

    # Build controls from pose data (or mock)
    controls = build_control_covariates(
        n_bins=7200,
        bin_size_sec=1.0,
        pose_data=None,  # Uses mock if None
    )

    # Residualize neural data on controls
    X_residual = residualize_on_controls(
        X=neural_rates,
        controls=controls,
        train_mask=train_indices,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class ControlCovariates:
    """Container for control covariate matrices.

    Attributes:
        self_kinematics_m1: Self kinematics for monkey 1, shape (n_bins, n_features)
        self_kinematics_m2: Self kinematics for monkey 2, shape (n_bins, n_features)
        dyad_geometry: Dyad geometry features, shape (n_bins, n_features)
        feature_names: Dict mapping feature group to list of names
        n_bins: Number of time bins
        bin_size_sec: Bin size in seconds
    """

    self_kinematics_m1: np.ndarray
    self_kinematics_m2: np.ndarray
    dyad_geometry: np.ndarray
    feature_names: Dict[str, List[str]]
    n_bins: int
    bin_size_sec: float

    @property
    def all_features_m1(self) -> np.ndarray:
        """Get all control features for monkey 1 (self kin + dyad geo)."""
        return np.hstack([self.self_kinematics_m1, self.dyad_geometry])

    @property
    def all_features_m2(self) -> np.ndarray:
        """Get all control features for monkey 2 (self kin + dyad geo)."""
        return np.hstack([self.self_kinematics_m2, self.dyad_geometry])

    @property
    def all_feature_names(self) -> List[str]:
        """Get all feature names in order."""
        return (
            self.feature_names.get("self_kinematics", [])
            + self.feature_names.get("dyad_geometry", [])
        )

    def get_for_subject(self, subject: str) -> np.ndarray:
        """Get control features for a specific subject.

        Args:
            subject: Subject identifier ('m1', 'm2', 'M1', 'M2')

        Returns:
            Control matrix of shape (n_bins, n_features)
        """
        subject = subject.lower()
        if subject in ("m1", "M1"):
            return self.all_features_m1
        elif subject in ("m2", "M2"):
            return self.all_features_m2
        else:
            raise ValueError(f"Unknown subject: {subject}")


def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Z-score each column of a matrix."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)  # Avoid division by zero
    return (X - mu) / sd


def _generate_mock_kinematics(
    n_bins: int,
    bin_size_sec: float,
    smoothing_sigma_sec: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Generate mock self-kinematics features.

    Returns:
        Tuple of (features array, feature names list)
    """
    rng = np.random.default_rng(seed)

    # Feature names
    names = ["speed", "accel", "yaw_rate", "heading_sin", "heading_cos"]

    # Generate correlated mock features
    features = []

    # Speed: always positive, AR(1) process
    speed = np.cumsum(rng.standard_normal(n_bins) * 0.1) + 2.0
    speed = np.clip(speed, 0.1, 10.0)
    features.append(speed)

    # Acceleration: derivative of speed + noise
    accel = np.diff(speed, prepend=speed[0]) / bin_size_sec + rng.standard_normal(n_bins) * 0.1
    features.append(accel)

    # Yaw rate: smooth random walk
    yaw_rate = np.cumsum(rng.standard_normal(n_bins) * 0.05)
    features.append(yaw_rate)

    # Heading: circular (sin/cos representation)
    heading = np.cumsum(yaw_rate) * 0.1
    features.append(np.sin(heading))
    features.append(np.cos(heading))

    X = np.column_stack(features)

    # Smooth
    if smoothing_sigma_sec > 0:
        sigma_bins = smoothing_sigma_sec / bin_size_sec
        X = gaussian_filter1d(X, sigma=max(1e-9, sigma_bins), axis=0)

    return _zscore_columns(X), names


def _generate_mock_dyad_geometry(
    n_bins: int,
    bin_size_sec: float,
    smoothing_sigma_sec: float = 0.5,
    seed: int = 123,
) -> Tuple[np.ndarray, List[str]]:
    """Generate mock dyad geometry features.

    Returns:
        Tuple of (features array, feature names list)
    """
    rng = np.random.default_rng(seed)

    names = ["distance", "bearing_sin", "bearing_cos", "approach_velocity", "relative_orientation"]

    features = []

    # Distance: bounded random walk (typically 0.5 - 5 meters)
    distance = np.cumsum(rng.standard_normal(n_bins) * 0.02) + 2.0
    distance = np.clip(distance, 0.3, 6.0)
    features.append(distance)

    # Bearing: circular (where is partner relative to self heading)
    bearing = rng.standard_normal(n_bins) * 0.5
    bearing = np.cumsum(bearing)
    features.append(np.sin(bearing))
    features.append(np.cos(bearing))

    # Approach velocity: negative distance derivative + noise
    approach_vel = -np.diff(distance, prepend=distance[0]) / bin_size_sec
    approach_vel += rng.standard_normal(n_bins) * 0.1
    features.append(approach_vel)

    # Relative orientation: how aligned are headings
    rel_orient = np.cos(rng.standard_normal(n_bins) * 0.3)
    rel_orient = np.cumsum(rel_orient * 0.01 + rng.standard_normal(n_bins) * 0.01)
    features.append(rel_orient)

    X = np.column_stack(features)

    # Smooth
    if smoothing_sigma_sec > 0:
        sigma_bins = smoothing_sigma_sec / bin_size_sec
        X = gaussian_filter1d(X, sigma=max(1e-9, sigma_bins), axis=0)

    return _zscore_columns(X), names


def build_control_covariates(
    n_bins: int,
    bin_size_sec: float,
    pose_data_m1: Optional[np.ndarray] = None,
    pose_data_m2: Optional[np.ndarray] = None,
    smoothing_sigma_sec: float = 0.5,
    use_mock: bool = True,
) -> ControlCovariates:
    """Build control covariate matrices from pose data.

    Args:
        n_bins: Number of time bins
        bin_size_sec: Bin size in seconds
        pose_data_m1: 3D pose data for monkey 1, shape (n_frames, n_joints, 3)
        pose_data_m2: 3D pose data for monkey 2, shape (n_frames, n_joints, 3)
        smoothing_sigma_sec: Gaussian smoothing sigma in seconds
        use_mock: If True (default) and pose data is None, generate mock features

    Returns:
        ControlCovariates object with all features

    Example:
        >>> controls = build_control_covariates(n_bins=7200, bin_size_sec=1.0)
        >>> X_ctrl = controls.get_for_subject('M1')  # shape (7200, n_features)
    """
    # For now, use mock data
    # TODO: Implement real pose feature extraction when pose data is available
    if pose_data_m1 is None or pose_data_m2 is None:
        if not use_mock:
            raise ValueError("Pose data not provided and use_mock=False")

        # Generate mock features
        self_kin_m1, kin_names = _generate_mock_kinematics(
            n_bins, bin_size_sec, smoothing_sigma_sec, seed=42
        )
        self_kin_m2, _ = _generate_mock_kinematics(
            n_bins, bin_size_sec, smoothing_sigma_sec, seed=43
        )
        dyad_geo, geo_names = _generate_mock_dyad_geometry(
            n_bins, bin_size_sec, smoothing_sigma_sec, seed=123
        )

        return ControlCovariates(
            self_kinematics_m1=self_kin_m1,
            self_kinematics_m2=self_kin_m2,
            dyad_geometry=dyad_geo,
            feature_names={
                "self_kinematics": kin_names,
                "dyad_geometry": geo_names,
            },
            n_bins=n_bins,
            bin_size_sec=bin_size_sec,
        )

    # TODO: Extract real features from pose data
    raise NotImplementedError("Real pose feature extraction not yet implemented")


def residualize_on_controls(
    X: np.ndarray,
    controls: ControlCovariates,
    subject: str,
    train_mask: Optional[np.ndarray] = None,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Residualize neural data by regressing out control covariates.

    Fits a ridge regression model on training data and returns residuals
    for all data. This is used to isolate neural variance unexplained by
    movement and dyad geometry.

    Args:
        X: Neural data, shape (n_bins, n_units)
        controls: ControlCovariates object
        subject: Subject identifier ('m1', 'm2', 'M1', 'M2')
        train_mask: Boolean mask for training samples (fit on True, predict all)
        ridge_alpha: Ridge regularization strength

    Returns:
        Residualized neural data, shape (n_bins, n_units)

    Example:
        >>> X_resid = residualize_on_controls(
        ...     X=neural_rates,
        ...     controls=controls,
        ...     subject='M1',
        ...     train_mask=train_indices,
        ... )
    """
    C = controls.get_for_subject(subject)

    if X.shape[0] != C.shape[0]:
        raise ValueError(f"Shape mismatch: X has {X.shape[0]} bins, controls have {C.shape[0]}")

    n_bins, n_units = X.shape
    n_features = C.shape[1]

    if train_mask is None:
        train_mask = np.ones(n_bins, dtype=bool)

    # Fit on training data
    C_train = C[train_mask]
    X_train = X[train_mask]

    # Ridge regression: (C'C + alpha*I)^-1 C' X
    CTC = C_train.T @ C_train
    regularizer = ridge_alpha * np.eye(n_features)
    W = np.linalg.solve(CTC + regularizer, C_train.T @ X_train)  # (n_features, n_units)

    # Predict and compute residuals for all data
    X_pred = C @ W
    X_resid = X - X_pred

    return X_resid


def partial_out_controls_fold_aware(
    X: np.ndarray,
    controls: ControlCovariates,
    subject: str,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Residualize neural data with fold-aware fitting to prevent leakage.

    For each fold, fits the control regression on training data only,
    then computes residuals for test data. This prevents information leakage
    when residualized data is used for downstream CV analysis.

    Args:
        X: Neural data, shape (n_bins, n_units)
        controls: ControlCovariates object
        subject: Subject identifier
        fold_indices: List of (train_idx, test_idx) tuples
        ridge_alpha: Ridge regularization strength

    Returns:
        Residualized neural data, shape (n_bins, n_units)
    """
    C = controls.get_for_subject(subject)
    n_bins, n_units = X.shape
    n_features = C.shape[1]

    X_resid = np.full_like(X, np.nan)

    for train_idx, test_idx in fold_indices:
        # Fit on training fold
        C_train = C[train_idx]
        X_train = X[train_idx]

        CTC = C_train.T @ C_train
        regularizer = ridge_alpha * np.eye(n_features)
        W = np.linalg.solve(CTC + regularizer, C_train.T @ X_train)

        # Compute residuals for test fold
        X_pred_test = C[test_idx] @ W
        X_resid[test_idx] = X[test_idx] - X_pred_test

    return X_resid
