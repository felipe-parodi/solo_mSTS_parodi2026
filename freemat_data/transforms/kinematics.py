"""Kinematic feature extraction utilities for Figure 3.

Provides allocentric and egocentric kinematic features aligned to neural bins.
Supports mock generation for development (no pose available) and a pluggable
path for real pose NPZ/arrays to be added later.

Features (per session):
- Allocentric (partner-focused):
  - partner_speed: instantaneous CoM speed
  - partner_pose_expansion: proxy for pose expansion (convex hull volume or variance proxy)
  - partner_head_dir_{sin,cos}: head direction as sin/cos
- Egocentric (relative to subject):
  - relative_distance: distance between subject and partner CoMs
  - relative_bearing_deg: angle between subject head direction and to-partner vector
  - relative_velocity_towardself: projection of partner velocity onto to-subject unit vector

All outputs are returned as an (n_bins × n_features) matrix plus feature names.
Values are z-scored per feature by default (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class KinematicFeatures:
    X: np.ndarray
    names: List[str]
    meta: Dict[str, Any]
    raw_by_name: Dict[str, np.ndarray]


def _zscore_cols(mat: np.ndarray) -> np.ndarray:
    if mat is None or mat.size == 0:
        return mat
    m = np.nanmean(mat, axis=0, keepdims=True)
    s = np.nanstd(mat, axis=0, keepdims=True)
    s[s <= 1e-12] = 1.0
    return (mat - m) / s


def _smooth(series: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins and sigma_bins > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(series.astype(float), sigma=max(1e-9, float(sigma_bins)))
        except Exception:
            pass
    return series


def _make_mock_series(n: int, rng: np.random.RandomState, smooth_sigma_bins: float = 2.0) -> np.ndarray:
    x = rng.randn(n).astype(float)
    if smooth_sigma_bins > 0:
        x = _smooth(x, smooth_sigma_bins)
    return x


def extract_kinematics_for_session(
    n_bins: int,
    bin_size_sec: float,
    use_mock: bool = True,
    smoothing_sigma_sec: float = 0.1,
    zscore: bool = True,
    fov_deg: float = 60.0,
) -> KinematicFeatures:
    """Return kinematic features aligned to neural bins.

    Currently uses mock generation when use_mock=True. Real pose integration
    can replace the mock generation path by computing the same features.
    """
    names: List[str] = []
    cols_raw: List[np.ndarray] = []
    meta: Dict[str, Any] = {
        "source": "mock" if use_mock else "real",
        "bin_size_sec": float(bin_size_sec),
        "smoothing_sigma_sec": float(smoothing_sigma_sec),
        "fov_deg": float(fov_deg),
    }
    sigma_bins = float(smoothing_sigma_sec) / float(bin_size_sec) if smoothing_sigma_sec else 0.0

    rng = np.random.RandomState(42)

    # Self (subject) mock signals (controls)
    self_speed = np.abs(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins))
    self_head_angle = np.mod(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins) * 180.0, 360.0) - 180.0
    self_head_dir_sin = np.sin(np.deg2rad(self_head_angle))
    self_head_dir_cos = np.cos(np.deg2rad(self_head_angle))

    names += [
        "self_speed",
        "self_head_dir_sin",
        "self_head_dir_cos",
    ]
    cols_raw += [self_speed, self_head_dir_sin, self_head_dir_cos]

    # Allocentric (partner-focused) mock signals
    partner_speed = np.abs(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins))
    partner_pose_expansion = np.abs(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins))
    partner_head_angle = np.mod(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins) * 180.0, 360.0) - 180.0
    partner_head_dir_sin = np.sin(np.deg2rad(partner_head_angle))
    partner_head_dir_cos = np.cos(np.deg2rad(partner_head_angle))

    names += [
        "partner_speed",
        "partner_pose_expansion",
        "partner_head_dir_sin",
        "partner_head_dir_cos",
    ]
    cols_raw += [partner_speed, partner_pose_expansion, partner_head_dir_sin, partner_head_dir_cos]

    # Egocentric (subject-relative) mock signals
    relative_distance = np.abs(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins)) + 0.5
    relative_bearing_deg = np.mod(_make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins) * 90.0, 360.0)
    relative_velocity_towardself = _make_mock_series(n_bins, rng, smooth_sigma_bins=sigma_bins)

    names += [
        "relative_distance",
        "relative_bearing_deg",
        "relative_velocity_towardself",
    ]
    cols_raw += [relative_distance, relative_bearing_deg, relative_velocity_towardself]

    X_raw = np.column_stack(cols_raw).astype(float)
    X = _zscore_cols(X_raw) if zscore else X_raw.copy()

    meta["feature_names"] = names
    meta["n_bins"] = int(n_bins)
    raw_by_name = {n: X_raw[:, i].copy() for i, n in enumerate(names)}
    return KinematicFeatures(X=X, names=names, meta=meta, raw_by_name=raw_by_name)
