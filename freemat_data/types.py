from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Pose3D:
    """Canonical 3D pose container for a single animal.

    Attributes:
        X: (T, J, 3) float32 array of 3D joint coordinates.
        C: Optional (T, J) float32 confidences per joint.
        fids: (T,) int64 frame indices (monotonic).
        fps: Frames per second used in triangulation/export.
        joint_names: Optional list of J joint names.
        up_axis: 'Y' or 'Z' indicating which axis is vertical.
        meta: Optional metadata parsed from NPZ.
        path: Optional source path for bookkeeping.
    """

    X: np.ndarray
    C: Optional[np.ndarray]
    fids: np.ndarray
    fps: float
    joint_names: Optional[List[str]]
    up_axis: str
    meta: Optional[Dict[str, Any]]
    path: Optional[str] = None

    @property
    def T(self) -> int:
        return int(self.X.shape[0])

    @property
    def J(self) -> int:
        return int(self.X.shape[1])


@dataclass
class SpikesEvents:
    """Event-coded spikes for a session.

    Attributes:
        t: (E,) float32 spike timestamps in seconds.
        unit_index: (E,) int32 unit indices aligned with unit_ids.
        unit_ids: (U,) array-like of unit identifiers.
        session_id: Session identifier string.
        meta: Optional metadata.
    """

    t: np.ndarray
    unit_index: np.ndarray
    unit_ids: np.ndarray
    session_id: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SessionRef:
    """Normalized view of a session entry from the registry.

    Attributes:
        session_id: Canonical session ID (e.g., 'session_001').
        session_type: 'social' or 'asocial'.
        monkeys: List of subject IDs present.
        data_files: Mapping of modality keys to absolute/normalized paths.
        metadata: Free-form metadata.
    """

    session_id: str
    session_type: str
    monkeys: List[str]
    data_files: Dict[str, Optional[str]]
    metadata: Dict[str, Any]


@dataclass
class LFPTimeSeries:
    """Continuous LFP recording for a session.

    Args:
        t: 1D array of time stamps in seconds (video- or logger-aligned; specify in meta).
        x: 2D array of LFP values with shape (T, C).
        fs: Sampling frequency (Hz). If t is provided, fs is optional.
        channel_ids: Optional list of channel identifiers (length C).
        meta: Optional metadata (e.g., reference, units, alignment base).
    """

    t: Optional[np.ndarray]
    x: np.ndarray
    fs: Optional[float]
    channel_ids: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class StateSeries:
    """Behavioral state annotations.

    Supports either per-sample labels (t, labels) or segment form
    (segments start/end in seconds with labels). Only one form needs to
    be populated for a given instance.

    Args:
        t: Optional per-sample times (s).
        labels: Optional per-sample labels (same length as t).
        segments: Optional array of shape (K, 2) with start/end seconds.
        segment_labels: Optional list of K labels for segments.
        label_set: Optional list of all possible labels.
        meta: Optional metadata.
    """

    t: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    segments: Optional[np.ndarray] = None
    segment_labels: Optional[List[str]] = None
    label_set: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class MotifSeries:
    """Motif assignments over time.

    Args:
        t: Optional per-sample times (s).
        motif_id: Optional per-sample motif IDs (ints, same length as t).
        segments: Optional (K, 2) start/end seconds.
        segment_ids: Optional list of K motif IDs.
        names: Optional mapping from motif_id to string names.
        meta: Optional metadata.
    """

    t: Optional[np.ndarray] = None
    motif_id: Optional[np.ndarray] = None
    segments: Optional[np.ndarray] = None
    segment_ids: Optional[List[int]] = None
    names: Optional[Dict[int, str]] = None
    meta: Optional[Dict[str, Any]] = None
