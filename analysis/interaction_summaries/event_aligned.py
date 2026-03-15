"""Event-aligned analysis for interaction summaries.

Aligns summary variables (alignment, surprise, ledger) to grooming onset
and stable grooming epochs to demonstrate the "grooming is privileged" claim:
- Alignment rises around grooming onset
- Surprise drops during stable grooming

Usage:
    from analysis.interaction_summaries.event_aligned import (
        compute_event_aligned_traces,
        find_grooming_onsets,
    )

    onsets = find_grooming_onsets(
        m1_labels=hooke_labels,
        m2_labels=hedy_labels,
        min_duration_sec=10.0,
        bin_size_sec=1.0,
    )

    result = compute_event_aligned_traces(
        alignment=alignment_series,
        surprise=surprise_series,
        event_indices=onsets,
        pre_bins=100,
        post_bins=200,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EventAlignedResult:
    """Result of event-aligned analysis.

    Attributes:
        time_axis: Time axis relative to event (in bins or seconds)
        traces: Dict mapping variable name to mean trace
        traces_sem: Dict mapping variable name to SEM
        traces_ci_low: Dict mapping variable name to lower CI
        traces_ci_high: Dict mapping variable name to upper CI
        n_events: Number of events included
        event_indices: Original event indices
    """

    time_axis: np.ndarray
    traces: Dict[str, np.ndarray]
    traces_sem: Dict[str, np.ndarray]
    traces_ci_low: Dict[str, np.ndarray]
    traces_ci_high: Dict[str, np.ndarray]
    n_events: int
    event_indices: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "time_axis": self.time_axis.tolist(),
            "traces": {k: v.tolist() for k, v in self.traces.items()},
            "traces_sem": {k: v.tolist() for k, v in self.traces_sem.items()},
            "n_events": self.n_events,
        }


def find_grooming_onsets(
    m1_labels: List[Optional[str]],
    m2_labels: List[Optional[str]],
    min_duration_sec: float = 10.0,
    bin_size_sec: float = 1.0,
    require_mutual: bool = True,
) -> np.ndarray:
    """Find grooming onset indices.

    Args:
        m1_labels: Behavior labels for monkey 1
        m2_labels: Behavior labels for monkey 2
        min_duration_sec: Minimum grooming bout duration
        bin_size_sec: Bin size in seconds
        require_mutual: If True, require both monkeys grooming

    Returns:
        Array of onset bin indices
    """
    n_bins = min(len(m1_labels), len(m2_labels))
    min_duration_bins = int(round(min_duration_sec / bin_size_sec))

    def _is_grooming(label: Optional[str]) -> bool:
        if label is None:
            return False
        s = str(label).lower()
        return "groom" in s

    # Find grooming bins
    if require_mutual:
        groom_mask = np.array([
            _is_grooming(m1_labels[i]) and _is_grooming(m2_labels[i])
            for i in range(n_bins)
        ], dtype=bool)
    else:
        groom_mask = np.array([
            _is_grooming(m1_labels[i]) or _is_grooming(m2_labels[i])
            for i in range(n_bins)
        ], dtype=bool)

    # Find onset transitions (0 -> 1)
    transitions = np.diff(groom_mask.astype(int))
    potential_onsets = np.where(transitions == 1)[0] + 1

    # Filter by minimum duration
    valid_onsets = []
    for onset in potential_onsets:
        # Find end of this grooming bout
        bout_end = onset
        while bout_end < n_bins and groom_mask[bout_end]:
            bout_end += 1

        duration_bins = bout_end - onset
        if duration_bins >= min_duration_bins:
            valid_onsets.append(onset)

    return np.array(valid_onsets, dtype=int)


def find_grooming_stable_epochs(
    m1_labels: List[Optional[str]],
    m2_labels: List[Optional[str]],
    min_duration_sec: float = 10.0,
    bin_size_sec: float = 1.0,
    stable_offset_sec: float = 5.0,
) -> List[Tuple[int, int]]:
    """Find stable grooming epochs (excluding onset/offset transitions).

    Args:
        m1_labels: Behavior labels for monkey 1
        m2_labels: Behavior labels for monkey 2
        min_duration_sec: Minimum bout duration to consider
        bin_size_sec: Bin size in seconds
        stable_offset_sec: Seconds after onset and before offset to consider "stable"

    Returns:
        List of (start_bin, end_bin) tuples for stable epochs
    """
    n_bins = min(len(m1_labels), len(m2_labels))
    min_duration_bins = int(round(min_duration_sec / bin_size_sec))
    offset_bins = int(round(stable_offset_sec / bin_size_sec))

    def _is_grooming(label: Optional[str]) -> bool:
        if label is None:
            return False
        return "groom" in str(label).lower()

    # Find mutual grooming bouts
    groom_mask = np.array([
        _is_grooming(m1_labels[i]) and _is_grooming(m2_labels[i])
        for i in range(n_bins)
    ], dtype=bool)

    # Find bout boundaries
    transitions = np.diff(np.concatenate([[False], groom_mask, [False]]).astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    stable_epochs = []
    for start, end in zip(starts, ends):
        duration = end - start
        if duration >= min_duration_bins:
            # Stable epoch = [start + offset, end - offset]
            stable_start = start + offset_bins
            stable_end = end - offset_bins

            if stable_end > stable_start:
                stable_epochs.append((stable_start, stable_end))

    return stable_epochs


def compute_event_aligned_traces(
    event_indices: np.ndarray,
    pre_bins: int,
    post_bins: int,
    alignment: Optional[np.ndarray] = None,
    surprise: Optional[np.ndarray] = None,
    ledger: Optional[np.ndarray] = None,
    other_traces: Optional[Dict[str, np.ndarray]] = None,
    bin_size_sec: float = 1.0,
    ci_percentile: float = 95.0,
) -> EventAlignedResult:
    """Compute event-aligned traces for multiple variables.

    Args:
        event_indices: Array of event onset indices
        pre_bins: Number of bins before event
        post_bins: Number of bins after event
        alignment: Alignment time series (optional)
        surprise: Surprise time series (optional)
        ledger: Ledger time series (optional)
        other_traces: Dict of additional traces to align
        bin_size_sec: Bin size for time axis
        ci_percentile: Percentile for confidence intervals

    Returns:
        EventAlignedResult with aligned traces
    """
    # Collect all traces
    traces_to_align: Dict[str, np.ndarray] = {}
    if alignment is not None:
        traces_to_align["alignment"] = alignment
    if surprise is not None:
        traces_to_align["surprise"] = surprise
    if ledger is not None:
        traces_to_align["ledger"] = ledger
    if other_traces is not None:
        traces_to_align.update(other_traces)

    if not traces_to_align:
        raise ValueError("At least one trace must be provided")

    # Get trace length
    n_bins = len(next(iter(traces_to_align.values())))
    total_window = pre_bins + post_bins

    # Time axis
    time_axis = np.arange(-pre_bins, post_bins) * bin_size_sec

    # Align each trace
    aligned: Dict[str, List[np.ndarray]] = {k: [] for k in traces_to_align}

    valid_events = []
    for event_idx in event_indices:
        start = event_idx - pre_bins
        end = event_idx + post_bins

        if start < 0 or end > n_bins:
            continue

        valid_events.append(event_idx)

        for name, trace in traces_to_align.items():
            segment = trace[start:end]
            aligned[name].append(segment)

    n_events = len(valid_events)

    # Compute statistics
    traces_mean: Dict[str, np.ndarray] = {}
    traces_sem: Dict[str, np.ndarray] = {}
    traces_ci_low: Dict[str, np.ndarray] = {}
    traces_ci_high: Dict[str, np.ndarray] = {}

    for name, segments in aligned.items():
        if not segments:
            traces_mean[name] = np.full(total_window, np.nan)
            traces_sem[name] = np.full(total_window, np.nan)
            traces_ci_low[name] = np.full(total_window, np.nan)
            traces_ci_high[name] = np.full(total_window, np.nan)
            continue

        stacked = np.vstack(segments)  # (n_events, total_window)

        # Mean and SEM
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        sem = std / np.sqrt(np.sum(~np.isnan(stacked), axis=0) + 1e-12)

        # Percentile CIs
        ci_low = np.nanpercentile(stacked, (100 - ci_percentile) / 2, axis=0)
        ci_high = np.nanpercentile(stacked, 100 - (100 - ci_percentile) / 2, axis=0)

        traces_mean[name] = mean
        traces_sem[name] = sem
        traces_ci_low[name] = ci_low
        traces_ci_high[name] = ci_high

    return EventAlignedResult(
        time_axis=time_axis,
        traces=traces_mean,
        traces_sem=traces_sem,
        traces_ci_low=traces_ci_low,
        traces_ci_high=traces_ci_high,
        n_events=n_events,
        event_indices=np.array(valid_events),
    )


def compute_stable_epoch_comparison(
    stable_epochs: List[Tuple[int, int]],
    non_grooming_bins: np.ndarray,
    alignment: np.ndarray,
    surprise: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare alignment and surprise between stable grooming and non-grooming.

    Args:
        stable_epochs: List of (start, end) tuples for stable grooming
        non_grooming_bins: Array of non-grooming bin indices
        alignment: Alignment time series
        surprise: Surprise time series

    Returns:
        Dict with comparison statistics
    """
    # Collect values during stable grooming
    groom_alignment = []
    groom_surprise = []

    for start, end in stable_epochs:
        for idx in range(start, end):
            if not np.isnan(alignment[idx]):
                groom_alignment.append(alignment[idx])
            if not np.isnan(surprise[idx]):
                groom_surprise.append(surprise[idx])

    # Collect values during non-grooming
    non_groom_alignment = []
    non_groom_surprise = []

    for idx in non_grooming_bins:
        if idx < len(alignment) and not np.isnan(alignment[idx]):
            non_groom_alignment.append(alignment[idx])
        if idx < len(surprise) and not np.isnan(surprise[idx]):
            non_groom_surprise.append(surprise[idx])

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": np.nan, "std": np.nan, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    return {
        "alignment_grooming": _stats(groom_alignment),
        "alignment_non_grooming": _stats(non_groom_alignment),
        "surprise_grooming": _stats(groom_surprise),
        "surprise_non_grooming": _stats(non_groom_surprise),
        "alignment_difference": float(
            np.mean(groom_alignment) - np.mean(non_groom_alignment)
            if groom_alignment and non_groom_alignment else np.nan
        ),
        "surprise_difference": float(
            np.mean(groom_surprise) - np.mean(non_groom_surprise)
            if groom_surprise and non_groom_surprise else np.nan
        ),
    }
