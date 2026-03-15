"""Syllable bout segmentation and vocabulary management.

This module handles:
- Segmenting continuous frame-wise syllable labels into discrete bouts
- Filtering bouts by minimum duration
- Detecting transition/onset times
- Remapping syllables to a top-N vocabulary
- Computing descriptive statistics (durations, transition rates, transition matrix)

Core Algorithms:

    Bout Segmentation:
        Input: frame-wise labels [s₁, s₁, s₁, s₂, s₂, s₃, ...]
        Output: bouts [(id=s₁, start, end), (id=s₂, start, end), ...]

        A bout is a contiguous run of identical syllable IDs.
        Filter: duration(bout) ≥ min_duration_sec (default 400ms)

    Transition Detection:
        onset[t] = 𝟙[syllable[t] ≠ syllable[t-1]]

        An onset occurs at frame t when the syllable changes from t-1 to t.

    Transition Rate:
        rate = n_transitions / (T_total / 60)  [transitions per minute]

    Transition Matrix:
        T[i,j] = P(next = j | current = i) = count(i→j) / Σ_k count(i→k)

The minimum bout duration (default 400ms) filters out single-frame noise
and ensures meaningful behavioral units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import Counter


@dataclass
class SyllableBout:
    """A single syllable bout (contiguous run of same syllable).

    Attributes:
        syllable_id: Integer ID of the syllable.
        start_idx: Start frame index (inclusive).
        end_idx: End frame index (exclusive).
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        duration_sec: Duration in seconds.
    """

    syllable_id: int
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    duration_sec: float


def segment_bouts(
    motif_ids: np.ndarray,
    times: Optional[np.ndarray] = None,
    min_duration_sec: float = 0.4,
    fps: float = 30.0,
) -> List[SyllableBout]:
    """Segment continuous frame-wise motif labels into discrete bouts.

    Identifies contiguous runs of the same syllable ID and filters by
    minimum duration.

    Args:
        motif_ids: 1D array of integer syllable IDs per frame.
        times: Optional 1D array of frame times in seconds. If None,
            times are computed from fps.
        min_duration_sec: Minimum bout duration in seconds (default 0.4).
            Bouts shorter than this are excluded.
        fps: Frames per second, used if times is None.

    Returns:
        List of SyllableBout objects, sorted by start time.
    """
    if motif_ids.ndim != 1:
        raise ValueError(f"motif_ids must be 1D, got shape {motif_ids.shape}")

    n_frames = len(motif_ids)
    if n_frames == 0:
        return []

    # Generate times if not provided
    if times is None:
        times = np.arange(n_frames) / fps
    else:
        if len(times) != n_frames:
            raise ValueError(
                f"times length {len(times)} != motif_ids length {n_frames}"
            )

    # Find run boundaries using diff
    # A change occurs where motif_ids[i] != motif_ids[i-1]
    changes = np.concatenate([[True], motif_ids[1:] != motif_ids[:-1]])
    run_starts = np.where(changes)[0]

    # Build bouts
    bouts = []
    for i, start_idx in enumerate(run_starts):
        # End is the start of next run or end of array
        end_idx = run_starts[i + 1] if i + 1 < len(run_starts) else n_frames

        syllable_id = int(motif_ids[start_idx])
        start_sec = float(times[start_idx])
        # end_sec is the time of the last frame in the bout
        end_sec = float(times[end_idx - 1]) if end_idx > start_idx else start_sec

        # Duration: time span of the bout
        # If we have frame times, duration = time[end-1] - time[start] + frame_duration
        # Approximate: end_sec - start_sec + 1/fps (to include the last frame)
        if end_idx > start_idx + 1:
            # Multiple frames: use actual time span
            duration_sec = end_sec - start_sec
            # Add one frame duration to account for the last frame
            if len(times) > 1:
                avg_dt = (times[-1] - times[0]) / max(1, len(times) - 1)
                duration_sec += avg_dt
            else:
                duration_sec += 1.0 / fps
        else:
            # Single frame
            duration_sec = 1.0 / fps

        # Filter by minimum duration
        if duration_sec >= min_duration_sec:
            bouts.append(
                SyllableBout(
                    syllable_id=syllable_id,
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    start_sec=start_sec,
                    end_sec=end_sec + (1.0 / fps if times is None else 0),
                    duration_sec=duration_sec,
                )
            )

    return bouts


def detect_onsets(
    motif_ids: np.ndarray,
    times: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect syllable transition (onset) times.

    An onset occurs when syllable[t] != syllable[t-1].

    Args:
        motif_ids: 1D array of integer syllable IDs per frame.
        times: Optional 1D array of frame times in seconds.
        fps: Frames per second if times is None.

    Returns:
        Tuple of:
            onset_indices: Frame indices where transitions occur.
            onset_times: Times in seconds of transitions.
            post_syllables: Syllable IDs after each transition.
    """
    if motif_ids.ndim != 1 or len(motif_ids) == 0:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    n_frames = len(motif_ids)
    if times is None:
        times = np.arange(n_frames) / fps

    # Onset at frame i means syllable[i] != syllable[i-1]
    # Skip frame 0 (no preceding frame)
    changes = motif_ids[1:] != motif_ids[:-1]
    onset_indices = np.where(changes)[0] + 1  # +1 because we started from index 1

    onset_times = times[onset_indices]
    post_syllables = motif_ids[onset_indices]

    return onset_indices, onset_times, post_syllables


def remap_to_top_n(
    syllable_ids: np.ndarray,
    n: int = 10,
    other_id: int = -1,
    frequency_counts: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Remap syllable IDs to top-N vocabulary, mapping others to OTHER.

    Args:
        syllable_ids: 1D array of integer syllable IDs.
        n: Number of top syllables to keep.
        other_id: ID to assign to non-top-N syllables (default -1).
        frequency_counts: Optional pre-computed frequency counts.
            If None, computed from syllable_ids.

    Returns:
        Tuple of:
            remapped: Array with same shape, IDs remapped (top-N kept, rest → other_id).
            mapping: Dict mapping original ID → new ID (or other_id).
    """
    if frequency_counts is None:
        counts = Counter(syllable_ids.flatten())
    else:
        counts = Counter(frequency_counts)

    # Get top-N by frequency
    top_n_ids = [syll_id for syll_id, _ in counts.most_common(n)]
    top_n_set = set(top_n_ids)

    # Build mapping: top-N keep their ID, others → other_id
    mapping = {}
    for syll_id in counts.keys():
        if syll_id in top_n_set:
            mapping[syll_id] = syll_id
        else:
            mapping[syll_id] = other_id

    # Apply mapping
    remapped = np.array([mapping.get(s, other_id) for s in syllable_ids.flatten()])
    remapped = remapped.reshape(syllable_ids.shape)

    return remapped, mapping


def compute_duration_stats(bouts: List[SyllableBout]) -> Dict[int, Dict[str, float]]:
    """Compute duration statistics per syllable class.

    Args:
        bouts: List of SyllableBout objects.

    Returns:
        Dict mapping syllable_id → {mean, median, std, min, max, count}.
    """
    from collections import defaultdict

    durations_by_syllable: Dict[int, List[float]] = defaultdict(list)
    for bout in bouts:
        durations_by_syllable[bout.syllable_id].append(bout.duration_sec)

    stats = {}
    for syll_id, durations in durations_by_syllable.items():
        arr = np.array(durations)
        stats[syll_id] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    return stats


def compute_transition_rate(
    motif_ids: np.ndarray,
    times: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> float:
    """Compute transition rate (transitions per minute).

    Args:
        motif_ids: 1D array of syllable IDs.
        times: Optional frame times in seconds.
        fps: Frames per second if times is None.

    Returns:
        Transitions per minute.
    """
    onset_indices, _, _ = detect_onsets(motif_ids, times, fps)
    n_transitions = len(onset_indices)

    # Total duration
    n_frames = len(motif_ids)
    if times is not None and len(times) > 1:
        total_duration_sec = times[-1] - times[0]
    else:
        total_duration_sec = n_frames / fps

    if total_duration_sec <= 0:
        return 0.0

    return n_transitions / (total_duration_sec / 60.0)


def compute_transition_matrix(
    motif_ids: np.ndarray,
    syllable_set: Optional[List[int]] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """Compute syllable → syllable transition probability matrix.

    Args:
        motif_ids: 1D array of syllable IDs.
        syllable_set: Optional list of syllable IDs to include.
            If None, uses all unique IDs in motif_ids.
        normalize: If True, normalize rows to probabilities. If False,
            return raw counts.

    Returns:
        Tuple of:
            matrix: (N, N) transition matrix where matrix[i,j] = P(j|i).
            syllable_list: List of syllable IDs corresponding to matrix indices.
    """
    if len(motif_ids) < 2:
        return np.array([[]]), []

    if syllable_set is None:
        syllable_list = sorted(list(set(motif_ids)))
    else:
        syllable_list = sorted(syllable_set)

    n = len(syllable_list)
    if n == 0:
        return np.array([[]]), []

    id_to_idx = {syll_id: i for i, syll_id in enumerate(syllable_list)}

    # Count transitions
    matrix = np.zeros((n, n), dtype=float)
    for t in range(len(motif_ids) - 1):
        from_id = motif_ids[t]
        to_id = motif_ids[t + 1]
        if from_id in id_to_idx and to_id in id_to_idx:
            i, j = id_to_idx[from_id], id_to_idx[to_id]
            matrix[i, j] += 1

    # Normalize rows
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums

    return matrix, syllable_list
