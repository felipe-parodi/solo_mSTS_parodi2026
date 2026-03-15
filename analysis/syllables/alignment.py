"""Syllable alignment to neural time bins.

This module handles aligning frame-wise syllable labels to neural analysis
time bins, using bin-center majority voting for assignment.

The standard approach is:
1. Define neural time bins (e.g., 100ms bins)
2. For each bin, find all frame times within the bin
3. Assign the majority syllable label to that bin
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_bin_centers(
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Compute bin centers from bin edges.

    Args:
        bin_edges: 1D array of bin edges (N+1 edges for N bins).

    Returns:
        1D array of N bin centers.
    """
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def align_syllables_to_bins(
    motif_ids: np.ndarray,
    frame_times: np.ndarray,
    bin_edges: np.ndarray,
    method: str = "center",
    missing_value: int = -999,
) -> np.ndarray:
    """Align frame-wise syllable labels to neural time bins.

    Args:
        motif_ids: 1D array of syllable IDs per frame.
        frame_times: 1D array of frame times in seconds.
        bin_edges: 1D array of bin edges (N+1 edges for N bins).
        method: How to assign labels to bins:
            - "center": Use label at bin center time (interpolated).
            - "majority": Use most common label among frames in bin.
        missing_value: Value to assign to bins with no data.

    Returns:
        1D array of syllable IDs per bin (length N = len(bin_edges) - 1).
    """
    if len(motif_ids) != len(frame_times):
        raise ValueError(
            f"motif_ids length {len(motif_ids)} != frame_times length {len(frame_times)}"
        )

    n_bins = len(bin_edges) - 1
    if n_bins <= 0:
        return np.array([], dtype=int)

    bin_labels = np.full(n_bins, missing_value, dtype=int)

    if method == "center":
        # Use nearest frame to bin center
        bin_centers = compute_bin_centers(bin_edges)
        for i, center in enumerate(bin_centers):
            # Find nearest frame time
            if len(frame_times) > 0:
                idx = np.argmin(np.abs(frame_times - center))
                # Only assign if frame is reasonably close to bin center
                # (within half bin width)
                bin_width = bin_edges[i + 1] - bin_edges[i]
                if np.abs(frame_times[idx] - center) <= bin_width:
                    bin_labels[i] = motif_ids[idx]

    elif method == "majority":
        # Use most common label in each bin
        for i in range(n_bins):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            mask = (frame_times >= bin_start) & (frame_times < bin_end)
            if np.any(mask):
                labels_in_bin = motif_ids[mask]
                # Find most common (mode)
                unique, counts = np.unique(labels_in_bin, return_counts=True)
                bin_labels[i] = unique[np.argmax(counts)]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'center' or 'majority'.")

    return bin_labels


def align_syllables_for_two_subjects(
    m1_motif_ids: np.ndarray,
    m1_frame_times: np.ndarray,
    m2_motif_ids: np.ndarray,
    m2_frame_times: np.ndarray,
    bin_edges: np.ndarray,
    method: str = "center",
    missing_value: int = -999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align syllables for both subjects to common neural bins.

    Convenience function for social sessions with two monkeys.

    Args:
        m1_motif_ids: Subject 1 syllable IDs per frame.
        m1_frame_times: Subject 1 frame times.
        m2_motif_ids: Subject 2 syllable IDs per frame.
        m2_frame_times: Subject 2 frame times.
        bin_edges: Common bin edges for both subjects.
        method: Assignment method ("center" or "majority").
        missing_value: Value for bins with no data.

    Returns:
        Tuple of (m1_bin_labels, m2_bin_labels).
    """
    m1_labels = align_syllables_to_bins(
        m1_motif_ids, m1_frame_times, bin_edges, method, missing_value
    )
    m2_labels = align_syllables_to_bins(
        m2_motif_ids, m2_frame_times, bin_edges, method, missing_value
    )
    return m1_labels, m2_labels


def detect_bin_onsets(
    bin_syllables: np.ndarray,
    missing_value: int = -999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect syllable transitions in binned data.

    An onset occurs at bin t if syllables[t] != syllables[t-1] and
    neither is missing.

    Args:
        bin_syllables: 1D array of syllable IDs per bin.
        missing_value: Value indicating missing data.

    Returns:
        Tuple of:
            onset_bins: Bin indices where transitions occur.
            post_syllables: Syllable IDs after each transition.
    """
    if len(bin_syllables) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Find transitions (excluding missing values)
    valid_prev = bin_syllables[:-1] != missing_value
    valid_curr = bin_syllables[1:] != missing_value
    changed = bin_syllables[1:] != bin_syllables[:-1]

    onset_mask = valid_prev & valid_curr & changed
    onset_bins = np.where(onset_mask)[0] + 1  # +1 because we compare with previous

    post_syllables = bin_syllables[onset_bins]

    return onset_bins, post_syllables


def get_onset_aligned_indices(
    onset_bins: np.ndarray,
    n_bins: int,
    pre_bins: int,
    post_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get bin indices aligned to each onset for window extraction.

    For each onset, returns indices spanning [onset - pre_bins, onset + post_bins).

    Args:
        onset_bins: Array of onset bin indices.
        n_bins: Total number of bins in the session.
        pre_bins: Number of bins before onset.
        post_bins: Number of bins after onset (including onset).

    Returns:
        Tuple of:
            window_indices: 2D array (n_onsets, window_size) of bin indices.
                Invalid indices (< 0 or >= n_bins) are set to -1.
            valid_mask: 2D boolean array indicating valid indices.
    """
    window_size = pre_bins + post_bins
    n_onsets = len(onset_bins)

    if n_onsets == 0:
        return np.array([], dtype=int).reshape(0, window_size), np.array(
            [], dtype=bool
        ).reshape(0, window_size)

    # Create relative indices
    rel_indices = np.arange(-pre_bins, post_bins)

    # Broadcast to get absolute indices for each onset
    # Shape: (n_onsets, window_size)
    window_indices = onset_bins[:, np.newaxis] + rel_indices[np.newaxis, :]

    # Mark invalid indices
    valid_mask = (window_indices >= 0) & (window_indices < n_bins)

    # Set invalid to -1 (caller should use valid_mask)
    window_indices = np.where(valid_mask, window_indices, -1)

    return window_indices, valid_mask
