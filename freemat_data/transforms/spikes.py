from __future__ import annotations

"""Spike transforms: binning, firing rate, z-scoring, filtering.

These functions operate on event-coded spikes (t, unit_index) or on
unit-by-time matrices and avoid hidden defaults.
"""

from typing import List, Optional, Tuple
import numpy as np


def bin_events(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    bin_s: float = 1.0,
    n_units: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram events into fixed-width bins (vectorized, fast).

    Args:
        t: Event times in seconds (E,).
        unit_index: Unit indices (E,) in [0, U).
        t_start: Inclusive start time (s).
        t_end: End time (s). Matches numpy.histogram semantics: last bin
            includes right edge (i.e., [t_start, t_end]).
        bin_s: Bin width (s).
        n_units: Number of units U; if None, inferred from unit_index.

    Returns:
        (edges, counts) where edges is (T+1,), counts is (T, U).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    unit_index = np.asarray(unit_index, dtype=int).reshape(-1)
    if n_units is None:
        n_units = int(unit_index.max()) + 1 if unit_index.size else 0
    # Build edges identical to np.histogram for parity
    edges = np.arange(float(t_start), float(t_end) + float(bin_s), float(bin_s), dtype=float)
    T = max(len(edges) - 1, 0)
    counts = np.zeros((T, n_units), dtype=np.int32)
    if t.size == 0 or n_units == 0 or T == 0:
        return edges, counts

    # Map all events to bin indices using searchsorted on edges (right-inclusive last bin)
    idx = np.searchsorted(edges, t, side='right') - 1  # [-1..T]
    # Events exactly at t_end produce idx == T → clamp to last bin
    idx[idx == T] = T - 1
    # Keep only in-range events and valid unit indices
    m = (idx >= 0) & (idx < T) & (unit_index >= 0) & (unit_index < n_units)
    if not np.any(m):
        return edges, counts
    ii = idx[m].astype(np.int64, copy=False)
    uu = unit_index[m].astype(np.int64, copy=False)
    # Accumulate counts in-place (O(E))
    np.add.at(counts, (ii, uu), 1)
    return edges, counts


def firing_rate(counts: np.ndarray, bin_s: float) -> np.ndarray:
    """Convert spike counts to firing rate (Hz)."""
    counts = np.asarray(counts, dtype=float)
    return counts / float(bin_s)


def zscore_by_unit(X: np.ndarray, *, ddof: int = 0) -> np.ndarray:
    """Z-score per column (unit). NaN-safe (ignores NaNs in mean/std).

    Args:
        X: Array (T, U).
        ddof: Degrees of freedom for std (default 0).

    Returns:
        Z-scored array (T, U).
    """
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=ddof, keepdims=True)
    sd = np.where(sd <= 0, 1.0, sd)
    return (X - mu) / sd


def filter_units(
    unit_channels: np.ndarray,
    unit_kind: Optional[np.ndarray] = None,
    *,
    include_channels: Optional[List[int]] = None,
    exclude_channels: Optional[List[int]] = None,
    include_mua: bool = True,
    include_sua: bool = True,
) -> np.ndarray:
    """Create boolean mask for units based on channel and type filtering.

    Args:
        unit_channels: Array of channel IDs per unit (U,).
        unit_kind: Array of unit types ('MUA', 'SUA') per unit (U,). Optional.
        include_channels: If provided, only include units from these channels.
        exclude_channels: If provided, exclude units from these channels.
        include_mua: Whether to include multi-unit activity.
        include_sua: Whether to include single-unit activity.

    Returns:
        Boolean mask (U,) where True means unit should be included.
    """
    n_units = len(unit_channels)
    mask = np.ones(n_units, dtype=bool)

    # Channel filtering
    if include_channels is not None:
        include_set = set(include_channels)
        for i in range(n_units):
            if int(unit_channels[i]) not in include_set:
                mask[i] = False

    if exclude_channels is not None:
        exclude_set = set(exclude_channels)
        for i in range(n_units):
            if int(unit_channels[i]) in exclude_set:
                mask[i] = False

    # Unit type filtering
    if unit_kind is not None and len(unit_kind) == n_units:
        for i in range(n_units):
            kind = str(unit_kind[i]).upper()
            if kind == "MUA" and not include_mua:
                mask[i] = False
            elif kind == "SUA" and not include_sua:
                mask[i] = False

    return mask


def time_shift_rasters(
    rasters: np.ndarray,
    shift_bins: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Shift rasters by N bins, truncating the result.

    Args:
        rasters: Array (U, T) of spike counts.
        shift_bins: Number of bins to shift.
            Positive = shift forward (remove first N bins).
            Negative = shift backward (remove last N bins).
        fill_value: Not used (kept for API compatibility). Rasters are truncated.

    Returns:
        Shifted rasters (U, T - |shift_bins|).

    Raises:
        ValueError: If shift would result in empty rasters.
    """
    rasters = np.asarray(rasters)
    n_time = rasters.shape[1] if rasters.ndim == 2 else rasters.shape[0]

    if abs(shift_bins) >= n_time:
        raise ValueError(
            f"Shift of {shift_bins} bins too large for {n_time} time bins"
        )

    if shift_bins == 0:
        return rasters.copy()

    if rasters.ndim == 2:
        # (U, T) format
        if shift_bins > 0:
            return rasters[:, shift_bins:]
        else:
            return rasters[:, :shift_bins]
    else:
        # (T,) or (T, U) format
        if shift_bins > 0:
            return rasters[shift_bins:]
        else:
            return rasters[:shift_bins]


def smooth_firing_rates(
    rates: np.ndarray,
    window_bins: int,
    axis: int = 0,
) -> np.ndarray:
    """Smooth firing rates with a moving average.

    Args:
        rates: Firing rate array. Can be (T,), (T, U), or (U, T).
        window_bins: Number of bins for the smoothing window.
        axis: Axis along which to smooth (0 = rows/time, 1 = columns).

    Returns:
        Smoothed firing rates with same shape as input.
    """
    rates = np.asarray(rates, dtype=float)

    if window_bins <= 1:
        return rates.copy()

    kernel = np.ones(window_bins) / window_bins

    if rates.ndim == 1:
        return np.convolve(rates, kernel, mode="same")
    elif rates.ndim == 2:
        result = np.zeros_like(rates)
        if axis == 0:
            for i in range(rates.shape[1]):
                result[:, i] = np.convolve(rates[:, i], kernel, mode="same")
        else:
            for i in range(rates.shape[0]):
                result[i, :] = np.convolve(rates[i, :], kernel, mode="same")
        return result
    else:
        raise ValueError(f"Unsupported array shape: {rates.shape}")
