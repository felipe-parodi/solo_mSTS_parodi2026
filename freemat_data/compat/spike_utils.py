"""Legacy-compatible wrapper around freemat_data spike loaders.

This module provides backward compatibility with data_utils.spike_utils.
New code should use freemat_data.loaders.spikes and freemat_data.transforms.spikes directly.

Example:
    # Legacy usage (still works)
    from freemat_data.compat.spike_utils import load_and_prep_neural_data
    rasters, metadata = load_and_prep_neural_data(mat_file, bin_duration_sec=0.5)

    # Modern usage (preferred)
    from freemat_data.loaders.spikes import load_spikes_events_mat
    from freemat_data.transforms.spikes import bin_events
    events = load_spikes_events_mat(mat_file)
    edges, counts = bin_events(events['spike_t'], events['spike_unit_index'], ...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..loaders.spikes import load_spikes_events_mat
from ..transforms.spikes import bin_events

logger = logging.getLogger(__name__)


def load_and_prep_neural_data(
    mat_file: str,
    num_units_to_load: Optional[int] = None,
    use_multiunit: bool = False,
    time_shift_sec: int = 0,
    data_is_1s_binned: bool = True,
    target_deuteron_channels: Optional[List[int]] = None,
    channels_to_exclude: Optional[List[int]] = None,
    bin_from_raw_spikes: bool = False,
    bin_duration_sec: float = 1.0,
    recording_duration_sec: Optional[float] = None,
    channel_map: Optional[Dict[int, str]] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """Load neural data from .mat file with legacy-compatible interface.

    Wraps the modern freemat_data API to provide backward compatibility with
    data_utils.spike_utils.load_and_prep_neural_data().

    Args:
        mat_file: Path to the .mat file.
        num_units_to_load: Max number of units to load after filtering.
        use_multiunit: If False, exclude MUA (first unit per channel when multiple exist).
        time_shift_sec: Time shift in seconds (positive = forward, negative = backward).
        data_is_1s_binned: Indicates if pre-loaded Unit_rasters are 1s binned.
        target_deuteron_channels: If provided, only include units from these 0-indexed channels.
        channels_to_exclude: If provided, exclude units from these 0-indexed channels.
        bin_from_raw_spikes: If True, bin from SpikeData instead of using Unit_rasters.
        bin_duration_sec: Bin width in seconds when binning from raw spikes.
        recording_duration_sec: Total recording duration; if None, inferred from data.
        channel_map: Optional dict mapping channel IDs to area names (e.g., {0: 'dorsal'}).

    Returns:
        Tuple of (rasters, metadata) where:
        - rasters: numpy array (U, T) of spike counts per unit per time bin, or None
        - metadata: list of dicts with {'deuteron_id': int, 'area': str} per unit, or None
    """
    # Load events using modern API
    events = load_spikes_events_mat(mat_file, prefer_spikedata=bin_from_raw_spikes)

    if events is None:
        logger.warning(f"Failed to load spike data from {mat_file}")
        return None, None

    spike_t = events.get("spike_t")
    spike_unit_index = events.get("spike_unit_index")
    unit_channels = events.get("unit_channels")
    unit_kind = events.get("unit_kind")
    event_source = events.get("event_source", "unknown")
    bin_width_from_file = events.get("bin_width_s")

    if spike_t is None or spike_unit_index is None:
        logger.warning(f"No spike events found in {mat_file}")
        return None, None

    n_units_total = int(spike_unit_index.max()) + 1 if spike_unit_index.size > 0 else 0

    if n_units_total == 0:
        logger.warning(f"No units found in {mat_file}")
        return None, None

    # Build unit mask based on channel/MUA filtering
    unit_mask = np.ones(n_units_total, dtype=bool)

    # Channel-based filtering (requires unit_channels metadata)
    if unit_channels is not None and len(unit_channels) == n_units_total:
        if target_deuteron_channels is not None:
            target_set = set(target_deuteron_channels)
            for i in range(n_units_total):
                if unit_channels[i] not in target_set:
                    unit_mask[i] = False

        if channels_to_exclude is not None:
            exclude_set = set(channels_to_exclude)
            for i in range(n_units_total):
                if unit_channels[i] in exclude_set:
                    unit_mask[i] = False
    elif target_deuteron_channels is not None or channels_to_exclude is not None:
        logger.warning(
            "Channel filtering requested but unit_channels metadata not available. "
            "Loading all units without channel filtering."
        )

    # MUA filtering (requires unit_kind metadata)
    if not use_multiunit and unit_kind is not None and len(unit_kind) == n_units_total:
        for i in range(n_units_total):
            if unit_kind[i] == "MUA":
                unit_mask[i] = False
    elif not use_multiunit and unit_kind is None:
        logger.warning(
            "use_multiunit=False but unit_kind metadata not available. "
            "Cannot filter MUA units."
        )

    # Get selected unit indices
    selected_units = np.where(unit_mask)[0]

    if len(selected_units) == 0:
        logger.warning(f"No units remaining after filtering in {mat_file}")
        return None, None

    # Determine recording duration
    if recording_duration_sec is not None:
        t_end = recording_duration_sec
    elif spike_t.size > 0:
        t_end = float(np.max(spike_t)) + bin_duration_sec
    else:
        logger.warning("Cannot determine recording duration")
        return None, None

    # Determine bin size
    if event_source == "Unit_rasters" and bin_width_from_file is not None:
        # Data came from pre-binned Unit_rasters
        actual_bin_sec = float(bin_width_from_file)
        if data_is_1s_binned:
            actual_bin_sec = 1.0
    else:
        # Data came from SpikeData (raw spikes)
        actual_bin_sec = bin_duration_sec

    # Bin the events
    edges, counts = bin_events(
        spike_t,
        spike_unit_index,
        t_start=0.0,
        t_end=t_end,
        bin_s=actual_bin_sec,
        n_units=n_units_total,
    )

    # counts is (T, U), we need (U, T)
    rasters = counts.T  # Now (U, T)

    # Apply unit filtering
    rasters = rasters[selected_units, :]

    # Apply time shift
    if time_shift_sec != 0:
        shift_bins = int(round(time_shift_sec / actual_bin_sec))
        n_time_bins = rasters.shape[1]

        if shift_bins > 0:
            # Forward shift: remove first N bins
            if shift_bins >= n_time_bins:
                logger.warning(
                    f"time_shift_sec ({time_shift_sec}s) too large. Raster would be empty."
                )
                return None, None
            rasters = rasters[:, shift_bins:]
        elif shift_bins < 0:
            # Backward shift: remove last N bins
            abs_shift = abs(shift_bins)
            if abs_shift >= n_time_bins:
                logger.warning(
                    f"time_shift_sec ({time_shift_sec}s) too large. Raster would be empty."
                )
                return None, None
            rasters = rasters[:, :-abs_shift]

    # Limit number of units if requested
    if num_units_to_load is not None and num_units_to_load > 0:
        if num_units_to_load < rasters.shape[0]:
            rasters = rasters[:num_units_to_load, :]
            selected_units = selected_units[:num_units_to_load]

    # Build metadata
    unit_metadata: List[Dict[str, Any]] = []
    for unit_idx in selected_units:
        meta: Dict[str, Any] = {}

        if unit_channels is not None and unit_idx < len(unit_channels):
            deuteron_id = int(unit_channels[unit_idx])
            meta["deuteron_id"] = deuteron_id

            if channel_map is not None:
                meta["area"] = channel_map.get(deuteron_id, "unknown")
            else:
                meta["area"] = "unknown"
        else:
            meta["deuteron_id"] = -1
            meta["area"] = "unknown"

        if unit_kind is not None and unit_idx < len(unit_kind):
            meta["kind"] = str(unit_kind[unit_idx])

        unit_metadata.append(meta)

    # Limit metadata to match rasters
    if num_units_to_load is not None and num_units_to_load > 0:
        unit_metadata = unit_metadata[:num_units_to_load]

    if rasters.shape[0] == 0:
        logger.warning(f"Final rasters are empty for {mat_file}")
        return None, None

    logger.info(
        f"Loaded {rasters.shape[0]} units ({rasters.shape[1]} time bins) from {mat_file}"
    )

    return rasters, unit_metadata


def filter_units_by_channel(
    unit_channels: np.ndarray,
    include_channels: Optional[List[int]] = None,
    exclude_channels: Optional[List[int]] = None,
) -> np.ndarray:
    """Create boolean mask for units based on channel filtering.

    Args:
        unit_channels: Array of channel IDs per unit (U,).
        include_channels: If provided, only include units from these channels.
        exclude_channels: If provided, exclude units from these channels.

    Returns:
        Boolean mask (U,) where True means unit should be included.
    """
    n_units = len(unit_channels)
    mask = np.ones(n_units, dtype=bool)

    if include_channels is not None:
        include_set = set(include_channels)
        for i in range(n_units):
            if unit_channels[i] not in include_set:
                mask[i] = False

    if exclude_channels is not None:
        exclude_set = set(exclude_channels)
        for i in range(n_units):
            if unit_channels[i] in exclude_set:
                mask[i] = False

    return mask


def filter_units_by_kind(
    unit_kind: np.ndarray,
    include_mua: bool = True,
    include_sua: bool = True,
) -> np.ndarray:
    """Create boolean mask for units based on unit type (MUA/SUA).

    Args:
        unit_kind: Array of unit types ('MUA', 'SUA', etc.) per unit (U,).
        include_mua: Whether to include multi-unit activity.
        include_sua: Whether to include single-unit activity.

    Returns:
        Boolean mask (U,) where True means unit should be included.
    """
    n_units = len(unit_kind)
    mask = np.ones(n_units, dtype=bool)

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
) -> np.ndarray:
    """Shift rasters by N bins.

    Args:
        rasters: Array (U, T) of spike counts.
        shift_bins: Number of bins to shift (positive = forward/right, negative = backward/left).

    Returns:
        Shifted rasters with reduced time dimension.

    Raises:
        ValueError: If shift would result in empty rasters.
    """
    n_time = rasters.shape[1]

    if abs(shift_bins) >= n_time:
        raise ValueError(
            f"Shift of {shift_bins} bins too large for {n_time} time bins"
        )

    if shift_bins > 0:
        return rasters[:, shift_bins:]
    elif shift_bins < 0:
        return rasters[:, :shift_bins]
    else:
        return rasters.copy()
