"""Spike utilities for Single-FM analyses.

This module composes project-agnostic freemat_data APIs into
bank/subject-aware helpers for Single-FM notebooks and pipelines.

Functions here should be small, typed, and fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from freemat_data import (
    SessionRef,
    SpikesEvents,
    load_spikes_from_session,
    load_banks_config,
    infer_subject,
    mask_excluded_units,
    zscore_by_unit,
    bin_events,
    firing_rate,
    map_channels_to_bank,
    channels_from_unit_ids,
    normalize_channels_for_banks,
)
from freemat_data.qc.spikes import compute_qc, SpikesQCParams


@dataclass
class SpikesBundle:
    """Spike events with optional per-unit metadata.

    Attributes:
        spikes: Event-coded spikes (neural time)
        keep_units: Optional boolean mask (U,) indicating units to keep
        unit_bank: Optional per-unit bank labels (U,)
    """

    spikes: SpikesEvents
    keep_units: Optional[np.ndarray]
    unit_bank: Optional[np.ndarray]


def load_with_banks(sess: SessionRef, *, banks_yaml: Optional[str] = None) -> SpikesBundle:
    """Load spikes and apply subject/bank-aware exclusions if configured.

    Args:
        sess: Session reference
        banks_yaml: Optional path to `configs/banks.yaml`

    Returns:
        SpikesBundle with spikes, keep_units mask, and unit_bank labels (if present)
    """

    spikes = load_spikes_from_session(sess)
    keep_units = None
    unit_bank = None

    if banks_yaml:
        banks_cfg = load_banks_config(banks_yaml)
        subject = infer_subject(sess)

        # Resolve per-unit channel IDs from meta or unit_ids
        U = int(spikes.unit_index.max()) + 1 if spikes.unit_index.size else 0
        unit_channels: Optional[np.ndarray] = None
        if hasattr(spikes, "meta") and isinstance(spikes.meta, dict):
            ch = spikes.meta.get("unit_channels")
            if ch is not None and np.asarray(ch).size in (U,):
                unit_channels = np.asarray(ch, dtype=int)
        if unit_channels is None:
            # Strict parse: expects patterns like 'ch034_*'; returns None on mismatch
            unit_channels = channels_from_unit_ids(spikes.unit_ids, expect_U=U)

        if unit_channels is not None and unit_channels.size == U:
            # Normalize channels into banks.yaml domain; avoid remapping sentinels
            ch_norm, rule, _unk = normalize_channels_for_banks(unit_channels, banks_cfg)
            # Subject-specific keep mask and bank labels
            keep_units = mask_excluded_units(ch_norm, subject=subject, banks_cfg=banks_cfg)
            unit_bank = map_channels_to_bank(ch_norm, banks_cfg)

    return SpikesBundle(spikes=spikes, keep_units=keep_units, unit_bank=unit_bank)


def bin_to_tidy(
    t_spk_vid: np.ndarray,
    unit_index: np.ndarray,
    *,
    session_id: str,
    bin_s: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> pd.DataFrame:
    """Bin events to rates and return a tidy dataframe with FR and FR_z.

    Args:
        t_spk_vid: Event times (video seconds), shape (E,)
        unit_index: Unit indices (E,) in [0,U)
        session_id: Session ID to annotate output
        bin_s: Bin width (seconds)
        t_start: Optional start bound; defaults to min(t)
        t_end: Optional end bound; defaults to max(t)

    Returns:
        DataFrame with columns: t, bin_idx, unit_index, fr, fr_z, session_id, bin_s
    """

    t = np.asarray(t_spk_vid, dtype=float).reshape(-1)
    u = np.asarray(unit_index, dtype=int).reshape(-1)
    if t.size == 0:
        return pd.DataFrame(columns=["t", "bin_idx", "unit_index", "fr", "fr_z", "session_id", "bin_s"])  # empty

    if t_start is None:
        t_start = float(np.min(t))
    if t_end is None:
        t_end = float(np.max(t))
    edges, counts = bin_events(t, u, t_start=t_start, t_end=t_end, bin_s=float(bin_s))
    if counts.size == 0:
        return pd.DataFrame(columns=["t", "bin_idx", "unit_index", "fr", "fr_z", "session_id", "bin_s"])  # empty
    fr = firing_rate(counts, float(bin_s))  # (T,U)
    T, U = fr.shape
    bin_idx = np.arange(T, dtype=int)
    t_cent = (edges[:-1] + edges[1:]) / 2.0
    unit_list = np.arange(U, dtype=int)
    df = pd.DataFrame(
        {
            "bin_idx": np.repeat(bin_idx, U),
            "t": np.repeat(t_cent, U),
            "unit_index": np.tile(unit_list, T),
            "fr": fr.reshape(-1),
        }
    )
    df["session_id"] = session_id
    df["bin_s"] = float(bin_s)
    # Z-score per unit using the matrix form for numerical stability
    Z = zscore_by_unit(fr)
    df["fr_z"] = Z.reshape(-1)
    return df


def qc_summary(
    t_spk_vid: np.ndarray,
    unit_index: np.ndarray,
    *,
    bin_s: float = 1.0,
    unit_channels: Optional[np.ndarray] = None,
    banks: Optional[np.ndarray] = None,
    params: Optional[SpikesQCParams] = None,
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Compute per-unit and population QC metrics using freemat_data.qc.

    Args:
        t_spk_vid: Event times (video seconds)
        unit_index: Unit indices for events
        bin_s: Base bin width for QC-derived metrics
        unit_channels: Optional per-unit channels
        banks: Optional per-unit bank labels
        params: Optional QC parameter overrides

    Returns:
        (per_unit, summary) metric mappings
    """

    return compute_qc(
        np.asarray(t_spk_vid),
        np.asarray(unit_index, dtype=int),
        bin_s=float(bin_s),
        unit_channels=unit_channels,
        banks=banks,
        params=params,
    )


__all__ = [
    "SpikesBundle",
    "load_with_banks",
    "bin_to_tidy",
    "qc_summary",
]
