from __future__ import annotations

"""Placeholders for LFP loading.

These functions define the public API; implementations will be added once
schemas are finalized. Keep failure explicit to avoid silent fallbacks.
"""

from typing import Optional

from ..types import LFPTimeSeries, SessionRef


def load_lfp_npz(npz_path: str) -> LFPTimeSeries:
    """Load LFP from an NPZ file.

    Args:
        npz_path: Path to NPZ file containing LFP arrays.

    Returns:
        LFPTimeSeries object with time vector and channels.

    Raises:
        NotImplementedError: Placeholder; to be implemented later.
    """
    raise NotImplementedError("LFP NPZ loader not implemented yet.")


def load_lfp_from_session(session: SessionRef) -> Optional[LFPTimeSeries]:
    """Load LFP time series for a session using configured registry paths.

    Args:
        session: SessionRef from registry or YAML.

    Returns:
        LFPTimeSeries if available, else None.

    Raises:
        NotImplementedError: Placeholder; to be implemented later.
    """
    raise NotImplementedError("LFP session loader not implemented yet.")

