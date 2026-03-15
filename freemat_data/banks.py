from __future__ import annotations

"""Bank mapping utilities backed by configs/banks.yaml.

Channels are assumed to be Deuteron IDs (0-indexed). The YAML provides
lists of channel IDs for each bank and optional subject-specific exclusion
lists.
"""

from typing import Dict, Iterable, Optional, Tuple
import os
import yaml
import numpy as np


def load_banks_config(path: str = "configs/banks.yaml") -> Dict[str, list]:
    """Load bank/channel configuration from YAML.

    Args:
        path: Path to YAML file (default: configs/banks.yaml relative to CWD).

    Returns:
        Dict with keys: 'ventral_channels', 'dorsal_channels', and optional
        'M1_exclude', 'M2_exclude'.

    Raises:
        FileNotFoundError: If the YAML does not exist.
        ValueError: If required keys are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        obj = yaml.safe_load(f) or {}
    for k in ("ventral_channels", "dorsal_channels"):
        if k not in obj or not isinstance(obj[k], list):
            raise ValueError(f"banks.yaml missing key '{k}' or not a list")
    return obj


def map_channels_to_bank(
    unit_channels: np.ndarray,
    banks_cfg: Dict[str, list],
) -> np.ndarray:
    """Map per-unit channel indices to bank labels ('dorsal'/'ventral'/'unknown').

    Args:
        unit_channels: (U,) array of channel IDs (int, 0-indexed Deuteron IDs).
        banks_cfg: Loaded YAML dict.

    Returns:
        Array of shape (U,) with strings: 'dorsal', 'ventral', or 'unknown'.
    """
    unit_channels = np.asarray(unit_channels).astype(int, copy=False)
    vent = set(int(x) for x in banks_cfg.get("ventral_channels", []))
    dors = set(int(x) for x in banks_cfg.get("dorsal_channels", []))
    out = np.empty(unit_channels.shape[0], dtype=object)
    for i, ch in enumerate(unit_channels):
        if ch in vent:
            out[i] = "ventral"
        elif ch in dors:
            out[i] = "dorsal"
        else:
            out[i] = "unknown"
    return out


def excluded_channels_for_subject(subject: Optional[str], banks_cfg: Dict[str, list]) -> set:
    """Return the set of channels to exclude for a given subject name.

    Args:
        subject: e.g., 'M2' or 'M1'. Case-insensitive; None → empty set.
        banks_cfg: Loaded YAML dict.

    Returns:
        Set of channel IDs to exclude.
    """
    if not subject:
        return set()
    s = str(subject).strip().lower()
    key = f"{s}_exclude"
    lst = banks_cfg.get(key, [])
    try:
        return set(int(x) for x in lst)
    except Exception:
        return set()


def mask_excluded_units(
    unit_channels: np.ndarray,
    subject: Optional[str],
    banks_cfg: Dict[str, list],
) -> np.ndarray:
    """Boolean mask for units to keep (channel not in subject-specific excludes).

    Args:
        unit_channels: (U,) array of channel IDs.
        subject: Subject name; see excluded_channels_for_subject.
        banks_cfg: YAML dict.

    Returns:
        keep_mask: (U,) boolean array; True means keep unit.
    """
    unit_channels = np.asarray(unit_channels).astype(int, copy=False)
    excl = excluded_channels_for_subject(subject, banks_cfg)
    if not excl:
        return np.ones(unit_channels.shape[0], dtype=bool)
    return ~np.isin(unit_channels, np.fromiter(excl, dtype=int))


def channels_from_unit_ids(unit_ids: Optional[np.ndarray], expect_U: Optional[int] = None) -> Optional[np.ndarray]:
    """Parse channel numbers from unit_ids strings of the form 'ch###'.

    This is strict by design: only explicit 'ch' patterns followed by three
    digits are accepted (case-insensitive). Examples: 'ch034', 'CH127_SUA'.

    Args:
        unit_ids: Array-like of unit identifiers (strings or bytes).
        expect_U: Optional expected number of units; when provided and the
            parsed array size mismatches, returns None.

    Returns:
        int array of channel IDs with shape (U,) when successful; otherwise None.
    """
    try:
        if unit_ids is None:
            return None
        arr = np.asarray(unit_ids)
        U = int(arr.size)
        if expect_U is not None and int(expect_U) != U:
            # If provided, insist on exact length match
            return None
        out = np.full((U,), -1, dtype=int)
        for i in range(U):
            s = arr[i]
            if isinstance(s, (bytes, bytearray)):
                try:
                    s = s.decode("utf-8", errors="ignore")
                except Exception:
                    s = str(s)
            s = str(s)
            s_low = s.lower()
            if len(s_low) >= 5 and s_low.startswith("ch") and s_low[2:5].isdigit():
                out[i] = int(s_low[2:5])
            else:
                out[i] = -1
        if np.all(out >= 0):
            return out
    except Exception:
        return None
    return None


def normalize_channels_for_banks(
    unit_channels: np.ndarray,
    banks_cfg: Dict[str, list],
) -> Tuple[np.ndarray, str, Dict[str, int]]:
    """Return channels as-is (identity) for banks.yaml mapping.

    freemat_data loaders already produce 0-indexed Deuteron channel IDs, and
    configs/banks.yaml is authored in that same 0-index domain. To avoid
    silent remapping, we keep identity only and report unknowns under identity.

    Args:
        unit_channels: Per-unit channel IDs (int array, shape (U,)).
        banks_cfg: Loaded banks.yaml dictionary.

    Returns:
        (channels_identity, 'identity', {'identity': unknown_count})
    """
    ch = np.asarray(unit_channels).astype(int, copy=False)
    try:
        labels = map_channels_to_bank(ch, banks_cfg)
        unk = int(np.sum(labels == "unknown"))
    except Exception:
        # If mapping fails, treat all as unknown
        unk = int(ch.size)
    return ch, "identity", {"identity": unk}
