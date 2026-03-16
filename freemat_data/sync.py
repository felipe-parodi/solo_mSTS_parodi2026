from __future__ import annotations

"""Synchronization helpers leveraging scripts/synchronization outputs.

This module provides functional sync utilities that use SessionSynchronizer
from the compat layer for time conversions across timebases.

The canonical SessionSynchronizer is now in freemat_data.compat.sync_utils.
"""

from typing import Optional, Any, Mapping
from pathlib import Path

from .types import SessionRef, SpikesEvents


def _require_synchronizer():
    """Return the SessionSynchronizer class.

    Imports from the compat layer which is the canonical location.
    """
    from .compat.sync_utils import SessionSynchronizer
    return SessionSynchronizer


def load_syncmap(path: str | Path):
    """Load a sync_map.json into a SessionSynchronizer."""
    SessionSynchronizer = _require_synchronizer()
    return SessionSynchronizer(path)


def get_syncmap_from_session(session: SessionRef):
    """Return SessionSynchronizer if session config provides 'syncmap_json'."""
    p = session.data_files.get("syncmap_json") or session.metadata.get("syncmap_json")
    if not p:
        return None
    return load_syncmap(str(p))


def _maybe_load_record_pc(session: Optional[SessionRef], tag: str) -> Optional[float]:
    """Return T_pc_record for a logger tag if session provides a record JSON.

    Looks for `data_files['{tag}_record']`; if absent, tries next to the
    syncmap_json as `<tag>_record.json`. Returns None when not found or invalid.
    """
    if session is None:
        return None
    try:
        key = f"{tag}_record"
        p = session.data_files.get(key)
        if not p:
            sync_p = session.data_files.get("syncmap_json")
            if sync_p:
                candidate = Path(sync_p).with_name(f"{tag}_record.json")
                if candidate.exists():
                    p = str(candidate)
        if not p:
            return None
        import json as _json
        val = _json.loads(Path(p).read_text()).get("T_pc_record")
        return float(val) if val is not None else None
    except Exception:
        return None


def _looks_local_logger(arr, sync) -> bool:
    """Heuristic: logger times are local (seconds from record) vs seconds-of-day.

    - If median logger time < 6h while `video.pc_flash_sec` is > 6h, treat as local.
    This mirrors notebook heuristics and avoids misusing direct logger→video fits.
    """
    try:
        import numpy as _np
        a = _np.asarray(arr, dtype=float).ravel()
        if a.size == 0:
            return False
        med = float(_np.nanmedian(a))
        v = getattr(sync, "map", {}).get("video", {})
        pc_flash = float(v.get("pc_flash_sec", _np.nan))
        if _np.isfinite(med) and _np.isfinite(pc_flash):
            return (med < (6.0 * 3600.0)) and (pc_flash > (6.0 * 3600.0))
    except Exception:
        return False
    return False


def map_spikes_to_video(
    spikes: SpikesEvents,
    sync,
    *,
    logger: str = "m1",
    session: Optional[SessionRef] = None,
):
    """Map spike timestamps to video seconds using the session sync map.

    Behavior:
    - If a direct logger→video mapping exists and the input looks like
      absolute logger seconds-of-day, apply it with day-wrap correction.
    - If inputs look like local seconds-from-record and a record anchor
      (T_pc_record) is available, map via PC: pc_to_video(t + T_pc_record).
    - Otherwise, compose via logger→PC→video using SessionSynchronizer.

    Args:
        spikes: SpikesEvents with `t` in logger time.
        sync: SessionSynchronizer for this session.
        logger: 'm1' or 'm2'.
        session: Optional SessionRef to resolve T_pc_record when needed.
    """
    import numpy as np
    tag = (logger or "m1").strip().lower()
    arr = np.asarray(spikes.t, dtype=float)

    # Prefer direct mapping when anchor and input are in the same time domain
    try:
        mp: Mapping[str, Any] = getattr(sync, "map", None) or {}
        if isinstance(mp, dict) and tag in mp and isinstance(mp[tag], dict):
            lv = mp[tag]
            ltov = lv.get("logger_to_video") if isinstance(lv, dict) else None
            if ltov:
                slope = float(ltov.get("slope", 1.0))
                t0_l = float(ltov.get("anchor_logger_sec"))
                t0_v = float(ltov.get("anchor_video_sec"))

                # Check if anchor is local (< 6h) or absolute (> 6h)
                anchor_is_local = t0_l < (6.0 * 3600.0)  # 21600s
                input_is_local = _looks_local_logger(arr, sync)

                # Use direct mapping if anchor and input are in the same domain
                if anchor_is_local == input_is_local:
                    out = arr.astype(float, copy=True)
                    if not input_is_local:
                        # Day-wrap correction for absolute times
                        day = 86400.0
                        mean = float(np.nanmean(out))
                        k = round((mean - t0_l) / day)
                        if abs(k) >= 1:
                            out = out - (k * day)
                    return slope * (out - t0_l) + t0_v

                # Domains don't match - try record anchor path for local times
                if input_is_local:
                    t_pc_rec = _maybe_load_record_pc(session, tag)
                    if t_pc_rec is not None:
                        return np.asarray(sync.pc_to_video(arr + float(t_pc_rec)), dtype=float)
                # No record anchor available → compose via SessionSynchronizer (may be approximate)
    except Exception:
        pass

    if tag == "m1":
        return np.asarray(sync.m1_to_video(arr), dtype=float)
    if tag == "m2":
        return np.asarray(sync.m2_to_video(arr), dtype=float)
    raise ValueError("logger must be 'm1' or 'm2'")


def map_times_to_video(
    t: "np.ndarray",
    sync,
    *,
    domain: str = "logger",
    logger: str = "m1",
    session: Optional[SessionRef] = None,
):
    """Map arbitrary times to video seconds using the session synchronizer.

    Mirrors map_spikes_to_video heuristics:
    - Prefer direct logger→video mapping when available under the tag in sync.map.
    - Otherwise compose via logger→PC→video using SessionSynchronizer helpers.
    - For PC-domain inputs, map via PC→video directly.
    - Applies day-wrap correction for absolute logger seconds; when inputs
      look like seconds-from-record and T_pc_record is available, use
      pc_to_video(t + T_pc_record).

    Args:
        t: 1D array-like of times in the specified domain.
        sync: Loaded SessionSynchronizer for the session.
        domain: 'logger' or 'pc' — timebase of input array.
        logger: Which logger ('m1' or 'm2') when domain='logger'.

    Returns:
        1D numpy array of video times (seconds).
    """
    import numpy as np

    arr = np.asarray(t, dtype=float).ravel()
    if arr.size == 0:
        return arr.copy()

    d = (domain or "").strip().lower()
    lg = (logger or "").strip().lower()

    if d not in ("logger", "pc"):
        raise ValueError("domain must be 'logger' or 'pc'")

    if d == "pc":
        # Use synchronizer's anchored affine mapping (flash-based)
        return np.asarray(sync.pc_to_video(arr), dtype=float).ravel()

    # domain == 'logger'
    if lg not in ("m1", "m2"):
        raise ValueError("logger must be 'm1' or 'm2' when domain='logger'")

    # Prefer direct logger→video mapping when anchor and input are in the same domain
    try:
        mp = getattr(sync, "map", None) or {}
        if isinstance(mp, dict) and lg in mp:
            lv = mp[lg]
            if isinstance(lv, dict) and "logger_to_video" in lv and lv["logger_to_video"]:
                ltov = lv["logger_to_video"] or {}
                slope = float(ltov.get("slope", 1.0))
                t0_l = float(ltov.get("anchor_logger_sec"))
                t0_v = float(ltov.get("anchor_video_sec"))

                # Check if anchor is local (< 6h) or absolute (> 6h)
                anchor_is_local = t0_l < (6.0 * 3600.0)  # 21600s
                input_is_local = _looks_local_logger(arr, sync)

                # Use direct mapping if anchor and input are in the same domain
                if anchor_is_local == input_is_local:
                    out = arr.astype(float, copy=True)
                    if not input_is_local:
                        # Day-wrap correction for absolute times
                        day = 86400.0
                        mean = float(np.nanmean(out))
                        k = round((mean - t0_l) / day)
                        if abs(k) >= 1:
                            out = out - (k * day)
                    return slope * (out - t0_l) + t0_v

                # Domains don't match - try record anchor path for local times
                if input_is_local:
                    t_pc_rec = _maybe_load_record_pc(session, lg)
                    if t_pc_rec is not None:
                        return np.asarray(sync.pc_to_video(arr + float(t_pc_rec)), dtype=float).ravel()
                # Otherwise, fall through to composed mapping
    except Exception:
        # Fall through to composed mapping
        pass

    # Fallback: composed mapping via PC
    if lg == "m1":
        return np.asarray(sync.m1_to_video(arr), dtype=float).ravel()
    else:
        return np.asarray(sync.m2_to_video(arr), dtype=float).ravel()
