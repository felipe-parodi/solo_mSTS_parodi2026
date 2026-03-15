from __future__ import annotations

"""Motif loader for ethogram outputs produced by the motifs pipeline.

This reads frame-wise motif labels from a motifs run directory and returns
per-sample times and motif IDs for a specific session. It prefers the most
curated labels available in the run directory:

1) ethogram_final.parquet (after optional label merging)
2) ethogram_smoothed.parquet
3) ethogram.parquet (raw)

To resolve paths, the loader checks a small set of keys in the session
registry's ``data_files`` mapping:
- ``motifs_dir`` or ``motifs_run``: directory containing the parquet files
- ``motifs_ethogram``: direct path to a specific ethogram parquet
- ``motifs_frames_map``: optional override path to frames_map.parquet

If a directory is provided, the loader expects the parquet files produced by
``poyopo.freemat_vid_inference.motifs.cli single`` to be present. If a direct
ethogram path is given, the loader will try to find ``frames_map.parquet`` in
the same directory to recover per-frame times; if unavailable, it will fall
back to ``embedding.parquet`` (which contains ``t_sec``) or return a series
without times.
"""

from typing import Dict, Optional, Tuple
import os

import numpy as np
import polars as pl

from ..pathing import normalize
from ..types import MotifSeries, SessionRef


def _resolve_run_dir(session: SessionRef) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve motifs paths from a session.

    Returns a tuple ``(run_dir, eth_path, frames_map_path)`` where any element
    may be None if not configured.
    """
    files: Dict[str, Optional[str]] = dict(session.data_files or {})
    # Normalize a few accepted keys
    run_dir = files.get("motifs_dir") or files.get("motifs_run") or files.get("motifs")
    eth_path = files.get("motifs_ethogram") or files.get("ethogram_parquet")
    fmap_path = files.get("motifs_frames_map")

    if eth_path and not run_dir:
        # Infer directory from ethogram path
        run_dir = os.path.dirname(str(eth_path))

    # Normalize (using shared PathHandler when available)
    run_dir_n = normalize(run_dir)
    eth_path_n = normalize(eth_path)
    fmap_path_n = normalize(fmap_path)
    return (
        str(run_dir_n) if run_dir_n is not None else None,
        str(eth_path_n) if eth_path_n is not None else None,
        str(fmap_path_n) if fmap_path_n is not None else None,
    )


def _pick_ethogram_path(run_dir: str) -> Optional[str]:
    """Choose best-available ethogram parquet under ``run_dir``.

    Prefers final → smoothed → raw.
    """
    candidates = [
        os.path.join(run_dir, "ethogram_final.parquet"),
        os.path.join(run_dir, "ethogram_smoothed.parquet"),
        os.path.join(run_dir, "ethogram.parquet"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _frames_map_path(run_dir: str) -> Optional[str]:
    p = os.path.join(run_dir, "frames_map.parquet")
    return p if os.path.isfile(p) else None


def load_motifs_from_session(session: SessionRef) -> Optional[MotifSeries]:
    """Load motif assignments for a session from a motifs run directory.

    The loader expects the standard motifs outputs under a configured directory
    (``ethogram*.parquet``, ``frames_map.parquet``). It filters rows by the
    provided ``session.session_id`` and returns a ``MotifSeries`` with per-
    sample times (seconds) and motif IDs. If frame times are not available, the
    returned ``t`` is ``None`` and only ``motif_id`` is populated.

    Args:
        session: SessionRef for which to load motifs.

    Returns:
        MotifSeries if the run directory and at least one ethogram parquet are
        found; otherwise None.
    """
    run_dir, eth_path, fmap_override = _resolve_run_dir(session)
    if not run_dir and not eth_path:
        return None

    # Decide which ethogram to use
    eth_file = eth_path if eth_path is not None else _pick_ethogram_path(str(run_dir))  # type: ignore[arg-type]
    if not eth_file or not os.path.isfile(eth_file):
        return None

    # Read ethogram parquet and filter to this session
    eth = pl.read_parquet(eth_file)
    if "session" not in eth.columns:
        raise ValueError(f"Ethogram parquet missing 'session' column: {eth_file}")
    if "frame_id" not in eth.columns or "llac" not in eth.columns:
        raise ValueError(f"Ethogram parquet missing required columns in {eth_file}")

    sid = session.metadata.get("ethogram_session_id", session.session_id)
    eth_s = eth.filter(pl.col("session") == sid)
    if eth_s.height == 0:
        # If the ethogram has no rows for this session, nothing to return
        return None

    # Optional: drop invalid frames if present
    if "valid" in eth_s.columns:
        eth_s = eth_s.filter(pl.col("valid") == True)  # noqa: E712

    # Resolve frame times via frames_map or embedding as fallback
    t_sec = None
    run_dir_eff = os.path.dirname(eth_file)
    fmap = fmap_override or _frames_map_path(run_dir_eff)
    if fmap and os.path.isfile(fmap):
        fm = pl.read_parquet(fmap).filter(pl.col("session") == sid)
        if "frame_id" in fm.columns and "t_sec" in fm.columns:
            joined = eth_s.join(fm.select(["frame_id", "t_sec"]), on="frame_id", how="left")
            t_sec = joined.get_column("t_sec").to_numpy().astype(np.float32, copy=False)
            eth_s = joined
    else:
        # Fallback: read embedding.parquet for times
        emb_path = os.path.join(run_dir_eff, "embedding.parquet")
        if os.path.isfile(emb_path):
            emb = pl.read_parquet(emb_path).filter(pl.col("session") == sid)
            if {"frame_id", "t_sec"}.issubset(set(emb.columns)):
                joined = eth_s.join(emb.select(["frame_id", "t_sec"]), on="frame_id", how="left")
                t_sec = joined.get_column("t_sec").to_numpy().astype(np.float32, copy=False)
                eth_s = joined

    # Sort by frame_id to ensure monotonic order
    eth_s = eth_s.sort(by=["frame_id"])

    motif_id = eth_s.get_column("llac").to_numpy().astype(np.int32, copy=False)
    t = None
    if t_sec is not None and isinstance(t_sec, np.ndarray) and t_sec.shape[0] == motif_id.shape[0]:
        t = t_sec

    meta = {
        "session_id": sid,
        "source": os.path.basename(eth_file),
        "run_dir": run_dir or run_dir_eff,
    }
    return MotifSeries(t=t, motif_id=motif_id, segments=None, segment_ids=None, names=None, meta=meta)
