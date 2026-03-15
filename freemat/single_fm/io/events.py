"""Single-FM events processing glue (scaffold).

This module ties together freemat_data loaders with session config and sync
to produce aligned event points and segments, and persists them alongside
aligned neural outputs. Implementation is deferred; this file defines the
API and responsibilities for NB01 usage.

Example (planned):
    >>> from freemat.single_fm.io.events import process_and_persist_events
    >>> out = process_and_persist_events(
    ...     session_yaml="configs/single_fm/session_hedy_0526.yaml",
    ...     out_dir="results/single_fm/session_hedy_0526/events",
    ...     pattern_config="configs/events_patterns.yaml",
    ...     prefer_logger_tag="m1",
    ... )
    >>> out.events_parquet, out.segments_parquet

Notes:
    - Only map to video seconds when a sync map is available (same pass as spike syncing).
    - Fail loudly on missing files, zero matches, or mismatched starts/ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EventsOutputs:
    """Artifacts produced by the events pipeline (paths only)."""

    events_parquet: Optional[str]
    segments_parquet: Optional[str]
    raw_text_copy: Optional[str]
    mapped: bool


def process_and_persist_events(
    session_yaml: str,
    *,
    out_dir: str,
    pattern_config: Optional[str] = None,
    prefer_logger_tag: str = "m1",
) -> EventsOutputs:
    """Parse logger text, extract events/segments, optionally map to video, and persist.

    Args:
        session_yaml: Path to per-session YAML.
        out_dir: Output directory for events artifacts.
        pattern_config: Optional YAML with canonical label patterns.
        prefer_logger_tag: Which tag ('m1'/'m2') to select in single-monkey sessions.

    Returns:
        EventsOutputs with paths to written Parquet files and mapped status.

    Raises:
        NotImplementedError: This is a scaffold; implementation pending.
    """
    # TODO(future):
    # - Load session YAML, resolve data_files.events_text and syncmap_json
    # - Parse logger text → df_raw
    # - Load pattern_config (or built-in defaults) → include/exclude dicts
    # - find_events(..., tag=prefer_logger_tag) → df_evt
    # - to_segments(...) for labels of interest (e.g., baseline)
    # - If sync present: map times to video via freemat_data.sync.map_times_to_video
    # - Write Parquet files for events and segments; copy raw text for provenance
    # - Return outputs and mapped flag; update aligned parquet sidecar in calling code
    raise NotImplementedError("process_and_persist_events is not implemented yet.")

