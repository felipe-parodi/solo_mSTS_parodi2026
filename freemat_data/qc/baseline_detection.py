"""Baseline detection from events text files.

Extracts baseline start/end segments from session event logs and computes
time windows for raster plots. Separated from plotting to improve testability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaselineDetectionError(RuntimeError):
    """Critical error when baseline detection fails and windowed plots are requested.

    This exception is not caught by _try_plot and will halt the entire QC process.
    """


# Default text patterns for baseline events across session styles
_BASELINE_INCLUDE: Dict[str, List[str]] = {
    "baseline.start": [
        "exit room & start baseline",
        "start baseline",
        "start 5 min baseline",
    ],
    "baseline.end": [
        "end baseline and place in quad",
        "end baseline",
        "end 5 min baseline",
    ],
}

# Default duration used to synthesize a missing baseline.end:
# 30s (exit room) + 250s (baseline) = 280s
_SYNTH_END_OFFSET_S: float = 280.0


def detect_baseline_span(
    events_text_path: str,
    *,
    session,
    logger_tag: Optional[str] = None,
    sync=None,
    errlog: Optional[Path] = None,
) -> Optional[Tuple[float, float]]:
    """Detect baseline time span from an events text file.

    Args:
        events_text_path: Path to the events text file.
        session: Loaded session YAML (used for logger inference).
        logger_tag: Logger tag (e.g., 'm1', 'm2').
        sync: Sync map object (if available, maps to video seconds).
        errlog: Path to error log file for debug output.

    Returns:
        (start_s, end_s) tuple in mapped time domain, or None if
        baseline could not be detected.

    Raises:
        BaselineDetectionError: If multiple unmatched starts require user input
            and no single start can be auto-resolved.
    """
    from .. import (
        infer_logger,
        map_times_to_video,
        parse_logger_text,
        find_events,
        to_segments,
    )

    if not os.path.isfile(str(events_text_path)):
        _log(errlog, f"events_text not found: {events_text_path}")
        return None

    _log(errlog, f"events_path={events_text_path}")
    parsed = parse_logger_text(str(events_text_path))

    # Debug parsed tags
    meta_tags = (
        getattr(parsed, "meta", {}).get("tags", {})
        if hasattr(parsed, "meta")
        else {}
    )
    _log(errlog, f"parsed.tags keys={list(meta_tags.keys())}")

    tag = infer_logger(session)
    events = find_events(parsed.df, include=_BASELINE_INCLUDE, tag=tag)
    _log(errlog, f"inferred_tag={tag} matches={len(events)}")

    if events.empty:
        # Fallback: search across all tags
        events_any = find_events(parsed.df, include=_BASELINE_INCLUDE, tag=None)
        if not events_any.empty:
            _log(errlog, f"no baseline for tag; any-tag matches={len(events_any)}")
            events = events_any

    # Choose time domain
    if sync is not None:
        time_field = "pc"
    elif "logger_sec_rel" in events.columns and np.isfinite(events["logger_sec_rel"]).any():
        time_field = "logger_rel"
    else:
        time_field = "logger"

    # Handle missing baseline.end events
    events = _handle_missing_end(
        events,
        time_field=time_field,
        logger_tag=logger_tag,
        events_text_path=events_text_path,
        errlog=errlog,
    )

    seg = to_segments(
        events,
        start_keys=["baseline.start"],
        end_keys=["baseline.end"],
        label="baseline",
        time_field=time_field,
    )

    if seg.segments.shape[0] < 1:
        if "label" in events.columns:
            n_start = int((events["label"] == "baseline.start").sum())
            n_end = int((events["label"] == "baseline.end").sum())
            _log(errlog, f"baseline not found after tag/any-tag; starts={n_start} ends={n_end}")
        return None

    s0, s1 = float(seg.segments[0, 0]), float(seg.segments[0, 1])
    _log(errlog, f"baseline domain={time_field} raw=({s0:.3f},{s1:.3f})")

    # Map to video seconds if sync available and using PC domain
    if sync is not None and time_field == "pc":
        mapped = map_times_to_video(
            np.array([s0, s1]), sync, domain="pc", logger=logger_tag,
        )
        s0, s1 = float(mapped[0]), float(mapped[1])
        _log(errlog, f"baseline mapped video=({s0:.3f},{s1:.3f})")

    return (s0, s1)


def compute_raster_window(
    baseline_span: Tuple[float, float],
    window_s: float,
    t_min: float,
    t_max: float,
) -> Tuple[float, float]:
    """Compute raster time window centered on baseline.

    Args:
        baseline_span: (start, end) of baseline period.
        window_s: Desired window width in seconds.
        t_min: Minimum time in the data.
        t_max: Maximum time in the data.

    Returns:
        (window_start, window_end) tuple.
    """
    s0, s1 = baseline_span
    W = float(window_s)
    baseline_len = max(float(s1 - s0), 0.0)

    if baseline_len >= W:
        return (float(s0), float(s0) + W)

    # Center window on baseline midpoint where possible
    c = float(0.5 * (s0 + s1))
    ws = max(min(c - 0.5 * W, t_max - W), t_min)
    we = min(ws + W, t_max)

    # Shift to get full W when possible
    if (we - ws) < W and (t_max - t_min) >= W:
        if ws <= t_min + 1.0:
            we = min(t_min + W, t_max)
            ws = max(we - W, t_min)
        elif we >= t_max - 1.0:
            ws = max(t_max - W, t_min)
            we = min(ws + W, t_max)

    return (float(ws), float(we))


def validate_baseline_for_window(
    baseline_span: Optional[Tuple[float, float]],
    *,
    window_s: Optional[float],
    events_text_path: Optional[str],
    logger_tag: Optional[str],
    errlog: Optional[Path] = None,
) -> None:
    """Raise BaselineDetectionError if windowed raster requires baseline but none found.

    Args:
        baseline_span: Detected baseline span, or None.
        window_s: Requested window size (None = no windowing).
        events_text_path: Events text path for error messaging.
        logger_tag: Logger tag for error messaging.
        errlog: Error log path for error messaging.
    """
    if window_s is None:
        return
    if baseline_span is not None and all(np.isfinite(baseline_span)):
        return

    err_details: List[str] = []
    if not events_text_path:
        err_details.append("events_text_path not provided in session config")
    elif not os.path.isfile(str(events_text_path)):
        err_details.append(f"events file not found: {events_text_path}")
    else:
        err_details.append("baseline events not found or mapping to video failed")
        err_details.append(f"  Expected baseline.start/end events in: {events_text_path}")
        if baseline_span is not None:
            err_details.append(f"  Baseline span: {baseline_span} (contains NaN)")

    raise BaselineDetectionError(
        f"Baseline detection failed for {logger_tag or 'session'}.\n"
        f"Windowed raster (window_s={window_s}) requires valid baseline.\n"
        + "\n".join(err_details)
        + "\n"
        + (f"Check error log: {errlog}" if errlog else "")
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log(errlog: Optional[Path], msg: str) -> None:
    """Append a debug line to the error log (if provided)."""
    if errlog is None:
        return
    with errlog.open("a", encoding="utf-8") as f:
        f.write(f"[baseline] {msg}\n")


def _handle_missing_end(
    events: pd.DataFrame,
    *,
    time_field: str,
    logger_tag: Optional[str],
    events_text_path: str,
    errlog: Optional[Path],
) -> pd.DataFrame:
    """Synthesize a baseline.end event when only starts are present.

    For a single unmatched start, auto-synthesizes end = start + 280s.
    For multiple unmatched starts, prompts the user to select one.

    Args:
        events: Events DataFrame with 'label' column.
        time_field: Time domain key ('logger', 'pc', 'logger_rel').
        logger_tag: Logger tag for error messages.
        events_text_path: Events file path for error messages.
        errlog: Error log path.

    Returns:
        Updated events DataFrame (may have a synthetic end row appended).
    """
    if events.empty or "label" not in events.columns:
        return events

    starts_mask = events["label"] == "baseline.start"
    ends_mask = events["label"] == "baseline.end"
    n_starts = int(starts_mask.sum())
    n_ends = int(ends_mask.sum())

    _log(errlog, f"baseline events: n_starts={n_starts}, n_ends={n_ends}, total={len(events)}")

    if n_starts == 0 or n_ends > 0:
        return events

    tf_map = {
        "logger": "logger_sec",
        "pc": "pc_sec",
        "logger_tod": "logger_tod_sec",
        "logger_rel": "logger_sec_rel",
    }
    time_col = tf_map.get(time_field, "logger_sec")

    if n_starts == 1:
        start_row = events[starts_mask].iloc[0]
        start_time = float(start_row[time_col])
        end_time = start_time + _SYNTH_END_OFFSET_S

        end_row = start_row.copy()
        end_row["label"] = "baseline.end"
        end_row[time_col] = end_time

        events = pd.concat([events, pd.DataFrame([end_row])], ignore_index=True)
        _log(
            errlog,
            f"WARNING: No 'end 5 min baseline' found. "
            f"Synthesizing end at {end_time:.3f}s ({time_col}) = start + {_SYNTH_END_OFFSET_S}s",
        )
    else:
        # Multiple starts without ends — prompt user
        print(f"\n{'=' * 70}")
        print(f"BASELINE DETECTION: Multiple unmatched starts found for {logger_tag or 'session'}")
        print(f"{'=' * 70}")
        print(f"Found {n_starts} 'baseline.start' events but 0 'baseline.end' events.")
        print(f"\nAvailable start events in {events_text_path}:")

        start_events = events[starts_mask].copy()
        for idx, row in start_events.iterrows():
            t_val = row.get(time_col, np.nan)
            print(f"  [{idx}] {time_col}={t_val:.3f}s - {row.get('text', 'N/A')}")

        print("\nWhich start event should be used? Enter index number (or 'q' to quit): ", end="")
        user_choice = input().strip()

        if user_choice.lower() == "q":
            raise BaselineDetectionError(
                f"User aborted baseline selection for {logger_tag or 'session'}.\n"
                f"Found {n_starts} unmatched baseline.start events.\n"
                + (f"Check error log: {errlog}" if errlog else "")
            )

        selected_idx = int(user_choice)
        if selected_idx not in start_events.index:
            raise BaselineDetectionError(
                f"Invalid selection: {selected_idx}. "
                f"Valid indices: {list(start_events.index)}\n"
                + (f"Check error log: {errlog}" if errlog else "")
            )

        start_row = events.loc[selected_idx]
        start_time = float(start_row[time_col])
        end_time = start_time + _SYNTH_END_OFFSET_S

        end_row = start_row.copy()
        end_row["label"] = "baseline.end"
        end_row[time_col] = end_time

        events = pd.concat([events, pd.DataFrame([end_row])], ignore_index=True)
        _log(
            errlog,
            f"User selected start event at index {selected_idx}, "
            f"time={start_time:.3f}s. Synthesized end at {end_time:.3f}s",
        )

    return events
