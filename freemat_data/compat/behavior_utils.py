"""Legacy-compatible wrapper around freemat_data behavior loaders.

This module provides backward compatibility with data_utils.behavior_utils.
New code should use freemat_data.loaders.behavior directly.

Example:
    # Legacy usage (still works)
    from freemat_data.compat.behavior_utils import load_and_align_behavioral_data
    df = load_and_align_behavioral_data('behavior.csv')

    # Modern usage (preferred)
    from freemat_data.loaders.behavior import load_behavior_states_from_csv
    states = load_behavior_states_from_csv('behavior.csv')
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..loaders.behavior import (
    load_behavior_states_from_csv,
    EXCLUDE_DEFAULT,
)
from ..types import StateSeries

logger = logging.getLogger(__name__)


def load_and_align_behavioral_data(
    behavioral_labels_file: str,
    use_gpu: bool = False,
    behaviors_to_exclude: Optional[List[str]] = None,
    min_behavior_duration_sec: float = 0,
) -> Optional[pd.DataFrame]:
    """Load behavioral data with legacy-compatible interface.

    Wraps the modern freemat_data API to provide backward compatibility with
    data_utils.behavior_utils.load_and_align_behavioral_data().

    Args:
        behavioral_labels_file: Path to the behavioral labels CSV file.
        use_gpu: Ignored (kept for API compatibility). Modern loader is CPU-only.
        behaviors_to_exclude: List of behavior names to exclude.
        min_behavior_duration_sec: Minimum total duration for a behavior to be included.

    Returns:
        DataFrame with columns: Time-M1, M1-Behavior, Time-M2, M2-Behavior.
        Returns None if loading fails.
    """
    if use_gpu:
        logger.warning(
            "use_gpu=True is deprecated. Modern loader uses CPU-only pandas. "
            "Ignoring GPU flag."
        )

    path = Path(behavioral_labels_file)
    if not path.exists():
        logger.error(f"Behavioral file not found: {path}")
        return None

    # Load using modern API
    states = load_behavior_states_from_csv(str(path))

    if not states:
        logger.warning(f"No behavioral states loaded from {path}")
        return None

    # Convert StateSeries back to legacy DataFrame format
    rows: List[Dict[str, Any]] = []

    # Get states for each subject
    m1_states = states.get("m1")
    m2_states = states.get("m2")

    # Build exclusion set
    exclude_set: Set[str] = set(EXCLUDE_DEFAULT)
    if behaviors_to_exclude:
        exclude_set.update(b.lower().strip() for b in behaviors_to_exclude)

    def _states_to_rows(
        state_series: Optional[StateSeries],
        time_col: str,
        behavior_col: str,
    ) -> List[Tuple[float, str]]:
        """Convert StateSeries to (time, behavior) tuples."""
        if state_series is None:
            return []
        if state_series.segments is None or state_series.segment_labels is None:
            return []

        segments = np.asarray(state_series.segments)
        labels = state_series.segment_labels

        result = []
        for i, (seg, lbl) in enumerate(zip(segments, labels)):
            t_start = float(seg[0])
            label = str(lbl).lower().strip() if lbl else ""

            # Skip excluded behaviors
            if label in exclude_set:
                continue

            result.append((t_start, label))

        return result

    # Get rows for each subject
    M1_rows = _states_to_rows(m1_states, "Time-M1", "M1-Behavior")
    M2_rows = _states_to_rows(m2_states, "Time-M2", "M2-Behavior")

    # Apply minimum duration filter if specified
    if min_behavior_duration_sec > 0:
        M1_rows = _filter_by_duration(M1_rows, min_behavior_duration_sec)
        M2_rows = _filter_by_duration(M2_rows, min_behavior_duration_sec)

    # Build output DataFrame in legacy format
    # Legacy format has paired rows for both subjects
    max_len = max(len(M1_rows), len(M2_rows))

    df_data = {
        "Time-M1": [],
        "M1-Behavior": [],
        "Time-M2": [],
        "M2-Behavior": [],
    }

    for i in range(max_len):
        if i < len(M1_rows):
            t_h, b_h = M1_rows[i]
            df_data["Time-M1"].append(_seconds_to_time_str(t_h))
            df_data["M1-Behavior"].append(b_h)
        else:
            df_data["Time-M1"].append(None)
            df_data["M1-Behavior"].append(None)

        if i < len(M2_rows):
            t_d, b_d = M2_rows[i]
            df_data["Time-M2"].append(_seconds_to_time_str(t_d))
            df_data["M2-Behavior"].append(b_d)
        else:
            df_data["Time-M2"].append(None)
            df_data["M2-Behavior"].append(None)

    df = pd.DataFrame(df_data)

    if df.empty:
        logger.warning(f"No behaviors remaining after filtering for {path}")
        return None

    logger.info(
        f"Loaded {len(df)} behavioral events from {path} "
        f"(M1: {len(M1_rows)}, M2: {len(M2_rows)})"
    )

    return df


def _filter_by_duration(
    rows: List[Tuple[float, str]],
    min_duration_sec: float,
) -> List[Tuple[float, str]]:
    """Filter behaviors by minimum total duration.

    Args:
        rows: List of (time, behavior) tuples, sorted by time.
        min_duration_sec: Minimum total duration to keep a behavior.

    Returns:
        Filtered list with low-duration behaviors replaced with 'misc'.
    """
    if not rows:
        return rows

    # Calculate duration for each behavior
    behavior_durations: Dict[str, float] = {}

    for i in range(len(rows)):
        t_start = rows[i][0]
        behavior = rows[i][1]

        # Duration is time until next event
        if i + 1 < len(rows):
            t_end = rows[i + 1][0]
            duration = t_end - t_start
        else:
            # Last event: assume 1 second duration
            duration = 1.0

        if duration > 0:
            behavior_durations[behavior] = (
                behavior_durations.get(behavior, 0.0) + duration
            )

    # Find behaviors that meet the threshold
    behaviors_to_keep = {
        b for b, d in behavior_durations.items() if d >= min_duration_sec
    }

    # Replace low-duration behaviors with 'misc'
    result = []
    for t, b in rows:
        if b in behaviors_to_keep:
            result.append((t, b))
        else:
            result.append((t, "misc"))

    return result


def _seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to H:M:S time string.

    Args:
        seconds: Time in seconds from start.

    Returns:
        Time string in H:M:S format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def get_event_onsets(
    aligned_df: pd.DataFrame,
    monkey_behavior_col: str,
    target_behavior: str,
) -> np.ndarray:
    """Get onset times for a specific behavior.

    Args:
        aligned_df: DataFrame from load_and_align_behavioral_data().
        monkey_behavior_col: Column name for behavior (e.g., 'M1-Behavior').
        target_behavior: Behavior name to find onsets for.

    Returns:
        Array of onset times in seconds.
    """
    time_col = monkey_behavior_col.replace("-Behavior", "").replace(
        "Behavior", ""
    )
    time_col = f"Time-{time_col.strip('-')}"

    if time_col not in aligned_df.columns:
        logger.warning(f"Time column {time_col} not found in DataFrame")
        return np.array([])

    mask = aligned_df[monkey_behavior_col].str.lower() == target_behavior.lower()
    time_strs = aligned_df.loc[mask, time_col].dropna()

    onsets = []
    for ts in time_strs:
        sec = _time_str_to_seconds(str(ts))
        if sec is not None:
            onsets.append(sec)

    return np.array(onsets, dtype=float)


def _time_str_to_seconds(ts: str) -> Optional[float]:
    """Convert H:M:S time string to seconds.

    Args:
        ts: Time string in H:M:S or M:S format.

    Returns:
        Time in seconds, or None if parsing fails.
    """
    parts = ts.strip().split(":")
    # TODO: Remove try-except per CLAUDE.md rules in tech debt cleanup
    try:
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return float(m * 60 + s)
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return float(h * 3600 + m * 60 + s)
    except (ValueError, TypeError):
        pass
    return None


def clean_behavior(behavior: Optional[str]) -> Optional[List[str]]:
    """Clean behavior string for analysis.

    Normalizes behavior labels by:
    1. Removing trailing/leading spaces
    2. Removing parenthetical content
    3. Converting 'pace on..' to just 'pace'
    4. Normalizing climb behaviors
    5. Converting misc to Miscellaneous
    6. Splitting compound events on '+'
    7. Normalizing case to lowercase

    Args:
        behavior: Raw behavior string from ethogram.

    Returns:
        List of cleaned behavior strings, or None if input is None.
    """
    if behavior is None:
        return None

    # Clean up the string first
    behavior = behavior.strip().lower()

    # Remove parenthetical content
    if "(" in behavior:
        behavior = behavior[: behavior.find("(")].strip()

    # Normalize specific behaviors
    if behavior.startswith("pace"):
        behavior = "pace"
    elif behavior.startswith("climb up"):
        behavior = "climb up"
    elif behavior.startswith("climb down"):
        behavior = "climb down"
    elif behavior.startswith("misc"):
        behavior = "miscellaneous"
    elif "tube bench" in behavior:
        behavior = behavior.replace("tube bench", "").strip()
    elif "bench" in behavior:
        behavior = behavior.replace("bench", "").strip()

    # Split compound behaviors and clean each part
    if "+" in behavior:
        parts = []
        for part in behavior.split("+"):
            cleaned_part = part.strip()
            # Apply the same cleaning rules to each part
            if cleaned_part.startswith("pace"):
                cleaned_part = "pace"
            elif cleaned_part.startswith("climb up"):
                cleaned_part = "climb up"
            elif cleaned_part.startswith("climb down"):
                cleaned_part = "climb down"
            elif cleaned_part.startswith("misc"):
                cleaned_part = "miscellaneous"
            elif "tube bench" in cleaned_part:
                cleaned_part = cleaned_part.replace("tube bench", "").strip()
            elif "bench" in cleaned_part:
                cleaned_part = cleaned_part.replace("bench", "").strip()
            parts.append(cleaned_part)
        return parts

    return [behavior]


def parse_time(time_str: Union[str, Any]) -> Optional[datetime]:
    """Parse time string to datetime, handling different formats including M:S.

    Args:
        time_str: Time string in various formats (H:M:S, M:S, etc.)

    Returns:
        datetime object with placeholder date (1900-01-01), or None if parsing fails.
        Returns input unchanged if not a string.
    """
    if not isinstance(time_str, str):
        return time_str

    parts = time_str.split(":")

    # TODO: Remove try-except per CLAUDE.md rules in tech debt cleanup
    try:
        if len(parts) == 2:
            # Assumes M:S format
            m, s = map(int, parts)
            h, m = divmod(m, 60)
            return datetime(1900, 1, 1, h, m, s)

        elif len(parts) == 3:
            h, m, s = map(int, parts)
            if h >= 24:
                # If "hour" is >= 24, treat as M:S:00
                m_from_h, s_from_m = h, m
                h_new, m_new = divmod(m_from_h, 60)
                return datetime(1900, 1, 1, h_new, m_new, s_from_m)
            elif h > 12:
                # If hour > 12, likely M:S format not H:M:S
                total_minutes = h
                seconds = m
                h_new, m_new = divmod(total_minutes, 60)
                return datetime(1900, 1, 1, h_new, m_new, seconds)
            else:
                return datetime.strptime(time_str, "%H:%M:%S")
        else:
            return None

    except (ValueError, TypeError):
        return None
