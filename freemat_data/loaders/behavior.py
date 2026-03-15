from __future__ import annotations

"""Behavioral state loading from CSVs aligned to video seconds.

Implements a typed loader that reads the canonical dual-monkey CSV schema:

    Time-Hooke, Hooke-Behavior, Time-Hedy, Hedy-Behavior

Times may be expressed as H:M:S, M:S, or commonly misformatted entries such as
"24:10:00" that indicate 24 minutes 10 seconds. This module applies strict,
documented heuristics to normalize times to video seconds and cleans labels by
removing parenthetical notes and standardizing common variants (pace → pace,
misc → miscellaneous, climb up/down, remove "bench" tokens).

Composite labels separated by '+' are expanded into overlapping segments per
atomic label as requested by analysis code.
"""

from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

from ..types import StateSeries, SessionRef


# Default exclusions (after cleaning/normalization)
EXCLUDE_DEFAULT: Tuple[str, ...] = (
    "start",
    "experimenter enters",
    "experimenter leaves",
    "miscellaneous",
)


def _parse_time_to_seconds(ts: str) -> Optional[float]:
    """Parse a time string into seconds with robust heuristics.

    Supported patterns:
      - M:S            → minutes, seconds
      - H:M:S          → hours, minutes, seconds
      - Misformatted   → e.g., "24:10:00" intended as 24 minutes 10 seconds

    Heuristics for three-field strings:
      - If hours >= 24: treat as M:S using (hours, minutes) and ignore the
        third field. Example: "26:35:00" → 26 min 35 s.
      - Else if hours > 12: treat as M:S using (hours, minutes) (annotation
        convenience where H was actually minutes) and ignore seconds.
      - Else: interpret as conventional H:M:S.
    """
    if ts is None:
        return None
    if not isinstance(ts, str):
        try:
            # Pass-through for numeric input
            v = float(ts)  # type: ignore[arg-type]
            return v if math.isfinite(v) else None
        except Exception:
            return None
    s = ts.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), int(parts[1])
            return float(m * 60 + sec)
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            # Handle common annotation patterns
            if h >= 24:
                # Use (h, m) as (M, S); ignore third field
                return float(h * 60 + m)
            if h > 12:
                return float(h * 60 + m)
            return float(h * 3600 + m * 60 + sec)
    except Exception:
        return None
    # Fallback: single number etc. → None
    return None


def _clean_and_split_label(lbl: Optional[str]) -> List[str]:
    """Normalize a behavior label and split on '+' into atomic labels.

    Rules:
      - Lowercase, strip
      - Remove parenthetical suffixes "(...)"
      - Normalize prefixes: pace*, climb up*, climb down*, misc* → standardized
      - Remove "tube bench" and "bench" tokens
      - Split on '+' and normalize each part recursively
    """
    if lbl is None:
        return []
    s = str(lbl).strip().lower()
    if not s:
        return []
    # Remove inline comments after ';'
    if ";" in s:
        s = s.split(";")[0].strip()
    # Remove parenthetical notes
    if "(" in s:
        s = s[: s.find("(")].strip()
    # Replace tokens
    s = s.replace("tube bench", " ")
    s = s.replace("bench", " ")
    # Split composites
    parts = [p.strip() for p in s.split("+")]

    def _norm_one(x: str) -> Optional[str]:
        if not x:
            return None
        if x.startswith("pace"):
            return "pace"
        if x.startswith("climb up"):
            return "climb up"
        if x.startswith("climb down"):
            return "climb down"
        if x.startswith("misc"):
            return "miscellaneous"
        return x.strip()

    out: List[str] = []
    for p in parts:
        v = _norm_one(p)
        if v:
            out.append(v)
    return out


def _rows_to_segments(times: np.ndarray, labels_list: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """Expand onsets into overlapping segments per label.

    For each onset time t[i] with atomic labels L[i], generate one segment per
    label from t[i] to t[i+1]. The last onset ends at NaN (open segment).
    """
    assert times.ndim == 1
    n = int(times.size)
    if n == 0:
        return np.zeros((0, 2), dtype=float), []
    segs: List[Tuple[float, float, str]] = []
    for i in range(n):
        t0 = float(times[i])
        t1 = float(times[i + 1]) if i + 1 < n else float("nan")
        for lbl in labels_list[i]:
            # Skip empty
            if not lbl:
                continue
            segs.append((t0, t1, lbl))
    if not segs:
        return np.zeros((0, 2), dtype=float), []
    # Build arrays
    segments = np.asarray([[a, b] for (a, b, _) in segs], dtype=float)
    seg_labels = [lbl for (_, _, lbl) in segs]
    return segments, seg_labels


def _build_states_for_subject(
    df: pd.DataFrame,
    *,
    time_col: str,
    label_col: str,
    excludes: Tuple[str, ...],
) -> StateSeries:
    """Construct a StateSeries for a single subject from a CSV subset."""
    t_raw = df[time_col].tolist()
    l_raw = df[label_col].tolist()

    # Parse times and clean labels
    tt: List[float] = []
    labs: List[List[str]] = []
    for t_str, lb in zip(t_raw, l_raw):
        t = _parse_time_to_seconds(t_str)
        if t is None or not math.isfinite(t):
            continue
        atomic = _clean_and_split_label(lb)
        # Apply exclusions after cleaning
        atomic = [x for x in atomic if x and x not in excludes]
        if not atomic:
            # Row excluded entirely
            continue
        tt.append(float(t))
        labs.append(atomic)

    if not tt:
        return StateSeries(segments=np.zeros((0, 2), dtype=float), segment_labels=[], label_set=[], meta={})

    # Sort by time and de-duplicate near-duplicates (<= 0.5 s)
    order = np.argsort(np.asarray(tt, dtype=float), kind="mergesort")
    tt_sorted = [tt[i] for i in order]
    labs_sorted = [labs[i] for i in order]

    tt_dedup: List[float] = []
    labs_dedup: List[List[str]] = []
    EPS = 0.5
    for t, l in zip(tt_sorted, labs_sorted):
        if (len(tt_dedup) > 0) and (abs(t - tt_dedup[-1]) <= EPS):
            # Prefer the latest occurrence by replacing
            tt_dedup[-1] = t
            labs_dedup[-1] = l
        else:
            tt_dedup.append(t)
            labs_dedup.append(l)

    times_arr = np.asarray(tt_dedup, dtype=float)
    segments, seg_labels = _rows_to_segments(times_arr, labs_dedup)
    label_set = sorted({s for s in sum(labs_dedup, [])})

    return StateSeries(
        segments=segments,
        segment_labels=seg_labels,
        label_set=label_set,
        meta={
            "time_domain": "video",
            "excludes": list(excludes),
            "n_onsets": int(times_arr.size),
        },
    )


def load_behavior_states_from_csv(
    csv_path: str,
    *,
    excludes: Tuple[str, ...] = EXCLUDE_DEFAULT,
    subject_map: Optional[Dict[str, str]] = None,
) -> Dict[str, StateSeries]:
    """Load behavioral states from a dual-subject CSV into StateSeries objects.

    Args:
        csv_path: Path to CSV with columns: Time-Hooke, Hooke-Behavior,
            Time-Hedy, Hedy-Behavior.
        excludes: Labels to drop after normalization (default excludes misc).
        subject_map: Optional mapping from CSV subject names to output keys.
            Example: {"Hooke": "m1", "Hedy": "m2"}.

    Returns:
        Dict mapping subject key → StateSeries.
    """
    df = pd.read_csv(csv_path)
    cols = set(df.columns.astype(str))
    need = {"Time-Hooke", "Hooke-Behavior", "Time-Hedy", "Hedy-Behavior"}
    missing = need - cols
    if missing:
        raise ValueError(f"Behavior CSV missing required columns: {sorted(missing)}")

    # Default subject mapping (Hooke→m1, Hedy→m2)
    sub_map = subject_map or {"Hooke": "m1", "Hedy": "m2"}
    out: Dict[str, StateSeries] = {}

    # Hooke
    hooke = _build_states_for_subject(
        df,
        time_col="Time-Hooke",
        label_col="Hooke-Behavior",
        excludes=excludes,
    )
    hooke.meta = {
        **(hooke.meta or {}),
        "source": csv_path,
        "subject": "Hooke",
        "tag": sub_map.get("Hooke", "m1"),
    }
    out[sub_map.get("Hooke", "m1")] = hooke

    # Hedy
    hedy = _build_states_for_subject(
        df,
        time_col="Time-Hedy",
        label_col="Hedy-Behavior",
        excludes=excludes,
    )
    hedy.meta = {
        **(hedy.meta or {}),
        "source": csv_path,
        "subject": "Hedy",
        "tag": sub_map.get("Hedy", "m2"),
    }
    out[sub_map.get("Hedy", "m2")] = hedy

    return out


def load_behavior_states_from_session(session: SessionRef) -> Dict[str, StateSeries]:
    """Load behavioral states from a session's configured CSV.

    This returns a dictionary keyed by 'm1' and 'm2' by default.

    Args:
        session: SessionRef.

    Returns:
        Mapping from subject tag ('m1','m2') to StateSeries.
    """
    csv_path = (
        session.data_files.get("behavior_csv")
        or session.data_files.get("behavior")
        or session.metadata.get("behavior_csv")
    )
    if not csv_path:
        raise FileNotFoundError("Session is missing 'behavior_csv' in data_files or metadata")
    return load_behavior_states_from_csv(
        str(csv_path), subject_map={"Hooke": "m1", "Hedy": "m2"}
    )
