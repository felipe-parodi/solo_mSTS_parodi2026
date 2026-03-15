"""Experimenter event parsing from neural logger text files.

This module provides project-agnostic helpers to parse combined logger
text files (Spikelog/CerePlex), match messages to canonical labels, and
derive labeled segments suitable for downstream analyses.

Example:
    >>> from freemat_data.loaders import events as evt
    >>> df_raw = evt.parse_logger_text("/path/to/AutoLOG_Subject001.txt")
    >>> df_evt = evt.find_events(
    ...     df_raw,
    ...     include={
    ...         "baseline.start": [
    ...             "exit room & start baseline",
    ...             "start baseline",
    ...             "start 5 min baseline",
    ...             "5 min baseline",
    ...         ],
    ...         "baseline.end": [
    ...             "end baseline and place in quad",
    ...             "end baseline",
    ...         ],
    ...     },
    ...     tag="m1",
    ... )
    >>> seg = evt.to_segments(
    ...     df_evt,
    ...     start_keys=["baseline.start"],
    ...     end_keys=["baseline.end"],
    ...     label="baseline",
    ...     time_field="logger",
    ... )

Notes:
    - Mapping to video seconds is handled in freemat_data.sync via a
      separate helper (see map_times_to_video). This module operates in
      logger/PC time domains only.

    Baseline Period Convention:
    - When "start 5 min baseline" is logged but no corresponding "end 5 min baseline"
      exists, the end time should be estimated as: start_time + 30s + 250s = start + 280s.
    - Rationale: The experimenter typically logs the start event, then takes ~30 seconds
      to exit the room. The actual baseline period (monkeys alone) is then ~250 seconds,
      giving a total of 280 seconds from the logged start event.
    - This estimation should only be applied when exactly one unmatched start exists.
      Multiple unmatched starts require manual user intervention to select the correct one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from freemat_data.types import StateSeries

import re
import datetime as dt
import numpy as np
import pandas as pd


# Default serial-number to tag mapping for single_fm context
DEFAULT_SN_TO_TAG: Dict[int, str] = {4: "m1", 3: "m2"}


@dataclass
class ParsedLoggerText:
    """Container for parsed logger text lines.

    Attributes:
        df: DataFrame with columns including:
            - pc_sec: float seconds-of-day derived from PC timestamp
            - logger_ctr: integer/logger counter value as parsed
            - logger_sec: float seconds (derived from logger_ctr when possible)
            - logger_tod_sec: optional float seconds-of-day parsed from embedded tokens
            - text: raw message text
            - tag: 'm1' or 'm2' when SN mapping is available
            - raw_line: original line content
        meta: Parse metadata (e.g., counters skipped, scale inference)
    """

    df: pd.DataFrame
    meta: Dict[str, Any]


def parse_logger_text(
    path: str,
    *,
    emit_logger_tod: bool = True,
    sn_to_tag: Optional[Dict[int, str]] = None,
) -> ParsedLoggerText:
    """Parse a combined logger text file into a structured table.

    Reuses regex anchors from the synchronization scripts to extract:
    - PC time-of-day → seconds-of-day (pc_sec)
    - Logger serial number (SN) → tag via mapping ({4:'m1',3:'m2'} by default)
    - Logger counter (2nd CSV field) → inferred units and seconds
    - Optional embedded logger time-of-day tokens (e.g., 't=10:06:43.457227')

    The function infers the counter interpretation per tag using a simple
    heuristic on d(counter)/d(pc_sec). When it looks like ~1000, the counter
    is treated as milliseconds-of-day. If a sampling frequency is detected in
    the text ("Sampling frequency: <Hz> Hz"), the counter is treated as
    samples and seconds are derived as 1/Hz. If neither is available, the
    default is milliseconds-of-day.

    Columns in the returned DataFrame:
    - pc_tod: original PC time-of-day string
    - pc_sec: float seconds-of-day (PC clock)
    - logger_ctr: raw counter value (int when parseable)
    - logger_sec: seconds in logger domain
        - ms-of-day → absolute seconds-of-day
        - samples   → seconds since 'Record pressed' when anchor present; else NaN
    - logger_sec_rel: seconds since 'Record pressed' when anchor present; else NaN
    - logger_tod: optional logger time-of-day string (from embedded token)
    - logger_tod_sec: optional float seconds-of-day from logger_tod
    - tag: 'm1' or 'm2' when SN can be mapped
    - text: message portion (everything after the second comma)
    - raw_line: original line

    Args:
        path: Path to the combined Spikelog/CerePlex text file.
        emit_logger_tod: Whether to attempt extracting logger time-of-day tokens.
        sn_to_tag: Optional mapping from serial number to tag ('m1','m2').

    Returns:
        ParsedLoggerText object with DataFrame and parse metadata.
    """
    # Patterns borrowed from scripts/synchronization/sync/parse_log.py
    _STAMP = r"(?P<pc>\d{1,2}:\d{2}:\d{2}\s[AP]M)"
    _SERIAL = r"Spikelog128\s+for\s+Cereport\s+SN\s+(?P<sn>\d+)"
    REC_RE = re.compile(fr"{_STAMP}.*{_SERIAL}.*Record\s+pressed", re.IGNORECASE)
    HZ_RE = re.compile(r"Sampling\s+frequency:\s*(?P<hz>\d+(?:\.\d+)?)\s*Hz", re.IGNORECASE)
    LOGGER_TOD_RE = re.compile(r"t=(?P<ltod>\d{1,2}:\d{2}:\d{2}\.\d+)")
    SN_RE = re.compile(_SERIAL, re.IGNORECASE)

    def _pc_to_sec(tod: str) -> float:
        t = dt.datetime.strptime(tod, "%I:%M:%S %p")
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

    def _logger_tod_to_sec(tod: str) -> float:
        h, m, s = tod.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    sn_map = sn_to_tag or DEFAULT_SN_TO_TAG

    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "lines_total": 0,
        "lines_parsed": 0,
        "lines_skipped": 0,
        "tags": {},
    }

    # First pass: parse lines and collect per-tag counters for unit inference
    per_tag_pc: Dict[str, List[float]] = {}
    per_tag_ctr: Dict[str, List[int]] = {}
    per_tag_rec: Dict[str, Dict[str, Any]] = {}
    detected_hz: Optional[float] = None

    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            meta["lines_total"] += 1
            line = ln.rstrip("\n")

            # Try to extract PC time-of-day: rely on leading CSV field
            parts = line.split(",", 2)
            if len(parts) < 2:
                meta["lines_skipped"] += 1
                continue
            pc_tod = parts[0].strip()
            try:
                pc_sec = _pc_to_sec(pc_tod)
            except Exception:
                # As a fallback, attempt regex-based extraction
                m_stamp = re.search(_STAMP, line)
                if not m_stamp:
                    meta["lines_skipped"] += 1
                    continue
                pc_tod = m_stamp.group("pc")
                pc_sec = _pc_to_sec(pc_tod)

            # Logger counter (2nd field)
            ctr_s = parts[1].strip()
            try:
                logger_ctr = int(ctr_s)
            except Exception:
                # If unparsable, mark and continue; we still keep the row
                logger_ctr = np.int64(-1)

            # Extract SN and map to tag if possible
            m_sn = SN_RE.search(line)
            tag: Optional[str] = None
            if m_sn and m_sn.group("sn"):
                try:
                    sn = int(m_sn.group("sn"))
                    tag = sn_map.get(sn)
                except Exception:
                    tag = None

            # Detect 'Record pressed' anchor (per tag)
            if tag:
                m_rec = REC_RE.search(line)
                if m_rec:
                    per_tag_rec.setdefault(tag, {})["T_pc_record"] = _pc_to_sec(m_rec.group("pc"))
                    # Store counter at record (even if unparsable, skip)
                    try:
                        per_tag_rec[tag]["ctr_record"] = int(ctr_s)
                    except Exception:
                        pass

            # Sampling frequency (global across tags)
            if detected_hz is None:
                m_hz = HZ_RE.search(line)
                if m_hz:
                    try:
                        detected_hz = float(m_hz.group("hz"))
                    except Exception:
                        detected_hz = None

            # Optional embedded logger time-of-day
            ltod_s: Optional[str] = None
            ltod_sec: Optional[float] = None
            if emit_logger_tod:
                m_ltod = LOGGER_TOD_RE.search(line)
                if m_ltod:
                    ltod_s = m_ltod.group("ltod")
                    try:
                        ltod_sec = _logger_tod_to_sec(ltod_s)
                    except Exception:
                        ltod_sec = None

            text = parts[2].strip() if len(parts) >= 3 else ""

            rows.append(
                dict(
                    pc_tod=pc_tod,
                    pc_sec=float(pc_sec),
                    logger_ctr=int(logger_ctr) if logger_ctr != -1 else pd.NA,
                    logger_tod=ltod_s,
                    logger_tod_sec=float(ltod_sec) if ltod_sec is not None else pd.NA,
                    tag=tag,
                    text=text,
                    raw_line=line,
                )
            )
            meta["lines_parsed"] += 1

            # Accumulate for unit inference by tag
            if tag is not None and logger_ctr != -1:
                per_tag_pc.setdefault(tag, []).append(float(pc_sec))
                per_tag_ctr.setdefault(tag, []).append(int(logger_ctr))

    df = pd.DataFrame(rows)

    # Per-tag unit inference and seconds computation
    meta["detected_hz"] = float(detected_hz) if detected_hz is not None else None
    # Collect unique tags robustly (avoid ambiguous truth of Series)
    if "tag" in df.columns:
        tag_values = [str(x) for x in df["tag"].dropna().tolist()]
    else:
        tag_values = []
    for tag in sorted(set(tag_values)):
        # Prepare meta bucket
        tmeta = meta["tags"].setdefault(tag, {})
        t_pc = np.asarray(per_tag_pc.get(tag, []), dtype=float)
        t_ctr = np.asarray(per_tag_ctr.get(tag, []), dtype=float)

        # Default interpretation
        counter_unit = "ms_of_day"
        counter_scale_sec = 1.0 / 1000.0

        # Prefer explicit Hz
        if detected_hz is not None and detected_hz > 0:
            counter_unit = "samples"
            counter_scale_sec = 1.0 / float(detected_hz)
        else:
            # Ratio heuristic: median(dctr/dpc) ~ 1000 → ms-of-day
            if len(t_ctr) >= 3 and len(t_pc) >= 3:
                d_ctr = np.diff(t_ctr)
                d_pc = np.diff(t_pc)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.nanmedian(d_ctr / np.where(d_pc == 0, np.nan, d_pc))
                tmeta["ratio_ctr_per_sec"] = float(ratio) if np.isfinite(ratio) else None
                if np.isfinite(ratio) and not (800.0 <= float(ratio) <= 1200.0):
                    # Not obviously ms-of-day; keep ms-of-day fallback but record ambiguity
                    tmeta["ambiguous_units"] = True
                else:
                    tmeta["ambiguous_units"] = False

        tmeta["counter_unit"] = counter_unit
        tmeta["counter_scale_sec"] = float(counter_scale_sec)

        # Attach record anchor if present
        if tag in per_tag_rec:
            tmeta.update(per_tag_rec[tag])

        # Compute seconds columns for this tag
        mask_tag = (df["tag"] == tag) & df["logger_ctr"].notna()
        ctr_arr = df.loc[mask_tag, "logger_ctr"].astype("float64").to_numpy()

        # Absolute seconds-of-day if ms-of-day; else NaN
        if counter_unit == "ms_of_day":
            df.loc[mask_tag, "logger_sec"] = ctr_arr * counter_scale_sec
        else:
            df.loc[mask_tag, "logger_sec"] = pd.NA

        # Relative seconds since 'Record pressed' when anchor available
        if "ctr_record" in tmeta:
            rec_ctr = float(tmeta["ctr_record"])  # type: ignore[index]
            df.loc[mask_tag, "logger_sec_rel"] = (ctr_arr - rec_ctr) * counter_scale_sec
        else:
            df.loc[mask_tag, "logger_sec_rel"] = pd.NA

    # Ensure columns exist even if empty
    for col in ("logger_sec", "logger_sec_rel"):
        if col not in df.columns:
            df[col] = pd.NA

    return ParsedLoggerText(df=df, meta=meta)


def find_events(
    df: pd.DataFrame,
    *,
    include: Optional[Dict[str, List[str]]] = None,
    exclude: Optional[List[str]] = None,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """Match message text to canonical labels using pattern sets.

    Args:
        df: Parsed DataFrame from parse_logger_text.
        include: Mapping from canonical label → list of substrings/regex patterns (case-insensitive).
        exclude: Optional list of substrings/regex patterns to ignore.
        tag: If provided, filter to a single tag (e.g., 'm1').

    Returns:
        Tidy DataFrame with time fields (pc_sec, logger_sec, logger_tod_sec),
        'label', 'tag', and 'raw_line' for matched events.

    Notes:
        - Patterns in 'include' are treated as case-insensitive regex; plain
          strings also work (interpreted literally unless they contain regex
          metacharacters).
        - Excludes are applied first.
        - Returns one row per matched label; if a row matches multiple labels,
          it will appear multiple times with different 'label' values.
    """
    if include is None or len(include) == 0:
        # Nothing to match — return empty dataframe with expected columns
        base_cols = [c for c in ("pc_sec", "logger_sec", "logger_tod_sec", "tag", "text", "raw_line") if c in df.columns]
        return pd.DataFrame({**{c: pd.Series(dtype=df[c].dtype if c in df.columns else "float64") for c in base_cols}, "label": pd.Series(dtype="string")})

    work = df.copy()
    if tag is not None:
        work = work[work.get("tag").fillna("").astype(str).str.lower() == tag.lower()]

    # Apply excludes
    if exclude:
        ex_mask = pd.Series(False, index=work.index)
        for pat in exclude:
            try:
                ex_mask = ex_mask | work["text"].astype(str).str.contains(pat, case=False, regex=True)
            except re.error:
                # Fallback to literal substring
                ex_mask = ex_mask | work["text"].astype(str).str.contains(re.escape(pat), case=False, regex=True)
        work = work[~ex_mask]

    out_frames: List[pd.DataFrame] = []
    for label_name, patterns in include.items():
        if not patterns:
            continue
        mask = pd.Series(False, index=work.index)
        for pat in patterns:
            try:
                mask = mask | work["text"].astype(str).str.contains(pat, case=False, regex=True)
            except re.error:
                mask = mask | work["text"].astype(str).str.contains(re.escape(pat), case=False, regex=True)
        sub = work.loc[mask].copy()
        if sub.empty:
            continue
        sub["label"] = label_name
        out_frames.append(sub)

    if not out_frames:
        # Return empty with schema if no matches
        base_cols = [c for c in ("pc_sec", "logger_sec", "logger_tod_sec", "tag", "text", "raw_line") if c in df.columns]
        return pd.DataFrame({**{c: pd.Series(dtype=df[c].dtype if c in df.columns else "float64") for c in base_cols}, "label": pd.Series(dtype="string")})

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    # Sort for deterministic downstream behavior
    sort_cols = [c for c in ["logger_sec", "pc_sec", "logger_tod_sec"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def to_segments(
    events: pd.DataFrame,
    *,
    start_keys: List[str],
    end_keys: List[str],
    label: str,
    time_field: Literal["logger", "pc", "logger_tod", "logger_rel"] = "logger",
) -> "StateSeries":
    """Convert matched start/end events into labeled segments.

    Args:
        events: Tidy events DataFrame with canonical 'label' column.
        start_keys: List of canonical labels that denote start of intervals.
        end_keys: List of canonical labels that denote end of intervals.
        label: Label to assign to produced segments (e.g., 'baseline').
        time_field: Which time to use ('logger' → logger_sec,
            'pc' → pc_sec, 'logger_tod' → logger_tod_sec).

    Returns:
        freemat_data.types.StateSeries with segments in the chosen time domain.

    Returns:
        StateSeries with segments in the chosen time domain and metadata capturing
        pairing stats (counts, duplicates ignored, mismatches).

    Raises:
        ValueError: If the requested time field is missing or pairing fails due to
            irreconcilable mismatches (e.g., end without prior start far from last end).
    """
    from freemat_data.types import StateSeries

    tf_map = {
        "logger": "logger_sec",
        "pc": "pc_sec",
        "logger_tod": "logger_tod_sec",
        "logger_rel": "logger_sec_rel",
    }
    if time_field not in tf_map:
        raise ValueError("time_field must be one of: 'logger', 'pc', 'logger_tod', 'logger_rel'")
    col = tf_map[time_field]
    if col not in events.columns:
        raise ValueError(f"events DataFrame missing required time column '{col}'")

    # Filter to relevant labels
    use = events[events["label"].isin(list(start_keys) + list(end_keys))].copy()
    if use.empty:
        return StateSeries(segments=np.zeros((0, 2), dtype=float), segment_labels=[], label_set=[label], meta={"paired": 0, "starts": 0, "ends": 0})

    # Sort by time; stable within equal-time blocks
    use = use.sort_values([col], kind="mergesort").reset_index(drop=True)

    # Greedy pairing with duplicate suppression within epsilon
    EPS_DUP = 1.0  # seconds; drop starts/ends within this window as duplicates
    segments: List[List[float]] = []
    seg_labels: List[str] = []
    pending_start: Optional[float] = None
    last_end: Optional[float] = None
    n_starts = 0
    n_ends = 0
    n_dupe_drop = 0

    times = use[col].to_numpy(dtype=float)
    labels = use["label"].astype(str).to_list()

    for t, lbl in zip(times, labels):
        if lbl in start_keys:
            n_starts += 1
            # Duplicate start within epsilon → drop
            if pending_start is not None and abs(t - pending_start) <= EPS_DUP:
                n_dupe_drop += 1
                continue
            pending_start = float(t)
        elif lbl in end_keys:
            n_ends += 1
            # Duplicate end within epsilon of last_end → drop
            if last_end is not None and abs(t - last_end) <= EPS_DUP:
                n_dupe_drop += 1
                continue
            if pending_start is None:
                # No active start; treat as spurious if near previous end; else error
                if last_end is not None and abs(t - last_end) <= EPS_DUP:
                    n_dupe_drop += 1
                    continue
                raise ValueError("Encountered end event without a preceding start; check for out-of-order or missing start events.")
            if t <= pending_start:
                # Non-positive duration; drop as malformed duplicate
                n_dupe_drop += 1
                continue
            segments.append([pending_start, float(t)])
            seg_labels.append(label)
            last_end = float(t)
            pending_start = None

    if pending_start is not None:
        raise ValueError("Unmatched start event at end of sequence; expected a corresponding end event.")

    seg_arr = np.asarray(segments, dtype=float) if segments else np.zeros((0, 2), dtype=float)
    meta = {
        "paired": int(len(segments)),
        "starts": int(n_starts),
        "ends": int(n_ends),
        "dropped_duplicates": int(n_dupe_drop),
        "time_field": time_field,
    }
    return StateSeries(segments=seg_arr, segment_labels=seg_labels, label_set=[label], meta=meta)


# ---------------------------------------------------------------------------
# RSSI extraction
# ---------------------------------------------------------------------------

_RSSI_RE = re.compile(r"RSSI=(-?\d+\.?\d*)")
_RECORD_PRESSED_RE = re.compile(r"Record\s+pressed", re.IGNORECASE)
_STOP_RECORDING_RE = re.compile(r"Stop\s+recording", re.IGNORECASE)


def parse_rssi(
    path: str,
    *,
    recording_only: bool = True,
    tag_filter: Optional[str] = None,
    sn_to_tag: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Extract RSSI time series from an AutoLOG text file.

    Reuses :func:`parse_logger_text` for line parsing, then extracts RSSI
    values from any line containing ``RSSI=`` (covers both "System check"
    and "Warning: weak radio link" messages).

    Args:
        path: Path to AutoLOG text file.
        recording_only: If True, restrict to the recording window defined
            by "Record pressed" / "Stop recording" anchors.
        tag_filter: Optional tag to keep (e.g., ``'m1'``).  ``None`` keeps
            all tags.
        sn_to_tag: SN→tag mapping (default: ``{4:'m1', 3:'m2'}``).

    Returns:
        DataFrame with columns ``pc_sec``, ``rssi_dbm``, ``tag``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If no RSSI values are found (after filtering).
    """
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"AutoLOG file not found: {path}")

    parsed = parse_logger_text(path, sn_to_tag=sn_to_tag)
    df = parsed.df

    if df.empty:
        raise ValueError(f"No lines parsed from AutoLOG: {path}")

    # --- Optional tag filter ---
    if tag_filter is not None:
        df = df[df["tag"] == tag_filter]
        if df.empty:
            raise ValueError(
                f"No lines with tag='{tag_filter}' in AutoLOG: {path}"
            )

    # --- Recording window via anchors ---
    if recording_only:
        rec_start_mask = df["text"].str.contains(
            _RECORD_PRESSED_RE, na=False
        )
        rec_stop_mask = df["text"].str.contains(
            _STOP_RECORDING_RE, na=False
        )

        rec_starts = df.loc[rec_start_mask, "pc_sec"]
        rec_stops = df.loc[rec_stop_mask, "pc_sec"]

        if rec_starts.empty:
            raise ValueError(
                f"No 'Record pressed' anchor found in AutoLOG: {path}"
            )

        t_rec_start = float(rec_starts.iloc[0])
        t_rec_stop = (
            float(rec_stops.iloc[-1]) if not rec_stops.empty else np.inf
        )
        df = df[
            (df["pc_sec"] >= t_rec_start) & (df["pc_sec"] <= t_rec_stop)
        ]

    # --- Extract RSSI from text ---
    rssi_values: List[float] = []
    rssi_pc_sec: List[float] = []
    rssi_tags: List[Optional[str]] = []

    for _, row in df.iterrows():
        m = _RSSI_RE.search(str(row["text"]))
        if m is not None:
            rssi_values.append(float(m.group(1)))
            rssi_pc_sec.append(float(row["pc_sec"]))
            rssi_tags.append(row.get("tag"))

    if len(rssi_values) == 0:
        raise ValueError(
            f"No RSSI values found in AutoLOG "
            f"(recording_only={recording_only}): {path}"
        )

    return pd.DataFrame(
        {"pc_sec": rssi_pc_sec, "rssi_dbm": rssi_values, "tag": rssi_tags}
    )
