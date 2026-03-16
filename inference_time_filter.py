"""Utility module for filtering inference by session time ranges.

Filters out frames outside the valid inference window (e.g., experimenter setup/cleanup).
Used across detection, tracking, ID correction, anchor annotation, and triangulation.

TSV format:
    Session, Inf Start:Stop
    240531, 00:26:00-01:55:00

Usage:
    from inference_time_filter import InferenceTimeFilter

    time_filter = InferenceTimeFilter.from_tsv("inf_start_stop_times.tsv")
    start_frame, end_frame = time_filter.get_frame_range("240531", fps=40)

    # Check if a frame is valid
    if time_filter.is_frame_valid("240531", frame_id, fps=40):
        process_frame(frame_id)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class SessionTimeRange:
    """Time range for a session."""
    session: str
    start_time: str  # HH:MM:SS format
    end_time: str    # HH:MM:SS format

    @property
    def start_seconds(self) -> float:
        """Convert start time to seconds."""
        return parse_time_to_seconds(self.start_time)

    @property
    def end_seconds(self) -> float:
        """Convert end time to seconds."""
        return parse_time_to_seconds(self.end_time)

    def get_frame_range(self, fps: float = 40.0) -> Tuple[int, int]:
        """Get (start_frame, end_frame) for this session.

        Args:
            fps: Video frame rate (default 40 for macaque videos)

        Returns:
            (start_frame, end_frame) tuple
        """
        start_frame = int(self.start_seconds * fps)
        end_frame = int(self.end_seconds * fps)
        return start_frame, end_frame

    def is_frame_valid(self, frame_id: int, fps: float = 40.0) -> bool:
        """Check if a frame is within the valid range."""
        start_frame, end_frame = self.get_frame_range(fps)
        return start_frame <= frame_id <= end_frame

    def __str__(self) -> str:
        start_f, end_f = self.get_frame_range()
        return f"{self.session}: {self.start_time}-{self.end_time} (frames {start_f:,}-{end_f:,})"


def parse_time_to_seconds(time_str: str) -> float:
    """Parse time string (H:MM:SS or HH:MM:SS) to seconds.

    Handles formats:
        - 00:26:00 (HH:MM:SS)
        - 0:23:05 (H:MM:SS)
        - 1:55:00 (H:MM:SS)
    """
    time_str = time_str.strip()
    parts = time_str.split(":")

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    elif len(parts) == 2:
        # MM:SS format
        hours = 0
        minutes = int(parts[0])
        seconds = int(parts[1])
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def frame_to_time_str(frame_id: int, fps: float = 40.0) -> str:
    """Convert frame ID to HH:MM:SS string."""
    seconds = frame_id / fps
    return seconds_to_time_str(seconds)


class InferenceTimeFilter:
    """Manages inference time ranges for multiple sessions."""

    # Default TSV location (relative to this module)
    DEFAULT_TSV_PATH = Path(__file__).parent / "inf_start_stop_times.tsv"

    def __init__(self, session_ranges: Dict[str, SessionTimeRange] = None):
        """Initialize with optional pre-loaded ranges.

        Args:
            session_ranges: Dict mapping session prefix to SessionTimeRange
        """
        self.session_ranges = session_ranges or {}

    @classmethod
    def from_tsv(cls, tsv_path: Optional[str] = None) -> "InferenceTimeFilter":
        """Load time ranges from TSV file.

        Args:
            tsv_path: Path to TSV file. If None, uses default location.

        Returns:
            InferenceTimeFilter instance
        """
        if tsv_path is None:
            tsv_path = cls.DEFAULT_TSV_PATH

        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            print(f"Warning: Inference time filter TSV not found: {tsv_path}")
            return cls({})

        session_ranges = {}

        with open(tsv_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or i == 0:  # Skip empty lines and header
                continue

            # Parse: "240531, 00:26:00-01:55:00"
            # Handle both comma and tab separators
            if ',' in line:
                parts = line.split(',', 1)
            elif '\t' in line:
                parts = line.split('\t', 1)
            else:
                continue

            if len(parts) != 2:
                continue

            session = parts[0].strip()
            time_range = parts[1].strip()

            # Parse time range: "00:26:00-01:55:00"
            if '-' not in time_range:
                continue

            # Split on the dash between times (careful: times contain colons)
            # Format: HH:MM:SS-HH:MM:SS or H:MM:SS-H:MM:SS
            time_parts = time_range.split('-')
            if len(time_parts) != 2:
                continue

            start_time = time_parts[0].strip()
            end_time = time_parts[1].strip()

            try:
                # Validate times parse correctly
                _ = parse_time_to_seconds(start_time)
                _ = parse_time_to_seconds(end_time)

                session_ranges[session] = SessionTimeRange(
                    session=session,
                    start_time=start_time,
                    end_time=end_time,
                )
            except ValueError as e:
                print(f"Warning: Could not parse time range for session {session}: {e}")
                continue

        return cls(session_ranges)

    def find_session(self, session_or_path: str) -> Optional[str]:
        """Find matching session key by prefix matching.

        Args:
            session_or_path: Session name or path containing session (e.g., "240531-latesync")

        Returns:
            Matching session key or None
        """
        # Extract potential session prefix from path
        basename = os.path.basename(session_or_path.rstrip('/\\'))

        # Try exact match first
        if basename in self.session_ranges:
            return basename

        # Try prefix match (e.g., "240531" matches "240531-latesync")
        for session_key in self.session_ranges:
            if basename.startswith(session_key) or session_key.startswith(basename):
                return session_key

        # Try to extract 6-digit date pattern (YYMMDD)
        match = re.search(r'(\d{6})', basename)
        if match:
            date_prefix = match.group(1)
            if date_prefix in self.session_ranges:
                return date_prefix

        return None

    def get_range(self, session_or_path: str) -> Optional[SessionTimeRange]:
        """Get time range for a session.

        Args:
            session_or_path: Session name or path

        Returns:
            SessionTimeRange or None if not found
        """
        session_key = self.find_session(session_or_path)
        if session_key:
            return self.session_ranges[session_key]
        return None

    def get_frame_range(
        self,
        session_or_path: str,
        fps: float = 40.0
    ) -> Optional[Tuple[int, int]]:
        """Get valid frame range for a session.

        Args:
            session_or_path: Session name or path
            fps: Video frame rate

        Returns:
            (start_frame, end_frame) or None if session not found
        """
        time_range = self.get_range(session_or_path)
        if time_range:
            return time_range.get_frame_range(fps)
        return None

    def is_frame_valid(
        self,
        session_or_path: str,
        frame_id: int,
        fps: float = 40.0
    ) -> bool:
        """Check if a frame is within the valid inference range.

        Args:
            session_or_path: Session name or path
            frame_id: Frame ID to check
            fps: Video frame rate

        Returns:
            True if valid, False if outside range.
            Returns True if session not in TSV (permissive).
        """
        time_range = self.get_range(session_or_path)
        if time_range is None:
            # Session not in TSV - permissive behavior
            return True
        return time_range.is_frame_valid(frame_id, fps)

    def filter_frames(
        self,
        session_or_path: str,
        frame_ids: list,
        fps: float = 40.0,
    ) -> list:
        """Filter a list of frame IDs to only valid ones.

        Args:
            session_or_path: Session name or path
            frame_ids: List of frame IDs
            fps: Video frame rate

        Returns:
            Filtered list of frame IDs within valid range
        """
        frame_range = self.get_frame_range(session_or_path, fps)
        if frame_range is None:
            return frame_ids

        start_frame, end_frame = frame_range
        return [f for f in frame_ids if start_frame <= f <= end_frame]

    def print_summary(self):
        """Print summary of all loaded session ranges."""
        print(f"\nInference Time Ranges ({len(self.session_ranges)} sessions):")
        print("-" * 60)
        for session in sorted(self.session_ranges.keys()):
            print(f"  {self.session_ranges[session]}")
        print("-" * 60)


# Convenience function for quick access
def get_default_filter() -> InferenceTimeFilter:
    """Get the default inference time filter (loads from default TSV)."""
    return InferenceTimeFilter.from_tsv()


if __name__ == "__main__":
    # Test/demo
    print("Testing InferenceTimeFilter...")

    time_filter = InferenceTimeFilter.from_tsv()
    time_filter.print_summary()

    # Test with 240531
    session = "240531-latesync"
    frame_range = time_filter.get_frame_range(session)
    if frame_range:
        start, end = frame_range
        print(f"\n{session}:")
        print(f"  Frame range: {start:,} - {end:,}")
        print(f"  Duration: {(end - start) / 40 / 60:.1f} minutes")
        print(f"  Frame 62400 valid: {time_filter.is_frame_valid(session, 62400)}")
        print(f"  Frame 1000 valid: {time_filter.is_frame_valid(session, 1000)}")
