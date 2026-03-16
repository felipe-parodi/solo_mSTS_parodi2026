"""Session-level synchronization utilities.

This module provides a `SessionSynchronizer` that loads a per-session
`sync_map.json` and exposes typed, Google-style documented conversions across
timebases (logger ↔ PC ↔ video ↔ audio).

This is the canonical location for SessionSynchronizer. The legacy
data_utils.sync_utils module will re-export from here.

Example:
    >>> from freemat_data.compat.sync_utils import SessionSynchronizer
    >>> sync = SessionSynchronizer("scripts/synchronization/sync/sync_map.json")
    >>> t_vid = sync.m1_to_video([1.0, 2.0, 3.0])
    >>> t_m1  = sync.video_to_m1(t_vid)

Notes:
    - PC↔video uses an anchored affine if available:
      tv = s * (t_pc - pc_flash) + T_vid_flash
      and its inverse.
    - If no slope is present, falls back to a simple shift using T_vid_flash.
    - Audio↔video uses a clapper anchor when present; otherwise LED flash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np


ArrayLike = Union[float, int, Sequence[float], np.ndarray]


def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert a scalar/sequence to 1D numpy array without copying when possible."""
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False).ravel()
    return np.asarray(
        list(x) if isinstance(x, (list, tuple)) else [x], dtype=float
    ).ravel()


def _maybe_scalar(y: np.ndarray, like: ArrayLike) -> Union[float, np.ndarray]:
    """Return a scalar if the input 'like' was scalar; else return the array."""
    if isinstance(like, (float, int)) or (
        isinstance(like, np.ndarray) and like.ndim == 0
    ):
        return float(y.ravel()[0])
    if isinstance(like, (list, tuple)) and len(like) == 1:
        return float(y.ravel()[0])
    return y


@dataclass
class SessionSynchronizer:
    """Session-level synchronizer backed by a sync_map.json.

    Attributes:
        path: Path to the `sync_map.json` for this session.
        map:  Loaded mapping dictionary.

    The mapping format is produced by the synchronization scripts under
    `scripts/synchronization/sync`. This class is resilient to missing
    sub-mappings and raises clear errors when a conversion cannot be performed.
    """

    path: Union[str, Path]
    map: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.reload()

    # ------------------------------- loading
    def reload(self) -> None:
        """Reload sync map from disk."""
        with open(self.path, "r") as f:
            self.map = json.load(f)

    # ------------------------------- helpers
    @property
    def has_m1(self) -> bool:
        return isinstance(self.map, dict) and "m1" in self.map

    @property
    def has_m2(self) -> bool:
        return isinstance(self.map, dict) and "m2" in self.map

    @property
    def has_audio(self) -> bool:
        return isinstance(self.map, dict) and "audio" in self.map

    @property
    def has_video(self) -> bool:
        return isinstance(self.map, dict) and "video" in self.map

    # ------------------------------- pc ↔ video
    def pc_to_video(self, t_pc: ArrayLike) -> Union[float, np.ndarray]:
        """PC seconds-of-day to video seconds.

        Args:
            t_pc: Single time or sequence of times in PC seconds.

        Returns:
            Video time(s) in seconds.
        """
        if not self.has_video:
            raise KeyError("Missing 'video' mapping in sync_map.json")
        v = self.map["video"]
        arr = _to_array(t_pc)
        if "pc_to_video" in v and "pc_flash_sec" in v and "T_vid_flash" in v:
            s = float(v["pc_to_video"].get("slope", 1.0))
            pc0 = float(v["pc_flash_sec"])
            tv0 = float(v["T_vid_flash"])
            out = s * (arr - pc0) + tv0
        elif "T_vid_flash" in v:
            # Legacy: shift only
            tv0 = float(v["T_vid_flash"])
            out = arr + tv0
        else:
            raise KeyError("Video mapping incomplete in sync_map.json")
        return _maybe_scalar(out, t_pc)

    def video_to_pc(self, t_vid: ArrayLike) -> Union[float, np.ndarray]:
        """Video seconds to PC seconds-of-day."""
        if not self.has_video:
            raise KeyError("Missing 'video' mapping in sync_map.json")
        v = self.map["video"]
        arr = _to_array(t_vid)
        if "pc_to_video" in v and "pc_flash_sec" in v and "T_vid_flash" in v:
            s = float(v["pc_to_video"].get("slope", 1.0))
            if s == 0:
                raise ZeroDivisionError("video slope is zero in sync_map.json")
            pc0 = float(v["pc_flash_sec"])
            tv0 = float(v["T_vid_flash"])
            out = (arr - tv0) / s + pc0
        elif "T_vid_flash" in v:
            tv0 = float(v["T_vid_flash"])
            out = arr - tv0
        else:
            raise KeyError("Video mapping incomplete in sync_map.json")
        return _maybe_scalar(out, t_vid)

    # ------------------------------- pc ↔ loggers
    def m1_to_pc(self, t_m1: ArrayLike) -> Union[float, np.ndarray]:
        """Convert m1 logger time to PC seconds."""
        if not self.has_m1:
            raise KeyError("Missing 'm1' mapping in sync_map.json")
        s = float(self.map["m1"]["slope"])
        off = float(self.map["m1"]["offset_sec"])
        arr = _to_array(t_m1)
        out = (arr - off) / s
        return _maybe_scalar(out, t_m1)

    def m2_to_pc(self, t_m2: ArrayLike) -> Union[float, np.ndarray]:
        """Convert m2 logger time to PC seconds."""
        if not self.has_m2:
            raise KeyError("Missing 'm2' mapping in sync_map.json")
        s = float(self.map["m2"]["slope"])
        off = float(self.map["m2"]["offset_sec"])
        arr = _to_array(t_m2)
        out = (arr - off) / s
        return _maybe_scalar(out, t_m2)

    def pc_to_m1(self, t_pc: ArrayLike) -> Union[float, np.ndarray]:
        """Convert PC seconds to m1 logger time."""
        if not self.has_m1:
            raise KeyError("Missing 'm1' mapping in sync_map.json")
        s = float(self.map["m1"]["slope"])
        off = float(self.map["m1"]["offset_sec"])
        arr = _to_array(t_pc)
        out = s * arr + off
        return _maybe_scalar(out, t_pc)

    def pc_to_m2(self, t_pc: ArrayLike) -> Union[float, np.ndarray]:
        """Convert PC seconds to m2 logger time."""
        if not self.has_m2:
            raise KeyError("Missing 'm2' mapping in sync_map.json")
        s = float(self.map["m2"]["slope"])
        off = float(self.map["m2"]["offset_sec"])
        arr = _to_array(t_pc)
        out = s * arr + off
        return _maybe_scalar(out, t_pc)

    # ------------------------------- composed: logger ↔ video
    def m1_to_video(self, t_m1: ArrayLike) -> Union[float, np.ndarray]:
        """Convert m1 logger time directly to video seconds."""
        return self.pc_to_video(self.m1_to_pc(t_m1))

    def m2_to_video(self, t_m2: ArrayLike) -> Union[float, np.ndarray]:
        """Convert m2 logger time directly to video seconds."""
        return self.pc_to_video(self.m2_to_pc(t_m2))

    def video_to_m1(self, t_vid: ArrayLike) -> Union[float, np.ndarray]:
        """Convert video seconds to m1 logger time."""
        return self.pc_to_m1(self.video_to_pc(t_vid))

    def video_to_m2(self, t_vid: ArrayLike) -> Union[float, np.ndarray]:
        """Convert video seconds to m2 logger time."""
        return self.pc_to_m2(self.video_to_pc(t_vid))

    # ------------------------------- audio ↔ video
    def audio_to_video(self, t_aud: ArrayLike) -> Union[float, np.ndarray]:
        """Audio seconds to video seconds.

        Prefers `video.T_vid_clap` if available; otherwise falls back to
        `video.T_vid_flash`. Uses `audio.slope` if provided (defaults to 1.0).
        """
        if not self.has_video:
            raise KeyError("Missing 'video' mapping in sync_map.json")
        aud = self.map.get("audio", {})
        va = self.map["video"]
        t_a_clap = float(aud.get("T_aud_clap", 0.0))
        t_v_ref = float(va.get("T_vid_clap", va.get("T_vid_flash", 0.0)))
        s = float(aud.get("slope", 1.0))
        arr = _to_array(t_aud)
        out = s * (arr - t_a_clap) + t_v_ref
        return _maybe_scalar(out, t_aud)

    def video_to_audio(self, t_vid: ArrayLike) -> Union[float, np.ndarray]:
        """Video seconds to audio seconds."""
        if not self.has_video:
            raise KeyError("Missing 'video' mapping in sync_map.json")
        aud = self.map.get("audio", {})
        va = self.map["video"]
        t_a_clap = float(aud.get("T_aud_clap", 0.0))
        t_v_ref = float(va.get("T_vid_clap", va.get("T_vid_flash", 0.0)))
        s = float(aud.get("slope", 1.0))
        if s == 0:
            raise ZeroDivisionError("audio slope is zero in sync_map.json")
        arr = _to_array(t_vid)
        out = (arr - t_v_ref) / s + t_a_clap
        return _maybe_scalar(out, t_vid)
