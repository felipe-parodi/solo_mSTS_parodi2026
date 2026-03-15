from __future__ import annotations

"""Generic video utilities (timestamps and frame iteration).

These helpers are project-agnostic and support NB03 visual feature extraction
and any downstream tools that need lightweight access to video timing.

Design:
- Prefer PyAV for robust timestamp extraction (PTS→seconds)
- Fall back to OpenCV FPS-derived timestamps when PyAV is unavailable
- Provide a small iterator to traverse consecutive frame pairs with times

Notes:
- All functions fail loudly with clear error messages when prerequisites are
  missing (e.g., OpenCV not installed) or files are unreadable.
"""

from typing import Dict, Iterator, Optional, Tuple
import os
import numpy as np


def video_timestamp_index(video_path: str) -> Tuple[np.ndarray, float]:
    """Return (t_video_s, fps) for the given video file.

    Prefers PyAV PTS timestamps when available; otherwise, uses OpenCV to
    derive timestamps from the FPS property. Times are in seconds on the
    video timeline, starting at the first decoded frame.

    Args:
        video_path: Path to a readable video file.

    Returns:
        t_video_s: (N,) float array of per-frame timestamps in seconds.
        fps: Estimated frame rate in frames per second.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the file cannot be decoded or FPS is invalid.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    # Try PyAV for PTS timestamps
    try:
        import av  # type: ignore

        container = av.open(video_path)
        stream = next((s for s in container.streams.video), None)
        if stream is None:
            raise RuntimeError(f"No video stream found in {video_path}")
        tb = float(stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        times = []
        for idx, frame in enumerate(container.decode(stream)):
            pts = frame.pts
            t = float(pts * tb) if pts is not None else (idx / fps if fps > 0 else None)
            if t is not None:
                times.append(t)
        if not times:
            raise RuntimeError(f"No timestamps decoded for {video_path}")
        t_video_s = np.asarray(times, dtype=float)
        if not np.isfinite(fps) or fps <= 0:
            # Estimate FPS from median dt
            dts = np.diff(t_video_s)
            med = float(np.median(dts)) if dts.size else 0.0
            fps = 1.0 / med if med > 0 else 0.0
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Invalid FPS derived for {video_path}")
        return t_video_s, fps
    except Exception:
        pass  # Fall back to OpenCV

    # OpenCV fallback
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Neither PyAV nor OpenCV available for timestamps; install 'av' or 'opencv-python'"
        ) from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Some containers do not report frame count; derive by walking
    if n <= 0:
        times = []
        idx = 0
        ok, _ = cap.read()
        if ok:
            times.append(0.0)
            idx = 1
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            times.append(idx / fps)
            idx += 1
        cap.release()
        if not times:
            raise RuntimeError(f"Could not decode any frames for {video_path}")
        return np.asarray(times, dtype=float), fps
    # Construct evenly spaced timeline from FPS and frame count
    cap.release()
    t = np.arange(n, dtype=float) / float(fps)
    return t, float(fps)


def write_timestamp_index_csv(video_path: str, out_csv: str) -> Dict[str, float]:
    """Write a frame index → video seconds CSV for a video file.

    Args:
        video_path: Video file to index.
        out_csv: Output CSV path; parents are created if needed.

    Returns:
        Summary dict with keys: n_frames, t_start, t_end, fps.
    """
    t, fps = video_timestamp_index(video_path)
    import csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_video_s"])
        for i, tv in enumerate(t):
            w.writerow([i, f"{float(tv):.9f}"])
    return {
        "n_frames": float(t.size),
        "t_start": float(t[0]) if t.size else float("nan"),
        "t_end": float(t[-1]) if t.size else float("nan"),
        "fps": float(fps),
    }


def iter_frame_pairs(
    video_path: str,
    *,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> Iterator[Tuple[Tuple[np.ndarray, float], Tuple[np.ndarray, float]]]:
    """Yield successive frame pairs with timestamps using OpenCV with optional seeking.

    Args:
        video_path: Path to video file.
        start_s: Optional approximate start time (seconds) on video timeline. Decoder seeks
                 near this time using CAP_PROP_POS_MSEC (or frame index fallback).
        end_s: Optional end time (seconds) to stop iteration when exceeded.

    Yields:
        ((img_prev, t_prev), (img_curr, t_curr)) with t_* in seconds (FPS-derived).

    Raises:
        RuntimeError: If OpenCV is unavailable or the video cannot be opened.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV is required for frame iteration; install opencv-python") from e
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    idx = 0
    # Optional seek
    if start_s is not None and start_s > 0:
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
        except Exception:
            pass
        # Fallback: frame index seek
        try:
            i0 = int(max(0, round(float(start_s) * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
            idx = i0
        except Exception:
            pass
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return
    t_prev = idx / fps
    idx += 1
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        t_cur = idx / fps
        if end_s is not None and t_cur > float(end_s):
            break
        yield (prev, t_prev), (cur, t_cur)
        prev, t_prev = cur, t_cur
        idx += 1
    cap.release()
