from __future__ import annotations

"""Small utilities for working with Pose3D, spikes, alignment, and JSON safety.

This module collects lightweight helpers that are reused across projects. It
keeps math/data logic here and defers any plotting to dedicated modules.
"""

import os
import json
import math
from typing import Any, Optional, Tuple, List, Mapping

import numpy as np

from .types import Pose3D, SessionRef
from .calibration import rotation_from_floor_json


def pose_time_seconds(pose: Pose3D) -> np.ndarray:
    """Return per-frame video time in seconds using (fids - fids[0]) / fps.

    Triangulation exports 1-based absolute video frame IDs. Converting to a
    zero-based time axis anchored at the first pose frame makes it easy to
    align spikes mapped to video seconds.
    """
    fps = float(pose.fps or 0.0)
    if fps <= 0.0:
        # Fallback to even sampling if fps is missing
        return np.arange(pose.T, dtype=np.float32)
    f0 = int(pose.fids[0]) if pose.fids.size else 1
    return (pose.fids.astype(float) - float(f0)) / fps


def orient_pose_upright(pose: Pose3D, prefer: str = "Z") -> Pose3D:
    """Return a new Pose3D with vertical axis standardized.

    If `pose.up_axis` differs from `prefer` ('Z' default), swap Y/Z so that
    downstream code can assume Z-up coordinates. This is a minimal correction
    and does not apply any additional calibration-based rotations.
    """
    prefer = str(prefer).strip().upper() or "Z"
    if pose.up_axis.strip().upper() == prefer:
        return pose
    X = pose.X.copy()
    # Swap Y and Z to convert Y-up to Z-up (or vice-versa)
    X = X[:, :, [0, 2, 1]]
    new_axis = prefer
    return Pose3D(
        X=X,
        C=pose.C,
        fids=pose.fids.copy(),
        fps=pose.fps,
        joint_names=pose.joint_names,
        up_axis=new_axis,
        meta=pose.meta,
        path=pose.path,
    )


def _rotation_align_vector_to_axis(v: np.ndarray | list[float], up_axis: str = "Z") -> np.ndarray:
    """Minimal rotation aligning vector ``v`` to the world up-axis.

    Args:
        v: 3-vector to align.
        up_axis: Target up-axis label ('Y' or 'Z').

    Returns:
        3x3 rotation matrix.
    """
    v = np.asarray(v, dtype=np.float64).reshape(3)
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return np.eye(3)
    v = v / nv
    target = np.array([0.0, 1.0, 0.0]) if str(up_axis).upper() == "Y" else np.array([0.0, 0.0, 1.0])
    c = float(np.dot(v, target))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - v * float(np.dot(axis, v))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
        )
        return np.eye(3) + 2 * K @ K
    axis = np.cross(v, target)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.linalg.norm(np.cross(v, target))
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    return np.eye(3) + K + K @ K * ((1 - c) / (s**2 + 1e-12))


def orient_pose_with_floor(pose: Pose3D, prefer: str = "upright") -> Tuple[Pose3D, Optional[str]]:
    """Orient pose using ``floor_plane.json`` if present next to the NPZ.

    Behavior:
    - If the JSON has upright.R_align/R_yaw, applies ``R = R_yaw @ R_align``.
    - Else, if a plane is provided with normal (a,b,c), rotates minimally to
      align the plane normal to the reported ``up_axis`` (defaults to 'Z').
    - If the JSON is missing, returns the original pose.

    Args:
        pose: Input Pose3D with ``path`` pointing to its NPZ.
        prefer: Unused (reserved for future policy hooks); kept for API
            compatibility.

    Returns:
        Tuple of (pose_oriented, up_axis_used_or_None).
    """
    # Resolve floor_plane.json next to the NPZ path
    base = os.path.dirname(os.path.abspath(pose.path)) if pose.path else ""
    floor_path = os.path.join(base, "floor_plane.json")
    if not (floor_path and os.path.isfile(floor_path)):
        return pose, None

    # Parse upright rotation/plane info centrally
    R_upr, plane_upr, up_axis = rotation_from_floor_json(floor_path)

    R = None
    if R_upr is not None:
        R = np.asarray(R_upr, dtype=float).reshape(3, 3)
    else:
        # Validate plane metadata strictly if present
        if isinstance(plane_upr, dict):
            has_norm = all(k in plane_upr for k in ("a", "b", "c"))
            if not has_norm:
                raise ValueError(
                    "floor_plane.json present but missing plane normal keys 'a','b','c'"
                )
            n = np.array(
                [
                    float(plane_upr["a"]),
                    float(plane_upr["b"]),
                    float(plane_upr["c"]),
                ],
                dtype=float,
            )
            if not np.isfinite(n).all() or float(np.linalg.norm(n)) < 1e-9:
                raise ValueError("floor_plane.json contains invalid plane normal values")
            R = _rotation_align_vector_to_axis(n, up_axis or "Z")

    if R is None:
        # File exists but neither upright nor valid plane provided → fail loudly
        raise ValueError(
            "floor_plane.json present but missing upright rotations and plane; cannot orient pose"
        )

    X = (np.asarray(R, dtype=float) @ pose.X.reshape(-1, 3).T).T.reshape(pose.X.shape)
    out = Pose3D(
        X=X.astype(pose.X.dtype, copy=False),
        C=pose.C,
        fids=pose.fids.copy(),
        fps=pose.fps,
        joint_names=pose.joint_names,
        up_axis=(up_axis or pose.up_axis or "Z"),
        meta=pose.meta,
        path=pose.path,
    )
    return out, (up_axis or "Z")


def times_to_frame_index(t_video: np.ndarray, pose: Pose3D) -> np.ndarray:
    """Map video times (s) to nearest pose frame indices.

    Uses the pose time axis computed from fids/fps and clamps to [0, T-1].
    """
    t_pose = pose_time_seconds(pose)
    # searchsorted on midpoints for nearest
    # Construct midpoints between pose times to choose nearest frame
    mids = (t_pose[1:] + t_pose[:-1]) / 2.0
    idx = np.searchsorted(mids, t_video).astype(int)
    idx = np.clip(idx, 0, pose.T - 1)
    return idx


def spike_histogram(t_spk_video: np.ndarray, t_start: float, t_end: float, bin_s: float = 1.0):
    """Histogram spike times (video seconds) into fixed-width bins.

    Returns (edges, counts) where counts[i] corresponds to [edges[i], edges[i+1]).
    """
    if t_end <= t_start:
        t_end = t_start + max(bin_s, 1.0)
    edges = np.arange(t_start, t_end + bin_s, bin_s, dtype=float)
    h, e = np.histogram(t_spk_video, bins=edges)
    return e, h


def infer_logger(session: SessionRef) -> str:
    """Infer default logger key ('m1' or 'm2') from session config.

    Precedence:
      1) Explicit metadata 'logger' (e.g., 'm1' or 'm2')
      2) Presence of data_files keys (exactly one of m1_*_neural or m2_*_neural)
      3) Fallback to 'm1'
    """
    # 1) Explicit metadata override
    meta_logger = (session.metadata or {}).get('logger')
    if isinstance(meta_logger, str) and meta_logger.strip().lower() in {'m1', 'm2'}:
        return meta_logger.strip().lower()
    # 2) Infer from configured file keys
    keys = list(session.data_files.keys())
    has_m1 = any(k.lower().startswith('m1_') and k.lower().endswith('_neural') for k in keys)
    has_m2 = any(k.lower().startswith('m2_') and k.lower().endswith('_neural') for k in keys)
    if has_m1 and not has_m2:
        return 'm1'
    if has_m2 and not has_m1:
        return 'm2'
    # 3) Default
    return 'm1'


def infer_subject(session: SessionRef) -> Optional[str]:
    """Infer subject name ('M2' or 'M1') from session metadata or paths.

    Precedence:
      1) Explicit metadata 'subject'
      2) Heuristic: scan data_files values and session_id for tokens
      3) None if unknown
    """
    subj = (session.metadata or {}).get('subject')
    if isinstance(subj, str) and subj.strip():
        s = subj.strip().lower()
        if s in {"M2", "M1"}:
            return s
    hay = " ".join([str(session.session_id), *(str(v) for v in (session.data_files or {}).values() if v)])
    low = hay.lower()
    if "M2" in low:
        return "M2"
    if "M1" in low:
        return "M1"
    return None


def video_time_to_pose_axis(t_video: np.ndarray, pose: Pose3D) -> np.ndarray:
    """Shift absolute video times to the pose-centered axis used by pose_time_seconds.

    pose_time_seconds(pose) uses (fids - f0) / fps, i.e., it is anchored at the
    first pose frame. Sync mappings (logger→video) typically return video seconds
    on the absolute video timeline. To compare or bin against t_pose, shift
    t_video by f0/fps.
    """
    if pose is None or pose.fps <= 0 or pose.fids.size == 0:
        return t_video
    # Align absolute video seconds to pose-centered axis by subtracting first pose frame time.
    offset = float(pose.fids[0]) / float(pose.fps)
    return (t_video - offset).astype(float, copy=False)


def json_sanitize(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types.

    - numpy scalars -> Python scalars via .item()
    - numpy arrays -> lists via .tolist()
    - dict/list/tuple -> recurse
    - other objects are returned as-is
    """
    import numpy as _np
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [json_sanitize(x) for x in obj]
        return t if isinstance(obj, list) else tuple(t)
    return obj


def append_manifest_row(path: str, headers: List[str], row: List[Any]) -> None:
    """Append a row to a CSV manifest, writing headers if file is new.

    Args:
        path: CSV file path to write/append.
        headers: Column headers to ensure in the manifest.
        row: Row values in the same order as `headers`.

    Notes:
        - Creates parent directory if needed.
        - Does not deduplicate; callers should manage uniqueness.
    """
    import csv
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerow([json_sanitize(x) for x in row])


# --- Null generation utilities for continuous time series ---

def _block_boundaries(T: int, block_len: int) -> List[Tuple[int, int]]:
    """Return non-overlapping [start, end) block boundaries covering 0..T.

    Drops a trailing short block shorter than block_len to keep block sizes
    comparable (common for null generation).
    """
    T = int(T)
    block_len = max(1, int(block_len))
    n_full = T // block_len
    bounds: List[Tuple[int, int]] = []
    for k in range(n_full):
        s = k * block_len
        e = s + block_len
        bounds.append((s, e))
    return bounds


def block_permutation_indices(T: int, *, block_len: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Permutation indices that randomly permute non-overlapping blocks.

    Args:
        T: Length of the time series.
        block_len: Block length in samples (choose > max lag in samples).
        rng: Optional RandomState for reproducibility.

    Returns:
        idx: (T,) integer indices such that x[idx] is block-permuted.
             Trailing samples not forming a full block are dropped.
    """
    if rng is None:
        rng = np.random.RandomState()
    bounds = _block_boundaries(T, block_len)
    if not bounds:
        return np.arange(T, dtype=int)
    order = np.arange(len(bounds))
    rng.shuffle(order)
    pieces = []
    for k in order:
        s, e = bounds[k]
        pieces.append(np.arange(s, e, dtype=int))
    idx = np.concatenate(pieces, axis=0)
    return idx


def within_block_scramble_indices(T: int, *, block_len: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Permutation indices that shuffle time within each non-overlapping block.

    Args:
        T: Length of the time series.
        block_len: Block length in samples.
        rng: Optional RandomState.

    Returns:
        idx: (T,) integer indices with within-block shuffling; trailing
             samples not forming a full block are dropped.
    """
    if rng is None:
        rng = np.random.RandomState()
    bounds = _block_boundaries(T, block_len)
    if not bounds:
        return np.arange(T, dtype=int)
    pieces = []
    for s, e in bounds:
        r = np.arange(s, e, dtype=int)
        rng.shuffle(r)
        pieces.append(r)
    idx = np.concatenate(pieces, axis=0)
    return idx


def apply_permutation_to_columns(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Apply a time-index permutation to each column of 2D array X.

    Args:
        X: (T, D) matrix.
        idx: (K,) indices into first axis; if K < T, the output is (K, D).

    Returns:
        X_perm: X[idx, :].
    """
    X = np.asarray(X)
    return X[idx, ...]


def segments_to_bin_labels(
    segments: np.ndarray,
    labels: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    bin_s: float,
    label_map: Optional[Mapping[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign a label to each fixed-width time bin from labeled segments.

    Conventions:
    - Bins are half-open intervals [t, t+bin_s); returned `t_bins` are bin
      start times. A segment [t0, t1] covers all bins with t ∈ [t0, t1).
    - Non-finite starts are ignored; non-finite ends are treated as `t_max`.
    - When multiple segments overlap a bin, later rows take precedence.

    Args:
        segments: (N, 2) array of [start, end] seconds.
        labels: (N,) labels per segment.
        t_min: Start time for binning (seconds).
        t_max: End time for binning (seconds).
        bin_s: Bin size (seconds), must be > 0.
        label_map: Optional map applied to `str(label).lower().strip()`.

    Returns:
        t_bins: (K,) bin start times.
        labels_per_bin: (K,) labels (empty string for unlabeled bins).
    """
    if not (bin_s > 0.0):
        raise ValueError("bin_s must be > 0")
    if not (t_max > t_min):
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)

    segs = np.asarray(segments, dtype=float).reshape(-1, 2)
    lbls = np.asarray(labels, dtype=object).reshape(-1)
    if segs.shape[0] != lbls.shape[0]:
        raise ValueError("segments and labels must have matching lengths")

    t_bins = np.arange(float(t_min), float(t_max), float(bin_s), dtype=float)
    K = int(t_bins.size)
    if K == 0:
        return t_bins, np.zeros((0,), dtype=object)

    out = np.full(K, "", dtype=object)

    for i in range(segs.shape[0]):
        t0 = float(segs[i, 0])
        t1 = float(segs[i, 1])
        if not np.isfinite(t0):
            continue
        if not np.isfinite(t1):
            t1 = float(t_max)
        # Clip to domain
        a = max(t0, float(t_min))
        b = min(t1, float(t_max))
        if b <= a:
            continue
        raw = str(lbls[i]).strip().lower()
        lab = label_map.get(raw, raw) if label_map is not None else raw
        mask = (t_bins >= a) & (t_bins < b)
        out[mask] = lab

    return t_bins, out


def contiguous_bout_ids(labels_per_bin: np.ndarray) -> np.ndarray:
    """Return per-bin bout IDs that increment on label changes/gaps.

    Unlabeled bins receive bout_id = -1. Labeled runs receive 0,1,2,... in
    order of appearance.

    Args:
        labels_per_bin: (K,) array of labels (strings).

    Returns:
        bout_id: (K,) int array with -1 for unlabeled.
    """
    lbl = np.asarray(labels_per_bin, dtype=object).reshape(-1)
    K = int(lbl.size)
    if K == 0:
        return np.zeros((0,), dtype=int)
    out = np.full(K, -1, dtype=int)
    prev = None  # type: Optional[str]
    on = False
    bid = -1
    for i in range(K):
        s = str(lbl[i]).strip()
        if not s:
            out[i] = -1
            on = False
            prev = None
            continue
        if (not on) or (prev != s):
            bid += 1
        out[i] = bid
        on = True
        prev = s
    return out
