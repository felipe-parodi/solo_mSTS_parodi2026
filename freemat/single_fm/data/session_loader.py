"""SingleFMSessionLoader: Single source of truth for aligned single-FM data.

This module solves the time alignment bug where features were computed in
pose-relative time while spikes were in video time, causing Hedy sessions
to have 0 merged rows.

The key insight: pose NPZ files contain video frame IDs (fids) and video_fps.
Video time = fids / video_fps, NOT (fids - fids[0]) / pose_fps.

Usage:
    >>> loader = SingleFMSessionLoader('configs/single_fm/session_hedy_0526.yaml')
    >>> print(f"Video FPS: {loader.video_fps}")  # 40
    >>> print(f"Time window: {loader.time_window}")  # (580, 3000)
    >>> merged = loader.get_aligned_data(bin_s=0.5)
    >>> print(f"Merged rows: {len(merged)}")  # > 0!
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from freemat.single_fm.data.arena import ArenaGeometry

from freemat_data import (
    load_session_yaml,
    load_pose3d_from_session,
)
from freemat.single_fm.spikes import load_with_banks, bin_to_tidy, SpikesBundle
from freemat.single_fm.align import to_video_time
from freemat.single_fm.data.calibration import load_calibration_json, apply_rotation_to_pose
from freemat.single_fm.data.arena import ArenaGeometry as _ArenaGeometry  # noqa: F401


@dataclass
class LoaderDiagnostics:
    """Diagnostic info for debugging alignment issues."""
    session_id: str
    video_fps: float
    pose_fps: float
    time_window: Tuple[float, float]
    spike_time_range: Tuple[float, float]
    pose_video_time_range: Tuple[float, float]
    n_spikes_raw: int
    n_spikes_filtered: int
    n_pose_frames_raw: int
    n_pose_frames_filtered: int
    overlap_range: Optional[Tuple[float, float]]


def _extract_session_date(session_id: str) -> Optional[str]:
    """Extract 6-digit session date (YYMMDD) from session ID.

    Args:
        session_id: Session identifier like 'hooke_0525' or 'hedy_0526'

    Returns:
        6-digit date string like '240525' or None if not found
    """
    patterns = [
        r"(\d{6})",  # Direct 6-digit (240525)
        r"_(\d{4})$",  # _0525 at end -> 240525
        r"_(\d{4})_",  # _0525_ -> 240525
        r"(\d{4})[_-]",  # 0525- -> 240525
    ]

    for pattern in patterns:
        match = re.search(pattern, session_id)
        if match:
            found = match.group(1)
            if len(found) == 6:
                return found
            elif len(found) == 4:
                return "24" + found  # Assume 2024
    return None


def _get_video_fps_from_npz(npz_path: str) -> float:
    """Extract video_fps from pose NPZ file.

    Args:
        npz_path: Path to pose NPZ file

    Returns:
        Video fps (typically 40.0)

    Raises:
        ValueError: If video_fps not found in NPZ
    """
    data = np.load(npz_path, allow_pickle=True)
    if "video_fps" in data:
        return float(data["video_fps"].flat[0])

    # Fallback: infer from fids step
    if "fids" in data:
        fids = data["fids"]
        if len(fids) > 1:
            step = int(np.median(np.diff(fids)))
            # If step is ~8 and pose is 5Hz, video is 40Hz
            fps = data.get("fps", [5.0])
            pose_fps = float(fps.flat[0]) if hasattr(fps, "flat") else float(fps)
            return pose_fps * step if step > 0 else 40.0

    # Conservative default
    return 40.0


class SingleFMSessionLoader:
    """Single source of truth for aligned single-FM data.

    This loader ensures that:
    1. All timestamps use VIDEO TIME (fids / video_fps)
    2. Inference start-stop filtering is applied from TSV
    3. Spikes and features can be merged on t_bin_s

    Attributes:
        session_id: Session identifier
        video_fps: Video frame rate (e.g., 40 Hz)
        pose_fps: Pose output rate (e.g., 5 Hz)
        time_window: Valid inference window (start_s, end_s) in video time
    """

    def __init__(
        self,
        session_yaml: str,
        start_stop_tsv: Optional[str] = None,
        banks_yaml: Optional[str] = None,
        canonical_alignment_path: Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize loader from session YAML.

        Args:
            session_yaml: Path to session YAML config
            start_stop_tsv: Path to inference time filter TSV
                (default: docs/inf_start_stop_times.tsv)
            banks_yaml: Path to banks.yaml for spike bank annotations
            canonical_alignment_path: Path to canonical_alignment.json for
                cross-session arena alignment. When set, Procrustes alignment
                (R, t) is applied to pose after upright rotation, putting all
                sessions into the reference session's coordinate frame.
            debug: Print debug info
        """
        self._session_yaml = Path(session_yaml)
        self._debug = debug
        self._canonical_alignment_path = canonical_alignment_path

        # Resolve default TSV path
        if start_stop_tsv is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            start_stop_tsv = str(repo_root / "docs" / "inf_start_stop_times.tsv")
        self._start_stop_tsv = start_stop_tsv

        # Resolve banks yaml
        if banks_yaml is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            candidate = repo_root / "configs" / "banks.yaml"
            banks_yaml = str(candidate) if candidate.exists() else None
        self._banks_yaml = banks_yaml

        # Load session
        self._session = load_session_yaml(str(session_yaml))
        self.session_id = self._session.session_id

        # Lazy-loaded attributes
        self._pose = None
        self._pose_oriented = None
        self._video_fps: Optional[float] = None
        self._pose_fps: Optional[float] = None
        self._time_window: Optional[Tuple[float, float]] = None
        self._spikes_bundle: Optional[SpikesBundle] = None
        self._t_video_spikes: Optional[np.ndarray] = None
        self._u_index_spikes: Optional[np.ndarray] = None
        self._arena: Optional["ArenaGeometry"] = None

        # Calibration and triangulation config (computed fresh, not from cache)
        self._triang_config: Optional[Dict] = None
        self._calib: Optional[Dict[str, Dict]] = None
        self._up_axis: str = "Y"
        self._floor_level: Optional[float] = None
        self._t_total: Optional[np.ndarray] = None  # Procrustes translation
        self._R_total: Optional[np.ndarray] = None  # Rotation applied to pose

    @property
    def video_fps(self) -> float:
        """Get video fps from pose NPZ (e.g., 40 Hz)."""
        if self._video_fps is None:
            self._load_pose()
        return self._video_fps

    @property
    def pose_fps(self) -> float:
        """Get pose output fps (e.g., 5 Hz)."""
        if self._pose_fps is None:
            self._load_pose()
        return self._pose_fps

    @property
    def time_window(self) -> Tuple[float, float]:
        """Get valid inference window in video seconds."""
        if self._time_window is None:
            self._time_window = self._compute_time_window()
        return self._time_window

    def _load_pose(self) -> None:
        """Load pose and extract video_fps from NPZ."""
        self._pose = load_pose3d_from_session(self._session)
        npz_path = self._pose.path if hasattr(self._pose, "path") else None

        # Get video_fps from NPZ
        if npz_path and Path(npz_path).exists():
            self._video_fps = _get_video_fps_from_npz(npz_path)
        else:
            # Fallback: infer from fids step
            if self._pose.fids is not None and len(self._pose.fids) > 1:
                step = int(np.median(np.diff(self._pose.fids)))
                self._video_fps = (self._pose.fps or 5.0) * step if step > 0 else 40.0
            else:
                self._video_fps = 40.0  # Conservative default

        self._pose_fps = float(self._pose.fps) if self._pose.fps else 5.0

        if self._debug:
            print(f"[Loader] session={self.session_id} video_fps={self._video_fps} pose_fps={self._pose_fps}")
            print(f"[Loader] pose T={self._pose.T} fids=[{self._pose.fids[0]}, {self._pose.fids[-1]}]")

    def _compute_time_window(self) -> Tuple[float, float]:
        """Compute valid inference window from TSV."""
        from inference_time_filter import InferenceTimeFilter

        session_date = _extract_session_date(self.session_id)
        if session_date is None:
            if self._debug:
                print(f"[Loader][warn] Could not extract date from session_id: {self.session_id}")
            # Fallback: use full pose range
            self.get_pose()  # ensure pose is loaded
            t_video = self.pose_video_time()
            return (float(t_video[0]), float(t_video[-1]))

        time_filter = InferenceTimeFilter.from_tsv(self._start_stop_tsv)
        time_range = time_filter.get_range(session_date)

        if time_range is None:
            if self._debug:
                print(f"[Loader][warn] No time range found for {session_date}")
            self.get_pose()  # ensure pose is loaded
            t_video = self.pose_video_time()
            return (float(t_video[0]), float(t_video[-1]))

        t_start = time_range.start_seconds
        t_end = time_range.end_seconds

        if self._debug:
            print(f"[Loader] time_window from TSV: [{t_start:.1f}, {t_end:.1f}]s")

        return (t_start, t_end)

    def get_pose(self):
        """Load raw pose (not floor-oriented)."""
        if self._pose is None:
            self._load_pose()
        return self._pose

    def get_pose_oriented(self):
        """Load pose oriented to floor plane using fresh rotation from calibration.

        Computes rotations fresh from calib.pkl (like viser_pose) instead of
        using cached rotations from floor_plane.json.
        """
        if self._pose_oriented is None:
            self._load_pose_with_fresh_rotation()
        return self._pose_oriented

    @property
    def floor_level(self) -> float:
        """Get floor level derived from bottom camera positions."""
        if self._floor_level is None:
            self._load_pose_with_fresh_rotation()
        return self._floor_level

    @property
    def plane_dict(self) -> Dict[str, float]:
        """Get floor plane equation dict for feature computation.

        Constructed dynamically from up_axis and floor_level.
        Plane equation: ax + by + cz + d = 0.

        Returns:
            Dict with keys 'a', 'b', 'c', 'd'.
        """
        floor_level = self.floor_level
        up_axis = self._up_axis.upper()
        if up_axis == "Y":
            return {'a': 0.0, 'b': 1.0, 'c': 0.0, 'd': -floor_level}
        elif up_axis == "Z":
            return {'a': 0.0, 'b': 0.0, 'c': 1.0, 'd': -floor_level}
        else:  # X
            return {'a': 1.0, 'b': 0.0, 'c': 0.0, 'd': -floor_level}

    def _get_vertical_axis(self) -> int:
        """Return integer index of the vertical (up) axis.

        Returns:
            0 for X, 1 for Y, 2 for Z.
        """
        return {"X": 0, "Y": 1, "Z": 2}.get(self._up_axis.upper(), 1)

    def _resolve_calibration_json_path(self) -> Path:
        """Resolve path to pre-computed calibration JSON for this session.

        Returns:
            Path to configs/calibration_<session_id>.json.

        Raises:
            FileNotFoundError: If calibration JSON does not exist.
        """
        repo_root = Path(__file__).parent.parent.parent.parent
        calib_path = repo_root / "configs" / f"calibration_{self.session_id}.json"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"Calibration JSON not found: {calib_path}. "
                "Run extract_calibration_json.py to generate it."
            )
        return calib_path

    def _load_canonical_alignment(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load this session's Procrustes alignment (R, t) from JSON.

        Matches the session_id to a JSON key via the 4-digit date substring
        (e.g. session_id "single_0526_hedy" matches JSON key "hedy_0526").

        Returns:
            (R_align, t_align) where R_align is (3,3) and t_align is (3,).

        Raises:
            ValueError: If alignment file or session key not found.
        """
        import json

        align_path = Path(self._canonical_alignment_path)
        if not align_path.exists():
            raise FileNotFoundError(
                f"Canonical alignment file not found: {align_path}"
            )

        with open(align_path, "r") as f:
            align_data = json.load(f)

        # Extract 4-digit date from session_id for matching
        date_4 = _extract_session_date(self.session_id)
        if date_4 is not None:
            # date_4 is "24MMDD", we want "MMDD" for matching
            date_suffix = date_4[2:]  # e.g. "0526"
        else:
            date_suffix = None

        # Find matching key in JSON
        matched_key = None
        for key in align_data["sessions"]:
            if date_suffix and date_suffix in key:
                matched_key = key
                break

        if matched_key is None:
            raise ValueError(
                f"Session '{self.session_id}' (date={date_suffix}) not found "
                f"in canonical alignment JSON. Available: {list(align_data['sessions'].keys())}"
            )

        entry = align_data["sessions"][matched_key]
        R_align = np.array(entry["R"], dtype=np.float64)
        t_align = np.array(entry["t"], dtype=np.float64)

        quality = entry.get("quality", "unknown")
        if quality == "poor" and self._debug:
            print(
                f"[Loader][warn] Session {self.session_id} ({matched_key}) has "
                f"poor alignment quality (RMSE={entry['rmse_cm']:.1f}cm)"
            )

        if self._debug:
            print(
                f"[Loader] Canonical alignment: {matched_key}, "
                f"RMSE={entry['rmse_cm']:.2f}cm, quality={quality}"
            )

        return R_align, t_align

    def _load_pose_with_fresh_rotation(self) -> None:
        """Load pose and apply pre-computed rotation from calibration JSON.

        Loads R_total, floor_level, and up_axis from the pre-computed
        calibration JSON instead of deriving them from multical workspace.
        """
        from freemat_data.types import Pose3D

        # Load raw pose
        pose = self.get_pose()

        # Load pre-computed calibration from JSON
        calib_json_path = self._resolve_calibration_json_path()
        calib_data = load_calibration_json(calib_json_path, debug=self._debug)

        R_total = calib_data["R_total"]
        floor_level = calib_data["floor_level"]
        up_axis = calib_data["up_axis"]

        # Apply rotation to pose
        X_rotated = apply_rotation_to_pose(pose.X, R_total)

        self._up_axis = up_axis
        self._floor_level = floor_level
        self._R_total = R_total

        # Apply canonical alignment if configured
        if self._canonical_alignment_path is not None:
            R_align, t_align = self._load_canonical_alignment()
            up_idx = {"X": 0, "Y": 1, "Z": 2}[up_axis]

            X_rotated = apply_rotation_to_pose(X_rotated, R_align, t_align)

            self._floor_level = floor_level + t_align[up_idx]
            self._R_total = R_align @ R_total
            self._t_total = t_align

            if self._debug:
                print(
                    f"[Loader] Canonical alignment applied: "
                    f"floor {floor_level:.4f} -> {self._floor_level:.4f}m"
                )

        # Create new Pose3D with rotated coordinates
        self._pose_oriented = Pose3D(
            X=X_rotated.astype(pose.X.dtype, copy=False),
            C=pose.C,
            fids=pose.fids.copy(),
            fps=pose.fps,
            joint_names=pose.joint_names,
            up_axis=self._up_axis,
            meta=pose.meta,
            path=pose.path,
        )

    def pose_video_time(self) -> np.ndarray:
        """Get pose time axis in VIDEO seconds.

        This is the KEY FIX: use fids / video_fps, NOT (fids - fids[0]) / pose_fps.
        """
        pose = self.get_pose()
        return pose.fids.astype(float) / self.video_fps

    @property
    def arena(self) -> "ArenaGeometry":
        """Get arena geometry (lazy loaded from calibration PKL).

        Raises:
            ValueError: If arena geometry cannot be loaded.
        """
        if self._arena is None:
            self._arena = self._load_arena_geometry()
        return self._arena

    def _load_arena_geometry(self) -> "ArenaGeometry":
        """Load hexagonal arena geometry from pre-computed calibration JSON.

        Raises:
            ValueError: If pose orientation is unavailable.
        """
        # Ensure pose rotation has been applied (sets _floor_level, _up_axis)
        self.get_pose_oriented()

        calib_json_path = self._resolve_calibration_json_path()
        arena = _ArenaGeometry.from_json(calib_json_path)

        if self._debug:
            print(
                f"[Loader] ArenaGeometry loaded: {len(arena.walls)} walls, "
                f"radius={arena.hexagon_radius:.2f}m, up_axis={arena.up_axis}, "
                f"floor={arena.floor_height:.2f}m"
            )

        return arena

    def _load_spikes(self) -> Tuple[np.ndarray, np.ndarray, SpikesBundle]:
        """Load spikes and convert to video time."""
        if self._t_video_spikes is not None:
            return self._t_video_spikes, self._u_index_spikes, self._spikes_bundle

        bundle = load_with_banks(self._session, banks_yaml=self._banks_yaml)
        spikes = bundle.spikes

        # Convert to video time using sync map
        t_video = to_video_time(self._session, spikes)
        u_index = spikes.unit_index.astype(int)

        # Store unit_kind metadata ('MUA' or 'SUA' for each unit)
        self._unit_kind = spikes.meta.get('unit_kind') if spikes.meta else None

        # Store unit_bank metadata ('ventral' or 'dorsal' for each unit)
        self._unit_bank = bundle.unit_bank

        self._t_video_spikes = t_video
        self._u_index_spikes = u_index
        self._spikes_bundle = bundle

        if self._debug:
            print(f"[Loader] spikes: {len(t_video)} events, range=[{t_video.min():.1f}, {t_video.max():.1f}]s")
            if self._unit_kind is not None:
                n_mua = sum(1 for k in self._unit_kind if k == 'MUA')
                n_sua = sum(1 for k in self._unit_kind if k == 'SUA')
                print(f"[Loader] unit_kind: {n_mua} MUA, {n_sua} SUA")

        return t_video, u_index, bundle

    def get_spikes(self, bin_s: float = 0.5) -> pd.DataFrame:
        """Get binned spikes in video time, filtered to valid window.

        Args:
            bin_s: Bin size in seconds

        Returns:
            DataFrame with columns: t, bin_idx, unit_index, fr, fr_z, session_id, bin_s
            where t is the bin CENTER in video seconds.
        """
        t_video, u_index, bundle = self._load_spikes()

        # Filter to valid window
        t_start, t_end = self.time_window
        mask = (t_video >= t_start) & (t_video <= t_end)
        t_filtered = t_video[mask]
        u_filtered = u_index[mask]

        if self._debug:
            print(f"[Loader] spikes after time filter: {len(t_filtered)}/{len(t_video)} events")

        if len(t_filtered) == 0:
            return pd.DataFrame(
                columns=["t", "bin_idx", "unit_index", "fr", "fr_z", "session_id", "bin_s"]
            )

        # Bin spikes
        df = bin_to_tidy(
            t_filtered,
            u_filtered,
            session_id=self.session_id,
            bin_s=bin_s,
            t_start=t_start,
            t_end=t_end,
        )

        # Rename 't' to 't_bin_s' for clarity and merging
        df = df.rename(columns={"t": "t_bin_s"})

        # Add unit_kind column (MUA or SUA) from metadata
        if self._unit_kind is not None and len(self._unit_kind) > 0:
            kind_map = {i: str(k) for i, k in enumerate(self._unit_kind)}
            df['unit_kind'] = df['unit_index'].map(kind_map)

        # Add unit_bank column (ventral or dorsal) from metadata
        if self._unit_bank is not None and len(self._unit_bank) > 0:
            bank_map = {i: str(b) for i, b in enumerate(self._unit_bank)}
            df['unit_bank'] = df['unit_index'].map(bank_map)

        return df

    def get_pose_filtered(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get pose data filtered to valid time window.

        Returns:
            (X, t_video) where:
                X: Pose array (T_filtered, J, 3)
                t_video: Time in video seconds (T_filtered,)
        """
        pose = self.get_pose_oriented()
        t_video = self.pose_video_time()
        t_start, t_end = self.time_window

        mask = (t_video >= t_start) & (t_video <= t_end)
        X_filtered = pose.X[mask]
        t_filtered = t_video[mask]

        if self._debug:
            print(f"[Loader] pose after time filter: {len(t_filtered)}/{len(t_video)} frames")

        return X_filtered, t_filtered

    def _build_features_video_time(
        self,
        bin_s: float = 0.5,
        floor_filter: bool = True,
        floor_height_threshold: float = 0.5,
        floor_filter_before_features: bool = False,
        binning_method: str = 'mean',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build VIS and KIN features using VIDEO time axis.

        Delegates to :func:`feature_builder.build_features_video_time` for
        the actual computation.  See that function for full documentation.

        Args:
            bin_s: Bin size in seconds.
            floor_filter: If True, filter to on-floor bins.
            floor_height_threshold: Height threshold in meters (default 0.5m).
            floor_filter_before_features: If True, NaN-mask off-floor frames
                before feature computation.
            binning_method: 'mean' or 'interp'.

        Returns:
            (vis_df, kin_df) DataFrames with features.
        """
        from freemat.single_fm.data.feature_builder import (
            FeatureBuildContext,
            build_features_video_time,
        )

        pose = self.get_pose_oriented()
        ctx = FeatureBuildContext(
            session_id=self.session_id,
            time_window=self.time_window,
            floor_level=self.floor_level,
            up_axis=self._up_axis,
            pose_fps=float(pose.fps or 5.0),
            debug=self._debug,
        )
        return build_features_video_time(
            pose.X,
            self.pose_video_time(),
            self.arena,
            ctx,
            bin_s=bin_s,
            floor_filter=floor_filter,
            floor_height_threshold=floor_height_threshold,
            floor_filter_before_features=floor_filter_before_features,
            binning_method=binning_method,
        )

    def get_aligned_data(
        self,
        bin_s: float = 0.5,
        floor_filter: bool = True,
        floor_height_threshold: float = 0.5,
        floor_filter_before_features: bool = False,
        binning_method: str = 'mean',
        min_r2_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Get merged spikes + features, ready for encoding.

        Args:
            bin_s: Bin size in seconds.
            floor_filter: If True, filter to on-floor bins.
            floor_height_threshold: Height threshold in meters (default 0.5m).
            floor_filter_before_features: If True, NaN-mask off-floor frames
                in pose BEFORE feature computation. If False (default), compute
                features on full timeline then NaN-mask.
            binning_method: 'mean' (average frames in bin) or 'interp'
                (linear interpolation at bin center).
            min_r2_threshold: Minimum R2 threshold (not used here, for API compat).

        Returns:
            Merged DataFrame with spike rates and features aligned on t_bin_s.
            Only includes bins where animal was on floor (if floor_filter=True).
        """
        spikes_df = self.get_spikes(bin_s)
        vis_df, kin_df = self._build_features_video_time(
            bin_s,
            floor_filter=floor_filter,
            floor_height_threshold=floor_height_threshold,
            floor_filter_before_features=floor_filter_before_features,
            binning_method=binning_method,
        )

        if len(spikes_df) == 0:
            if self._debug:
                print("[Loader][warn] No spikes in time window")
            return pd.DataFrame()

        if len(vis_df) == 0 or len(kin_df) == 0:
            if self._debug:
                print("[Loader][warn] No features in time window")
            return pd.DataFrame()

        # Round t_bin_s for merging (handle floating point)
        decimals = max(0, -int(np.floor(np.log10(bin_s))) + 1)
        spikes_df["t_bin_s"] = spikes_df["t_bin_s"].round(decimals)
        vis_df["t_bin_s"] = vis_df["t_bin_s"].round(decimals)
        kin_df["t_bin_s"] = kin_df["t_bin_s"].round(decimals)

        # Merge features first (drop on_floor from kin_df since it's redundant)
        features_df = vis_df.merge(
            kin_df.drop(columns=["session_id", "on_floor"], errors="ignore"),
            on="t_bin_s",
            how="inner",
        )

        if self._debug:
            print(f"[Loader] features merged: {len(features_df)} rows")

        # Merge with spikes
        merged = spikes_df.merge(
            features_df.drop(columns=["session_id"], errors="ignore"),
            on="t_bin_s",
            how="inner",
        )

        if self._debug:
            print(f"[Loader] final merged: {len(merged)} rows")
            if len(merged) == 0:
                print(f"[Loader] DEBUG: spikes t_bin_s range: [{spikes_df['t_bin_s'].min():.1f}, {spikes_df['t_bin_s'].max():.1f}]")
                print(f"[Loader] DEBUG: features t_bin_s range: [{features_df['t_bin_s'].min():.1f}, {features_df['t_bin_s'].max():.1f}]")

        return merged

    def get_diagnostics(self) -> LoaderDiagnostics:
        """Get diagnostic info for debugging alignment issues."""
        t_video_spikes, u_index, bundle = self._load_spikes()
        t_video_pose = self.pose_video_time()
        t_start, t_end = self.time_window

        # Filter counts
        spike_mask = (t_video_spikes >= t_start) & (t_video_spikes <= t_end)
        pose_mask = (t_video_pose >= t_start) & (t_video_pose <= t_end)

        # Compute overlap
        spike_range = (float(t_video_spikes.min()), float(t_video_spikes.max()))
        pose_range = (float(t_video_pose.min()), float(t_video_pose.max()))

        overlap_start = max(spike_range[0], pose_range[0], t_start)
        overlap_end = min(spike_range[1], pose_range[1], t_end)
        overlap = (overlap_start, overlap_end) if overlap_start < overlap_end else None

        return LoaderDiagnostics(
            session_id=self.session_id,
            video_fps=self.video_fps,
            pose_fps=self.pose_fps,
            time_window=(t_start, t_end),
            spike_time_range=spike_range,
            pose_video_time_range=pose_range,
            n_spikes_raw=len(t_video_spikes),
            n_spikes_filtered=int(spike_mask.sum()),
            n_pose_frames_raw=len(t_video_pose),
            n_pose_frames_filtered=int(pose_mask.sum()),
            overlap_range=overlap,
        )

    def print_diagnostics(self) -> None:
        """Print diagnostic info."""
        diag = self.get_diagnostics()
        print(f"\n{'='*60}")
        print(f"SingleFMSessionLoader Diagnostics: {diag.session_id}")
        print(f"{'='*60}")
        print(f"Video FPS: {diag.video_fps}")
        print(f"Pose FPS: {diag.pose_fps}")
        print(f"Time window (from TSV): [{diag.time_window[0]:.1f}, {diag.time_window[1]:.1f}]s")
        print(f"Spike time range (video): [{diag.spike_time_range[0]:.1f}, {diag.spike_time_range[1]:.1f}]s")
        print(f"Pose time range (video): [{diag.pose_video_time_range[0]:.1f}, {diag.pose_video_time_range[1]:.1f}]s")
        print(f"Spikes: {diag.n_spikes_filtered}/{diag.n_spikes_raw} after filter")
        print(f"Pose frames: {diag.n_pose_frames_filtered}/{diag.n_pose_frames_raw} after filter")
        if diag.overlap_range:
            duration = diag.overlap_range[1] - diag.overlap_range[0]
            print(f"Overlap range: [{diag.overlap_range[0]:.1f}, {diag.overlap_range[1]:.1f}]s ({duration:.1f}s)")
        else:
            print("Overlap range: NONE - data cannot be merged!")
        print(f"{'='*60}\n")


__all__ = [
    "SingleFMSessionLoader",
    "LoaderDiagnostics",
]
