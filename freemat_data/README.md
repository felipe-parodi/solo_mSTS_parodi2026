Project‑agnostic data IO for the FreeMat (mid‑STS) ecosystem.

Scope
- Standardized loaders for core data types: spikes, 3D pose, calibration; CSV-based behavioral states; and stubs for LFP and motifs.
- Minimal, composable transforms for spikes and pose (binning, z‑scoring, interpolation, speed, upright orientation).
- Lightweight visualization helpers for common QA/figure panels (rasters, rate heatmaps, XY occupancy, speed traces).

Installation
- Add this repository root to `PYTHONPATH` or ensure notebooks prepend it to `sys.path`.
- Core dependencies: numpy, pyyaml.
- Optional:
  - matplotlib (2D/3D plotting via `freemat_data.viz.mpl`)
  - opencv-python or av (video timestamps in `loaders.video`)
  - scipy (Savitzky–Golay smoothing in transforms; .mat spikes loading)
  - ipywidgets (interactive widgets in notebooks)
  - plotly (optional for 3D demos if desired)

Nonstationarity note
- Data in FreeMat projects are collected under freely moving, naturalistic conditions. Many traditional metrics (e.g., global Fano factor) assume stationarity and can be misleading on long, behaviorally diverse sessions. Prefer local or chunked summaries (e.g., sliding‑window CV2/LV, short‑window Fano) and interpret dispersion measures with caution. Guardrails (minimum blocks, low‑mean filtering) are provided for descriptive plots; quality screening should rely on explicit QC flags (refractory, prolonged silence, drift) rather than Fano alone.

Design principles
- Immutable interface: downstream projects consume data via this package; they do not redefine loading.
- No hidden defaults: do not assume fps or axis conventions — parse from files and metadata.
- Fail loudly: when a file is missing or a schema is unsupported, raise with a clear error.
- Small, testable pieces: simple functions with type hints and docstrings.

Projects using this package
- dual_monkey_fm: dyadic social interactions
- single_fm: single asocial behavior (current focus)
- hipoyo: decoding of 3D pose, motifs, and states
- ibns: inter‑brain LFP coupling

Data types and schemas
- SpikesEvents: event‑coded spikes with `t: float32[E]` (video seconds) and `unit_index: int32[E]`, plus optional `unit_ids`.
- Pose3D: `X[T,J,3]`, `C[T,J]` joint confidence, `fids[T]` absolute video frame IDs, `fps: float`, `up_axis: {'Y','Z'}`.
- LFPTimeSeries, MotifSeries, StateSeries: typed placeholders (TBD schemas).

Multi‑animal future‑proofing
- Current loaders target single‑animal NPZ (`X[T,J,3]`). We will extend to `X[T,N,J,3]` with an `animal` selector or `animal='all'` to return a list of `Pose3D`.
- Downstream code should avoid hard‑coding `J` and instead query `pose.J` and `pose.joint_names`.

API index
- Loaders: `loaders.pose3d`, `loaders.spikes`, `loaders.lfp` (stub), `loaders.motifs` (stub), `loaders.behavior`
- Registry/paths: `Registry`, `load_session_yaml`, `pathing.normalize`, `pathing.data_root`, `pathing.cache_dir`
- Calibration: `calibration.load_calibration_{pkl,json}`, `apply_world_rotation_to_calib`, `apply_center_translation_to_calib`, `apply_upright_from_floor`
- Transforms (pose): `interpolate_pose_linear`, `smooth_pose`, `frame_valid_fraction`, `longest_contiguous_segment`, `compute_center_of_mass`, `compute_speed_from_com`, `compute_derivatives`, `heights_from_floor`, `remove_outlier_frames`
- Transforms (spikes): `bin_events`, `firing_rate`, `zscore_by_unit`
- Sync: `sync.load_syncmap`, `sync.get_syncmap_from_session`, `sync.map_spikes_to_video`
- Visualization: `viz.geometry` (arrays), `viz.mpl` (Matplotlib)

Loaders
- Spikes: `freemat_data.loaders.spikes`
  - `load_spikes_events_npz(npz_path)` parses `spike_t`/`spike_unit_index` or reconstructs from `binned_spikes`+`timestamps`.
  - `load_spikes_from_session(SessionRef)` resolves an NPZ or MAT from the session registry/YAML.
- Pose3D: `freemat_data.loaders.pose3d`
  - `load_pose3d(npz_path)`: robust NPZ meta parsing (fps, joint_names, up_axis); no fps assumptions.
  - `load_pose3d_from_session(SessionRef)`: resolves pose NPZ via registry.
- Calibration: `freemat_data.calibration` loads camera intrinsics/extrinsics JSON/PKL (read‑only, no side effects).
- LFP/Motifs: `freemat_data.loaders.{lfp,motifs}` are stubs. Behavioral states are loaded via a dual-subject CSV loader (`load_behavior_states_from_csv`, `load_behavior_states_from_session`).

Calibration examples
- Apply a world-space center shift (e.g., to center cameras on the enclosure center) while keeping rotations unchanged:

```python
from freemat_data import load_session_yaml, load_calibration_from_session
from freemat_data.calibration import apply_center_translation_to_calib
sess = load_session_yaml('configs/session_example.yaml')
calib = load_calibration_from_session(sess)
center_shift = com_xyz.mean(axis=0)  # e.g., CoM mean over a stable segment
calib_centered = apply_center_translation_to_calib(calib, center_shift)
```

Transforms
- Pose: `freemat_data.transforms.pose`
  - `interpolate_pose_linear` — fills short gaps per joint using confidence‑based masking.
  - `compute_center_of_mass` — weighted joint average and `compute_speed_from_com` — finite‑difference speed (m/s).
  - `apply_upright` — standardize vertical axis; `orient_with_floor` — apply floor_plane upright if available.
- Spikes: `freemat_data.transforms.spikes`
  - `bin_events` — histogram events into `[t_start,t_end)` with bin size `bin_s`.
  - `firing_rate` — counts to Hz; `zscore_by_unit` — per‑unit z‑scoring (NaN‑safe).

Visualization helpers (two layers, one namespace)
- Geometry (renderer-agnostic): `freemat_data.viz.geometry`
  - Plane/UV helpers (`plane_basis_from_dict`, `project_points_to_uv`, `uv_to_xyz`), hex footprint (`load_hex_uv`, `regular_hex_from_uv`), checkerboard meshes, camera gizmo segments. Returns NumPy arrays only.
- Matplotlib renderers: `freemat_data.viz.mpl`
  - 3D scene: `plot_com_trajectory_3d`, `draw_camera_gizmos_mpl`, `plot_hex_outline_mpl`, `set_axes_equal_3d`.
  - 2D QA: `plot_xy_occupancy`, `plot_speed_timeseries`, `plot_valid_fraction`, `plot_pose_traj_xy_xz`, `plot_occupancy_projections_grid`, `plot_event_raster`, `plot_rate_heatmap`, `plot_fr_histogram`.

Notes:
- The top-level `freemat_data` package does not re-export plotting functions to keep imports light.
- Legacy aliases remain available: `freemat_data.vis` (geometry) and `freemat_data.vis_mpl` (MPL renderers).

Interactive rasters (widgets)
- For long sessions, use an `ipywidgets` slider‑driven window. Best practice:
  - Create the figure once off‑screen (`plt.ioff()` + `plt.close(fig)`).
  - Display it inside a single `widgets.Output()` area and redraw in‑place.
  - Keep a module/global state dict to unobserve previous handlers when re‑running cells.
  - Set `continuous_update=False` on sliders to limit callback frequency.

Figure 1 (single_fm) integration
- NB01 writes aligned parquet(s) and optional `pose_upright.npz`/pose features.
- NB02 loads the 1.0 s parquet and uses event rasters when available; otherwise, it reconstructs ticks from rates.
- All panels save PNG and SVG next to the aligned parquet; fps and coordinate conventions come from the NPZ meta.

Notes on centralization
- Historical pipelines (triangulation, motifs, torch_brain examples, pose_derivatives) loaded 3D pose slightly differently.
- This package consolidates pose loading and orientation logic (including `floor_plane.json` upright), and will host interpolation utilities previously prototyped in `triangulation/cli/interpolate_pose_npz.py`.
- Future work: migrate motif and torch_brain loaders to call `freemat_data` and add unit tests for `load_pose3d` and transforms.

Quick start (per‑session YAML)
- Create `configs/session_<id>.yaml` with `pose_3d`, `m1_neural` (or `m2_*_neural`), `syncmap_json`, and optional `calib_pkl`.
- Minimal usage:

```python
import os
from freemat_data import (
    load_session_yaml, load_pose3d_from_session, load_spikes_from_session,
    get_syncmap_from_session, map_spikes_to_video,
    load_calibration_from_session, apply_upright_from_floor,
    load_hex_uv, checker_mesh_from_hex
)

sess = load_session_yaml('configs/session_example.yaml')
pose = load_pose3d_from_session(sess)
spk = load_spikes_from_session(sess)
sync = get_syncmap_from_session(sess)
t_spk_vid = map_spikes_to_video(spk, sync, logger='m1')

calib = load_calibration_from_session(sess)
npz_dir = os.path.dirname(pose.path)
calib_upr, R_upr, plane_upr, _ = apply_upright_from_floor(calib, os.path.join(npz_dir, 'floor_plane.json'))
hex_uv = load_hex_uv(npz_dir)
Vd, Fd, Vl, Fl = checker_mesh_from_hex(plane_upr, hex_uv, cell=0.20)
```

Sync usage
- Synchronization helpers are imported lazily; ensure `data_utils.sync_utils.SessionSynchronizer` is available when calling them.

```python
from freemat_data import load_session_yaml, get_syncmap_from_session, map_spikes_to_video

sess = load_session_yaml('configs/session_example.yaml')
sync = get_syncmap_from_session(sess)
if sync is not None:
    t_spk_video = map_spikes_to_video(spikes, sync, logger='m1')
```

Tips
- Prefer forward‑slash or single‑quoted Windows paths in YAML to avoid escape issues.
- Pose is rotated upright via `floor_plane.json` when present — do not rotate again downstream.
- `floor_plane.json` must live next to the source `pose_3d` NPZ. If the file exists but lacks valid `upright` rotations or a plane normal (`a,b,c`), orientation will raise with a clear error message.
- For Plotly, build `go.Mesh3d` from `(V*,F*)` and draw camera wireframes from `calib_upr` with your renderer of choice.

Testing (recommended)
- Unit tests for `load_pose3d` edge cases (meta parsing, fps scalar/array, up‑axis fallback).
- Integration tests for transforms (interpolation, speed) and viz (raster filtering, heatmap shape) on synthetic data.


Notes:
- In raster plots, look for:
    a. vertical bands: as this indicates recording artifacts & synchronous noise
    b. horizontal gaps: dead periods or unit loss
    c. checkerboard patterns: sorting artifacts
    d. gradual thinning: electrode drift
