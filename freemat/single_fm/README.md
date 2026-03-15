Single‑FM (single‑monkey) project

Mission
- Understand how macaque mid‑STS encodes self‑motion during unconstrained, freely moving behavior. Single_fm provides a clean, single‑animal foundation to design and validate kinematic (KIN) and visual (VIS) covariates before scaling to social contexts (dual_monkey).

Note: All data IO comes from `freemat_data` (project‑agnostic). Notebooks add analysis/QA and write results.

Coordinate convention
- Single‑FM enforces Z‑up in all downstream visualizations and analyses. If triangulation outputs are upright Y‑up, the loader converts pose and the floor plane to Z‑up once on load so heights and centering are consistent.

YOUR JOB AS AN AGENT IS TO BUILD OUT THE MODULES THAT I THE RESEARCHER TESTS STEP BY STEP IN JUPYTER NOTEBOOKS.

Purpose
- Load spikes + 3D pose via `freemat_data`, align to video seconds using `syncmap_json`, verify upright orientation from `floor_plane.json`, and write aligned parquets + QC for downstream figures.

Docs
- Project plan: `docs/projects/single_fm/analyses.md`
- Data IO: `freemat_data/README.md`
- Session configs: `configs/single_fm/*.yaml`

Step-A: Looking vs Movement
- CLI:
  - `python -m freemat.single_fm.encoding_stepA --neural neural.parquet --pose pose.parquet`  
    Optional: `--outdir results` (default), `--s0q 0.15`, `--smooth_win_s 0.05`.
  - Or derive pose geometry on-the-fly from an upright NPZ:  
    `python -m freemat.single_fm.encoding_stepA --neural neural.parquet --pose-npz results/single_fm/<sid>/pose_processed.npz`
    - When using `--pose-npz`, the CLI also saves the derived pose parquet to the chosen `--outdir` as `pose_from_npz_<sid>.parquet`.
- Inputs:
  - `neural.parquet` with columns: `session_id, unit_id, bank, t_bin_s, fr`.
  - `pose.parquet` with columns: `session_id, t_bin_s, P_x,P_y,P_z, fhead_x,fhead_y,fhead_z, n_x,n_y,n_z`.  
    If not provided, you may pass `--pose-npz` and the CLI will compute these per-bin from the NPZ + floor plane.
- What it does:
  - Computes KIN‑lite features per session (light smoothing on COM, central diffs,
    stable plane basis, unwrapped angles, Fourier(theta_look)).
  - Builds 5‑fold blocked CV per session (leakage‑safe scaling).
  - Runs three GAMs per unit: M1 (look while still), M2 (move unique vs look), M3 (alignment).
- Outputs:
  - `results/stepA_metrics.parquet` with per‑unit metrics:
    `unit_id, bank, r2_look_still, r2_move_plus_look, r2_look_all, move_unique, r2_align`.
  - Figures (PNG+SVG) under `results/stepA_figs/` for the top 6 units by `move_unique`:
    theta tuning (M1), partial for `v_parallel` (M2), partial for `align_cos` (M3), and yaw‑rate vs FR scatter.
- Console summary:
  - Overall and per‑bank % look‑tuned (r2_look_still>0.01),
    median [95% CI bootstrap] for `r2_look_still`, `move_unique`, `r2_align`.
- Notes:
  - Conservative defaults: light smoothing (50 ms), NaN propagation, winsorization and
    per‑session scaling within CV folds, blocked CV. No VIS features in this pass.
  - When deriving pose from NPZ, head‑forward uses a robust estimator with in‑plane norm gating (>=0.05) and edge fill to handle brief gaps.
  - Hooks for Step‑B/C (VIS proxy and minimal image control) are left commented in the CLI.

Pose Decoding (ridge)
- CLI:
  - `python freemat/single_fm/decode_pose.py --npz data/session01.npz --neural_parquet results/single_fm/aligned/session_001_bin0p025.parquet --outdir results/single_fm/decoding/session01 --bin_hz 40 --history_s 1.0 --lags_ms -300:300:20 --k_pca 12 --cv_folds 5 --purge_s 1.0 --alpha_grid -3:3:13 --null_perm 50 --banks dorsal,ventral,both`
- Expected inputs
  - Pose NPZ: `X[T,J,3]`, optional `C[T,J]`, `fps: float`, `joint_names` (COCO‑17 order or `J0..J16`), `up_axis` in meta.
  - Neural parquet: tidy rows with `t_bin_s, unit_id, bank, fr` (Hz). Bank mapping is normalized via `configs/banks.yaml` when needed.
- Targets
  - JOINTS: root‑centered positions flattened to `(J-1)*3`.
  - VELOCITIES: central differences on smoothed positions.
  - POSE_PCS: PCA fit on training joint positions; decoder trains on PC scores and reports reconstruction metrics.
- Modeling
  - X: spike history design using causal windows (`history_s`) built with stride‑tricks (no Python loops).
  - Hyperparameters: nested selection for global lag (bins from `--lags_ms`) and ridge `alpha` (logspace grid), per family and per bank.
  - CV: chronological folds (`--cv_folds`) over valid frames with `--purge_s` bins dropped around boundaries.
  - Banks: runs for `{dorsal, ventral, both}`; optional unit matching and learning curves are supported (kept minimal in v1).
- Metrics
  - Primary: out‑of‑sample CV R² per joint‑dim and overall; JOINTS adds MPJPE; VELOCITIES/JOINTS add Pearson r per dim.
  - Nulls: time‑shift within folds to produce null R² distribution (controlled by `--null_perm`).
- Outputs
  - `outdir/{bank}/{family}/metrics.json` (overall R², alpha, lag, per‑dim R², nulls), `y_true.npy`, `y_pred.npy`, `r2_per_dim.npy`.
  - `outdir/run.json` summarizing args, pose meta, and per‑bank/family metrics.
  - Plots: per‑dim R² bars, lag curve (R² vs lag), null violin. Additional diagnostics can be extended.
  - Optionally saves `pca.pkl` for POSE_PCS.
  

Prerequisites
- Ensure the repo root is on `PYTHONPATH` (the notebooks add it automatically) so `freemat_data` imports cleanly.
- Install requirements: `pip install -r requirements.txt`.
- Optional deps:
  - `matplotlib` for 2D/3D plots via `freemat_data.viz.mpl` (recommended)
  - `opencv-python` or `av` for video timestamps in NB03
  - `plotly` if you want interactive 3D (optional)
- Prepare a per‑session YAML under `configs/single_fm/` (see below).

Layout
- `notebooks/`
  - `01_load_qc_align.ipynb` — data IO, QA, upright pose viz (interactive Plotly), writes aligned parquets (1.0 s, 0.5 s) and optional `pose_upright.npz`.
  - `02_figure1_overview.ipynb` — dataset overview panels (occupancy, speed, exemplar rasters).
 - `03_visual_covariates.ipynb` — head‑anchored ROI optic flow + motion‑energy PCs (VIS features).
 - `03b_kin_covariates.ipynb` — head‑frame kinematic features (KIN) with lags, aligned to the chosen bin grid.
- `04_gam_modeling.ipynb` — GAM modeling of neural firing vs KIN/VIS covariates with blocked CV and nulls.

Visualization modules (use these in notebooks)
- Geometry: `from freemat_data.viz.geometry import load_hex_uv, checker_mesh_from_hex, camera_gizmo_segments`
- Matplotlib: `from freemat_data.viz.mpl import plot_com_trajectory_3d, draw_camera_gizmos_mpl, plot_hex_outline_mpl, set_axes_equal_3d`

KIN pipeline (head‑frame)
- CLI example:
  - `python -m freemat.single_fm.pipeline.kin_cli --session-yaml configs/single_fm/session_001.yaml --bin-s 0.05 --smooth-win-s 1.0 --lags-ms -800:800`
- Outputs under `results/single_fm/<sid>/kin_features/`:
  - `kin_features_<sid>.parquet` (base features at τ=0)
  - `kin_features_lagged_<sid>.parquet` (long‑form table with `lag_ms`)
  - `kin_qc_<sid>.json`
  - Defaults: head‑only frame, 1 s smoothing, POS excluded, per‑session z‑scoring enabled.
  - Diagnostics (PNG+SVG): timeseries, histograms, velocity‑plane hexbin, autocorrelation (disable with `--no-diagnostics`).

Upright policy
- Pose orientation uses `floor_plane.json` next to the pose NPZ. If this file exists but lacks upright matrices and a valid plane normal (`a,b,c`), orientation raises with a clear error. If absent, pose is returned unchanged.
- `results/single_fm/` — per‑session outputs:
  - `aligned/session_<id>_bin{1p0,0p5}.parquet`
  - `<id>/qc_<id>.json`, `<id>/session.yaml`, optional `<id>/pose_upright.npz`
  - `<id>/pose_processed.npz` — processed 3D pose (upright, centered, trimmed to longest valid segment, interpolated, smoothed)

Quickstart (NB01 modules only)
- Resolve session: `from freemat.single_fm.core.session import resolve_session`
- Load spikes (+optional banks) and map to video seconds:
  - `from freemat.single_fm.spikes import load_with_banks, qc_summary`
  - `from freemat.single_fm.align import to_video_time, bin_aligned, merge_bins_with_pose`
- Load + process pose (upright + pipeline):
  - `from freemat.single_fm.pose import load_and_process`
- Save outputs + manifest:
  - `from freemat.single_fm.io.save import save_aligned_parquet, save_qc_json, copy_session_yaml, update_manifest_row`
- Optional animation (HTML/MP4):
  - `from freemat.single_fm.viz.animation import AnimOptions, build_com_fr_animation, save_animation`

Example skeleton (per session):
```python
sess = resolve_session(session_yaml='configs/single_fm/session_<id>.yaml')

# Spikes → map → QC → bin
spk = load_with_banks(sess, banks_yaml='configs/banks.yaml')
t_evt = to_video_time(sess, spk.spikes)
per_unit, summary = qc_summary(t_evt, spk.spikes.unit_index)
aligned = bin_aligned(sess, spk.spikes, session_id=sess.session_id, bins=(1.0, 0.5), keep_units=spk.keep_units)

# Pose → process (requires floor_plane.json and hex footprint next to pose NPZ)
pose_proc = load_and_process(sess)

# Merge (1.0 s) for quick diagnostics and animation inputs
df1 = aligned[1.0]
bins_pose = merge_bins_with_pose(df1, pose_proc.pose)

# Save aligned + QC + YAML and update manifest
save_aligned_parquet(df1, f'results/single_fm/aligned/session_{sess.session_id}_bin1p0.parquet',
                     metadata={'session_id': sess.session_id, 'bin_s': 1.0})
save_qc_json({'spikes': summary}, f'results/single_fm/{sess.session_id}/qc_{sess.session_id}.json')
copy_session_yaml(sess.source_yaml, f'results/single_fm/{sess.session_id}')
update_manifest_row('results/single_fm/_manifest.csv', sess.session_id, {
    'parquet_bin1p0': f'results/single_fm/aligned/session_{sess.session_id}_bin1p0.parquet',
    'yaml': f'results/single_fm/{sess.session_id}/session.yaml'
})

# Optional: Plotly animation (HTML recommended)
fig = build_com_fr_animation(bins_pose['t'].to_numpy(),
                             bins_pose[['com_x','com_y','com_z']].to_numpy(),
                             bins_pose['fr_pop_mean'].to_numpy(),
                             options=AnimOptions(fps=20, title=f'{sess.session_id}'))
save_animation(fig, html_path=f'results/single_fm/{sess.session_id}/anim_com_fr.html')
```

Notes
- We removed notebook-specific helpers (e.g., `pipeline/nb01.py`). Use the modules above directly.
- Manifest canonical columns for downstream use: `parquet_bin1p0`, `parquet_bin0p5`, `yaml`, `qc`, `pose_upright_npz`, `pose_processed_npz`, `pose_features`.

NB03 — Visual Covariates (ROI flow + ME PCs)
- Purpose: conservative visual‑only features from scene cameras to control for visual drive when modeling neural activity.
- Inputs:
  - 1.0 s aligned parquet (for session metadata/time base)
  - Upright pose NPZ or pose features parquet (head position/direction)
  - Calibration intrinsics/extrinsics (JSON/PKL) — rotated upright via `floor_plane.json`
  - Scene video root and (preferred) `extraction_report` mapping device prefixes → calibration camera names
- Outputs:
  - `results/single_fm/<sid>/vis_features/vis_features_<sid>.parquet` with per‑bin columns:
    - `flow_mag, flow_div, flow_curl, flow_dir_x, flow_dir_y`
    - `me_pc1 … me_pcK` (motion‑energy PCs)
    - `wsum, cams_used` (diagnostics)
- Parameters (top of notebook):
  - `FLOW_ALGO`: `'farneback'` (default) or `'tvl1'` (if available in OpenCV)
  - `ROI_RADIUS_DEG_GRID`: candidate angular half‑radii; current implementation uses `ROI_RADIUS_PX` as a fallback
  - `ME_K`: number of motion‑energy PCs (e.g., 8–16)
  - `BIN_S`: feature bin size; default 0.05–0.10 s recommended
- Camera discovery and mapping:
  - Preferred: use triangulation `extraction_report` via `freemat_data.calibration.load_extraction_report`
  - Fallback: add to the session YAML under `video_extraction.camera_name_map` with per‑camera glob(s)
- Verification checkpoints:
  - Divergence vs forward speed: positive correlation
  - Flow curl vs head yaw‑rate: positive correlation
  - Coverage: non‑zero `wsum` during movement; ROI overlay sanity on sampled frames
- Requirements: OpenCV (`cv2`) for optic flow and motion‑energy; PyAV (`av`) for robust timestamps (falls back to FPS indexing if unavailable). Notebook fails loudly if prerequisites or inputs are missing.

Module usage (preferred inside the notebook)
```python
from freemat.single_fm.visual.features import VisualParams, build_visual_features

params = VisualParams(bin_s=0.05, me_k=12, flow_algo='farneback', lookahead_m=1.0, roi_radius_deg=12)
df, qc = build_visual_features(sess_yaml, params=params)
df.to_parquet(os.path.join(vis_dir, f'vis_features_{SESSION_ID}.parquet'), index=False)
```

YAML additions (optional fallback when no extraction_report)
```yaml
video_extraction:
  camera_name_map:
    Cam_001: "**/e3v825d-*.mp4"  # glob(s) under video_dir
    Cam_002: ["**/e3v825e-*.mp4", "**/altprefix-*.mp4"]
  # Optional explicit outputs for timestamp indices
  timestamp_index:
    Cam_001: results/single_fm/<sid>/vis_features/ts_index/Cam_001.csv
    Cam_002: results/single_fm/<sid>/vis_features/ts_index/Cam_002.csv
```

Quick start
1) Create a per‑session YAML in `configs/single_fm/` with:
   - `data_files.pose_3d`, `data_files.m1_neural` (or `m2_*_neural`), `data_files.syncmap_json`
   - Optional calibration: `data_files.calib_pkl`; metadata: `reference_camera`, `units`, `subject` (hedy|hooke)
2) Open `notebooks/01_load_qc_align.ipynb` and set `SESSION_YAML = 'configs/single_fm/session_<id>.yaml'` (replace placeholder Windows paths shown as examples). Run all cells.
3) Outputs are written under `results/single_fm/<id>/` and `results/single_fm/aligned/` and a manifest row is updated.
4) Alternative: use a central registry via `Registry('configs/sessions.yaml')` with `SESSION_ID` to resolve sessions without editing paths in the notebook.

Notes
- Pose orientation: do not re‑rotate pose in the Plotly cell — `freemat_data.orient_pose_with_floor` already applies upright `R_yaw @ R_align`.
- Calibration display: always use `calib_upr` (also rotated upright) to draw camera gizmos.
- Visual floor: if `floor_hex_uv.npz` is present next to the pose NPZ, the notebook uses it to draw the hex checker correctly.
- Manifest‑first: prefer paths from `results/single_fm/_manifest.csv` when present; normalize Windows backslashes to forward‑slashes.
- Data safety: never mix real and mock data in a run. If using mock data, set flags consistently and keep outputs separate (do not reuse manifest rows).

Bank mapping and exclusions
- Configure electrode banks and subject‑specific exclusions in `configs/banks.yaml`.
- NB01 uses `freemat_data.banks` to:
  - Map unit channels → bank labels (`map_channels_to_bank`).
  - Apply subject‑specific channel/unit masks (`excluded_channels_for_subject`, `mask_excluded_units`).
- Include `subject: hedy|hooke` in each session YAML so the notebook can resolve the correct exclude list.

**Motifs Integration**
- Source for motifs: use the processed pose saved by NB01 at `results/single_fm/<sid>/pose_processed.npz`.
- Why processed: NB01 standardizes orientation (upright), centers to the enclosure, trims to the longest valid segment, interpolates short gaps, and applies light smoothing. Motifs can then focus on feature/embedding/segmentation without re‑doing preproc.
- Floor metadata: the motifs pipeline expects `floor_plane.json` (and `floor_hex_uv.npz`) next to the NPZ it reads. If running motifs on `pose_processed.npz`, copy or symlink the original `floor_plane.json` (and `floor_hex_uv.npz`) from the triangulation output directory into `results/single_fm/<sid>/`.
- Running motifs (current):
  - Single‑animal: `python -m poyopo.freemat_vid_inference.motifs.cli single --npz results/single_fm/<sid>/pose_processed.npz --out runs/motifs/<sid> --assume_processed --zup`
  - If `floor_plane.json` is not next to the NPZ, add `--plane_json <path/to/floor_plane.json>` (found in the triangulation folder).
  - The loader accepts both raw triangulation NPZs `(T,N,J,3)` and processed NPZs `(T,J,3)` — the latter is handled as `N=1` automatically.
- Downstream loading: in neural notebooks (NB05 motifs‑spike), load motif assignments via `freemat_data.loaders.motifs` (to be implemented) or by reading the motifs run `ethogram_smoothed.parquet` until the loader is added.

Notebook flow

- NB01 `01_load_qc_align.ipynb`
  - Loads session config via `freemat_data.Registry`.
  - Loads spike events and 3D pose (do not assume fps; parse from files).
  - Aligns spikes to video time using registry sync mappings.
  - Bins spikes (1.0 s and 0.5 s) and computes `fr`, optional `fr_z` per unit.
  - Saves outputs:
    - `results/single_fm/aligned/session_{SESSION_ID}_bin1p0.parquet`
    - `results/single_fm/aligned/session_{SESSION_ID}_bin0p5.parquet`
    - `results/single_fm/{SESSION_ID}/qc_{SESSION_ID}.json`
    - `results/single_fm/{SESSION_ID}/session.yaml` (copied provenance)
    - Optional: `results/single_fm/{SESSION_ID}/pose_upright.npz`, `results/single_fm/pose_features/pose_features_{SESSION_ID}.parquet`
  - Updates manifest row in `results/single_fm/_manifest.csv` with discovered paths.
  - Bank mapping: applies `configs/banks.yaml` to annotate units by bank and subject and to mask excluded units for analysis/QC.

- NB02 `02_figure1_overview.ipynb`
  - Loads the 1.0 s aligned parquet and, if available, pose features and `pose_upright.npz`.
  - Produces Figure 1 panels (see docs/projects/single_fm/analyses.md):
    - Spike firing rate histogram (population)
    - Short‑window event rasters (10 s, 30 s, 60 s); fallback to tick reconstruction from rates if events unavailable
    - Long‑window rate heatmaps (5 min, 30 min)
    - Speed timeseries + histogram (if pose available)
    - 3D occupancy projections (XY/XZ/YZ) and height-tiers summary (if pose available)
  - Default filtering: panels use `qc_keep_analysis == True` by default and cap rasters/heatmaps at 64 units; toggles can expose dropped units if desired.
  - Saves each panel as PNG and SVG next to the aligned parquet, e.g., `results/single_fm/aligned/fig1_{SESSION_ID}_raster_10s.png`.

- NB03 `03_visual_covariates.ipynb`
  - Loads session via manifest; uses `data_files.video_dir` for scene videos.
  - Computes VIS features via a virtual head view (planar warp): for each bin, intersect head-forward with the floor plane to define a small ground patch ~1 m ahead; warp frames from each camera into this canonical patch; compute dense flow inside a central ROI; summarize magnitude, divergence, curl, and dominant direction; visibility-weight features by `w = max(0, cos(fhead, fcam))` and aggregate across cameras.
  - Extracts motion-energy features on the patch difference (tiny Gabor bank); runs PCA to `me_pc1..K`.
  - Produces `results/single_fm/{SESSION_ID}/vis_features/vis_features_{SESSION_ID}.parquet`.
  - Verification checkpoints:
    - Flow divergence correlates with forward speed; curl with head yaw‑rate (smoothed).
    - Camera weighting sane: cameras opposite to head‑forward have near‑zero weights.
    - Coverage plots: ROI path overlays and non‑zero `wsum` during movement.
  - Common toggles: `FLOW_ALGO={'farneback','tvl1'}`, `ROI_RADIUS_PX`, `ME_K`, `BIN_S` (default 0.05 s).

NB03 minimal usage (module-centric)

```python
from freemat.single_fm.visual.features import build_visual_features, VisualParams
from freemat.single_fm.viz import save_vis_diagnostics

session_yaml = 'configs/single_fm/session_<id>.yaml'
params = VisualParams(
    bin_s=0.05,
    flow_algo='farneback',
    roi_radius_px=60,
    patch_hw=(200, 200),
    mpp=0.005,
    me_k=6,
    cams_include=['Cam_001'],   # optional: run only this camera
    frame_stride=2,             # optional: process every other frame-pair
)

df, qc = build_visual_features(session_yaml, params=params)
out_dir = f"results/single_fm/{qc.get('session_id', '<id>')}/vis_features"
save_vis_diagnostics(session_yaml, df, out_dir=out_dir)
```

NB03 CLI (quick tests)

```bash
python -m freemat.single_fm.pipeline.vis_cli \
  --session-yaml configs/single_fm/session_<id>.yaml \
  --cams Cam_001 \
  --bin-s 0.05 --roi-radius-px 60 --flow farneback \
  --patch-h 200 --patch-w 200 --mpp 0.005 \
  --me-k 6 --lookahead-m 1.0 --frame-stride 2 \
  --patch-overlays 3
```

Outputs
- Parquet: `results/single_fm/<sid>/vis_features/vis_features_<sid>.parquet`
- QC JSON: `results/single_fm/<sid>/vis_features/vis_qc_<sid>.json`
- Diagnostics (PNG+SVG): same directory unless `--no-diagnostics`

Notes
- Pose: this pipeline prefers `results/single_fm/<sid>/pose_processed.npz` (upright, leveled) when present; otherwise it loads from the session YAML and orients using `floor_plane.json`.
- Kinematics: forward speed and yaw rate are computed internally for validation (smoothed with a short moving average). Figures are saved as PNG+SVG.

Key conventions

- Always infer fps from source files (NPZ meta) — do not hardcode.
- Paths: prefer entries from `results/single_fm/_manifest.csv` when present; normalize backslashes to forward‑slashes.
- Camera mapping (NB03): prefer `data_files.extraction_report` from triangulation to map calibration cameras (e.g., `Cam_001`) to device/video prefixes (e.g., `e3v825d-…`). The notebook parses this report (via `freemat_data.calibration.load_extraction_report`) and searches under `data_files.video_dir` to find files for each calibration camera without relying on hand-written globs. If the report is missing, you may supply fallback globs under `video_extraction.camera_name_map` in the session YAML.
- Coordinate frame: pose is expected to be upright (Z‑up); notebook checks `pose_upright.npz` metadata and applies minimal orientation if needed.
- Event rasters: prefer event‑coded spikes resolved from the per‑session YAML in the manifest; if missing, NB02 falls back to tick reconstruction from rates.
- Plot style: use `visualization/nature_neuro_style.py` when available; otherwise fall back to Matplotlib defaults.

Stationarity disclaimer

- Sessions are long and naturalistic (freely moving). Global stationarity rarely holds. QC and diagnostics emphasize per‑bank summaries, sliding‑window metrics (e.g., CV2/LV), and short‑window rasters. Global dispersion (e.g., Fano) must be interpreted cautiously; we include guardrails for descriptive plots but do not use Fano for exclusion decisions by default.
- ACF vs “ACG”: when event times are unavailable, we plot rate autocorrelation (ACF) from z‑scored 1 s firing rates (y ∈ [‑1,1]) and a high‑pass variant. These replace classical millisecond spike autocorrelograms.

Acceptance checklist
- NB01 writes: aligned 1.0 s/0.5 s parquets; per‑session QC JSON; session YAML copy; updates the manifest row; optional pose_upright.npz and pose features parquet.
- NB02 reads: manifest‑resolved 1.0 s parquet; applies default filtering (qc_keep_analysis) and unit cap; saves Figure 1 panels (PNG+SVG) next to the aligned parquet.
 - NB03 verifies: divergence↔forward speed and curl↔yaw‑rate correlations; non‑zero coverage during movement; writes VIS features parquet with diagnostics (`wsum`, `cams_used`).

Troubleshooting

- If short‑window rasters show a single vertical line, ensure events are loaded; the notebook falls back to a tick raster by scanning for the earliest non‑empty window.
- If speed/XY panels are skipped, confirm `pose_features` parquet and/or `pose_upright.npz` exist from NB01.
- If `visualization` import fails, NB02 adds repo‑root to `sys.path` and provides a local style fallback.
- If camera files are not found in NB03: ensure `data_files.video_dir` points to the scene video root and `data_files.extraction_report` exists. The extraction report maps device prefixes to calibration camera names; NB03 will invert this mapping and rglob matching files. As a fallback, define `video_extraction.camera_name_map` globs in the session YAML.

---
Single_fm project

This contains the configs for the single_fm FreeMat project.

Relevant docs:
1. See freemat/single_fm/README.md for the project explanation
2. See docs/projects/single_fm/analyses.md for analyses and TODOs.md for pending TODO items.
3. See configs/single_fm/README.md for the single_fm config files.
4. See freemat_data/README.md for data IO package.

Other:
Per-session YAMLs for Single‑FM

Each session has its own YAML describing paths and metadata. Save them here or alongside the session data. When producing outputs (e.g., parquet), copy the YAML next to the outputs for provenance.

Minimal schema (example in session_example.yaml):
- session_id: unique ID (e.g., single_001)
- project: single_fm
- type: asocial  # single-monkey
- monkeys: [M1]  # or the subject name
- subject: hedy   # required to resolve subject-specific exclusions (hedy|hooke)
- data_files:
  m1_neural: A:/.../Neural_data_sorted.mat
  pose_3d: A:/.../triangulation/pose3d_smoothed.npz
  syncmap_json: A:/.../scripts/synchronization/sync/sync_map.json  # required for alignment
- metadata:
  notes: any

YAML path tips (Windows):
- Prefer single quotes around Windows paths, or convert to forward slashes:
  m1_neural: 'A:/EnclosureProjects/.../Neural_data_sorted.mat'
  # or escape backslashes
  m1_neural: 'A:\\EnclosureProjects\\...\\Neural_data_sorted.mat'


NB04 — GAM Modeling (VIS vs KIN)

- Purpose: quantify variance explained (R²) in per‑unit z‑scored firing rates by kinematic (KIN), visual (VIS), and combined (FULL) covariates. Scan temporal lags, use blocked cross‑validation with gaps, and compute null distributions via block permutation or within‑block shuffling.
- Inputs:
  - Aligned neural table (e.g., `aligned/session_<id>_bin1p0.parquet` or `bin0p5` when events available)
  - KIN features: from pose features parquet (NB01/pose pipeline output) or derived on the fly
  - VIS features: `results/single_fm/<sid>/vis_features/vis_features_<sid>.parquet` (from NB03)
- Core API (module): `freemat/single_fm/modeling/gam.py`
  - `make_block_folds(T, bin_s, folds, block_s, gap_s)` — contiguous folds with exclusion gaps
  - `build_spline_design(X, df)` — cubic regression spline basis per covariate (via `patsy`)
  - `evaluate_family(X_family, y, bin_s, cv_cfg, lag_ms, alphas)` — best R² and lag for one family
  - `family_null_r2(X_family, y, bin_s, cv_cfg, lag_ms, alphas, method, block_s, n_null)` — null R² samples
  - Convenience: `design_from_df(df, cols, df_spline, standardize=True)` and
    `evaluate_families_from_df(df, y_col, kin_cols, vis_cols, bin_s, ...)`
- Recommended defaults:
  - `bin_s ∈ {0.1, 0.25, 0.5, 1.0}` (prefer events for <1.0 s; otherwise aggregate)
  - CV: `folds=5`, `block_s=60 s`, `gap_s=10 s`
  - Lag scan: `±1000 ms` around response; choose best via inner‑CV alpha selection
  - Splines: `df=6` per covariate; ridge alphas grid `[0.1, 1.0, 10.0]`
  - Nulls: `method='block_permute'`, `block_s ≥ max tested lag`
- Usage example (per‑unit):
  ```python
  from freemat.single_fm.modeling.gam import (
      CVConfig, design_from_df, evaluate_family, evaluate_families_from_df,
      family_null_r2, empirical_p
  )

  # df_neural: tidy neural with columns ['t','unit_index','fr_z',...]
  # df_kin, df_vis: per‑bin features merged on ['t'] (and optionally ['bin_s'])
  df = df_neural.merge(df_kin, on='t', how='inner').merge(df_vis, on='t', how='left')
  y = df[df['unit_index'] == u]['fr_z'].to_numpy()
  Xkin = design_from_df(df[df['unit_index'] == u], kin_cols, df_spline=6)
  Xvis = design_from_df(df[df['unit_index'] == u], vis_cols, df_spline=6)
  r2_kin, best_lag_kin = evaluate_family(Xkin, y, bin_s=0.5, cv_cfg=CVConfig(), lag_ms=1000, alphas=[0.1,1,10])
  r2_vis, best_lag_vis = evaluate_family(Xvis, y, bin_s=0.5, cv_cfg=CVConfig(), lag_ms=1000, alphas=[0.1,1,10])
  r2_full, best_lag_full = evaluate_family(np.hstack([Xkin, Xvis]), y, bin_s=0.5, cv_cfg=CVConfig(), lag_ms=1000, alphas=[0.1,1,10])
  # Unique contributions (heuristic):
  uniq_kin = r2_full - r2_vis
  uniq_vis = r2_full - r2_kin
  # Null (optional):
  null = family_null_r2(Xkin, y, bin_s=0.5, cv_cfg=CVConfig(), lag_ms=1000, alphas=[0.1,1,10], method='block_permute', block_s=60.0, n_null=100)
  pval = empirical_p(r2_kin, null)
  ```
- Notes:
  - Ensure no future leakage: apply lag to predictors only; response stays fixed.
  - Drop rows with NaNs across selected columns before modeling.
  - Standardize covariates before spline bases for numerical stability when needed.
  - Interpret unique fractions cautiously; formal variance partitioning can be added later.
