Single‑FM Configs

Purpose
- Define per‑session YAMLs and optional multi‑session registry for the single_fm project.

Minimal per‑session YAML schema
```yaml
session_id: single_001             # unique ID
project: single_fm                  # fixed
type: asocial                       # single‑monkey
subject: hedy                       # hedy|hooke
monkeys: [M1]
data_files:
  m1_neural: A:/.../Neural_data_sorted.mat   # or *.npz with spike events
  pose_3d: A:/.../triangulation/pose3d_smoothed.npz
  syncmap_json: A:/.../sync/sync_map.json
  calib_pkl: A:/.../calibration/calibration.pkl  # optional
metadata:
  reference_camera: Cam_001         # optional
  notes: any

# Optional (NB03 fallback when extraction_report is unavailable)
video_extraction:
  camera_name_map:
    Cam_001: "**/e3v825d-*.mp4"
    Cam_002: ["**/e3v825e-*.mp4", "**/altprefix-*.mp4"]
  timestamp_index:
    Cam_001: results/<sid>/vis_features/ts_index/Cam_001.csv
    Cam_002: results/<sid>/vis_features/ts_index/Cam_002.csv
```

Tips
- Use single quotes or forward‑slashes for Windows paths to avoid escapes.
- Keep YAMLs alongside data or under `configs/`; NB01 copies the YAML next to results for provenance.
- Most notebooks resolve inputs via `results/_manifest.csv` first, then fall back to explicit paths.

Manifest columns (written by NB01 helpers)
- `aligned_1p0` → `results/aligned/session_<sid>_bin1p0.parquet`
- `aligned_0p5` → `results/aligned/session_<sid>_bin0p5.parquet` (when available)
- `session_yaml` → copied YAML path for the session

VIS features (NB03)
- Writes `results/<sid>/vis_features/vis_features_<sid>.parquet`.
- Preferred camera discovery via `data_files.extraction_report`; fallback to `video_extraction.camera_name_map`.

KIN features
- If pose features parquet is produced, prefer it (e.g., `results/<sid>/features/pose_features_<sid>.parquet`).
- Otherwise, downstream analyses may derive KIN from processed pose.

