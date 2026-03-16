# freemat_data

Data loading and transformation utilities for neural-behavioral analysis.

Provides typed loaders for:
- **3D pose** — triangulated keypoints from multi-camera tracking
- **Spikes** — sorted single/multi-unit events from wireless arrays
- **Motifs** — behavioral syllable labels from unsupervised segmentation
- **Sync** — neural-to-video timestamp alignment

Also includes feature transforms (kinematics, spike binning, visual features), electrode bank configuration, and spike QC metrics.

Used by `freemat/single_fm/` analysis modules. Session configs in `configs/*.yaml` define per-session data paths.
