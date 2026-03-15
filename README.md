# solo_mSTS_parodi2026

Analysis code for: **Embodied state coding in the mid superior temporal sulcus of freely moving macaques**

Parodi et al. (2026), *Nature Communications*

## Overview

This repository contains the complete analysis pipeline for reproducing all results (Figs 2-5) from raw neural and behavioral data. Two rhesus macaques were recorded with wireless 128-channel arrays in mid-STS while freely navigating a hexagonal arena.

## Data

Raw data and pre-computed results are available on Zenodo (DOI: TBD).

After downloading, place data in `data/` with this layout:

```
data/
  240525-hooke/
    Neural_data_sorted_final.mat
    pose3d_smoothed.npz
    sync_map.json
  240526-hedy/
    ...
  240605-hedy/
    ...
  240609-hooke/
    ...
  240613-hooke/
    ...
  motifs/
    ethogram_final.parquet
    clusters_summary.parquet
    ...
```

Pre-computed results go in `results/`:

```
results/
  production/          # Encoding/decoding parquets (Figs 2-4)
  fig3_rf/             # Reference-frame index results
  fig4_similarity/     # Neural-behavioral similarity
  fig5_cebra/          # CEBRA manifold embeddings and results
```

## Installation

```bash
git clone https://github.com/felipe-parodi/solo_mSTS_parodi2026.git
cd solo_mSTS_parodi2026
pip install -e .
```

For GPU-accelerated CEBRA analysis (Fig 5):

```bash
pip install -e ".[gpu]"
# CEBRA v0.4.0 must be installed from source:
git clone https://github.com/AdaptiveMotorControlLab/CEBRA.git
cd CEBRA && git checkout v0.4.0 && pip install -e . && cd ..
```

## Reproduction

### From pre-computed results

Pre-computed parquets contain all numbers reported in the paper. Load them directly:

```python
import pandas as pd
fig2 = pd.read_parquet("results/production/hedy_bin05/summary/fig2b_summary.parquet")
fig3 = pd.read_parquet("results/production/fig3_bin05/summary/fig3_summary.parquet")
```

### From raw data (full pipeline)

```bash
# Fig 2: Encoding model (Hedy sessions, ~1h)
python scripts/run_fig2b_all_sessions.py \
    --sessions hedy_0526 hedy_0605 --bin-width 0.5 --save-residuals

# Fig 3: Reference frames (all sessions, ~1h)
python scripts/run_fig3_all_sessions.py

# Fig 4: Motif decoding (all sessions, ~2h)
python scripts/run_fig4_all_sessions.py \
    --output-dir results/production/fig4_all4 --n-perm 1000

# Fig 5: CEBRA manifold (all sessions, ~45min with GPU)
python scripts/run_fig5_cebra.py \
    --output-dir results/fig5_cebra
```

## Repository Structure

```
solo_mSTS_parodi2026/
  freemat/single_fm/       # Core analysis modules
    data/                  # Session loading, feature computation, calibration
    encoding/              # Ridge regression, variance decomposition, decoding
    features/              # VIS/KIN/POS feature computation
    analysis/              # Syllable analysis (follow-up)
  freemat_data/            # Data infrastructure (loaders, types, sync)
  analysis/                # Decoding framework, syllable alignment
  scripts/                 # Runnable analysis pipelines (Figs 2-5)
  configs/                 # Session configs, calibration JSONs, electrode banks
  tests/                   # Unit tests
```

## Key Analysis Modules

| Module | Purpose |
|--------|---------|
| `encoding/run_fig2b.py` | Ridge encoding + unique variance decomposition |
| `encoding/fig2d_spatial_decoder.py` | Height/zone spatial decoding |
| `encoding/fig3_rf_index.py` | Reference-frame preference (allo/body/head) |
| `encoding/run_fig4.py` | Behavioral syllable decoding + PETH |
| `data/session_loader.py` | Aligned spike-pose data loading |
| `data/feature_builder.py` | VIS, BODY_KIN, HEAD_KIN, POS features |
| `data/arena.py` | Hexagonal arena geometry (d_wall_horiz) |

## Sessions

| Session | Subject | Units | Duration | FPS |
|---------|---------|-------|----------|-----|
| hedy_0526 | M2 (Hedy) | 443 | 40 min | 40 |
| hedy_0605 | M2 (Hedy) | 400 | 40 min | 40 |
| hooke_0525 | M1 (Hooke) | 186 | 31 min | 40 |
| hooke_0609 | M1 (Hooke) | 191 | 17 min | 30 |
| hooke_0613 | M1 (Hooke) | 184 | 37 min | 40 |

**Total**: 1,404 units (860 SUA + 544 MUA) across 5 sessions.

## License

MIT. See [LICENSE](LICENSE).
