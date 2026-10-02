# Wildfire Recovery Geospatial Modeling

**Spatially explicit modelling of post-fire vegetation recovery across contrasting ecosystems using multi-temporal Earth observation.**

[![Status](https://img.shields.io/badge/status-analysis%20complete-blue)](#project-status)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project models post-fire vegetation recovery across 15 California wildfires (2018–2022) using Sentinel-2 satellite imagery, terrain, climate, and land-cover data. The goal is to understand how burn severity, terrain, climate, pre-fire vegetation state, and ecosystem type control the spatial pattern of recovery after wildfire.

The pipeline is fully reproducible and evaluates three models — a persistence baseline, Random Forest, and XGBoost, under leave-one-fire-out and leave-one-ecosystem-out cross-validation. All decisions (feature list, model configuration, test fire split) were frozen in `PROTOCOL.md` before any model was trained.

## Project status

**Analysis complete.** Manuscript in preparation.

## Key findings

- **Random Forest modestly beats the persistence baseline** under leave-one-fire-out cross-validation (RMSE 0.197 vs 0.216).
- **Model skill is fire-dependent.** On the frozen test set, Random Forest outperformed the baseline on BOBCAT (0.199 vs 0.241) but underperformed on CREEK (0.217 vs 0.211) and TUCKER (0.238 vs 0.137).
- **Leave-one-ecosystem-out generalization is possible.** Random Forest achieved positive mean R² (+0.036) when holding out entire ecosystems higher than under leave-one-fire-out (−0.10).
- **Seed-based uncertainty intervals substantially undercover** (5–13% coverage vs. 95% nominal), indicating that seed variance captures model initialization noise rather than true predictive uncertainty. Reported as a limitation.
- The strongest predictors are **pre-fire vegetation state**, **precipitation**, and **temperature** — consistent with established fire ecology.

## Data sources

| Data | Source | Resolution |
|------|--------|------------|
| Optical imagery | Sentinel-2 L2A via Microsoft Planetary Computer | 10 m |
| Fire perimeters & dNBR | MTBS (Monitoring Trends in Burn Severity) | 30 m |
| Elevation | NASADEM via Microsoft Planetary Computer | 30 m |
| Climate | ERA5-Land via Copernicus CDS | ~9 km |
| Land cover | ESA WorldCover | 10 m |

## Pipeline overview

| Stage | Input | Output |
|-------|-------|--------|
| 1. Fire selection | MTBS fire occurrence (2018–2022) | 15 California fires, 3 ecosystems, 5 years |
| 2. Optical processing | Sentinel-2 L2A scenes | Pre/post composites, tile-stratified |
| 3. Indices | Composite bands | NDVI, NBR, dNBR at MTBS pre/post dates ± 30 days |
| 4. Auxiliary layers | NASADEM, ERA5-Land, ESA WorldCover | Terrain, climate, land cover aligned to fire grids |
| 5. Feature tables | All aligned rasters | 20,000 sampled pixels per fire, 9 predictors |
| 6. Modelling | Feature tables | Persistence, RF, XGBoost under LOFO/LOEO |
| 7. Evaluation | Frozen test fires | Per-fire metrics on CREEK, BOBCAT, TUCKER |

## Repository structure

```
wildfire-recovery-geospatial-modeling/
├── PROJECT_PLAN.md          # Full research plan
├── PROTOCOL.md              # Frozen study design + stage outcomes
├── README.md                # This file
├── CITATION.cff             # Citation metadata
├── CHANGELOG.md             # Research progression log
├── configs/                 # Configuration files
├── data/                    # Source and processed data (not committed)
├── docs/
│   ├── METHODS.md           # Technical methods
│   ├── DATA_DICTIONARY.md   # Feature table schema
│   └── REPRODUCIBILITY.md   # How to regenerate the pipeline
├── scripts/                 # Analysis scripts (numbered in order)
├── src/wildfire_recovery/   # Reusable package code
├── reports/
│   ├── *.json               # Per-stage results
│   └── figures/             # Generated figures
└── tests/                   # Unit tests
```

## Scripts

| Script | Purpose |
|--------|---------|
| `05_classify_fires.py` | Classify candidate fires by ecosystem (ESA WorldCover) |
| `06_process_selected_fires.py` | Build Sentinel-2 pre/post composites, compute dNBR |
| `07_load_perimeters.py` | Load MTBS fire perimeters, compute bounding boxes |
| `09_fetch_auxiliary_layers.py` | Fetch terrain, climate, land cover per fire |
| `10_build_feature_tables.py` | Sample 20k pixels per fire → Parquet |
| `11_baseline_persistence.py` | Persistence baseline (LOFO) |
| `12_random_forest.py` | Random Forest (LOFO) |
| `13_xgboost.py` | XGBoost (LOFO) |
| `14_leave_one_ecosystem_out.py` | Leave-one-ecosystem-out validation |
| `15_uncertainty.py` | Seed-based uncertainty quantification |
| `16_final_test.py` | Frozen test fire evaluation |

## Reproducing

See `docs/REPRODUCIBILITY.md` for full instructions. In brief:

```bash
conda env create -f environment.yml
conda activate wildfire-recovery
# Download source data per docs/METHODS.md
python scripts/05_classify_fires.py
python scripts/06_process_selected_fires.py --batch 1
python scripts/06_process_selected_fires.py --batch 2
# ... see REPRODUCIBILITY.md
```

## Results summary

| Model | LOFO RMSE | LOFO R2 | LOEO RMSE | LOEO R2 |
|-------|-----------|---------|-----------|---------|
| Persistence | 0.216 | -0.34 | - | - |
| Random Forest | 0.197 | -0.10 | 0.204 | 0.036 |
| XGBoost | 0.207 | -0.20 | 0.217 | -0.12 |

Test set (frozen):

| Fire | Ecosystem | RF RMSE | Persistence RMSE |
|------|-----------|---------|------------------|
| CREEK | forest | 0.217 | 0.211 |
| BOBCAT | shrubland | 0.199 | 0.241 |
| TUCKER | grassland | 0.238 | 0.137 |

## Author

**Aamna Naveed**
Biodiversity and geospatial data researcher
Bahawalpur, Pakistan

## Citation

Manuscript in preparation. If you use this code, please cite this repository via the metadata in `CITATION.cff`.

## License

MIT — see `LICENSE`.
