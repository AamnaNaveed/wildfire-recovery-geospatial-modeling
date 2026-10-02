# Wildfire Recovery Geospatial Modeling

**Spatially explicit modelling of post-fire vegetation recovery across contrasting ecosystems using multi-temporal Earth observation.**

[![Status](https://img.shields.io/badge/status-analysis%20complete-blue)](#project-status)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project models post-fire vegetation recovery across 15 California wildfires (2018–2022) using Sentinel-2 satellite imagery, terrain, climate, and land-cover data. The goal is to understand how burn severity, terrain, climate, pre-fire vegetation state, and ecosystem type control the spatial pattern of recovery after wildfire.

The pipeline is fully reproducible and evaluates three models — a persistence baseline, Random Forest, and XGBoost — under leave-one-fire-out and leave-one-ecosystem-out cross-validation. All decisions (feature list, model configuration, test fire split) were frozen in `PROTOCOL.md` before any model was trained.

## Project status

**Analysis complete.** Manuscript in preparation.

## Key findings

- **Random Forest modestly beats the persistence baseline** under leave-one-fire-out cross-validation (RMSE 0.197 vs 0.216).
- **Model skill is fire-dependent.** On the frozen test set, Random Forest outperformed the baseline on BOBCAT (0.199 vs 0.241) but underperformed on CREEK (0.217 vs 0.211) and TUCKER (0.238 vs 0.137).
- **Leave-one-ecosystem-out generalization is possible.** Random Forest achieved positive mean R² (+0.036) when holding out entire ecosystems — higher than under leave-one-fire-out (−0.10).
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
