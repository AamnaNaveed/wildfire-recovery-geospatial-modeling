# Changelog

All notable changes to this project are documented here.

The project followed a staged pipeline: data acquisition, processing,
modelling, and evaluation. Each stage produced a committed script and
a JSON report. This log summarizes the progression.

## [0.9.0] — 2026-10-02

### Analysis complete

- **Stage 0–1** — Project skeleton, synthetic raster pipeline test
- **Stage 2** — Fire catalog: 145 California fires (2018–2022) from MTBS
- **Stage 3** — Data-access test: one fire processed end-to-end with Sentinel-2
- **Stage 4** — Raster QA: terrain, climate, and land cover aligned
- **Stage 5** — Selected 15 fires stratified by ecosystem and year
- **Stage 6a** — Auxiliary layers (NASADEM, ERA5-Land, WorldCover) for 15 fires
- **Stage 6b** — Feature tables: 20,000 pixels per fire, 9 predictors
- **Stage 7** — Protocol frozen; test fires assigned (CREEK, BOBCAT, TUCKER)
- **Stage 8** — Persistence baseline: RMSE 0.216, R² −0.34
- **Stage 9** — Random Forest: RMSE 0.197, R² −0.10, Spearman 0.336
- **Stage 10** — XGBoost: RMSE 0.207, R² −0.20, Spearman 0.194
- **Stage 11** — Leave-one-ecosystem-out: RF RMSE 0.204, R² +0.036
- **Stage 12** — Uncertainty: seed-based intervals undercover (5–13%)
- **Stage 14** — Final test: RF wins on BOBCAT, loses to baseline on CREEK/TUCKER

### Known limitations

- 5 of 15 selected fires excluded due to insufficient burn signal
- Uneven ecosystem balance: 5 forest, 3 shrubland, 2 grassland
- Seed-based uncertainty intervals are not well-calibrated
- Model skill is fire-dependent, not uniformly better than baseline

### Manuscript

In preparation.