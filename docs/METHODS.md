# Methods

Technical summary of the analysis pipeline. For the frozen study design,
see `PROTOCOL.md`. For data sources, see the README.

## Data acquisition

All source data is publicly available and downloaded programmatically.

| Layer | Source | Access method | Resolution |
|-------|--------|---------------|------------|
| Sentinel-2 L2A | Microsoft Planetary Computer | STAC API (`pystac-client`) | 10 m |
| MTBS perimeters | USGS MTBS | Direct download (annual zips) | vector polygons |
| MTBS fire occurrence | USGS MTBS | Direct download (shapefile) | point |
| NASADEM | Microsoft Planetary Computer | STAC API | 30 m |
| ERA5-Land | Copernicus CDS | CDS API (`cdsapi`) | ~9 km |
| ESA WorldCover | ESA / AWS S3 | Direct download (3° tiles) | 10 m |

## Fire selection

145 California fires from MTBS (2018–2022), filtered by:
- `map_prog == MTBS` and `incid_type == Wildfire`
- Event ID prefix `CA` (California only)
- Burn area 1,000–250,000 hectares

Each fire classified by majority ESA WorldCover class in a 3 km buffer
around the ignition point:
- class 10 → forest
- class 20 → shrubland
- class 30 → grassland

15 fires selected stratified across the three ecosystem groups and five years.

## Optical processing

For each fire:
1. Search Sentinel-2 L2A scenes at MTBS `pre_image_id` and `post_image_id`
   dates ± 30 days.
2. Group scenes by MGRS tile.
3. Load bands B04 (Red), B08 (NIR), B12 (SWIR2), SCL (cloud mask) at COG
   overview level 3.
4. Apply SCL mask (exclude values 0, 9, 10 — no-data and clouds).
5. Take pixel-wise median across the tile's scenes.
6. Mosaic across tiles by reprojecting each tile's composite to EPSG:5070
   (CONUS Albers Equal Area) and taking the median.

## Indices

- NDVI = (NIR − Red) / (NIR + Red)
- NBR = (NIR − SWIR2) / (NIR + SWIR2)
- dNBR = pre_fire_NBR − post_fire_NBR, clipped to [−1, 1]

## Auxiliary layers

For each fire, aligned to the optical grid:
- **Terrain** — NASADEM elevation; slope and aspect derived by finite
  differences on the aligned grid
- **Climate** — ERA5-Land 2 m temperature (annual mean) and total
  precipitation (annual sum) for the fire year
- **Land cover** — ESA WorldCover class, clipped to fire bounding box

## Feature tables

For each fire:
- Valid-pixel mask: all of dNBR, pre_NBR, post_NBR, elevation, temperature,
  and land cover must be finite
- 20,000 pixels sampled at random from the valid mask
- 19 columns per row: fire metadata, dNBR, pre/post NBR, pre/post NDVI,
  elevation, slope, aspect_sin, aspect_cos, temperature, precipitation,
  landcover_class, x, y, signal_quality

Fire-level signal quality assigned:
- `strong` — mean dNBR ≥ 0.10 and mean pre_NBR ≥ 0.10
- `weak` — mean dNBR ≥ 0.02
- `invalid` — mean dNBR < 0.02
- `insufficient_data` — fewer than 1,000 valid pixels

10 fires with `strong` or `weak` signal retained for modelling.

## Models

### Persistence baseline
Predict each pixel's dNBR as the mean dNBR of its fire's ecosystem group
in the training fires.

### Random Forest
500 trees, `max_features=sqrt`, `min_samples_leaf=5`, `random_state=42`.

### XGBoost
`n_estimators=800`, `learning_rate=0.03`, `max_depth=6`,
`min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`,
`reg_lambda=1`, `objective=reg:squarederror`, `random_state=42`.

## Validation

- **Leave-one-fire-out (LOFO)** — each fire held out once
- **Leave-one-ecosystem-out (LOEO)** — train on two ecosystems, test on the third
- **Frozen test set** — CREEK, BOBCAT, TUCKER, evaluated once

Random pixel splits are prohibited due to spatial autocorrelation.

## Metrics

RMSE, MAE, R², Spearman correlation. Reported per fire and aggregated as
mean ± SD.

## Uncertainty

Five random seeds per model. Prediction variance across seeds, 95%
prediction interval width, and correlation between uncertainty and error.