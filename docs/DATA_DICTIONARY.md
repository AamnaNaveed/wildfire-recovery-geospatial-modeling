# Data Dictionary

Schema of the per-fire feature tables produced by
`scripts/10_build_feature_tables.py`. One row per sampled pixel.

## Columns

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `event_id` | string | — | MTBS event identifier (e.g. `CA3720111927220200905`) |
| `fire_name` | string | — | Fire name from MTBS |
| `year` | int | — | Year of ignition |
| `ecosystem` | string | — | One of: `forest`, `shrubland`, `grassland` |
| `dnbr` | float32 | index | Differenced Normalized Burn Ratio (target variable) |
| `pre_nbr` | float32 | index | Pre-fire Normalized Burn Ratio |
| `post_nbr` | float32 | index | Post-fire Normalized Burn Ratio |
| `pre_ndvi` | float32 | index | Pre-fire Normalized Difference Vegetation Index |
| `post_ndvi` | float32 | index | Post-fire Normalized Difference Vegetation Index |
| `elevation` | float32 | metres | NASADEM elevation |
| `slope` | float32 | degrees | Terrain slope derived from elevation |
| `aspect_sin` | float32 | — | Sine of aspect (circular → linear transform) |
| `aspect_cos` | float32 | — | Cosine of aspect (circular → linear transform) |
| `temperature` | float32 | °C | Annual mean 2 m air temperature (ERA5-Land) |
| `precipitation` | float32 | mm | Annual total precipitation (ERA5-Land) |
| `landcover_class` | int32 | — | ESA WorldCover class code |
| `x` | float64 | metres | Projected X coordinate (EPSG:5070) |
| `y` | float64 | metres | Projected Y coordinate (EPSG:5070) |
| `fire_signal_quality` | string | — | One of: `strong`, `weak`, `invalid`, `insufficient_data` |

## ESA WorldCover class codes

| Code | Class |
|------|-------|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare / sparse vegetation |
| 70 | Snow and ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss and lichen |

## Coordinate reference systems

- Feature table `x`, `y`: **EPSG:5070** (NAD83 / Conus Albers)
- MTBS perimeters and fire occurrence: **EPSG:4269** (NAD83)
- Sentinel-2 tiles: **EPSG:326xx** (UTM zones 10N and 11N for California)
- ERA5-Land: **EPSG:4326** (WGS84)
- ESA WorldCover: **EPSG:4326** (WGS84)

All rasters aligned to a common grid per fire, in EPSG:5070.

## File locations

- Feature tables: `data/interim/features/{event_id}.parquet`
- Fire folder per fire: `data/interim/fires/{event_id}/` containing
  `pre_nbr.tif`, `post_nbr.tif`, `dnbr.tif`, `pre_ndvi.tif`,
  `post_ndvi.tif`, `dem.tif`, `slope.tif`, `aspect.tif`, `temp.tif`,
  `precip.tif`, `landcover.tif`, `metadata.json`, `aux_metadata.json`
- Stage reports: `reports/*.json`
- Figures: `reports/figures/*.png`