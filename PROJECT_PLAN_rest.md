# Wildfire Recovery Geospatial Modeling

Spatially explicit modelling of post-fire vegetation recovery across contrasting ecosystems using multi-temporal Earth observation.

## Important feasibility decision

Skardu is discontinued. Google Earth Engine is optional. The project must still work through Python and public downloads if GEE fails. Never allow a broken data platform to invalidate the research design.

## 1. Project rationale

This project treats satellite imagery as spatiotemporal evidence and builds a transparent environmental model - not just a map classifier. It uses GIS programming, Earth observation, spatial statistics, machine learning, and ecological interpretation to understand post-fire land dynamics.

The methods deliberately span several areas of modern environmental modelling: GIS programming for spatial data acquisition, raster alignment, tiling, QA, modelling, and map export; geospatial AI through Random Forest, XGBoost, temporal models, explainability, and uncertainty analysis; Earth observation using Sentinel-2, Landsat, and MODIS/VIIRS spectral indices, burn severity, and multi-temporal composites; advanced satellite remote sensing for pre/post-fire change detection, dNBR, NBR/NDVI/NDMI recovery trajectories, cloud masking, and sensor harmonisation; spatial modelling of land dynamics through recovery trajectories, spatial autocorrelation, spatial blocks, ecosystem transfer, and optional cellular-automata simulation; and environmental/bioscience modelling of ecological recovery response as a function of burn severity, climate, terrain, pre-fire vegetation, and ecosystem type. The scientific process is grounded in pre-registered hypotheses, independent event validation, reproducible code, uncertainty, limitations, and manuscript preparation.

| Methodological area | Where this project demonstrates it |
|---|---|
| GIS programming | Python package for spatial data acquisition, raster alignment, tiling, QA, modelling, and map export. |
| Geospatial AI | Random Forest, XGBoost, temporal models, explainability, and uncertainty analysis. |
| Earth observation | Sentinel-2, Landsat, MODIS/VIIRS, spectral indices, burn severity, and multi-temporal composites. |
| Advanced satellite remote sensing | Pre/post-fire change detection, dNBR, NBR/NDVI/NDMI recovery trajectories, cloud masking, and sensor harmonisation. |
| Spatial modelling of land dynamics | Recovery trajectories, spatial autocorrelation, spatial blocks, ecosystem transfer, and optional cellular-automata simulation. |
| Environmental/bioscience modelling | Ecological recovery response as a function of burn severity, climate, terrain, pre-fire vegetation, and ecosystem type. |
| Scientific process | Pre-registered hypotheses, independent event validation, reproducible code, uncertainty, limitations, and manuscript preparation. |

## 2. Final project definition

Working title: Spatially Explicit Modelling of Post-Fire Vegetation Recovery Across Contrasting Ecosystems Using Multi-Temporal Earth Observation.

Core question: How do burn severity, terrain, climate, pre-fire vegetation, and ecosystem type control the rate and spatial pattern of vegetation recovery after wildfire?

Secondary question: Can a model trained on some ecosystem types predict recovery in a held-out ecosystem or held-out fire event without leaking spatial information?

| Item | Fixed choice |
|---|---|
| Study region | California, United States; 10-15 official fires from 2018-2022. |
| Ecosystems | Mediterranean shrubland, conifer forest, and grassland/open woodland. |
| Observation period | At least 2 years pre-fire and 3 years post-fire where imagery permits. |
| Main response | Fractional vegetation recovery from post-fire NDVI/NBR/NDMI relative to pre-fire baseline. |
| Main predictors | Burn severity, pre-fire vegetation, terrain, climate, fire timing, ecosystem, distance to fire edge, and radar change. |
| Primary task | Predict continuous recovery at year 1, year 2, and year 3 after fire. |
| Secondary task | Classify recovery status: low, moderate, high using thresholds defined before test evaluation. |
| Main models | Mixed-effects model, Random Forest, XGBoost, and temporal baseline; optional LSTM only after baselines pass. |
| Validation | Leave-one-fire-out and leave-one-ecosystem-out; random pixel splits are prohibited. |
| Main scientific output | Recovery-rate maps, driver importance, cross-ecosystem transfer results, uncertainty, and failure analysis. |

## 3. Research hypotheses

1. H1: Higher burn severity is associated with slower vegetation recovery after controlling for climate, terrain, and pre-fire vegetation.
2. H2: Multi-temporal models using burn severity, pre-fire state, terrain, and climate outperform spectral-only models under leave-one-fire-out validation.
3. H3: A model trained in two ecosystem types will show a measurable performance decrease in a held-out ecosystem, revealing domain shift.
4. H4: Terrain and climate variables explain a larger share of recovery variation in conifer forest than in grassland/open woodland.
5. H5: Spatially clustered residuals remain after non-spatial models, and spatial diagnostics reveal where important environmental processes are missing.
6. H6: Ensemble uncertainty and feature-space novelty are higher in rare ecosystem conditions and poorly represented fire-severity regimes.

### No guaranteed positive result

These hypotheses are tests, not promises. If H2 or H3 fails, that is still scientifically valuable if the comparison is fair, the validation is independent, and the result explains why. Never manipulate the method to force a high score.

## 4. Data plan with GEE-independent fallback

The project must work through a data abstraction layer. The modelling code receives standard GeoTIFF/Parquet files, not raw GEE objects. This means the data source can change without rewriting the scientific pipeline.

| Data | Primary source | Fallback | Use |
|---|---|---|---|
| Fire perimeter | MTBS official fire perimeter | NIFC/USGS fire perimeter archive | Event boundary and event ID |
| Burn severity | MTBS dNBR/RdNBR where available | Compute dNBR from Landsat/Sentinel-2 pre/post composites | Main disturbance predictor and response check |
| Optical imagery | Sentinel-2 L2A and Landsat Collection 2 | Microsoft Planetary Computer STAC/direct USGS/NASA downloads | NDVI, NBR, NDMI, EVI and time series |
| Fire detections | MODIS/VIIRS active fire products | NASA FIRMS downloads | Fire timing and event confirmation |
| Terrain | NASADEM or Copernicus DEM | SRTM/NASADEM local tile | Elevation, slope, aspect, curvature, TPI, TRI |
| Climate | ERA5-Land or PRISM | CHIRPS precipitation plus temperature reanalysis | Rainfall, temperature, drought and seasonal anomalies |
| Land cover | NLCD or Dynamic World | ESA WorldCover | Ecosystem stratification and predictor |

## 5. Exact variables and formulas

| Variable | Definition |
|---|---|
| NBR | (NIR - SWIR2) / (NIR + SWIR2) |
| NDVI | (NIR - Red) / (NIR + Red) |
| NDMI | (NIR - SWIR1) / (NIR + SWIR1) |
| dNBR | Pre-fire NBR - post-fire NBR |
| Recovery index | (Index_t - Index_postfire) / (Index_prefire - Index_postfire), clipped only for reporting, not during model fitting |
| Pre-fire baseline | Median of valid observations from 24 to 2 months before fire |
| Post-fire baseline | Median of valid observations from 1 to 3 months after fire, or nearest valid seasonal composite |
| Year-1 recovery | Median recovery index in months 10-14 after fire |
| Year-2 recovery | Median recovery index in months 22-26 after fire |
| Year-3 recovery | Median recovery index in months 34-38 after fire |
| Climate anomaly | Fire-year seasonal value minus 10-year seasonal baseline |
| Topographic exposure | Aspect encoded as sine and cosine; do not use raw circular degrees |

## 6. Study-event selection

Use complete fire events as the independent units. Do not select fires because they produce attractive maps or high model scores.

1. Build a candidate list of California fires from 2018-2022 with official perimeters and at least one valid burn-severity product.
2. Keep fires larger than 1,000 hectares and smaller than 250,000 hectares to balance coverage and processing cost.
3. Require at least 60% usable optical observations in the target seasonal windows after cloud masking.
4. Require at least 20,000 valid modelling cells per fire after masking water, urban areas, and permanent snow where relevant.
5. Stratify the final 10-15 fires across the three ecosystem groups and across years.
6. Freeze the event list in protocol.md before looking at final model performance.
7. Assign three groups: development fires, validation fires, and final test fires. Keep the final test fires hidden from tuning.

| Group | Minimum | Use |
|---|---|---|
| Development | 6 fires | Feature engineering, model development, and internal spatial cross-validation |
| Validation | 2-3 fires | Model selection and threshold/calibration decisions |
| Final test | 2-3 fires | One-time final evaluation; complete fire events held out |
| Ecosystem transfer holdout | At least 1 ecosystem with 2 fires if possible | Test generalisation to an unseen ecological domain |

## 7. Processing pipeline

1. Create a project boundary for each fire with a 2 km exterior buffer.
2. Acquire and catalogue imagery with source, date, cloud cover, scene ID, checksum, and download URL.
3. Apply cloud and cloud-shadow masks; retain quality flags and count valid observations.
4. Reproject continuous rasters to EPSG:3310 for California modelling; use nearest neighbour for masks and categorical data, bilinear for continuous data.
5. Create seasonal composites using the median of valid observations, not a single lucky image.
6. Calculate spectral indices and pre/post-fire change metrics.
7. Derive terrain variables once from the DEM and validate their units and ranges.
8. Join climate variables by fire and seasonal window; document the coarse spatial resolution of climate data.
9. Mask water, urban areas, permanent snow/ice, and invalid/no-data cells using fixed rules.
10. Write aligned GeoTIFF stacks and a Parquet modelling table with unique cell IDs.
11. Create a data-quality report for every fire: pixel count, valid fraction, missingness, min/max/median, CRS, resolution, and date coverage.
12. Only after the data audit passes, create training/validation/test manifests.

### Expected output after this stage

For each fire: aligned raster stack, valid-data mask, modelling table, preview map, metadata JSON, and QA report. If any fire has excessive missing data, remove it before modelling and document the exclusion. Do not fill large missing areas with zero.

## 8. Modelling architecture

### 8.1 Baseline 1 - persistence model

Predict that post-fire recovery follows the mean recovery of the same ecosystem and burn-severity class in the training fires. This simple baseline is required. If the machine-learning model cannot beat it under independent validation, the paper must say so.

### 8.2 Baseline 2 - mixed-effects model

recovery_y1 ~ burn_severity + prefire_ndvi + slope + elevation + rainfall_anomaly + temperature_anomaly + ecosystem + (1 | fire_id)

Use a random intercept for fire ID during development, but evaluate on fire IDs never seen during fitting. Report fixed effects, confidence intervals, random-effect variance, residual diagnostics, and partial R2 where appropriate.

### 8.3 Baseline 3 - Random Forest

Use a spatially blocked Random Forest regression model. Fix the starting configuration: 500 trees, max_features = sqrt, min_samples_leaf = 5, n_jobs = -1, random_state = 42. Tune only within the development fires.

### 8.4 Main model - XGBoost

Starting configuration: n_estimators = 800, learning_rate = 0.03, max_depth = 6, min_child_weight = 5, subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1, objective = reg:squarederror, random_state = 42. Use early stopping only against validation fires, never the final test fires.

### 8.5 Optional temporal model

Only build an LSTM/temporal neural model after the three baselines are complete and tested. Input: monthly or seasonal index sequence plus static terrain. Do not start with deep learning. The publication contribution is the environmental modelling and independent validation, not the number of neural networks.

## 9. Validation and non-negotiable result standards

| Evaluation | How it is done | Required report |
|---|---|---|
| Leave-one-fire-out | Train on all development fires except one, test on the held-out fire | RMSE, MAE, R2, Spearman correlation, calibration/residual plots |
| Leave-one-ecosystem-out | Train on two ecosystem classes, test on the third | Metrics by ecosystem and performance drop from in-domain |
| Spatial block | Divide each fire into spatial blocks; keep blocks intact | Block-level metric distribution and map of blocks |
| Temporal holdout | Train on earlier fires, test on later fires | Year-wise performance and drift |
| Uncertainty | Five seeds or bootstrap resampling for ML models | Mean +/- SD, prediction intervals, error versus uncertainty |

### Minimum acceptable scientific result

Do not proceed to manuscript claims unless the final test set has: (1) a functioning baseline, (2) independent fire-event validation, (3) a complete error table, and (4) evidence that the model is not merely learning fire identity or spatial location. A low result is not a reason to weaken validation.

## 10. Expected results and interpretation rules

| Result pattern | Correct interpretation | Incorrect interpretation |
|---|---|---|
| High in-domain, low held-out-fire performance | Strong event/domain shift; model may not generalise. | The model is highly accurate. |
| XGBoost beats RF slightly | Nonlinear feature interactions help modestly. | AI solves ecological recovery. |
| Spectral-only beats climate+terrain | Data and temporal signal may dominate this region; investigate leakage and scale. | Terrain is unimportant everywhere. |
| Deep model does not beat XGBoost | Tabular environmental modelling may be more appropriate for this sample size. | The project failed. |
| Uncertainty high in rare ecosystems | Model support is limited there. | The uncertainty map is an error map. |
| Null relationship with burn severity after controls | Recovery may be controlled by other factors or measurement limitations. | Change the outcome until significance appears. |

## 11. Publication-grade controls

1. Write protocol.md before final event selection and model tuning.
2. Freeze the primary outcome: year-1 recovery index, with year-2 and year-3 secondary outcomes.
3. Freeze the primary comparison: XGBoost versus mixed-effects and Random Forest under leave-one-fire-out validation.
4. Keep final test fire labels and outcomes out of all tuning code.
5. Report all planned models, including models that perform poorly.
6. Use paired bootstrap confidence intervals for model differences over held-out fire blocks.
7. Report missingness and exclusions; never silently discard difficult fires.
8. Archive code, environment, manifests, configurations, plots, and a DOI release.
9. Write a limitations section covering sensor resolution, cloud gaps, fire-boundary uncertainty, ecological heterogeneity, and observational - not causal - interpretation.

## 12. Professional GitHub repository design

Use GitHub Desktop for commits and pushes, but let the AI create the files and tell you exactly when to commit. The repository must be understandable to a scientist who did not build it.

See the full folder tree in the repository. Key directories: data/, configs/, notebooks/, src/wildfire_recovery/, scripts/, tests/, reports/, manuscript/, docs/.

## 13. GitHub Desktop workflow

1. Create the repository on GitHub as private until the first stable release.
2. Clone it using GitHub Desktop to a known folder.
3. Open the folder in the AI editor as a project.
4. Create a branch named setup/environment.
5. Ask the AI to implement exactly one approved stage.
6. Review the changed files and run the stated test yourself.
7. Commit through GitHub Desktop with a scientific message.
8. Push after each stable milestone.
9. Use branches for experiments; never overwrite the final branch with unreviewed AI changes.
10. Tag releases such as v0.1-data-audit, v0.2-features, v0.3-baselines, and v1.0-manuscript.

| Milestone | Commit message |
|---|---|
| Repository skeleton | chore: create reproducible project structure |
| Environment passes | chore: add locked environment and environment check |
| Data catalog passes | feat: add fire event catalog and provenance manifest |
| Raster QA passes | feat: add aligned raster preparation and QA reports |
| Feature table passes | feat: add post-fire recovery feature engineering |
| Splits pass | test: add fire-event and spatial-block split audit |

## 14. AI instruction file

See AGENT_RULES.md for the full text.

## 15. Step-by-step execution roadmap

This is the order. Do not skip ahead because a later model looks more exciting.

| Stage | What the AI does | What you should see before committing |
|---|---|---|
| 0. Project setup | Create repository skeleton, environment.yml, config, README, AGENT_RULES.md. | Project opens; environment check reports Python and package versions. |
| 1. Environment test | Run a tiny synthetic raster and feature test; no satellite download yet. | A small GeoTIFF and JSON QA report are created. |
| 2. Fire catalog | Download/list official fire perimeters and metadata. | CSV with event IDs, dates, areas, ecosystems, and source URLs. |
| 3. Data-access test | Process one small fire only using the chosen non-GEE path. | One valid optical composite and preview map. |
| 4. Raster QA | Align optical, terrain, climate, masks, and indices for one fire. | QA report shows matching grids, CRS, ranges, and valid-pixel counts. |
| 5. Scale to events | Run the tested pipeline for all selected fires. | One folder per fire with immutable metadata and processing log. |
| 6. Feature table | Build modelling Parquet table with one row per valid cell/time. | Schema, row count, missingness table, and feature preview. |
| 7. Protocol freeze | Create PROTOCOL.md, freeze fire groups, splits, outcomes, metrics. | Git tag protocol-v1.0. No final tuning before this. |
| 8. Baselines | Implement persistence and mixed-effects models. | Baseline metrics and residual plots. |
| 9. Random Forest | Train spatially validated RF with fixed starting config. | Fold metrics, feature importance, predictions. |
| 10. XGBoost | Train and tune only on development/validation fires. | Held-out validation metrics and saved model. |
| 11. Transfer test | Run leave-one-fire-out and leave-one-ecosystem-out. | Performance table and domain-shift analysis. |
| 12. Uncertainty | Run five seeds/bootstrap predictions and novelty analysis. | Mean, spread, interval, and uncertainty-error plots. |
| 13. Optional temporal model | Only if baselines and data support it. | Fair comparison under identical splits. |
| 14. Final evaluation | Use frozen config on final test fires once. | Final metrics, error analysis, and audit log. |
| 15. Manuscript package | Generate figures, tables, methods, limitations, and archive. | Reproducible report and release candidate. |

## 16. Stage gates and expected numbers

| Gate | Expected minimum | Action if below |
|---|---|---|
| Event inventory | 10 fires, 3 ecosystem groups, documented dates and areas. | Expand or revise selection; do not proceed with one convenient fire. |
| Valid imagery | At least 60% valid target-season observations per fire. | Change source/composite window or exclude fire with a written reason. |
| Modelling rows | At least 20,000 valid cells per fire after masking. | Investigate masks and resolution; never manufacture rows. |
| Feature missingness | Primary features under 15% missing in training rows. | Repair acquisition or revise feature; document any imputation. |
| Baseline | Persistence baseline runs and is reported. | Stop; no ML model until baseline works. |
| Model stability | Five seeds or bootstrap replicates complete; no NaN metrics. | Fix instability or report it; never select only best run. |

## 17. Visual quality control

- Open every first-fire preview in QGIS or a raster viewer.
- Check that fire perimeter overlays the burned area.
- Check that no-data is not rendered as black vegetation.
- Check that slope, elevation, and aspect look physically plausible.
- Check that seasonal composites do not contain obvious clouds or shadows.
- Check that recovery increases or decreases in plausible time patterns rather than jumping because of a bad image.
- Check maps from at least one easy fire and one difficult fire.
- Save screenshots or PNG previews in reports/figures/quality_control/.

## 18. Scientific interpretation and manuscript plan

| Manuscript section | What to write |
|---|---|
| Introduction | Gap in spatially independent modelling of post-fire recovery across ecosystem domains. |
| Data | Fire events, sensors, temporal windows, masks, climate and terrain sources. |
| Methods | Indices, recovery definition, predictors, models, splits, uncertainty, and statistical rules. |
| Results | Data quality, baseline, model comparison, held-out fire results, ecosystem transfer, uncertainty. |
| Discussion | Ecological meaning, why predictors matter, domain shift, limitations, and practical value. |
| Conclusion | Narrow evidence-based answer to the main question. |

Never describe the maps as official hazard maps or causal ecological forecasts. They are research estimates from observational Earth-observation data. Clearly separate association, prediction, and mechanism.

## 19. First action today

Do not download imagery yet. Start with the repository and the synthetic test. This proves the AI can work correctly before expensive data processing begins.

## 20. References and official sources

MTBS: https://www.mtbs.gov/

USGS Landsat Collection 2: https://www.usgs.gov/landsat-missions/landsat-collection-2

Copernicus Sentinel-2: https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2

NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/

NASADEM: https://www.earthdata.nasa.gov/data/catalog/lpcloud-nasadem-hgt-001

CHIRPS: https://www.chc.ucsb.edu/data/chirps

Microsoft Planetary Computer: https://planetarycomputer.microsoft.com/

XGBoost documentation: https://xgboost.readthedocs.io/

GeoPandas documentation: https://geopandas.org/en/stable/

Rasterio documentation: https://rasterio.readthedocs.io/

## 21. Final standard

### Definition of done

The project is journal-ready only when the data pipeline is reproducible, the final fire events were held out, all planned baselines and models are reported, low results are not hidden, uncertainty and failure cases are analysed, and a clean environment regenerates the final tables and figures. The purpose is to prove that you can think and work like a geospatial modeller - carefully, transparently, and scientifically.