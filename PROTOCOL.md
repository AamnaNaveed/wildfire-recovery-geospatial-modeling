# Study Protocol

This document freezes the study design before any modelling is performed.
It was written on 2026-10-01, before any baseline or machine-learning
model was trained on the feature tables.

Any deviation from this protocol must be documented in the manuscript's
methods section as a post-hoc change, with justification.

---

## 1. Study question

How do burn severity, terrain, climate, pre-fire vegetation, and ecosystem
type control the rate and spatial pattern of vegetation recovery after
wildfire in California?

Secondary: Can a model trained on some ecosystem types predict recovery in
a held-out ecosystem or held-out fire event without leaking spatial
information?

## 2. Study region

California, United States. Fires from 2018 to 2022.

## 3. Fire selection

15 fires selected from the MTBS Fire Occurrence Dataset, stratified across
three ecosystem groups (forest, shrubland, grassland) and across the five
years. Selection was performed in `05_classify_fires.py` before any
processing.

After feature-table construction, 10 fires remain in the usable pool
(6 "strong", 4 "weak"). Five fires are excluded:

- 3 invalid (MOJAVE, LOST LAKE, MILL): mean dNBR < 0.02, no burn signal
- 2 insufficient data (RANCH, WOOLSEY): fewer than 1,000 valid pixels

Exclusion criteria are defined here, before modelling, based only on
signal quality computed in `10_build_feature_tables.py`. They do not
depend on model performance.

## 4. Response variable

Primary: `dnbr` (differenced Normalized Burn Ratio), continuous.

dNBR = pre_fire_NBR - post_fire_NBR, computed from Sentinel-2 L2A
composites at MTBS pre- and post-image dates ± 30 days.

dNBR is clipped to [-1, 1] to remove physically impossible values.

## 5. Predictors

Fixed list. No feature will be added after seeing model performance.

- `pre_nbr` — pre-fire vegetation state
- `pre_ndvi` — pre-fire vegetation index
- `elevation` — metres
- `slope` — degrees
- `aspect_sin`, `aspect_cos` — aspect components
- `temperature` — annual mean, Celsius (ERA5-Land)
- `precipitation` — annual total, mm (ERA5-Land)
- `landcover_class` — ESA WorldCover class

Categorical predictors will be one-hot encoded before modelling.

**Feature engineering is frozen at this point.** No interactions, no
derived ratios, no automated feature selection that depends on outcomes.

## 6. Models

Three models are pre-registered. All three will be reported regardless of
performance.

### 6.1 Persistence baseline

Predict dNBR as the mean dNBR of the same ecosystem group in the training
fires. Required reference point. If the ML model cannot beat this under
leave-one-fire-out validation, the manuscript must say so.

### 6.2 Random Forest

Spatially blocked. Fixed starting configuration:
- 500 trees
- max_features = sqrt
- min_samples_leaf = 5
- random_state = 42

Tuning (if performed) is restricted to development fires only.

### 6.3 XGBoost

Starting configuration:
- n_estimators = 800
- learning_rate = 0.03
- max_depth = 6
- min_child_weight = 5
- subsample = 0.8
- colsample_bytree = 0.8
- reg_lambda = 1
- objective = reg:squarederror
- random_state = 42

Early stopping only against validation folds; never against test.

## 7. Validation

Primary evaluation: **leave-one-fire-out (LOFO)** cross-validation over the
10 usable fires. Each fire is held out once; the model trains on the other
9 and predicts the held-out fire.

Secondary evaluation: **leave-one-ecosystem-out (LOEO)**.
Train on two ecosystem groups, predict the third. Reported separately.

Spatial diagnostics: Moran's I of residuals per held-out fire.

**Random pixel splits are prohibited.** They leak spatial information and
produce inflated performance estimates.

## 8. Metrics

For each held-out fire and aggregated:
- RMSE
- MAE
- R²
- Spearman correlation

Reported as mean ± SD across fires.

## 9. Uncertainty

Five random seeds per ML model. Report:
- Mean prediction and SD across seeds
- Prediction interval width
- Correlation between uncertainty and error

## 10. Test set

Held-out test fires are frozen at this point. Only one evaluation run is
permitted on them, using the frozen final configuration.

Test fire assignment:
- Training fires: 7 (of the 10 usable)
- Test fires: 3

Test fires will be named in a separate file `data/interim/test_fires.txt`
and will not be inspected by any tuning code.

## 11. Reporting rules

- All three models will be reported, including those that perform poorly.
- Low results will not be hidden.
- Null results will not be reframed as positive.
- If the ML model cannot beat the persistence baseline, that is reported.
- If cross-ecosystem transfer fails, that is reported.
- All code, configs, and outputs are preserved for reproducibility.

## 12. Known limitations (declared before modelling)

- 5 of 15 fires were excluded due to insufficient burn signal or
  insufficient valid pixels.
- Biome composition is uneven: 5 forest, 3 shrubland, 2 grassland in the
  usable pool.
- One fire (SCU) has a WorldCover label ("grassland") that may not match
  its actual vegetation type.
- MTBS pre/post image dates are 6–12 months apart for most fires, not
  immediate pre/post.
- NBR is a proxy for burn severity, not a direct measurement of
  vegetation recovery.

## 13. Stage outcomes and protocol amendments

Any deviation from the protocol, or any post-hoc finding that affects
interpretation, is recorded here.

### Stage 9–10 (Random Forest and XGBoost) — outcome note

Both models were trained under the frozen configurations. Random Forest
outperformed XGBoost on every metric (LOFO RMSE 0.197 vs 0.207; LOFO
Spearman 0.336 vs 0.194). Both models modestly beat the persistence
baseline (RMSE 0.216). Random Forest is therefore the primary model for
interpretation and reporting; XGBoost is reported as a secondary model.

### Stage 11 (Leave-one-ecosystem-out) — outcome note

Random Forest achieved positive mean R² (+0.036) under LOEO, higher than
its LOFO mean R² (−0.10). The hardest ecosystem to predict was forest
(RMSE 0.226), which is consistent with the high within-fire heterogeneity
of large forest fires. XGBoost underperformed RF under LOEO as well
(mean R² −0.12).

### Stage 12 (Uncertainty quantification) — outcome note

Seed-based uncertainty quantification (5 random seeds per model) revealed
that Random Forest predictions are highly stable across seeds (mean
prediction std = 0.006; XGBoost = 0.016). This indicates that seed
variance captures model initialization noise only, not true predictive
uncertainty. The resulting 95% prediction intervals show coverage of only
5% (RF) and 13% (XGB), far below the nominal 95%. This is reported as a
limitation of the uncertainty method, not as a model failure. True
predictive intervals would require alternative methods (e.g. quantile
regression forests, conformal prediction); these are proposed as future
work rather than added post-hoc.

---

**Frozen at**: 2026-10-01
**Stage outcome notes added**: 2026-10-01
**Signed**: Aamna Naveed