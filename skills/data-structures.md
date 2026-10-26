---
name: data-structures
description: >-
  Canonical reference for all data structures, Zarr coordinates, dimensions,
  dtypes, chunking, canonical PUB (12-fold) and PUR (18-fold) split schemas,
  observed target Zarr stores, model hydrograph Zarr stores, and CAMELS-US
  ERA input tensors in camels-us-mip and gs://camels-us-mip/. Use whenever
  reading, writing, transforming, validating, or scoring data in metrics/,
  models/, era_experiments/, or frontend/.
---

# Data Structures, Coordinates & Schemas (`camels-us-mip`)

This skill defines the schemas, coordinate systems, dimensions, data types, and chunk layouts across `metrics/`, `models/`, `era_experiments/`, and `gs://camels-us-mip/`.

---

## 1. Unified Hydrograph Zarr Store Specification (`gs://camels-us-mip/hydrographs/`)

Every model hydrograph store written by `models/run_model.py` (or converted from published external hydrograph archives) and scored by `metrics/` must strictly adhere to the following `xarray.Dataset` Zarr schema:

| Property | Standard Specification |
| :--- | :--- |
| **Store URI Pattern** | `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` |
| **Primary 2D Dimensions** | `("basin", "date")` of exact shape `(531, 3652)` |
| **Optional 3D Member Dimensions** | `("member", "basin", "date")` of shape `(N_members, 531, 3652)` |
| **`basin` Coordinate** | `str` (`<U8`), length `531`, monotonically sorted canonical 8-digit USGS gauge IDs (`"01022500"` through `"14309500"` matching `splits/basins_531.txt`). |
| **`date` Coordinate** | `datetime64[ns]`, length `3652`, strictly contiguous daily (`1D`) timestamps from `1989-10-01T00:00:00` to `1999-09-30T00:00:00` (water years 1990–1999). |
| **`huc_02` Coordinate** | Non-dimension coordinate on `("basin",)`, `str` (`<U2`), length `531`, 2-digit USGS hydrologic region code (`"01"` through `"18"`). |
| **`member` Coordinate** *(optional)* | `int64`, `0 .. N_members - 1` when multi-seed / ensemble member trajectories are stored alongside the primary `qsim` array. |
| **`qsim` Data Variable** | `float32`, dims `("basin", "date")`, shape `(531, 3652)`, units `"mm/day"`, `fill_value = np.nan`. Contains the daily specific discharge predictions to be scored for the leaderboard entry. |
| **`qsim_members` Data Variable** *(optional)* | `float32`, dims `("member", "basin", "date")`, shape `(N_members, 531, 3652)`, units `"mm/day"`, `fill_value = np.nan`. |
| **Chunk Layout** | `{"basin": 531, "date": 3652}` (single consolidated chunk for `(531, 3652)` 2D array, ~7.7 MB uncompressed / ~2.5 MB compressed) with consolidated Zarr metadata (`consolidated=True`). |
| **Required Dataset Attributes (`ds.attrs`)** | Bucket / `metrics.evaluator` vocabulary (validated by `models/zarr_writer.py`): `entry_id` (`str`, == `model_id`), `model_name` (`str`), `family_id` (`str`), `provenance_category` (`Literature` / `MIP Re-run` / `Archive` / `Archive*` / `Contribution`), `leaderboard_id` (one of the 12 ids `gauged__single_camels` .. `pur__freeform`), `spatial_regime` (`gauged` / `pub` / `pur`), `forcing_regime` (`single_camels` / `multi_camels` / `lagged_streamflow` / `freeform`), `forcings_used` (`str`, e.g. `"Daymet, Maurer, NLDAS"`), `is_ensemble` (`bool`, == `num_members > 1`), `num_members` (`int`, == size of `member` when `qsim_members` exists), `train_period` (`"1999-10-01/2008-09-30"`), `test_period` (`"1989-10-01/1999-09-30"`, must equal the `date` coordinate range), `units` (`"mm/day"`), `code_url` (`str`), `archive_url` (`str`). Stores written by `run_model.py` additionally carry the aliases `model_id`, `model_family`, `table_id`, `spatial_protocol` (`gauged` / `pub_12fold` / `pur_18huc02`), `forcing_products` (`list[str]`), `training_period`, `evaluation_period`, `data_url`, and run provenance (`seeds`, `fold_ids`, `runner_kwargs_json`, `git_commit`, `created_utc`, `data_source`). `qsim` must equal the arithmetic mean of `qsim_members`. |

---

## 2. Canonical Observed Streamflow Target Store (`gs://camels-us-mip/observations/newman_eval_targets.zarr`)

The ground-truth observations used by `metrics/` are stored in `gs://camels-us-mip/observations/newman_eval_targets.zarr`:

| Coordinate / Variable | Dims & Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| **`basin`** | `("basin",)` `(531,)` | `<U8` | Canonical sorted 531 USGS 8-digit gauge IDs. |
| **`date`** | `("date",)` `(3652,)` | `datetime64[ns]` | Newman 10-water-year evaluation window (`1989-10-01` to `1999-09-30`). |
| **`date_all`** | `("date_all",)` `(10593,)` | `datetime64[ns]` | Full 29-year CAMELS-US record (`1980-01-01` to `2008-12-31`). |
| **`huc_02`** | `("basin",)` `(531,)` | `<U2` | Official 2-digit USGS hydrologic region code (`01` .. `18`). |
| **`qobs`** | `("basin", "date")` `(531, 3652)` | `float32` | Observed daily specific discharge (`mm/day`) during the Newman evaluation decade (`1989-10-01` to `1999-09-30`). Missing USGS observations are `np.nan`. |
| **`qobs_all`** | `("basin", "date_all")` `(531, 10593)` | `float32` | Observed daily specific discharge (`mm/day`) over `1980-01-01` to `2008-12-31`. |
| **`newman_train_mask`** | `("date_all",)` `(10593,)` | `bool` | `True` on `1999-10-01` to `2008-09-30` (3,288 days; water years 2000–2008). |
| **`newman_test_mask`** | `("date_all",)` `(10593,)` | `bool` | `True` on `1989-10-01` to `1999-09-30` (3,652 days; water years 1990–1999). |

---

## 3. Canonical Spatial Split Files (`splits/` and `gs://camels-us-mip/splits/`)

1. **`basins_531.txt` / `basins_531.json`:**
   - The 531 canonical Newman et al. (2015, 2017) / Kratzert et al. (2019b) 8-digit USGS gauge IDs in sorted order.
2. **`pub_12fold_random_splits.json` & `pub_12fold/fold_{00..11}_{train,test}.txt`:**
   - 12-fold random cross-validation across the 531 sorted basins (`np.array_split(np.random.RandomState(42).permutation(531), 12)`):
     - Folds `fold_00` .. `fold_02`: `486` training basins and `45` held-out test basins per fold.
     - Folds `fold_03` .. `fold_11`: `487` training basins and `44` held-out test basins per fold.
     - Disjoint test union across all 12 folds equals all `531` basins.
3. **`pur_18fold_huc02_splits.json` & `pur_18fold/huc_{01..18}_{train,test}.txt`:**
   - 18-fold Leave-One-HUC-02-Region-Out cross-validation across all 18 USGS two-digit hydrologic regions (`huc_01` through `huc_18`):
     - Each fold holds out 100% of basins in one HUC-02 region (`2` to `79` test basins) and trains on the remaining 17 regions (`452` to `529` training basins).
     - Disjoint test union across all 18 folds equals all `531` basins.

---

## 4. Standardized Five Leaderboard Metrics & Regional Breakdown (`metrics/evaluator.py`)

Given `qsim` of shape `(531, 3652)` (`mm/day`) from `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` and `qobs` of shape `(531, 3652)` (`mm/day`) from `gs://camels-us-mip/observations/newman_eval_targets.zarr`:

1. **Precondition & Anti-Masking Validation:**
   - Verify `qsim` has dimensions `("basin", "date")`, exact shape `(531, 3652)`, and coordinates identical to `qobs`.
   - Count `pred_nan_when_obs_valid = int(np.sum((~np.isnan(qobs)) & (qobs >= 0.0) & np.isnan(qsim)))`.
   - If `pred_nan_when_obs_valid > 0` for any basin that has valid observations, raise a `ValueError` (never silently drop predicted `NaN`s on valid observation days).
2. **Per-Basin Metric Definitions (on valid observation days `(~np.isnan(qobs)) & (qobs >= 0.0)`, requiring `>= 30` valid days and `SST > 1e-12`):**
   - **Nash–Sutcliffe Efficiency (`NSE`):**
     $$\text{NSE}_b = 1 - \frac{\sum_{t \in \mathcal{T}_b} (Q_{\text{sim},b,t} - Q_{\text{obs},b,t})^2}{\sum_{t \in \mathcal{T}_b} (Q_{\text{obs},b,t} - \bar{Q}_{\text{obs},b})^2}$$
   - **Kling–Gupta Efficiency (`KGE`, Gupta et al., 2009):**
     $$\text{KGE}_b = 1 - \sqrt{(r_b - 1)^2 + (\alpha_b - 1)^2 + (\beta_b - 1)^2}, \qquad \alpha_b = \frac{\sigma_{\text{sim},b}}{\sigma_{\text{obs},b}}, \quad \beta_b = \frac{\mu_{\text{sim},b}}{\mu_{\text{obs},b}}$$
   - **Top 2% Peak Flow Bias Magnitude (`|FHV|`, %, Yilmaz et al., 2008):**
     $$|\text{FHV}|_b = \left| \frac{\sum_{m=1}^{M_b} Q_{\text{sim},b,(m)} - \sum_{m=1}^{M_b} Q_{\text{obs},b,(m)}}{\sum_{m=1}^{M_b} Q_{\text{obs},b,(m)}} \right| \times 100\%, \qquad M_b = \max(1, \lfloor 0.02 \cdot |\mathcal{T}_b| \rfloor)$$
     where $Q_{\text{sim},b,(m)}$ and $Q_{\text{obs},b,(m)}$ are the top $M_b$ highest flows sorted independently per series (flow duration curve high-flow segment).
3. **Summary Aggregations Across the 531 Basins:**
   - **Headline 5 Metrics:** `median_nse`, `mean_nse`, `median_kge`, `median_abs_fhv` (%), and `nse_le_zero_count` (number of basins with $\text{NSE}_b \le 0.0$).
   - **Lower-Tail & Quartile Distribution:** `min_nse`, `p01_nse`, `p05_nse`, `p10_nse`, `q25_nse`, `q75_nse`, plus 18-region HUC-02 median NSE (`regional_huc02`).

---

## 5. CAMELS-US ERA Experiment Input & Time-Split Tensors (`era_experiments/`)

In the 6 ERA prompt notebooks (`era_gauged_minimal_guidance.ipynb`, `era_gauged_physics_informed.ipynb`, `era_pub_minimal_guidance.ipynb`, `era_pub_physics_informed.ipynb`, `era_pur_minimal_guidance.ipynb`, `era_pur_physics_informed.ipynb`):

- **Time Axis (`DATES`):** `3,653` daily steps (`1998-10-01` to `2008-09-30`, restricted strictly to the Newman training decade plus 1 year of meteorological spin-up).
  - `ERA_TRAIN_MASK`: `True` on `2003-10-01` to `2008-09-30` (water years 2004–2008; 1,827 days).
  - `ERA_EVAL_MASK`: `True` on `1999-10-01` to `2003-09-30` (`ERA_EVAL_MASK_00S`, water years 2000–2003; 1,461 days).
  - **Newman Test Period (`1989-10-01` to `1999-09-30`):** Completely excluded from the ERA dataset (only meteorological spin-up `1998-10-01` to `1999-09-30` is present, with `QObs` masked to `NaN`).
- **Starter Baseline Inputs (Kratzert Multi-Forcing CudaLSTM):**
  - `DYNAMIC_INPUTS`: `(531, 3653, 15)` `float32` — 5 meteorological variables (`prcp(mm/day)`, `srad(W/m2)`, `tmax(C)`, `tmin(C)`, `vp(Pa)`) across all 3 CAMELS-US forcing products (`daymet`, `maurer_extended`, `nldas_extended`).
  - `STATIC_ATTRIBUTES`: `(531, 27)` `float32` — the 27 Kratzert et al. (2019b, 2021) CAMELS catchment attributes.
- **Additional Official CAMELS Variables Available in Scope (Documented in Paper Appendix):**
  - `AUX_FORCINGS`: `(531, 3653, 6)` `float32` — `dayl(s)` (daylength) and `swe(mm)` (simulated snow water equivalent) from each of the 3 CAMELS forcing products.
  - `LAT`, `LON`, `BASIN_AREAS`, `ATTRIBUTES_DF`: official CAMELS gauge coordinates, drainage area, and attribute table.
