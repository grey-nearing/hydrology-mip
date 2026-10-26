---
name: algorithm-rules-and-norms
description: >-
  Non-negotiable algorithmic rules, data-integrity invariants, error-handling
  norms, centralized split/time-period enforcement, and cloud-only metric
  calculation requirements across all metrics, models, era_experiments, and
  frontend code in camels-us-mip. Use whenever writing, modifying, or reviewing
  algorithms, model runners, ERA notebooks, and data pipelines.
---

# Algorithm Rules, Assumptions & Data-Integrity Norms (`camels-us-mip`)

Every model runner, metrics evaluator, Zarr writer, ERA experiment notebook, and leaderboard pipeline in this repository must strictly obey the following algorithmic rules and data-integrity norms.

---

## 1. Missing Data In Always Means `NaN` Out (Zero Imputation or Masking)

- **Zero Silent Imputation or Gap-Filling:**
  - **Never** replace missing dates, missing streamflow observations, missing meteorological forcings, or missing catchment predictions with `0.0`, forward-filled values (`ffill`), backward-filled values (`bfill`), linear interpolation, or climatological averages.
  - **Never** substitute backup or fallback data products or local files when a requested cloud Zarr store or forcing dataset is unavailable.
  - **Never** mask out missing model predictions (`np.isnan(qsim)`) when observed discharge (`qobs`) is valid (`~np.isnan(qobs)`). Any `NaN` in model predictions on a day with valid observed discharge (`pred_nan_when_obs_valid > 0`) must immediately raise a `ValueError` or be explicitly counted as a catastrophic failure—never silently dropped from a pairwise mask `(~np.isnan(obs)) & (~np.isnan(sim))` to inflate metrics.
- **Explicit Missing-Data Representation:**
  - Missing numerical values in Zarr stores and arrays must always be represented as `np.nan` (`float('nan')`).
  - Missing required metadata fields, basin IDs, or dates must raise `KeyError` or `ValueError`—never default to `0.0` or empty arrays.

---

## 2. Zero `try`/`except`/`finally` Blocks That Mask Failures

- **Do Not Catch Exceptions to Hide Bad Data, Missing Files, or Broken Code:**
  - **Zero `try`/`except` blocks** are permitted for data loading, Zarr store reading, split parsing, model execution, or metric calculation across `metrics/`, `models/`, `era_experiments/`, and `benchmarks/`.
  - Never catch exceptions (`Exception`, `KeyError`, `ValueError`, `OSError`, `FileNotFoundError`) to log a warning, fall back to an alternate path/key, or return `NaN`, `None`, or a default dictionary.
- **Validate Preconditions Explicitly:**
  - Check local file/directory existence explicitly with `os.path.exists` / `Path.exists()` and GCS store existence explicitly via `gcsfs.GCSFileSystem.exists()`.
  - Check dictionary keys, DataFrame columns, and Zarr variable names, coordinates (`basin`, `date`), and dimensions (`(531, 3652)`) explicitly with `in` and shape equality checks, raising descriptive `ValueError`, `KeyError`, or `FileNotFoundError` immediately when expectations are violated.

---

## 3. Cloud-Only Metrics Calculation (`gs://camels-us-mip/`) & Gauch et al. (2023) Metrics Suite

- **Single Source of Truth for Hydrograph Scoring (`metrics/`):**
  - All metrics for models in this study (and all archived hydrograph contributions) **must** be calculated directly from the standardized Zarr hydrograph stores in **`gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`** against the canonical target store **`gs://camels-us-mip/observations/newman_eval_targets.zarr`**.
  - This rule is **without exception**:
    - No fallback paths to local `.npz`, `.npy`, `.csv`, or `.nc` files.
    - No `try`/`except` loops probing multiple candidate filenames or dictionary keys (`qsim`, `preds`, `oof_preds`, `y_hat`).
    - One standalone metrics calculation pipeline in `metrics/` with a single public API (`calculate_model_leaderboard_metrics(model_id, leaderboard_id, bucket_uri="gs://camels-us-mip")`) that computes the metrics for **one model and one leaderboard** directly from `gs://camels-us-mip/`.
- **Exact Gauch et al. (2023) *"In Defense of Metrics"* Suite (`metrics/metrics.py`):**
  - All per-basin hydrograph metrics must be computed via `metrics/metrics.py` (copied directly from Gauch et al., 2023, *Water Resources Research*, `10.1029/2022WR033918` / `neuralhydrology/evaluation/metrics.py`), which computes all 14 standard hydrological metrics:
    `NSE`, `MSE`, `RMSE`, `KGE`, `Alpha-NSE`, `Pearson-r`, `Beta-KGE`, `Beta-NSE`, `FHV` (top 2% peak bias), `FMS` (20%–70% FDC mid-slope bias), `FLV` (bottom 30% low-flow bias), `Peak-Timing` (mean absolute peak timing error in days), `Missed-Peaks` (fraction of observed peaks above the 80th percentile missed within $\pm 1$ day), and `Peak-MAPE` (mean absolute percentage error on observed peaks).
- **Strict Complete-Basin Rule (No Missing Basins Allowed):**
  - Every model Zarr store must have the full canonical `(531, 3652)` coordinate grid covering all 531 canonical Newman basins (`splits/basins_531.txt`), all 531 of which have complete valid USGS discharge observations (`3,652 / 3,652` days) in the 1989–1999 evaluation period.
  - **No missing basins and no partial time-series masking are allowed:** `qsim` must be finite on 100% of days and basins where `qobs` is valid (`pred_nan_when_obs_valid == 0`). Any model store missing even a single basin or single valid observation day immediately raises a `ValueError`.

---

## 4. Centralized Split, Basin, & Time-Period Control (`models/run_model.py`)

- **All Model Runs Must Launch via `models/run_model.py`:**
  - Every model in `models/<model_name>/` must be trained and evaluated via the single central deployment script `models/run_model.py`.
  - **Strict Separation of Responsibilities:**
    - `models/run_model.py` is the **only** script allowed to:
      1. Load the canonical 531-basin list, 12-fold PUB splits (`splits/pub_12fold_random_splits.json`), and 18-fold HUC-02 PUR splits (`splits/pur_18fold_huc02_splits.json`).
      2. Define the canonical training time period (`1999-10-01` to `2008-09-30`) and out-of-sample evaluation time period (`1989-10-01` to `1999-09-30`).
      3. Perform the cross-validation fold looping or parallel fold deployment across the 12 PUB folds or 18 PUR folds.
      4. Pass the explicit `train_basins`, `test_basins`, `train_start_date`, `train_end_date`, `eval_start_date`, and `eval_end_date` arguments to the model-specific runner in `models/<model_name>/`.
      5. Collect all fold/seed outputs, verify 100% out-of-sample coverage across all 531 basins and 3,652 evaluation days (`(531, 3652)`), standardize them into a canonical Zarr store (`dims: ("basin", "date")`), and upload them to `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`.
    - Individual model scripts inside `models/<model_name>/` **must NEVER**:
      - Load basin split files or `.txt`/`.json` basin lists on their own.
      - Choose or hardcode training or testing date ranges.
      - Assign basins to folds or filter the 531-basin evaluation cohort.

---

## 5. Strict Temporal & Spatial Anti-Leakage Rules (Including ERA Experiments)

- **Zero Test-Period Data Leakage:**
  - Normalization statistics (feature means, standard deviations, target scalers, and per-basin loss weights) must be computed **strictly** on the training basins (`train_basins`) and training time period (`train_start_date` .. `train_end_date`). Never compute normalization statistics across test dates or held-out PUB/PUR test basins.
  - When evaluating out-of-sample test periods (`1989-10-01` to `1999-09-30`), meteorological spin-up/warmup (e.g., 365 days prior to `1989-10-01`) may use meteorological forcing inputs (`1988-10-01` to `1989-09-30`), but observed streamflow (`QObs`) from the evaluation window must never be fed into non-autoregressive models or used for weight selection/early stopping.
- **ERA Hillclimbing Anti-Leakage Rules (`era_experiments/`):**
  - ERA hillclimbing notebooks must **never** touch the Newman test period (`1989-10-01` to `1999-09-30`) or the pre-1989 period (`1980-01-01` to `1989-09-30`) for either training or candidate selection.
  - All 6 ERA prompt notebooks (`era_gauged_minimal_guidance.ipynb`, `era_gauged_physics_informed.ipynb`, `era_pub_minimal_guidance.ipynb`, `era_pub_physics_informed.ipynb`, `era_pur_minimal_guidance.ipynb`, `era_pur_physics_informed.ipynb`) must train on `2003-10-01` to `2008-09-30` (`ERA_TRAIN_MASK`) and evaluate/hillclimb strictly within the Newman training period on `1999-10-01` to `2003-09-30` (`ERA_EVAL_MASK`).
  - For PUB (`12` folds) and PUR (`18` folds), every ERA candidate trial must execute an actual loop over all $k$ canonical folds (training only on each fold's training basins during `ERA_TRAIN_MASK` and predicting on that fold's held-out test basins), assemble the complete 531-basin out-of-sample hydrograph matrix before evaluation, and hillclimb on Median NSE computed from the full stitched out-of-fold hydrographs over `ERA_EVAL_MASK`.

---

## 6. Mandatory Comprehensive Canonical Benchmarks (`benchmarks/`)

- In addition to fast, hermetic unit and integration tests in `<package>/tests/`, core algorithmic packages (`metrics/`, `models/`) must have companion verification harnesses in `benchmarks/` (`benchmarks/metrics_pipeline.py`) with unit tests in `benchmarks/tests/`.
- All benchmarks must strictly follow [`skills/benchmarking.md`](./benchmarking.md): zero fallback to local/reference data, zero silent `NaN` masking (`pred_nan_when_obs_valid` must be explicitly counted and rejected), zero imputation, zero dropping of failed/outlier basins from headline tables, and mandatory lower-tail (`[Min, P1, P5, P10, P25, P50]`) and failure-rate (`NSE <= 0`) reporting.
