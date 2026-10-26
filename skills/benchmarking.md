---
name: benchmarking
description: >-
  End-to-end canonical benchmarking protocol, anti-masking/anti-imputation/zero-fallback
  hard rules, lower-tail and failure-rate reporting standards, and CLI runbooks
  for the root benchmarks/ package in camels-us-mip. Use whenever running,
  adding, updating, or reviewing canonical benchmarks and leaderboard tables.
---

# Canonical Benchmarking Protocol & Standards (`skills/benchmarking.md`)

This skill defines the mandatory rules, lower-tail/failure-rate reporting standards, and CLI runbooks for benchmarking and scoring all models in `camels-us-mip` against the canonical 531-basin Newman et al. (2015, 2017) / Kratzert et al. (2019b, 2021) evaluation protocol on `gs://camels-us-mip/`.

---

## 1. Non-Negotiable Anti-Masking, Anti-Imputation & Failure-Reporting Hard Rules

> **CRITICAL HARD RULE:** Under no circumstances may any model runner, Zarr converter, benchmark script, or metrics evaluator mask bad data, impute missing values, drop failed/outlier basins from headline metrics, or fall back to alternate/local data when a cloud Zarr store or model prediction fails.

Every model evaluation and leaderboard table MUST strictly enforce the following **five invariants**:

1. **Zero Fallback Paths:**
   - Metrics in `metrics/` must be computed **exclusively** from the standardized Zarr hydrographs in `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` and the canonical observations in `gs://camels-us-mip/observations/newman_eval_targets.zarr`.
   - Never fall back to local `.npz` files, alternate keys, or backup directories.
2. **Zero Silent `NaN` Masking of Predictions (`pred_nan_when_obs_valid == 0`):**
   - A pairwise mask `(~np.isnan(obs)) & (~np.isnan(sim))` must **never** be used without first verifying that `sim` has zero `NaN`s wherever `obs` is valid (`pred_nan_when_obs_valid = int(np.sum((~np.isnan(obs)) & (obs >= 0.0) & np.isnan(sim))) == 0`).
   - Any model hydrograph with `pred_nan_when_obs_valid > 0` must raise a `ValueError` immediately.
3. **Strict Complete-Basin Rule (No Missing Basins Allowed — All 531 Basins Required):**
   - All 531 canonical CAMELS-US basins (`splits/basins_531.txt`) have complete valid USGS discharge observations (`3,652 / 3,652` days) in the 1989–1999 test decade and must be evaluated in every Gauged, 12-fold PUB, and 18-fold PUR run.
   - No missing basins are allowed: any submission or Zarr store missing even a single basin or valid observation day (`pred_nan_when_obs_valid > 0`) immediately raises a `ValueError`.
   - Never drop catastrophic failure basins (`NSE <= 0`) when computing `Mean NSE`, `Median NSE`, `Median KGE`, or `Median |FHV|`.
4. **Zero Imputation or Interpolation:**
   - Missing observations (`qobs` `NaN`s) must remain `NaN`, and model predictions must never be interpolated or filled across missing days.
5. **Exact Gauch et al. (2023) 14-Metric Suite (`metrics/metrics.py`) + Lower-Tail & Failure-Rate Reporting:**
   - Every evaluation summary and benchmark report MUST compute all 14 Gauch et al. (2023) metrics (`NSE`, `MSE`, `RMSE`, `KGE`, `Alpha-NSE`, `Pearson-r`, `Beta-KGE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`, `Peak-Timing`, `Missed-Peaks`, `Peak-MAPE`) via `metrics/metrics.py`, alongside lower-tail skill percentiles (`Min`, `P1`, `P5`, `P10`, `P25`, `P50`, `P75`), upper-tail error percentiles (`P50`, `P75`, `P90`, `P95`, `P99`, `Max`), `NSE <= 0` failure counts, and 18-region USGS HUC-02 breakdowns.

---

## 2. Canonical CLI Runbook (`benchmarks/` & `metrics/`)

### 2.1 Score a Single Model & Leaderboard from `gs://camels-us-mip/`
```bash
mip-score-model \
  --model-id t4_06 \
  --leaderboard-id gauged__multi_camels \
  --bucket-uri gs://camels-us-mip
```

### 2.2 Run the Full End-to-End Cloud Zarr Metrics Benchmark Suite
```bash
benchmark-mip-metrics \
  --bucket-uri gs://camels-us-mip \
  --output-json leaderboards/tables_3_to_14.json
```
