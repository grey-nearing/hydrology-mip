# `metrics`: Scoring Package

The `metrics` package scores daily streamflow predictions (in mm/day) for the 531 CAMELS-US
basins over the evaluation period, 1989-10-01 to 1999-09-30 (3,652 days). It reads one
hydrograph store and the observation targets store from a root folder. The root can be a folder
on your own computer or the maintainers' bucket. It computes the 14 metrics of the Gauch et al.
(2023) suite: `NSE`, `MSE`, `RMSE`, `KGE`, `logNSE`, `logKGE`, `Alpha-NSE`, `Pearson-r`,
`Beta-KGE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`, and `Peak-Timing`.

The metric code itself lives in
[`hydromip/metrics/deterministic_daily.py`](../hydromip/metrics/deterministic_daily.py).
[`metrics/metrics.py`](metrics.py) only re-exports it.

All 531 basins have valid observations on all 3,652 evaluation days. So every basin is scored,
and every prediction must be a finite number. A store with a missing basin or a missing value
fails.

## Overview

| Component | Entry point | Inputs | Output |
| :--- | :--- | :--- | :--- |
| Command line | `mip-score-model` (or `python -m metrics.cli`) | `<root>/hydrographs/<leaderboard_id>/<model_id>.zarr` and `<root>/observations/newman_eval_targets.zarr` | Prints a JSON summary. Can save the full result as JSON. |
| Observation targets | `python -m metrics.build_eval_targets` | The raw CAMELS-US v1.2 files and the split files | `newman_eval_targets.zarr` |
| Python API | `calculate_model_leaderboard_metrics(model_id, leaderboard_id, bucket_uri=...)` | Same as the command line | A dictionary with all scores |
| One basin | `calculate_all_metrics(obs, sim, resolution="1D", datetime_coord=None)` | Two 1-D `xarray.DataArray` time series | A dictionary with the 14 metrics |

## Prerequisites

- Python 3.11 or newer, with the repository installed: `pip install -e .` from the repository root.
- Run every command from the repository root.
- To score on your own computer, you need the raw CAMELS-US v1.2 time series (see
  [`mips/camels_us_daily/data/README.md`](../mips/camels_us_daily/data/README.md)). Only two raw
  folders are used here: `usgs_streamflow/` and `basin_mean_forcing/daymet/`.
- You do not need a Google Cloud account.

## Quick start: score a model on your own computer

The scorer reads a root folder with this layout:

```text
local_bucket/
├── observations/newman_eval_targets.zarr
└── hydrographs/<leaderboard_id>/<model_id>.zarr
```

Git ignores Zarr stores (`*.zarr/`), so nothing in `./local_bucket/` is committed.

**Step 1. Build the observation targets once.**

```bash
python -m metrics.build_eval_targets \
  --camels-dir /path/to/camels_us/basin_dataset_public_v1p2 \
  --splits-dir mips/camels_us_daily/splits \
  --output ./local_bucket/observations/newman_eval_targets.zarr
```

**Step 2. Put your hydrograph store in the folder of its leaderboard.** For example, run a model
straight into `./local_bucket/` (see [`models/README.md`](../models/README.md)). The persistence
benchmark below has no parameters and runs in seconds:

```bash
mip-run-model --model naive_lag1_persistence --spatial-regime gauged \
  --forcings daymet --forcing-regime lagged_streamflow \
  --model-id persistence_test \
  --output-zarr ./local_bucket/hydrographs/gauged__lagged_streamflow/persistence_test.zarr
```

Without `--output-zarr`, `mip-run-model` writes to `./outputs/<leaderboard_id>/<model_id>.zarr`.
Then copy that store into `./local_bucket/hydrographs/<leaderboard_id>/`. To package predictions
from your own code instead, use `models/zarr_writer.py` (see
[`models/README.md`](../models/README.md)).

**Step 3. Score the store.**

```bash
mip-score-model \
  --model-id persistence_test \
  --leaderboard-id gauged__lagged_streamflow \
  --bucket-uri ./local_bucket \
  --output-json ./outputs/persistence_test_scores.json
```

The printed summary must show `"num_evaluated_basins": 531` and `"pred_nan_when_obs_valid": 0`.
For this persistence model, `"median_nse"` is `0.4542`.

**Check in the browser.** The Score page of <https://hydrologymip.org> runs the same checks. Upload
a zipped Zarr store (`.zarr.zip`) or a `.npz` file with the arrays `qsim`, `basin`, and `date`.
Bare `.npy` files are not accepted.

To submit your result, follow [`CONTRIBUTING.md`](../CONTRIBUTING.md#submit).

## Official scores (Maintainers only)

The official leaderboard scores come from the stores in the private bucket `gs://camels-us-mip`.
Without `--bucket-uri`, `mip-score-model` uses `storage.bucket_uri` from
`mips/camels_us_daily/protocol.json`, which is that bucket. Reading it needs Google Cloud
credentials (see [`hydromip/README.md`](../hydromip/README.md#zarr-io-and-cloud-credentials)).

```bash
mip-score-model --model-id <model_id> --leaderboard-id gauged__multi_camels
```

## What the scorer checks

The observation targets store must have:

- a `qobs` variable with dimensions `(basin, date)` and shape `(531, 3652)`;
- `basin` coordinates with 531 sorted 8-digit gauge IDs, `date` coordinates for every day from
  1989-10-01 to 1999-09-30, and `huc_02` coordinates.

The hydrograph store must have:

- a `qsim` variable with dimensions `(basin, date)` and shape `(531, 3652)`, in mm/day;
- the same `basin` IDs in the same order, and the same `date` values, as the targets;
- if it has `huc_02` coordinates, the same regions as the targets;
- no infinite value, and no missing value on a day with a valid observation
  (`pred_nan_when_obs_valid` must be 0);
- if it has `qsim_members` (`(member, basin, date)`): an ensemble store (`is_ensemble` true) must
  have `qsim` equal to the mean of the members, or to the reduction named in its
  `ensemble_reduction` attribute; a single-run store must have `qsim` equal to exactly one member.

A basin is scored when it has at least 30 valid observation days and its observations vary.
All 531 basins pass this test. Any failed check raises an error that names the problem.

## The 14 metrics

Each metric is computed per basin on the days with valid observations. The leaderboards summarize
them over the basins.

| Metric | Key in `per_basin` | Meaning | Best value |
| :--- | :--- | :--- | :--- |
| `NSE` | `nse` | Nash-Sutcliffe efficiency: 1 minus the squared error divided by the variance of the observations. 0 means "as good as the mean observed flow". | 1 |
| `MSE` | `mse` | Mean squared error. | 0 |
| `RMSE` | `rmse` | Root mean squared error. | 0 |
| `KGE` | `kge` | Kling-Gupta efficiency. It combines correlation, the variability ratio, and the mean ratio. | 1 |
| `logNSE` | `log_nse` | NSE of log-transformed flows, `log(max(Q, 0) + 1e-5)` for both observed and simulated flow. It puts more weight on low flows. | 1 |
| `logKGE` | `log_kge` | KGE of the same log-transformed flows. | 1 |
| `Alpha-NSE` | `alpha_nse` | Ratio of the standard deviations (simulated / observed). | 1 |
| `Pearson-r` | `pearson_r` | Pearson correlation. | 1 |
| `Beta-KGE` | `beta_kge` | Ratio of the means (simulated / observed). | 1 |
| `Beta-NSE` | `beta_nse` | Difference of the means, divided by the standard deviation of the observations. | 0 |
| `FHV` | `fhv` | Volume bias of the highest 2% of flows (peak flows), in percent. | 0 |
| `FMS` | `fms` | Bias of the middle slope of the flow duration curve (20% to 70% exceedance), in percent. | 0 |
| `FLV` | `flv` | Volume bias of the lowest 30% of flows, in percent. | 0 |
| `Peak-Timing` | `peak_timing` | Mean timing error of flow peaks, in days. | 0 |

The five **headline scores** of a leaderboard row are:

1. **Median NSE** (`median_nse`): the middle NSE over the basins. Leaderboards are ranked by it.
2. **Mean NSE** (`mean_nse`).
3. **Median KGE** (`median_kge`).
4. **Median |FHV|** (`median_abs_fhv`): the middle absolute peak-flow bias, in percent.
5. **Failure count** (`nse_le_zero_count`): the number of basins with NSE ≤ 0.

The leaderboard tables show 16 summary columns: the median of each of the 14 metrics, plus the
mean NSE and the mean KGE.

## Output

`calculate_model_leaderboard_metrics` returns one dictionary:

| Key | Content |
| :--- | :--- |
| `model_id`, `leaderboard_id` | The scored entry. |
| `bucket_uri`, `hydrograph_uri`, `targets_uri` | The root and the two stores that were read. |
| `available_gauch_2023_metrics` | The 14 metric names. |
| `summary` | All summary scores (see below). Every `summary` key is also copied to the top level. |
| `regional_huc02` | The median NSE of each HUC-02 region. |
| `per_basin` | `basins`, `huc_02`, `is_evaluable`, and one list of 531 values per metric, under both names (for example `NSE` and `nse`), plus `abs_fhv`, `abs_fms`, and `abs_flv`. A basin that is not scored has `NaN`. |
| `attributes` | The attributes of the hydrograph store. |

`summary` holds:

| Group | Keys |
| :--- | :--- |
| Counts | `num_total_basins`, `num_evaluated_basins`, `num_eval_days`, `pred_nan_when_obs_valid`, `allow_partial_archive`, `num_excluded_basins`, `excluded_basins` |
| Ensemble | `is_ensemble`, `num_members`, `has_qsim_members`, `ensemble_source`, `single_seed_member_index`, `single_seed_median_nse_mean`, `single_seed_median_nse_std`, `single_seed_median_nse_per_member` |
| Headline scores | `median_nse`, `mean_nse`, `median_kge`, `median_abs_fhv`, `nse_le_zero_count` |
| NSE and KGE spread | `min_`, `p01_`, `p05_`, `p10_`, `q25_` (= `p25_`), `p50_`, `q75_` (= `p75_`), and `max_` with the suffix `nse` or `kge`, plus `mean_kge` |
| Peak-flow bias spread | `min_abs_fhv`, `q25_abs_fhv`, `p50_abs_fhv`, `p75_abs_fhv` (= `q75_abs_fhv`), `p90_abs_fhv`, `p95_abs_fhv`, `p99_abs_fhv`, `max_abs_fhv`, `mean_abs_fhv` |
| Other metrics | `median_`, `q25_`, and `q75_` with the suffix `mse`, `rmse`, `log_nse`, `log_kge`, `alpha_nse`, `pearson_r`, `beta_kge`, `beta_nse`, `fhv`, `fms`, `abs_fms`, `flv`, `abs_flv`, or `peak_timing` |
| Regions | `median_of_18_huc02_medians`, `worst_huc02_region`, `best_huc02_region` |
| All metrics | `gauch_2023_metrics`: for each of the 14 metrics, `median`, `mean`, `min`, `p01`, `p05`, `p10`, `q25`, `q75`, `p90`, `p95`, `p99`, and `max` |

`mip-score-model` prints `model_id`, `leaderboard_id`, `hydrograph_uri`, `summary`, and
`regional_huc02`. `--output-json` saves the full dictionary.

## Python API

```python
from metrics import calculate_model_leaderboard_metrics

result = calculate_model_leaderboard_metrics(
    model_id="persistence_test",
    leaderboard_id="gauged__lagged_streamflow",
    bucket_uri="./local_bucket",
)
print("Median NSE:", result["median_nse"])
print("Mean NSE:", result["mean_nse"])
print("Median KGE:", result["median_kge"])
print("Median |FHV| (%):", result["median_abs_fhv"])
print("Basins with NSE <= 0:", result["nse_le_zero_count"])
print("Metrics:", list(result["gauch_2023_metrics"]))
```

To score one basin, pass two time series with a date coordinate:

```python
import numpy as np
import pandas as pd
import xarray as xr

from metrics import calculate_all_metrics

dates = pd.date_range("1989-10-01", "1999-09-30", freq="D")
rng = np.random.default_rng(0)
obs = xr.DataArray(rng.gamma(2.0, 1.0, dates.size), coords={"date": dates}, dims="date")
sim = 0.9 * obs + 0.1
scores = calculate_all_metrics(obs, sim)
print(round(scores["NSE"], 3), round(scores["logNSE"], 3), len(scores))
```

### `calculate_model_leaderboard_metrics` arguments

| Argument | Type | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `model_id` | `str` | required | The store name: `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr`. Letters, digits, `.`, `_`, and `-` only. |
| `leaderboard_id` | `str` | required | One of the 12 leaderboard IDs (`CANONICAL_LEADERBOARD_IDS`). |
| `bucket_uri` | `str` | `"gs://camels-us-mip"` | The root folder or bucket. Use a local folder such as `"./local_bucket"`. |
| `allow_partial_archive` | `bool` | `False` | Maintainers only, for author archives that miss whole basins (`Archive*` rows). Allows whole basins to be missing, never single days. |
| `evaluation_mode` | `str` or `None` | `None` | `None` or `"ensemble"` scores `qsim`. `"single_member_mean"` scores each member and averages; it needs at least two members. |

### Other names

`metrics` also exports `calculate_all_metrics`, `calculate_metrics`, `get_available_metrics`, the
single-metric functions (`nse`, `mse`, `rmse`, `kge`, `log_nse`, `log_kge`, `alpha_nse`, `pearsonr`,
`beta_kge`, `beta_nse`, `fdc_fhv`, `fdc_fms`, `fdc_flv`, `mean_peak_timing`), the helpers
`infer_datetime_coord`, `get_frequency_factor`, and `AllNaNError`, and these constants:

| Constant | Value |
| :--- | :--- |
| `CANONICAL_LEADERBOARD_IDS` | The 12 leaderboard IDs. |
| `DEFAULT_BUCKET_URI` | `"gs://camels-us-mip"` |
| `EVAL_START_DATE`, `EVAL_END_DATE` | `"1989-10-01"`, `"1999-09-30"` |
| `NUM_BASINS`, `NUM_EVAL_DAYS` | `531`, `3652` |
| `GAUCH_2023_METRICS` | The 14 metric names, in the order above. |

## Argument reference

### `mip-score-model`

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--mip MIP` | `camels_us_daily` | Registered MIP. This scorer supports only `camels_us_daily`. |
| `--model-id MODEL_ID` | required | The store name in `hydrographs/<leaderboard_id>/<MODEL_ID>.zarr`. |
| `--leaderboard-id ID` | required | One of the 12 leaderboard IDs. It must be declared by the MIP. |
| `--bucket-uri URI` | `storage.bucket_uri` of the MIP (`gs://camels-us-mip`) | The root folder or bucket. Contributors pass a local folder, for example `./local_bucket`. |
| `--output-json PATH` | none | Save the full result as JSON. Missing parent folders are created. |
| `-h`, `--help` | — | Show the help text. |

### `python -m metrics.build_eval_targets`

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--camels-dir PATH` | required | The raw `basin_dataset_public_v1p2` folder. It needs `usgs_streamflow/` and `basin_mean_forcing/daymet/`. |
| `--splits-dir PATH` | required | The folder with `basins_531.json` and `splits_and_regimes.json`: `mips/camels_us_daily/splits`. |
| `--output PATH_OR_URI` | required | Where to write the store: a local path or a `gs://` URI. An existing store is overwritten. |
| `--verify-against PATH_OR_URI` | none | An existing targets store. The new build must match it before anything is written; otherwise the command fails. |
| `--atol VALUE` | `0.001` | Allowed difference in mm/day for `--verify-against`. |
| `-h`, `--help` | — | Show the help text. |

## The observation targets store

`build_eval_targets` rebuilds `newman_eval_targets.zarr` from the raw files. Anyone can repeat
the build; the same raw files always give the same store.

- **Dimensions:** `basin` (531), `date` (3,652 days, 1989-10-01 to 1999-09-30), and `date_all`
  (10,593 days, 1980-01-01 to 2008-12-31).
- **Coordinates:** `basin` (sorted gauge IDs), `date`, `date_all`, and `huc_02` (the region folder
  of each basin's files).
- **Variables:** `qobs` (`basin`, `date`) and `qobs_all` (`basin`, `date_all`) in mm/day, plus the
  masks `newman_test_mask` and `newman_train_mask` (`date_all`).
- **Units:** `mm/day = cfs * 0.028316846592 * 86400 * 1000 / area_m2`. `area_m2` is the basin area
  from the third header line of the basin's Daymet forcing file.
- **Missing values:** `-999` and any negative discharge become NaN, never 0.
- **Checks:** every basin needs a streamflow file and a Daymet file in the same HUC-02 folder, and
  that folder must match `basin_to_huc02` in `splits_and_regimes.json`. A missing file stops the
  build.

**Maintainers only:** pass
`--verify-against gs://camels-us-mip/observations/newman_eval_targets.zarr` to prove that a
rebuild matches the official store.

## Tests

```bash
pytest metrics/tests
```

The tests cover the 14 metrics on known values, the completeness and anti-masking checks, the
ensemble and single-run rules, the store ID rules, and the targets builder. One test compares a
full rebuild with the official store; it needs the raw data and bucket access.
