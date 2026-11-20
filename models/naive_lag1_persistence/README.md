# Naive Lag-1 Streamflow Persistence (`models/naive_lag1_persistence/`)

This is a zero-parameter, deterministic lower benchmark for the three **With Lagged Streamflow**
leaderboards: `gauged__lagged_streamflow` (Table 5), `pub__lagged_streamflow` (Table 9) and
`pur__lagged_streamflow` (Table 13).

$$\hat{Q}(t) = Q_{\text{obs}}(t - 1)$$

## Overview

| | |
|---|---|
| Model family ID | `naive_lag1_persistence` |
| Category | `Contribution` (statistical lower benchmark) |
| Parameters | 0. There is no training, so predictions are identical in the gauged, 12-fold PUB and 18-fold HUC-02 PUR protocols. |
| Inputs | Observed discharge of the **test** basin on earlier days only. No forcings are used. |
| Allowed regimes | `run_model.py` registers this model for `lagged_streamflow` only, in all three spatial protocols. Any other forcing regime is rejected. |
| Citation | Nearing et al. (2026), HydrologyMIP CAMELS-US MIP |

## Missing-day handling

The model is strictly causal, and nothing is fabricated without being declared.

- If `Q_obs(t-1)` is missing, the most recent finite observation `Q_obs(t')` with `t' < t` is carried forward. The runner reads a 30-day lookback before the evaluation window (`lookback_days`). It reads this window without spin-up masking, because the lagged-streamflow protocol allows past discharge.
- Two of the 531 basins, `02427250` and `09484600`, have gauge records that begin on the first day of the evaluation window, so there is nothing to persist on that day.
  - These two "cold-start" basin-days raise an error unless `cold_start_value` is passed explicitly.
  - The registry passes `cold_start_value=0.0` through `ModelRegistryEntry.default_kwargs`. That value is recorded in each store's `runner_kwargs_json` attribute and in the Stage-1 JSON.
  - The runner prints how many basin-days used it: 2 in the gauged protocol, and 1 in each of two folds for PUB and PUR.

## Running

From the repository root:

```bash
# One store per spatial protocol (written locally to ./outputs/<leaderboard_id>/<model_id>.zarr)
python -m models.run_model --model naive_lag1_persistence --spatial-regime gauged \
  --forcings daymet --forcing-regime lagged_streamflow --model-id naive_lag1_persistence_gauged
python -m models.run_model --model naive_lag1_persistence --spatial-regime pub_12fold \
  --forcings daymet --forcing-regime lagged_streamflow --model-id naive_lag1_persistence_pub
python -m models.run_model --model naive_lag1_persistence --spatial-regime pur_18huc02 \
  --forcings daymet --forcing-regime lagged_streamflow --model-id naive_lag1_persistence_pur

# Stage-1 record (no published reference; writes stage1_author_reproduction_results.local.json)
python -m models.naive_lag1_persistence.reproduce_author_metrics --mode train_and_eval
```

`--forcings daymet` only selects the product that the loader validates; the predictions use
discharge alone.

## Results (531 basins, evaluation 1989-10-01 to 1999-09-30)

These are the same in all three protocols, because the model has no parameters:

- Median NSE **0.454**, mean NSE 0.412, with 86 of 531 basins at NSE ≤ 0
- Median KGE **0.727**, mean KGE 0.706
- Median |FHV| 0.0 %
- Median peak-timing error 1.0 day, an exact one-day phase shift

The canonical stores are:
- `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/naive_lag1_persistence_gauged.zarr`
- `gs://camels-us-mip/hydrographs/pub__lagged_streamflow/naive_lag1_persistence_pub.zarr`
- `gs://camels-us-mip/hydrographs/pur__lagged_streamflow/naive_lag1_persistence_pur.zarr`

All three were produced with `models/run_model.py` and scored with
`metrics.evaluator.calculate_model_leaderboard_metrics`.
