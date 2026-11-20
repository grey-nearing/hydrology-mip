# CAMELS-US Daily Streamflow MIP (`mips/camels_us_daily/`)

This is the first benchmark of HydrologyMIP (<https://hydrologymip.org>). Models predict daily
streamflow for 531 river basins in the United States from the CAMELS-US data set. They learn from
a nine-year training period and predict the ten years before it. There are 12 leaderboards: three
spatial regimes (how basins are held out) times four forcing regimes (which inputs a model uses).
Each leaderboard ranks models by the median Nash-Sutcliffe efficiency (NSE) over the 531 basins.
HydrologyMIP scores every submitted hydrograph with the 14 metrics of the Gauch et al. (2023)
suite.

The machine-readable rules are in [`protocol.json`](protocol.json). If this page and
`protocol.json` disagree, `protocol.json` is right.

## Overview

| Item | Value |
| :--- | :--- |
| Basins | 531 CAMELS-US basins ([`splits/basins_531.txt`](splits/basins_531.txt)), the Newman et al. (2017) subset of the 671 CAMELS-US basins |
| Target | Daily streamflow `qobs` in mm/day (discharge divided by basin area) |
| Training period | 1999-10-01 to 2008-09-30 (water years 2000 to 2008, 3,288 days) |
| Evaluation period | 1989-10-01 to 1999-09-30 (water years 1990 to 1999, 3,652 days) |
| Spin-up | 365 days of inputs before each period (`spinup_days` in `protocol.json`) |
| Spatial regimes | `gauged`, `pub_12fold`, `pur_18huc02` |
| Forcing regimes | `single_camels`, `multi_camels`, `lagged_streamflow`, `freeform` |
| Leaderboards | 12 (Tables 3 to 14) |
| Metrics | 14 metrics: `NSE`, `MSE`, `RMSE`, `KGE`, `logNSE`, `logKGE`, `Alpha-NSE`, `Pearson-r`, `Beta-KGE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`, `Peak-Timing` (suite `gauch_2023_deterministic`) |
| Ranking metric | Median NSE over the 531 basins (higher is better) |
| Summary columns | 16 per leaderboard: the median of each of the 14 metrics, plus the mean NSE and the mean KGE |
| Status | `active`, version 1.0.0 |

A water year runs from 1 October to 30 September. Water year 1990 starts on 1989-10-01.

## Folder contents

| Path | What it holds |
| :--- | :--- |
| [`protocol.json`](protocol.json) | The machine-readable protocol: periods, regimes, leaderboards, metric suite, and storage. |
| [`adapter.py`](adapter.py), [`__init__.py`](__init__.py) | Loads the basin list, the basin-to-region map, and the folds, and checks them for leakage. |
| [`splits/`](splits/README.md) | The basin lists and the cross-validation folds. |
| [`data/`](data/README.md) | How to download and build the input data. The data files are not in git. |
| [`leaderboards/`](leaderboards/README.md) | The 12 leaderboards, the model families, and the Stage-1 checks. |
| [`figures/`](figures/README.md) | Figures drawn from the leaderboards. |
| [`era_experiments/`](era_experiments/README.md) | The ERA notebooks that produced the ERA models in `models/era_*`. |

## Data sources

| Data | Source | DOI |
| :--- | :--- | :--- |
| CAMELS-US v1.2 forcings and USGS streamflow (Newman et al., 2015) | NCAR GDEX, <https://gdex.ucar.edu/dataset/camels.html> | [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D) |
| CAMELS-US catchment attributes v2.0 (Addor et al., 2017) | NCAR GDEX, same page | [10.5065/D6G73C3Q](https://doi.org/10.5065/D6G73C3Q) |
| Extended Maurer forcings (with daily minimum and maximum temperature) | HydroShare | [10.4211/hs.17c896843cf940339c3c3496d0c1c077](https://doi.org/10.4211/hs.17c896843cf940339c3c3496d0c1c077) |
| Extended NLDAS forcings | HydroShare | [10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c](https://doi.org/10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c) |
| HydrologyMIP v1.0 release (code, splits, and leaderboards) | Zenodo | [10.5281/zenodo.24383448](https://doi.org/10.5281/zenodo.24383448) |
| Ongoing public Zarr v2 hydrograph & evaluation-target mirror | Cloudflare R2 | <https://data.hydrologymip.org/camels_us_daily/> |

Download and build steps are in [`data/README.md`](data/README.md). The licenses of the data are
in [`DATA_LICENSE.md`](../../DATA_LICENSE.md).

## Protocol rules

These rules make the comparison fair. A pull request that breaks one of them is rejected.

1. **Use all 531 basins.** Do not add, drop, or replace basins.
2. **Train only on the training period.** Fit models, tune hyperparameters, choose architectures,
   and stop training using 1999-10-01 to 2008-09-30 only.
3. **Predict the evaluation period.** Predict every basin on every day from 1989-10-01 to
   1999-09-30. The evaluation period comes *before* the training period.
4. **Never use evaluation streamflow.** Observed streamflow from the evaluation period must not
   enter training, tuning, early stopping, model selection, or normalization statistics.
5. **Spin-up uses inputs only.** Before each period, a model may read forcing data to warm up. The
   standard spin-up is 365 days. A model must not use observed streamflow from the spin-up days;
   the data loader hides it (sets it to missing). This matters for the training period: its
   spin-up year, 1998-10-01 to 1999-09-30, lies inside the evaluation period.
6. **Keep held-out basins out.** In the `pub_12fold` and `pur_18huc02` regimes, the test basins of
   a fold must not affect anything in that fold: model weights, normalization statistics,
   hyperparameters, or the choice of donor basins.
7. **Use only the inputs of your forcing regime.** Past observed streamflow is an allowed input
   only on the `lagged_streamflow` leaderboards.
8. **Submit complete predictions.** A submission has one value for each basin and each day:
   shape (531, 3652), in mm/day. Every basin has valid observations on all 3,652 evaluation days,
   so no basin is left out of scoring, and any missing prediction makes the submission fail.

### Spatial regimes

| Regime ID | Leaderboard prefix | Folds | How basins are held out |
| :--- | :--- | :--- | :--- |
| `gauged` | `gauged` | 1 | No basin is held out. Train on all 531 basins (training period) and predict all 531 basins (evaluation period). |
| `pub_12fold` | `pub` | 12 | Prediction in ungauged basins. The basins are split at random into 12 folds of 44 or 45 basins. Each fold is predicted by a model that never saw those basins. |
| `pur_18huc02` | `pur` | 18 | Prediction in ungauged regions. Each fold holds out one of the 18 USGS two-digit hydrologic regions (HUC-02). |

In `pub_12fold` and `pur_18huc02`, the test predictions of all folds are joined into one
531-basin hydrograph. That hydrograph is scored.

### Forcing regimes

| Regime ID | Inputs allowed |
| :--- | :--- |
| `single_camels` | One CAMELS-US forcing product: Daymet, Maurer, or NLDAS. |
| `multi_camels` | Two or three CAMELS-US forcing products. |
| `lagged_streamflow` | CAMELS-US forcings plus past observed streamflow. |
| `freeform` | Other inputs: forcings from outside CAMELS-US (for example ERA5-Land), extra attributes, or pretraining on other data. |

### Regime IDs and folder names

The PUR regime has two names. Its ID is `pur_18huc02` in `protocol.json`, on the command line
(`--spatial-regime pur_18huc02`), and in hydrograph stores. Its split file and folder use the
older name `pur_18fold`. Both names mean the same 18 folds.

| Regime ID | Split file | Plain-text fold lists |
| :--- | :--- | :--- |
| `gauged` | none (all 531 basins) | none |
| `pub_12fold` | [`splits/pub_12fold_random_splits.json`](splits/pub_12fold_random_splits.json) | [`splits/pub_12fold/`](splits/pub_12fold/) (`fold_00` to `fold_11`) |
| `pur_18huc02` | [`splits/pur_18fold_huc02_splits.json`](splits/pur_18fold_huc02_splits.json) | [`splits/pur_18fold/`](splits/pur_18fold/) (`huc_01` to `huc_18`) |

### Leaderboards

A leaderboard ID is `<leaderboard prefix>__<forcing regime>`.

| Prefix | `single_camels` | `multi_camels` | `lagged_streamflow` | `freeform` |
| :--- | :--- | :--- | :--- | :--- |
| `gauged` | Table 3 | Table 4 | Table 5 | Table 6 |
| `pub` | Table 7 | Table 8 | Table 9 | Table 10 |
| `pur` | Table 11 | Table 12 | Table 13 | Table 14 |

For example, Table 8 is `pub__multi_camels`. The leaderboard files and their fields are described
in [`leaderboards/README.md`](leaderboards/README.md).

## How to take part

1. **Get the data:** [`data/README.md`](data/README.md).
2. **Run a model:** [`models/README.md`](../../models/README.md). `mip-run-model` writes a local
   store to `./outputs/<leaderboard_id>/<model_id>.zarr` by default.
3. **Score it on your own computer:** [`metrics/README.md`](../../metrics/README.md). You can also
   check a file in your browser on the Score page of <https://hydrologymip.org>.
4. **Submit:** open a pull request on GitHub, or use an issue form in
   [`.github/ISSUE_TEMPLATE/`](../../.github/ISSUE_TEMPLATE/). Maintainers score every submission
   and update the leaderboards. You never need access to Google Cloud.

The full guide for contributors is [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

To load the checked basin list and folds in Python:

```python
from mips.camels_us_daily import load_canonical_basins, load_splits_for_regime

basins = load_canonical_basins()               # 531 sorted USGS gauge IDs
folds = load_splits_for_regime("pur_18huc02")  # 18 checked folds
print(len(basins), len(folds), folds[0].fold_id)  # 531 18 huc_01
```

| Function | What it does |
| :--- | :--- |
| `load_protocol(repo_root=None)` | Returns the `MipProtocol` of this MIP. |
| `load_canonical_basins(splits_dir=None)` | Returns the 531 sorted gauge IDs. Checks the count, the order, and the 8-digit format. |
| `load_basin_to_huc02(splits_dir=None)` | Returns the map from gauge ID to its two-digit HUC-02 region. |
| `load_splits_for_regime(spatial_regime, splits_dir=None, repo_root=None)` | Returns the folds of one spatial regime as a list of `CvFold`, after strict leakage checks. In `pur_18huc02`, every fold must hold out exactly one HUC-02 region. |

## Citation

Please cite the benchmark paper:

> Nearing, G. S., Klotz, D., Ebel, P., & Gauch, M. (2026). HydrologyMIP: A Rainfall-Runoff Model
> Intercomparison Project. Hydrology and Earth System Sciences (submitted).

Please also cite the data and the metrics:

- Newman, A. J., et al. (2015). *Hydrology and Earth System Sciences*.
  <https://doi.org/10.5194/hess-19-209-2015>
- Addor, N., et al. (2017). *Hydrology and Earth System Sciences*.
  <https://doi.org/10.5194/hess-21-5293-2017>
- Newman, A. J., et al. (2017). *Journal of Hydrometeorology*.
  <https://doi.org/10.1175/JHM-D-16-0284.1>
- Gauch, M., et al. (2023). *Water Resources Research*. <https://doi.org/10.1029/2022WR033918>
