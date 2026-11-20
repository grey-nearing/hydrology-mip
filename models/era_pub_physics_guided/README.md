# ERA-Physics-Guided-PUB (10-model ensemble) (`era_pub_physics_guided`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `pub__multi_camels` (`pub_12fold`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639402`](https://doi.org/10.5281/zenodo.23639402)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_pub_physics_informed.ipynb`](../../mips/camels_us_daily/era_experiments/era_pub_physics_informed.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_pub_physics_informed_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_pub_physics_informed_winner.ipynb) (Winning Tree-Search Node: **Node 203**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
`PhysicsEALSTM` (1 network per seed per fold, 10 seeds across canonical 12-fold PUB):
- **Backbone:** 1-layer LSTM (`hidden_size = 160`) over 14 physics-engineered dynamic channels (multi-forcing mean precipitation and its `log1p`, precipitation spread across forcings, temperature mean/range, vapour pressure, Hargreaves PET, liquid/solid precipitation split, melt potential, SWE, daylength, day-of-year `sin`/`cos`), with static hidden/cell state initialization (`h0`, `c0`) from a 2-layer `Linear -> SiLU -> Dropout(0.1) -> Linear` static encoder (`stat_emb_size = 64`, no LayerNorm) over the 27 CAMELS attributes only (no lat/lon/area features are appended), and a Softplus discharge readout head.
- **Loss & Protocol:** Basin-std-scaled MSE (`mean(((y_hat - y_true) / (sigma_b + 0.1))^2)`) plus `0.20 * MSE(log1p(clamp(y_hat, 0)), log1p(clamp(y_true, 0)))` with AdamW (`weight_decay = 0.05`). Architecture selected via 256-iteration tree search in which every candidate was trained on all 12 canonical PUB folds (weights fit on `2003-10-01`–`2008-09-30`, training-basin discharge only) and ranked by the median NSE of the stitched out-of-fold predictions on `1999-10-01`–`2003-09-30`; the winner was then retrained from scratch across all 12 PUB folds with no target-basin discharge used in training.

## Packaging Notes & Known Deviations (Code-Verified)
- **Samples per basin.** The packaged runner draws `samples_per_basin_per_epoch = 64` random 365-day windows per training basin per epoch (`runner.py`, `RUNNER_DEFAULTS`); the Node 203 winner notebook (`era_pub_physics_informed_winner.ipynb` cell 5 L12) used `SAMPLES_PER_BASIN = 180`. The published hydrographs were produced with 64.
- **Static log-transform columns.** `runner.py` (`_prepare_features`) applies `log1p(max(0, x))` to columns 0–4 of the **alphabetically sorted** static attributes, i.e. `area_gages2`, `aridity`, `carbonate_rocks_frac`, `clay_frac`, `elev_mean`. The notebook applies the same transform to columns 0–4 of `STATIC_ATTRIBUTES` in the Kaggle npz's own (unsorted) order, with the comment "Assuming first few are area/elev related". If the Kaggle order differs from alphabetical, the packaged model transforms a different set of statics than Node 203. The packaged choice is kept because the published hydrographs were produced with it.
- **Normalization statistics include the spin-up year.** The dynamic-input mean/std (`runner.py`, `d_mean`/`d_std`, L184–L185) are computed over the full training slice, including the 365-day WY1999 forcing-only spin-up (`1998-10-01`–`1999-09-30`), which overlaps the last year of the evaluation decade. Only forcings are involved (no discharge). This is kept as-is to stay consistent with the published hydrograph stores.
- **Spatial selection caveat.** The PUB test basins are ungauged with respect to weight training, but their WY2000–2003 streamflow was used to rank candidate architectures during the tree search (see [`mips/camels_us_daily/era_experiments/README.md`](../../mips/camels_us_daily/era_experiments/README.md)).
- **Train vs. inference protocol:** training samples random `365`-day windows (`seq_len = 365`), but inference runs **one stateful pass** over the 365-day spin-up plus the full 3,652-day evaluation decade (about 4,017 days) with the EMA weights, discarding the first 365 predicted days. Predictions are therefore not windowed sequence-to-one outputs.

## Benchmark Performance (531 Newman Basins, 12-Fold PUB, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_pub_physics_guided`) | Single-Seed Realization (`era_pub_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.725`** | `0.714 ± 0.004` |
| **Mean NSE** | `0.645` | `0.625` |
| **Median KGE** | `0.664` | `0.662` |
| **Basins with `NSE <= 0`** | `10 / 531` | `11 / 531` |
