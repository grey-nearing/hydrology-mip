# ERA-Physics-Guided-PUR (10-model ensemble) (`era_pur_physics_guided`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `pur__multi_camels` (`pur_18huc02`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639461`](https://doi.org/10.5281/zenodo.23639461)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_pur_physics_informed.ipynb`](../../mips/camels_us_daily/era_experiments/era_pur_physics_informed.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_pur_physics_informed_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_pur_physics_informed_winner.ipynb) (Winning Tree-Search Node: **Node 206**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
`PURPhysicsModel` (1 network per seed per fold, 10 seeds across canonical 18-fold leave-one-HUC-02-out PUR):
- **Inputs:** 26 dynamic channels combining three-forcing meteorological variables, Hargreaves potential evapotranspiration (PET), and four exponential Antecedent Precipitation Indices (APIs); static encoder (`128 -> stat_emb`, LayerNorm+ReLU+Dropout, Gaussian noise augmentation `std = 0.02`) over 27 CAMELS attributes + `lat`, `lon`, `log1p(area_km2)`.
- **Backbone & Loss:** 1-layer 128-unit FiLM-input-gated LSTM (`x * (1 + gamma) + beta`, static-initialized `h0`, `c0`), trained for 40 epochs (`batch_size = 512`, `365`-day sequences, a fixed pool of up to 128 sequences per basin drawn once per seed without replacement and reshuffled each epoch, AdamW `lr = 1e-3`, `weight_decay = 0.05`, OneCycleLR) minimizing basin-variance-scaled MSE (`scaled_mse_loss`). Architecture selected via 256-iteration tree search in which every candidate was trained on all 18 canonical leave-one-HUC-02-out PUR folds (weights fit on `2003-10-01`–`2008-09-30`, training-basin discharge only) and ranked by the median NSE of the stitched out-of-fold predictions on `1999-10-01`–`2003-09-30`; the winner was then retrained across all 18 PUR folds on the full Newman training window.

## Packaging Notes & Known Deviations (Code-Verified)
- **Samples per basin.** The packaged runner uses `samples_per_basin = 128` (`runner.py`, `RUNNER_DEFAULTS`); the Node 206 winner notebook (`era_pur_physics_informed_winner.ipynb` cell 5 L13) used `SAMPLES_PER_BASIN = 400`. The published hydrographs were produced with 128.
- **Normalization statistics include the spin-up year.** The dynamic-input mean/std (`runner.py`, `d_mean`/`d_std`, L186–L187) are computed over the full training slice, including the 365-day WY1999 forcing-only spin-up (`1998-10-01`–`1999-09-30`), which overlaps the last year of the evaluation decade. Only forcings are involved (no discharge). This is kept as-is to stay consistent with the published hydrograph stores.
- **Spatial selection caveat.** The PUR test basins are ungauged with respect to weight training, but their WY2000–2003 streamflow was used to rank candidate architectures during the tree search (see [`mips/camels_us_daily/era_experiments/README.md`](../../mips/camels_us_daily/era_experiments/README.md)).
- **Train vs. inference protocol:** training samples `365`-day windows (`seq_len = 365`), but inference runs **one stateful pass** over the 365-day spin-up plus the full 3,652-day evaluation decade (about 4,017 days), discarding the first 365 predicted days. Predictions are therefore not windowed sequence-to-one outputs.

## Benchmark Performance (531 Newman Basins, 18-Fold PUR, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_pur_physics_guided`) | Single-Seed Realization (`era_pur_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.684`** | `0.653 ± 0.007` |
| **Mean NSE** | `0.524` | `0.453` |
| **Median KGE** | `0.657` | `0.636` |
| **Basins with `NSE <= 0`** | `28 / 531` | `33 / 531` |
