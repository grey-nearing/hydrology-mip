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
- **Backbone & Loss:** 1-layer 128-unit FiLM-input-gated LSTM (`x * (1 + gamma) + beta`, static-initialized `h0`, `c0`), trained for 40 epochs (`batch_size = 512`, `365`-day sequences, 400 sampled sequences per basin per seed, AdamW `lr = 1e-3`, `weight_decay = 0.05`, OneCycleLR) minimizing basin-variance-scaled MSE (`scaled_mse_loss`). Architecture selected via 256-iteration tree search holding out HUC-02 Region `01` strictly inside `1999-10-01` to `2008-09-30`, then retrained across all 18 PUR folds.

## Benchmark Performance (531 Newman Basins, 18-Fold PUR, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_pur_physics_guided`) | Single-Seed Realization (`era_pur_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.684`** | `0.653 ± 0.007` |
| **Mean NSE** | `0.524` | `0.453` |
| **Median KGE** | `0.657` | `0.636` |
| **Basins with `NSE <= 0`** | `28 / 531` | `33 / 531` |
