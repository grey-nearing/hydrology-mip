# ERA-Unguided-PUR (10-model ensemble) (`era_pur_minimal_guidance`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `pur__multi_camels` (`pur_18huc02`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639429`](https://doi.org/10.5281/zenodo.23639429)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_pur_minimal_guidance.ipynb`](../../mips/camels_us_daily/era_experiments/era_pur_minimal_guidance.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_pur_minimal_guidance_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_pur_minimal_guidance_winner.ipynb) (Winning Tree-Search Node: **Node 135**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
`DualFiLMLSTM` (`StatEncoder` + FiLM-modulated 1-layer LSTM, 1 network per seed per fold, 10 seeds across canonical 18-fold leave-one-HUC-02-out PUR):
- **Inputs:** 23 dynamic channels (15 three-forcing meteorological variables + 6 auxiliary daylength/SWE variables + 2 cyclical day-of-year `sin`/`cos` encodings) and 33 static features (27 CAMELS attributes + `lat`, `lon`, `log1p(area_km2)`, training-window `log1p(mean_prcp)`, `mean_temp`, and thermal aridity index `mean_prcp / (max(mean_temp, -5.0) + 10.0)`), with static scalers fit strictly on the 17 training HUC-02 regions per fold.
- **Backbone & Loss:** 256-unit LSTM (`dropout = 0.3`, static noise augmentation `std = 0.015`, 50 epochs, AdamW `lr = 1e-3`, `weight_decay = 0.05`) predicting standardized `sqrt(Q)` under MSE loss (inverted via `clamp(y_hat, 0)^2` at inference). Architecture selected via 256-iteration tree search holding out HUC-02 Region `01` strictly inside `1999-10-01` to `2008-09-30`, then retrained across all 18 PUR folds.

## Benchmark Performance (531 Newman Basins, 18-Fold PUR, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_pur_minimal_guidance`) | Single-Seed Realization (`era_pur_minimal_guidance_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.696`** | `0.686 ± 0.004` |
| **Mean NSE** | `0.494` | `0.466` |
| **Median KGE** | `0.642` | `0.635` |
| **Basins with `NSE <= 0`** | `35 / 531` | `35 / 531` |
