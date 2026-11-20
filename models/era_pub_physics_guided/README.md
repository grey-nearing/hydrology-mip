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
- **Backbone:** 1-layer LSTM (`hidden_size = 160`) with static hidden/cell state initialization (`h0`, `c0`), LayerNorm+SiLU static encoder (27 CAMELS attributes with log-transformed skewed attributes + coordinates/log area), and Softplus discharge readout head.
- **Loss & Protocol:** Basin-std-scaled MSE (`mean(((y_hat - y_true) / (sigma_b + 0.1))^2)`) plus `0.20 * MSE(log1p(clamp(y_hat, 0)), log1p(clamp(y_true, 0)))` with AdamW (`weight_decay = 0.05`). Architecture selected via 256-iteration tree search holding out canonical Fold 0 strictly inside `1999-10-01` to `2008-09-30`, then retrained from scratch across all 12 PUB folds with zero target-basin discharge access.

## Benchmark Performance (531 Newman Basins, 12-Fold PUB, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_pub_physics_guided`) | Single-Seed Realization (`era_pub_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.725`** | `0.714 ± 0.004` |
| **Mean NSE** | `0.645` | `0.625` |
| **Median KGE** | `0.664` | `0.662` |
| **Basins with `NSE <= 0`** | `10 / 531` | `11 / 531` |
