# ERA-Multi-Paradigm-Gauged (10-model ensemble) (`era_gauged_multi_paradigm`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `gauged__multi_camels` (`gauged`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23641695`](https://doi.org/10.5281/zenodo.23641695)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_gauged_newman_cv.ipynb`](../../mips/camels_us_daily/era_experiments/era_gauged_newman_cv.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_gauged_newman_cv_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_gauged_newman_cv_winner.ipynb) (Winning Tree-Search Node: **Node 2**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
Four-architecture multi-paradigm ensemble per random seed (40 sub-networks total across 10 independent seeds) blending two complementary conditioning paradigms (`family_weights = 0.38, 0.12, 0.36, 0.14` fit strictly on out-of-fold inner validation predictions inside Water Years 2000–2008):
1. **Paradigm A (`CausalConvGRULSTM` + `MultiScaleFiLMEALSTM`, 60 engineered dynamic features, 27 static attributes + 32D learned basin-ID embeddings):** Trained with trajectory Huber-NSE loss plus `0.12 * MSE(log1p(clamp(Q_hat, 0)), log1p(clamp(Q_obs, 0)))` and AdamW (`weight_decay = 0.05`).
2. **Paradigm B (`DualPathwayFiLMGRULSTM` + `EntityAwareHydroLSTM`, 36 dynamic features including continuous 3-forcing degree-day snowpack states, 27 static attributes + 32D learned basin-ID embeddings for `DualPathwayFiLMGRULSTM`):** Trained with basin-variance-normalized NSE loss plus `0.10 * MSE(log1p(clamp(Q_hat, 0)), log1p(clamp(Q_obs, 0)))` and AdamW (`weight_decay = 0.10`).
3. **Inference Rollout:** Equal `0.5 / 0.5` blend of a continuous 10-year rollout and overlapping 730-day sliding-window rollouts (`stride = 182` days, `365`-day warm-up discard), clipped at `0.0 mm/day`.

## Benchmark Performance (531 Newman Basins, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_gauged_multi_paradigm`) | Single-Seed Realization (`era_gauged_multi_paradigm_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.827`** | `0.822 ± 0.002` |
| **Mean NSE** | `0.776` | `0.771` |
| **Median KGE** | `0.806` | `0.805` |
| **Basins with `NSE <= 0`** | `2 / 531` | `2 / 531` |
