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
Four-architecture multi-paradigm ensemble per random seed (40 sub-networks total across 10 independent seeds) blending two complementary conditioning paradigms with `family_weights = 0.38, 0.12, 0.36, 0.14` (see the weight-provenance note below):
1. **Paradigm A (`CausalConvGRULSTM` + `MultiScaleFiLMEALSTM`, 60 engineered dynamic features, 27 static attributes + 32D learned basin-ID embeddings):** Trained with trajectory Huber-NSE loss plus `0.12 * MSE(log1p(clamp(Q_hat, 0)), log1p(clamp(Q_obs, 0)))` and AdamW (`weight_decay = 0.05`).
2. **Paradigm B (`DualPathwayFiLMGRULSTM` + `EntityAwareHydroLSTM`, 36 dynamic features including continuous 3-forcing degree-day snowpack states, 27 static attributes only; neither Paradigm B network has learned basin-ID embeddings):** Trained with basin-variance-normalized NSE loss plus `0.10 * MSE(log1p(clamp(Q_hat, 0)), log1p(clamp(Q_obs, 0)))` and AdamW (`weight_decay = 0.10`).
3. **Inference Rollout:** All networks are trained on random `365`-day windows (`seq_len = 365`). At inference each network's prediction is an equal `0.5 / 0.5` blend of one stateful pass over the 365-day spin-up plus the full 10-year evaluation decade, and overlapping 730-day sliding-window rollouts (`stride = 182` days, `365`-day warm-up discard), clipped at `0.0 mm/day`.

## Packaging Notes & Deviations From the Winner Notebook
- **Family weights.** The Node 2 winner notebook (`era_gauged_newman_cv_winner.ipynb`, cell 5) blends the four families with fixed weights `0.35 / 0.15 / 0.35 / 0.15`. The packaged runner default is `family_weights = (0.38, 0.12, 0.36, 0.14)` (`runner.py`, `RUNNER_DEFAULTS`). These packaged weights were reported to have been fit on inner-validation predictions inside the Newman training decade (`1999-10-01`–`2008-09-30`), but **no weight-fitting script, inputs, or outputs exist in this repository**, so that provenance cannot be verified from the repository alone.
- **Ensemble membership.** The winner notebook trains a 10-member set (3 × conv, 2 × FiLM, 3 × snow-FiLM, 2 × snow-EA); the packaged runner trains one network per family per seed (4 per seed, 40 across 10 seeds).

## Benchmark Performance (531 Newman Basins, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_gauged_multi_paradigm`) | Single-Seed Realization (`era_gauged_multi_paradigm_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.827`** | `0.822 ± 0.002` |
| **Mean NSE** | `0.776` | `0.771` |
| **Median KGE** | `0.806` | `0.805` |
| **Basins with `NSE <= 0`** | `2 / 531` | `2 / 531` |
