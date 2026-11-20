# ERA-Unguided-Gauged (10-model ensemble) (`era_gauged_minimal_guidance`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `gauged__multi_camels` (`gauged`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639347`](https://doi.org/10.5281/zenodo.23639347)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_gauged_minimal_guidance.ipynb`](../../mips/camels_us_daily/era_experiments/era_gauged_minimal_guidance.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_gauged_minimal_guidance_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_gauged_minimal_guidance_winner.ipynb) (Winning Tree-Search Node: **Node 73**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
`EntityAwareLSTM` (1 network per seed, 10 networks across 10 seeds):
- **Backbone:** 2-layer LSTM (`hidden_size = 320`, `dropout = 0.4`, orthogonal weight initialization, forget-gate bias `4.0`) with a 3-layer LayerNorm+SiLU static context encoder over the 27 CAMELS static attributes (no FiLM modulation and no learned basin-ID embeddings), projecting static context into initial `(h0, c0)` states and concatenating static context at every timestep.
- **Inputs & Loss:** 30 dynamic multi-forcing and rolling features; trained for 70 epochs (`batch_size = 128`, AdamW `lr = 1.2e-3`, `weight_decay = 0.1`, OneCycleLR) minimizing basin-variance-normalized NSE loss plus `0.10 * MSE(log1p(clamp(Q_hat, 0)), log1p(clamp(Q_obs, 0)))`.

## Benchmark Performance (531 Newman Basins, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_gauged_minimal_guidance`) | Single-Seed Realization (`era_gauged_minimal_guidance_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.819`** | `0.810 ± 0.002` |
| **Mean NSE** | `0.769` | `0.755` |
| **Median KGE** | `0.802` | `0.798` |
| **Basins with `NSE <= 0`** | `3 / 531` | `3 / 531` |
