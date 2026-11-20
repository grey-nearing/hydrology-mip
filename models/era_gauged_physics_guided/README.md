# ERA-Physics-Guided-Gauged (10-model ensemble) (`era_gauged_physics_guided`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `gauged__multi_camels` (`gauged`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639367`](https://doi.org/10.5281/zenodo.23639367)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_gauged_physics_informed.ipynb`](../../mips/camels_us_daily/era_experiments/era_gauged_physics_informed.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_gauged_physics_informed_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_gauged_physics_informed_winner.ipynb) (Winning Tree-Search Node: **Node 187**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
`PhysicsHybridEALSTM` (`DifferentiableSnowPack` + Entity-Aware LSTM, 1 network per seed, 10 networks across 10 seeds):
- **Differentiable Snowpack Module (`DifferentiableSnowPack`):** TorchScript causal scan computing basin-conditioned degree-day melt factor (`mf`) and rain/snow temperature threshold (`tc`) to output continuous snow water equivalent and liquid water delivery.
- **Sequence Backbone:** 256-unit Entity-Aware LSTM over 30 dynamic meteorological/rolling features plus snowpack liquid-water delivery, conditioned on 27 CAMELS static attributes plus learned per-basin ID embeddings.
- **Training & Inference:** AdamW with OneCycleLR on `365`-day spin-up + Newman training window (`1999-10-01` to `2008-09-30`); continuous evaluation rollout clipped at `0.0 mm/day`.
- **Train vs. inference protocol:** training samples random `365`-day windows (`seq_len = 365`), but inference runs **one stateful pass** over the 365-day spin-up plus the full 3,652-day evaluation decade (about 4,017 days), discarding the first 365 predicted days. Predictions are therefore not windowed sequence-to-one outputs.

## Benchmark Performance (531 Newman Basins, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_gauged_physics_guided`) | Single-Seed Realization (`era_gauged_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.820`** | `0.800 ± 0.003` |
| **Mean NSE** | `0.778` | `0.755` |
| **Median KGE** | `0.821` | `0.818` |
| **Basins with `NSE <= 0`** | `1 / 531` | `2 / 531` |
