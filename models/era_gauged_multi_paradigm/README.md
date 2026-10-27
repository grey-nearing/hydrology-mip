# ERA-Multi-Paradigm-Gauged (10-model ensemble) (`era_gauged_multi_paradigm`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `gauged__multi_camels` (`gauged`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23641695`](https://doi.org/10.5281/zenodo.23641695)
- **ERA Starting Notebook**: [`era_experiments/era_gauged_newman_cv.ipynb`](../../era_experiments/era_gauged_newman_cv.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_gauged_newman_cv_winner.ipynb`](../../era_experiments/era_gauged_newman_cv_winner.ipynb) (Winning Node `2`, `1999–2003` Validation Median NSE = `0.8417`)

## Architecture Summary
Four-Family Multi-Paradigm Hydrological Ensemble combining two complementary feature-and-conditioning paradigms per seed: (1) Paradigm A (60 multi-scale hydroclimatic, PET, rolling water-balance deficit, and cross-forcing agreement features with basin-conditioned static embeddings, trajectory Huber-NSE + log1p MSE loss, and exponential moving average weight averaging): `CausalConvGRULSTM` (multi-scale causal dilated Conv1D d=1,3,9 + coupled GRU-LSTM) and `MultiScaleFiLMEALSTM` (2-layer FiLM-modulated Entity-Aware LSTM); and (2) Paradigm B (36 clean meteorological features augmented with continuous 10-year 3-forcing degree-day snowpack accumulation/melt and liquid water delivery across Daymet, Maurer, and NLDAS, conditioned strictly on 27 static catchment attributes): `DualPathwayFiLMGRULSTM` (parallel 2-layer LSTM + 2-layer GRU with static FiLM gates) and `EntityAwareHydroLSTM` (2-layer Entity-Aware HydroLSTM), evaluated with hybrid continuous 10-year rollout + overlapping 730-day sliding-window inference.

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_gauged_multi_paradigm`) | Single Model (`era_gauged_multi_paradigm_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.827`** (`0.8265`) | **`0.822 ± 0.002`** |
| **Mean NSE** | `0.776` | `0.771` |
| **Median KGE** | `0.806` | `0.805` |
| **Median `|FHV|` (%)** | `13.28%` | `13.15%` |
| **Basins with `NSE <= 0`** | `2 / 531` | `2 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_gauged_multi_paradigm \
  --spatial-regime gauged \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_gauged_multi_paradigm_gauged__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_gauged_multi_paradigm.reproduce_author_metrics --mode smoke
```
