# ERA-Unguided-Gauged (10-model ensemble) (`era_gauged_minimal_guidance`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `gauged__multi_camels` (`gauged`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639347`](https://doi.org/10.5281/zenodo.23639347)
- **ERA Starting Notebook**: [`era_experiments/era_gauged_minimal_guidance.ipynb`](../../era_experiments/era_gauged_minimal_guidance.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_gauged_minimal_guidance_winner.ipynb`](../../era_experiments/era_gauged_minimal_guidance_winner.ipynb) (Winning Node `73`, `1999–2003` Validation Median NSE = `0.8241`)

## Architecture Summary
Multi-Scale Static-Conditioned GRU-LSTM Hybrid (`HydroMultiScaleModel`) combining a 2-layer GRU and 2-layer LSTM (`hidden_dim=256`) with static attribute conditioning and 30 multi-timescale rolling meteorological features.

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_gauged_minimal_guidance`) | Single Model (`era_gauged_minimal_guidance_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.819`** (`0.8192`) | **`0.810 ± 0.002`** |
| **Mean NSE** | `0.769` | `0.755` |
| **Median KGE** | `0.802` | `0.798` |
| **Median `|FHV|` (%)** | `13.62%` | `13.26%` |
| **Basins with `NSE <= 0`** | `3 / 531` | `3 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_gauged_minimal_guidance \
  --spatial-regime gauged \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_gauged_minimal_guidance_gauged__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_gauged_minimal_guidance.reproduce_author_metrics --mode smoke
```
