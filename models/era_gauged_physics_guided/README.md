# ERA-Physics-Guided-Gauged (10-model ensemble) (`era_gauged_physics_guided`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `gauged__multi_camels` (`gauged`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639367`](https://doi.org/10.5281/zenodo.23639367)
- **ERA Starting Notebook**: [`era_experiments/era_gauged_physics_informed.ipynb`](../../era_experiments/era_gauged_physics_informed.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_gauged_physics_informed_winner.ipynb`](../../era_experiments/era_gauged_physics_informed_winner.ipynb) (Winning Node `187`, `1999–2003` Validation Median NSE = `0.8204`)

## Architecture Summary
Physics-Guided Hydrological Ensemble combining a 2-layer Entity-Aware LSTM and a 2-layer Static-Conditioned GRU trained with explicit snowpack/melt degree-day state variables, multi-timescale antecedent precipitation indices, Hamon PET, and dual-objective NSE + log-discharge loss.

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_gauged_physics_guided`) | Single Model (`era_gauged_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.820`** (`0.8203`) | **`0.800 ± 0.003`** |
| **Mean NSE** | `0.778` | `0.755` |
| **Median KGE** | `0.821` | `0.818` |
| **Median `|FHV|` (%)** | `11.48%` | `11.14%` |
| **Basins with `NSE <= 0`** | `1 / 531` | `2 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_gauged_physics_guided \
  --spatial-regime gauged \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_gauged_physics_guided_gauged__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_gauged_physics_guided.reproduce_author_metrics --mode smoke
```
