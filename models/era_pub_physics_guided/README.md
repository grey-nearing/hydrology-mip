# ERA-Physics-Guided-PUB (10-model ensemble) (`era_pub_physics_guided`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `pub__multi_camels` (`pub_12fold`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639402`](https://doi.org/10.5281/zenodo.23639402)
- **ERA Starting Notebook**: [`era_experiments/era_pub_physics_informed.ipynb`](../../era_experiments/era_pub_physics_informed.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_pub_physics_informed_winner.ipynb`](../../era_experiments/era_pub_physics_informed_winner.ipynb) (Winning Node `203`, `1999–2003` Validation Median NSE = `0.7229`)

## Architecture Summary
Physics-Informed Entity-Aware LSTM (`PhysicsEALSTM`) for 12-Fold Prediction in Ungauged Basins (PUB) featuring cross-forcing precipitation/temperature uncertainty features, radiation-temperature PET proxy, rolling water-balance indices, and static attribute Gaussian jitter regularization (`std=0.05`).

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_pub_physics_guided`) | Single Model (`era_pub_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.725`** (`0.7254`) | **`0.714 ± 0.004`** |
| **Mean NSE** | `0.645` | `0.625` |
| **Median KGE** | `0.663` | `0.662` |
| **Median `|FHV|` (%)** | `24.58%` | `24.11%` |
| **Basins with `NSE <= 0`** | `10 / 531` | `11 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_pub_physics_guided \
  --spatial-regime pub_12fold \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_pub_physics_guided_pub__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_pub_physics_guided.reproduce_author_metrics --mode smoke
```
