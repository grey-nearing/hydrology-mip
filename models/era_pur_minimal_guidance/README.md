# ERA-Unguided-PUR (10-model ensemble) (`era_pur_minimal_guidance`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `pur__multi_camels` (`pur_18huc02`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639429`](https://doi.org/10.5281/zenodo.23639429)
- **ERA Starting Notebook**: [`era_experiments/era_pur_minimal_guidance.ipynb`](../../era_experiments/era_pur_minimal_guidance.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_pur_minimal_guidance_winner.ipynb`](../../era_experiments/era_pur_minimal_guidance_winner.ipynb) (Winning Node `135`, `1999–2003` Validation Median NSE = `0.6981`)

## Architecture Summary
Static-FiLM Dual-Pathway Recurrent Network (`HydroPURNet`) for 18-Fold HUC-02 Prediction in Ungauged Regions (PUR) combining parallel 2-layer LSTM and 2-layer GRU recurrent cores (`hidden_dim=256`) with static catchment attribute FiLM modulation (`1 + 0.25 * tanh(gamma)`), static Gaussian noise + dropout regularization against regional overfitting, and 38 engineered cross-forcing hydrological features.

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_pur_minimal_guidance`) | Single Model (`era_pur_minimal_guidance_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.696`** (`0.6956`) | **`0.686 ± 0.004`** |
| **Mean NSE** | `0.494` | `0.466` |
| **Median KGE** | `0.642` | `0.635` |
| **Median `|FHV|` (%)** | `24.95%` | `24.71%` |
| **Basins with `NSE <= 0`** | `35 / 531` | `35 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_pur_minimal_guidance \
  --spatial-regime pur_18huc02 \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_pur_minimal_guidance_pur__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_pur_minimal_guidance.reproduce_author_metrics --mode smoke
```
