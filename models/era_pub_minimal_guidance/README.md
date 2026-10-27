# ERA-Unguided-PUB (10-model ensemble) (`era_pub_minimal_guidance`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `pub__multi_camels` (`pub_12fold`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639387`](https://doi.org/10.5281/zenodo.23639387)
- **ERA Starting Notebook**: [`era_experiments/era_pub_minimal_guidance.ipynb`](../../era_experiments/era_pub_minimal_guidance.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_pub_minimal_guidance_winner.ipynb`](../../era_experiments/era_pub_minimal_guidance_winner.ipynb) (Winning Node `82`, `1999–2003` Validation Median NSE = `0.7304`)

## Architecture Summary
Enhanced Entity-Aware LSTM (`EnhancedEALSTM`) for 12-Fold PUB with static attribute input-gate conditioning, multi-timescale causal rolling precipitation/temperature aggregates, and OneCycleLR cosine annealing.

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_pub_minimal_guidance`) | Single Model (`era_pub_minimal_guidance_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.720`** (`0.7197`) | **`0.714 ± 0.004`** |
| **Mean NSE** | `0.648` | `0.636` |
| **Median KGE** | `0.644` | `0.641` |
| **Median `|FHV|` (%)** | `27.23%` | `26.94%` |
| **Basins with `NSE <= 0`** | `5 / 531` | `5 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_pub_minimal_guidance \
  --spatial-regime pub_12fold \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_pub_minimal_guidance_pub__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_pub_minimal_guidance.reproduce_author_metrics --mode smoke
```
