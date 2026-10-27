# ERA-Physics-Guided-PUR (10-model ensemble) (`era_pur_physics_guided`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistant -- LSTM Seeded` (`era_lstm_seeded`)
- **Citation**: `Nearing (2026)`
- **Target Leaderboard**: `pur__multi_camels` (`pur_18huc02`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639461`](https://doi.org/10.5281/zenodo.23639461)
- **ERA Starting Notebook**: [`era_experiments/era_pur_physics_informed.ipynb`](../../era_experiments/era_pur_physics_informed.ipynb)
- **ERA Winner Notebook**: [`era_experiments/era_pur_physics_informed_winner.ipynb`](../../era_experiments/era_pur_physics_informed_winner.ipynb) (Winning Node `206`, `1999–2003` Validation Median NSE = `0.6781`)

## Architecture Summary
Physics-Informed FiLM-Modulated API & PET LSTM (`PhysicsFiLMLSTM`) for 18-Fold HUC-02 PUR incorporating exponential antecedent precipitation index (API) state features, radiation-driven PET proxies, rain/snow partitioning, and bounded static FiLM conditioning (`gamma = 1 + 0.3 * tanh(...)`) with static attribute noise regularization (`std=0.02`).

## Newman Benchmark Performance (`531` Basins, `1989-10-01` to `1999-09-30`, `n = 10` Seeds)

| Metric | 10-Model Ensemble (`era_pur_physics_guided`) | Single Model (`era_pur_physics_guided_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.684`** (`0.6837`) | **`0.653 ± 0.007`** |
| **Mean NSE** | `0.524` | `0.453` |
| **Median KGE** | `0.657` | `0.636` |
| **Median `|FHV|` (%)** | `23.40%` | `22.28%` |
| **Basins with `NSE <= 0`** | `28 / 531` | `33 / 531` |

## Reproducing From Scratch

### 1. Full 10-Seed Newman Benchmark Evaluation (`models/run_model.py`)
```bash
python -m models.run_model \
  --model era_pur_physics_guided \
  --spatial-regime pur_18huc02 \
  --forcings daymet,maurer,nldas \
  --output-zarr /tmp/era_pur_physics_guided_pur__multi_camels.zarr
```

### 2. Hermetic Smoke / Reproduction Test (`reproduce_author_metrics.py`)
```bash
python -m models.era_pur_physics_guided.reproduce_author_metrics --mode smoke
```
