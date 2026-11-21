# CAMELS-US MIP — Empirical Research Assistance (ERA) Experiments

This directory contains the **7 starting notebooks** and **7 final winning notebooks** from the Empirical Research Assistance (ERA) hydrological model discovery experiments in **Nearing et al. (2026)**, strictly aligned with the **Newman et al. (2015)** temporal split.

## Newman Temporal Protocol and Non-Leakage

All 7 ERA tree-search experiments read data only from `1998-10-01` to `2008-09-30` (`3,653` days across all `531` CAMELS-US basins) using the Kaggle dataset `greynearing/camels-us-531-era`. The Newman et al. (2015) training decade is `1999-10-01` to `2008-09-30`; the preceding water year WY1999 (`1998-10-01` to `1999-09-30`) supplies **forcing-only** meteorological spin-up and overlaps the final year of the blind Newman benchmark evaluation decade (`1989-10-01` to `1999-09-30`, `3,652` days). **No streamflow from `1989-10-01` to `1999-09-30` is used** in any search, selection or final training step:

- **ERA Model Weights Training Window (`2003-10-01` to `2008-09-30`, `1,827` days)**: Preceded by a 365-day meteorological spin-up (`2002-10-01` to `2003-09-30`).
- **ERA Inner Validation Window (`1999-10-01` to `2003-09-30`, `1,461` days)**: Preceded by a 365-day meteorological spin-up (`1998-10-01` to `1999-09-30`, forcings only). Candidate architectures are ranked by 531-basin Median NSE on this inner validation window.
- **Blind Newman Benchmark Evaluation (`1989-10-01` to `1999-09-30`, `3,652` days)**: Once each ERA experiment completes, its winning architecture is extracted into `models/era_*/` and trained from scratch with `n = 10` independent random seeds on the full Newman training window (`1999-10-01` to `2008-09-30`, with 365-day forcing-only spin-up `1998-10-01` to `1999-09-30`) and evaluated out-of-sample on `1989-10-01` to `1999-09-30`.
- **Spatial selection caveat (PUB/PUR)**: In the four PUB/PUR experiments, every candidate architecture was trained on all folds (`pub_12fold_random_splits.json` / `pur_18fold_huc02_splits.json`) and ranked on the median out-of-fold NSE of the **test** basins in WY2000–2003 (`1999-10-01` to `2003-09-30`). These basins are therefore ungauged with respect to weight training, but not with respect to architecture selection: their WY2000–2003 streamflow informed which architecture was chosen. The selection decade does not overlap the benchmark evaluation decade.

## Summary of the 7 ERA Experiments & Extracted Models

| # | Spatial Regime | Guidance Protocol | Starting Notebook | Final Winner Notebook | Extracted Model Package | 10-Seed Ensemble Med NSE | Single-Seed Med NSE | Zenodo DOI |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | `gauged` | `ERA-Multi-Paradigm-Gauged` | [`era_gauged_newman_cv.ipynb`](./era_gauged_newman_cv.ipynb) | [`era_gauged_newman_cv_winner.ipynb`](./era_gauged_newman_cv_winner.ipynb) | [`models/era_gauged_multi_paradigm`](../../../models/era_gauged_multi_paradigm) | **`0.827`** | `0.822 ± 0.002` | [`10.5281/zenodo.23641695`](https://doi.org/10.5281/zenodo.23641695) |
| **2** | `gauged` | `ERA-Physics-Guided-Gauged` | [`era_gauged_physics_informed.ipynb`](./era_gauged_physics_informed.ipynb) | [`era_gauged_physics_informed_winner.ipynb`](./era_gauged_physics_informed_winner.ipynb) | [`models/era_gauged_physics_guided`](../../../models/era_gauged_physics_guided) | **`0.820`** | `0.800 ± 0.003` | [`10.5281/zenodo.23639367`](https://doi.org/10.5281/zenodo.23639367) |
| **3** | `gauged` | `ERA-Unguided-Gauged` | [`era_gauged_minimal_guidance.ipynb`](./era_gauged_minimal_guidance.ipynb) | [`era_gauged_minimal_guidance_winner.ipynb`](./era_gauged_minimal_guidance_winner.ipynb) | [`models/era_gauged_minimal_guidance`](../../../models/era_gauged_minimal_guidance) | **`0.819`** | `0.810 ± 0.002` | [`10.5281/zenodo.23639347`](https://doi.org/10.5281/zenodo.23639347) |
| **4** | `pub_12fold` | `ERA-Physics-Guided-PUB` | [`era_pub_physics_informed.ipynb`](./era_pub_physics_informed.ipynb) | [`era_pub_physics_informed_winner.ipynb`](./era_pub_physics_informed_winner.ipynb) | [`models/era_pub_physics_guided`](../../../models/era_pub_physics_guided) | **`0.725`** | `0.714 ± 0.004` | [`10.5281/zenodo.23639402`](https://doi.org/10.5281/zenodo.23639402) |
| **5** | `pub_12fold` | `ERA-Unguided-PUB` | [`era_pub_minimal_guidance.ipynb`](./era_pub_minimal_guidance.ipynb) | [`era_pub_minimal_guidance_winner.ipynb`](./era_pub_minimal_guidance_winner.ipynb) | [`models/era_pub_minimal_guidance`](../../../models/era_pub_minimal_guidance) | **`0.720`** | `0.714 ± 0.004` | [`10.5281/zenodo.23639387`](https://doi.org/10.5281/zenodo.23639387) |
| **6** | `pur_18huc02` | `ERA-Unguided-PUR` | [`era_pur_minimal_guidance.ipynb`](./era_pur_minimal_guidance.ipynb) | [`era_pur_minimal_guidance_winner.ipynb`](./era_pur_minimal_guidance_winner.ipynb) | [`models/era_pur_minimal_guidance`](../../../models/era_pur_minimal_guidance) | **`0.696`** | `0.686 ± 0.004` | [`10.5281/zenodo.23639429`](https://doi.org/10.5281/zenodo.23639429) |
| **7** | `pur_18huc02` | `ERA-Physics-Guided-PUR` | [`era_pur_physics_informed.ipynb`](./era_pur_physics_informed.ipynb) | [`era_pur_physics_informed_winner.ipynb`](./era_pur_physics_informed_winner.ipynb) | [`models/era_pur_physics_guided`](../../../models/era_pur_physics_guided) | **`0.684`** | `0.653 ± 0.007` | [`10.5281/zenodo.23639461`](https://doi.org/10.5281/zenodo.23639461) |

## Reproducing Any Extracted ERA Model

All 7 winning ERA models are extracted as standalone, testable PyTorch packages under `models/era_*/` and registered in `models/run_model.py`. Run these commands from the repository root:

```bash
# List all registered models (including all 7 era_* models)
python -m models.run_model --list-models

# Run full 10-seed Newman evaluation for any ERA model
python -m models.run_model --model era_gauged_multi_paradigm --spatial-regime gauged --forcings daymet maurer nldas
```

## Re-running the Notebooks

- **Working directory:** run the notebooks with this directory (`mips/camels_us_daily/era_experiments/`) as the current working directory, so that `../splits` resolves to `mips/camels_us_daily/splits/` (`basins_531.json`, `pub_12fold_random_splits.json`, `pur_18fold_huc02_splits.json`).
- **Data files:** the notebooks require the Kaggle data files `camels_era_train.npz` and `era_eval_targets.npz` from the Kaggle dataset `greynearing/camels-us-531-era`. They are not included in this repository; place them in `/kaggle/input/camels-us-531-era/` (the Kaggle default) or in one of the other directories listed in `_SEARCH_DIRS` (cell 2).
- **Hardware:** a CUDA GPU is required (the searches ran on Kaggle 2 × NVIDIA T4); several winner notebooks hard-code `cuda` devices and CUDA AMP and will not run on CPU.
