# CAMELS-US MIP — Empirical Research Assistance (ERA) Experiments

This directory contains the **7 starting notebooks** and **7 final winning notebooks** from the Empirical Research Assistance (ERA) hydrological model discovery experiments in **Nearing (2026)**, strictly aligned with the canonical **Newman et al. (2015)** temporal split.

## Strict Non-Leakage Newman Temporal Protocol

All 7 ERA tree-search experiments operate strictly inside the Newman et al. (2015) training decade (`1998-10-01` to `2008-09-30`, `3,653` days across all `531` canonical CAMELS-US basins) using the Kaggle dataset `greynearing/camels-us-531-era`, with **zero access** to the blind Newman benchmark evaluation decade (`1989-10-01` to `1999-09-30`, `3,652` days):

- **ERA Model Weights Training Window (`2003-10-01` to `2008-09-30`, `1,827` days)**: Preceded by a 365-day meteorological spin-up (`2002-10-01` to `2003-09-30`).
- **ERA Inner Validation Window (`1999-10-01` to `2003-09-30`, `1,461` days)**: Preceded by a 365-day meteorological spin-up (`1998-10-01` to `1999-09-30`). Candidate architectures are ranked by 531-basin Median NSE on this inner validation window.
- **Blind Newman Benchmark Evaluation (`1989-10-01` to `1999-09-30`, `3,652` days)**: Once each ERA experiment completes, its winning architecture is extracted into `models/era_*/` and trained from scratch with `n = 10` independent random seeds on the full Newman training window (`1999-10-01` to `2008-09-30`, with 365-day spin-up `1998-10-01` to `1999-09-30`) and evaluated out-of-sample on `1989-10-01` to `1999-09-30`.

## Summary of the 7 ERA Experiments & Extracted Models

| # | Spatial Regime | Guidance Protocol | Starting Notebook | Final Winner Notebook | Extracted Model Package | 10-Seed Ensemble Med NSE | Single-Seed Med NSE | Zenodo DOI |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | `gauged` | `ERA-Multi-Paradigm-Gauged` | [`era_gauged_newman_cv.ipynb`](./era_gauged_newman_cv.ipynb) | [`era_gauged_newman_cv_winner.ipynb`](./era_gauged_newman_cv_winner.ipynb) | [`models/era_gauged_multi_paradigm`](../models/era_gauged_multi_paradigm) | **`0.827`** | `0.822 ± 0.002` | [`10.5281/zenodo.23641695`](https://doi.org/10.5281/zenodo.23641695) |
| **2** | `gauged` | `ERA-Physics-Guided-Gauged` | [`era_gauged_physics_informed.ipynb`](./era_gauged_physics_informed.ipynb) | [`era_gauged_physics_informed_winner.ipynb`](./era_gauged_physics_informed_winner.ipynb) | [`models/era_gauged_physics_guided`](../models/era_gauged_physics_guided) | **`0.820`** | `0.800 ± 0.003` | [`10.5281/zenodo.23639367`](https://doi.org/10.5281/zenodo.23639367) |
| **3** | `gauged` | `ERA-Unguided-Gauged` | [`era_gauged_minimal_guidance.ipynb`](./era_gauged_minimal_guidance.ipynb) | [`era_gauged_minimal_guidance_winner.ipynb`](./era_gauged_minimal_guidance_winner.ipynb) | [`models/era_gauged_minimal_guidance`](../models/era_gauged_minimal_guidance) | **`0.819`** | `0.810 ± 0.002` | [`10.5281/zenodo.23639347`](https://doi.org/10.5281/zenodo.23639347) |
| **4** | `pub_12fold` | `ERA-Physics-Guided-PUB` | [`era_pub_physics_informed.ipynb`](./era_pub_physics_informed.ipynb) | [`era_pub_physics_informed_winner.ipynb`](./era_pub_physics_informed_winner.ipynb) | [`models/era_pub_physics_guided`](../models/era_pub_physics_guided) | **`0.725`** | `0.714 ± 0.004` | [`10.5281/zenodo.23639402`](https://doi.org/10.5281/zenodo.23639402) |
| **5** | `pub_12fold` | `ERA-Unguided-PUB` | [`era_pub_minimal_guidance.ipynb`](./era_pub_minimal_guidance.ipynb) | [`era_pub_minimal_guidance_winner.ipynb`](./era_pub_minimal_guidance_winner.ipynb) | [`models/era_pub_minimal_guidance`](../models/era_pub_minimal_guidance) | **`0.720`** | `0.714 ± 0.004` | [`10.5281/zenodo.23639387`](https://doi.org/10.5281/zenodo.23639387) |
| **6** | `pur_18huc02` | `ERA-Unguided-PUR` | [`era_pur_minimal_guidance.ipynb`](./era_pur_minimal_guidance.ipynb) | [`era_pur_minimal_guidance_winner.ipynb`](./era_pur_minimal_guidance_winner.ipynb) | [`models/era_pur_minimal_guidance`](../models/era_pur_minimal_guidance) | **`0.696`** | `0.686 ± 0.004` | [`10.5281/zenodo.23639429`](https://doi.org/10.5281/zenodo.23639429) |
| **7** | `pur_18huc02` | `ERA-Physics-Guided-PUR` | [`era_pur_physics_informed.ipynb`](./era_pur_physics_informed.ipynb) | [`era_pur_physics_informed_winner.ipynb`](./era_pur_physics_informed_winner.ipynb) | [`models/era_pur_physics_guided`](../models/era_pur_physics_guided) | **`0.684`** | `0.653 ± 0.007` | [`10.5281/zenodo.23639461`](https://doi.org/10.5281/zenodo.23639461) |

## Reproducing Any Extracted ERA Model

All 7 winning ERA models are extracted as standalone, testable PyTorch packages under `models/era_*/` and registered in `models/run_model.py`:

```bash
# List all registered models (including all 7 era_* models)
python -m models.run_model --list-models

# Run full 10-seed Newman evaluation for any ERA model
python -m models.run_model --model era_gauged_multi_paradigm --spatial-regime gauged --forcings daymet,maurer,nldas
```
