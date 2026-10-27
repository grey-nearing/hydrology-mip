# Models and Central Runner

The `models/` directory contains 19 rainfall-runoff model implementations (12 `MIP Re-run` implementations and 7 `ERA Contribution` implementations), a central deployment runner
(`run_model.py`), the shared CAMELS-US data loader (`camels_data.py`), and the consolidated Zarr
hydrograph writer (`zarr_writer.py`).

## Running a Model

`models/run_model.py` is the **only** place that loads the canonical 531-basin list and the fold files
from `splits/`, defines the canonical training (`1999-10-01`..`2008-09-30`) and evaluation
(`1989-10-01`..`1999-09-30`) windows, loops over every fold of the selected spatial regime **once per
random seed**, stitches each seed's out-of-fold predictions into a `(531, 3652)` array, and writes one
consolidated Zarr store with `qsim` (seed mean) and `qsim_members` (per-seed hydrographs).

```bash
python -m models.run_model --list-models

# Gauged regime, single forcing (Daymet) -> leaderboard gauged__single_camels (Table 3)
python -m models.run_model --model kratzert_lstm --spatial-regime gauged --forcings daymet \
  --data-dir data --output-zarr ./kratzert_lstm_gauged_single_camels.zarr

# 12 random PUB folds, three forcings -> pub__multi_camels (Table 8)
python -m models.run_model --model feng_dhbv10 --spatial-regime pub_12fold --forcings daymet maurer nldas \
  --data-dir data

# 18 leave-one-HUC02-out PUR folds, explicit seeds
python -m models.run_model --model era_pur_minimal_guidance --spatial-regime pur_18huc02 --forcings daymet \
  --seeds 42 43
```

* `--data-dir` (default `data/`) must point at the canonical bundle `data/camels_671_1980_2014.npz` (or
  the directory containing it), a raw NCAR CAMELS-US v1.2 directory, or a per-basin NetCDF layout; see
  [`data/README.md`](../data/README.md).
* The forcing regime is derived from `--forcings`: one product -> `single_camels`, two or three distinct
  products -> `multi_camels`. Duplicate products are an error. `--forcing-regime lagged_streamflow|freeform`
  is only for submissions that legitimately belong to those leaderboards.
* Without `--output-zarr`, the store is written to
  `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` where `leaderboard_id` is one of the 12
  canonical IDs (`gauged__single_camels`, `gauged__multi_camels`, `pub__single_camels`, ...,
  `pur__freeform`) shared with `metrics/evaluator.py` and `leaderboards/`.
* `--max-workers > 1` runs `(fold, seed)` tasks in threads; results are only reproducible if the runner
  uses seed-local random generators.

## Model Directories

Each model family lives in its own subdirectory `models/<model_name>/`:

| Directory | Model Family | Category |
| :--- | :--- | :--- |
| `kratzert_lstm/` | Kratzert et al. (2019, 2021) CudaLSTM (PUB & PUR MIP Re-run) | MIP Re-run |
| `kratzert_ealstm/` | Kratzert et al. (2019b) Entity-Aware LSTM (EA-LSTM, Stage-1 Confirmation) | MIP Re-run |
| `feng2020_di_lstm/` | Feng et al. (2020) Data-Integration LSTM (`DI(1)-LSTM`) | MIP Re-run |
| `newman_sacsma/` | Nearing et al. (2020) Snow-17 + SAC-SMA (DDS & Donor Transfer) | MIP Re-run |
| `hoedt_mclstm/` | Hoedt et al. (2021) Mass-Conserving LSTM (MC-LSTM) | MIP Re-run |
| `feng_dhbv10/` | Feng et al. (2022, 2023) Differentiable HBV 1.0 (`δHBV 1.0`) | MIP Re-run |
| `yin2022_rrformer/` | Yin et al. (2022) Rainfall-Runoff Transformer (`RR-Former`) | MIP Re-run |
| `rasiya_koya_tft/` | Rasiya Koya & Roy (2024) Temporal Fusion Transformer (`TFT`) | MIP Re-run |
| `song_dhbv11p/` | Song et al. (2024) Differentiable HBV 1.1p (`δHBV 1.1p / δMG`) | MIP Re-run |
| `li2025_lstm_dhbv_ens/` | Li et al. (2025) `LSTM + δHBV 1.1p` Multi-Model Ensemble | MIP Re-run |
| `poudel_mlp_hbv/` | Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (`diff-hydro`) | MIP Re-run |
| `era_gauged_multi_paradigm/` | ERA-Multi-Paradigm-Gauged (Node 248 `MultiScaleRoutingLSTM`) | Contribution |
| `era_gauged_physics_guided/` | ERA-Physics-Guided-Gauged (Node 228 `PhysicsGuidedHydroModel`) | Contribution |
| `era_gauged_minimal_guidance/` | ERA-Unguided-Gauged (Node 73 `EntityAwareLSTM`) | Contribution |
| `era_pub_physics_guided/` | ERA-Physics-Guided-PUB (Node 184 `HydroBottleneckLSTM`) | Contribution |
| `era_pub_minimal_guidance/` | ERA-Unguided-PUB (Node 178 `CrossForcingInteractionLSTM`) | Contribution |
| `era_pur_minimal_guidance/` | ERA-Unguided-PUR (Node 135 `DualFiLMLSTM`) | Contribution |
| `era_pur_physics_guided/` | ERA-Physics-Guided-PUR (Node 134 `PhysicsGuidedHydroLSTM`) | Contribution |

## Adding a New Model

1. Create `models/<your_model>/model.py` and `models/<your_model>/runner.py`.
2. Define `run_fold` in `runner.py`:

```python
from collections.abc import Sequence
from pathlib import Path
import numpy as np

from models.camels_data import load_camels_slice


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (123,),
    **kwargs: object,
) -> np.ndarray:
    """Return simulated daily discharge, shape (len(test_basins), n_eval_days), mm/day, float32.

    `run_model.py` calls this once per seed (`seeds=(seed,)`); never average seeds inside the runner.
    """
    spinup_days = 365
    train = load_camels_slice(train_basins, train_period, forcings, data_dir,
                              spinup_days=spinup_days, require_valid_targets=True)
    test = load_camels_slice(test_basins, eval_period, forcings, data_dir, spinup_days=spinup_days)
    # train["qobs"][:, :spinup_days] is NaN by construction; compute every statistic on
    # train["x_dyn"][:, spinup_days:] and train["target_qobs"], and emit the last
    # int(test["n_target_days"]) days of the test simulation.
    ...
```

3. Register the model in `MODEL_REGISTRY` inside `models/run_model.py`.

Model directories never load split files or hard-code dates: they receive `train_basins`,
`test_basins`, `train_period`, and `eval_period` from `run_model.py`, and they obtain data only through
`load_camels_slice`, which masks spin-up discharge, sanitizes negative sentinels, and raises on missing
forcings or attributes.

## Zarr Hydrograph Output

`models/zarr_writer.py` builds and validates an `xarray.Dataset` before writing a consolidated Zarr v2 store:

- **Dimensions**: `("basin", "date")` of shape `(531, 3652)`, plus `("member", "basin", "date")` for `qsim_members`.
- **Coordinates**: `basin` (`<U8`, sorted canonical IDs), `date` (`datetime64[ns]`, `1989-10-01`..`1999-09-30`,
  derived from the evaluation period), `huc_02` (`<U2` on `basin`), `member` (`int64`).
- **Variables**: `qsim` (`float32`, mm/day, finite everywhere, equal to the mean of `qsim_members`), `qsim_members`.
- **Attributes** (bucket / `metrics.evaluator` vocabulary): `entry_id`/`model_id`, `model_name`, `family_id`,
  `model_family`, `provenance_category` (`Literature`, `MIP Re-run`, `Archive`, `Archive*`, `Contribution`), `leaderboard_id`, `table_id`,
  `spatial_regime` (`gauged`/`pub`/`pur`) + `spatial_protocol`, `forcing_regime`, `forcings_used` (string) +
  `forcing_products` (list), `is_ensemble`, `num_members`, `train_period`/`test_period` (`A/B`) +
  `training_period`/`evaluation_period`, `units`, `code_url`, `archive_url`/`data_url`, and run provenance
  (`seeds`, `fold_ids`, `runner_kwargs_json`, `git_commit`, `created_utc`, `data_source`).
