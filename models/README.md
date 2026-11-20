# Models and Central Runner

`models/` holds **20 rainfall-runoff model implementations**: 12 `MIP Re-run` re-implementations of
published models and 8 `Contribution` models (7 ERA winners and 1 statistical lower benchmark). It
also holds the shared infrastructure:

| File | Role |
| :--- | :--- |
| `run_model.py` | The central runner, which owns the model registry |
| `catalog.json` | Machine-readable mirror of the registry |
| `camels_data.py` | Shared CAMELS-US loader |
| `zarr_writer.py` | Consolidated Zarr v2 hydrograph writer and validator |

Run every command below from the repository root.

## Running a Model

`models/run_model.py` is the **only** place that loads the canonical 531-basin list and the fold files
from `mips/<mip>/splits/`, and the only place that defines the canonical windows: training
`1999-10-01`..`2008-09-30` and evaluation `1989-10-01`..`1999-09-30`. It:

1. Checks that the model is registered for the requested spatial protocol and forcing regime (see
   [Allowed regimes](#allowed-regimes-anti-leakage-guardrail)).
2. Checks that the output target is writable, before any training starts.
3. Runs every fold of the spatial regime **once per random seed**.
4. Stitches each seed's out-of-fold predictions into a `(531, 3652)` array.
5. Writes one consolidated Zarr store with `qsim` (the seed mean) and `qsim_members` (the per-seed
   hydrographs).

```bash
python -m models.run_model --list-models

# Gauged regime, single forcing (Daymet) -> gauged__single_camels (Table 3)
python -m models.run_model --model kratzert_lstm --spatial-regime gauged --forcings daymet

# 12 random PUB folds, three forcings -> pub__multi_camels (Table 8)
python -m models.run_model --model feng_dhbv10 --spatial-regime pub_12fold --forcings daymet maurer nldas

# 18 leave-one-HUC02-out PUR folds, explicit seeds, 4 process workers
python -m models.run_model --model era_pur_minimal_guidance --spatial-regime pur_18huc02 \
  --forcings daymet maurer nldas --seeds 42 43 --max-workers 4 --executor process

# Lagged-streamflow models must name their leaderboard explicitly -> gauged__lagged_streamflow (Table 5)
python -m models.run_model --model naive_lag1_persistence --spatial-regime gauged \
  --forcings daymet --forcing-regime lagged_streamflow
```

| Flag | Meaning |
| :--- | :--- |
| `--model` | Registered model key (see the table below). |
| `--spatial-regime` | `gauged` (default), `pub_12fold`, or `pur_18huc02`. |
| `--forcings` | One or more **space-separated** products from `daymet`, `maurer`, `nldas`. One product gives `single_camels`; two or three give `multi_camels`. Duplicates are an error. |
| `--forcing-regime` | Optional explicit regime. Required for `lagged_streamflow` models; a regime that contradicts `--forcings` is an error. |
| `--data-dir` | The canonical bundle `camels_671_1980_2014.npz` or the directory that contains it, a raw NCAR CAMELS-US v1.2 directory, or a per-basin NetCDF layout. Default: `mips/<mip>/data`. See [`mips/camels_us_daily/data/README.md`](../mips/camels_us_daily/data/README.md). |
| `--splits-dir` | Directory with the canonical split files. Default: `mips/<mip>/splits`. |
| `--mip` | Registered MIP under `mips/`. Default: `camels_us_daily`. |
| `--output-zarr` | Target store path or `gs://` URI. Default: the local `./outputs/<leaderboard_id>/<model_id>.zarr`. The shared bucket is never written unless you pass a `gs://` URI and have write access. |
| `--model-id` | Explicit entry ID, used for the attributes and the store name. Default: `<model>[_<model_kind>]_<spatial_regime>_<forcing_regime>`. |
| `--seeds` | Override the registry's default seed list. Each seed is a full independent run of every fold. |
| `--max-workers`, `--executor`, `--threads-per-worker` | Parallelism over `(fold, seed)` tasks. With `thread` (the default), runs are reproducible only if the runner uses seed-local RNGs. `process` uses spawned workers, each capped at `--threads-per-worker` BLAS/torch threads; prefer it for CPU-bound calibrations. |
| `--runner-kwarg KEY=JSON` | Extra `run_fold` keyword argument, repeatable. The value is parsed as JSON. Unknown keys raise inside the runner. Every kwarg is recorded in the store's `runner_kwargs_json` attribute. |

The store's `archive_url`/`data_url` attributes always record the canonical bucket location
`gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`, which is used for registration.
The run provenance attributes are `seeds`, `fold_ids`, `runner_kwargs_json`, `git_commit`, `git_dirty`,
`created_utc` and `data_source`. `data_source` records only the final path component, so stores never
contain workstation paths.

## Allowed regimes (anti-leakage guardrail)

Each `ModelRegistryEntry` declares `allowed_forcing_regimes` and `allowed_spatial_regimes`, and
`run_model_experiment` raises before training on any mismatch.

- Models that read **observed discharge of the test basins** (`feng2020_di_lstm`, `yin2022_rrformer` and
  `naive_lag1_persistence`) are allowed only on `lagged_streamflow` leaderboards. Forcings-only models
  can never be filed there.
- The ERA gauged winners learn per-basin embeddings, so they are allowed only in the `gauged` regime. Their
  runners also raise on any unseen basin.
- The ERA PUB and PUR winners are registered only for the protocol they were selected in.

`models/catalog.json` mirrors the registry exactly: name, family, provenance, runner, code URL and
regimes. `models/tests/test_run_model.py` enforces the match.

## Model Directories

| Directory | Model | Category | Forcing regimes | Spatial regimes |
| :--- | :--- | :--- | :--- | :--- |
| `kratzert_lstm/` | Kratzert et al. (2019, 2021) CudaLSTM | MIP Re-run | single, multi | all |
| `kratzert_ealstm/` | Kratzert et al. (2019a) Entity-Aware LSTM (EA-LSTM) | MIP Re-run | single, multi | all |
| `hoedt_mclstm/` | Hoedt et al. (2021) Mass-Conserving LSTM (MC-LSTM) | MIP Re-run | single, multi | all |
| `rasiya_koya_tft/` | Rasiya Koya & Roy (2024) Temporal Fusion Transformer | MIP Re-run | single, multi | all |
| `shahriar2026_mamba3/` | Shahriar (2026) Mamba-3 Static-Dynamic Adapter (pure PyTorch) | MIP Re-run | single, multi | all |
| `poudel_mlp_hbv/` | Poudel & Steinschneider (2026) MLP + 16-zone differentiable HBV | MIP Re-run | single, multi | all |
| `feng_dhbv10/` | Feng et al. (2022) δHBV 1.0 | MIP Re-run | single, multi | all |
| `song_dhbv11p/` | Song et al. (2025) δHBV 1.1p | MIP Re-run | single, multi | all |
| `li2025_lstm_dhbv_ens/` | Li et al. (2025) LSTM + δHBV 1.1p multi-model ensemble | MIP Re-run | single, multi | all |
| `newman_sacsma/` | Snow-17 + SAC-SMA (per-basin DDS calibration; donor transfer in PUB/PUR) | MIP Re-run | single | all |
| `feng2020_di_lstm/` | Feng, Fang & Shen (2020) DI(1) lagged-discharge LSTM | MIP Re-run | lagged_streamflow | all |
| `yin2022_rrformer/` | Yin et al. (2022) RR-Former encoder-decoder Transformer | MIP Re-run | lagged_streamflow | all |
| `era_gauged_multi_paradigm/` | ERA-Multi-Paradigm-Gauged (Node 2; `CausalConvGRULSTM`, `MultiScaleFiLMEALSTM`, `DualPathwayFiLMGRULSTM`, `EntityAwareHydroLSTM` ensemble) | Contribution | multi | gauged |
| `era_gauged_physics_guided/` | ERA-Physics-Guided-Gauged (Node 187; `PhysicsHybridEALSTM` + `DifferentiableSnowPack`) | Contribution | multi | gauged |
| `era_gauged_minimal_guidance/` | ERA-Unguided-Gauged (Node 73; `EntityAwareLSTM`) | Contribution | multi | gauged |
| `era_pub_physics_guided/` | ERA-Physics-Guided-PUB (Node 203; `PhysicsEALSTM`) | Contribution | multi | pub_12fold |
| `era_pub_minimal_guidance/` | ERA-Unguided-PUB (Node 82; `EALSTM`) | Contribution | multi | pub_12fold |
| `era_pur_minimal_guidance/` | ERA-Unguided-PUR (Node 135; `DualFiLMLSTM` + `StatEncoder`) | Contribution | multi | pur_18huc02 |
| `era_pur_physics_guided/` | ERA-Physics-Guided-PUR (Node 206; `PURPhysicsModel`) | Contribution | multi | pur_18huc02 |
| `naive_lag1_persistence/` | Naive lag-1 streamflow persistence, a zero-parameter lower benchmark | Contribution | lagged_streamflow | all |

In this table, "single" and "multi" mean `single_camels` and `multi_camels`, and "all" means `gauged`,
`pub_12fold` and `pur_18huc02`. Each directory has its own `README.md`, a
`reproduce_author_metrics.py` Stage-1 script and its `stage1_author_reproduction_results.json` record.
Stage-1 scripts write untracked `*.local.json` results unless you explicitly ask them to regenerate
the tracked record. Uploads to GCS are opt-in.

## Build Prerequisites

- **Python packages**: install the repository with its model extras, `pip install -e ".[models]"`.
  This gives you `torch` and `ninja`.
- **C++ extensions.** `newman_sacsma`, `feng_dhbv10`, `song_dhbv11p` (shared core in `_dhbv_common/`),
  `li2025_lstm_dhbv_ens` and `poudel_mlp_hbv` compile small C++ (and, for GPU training, CUDA)
  kernels on first use with `torch.utils.cpp_extension.load_inline`. You need:
  - a C++17 compiler; on Linux, `g++` with OpenMP. On macOS the kernels build without OpenMP and run
    serially.
  - `ninja`.
  - for GPU training only, an `nvcc` toolkit that matches `torch.version.cuda`, located through
    `CUDA_HOME`.
- **Build cache.** The first call compiles for a minute or two. Results are cached under
  `~/.cache/torch_extensions`; set `TORCH_EXTENSIONS_DIR` to use a per-run cache when many processes
  start at once.
- **No compile step**: the LSTM family, the Transformers, Mamba-3, the ERA models and the persistence
  benchmark are pure PyTorch or NumPy.

## Adding a New Model

1. Create `models/<your_model>/model.py`, `runner.py`, `__init__.py` and `README.md`. Put the
   standard two-line Apache-2.0 SPDX header at the top of every `.py` file.
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
    **kwargs: object,  # receives fold_id / fold_index; raise on any other unknown key
) -> np.ndarray:
    """Return simulated daily discharge, shape (len(test_basins), n_eval_days), mm/day, float32.

    `run_model.py` calls this once per seed (`seeds=(seed,)`); never average seeds inside the runner.
    Never replace NaN predictions with numbers: run_model rejects non-finite output on purpose.
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

3. Register the model in `MODEL_REGISTRY` in `models/run_model.py`, including
   `allowed_forcing_regimes` and `allowed_spatial_regimes`. Add the identical entry to
   `models/catalog.json`.
4. Add the directory name to `EXPECTED_MODEL_DIRS` in `models/tests/test_run_model.py`, then run
   `pytest models/tests` and `ruff check models`.

Model directories never load split files or hard-code canonical dates. They receive `train_basins`,
`test_basins`, `train_period` and `eval_period` from `run_model.py`. Reproduction scripts import
`CANONICAL_*` from `models.zarr_writer`. All data comes through `load_camels_slice`, which:
- masks spin-up discharge;
- sanitizes negative sentinels;
- raises on missing forcings or attributes;
- returns the auxiliary `dayl`/`swe` channels (`x_aux`) only when the source provides them. The
  canonical bundle provides Daymet's for every product, and `aux_sources` records this.

## Zarr Hydrograph Output

`models/zarr_writer.py` builds and validates an `xarray.Dataset` before writing a consolidated Zarr v2
store:

- **Dimensions**: `("basin", "date")` with shape `(531, 3652)`, plus `("member", "basin", "date")` for
  `qsim_members`.
- **Coordinates**: `basin` (`<U8`, sorted canonical IDs); `date` (`datetime64[ns]`,
  `1989-10-01`..`1999-09-30`, derived from the evaluation period); `huc_02` (`<U2` on `basin`);
  `member` (`int64`).
- **Variables**: `qsim` (`float32`, mm/day, finite everywhere, equal to the mean of `qsim_members`) and
  `qsim_members`.
- **Attributes** (bucket and `metrics.evaluator` vocabulary):
  - `entry_id`/`model_id`, `model_name`, `family_id`, `model_family`
  - `provenance_category`: one of `Literature`, `MIP Re-run`, `Archive`, `Archive*`, `Contribution`
  - `leaderboard_id`, `table_id`
  - `spatial_regime` (`gauged`/`pub`/`pur`) and `spatial_protocol`
  - `forcing_regime`, `forcings_used` (string) and `forcing_products` (list)
  - `is_ensemble`, `num_members`
  - `train_period`/`test_period` (`A/B`) and `training_period`/`evaluation_period`
  - `units`, `code_url`, `archive_url`/`data_url`
  - the run provenance attributes listed above
