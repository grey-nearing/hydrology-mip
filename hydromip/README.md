# `hydromip`: Shared Core Package

`hydromip` is the shared core of HydrologyMIP (<https://hydrologymip.org>). Every benchmark uses
it. In this repository a benchmark is called a **MIP** (model intercomparison project). Each MIP
lives in its own folder, `mips/<mip_id>/`, and describes itself in a `protocol.json` file.

`hydromip` does five jobs:

1. It finds every MIP and checks its `protocol.json` file.
2. It gives the folder paths of each MIP.
3. It loads cross-validation (CV) folds and checks them for data leakage.
4. It keeps a registry of metric suites, including the 14 daily streamflow metrics.
5. It opens Zarr stores on a local disk or in Google Cloud Storage.

It also holds the repository checks that CI runs. It has no model code and no data. Models live
in [`models/`](../models/README.md), scoring lives in [`metrics/`](../metrics/README.md), and the
benchmark folders live in [`mips/`](../mips/README.md).

## Overview

| Module | What it does | Main names |
| :--- | :--- | :--- |
| [`protocol.py`](protocol.py) | Loads and checks `mips/<mip_id>/protocol.json`. Finds all MIPs. | `discover_mips`, `get_mip`, `load_mip_protocol`, `validate_protocol_dict`, `validate_mip_assets`, `MipProtocol` |
| [`paths.py`](paths.py) | Gives the repository root and the folders of each MIP. | `resolve_mip_paths`, `resolve_mip_subdir`, `MipPaths`, `REPO_ROOT` |
| [`splits.py`](splits.py) | Loads CV folds and checks them for leakage. | `CvFold`, `load_cv_splits_json`, `validate_cv_folds` |
| [`metrics/registry.py`](metrics/registry.py) | Registry of metric suites. | `MetricSuiteSpec`, `register_metric_suite`, `get_metric_suite`, `list_metric_suites` |
| [`metrics/deterministic_daily.py`](metrics/deterministic_daily.py) | The 14 daily streamflow metrics of the Gauch et al. (2023) suite. This is the only implementation; `metrics/metrics.py` re-exports it. | `calculate_all_metrics`, `calculate_metrics`, `nse`, `kge`, `log_nse`, `log_kge`, and the other metric functions |
| [`evaluation/zarr_io.py`](evaluation/zarr_io.py) | Opens local, public `https://`, or `gs://` Zarr stores and chooses Google Cloud credentials. | `open_zarr_store`, `zarr_store_exists`, `get_gcs_storage_options`, `deterministic_io`, `validate_store_id` |
| [`io/`](io/__init__.py) | Safe multi-format prediction ingestion (`NetCDF`, `Zarr`, `CSV`, `Parquet`, `NPZ`), 531-basin physical sanity checks, and optional schema fallback. | `ingest_hydrograph`, `extract_file_metadata`, `resolve_deterministic_recipe`, `run_physical_sanity_checks`, `write_canonical_zarr_v2` |
| [`leaderboards/registry.py`](leaderboards/registry.py) | Gives the leaderboard and figure folders of a MIP. | `get_mip_leaderboards_dir`, `get_mip_figures_dir`, `get_mip_leaderboard_ids` |
| [`tools/cli.py`](tools/cli.py) | Unified CLI entrypoint (`hydromip package`, `hydromip-package`, `process-submission`, `guardrails`). | `hydromip package --input ... --output ...` |
| [`tools/ci_guardrails.py`](tools/ci_guardrails.py) | The repository checks that CI runs. | `python -m hydromip.tools.ci_guardrails` |
| [`tools/enable_public_launch_guardrails.sh`](tools/enable_public_launch_guardrails.sh) | **Maintainers only.** Turns on branch protection for `main` on GitHub. | `bash hydromip/tools/enable_public_launch_guardrails.sh` |
| [`tests/`](tests/) | Tests for this package. | `pytest hydromip/tests` |

## Prerequisites

- Python 3.11 or newer.
- A copy of the repository. Run every command from the repository root.
- Install the package:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```

  The core install is enough for `hydromip`. To run the tests and the CI checks, install the
  `dev` extra instead (`pip install -e ".[dev]"`). It adds `pytest`, `ruff`, and `pyyaml`, plus
  the `portal` and `models` extras.

You do not need a Google Cloud account to use `hydromip`.

## Quick start

```python
from hydromip import discover_mips, get_mip

print(sorted(discover_mips()))     # ['camels_us_daily']

proto = get_mip("camels_us_daily")
print(proto.task_type)             # streamflow_daily_deterministic
print(len(proto.leaderboard_ids))  # 12
print(proto.leaderboard_ids[0])    # gauged__single_camels
print(proto.temporal_windows["eval_start"], proto.temporal_windows["eval_end"])
print(proto.paths.splits_dir)      # <repository>/mips/camels_us_daily/splits
```

## Public API

`from hydromip import ...` gives these names (`hydromip.__all__`):

| Group | Names |
| :--- | :--- |
| Protocol | `SCHEMA_VERSION`, `SUPPORTED_TASK_TYPES`, `MipProtocol`, `SpatialRegimeSpec`, `ForcingRegimeSpec`, `LeaderboardSpec`, `discover_mips`, `get_mip`, `load_mip_protocol`, `validate_protocol_dict` |
| Paths | `DEFAULT_MIP_ID`, `REPO_ROOT`, `MIPS_DIR`, `MODELS_DIR`, `MipPaths`, `get_repo_root`, `resolve_mip_paths`, `resolve_mip_subdir` |
| Splits | `CvFold`, `load_cv_splits_json`, `validate_cv_folds` |
| Metric suites | `MetricSuiteSpec`, `METRIC_SUITES`, `GAUCH_2023_DETERMINISTIC_SUITE`, `register_metric_suite`, `get_metric_suite`, `list_metric_suites` |

The subpackages export more names:

| Subpackage | Names |
| :--- | :--- |
| `hydromip.metrics` | The metric suite names above, plus `calculate_all_metrics`, `calculate_metrics`, `get_available_metrics`, `nse`, `mse`, `rmse`, `kge`, `log_nse`, `log_kge`, `alpha_nse`, `pearsonr`, `beta_kge`, `beta_nse`, `fdc_fhv`, `fdc_fms`, `fdc_flv`, `mean_peak_timing`, `infer_datetime_coord`, `get_frequency_factor`, `AllNaNError` |
| `hydromip.evaluation` | `open_zarr_store`, `zarr_store_exists`, `get_gcs_storage_options`, `running_on_gce`, `deterministic_io`, `validate_store_id`, `STORE_ID_PATTERN` |
| `hydromip.leaderboards` | `get_mip_leaderboards_dir`, `get_mip_figures_dir`, `get_mip_leaderboard_ids` |

These names are public but are not exported from the top level: `hydromip.protocol.validate_mip_assets`,
`hydromip.protocol.SUPPORTED_STATUSES`, `hydromip.paths.validate_mip_id`,
`hydromip.paths.MIP_SUBDIRS`, and `hydromip.paths.MIP_ID_PATTERN`.

### Functions

| Function | What it does |
| :--- | :--- |
| `discover_mips(repo_root=None)` | Loads every `<repo_root>/mips/*/protocol.json`. Returns `{mip_id: MipProtocol}`. Raises `FileNotFoundError` if there is no `mips/` folder or no protocol file. |
| `get_mip(mip_id="camels_us_daily", repo_root=None)` | Returns one `MipProtocol`. Raises `KeyError` for an unknown ID. |
| `load_mip_protocol(protocol_path, repo_root=None)` | Loads and checks one `protocol.json` file. The folder name must equal `mip_id`. |
| `validate_protocol_dict(payload, expected_mip_id=None)` | Checks a `protocol.json` dictionary (see [the schema](#the-protocoljson-schema)). Raises `KeyError` or `ValueError`. |
| `validate_mip_assets(proto)` | Checks the files on disk (see [Folder layout](#folder-layout-of-a-mip)). |
| `resolve_mip_paths(mip_id="camels_us_daily", repo_root=None)` | Returns the `MipPaths` of a MIP. It does not check that the folders exist. |
| `resolve_mip_subdir(data_root, subdir, mip_id="camels_us_daily", marker_filename=None)` | Finds one MIP folder below a data root (see [Folder layout](#folder-layout-of-a-mip)). |
| `get_repo_root(repo_root=None)` | Returns `repo_root` as an absolute path, or `REPO_ROOT` when it is `None`. |
| `load_cv_splits_json(...)`, `validate_cv_folds(...)` | See [CV hygiene](#cv-hygiene). |
| `register_metric_suite(spec, *, overwrite=False)`, `get_metric_suite(suite_id)`, `list_metric_suites()` | See [Register a metric suite](#register-a-metric-suite). |

### `MipProtocol`

`MipProtocol` is a frozen dataclass. It has one field per required key of `protocol.json`.
`spatial_regimes`, `forcing_regimes`, and `leaderboards` are tuples of `SpatialRegimeSpec`,
`ForcingRegimeSpec`, and `LeaderboardSpec`. It also has:

| Member | Meaning |
| :--- | :--- |
| `paths` | The `MipPaths` of the MIP. |
| `raw` | The full `protocol.json` dictionary, including keys that are not checked. |
| `leaderboard_ids`, `spatial_regime_ids`, `forcing_regime_ids` | Tuples of IDs, in file order. |
| `get_spatial_regime(regime_id)` | Returns one `SpatialRegimeSpec`, or raises `KeyError`. |
| `to_summary_dict()` | A JSON-ready summary. The web portal serves it. |

## The `protocol.json` schema

Each MIP has one `protocol.json` file. `validate_protocol_dict` checks it every time it is loaded.
The schema version is `hydromip-protocol-v1` (`SCHEMA_VERSION`).

### Required top-level keys

| Key | Type | Rule |
| :--- | :--- | :--- |
| `schema_version` | string | Must be `"hydromip-protocol-v1"`. |
| `mip_id` | string | Must equal the folder name. Allowed: letters, digits, `_` and `-`; it starts with a letter or digit; at most 64 characters. |
| `name`, `short_name`, `version` | string | Not empty. |
| `status` | string | One of `active`, `beta`, `planned`, `deprecated`, `archived`. |
| `task_type` | string | One of the task types listed below. |
| `metric_suite` | string | A suite ID in the metric registry (`METRIC_SUITES`). |
| `primary_metric` | string | One of the metric names of that suite. |
| `primary_metric_higher_is_better` | boolean | Must agree with the suite when `primary_metric` is the suite's own primary metric. |
| `target_variable`, `target_units` | string | Not empty (for example `qobs` and `mm/day`). |
| `prediction_dims` | list of strings | Not empty. No repeated names. |
| `entity_unit` | string | Must be one of `prediction_dims` (for example `basin`). |
| `num_entities`, `num_eval_steps` | integer | Positive. |
| `temporal_windows` | object | Not empty. The keys are free (for example `train_start`, `eval_end`). |
| `spatial_regimes` | list of objects | Not empty. See below. |
| `forcing_regimes` | list of objects | Not empty. See below. |
| `leaderboards` | list of objects | Not empty. See below. |
| `storage` | object | May be empty. `mip-score-model` reads `storage.bucket_uri` as its default bucket. |

Supported task types (`SUPPORTED_TASK_TYPES`): `streamflow_daily_deterministic`,
`streamflow_daily_probabilistic`, `streamflow_forecast_deterministic`,
`streamflow_forecast_probabilistic`, `streamflow_realtime`, `soil_moisture_daily`, and
`flood_inundation_2d`. Today only one metric suite is registered:
`gauch_2023_deterministic` (14 metrics, primary metric `NSE`).

### List entries

| List | Required keys | Optional keys | Rules |
| :--- | :--- | :--- | :--- |
| `spatial_regimes` | `id`, `leaderboard_prefix`, `title`, `num_folds`, `description` | `split_file`, `section` | `num_folds` is a positive integer. If `num_folds` is more than 1, `split_file` must name a file in `splits/`. IDs and prefixes are unique. |
| `forcing_regimes` | `id`, `title`, `description` | — | IDs are unique. |
| `leaderboards` | `id`, `spatial_regime`, `forcing_regime`, `title` | `table_number`, `section` | `spatial_regime` and `forcing_regime` must name declared IDs. `id` must equal `<leaderboard_prefix>__<forcing_regime>`. Each pair appears once. `table_number` is a positive integer or `null`. |

All required keys of the list entries are non-empty strings, except `num_folds`.

### The `paths` key

`paths` is optional. The folder names of a MIP are fixed (see [Folder layout](#folder-layout-of-a-mip)).
A `paths` block may only repeat them, for example `{"splits": "splits"}`. Any other value raises
`ValueError`. The `camels_us_daily` protocol has no `paths` block.

### Other keys

You may add other keys, for example `authors`, `citation`, `temporal_resolution`, or
`ensemble_prediction_dims` (all used in `mips/camels_us_daily/protocol.json`). They are not
checked. You can read them from `MipProtocol.raw`.

### Example

This file is valid for a folder named `mips/example_daily/`:

```json
{
  "schema_version": "hydromip-protocol-v1",
  "mip_id": "example_daily",
  "name": "Example Daily Streamflow MIP",
  "short_name": "Example Daily",
  "status": "planned",
  "version": "0.1.0",
  "task_type": "streamflow_daily_deterministic",
  "metric_suite": "gauch_2023_deterministic",
  "primary_metric": "NSE",
  "primary_metric_higher_is_better": true,
  "target_variable": "qobs",
  "target_units": "mm/day",
  "prediction_dims": ["basin", "date"],
  "entity_unit": "basin",
  "num_entities": 100,
  "num_eval_steps": 3653,
  "temporal_windows": {
    "train_start": "2000-10-01",
    "train_end": "2010-09-30",
    "eval_start": "2010-10-01",
    "eval_end": "2020-09-30"
  },
  "spatial_regimes": [
    {
      "id": "gauged",
      "leaderboard_prefix": "gauged",
      "title": "Gauged",
      "num_folds": 1,
      "split_file": null,
      "description": "Train and test on all basins."
    },
    {
      "id": "pub_5fold",
      "leaderboard_prefix": "pub",
      "title": "5-Fold PUB",
      "num_folds": 5,
      "split_file": "pub_5fold_splits.json",
      "description": "5-fold random cross-validation over basins."
    }
  ],
  "forcing_regimes": [
    {"id": "era5land", "title": "ERA5-Land", "description": "ERA5-Land forcing only."}
  ],
  "leaderboards": [
    {
      "id": "gauged__era5land",
      "spatial_regime": "gauged",
      "forcing_regime": "era5land",
      "title": "Gauged - ERA5-Land",
      "table_number": 1
    },
    {
      "id": "pub__era5land",
      "spatial_regime": "pub_5fold",
      "forcing_regime": "era5land",
      "title": "PUB - ERA5-Land",
      "table_number": 2
    }
  ],
  "storage": {}
}
```

### Check a protocol

```bash
python -c "from hydromip import get_mip; print(get_mip('camels_us_daily').to_summary_dict()['n_leaderboards'])"
python -c "from hydromip import get_mip; from hydromip.protocol import validate_mip_assets; validate_mip_assets(get_mip('camels_us_daily')); print('assets OK')"
```

## Folder layout of a MIP

`resolve_mip_paths(mip_id)` returns a `MipPaths` object with these fields:

| Field | Path |
| :--- | :--- |
| `root` | `mips/<mip_id>/` |
| `protocol_file` | `mips/<mip_id>/protocol.json` |
| `splits_dir` | `mips/<mip_id>/splits/` |
| `leaderboards_dir` | `mips/<mip_id>/leaderboards/` |
| `figures_dir` | `mips/<mip_id>/figures/` |
| `data_dir` | `mips/<mip_id>/data/` |
| `era_experiments_dir` | `mips/<mip_id>/era_experiments/` |

The subfolder names are fixed (`MIP_SUBDIRS`). `resolve_mip_paths` does not check that the
folders exist, so a new MIP can be set up step by step.

`validate_mip_assets(proto)` checks the files on disk:

- `splits/`, `leaderboards/`, and `figures/` exist;
- every `split_file` named in `protocol.json` exists, and a JSON split file has exactly
  `num_folds` folds under its `folds` key;
- every declared leaderboard has a file `leaderboards/<leaderboard_id>.json` whose `id` matches.

`resolve_mip_subdir(data_root, subdir, mip_id, marker_filename=None)` finds one folder below a
data root. If `<data_root>/mips/` exists, it returns `<data_root>/mips/<mip_id>/<subdir>`.
Otherwise it treats `data_root` as a flat data folder for one MIP (for example a test fixture) and
returns `<data_root>/<subdir>`. It never uses the flat layout at the repository root. It raises
`FileNotFoundError` if the folder (or `marker_filename` inside it) is missing. The web portal uses
this function.

## Register a metric suite

A metric suite is a named set of metrics. `protocol.json` names one suite in `metric_suite`.

```python
import numpy as np

from hydromip import MetricSuiteSpec, get_metric_suite, register_metric_suite


def compute_error_suite(obs, sim):
    """Return the mean absolute error and the mean bias on days where both values are finite."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return {
        "MAE": float(np.mean(np.abs(sim[ok] - obs[ok]))),
        "Bias": float(np.mean(sim[ok] - obs[ok])),
    }


register_metric_suite(
    MetricSuiteSpec(
        suite_id="daily_error_demo",
        description="Mean absolute error and mean bias of daily values (example).",
        metric_names=("MAE", "Bias"),
        primary_metric="MAE",
        higher_is_better=False,
        compute_fn=compute_error_suite,
        citation="Your paper (year), DOI",
    )
)

suite = get_metric_suite("daily_error_demo")
print(suite.compute_fn([1.0, 2.0, 3.0], [1.5, 2.0, 2.0]))  # {'MAE': 0.5, 'Bias': -0.1666...}
```

`MetricSuiteSpec` fields:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `suite_id` | `str` | The ID that `protocol.json` uses in `metric_suite`. Not empty. |
| `description` | `str` | Short description. |
| `metric_names` | `tuple[str, ...]` | All metric names of the suite. |
| `primary_metric` | `str` | The ranking metric. It must be in `metric_names`. |
| `higher_is_better` | `bool` | Direction of the primary metric. |
| `compute_fn` | callable | Computes the metrics. Returns a dictionary from metric name to `float`. |
| `citation` | `str` | Reference for the suite. |

Rules:

- `register_metric_suite` raises `ValueError` if `suite_id` is empty, if `primary_metric` is not in
  `metric_names`, or if the ID is already registered. Pass `overwrite=True` to replace a suite.
- A registration lasts only while the Python process runs. To add a permanent suite, put its code
  in a new module under `hydromip/metrics/` and add its `MetricSuiteSpec` to `METRIC_SUITES` in
  [`metrics/registry.py`](metrics/registry.py). A `protocol.json` that names a suite fails to load
  until the suite is registered.
- The CAMELS-US evaluator (`metrics/evaluator.py`) calls the metric functions of
  `gauch_2023_deterministic` directly. A MIP with a new suite also needs its own scoring code.

## CV hygiene

These tools check that cross-validation folds do not leak data between training and testing.

`CvFold` is a frozen dataclass with four fields: `fold_id` (`str`), `fold_index` (`int`),
`train_entities` (`tuple[str, ...]`), and `test_entities` (`tuple[str, ...]`).

`validate_cv_folds(folds, canonical_entities, *, ...)` always checks that:

- there is at least one fold, and fold IDs and fold indexes are unique;
- no fold repeats an entity, and no fold has an empty train or test set;
- every entity belongs to `canonical_entities`, and `canonical_entities` has no repeats.

The keyword options add more checks:

| Option | Default | Extra check |
| :--- | :--- | :--- |
| `require_disjoint_train_test` | `True` | No entity is in both the train set and the test set of a fold. |
| `require_exact_test_partition` | `True` | Every canonical entity is tested exactly once over all folds. |
| `entity_to_group` | `None` | A map from entity to group (for example basin to HUC-02 region). With `require_disjoint_train_test=True`, no group is in both the train and the test set of a fold. |
| `require_train_test_union_equals_cohort` | `False` | In every fold, train plus test covers the whole cohort. |
| `require_single_test_group` | `False` | Every test set is exactly one group (leave one group out). Needs `entity_to_group`. |

Any failure raises `ValueError` (or `KeyError` if `entity_to_group` misses an entity).

`load_cv_splits_json(split_json_path, canonical_entities, *, expected_num_folds=None, ...)` reads a
split file and then calls `validate_cv_folds` with the same keyword options. The file is a JSON
object with a `folds` object. Each fold has `train_basins` and `test_basins` (or `train_entities`
and `test_entities`), and may have `fold_index`. Folds are sorted by fold ID. Without
`fold_index`, a fold gets its sorted position.

Example: load the 18 leave-one-region-out folds of `camels_us_daily` with every check turned on.

```python
from hydromip import load_cv_splits_json, resolve_mip_paths
from mips.camels_us_daily import load_basin_to_huc02, load_canonical_basins

splits_dir = resolve_mip_paths("camels_us_daily").splits_dir
basins = load_canonical_basins()
folds = load_cv_splits_json(
    splits_dir / "pur_18fold_huc02_splits.json",
    basins,
    expected_num_folds=18,
    entity_to_group=load_basin_to_huc02(),
    require_train_test_union_equals_cohort=True,
    require_single_test_group=True,
)
print(len(folds), folds[0].fold_id, len(folds[0].test_entities))  # 18 huc_01 23
```

## Zarr I/O and cloud credentials

| Function | What it does |
| :--- | :--- |
| `open_zarr_store(store_uri)` | Opens a Zarr store with `xarray`. Accepts a local path, a `file://` path, or a `gs://` URI. Raises `FileNotFoundError` if the store does not exist. Uses consolidated metadata when the store has it. |
| `zarr_store_exists(store_uri, storage_options=None)` | Returns `True` if the store exists. |
| `validate_store_id(value, field="model_id")` | Returns `value` if it is a safe store or entry ID: a letter or digit, then up to 127 letters, digits, `.`, `_` or `-`, and never `..`. Otherwise raises `ValueError`. |
| `deterministic_io()` | Context manager. Inside the block, Zarr runs one read request at a time and `dask` uses its synchronous scheduler. Both settings are restored after the block. |
| `get_gcs_storage_options()` | Returns the credentials for `gs://` access (see below). |
| `running_on_gce()` | Returns `True` on a Google Compute Engine VM. `NO_GCE_CHECK=1` makes it return `False`. |

Local paths never need credentials. Contributors only use local paths. The shared bucket
`gs://camels-us-mip` is for maintainers only.

**Maintainers only.** For `gs://` URIs, `get_gcs_storage_options()` uses the first match of:

1. `HYDROMIP_GCS_ANON=1` (or `true`, `yes`): anonymous access.
2. `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, then `GOOGLE_APPLICATION_CREDENTIALS`: that
   credentials file. A variable that points to a missing file raises `FileNotFoundError`.
3. The gcloud application-default credentials file,
   `~/.config/gcloud/application_default_credentials.json`.
4. On a Google Compute Engine VM: the VM's default credentials.
5. Otherwise: anonymous access. This can read public buckets only.

## Leaderboard folders

| Function | Returns |
| :--- | :--- |
| `get_mip_leaderboards_dir(mip_id="camels_us_daily", repo_root=None)` | `mips/<mip_id>/leaderboards/` |
| `get_mip_figures_dir(mip_id="camels_us_daily", repo_root=None)` | `mips/<mip_id>/figures/` |
| `get_mip_leaderboard_ids(mip_id="camels_us_daily", repo_root=None)` | The leaderboard IDs from `protocol.json`, in file order. |

Import them from `hydromip.leaderboards`. The tools that write leaderboard files live in the
separate [`leaderboards/`](../leaderboards/README.md) package.

## Repository checks (`ci_guardrails`)

CI runs these checks on every pull request. Run them yourself before you open one:

```bash
python -m hydromip.tools.ci_guardrails --mode all
python -m hydromip.tools.ci_guardrails --self-test
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--mode {code-guardrails,docs-and-registry,all}` | `all` | Which checks to run. |
| `--self-test` | off | Run the built-in tests of the checks on a temporary folder, then exit. |

The exit code is 0 when there are no problems. Otherwise it is 1 and every problem is printed.
The checks read all files that git tracks, plus new files that git does not ignore.

`code-guardrails` checks that:

- no file is larger than 5 MiB, and no data, checkpoint, or Zarr file is committed;
- no file contains a home-folder path, an internal company path or host name, or a secret key;
- Python code outside `tests/` folders has no `try`/`except` block and does not hide errors or
  warnings;
- `np.load` passes `allow_pickle=False`, and `torch.load` passes `weights_only=True`;
- non-test Python files and notebooks do not hard-code temporary-folder paths;
- model code in `models/<name>/` does not name split files or the canonical protocol dates
  (`reproduce_author_metrics.py` files and tests are exempt).

`docs-and-registry` checks that:

- every relative link in a Markdown file points to a file or folder that exists;
- Markdown files and skills contain no banned words and no internal paths;
- when `AGENTS.md` exists, the agent skills follow the layout rules
  (`skills/<name>/SKILL.md` with matching `name`, `.claude/skills` and `.agents/skills` links,
  `CLAUDE.md` and `GEMINI.md`, and every skill listed in `AGENTS.md`);
- `CITATION.cff`, `.zenodo.json`, and `pyproject.toml` agree on authors, version, and license;
- the model folders, `models/catalog.json`, and `MODEL_REGISTRY` in `models/run_model.py` list the
  same models;
- for every active MIP, the leaderboard files, `tables_3_to_14.json`, `table_2_registry.json`, and
  `figures/figures_manifest.json` agree with each other.

**Maintainers only:** `hydromip/tools/enable_public_launch_guardrails.sh` sets the GitHub branch
protection of `main` (required `CI Gate` check and code-owner review). It needs the GitHub CLI
`gh` with admin rights on the repository, and a passing `CI Gate` check on `main`.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--enforce-admins true\|false` | `true` | Whether the rules also apply to repository admins. |
| `--repo owner/name` | `grey-nearing/camels-us-mip` | The GitHub repository to configure. |
| `-h`, `--help` | — | Print the usage text. |

## Tests

```bash
pytest hydromip/tests
```

The tests check, among other things, that `camels_us_daily` is found and its files are complete,
that bad protocols are rejected, that the CV checks catch leakage, that the metric registry accepts
new suites, and that a new MIP folder is found without code changes.
