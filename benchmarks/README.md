# `benchmarks`: Verification Suite

The `benchmarks` package scores many hydrograph stores in one run and checks that each one
follows the benchmark rules. It finds every store under `<root>/hydrographs/`, scores each one
with the `metrics` package, and writes a ranked report. The root can be a folder on your own
computer or the maintainers' bucket.

For each store it checks five rules. The run stops with an error at the first store that breaks
one:

1. no missing prediction on a day with a valid observation (`pred_nan_when_obs_valid` is 0);
2. all 531 basins are scored and the period has 3,652 days;
3. every headline score and every percentile is a finite number;
4. the NSE, KGE, and peak-flow-bias percentiles are in increasing order;
5. all 18 HUC-02 regions have a finite median NSE.

The report has, per store, the five headline scores, the low tail of the NSE and KGE spread (from
the minimum to the 75th percentile), the high tail of the absolute peak-flow bias (from the 50th
percentile to the maximum), and the 18 regional medians. The full scorer output, with all 14
metrics, is kept for every store.

## Overview

| Component | Entry point | Inputs | Output |
| :--- | :--- | :--- | :--- |
| Command line | `benchmark-mip-metrics` (or `python -m benchmarks.metrics_pipeline`) | `<root>/hydrographs/<leaderboard_id>/<model_id>.zarr` and `<root>/observations/newman_eval_targets.zarr` | Prints a Markdown table. Can save a JSON report. |
| Python API | `run_metrics_benchmark_pipeline(...)` | Same | A report dictionary |

## Prerequisites

- Python 3.11 or newer, with the repository installed: `pip install -e .` from the repository root.
- Run every command from the repository root.
- A root folder with the observation targets and the stores:

  ```text
  local_bucket/
  ├── observations/newman_eval_targets.zarr
  └── hydrographs/<leaderboard_id>/<model_id>.zarr
  ```

  [`metrics/README.md`](../metrics/README.md) shows how to build the targets and fill this folder.
  You do not need a Google Cloud account.

## Quick start

Score every store in a local root folder:

```bash
benchmark-mip-metrics \
  --bucket-uri ./local_bucket \
  --output-json ./outputs/benchmark_report.json
```

Score only some leaderboards (repeat the flag):

```bash
benchmark-mip-metrics \
  --bucket-uri ./local_bucket \
  --leaderboard-id gauged__lagged_streamflow \
  --leaderboard-id pub__lagged_streamflow
```

Score only the stores listed in a manifest file:

```bash
benchmark-mip-metrics --bucket-uri ./local_bucket --manifest-path ./my_manifest.json
```

A manifest is a JSON list of objects with `leaderboard_id` and `model_id`:

```json
[
  {"leaderboard_id": "gauged__lagged_streamflow", "model_id": "persistence_test"}
]
```

A JSON object with an `entries` key also works. `entries` can be such a list, or an object keyed
by entry ID whose values have `leaderboard_id` or `target_leaderboard_tables`. That is the shape of
[`mips/camels_us_daily/leaderboards/contributions_index.json`](../mips/camels_us_daily/leaderboards/contributions_index.json).

> [!WARNING]
> Never write a report into `leaderboards/` or into `mips/<mip_id>/leaderboards/`. A report is not
> a leaderboard file. If it replaced one, it would break the leaderboards and CI. Write reports to
> `./outputs/` (git ignores it) or another folder outside the repository data.

## Python API

```python
from benchmarks import format_verification_markdown_table, run_metrics_benchmark_pipeline

report = run_metrics_benchmark_pipeline(
    bucket_uri="./local_bucket",
    leaderboard_ids=["gauged__lagged_streamflow"],
)
print("Stores scored:", report["total_evaluated_entries"])
print(format_verification_markdown_table(report))
```

| Function | What it does |
| :--- | :--- |
| `run_metrics_benchmark_pipeline(bucket_uri="gs://camels-us-mip", entries=None, manifest_path=None, leaderboard_ids=None, output_json=None)` | Scores and checks the stores. Returns the report. |
| `discover_hydrograph_zarr_entries(bucket_uri="gs://camels-us-mip", leaderboard_ids=None)` | Lists the `(leaderboard_id, model_id)` pairs of all stores under `<root>/hydrographs/`. Raises `FileNotFoundError` when it finds none. |
| `load_manifest_entries(manifest_path)` | Reads the `(leaderboard_id, model_id)` pairs from a manifest file. |
| `verify_single_entry_report(entry_metrics)` | Checks the five rules on one scorer result. Raises `KeyError` or `ValueError`. |
| `format_verification_markdown_table(report)` | Returns the report as a Markdown table. |

The report has these keys:

| Key | Content |
| :--- | :--- |
| `bucket_uri` | The root that was scored. |
| `verified_at_utc` | The time of the run. |
| `total_evaluated_entries` | The number of stores scored. |
| `total_pred_nan_when_obs_valid` | The total number of missing predictions (always 0 in a finished report). |
| `all_invariants_verified` | Always `true` in a finished report. |
| `leaderboards` | For each leaderboard: `num_entries`, `top_model_id`, `top_median_nse`, and the ranked `rows`. Rows are sorted by median NSE, then mean NSE. |
| `entries` | The full scorer output of each store, keyed by `<leaderboard_id>/<model_id>`. |

## Argument reference

### `benchmark-mip-metrics` and `run_metrics_benchmark_pipeline`

| Flag / argument | Default | Meaning |
| :--- | :--- | :--- |
| `--bucket-uri` / `bucket_uri` | `gs://camels-us-mip` | The root folder or bucket. Contributors pass a local folder, for example `./local_bucket`. |
| `--manifest-path` / `manifest_path` | none | A manifest file with the stores to score. Without it, all stores under `<root>/hydrographs/` are found automatically. |
| `--leaderboard-id` / `leaderboard_ids` | all 12 leaderboards | Only look in these leaderboards. Repeat the flag for more than one. Used only when stores are found automatically. |
| `entries` (Python only) | none | An explicit list of `(leaderboard_id, model_id)` pairs. It wins over `manifest_path` and `leaderboard_ids`. |
| `--output-json` / `output_json` | none | Save the full JSON report to this file. Missing parent folders are created. |
| `-h`, `--help` | — | Show the help text. |

## Maintainers only: the official bucket

Without `--bucket-uri`, the command reads the private bucket `gs://camels-us-mip`. That needs
Google Cloud credentials (see
[`hydromip/README.md`](../hydromip/README.md#zarr-io-and-cloud-credentials)). The checks require
all 531 basins, so a store that misses whole basins (an `Archive*` author archive) stops the run.
To skip such stores, list the stores you want in a manifest.

```bash
benchmark-mip-metrics --output-json ./outputs/bucket_report.json
```

## Tests

```bash
pytest benchmarks/tests
```

The tests write full-size stores (531 basins × 3,652 days) to a temporary folder. They check store
discovery, manifest loading, the command line, and that the rule checks reject bad results.
