# Benchmark Verification Suite (`benchmarks/`)

The `benchmarks/` package scores and checks all model streamflow predictions (`mm/day`) across the 12 CAMELS-US Model Intercomparison Project (MIP) leaderboards in `gs://camels-us-mip/`. It verifies that every model store has shape `(531, 3652)` over `1989-10-01` to `1999-09-30` (`3,652` days), has zero missing basins and zero missing predictions on days with valid USGS flow observations (`pred_nan_when_obs_valid == 0`), and reports all 5 headline metrics, all 12 Gauch et al. (2023) metric distributions, lower-tail skill percentiles, peak-flow error percentiles, and 18-region summaries.

## 1. Overview & Summary Table

| Component | Entry Point | Inputs | Output |
| :--- | :--- | :--- | :--- |
| **Python API** | `run_metrics_benchmark_pipeline(...)` | Zarr hydrograph stores in `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr` and observed targets in `<bucket_uri>/observations/newman_eval_targets.zarr` | Structured Python `dict` with ranked leaderboard tables, percentile distributions, and per-basin scores |
| **Command Line (CLI)** | `benchmark-mip-metrics` (or `python -m benchmarks.metrics_pipeline`) | Same Zarr stores (plus optional `--manifest-path` or `--leaderboard-id` filters) | Prints a Markdown summary table to the terminal and optionally writes a JSON report file |

---

## 2. Prerequisites & Credentials

Activate the Python environment and sign in to Google Cloud Storage so the benchmark runner can read `gs://camels-us-mip/`:

```bash
conda activate googlehydrology
gcloud auth application-default login
```

---

## 3. Quick Start Examples

### Score All Zarr Hydrographs from the Command Line

```bash
python -m benchmarks.metrics_pipeline \
  --bucket-uri gs://camels-us-mip \
  --output-json mip_benchmark_report.json
```

### Filter by Leaderboard or Manifest File

```bash
python -m benchmarks.metrics_pipeline \
  --bucket-uri gs://camels-us-mip \
  --leaderboard-id gauged__single_camels \
  --leaderboard-id gauged__multi_camels \
  --output-json gauged_report.json
```

### Run the Benchmark Pipeline in Python

```python
from benchmarks import (
    format_verification_markdown_table,
    run_metrics_benchmark_pipeline,
)

report = run_metrics_benchmark_pipeline(
    bucket_uri="gs://camels-us-mip",
    leaderboard_ids=["gauged__multi_camels"],
)

print("Total models scored:", report["total_evaluated_entries"])
print(format_verification_markdown_table(report))
```

---

## 4. Arguments Reference

### Python API (`run_metrics_benchmark_pipeline`) and CLI (`benchmark-mip-metrics`)

| Argument / Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bucket_uri` / `--bucket-uri` | `str` | `"gs://camels-us-mip"` | Root URI of the cloud bucket or local Zarr directory containing `observations/` and `hydrographs/`. |
| `manifest_path` / `--manifest-path` | `str` or `Path` | `None` | Optional path to a JSON manifest file listing the `(leaderboard_id, model_id)` entries to score. If omitted, all `.zarr` stores under `hydrographs/` are discovered automatically. |
| `leaderboard_ids` / `--leaderboard-id` | `list[str]` | `None` (all 12 leaderboards) | Optional subset of leaderboard IDs to scan (can be passed multiple times on the command line). |
| `entries` *(Python API only)* | `Sequence[tuple[str, str]]` | `None` | Optional explicit list of `(leaderboard_id, model_id)` pairs to score. |
| `output_json` / `--output-json` | `str` or `Path` | `None` | Optional path where the full JSON verification report is saved. |
