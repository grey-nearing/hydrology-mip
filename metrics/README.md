# Hydrograph Metrics Package (`metrics/`)

The `metrics/` package scores daily streamflow predictions (`mm/day`) across the 531 CAMELS-US river basins over the 10-water-year evaluation period (`1989-10-01` to `1999-09-30`, `3,652` days). It reads model predictions and observed USGS river flow directly from Zarr stores in the `gs://camels-us-mip/` Google Cloud Storage bucket and computes all 14 hydrological metrics from Gauch et al. (2023) (`metrics/metrics.py`).

Every model hydrograph Zarr store must have shape `(531, 3652)` and provide finite daily predictions on every basin and day with valid USGS observations (`530` evaluable basins; basin `06775500` has zero valid USGS observations during `1989-10-01` to `1999-09-30`). Missing basins or missing predictions on valid observation days raise a `ValueError`.

## 1. Overview & Summary Table

| Component | Entry Point | Input Zarr Stores | Output |
| :--- | :--- | :--- | :--- |
| **Python API** | `calculate_model_leaderboard_metrics(model_id, leaderboard_id, bucket_uri)` | `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr` and `<bucket_uri>/observations/newman_eval_targets.zarr` | Python `dict` with 5 headline leaderboard metrics, all 14 Gauch et al. (2023) metric summaries, percentiles, 18-region summaries, and per-basin scores |
| **Single-Basin Metrics** | `calculate_all_metrics(obs, sim, resolution="1D", datetime_coord="date")` | 1D `xarray.DataArray` time series for observed (`obs`) and simulated (`sim`) flow | Python `dict` of the 14 Gauch et al. (2023) metrics for one basin |
| **Command Line (CLI)** | `mip-score-model` (or `python -m metrics.cli`) | Same Zarr stores in `gs://camels-us-mip/` | Prints JSON summary to terminal and optionally saves full JSON file |

### The 5 Headline Leaderboard Metrics & 14 Gauch et al. (2023) Metrics

Each basin is scored on days with valid USGS streamflow observations (`mm/day`, daily water volume divided by basin area):

1. **Median NSE (`median_nse`):** The middle (50th percentile) Nash-Sutcliffe Efficiency across evaluated basins. NSE compares model squared error to the variance of observed flow (`1.0` is a perfect match; `0.0` matches the long-term average flow).
2. **Mean NSE (`mean_nse`):** The average NSE across all evaluated basins.
3. **Median KGE (`median_kge`):** The middle Kling-Gupta Efficiency across evaluated basins. KGE combines correlation (`Pearson-r`), flow variability ratio (`Alpha-NSE`), and average flow ratio (`Beta-KGE`) (`1.0` is ideal).
4. **Median `|FHV|` (`median_abs_fhv`, `%`):** The middle unsigned percentage bias on the highest 2% of daily flows (flood peaks). `0.0%` means zero peak-flow volume bias.
5. **Catastrophic Failure Count (`nse_le_zero_count`):** The number of basins where `NSE <= 0.0` (where predicting the simple average observed flow beats the model).

In addition to `NSE`, `KGE`, and `FHV`, `metrics/metrics.py` computes `MSE` (mean squared error), `RMSE` (root mean squared error), `Alpha-NSE` (variability ratio), `Pearson-r` (correlation), `Beta-KGE` (mean ratio), `Beta-NSE` (normalized mean bias), `FMS` (flow duration curve mid-slope bias, `%`), `FLV` (low-flow bias on the lowest 30% of flows, `%`), `Peak-Timing` (mean timing error in days across flood peaks), `Missed-Peaks` (fraction of observed peaks above the 80th percentile missed within 1 day), and `Peak-MAPE` (mean absolute percentage error on observed peak flows, `%`).

---

## 2. Prerequisites & Credentials

Activate the Python environment and authenticate with Google Cloud Storage so Python can read `gs://camels-us-mip/`:

```bash
conda activate googlehydrology
gcloud auth application-default login
```

---

## 3. Quick Start Examples

### Score a Model from the Command Line

```bash
python -m metrics.cli \
  --model-id t4_06 \
  --leaderboard-id gauged__multi_camels \
  --bucket-uri gs://camels-us-mip \
  --output-json t4_06_metrics.json
```

### Score a Model in Python

```python
from metrics import calculate_model_leaderboard_metrics

result = calculate_model_leaderboard_metrics(
    model_id="t4_06",
    leaderboard_id="gauged__multi_camels",
    bucket_uri="gs://camels-us-mip",
)

print("Median NSE:", result["median_nse"])
print("Mean NSE:", result["mean_nse"])
print("Median KGE:", result["median_kge"])
print("Median |FHV| (%):", result["median_abs_fhv"])
print("Basins with NSE <= 0:", result["nse_le_zero_count"])
print("14 Gauch et al. (2023) summaries:", result["gauch_2023_metrics"].keys())
```

---

## 4. Arguments Reference

### Python API (`calculate_model_leaderboard_metrics`) and CLI (`mip-score-model`)

| Argument / Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_id` / `--model-id` | `str` | *(Required)* | Model entry ID matching `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr` (for example, `t4_06`). |
| `leaderboard_id` / `--leaderboard-id` | `str` | *(Required)* | One of the 12 leaderboard IDs: `gauged__single_camels`, `gauged__multi_camels`, `gauged__lagged_streamflow`, `gauged__freeform`, `pub__single_camels`, `pub__multi_camels`, `pub__lagged_streamflow`, `pub__freeform`, `pur__single_camels`, `pur__multi_camels`, `pur__lagged_streamflow`, or `pur__freeform`. |
| `bucket_uri` / `--bucket-uri` | `str` | `"gs://camels-us-mip"` | Root URI of the cloud bucket or local Zarr directory containing `observations/` and `hydrographs/`. |
| `--output-json` *(CLI only)* | `str` | `None` | Optional file path where the full JSON output (including all 531 per-basin scores) is saved. |
