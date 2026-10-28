# Leaderboard Website Frontend & Backend (`frontend/`)

`frontend/` runs the `hydrologymip.org` web application. It serves the 12 CAMELS-US stream-gauge leaderboards (Tables 3 to 14), the model-family registry (Table 2, counted from `table_2_registry.json`), the benchmark split protocol, and an interactive tool that scores uploaded daily streamflow predictions (`mm/day` over `1989-10-01` to `1999-09-30`, shape `531` basins by `3,652` days).

---

## 1. Overview & Summary Table

| Component | Entry Point / Route | Input | Output |
| :--- | :--- | :--- | :--- |
| **Web Server** | `uvicorn frontend.backend.app:app` | `leaderboards/*.json`, `splits/splits_and_regimes.json` | Web UI at `http://localhost:8080/` and JSON API at `/api/*` |
| **Health Check** | `GET /api/health` | None | Server status, leaderboard count (`12`), and model count (`32`) |
| **Protocol Metadata** | `GET /api/protocol` | None | 531 USGS 8-digit basin IDs, date windows, 12-fold PUB splits, and 18-fold HUC-02 PUR splits |
| **All Leaderboards** | `GET /api/leaderboards` | Optional query filters (`spatial_regime`, `forcing_regime`, `category`, `search`) | All 12 leaderboard tables and the registry model families |
| **Single Leaderboard** | `GET /api/leaderboards/{leaderboard_id}` | `leaderboard_id` (e.g., `gauged__single_camels`) | Ranked rows for one leaderboard table |
| **Model Registry** | `GET /api/registry` | None | Table 2 model families (`32` models) and contributed runs |
| **Score Hydrographs** | `POST /api/score` | Multipart file (`.zarr` zip, `.npz`, or `.npy` of shape `(531, 3652)` in `mm/day`) | 5 headline metrics, lower-tail percentiles, and 18-region HUC-02 breakdown |
| **Submit Model** | `POST /api/submit` | Multipart file (`(531, 3652)` in `mm/day`) + model metadata and URLs | Verified score and new row added to the selected leaderboard |

### Plain-Language Term Guide

- **Gauged:** Models trained and tested on the same 531 river basins during different time periods.
- **PUB (Prediction in Ungauged Basins):** 12-fold cross-validation where models are tested on randomly held-out river basins never seen during training.
- **PUR (Prediction in Ungauged Regions):** 18-fold cross-validation where models are tested on an entire held-out USGS 2-digit hydrologic region (`HUC-02` `01` through `18`).
- **NSE (Nash-Sutcliffe Efficiency):** Accuracy score comparing predictions to the observed average flow (`1.0` is perfect; `<= 0.0` is worse than predicting the constant mean).
- **KGE (Kling-Gupta Efficiency):** Accuracy score combining correlation, flow variability bias, and total volume bias (`1.0` is perfect).
- **|FHV| (%):** Unsigned percentage bias on the highest 2% of daily peak flows (`0%` is perfect).

---

## 2. Prerequisites & Credentials

1. Install Python dependencies in the `googlehydrology` Conda environment:
   ```bash
   pip install -r frontend/requirements.txt
   ```
2. Authenticate with Google Cloud Storage so `/api/score` and `/api/submit` can read observed discharge from `gs://camels-us-mip/observations/newman_eval_targets.zarr`:
   ```bash
   gcloud auth application-default login
   ```

---

## 3. Copy-Pasteable Quick Start Examples

### 3.1 Launch the Local Web Server

```bash
python -m uvicorn frontend.backend.app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` in your browser.

### 3.2 Query Leaderboards and Protocol from Python

```python
from fastapi.testclient import TestClient
from frontend.backend.app import app

client = TestClient(app)

# 1. Check server health
print(client.get("/api/health").json())

# 2. Fetch Table 3 (Gauged Single CAMELS Forcing)
table_3 = client.get("/api/leaderboards/gauged__single_camels").json()
print(table_3["title"], "Top model:", table_3["top_model"], "Median NSE:", table_3["top_nse"])
```

### 3.3 Deploy to Google Cloud Run

```bash
gcloud run deploy hydrologymip-web \
  --source . \
  --dockerfile frontend/Dockerfile \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### 3.4 Run the Test Suite

```bash
pytest frontend/tests
```

---

## 4. Complete Arguments & Configuration Reference

### 4.1 Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `CAMELS_MIP_DATA_DIR` | `str` (path) | Repository root | Directory containing `leaderboards/` (`tables_3_to_14.json`, `table_2_registry.json`, `contributions_index.json`) and `splits/` (`splits_and_regimes.json`). |
| `CAMELS_MIP_TARGETS_ZARR` | `str` (URI/path) | `gs://camels-us-mip/observations/newman_eval_targets.zarr` | Path or GCS URI to the observed streamflow Zarr dataset (`(531, 3652)` in `mm/day`). |
| `CAMELS_MIP_GCS_ROOT` | `str` (path) | Unset | Optional local directory mapping for `gs://camels-us-mip/` during offline testing. |
| `CAMELS_MIP_SUBMISSIONS_DIR` | `str` (path) | `<CAMELS_MIP_DATA_DIR or repository root>/submissions` | Writable root where `/api/submit` persists every accepted community hydrograph as a consolidated Zarr v2 store at `<root>/<leaderboard_id>/<entry_id>.zarr` (`qsim` plus `qsim_members` when supplied) **before** the leaderboard row is written; the row records that real path as `hydrograph_path`. If the directory cannot be created or written, `/api/submit` refuses the submission with HTTP 503. |
| `PORT` | `int` | `8080` | TCP port used inside the Docker container. |

### 4.2 HTTP API Endpoints & Parameters

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | None | Returns `status`, `n_leaderboards` (`12`), `n_families` (read from `table_2_registry.json`), `total_entries`, and `total_contributions`. |
| `/api/protocol` | `GET` | None | Returns the 531 USGS gauge IDs, training dates (`1999-10-01` to `2008-09-30`), evaluation dates (`1989-10-01` to `1999-09-30`, `3,652` days), 12 PUB folds, and 18 HUC-02 PUR folds. |
| `/api/leaderboards` | `GET` | `spatial_regime` (`str`, optional: `gauged`, `pub`, `pur`), `forcing_regime` (`str`, optional: `single_camels`, `multi_camels`, `lagged_streamflow`, `freeform`), `paradigm` (`str`, optional), `category` (`str`, optional), `search` (`str`, optional) | Returns all matching leaderboards and the registry model families. `updated_at` is the UTC modification time of the newest leaderboard JSON file; rows without a numeric `primary_ranking_nse` carry `null` and sort last. |
| `/api/leaderboards/{leaderboard_id}` | `GET` | `leaderboard_id` (`str`, required: one of the 12 canonical IDs) | Returns a single leaderboard table and its ranked model rows. |
| `/api/registry` | `GET` | None | Returns the model families (`table_2_registry.json`) and the contributions index (`contributions_index.json`). |
| `/api/entry/{entry_id}/basins` | `GET` | `entry_id` (`str`, required) | Returns the empirical 101-point per-basin NSE CDF (`np.quantile` over the entry's stored `per_basin_nse`) with `source: "per_basin_metrics"`, or `source: "unavailable"` with `nse_cdf_101: null` for literature-only rows. Reference curves (`ref_3f_lstm_cdf_101`, `ref_1f_daymet_cdf_101`) come from the real rows `t4_05` / `t3_02` and are `null` until those rows carry per-basin values. Nothing is synthesized. |
| `/api/score` | `POST` | `file` (`UploadFile`, required: zipped `.zarr`, `.npz`, or `.npy` with `qsim` of shape `(531, 3652)` in `mm/day`; Zarr stores may add `qsim_members` `(member, basin, date)` and the attributes `is_ensemble` / `ensemble_reduction`), `leaderboard_id` (`str`, default `gauged__single_camels`) | Scores exactly like `metrics.evaluator.calculate_model_leaderboard_metrics` (float64, no clipping, `obs_valid = ~isnan(qobs) & qobs >= 0`, evaluable basins `>= 30` valid days and `SST > 1e-12`). Rejects `Inf`, any `NaN` on a valid observation day (`pred_nan_when_obs_valid > 0`), and any `qsim` that is not the declared reduction of its members (or exactly one member for `is_ensemble=False`). Returns the 5 headline metrics, `ensemble_source`, per-member single-seed skill, regional HUC-02 scores, and the per-basin NSE vector. |
| `/api/submit` | `POST` | `file` (`UploadFile`, required), `leaderboard_id` (`str`, required), `model_name` (`str`, required), `forcing` (`str`, required), `github_url` (`str`, optional), `zenodo_doi` (`str`, optional), `citation` (`str`, optional) | Scores the uploaded `(531, 3652)` hydrographs, persists them under `CAMELS_MIP_SUBMISSIONS_DIR`, and saves the verified entry (single-seed rows: `ens_med_nse: null` / `"---"`; ensembles: `single_med_nse` = per-member mean with `"mean ± std"` display) to the target leaderboard. |
