# CAMELS-US Model Intercomparison Project (`camels-us-mip`)

`camels-us-mip` is the repository for the CAMELS-US Model Intercomparison Project (`hydrologymip.org`). It provides:

1. **Canonical Basin & Fold Splits ([`splits/`](./splits/README.md)):** The 531 Newman et al. (2015, 2017) USGS stream gauges, the 12-fold random Predictions in Ungauged Basins (`pub_12fold`) splits, and the 18-fold Leave-One-HUC02-Out Predictions in Ungauged Regions (`pur_18huc02`) splits, mirrored to `gs://camels-us-mip/splits/`.
2. **Cloud-Only Zarr Hydrograph Metrics Pipeline ([`metrics/`](./metrics/README.md)):** Standalone evaluation package with a single public API (`calculate_model_leaderboard_metrics`) that reads model hydrographs (`gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`) and USGS observations (`gs://camels-us-mip/observations/newman_eval_targets.zarr`) directly from Google Cloud Storage and computes all 14 hydrological metrics from Gauch et al. (2023) (*"In Defense of Metrics"*, [`metrics/metrics.py`](./metrics/metrics.py)).
3. **Centralized Literature & Study Model Runner ([`models/`](./models/README.md)):** A single central deployment script ([`models/run_model.py`](./models/run_model.py)) that loads the canonical splits and time windows (`1999-10-01` to `2008-09-30` training; `1989-10-01` to `1999-09-30` testing), orchestrates cross-validation across all 15 model subdirectories (`models/<model_name>/`), standardizes `(531, 3652)` outputs into consolidated Zarr stores ([`models/zarr_writer.py`](./models/zarr_writer.py)), and writes them to `gs://camels-us-mip/`.
4. **ERA Hillclimbing Prompt Notebooks (`era_experiments/`):** Six 256-iteration ERA prompt notebooks (`era_gauged_minimal_guidance.ipynb`, `era_gauged_physics_informed.ipynb`, `era_pub_minimal_guidance.ipynb`, `era_pub_physics_informed.ipynb`, `era_pur_minimal_guidance.ipynb`, `era_pur_physics_informed.ipynb`) initialized from a Kratzert-style multi-forcing `CudaLSTM` (3 CAMELS forcing products + 27 static catchment attributes) that load the canonical split files and hillclimb on stitched out-of-sample hydrographs strictly inside the Newman training period (`2003-10-01` to `2008-09-30` training, `1999-10-01` to `2003-09-30` evaluation) without touching the 1989–1999 Newman test decade.
5. **Leaderboard Website & FastAPI Backend ([`frontend/`](./frontend/README.md)):** Full web application for `hydrologymip.org` serving the $3 \times 4 = 12$ leaderboards (`Tables 3–14`), the 28-model family registry (`Table 2`), and interactive Zarr hydrograph scoring.
6. **End-to-End Verification Benchmarks ([`benchmarks/`](./benchmarks/README.md)), Codebase Management Skills ([`skills/`](./skills/README.md)), and Licensing Audit ([`LICENSE_ASSESSMENT.md`](./LICENSE_ASSESSMENT.md)).**

---

## Repository Structure

```text
camels-us-mip/
├── README.md                  # Repository overview and quickstart
├── LICENSE_ASSESSMENT.md      # Audit of code and hydrograph archive licenses
├── pyproject.toml             # Package configuration, CLI entrypoints, and pytest/ruff config
├── setup.py                   # Editable installation shim
├── skills/                    # Codebase management & data-integrity skills (start at skills/README.md)
├── splits/                    # Canonical 531-basin list, 12-fold PUB splits, and 18-fold HUC-02 PUR splits
├── metrics/                   # Standalone cloud-only Zarr metrics pipeline + Gauch et al. (2023) metrics.py
├── models/                    # Central deployment script (run_model.py) + 15 per-model subdirectories
├── era_experiments/           # 6 canonical 256-iteration ERA prompt notebooks
├── leaderboards/              # Leaderboard JSON tables, submission updater, and dynamic figure generator
├── figures/                   # Manuscript figures generated directly from leaderboards/*.json on each commit
├── frontend/                  # hydrologymip.org web UI and FastAPI backend
└── benchmarks/                # End-to-end cloud Zarr metrics benchmark suite
```

---

## Quickstart

### 1. Install and Authenticate with `gs://camels-us-mip/`
```bash
pip install -e ".[dev]"
export CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=false
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/.config/gcloud/application_default_credentials.json"
```

### 2. Calculate Metrics for One Model and One Leaderboard from `gs://camels-us-mip/`
```bash
mip-score-model \
  --model-id t4_06 \
  --leaderboard-id gauged__multi_camels \
  --bucket-uri gs://camels-us-mip
```

Or in Python:
```python
from metrics import calculate_model_leaderboard_metrics

summary = calculate_model_leaderboard_metrics(
    model_id="t4_06",
    leaderboard_id="gauged__multi_camels",
    bucket_uri="gs://camels-us-mip",
)
print(summary["median_nse"], summary["nse_le_zero_count"])
```

### 3. Train and Evaluate a Model via the Central Deployment Script
```bash
mip-run-model \
  --model-id kratzert_lstm \
  --spatial-regime pub_12fold \
  --forcing-regime multi_forcing \
  --data-dir /path/to/camels_us \
  --output-zarr gs://camels-us-mip/hydrographs/pub__multi_camels/kratzert_lstm_pub_3f.zarr
```

### 4. Run the Full Test Suite
```bash
pytest
```
