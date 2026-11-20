---
name: repo-organization
description: >-
  Repository architecture, top-level package layout, centralized model runner
  design (models/run_model.py), standalone cloud-only metrics package
  (metrics/), ERA experiment notebooks (era_experiments/), leaderboard website
  frontend (frontend/), and packaging/CI synchronization for camels-us-mip. Use
  whenever adding, moving, refactoring, or reviewing modules, model scripts,
  ERA notebooks, or CLI entry points.
---

# Repository Organization (`camels-us-mip`)

This skill defines the package layout, directory boundaries, centralized execution rules, and packaging conventions for the `camels-us-mip` repository.

---

## 1. Top-Level Repository Layout

The repository is organized into self-contained top-level directories (`splits/`, `metrics/`, `models/`, `era_experiments/`, `frontend/`, `leaderboards/`, `benchmarks/`, `skills/`):

```text
camels-us-mip/
├── .github/workflows/             # All GitHub Actions CI workflows (root-only)
├── skills/                        # Project-level AI agent skills
│   ├── README.md
│   ├── repo-organization.md
│   ├── algorithm-rules-and-norms.md
│   ├── model-compliance-and-leaderboard-norms.md
│   ├── gcs-bucket-organization.md
│   ├── data-structures.md
│   ├── testing.md
│   ├── documentation.md
│   ├── pr-review.md
│   ├── benchmarking.md
│   └── community-contribution.md
├── splits/                        # Canonical 531-basin list, 12-fold PUB splits, and 18-fold HUC-02 PUR splits
│   ├── README.md
│   ├── basins_531.txt
│   ├── basins_531.json
│   ├── basins_447_kratzert2019b.txt
│   ├── basins_447_kratzert2019b.json
│   ├── splits_and_regimes.json
│   ├── pub_12fold_random_splits.json
│   ├── pub_12fold/                # fold_00..11_{train,test}.txt
│   ├── pur_18fold_huc02_splits.json
│   └── pur_18fold/                # huc_01..18_{train,test}.txt
├── metrics/                       # Standalone metrics calculation pipeline (cloud Zarr only, zero fallbacks)
│   ├── __init__.py
│   ├── evaluator.py               # Single API: calculate_model_leaderboard_metrics(model_id, leaderboard_id, ...)
│   ├── cli.py                     # CLI entrypoint (mip-score-model)
│   ├── README.md
│   └── tests/                     # Co-located unit & data-integrity regression tests
├── models/                        # All reproduced literature & study models + central deployment script
│   ├── __init__.py
│   ├── run_model.py               # Single central script for running Gauged, 12-fold PUB, and 18-fold PUR models
│   ├── zarr_writer.py             # Standardized Zarr hydrograph builder & GCS uploader
│   ├── README.md
│   ├── tests/                     # Unit & integration tests for central runner & Zarr standardization
│   ├── kratzert_lstm/             # NeuralHydrology CudaLSTM code & fold runner
│   ├── kratzert_ealstm/           # Entity-Aware LSTM code & fold runner
│   ├── hoedt_mclstm/              # Mass-Conserving LSTM code & fold runner
│   ├── rasiya_koya_tft/           # Temporal Fusion Transformer code & fold runner
│   ├── gauch_xgboost/             # XGBoost code & fold runner
│   ├── das_tide/                  # TiDE code & fold runner
│   ├── wang_s4dft/                # S4D & S4D-FT code & fold runner
│   ├── poudel_mlp_hbv/            # Poudel & Steinschneider (2026) MLP + 16HBV code & fold runner
│   ├── hydrodiffusion/            # HydroDiffusion code & fold runner
│   ├── feng_dhbv10/               # dHBV 1.0 (hydroDL) code & fold runner
│   ├── song_dhbv11p/              # dHBV 1.1p (hydroDL) code & fold runner
│   ├── newman_sacsma/             # SAC-SMA + Snow-17 DDS calibration & ensemble code & fold runner
│   ├── era_gauged_minimal_guidance/ # ERA Gauged Minimal Guidance (EntityAwareLSTM) code & fold runner
│   └── era_pur_minimal_guidance/  # ERA PUR Minimal Guidance (DualFiLMLSTM) code & fold runner
├── era_experiments/               # 6 canonical 256-iteration ERA hillclimbing prompt notebooks
│   ├── era_gauged_minimal_guidance.ipynb
│   ├── era_gauged_physics_informed.ipynb
│   ├── era_pub_minimal_guidance.ipynb
│   ├── era_pub_physics_informed.ipynb
│   ├── era_pur_minimal_guidance.ipynb
│   └── era_pur_physics_informed.ipynb
├── frontend/                      # Leaderboard web portal (hydromip.org) UI & FastAPI backend
│   ├── __init__.py
│   ├── README.md
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── backend/                   # FastAPI application serving leaderboards & live Zarr scoring
│   ├── static/                    # HTML/CSS/JS frontend assets
│   └── tests/                     # Co-located backend & UI tests
├── leaderboards/                  # Leaderboard JSON tables & model family registry
│   ├── contributions_index.json
│   ├── table_2_registry.json
│   └── tables_3_to_14.json
├── benchmarks/                    # Canonical benchmark suite verifying metrics/ and models/
│   ├── __init__.py
│   ├── README.md
│   ├── metrics_pipeline.py
│   └── tests/
├── LICENSE_ASSESSMENT.md          # Comprehensive license audit for all model repos and hydrograph archives
├── pyproject.toml                 # Ruff linter and Pytest configuration
├── setup.py                       # Package installation and console_scripts entry points
└── README.md                      # Top-level repository guide
```

---

## 2. Package Rules & Architectural Boundaries

1. **No Tracked Hydrograph Binary Blobs in Git:**
   - Do **not** commit `.npz`, `.npy`, or `.zarr` hydrograph outputs into the GitHub repository (`hydrographs/` and `protocol/newman_eval_targets.npz` must not live in Git).
   - All daily model hydrographs (`(531, 3652)`) and the canonical evaluation target store (`newman_eval_targets.zarr`) live in the private Google Cloud Storage bucket **`gs://camels-us-mip/`**.
2. **Standalone `metrics/` Package With a Single Public API:**
   - `metrics/` contains the standalone scoring pipeline (`metrics/evaluator.py`).
   - It exposes a **single public API function** (`calculate_model_leaderboard_metrics(model_id: str, leaderboard_id: str, bucket_uri: str = "gs://camels-us-mip") -> dict`) and CLI (`mip-score-model`) that computes the metrics for **one model and one leaderboard** directly from `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` and `gs://camels-us-mip/observations/newman_eval_targets.zarr`.
   - Zero fallback paths, zero `try`/`except` blocks, and zero local file fallbacks.
3. **Centralized Model Execution in `models/run_model.py`:**
   - Every reproduced literature model and study model lives in its own subdirectory `models/<model_name>/`, which includes the original upstream model code and a model-specific `runner.py` (exposing a standard interface `run_fold(train_basins, test_basins, train_dates, eval_dates, forcings, data_dir, **kwargs) -> np.ndarray`).
   - Individual model subdirectories must **never** load split files, choose training/evaluation time periods, or assign basins on their own.
   - `models/run_model.py` is the **only** entry point for running any model across `gauged`, `pub_12fold`, or `pur_18huc02`: it loads the canonical splits from `splits/` (or `gs://camels-us-mip/splits/`), orchestrates the cross-validation loop or parallel fold execution, collects predictions, converts them to a standardized Zarr store via `models/zarr_writer.py`, and writes them directly to `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`.
4. **No Root `scripts/` or `tools/` Folders:**
   - Never create top-level `scripts/`, `tools/`, or `scratch/` folders in the repository. Any package-specific helper scripts belong inside `<package>/tools/`.
5. **Keep Packaging & CI Synchronized:**
   - Whenever adding or modifying packages, CLI entry points, or test suites, keep `setup.py`, `pyproject.toml` (`[tool.pytest.ini_options] testpaths`), and `.github/workflows/` synchronized.
