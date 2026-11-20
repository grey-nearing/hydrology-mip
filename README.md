# HydroMIP (`camels-us-mip`) — Multi-MIP Hydrological Model Intercomparison Platform

This repository hosts the **HydroMIP** (`hydromip.org`) multi-MIP architecture—starting with the canonical **CAMELS-US Daily Streamflow MIP (`mips/camels_us_daily/`)** and engineered so future MIPs (global Caravan nowcasting, probabilistic streamflow, soil moisture, 2D flood inundation, real-time operational, and weather-driven forecasting) plug in cleanly under `mips/<mip_id>/` without duplicating models, metrics, or web portal code.

---

## Architecture Overview

1. **Shared Multi-MIP Core ([`hydromip/`](./hydromip/)):** MIP-agnostic protocol schema & auto-discovery (`mips/*/protocol.json`), canonical path resolution (`hydromip/paths.py`), generic anti-leakage CV hygiene verification (`hydromip/splits.py`), pluggable metric-suite registry (`hydromip/metrics/`), and shared cloud Zarr I/O (`hydromip/evaluation/`).
2. **Self-Contained MIP Registry ([`mips/`](./mips/README.md)):** Every benchmark lives under `mips/<mip_id>/`. Currently active:
   - **[`mips/camels_us_daily/`](./mips/camels_us_daily/README.md):** The 531-basin Newman et al. (2015, 2017) / Kratzert et al. (2019, 2021) CAMELS-US Daily Streamflow MIP (`protocol.json`, `splits/`, `leaderboards/`, `figures/`, `era_experiments/`, `data/`, and `adapter.py`).
3. **Shared Model Zoo & Centralized Runner ([`models/`](./models/README.md)):** Machine-readable capability catalog ([`models/catalog.json`](./models/catalog.json)), centralized CV runner ([`models/run_model.py`](./models/run_model.py), supporting `--mip <mip_id>`), standardized Zarr writer ([`models/zarr_writer.py`](./models/zarr_writer.py)), and all 20 verified model subdirectories (`models/<model_name>/`, comprising 12 `MIP Re-run`, 7 `ERA Contribution`, and 1 zero-parameter `Naive Lag-1 Persistence` lower benchmark implementations).
4. **Cloud-Only Zarr Metrics Pipeline ([`metrics/`](./metrics/README.md)):** Standalone evaluation package exposing `calculate_model_leaderboard_metrics` and `mip-score-model`, backed by `hydromip.metrics.deterministic_daily` (the 12 Gauch et al., 2023 metrics).
5. **Leaderboard & Figure Tooling ([`leaderboards/`](./leaderboards/)):** Pure Python leaderboard synchronization, ranking, and figure-generation tooling (`mip-update-leaderboard`, `mip-generate-figures`) operating on `mips/<mip_id>/leaderboards/` and `mips/<mip_id>/figures/`.
6. **Multi-MIP Web Portal ([`frontend/`](./frontend/README.md)):** FastAPI backend and static UI for `hydromip.org`, auto-discovering registered MIPs via `/api/mips` while preserving full backwards-compatible CAMELS-US endpoints (`/api/leaderboards`, `/api/protocol`, `/api/splits/*`, `/api/figures/*`, `/api/score`, `/api/submit`).
7. **Verification Benchmarks ([`benchmarks/`](./benchmarks/README.md)), Agent Skills ([`skills/`](./skills/README.md)), and Licensing ([`LICENSE`](./LICENSE), [`NOTICE`](./NOTICE), [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md), [`DATA_LICENSE.md`](./DATA_LICENSE.md), [`LICENSE_ASSESSMENT.md`](./LICENSE_ASSESSMENT.md)).**

---

## Repository Structure

```text
camels-us-mip/
├── README.md                              # Multi-MIP platform overview and quickstart
├── LICENSE / NOTICE / DATA_LICENSE.md     # Apache-2.0 (code) and CC-BY-4.0 (first-party data)
├── THIRD_PARTY_NOTICES.md / CITATION.cff  # Third-party attributions & citation metadata
├── LICENSE_ASSESSMENT.md                  # Audit of code and hydrograph archive licenses
├── pyproject.toml / setup.py              # Package configuration, CLI entrypoints, pytest/ruff config
├── .github/workflows/                     # CI workflows for leaderboard sync & figure regeneration
├── hydromip/                              # Shared MIP-agnostic core (protocol, paths, splits, pluggable metrics, Zarr I/O)
├── mips/                                  # Self-contained per-MIP directories (purely additive for new MIPs)
│   ├── README.md                          # Guide for onboarding future MIPs
│   └── camels_us_daily/                   # Active MIP: CAMELS-US Daily Streamflow (v1.0.0)
│       ├── protocol.json                  # Declarative MIP protocol spec
│       ├── adapter.py                     # Verified split/data adapter
│       ├── splits/                        # 531-basin list, 12-fold PUB, 18-fold HUC-02 PUR
│       ├── leaderboards/                  # 12 leaderboard JSONs, Table 2 registry, Stage-1 appendix
│       ├── figures/                       # Manuscript & portal figures + figures_manifest.json
│       ├── era_experiments/               # 14 canonical 256-iteration ERA notebooks (7 tracks)
│       └── data/                          # Raw/compiled CAMELS-US dataset docs (binaries gitignored)
├── models/                                # Shared Model Zoo (catalog.json, run_model.py, zarr_writer.py, 19 model dirs)
├── metrics/                               # Cloud-only Zarr metrics scoring pipeline & CLI
├── leaderboards/                          # Reusable leaderboard ranking & figure-generation code
├── frontend/                              # Multi-MIP aware hydromip.org web UI & FastAPI backend
├── benchmarks/                            # End-to-end cloud Zarr verification suite
└── skills/                                # Codebase management & data-integrity skills
```

---

## Quickstart

### 1. Install the Package
```bash
pip install -e ".[dev]"
```

For authenticated writes to `gs://camels-us-mip/`, configure Google Cloud Application Default Credentials (`gcloud auth application-default login`).

### 2. Discover Registered MIPs in Python
```python
from hydromip import discover_mips, get_mip

print(list(discover_mips().keys()))  # ['camels_us_daily']
proto = get_mip("camels_us_daily")
print(proto.name, proto.num_entities, len(proto.leaderboard_ids))
```

### 3. Calculate Metrics for One Model and One Leaderboard from `gs://camels-us-mip/`
```bash
mip-score-model \
  --mip camels_us_daily \
  --model-id t4_06 \
  --leaderboard-id gauged__multi_camels \
  --bucket-uri gs://camels-us-mip
```

### 4. Train and Evaluate a Model via the Central Deployment Script
```bash
mip-run-model \
  --mip camels_us_daily \
  --model kratzert_lstm \
  --spatial-regime pub_12fold \
  --forcings daymet maurer nldas \
  --output-zarr gs://camels-us-mip/hydrographs/pub__multi_camels/kratzert_lstm_pub_3f.zarr
```

### 5. Run the Full Test Suite
```bash
pytest
```

---

## Citation & Archive

* **CAMELS-US Master Release Archive (`v1.0.0`):** `https://doi.org/10.5281/zenodo.24383448`
* **Paper:** Nearing, G. S., Klotz, D., Ebel, P., & Gauch, M. (2026). *HydroMIP: A Rainfall-Runoff Model Intercomparison Project*. *Hydrology and Earth System Sciences* (submitted). See [`CITATION.cff`](./CITATION.cff).

---

## License

* **Software & Source Code (`Apache-2.0`):** Licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE), [`NOTICE`](./NOTICE), and [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
* **Datasets, Splits, Leaderboards, & Hydrograph Zarr Stores (`CC-BY-4.0`):** All first-party splits (`mips/*/splits/`), leaderboard tables (`mips/*/leaderboards/`), figures (`mips/*/figures/`), and `MIP Re-run` / `Contribution` hydrograph Zarr stores in `gs://camels-us-mip/` are licensed under the Creative Commons Attribution 4.0 International License (`CC-BY-4.0`). External author-archived hydrographs retain their original upstream licenses; see [`DATA_LICENSE.md`](./DATA_LICENSE.md).
