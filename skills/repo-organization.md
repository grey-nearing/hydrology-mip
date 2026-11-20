---
name: repo-organization
description: >-
  Multi-MIP repository architecture, shared hydromip/ core package, per-MIP
  directory contract under mips/<mip_id>/, shared Model Zoo (models/),
  standalone cloud-only metrics package (metrics/), leaderboard tooling
  (leaderboards/), multi-MIP web portal (frontend/), and packaging/CI
  synchronization for HydroMIP (camels-us-mip). Use whenever adding a new MIP,
  moving, refactoring, or reviewing modules, model scripts, or CLI entry points.
---

# Multi-MIP Repository Organization (`HydroMIP` / `camels-us-mip`)

This skill defines the multi-MIP monorepo layout, package boundaries, centralized execution rules, and packaging conventions so new MIPs (global Caravan, probabilistic streamflow, soil moisture, 2D flood inundation, real-time, forecasting) can be added cleanly under `mips/<mip_id>/` without duplicating models, metrics, or website infrastructure.

---

## 1. Top-Level Repository Layout

```text
camels-us-mip/
├── .github/workflows/             # All GitHub Actions CI workflows (root-only)
├── hydromip/                      # Shared, MIP-agnostic core library (zero try/except)
│   ├── protocol.py                #   Declarative MipProtocol schema & auto-discovery (mips/*/protocol.json)
│   ├── paths.py                   #   Canonical repo & per-MIP path resolution
│   ├── splits.py                  #   Generic CV split loader & strict anti-leakage hygiene verifier
│   ├── metrics/                   #   Pluggable METRIC_SUITES registry + deterministic_daily.py (Gauch 2023)
│   ├── evaluation/                #   Shared GCS/local Zarr store I/O helpers
│   ├── leaderboards/              #   Per-MIP leaderboard/figure directory resolvers
│   └── tests/                     #   Core, CV-hygiene, and multi-MIP plugin extensibility tests
├── mips/                          # Self-contained per-MIP benchmark directories (purely additive!)
│   ├── README.md                  #   Plugin contract & onboarding checklist for new MIPs
│   └── camels_us_daily/           #   Active MIP: CAMELS-US Daily Streamflow (v1.0.0)
│       ├── protocol.json          #     Machine-readable MIP protocol spec
│       ├── adapter.py             #     Verified CAMELS-US split & data adapter
│       ├── splits/                #     Canonical 531/671/447 basin lists, 12-fold PUB, 18-fold HUC-02 PUR
│       ├── leaderboards/          #     12 leaderboard JSONs, tables_3_to_14.json, table_2_registry.json, Stage-1 appendix
│       ├── figures/               #     Manuscript & portal figures + figures_manifest.json
│       ├── era_experiments/       #     14 canonical 256-iteration ERA prompt & winner notebooks
│       └── data/                  #     Raw/compiled dataset docs (binary bundles gitignored)
├── models/                        # Shared Model Zoo (model code lives ONCE, runs across compatible MIPs)
│   ├── catalog.json               #   Machine-readable model capability manifest
│   ├── run_model.py               #   Central CV runner (`--mip <mip_id>`, default `camels_us_daily`)
│   ├── zarr_writer.py             #   Standardized Zarr hydrograph builder & GCS uploader
│   ├── camels_data.py             #   Shared CAMELS-US data loader
│   ├── tests/                     #   Unit & integration tests for central runner & Zarr standardization
│   └── <model_name>/              #   19 verified model subdirectories (unchanged top-level URLs)
├── metrics/                       # Standalone cloud-only Zarr metrics scoring pipeline & CLI
├── leaderboards/                  # Pure Python leaderboard ranking & figure-generation tooling (no data JSONs)
├── frontend/                      # Multi-MIP aware web portal (hydromip.org) UI & FastAPI backend
├── benchmarks/                    # Canonical cloud Zarr benchmark verification suite
├── skills/                        # Project-level AI agent skills
├── LICENSE_ASSESSMENT.md          # Comprehensive license audit
├── pyproject.toml / setup.py      # Packaging, console_scripts, ruff, and pytest config
└── README.md                      # Top-level platform overview
```

---

## 2. Package Rules & Architectural Boundaries

1. **Strict Separation of Shared Code vs. Per-MIP Data (`mips/<mip_id>/`):**
   - Benchmark-specific splits, leaderboard JSONs, generated figures, dataset bundles, and MIP-specific experiment notebooks **must** live inside `mips/<mip_id>/`.
   - Shared Python packages (`hydromip/`, `models/`, `metrics/`, `leaderboards/`, `frontend/`, `benchmarks/`) must never store per-MIP JSON tables or PNG/PDF figures inside their package directories.
   - Adding a new MIP must be **purely additive**: create `mips/<new_mip_id>/` with a valid `protocol.json` (plus a new metric module in `hydromip/metrics/` if needed) without editing existing MIPs.
2. **No Tracked Hydrograph Binary Blobs in Git:**
   - Never commit `.npz`, `.npy`, or `.zarr` hydrographs into Git. Standardized daily hydrographs and observation targets live in cloud Zarr storage (`gs://camels-us-mip/` for `camels_us_daily`).
3. **Standalone `metrics/` & Pluggable `hydromip.metrics` Registry:**
   - `hydromip/metrics/deterministic_daily.py` is the canonical source of the 12 Gauch et al. (2023) metrics (re-exported identically by `metrics/metrics.py`).
   - Future MIP metric suites (probabilistic CRPS/PIT, 2D flood inundation CSI/IoU, soil moisture ubRMSE, forecast lead-time skill) live in `hydromip/metrics/` and register via `register_metric_suite()`.
   - Zero fallback paths and zero `try`/`except` blocks across `hydromip/`, `mips/*/adapter.py`, `metrics/`, `models/`, `leaderboards/`, and `benchmarks/`.
4. **Centralized Model Execution in `models/run_model.py`:**
   - All model architectures live once under `models/<model_name>/` (preserving public GitHub links) and declare compatible tasks/MIPs in `models/catalog.json`.
   - Individual model runners never load split files or hardcode dates; `models/run_model.py` loads canonical splits from `mips/<mip_id>/splits/` via `hydromip.paths`.
5. **No Root `scripts/` or `tools/` Folders:**
   - Keep root clean; package-specific helpers belong inside `<package>/tools/`.
6. **Keep Packaging, Portal, & CI Synchronized:**
   - Keep `setup.py`, `pyproject.toml` (`packages.find` and `pytest.ini_options.testpaths`), `frontend/Dockerfile`, and `.github/workflows/` synchronized whenever packages or MIPs change.
