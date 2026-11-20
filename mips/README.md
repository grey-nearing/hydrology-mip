# Registered Model Intercomparison Projects (`mips/`)

Every benchmark in HydroMIP lives in its own self-contained subdirectory `mips/<mip_id>/`. Adding a new MIP is **purely additive**: create a new directory here with a valid `protocol.json`, canonical splits, leaderboard tables, and adapter—without modifying any existing MIP or duplicating models from `models/`.

---

## Active MIPs

| `mip_id` | Name | Task Type | Metric Suite | Entities × Eval Steps | Leaderboards | Status |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| [`camels_us_daily`](./camels_us_daily/README.md) | CAMELS-US Daily Streamflow MIP | `streamflow_daily_deterministic` | `gauch_2023_deterministic` | 531 basins × 3,652 days | 12 (`3 × 4`) | `active` (`v1.0.0`) |

---

## Per-MIP Directory Contract

```text
mips/<mip_id>/
├── protocol.json          # Required: declarative spec validated by hydromip.protocol
├── README.md              # Required: benchmark overview, data source DOIs, and protocol rules
├── adapter.py             # Required: binds dataset loader, splits, and CV hygiene to hydromip
├── splits/                # Required: canonical entity lists and CV fold JSON/TXT files
├── leaderboards/          # Required: per-leaderboard JSONs, model registry, contributions index
├── figures/               # Generated publication & dashboard figures + figures_manifest.json
├── data/                  # Dataset acquisition docs (raw/compiled binary data gitignored)
└── era_experiments/       # Optional: MIP-specific experiment or hillclimbing notebooks
```

---

## Onboarding Future MIPs

The `hydromip` core schema (`hydromip/protocol.py`) and metric-suite registry (`hydromip/metrics/registry.py`) are designed to support diverse hydrological benchmarks side by side:

| Future MIP Domain | `task_type` | `prediction_dims` Example | Metric Suite Example |
| :--- | :--- | :--- | :--- |
| Global Caravan daily nowcasting | `streamflow_daily_deterministic` | `["basin", "date"]` | `gauch_2023_deterministic` |
| Probabilistic daily streamflow | `streamflow_daily_probabilistic` | `["member", "basin", "date"]` | `probabilistic_crps` (CRPS, PIT, reliability, spread-skill) |
| Medium-range streamflow forecasting | `streamflow_forecast_deterministic` / `streamflow_forecast_probabilistic` | `["basin", "issue_date", "lead_time"]` | `forecast_leadtime` (lead-time NSE/CRPS decay, Brier, ROC) |
| Real-time operational streamflow | `streamflow_realtime` | `["basin", "issue_time", "lead_time"]` | `realtime_operational` (latency-aware skill, warning lead-time) |
| Soil moisture | `soil_moisture_daily` | `["station", "date"]` | `soil_moisture_daily` (ubRMSE, anomaly $r$, KGE, bias) |
| 2D flood inundation mapping | `flood_inundation_2d` | `["event", "y", "x"]` | `flood_inundation_2d` (CSI, IoU, F1, false-alarm ratio) |

### Checklist for Adding a New MIP

1. Create `mips/<new_mip_id>/protocol.json` conforming to `hydromip-protocol-v1`.
2. If the new MIP uses a new metric suite, add it under `hydromip/metrics/` and register it via `hydromip.metrics.register_metric_suite`.
3. Add canonical entity lists and CV fold files under `mips/<new_mip_id>/splits/` and verify zero leakage with `hydromip.splits.validate_cv_folds`.
4. Add `mips/<new_mip_id>/adapter.py` and initialize empty leaderboard JSONs under `mips/<new_mip_id>/leaderboards/`.
5. Declare compatible models in `models/catalog.json`.
6. Run `pytest hydromip/tests` — `hydromip.discover_mips()` and `frontend/` (`GET /api/mips`) pick up the new MIP automatically.
