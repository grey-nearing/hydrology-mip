# CAMELS-US MIP Manuscript Figures (`figures/`)

All manuscript figures in this directory are generated **directly from the live leaderboard JSON files** in [`../leaderboards/`](../leaderboards/) via [`../leaderboards/generate_figures.py`](../leaderboards/generate_figures.py) (or the thin wrapper [`figures/generate_figures.py`](generate_figures.py)) and are automatically regenerated on every community submission and leaderboard update.

## Generated Figures

| File | Manuscript Reference | Source JSON Files |
| :--- | :--- | :--- |
| `figure_1_water_year_coverage.png` / `.pdf` | **Figure 1.** Streamflow Observation Completeness by Water Year Across the 531 CAMELS-US Benchmark Basins (Water Years 1981–2014). | `leaderboards/<leaderboard_id>.json` → `protocol.water_year_completeness` (mirrored in `leaderboards/tables_3_to_14.json`) |
| `figure_2_leaderboard_progress_all_regimes.png` / `.pdf` | **Figure 2.** 3-subpanel CAMELS-531 Leaderboard Progress Over Time (2015–2026) across (a) Gauged Simulation, (b) 12-Fold Random PUB, and (c) 18-Fold HUC-02 PUR. | `leaderboards/gauged__*.json`, `leaderboards/pub__*.json`, `leaderboards/pur__*.json` |
| `figure_2a_leaderboard_progress_gauged.png` | **Figure 2a.** Standalone Gauged CAMELS-531 Leaderboard Progress Over Time (Tables 3–6). | `leaderboards/gauged__*.json` |
| `figure_2b_leaderboard_progress_pub.png` | **Figure 2b.** Standalone Random Ungauged (12-Fold PUB) Leaderboard Progress Over Time (Tables 7–10). | `leaderboards/pub__*.json` |
| `figure_2c_leaderboard_progress_pur.png` | **Figure 2c.** Standalone Regionally Ungauged (18-Fold HUC-02 PUR) Leaderboard Progress Over Time (Tables 11–14). | `leaderboards/pur__*.json` |
| `figures_manifest.json` | Machine-readable manifest: plotted entry counts, excluded rows (with reasons), per-panel axis limits, active frontier leaders per regime (with source category and provenance), and output paths. | `leaderboards/*.json` |

## Figure 1: Water-Year Completeness Comes Only From the Protocol Block

Figure 1 is drawn **exclusively** from `protocol["water_year_completeness"]` in the leaderboard JSON protocol block (keys `n_basins`, `water_years`, `basins_with_100pct_complete`, `basins_with_any_valid_data`, `mean_daily_coverage_fraction`, plus the optional `newman_eval_water_years` / `newman_train_water_years` shading windows). There is **no hard-coded fallback series** in the generator: a missing or malformed block raises `ValueError` with a message naming the offending key.

The block can be recomputed from the canonical observation store (`gs://camels-us-mip/observations/newman_eval_targets.zarr`, variables `qobs_all` of shape `(531, n_days)` and `date_all`) with:

```python
from leaderboards.generate_figures import compute_water_year_completeness

block = compute_water_year_completeness(qobs_all, date_all)
# -> {"n_basins": 531, "water_years": [...], "basins_with_100pct_complete": [...],
#     "basins_with_any_valid_data": [...], "mean_daily_coverage_fraction": [...],
#     "newman_eval_water_years": [1990, 1999], "newman_train_water_years": [2000, 2008],
#     "partial_water_years_excluded": [...]}
```

Definitions (water year `Y` = 1 Oct `Y-1` – 30 Sep `Y`): a basin is *100 % complete* in a water year when it has no missing (NaN / non-finite) day in it; it has *any valid data* when it has at least one valid day; `mean_daily_coverage_fraction` is the mean over basins of the fraction of valid days. Water years whose full calendar is not covered by `date_all` are dropped by default (`include_partial_water_years=False`) because their completeness cannot be assessed.

## Figure 2: Every Scored Row Is Plotted by Original Publication Year

- **No cropping.** Axis limits are derived from the data: the y-range of each panel is the union of a per-panel minimum extent (gauged 0.42–0.91, PUB 0.40–0.80, PUR 0.30–0.76) and `[min NSE − 0.02, max NSE + 0.02]` of the plotted rows; the x-range likewise expands beyond 2015–2026 when needed. Every leaderboard row with a finite primary NSE is therefore drawn, and the generator raises if it ever draws fewer points than it was given.
- **Explicit exclusions.** Rows without a usable NSE (`primary_ranking_nse` / `ens_med_nse` / `single_med_nse` all null, non-numeric, non-finite, or a legacy `-999` sentinel) are skipped explicitly and listed in `figures_manifest.json → excluded_rows[<spatial_regime>]` with `entry_id`, `leaderboard_id`, `model_name` and a `reason`. `panel_entry_counts` are the numbers of points actually drawn.
- **Provenance and publication year.** `Literature`, `Archive`, and `Archive*` results are plotted at the year of the cited paper with **filled** markers. `MIP Re-run` results are also plotted at the year of the original published paper with **hollow** markers (so credit goes to the original authors while transparently indicating that the run was re-executed on standard MIP splits). `Contribution` models from this study or subsequent community submissions are plotted at the year of their submission or `2026` with **hollow** markers. A row with no 1990–2035 year in `publication_year`, `model_name` or `citation` makes the generator raise `ValueError` naming the entry; there is no silent default year.
- The step-line frontier of each input-forcing regime is still the running maximum of the median NSE over publication years (rows containing "oracle" are excluded from the frontier but still plotted).

## Automatic Triggering on Community Commits & Submissions

1. **In the Python Submission Pipeline (`leaderboards/update_leaderboard.py`)**:
   Calling `upsert_leaderboard_entry(...)`, `trigger_submission_from_gcs(...)`, or `recalculate_all_self_run_models(...)` automatically calls `generate_all_paper_figures(...)` so that `figures/*.png` and `figures/*.pdf` are updated immediately whenever a new model is scored or added.
2. **In Git / GitHub Actions (`.github/workflows/update_leaderboards_and_figures.yml` & `leaderboards/on_community_commit.sh`)**:
   Every push or pull request touching `leaderboards/**` or `submissions/**` (as well as the local git `pre-commit` hook installed via `bash leaderboards/on_community_commit.sh --install-hook`) runs `python -m leaderboards.generate_figures` and commits the updated figures directly in `figures/`.
   The script uses `${PYTHON_BIN:-$(command -v python3)}` as its interpreter (no machine-specific paths) and fails with a clear message if that interpreter cannot `import leaderboards`; set `PYTHON_BIN=/path/to/python` to pick an environment with the project dependencies installed.

## Manual CLI Usage

```bash
python -m leaderboards.generate_figures            # regenerate into figures/
python -m leaderboards.generate_figures --leaderboards-dir leaderboards --output-dir /tmp/mip_figures
# or:
mip-generate-figures
```

## Tests

`leaderboards/tests/test_generate_figures.py` regenerates all figures from a copy of the committed leaderboards into a temporary directory and asserts that `panel_entry_counts` equals the number of rows with a finite NSE per spatial regime, that `excluded_rows` lists exactly the rows without one, that every plotted row lies inside the recorded axis limits, that no MIP-produced frontier leader is dated before 2026, that the hard-coded Figure-1 series is gone (a protocol without `water_year_completeness` raises), and it unit-tests `compute_water_year_completeness`, `extract_publication_year`, `format_leader_label` and `resolve_primary_nse`.
