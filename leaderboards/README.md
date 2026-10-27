# CAMELS-US MIP Leaderboard JSON Files & Automated Submission Updater (`leaderboards/`)

## 1. Overview & Purpose
The `leaderboards/` package stores the JSON data files for all 12 CAMELS-US Model Intercomparison Project (MIP) leaderboards (`Tables 3–14`, 126 active entries across 9 populated leaderboards and 3 open Freeform tracks), the 32-model family registry (`Table 2`), the Stage-1 confirmation runs on the original authors' splits (`appendix_stage1_confirmation_runs.{json,csv,md}`), and the automated scripts (`leaderboards/update_leaderboard.py`) that score new community Zarr submissions on `gs://camels-us-mip/` and update the leaderboard JSON files.

The `frontend/` web application reads directly from these leaderboard JSON files and calls `leaderboards.update_leaderboard` whenever a community submission is accepted.

| File / Script | Role |
| :--- | :--- |
| `gauged__single_camels.json` .. `pur__freeform.json` (12 files) | Standalone JSON file for each of the $3 \times 4 = 12$ leaderboards (`Tables 3–14`, 126 total rows). |
| `tables_3_to_14.json` | Combined 12-leaderboard bundle automatically synchronized from the 12 individual `<leaderboard_id>.json` files. |
| `table_2_registry.json` | Registry of all 32 hydrological model families (`Table 2`). |
| `contributions_index.json` | Index of the 65 model entries (72 leaderboard rows: 58 `MIP Re-run` rows + 14 `ERA Contribution` rows) contributed by this study (`gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`). |
| `appendix_stage1_confirmation_runs.{md,csv,json}` | Side-by-side comparison of original authors' published metrics vs. our Stage-1 confirmation runs on the authors' exact splits and time periods. |
| `update_leaderboard.py` (`mip-update-leaderboard`) | Script and Python API triggered upon each community submission to score the Zarr hydrograph on `gs://camels-us-mip/`, update `<leaderboard_id>.json` and `tables_3_to_14.json`, and regenerate all manuscript figures in `figures/`. |
| `generate_figures.py` (`mip-generate-figures`) | Generates all manuscript figures (`figures/figure_1_water_year_coverage.{png,pdf}`, `figures/figure_2_leaderboard_progress_all_regimes.{png,pdf}`, and single-panel variants `figure_2{a,b,c}_*.png`) directly from the live leaderboard JSON files. |
| `on_community_commit.sh` | Hook and CI script triggered on each community commit to synchronize `tables_3_to_14.json` and regenerate `figures/*.{png,pdf}`. |

---

## 2. How to Trigger a Leaderboard Update Upon Community Submission

### Command Line (`mip-update-leaderboard`)
```bash
mip-update-leaderboard \
  --model-id community_lstm_v1 \
  --leaderboard-id gauged__single_camels \
  --model-name "Community LSTM v1" \
  --family-id kratzert_lstm \
  --forcing Daymet \
  --code-url https://github.com/example/community-lstm \
  --bucket-uri gs://camels-us-mip
```

### Python API
```python
from leaderboards import trigger_submission_from_gcs

result = trigger_submission_from_gcs(
    model_id="community_lstm_v1",
    leaderboard_id="gauged__single_camels",
    model_name="Community LSTM v1",
    family_id="kratzert_lstm",
    forcing="Daymet",
    code_url="https://github.com/example/community-lstm",
    bucket_uri="gs://camels-us-mip",
)
print("New Rank:", result["updated_entry"]["rank"])
```

---

## 3. Ensemble vs. Single-Seed Columns (`ens_med_nse` / `single_med_nse`)

The two NSE columns of every row are populated from **different quantities** and are never copies
of each other (`leaderboards.update_leaderboard.format_row_metrics_from_evaluator`):

| Store content | `ens_med_nse` | `single_med_nse` (`single_med_nse_std`) |
| :--- | :--- | :--- |
| `is_ensemble=True` **with** `qsim_members` | median NSE of the stored ensemble-mean `qsim` | mean (± population std) over members of each member's 531-basin median NSE |
| `is_ensemble=True` **without** `qsim_members` | median NSE of `qsim` | `None` / `"---"` (unknown; never substituted with the ensemble score) |
| `is_ensemble=False` | `None` / `"---"` | median NSE of `qsim` |

`primary_ranking_nse` (the ranking key) is always the median NSE of the stored hydrograph. The
ensemble status is read from the Zarr store (`is_ensemble` attribute, `qsim_members`), never from
the command line or the model name; `metrics.evaluator` rejects an ensemble store whose `qsim` is
not the declared reduction (`ensemble_reduction`: `mean` (default) or `median`) of its members, and a
single-model store whose `qsim` matches none of its shipped members. Rows written by the evaluator
also carry `num_members`, `has_qsim_members`, `ensemble_source`, `single_med_nse_per_member`, and
`per_basin_nse` (531 values, `null` where a basin is not evaluable).

Legacy rows (seeded before this schema) held the same value in both columns. They were repaired once
with

```bash
python -m leaderboards.update_leaderboard --normalize-legacy-rows
```

which (a) replaces non-finite / `-999.0` sentinels by `null`, (b) keeps only the column the row can
support (ensemble rows keep `ens_med_nse`, single-model rows keep `single_med_nse`), and (c) removes
residual local `.npz` / `submissions/` paths and placeholder DOIs. Duplicate rows (identical headline
metrics under two `entry_id`s) are detected with `find_duplicate_metric_rows()` and removed with
`remove_leaderboard_entry(leaderboard_id, entry_id, reason)`, which keeps the full original row in the
file's `removed_rows` ledger.

## 4. Reconciling Literature Rows With Archived Hydrographs

`Literature` / `Archive*` rows were seeded with author-reported numbers. Where the archived hydrograph
exists in the bucket they can be re-scored with the canonical evaluator:

```bash
# report only (no files modified)
python -m leaderboards.update_leaderboard --reconcile-archived --bucket-uri gs://camels-us-mip --output-json /path/to/report.json
# rewrite those rows from the evaluator (previous value kept as `author_reported_median_nse`)
python -m leaderboards.update_leaderboard --reconcile-archived --apply --bucket-uri gs://camels-us-mip
```

`--recalculate-all-self-run` re-scores every `MIP Re-run` / `Contribution` row from the bucket and fails
loudly (`FileNotFoundError`) for any row whose store is missing; it never falls back to local files.
