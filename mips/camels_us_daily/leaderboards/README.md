# CAMELS-US Daily Leaderboard Data (`mips/camels_us_daily/leaderboards/`)

This folder holds the published leaderboards of the CAMELS-US Daily MIP. Every file here is
generated. Do not edit these files by hand. Maintainers update them with the tools in
[`leaderboards/`](../../../leaderboards/README.md), and CI checks that the files agree with each
other.

The numbers of rows, families, and entries change over time. Read them from the files:

- `n_entries` in each `<leaderboard_id>.json` file;
- `total_entries` in `tables_3_to_14.json`;
- `total_families` in `table_2_registry.json`;
- `total_contributions` in `contributions_index.json`.

## Files

| File | Role |
| :--- | :--- |
| `<leaderboard_id>.json` (12 files) | One leaderboard each (Tables 3 to 14). `tables_3_to_14.json` and the figures are built from these 12 files. |
| `tables_3_to_14.json` | All 12 leaderboards in one file, with one shared `protocol` block and `total_entries`. |
| `table_2_registry.json` | The model families (Table 2). One record per family. |
| `contributions_index.json` | The entries that HydrologyMIP produced itself (`MIP Re-run` and `Contribution` rows), with their store locations and scores. |
| `appendix_stage1_confirmation_runs.json` | The Stage-1 checks. Each re-implemented model was run on the original authors' own setup and compared with the published score. |
| `appendix_stage1_confirmation_runs.md`, `appendix_stage1_confirmation_runs.csv` | The same Stage-1 table as Markdown and as CSV, written from the JSON file. |

## Leaderboard IDs

A leaderboard ID is `<spatial prefix>__<forcing regime>`. The rules of each regime are in
[`../README.md`](../README.md).

| Spatial prefix | `single_camels` | `multi_camels` | `lagged_streamflow` | `freeform` |
| :--- | :--- | :--- | :--- | :--- |
| `gauged` | Table 3: `gauged__single_camels` | Table 4: `gauged__multi_camels` | Table 5: `gauged__lagged_streamflow` | Table 6: `gauged__freeform` |
| `pub` | Table 7: `pub__single_camels` | Table 8: `pub__multi_camels` | Table 9: `pub__lagged_streamflow` | Table 10: `pub__freeform` |
| `pur` | Table 11: `pur__single_camels` | Table 12: `pur__multi_camels` | Table 13: `pur__lagged_streamflow` | Table 14: `pur__freeform` |

## Leaderboard file schema

Each `<leaderboard_id>.json` file is one JSON object:

| Key | Meaning |
| :--- | :--- |
| `id` | The leaderboard ID. |
| `table_number` | The table name, for example `"Table 3"`. |
| `section` | The section number in the paper, for example `"4.1.1"`. |
| `spatial_regime` | The spatial prefix: `gauged`, `pub`, or `pur`. |
| `forcing_regime` | `single_camels`, `multi_camels`, `lagged_streamflow`, or `freeform`. |
| `title`, `short_title`, `tab_label`, `subtitle` | Display text for the web portal. |
| `n_entries` | The number of rows. |
| `top_model`, `top_nse` | The name and the ranking NSE of the first row (`null` when there are no rows). |
| `rows` | The rows, in rank order. |
| `protocol` | Shared protocol facts. The block is the same in all 12 files. |
| `removed_rows` | Present only after a row was removed. Each item has `reason`, `removed_at_utc`, and the full `row`. |

The `protocol` block has these keys: `benchmark_name`, `n_basins`, `train_period`, `eval_period`,
`spatial_regimes`, `forcing_regimes`, `n_leaderboards`, `targets_loaded`,
`water_year_completeness` (the data of Figure 1; see [`../figures/README.md`](../figures/README.md)),
and `provenance_categories` (the definitions of the row categories).

`tables_3_to_14.json` has three keys: `protocol`, `total_entries`, and `leaderboards` (the 12
leaderboard objects, without their `protocol` blocks).

## Row fields

### Identity

| Field | Meaning |
| :--- | :--- |
| `entry_id` | The ID of the row. It is unique within a leaderboard. |
| `leaderboard_id` | The leaderboard of the row. |
| `rank` | The position of the row, starting at 1. |
| `model_name` | The display name. It never contains author names or years. |
| `family_id` | The model family. It must exist in `table_2_registry.json`. |
| `family_name`, `family_description`, `paradigm` | Text about the family and the model type. |
| `model_category` | The architecture group: `Machine learning`, `Hybrid (ML + Physics)`, `Conceptual`, `Process-based`, or `Statistical`. |
| `forcing` | The forcing label, for example `Daymet` or `Daymet + Maurer + NLDAS`. |
| `eval_period` | The evaluation period: `1989-10-01 to 1999-09-30`. |
| `publication_year` | The year at which Figure 2 plots the row. |
| `notes` | Free text. |

### Provenance and links

| Field | Meaning |
| :--- | :--- |
| `category` | Where the result comes from. See [Provenance vocabulary](#provenance-vocabulary). |
| `code_provenance` | Whose work produced the result. See [Provenance vocabulary](#provenance-vocabulary). |
| `asterisk`, `asterisk_reason` | `true`, with a reason, when the result does not use the full standard setup. |
| `citation`, `paper_url` | The paper that the row cites. |
| `code_url`, `github_url` | The code repository. |
| `data_archive_url`, `zenodo_doi`, `zenodo_url`, `hydroshare_url` | Public archives of the hydrographs. |
| `zarr_path` | The scored store: `gs://camels-us-mip/hydrographs/<leaderboard_id>/<store_id>.zarr`. Rows without a store (for example `Literature` rows) do not have it. |
| `archive_key` | The key of the hydrograph archive behind the row. It is often, but not always, equal to `entry_id`. |

### Scores

| Field | Meaning |
| :--- | :--- |
| `primary_ranking_nse` | The ranking value. For a row with a store, it is the median NSE over the basins, computed by HydrologyMIP. For a row without a store, it is the median NSE reported in the paper. |
| `mean_nse` | The mean NSE over the scored basins. |
| `med_kge` | The median KGE. |
| `med_fhv_abs` | The median absolute peak-flow bias (FHV), in percent. |
| `nse_le_0` | The number of basins with NSE ≤ 0. |
| `num_evaluated_basins` | The number of scored basins. It is 531 for a complete store. `Archive*` and `Literature*` rows can have fewer. |
| `pred_nan_when_obs_valid` | The number of missing predictions on days with valid observations. It is 0 for every scored row. |
| `per_basin_nse` | 531 NSE values in the order of [`../splits/basins_531.txt`](../splits/basins_531.txt). The value is `null` for a basin that was not scored. Only rows with a store have this field. |
| `regional_huc02` | The median NSE in each HUC-02 region (`"01"` to `"18"`). It is `null` for rows without a store. |
| `gauch_2023_metrics` | One summary per metric of the 14-metric suite, in this order: `NSE`, `MSE`, `RMSE`, `KGE`, `logNSE`, `logKGE`, `Alpha-NSE`, `Pearson-r`, `Beta-KGE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`, `Peak-Timing`. Each summary has `median`, `mean`, `min`, `p01`, `p05`, `p10`, `q25`, `q75`, `p90`, `p95`, `p99`, and `max` over the scored basins. Rows without a store only have the values that the paper reports. |

Rows are sorted by `primary_ranking_nse` (high to low). Ties are broken by `mean_nse`, then
`med_kge`, then `model_name`. Rows without a finite NSE go last.

The leaderboard tables show 16 summary columns: the median over the basins of each of the 14
metrics, plus the mean NSE and the mean KGE. All of them come from `gauch_2023_metrics`: the
`median` of each metric, and the `mean` of `NSE` and of `KGE`.

### Ensemble and single-run fields

| Field | Meaning |
| :--- | :--- |
| `is_ensemble` | `true` when the stored hydrograph `qsim` combines several runs (members). |
| `num_members` | The number of members. It is 1 for a single run. |
| `has_qsim_members` | `true` when the store also holds every member (`qsim_members`). |
| `ensemble_source` | How `qsim` relates to the members. See [Ensemble and single-run columns](#ensemble-and-single-run-columns). |
| `ens_med_nse`, `ens_med_nse_display` | The median NSE of the ensemble hydrograph, and its display text. |
| `single_med_nse`, `single_med_nse_std`, `single_med_nse_display` | The skill of a single run, its spread over members, and its display text. |
| `single_med_nse_per_member` | The median NSE of each member. |

### Fields that only some rows have

- Rows of the ERA models: `model_key`, `hydrograph_model_key`, `has_hydrograph`, `seeds`,
  `submitted_at`, `winner_notebook_url`, and flat copies of the metric medians named
  `gauch_<metric>` (for example `gauch_nse`).
- `author_reported_median_nse`: the median NSE that a row had before HydrologyMIP scored the
  archived store again.

## Provenance vocabulary

Two row fields describe where a result comes from: `category` and `code_provenance`.

| `category` | Meaning | `code_provenance` | Store in the bucket? |
| :--- | :--- | :--- | :--- |
| `Literature` | Scores copied from a published paper that used the standard setup. | `Published Paper` | No |
| `Literature*` | Scores copied from a paper that used a different setup, for example fewer basins, other years, or other folds. | `Published Paper` | No |
| `Archive` | HydrologyMIP scored the hydrographs that the authors archived for their paper. | `Author Archive` | Usually |
| `Archive*` | Like `Archive`, but the archive covers fewer than 531 basins or a different period. | `Author Archive` | Yes |
| `MIP Re-run` | HydrologyMIP re-implemented a published model and ran it on the standard setup. | `Our Code` | Yes |
| `Contribution` | A new model contributed to the benchmark. | `Our Code` | Yes |

The `*` marks a result that does not use the full standard setup. These rows have
`asterisk: true` and give the reason in `asterisk_reason`. The `protocol.provenance_categories`
block in each file holds the official definitions of `Literature`, `Archive`, `Archive*`,
`MIP Re-run`, and `Contribution`.

The same idea appears in two other places, with different spelling:

- Families in `table_2_registry.json` use one combined value in `provenance_category`, for example
  `MIP Re-run (Our Code)`.
- Hydrograph stores carry a `provenance_category` attribute. It must be one of `Literature`,
  `MIP Re-run`, `Archive`, `Archive*`, or `Contribution` (`VALID_PROVENANCE_CATEGORIES` in
  `models/zarr_writer.py`).

## Ensemble and single-run columns

The ensemble status comes from the store: its `is_ensemble` attribute and its `qsim_members`
variable. It never comes from the model name. The scorer checks the store. An ensemble's `qsim`
must equal the mean of its members (or the reduction named in the store's `ensemble_reduction`
attribute). A single run's `qsim` must equal exactly one member.

The two NSE columns hold different numbers. One is never a copy of the other.

| Store | `ens_med_nse` | `single_med_nse` (± `single_med_nse_std`) | `primary_ranking_nse` |
| :--- | :--- | :--- | :--- |
| Ensemble with `qsim_members` | Median NSE of the ensemble `qsim` | Mean (± population standard deviation) over members of each member's median NSE | Same as `ens_med_nse` |
| Ensemble without `qsim_members` | Median NSE of `qsim` | `null` (shown as `---`) | Same as `ens_med_nse` |
| Single run | `null` (shown as `---`) | Median NSE of `qsim` | Same as `single_med_nse` |
| Single-run row scored over several members (`ensemble_source` is `mean_across_individual_members`) | `null` (shown as `---`) | Mean (± standard deviation) over members of each member's median NSE | Same as `single_med_nse` |

`ensemble_source` values:

| Value | Meaning |
| :--- | :--- |
| `mean_of_qsim_members` | `qsim` is the mean of the members. |
| `single_member` | The store is a single run, and `qsim` equals one member. |
| `stored_qsim_without_members` | The store has no `qsim_members`. |
| `mean_across_individual_members` | The single-run score is averaged over the members. |
| `null` | The row has no store. |

## `table_2_registry.json`

The file has a `families` list and three count fields: `total_families`, `num_families`, and
`n_families`. All three must equal the length of `families` (CI checks this).

| Family field | Meaning |
| :--- | :--- |
| `family_id` | The ID that rows use in `family_id`. |
| `name`, `paradigm`, `description` | Display text. |
| `category`, `model_category` | The architecture group (the same values as the row field `model_category`). |
| `provenance_category` | The combined provenance value, for example `MIP Re-run (Our Code)`. |
| `citation`, `paper_url`, `year` | The paper of the family and its year. |
| `github_url`, `zenodo_doi`, `zenodo_url`, `hydroshare_url` | Links to code and archives. |
| `code_dir` | The model folder in this repository, for example `models/kratzert_lstm`, or `null`. |
| `asterisk`, `asterisk_reason` | Set when the family's results do not use the full standard setup. |
| `n_entries` | The number of rows of this family over all 12 leaderboards (CI checks this). |
| `leaderboards_covered` | The leaderboards that have at least one row of this family (CI checks this). |
| `best_nse` | The best ranking NSE of the family. |

## `contributions_index.json`

The file has `total_contributions` and `entries`. `entries` is an object keyed by `entry_id`.

| Entry field | Meaning |
| :--- | :--- |
| `entry_id`, `leaderboard_id`, `model_name`, `family_id` | The row that the entry belongs to. |
| `category`, `code_provenance` | As in the rows. |
| `spatial_regime`, `forcing_regime`, `forcings_used` | The setup of the run. |
| `is_ensemble`, `num_members`, `has_qsim_members`, `ensemble_source` | As in the rows. |
| `metrics` | The headline scores: median and mean NSE, median KGE, median absolute FHV, the number of basins with NSE ≤ 0, and the single-run statistics. |
| `gauch_2023_metrics` | As in the rows. |
| `hydrograph_path`, `data_url` | The store location. |
| `code_url` | The code of the model. |
| `target_leaderboard_tables` | The table name and the leaderboard ID, for example `["table_3", "gauged__single_camels"]`. |

Some entries have more fields, for example `citation`, `submitted_at`, `winner_notebook_url`, or
`removed_from_leaderboard` (after a row was removed).

## Stage-1 appendix

`appendix_stage1_confirmation_runs.json` has `generated_at`, `n_rows`, and `rows`. Each row has:

| Field | Meaning |
| :--- | :--- |
| `family_id`, `model_dir`, `model_name`, `citation`, `paper_url` | The model and its paper. |
| `author_split_protocol`, `forcing`, `num_basins`, `num_seeds` | The authors' original setup. |
| `author_published_med_nse`, `author_published_single_med_nse` | The scores that the authors published. |
| `stage1_reproduced_med_nse`, `stage1_reproduced_single_med_nse` | The scores of the HydrologyMIP run on that setup. |
| `delta_med_nse` | The reproduced median NSE minus the published median NSE. |
| `tolerance_abs_delta_med_nse` | The allowed absolute difference, or `null`. |
| `status`, `status_reason` | `PASS` or `PARTIAL`, and the reason when it is not `PASS`. |
| `zarr_path`, `data_archive_url`, `zenodo_url` | Where the Stage-1 hydrographs are stored. |
| `notes` | Free text. |

A row with a tolerance has `status` `PASS` when the absolute value of `delta_med_nse` is at most
the tolerance, and `PARTIAL` otherwise. A row without a tolerance keeps the status recorded in its
model's `stage1_author_reproduction_results.json`. The `.md` and `.csv` files are written from the
JSON file and are never edited on their own.

## What CI checks

`python -m hydromip.tools.ci_guardrails --mode docs-and-registry` checks that:

- each leaderboard file exists, and its `n_entries` equals its number of rows;
- each leaderboard file has the same `entry_id` and `primary_ranking_nse` values, in the same
  order, as `tables_3_to_14.json`;
- every row's `family_id` exists in `table_2_registry.json`, and each family's `n_entries` and
  `leaderboards_covered` match the rows;
- `figures/figures_manifest.json` counts the same number of rows, and every figure that it lists
  exists.
