# `leaderboards`: Leaderboard and Figure Tools

> [!IMPORTANT]
> **Maintainers only.** Contributors never run these tools. To submit a model or predictions, open
> a pull request or use an issue form in [`.github/ISSUE_TEMPLATE/`](../.github/ISSUE_TEMPLATE/)
> (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)). A maintainer then scores the entry and updates the
> leaderboards with the tools on this page.

This package holds the code that writes the leaderboard files and draws the figures of a MIP. It
holds no data. The files that it writes live in two folders:

- `mips/<mip_id>/leaderboards/`: the leaderboard JSON files. Their schema is in
  [`mips/camels_us_daily/leaderboards/README.md`](../mips/camels_us_daily/leaderboards/README.md).
- `mips/<mip_id>/figures/`: the figures. They are described in
  [`mips/camels_us_daily/figures/README.md`](../mips/camels_us_daily/figures/README.md).

All tools support only the `camels_us_daily` MIP today. Any other `--mip` value stops with an
error.

## Overview

| File | Command | What it does |
| :--- | :--- | :--- |
| [`update_leaderboard.py`](update_leaderboard.py) | `mip-update-leaderboard` (or `python -m leaderboards.update_leaderboard`) | Scores one hydrograph store and adds or updates its leaderboard row. Also has maintenance modes. |
| [`generate_figures.py`](generate_figures.py) | `mip-generate-figures` (or `python -m leaderboards.generate_figures`) | Draws all figures from the leaderboard JSON files. |
| [`on_community_commit.sh`](on_community_commit.sh) | `bash leaderboards/on_community_commit.sh` | Rebuilds `tables_3_to_14.json` and all figures for each MIP. Can install a git pre-commit hook. |
| [`tests/`](tests/) | `pytest leaderboards/tests` | Tests for this package. |

## Prerequisites

- Python 3.11 or newer, with the repository installed (`pip install -e .`, or
  `pip install -e ".[dev]"` to run the tests).
- Run every command from the repository root.
- The scoring commands read stores from a bucket root. The default root is `gs://camels-us-mip`.
  Reading it needs Google Cloud credentials with access to the bucket (see
  [hydromip/README.md](../hydromip/README.md#zarr-io-and-cloud-credentials)). `--bucket-uri` also
  accepts a local folder with the same layout:
  - `<root>/observations/newman_eval_targets.zarr`
  - `<root>/hydrographs/<leaderboard_id>/<model_id>.zarr`
- These commands never read a bucket: `--render-stage1-appendix`, `--render-hydrograph-manifest`,
  `--normalize-legacy-rows`, `mip-generate-figures`, and `on_community_commit.sh`.

## Add or update a leaderboard row (Maintainers only)

The full ingestion procedure is in
[`skills/maintainer-ingest-submission/SKILL.md`](../skills/maintainer-ingest-submission/SKILL.md).
The bucket steps are in [`skills/maintainer-gcs-bucket/SKILL.md`](../skills/maintainer-gcs-bucket/SKILL.md).
The short version:

1. Put the store at `<bucket>/hydrographs/<leaderboard_id>/<model_id>.zarr`.
2. Make sure that the row's model family exists in `table_2_registry.json` (see
   [Families and contributions](#families-and-contributions)).
3. Run the command. These five flags are required: `--model-id`, `--leaderboard-id`,
   `--model-name`, `--family-id`, and `--forcing`.

```bash
mip-update-leaderboard \
  --model-id community_lstm_v1 \
  --leaderboard-id gauged__single_camels \
  --model-name "Community LSTM" \
  --family-id community_lstm \
  --forcing Daymet \
  --citation "Smith et al. (2026)" \
  --code-url https://github.com/example/community-lstm \
  --data-archive-url https://doi.org/10.5281/zenodo.0000000
```

The command does these steps:

1. It scores the store with `metrics.evaluator.calculate_model_leaderboard_metrics`.
2. It builds the row: the scores, `zarr_path`, the citation, and the links.
3. It adds the row, or replaces the row that has the same `entry_id`. Then it ranks the rows again
   and writes `<leaderboard_id>.json`.
4. It rebuilds `tables_3_to_14.json` from the 12 leaderboard files.
5. It draws all figures again and rewrites `figures_manifest.json`.
6. It prints the new row as JSON.

The store decides whether the row is an ensemble (the `is_ensemble` attribute and the
`qsim_members` variable). `--is-ensemble` is only a check: if it disagrees with the store, the
command stops with an error.

### Families and contributions

Every row sets `code_provenance`. You can give it with `--code-provenance`. If you do not, it comes
from `--category`:

- `Contribution` and `MIP Re-run` become `Our Code`;
- `Archive` and `Archive*` become `Author Archive`;
- `Literature` and `Literature*` become `Published Paper`.

An unknown category without `--code-provenance` stops with an error.

After every row change (add, update, or remove), the tools recount `table_2_registry.json`: each
family's `n_entries` and `leaderboards_covered`, and the family totals (`total_families`,
`num_families`, `n_families`). You never set these counts by hand.

Two things still need a separate step:

- **A new model family.** CI checks that every row's `family_id` exists in `table_2_registry.json`.
  Add a new family with the Python function `upsert_leaderboard_entry(..., family_record=...)` (see
  [Python API](#python-api)). It merges `family_record` into the family with the same `family_id`,
  or adds a new family.
- **`contributions_index.json`.** The row command does not change it. Pass `contribution_record=`
  to `upsert_leaderboard_entry` to add or update an entry.

Then check the registries:

```bash
python -m hydromip.tools.ci_guardrails --mode docs-and-registry
```

## Maintenance modes (Maintainers only)

| Flag | Reads the bucket? | What it does |
| :--- | :--- | :--- |
| `--render-stage1-appendix` | no | Checks `appendix_stage1_confirmation_runs.json`, then writes its `.md` and `.csv` copies from it. |
| `--render-hydrograph-manifest OUTPUT_PATH` | no | Writes a JSON list of every store that a row names in `zarr_path`, with the rows that use it, to the local file `OUTPUT_PATH`. Uses `--bucket-uri` as the prefix that every `zarr_path` must have. It never uploads. Publishing the file to `<bucket>/hydrographs/manifest_index.json` is a separate step. |
| `--normalize-legacy-rows` | no | Cleans every row: non-finite or `-999` NSE values become `null`, a row keeps only the NSE column it can support, and leftover local paths and placeholder DOIs are removed. It also cleans the entries of `contributions_index.json`. Then it ranks again, rebuilds `tables_3_to_14.json`, and draws the figures. |
| `--reconcile-archived` | yes | Scores every row whose store exists in the bucket and reports the difference from the stored numbers. It changes no files. It stops with an error if a `MIP Re-run`, `Contribution`, or bucket-backed `Archive` row has no store. |
| `--reconcile-archived --apply` | yes | As above, and it rewrites those rows from the scores. The old median NSE is kept in `author_reported_median_nse`. Then it ranks again, rebuilds `tables_3_to_14.json`, and draws the figures. |
| `--recalculate-all-self-run` | yes | Scores again every `MIP Re-run` and `Contribution` row, and every `Archive`/`Archive*` row whose archive URL points into the bucket. It updates the rows, the metrics in `contributions_index.json`, `tables_3_to_14.json`, and the figures. It first checks that every store exists and stops with one error that lists all missing stores. |

Run one mode at a time. If you pass more than one, only the first in this order runs:
`--render-stage1-appendix`, `--render-hydrograph-manifest`, `--normalize-legacy-rows`,
`--reconcile-archived`, `--recalculate-all-self-run`. Without a mode flag, the command adds a row.

Every mode prints a JSON report. `--output-json PATH` also saves it. Keep reports outside
`leaderboards/` and outside `mips/<mip_id>/leaderboards/`.

```bash
# Compare stored numbers with fresh scores of the archived stores. No files change.
mip-update-leaderboard --reconcile-archived --output-json ./outputs/reconcile.json

# Write the hydrograph store manifest to a local file.
mip-update-leaderboard --render-hydrograph-manifest ./outputs/manifest_index.json

# Rewrite the Stage-1 appendix .md and .csv from its JSON file.
mip-update-leaderboard --render-stage1-appendix
```

## `mip-update-leaderboard` argument reference

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--model-id MODEL_ID` | — | Store ID: the store is `<bucket>/hydrographs/<leaderboard_id>/<MODEL_ID>.zarr`. It also becomes the row's `entry_id`. Required to add a row. |
| `--leaderboard-id ID` | — | One of the 12 leaderboard IDs, for example `gauged__single_camels`. Required to add a row. |
| `--model-name NAME` | — | Display name of the row. Author names and years are removed from it; they belong in `--citation`. Required to add a row. |
| `--family-id ID` | — | Model family ID. It must exist in `table_2_registry.json`. Required to add a row. |
| `--forcing LABEL` | — | Forcing label shown in the row, for example `Daymet`. Required to add a row. |
| `--is-ensemble` | not set | Check that the store is an ensemble. The store always decides. |
| `--category CATEGORY` | `Contribution` | Row `category` (provenance). See the [leaderboard data README](../mips/camels_us_daily/leaderboards/README.md). |
| `--code-provenance LABEL` | from `--category` | Row `code_provenance`: `Our Code`, `Author Archive`, or `Published Paper`. Without it, `Contribution` and `MIP Re-run` give `Our Code`, `Archive`/`Archive*` give `Author Archive`, and `Literature`/`Literature*` give `Published Paper`. |
| `--citation TEXT` | `Community submission` | Citation shown under the model name. |
| `--code-url URL` | empty | Public code repository. Stored as `code_url` and `github_url`. |
| `--data-archive-url URL` | empty | Public archive of the hydrographs (for example a Zenodo DOI). When empty, the store URI is used. |
| `--bucket-uri URI` | `gs://camels-us-mip` | Bucket root, or a local folder with the same layout. |
| `--mip MIP` | `camels_us_daily` | Registered MIP. Only `camels_us_daily` is supported. |
| `--leaderboards-dir PATH` | `mips/<mip>/leaderboards` | Leaderboard folder to read and write. With a copy of the folder, the figures go to a `figures/` folder next to the copy. |
| `--recalculate-all-self-run` | off | Mode: see [Maintenance modes](#maintenance-modes-maintainers-only). |
| `--normalize-legacy-rows` | off | Mode: see above. |
| `--reconcile-archived` | off | Mode: see above. |
| `--apply` | off | With `--reconcile-archived` only: write the new scores into the rows. |
| `--render-stage1-appendix` | off | Mode: see above. |
| `--render-hydrograph-manifest OUTPUT_PATH` | off | Mode: see above. |
| `--output-json OUTPUT_JSON` | none | Also save the printed report of a mode to this file. |
| `-h`, `--help` | — | Show the help text. |

## `mip-generate-figures`

Draws every figure of the MIP from its leaderboard JSON files and writes `figures_manifest.json`.
It prints the paths of the figures as JSON.

```bash
# Redraw the figures in mips/camels_us_daily/figures/
mip-generate-figures

# Draw the figures of a copied leaderboard folder into another folder
mip-generate-figures --leaderboards-dir ./lb_copy/leaderboards --output-dir ./lb_copy/figures
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--mip MIP` | `camels_us_daily` | Registered MIP. Only `camels_us_daily` is supported. |
| `--leaderboards-dir PATH` | `mips/<mip>/leaderboards` | Folder with the leaderboard JSON files. |
| `--output-dir PATH` | `mips/<mip>/figures` | Folder for the figures. If you give `--leaderboards-dir` but not `--output-dir`, the figures go to a `figures/` folder next to the leaderboard folder. |
| `-h`, `--help` | — | Show the help text. |

## `on_community_commit.sh`

Rebuilds the derived files of each MIP from its leaderboard files. For each MIP, it:

1. rebuilds `tables_3_to_14.json` from the 12 `<leaderboard_id>.json` files;
2. runs `python -m leaderboards.generate_figures --mip <mip_id>`.

```bash
# All MIPs
bash leaderboards/on_community_commit.sh

# One MIP
MIP_ID=camels_us_daily bash leaderboards/on_community_commit.sh

# Use a specific Python interpreter
PYTHON_BIN=.venv/bin/python bash leaderboards/on_community_commit.sh

# Install the git pre-commit hook, then exit
bash leaderboards/on_community_commit.sh --install-hook
```

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| `--install-hook` (first argument) | — | Writes `.git/hooks/pre-commit` and exits. It replaces any existing pre-commit hook. The hook runs this script when the staged files touch `mips/<mip_id>/leaderboards/`, `mips/<mip_id>/protocol.json`, or `leaderboards/`. Then it stages each MIP's `tables_3_to_14.json` and `figures/`. |
| `MIP_ID` (environment variable) | all MIPs found by `discover_mips()` | Process only this MIP. |
| `PYTHON_BIN` (environment variable) | the first `python3` on `PATH` | The Python interpreter to use. It must be able to import `hydromip` and `leaderboards`; if not, the script stops with an error. |

GitHub Actions has a matching workflow, **Regenerate Leaderboard Bundle and Figures**
([`.github/workflows/update_leaderboards_and_figures.yml`](../.github/workflows/update_leaderboards_and_figures.yml)).
A maintainer starts it by hand from the Actions tab. It has one optional input, `mip_id`. It runs
this script, then `pytest hydromip/tests leaderboards/tests`. If any file changed, it opens a pull
request from a new branch. It never pushes to `main`.

## Python API

Import these names from `leaderboards` (`leaderboards.__all__`). The default leaderboard folder is
`mips/camels_us_daily/leaderboards`. The default bucket root is `gs://camels-us-mip`. Functions that
take `update_figures=True` also draw all figures again.

### Constants

| Name | Value |
| :--- | :--- |
| `CANONICAL_LEADERBOARD_IDS` | The 12 leaderboard IDs, in table order. |
| `DEFAULT_LEADERBOARDS_DIR` | `mips/camels_us_daily/leaderboards` (absolute path). |
| `DEFAULT_FIGURES_DIR` | `mips/camels_us_daily/figures` (absolute path). |
| `HYDROGRAPH_MANIFEST_NAME` | `"manifest_index.json"` |

### Functions that only read

| Function | Returns |
| :--- | :--- |
| `load_leaderboard(leaderboard_id, leaderboards_dir=...)` | One leaderboard file as a dictionary. |
| `load_all_leaderboards(leaderboards_dir=...)` | `{"protocol": ..., "total_entries": ..., "leaderboards": [...]}`. |
| `find_duplicate_metric_rows(leaderboards_dir=...)` | Groups of rows in one leaderboard with the same headline scores. |
| `resolve_row_store_id(row, leaderboard_id)` | The store ID of a row: from `zarr_path`, or else `entry_id`. |
| `build_hydrograph_manifest(bucket_uri=..., leaderboards_dir=..., updated_utc=None)` | The store manifest as a dictionary. |
| `format_row_metrics_from_evaluator(metrics_summary)` | The score columns of a row, built from the evaluator output. |
| `rank_leaderboard_rows(rows)` | A ranked copy of the rows: by `primary_ranking_nse`, then `mean_nse`, `med_kge`, and `model_name`. Rows without a finite NSE go last. |
| `clean_model_name(name, *, is_ensemble=None, num_members=None, include_ensemble_suffix=True)` | The name without author prefixes, with an ensemble suffix when needed. |
| `extract_publication_year(row)` | The year at which Figure 2 plots the row. Raises `ValueError` if the row has no usable year. |
| `classify_model_category(row)` | The architecture group used for figure colors. |
| `validate_stage1_appendix(payload)` | Nothing. Raises if the Stage-1 appendix JSON is not valid. |
| `render_stage1_appendix_markdown(payload)`, `render_stage1_appendix_csv(payload)` | The appendix as Markdown or CSV text. |
| `normalize_legacy_row(row, contribution_entry=None)` | `(clean_row, changes)` for one row, as `--normalize-legacy-rows` would clean it. |

### Functions that write files

| Function | What it changes |
| :--- | :--- |
| `trigger_submission_from_gcs(model_id, leaderboard_id, model_name, family_id, forcing, is_ensemble=None, category="Contribution", asterisk=False, code_provenance="", family_name="", paradigm="Community Contribution", citation="Community submission", paper_url="", code_url="", data_archive_url="", bucket_uri=..., leaderboards_dir=..., update_figures=True)` | Scores a store and adds its row. This is the function behind the row command. Returns `{"updated_entry": ..., "metrics_summary": ...}`. |
| `upsert_leaderboard_entry(leaderboard_id, entry, leaderboards_dir=..., family_record=None, contribution_record=None, update_figures=True)` | Adds or replaces a row (`entry` needs `entry_id` and `model_name`), ranks again, and rebuilds `tables_3_to_14.json`. Sets `code_provenance` from the category if the row has none. Optionally merges `family_record` into `table_2_registry.json` and `contribution_record` into `contributions_index.json`. Then recounts the family counts in `table_2_registry.json`. Returns the ranked row. |
| `remove_leaderboard_entry(leaderboard_id, entry_id, reason, leaderboards_dir=..., update_figures=True)` | Moves a row into the file's `removed_rows` list, with the reason and the time. Ranks again, rebuilds `tables_3_to_14.json`, marks the matching `contributions_index.json` entry, and recounts the family counts in `table_2_registry.json`. Returns the removal record. |
| `refresh_registry_counts(leaderboards_dir=...)` | Recomputes each family's `n_entries` and `leaderboards_covered`, and the family totals, in `table_2_registry.json` from the rows. Returns `True` if the file changed. |
| `recalculate_all_self_run_models(bucket_uri=..., leaderboards_dir=..., update_figures=True)` | The `--recalculate-all-self-run` mode. |
| `reconcile_archived_rows(bucket_uri=..., leaderboards_dir=..., apply=False, update_figures=True)` | The `--reconcile-archived` mode. |
| `normalize_legacy_leaderboards(leaderboards_dir=..., update_figures=True)` | The `--normalize-legacy-rows` mode. |
| `sync_master_tables_from_per_leaderboard_files(leaderboards_dir=...)` | Rebuilds `tables_3_to_14.json`. |
| `ensure_per_leaderboard_files(leaderboards_dir=...)` | If a `<leaderboard_id>.json` file is missing, it creates it from `tables_3_to_14.json`. |
| `write_stage1_appendix(leaderboards_dir=...)` | The `--render-stage1-appendix` mode. |
| `write_hydrograph_manifest(output_path, bucket_uri=..., leaderboards_dir=..., updated_utc=None)` | The `--render-hydrograph-manifest` mode. |
| `generate_all_paper_figures(leaderboards_dir=..., output_dir=None)` | Draws all figures and writes `figures_manifest.json`. Returns the manifest. |
| `generate_figure_1_water_year_coverage(protocol, output_dir=...)` | Draws Figure 1. Returns `{"png": ..., "pdf": ...}`. |
| `generate_figure_2_leaderboard_progress(all_leaderboards, output_dir=...)` | Draws Figure 2, its three single panels, and one chart per leaderboard. |

### Examples

Read the leaderboards (safe for everyone; changes no files):

```python
from leaderboards import find_duplicate_metric_rows, load_all_leaderboards

bundle = load_all_leaderboards()
print(len(bundle["leaderboards"]))   # 12
print(bundle["total_entries"])       # number of rows over all 12 leaderboards
print(find_duplicate_metric_rows())  # [] when no two rows share the same headline scores
```

Add a row from Python (maintainers only; reads the bucket and writes files):

```python
from leaderboards import trigger_submission_from_gcs

result = trigger_submission_from_gcs(
    model_id="community_lstm_v1",
    leaderboard_id="gauged__single_camels",
    model_name="Community LSTM",
    family_id="community_lstm",
    forcing="Daymet",
    citation="Smith et al. (2026)",
    code_url="https://github.com/example/community-lstm",
    data_archive_url="https://doi.org/10.5281/zenodo.0000000",
)
print(result["updated_entry"]["rank"])
```

## Tests

```bash
pytest leaderboards/tests
```

Some tests check the committed leaderboard and figure files. The others write only to temporary
folders. Together they check ranking, the ensemble and single-seed columns, the family counts in
`table_2_registry.json`, the Stage-1 appendix, the store manifest, and that the figures show every
row with a finite NSE.
