# CAMELS-US Daily Figures (`mips/camels_us_daily/figures/`)

The figures in this folder are drawn from the leaderboard JSON files in
[`../leaderboards/`](../leaderboards/README.md) by
[`leaderboards/generate_figures.py`](../../../leaderboards/generate_figures.py). They are generated
files. Do not edit them by hand. The maintainer tools that change a leaderboard also draw the
figures again (see [When the figures are drawn again](#when-the-figures-are-drawn-again)).

## Files

| File | Content | Data source |
| :--- | :--- | :--- |
| `figure_1_water_year_coverage.png`, `.pdf` | **Figure 1.** How complete the streamflow records of the 531 basins are in each water year. | `protocol.water_year_completeness` in the leaderboard JSON files |
| `figure_2_leaderboard_progress_all_regimes.png`, `.pdf` | **Figure 2.** Leaderboard progress over time in three panels: (a) gauged, (b) 12-fold PUB, (c) 18-fold HUC-02 PUR. | All 12 leaderboard files |
| `figure_2a_leaderboard_progress_gauged.png` | Panel (a) alone: gauged (Tables 3 to 6). | `gauged__*.json` |
| `figure_2b_leaderboard_progress_pub.png` | Panel (b) alone: PUB (Tables 7 to 10). | `pub__*.json` |
| `figure_2c_leaderboard_progress_pur.png` | Panel (c) alone: PUR (Tables 11 to 14). | `pur__*.json` |
| `figure_<leaderboard_id>.png` (12 files) | Progress over time for one leaderboard. A leaderboard without rows shows a "No entries yet" message. | `<leaderboard_id>.json` |
| `figures_manifest.json` | A machine-readable record of the last drawing (see below). | All of the above |

## Figure 1: water-year completeness

Figure 1 uses only the `water_year_completeness` block inside the `protocol` block of the
leaderboard files. That block has these keys:

| Key | Meaning |
| :--- | :--- |
| `n_basins` | The number of basins (531). |
| `water_years` | The water years shown, in increasing order. |
| `basins_with_100pct_complete` | For each water year, the number of basins with no missing day. |
| `basins_with_any_valid_data` | For each water year, the number of basins with at least one valid day. |
| `mean_daily_coverage_fraction` | For each water year, the mean over basins of the fraction of valid days. |
| `newman_eval_water_years`, `newman_train_water_years` | Optional. The shaded evaluation and training windows. Defaults: 1990–1999 and 2000–2008. |

A water year `Y` runs from 1 October of year `Y-1` to 30 September of year `Y`. There is no
built-in fallback: if the block is missing or broken, drawing stops with a `ValueError` that names
the bad key.

To compute the block again, pass a `(basins, days)` streamflow array and its dates to
`compute_water_year_completeness`. The committed block covers water years 1981 to 2014. It comes
from the 531 benchmark basins in the input bundle (see [`../data/README.md`](../data/README.md)):

```python
import numpy as np

from leaderboards.generate_figures import compute_water_year_completeness
from mips.camels_us_daily import load_canonical_basins

basins = load_canonical_basins()
with np.load("mips/camels_us_daily/data/camels_671_1980_2014.npz", allow_pickle=False) as bundle:
    row = {b: i for i, b in enumerate(bundle["basins"].tolist())}
    qobs = bundle["qobs"][[row[b] for b in basins]]
    dates = bundle["dates"].astype("datetime64[D]")

block = compute_water_year_completeness(qobs, dates)
print(block["n_basins"], block["water_years"][0], block["water_years"][-1])  # 531 1981 2014
```

The observation targets store (`newman_eval_targets.zarr`, variables `qobs_all` and `date_all`)
also works, but it ends in 2008, so its block stops at water year 2008.

`compute_water_year_completeness(qobs_all, dates_all, *, include_partial_water_years=False, ...)`
drops water years that the dates do not fully cover, because their completeness is unknown. It
lists them in `partial_water_years_excluded`.

## Figure 2: leaderboard progress

Each point is one leaderboard row. The x-axis is a year, and the y-axis is the median NSE of the
row over the basins. The marker shape shows the forcing regime, and the color shows the
architecture group (`model_category`).

- **Every row with a finite NSE is drawn.** The axes grow to fit all points; no point is cut off.
  The y-axis of each panel covers at least these ranges: gauged 0.42–0.91, PUB 0.40–0.80, and
  PUR 0.30–0.76. The x-axis covers at least 2015–2026.
- **Rows without a usable NSE are skipped and listed.** A row is skipped when
  `primary_ranking_nse`, `ens_med_nse`, and `single_med_nse` are all missing, not numbers, not
  finite, or the old `-999` placeholder. Skipped rows appear in `figures_manifest.json` under
  `excluded_rows`, with a reason.
- **Year of each point.** `Literature`, `Literature*`, `Archive`, and `Archive*` rows use the year
  of the cited paper and have filled markers. `MIP Re-run` rows use the year of the original paper
  of the model and have hollow markers. `Contribution` rows use the year of their submission
  (`submitted_at`), or 2026 when there is none, and have hollow markers. A literature row without
  any year between 1990 and 2035 stops the drawing with a `ValueError`.
- **Frontier line.** For each forcing regime, a step line connects the best NSE so far over time.
  Rows whose model name contains "oracle", `Literature*` rows (and `Literature` rows with an
  asterisk), and the persistence benchmark `naive_lag1_persistence` are drawn but never set the
  frontier.

## `figures_manifest.json`

| Key | Meaning |
| :--- | :--- |
| `source_leaderboards_dir`, `output_figures_dir` | The folders used, relative to the repository root. |
| `total_leaderboard_entries` | The number of rows over all 12 leaderboards. CI checks that it matches the leaderboard files. |
| `mip_study_year` | The year used for new contributions (2026). |
| `panel_entry_counts` | The number of points drawn in each Figure 2 panel. |
| `excluded_rows` | The skipped rows of each panel, with the reason. |
| `panel_axis_limits` | The x- and y-limits of each panel. |
| `figure_1_source` | Always `protocol.water_year_completeness`. |
| `figures` | The path of every figure file. CI checks that each one exists. |
| `frontier_leaders` | For each panel, the rows on the frontier line, with year, NSE, and provenance. |

## When the figures are drawn again

- **By the Python tools (maintainers).** `trigger_submission_from_gcs`, `upsert_leaderboard_entry`,
  `remove_leaderboard_entry`, `recalculate_all_self_run_models`, `normalize_legacy_leaderboards`,
  and `reconcile_archived_rows` (with `apply=True`) call `generate_all_paper_figures`. The
  `mip-update-leaderboard` command uses these functions.
- **By the git pre-commit hook (optional).** Install it with
  `bash leaderboards/on_community_commit.sh --install-hook`. When a commit changes files in
  `mips/<mip_id>/leaderboards/`, `mips/<mip_id>/protocol.json`, or `leaderboards/`, the hook runs
  `leaderboards/on_community_commit.sh`. Then it adds the new `tables_3_to_14.json` and `figures/`
  to the commit.
- **By GitHub Actions.** The workflow **Regenerate Leaderboard Bundle and Figures**
  ([`.github/workflows/update_leaderboards_and_figures.yml`](../../../.github/workflows/update_leaderboards_and_figures.yml))
  runs only when a maintainer starts it by hand. It runs `leaderboards/on_community_commit.sh` and
  the tests. If any file changed, it opens a pull request. It never pushes to `main` directly.

`on_community_commit.sh` uses `PYTHON_BIN` as its Python interpreter (default: the first `python3`
on `PATH`). That interpreter must be able to import `hydromip` and `leaderboards`. Set `MIP_ID`
to process only one MIP. Details are in [`leaderboards/README.md`](../../../leaderboards/README.md).

## Draw the figures yourself

```bash
# Redraw the figures in this folder
mip-generate-figures

# Draw the figures of a copied leaderboard folder into another folder
mip-generate-figures --leaderboards-dir ./lb_copy/leaderboards --output-dir ./lb_copy/figures
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--mip MIP` | `camels_us_daily` | Registered MIP. Only `camels_us_daily` is supported. |
| `--leaderboards-dir PATH` | `mips/camels_us_daily/leaderboards` | Folder with the leaderboard JSON files. |
| `--output-dir PATH` | `mips/camels_us_daily/figures` | Folder for the figures. With `--leaderboards-dir` but no `--output-dir`, the figures go to a `figures/` folder next to the leaderboard folder. |
| `-h`, `--help` | — | Show the help text. |

`python -m leaderboards.generate_figures` is the same command.

## Tests

```bash
pytest leaderboards/tests/test_generate_figures.py
```

The tests draw all figures from the leaderboards into a temporary folder. They check that the
number of points equals the number of rows with a finite NSE, that the skipped rows are listed,
that Figure 1 fails without its protocol block, and that the committed figures match
`figures_manifest.json`.
