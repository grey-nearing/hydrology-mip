# Benchmarks (`mips/`)

Every benchmark of HydrologyMIP (<https://hydrologymip.org>) lives in its own folder,
`mips/<mip_id>/`. In this repository a benchmark is called a **MIP** (model intercomparison
project). The shared core package, [`hydromip`](../hydromip/README.md), finds each MIP through its
`protocol.json` file. Models in [`models/`](../models/README.md) are shared by all MIPs.

## Registered MIPs

| `mip_id` | Name | Task type | Metric suite | Entities × evaluation days | Leaderboards | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`camels_us_daily`](camels_us_daily/README.md) | CAMELS-US Daily Streamflow Model Intercomparison Project | `streamflow_daily_deterministic` | `gauch_2023_deterministic` (14 metrics) | 531 basins × 3,652 days | 12 (3 spatial regimes × 4 forcing regimes) | `active`, version 1.0.0 |

To list the registered MIPs from Python:

```bash
python -c "from hydromip import discover_mips; print(sorted(discover_mips()))"
```

## Folder contract

The subfolder names are fixed. `protocol.json` cannot rename them (see
[`hydromip/README.md`](../hydromip/README.md#the-paths-key)).

| Path | Required? | Purpose | Checked by |
| :--- | :--- | :--- | :--- |
| `protocol.json` | Required | The machine-readable description of the MIP: task, metric suite, time windows, regimes, leaderboards, storage. The schema is in [`hydromip/README.md`](../hydromip/README.md#the-protocoljson-schema). | `discover_mips()` loads it and `validate_protocol_dict` checks it. |
| `splits/` | Required | Entity lists and cross-validation fold files. It must hold every `split_file` named in `protocol.json`. A JSON split file has exactly `num_folds` folds under its `folds` key. | `validate_mip_assets` |
| `leaderboards/` | Required | One `<leaderboard_id>.json` file per declared leaderboard, with a matching `id`. An `active` MIP also needs `table_2_registry.json` and `tables_3_to_14.json`. | `validate_mip_assets` and `ci_guardrails` |
| `figures/` | Required | Generated figures. If `figures_manifest.json` exists, its row count and figure paths must be correct. | `validate_mip_assets` and `ci_guardrails` |
| `README.md` | Required | Benchmark overview, data sources with DOIs and licenses, and the protocol rules in plain words. | Reviewers |
| `data/` | Optional | A `README.md` with download and build steps. Large data files stay out of git. | — |
| `adapter.py` and `__init__.py` | Optional | A small Python module that loads the MIP's splits and checks them. | The MIP's own tests |
| `era_experiments/` | Optional | Notebooks for experiments that belong to this MIP. | — |

`validate_mip_assets` is in `hydromip.protocol`. `ci_guardrails` is
`python -m hydromip.tools.ci_guardrails --mode docs-and-registry`.

### What `adapter.py` does

`hydromip` never imports `adapter.py`. In `camels_us_daily`, the adapter loads the protocol, the
standard basin list, and the basin-to-region map. It also loads the folds of each spatial regime
and checks them with `hydromip.splits.validate_cv_folds`. The tests use it to prove that the
splits do not leak.

The adapter does not load forcing or streamflow data. Model runs read data through
`models/camels_data.py`, and `models/run_model.py` reads the split files itself.

## Shared code and CAMELS-specific code

These parts work for any MIP:

- `hydromip`: protocol loading, folder paths, CV checks, the metric registry, and Zarr I/O;
- the web portal's MIP list (`/api/mips`), which uses `discover_mips()`;
- `leaderboards/on_community_commit.sh`, which loops over all MIPs.

These parts support only `camels_us_daily` today:

- `models/run_model.py`: the split file names, the three spatial regimes, and the
  531 × 3,652 output shape;
- `models/camels_data.py`, the CAMELS-US data loader;
- `models/zarr_writer.py`: 531 basins, 3,652 days, and the 12 leaderboard IDs;
- `metrics/evaluator.py` and `mip-score-model` (`--mip` accepts only `camels_us_daily`);
- the `leaderboards/` tools and `benchmarks/`;
- `hydromip/tests/test_hydromip_core.py`, which expects `camels_us_daily` to be the only
  registered MIP.

A new MIP therefore needs work in these places. Plan it in a proposal issue first.

## Add a new MIP

1. Open a **New MIP Proposal** issue
   ([`.github/ISSUE_TEMPLATE/new-mip-proposal.yml`](../.github/ISSUE_TEMPLATE/new-mip-proposal.yml)).
   Agree on the data, the splits, the metrics, and the storage before you write code.
2. Create `mips/<new_mip_id>/protocol.json`. `mip_id` must equal the folder name. Start from the
   example in [`hydromip/README.md`](../hydromip/README.md#example). Use `"status": "planned"` or
   `"beta"` while you build the MIP. CI checks the leaderboard bundle files only for `active` MIPs.
3. If the MIP needs new metrics, add a metric suite module under `hydromip/metrics/` and add it to
   `METRIC_SUITES` in `hydromip/metrics/registry.py`.
4. Add the entity lists and the fold files to `splits/`. Check them with
   `hydromip.load_cv_splits_json` (use `entity_to_group` for region-based folds).
5. Add one `<leaderboard_id>.json` per leaderboard to `leaderboards/`, each with its `id`,
   `"n_entries": 0`, and `"rows": []`. Add a `figures/` folder. Git keeps a folder only if it
   holds a file, so put a short `README.md` in it.
6. Write `README.md` and `data/README.md`: data sources, DOIs, licenses, and protocol rules.
7. Optional: add `adapter.py` and `__init__.py`, following `camels_us_daily/adapter.py`.
8. Add the MIP's data files to `[tool.setuptools.package-data]` in `pyproject.toml`, so that
   installed packages include them.
9. In `models/catalog.json`, add the new `mip_id` to `supported_mips` only for models that really
   support it.
10. Update the tests that expect one MIP. Then run:

    ```bash
    python -c "from hydromip import get_mip; from hydromip.protocol import validate_mip_assets; validate_mip_assets(get_mip('<new_mip_id>')); print('assets OK')"
    pytest hydromip/tests
    python -m hydromip.tools.ci_guardrails --mode all
    ```

11. Add a row for the new MIP to the table at the top of this page.

AI coding agents follow the skill
[`skills/contribute-mip/SKILL.md`](../skills/contribute-mip/SKILL.md) for these steps.
