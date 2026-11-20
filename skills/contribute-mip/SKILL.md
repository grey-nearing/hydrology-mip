---
name: contribute-mip
description: >-
  Workflow and contract for adding a new self-contained benchmark (MIP) under mips/<mip_id>/ in
  HydrologyMIP: proposal issue, protocol.json schema and validation, canonical splits with
  cross-validation hygiene checks, adapter.py, empty leaderboard files, optional metric-suite
  registration, model catalog updates, tests, and docs, plus the parts of the toolchain that are
  still specific to CAMELS-US. Use when proposing, scaffolding, or reviewing a new MIP.
license: Apache-2.0
metadata:
  audience: contributor
---

# Contribute a new MIP (`mips/<mip_id>/`)

Adding a MIP should only **add** files. Existing MIPs and shared models stay unchanged. Read
`mips/README.md` (the per-MIP contract) and `hydromip/README.md` (core API and `protocol.json`
schema) first. All paths are relative to the repository root.

## 0. Propose first

Open a **New MIP Proposal** issue (`.github/ISSUE_TEMPLATE/new-mip-proposal.yml`) before you write
code. Describe:

- the task type, target variable, and units;
- the dataset and its open license;
- the number of entities;
- the temporal windows and spatial regimes, and how they prevent leakage;
- the metric suite and primary metric;
- the baseline models.

Wait for a maintainer to agree.

## 1. Folder layout

```text
mips/<mip_id>/
├── protocol.json        # required: declarative spec, validated by hydromip.protocol
├── README.md            # required: overview, data-source DOIs and licenses, protocol rules in plain words
├── __init__.py          # required (package data and imports)
├── adapter.py           # required: loads and validates entities and folds
├── splits/              # required: entity list + fold files
├── leaderboards/        # required: one <leaderboard_id>.json per declared leaderboard
├── figures/             # required folder (generated content)
├── data/                # optional: README with download/build steps; binaries are gitignored
└── era_experiments/     # optional
```

`<mip_id>` is lowercase with underscores, for example `caravan_daily`. It must equal the `mip_id` in
`protocol.json`.

## 2. `protocol.json`

- Required top-level keys: `schema_version` (`hydromip-protocol-v1`), `mip_id`, `name`,
  `short_name`, `status`, `version`, `task_type`, `metric_suite`, `primary_metric`,
  `primary_metric_higher_is_better`, `target_variable`, `target_units`, `prediction_dims`,
  `entity_unit`, `num_entities`, `num_eval_steps`, `temporal_windows`, `spatial_regimes`,
  `forcing_regimes`, `leaderboards`, `storage`.
- `task_type` must be one of `hydromip.SUPPORTED_TASK_TYPES`. `metric_suite` must be registered,
  and `primary_metric` and its direction must match the suite. `entity_unit` must be one of
  `prediction_dims`.
- Every spatial regime has `id`, `leaderboard_prefix`, `title`, `num_folds`, `description`, and a
  `split_file` when `num_folds > 1`. Every forcing regime has `id`, `title`, `description`.
- Every leaderboard has `id`, `spatial_regime`, `forcing_regime`, `title`. Its `id` must equal
  `<leaderboard_prefix>__<forcing_regime>`, and each regime pair appears once.
- `paths` is optional. It may only repeat the standard folder names; custom paths are not supported.
- Use `mips/camels_us_daily/protocol.json` as the worked example. The validator is the final word:

```bash
python -c "from hydromip.protocol import get_mip, validate_mip_assets; p = get_mip('<mip_id>'); validate_mip_assets(p); print(p.to_summary_dict())"
```

## 3. Splits and cross-validation hygiene

- Store the canonical entity list (sorted, unique) and one JSON file per multi-fold regime, with a
  `folds` object. Each fold has `train_basins` and `test_basins` (or your entity name). Follow
  `mips/camels_us_daily/splits/pub_12fold_random_splits.json`.
- Load and check folds with `hydromip.splits.load_cv_splits_json(path, canonical_entities, expected_num_folds=..., require_disjoint_train_test=True, require_exact_test_partition=True, entity_to_group=...)`.
  Pass `entity_to_group` for group-separated regimes such as regions or river networks.
- Write down how the folds were built (seed, grouping) in `splits/` and in the README.

## 4. `adapter.py`

Follow `mips/camels_us_daily/adapter.py`: small functions that load the protocol, the canonical
entities, and the folds for each regime, and that validate everything with `hydromip.splits`. Zero
`try`/`except`; raise on any problem.

## 5. Metric suite (only if the existing one does not fit)

Put the metric code in a new module, `hydromip/metrics/<suite>.py`. Add its `MetricSuiteSpec` to
`METRIC_SUITES` in `hydromip/metrics/registry.py`, so it is registered whenever `hydromip` is
imported. A runtime `register_metric_suite(...)` call lasts only for the current process. A
`protocol.json` that names an unregistered suite fails to load.

```python
MetricSuiteSpec(
    suite_id="<suite_id>", description="...", metric_names=("CRPS", "..."),
    primary_metric="CRPS", higher_is_better=False, compute_fn=compute_all, citation="...",
)
```

`primary_metric` must be one of `metric_names`. A new suite changes shared code, so add unit tests in
`hydromip/tests/`. The CAMELS-US evaluator calls the `gauch_2023_deterministic` metric functions
directly, so a MIP with a new suite also needs its own scoring code (see section 7).
See `hydromip/README.md`, section "Register a metric suite".

## 6. Leaderboards, models, and the portal

- Create one `<leaderboard_id>.json` per declared leaderboard with the same top-level fields as the
  CAMELS files and `"rows": []`. See `skills/data-schemas/SKILL.md`.
- Add `<mip_id>` to `supported_mips` in `models/catalog.json` **only** for models that really support
  the new task.
- `hydromip.discover_mips()` and the portal's `/api/mips` pick up the MIP automatically.

## 7. Toolchain limits (still CAMELS-US specific)

These parts assume the CAMELS-US shape (531 basins, 3,652 days, the 12 CAMELS leaderboard IDs):

- the split handling in `models/run_model.py`;
- `models/zarr_writer.py`;
- `metrics/evaluator.py` (and `mip-score-model` accepts only the supported MIP IDs);
- `benchmarks/metrics_pipeline.py`.

Do not patch around them inside your MIP. Open an issue and plan a generalization with the
maintainers.

## 8. Tests and docs

- Add tests that check the protocol loads, `validate_mip_assets` passes, all folds are valid, and
  `/api/mips` lists the MIP. The core tests in `hydromip/tests/` accept any number of registered
  MIPs, so a new MIP should not break them.
- Write `mips/<mip_id>/README.md` and add a row to the table in `mips/README.md`.
- Add the new files to `[tool.setuptools.package-data]` in `pyproject.toml`.
- Run:

```bash
ruff check .
python -m hydromip.tools.ci_guardrails --mode all
pytest hydromip/tests
```

## Stop and ask a human if

- the benchmark needs data that cannot be shared openly;
- the task needs a new `task_type`, changes to shared code, or a generalization of the tools in
  section 7;
- you are not sure the folds prevent leakage.
