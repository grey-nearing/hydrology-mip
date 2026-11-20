---
name: contribute-model
description: >-
  Step-by-step workflow for adding a runnable rainfall-runoff model to HydrologyMIP under
  models/<name>/: the exact run_fold contract, MODEL_REGISTRY and models/catalog.json registration,
  per-model README, tests, Stage-1 author reproduction for MIP Re-runs, a local run, local
  self-scoring without any Google Cloud account, and the pull request. Use when implementing,
  porting, or re-running a model inside this repository.
license: Apache-2.0
metadata:
  audience: contributor
---

# Contribute a runnable model (`models/<name>/`)

Read `models/README.md` (the human guide) and `skills/protocol-guardrails/SKILL.md` first. All paths
are relative to the repository root. You never need write access to `gs://camels-us-mip/`.

## 0. Decide what you are adding

- **`Contribution`**: a new model or architecture.
- **`MIP Re-run`**: a published model, re-implemented and run on the MIP protocol. Run only the
  configurations (forcings, spatial regimes) that the original authors ran. You also need a Stage-1
  reproduction (step 5). See `skills/leaderboard-norms/SKILL.md`.
- Pick a folder key: lowercase, with underscores, such as `smith2027_gru`. Model IDs and entry IDs
  must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`.

## 1. Create the folder

```text
models/<key>/
├── __init__.py
├── model.py        # architecture and equations
├── runner.py       # run_fold(...)
└── README.md       # use references/model_readme_template.md
```

Start every `.py` file with a two-line header:

```python
# Copyright 2026 <your name or organization>
# SPDX-License-Identifier: Apache-2.0
```

Only add third-party code whose license allows it (MIT, BSD, Apache-2.0). Credit it in
`THIRD_PARTY_NOTICES.md`. Never copy GPL or non-commercial code.

## 2. Implement `run_fold`

Copy the template in `references/run_fold_template.md`. The contract:

- The signature is exactly `run_fold(train_basins, test_basins, train_period, eval_period, forcings, data_dir, seeds=(123,), **kwargs)`.
  `kwargs` always contains `fold_id` and `fold_index`. Raise `TypeError` on any other unknown key.
- Load data only with `load_camels_slice(basins, period, forcings, data_dir, spinup_days=..., require_valid_targets=...)`.
  Fit statistics only with `compute_train_stats(train)`.
- Return `float32` with shape `(len(test_basins), n_eval_days)` in mm/day, rows in `test_basins`
  order, and no NaN.
- Use the single seed in `seeds` with seed-local random generators. Never average seeds.
- Never read split files, never hard-code dates or basins, never write files.

## 3. Register the model in both places

1. Add a `ModelRegistryEntry` to `MODEL_REGISTRY` in `models/run_model.py`.
2. Add the matching entry to `models/catalog.json`.
3. Add the folder name to `EXPECTED_MODEL_DIRS` in `models/tests/test_run_model.py`, and update the
   expected model count in the same test.

Examples are in `references/registration_examples.md`. The name, family, provenance, runner, code
URL, and regimes must be identical in the registry and the catalog. Tests check this.

## 4. Add tests

Add `models/tests/test_<key>.py`, which is already a test path. Include:

- a fast CPU smoke test of `run_fold` on small synthetic inputs;
- a check that unknown keyword arguments raise;
- a check that the output shape, dtype, and finiteness are correct.

Tests of stores, splits, or scoring must use the full 531 × 3,652 size (see
`skills/testing/SKILL.md`).

## 5. Stage-1 reproduction (MIP Re-run only)

- Add `models/<key>/reproduce_author_metrics.py`. It runs the **authors'** split, basin subset,
  forcing, and hyperparameters, and checks that your code reproduces the paper's reported numbers.
- Note the table or figure in the paper for every target number.
- The script writes an untracked `*.local.json` by default. Only regenerate
  `stage1_author_reproduction_results.json` when asked. Uploads are opt-in.
- Import `CANONICAL_*` from `models.zarr_writer` if you need the MIP windows.

## 6. Get the input data

Follow `mips/camels_us_daily/data/README.md`. You need the NCAR CAMELS-US v1.2 download and the
extended Maurer and NLDAS forcings. Then build the bundle into `mips/camels_us_daily/data/`, which is
gitignored.

## 7. Run locally

```bash
# quick smoke run with one seed, written to ./outputs/<leaderboard_id>/<model_id>.zarr
python -m models.run_model --model <key> --spatial-regime gauged --forcings daymet --seeds 1
# full run straight into a local scoring layout
python -m models.run_model --model <key> --spatial-regime pub_12fold --forcings daymet maurer nldas \
  --output-zarr ./local_bucket/hydrographs/pub__multi_camels/<key>_pub_12fold_multi_camels.zarr
```

Lagged-streamflow models must add `--forcing-regime lagged_streamflow`.

## 8. Self-score locally (no Google account needed)

```bash
# once: rebuild the observation targets from the raw NCAR download
python -m metrics.build_eval_targets --camels-dir /path/to/basin_dataset_public_v1p2 \
  --splits-dir mips/camels_us_daily/splits \
  --output ./local_bucket/observations/newman_eval_targets.zarr
# score one store under ./local_bucket/hydrographs/<leaderboard_id>/<model_id>.zarr
mip-score-model --model-id <model_id> --leaderboard-id <leaderboard_id> \
  --bucket-uri ./local_bucket --output-json ./outputs/<model_id>_score.json
```

In the JSON, check that `summary.num_evaluated_basins` is 531 and `summary.pred_nan_when_obs_valid`
is 0. Keep the file; you attach it to the PR.

## 9. Check, then open the PR

```bash
ruff check .
python -m hydromip.tools.ci_guardrails --mode all
pytest models/tests
```

- Open the PR from your fork and fill in every item of the template. Attach the score JSON. Say
  which AI tools you used.
- Do **not** edit the leaderboard JSONs, `table_2_registry.json`, `contributions_index.json`, or the
  figures. Maintainers add your row after the official run.
- If you also have hydrographs to submit, follow `skills/contribute-hydrographs/SKILL.md` (Zenodo
  deposit and issue form).

## Stop and ask a human if

- the paper's configuration is unclear, or Stage-1 does not reproduce the paper;
- your model needs a new forcing regime, new input data, or changes to `run_model.py`,
  `camels_data.py`, or `zarr_writer.py`;
- you cannot meet a rule in `skills/protocol-guardrails/SKILL.md`.
