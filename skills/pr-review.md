---
name: pr-review
description: >-
  Comprehensive Pull Request (PR) and code review checklist, standards, and
  workflow for camels-us-mip. Covers zero-tolerance checks for
  phantom/fake/masked/imputed data, removal of try/except error suppression,
  zero fallback data paths, centralized split/date injection, cloud-only Zarr
  metric calculation, mathematical and autograd rigor, test and mock audits,
  Google Python readability, and documentation compliance. Use whenever
  reviewing a PR or auditing code changes before commit/merge.
---

# Pull Request & Code Review Standards (`camels-us-mip`)

Whenever reviewing or auditing code in `camels-us-mip`, load this skill alongside [`skills/algorithm-rules-and-norms.md`](./algorithm-rules-and-norms.md), [`skills/testing.md`](./testing.md), [`skills/repo-organization.md`](./repo-organization.md), and [`skills/documentation.md`](./documentation.md), and evaluate every change against the following mandatory pillars.

---

## 1. Zero Tolerance for Phantom, Fake, Masked, or Imputed Data

Audit every line of code for any potential to introduce, fabricate, impute, or mask bad or missing data:

1. **Missing Data In $\implies$ `NaN` Out (or Loud Exception):**
   - **Never** silently drop, zero-fill (`0.0`), forward-fill (`ffill`), backward-fill (`bfill`), linearly interpolate, or replace missing dates, basins, or predictions with climatology or constants.
   - **Never** substitute backup or fallback data paths when a requested Zarr store in `gs://camels-us-mip/` or input dataset is unavailable.
   - **Never** allow `qsim` `NaN` values on days with valid `qobs` (`pred_nan_when_obs_valid > 0`) to be silently masked out by `(~np.isnan(obs)) & (~np.isnan(sim))`.
2. **Centralized Splits & Time Periods Only (`models/run_model.py`):**
   - Verify that individual model scripts in `models/<model_name>/` **never** load basin files, **never** choose training or testing time periods, and **never** assign basins to folds. All split lists and date ranges must be injected by `models/run_model.py`.
3. **Cloud-Only Metrics Calculation (`metrics/`):**
   - Verify that `metrics/` reads model hydrographs **exclusively** from `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` and targets from `gs://camels-us-mip/observations/newman_eval_targets.zarr`, with a single public API for scoring one model and one leaderboard and zero fallback paths.

---

## 2. Eliminate All `try`/`except` Blocks and Error-Swallowing Logic

Flag and remove **every** `try`/`except` block and equivalent error-masking construct:

1. **No `try`/`except` Control Flow or Suppression:**
   - Flag all `try`/`except` blocks, `contextlib.suppress(...)`, `shutil.rmtree(..., ignore_errors=True)`, and `warnings.catch_warnings` blocks in data loading, model runners, Zarr writing, and metric calculation.
   - Flag silent-fallback patterns: `.get(key, default)` on required dictionary/config fields, probing multiple candidate keys (`for k in ['qsim', 'preds', ...]:`), or early `return None` / `return np.nan` / `continue` branches on corrupt or missing inputs.
2. **Explicit Precondition Validation:**
   - Require explicit precondition checks (`Path.exists()`, `fs.exists()`, `in` checks on Zarr variables/coordinates, and exact shape checks `(531, 3652)`) that immediately raise descriptive `ValueError`, `KeyError`, or `FileNotFoundError`.

---

## 3. Mathematical, Statistical, & Autograd Rigor

1. **Forward-Pass, Loss, & Autograd Gradient Integrity:**
   - Verify that normalization statistics (input means/stds, target means/stds, per-basin NSE loss weights) are computed strictly on `train_basins` and `train_dates`.
   - Watch for `0.0 * NaN = NaN` in PyTorch autograd and ensure missing target steps during training are masked before or after the loss without corrupting recurrent gradients.
2. **ERA Experiment Integrity (`era_experiments/`):**
   - Verify all 6 ERA prompt notebooks (`gauged`, `pub_12fold`, `pur_18fold` $\times$ `physics_guided`, `unguided`) start from the Kratzert multi-forcing CudaLSTM baseline, run for `256` iterations, never touch the Newman test period (`1989-10-01` to `1999-09-30`), and (for PUB and PUR) loop over all 12 or 18 canonical folds to assemble the full 531-basin out-of-sample hydrograph before computing the hillclimbing metric.

---

## 4. Test Suite Rigor & Mocking Audit

1. **Reject Trivial or Toy-Dimension Tests:**
   - Require tests to run at full `(531, 3652)` coordinate dimensions with real Zarr stores written to `tmp_path`.
2. **Audit Every Mock:**
   - Only external cloud/network transport boundaries may be redirected to `tmp_path` in hermetic unit tests; never mock Zarr parsing or metric math.
3. **Verify Negative Tests:**
   - Confirm explicit tests exist for missing stores, wrong shapes, mismatched basin order, and `pred_nan_when_obs_valid > 0`.

---

## 5. Python Readability & Documentation Compliance

1. Verify `ruff check` and `pytest` pass with zero errors.
2. Verify all public functions and classes have complete type annotations and Google-style docstrings.
3. Verify all `README.md` files obey [`skills/documentation.md`](./documentation.md) (document only what exists and how to use it; concise, high-school reading level suitable for non-experts and ESL readers).
