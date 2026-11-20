---
name: testing
description: >-
  How HydrologyMIP tests are organized, written, and run: co-located test folders registered in
  pyproject.toml, the strict pytest markers (unit, integration, e2e, slow, gpu, gcs), offline and
  hermetic rules, testing at the native 531 x 3652 size, where mocking is allowed, required
  negative tests, and the CI shards. Use when writing, moving, running, or reviewing tests.
license: Apache-2.0
metadata:
  audience: contributor
---

# Testing

All paths are relative to the repository root.

## 1. Where tests live

- Each package has its own `tests/` folder: `hydromip/tests`, `metrics/tests`, `models/tests`,
  `models/kratzert_lstm/tests`, `leaderboards/tests`, `web/tests`, `benchmarks/tests`.
- `pyproject.toml` lists them under `[tool.pytest.ini_options] testpaths`. A new test folder must be
  added there. For a new model, prefer `models/tests/test_<key>.py`, which is already covered.
- Repository guardrails test themselves: `python -m hydromip.tools.ci_guardrails --self-test`.

## 2. Markers (strict)

`pyproject.toml` runs pytest with `--strict-markers`, so only registered markers work:

| Marker | Use for |
| :--- | :--- |
| `unit` | fast, hermetic tests (the default kind) |
| `integration` | tests that combine several components |
| `e2e` | end-to-end pipeline checks |
| `slow` | long tests; skip them with `-m "not slow"` |
| `gpu` | tests that need a CUDA GPU |
| `gcs` | tests that need live Google Cloud Storage credentials and network access |

CI runs on CPU, without credentials or network. Mark anything that needs a GPU or GCS so it can be
deselected. To register a new marker, add it to `pyproject.toml` in the same PR.

## 3. How to run

```bash
pytest                                        # all test paths
pytest models/tests models/kratzert_lstm/tests
pytest -m "not slow and not gpu and not gcs"  # what a laptop can run
pytest -n auto                                # parallel (pytest-xdist, part of the dev extras)
```

CI (`.github/workflows/ci.yml`) runs the suites in parallel shards with CPU-only PyTorch and
`HYDROMIP_GCS_ANON=1`. Tests must pass with no Google credentials, no network, and no data bundle.

## 4. Native size, real stores

- Code that writes or reads hydrograph stores, validates splits, stitches folds, or computes
  metrics is tested at the real size: **531 basins × 3,652 days**. Write real Zarr stores to
  `tmp_path` and assert exact numbers. This takes milliseconds and catches ordering and
  regional-aggregation bugs that toy sizes hide.
- Never monkeypatch `531`, `3652`, or the evaluation window to toy values.
- Model forward-pass smoke tests may use small synthetic inputs, because they test the model, not
  the protocol.

## 5. Mocking

- Mock only external boundaries: GCS or network transport. Point a `gs://` URI at a real store on
  `tmp_path`, or pass a local root (`bucket_uri=tmp_path`).
- Never mock what is being tested: the evaluator, coordinate validation, metric formulas, or fold
  stitching.

## 6. Required negative tests

Every change to data handling needs tests showing that bad input fails loudly:

- a missing store, a missing `qsim`, wrong dimensions such as `(530, 3652)` or `(531, 3650)`,
  unsorted or wrong basin IDs, or wrong dates raise `FileNotFoundError`, `KeyError`, or `ValueError`;
- a single NaN in `qsim` on a day with a valid observation makes scoring raise and report
  `pred_nan_when_obs_valid`;
- incomplete or overlapping fold coverage raises before any store is written;
- a model runner raises on unknown keyword arguments, and never reads split files or hard-codes dates.

## Verify before a PR

```bash
ruff check .
python -m hydromip.tools.ci_guardrails --mode all
pytest <the suites you touched>
```
