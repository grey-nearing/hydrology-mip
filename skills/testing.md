---
name: testing
description: >-
  Test suite organization, co-located package test directories, native 531x3652
  synthetic testing requirements, strict mocking boundaries, and data-integrity
  regression testing rules for camels-us-mip. Use whenever writing, moving,
  running, or reviewing unit, integration, or benchmark tests.
---

# Testing Standards & Organization (`camels-us-mip`)

This skill defines how tests are organized, written, and reviewed across the `camels-us-mip` repository.

---

## 1. Co-Located Package Test Suites (`<package>/tests/`)

1. **Test Directory Layout:**
   - Every top-level package maintains its own co-located `tests/` directory:
     - `metrics/tests/`: Unit, integration, and data-integrity regression tests for `metrics/evaluator.py` and `metrics/cli.py`.
     - `models/tests/`: Unit and integration tests for `models/run_model.py`, `models/zarr_writer.py`, centralized split/date injection, and model runner contracts.
     - `frontend/tests/`: FastAPI backend endpoint and leaderboard verification tests.
     - `benchmarks/tests/`: Unit tests for the canonical benchmark harnesses in `benchmarks/`.
2. **Pytest & CI Configuration:**
   - Register every test directory in `pyproject.toml` under `[tool.pytest.ini_options] testpaths`.
   - Use standard Pytest markers registered in `pyproject.toml`:
     - `@pytest.mark.unit`: Fast, hermetic unit tests.
     - `@pytest.mark.integration`: End-to-end pipeline integration tests.
     - `@pytest.mark.slow`: Computationally intensive tests.
     - `@pytest.mark.gpu`: Tests requiring CUDA GPU hardware.
     - `@pytest.mark.gcs`: Live tests reading from `gs://camels-us-mip/`.

---

## 2. Native-Resolution Synthetic Testing (No Toy-Dimension Monkeypatching)

- **Always Test Hydrograph & Split Pipelines at Native Benchmark Dimensions:**
  - Hydrograph Zarr writers, split validators, and `metrics/evaluator.py` must be tested at the real production CAMELS-US MIP dimensions: **`531` basins $\times$ `3,652` daily steps** (`1989-10-01` to `1999-09-30`) using real Zarr stores written to `tmp_path` (or an in-memory/local fsspec store mapped to `gs://`).
  - **Never monkeypatch** `NUM_BASINS = 531` or `NUM_EVAL_DAYS = 3652` to tiny toy sizes (`3 basins x 10 days`) in tests. Testing at full `(531, 3652)` takes milliseconds in NumPy/Zarr and catches coordinate sorting, HUC-02 regional aggregation, and fold-stitching bugs that toy dimensions hide.

---

## 3. Strict Mocking Boundaries

- **Only Mock External Network or Cloud Transport Boundaries When Running Hermetic Unit Tests:**
  - When testing `gs://camels-us-mip/` URI resolution hermetically without network access, route `gs://camels-us-mip/...` to a real Zarr store on `tmp_path` at the `fsspec`/`zarr` storage-mapper boundary—never mock `xr.open_zarr` to return a fake in-memory object that bypasses actual Zarr serialization and coordinate decoding.
- **Never Mock Internal Logic Under Test:**
  - Never mock `calculate_model_leaderboard_metrics`, Zarr coordinate validation, NSE/KGE/FHV mathematical formulas, or fold-stitching masks. Write real Zarr stores to `tmp_path` and assert on exact numerical outputs.

---

## 4. Required Data-Integrity Regression Tests

Every module and PR must include explicit negative and edge-case tests verifying our algorithmic norms:

1. **Missing & Corrupt Inputs Fail Loudly (Zero Silent Fallbacks):**
   - Missing Zarr stores on GCS, missing `qsim` variable, wrong dimensions (e.g., `(530, 3652)` or `(531, 3650)`), unsorted or mismatched `basin` IDs, or wrong `date` coordinates raise `FileNotFoundError`, `KeyError`, or `ValueError` immediately.
2. **Predicted `NaN` on Valid Observation Day Fails Loudly:**
   - Injecting even a single `np.nan` into `qsim` at a `(basin, date)` entry where `qobs` is valid (`>= 0.0`) must cause `calculate_model_leaderboard_metrics` to raise a `ValueError` reporting `pred_nan_when_obs_valid`.
3. **Incomplete Cross-Validation Fold Coverage Fails Loudly:**
   - In `models/run_model.py`, if a 12-fold PUB or 18-fold PUR run leaves any of the 531 basins uncovered or produces overlapping test fold assignments, it must raise `ValueError` before writing any Zarr store.
4. **Model Runner Split/Date Independence:**
   - Verify that individual model runners in `models/<model_name>/` accept arbitrary `train_basins`, `test_basins`, `train_dates`, and `eval_dates` passed by `models/run_model.py` and never read split files or hardcode dates internally.
