---
name: community-contribution
description: End-to-end protocol and engineering guide for contributing a community hydrological model (`Contribution`) or a benchmark re-run (`MIP Re-run`) to the CAMELS-US Model Intercomparison Project (MIP). Use when adding a new model implementation to `models/<model_name>/`, packaging external `(531, 3652)` hydrographs into Zarr v2 on `gs://camels-us-mip/`, scoring all 12 Gauch et al. (2023) hydrological metrics, updating `leaderboards/*.json`, or regenerating manuscript tables and figures.
---

# Community Model Contribution Skill (`camels-us-mip`)

Use this skill whenever an author, contributor, or AI agent is contributing a new hydrological model (`Contribution`) or re-running an existing benchmark architecture (`MIP Re-run`) for the **CAMELS-US Model Intercomparison Project (MIP)**.

---

## 1. The Four Rules of Responsible Benchmarking (Protocol Requirements)

Every community contribution and MIP re-run is governed by the **Four Rules of Responsible Benchmarking** defined in Section 3.1 of the CAMELS-US MIP manuscript:

### Rule 1: Strict Adherence to Temporal and Spatial Splits
- **Canonical Basin Set (`531` Basins)**:
  - All evaluations use the exact **531 canonical Newman et al. (2017) basins** listed in [`mips/camels_us_daily/splits/basins_531.txt`](../mips/camels_us_daily/splits/basins_531.txt) (`gs://camels-us-mip/splits/basins_531.txt`), stored in strictly sorted 8-character zero-padded USGS gauge ID order (`dtype='<U8'`, e.g., `'01013500'`).
  - **100% Basin Completeness**: All 531 basins have `3,652 / 3,652` valid USGS observations during the evaluation period (`1989-10-01` to `1999-09-30`) and `3,288 / 3,288` valid observations during the training period (`1999-10-01` to `2008-09-30`). Omitting any basin—even a single catchment—is strictly rejected by the validator (`missing_basins == []`).
  - *(Note on Metric Aggregations)*: You must submit all **531** basins with zero `NaN` predictions (`pred_nan_when_obs_valid == 0`). During metric aggregation, [`metrics/evaluator.py`](../metrics/evaluator.py) automatically excludes basin `09535100` (*San Pedro River at Charleston, AZ*, listed in `CONSTANT_ZERO_OBS_BASINS`) from division-by-variance / zero-denominator summaries (`num_evaluated_basins == 530`), because its USGS observation record is constant zero (`0.0 mm/day`) across all 3,652 evaluation days.
- **Temporal Split (Newman et al., 2017 / Kratzert et al., 2019)**:
  - **Training / Calibration Period**: `1999-10-01` to `2008-09-30` (`3,288` daily steps; 9 water years WY2000–WY2008). Models requiring state warm-up may use 1 year (`365` days) of meteorological spin-up prior to `1999-10-01` (`1998-10-01` to `1999-09-30`), during which `qobs` is masked as `NaN` so spin-up days never enter the training loss.
  - **Blind Evaluation / Test Period**: `1989-10-01` to `1999-09-30` (`3,652` daily steps; 10 water years WY1990–WY1999). Models may use 1 year (`365` days) of meteorological spin-up prior to `1989-10-01` (`1988-10-01` to `1989-09-30`), during which `qobs` is masked as `NaN` and predictions are sliced off before returning `(len(test_basins), 3652)`.
  - **Zero Evaluation-Period Leakage**: All hyperparameter tuning, learning-rate scheduling, early stopping, donor-basin selection rules, and architecture search **must** be conducted using a validation split carved strictly out of `1999-10-01` to `2008-09-30` (for example, training on WY2000–WY2006 `1999-10-01` to `2006-09-30` and validating on WY2007–WY2008 `2006-10-01` to `2008-09-30`, or using `ERA_TRAIN_MASK` / `ERA_EVAL_MASK` in [`models/run_model.py`](../models/run_model.py)). Checking `1989-10-01` to `1999-09-30` metrics to select epochs, hyperparameters, or architectures invalidates the submission.
- **Spatial Split Regimes ($3$ Regimes)**:
  1. **`gauged` (Gauged Temporal Generalization — 1 Fold)**: Train/calibrate on all 531 basins during `1999-10-01` to `2008-09-30`; evaluate on all 531 basins during `1989-10-01` to `1999-09-30`.
  2. **`pub_12fold` (Ungauged Random Cross-Validation / PUB — 12 Folds)**: Defined in [`mips/camels_us_daily/splits/pub_12fold_random_splits.json`](../mips/camels_us_daily/splits/pub_12fold_random_splits.json) (`gs://camels-us-mip/splits/pub_12fold_random_splits.json`, deterministic `np.random.RandomState(42)` partition into 3 folds of 45 test basins and 9 folds of 44 test basins). For each fold $k \in \{0, \dots, 11\}$, train exclusively on `train_basins` during `1999-10-01` to `2008-09-30` and predict on the held-out `test_basins` during `1989-10-01` to `1999-09-30`. Stitch the out-of-sample test predictions across all 12 folds into a single `(531, 3652)` array ordered by `mips/camels_us_daily/splits/basins_531.txt`.
  3. **`pur_18huc02` (Regional Holdout / PUR — 18 Folds)**: Defined in [`mips/camels_us_daily/splits/pur_18fold_huc02_splits.json`](../mips/camels_us_daily/splits/pur_18fold_huc02_splits.json) (`gs://camels-us-mip/splits/pur_18fold_huc02_splits.json`). Each fold holds out an entire USGS 2-digit Hydrologic Unit Code (`HUC-02` region `01` through `18`). Train on the remaining 17 regions during `1999-10-01` to `2008-09-30`, predict on the held-out region during `1989-10-01` to `1999-09-30`, and stitch all 18 regional predictions into a single `(531, 3652)` array ordered by `mips/camels_us_daily/splits/basins_531.txt`.
  - **Normalization & Transfer Hygiene in `pub_12fold` / `pur_18huc02`**: Mean/standard-deviation scalers for forcings, static catchment attributes, and target streamflow, as well as donor-transfer similarity matrices, must be computed **strictly from `train_basins` during `1999-10-01` to `2008-09-30`** inside each fold.

### Rule 2: Standardized 12-Leaderboard Taxonomy ($3 \text{ Spatial} \times 4 \text{ Input/Forcing Regimes}$) & Competing Ensemble/Single Rows
Every model run is routed to one of the **12 canonical leaderboards** defined in [`metrics/evaluator.py`](../metrics/evaluator.py) (`CANONICAL_LEADERBOARD_IDS`) and [`skills/model-compliance-and-leaderboard-norms.md`](./model-compliance-and-leaderboard-norms.md):

| Spatial Regime | Single CAMELS Forcing (`single_camels`: Daymet, Maurer, or NLDAS) | Multi-CAMELS Forcing (`multi_camels`: $\ge 2$ of Daymet, Maurer, NLDAS) | Lagged Streamflow / Data Assimilation (`lagged_streamflow`: $Q(t-k)$, AR, DA) | Freeform / External Inputs (`freeform`: NLDAS-2, AORC, ERA5, or auxiliary flow quantiles) |
| :--- | :--- | :--- | :--- | :--- |
| **Gauged (`gauged`)** | `gauged__single_camels` (Table 3) | `gauged__multi_camels` (Table 4) | `gauged__lagged_streamflow` (Table 5) | `gauged__freeform` (Table 6) |
| **Ungauged Random (`pub_12fold`)** | `pub__single_camels` (Table 7) | `pub__multi_camels` (Table 8) | `pub__lagged_streamflow` (Table 9) | `pub__freeform` (Table 10) |
| **Regional Holdout (`pur_18huc02`)** | `pur__single_camels` (Table 11) | `pur__multi_camels` (Table 12) | `pur__lagged_streamflow` (Table 13) | `pur__freeform` (Table 14) |

- **Ensemble vs. Single-Model Competing Rows (`Option B`):**
  Whenever a model architecture is evaluated across $N \ge 2$ random seeds (`qsim_members` of shape `(N, 531, 3652)`), it appears as **two separate, competing rows** on the same leaderboard:
  1. **Ensemble Row (`is_ensemble: true`, `entry_id: "<base_id>"`, `model_name: "<Base Name> (<N>-model ensemble)"`)**: Scored and ranked on the ensemble-mean daily hydrograph `qsim = np.mean(qsim_members, axis=0)`.
  2. **Single-Model Row (`is_ensemble: false`, `entry_id: "<base_id>_single"`, `model_name: "<Base Name>"`)**: Scored and ranked using the **mean ($\pm$ std) across all $N$ individual seed members** (`qsim_members[0 .. N-1]`) across all 12 Gauch et al. (2023) metrics—never by `member[0]` (`seed 0`) alone.
- **Strictly One Canonical Citation & Strictly One Provenance Category per Model Entry:**
  Every model entry in `leaderboards/table_2_registry.json` has **strictly one canonical citation** (the first paper to run the model over the Newman split, or the first paper to run it on CAMELS data) and **strictly one provenance category** (`Archive`, `Archive*`, `Literature`, `Literature*`, `MIP Re-run (Our Code)`, `MIP Re-run (Author Code)`, or `Contribution`). If a model was published on the Newman split in a given configuration, we report the published `Literature` or `Archive` values and never replace them with a MIP re-run; any additional or competing `MIP Re-run (Our Code)` runs live under a separate `<family_id>_rerun` model entry.
- **No Unauthored Configurations for External `MIP Re-run` Models ("If They Didn't Run It, We Don't Run It"):**
  For `MIP Re-run` models from external authors, **only evaluate and list configurations (forcing regime and spatial split regime) that the original authors actually defined and ran in their paper**. Never invent multi-forcing (`multi_camels` / `_3f`) or spatial generalization (`pub_12fold`, `pur_18huc02`) adaptations for an external author's model if they did not run that configuration in their original paper. The only exception is when Grey Nearing was an author/co-author on the original paper (e.g., Kratzert et al., 2019/2021; Nearing et al., 2020; Hoedt et al., 2021; Frame et al., 2023) or explicitly authorizes a specific evaluation. See [`skills/model-compliance-and-leaderboard-norms.md`](./model-compliance-and-leaderboard-norms.md) for full details.

- **Model Categories (for Table 2 Registry and Leaderboard Color-Coding)**:
  Every model must declare one of the 4 canonical architectural categories in `leaderboards/table_2_registry.json`:
  1. **`Machine learning`** (shaded soft blue `#e8f1fb` in manuscript tables)
  2. **`Hybrid (ML + Physics)`** (shaded soft purple `#f2ebf9` in manuscript tables)
  3. **`Conceptual`** (shaded soft orange `#fff0e0` in manuscript tables)
  4. **`Process-based`** (shaded soft green `#e8f6ef` in manuscript tables)

- **Provenance / Source Categories (`provenance_category` in Zarr & JSON)**:
  - **`Contribution`**: New community-contributed or MIP-contributed model architectures evaluated on the strict MIP protocol with full `(531, 3652)` Zarr hydrographs in `gs://camels-us-mip/hydrographs/`.
  - **`MIP Re-run (Our Code)`** / **`MIP Re-run (Author Code)`**: Published benchmark architectures re-implemented (`Our Code`) or re-executed from the original authors' repository (`Author Code`) under the strict MIP protocol (requiring a Stage-1 verification run in `reproduce_author_metrics.py` that reproduces the original paper's reported metrics on its original split before running Stage-2 on the MIP protocol). Reported with the original authors' single canonical citation and original publication year.
  - **`Archive`** and **`Archive*`**: Pre-existing public hydrograph archives scored by `metrics.evaluator` (flagged as `Archive*` with `asterisk: true` if basin coverage is `< 531` or evaluation window differs from the Newman decade).
  - **`Literature`** and **`Literature*`**: Historical paper-reported metrics extracted directly from a published paper without re-scoring daily hydrographs (`NSE` only; all other metric cells are `--`; flagged as `Literature*` with `asterisk: true` if evaluated on a 9-water-year window such as `WY1991–1999`).

### Rule 3: Multi-Metric Hydrological Evaluation (All 12 Gauch et al., 2023 Metrics)
Never hand-type or hardcode metrics in JSON or Markdown. All metrics for `MIP Re-run` and `Contribution` models are computed automatically from the consolidated Zarr store on `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` by [`metrics.calculate_model_leaderboard_metrics`](../metrics/evaluator.py) (*"In Defense of Metrics"*, Gauch et al., 2023):
1. `NSE` (Median & Mean across evaluated basins; higher is better, optimal $= 1$)
2. `KGE` (Gupta et al., 2009 Kling-Gupta Efficiency, Median & Mean; higher is better, optimal $= 1$)
3. `MSE` (Mean Squared Error in $(\text{mm/day})^2$; lower is better, optimal $= 0$)
4. `RMSE` (Root Mean Squared Error in $\text{mm/day}$; lower is better, optimal $= 0$)
5. `Pearson-r` (Linear correlation coefficient $r$; higher is better, optimal $= 1$)
6. `Alpha-NSE` ($\alpha_{\text{NSE}} = \sigma_{\text{sim}} / \sigma_{\text{obs}}$; optimal $= 1$)
7. `Beta-NSE` ($\beta_{\text{NSE}} = (\mu_{\text{sim}} - \mu_{\text{obs}}) / \sigma_{\text{obs}}$; optimal $= 0$)
8. `Beta-KGE` ($\beta_{\text{KGE}} = \mu_{\text{sim}} / \mu_{\text{obs}}$; optimal $= 1$)
9. `FHV` (Top 2% peak flow bias in %; optimal $= 0\%$)
10. `FMS` (Mid-segment FDC slope bias between 20% and 70% exceedance in %; optimal $= 0\%$)
11. `FLV` (Bottom 30% low-flow log-space volume bias in %; optimal $= 0\%$)
12. `Peak-Timing` (Mean absolute peak timing lag in days across detected flood events; optimal $= 0\text{ days}$)
13. Plus `NSE<=0` count (number of basins with $\text{NSE} \le 0$) and paired two-sided Wilcoxon signed-rank $p$-values (`p_value_vs_sota_nse`, `p_value_vs_sota_kge`).

### Rule 4: Open Hydrograph Archiving, Open Code, and Architectural Documentation
Every community contribution requires:
1. **Consolidated Zarr v2 Hydrographs** in `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` (plus a public DOI archive link in `archive_url` for external archives, or the GCS URI `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`).
2. **Open-Source Code** (`code_url` pointing to the GitHub repository or `https://github.com/grey-nearing/camels-us-mip/tree/main/models/<model_name>`).
3. **Formal Mathematical & Architectural Documentation** (recorded in `leaderboards/table_2_registry.json`, `models/<model_name>/`, and Appendix A or Appendix B of the manuscript).

---

## 2. Two Contribution Pathways

Contributors can add a model to `camels-us-mip` via **Pathway A** (implementing runnable model code inside `models/<model_name>/` and executing via `models/run_model.py`) or **Pathway B** (converting pre-computed `(531, 3652)` daily predictions from an external repository/archive into a canonical Zarr store and submitting via CLI or Web UI).

---

### Pathway A: Implementing a Runnable Model in `models/<model_name>/`

Use Pathway A when adding a new model directory to [`models/`](../models/) so that [`models/run_model.py`](../models/run_model.py) can train, evaluate, stitch folds, and write the Zarr store automatically.

#### Step A1: Create `models/<model_name>/`
Follow [`skills/repo-organization.md`](repo-organization.md) and [`skills/algorithm-rules-and-norms.md`](algorithm-rules-and-norms.md):
```text
models/<model_name>/
├── __init__.py
├── model.py                     # Core model architecture class(es) and equations
├── runner.py                    # Exposes run_fold(...) called by models/run_model.py
└── reproduce_author_metrics.py  # Required ONLY for MIP Re-run of a published paper
```

#### Step A2: Implement `run_fold(...)` in `models/<model_name>/runner.py`
- **Strict Separation of Responsibilities**:
  - **NEVER** read split files (`splits/*.json`), hardcode date strings (`1989-10-01`), or write Zarr/CSV files inside `models/<model_name>/`.
  - **NEVER** average predictions across random seeds inside `run_fold()`. [`models/run_model.py`](../models/run_model.py) invokes `run_fold(..., seeds=(single_seed,))` once per seed, collects all $N_{\text{members}}$ member arrays into `qsim_members` of shape `(N_members, 531, 3652)`, and computes the ensemble mean `qsim = np.mean(qsim_members, axis=0)` centrally.
  - Always load CAMELS-US forcings, observations, and static catchment attributes through [`models.camels_data.load_camels_slice`](../models/camels_data.py):
    ```python
    from __future__ import annotations
    from pathlib import Path
    from typing import Sequence
    import numpy as np
    from models.camels_data import load_camels_slice


    def run_fold(
        train_basins: Sequence[str],
        test_basins: Sequence[str],
        train_period: tuple[str, str],
        eval_period: tuple[str, str],
        forcings: Sequence[str],
        data_dir: str | Path,
        seeds: Sequence[int] = (123,),
        epochs: int = 30,
        spinup_days: int = 0,
        era_train_mask: np.ndarray | None = None,
        era_eval_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Train/calibrate on train_basins in train_period; predict test_basins in eval_period.

        Returns
        -------
        np.ndarray
            Daily streamflow predictions in mm/day, dtype float32,
            strictly shaped (len(test_basins), 3652) for the eval_period
            (excluding any spinup_days prefix), with zero NaNs and all values >= 0.0.
        """
        # 1. Load training slice (with optional spinup_days=365; spin-up qobs is NaN)
        train_slice = load_camels_slice(
            data_dir=data_dir,
            basins=train_basins,
            start_date=train_period[0],
            end_date=train_period[1],
            forcings=forcings,
            spinup_days=spinup_days,
        )
        # 2. Load evaluation slice (with optional spinup_days=365)
        eval_slice = load_camels_slice(
            data_dir=data_dir,
            basins=test_basins,
            start_date=eval_period[0],
            end_date=eval_period[1],
            forcings=forcings,
            spinup_days=spinup_days,
        )
        # 3. Fit normalization scalers ONLY on train_slice (never eval_slice!)
        # 4. Train/calibrate model using seeds[0]
        # 5. Predict on eval_slice, slice off spinup_days if spinup_days > 0:
        #    preds_eval = preds_full[:, spinup_days:] -> shape (len(test_basins), 3652)
        # 6. Clip negative discharge to 0.0 and return float32 array of shape (len(test_basins), 3652)
        return np.clip(preds_eval.astype(np.float32), 0.0, None)
    ```

#### Step A3: Register the Model in [`models/run_model.py`](../models/run_model.py)
Add an entry to `MODEL_REGISTRY` in [`models/run_model.py`](../models/run_model.py):
```python
"<model_key>": {
    "module": "models.<model_name>.runner",
    "display_name": "<Human-Readable Display Name>",
    "family_id": "<family_id>",
    "category": "<Machine learning | Hybrid (ML + Physics) | Conceptual | Process-based>",
    "provenance_category": "Contribution",  # or "MIP Re-run"
    "code_url": "https://github.com/grey-nearing/camels-us-mip/tree/main/models/<model_name>",
},
```

#### Step A4: Stage-1 Verification (Required Only for `MIP Re-run` Models)
If you are re-running a published benchmark paper (`MIP Re-run`), you must first implement `models/<model_name>/reproduce_author_metrics.py` and verify that your implementation reproduces the original authors' reported metrics on their original basin subset, forcing product, and date split (recorded in `leaderboards/appendix_stage1_confirmation_runs.json` and Appendix A of the manuscript) before running Stage 2 on the MIP splits.

#### Step A5: Execute via `mip-run-model` (`models/run_model.py`)
Run the centralized runner, which loops over all folds (`1`, `12`, or `18`) and seeds, stitches the `(531, 3652)` array in canonical `splits/basins_531.txt` order, validates completeness, and writes the consolidated Zarr v2 store to `gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr`:
```bash
# Single-forcing, single-seed run (eligible for both single_camels and single_camels_ensemble):
mip-run-model \
  --model <model_key> \
  --spatial-regime gauged \
  --forcings daymet \
  --seeds 123 \
  --output-zarr gs://camels-us-mip/hydrographs/gauged__single_camels/<entry_id>.zarr

# Multi-seed ensemble on a single CAMELS forcing:
mip-run-model \
  --model <model_key> \
  --spatial-regime pub_12fold \
  --forcings daymet \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --output-zarr gs://camels-us-mip/hydrographs/pub__single_camels_ensemble/<entry_id>.zarr

# Multi-CAMELS forcing ensemble (Daymet + Maurer + NLDAS):
mip-run-model \
  --model <model_key> \
  --spatial-regime pur_18huc02 \
  --forcings daymet maurer nldas \
  --seeds 101 102 103 104 105 \
  --output-zarr gs://camels-us-mip/hydrographs/pur__multi_camels/<entry_id>.zarr
```

---

### Pathway B: Packaging & Submitting External `(531, 3652)` Hydrograph Archives

Use Pathway B when a community contributor has already trained/calibrated their model in an external open-source repository (hosted on GitHub with hydrographs archived on Zenodo, HydroShare, or GCS) and wants to package their `(531, 3652)` evaluation predictions into the canonical CAMELS-US MIP Zarr v2 schema.

#### Step B1: Verify Canonical Coordinates and Shape
1. **Basins (`basin`)**: Load `splits/basins_531.txt` (`531` strings, sorted `<U8`). Every single basin in `basins_531.txt` must be present.
2. **Time (`time`)**: `3,652` daily timestamps from `1989-10-01` to `1999-09-30` (`pd.date_range("1989-10-01", "1999-09-30", freq="D")`, stored as `datetime64[ns]`).
3. **Units & Validity**: Convert discharge to **specific runoff in `mm/day`** (`float32`). If raw predictions are in $\text{ft}^3/\text{s}$ (cfs) or $\text{m}^3/\text{s}$ (cms), normalize by the USGS gauge drainage area from CAMELS-US before packaging. Verify:
   - `qsim.shape == (531, 3652)`
   - `np.isfinite(qsim).all() == True` (zero `NaN` or `Inf` entries)
   - `qsim_members.shape == (N_members, 531, 3652)` (if `is_ensemble=True`, where `np.allclose(qsim, qsim_members.mean(axis=0), atol=1e-4)` is strictly enforced).

#### Step B2: Write the Canonical Zarr v2 Store Using [`models/zarr_writer.py`](../models/zarr_writer.py)
Never build ad-hoc Zarr stores manually; always use [`build_hydrograph_dataset`](../models/zarr_writer.py) and [`write_hydrograph_zarr`](../models/zarr_writer.py), which enforce all 15 `REQUIRED_GLOBAL_ATTRS`, Zarr format v2 (`zarr_format=2`), consolidated metadata (`.zmetadata`), and canonical chunking (`basin: 531, time: 3652`):

```python
import numpy as np
from models.zarr_writer import build_hydrograph_dataset, write_hydrograph_zarr

# Load canonical 531 basins in exact sorted order
with open("splits/basins_531.txt", "r", encoding="utf-8") as f:
    basins_531 = [line.strip() for line in f if line.strip()]

# qsim: float32 array of shape (531, 3652) in mm/day for 1989-10-01..1999-09-30
# qsim_members: optional float32 array of shape (N_members, 531, 3652) if is_ensemble=True
ds = build_hydrograph_dataset(
    qsim=qsim,
    basins=basins_531,
    entry_id="CommunityModel_2026_gauged_daymet",
    model_name="CommunityModel (Daymet)",
    family_id="CommunityModel_2026",
    provenance_category="Contribution",         # "Contribution" or "MIP Re-run"
    leaderboard_id="gauged__single_camels",     # Must be one of the 12 CANONICAL_LEADERBOARDS
    spatial_regime="gauged",                    # "gauged" | "pub_12fold" | "pur_18huc02"
    forcing_regime="single_camels",             # "single_camels" | "single_camels_ensemble" | "multi_camels" | "freeform"
    forcings_used=["daymet"],
    is_ensemble=False,                          # True if N_members > 1
    num_members=1,
    code_url="https://github.com/author/community-model",
    archive_url="https://doi.org/10.5281/zenodo.XXXXXXX",
    qsim_members=None,                          # Required of shape (N_members, 531, 3652) when is_ensemble=True
    extra_attrs={
        "category": "Hybrid (ML + Physics)",    # "Machine learning" | "Hybrid (ML + Physics)" | "Conceptual" | "Process-based"
        "citation": "Author et al. (2026)",
    },
)

write_hydrograph_zarr(
    ds,
    "gs://camels-us-mip/hydrographs/gauged__single_camels/CommunityModel_2026_gauged_daymet.zarr",
)
```

> **Alternative (Zero-Clone Browser / REST API Submission)**:
> Contributors with a local `.zarr.zip`, `.nc` (NetCDF), or `.npz` file containing `qsim (531, 3652)`, `basin (531,)`, and `time (3652,)` can upload directly via the web frontend (`frontend/backend/app.py`):
> - `POST /api/score`: Validates all 531 basins and 3,652 days in memory and returns all 12 Gauch et al. (2023) metrics plus live leaderboard rank without persisting.
> - `POST /api/submit`: Writes the validated Zarr v2 store to `gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr`, updates the leaderboard JSONs, and regenerates the figures.

---

## 3. Scoring, Updating Leaderboards, and Regenerating Figures

Once the consolidated Zarr v2 store is written to `gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr`, run the automated scoring and leaderboard update pipeline.

### Step 1: Score the Zarr Store Against `gs://camels-us-mip/observations/newman_eval_targets.zarr`
Run `mip-score-model` ([`metrics/evaluator.py`](../metrics/evaluator.py)):
```bash
mip-score-model \
  --model-id <entry_id> \
  --leaderboard-id <leaderboard_id> \
  --bucket-uri gs://camels-us-mip
```
Verify the JSON output:
- `"status": "SCORED"`
- `"num_Submitted_basins": 531` and `"num_evaluated_basins": 530` (531 basins verified with zero `NaN` predictions on valid observation days; constant-zero observation basin `09535100` excluded from variance-normalized summary statistics)
- All 12 Gauch et al. (2023) metrics (`NSE`, `KGE`, `MSE`, `RMSE`, `Pearson-r`, `Alpha-NSE`, `Beta-NSE`, `Beta-KGE`, `FHV`, `FMS`, `FLV`, `Peak-Timing`) and `NSE<=0` are finite numbers.

### Step 2: Update Leaderboards & Regenerate Manuscript Figures via `mip-update-leaderboard`
Run `mip-update-leaderboard` ([`leaderboards/update_leaderboard.py`](../leaderboards/update_leaderboard.py)):
```bash
mip-update-leaderboard \
  --model-id <entry_id> \
  --leaderboard-id <leaderboard_id> \
  --bucket-uri gs://camels-us-mip \
  --leaderboards-dir leaderboards
```
What `mip-update-leaderboard` (`trigger_submission_from_gcs`) does automatically:
1. Pulls the Zarr store from `gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr` and computes all 12 Gauch et al. (2023) metrics and paired two-sided Wilcoxon signed-rank $p$-values (`p_value_vs_sota_nse`, `p_value_vs_sota_kge`).
2. Upserts the entry into `leaderboards/<leaderboard_id>.json` and re-ranks all entries by **Median NSE** (descending, tie-broken by Mean NSE).
3. **Dual-Eligibility Routing (`single_camels` $\rightarrow$ `single_camels_ensemble`)**: If a submission is a single-seed, single-CAMELS model (`<regime>__single_camels`, `is_ensemble=False`), remember that every single-forcing single model is also a 1-member contender on `<regime>__single_camels_ensemble` (Tables 4, 8, and 12). Ensure the model is also registered/scored on `<regime>__single_camels_ensemble` if applicable.
4. Rebuilds the consolidated `mips/camels_us_daily/leaderboards/tables_3_to_14.json` and updates `mips/camels_us_daily/leaderboards/contributions_index.json` and `mips/camels_us_daily/leaderboards/table_2_registry.json`.
5. Automatically calls [`leaderboards/generate_figures.py`](../leaderboards/generate_figures.py) (`mip-generate-figures`) to regenerate:
   - `mips/camels_us_daily/figures/figure_1_water_year_coverage.{png,pdf}`
   - `mips/camels_us_daily/figures/figure_2_leaderboard_progress_all_regimes.{png,pdf}`

> **Pre-Commit / CI Hook**: You can also run [`bash leaderboards/on_community_commit.sh`](../leaderboards/on_community_commit.sh) at any time to synchronize `mips/camels_us_daily/leaderboards/tables_3_to_14.json` and regenerate `mips/camels_us_daily/figures/` from the 12 per-leaderboard JSON files.

### Step 3: Register Architectural Metadata in `mips/camels_us_daily/leaderboards/table_2_registry.json`
Ensure the model family is documented in [`mips/camels_us_daily/leaderboards/table_2_registry.json`](../mips/camels_us_daily/leaderboards/table_2_registry.json) with:
- `family_id`, `display_name`, and `citation`
- `category`: One of `"Machine learning"`, `"Hybrid (ML + Physics)"`, `"Conceptual"`, or `"Process-based"`
- `provenance_category`: `"Contribution"` or `"MIP Re-run"`
- `spatial_regimes`, `forcing_regimes`, `forcings`, `ensemble_size`, `code_url`, and `archive_url`

---

## 4. Pre-PR Verification Checklist

Before opening a Pull Request or committing a new community model contribution, verify every item below (see also [`skills/pr-review.md`](pr-review.md) and [`skills/testing.md`](testing.md)):

- [ ] **1. Split & Leakage Audit**:
  - Training/calibration used **only** `1999-10-01` to `2008-09-30` (plus at most 365 spin-up days `1998-10-01` to `1999-09-30` with `qobs=NaN`).
  - Hyperparameters, early stopping, and architecture choices were selected **strictly within `1999-10-01` to `2008-09-30`**, never inspecting `1989-10-01` to `1999-09-30`.
  - For `pub_12fold` and `pur_18huc02`, normalization statistics and donor selection used **only** `train_basins` from `1999-10-01` to `2008-09-30` inside each fold.
- [ ] **2. Zarr Schema & 531-Basin Completeness**:
  - Consolidated Zarr v2 store exists at `gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr`.
  - `qsim` has shape `(531, 3652)`, `dtype=float32`, units `mm/day`, zero `NaN`s, and exact `splits/basins_531.txt` ordering.
  - When `is_ensemble=True`, `qsim_members` has shape `(N_members, 531, 3652)` and `np.allclose(qsim, qsim_members.mean(axis=0), atol=1e-4)` passes.
  - All 15 `REQUIRED_GLOBAL_ATTRS` in [`models/zarr_writer.py`](../models/zarr_writer.py) are populated with non-empty `code_url` and `archive_url`.
- [ ] **3. Live Metric Calculation (`mip-score-model` & `mip-update-leaderboard`)**:
  - All 12 Gauch et al. (2023) metrics and Wilcoxon $p$-values were generated by [`metrics/evaluator.py`](../metrics/evaluator.py) from the Zarr store (zero hand-edited numbers).
  - `leaderboards/<leaderboard_id>.json`, `leaderboards/tables_3_to_14.json`, `leaderboards/table_2_registry.json`, `leaderboards/contributions_index.json`, and `figures/figure_2_leaderboard_progress_all_regimes.{png,pdf}` are updated and in sync.
- [ ] **4. Linter & Full-Dimension Unit Tests**:
  - `ruff check .` passes with zero warnings or errors.
  - `pytest` passes across the entire test suite (testing at native `(531, 3652)` dimensions per [`skills/testing.md`](testing.md)).
- [ ] **5. 6-Point Forensic Model Audit & Manuscript Reconciliation (for `MIP Re-run` & Leaderboard Audits)**:
  - Verify live GCS `.zarr` provenance & worker scripts, total gradient updates ($\text{epochs} \times \text{batches\_per\_epoch}$) & window stride, line-by-line fidelity against the original authors' upstream GitHub repository, cross-table monotonicity invariants (`Ensemble >= Single`, `Gauged > PUB > PUR`, `3F >= 1F`, `Lagged > Unlagged`), the *"If They Didn't Run It, We Don't Run It"* rule, and 1-to-1 bijection across `gs://camels-us-mip/hydrographs/`, `leaderboards/*.json`, `table_2_registry.json`, `appendix_stage1_confirmation_runs.{json,md}`, and the manuscript (see Section 8 of [`skills/model-compliance-and-leaderboard-norms.md`](./model-compliance-and-leaderboard-norms.md)).
