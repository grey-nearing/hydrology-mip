---
name: protocol-guardrails
description: >-
  Non-negotiable HydrologyMIP benchmark protocol and data-integrity rules: frozen splits and
  windows, the central run_model.py runner contract, anti-leakage rules (including spin-up inside
  the evaluation decade), missing-data and error-handling rules, safe I/O, and how official versus
  local scores are produced. Use when writing, changing, or reviewing any code that loads data,
  defines folds or dates, trains or calibrates a model, writes hydrograph stores, or computes
  metrics.
license: Apache-2.0
metadata:
  audience: contributor
---

# Protocol and data-integrity guardrails

These rules keep the benchmark fair. CI enforces many of them
(`python -m hydromip.tools.ci_guardrails --mode all`). Reviewers enforce the rest. All paths are
relative to the repository root.

## 1. The protocol is frozen

- The source of truth for the active MIP is `mips/camels_us_daily/protocol.json`. In code, use
  `CANONICAL_TRAIN_START/END` and `CANONICAL_EVAL_START/END` from `models/zarr_writer.py`.
- Training period: `1999-10-01` to `2008-09-30`. Evaluation period: `1989-10-01` to `1999-09-30`
  (3,652 days).
- Basins: the 531 IDs in `mips/camels_us_daily/splits/basins_531.txt`, in sorted order. Every one of
  them has complete observations in the evaluation period, so **no basin is ever excluded** from
  scoring.
- Spatial regimes: `gauged` (1 fold), `pub_12fold` (12 random folds), and `pur_18huc02` (18
  leave-one-HUC-02-region-out folds; files in `splits/pur_18fold/`).
- Never edit `protocol.json` or anything under `splits/` for an active MIP.

## 2. The central runner contract

`models/run_model.py` is the only code that loads basin lists and fold files, defines the windows,
loops over folds and seeds, stitches out-of-fold predictions, and writes the store. A model folder
provides one function:

```python
def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (123,),
    **kwargs: object,  # always receives fold_id and fold_index; raise on any other unknown key
) -> np.ndarray:  # float32, shape (len(test_basins), n_eval_days), mm/day, rows in test_basins order
```

- It is called **once per seed** with `seeds=(seed,)`. Never average seeds inside `run_fold`. The
  runner stores every seed in `qsim_members` and their mean in `qsim`.
- Never replace NaN predictions with numbers. The runner rejects non-finite output on purpose.
- Model code never reads split files and never hard-codes the canonical dates. CI checks this.
  Exception: `reproduce_author_metrics.py` may hold the *author's* dates, and it imports the
  canonical ones from `models.zarr_writer`.
- Each `ModelRegistryEntry` declares `allowed_forcing_regimes` and `allowed_spatial_regimes`. The
  runner refuses any other combination.

## 3. Data access

- Load data only with
  `models.camels_data.load_camels_slice(basins, period, forcings, data_dir, spinup_days=0, require_valid_targets=False)`.
  Use `require_valid_targets=True` for training slices.
- Compute standardization statistics only with `compute_train_stats(train_slice)`. It uses the
  target window of the **training** slice only.
- Forcings are `daymet`, `maurer`, `nldas`. One product means `single_camels`; two or three mean
  `multi_camels`. Pass them separated by spaces on the command line.

## 4. No leakage

- Scalers, per-basin loss weights, early stopping, learning-rate schedules, hyperparameter search,
  architecture choice, and donor-basin selection use **training basins and the training period
  only**. Take any validation split out of `1999-10-01`..`2008-09-30`.
- In `pub_12fold` and `pur_18huc02`, the held-out basins of a fold are never used to fit anything
  in that fold.
- Spin-up: a model may run up to 365 days of forcings before a window. Discharge inside the spin-up
  is NaN; the loader masks it. Never unmask it. **The training spin-up (`1998-10-01`..`1999-09-30`)
  lies inside the evaluation decade**, so only forcings may be used there.
- A model that reads observed discharge of the test basins as an input is allowed only on
  `lagged_streamflow` leaderboards. Models that use non-CAMELS forcings, extra attributes, or
  flow-derived features go to `freeform` leaderboards.
- ERA notebooks under `mips/camels_us_daily/era_experiments/` never use evaluation-period
  discharge. They train on `2003-10-01`..`2008-09-30` and select on `1999-10-01`..`2003-09-30`. For
  PUB and PUR they loop over all canonical folds.

## 5. Missing data and errors

- Missing data in means NaN out, or a clear error. Never fill gaps with `0.0`, forward-fill,
  backward-fill, interpolation, or climatology.
- Never drop a prediction NaN on a day with a valid observation. The evaluator counts these as
  `pred_nan_when_obs_valid`, and the count must be 0.
- No fallback data paths, no alternate file names, no "try this key, then that key".
- No `try`/`except`/`finally`, `contextlib.suppress`, `warnings.catch_warnings`, or
  `ignore_errors=True` in library, model, or API code. Check preconditions explicitly
  (`Path.exists()`, key and shape checks) and raise `ValueError`, `KeyError`, or `FileNotFoundError`
  with a clear message.

## 6. Safe I/O

- `np.load(..., allow_pickle=False)` and `torch.load(..., weights_only=True)`, always explicit.
- No hard-coded `/tmp` paths, home directories, or machine-specific paths. No secrets.
- Write stores only with `models.zarr_writer.write_hydrograph_zarr`. Never write to
  `gs://camels-us-mip/`; that is for maintainers only.

## 7. Scores

- All 14 metrics from Gauch et al. (2023) come from `hydromip/metrics/deterministic_daily.py`
  through `metrics.evaluator.calculate_model_leaderboard_metrics`. In order: `NSE`, `MSE`, `RMSE`,
  `KGE`, `logNSE`, `logKGE`, `Alpha-NSE`, `Pearson-r`, `Beta-KGE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`,
  `Peak-Timing`. `logNSE` and `logKGE` are NSE and KGE computed on `log(max(Q, 0) + 1e-5)` of the
  observed and simulated flows. Leaderboards show 16 summary columns: the 531-basin median of all
  14 metrics, plus the mean NSE and mean KGE. Never type or copy a metric by hand.
- **Official** leaderboard scores are computed by maintainers from the canonical store.
- **Contributors self-check** with the same code against a local copy of the layout:
  `python -m metrics.build_eval_targets ... --output ./local_bucket/observations/newman_eval_targets.zarr`,
  then `mip-score-model ... --bucket-uri ./local_bucket`. See `skills/contribute-model/SKILL.md`.
- Ensemble and single-model results are separate leaderboard rows. Single-model skill is the mean
  over all seeds, never seed 0 alone. See `skills/leaderboard-norms/SKILL.md`.

## Verify

```bash
ruff check .
python -m hydromip.tools.ci_guardrails --mode all
pytest models/tests metrics/tests
```

## Stop and ask a human if

- a change would alter a split, a window, or `protocol.json`;
- a model needs test-basin discharge outside `lagged_streamflow`;
- you believe a basin, a day, or a seed should be excluded;
- a rule here seems to block legitimate scientific work. Open an issue; do not work around it.
