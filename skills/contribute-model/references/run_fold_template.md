# `run_fold` template

Copy this into `models/<key>/runner.py` and fill in the two marked parts. It follows the contract in
`skills/protocol-guardrails/SKILL.md` and `models/README.md`.

```python
# Copyright 2026 <your name or organization>
# SPDX-License-Identifier: Apache-2.0

"""Fold runner for <Model Name> in HydrologyMIP."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.camels_data import compute_train_stats, load_camels_slice

# Keys that models/run_model.py always passes, plus your own tunable settings.
_LAUNCHER_KWARGS: frozenset[str] = frozenset({"fold_id", "fold_index"})
RUNNER_DEFAULTS: dict[str, object] = {"epochs": 30, "hidden_size": 128}
SPINUP_DAYS: int = 365


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (123,),
    **kwargs: object,
) -> np.ndarray:
    """Train on ``train_basins`` in ``train_period`` and predict ``test_basins`` in ``eval_period``.

    Returns:
        float32 discharge in mm/day with shape ``(len(test_basins), n_eval_days)``; rows follow
        the order of ``test_basins``.
    """
    unknown = set(kwargs) - _LAUNCHER_KWARGS - set(RUNNER_DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown run_fold keyword arguments: {sorted(unknown)}")
    if len(seeds) != 1:
        raise ValueError(f"run_fold expects exactly one seed per call, got {list(seeds)}.")
    config = {**RUNNER_DEFAULTS, **{k: v for k, v in kwargs.items() if k in RUNNER_DEFAULTS}}
    rng = np.random.default_rng(int(seeds[0]))  # seed-local; never use global RNG state

    train = load_camels_slice(
        train_basins, train_period, forcings, data_dir,
        spinup_days=SPINUP_DAYS, require_valid_targets=True,
    )
    test = load_camels_slice(test_basins, eval_period, forcings, data_dir, spinup_days=SPINUP_DAYS)
    # Statistics from the training basins and the training target window only.
    # Keys: xd_mean, xd_std, xs_mean, xs_std, y_mean, y_std, basin_stds.
    stats = compute_train_stats(train)

    # (1) Build and train the model with `config`, `rng`, `train`, and `stats`.
    #     train["qobs"][:, :SPINUP_DAYS] is NaN by design; train on train["target_qobs"].
    # (2) Simulate the whole test window (spin-up + evaluation days) into `sim_full`,
    #     shape (len(test_basins), n_days_including_spinup).
    sim_full = ...

    n_eval = int(test["n_target_days"])
    pred = np.asarray(sim_full[:, -n_eval:], dtype=np.float32)
    if pred.shape != (len(test_basins), n_eval):
        raise ValueError(f"Expected shape {(len(test_basins), n_eval)}, got {pred.shape}.")
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions: fix the model; never fill or mask values.")
    return pred
```

Notes:

- `load_camels_slice` returns `x_dyn`, `x_stat`, `qobs` (spin-up masked), `target_qobs`, `dates`,
  `target_dates`, `n_target_days`, `spinup_days`, basin metadata (`lat`, `lon`, `elev`, `area_km2`,
  `huc_02`), and the feature names. See its docstring in `models/camels_data.py`.
- PyTorch models: create the generator from the seed (`torch.Generator().manual_seed(seed)`) and
  pass it to everything random. Load weights with `torch.load(..., weights_only=True)`.
- Do not write files, print large logs, or read split files inside the runner.
