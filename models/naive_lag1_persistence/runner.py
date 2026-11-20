# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold runner for the zero-parameter Naive Lag-1 Persistence lower benchmark.

Valid only on the ``lagged_streamflow`` leaderboards: the prediction for day ``t`` is the observed
discharge of the *test* basin on day ``t - 1`` (or the most recent earlier observation when that day
is missing). ``models/run_model.py`` enforces the forcing regime through the model registry, and the
registry's ``default_kwargs`` pass ``cold_start_value`` explicitly so it is recorded in every store's
``runner_kwargs_json`` provenance attribute.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from models.camels_data import load_camels_slice
from models.naive_lag1_persistence.model import predict_lag1_persistence

RUNNER_DEFAULTS: dict[str, object] = {"lookback_days": 30, "cold_start_value": None}
_LAUNCHER_PASSTHROUGH: frozenset[str] = frozenset({"fold_id", "fold_index"})


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (0,),
    **kwargs: object,
) -> np.ndarray:
    """Return strictly causal lag-1 persistence predictions, shape ``(len(test_basins), n_eval_days)``.

    The model has no parameters, so ``train_basins``, ``train_period`` and ``seeds`` are unused.
    ``forcings`` only selects which forcing product the loader validates; predictions use discharge
    alone.
    """
    del train_basins, train_period, seeds
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS) - _LAUNCHER_PASSTHROUGH)
    if unknown:
        raise ValueError(f"Unknown runner kwargs for naive_lag1_persistence: {unknown}")
    cfg: dict[str, object] = dict(RUNNER_DEFAULTS)
    cfg.update({k: v for k, v in kwargs.items() if k in RUNNER_DEFAULTS})
    lookback_days = int(cfg["lookback_days"])  # type: ignore[arg-type]
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}.")
    cold_start_value = None if cfg["cold_start_value"] is None else float(cfg["cold_start_value"])  # type: ignore[arg-type]

    eval_start = pd.Timestamp(eval_period[0])
    lookback_start = eval_start - pd.Timedelta(days=lookback_days)
    # spinup_days=0 on purpose: the lagged-streamflow protocol allows observed discharge before
    # each prediction day, and the loader would otherwise mask the lookback window to NaN.
    eval_slice = load_camels_slice(
        basins=test_basins,
        period=(str(lookback_start.date()), str(eval_period[1])),
        forcings=forcings,
        data_dir=data_dir,
        spinup_days=0,
    )
    n_eval_days = len(pd.date_range(eval_start, pd.Timestamp(eval_period[1]), freq="D"))
    preds, n_cold = predict_lag1_persistence(
        eval_slice["qobs"], lookback_days=lookback_days, cold_start_value=cold_start_value
    )
    if preds.shape != (len(test_basins), n_eval_days):
        raise ValueError(f"Expected predictions of shape {(len(test_basins), n_eval_days)}, got {preds.shape}.")
    if n_cold:
        print(
            f"[naive_lag1_persistence] {n_cold} basin-day(s) had no prior observation and were set to "
            f"cold_start_value={cold_start_value} mm/day.",
            flush=True,
        )
    return preds
