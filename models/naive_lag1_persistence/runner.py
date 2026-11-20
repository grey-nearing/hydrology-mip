# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold runner for the zero-parameter Naive Lag-1 Persistence lower benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.camels_data import load_camels_slice
from models.naive_lag1_persistence.model import predict_lag1_persistence


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (0,),
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Produce strictly causal lag-1 streamflow persistence predictions on `test_basins`."""
    del train_basins, train_period
    return_members = bool(kwargs.pop("return_members", False))
    spinup_days = max(1, int(kwargs.pop("spinup_days", 365)))  # type: ignore[arg-type]
    forcing_list = list(forcings) if forcings else ["daymet"]

    eval_slice = load_camels_slice(
        data_dir=data_dir,
        basins=test_basins,
        start_date=eval_period[0],
        end_date=eval_period[1],
        forcings=forcing_list,
        spinup_days=spinup_days,
    )
    preds = predict_lag1_persistence(
        eval_slice["qobs"],
        spinup_days=int(eval_slice["spinup_days"]),
    )
    if return_members:
        n_seeds = max(1, len(seeds))
        members = np.repeat(preds[np.newaxis, :, :], n_seeds, axis=0)
        return preds, members
    return preds
