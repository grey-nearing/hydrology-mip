# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold execution interface for Kratzert et al. (2019) EA-LSTM."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.kratzert_ealstm.model import EALSTM
from models.kratzert_lstm.fold import run_sequence_fold, sequence_runner_defaults

# Kratzert et al. (2019, HESS 23, 5089) EA-LSTM configuration (`ealstm_regional_modeling`):
# hidden 256, seq_length 270, dropout 0.4, batch 256, 30 epochs, Adam with learning rate
# 1e-3 (epochs 0-9), 5e-4 (10-19), 1e-4 (20-29), NSE* loss, initial forget bias 3,
# gradient-norm clipping 1.
RUNNER_DEFAULTS: dict[str, object] = sequence_runner_defaults(
    seq_len=270, epochs=30, hidden_size=256, dropout=0.4, lr=1e-3, eval_batch_size=2048
)
RUNNER_DEFAULTS["lr_schedule"] = {0: 1e-3, 10: 5e-4, 20: 1e-4}


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (123,),
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train Kratzert EA-LSTM on `train_basins` over `train_period` and predict `test_basins` over `eval_period`.

    Supported keyword overrides are the keys of `RUNNER_DEFAULTS` (unknown names raise).
    Returns the seed-ensemble mean `(n_test_basins, n_eval_days)` in mm/d, or
    `(mean, members)` when `return_members=True`.
    """
    return run_sequence_fold(
        model_cls=EALSTM,
        defaults=RUNNER_DEFAULTS,
        train_basins=train_basins,
        test_basins=test_basins,
        train_period=train_period,
        eval_period=eval_period,
        forcings=forcings,
        data_dir=data_dir,
        seeds=seeds,
        kwargs=kwargs,
    )
