# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold execution interface for Rasiya Koya et al. (2023) Temporal Fusion Transformer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.kratzert_lstm.fold import run_sequence_fold, sequence_runner_defaults
from models.rasiya_koya_tft.model import TemporalFusionTransformerHydro

# TFT configuration used for the MIP re-run: hidden 128, 4 heads, dropout 0.15, 180-day
# encoder window, batch 256, 25 epochs, Adam {0: 1e-3, 10: 5e-4, 25: 1e-4} (the last step is
# never reached with 25 epochs), NSE* loss, gradient-norm clipping 1.  The evaluation batch is
# smaller than for the LSTMs because self-attention materializes (batch, heads, 180, 180).
RUNNER_DEFAULTS: dict[str, object] = sequence_runner_defaults(
    seq_len=180, epochs=25, hidden_size=128, dropout=0.15, lr=1e-3, eval_batch_size=256
)


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
    """Train the TFT on `train_basins` over `train_period` and predict `test_basins` over `eval_period`.

    Supported keyword overrides are the keys of `RUNNER_DEFAULTS` (unknown names raise).
    Returns the seed-ensemble mean `(n_test_basins, n_eval_days)` in mm/d, or
    `(mean, members)` when `return_members=True`.
    """
    return run_sequence_fold(
        model_cls=TemporalFusionTransformerHydro,
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
