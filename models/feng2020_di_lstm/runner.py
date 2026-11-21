# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold runner for Feng, Fang & Shen (2020, WRR) Data-Integration LSTM (`DI(1)`)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from models.camels_data import load_camels_slice
from models.feng2020_di_lstm.model import DILSTMTrainer

DEFAULT_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105, 106)


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    hidden_size: int = 256,
    seq_len: int = 365,
    dropout: float = 0.5,
    batch_size: int = 100,
    epochs: int = 300,
    lr: float = 1.0,
    n_batches_per_epoch: int | None = None,
    clip_grad_norm: float = 0.0,
    eval_batch_size: int = 256,
    lag_days: int = 1,
    lag_dropout_prob: float = 0.0,
    device: str | torch.device | None = None,
    return_members: bool = False,
    fold_id: str | None = None,
    fold_index: int | None = None,
    spatial_regime: str | None = None,
    forcing_regime: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train Feng et al. (2020) `DI(1)` on `train_basins` and predict `eval_period` on `test_basins`."""
    del fold_id, fold_index, spatial_regime, forcing_regime
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    spinup_days = seq_len - 1

    train_slice = load_camels_slice(
        basins=train_basins,
        period=train_period,
        forcings=forcings,
        data_dir=data_dir,
        spinup_days=spinup_days,
        require_valid_targets=True,
    )
    test_slice = load_camels_slice(
        basins=test_basins,
        period=eval_period,
        forcings=forcings,
        data_dir=data_dir,
        spinup_days=spinup_days,
        require_valid_targets=False,
    )

    trainer = DILSTMTrainer(
        hidden_size=hidden_size,
        seq_len=seq_len,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        n_batches_per_epoch=n_batches_per_epoch,
        clip_grad_norm=clip_grad_norm,
        eval_batch_size=eval_batch_size,
        lag_days=lag_days,
        lag_dropout_prob=lag_dropout_prob,
        device=device,
    )
    qsim_mean, qsim_members = trainer.fit_and_predict(
        x_dyn_train=train_slice["x_dyn"],
        x_stat_train=train_slice["x_stat"],
        qobs_train=train_slice["qobs"],
        spinup_days_train=int(train_slice["spinup_days"]),
        x_dyn_test=test_slice["x_dyn"],
        x_stat_test=test_slice["x_stat"],
        qobs_test=test_slice["qobs"],
        spinup_days_test=int(test_slice["spinup_days"]),
        x_aux_train=train_slice.get("x_aux"),
        x_aux_test=test_slice.get("x_aux"),
        train_basins=list(train_basins),
        test_basins=list(test_basins),
        seeds=seeds,
    )
    if return_members:
        return qsim_mean, qsim_members
    return qsim_mean
