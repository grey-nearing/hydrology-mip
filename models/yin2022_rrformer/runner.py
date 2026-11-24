# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold runner for Yin et al. (2022, J. Hydrol.) RR-Former."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from models.camels_data import load_camels_slice
from models.yin2022_rrformer.model import RRFormerTrainer

DEFAULT_SEEDS: tuple[int, ...] = (601, 602, 603)


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 256,
    past_len: int | None = None,
    pred_len: int = 1,
    seq_len: int | None = None,
    use_static: bool = False,
    use_future_fea: bool = True,
    decode_mode: str = "NAR",
    dropout: float = 0.1,
    batch_size: int = 512,
    epochs: int = 200,
    finetune_epochs: int | None = None,
    finetune_patience: int = 0,
    pretrain_patience: int = 0,
    lr: float = 1e-3,
    finetune_lr: float = 1e-3,
    n_batches_per_epoch: int | None = None,
    clip_grad_norm: float = 0.0,
    eval_batch_size: int = 4096,
    lag_dropout_prob: float = 0.0,
    use_amp: bool = False,
    use_compile: bool = False,
    clip_nonneg: bool = True,
    checkpoint_dir: str | Path | None = None,
    load_pretrain_checkpoint: str | Path | None = None,
    val_period: tuple[str, str] | None = None,
    device: str | torch.device | None = None,
    return_members: bool = False,
    return_lead_predictions: bool = False,
    fold_id: str | None = None,
    fold_index: int | None = None,
    spatial_regime: str | None = None,
    forcing_regime: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train Yin et al. (2022) RR-Former and predict `eval_period` on `test_basins`."""
    del fold_id, fold_index, spatial_regime, forcing_regime
    if past_len is None:
        resolved_past = (int(seq_len) - 1) if seq_len is not None else 15
    else:
        resolved_past = int(past_len)
    if resolved_past < 1 or int(pred_len) < 1:
        raise ValueError(f"past_len ({resolved_past}) and pred_len ({pred_len}) must be >= 1.")

    spinup_days = resolved_past
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
    val_slice = (
        load_camels_slice(
            basins=train_basins,
            period=val_period,
            forcings=forcings,
            data_dir=data_dir,
            spinup_days=spinup_days,
            require_valid_targets=True,
        )
        if val_period is not None
        else None
    )

    resolved_ft_epochs = int(finetune_epochs) if finetune_epochs is not None else int(epochs)
    trainer = RRFormerTrainer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        past_len=resolved_past,
        pred_len=int(pred_len),
        use_static=use_static,
        use_future_fea=use_future_fea,
        decode_mode=decode_mode,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        finetune_epochs=resolved_ft_epochs,
        finetune_patience=finetune_patience,
        pretrain_patience=pretrain_patience,
        lr=lr,
        finetune_lr=finetune_lr,
        n_batches_per_epoch=n_batches_per_epoch,
        clip_grad_norm=clip_grad_norm,
        eval_batch_size=eval_batch_size,
        lag_dropout_prob=lag_dropout_prob,
        use_amp=use_amp,
        use_compile=use_compile,
        clip_nonneg=clip_nonneg,
        checkpoint_dir=checkpoint_dir,
        load_pretrain_checkpoint=load_pretrain_checkpoint,
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
        x_dyn_val=val_slice["x_dyn"] if val_slice is not None else None,
        x_stat_val=val_slice["x_stat"] if val_slice is not None else None,
        qobs_val=val_slice["qobs"] if val_slice is not None else None,
        spinup_days_val=int(val_slice["spinup_days"]) if val_slice is not None else 0,
        train_basins=list(train_basins),
        test_basins=list(test_basins),
        seeds=seeds,
        return_lead_predictions=return_lead_predictions,
    )
    if return_members:
        return qsim_mean, qsim_members
    return qsim_mean
