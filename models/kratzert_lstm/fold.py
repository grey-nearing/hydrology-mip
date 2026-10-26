"""Shared `run_fold` implementation for the sequence models trained with `KratzertLSTMTrainer`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from torch import nn

from models.camels_data import load_camels_slice
from models.kratzert_lstm.model import KratzertLSTMTrainer
from models.kratzert_lstm.windows import resolve_runner_kwargs


def sequence_runner_defaults(
    *,
    seq_len: int,
    epochs: int,
    hidden_size: int,
    dropout: float,
    lr: float = 1e-3,
    eval_batch_size: int = 1024,
) -> dict[str, object]:
    """Default `run_fold` keyword overrides for a sequence model."""
    return {
        "seq_len": int(seq_len),
        "batch_size": 256,
        "epochs": int(epochs),
        "lr": float(lr),
        "lr_schedule": None,
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "initial_forget_bias": 3.0,
        "nse_eps": 0.1,
        "clip_grad_norm": 1.0,
        "n_batches_per_epoch": None,
        "eval_batch_size": int(eval_batch_size),
        "device": None,
        "return_members": False,
    }


def run_sequence_fold(
    *,
    model_cls: type[nn.Module],
    defaults: Mapping[str, object],
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int],
    kwargs: Mapping[str, object],
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load the training/evaluation slices, train `model_cls` per seed, and predict `test_basins`.

    Both slices are loaded with `spinup_days = seq_len - 1` so that every target day (the first
    training day and the first evaluation day included) has a full `seq_len` input window.  The
    trainer masks the training spin-up discharge and uses evaluation spin-up forcings only as
    warm-up inputs.
    """
    cfg = resolve_runner_kwargs(kwargs, defaults)
    seq_len = int(cfg["seq_len"])
    spinup_days = max(0, seq_len - 1)

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
    )
    if int(train_slice["spinup_days"]) != spinup_days or int(test_slice["spinup_days"]) != spinup_days:
        raise ValueError("Loader returned a slice whose spinup_days differs from the requested value.")

    trainer = KratzertLSTMTrainer(
        seq_len=seq_len,
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        hidden_size=int(cfg["hidden_size"]),
        dropout=float(cfg["dropout"]),
        device=cfg["device"],
        lr_schedule=cfg["lr_schedule"],
        initial_forget_bias=float(cfg["initial_forget_bias"]),
        nse_eps=float(cfg["nse_eps"]),
        clip_grad_norm=float(cfg["clip_grad_norm"]),
        n_batches_per_epoch=cfg["n_batches_per_epoch"],
        eval_batch_size=int(cfg["eval_batch_size"]),
    )
    mean_pred, member_preds = trainer.fit_and_predict(
        x_dyn_train=train_slice["x_dyn"],
        x_stat_train=train_slice["x_stat"],
        qobs_train=train_slice["qobs"],
        x_dyn_test=test_slice["x_dyn"],
        x_stat_test=test_slice["x_stat"],
        n_target_days=int(test_slice["n_target_days"]),
        spinup_days_train=spinup_days,
        seeds=seeds,
        model_cls=model_cls,
    )
    if bool(cfg["return_members"]):
        return mean_pred, member_preds
    return mean_pred
