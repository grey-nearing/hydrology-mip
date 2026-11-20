# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 and Stage-1 Fold Runner for Poudel & Steinschneider (2026) MLP + 16-HBV."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.camels_data import load_camels_slice
from models.poudel_mlp_hbv.model import PoudelHBVTrainer


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (101, 102, 103),
    *,
    fold_id: str = "gauged",
    fold_index: int = 0,
    spinup_days: int = 730,
    eval_spinup_days: int | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    rho: int = 365,
    stride: int = 60,
    nmul: int = 16,
    hidden_dim: int = 2048,
    dropout: float = 0.4,
    lr: float = 1e-4,
    device: str | None = None,
    return_members: bool = False,
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train Poudel & Steinschneider (2026) MLP + 16-HBV on train_basins and predict test_basins.

    ``eval_spinup_days`` is the number of forcing days prepended to ``eval_period`` (defaults to
    ``spinup_days``). Set it to 0 when ``eval_period`` starts on the first day of the data bundle.

    Returns:
        Float32 array of shape (len(test_basins), n_target_days) in mm/day,
        or (mean, members) when return_members=True.
    """
    del fold_id, fold_index
    if "warmup_days" in kwargs:
        spinup_days = int(kwargs["warmup_days"])  # type: ignore[arg-type]
    if "seq_length" in kwargs:
        rho = int(kwargs["seq_length"])  # type: ignore[arg-type]
    if "hidden_size" in kwargs:
        hidden_dim = int(kwargs["hidden_size"])  # type: ignore[arg-type]

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
        spinup_days=spinup_days if eval_spinup_days is None else int(eval_spinup_days),
        require_valid_targets=False,
    )

    train_spinup = int(train_slice["spinup_days"])
    eval_start_idx = int(test_slice["spinup_days"])
    n_target_days = int(test_slice["n_target_days"])
    train_doy = pd.DatetimeIndex(train_slice["dates"]).dayofyear.to_numpy(dtype=np.float32)
    test_doy = pd.DatetimeIndex(test_slice["dates"]).dayofyear.to_numpy(dtype=np.float32)

    torch_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    seed_preds: list[np.ndarray] = []
    for seed in seeds:
        trainer = PoudelHBVTrainer(device=torch_device, seed=int(seed))
        preds = trainer.fit_and_predict(
            x_dyn_train=train_slice["x_dyn"],
            x_stat_train=train_slice["x_stat"],
            y_obs_train=train_slice["qobs"],
            x_dyn_test=test_slice["x_dyn"],
            x_stat_test=test_slice["x_stat"],
            eval_start_idx=eval_start_idx,
            epochs=epochs,
            batch_size=batch_size,
            rho=rho,
            warmup=train_spinup,
            stride=stride,
            nmul=nmul,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            spinup_days=train_spinup,
            train_doy=train_doy,
            test_doy=test_doy,
            lat_train=train_slice["lat"],
            lat_test=test_slice["lat"],
            n_batches_per_epoch=(
                int(kwargs["n_batches_per_epoch"])  # type: ignore[arg-type]
                if "n_batches_per_epoch" in kwargs
                else None
            ),
        )
        if preds.shape != (len(test_basins), n_target_days):
            raise ValueError(
                f"Predictions have shape {preds.shape}, expected {(len(test_basins), n_target_days)}."
            )
        seed_preds.append(preds.astype(np.float32))

    members = np.stack(seed_preds, axis=0).astype(np.float32)
    ens_mean = np.mean(members, axis=0).astype(np.float32)
    if return_members:
        return ens_mean, members
    return ens_mean
