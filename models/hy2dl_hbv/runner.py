"""Stage-2 and Stage-1 Fold Runner for Acuña Espinoza et al. (2024) Hy2DL LSTM-HBV."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.camels_data import load_camels_slice
from models.hy2dl_hbv.model import Hy2DLHBVTrainer


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
    spinup_days: int = 365,
    epochs: int = 20,
    batch_size: int = 256,
    rho: int = 365,
    hidden_size: int = 256,
    n_conceptual_models: int = 16,
    dropout: float = 0.4,
    lr: float = 1e-3,
    device: str | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Train Acuña Espinoza et al. (2024) Hy2DL on train_basins and predict test_basins.

    Returns:
        Float32 array of shape (len(test_basins), n_target_days) in mm/day.
    """
    del fold_id, fold_index
    if "warmup_days" in kwargs:
        spinup_days = int(kwargs["warmup_days"])  # type: ignore[arg-type]
    if "seq_length" in kwargs:
        rho = int(kwargs["seq_length"])  # type: ignore[arg-type]

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
        trainer = Hy2DLHBVTrainer(device=torch_device, seed=int(seed))
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
            hidden_size=hidden_size,
            n_conceptual_models=n_conceptual_models,
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
        assert preds.shape == (len(test_basins), n_target_days)
        seed_preds.append(preds.astype(np.float32))

    if len(seed_preds) == 1:
        return seed_preds[0]
    return np.mean(np.stack(seed_preds, axis=0), axis=0).astype(np.float32)
