"""Fold execution interface for Hoedt et al. (2021) MC-LSTM."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.camels_data import load_camels_slice
from models.hoedt_mclstm.model import MCLSTMTrainer
from models.kratzert_lstm.windows import resolve_runner_kwargs

# Hoedt et al. (2021) / NeuralHydrology `mclstm` hydrology configuration: hidden 64,
# seq_length 365, batch 256, 30 epochs, Adam {0: 1e-2, 20: 5e-3, 25: 1e-3}, NSE* loss,
# initial (output-gate) forget bias 3, gradient-norm clipping 1, unnormalized precipitation
# mass input and unnormalized discharge target.
RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "batch_size": 256,
    "epochs": 30,
    "lr": 1e-2,
    "lr_schedule": None,
    "hidden_size": 64,
    "initial_forget_bias": 3.0,
    "nse_eps": 0.1,
    "clip_grad_norm": 1.0,
    "n_batches_per_epoch": None,
    "eval_batch_size": 1024,
    "device": None,
    "return_members": False,
}

PRECIPITATION_VARIABLE_PREFIX: str = "prcp(mm/day)"


def precipitation_column_indices(feature_names: Sequence[str]) -> list[int]:
    """Indices of the precipitation (`prcp(mm/day)_<product>`) columns among `feature_names`.

    The mass input is identified by *name* from the loader's `dynamic_feature_names`, never by
    position, so a re-ordered bundle can never silently turn another variable into "mass".
    """
    names = [str(n) for n in feature_names]
    cols = [i for i, name in enumerate(names) if name.startswith(PRECIPITATION_VARIABLE_PREFIX)]
    if not cols:
        raise ValueError(f"No precipitation column found among dynamic features {names}.")
    return cols


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
    """Train Hoedt MC-LSTM on `train_basins` over `train_period` and predict `test_basins` over `eval_period`.

    Supported keyword overrides are the keys of `RUNNER_DEFAULTS` (unknown names raise).
    Returns the seed-ensemble mean `(n_test_basins, n_eval_days)` in mm/d, or
    `(mean, members)` when `return_members=True`.
    """
    cfg = resolve_runner_kwargs(kwargs, RUNNER_DEFAULTS)
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

    feature_names = [str(n) for n in np.asarray(train_slice["dynamic_feature_names"]).tolist()]
    test_feature_names = [str(n) for n in np.asarray(test_slice["dynamic_feature_names"]).tolist()]
    if feature_names != test_feature_names or len(feature_names) != train_slice["x_dyn"].shape[2]:
        raise ValueError("Train/test dynamic feature names disagree with the loaded arrays.")
    prcp_cols = precipitation_column_indices(feature_names)

    trainer = MCLSTMTrainer(
        seq_len=seq_len,
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        hidden_size=int(cfg["hidden_size"]),
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
        prcp_col_indices=prcp_cols,
        seeds=seeds,
    )
    if bool(cfg["return_members"]):
        return mean_pred, member_preds
    return mean_pred
