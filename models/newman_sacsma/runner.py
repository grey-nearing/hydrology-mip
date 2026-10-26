"""Stage-2 and Stage-1 Fold Runner for Nearing et al. (2020) & Newman et al. (2017) Snow-17 + SAC-SMA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from models.camels_data import load_camels_slice
from models.newman_sacsma.model import SacSmaCalibratorAndDonorTransfer


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (123,),
    *,
    fold_id: str = "gauged",
    fold_index: int = 0,
    spinup_days: int = 365,
    n_iter: int = 400,
    k_donors: int = 1,
    **kwargs: object,
) -> np.ndarray:
    """Calibrate Snow-17 + SAC-SMA on train_basins and predict test_basins.

    Returns:
        Float32 array of shape (len(test_basins), n_target_days) in mm/day.
    """
    del fold_id, fold_index
    if "warmup_days" in kwargs:
        spinup_days = int(kwargs["warmup_days"])  # type: ignore[arg-type]
    if "epochs" in kwargs and "n_iter" not in kwargs:
        n_iter = max(10, int(kwargs["epochs"]) * 10)  # type: ignore[arg-type]

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

    seed_preds: list[np.ndarray] = []
    for seed in seeds:
        calibrator = SacSmaCalibratorAndDonorTransfer(
            n_iter=n_iter,
            seed=int(seed),
            k_donors=k_donors,
        )
        preds = calibrator.fit_and_predict(
            train_basins=list(train_basins),
            test_basins=list(test_basins),
            x_dyn_train=train_slice["x_dyn"],
            x_stat_train=train_slice["x_stat"],
            y_obs_train=train_slice["qobs"],
            x_dyn_test=test_slice["x_dyn"],
            x_stat_test=test_slice["x_stat"],
            train_doy=train_doy,
            test_doy=test_doy,
            eval_start_idx=eval_start_idx,
            spinup_days=train_spinup,
            lat_train=train_slice["lat"],
            lon_train=train_slice["lon"],
            elev_train=train_slice["elev"],
            lat_test=test_slice["lat"],
            lon_test=test_slice["lon"],
            elev_test=test_slice["elev"],
        )
        assert preds.shape == (len(test_basins), n_target_days)
        seed_preds.append(preds.astype(np.float32))

    if len(seed_preds) == 1:
        return seed_preds[0]
    return np.mean(np.stack(seed_preds, axis=0), axis=0).astype(np.float32)
