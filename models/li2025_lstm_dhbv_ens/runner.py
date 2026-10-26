"""Fold execution interface for the Li et al. (2025) (LSTM + dHBV 1.1p) Multi-Model Ensemble."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models.kratzert_lstm.runner import run_fold as run_lstm_fold
from models.li2025_lstm_dhbv_ens.model import Li2025LstmDhbvEnsemble
from models.song_dhbv11p.runner import run_fold as run_dhbv11p_fold


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (101, 102, 103, 104, 105),
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train CudaLSTM and dHBV 1.1p on `train_basins` and combine their `test_basins` predictions."""
    return_members = bool(kwargs.pop("return_members", False))
    q_lstm = np.asarray(
        run_lstm_fold(
            train_basins=train_basins,
            test_basins=test_basins,
            train_period=train_period,
            eval_period=eval_period,
            forcings=forcings,
            data_dir=data_dir,
            seeds=seeds,
            **kwargs,
        ),
        dtype=np.float32,
    )
    q_dhbv = np.asarray(
        run_dhbv11p_fold(
            train_basins=train_basins,
            test_basins=test_basins,
            train_period=train_period,
            eval_period=eval_period,
            forcings=forcings,
            data_dir=data_dir,
            seeds=seeds,
            **kwargs,
        ),
        dtype=np.float32,
    )
    mean, members = Li2025LstmDhbvEnsemble.combine_hydrographs(q_lstm, q_dhbv)
    if return_members:
        return mean, members
    return mean
