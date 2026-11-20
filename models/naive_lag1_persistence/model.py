# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Zero-parameter Naive Lag-1 Streamflow Persistence lower benchmark (`Q_hat(t) = Q_obs(t - 1)`)."""

from __future__ import annotations

import numpy as np


def predict_lag1_persistence(
    qobs_with_spinup: np.ndarray,
    spinup_days: int,
) -> np.ndarray:
    """Predict daily specific discharge as the most recent strictly causal observed value `Q_obs(t - 1)`.

    Args:
        qobs_with_spinup: `(n_basins, spinup_days + n_eval_days)` observed specific discharge in
            `mm/day` (with negative CAMELS sentinels or `NaN` indicating missing days).
        spinup_days: Number of preceding days prepended before the evaluation window (`>= 1`).

    Returns:
        `(n_basins, n_eval_days)` float32 persistence predictions with zero `NaN` values. If
        `Q_obs(t - 1)` is missing on an isolated day, the most recent finite prior observation
        `Q_obs(t')` (`t' < t`) is carried forward strictly causally (falling back to `0.0` only
        when no prior finite observation exists).
    """
    if spinup_days < 1:
        raise ValueError(f"spinup_days must be >= 1 for lag-1 persistence, got {spinup_days}.")
    q = np.asarray(qobs_with_spinup, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] <= spinup_days:
        raise ValueError(
            f"qobs_with_spinup must have shape (n_basins, >{spinup_days}), got {q.shape}."
        )
    q = np.where(np.isfinite(q) & (q >= 0.0), q, np.nan)
    n_basins, total_days = q.shape
    n_eval = total_days - spinup_days
    out = np.zeros((n_basins, n_eval), dtype=np.float32)

    for b in range(n_basins):
        last_valid = np.nan
        for t_pre in range(spinup_days):
            if np.isfinite(q[b, t_pre]):
                last_valid = q[b, t_pre]
        for j in range(n_eval):
            v = q[b, spinup_days + j - 1]
            if np.isfinite(v):
                last_valid = v
                out[b, j] = float(v)
            elif np.isfinite(last_valid):
                out[b, j] = float(last_valid)
            else:
                out[b, j] = 0.0
    return out
