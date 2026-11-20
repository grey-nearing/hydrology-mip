# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Zero-parameter Naive Lag-1 Streamflow Persistence lower benchmark (`Q_hat(t) = Q_obs(t - 1)`)."""

from __future__ import annotations

import numpy as np


def predict_lag1_persistence(
    qobs_with_lookback: np.ndarray,
    lookback_days: int,
    cold_start_value: float | None = None,
) -> tuple[np.ndarray, int]:
    """Predict daily specific discharge as the most recent strictly causal observation.

    ``Q_hat(t) = Q_obs(t - 1)``. When ``Q_obs(t - 1)`` is missing, the most recent finite
    observation ``Q_obs(t')`` with ``t' < t`` is carried forward (still strictly causal).

    A basin whose gauge record starts on or after an evaluation day has no observation to persist
    for that day (in the canonical 531-basin set this happens only for `02427250` and `09484600`,
    whose records start on the first day of the evaluation window). Such "cold-start" days raise a
    ``ValueError`` unless ``cold_start_value`` is given explicitly, in which case that constant is
    used and the number of filled days is returned so callers can report it.

    Args:
        qobs_with_lookback: ``(n_basins, lookback_days + n_eval_days)`` observed specific discharge
            in mm/day, **without** spin-up masking, starting ``lookback_days`` days before the first
            evaluation day. Negative sentinels and non-finite values mark missing days.
        lookback_days: Number of days preceding the evaluation window (``>= 1``).
        cold_start_value: Explicit constant (mm/day) for days with no prior observation, or
            ``None`` to raise on such days.

    Returns:
        ``(predictions, n_cold_start_days)`` with predictions of shape ``(n_basins, n_eval_days)``
        (float32, finite everywhere).
    """
    lookback = int(lookback_days)
    if lookback < 1:
        raise ValueError(f"lookback_days must be >= 1 for lag-1 persistence, got {lookback_days}.")
    if cold_start_value is not None and not (np.isfinite(cold_start_value) and cold_start_value >= 0.0):
        raise ValueError(f"cold_start_value must be a finite non-negative number, got {cold_start_value!r}.")
    q = np.asarray(qobs_with_lookback, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] <= lookback:
        raise ValueError(f"qobs_with_lookback must have shape (n_basins, >{lookback}), got {q.shape}.")
    q = np.where(np.isfinite(q) & (q >= 0.0), q, np.nan)
    n_basins, total_days = q.shape
    n_eval = total_days - lookback
    out = np.empty((n_basins, n_eval), dtype=np.float32)
    n_cold = 0

    for b in range(n_basins):
        last_valid = np.nan
        for t in range(total_days - 1):
            if np.isfinite(q[b, t]):
                last_valid = q[b, t]
            j = t + 1 - lookback  # evaluation-day index predicted from observations up to day t
            if j < 0:
                continue
            if np.isfinite(last_valid):
                out[b, j] = float(last_valid)
                continue
            if cold_start_value is None:
                raise ValueError(
                    f"Basin row {b} has no finite discharge observation before evaluation day index {j} "
                    f"(lookback {lookback} days); pass an explicit cold_start_value."
                )
            out[b, j] = float(cold_start_value)
            n_cold += 1
    return out, n_cold
