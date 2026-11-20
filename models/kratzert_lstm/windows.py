# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared training/inference utilities for the NeuralHydrology-family MIP models.

Used by `KratzertLSTMTrainer` (CudaLSTM, EA-LSTM, sMTS-LSTM, TFT) and `MCLSTMTrainer`.

Temporal anti-leakage contract
------------------------------
Every slice returned by `models.camels_data.load_camels_slice(period, spinup_days)` covers
`[period_start - spinup_days, period_end]`.  For the *training* slice the spin-up window
(`[:, :spinup_days]`) precedes the training period and, under the Newman protocol, lies
inside the evaluation decade.  Therefore:

* `mask_spinup_discharge` sets all spin-up discharge to NaN before anything else happens.
* All normalisation statistics (`feature_stats`, `target_stats`) are computed strictly on the
  training target window `[:, spinup_days:]` of the training basins.
* `training_pairs` only emits target indices `t_end >= max(seq_len - 1, spinup_days)`.
* Spin-up meteorological forcings are used exclusively as model warm-up inputs.

Evaluation contract
-------------------
`predict_windowed` evaluates one `seq_len`-day input window per evaluation day (the same
sequence-to-one protocol used in training and in NeuralHydrology) instead of a single
stateful pass over the whole evaluation period.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import torch

RUNNER_PASSTHROUGH_KWARGS: frozenset[str] = frozenset({"fold_id", "fold_index"})


def resolve_runner_kwargs(
    kwargs: Mapping[str, object], defaults: Mapping[str, object]
) -> dict[str, object]:
    """Merge `run_fold(**kwargs)` overrides into `defaults`, rejecting unknown keyword names.

    `models/run_model.py` always forwards `fold_id` and `fold_index`; those are accepted and
    ignored.  Any other unknown name raises so that a typo (e.g. `n_epochs` instead of
    `epochs`) can never be swallowed silently and turn a full run into a stub or vice versa.
    """
    unknown = sorted(set(kwargs) - set(defaults) - RUNNER_PASSTHROUGH_KWARGS)
    if unknown:
        raise ValueError(
            f"Unknown run_fold keyword argument(s) {unknown}; "
            f"supported overrides are {sorted(defaults)}."
        )
    resolved: dict[str, object] = dict(defaults)
    for key, value in kwargs.items():
        if key in defaults:
            resolved[key] = value
    return resolved


def normalize_lr_schedule(
    schedule: Mapping[int | str, float] | None, base_lr: float, default_steps: Mapping[int, float]
) -> dict[int, float]:
    """Return an epoch -> learning-rate step schedule with integer keys.

    If `schedule` is None the schedule is `{epoch: base_lr * factor}` for the
    `default_steps` factors (e.g. `{0: 1.0, 10: 0.5, 25: 0.1}` reproduces the NeuralHydrology
    `{0: 1e-3, 10: 5e-4, 25: 1e-4}` schedule for `base_lr=1e-3`).
    """
    if schedule is None:
        resolved = {int(k): float(base_lr) * float(v) for k, v in default_steps.items()}
    else:
        resolved = {int(k): float(v) for k, v in dict(schedule).items()}
    if 0 not in resolved:
        raise ValueError(f"Learning-rate schedule {resolved!r} must define epoch 0.")
    if any(v <= 0.0 for v in resolved.values()):
        raise ValueError(f"Learning-rate schedule {resolved!r} must be strictly positive.")
    return dict(sorted(resolved.items()))


def lr_for_epoch(schedule: Mapping[int, float], epoch: int) -> float:
    """Learning rate active at `epoch` (0-based) for a step schedule keyed by start epoch."""
    active = None
    for start_epoch in sorted(schedule):
        if epoch >= int(start_epoch):
            active = float(schedule[start_epoch])
    if active is None:
        raise ValueError(f"Schedule {dict(schedule)!r} has no entry covering epoch {epoch}.")
    return active


def mask_spinup_discharge(qobs: np.ndarray, spinup_days: int) -> np.ndarray:
    """Copy `qobs`, convert negative sentinels to NaN, and NaN-out the first `spinup_days`."""
    q = np.array(qobs, dtype=np.float32, copy=True)
    if q.ndim != 2:
        raise ValueError(f"qobs must be 2-D (basins, days); got shape {q.shape}.")
    if spinup_days < 0 or spinup_days >= q.shape[1]:
        raise ValueError(
            f"spinup_days={spinup_days} must satisfy 0 <= spinup_days < n_days={q.shape[1]}."
        )
    q[q < 0.0] = np.nan
    if spinup_days > 0:
        q[:, :spinup_days] = np.nan
    return q


def feature_stats(
    x_dyn_target_window: np.ndarray, x_stat: np.ndarray, floor: float = 1e-6
) -> dict[str, np.ndarray]:
    """Mean/std of dynamic (over basins x target days) and static (over basins) inputs."""
    if x_dyn_target_window.ndim != 3:
        raise ValueError(f"x_dyn must be 3-D (basins, days, features); got {x_dyn_target_window.shape}.")
    if x_stat.ndim != 2:
        raise ValueError(f"x_stat must be 2-D (basins, attributes); got {x_stat.shape}.")
    xd_mean = np.nanmean(x_dyn_target_window, axis=(0, 1), keepdims=True).astype(np.float32)
    xd_std = np.nanstd(x_dyn_target_window, axis=(0, 1), keepdims=True).astype(np.float32)
    xs_mean = np.nanmean(x_stat, axis=0, keepdims=True).astype(np.float32)
    xs_std = np.nanstd(x_stat, axis=0, keepdims=True).astype(np.float32)
    if not (np.all(np.isfinite(xd_mean)) and np.all(np.isfinite(xs_mean))):
        raise ValueError("Training features contain all-NaN columns; cannot standardize.")
    return {
        "xd_mean": xd_mean,
        "xd_std": np.maximum(xd_std, floor).astype(np.float32),
        "xs_mean": xs_mean,
        "xs_std": np.maximum(xs_std, floor).astype(np.float32),
    }


def target_stats(q_target_window: np.ndarray) -> dict[str, np.ndarray]:
    """Global target mean/std and per-basin std (mm/d) over valid training-window discharge.

    `basin_std_mm` is NaN for basins with fewer than two valid observations; such basins
    contribute no training samples and the NaN never reaches the loss.
    """
    valid = np.isfinite(q_target_window) & (q_target_window >= 0.0)
    n_valid_total = int(valid.sum())
    if n_valid_total == 0:
        raise ValueError("Training target window contains zero valid discharge observations.")
    y_mean = np.float32(np.mean(q_target_window[valid]))
    y_std = np.float32(max(float(np.std(q_target_window[valid])), 1e-6))
    n_valid = valid.sum(axis=1)
    basin_std = np.full(q_target_window.shape[0], np.nan, dtype=np.float32)
    for i in range(q_target_window.shape[0]):
        if n_valid[i] >= 2:
            basin_std[i] = np.float32(np.std(q_target_window[i][valid[i]]))
    return {
        "y_mean": y_mean,
        "y_std": y_std,
        "basin_std_mm": basin_std,
        "n_valid_per_basin": n_valid.astype(np.int64),
    }


def nse_star_weights(basin_std_mm: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Per-basin NSE* loss weights `1 / (std_b + eps)^2` (Kratzert et al., 2019; NH `NSELoss`)."""
    if eps <= 0.0:
        raise ValueError(f"NSE* eps must be positive; got {eps}.")
    return (1.0 / (basin_std_mm + float(eps)) ** 2).astype(np.float32)


def finite_window_mask(input_finite: np.ndarray, seq_len: int) -> np.ndarray:
    """Boolean (basins, days): True at `t_end` iff the whole window `[t_end-seq_len+1, t_end]` is finite."""
    if input_finite.ndim != 2:
        raise ValueError("input_finite must be 2-D (basins, days).")
    n_b, n_t = input_finite.shape
    if seq_len < 1 or seq_len > n_t:
        raise ValueError(f"seq_len={seq_len} must satisfy 1 <= seq_len <= n_days={n_t}.")
    invalid = (~input_finite).astype(np.int64)
    cum = np.concatenate([np.zeros((n_b, 1), dtype=np.int64), np.cumsum(invalid, axis=1)], axis=1)
    ok = np.zeros((n_b, n_t), dtype=bool)
    t_end = np.arange(seq_len - 1, n_t)
    ok[:, seq_len - 1 :] = (cum[:, t_end + 1] - cum[:, t_end - seq_len + 1]) == 0
    return ok


def training_pairs(
    q_masked: np.ndarray,
    window_ok: np.ndarray,
    seq_len: int,
    spinup_days: int,
    basin_weights: np.ndarray | None = None,
) -> np.ndarray:
    """All `(basin, t_end)` pairs with a valid target, a finite input window, and `t_end` inside the training period.

    Enforces `t_end >= max(seq_len - 1, spinup_days)` explicitly, so training targets can never
    fall into the spin-up window regardless of how `seq_len` and `spinup_days` relate.  When
    `basin_weights` is given, basins whose NSE* weight is not finite (fewer than two valid
    training observations) are excluded so that no NaN can enter the loss.
    """
    if q_masked.shape != window_ok.shape:
        raise ValueError("q_masked and window_ok must have identical shapes.")
    t_min = max(int(seq_len) - 1, int(spinup_days))
    valid = np.isfinite(q_masked) & (q_masked >= 0.0) & window_ok
    valid[:, :t_min] = False
    if basin_weights is not None:
        weights = np.asarray(basin_weights, dtype=np.float64)
        if weights.shape != (q_masked.shape[0],):
            raise ValueError("basin_weights must have one entry per basin.")
        valid &= np.isfinite(weights)[:, None]
    pairs = np.argwhere(valid).astype(np.int64)
    return pairs


def gather_windows(x: torch.Tensor, b_idx: torch.Tensor, t_end: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Gather `(B, seq_len, F)` input windows ending at `t_end` for basins `b_idx` from `x` `(n_b, T, F)`."""
    n_b, n_t, n_f = x.shape
    offsets = torch.arange(-int(seq_len) + 1, 1, device=x.device, dtype=torch.long)
    lin_idx = (b_idx.unsqueeze(1) * n_t + t_end.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    return x.reshape(n_b * n_t, n_f).index_select(0, lin_idx).view(b_idx.size(0), int(seq_len), n_f)


def gather_target_windows(
    y: torch.Tensor, b_idx: torch.Tensor, t_end: torch.Tensor, targ_len: int
) -> torch.Tensor:
    """Gather `(B, targ_len)` target windows ending at `t_end` for basins `b_idx` from `y` `(n_b, T)`."""
    n_b, n_t = y.shape
    offsets = torch.arange(-int(targ_len) + 1, 1, device=y.device, dtype=torch.long)
    lin_idx = (b_idx.unsqueeze(1) * n_t + t_end.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    return y.reshape(n_b * n_t).index_select(0, lin_idx).view(b_idx.size(0), int(targ_len))


@torch.no_grad()
def predict_windowed(
    predict_batch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_basins: int,
    t_first: int,
    n_target_days: int,
    batch_size: int,
    device: torch.device,
    stride: int = 1,
) -> np.ndarray:
    """Windowed inference covering every `(basin, target day)` pair with warm-up inputs.

    When `stride == 1`, `predict_batch(b_idx, t_end)` returns a 1-D `(B,)` tensor (or `(B, 1)`)
    for the single target day `t_end`.  When `stride > 1`, each window predicts its last
    `s = min(stride, n_target_days)` post-warmup steps (`(B, s)`), striding by `s` across the
    target window so every evaluation day receives at least `seq_len - s` warm-up days.
    Returns `(n_basins, n_target_days)`.
    """
    if n_target_days < 1 or n_basins < 1:
        raise ValueError("predict_windowed requires n_basins >= 1 and n_target_days >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    out = np.full((n_basins, n_target_days), np.nan, dtype=np.float32)
    s = min(int(stride), int(n_target_days))
    if s == 1:
        n_pairs = n_basins * n_target_days
        for start in range(0, n_pairs, int(batch_size)):
            flat = np.arange(start, min(start + int(batch_size), n_pairs), dtype=np.int64)
            b_np = flat // n_target_days
            k_np = flat % n_target_days
            b_idx = torch.as_tensor(b_np, dtype=torch.long, device=device)
            t_end = torch.as_tensor(k_np + int(t_first), dtype=torch.long, device=device)
            pred = predict_batch(b_idx, t_end)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if pred.ndim != 1 or pred.shape[0] != flat.shape[0]:
                raise ValueError(
                    f"predict_batch must return a 1-D tensor of length {flat.shape[0]}; got {tuple(pred.shape)}."
                )
            out[b_np, k_np] = pred.detach().float().cpu().numpy()
        return out

    k_ends = list(range(s - 1, n_target_days, s))
    if k_ends[-1] != n_target_days - 1:
        k_ends.append(n_target_days - 1)
    k_ends_arr = np.asarray(k_ends, dtype=np.int64)
    n_k = len(k_ends_arr)
    n_windows = n_basins * n_k
    for start in range(0, n_windows, int(batch_size)):
        flat = np.arange(start, min(start + int(batch_size), n_windows), dtype=np.int64)
        b_np = flat // n_k
        k_np = k_ends_arr[flat % n_k]
        b_idx = torch.as_tensor(b_np, dtype=torch.long, device=device)
        t_end = torch.as_tensor(k_np + int(t_first), dtype=torch.long, device=device)
        pred = predict_batch(b_idx, t_end)
        if pred.ndim != 2 or pred.shape != (flat.shape[0], s):
            raise ValueError(
                f"predict_batch must return a 2-D tensor of shape {(flat.shape[0], s)}; got {tuple(pred.shape)}."
            )
        pred_np = pred.detach().float().cpu().numpy()
        for i in range(flat.shape[0]):
            ke = int(k_np[i])
            out[int(b_np[i]), ke - s + 1 : ke + 1] = pred_np[i]
    return out


def n_batches_for_epoch(
    n_pairs: int, batch_size: int, cap: int | None, targ_len: int = 1
) -> int:
    """Number of mini-batches per epoch covering `n_pairs` target days, optionally capped."""
    if n_pairs <= 0:
        return 0
    max_possible = int(math.ceil(n_pairs / float(batch_size)))
    if cap is not None:
        if int(cap) < 1:
            raise ValueError("n_batches_per_epoch must be >= 1 when specified.")
        return min(max_possible, int(cap))
    if int(targ_len) <= 1:
        return max_possible
    covered = int(math.ceil(n_pairs / float(batch_size * int(targ_len))))
    return max(1, min(max_possible, covered))


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick the requested device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
