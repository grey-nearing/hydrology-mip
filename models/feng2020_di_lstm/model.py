# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Feng, Fang & Shen (2020, WRR) Data-Integration Lagged-Discharge LSTM (DI(1)).

Paper:
  Feng, D., Fang, K., & Shen, C. (2020). Enhancing Streamflow Forecast and
  Extracting Insights Using Long-Short Term Memory Networks With Data
  Integration at Continental Scales. Water Resources Research, 56(9),
  e2019WR026793. https://doi.org/10.1029/2019WR026793

Clean-room implementation matching `mhpi/hydroDL` (`flow_regime == 0`):
  - Dynamic inputs at day `t`:
      1. Meteorological forcings per product: `['dayl', 'prcp', 'srad', 'tmax', 'vp']`
         where `prcp` undergoes Eq. (11) `transNorm`: `log10(sqrt(max(x, 0)) + 0.1)`,
         and all meteorological inputs are z-score standardized on training-split
         target-window statistics.
      2. One-day-lagged observed streamflow transformed symmetrically by `basinNorm`
         (`Q_{t-1} / p_mean`) and `transNorm` (`log10(sqrt(max(x, 0)) + 0.1)`), then
         z-score standardized with training-split target statistics `(y_mean, y_std)`
         plus validity flag `q_lag_valid in {0.0, 1.0}`.
  - Static inputs: 27 CAMELS attributes z-score standardized on training basins.
  - Architecture: `Linear(nx, hidden_size) -> ReLU -> CudnnLstm(hidden_size, dr=0.5) -> Linear`.
  - Training: `RmseLoss` in z-scored transformed space, `Adadelta`, `rho = 365`,
    `batch_size = 100`, `epochs = 300`, hydroDL `nIterEp` sampling.
  - Inference: full continuous rollout over `[spinup + eval]` sliced after spin-up,
    inverted via `((max(10 ** (y_hat * y_std + y_mean) - 0.1, 0.0)) ** 2) * p_mean`.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models._dhbv_common.core import CudnnLstm
from models.kratzert_lstm.windows import (
    feature_stats,
    mask_spinup_discharge,
    resolve_device,
)


def trans_norm(x: np.ndarray) -> np.ndarray:
    """Apply Feng et al. (2020) Eq. (11) gamma/log10 transform: `log10(sqrt(max(x, 0)) + 0.1)`."""
    return np.log10(np.sqrt(np.maximum(np.asarray(x, dtype=np.float64), 0.0)) + 0.1)


def inv_trans_norm(y_log: np.ndarray, p_mean: np.ndarray | float) -> np.ndarray:
    """Invert Eq. (11) `transNorm` and `basinNorm`: `(max(10^y - 0.1, 0))^2 * p_mean`."""
    clipped = np.clip(np.asarray(y_log, dtype=np.float64), -20.0, 20.0)
    x_ratio = np.maximum(np.power(10.0, clipped) - 0.1, 0.0) ** 2
    return (x_ratio * np.asarray(p_mean, dtype=np.float64)).astype(np.float32)


def compute_hydrodl_iters_per_epoch(
    n_basins: int, n_target_days: int, batch_size: int, rho: int
) -> int:
    """Exact `hydroDL` iteration count covering ~99% of basin-days in expectation per epoch."""
    total_pts = max(1, int(n_basins) * int(n_target_days))
    batch_pts = max(1, int(batch_size) * int(rho))
    if batch_pts >= total_pts:
        return max(1, int(math.ceil(4.605170185988091)))
    frac = batch_pts / total_pts
    return max(1, int(math.ceil(math.log(0.01) / math.log(1.0 - frac))))


def assemble_di_met_features(
    x_dyn: np.ndarray,
    x_aux: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `[dayl, trans_norm(prcp), srad, tmax, vp]` meteorological array and raw `prcp`.

    Returns `(met_transformed, prcp_mean_daily)` where `met_transformed` has shape
    `(n_basins, n_days, 5 * n_products)` when `x_aux` is given (or raw `x_dyn` with `prcp`
    transformed when `x_aux` is omitted) and `prcp_mean_daily` has shape `(n_basins, n_days)`.
    """
    xd = np.asarray(x_dyn, dtype=np.float32)
    if xd.ndim != 3:
        raise ValueError(f"x_dyn must be 3D (n_basins, n_days, features), got {xd.shape}")
    n_ch = xd.shape[-1]
    if n_ch == 0 or n_ch % 5 != 0:
        prcp_raw = np.maximum(xd[:, :, 0], 0.0).astype(np.float64)
        xd_out = xd.astype(np.float32, copy=True)
        xd_out[:, :, 0] = trans_norm(xd_out[:, :, 0]).astype(np.float32)
        return xd_out, prcp_raw

    n_products = n_ch // 5
    prcp_raw = np.maximum(np.mean(xd[:, :, 0::5], axis=2, dtype=np.float64), 0.0)
    if x_aux is None:
        xd_out = xd.astype(np.float32, copy=True)
        for p in range(n_products):
            xd_out[:, :, 5 * p] = trans_norm(xd_out[:, :, 5 * p]).astype(np.float32)
        return xd_out, prcp_raw

    xa = np.asarray(x_aux, dtype=np.float32)
    if xa.ndim != 3 or xa.shape[:2] != xd.shape[:2] or xa.shape[-1] != 2 * n_products:
        raise ValueError(
            f"x_aux shape {xa.shape} incompatible with x_dyn shape {xd.shape}; "
            f"expected (n_basins, n_days, {2 * n_products})."
        )
    blocks: list[np.ndarray] = []
    for p in range(n_products):
        base = 5 * p
        dayl = xa[:, :, 2 * p : 2 * p + 1]
        prcp_t = trans_norm(xd[:, :, base + 0 : base + 1]).astype(np.float32)
        srad = xd[:, :, base + 1 : base + 2]
        tmax = xd[:, :, base + 2 : base + 3]
        vp = xd[:, :, base + 4 : base + 5]
        blocks.append(np.concatenate([dayl, prcp_t, srad, tmax, vp], axis=-1))
    return np.ascontiguousarray(np.concatenate(blocks, axis=-1), dtype=np.float32), prcp_raw


def build_lagged_discharge_features(
    qobs: np.ndarray,
    q_mean: float,
    q_std: float,
    lag_days: int = 1,
    p_mean: np.ndarray | float | None = None,
    apply_trans_norm: bool = False,
) -> np.ndarray:
    """Construct `(n_basins, n_days, 2)` `[q_lag_std, q_lag_valid]` features for `DI(lag_days)`.

    At timestep `t >= lag_days`, the lagged observation is `qobs[:, t - lag_days]`.
    At `t < lag_days` (or whenever `qobs[:, t - lag_days]` is NaN or negative),
    `q_lag_std` is zero-filled (`0.0`) and `q_lag_valid` is `0.0`.
    """
    if qobs.ndim != 2:
        raise ValueError(f"qobs must be 2D (n_basins, n_days), got shape {qobs.shape}")
    if lag_days < 1:
        raise ValueError(f"lag_days must be >= 1, got {lag_days}")

    n_basins, n_days = qobs.shape
    q_lag = np.full((n_basins, n_days), np.nan, dtype=np.float64)
    if n_days > lag_days:
        q_lag[:, lag_days:] = qobs[:, :-lag_days]

    valid = np.isfinite(q_lag) & (q_lag >= 0.0)
    if p_mean is not None:
        pm = np.asarray(p_mean, dtype=np.float64)
        if pm.ndim == 1:
            pm = pm[:, None]
        q_lag = np.where(valid, q_lag / np.maximum(pm, 1e-4), np.nan)
    if apply_trans_norm:
        safe_val = np.where(valid, trans_norm(np.where(valid, q_lag, 0.0)), q_mean)
    else:
        safe_val = np.where(valid, q_lag, q_mean)
    q_lag_std = np.where(valid, (safe_val - q_mean) / max(float(q_std), 1e-6), 0.0).astype(
        np.float32
    )
    q_lag_valid = valid.astype(np.float32)
    return np.stack([q_lag_std, q_lag_valid], axis=-1)


class DILSTM(nn.Module):
    """Input projection + DropConnect `CudnnLstm` + linear readout matching `mhpi/hydroDL`."""

    def __init__(
        self,
        dyn_in_dim: int,
        stat_in_dim: int = 27,
        hidden_size: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dyn_in_dim = int(dyn_in_dim)
        self.stat_in_dim = int(stat_in_dim)
        self.hidden_size = int(hidden_size)
        self.linear_in = nn.Linear(self.dyn_in_dim + self.stat_in_dim, self.hidden_size)
        self.lstm = CudnnLstm(nx=self.hidden_size, hidden_size=self.hidden_size, dr=float(dropout))
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        """Forward pass on `(batch, seq_len, dyn)` + `(batch, stat)` -> `(batch, seq_len)`."""
        seq_len = x_d.size(1)
        x_s_exp = x_s.unsqueeze(1).expand(-1, seq_len, -1)
        x_cat = torch.cat([x_d, x_s_exp], dim=-1).transpose(0, 1).contiguous()
        h_in = F.relu(self.linear_in(x_cat))
        lstm_out = self.lstm(h_in)
        return self.head(lstm_out).squeeze(-1).transpose(0, 1).contiguous()


class DILSTMTrainer:
    """Regional trainer for faithful Feng et al. (2020) `DI(1)` (`basinNorm` + `transNorm` + `RmseLoss` + `Adadelta`)."""

    def __init__(
        self,
        *,
        hidden_size: int = 256,
        seq_len: int = 365,
        dropout: float = 0.5,
        batch_size: int = 100,
        epochs: int = 300,
        lr: float = 1.0,
        n_batches_per_epoch: int | None = None,
        clip_grad_norm: float = 0.0,
        eval_batch_size: int = 256,
        lag_days: int = 1,
        lag_dropout_prob: float = 0.0,
        rmse_eps: float = 1e-6,
        device: str | torch.device | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        if lag_days < 1:
            raise ValueError(f"lag_days must be >= 1, got {lag_days}")
        if not 0.0 <= lag_dropout_prob < 1.0:
            raise ValueError(f"lag_dropout_prob must be in [0, 1), got {lag_dropout_prob}")

        self.hidden_size = int(hidden_size)
        self.seq_len = int(seq_len)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.n_batches_per_epoch = None if n_batches_per_epoch is None else int(n_batches_per_epoch)
        self.clip_grad_norm = float(clip_grad_norm)
        self.eval_batch_size = int(eval_batch_size)
        self.lag_days = int(lag_days)
        self.lag_dropout_prob = float(lag_dropout_prob)
        self.rmse_eps = float(rmse_eps)
        self.device = resolve_device(device)

    def _resolve_test_p_mean(
        self,
        p_mean_tr: np.ndarray,
        n_test: int,
        train_basins: Sequence[str] | None,
        test_basins: Sequence[str] | None,
    ) -> np.ndarray:
        global_p = float(np.mean(p_mean_tr))
        if train_basins is not None and test_basins is not None:
            tr_map = {str(b): i for i, b in enumerate(train_basins)}
            out = np.empty(n_test, dtype=np.float64)
            for i, bid in enumerate(test_basins):
                tr_pos = tr_map.get(str(bid))
                out[i] = float(p_mean_tr[tr_pos]) if tr_pos is not None else global_p
            return np.maximum(out, 1e-4)
        if n_test == p_mean_tr.shape[0]:
            return np.maximum(p_mean_tr.astype(np.float64), 1e-4)
        return np.full(n_test, max(global_p, 1e-4), dtype=np.float64)

    def fit_and_predict(
        self,
        *,
        x_dyn_train: np.ndarray,
        x_stat_train: np.ndarray,
        qobs_train: np.ndarray,
        spinup_days_train: int,
        x_dyn_test: np.ndarray,
        x_stat_test: np.ndarray,
        qobs_test: np.ndarray,
        spinup_days_test: int,
        x_aux_train: np.ndarray | None = None,
        x_aux_test: np.ndarray | None = None,
        train_basins: Sequence[str] | None = None,
        test_basins: Sequence[str] | None = None,
        seeds: Sequence[int] = (101, 102, 103, 104, 105, 106),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train one `DILSTM` per seed and return `(qsim_ensemble, qsim_members)`."""
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds must be non-empty.")

        x_stat_train = np.asarray(x_stat_train, dtype=np.float32)
        x_stat_test = np.asarray(x_stat_test, dtype=np.float32)
        met_tr_trans, prcp_tr_raw = assemble_di_met_features(x_dyn_train, x_aux_train)
        met_te_trans, _ = assemble_di_met_features(x_dyn_test, x_aux_test)

        # Mask spin-up discharge before any stat or lagged feature computation.
        q_tr_masked = mask_spinup_discharge(qobs_train, spinup_days_train)
        q_te_masked = mask_spinup_discharge(qobs_test, spinup_days_test)

        # Basin mean precipitation strictly on training-split basins & training target dates.
        p_mean_tr = np.maximum(np.mean(prcp_tr_raw[:, spinup_days_train:], axis=1), 1e-4)
        p_mean_te = self._resolve_test_p_mean(
            p_mean_tr, met_te_trans.shape[0], train_basins, test_basins
        )

        fstats = feature_stats(met_tr_trans[:, spinup_days_train:, :], x_stat_train)
        xd_tr_met = np.nan_to_num(
            ((met_tr_trans - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
        )
        xd_te_met = np.nan_to_num(
            ((met_te_trans - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
        )
        xs_tr_norm = ((x_stat_train - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
        xs_te_norm = ((x_stat_test - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)

        # Target Q: basinNorm (Q / p_mean) + transNorm (Eq. 11) + training z-score.
        valid_tr_target = np.isfinite(q_tr_masked[:, spinup_days_train:]) & (
            q_tr_masked[:, spinup_days_train:] >= 0.0
        )
        if not np.any(valid_tr_target):
            raise ValueError("Training target window contains zero valid discharge observations.")
        q_tr_ratio = q_tr_masked / p_mean_tr[:, None]
        q_tr_log = np.full(q_tr_masked.shape, np.nan, dtype=np.float64)
        valid_tr_all = np.isfinite(q_tr_masked) & (q_tr_masked >= 0.0)
        q_tr_log[valid_tr_all] = trans_norm(q_tr_ratio[valid_tr_all])
        target_logs = q_tr_log[:, spinup_days_train:][valid_tr_target]
        y_mean = float(np.mean(target_logs))
        y_std = float(max(np.std(target_logs), 1e-6))

        lag_tr = build_lagged_discharge_features(
            q_tr_masked,
            q_mean=y_mean,
            q_std=y_std,
            lag_days=self.lag_days,
            p_mean=p_mean_tr,
            apply_trans_norm=True,
        )
        lag_te = build_lagged_discharge_features(
            q_te_masked,
            q_mean=y_mean,
            q_std=y_std,
            lag_days=self.lag_days,
            p_mean=p_mean_te,
            apply_trans_norm=True,
        )

        if self.lag_dropout_prob > 0.0:
            drop_rng = np.random.RandomState(seed_list[0] + 7919)
            drop_mask = drop_rng.uniform(0.0, 1.0, size=lag_tr.shape[:2]) < self.lag_dropout_prob
            lag_tr = lag_tr.copy()
            lag_tr[drop_mask, 0] = 0.0
            lag_tr[drop_mask, 1] = 0.0

        xd_tr_norm = np.concatenate([xd_tr_met, lag_tr], axis=-1).astype(np.float32)
        xd_te_norm = np.concatenate([xd_te_met, lag_te], axis=-1).astype(np.float32)
        y_tr_norm = np.where(valid_tr_all, (q_tr_log - y_mean) / y_std, np.nan).astype(np.float32)

        n_tr_basins, n_tr_total_days, dyn_dim = xd_tr_norm.shape
        n_target_train_days = n_tr_total_days - int(spinup_days_train)
        eff_rho = min(self.seq_len, n_tr_total_days)
        valid_basins = np.where(np.sum(np.isfinite(y_tr_norm[:, spinup_days_train:]), axis=1) >= 2)[0]
        if valid_basins.size == 0:
            raise ValueError("No training basins have at least 2 valid target observations.")

        n_batches = (
            int(self.n_batches_per_epoch)
            if self.n_batches_per_epoch is not None and self.n_batches_per_epoch > 0
            else compute_hydrodl_iters_per_epoch(
                valid_basins.size, max(1, n_target_train_days), self.batch_size, eff_rho
            )
        )

        n_target_days = int(met_te_trans.shape[1]) - int(spinup_days_test)
        if n_target_days <= 0:
            raise ValueError(
                f"x_dyn_test has {met_te_trans.shape[1]} days <= spinup_days_test={spinup_days_test}"
            )

        dev = self.device
        use_amp = dev.type == "cuda"
        xd_tr_t = torch.from_numpy(xd_tr_norm).to(dev)
        xs_tr_t = torch.from_numpy(xs_tr_norm).to(dev)
        y_tr_t = torch.from_numpy(np.nan_to_num(y_tr_norm, nan=0.0)).to(dev)
        y_valid_t = torch.from_numpy(np.isfinite(y_tr_norm).astype(np.float32)).to(dev)
        xd_te_t = torch.from_numpy(xd_te_norm).to(dev)
        xs_te_t = torch.from_numpy(xs_te_norm).to(dev)

        stat_dim = xs_tr_norm.shape[1]
        n_te_basins = int(xd_te_t.shape[0])
        b_chunk = max(1, min(n_te_basins, self.eval_batch_size))
        b_act = min(self.batch_size, int(valid_basins.size))
        max_t0 = max(0, n_tr_total_days - eff_rho)
        min_t0 = int(spinup_days_train) if max_t0 >= int(spinup_days_train) else 0
        seq_offs = torch.arange(eff_rho, device=dev, dtype=torch.long)

        member_preds: list[np.ndarray] = []
        for seed in seed_list:
            torch.manual_seed(seed)
            if use_amp:
                torch.cuda.manual_seed_all(seed)
            rng = np.random.RandomState(seed)

            model = DILSTM(
                dyn_in_dim=dyn_dim,
                stat_in_dim=stat_dim,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            ).to(dev)

            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.lr, rho=0.9, eps=1e-6)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            model.train()
            for ep in range(self.epochs):
                log_ep = self.epochs >= 10 and (
                    ep == 0 or (ep + 1) % 50 == 0 or ep + 1 == self.epochs
                )
                last_loss = 0.0
                for b_i in range(n_batches):
                    b_np = rng.choice(valid_basins, size=b_act, replace=False)
                    if max_t0 > min_t0:
                        t0_np = rng.randint(min_t0, max_t0 + 1, size=b_act)
                    else:
                        t0_np = np.full(b_act, min_t0, dtype=np.int64)
                    b_idx = torch.as_tensor(b_np, dtype=torch.long, device=dev)
                    t0_t = torch.as_tensor(t0_np, dtype=torch.long, device=dev)
                    t_grid = t0_t.unsqueeze(1) + seq_offs.unsqueeze(0)
                    b_grid = b_idx.unsqueeze(1).expand(-1, eff_rho)

                    x_b = xd_tr_t[b_grid, t_grid]
                    s_b = xs_tr_t[b_idx]
                    y_b = y_tr_t[b_grid, t_grid]
                    m_b = y_valid_t[b_grid, t_grid]

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        y_hat = model(x_b, s_b).float()
                    sq_sum = (((y_hat - y_b) ** 2) * m_b).sum()
                    loss = torch.sqrt(sq_sum / torch.clamp(m_b.sum(), min=1.0) + self.rmse_eps)
                    if log_ep and b_i == n_batches - 1:
                        last_loss = float(loss.item())
                    scaler.scale(loss).backward()
                    if self.clip_grad_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                if log_ep:
                    print(
                        f"[DI(1) seed={seed}] ep={ep + 1}/{self.epochs} last_rmse={last_loss:.4f}",
                        flush=True,
                    )

            model.eval()
            chunks: list[np.ndarray] = []
            with torch.no_grad():
                for b0 in range(0, n_te_basins, b_chunk):
                    b1 = min(b0 + b_chunk, n_te_basins)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        out_b = model(xd_te_t[b0:b1], xs_te_t[b0:b1])[:, spinup_days_test:].float()
                    chunks.append(out_b.detach().cpu().numpy())
            p_norm = np.concatenate(chunks, axis=0).astype(np.float64)
            y_log = p_norm * y_std + y_mean
            member_preds.append(inv_trans_norm(y_log, p_mean_te[:, None]))

        members_arr = np.stack(member_preds, axis=0).astype(np.float32)
        mean_arr = np.mean(members_arr, axis=0, dtype=np.float32)
        return mean_arr, members_arr
