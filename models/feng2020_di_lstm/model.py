"""Feng, Fang & Shen (2020, WRR) Data-Integration Lagged-Discharge LSTM (DI(1)).

Paper:
  Feng, D., Fang, K., & Shen, C. (2020). Enhancing Streamflow Forecast and
  Extracting Insights Using Long-Short Term Memory Networks With Data
  Integration at Continental Scales. Water Resources Research, 56(9),
  e2019WR026793. https://doi.org/10.1029/2019WR026793

Architecture (`DI(1)`):
  - Single-layer 256-unit LSTM (matching `mhpi/hydroDL` / `CudaLSTM`) followed
    by dropout (`0.4`) and a linear readout head.
  - Dynamic inputs at day `t`:
      1. Meteorological forcing variables `x_dyn[:, t, :]` (standardized using
         training-period statistics).
      2. Standardized 1-day-lagged observed specific discharge
         `q_lag_std[:, t] = (Q_{t-1} - q_mean) / q_std` when `Q_{t-1}` is valid
         (`isfinite(Q_{t-1}) and Q_{t-1} >= 0`), and `0.0` when `Q_{t-1}` is
         missing/NaN.
      3. Binary validity indicator `q_lag_valid[:, t] in {0.0, 1.0}` (`1.0` when
         `Q_{t-1}` is valid, `0.0` when missing/NaN).
  - Static inputs: 27 catchment attributes (standardized using training-period
    statistics and broadcast across time).
  - Loss: Basin-averaged NSE* loss (`loss_fn="nse"`) over the post-spinup target
    tail of each 365-day sequence window (matching `mhpi/hydroDL` sequence-to-sequence
    training with 182 warmup steps + 183 active target steps).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.kratzert_lstm.model import (
    DEFAULT_LR_STEP_FACTORS,
    CudaLSTM,
)
from models.kratzert_lstm.windows import (
    feature_stats,
    finite_window_mask,
    gather_target_windows,
    gather_windows,
    lr_for_epoch,
    mask_spinup_discharge,
    n_batches_for_epoch,
    normalize_lr_schedule,
    nse_star_weights,
    resolve_device,
    target_stats,
    training_pairs,
)


def build_lagged_discharge_features(
    qobs: np.ndarray,
    q_mean: float,
    q_std: float,
    lag_days: int = 1,
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
    safe_q = np.where(valid, q_lag, q_mean)
    q_lag_std = np.where(valid, (safe_q - q_mean) / q_std, 0.0).astype(np.float32)
    q_lag_valid = valid.astype(np.float32)
    return np.stack([q_lag_std, q_lag_valid], axis=-1)


class DILSTM(CudaLSTM):
    """256-unit LSTM taking `[meteorological_forcings, q_lag_std, q_lag_valid]` + static attributes."""


class DILSTMTrainer:
    """Regional trainer for Feng et al. (2020) `DI(1)` Lagged-Discharge LSTM."""

    def __init__(
        self,
        *,
        hidden_size: int = 256,
        seq_len: int = 365,
        dropout: float = 0.4,
        batch_size: int = 256,
        epochs: int = 30,
        lr: float = 1e-3,
        lr_schedule: Mapping[int | str, float] | None = None,
        n_batches_per_epoch: int | None = None,
        clip_grad_norm: float = 1.0,
        eval_batch_size: int = 256,
        lag_days: int = 1,
        lag_dropout_prob: float = 0.02,
        nse_eps: float = 0.1,
        initial_forget_bias: float = 3.0,
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
        self.targ_len = max(1, min(183, self.seq_len - max(1, self.seq_len // 2)))
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.lr_schedule = normalize_lr_schedule(lr_schedule, self.lr, DEFAULT_LR_STEP_FACTORS)
        self.n_batches_per_epoch = None if n_batches_per_epoch is None else int(n_batches_per_epoch)
        self.clip_grad_norm = float(clip_grad_norm)
        self.eval_batch_size = int(eval_batch_size)
        self.lag_days = int(lag_days)
        self.lag_dropout_prob = float(lag_dropout_prob)
        self.nse_eps = float(nse_eps)
        self.initial_forget_bias = float(initial_forget_bias)
        self.device = resolve_device(device)

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
        seeds: Sequence[int] = (101, 102, 103, 104, 105, 106),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train one `DILSTM` per seed and return `(qsim_ensemble, qsim_members)`."""
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds must be non-empty.")

        x_dyn_train = np.asarray(x_dyn_train, dtype=np.float32)
        x_stat_train = np.asarray(x_stat_train, dtype=np.float32)
        x_dyn_test = np.asarray(x_dyn_test, dtype=np.float32)
        x_stat_test = np.asarray(x_stat_test, dtype=np.float32)

        # Mask spin-up discharge before computing training statistics or lagged features
        # so zero pre-train-period discharge ever leaks into training targets or inputs.
        q_tr_masked = mask_spinup_discharge(qobs_train, spinup_days_train)
        q_te_masked = mask_spinup_discharge(qobs_test, spinup_days_test)

        fstats = feature_stats(x_dyn_train[:, spinup_days_train:, :], x_stat_train)
        tstats = target_stats(q_tr_masked[:, spinup_days_train:])
        y_mean = float(tstats["y_mean"])
        y_std = float(tstats["y_std"])
        weights_np = nse_star_weights(tstats["basin_std_mm"], eps=self.nse_eps)

        xd_tr_met = np.nan_to_num(
            ((x_dyn_train - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
        )
        xd_te_met = np.nan_to_num(
            ((x_dyn_test - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
        )
        xs_tr_norm = ((x_stat_train - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
        xs_te_norm = ((x_stat_test - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)

        lag_tr = build_lagged_discharge_features(
            q_tr_masked, q_mean=y_mean, q_std=y_std, lag_days=self.lag_days
        )
        lag_te = build_lagged_discharge_features(
            q_te_masked, q_mean=y_mean, q_std=y_std, lag_days=self.lag_days
        )

        # Deterministic sparse lag-dropout during training so the model learns to
        # rely on meteorological + static state on any day with `q_lag_valid == 0.0`.
        if self.lag_dropout_prob > 0.0:
            drop_rng = np.random.RandomState(seed_list[0] + 7919)
            drop_mask = drop_rng.uniform(0.0, 1.0, size=lag_tr.shape[:2]) < self.lag_dropout_prob
            lag_tr = lag_tr.copy()
            lag_tr[drop_mask, 0] = 0.0
            lag_tr[drop_mask, 1] = 0.0

        xd_tr_norm = np.concatenate([xd_tr_met, lag_tr], axis=-1).astype(np.float32)
        xd_te_norm = np.concatenate([xd_te_met, lag_te], axis=-1).astype(np.float32)
        y_tr_norm = ((q_tr_masked - y_mean) / y_std).astype(np.float32)

        window_ok = finite_window_mask(np.all(np.isfinite(xd_tr_norm), axis=2), self.seq_len)
        pairs = training_pairs(
            q_tr_masked, window_ok, self.seq_len, spinup_days_train, basin_weights=weights_np
        )
        if pairs.shape[0] == 0:
            raise ValueError("No valid training windows in DILSTMTrainer.")

        n_batches = n_batches_for_epoch(
            pairs.shape[0], self.batch_size, self.n_batches_per_epoch, targ_len=self.targ_len
        )
        n_target_days = int(x_dyn_test.shape[1]) - int(spinup_days_test)
        if n_target_days <= 0:
            raise ValueError(
                f"x_dyn_test has {x_dyn_test.shape[1]} days <= spinup_days_test={spinup_days_test}"
            )

        dev = self.device
        use_amp = dev.type == "cuda"
        xd_tr_t = torch.from_numpy(xd_tr_norm).to(dev)
        xs_tr_t = torch.from_numpy(xs_tr_norm).to(dev)
        y_tr_t = torch.from_numpy(np.nan_to_num(y_tr_norm, nan=0.0)).to(dev)
        y_valid_t = torch.from_numpy(np.isfinite(y_tr_norm).astype(np.float32)).to(dev)
        w_tr_t = torch.from_numpy(weights_np).to(dev)
        xd_te_t = torch.from_numpy(xd_te_norm).to(dev)
        xs_te_t = torch.from_numpy(xs_te_norm).to(dev)
        y_std_t = torch.tensor(y_std, dtype=torch.float32, device=dev)
        pairs_t = torch.from_numpy(pairs).to(device=dev, dtype=torch.long)

        dyn_dim = xd_tr_norm.shape[2]
        stat_dim = xs_tr_norm.shape[1]
        n_te_basins = int(xd_te_t.shape[0])
        b_chunk = max(1, min(n_te_basins, self.eval_batch_size))

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
                initial_forget_bias=self.initial_forget_bias,
            ).to(dev)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr_for_epoch(self.lr_schedule, 0))
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            model.train()
            for epoch in range(self.epochs):
                lr_now = lr_for_epoch(self.lr_schedule, epoch)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                perm = rng.permutation(pairs.shape[0])
                if n_batches * self.batch_size <= perm.shape[0]:
                    idx_epoch_np = perm[: n_batches * self.batch_size]
                else:
                    idx_epoch_np = rng.choice(
                        pairs.shape[0], size=n_batches * self.batch_size, replace=True
                    )
                idx_epoch = torch.from_numpy(idx_epoch_np).to(device=dev, dtype=torch.long)
                for batch_i in range(n_batches):
                    batch_rows = pairs_t[
                        idx_epoch[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]
                    ]
                    b_idx = batch_rows[:, 0]
                    t_end = batch_rows[:, 1]
                    x_b = gather_windows(xd_tr_t, b_idx, t_end, self.seq_len)
                    s_b = xs_tr_t[b_idx]
                    y_b = gather_target_windows(y_tr_t, b_idx, t_end, self.targ_len)
                    m_b = gather_target_windows(y_valid_t, b_idx, t_end, self.targ_len)
                    w_b = w_tr_t[b_idx].unsqueeze(1)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        y_hat = model(x_b, s_b)[:, -self.targ_len :].float()
                    err_mm = (y_hat - y_b) * y_std_t
                    sq = (err_mm**2) * w_b * m_b
                    loss = sq.sum() / torch.clamp(m_b.sum(), min=1.0)
                    scaler.scale(loss).backward()
                    if self.clip_grad_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

            model.eval()
            chunks: list[np.ndarray] = []
            with torch.no_grad():
                for b0 in range(0, n_te_basins, b_chunk):
                    b1 = min(b0 + b_chunk, n_te_basins)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        out_b = model(xd_te_t[b0:b1], xs_te_t[b0:b1])[:, spinup_days_test:].float()
                    chunks.append(out_b.detach().cpu().numpy())
            p_norm = np.concatenate(chunks, axis=0)
            member_preds.append(np.clip(p_norm * y_std + y_mean, 0.0, None).astype(np.float32))

        members_arr = np.stack(member_preds, axis=0).astype(np.float32)
        mean_arr = np.mean(members_arr, axis=0, dtype=np.float32)
        return mean_arr, members_arr
