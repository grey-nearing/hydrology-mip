# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# The basin-averaged NSE* loss and step learning-rate schedule follow
# neuralhydrology/training/loss.py (Copyright (c) 2021, NeuralHydrology,
# BSD 3-Clause License) and https://github.com/kratzert/multiple_forcing
# (Copyright 2021 Frederik Kratzert, Apache License 2.0). See THIRD_PARTY_NOTICES.md.

"""Kratzert et al. (2019, 2021) CudaLSTM architecture and the shared NeuralHydrology-style trainer.

`KratzertLSTMTrainer` is reused by the EA-LSTM, sMTS-LSTM, and TFT runners.  It implements the
NeuralHydrology training protocol: sequence-to-one targets, basin-averaged NSE* loss, Adam with
a step learning-rate schedule, gradient-norm clipping, full-epoch coverage of all valid training
windows, and sequence-to-one windowed inference.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

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
    predict_windowed,
    resolve_device,
    target_stats,
    training_pairs,
)

# NeuralHydrology / Kratzert et al. (2021) step schedule expressed as multiples of the base LR.
DEFAULT_LR_STEP_FACTORS: dict[int, float] = {0: 1.0, 10: 0.5, 25: 0.1}


class CudaLSTM(nn.Module):
    """Regional LSTM with concatenated dynamic forcings and static catchment attributes.

    References
    ----------
    Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G. (2019):
    Towards learning universal, regional, and local hydrological behaviors via machine learning
    applied to large-sample datasets, Hydrol. Earth Syst. Sci., 23, 5089-5110.
    """

    def __init__(
        self,
        dyn_in_dim: int,
        stat_in_dim: int = 27,
        hidden_size: int = 256,
        dropout: float = 0.4,
        initial_forget_bias: float = 3.0,
    ) -> None:
        super().__init__()
        self.dyn_in_dim = dyn_in_dim
        self.stat_in_dim = stat_in_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=dyn_in_dim + stat_in_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, 1)
        self._reset_parameters(initial_forget_bias)

    def _reset_parameters(self, initial_forget_bias: float) -> None:
        self.lstm.bias_hh_l0.data[self.hidden_size : 2 * self.hidden_size].fill_(
            float(initial_forget_bias)
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        seq_len = x_d.size(1)
        x_s_exp = x_s.unsqueeze(1).expand(-1, seq_len, -1)
        x_in = torch.cat([x_d, x_s_exp], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        return self.head(self.dropout(lstm_out)).squeeze(-1)


def nse_star_loss(
    err_mm: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Basin-averaged NSE* loss: mean over the batch of `err_mm^2 / (std_b + eps)^2`.

    `err_mm` is the prediction error expressed in mm/d and `weights = 1 / (std_b_mm + eps)^2`
    with `std_b_mm` the per-basin training-period discharge standard deviation in mm/d and
    `eps = 0.1` mm/d (Kratzert et al., 2019; NeuralHydrology `NSELoss`).  Expressing the error
    in mm/d keeps numerator and denominator in identical units.
    """
    return torch.mean(weights * err_mm**2)


class KratzertLSTMTrainer:
    """NeuralHydrology-style sequence-to-one trainer with basin-averaged NSE* loss.

    Protocol
    --------
    * Inputs: dynamic forcings standardized with mean/std over training basins x training
      target days; static attributes standardized over training basins.
    * Targets: discharge standardized with the global training-window mean/std; the loss is
      formed in mm/d (see `nse_star_loss`).
    * Spin-up discharge (`[:, :spinup_days]`) is NaN-masked before any statistic is computed
      and training targets satisfy `t_end >= max(seq_len - 1, spinup_days)`.
    * One epoch visits every valid `(basin, day)` window once (random order); Adam with a step
      learning-rate schedule (default `{0: lr, 10: lr/2, 25: lr/10}`), gradient norm clipped to
      `clip_grad_norm`, `epochs` epochs, no early stopping and no validation period.
    * Inference: one `seq_len`-day window per evaluation day (`predict_windowed`), predictions
      de-standardized to mm/d and clipped at zero.
    """

    def __init__(
        self,
        seq_len: int = 365,
        batch_size: int = 256,
        epochs: int = 30,
        lr: float = 1e-3,
        hidden_size: int = 256,
        dropout: float = 0.4,
        device: str | torch.device | None = None,
        lr_schedule: Mapping[int | str, float] | None = None,
        initial_forget_bias: float = 3.0,
        nse_eps: float = 0.1,
        clip_grad_norm: float = 1.0,
        n_batches_per_epoch: int | None = None,
        eval_batch_size: int = 1024,
        model_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if epochs < 0:
            raise ValueError("epochs must be >= 0.")
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.device = resolve_device(device)
        self.lr_schedule = normalize_lr_schedule(lr_schedule, self.lr, DEFAULT_LR_STEP_FACTORS)
        self.initial_forget_bias = float(initial_forget_bias)
        self.nse_eps = float(nse_eps)
        self.clip_grad_norm = float(clip_grad_norm)
        self.n_batches_per_epoch = None if n_batches_per_epoch is None else int(n_batches_per_epoch)
        self.eval_batch_size = int(eval_batch_size)
        self.model_kwargs: dict[str, object] = dict(model_kwargs) if model_kwargs else {}

    def _build_model(self, model_cls: type[nn.Module], dyn_dim: int, stat_dim: int) -> nn.Module:
        params = inspect.signature(model_cls).parameters
        kwargs: dict[str, object] = {
            "dyn_in_dim": dyn_dim,
            "stat_in_dim": stat_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
        }
        if "initial_forget_bias" in params:
            kwargs["initial_forget_bias"] = self.initial_forget_bias
        kwargs.update(self.model_kwargs)
        return model_cls(**kwargs).to(self.device)

    def fit_and_predict(
        self,
        x_dyn_train: np.ndarray,
        x_stat_train: np.ndarray,
        qobs_train: np.ndarray,
        x_dyn_test: np.ndarray,
        x_stat_test: np.ndarray,
        n_target_days: int,
        spinup_days_train: int,
        seeds: Sequence[int] = (123,),
        model_cls: type[nn.Module] = CudaLSTM,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train one model per seed on the training slice and predict the evaluation slice.

        Args:
            x_dyn_train: `(n_train_basins, spinup_days_train + n_train_days, n_dyn)` forcings.
            x_stat_train: `(n_train_basins, n_static)` attributes.
            qobs_train: `(n_train_basins, spinup_days_train + n_train_days)` discharge in mm/d.
            x_dyn_test: `(n_test_basins, spinup_days_test + n_target_days, n_dyn)` forcings.
            x_stat_test: `(n_test_basins, n_static)` attributes.
            n_target_days: Number of evaluation target days (last days of `x_dyn_test`).
            spinup_days_train: Number of leading warm-up days in the training slice whose
                discharge must never be used.
            seeds: One model is trained per seed; the ensemble mean is returned first.
            model_cls: Architecture accepting `(dyn_in_dim, stat_in_dim, hidden_size, dropout)`
                and `forward(x_d, x_s) -> (batch, seq_len)`.

        Returns:
            `(mean_pred, member_preds)` with shapes `(n_test_basins, n_target_days)` and
            `(n_seeds, n_test_basins, n_target_days)` in mm/d.
        """
        if len(seeds) == 0:
            raise ValueError("At least one seed is required.")
        x_dyn_train = np.asarray(x_dyn_train, dtype=np.float32)
        x_stat_train = np.asarray(x_stat_train, dtype=np.float32)
        x_dyn_test = np.asarray(x_dyn_test, dtype=np.float32)
        x_stat_test = np.asarray(x_stat_test, dtype=np.float32)
        n_target_days = int(n_target_days)
        spinup_days_train = int(spinup_days_train)

        n_tr, total_tr_days, dyn_dim = x_dyn_train.shape
        if qobs_train.shape != (n_tr, total_tr_days):
            raise ValueError(
                f"qobs_train shape {qobs_train.shape} must match x_dyn_train[:, :, 0] shape {(n_tr, total_tr_days)}."
            )
        if x_dyn_test.shape[2] != dyn_dim or x_stat_test.shape[1] != x_stat_train.shape[1]:
            raise ValueError("Train/test feature dimensions differ.")
        spinup_days_test = int(x_dyn_test.shape[1]) - n_target_days
        if spinup_days_test < self.seq_len - 1:
            raise ValueError(
                f"Evaluation slice provides {spinup_days_test} warm-up days but seq_len={self.seq_len} "
                f"requires at least {self.seq_len - 1}; load the evaluation slice with spinup_days >= seq_len - 1."
            )
        if self.seq_len > total_tr_days:
            raise ValueError(
                f"seq_len={self.seq_len} exceeds the training slice length {total_tr_days}."
            )

        # ---- leak-free statistics: training basins x training target window only ----------
        q_masked = mask_spinup_discharge(qobs_train, spinup_days_train)
        fstats = feature_stats(x_dyn_train[:, spinup_days_train:, :], x_stat_train)
        tstats = target_stats(q_masked[:, spinup_days_train:])
        y_mean = float(tstats["y_mean"])
        y_std = float(tstats["y_std"])
        weights_np = nse_star_weights(tstats["basin_std_mm"], eps=self.nse_eps)

        xd_tr_norm = ((x_dyn_train - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32)
        xd_te_norm = ((x_dyn_test - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32)
        xs_tr_norm = ((x_stat_train - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
        xs_te_norm = ((x_stat_test - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
        if not np.all(np.isfinite(xs_tr_norm)) or not np.all(np.isfinite(xs_te_norm)):
            raise ValueError("Static attributes contain NaN/inf after standardization.")

        y_tr_norm = ((q_masked - y_mean) / y_std).astype(np.float32)
        window_ok = finite_window_mask(np.all(np.isfinite(xd_tr_norm), axis=2), self.seq_len)
        pairs = training_pairs(q_masked, window_ok, self.seq_len, spinup_days_train, basin_weights=weights_np)
        if pairs.shape[0] == 0:
            raise ValueError("No valid training windows: every training target is NaN or has NaN inputs.")

        targ_len = (
            max(1, self.seq_len // 3)
            if model_cls.__name__ == "TemporalFusionTransformerHydro" and self.seq_len >= 60
            else 1
        )
        if targ_len > 1:
            min_full_t = max(self.seq_len - 1, int(spinup_days_train) + targ_len - 1)
            full_mask = pairs[:, 1] >= min_full_t
            if np.any(full_mask):
                pairs = pairs[full_mask]

        xd_tr_t = torch.from_numpy(xd_tr_norm).to(self.device)
        xs_tr_t = torch.from_numpy(xs_tr_norm).to(self.device)
        y_tr_t = torch.from_numpy(np.nan_to_num(y_tr_norm, nan=0.0) if targ_len > 1 else y_tr_norm).to(
            self.device
        )
        y_valid_t = (
            torch.from_numpy(np.isfinite(y_tr_norm).astype(np.float32)).to(self.device)
            if targ_len > 1
            else None
        )
        w_tr_t = torch.from_numpy(weights_np).to(self.device)
        xd_te_t = torch.from_numpy(xd_te_norm).to(self.device)
        xs_te_t = torch.from_numpy(xs_te_norm).to(self.device)
        y_std_t = torch.tensor(y_std, dtype=torch.float32, device=self.device)
        pairs_t = torch.from_numpy(pairs).to(device=self.device, dtype=torch.long)

        n_batches = n_batches_for_epoch(
            pairs.shape[0], self.batch_size, self.n_batches_per_epoch, targ_len=targ_len
        )
        stat_dim = xs_tr_norm.shape[1]

        if len(seeds) >= 1 and self.device.type == "cuda" and model_cls is CudaLSTM and self.epochs > 0:
            from models.kratzert_lstm.cuda_multiseed import (
                fit_and_predict_multiseed_cuda,
                get_multiseed_cuda_ext,
            )

            ext = get_multiseed_cuda_ext()
            if ext is not None:
                models: list[nn.Module] = []
                rngs: list[np.random.RandomState] = []
                for seed in seeds:
                    torch.manual_seed(int(seed))
                    rngs.append(np.random.RandomState(int(seed)))
                    models.append(self._build_model(model_cls, dyn_dim, stat_dim))
                return fit_and_predict_multiseed_cuda(
                    ext=ext,
                    models=models,
                    rngs=rngs,
                    xd_tr_t=xd_tr_t,
                    xs_tr_t=xs_tr_t,
                    y_tr_t=y_tr_t,
                    w_tr_t=w_tr_t,
                    xd_te_t=xd_te_t,
                    xs_te_t=xs_te_t,
                    pairs_t=pairs_t,
                    n_batches=n_batches,
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    lr_schedule=self.lr_schedule,
                    clip_grad_norm=self.clip_grad_norm,
                    y_mean=y_mean,
                    y_std=y_std,
                    spinup_days_test=spinup_days_test,
                    n_target_days=n_target_days,
                    eval_batch_size=self.eval_batch_size,
                    device=self.device,
                )

        member_preds: list[np.ndarray] = []
        for seed in seeds:
            # Parameter initialization and dropout draw from the global torch RNG; mini-batch
            # sampling uses a seed-local generator so concurrent folds do not perturb it.
            torch.manual_seed(int(seed))
            rng = np.random.RandomState(int(seed))
            model = self._build_model(model_cls, dyn_dim, stat_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_for_epoch(self.lr_schedule, 0))
            use_amp = self.device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            use_last_only = bool(getattr(model, "supports_last_only", False))
            model.train()
            for epoch in range(self.epochs):
                lr_now = lr_for_epoch(self.lr_schedule, epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                perm = rng.permutation(pairs.shape[0])
                idx_epoch = torch.from_numpy(perm[: n_batches * self.batch_size]).to(
                    device=self.device, dtype=torch.long
                )
                for batch_i in range(n_batches):
                    batch_rows = pairs_t[
                        idx_epoch[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]
                    ]
                    b_idx = batch_rows[:, 0]
                    t_end = batch_rows[:, 1]
                    x_b = gather_windows(xd_tr_t, b_idx, t_end, self.seq_len)
                    s_b = xs_tr_t[b_idx]

                    optimizer.zero_grad(set_to_none=True)
                    if targ_len == 1:
                        y_true = y_tr_t[b_idx, t_end]
                        w_b = w_tr_t[b_idx]
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            if use_last_only:
                                y_hat = model(x_b, s_b, last_only=True)[:, -1].float()
                            else:
                                y_hat = model(x_b, s_b)[:, -1].float()
                        loss = nse_star_loss((y_hat - y_true) * y_std_t, w_b)
                    else:
                        assert y_valid_t is not None
                        y_true = gather_target_windows(y_tr_t, b_idx, t_end, targ_len)
                        m_b = gather_target_windows(y_valid_t, b_idx, t_end, targ_len)
                        w_b = w_tr_t[b_idx].unsqueeze(1) * m_b
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            y_hat = model(x_b, s_b)[:, -targ_len:].float()
                        err_sq = ((y_hat - y_true) * y_std_t) ** 2
                        loss = torch.sum(err_sq * w_b) / torch.clamp(m_b.sum(), min=1.0)
                    scaler.scale(loss).backward()
                    if self.clip_grad_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

            model.eval()
            if hasattr(model, "predict_eval_series"):
                p_norm = (
                    model.predict_eval_series(  # type: ignore[operator]
                        xd_te_t,
                        xs_te_t,
                        self.seq_len,
                        spinup_days_test,
                        n_target_days,
                        batch_size=self.eval_batch_size,
                        use_amp=use_amp,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:

                def _predict_batch(
                    b_idx: torch.Tensor,
                    t_end: torch.Tensor,
                    _m: nn.Module = model,
                    _amp: bool = use_amp,
                ) -> torch.Tensor:
                    x_b = gather_windows(xd_te_t, b_idx, t_end, self.seq_len)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=_amp):
                        return _m(x_b, xs_te_t[b_idx])[:, -1].float()

                p_norm = predict_windowed(
                    _predict_batch,
                    n_basins=int(xd_te_t.shape[0]),
                    t_first=spinup_days_test,
                    n_target_days=n_target_days,
                    batch_size=self.eval_batch_size,
                    device=self.device,
                )
            member_preds.append(np.clip(p_norm * y_std + y_mean, 0.0, None).astype(np.float32))

        members_arr = np.stack(member_preds, axis=0).astype(np.float32)
        mean_arr = np.mean(members_arr, axis=0, dtype=np.float32)
        return mean_arr, members_arr
