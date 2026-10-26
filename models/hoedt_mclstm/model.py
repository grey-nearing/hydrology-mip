"""Hoedt et al. (2021) Mass-Conserving LSTM (MC-LSTM) architecture and trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.kratzert_lstm.model import nse_star_loss
from models.kratzert_lstm.windows import (
    feature_stats,
    finite_window_mask,
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

# Hoedt et al. (2021) hydrology configuration (NeuralHydrology `mclstm`): Adam with
# learning rate 1e-2 (epochs 0-19), 5e-3 (20-24), 1e-3 (25-29).
DEFAULT_MCLSTM_LR_STEP_FACTORS: dict[int, float] = {0: 1.0, 20: 0.5, 25: 0.1}


class _Gate(nn.Module):
    """Sigmoid output gate for MC-LSTM."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class _NormalizedGate(nn.Module):
    """L1-normalized simplex gate (input distribution or mass redistribution matrix) for MC-LSTM."""

    def __init__(self, in_features: int, out_shape: tuple[int, int], normalizer: str) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape
        if normalizer == "normalized_sigmoid":
            self.activation: nn.Module = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported normalizer {normalizer!r}.")
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, *self.out_shape)
        return F.normalize(self.activation(h), p=1.0, dim=-1)


class MCLSTM(nn.Module):
    """Mass-Conserving LSTM (Hoedt et al., 2021) with trash cell at index 0.

    `initial_forget_bias` initializes the output-gate bias, following NeuralHydrology's
    `mclstm`: positive values open the output gate at the start of training (mass leaves the
    system readily), negative values close it (the model must learn when to release mass).
    The default `3.0` matches the NeuralHydrology configurations used for the MIP MC-LSTM runs.
    """

    def __init__(
        self,
        aux_dyn_dim: int,
        stat_in_dim: int = 27,
        hidden_size: int = 64,
        mass_input_size: int = 1,
        initial_forget_bias: float = 3.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 1:
            raise ValueError("MC-LSTM requires hidden_size >= 2 (1 trash cell + >=1 active mass cell).")
        if mass_input_size < 1:
            raise ValueError("MC-LSTM requires at least one mass input.")
        self.hidden_size = hidden_size
        self.mass_input_size = mass_input_size
        aux_input_size = aux_dyn_dim + stat_in_dim
        gate_inputs = aux_input_size + hidden_size + mass_input_size

        self.output_gate = _Gate(in_features=gate_inputs, out_features=hidden_size)
        self.input_gate = _NormalizedGate(
            in_features=gate_inputs,
            out_shape=(mass_input_size, hidden_size),
            normalizer="normalized_sigmoid",
        )
        self.redistribution = _NormalizedGate(
            in_features=gate_inputs,
            out_shape=(hidden_size, hidden_size),
            normalizer="normalized_relu",
        )
        nn.init.constant_(self.output_gate.fc.bias, val=float(initial_forget_bias))

    def forward(self, x_m: torch.Tensor, x_a_dyn: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_m.shape
        x_s_exp = x_s.unsqueeze(1).expand(-1, seq_len, -1)
        x_a = torch.cat([x_a_dyn, x_s_exp], dim=-1)
        ct = x_m.new_zeros((batch_size, self.hidden_size))

        q_out: list[torch.Tensor] = []
        for t in range(seq_len):
            xt_m = x_m[:, t, :]
            xt_a = x_a[:, t, :]
            c_norm = ct / (ct.norm(p=1, dim=-1, keepdim=True) + 1e-5)
            features = torch.cat([xt_m, xt_a, c_norm], dim=-1)
            i_gate = self.input_gate(features)
            r_gate = self.redistribution(features)
            o_gate = self.output_gate(features)
            m_in = torch.bmm(xt_m.unsqueeze(1), i_gate).squeeze(1)
            m_sys = torch.bmm(ct.unsqueeze(1), r_gate).squeeze(1)
            m_new = m_in + m_sys
            mt_out = o_gate * m_new
            ct = (1.0 - o_gate) * m_new
            q_out.append(mt_out[:, 1:].sum(dim=-1))

        return torch.stack(q_out, dim=1)


def _mean_over_products(x: np.ndarray) -> np.ndarray:
    """NaN-aware mean over the last axis without warnings; all-NaN rows stay NaN."""
    finite = np.isfinite(x)
    count = finite.sum(axis=-1)
    total = np.where(finite, x, 0.0).sum(axis=-1)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _validate_precipitation_columns(
    prcp_col_indices: Sequence[int], n_dyn: int, *arrays: np.ndarray
) -> list[int]:
    cols = [int(c) for c in prcp_col_indices]
    if not cols:
        raise ValueError("MC-LSTM requires at least one precipitation column as mass input.")
    if len(set(cols)) != len(cols) or min(cols) < 0 or max(cols) >= n_dyn:
        raise ValueError(f"Invalid precipitation column indices {cols} for {n_dyn} dynamic features.")
    for arr in arrays:
        vals = arr[:, :, cols]
        finite = vals[np.isfinite(vals)]
        if finite.size and float(finite.min()) < 0.0:
            raise ValueError(
                "Precipitation mass-input columns contain negative values; the column indices do not "
                "point at precipitation (expected 'prcp(mm/day)_<product>')."
            )
    return cols


class MCLSTMTrainer:
    """Trainer for Hoedt et al. (2021) MC-LSTM.

    * Mass input: precipitation in **raw mm/d** (never standardized); when several forcing
      products are supplied their precipitation is averaged so the mass budget stays physical.
    * Auxiliary inputs: the remaining forcings standardized over training basins x training
      target days plus standardized static attributes.
    * Target: discharge in raw mm/d; loss is the basin-averaged NSE* with per-basin std in mm/d
      (training target window only), `eps = 0.1`.
    * Optimization: Adam with the Hoedt et al. step schedule (default `{0: 1e-2, 20: 5e-3,
      25: 1e-3}`), gradient-norm clipping, full coverage of all valid windows per epoch.
    * Inference: sequence-to-one over `seq_len`-day windows for every evaluation day.
    """

    def __init__(
        self,
        seq_len: int = 365,
        batch_size: int = 256,
        epochs: int = 30,
        lr: float = 1e-2,
        hidden_size: int = 64,
        device: str | torch.device | None = None,
        lr_schedule: Mapping[int | str, float] | None = None,
        initial_forget_bias: float = 3.0,
        nse_eps: float = 0.1,
        clip_grad_norm: float = 1.0,
        n_batches_per_epoch: int | None = None,
        eval_batch_size: int = 1024,
    ) -> None:
        if seq_len < 1 or batch_size < 1 or epochs < 0:
            raise ValueError("seq_len and batch_size must be >= 1 and epochs >= 0.")
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.hidden_size = int(hidden_size)
        self.device = resolve_device(device)
        self.lr_schedule = normalize_lr_schedule(lr_schedule, self.lr, DEFAULT_MCLSTM_LR_STEP_FACTORS)
        self.initial_forget_bias = float(initial_forget_bias)
        self.nse_eps = float(nse_eps)
        self.clip_grad_norm = float(clip_grad_norm)
        self.n_batches_per_epoch = None if n_batches_per_epoch is None else int(n_batches_per_epoch)
        self.eval_batch_size = int(eval_batch_size)

    def fit_and_predict(
        self,
        x_dyn_train: np.ndarray,
        x_stat_train: np.ndarray,
        qobs_train: np.ndarray,
        x_dyn_test: np.ndarray,
        x_stat_test: np.ndarray,
        n_target_days: int,
        spinup_days_train: int,
        prcp_col_indices: Sequence[int],
        seeds: Sequence[int] = (123,),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train one MC-LSTM per seed and predict the evaluation slice (see `KratzertLSTMTrainer`)."""
        if len(seeds) == 0:
            raise ValueError("At least one seed is required.")
        x_dyn_train = np.asarray(x_dyn_train, dtype=np.float32)
        x_dyn_test = np.asarray(x_dyn_test, dtype=np.float32)
        x_stat_train = np.asarray(x_stat_train, dtype=np.float32)
        x_stat_test = np.asarray(x_stat_test, dtype=np.float32)
        n_target_days = int(n_target_days)
        spinup_days_train = int(spinup_days_train)

        n_tr, total_tr_days, n_dyn = x_dyn_train.shape
        if qobs_train.shape != (n_tr, total_tr_days):
            raise ValueError("qobs_train must have shape (n_train_basins, n_train_slice_days).")
        if x_dyn_test.shape[2] != n_dyn or x_stat_test.shape[1] != x_stat_train.shape[1]:
            raise ValueError("Train/test feature dimensions differ.")
        spinup_days_test = int(x_dyn_test.shape[1]) - n_target_days
        if spinup_days_test < self.seq_len - 1:
            raise ValueError(
                f"Evaluation slice provides {spinup_days_test} warm-up days but seq_len={self.seq_len} "
                f"requires at least {self.seq_len - 1}."
            )
        if self.seq_len > total_tr_days:
            raise ValueError(f"seq_len={self.seq_len} exceeds the training slice length {total_tr_days}.")

        prcp_cols = _validate_precipitation_columns(prcp_col_indices, n_dyn, x_dyn_train, x_dyn_test)
        aux_cols = [c for c in range(n_dyn) if c not in prcp_cols]
        if not aux_cols:
            raise ValueError("MC-LSTM requires at least one non-precipitation auxiliary forcing.")

        xm_tr = _mean_over_products(x_dyn_train[:, :, prcp_cols])[..., None]
        xm_te = _mean_over_products(x_dyn_test[:, :, prcp_cols])[..., None]

        # ---- leak-free statistics: training basins x training target window only ----------
        q_masked = mask_spinup_discharge(qobs_train, spinup_days_train)
        fstats = feature_stats(x_dyn_train[:, spinup_days_train:, :][:, :, aux_cols], x_stat_train)
        tstats = target_stats(q_masked[:, spinup_days_train:])
        weights_np = nse_star_weights(tstats["basin_std_mm"], eps=self.nse_eps)

        xa_tr_norm = ((x_dyn_train[:, :, aux_cols] - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32)
        xa_te_norm = ((x_dyn_test[:, :, aux_cols] - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32)
        xs_tr_norm = ((x_stat_train - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
        xs_te_norm = ((x_stat_test - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
        if not np.all(np.isfinite(xs_tr_norm)) or not np.all(np.isfinite(xs_te_norm)):
            raise ValueError("Static attributes contain NaN/inf after standardization.")

        inputs_finite = np.all(np.isfinite(xa_tr_norm), axis=2) & np.isfinite(xm_tr[:, :, 0])
        window_ok = finite_window_mask(inputs_finite, self.seq_len)
        pairs = training_pairs(q_masked, window_ok, self.seq_len, spinup_days_train, basin_weights=weights_np)
        if pairs.shape[0] == 0:
            raise ValueError("No valid training windows: every training target is NaN or has NaN inputs.")

        xm_tr_t = torch.from_numpy(xm_tr).to(self.device)
        xa_tr_t = torch.from_numpy(xa_tr_norm).to(self.device)
        xs_tr_t = torch.from_numpy(xs_tr_norm).to(self.device)
        y_tr_t = torch.from_numpy(q_masked).to(self.device)
        w_tr_t = torch.from_numpy(weights_np).to(self.device)
        xm_te_t = torch.from_numpy(xm_te).to(self.device)
        xa_te_t = torch.from_numpy(xa_te_norm).to(self.device)
        xs_te_t = torch.from_numpy(xs_te_norm).to(self.device)

        n_batches = n_batches_for_epoch(pairs.shape[0], self.batch_size, self.n_batches_per_epoch)

        member_preds: list[np.ndarray] = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            rng = np.random.RandomState(int(seed))
            model = MCLSTM(
                aux_dyn_dim=len(aux_cols),
                stat_in_dim=xs_tr_norm.shape[1],
                hidden_size=self.hidden_size,
                mass_input_size=1,
                initial_forget_bias=self.initial_forget_bias,
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_for_epoch(self.lr_schedule, 0))

            model.train()
            for epoch in range(self.epochs):
                lr_now = lr_for_epoch(self.lr_schedule, epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                perm = rng.permutation(pairs.shape[0])
                for batch_i in range(n_batches):
                    rows = pairs[perm[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]]
                    b_idx = torch.as_tensor(rows[:, 0], dtype=torch.long, device=self.device)
                    t_end = torch.as_tensor(rows[:, 1], dtype=torch.long, device=self.device)
                    xm_b = gather_windows(xm_tr_t, b_idx, t_end, self.seq_len)
                    xa_b = gather_windows(xa_tr_t, b_idx, t_end, self.seq_len)
                    xs_b = xs_tr_t[b_idx]
                    y_true = y_tr_t[b_idx, t_end]

                    optimizer.zero_grad(set_to_none=True)
                    y_hat = model(xm_b, xa_b, xs_b)[:, -1]
                    loss = nse_star_loss(y_hat - y_true, w_tr_t[b_idx])
                    loss.backward()
                    if self.clip_grad_norm > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                    optimizer.step()

            model.eval()

            def _predict_batch(b_idx: torch.Tensor, t_end: torch.Tensor, _m: nn.Module = model) -> torch.Tensor:
                xm_b = gather_windows(xm_te_t, b_idx, t_end, self.seq_len)
                xa_b = gather_windows(xa_te_t, b_idx, t_end, self.seq_len)
                return _m(xm_b, xa_b, xs_te_t[b_idx])[:, -1]

            p_mm = predict_windowed(
                _predict_batch,
                n_basins=int(xm_te_t.shape[0]),
                t_first=spinup_days_test,
                n_target_days=n_target_days,
                batch_size=self.eval_batch_size,
                device=self.device,
            )
            member_preds.append(np.clip(p_mm, 0.0, None).astype(np.float32))

        members_arr = np.stack(member_preds, axis=0).astype(np.float32)
        mean_arr = np.mean(members_arr, axis=0, dtype=np.float32)
        return mean_arr, members_arr
