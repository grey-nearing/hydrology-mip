# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Independent first-party PyTorch 2.0 implementation of the RR-Former architecture
# described in Yin et al. (2022, J. Hydrol., doi:10.1016/j.jhydrol.2022.127781);
# no upstream source code from https://github.com/iThronne/RR-Former was copied.

"""Yin et al. (2022, J. Hydrol.) RR-Former: Rainfall-Runoff Modeling Based on Transformer.

Paper:
  Yin, H., Guo, Z., Zhang, X., Chen, J., & Zhang, Y. (2022). RR-Former:
  Rainfall-runoff modeling based on Transformer. Journal of Hydrology, 609,
  127781. https://doi.org/10.1016/j.jhydrol.2022.127781
  Reference Code: https://github.com/iThronne/RR-Former

Architecture & Protocol from Yin et al. (2022):
  - Encoder-decoder Transformer (`d_model=64`, `nhead=4`, `num_encoder_layers=4`,
    `num_decoder_layers=4`, `dim_feedforward=256`, `activation="relu"`, `dropout=0.1`)
    with learnable positional embeddings (`nn.Embedding(max_len, d_model)`).
  - Sequence window `seq_len = 16` (`past_len = 15` historical days `t-15..t-1` +
    target day `t`):
    * Encoder input (`src_len = seq_len`): meteorological forcings `X_{t-15:t}` +
      static catchment attributes.
    * Decoder input (`tgt_len = seq_len`, non-autoregressive `NAR` mode):
      positions `0..seq_len-2` contain the 15 days of standardized lagged
      observed runoff `[q_lag_std, q_lag_valid]` (`Q_{t-15:t-1}`), while the
      target position `seq_len-1` is zero-padded (`0.0`) and masked in decoder
      self-attention (`tgt_key_padding_mask`) as specified in Yin et al. (2022),
      so the target query token attends to all 15 lagged
      runoff tokens and all 16 encoder meteorological memory tokens.
  - Two-stage training:
    1. Global pretraining across all training basins using basin-variance-weighted
       `NSELoss` (`eps = 0.1`), `Adam`, warmup + exponential decay (`gamma = 0.99`),
       and validation checkpoint selection.
    2. Per-basin fine-tuning initialized from the global pretrained checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.feng2020_di_lstm.model import build_lagged_discharge_features
from models.kratzert_lstm.model import nse_star_loss
from models.kratzert_lstm.windows import (
    feature_stats,
    finite_window_mask,
    gather_windows,
    mask_spinup_discharge,
    nse_star_weights,
    predict_windowed,
    resolve_device,
    target_stats,
    training_pairs,
)


class RRFormerEncoderLayer(nn.Module):
    """Post-LayerNorm Transformer encoder layer (Yin et al., 2022)."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.nhead = int(nhead)
        self.d_head = int(d_model) // int(nhead)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q, k, v = (
            self.qkv(x)
            .view(bsz, seq_len, 3, self.nhead, self.d_head)
            .unbind(dim=2)
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (
            F.scaled_dot_product_attention(q, k, v)
            .transpose(1, 2)
            .reshape(bsz, seq_len, d_model)
        )
        x = self.norm1(x + self.fc(attn))
        ff = self.dropout(self.ffn2(F.relu(self.ffn1(x))))
        return self.norm2(x + ff)


class RRFormerDecoderLayer(nn.Module):
    """Post-LayerNorm non-autoregressive Transformer decoder layer (Yin et al., 2022)."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.nhead = int(nhead)
        self.d_head = int(d_model) // int(nhead)
        self.self_q = nn.Linear(d_model, d_model, bias=False)
        self.self_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.self_fc = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross_fc = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, last_only: bool = False
    ) -> torch.Tensor:
        bsz, tgt_len, d_model = tgt.shape
        src_len = memory.size(1)
        # In NAR mode (pt=0), get_subsequence_mask masks the last column (-pred_len = -1)
        # with -inf, which is mathematically identical to attending over keys/values 0..tgt_len-2.
        kv_in = tgt[:, :-1, :] if tgt_len > 1 else tgt
        kv_len = kv_in.size(1)
        q_in = tgt[:, -1:, :] if last_only else tgt
        q_len = q_in.size(1)

        q = self.self_q(q_in).view(bsz, q_len, self.nhead, self.d_head).transpose(1, 2)
        k, v = (
            self.self_kv(kv_in)
            .view(bsz, kv_len, 2, self.nhead, self.d_head)
            .unbind(dim=2)
        )
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn1 = (
            F.scaled_dot_product_attention(q, k, v)
            .transpose(1, 2)
            .reshape(bsz, q_len, d_model)
        )
        x = self.norm1(q_in + self.self_fc(attn1))

        cq = self.cross_q(x).view(bsz, q_len, self.nhead, self.d_head).transpose(1, 2)
        ck, cv = (
            self.cross_kv(memory)
            .view(bsz, src_len, 2, self.nhead, self.d_head)
            .unbind(dim=2)
        )
        ck = ck.transpose(1, 2)
        cv = cv.transpose(1, 2)
        attn2 = (
            F.scaled_dot_product_attention(cq, ck, cv)
            .transpose(1, 2)
            .reshape(bsz, q_len, d_model)
        )
        x = self.norm2(x + self.cross_fc(attn2))

        ff = self.dropout(self.ffn2(F.relu(self.ffn1(x))))
        return self.norm3(x + ff)


class RRFormer(nn.Module):
    """4-encoder / 4-decoder RR-Former architecture (Yin et al., 2022)."""

    def __init__(
        self,
        dyn_in_dim: int,
        stat_in_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.dyn_in_dim = int(dyn_in_dim)
        self.stat_in_dim = int(stat_in_dim)
        self.d_model = int(d_model)
        self.met_in_dim = max(1, self.dyn_in_dim - 2)

        enc_in_dim = self.met_in_dim + self.stat_in_dim
        dec_in_dim = 2  # [q_lag_std, q_lag_valid] at each step (15 lagged steps + 1 zero token)

        self.enc_proj = nn.Linear(enc_in_dim, self.d_model)
        self.dec_proj = nn.Linear(dec_in_dim, self.d_model)
        self.enc_pos_emb = nn.Embedding(max_len, self.d_model)
        self.dec_pos_emb = nn.Embedding(max_len, self.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                RRFormerEncoderLayer(
                    d_model=self.d_model,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(int(num_encoder_layers))
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                RRFormerDecoderLayer(
                    d_model=self.d_model,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(int(num_decoder_layers))
            ]
        )
        self.head = nn.Linear(self.d_model, 1)

    def encode_features(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        """Return `(batch, d_model)` decoder representation at the target step."""
        bsz, seq_len, _ = x_dyn.shape
        x_met = x_dyn[:, :, : self.met_in_dim]
        x_lag = x_dyn[:, :, self.met_in_dim :]  # [q_lag_std, q_lag_valid] at each step tau -> Q_{tau-1}

        xs = x_stat.unsqueeze(1).expand(-1, seq_len, -1)
        enc_in = torch.cat([x_met, xs], dim=-1)

        pos = torch.arange(seq_len, device=x_dyn.device, dtype=torch.long).unsqueeze(0)
        memory = self.enc_proj(enc_in) + self.enc_pos_emb(pos)
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory)

        # Non-autoregressive (NAR) decoder input matching iThronne/RR-Former:
        # In x_lag[:, 1:, :], step k (k=0..seq_len-2, i.e. days t-(seq_len-2)..t) carries Q_{day-1},
        # which is the 15-day lagged runoff sequence Q_{t-15 : t-1}!
        # The last token (index seq_len-1, target day t) is zero-padded (0.0) and masked in
        # decoder self-attention so no token attends to the zero placeholder.
        dec_tokens = torch.zeros(
            (bsz, seq_len, 2), dtype=x_dyn.dtype, device=x_dyn.device
        )
        if seq_len > 1:
            dec_tokens[:, : seq_len - 1, :] = x_lag[:, 1:, :]
        dec_out = self.dec_proj(dec_tokens) + self.dec_pos_emb(pos)
        n_dec = len(self.decoder_layers)
        for idx, dec_layer in enumerate(self.decoder_layers):
            dec_out = dec_layer(dec_out, memory, last_only=(idx == n_dec - 1))
        return dec_out[:, -1, :]

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        """Return `(batch,)` standardized discharge prediction for the window endpoint."""
        h = self.encode_features(x_dyn, x_stat)
        return self.head(h).squeeze(-1)


class RRFormerTrainer:
    """Global pretraining + per-basin fine-tuning trainer for Yin et al. (2022) RR-Former."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        seq_len: int = 16,
        dropout: float = 0.1,
        batch_size: int = 512,
        epochs: int = 35,
        finetune_l2_reg: float = 10.0,
        lr: float = 1e-3,
        lr_schedule: Mapping[int | str, float] | None = None,
        n_batches_per_epoch: int | None = None,
        clip_grad_norm: float = 1.0,
        eval_batch_size: int = 4096,
        lag_days: int = 1,
        lag_dropout_prob: float = 0.02,
        nse_eps: float = 0.1,
        device: str | torch.device | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.seq_len = int(seq_len)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.finetune_l2_reg = float(finetune_l2_reg)
        self.lr = float(lr)
        self.lr_schedule = lr_schedule
        self.n_batches_per_epoch = (
            None if n_batches_per_epoch is None else int(n_batches_per_epoch)
        )
        self.clip_grad_norm = float(clip_grad_norm)
        self.eval_batch_size = int(eval_batch_size)
        self.lag_days = int(lag_days)
        self.lag_dropout_prob = float(lag_dropout_prob)
        self.nse_eps = float(nse_eps)
        self.device = resolve_device(device)

    def _lr_for_epoch(self, ep: int) -> float:
        """Warmup (5 epochs to 1e-3) + 0.99 exponential decay matching iThronne/RR-Former."""
        if self.lr_schedule is not None:
            best_k = -1
            cur = self.lr
            for k, v in self.lr_schedule.items():
                ki = int(k)
                if ki <= ep and ki >= best_k:
                    best_k = ki
                    cur = float(v)
            return cur
        warmup_ep = 5
        if ep < warmup_ep:
            return self.lr * float(ep + 1) / float(warmup_ep)
        return self.lr * (0.99 ** float(ep - warmup_ep + 1))

    @torch.no_grad()
    def _fit_per_basin_finetuned_heads(
        self,
        *,
        model: RRFormer,
        xd_tr_t: torch.Tensor,
        xs_tr_t: torch.Tensor,
        y_tr_t: torch.Tensor,
        q_tr_masked: np.ndarray,
        spinup_days_train: int,
        use_amp: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fine-tune per-basin parameters `(w_b, bias_b)` with validation-selected L2 regularization."""
        model.eval()
        n_tr, total_days, _ = xd_tr_t.shape
        t_first = max(self.seq_len - 1, int(spinup_days_train))
        n_tr_days = total_days - t_first

        w_global = model.head.weight.detach().float().squeeze(0)  # (D,)
        b_global = model.head.bias.detach().float().squeeze(0)  # ()
        d_dim = w_global.numel()

        w_basins = w_global.unsqueeze(0).repeat(n_tr, 1)
        b_basins = b_global.repeat(n_tr)
        if n_tr_days <= d_dim or self.finetune_l2_reg <= 0.0:
            return w_basins, b_basins

        n_pairs = n_tr * n_tr_days
        feats = torch.empty(
            (n_tr, n_tr_days, d_dim), dtype=torch.float32, device=xd_tr_t.device
        )
        for start in range(0, n_pairs, self.eval_batch_size):
            flat = np.arange(
                start, min(start + self.eval_batch_size, n_pairs), dtype=np.int64
            )
            b_np = flat // n_tr_days
            k_np = flat % n_tr_days
            b_idx = torch.as_tensor(b_np, dtype=torch.long, device=xd_tr_t.device)
            t_end = torch.as_tensor(
                k_np + t_first, dtype=torch.long, device=xd_tr_t.device
            )
            x_b = gather_windows(xd_tr_t, b_idx, t_end, self.seq_len)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                h_b = model.encode_features(x_b, xs_tr_t[b_idx]).float()
            feats[
                b_idx, torch.as_tensor(k_np, dtype=torch.long, device=xd_tr_t.device)
            ] = h_b

        valid_mask = torch.from_numpy(
            (
                np.isfinite(q_tr_masked[:, t_first:])
                & (q_tr_masked[:, t_first:] >= 0.0)
            ).astype(np.float32)
        ).to(xd_tr_t.device)
        y_target = y_tr_t[:, t_first:]
        ones = torch.ones(
            (n_tr, n_tr_days, 1), dtype=torch.float32, device=xd_tr_t.device
        )
        phi = torch.cat([feats, ones], dim=-1)  # (n_tr, T, D+1)
        theta_0 = torch.cat([w_global, b_global.unsqueeze(0)], dim=0)  # (D+1,)

        # Per-basin validation split (first 20% of training period, matching Yin et al. 2022 Task 2
        # which used 1999-2001 as validation and 2001-2008 as training) to select best regularization per basin
        val_len = max(1, n_tr_days // 5)
        if n_tr_days - val_len > d_dim + 10:
            val_m = valid_mask[:, :val_len]
            fit_m = valid_mask[:, val_len:]
            phi_fit = phi[:, val_len:, :] * fit_m.unsqueeze(-1)
            y_res_fit = (y_target[:, val_len:] - torch.matmul(phi[:, val_len:, :], theta_0)) * fit_m
            gram_fit = torch.bmm(phi_fit.transpose(1, 2), phi_fit)
            rhs_fit = torch.bmm(phi_fit.transpose(1, 2), y_res_fit.unsqueeze(-1)).squeeze(-1)

            eye = torch.eye(d_dim + 1, dtype=torch.float32, device=xd_tr_t.device).unsqueeze(0)
            best_val_mse = (
                ((torch.matmul(phi[:, :val_len, :], theta_0) - y_target[:, :val_len]) ** 2 * val_m).sum(dim=1)
                / torch.clamp(val_m.sum(dim=1), min=1.0)
            )
            best_lambda = torch.full((n_tr,), 1e6, dtype=torch.float32, device=xd_tr_t.device)
            for lam in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
                delta_c = torch.linalg.solve(gram_fit + lam * eye, rhs_fit)
                theta_c = theta_0.unsqueeze(0) + delta_c
                pred_val = torch.bmm(phi[:, :val_len, :], theta_c.unsqueeze(-1)).squeeze(-1)
                mse_c = (((pred_val - y_target[:, :val_len]) ** 2) * val_m).sum(dim=1) / torch.clamp(
                    val_m.sum(dim=1), min=1.0
                )
                improved = (mse_c < best_val_mse) & (val_m.sum(dim=1) >= 30.0)
                best_val_mse = torch.where(improved, mse_c, best_val_mse)
                best_lambda = torch.where(
                    improved, torch.full_like(best_lambda, lam), best_lambda
                )
        else:
            best_lambda = torch.full(
                (n_tr,), self.finetune_l2_reg, dtype=torch.float32, device=xd_tr_t.device
            )

        # Final refit on all training steps using the validation-selected lambda_b per basin
        phi_m = phi * valid_mask.unsqueeze(-1)
        y_res = (y_target - torch.matmul(phi, theta_0)) * valid_mask
        reg_mats = best_lambda.view(n_tr, 1, 1) * torch.eye(
            d_dim + 1, dtype=torch.float32, device=xd_tr_t.device
        ).unsqueeze(0)
        a_mat = torch.bmm(phi_m.transpose(1, 2), phi_m) + reg_mats
        rhs = torch.bmm(phi_m.transpose(1, 2), y_res.unsqueeze(-1)).squeeze(-1)
        delta = torch.linalg.solve(a_mat, rhs)
        theta = theta_0.unsqueeze(0) + delta
        has_enough = (valid_mask.sum(dim=1) >= 60.0) & (best_lambda < 1e5)
        w_out = torch.where(has_enough.unsqueeze(1), theta[:, :d_dim], w_basins)
        b_out = torch.where(has_enough, theta[:, d_dim], b_basins)
        return w_out, b_out

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
        train_basins: Sequence[str] | None = None,
        test_basins: Sequence[str] | None = None,
        seeds: Sequence[int] = (601, 602, 603),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pretrain globally across all training windows, fine-tune per-basin, and predict."""
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds must be non-empty.")

        x_dyn_train = np.asarray(x_dyn_train, dtype=np.float32)
        x_stat_train = np.asarray(x_stat_train, dtype=np.float32)
        x_dyn_test = np.asarray(x_dyn_test, dtype=np.float32)
        x_stat_test = np.asarray(x_stat_test, dtype=np.float32)

        q_tr_masked = mask_spinup_discharge(qobs_train, spinup_days_train)
        q_te_masked = mask_spinup_discharge(qobs_test, spinup_days_test)

        fstats = feature_stats(x_dyn_train[:, spinup_days_train:, :], x_stat_train)
        tstats = target_stats(q_tr_masked[:, spinup_days_train:])
        y_mean = float(tstats["y_mean"])
        y_std = float(tstats["y_std"])
        weights_np = nse_star_weights(tstats["basin_std_mm"], eps=self.nse_eps)

        xd_tr_met = np.nan_to_num(
            ((x_dyn_train - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32),
            nan=0.0,
        )
        xd_te_met = np.nan_to_num(
            ((x_dyn_test - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32),
            nan=0.0,
        )
        xs_tr_norm = ((x_stat_train - fstats["xs_mean"]) / fstats["xs_std"]).astype(
            np.float32
        )
        xs_te_norm = ((x_stat_test - fstats["xs_mean"]) / fstats["xs_std"]).astype(
            np.float32
        )

        lag_tr = build_lagged_discharge_features(
            q_tr_masked, q_mean=y_mean, q_std=y_std, lag_days=self.lag_days
        )
        lag_te = build_lagged_discharge_features(
            q_te_masked, q_mean=y_mean, q_std=y_std, lag_days=self.lag_days
        )

        if self.lag_dropout_prob > 0.0:
            drop_rng = np.random.RandomState(seed_list[0] + 6151)
            drop_mask = (
                drop_rng.uniform(0.0, 1.0, size=lag_tr.shape[:2]) < self.lag_dropout_prob
            )
            lag_tr = lag_tr.copy()
            lag_tr[drop_mask, 0] = 0.0
            lag_tr[drop_mask, 1] = 0.0

        xd_tr_norm = np.concatenate([xd_tr_met, lag_tr], axis=-1).astype(np.float32)
        xd_te_norm = np.concatenate([xd_te_met, lag_te], axis=-1).astype(np.float32)
        y_tr_norm = ((q_tr_masked - y_mean) / y_std).astype(np.float32)

        window_ok = finite_window_mask(
            np.all(np.isfinite(xd_tr_norm), axis=2), self.seq_len
        )
        pairs = training_pairs(
            q_tr_masked,
            window_ok,
            self.seq_len,
            spinup_days_train,
            basin_weights=weights_np,
        )
        if pairs.shape[0] == 0:
            raise ValueError("No valid training windows in RRFormerTrainer.")

        # Hold out 10% of windows as validation for checkpoint selection (matching pretrain.py saving_metric)
        split_rng = np.random.RandomState(seed_list[0] + 777)
        perm_p = split_rng.permutation(pairs.shape[0])
        if pairs.shape[0] >= 200:
            n_val_p = min(16384, max(512, pairs.shape[0] // 15))
            val_pairs = pairs[perm_p[:n_val_p]]
            train_pairs = pairs[perm_p[n_val_p:]]
        else:
            train_pairs = pairs
            val_pairs = pairs

        if self.n_batches_per_epoch is not None and self.n_batches_per_epoch > 0:
            n_batches = int(self.n_batches_per_epoch)
        else:
            n_batches = max(1, train_pairs.shape[0] // self.batch_size)
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
        w_tr_t = torch.from_numpy(weights_np).to(dev)
        xd_te_t = torch.from_numpy(xd_te_norm).to(dev)
        xs_te_t = torch.from_numpy(xs_te_norm).to(dev)
        y_std_t = torch.tensor(y_std, dtype=torch.float32, device=dev)
        train_pairs_t = torch.from_numpy(train_pairs).to(device=dev, dtype=torch.long)
        val_pairs_t = torch.from_numpy(val_pairs).to(device=dev, dtype=torch.long)

        tr_index_map: dict[str, int] = {}
        if train_basins is not None:
            for idx, b in enumerate(train_basins):
                tr_index_map[str(b)] = idx

        dyn_dim = xd_tr_norm.shape[2]
        stat_dim = xs_tr_norm.shape[1]
        member_preds: list[np.ndarray] = []

        for seed in seed_list:
            torch.manual_seed(seed)
            if use_amp:
                torch.cuda.manual_seed_all(seed)
            model = RRFormer(
                dyn_in_dim=dyn_dim,
                stat_in_dim=stat_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                max_len=max(512, self.seq_len + 16),
            ).to(dev)
            fwd_model: nn.Module = model
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed + 4919)

            best_val_loss = float("inf")
            best_state = copy.deepcopy(model.state_dict())

            for ep in range(self.epochs):
                model.train()
                cur_lr = self._lr_for_epoch(ep)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr
                perm = torch.randperm(
                    train_pairs_t.size(0), generator=gen, device=dev
                )
                for b in range(n_batches):
                    s0 = (b * self.batch_size) % train_pairs_t.size(0)
                    idx = perm[s0 : s0 + self.batch_size]
                    if idx.numel() == 0:
                        continue
                    batch_rows = train_pairs_t[idx]
                    b_idx = batch_rows[:, 0]
                    t_end = batch_rows[:, 1]
                    x_b = gather_windows(xd_tr_t, b_idx, t_end, self.seq_len)
                    s_b = xs_tr_t[b_idx]
                    y_true = y_tr_t[b_idx, t_end]
                    w_b = w_tr_t[b_idx]

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        "cuda", dtype=torch.float16, enabled=use_amp
                    ):
                        y_hat = fwd_model(x_b, s_b).float()
                    loss = nse_star_loss((y_hat - y_true) * y_std_t, w_b)
                    scaler.scale(loss).backward()
                    if self.clip_grad_norm > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=self.clip_grad_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()

                # Validation checkpoint selection matching pretrain.py
                model.eval()
                with torch.no_grad():
                    v_losses: list[float] = []
                    for vs in range(0, val_pairs_t.size(0), self.eval_batch_size):
                        v_rows = val_pairs_t[vs : vs + self.eval_batch_size]
                        vb_idx = v_rows[:, 0]
                        vt_end = v_rows[:, 1]
                        vx_b = gather_windows(xd_tr_t, vb_idx, vt_end, self.seq_len)
                        with torch.amp.autocast(
                            "cuda", dtype=torch.float16, enabled=use_amp
                        ):
                            vy_hat = model(vx_b, xs_tr_t[vb_idx]).float()
                        vl = nse_star_loss(
                            (vy_hat - y_tr_t[vb_idx, vt_end]) * y_std_t, w_tr_t[vb_idx]
                        )
                        v_losses.append(float(vl.item()))
                    mean_vl = float(np.mean(v_losses)) if v_losses else float("inf")
                    if mean_vl < best_val_loss:
                        best_val_loss = mean_vl
                        best_state = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_state)
            w_tr_basins, b_tr_basins = self._fit_per_basin_finetuned_heads(
                model=model,
                xd_tr_t=xd_tr_t,
                xs_tr_t=xs_tr_t,
                y_tr_t=y_tr_t,
                q_tr_masked=q_tr_masked,
                spinup_days_train=spinup_days_train,
                use_amp=use_amp,
            )

            n_te = x_dyn_test.shape[0]
            w_global = model.head.weight.detach().float().squeeze(0)
            b_global = model.head.bias.detach().float().squeeze(0)
            w_te = w_global.unsqueeze(0).repeat(n_te, 1)
            b_te = b_global.repeat(n_te)
            if test_basins is not None and tr_index_map:
                for i_te, b_id in enumerate(test_basins):
                    i_tr = tr_index_map.get(str(b_id))
                    if i_tr is not None:
                        w_te[i_te] = w_tr_basins[i_tr]
                        b_te[i_te] = b_tr_basins[i_tr]
            elif n_te == x_dyn_train.shape[0] and test_basins is None:
                w_te = w_tr_basins
                b_te = b_tr_basins

            model.eval()

            def _predict_batch(
                b_idx: torch.Tensor,
                t_end: torch.Tensor,
                _m: RRFormer = model,
                _w: torch.Tensor = w_te,
                _b: torch.Tensor = b_te,
                _amp: bool = use_amp,
            ) -> torch.Tensor:
                x_b = gather_windows(xd_te_t, b_idx, t_end, self.seq_len)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=_amp):
                    h_b = _m.encode_features(x_b, xs_te_t[b_idx]).float()
                return (h_b * _w[b_idx]).sum(dim=-1) + _b[b_idx]

            p_norm = predict_windowed(
                _predict_batch,
                n_basins=n_te,
                t_first=spinup_days_test,
                n_target_days=n_target_days,
                batch_size=self.eval_batch_size,
                device=dev,
            )
            member_preds.append(
                np.clip(p_norm * y_std + y_mean, 0.0, None).astype(np.float32)
            )

        members = np.stack(member_preds, axis=0)
        return np.mean(members, axis=0).astype(np.float32), members
