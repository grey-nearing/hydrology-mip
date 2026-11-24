# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Independent first-party PyTorch implementation of the RR-Former architecture
# described in Yin et al. (2022, J. Hydrol., doi:10.1016/j.jhydrol.2022.127781);
# no upstream source code from https://github.com/iThronne/RR-Former was copied.

"""Yin et al. (2022, J. Hydrol.) RR-Former: Rainfall-Runoff Modeling Based on Transformer.

Paper:
  Yin, H., Guo, Z., Zhang, X., Chen, J., & Zhang, Y. (2022). RR-Former:
  Rainfall-runoff modeling based on Transformer. Journal of Hydrology, 609,
  127781. https://doi.org/10.1016/j.jhydrol.2022.127781

Faithful architecture and two-stage protocol:
  - Encoder-decoder Transformer (`d_model=64`, `nhead=4`, `num_encoder_layers=4`,
    `num_decoder_layers=4`, `dim_feedforward=256`, `dropout=0.1`, `decode_mode="NAR"`,
    `use_static=False`, `use_future_fea=True`) with learnable positional embeddings
    (`src_pos_emb`, `tgt_pos_emb`), Post-LayerNorm residual blocks, dropout applied
    exclusively to the feed-forward output before the final residual add, and a
    non-causal historical-only decoder self-attention pattern (every decoder position
    attends to all `past_len` historical runoff tokens and never to future queries).
  - Window (`src_len = past_len + pred_len`, `tgt_len = past_len + pred_len`):
      * Encoder input: `src_len` days `[t_0 - past_len .. t_0 + pred_len - 1]` of
        standardized meteorological forcings (`prcp, srad, tmax, tmin, vp`).
      * Decoder input: single-channel tensor of shape `(batch, tgt_len, 1)` holding
        standardized observed runoff over `[t_0 - past_len .. t_0 - 1]` in positions
        `0 .. past_len - 1` and zeros in positions `past_len .. tgt_len - 1`.
      * Readout: `Linear(d_model, 1)` on the trailing `pred_len` decoder positions.
  - Two-stage training:
      1. Global pretraining across all training basins with global z-score normalization,
         basin-variance-weighted `NSELoss` (`mean((y_hat - y)^2 / (std_b_mm + 0.1)^2)`),
         `Adam(lr=1e-3)`, `batch_size=512`, full-epoch coverage (`n_batches_per_epoch=None`),
         linear warmup over `0.25 * epochs` followed by `0.99` exponential decay, selecting
         the best checkpoint by validation per-lead mean NSE across all epochs.
      2. Full-network per-basin fine-tuning initialized from the best global checkpoint,
         re-normalizing each basin's train/val/test forcings and discharge with that
         basin's own training-period mean and standard deviation, updating ALL Transformer
         parameters with `Adam(lr=1e-3)` and `MSELoss` over full `finetune_epochs` with
         linear warmup (`0.25 * finetune_epochs`) + `0.99` decay, selecting each basin's
         best checkpoint by per-basin validation NSE.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.kratzert_lstm.windows import (
    mask_spinup_discharge,
    nse_star_weights,
    resolve_device,
)


def _compute_per_lead_nse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute per-lead NSE array of shape `(pred_len,)` ignoring NaN/negative targets."""
    pred_2d = np.asarray(y_pred, dtype=np.float64)
    true_2d = np.asarray(y_true, dtype=np.float64)
    if pred_2d.ndim == 1:
        pred_2d = pred_2d[:, None]
        true_2d = true_2d[:, None]
    pred_len = pred_2d.shape[1]
    out = np.full(pred_len, np.nan, dtype=np.float64)
    for h in range(pred_len):
        valid = np.isfinite(true_2d[:, h]) & (true_2d[:, h] >= 0.0) & np.isfinite(pred_2d[:, h])
        if int(valid.sum()) < 2:
            continue
        o = true_2d[valid, h]
        s = pred_2d[valid, h]
        denom = float(np.sum((o - np.mean(o)) ** 2))
        if denom <= 1e-12:
            continue
        num = float(np.sum((s - o) ** 2))
        out[h] = 1.0 - num / denom
    return out


def _compute_per_lead_mean_nse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean of per-lead NSE values for one basin; returns `-inf` if no lead is finite."""
    leads = _compute_per_lead_nse(y_pred, y_true)
    finite = leads[np.isfinite(leads)]
    if finite.size == 0:
        return -float("inf")
    return float(np.mean(finite))


def _lr_warmup_exp(epoch: int, total_epochs: int, base_lr: float, decay: float = 0.99) -> float:
    """Linear warmup over first `0.25 * total_epochs` epochs followed by `decay^(ep - warmup)`."""
    if total_epochs <= 1:
        return float(base_lr)
    warmup_epochs = 0.25 * float(total_epochs)
    if float(epoch) < warmup_epochs:
        return float(base_lr) * float(epoch + 1) / warmup_epochs
    return float(base_lr) * (float(decay) ** (float(epoch) - warmup_epochs))


def _compute_global_stats(
    x_dyn_target: np.ndarray,
    x_stat: np.ndarray,
    q_target: np.ndarray,
    floor: float = 1e-6,
) -> dict[str, np.ndarray | float]:
    """Compute global z-score statistics on valid training records plus per-basin raw Q std."""
    valid_q = np.isfinite(q_target) & (q_target >= 0.0)
    if int(valid_q.sum()) == 0:
        raise ValueError("Training target window contains zero valid discharge observations.")
    x_valid = x_dyn_target[valid_q]
    xd_mean = np.mean(x_valid, axis=0, dtype=np.float64).astype(np.float32).reshape(1, 1, -1)
    xd_std = np.maximum(np.std(x_valid, axis=0, dtype=np.float64), floor).astype(np.float32).reshape(
        1, 1, -1
    )
    xs_mean = np.nanmean(x_stat, axis=0, keepdims=True).astype(np.float32)
    xs_std = np.maximum(np.nanstd(x_stat, axis=0, keepdims=True), floor).astype(np.float32)

    q_valid_vals = q_target[valid_q].astype(np.float64)
    y_mean = float(np.mean(q_valid_vals))
    y_std = max(float(np.std(q_valid_vals)), floor)

    n_b = q_target.shape[0]
    basin_std = np.full(n_b, np.nan, dtype=np.float32)
    for b in range(n_b):
        vb = valid_q[b]
        if int(vb.sum()) >= 2:
            basin_std[b] = np.float32(np.std(q_target[b, vb].astype(np.float64)))
    return {
        "xd_mean": xd_mean,
        "xd_std": xd_std,
        "xs_mean": xs_mean,
        "xs_std": xs_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "basin_std_mm": basin_std,
    }


def _compute_basin_stats(
    x_dyn_b_target: np.ndarray,
    q_b_target: np.ndarray,
    floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute per-basin z-score mean/std over valid training-period days for one basin."""
    valid_q = np.isfinite(q_b_target) & (q_b_target >= 0.0)
    if int(valid_q.sum()) < 2:
        x_mean = np.mean(x_dyn_b_target, axis=0, dtype=np.float64).astype(np.float32).reshape(1, -1)
        x_std = np.maximum(np.std(x_dyn_b_target, axis=0, dtype=np.float64), floor).astype(
            np.float32
        ).reshape(1, -1)
        q_vals = q_b_target[valid_q].astype(np.float64)
        y_mean = float(np.mean(q_vals)) if q_vals.size > 0 else 0.0
        return x_mean, x_std, y_mean, 1.0
    x_valid = x_dyn_b_target[valid_q]
    x_mean = np.mean(x_valid, axis=0, dtype=np.float64).astype(np.float32).reshape(1, -1)
    x_std = np.maximum(np.std(x_valid, axis=0, dtype=np.float64), floor).astype(np.float32).reshape(
        1, -1
    )
    q_vals = q_b_target[valid_q].astype(np.float64)
    y_mean = float(np.mean(q_vals))
    y_std = max(float(np.std(q_vals)), floor)
    return x_mean, x_std, y_mean, y_std


def build_window_starts(
    valid_q: np.ndarray,
    forcing_finite: np.ndarray,
    past_len: int,
    pred_len: int,
    t_first: int,
    require_full_past: bool = True,
    require_full_future: bool = True,
) -> np.ndarray:
    """Return `(basin, t_0)` window start pairs with finite forcings and valid runoff history/targets."""
    n_b, n_t = valid_q.shape
    min_t0 = max(int(past_len), int(t_first))
    max_t0 = n_t - int(pred_len)
    if max_t0 < min_t0:
        return np.zeros((0, 2), dtype=np.int64)

    inv_f = (~forcing_finite).astype(np.int64)
    cum_f = np.concatenate([np.zeros((n_b, 1), dtype=np.int64), np.cumsum(inv_f, axis=1)], axis=1)
    cum_q = np.concatenate(
        [np.zeros((n_b, 1), dtype=np.int64), np.cumsum(valid_q.astype(np.int64), axis=1)],
        axis=1,
    )

    t0_arr = np.arange(min_t0, max_t0 + 1, dtype=np.int64)
    f_bad = cum_f[:, t0_arr + int(pred_len)] - cum_f[:, t0_arr - int(past_len)]
    past_cnt = cum_q[:, t0_arr] - cum_q[:, t0_arr - int(past_len)]
    fut_cnt = cum_q[:, t0_arr + int(pred_len)] - cum_q[:, t0_arr]

    past_ok = (past_cnt == int(past_len)) if require_full_past else np.ones_like(past_cnt, dtype=bool)
    fut_ok = (fut_cnt == int(pred_len)) if require_full_future else (fut_cnt > 0)
    ok = (f_bad == 0) & past_ok & fut_ok
    rows = np.argwhere(ok).astype(np.int64)
    if rows.shape[0] > 0:
        rows[:, 1] += min_t0
    return rows


class RRFormerEncoderLayer(nn.Module):
    """Post-LayerNorm Transformer encoder layer matching Yin et al. (2022)."""

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
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        attn = (
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            .transpose(1, 2)
            .reshape(bsz, seq_len, d_model)
        )
        x = self.norm1(x + self.fc(attn))
        ff = self.dropout(self.ffn2(F.relu(self.ffn1(x))))
        return self.norm2(x + ff)


class RRFormerDecoderLayer(nn.Module):
    """Post-LayerNorm NAR decoder layer with non-causal historical-only self-attention."""

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
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        past_len: int,
    ) -> torch.Tensor:
        bsz, tgt_len, d_model = tgt.shape
        src_len = memory.size(1)

        # Every query attends over all `past_len` historical runoff tokens and never to future
        # query positions; slicing keys/values to `[:, :past_len]` is mathematically identical
        # to adding `-inf` to columns `past_len .. tgt_len - 1` before softmax.
        q = self.self_q(tgt).view(bsz, tgt_len, self.nhead, self.d_head).transpose(1, 2)
        k, v = (
            self.self_kv(tgt[:, :past_len, :])
            .view(bsz, past_len, 2, self.nhead, self.d_head)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        attn1 = (
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, d_model)
        )
        x = self.norm1(tgt + self.self_fc(attn1))

        cq = self.cross_q(x).view(bsz, tgt_len, self.nhead, self.d_head).transpose(1, 2)
        ck, cv = (
            self.cross_kv(memory)
            .view(bsz, src_len, 2, self.nhead, self.d_head)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        attn2 = (
            F.scaled_dot_product_attention(cq, ck, cv, dropout_p=0.0)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, d_model)
        )
        x = self.norm2(x + self.cross_fc(attn2))
        ff = self.dropout(self.ffn2(F.relu(self.ffn1(x))))
        return self.norm3(x + ff)


class RRFormer(nn.Module):
    """4-encoder / 4-decoder non-autoregressive RR-Former (Yin et al., 2022)."""

    def __init__(
        self,
        met_in_dim: int = 5,
        stat_in_dim: int = 0,
        past_len: int = 15,
        pred_len: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_static: bool = False,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.met_in_dim = int(met_in_dim)
        self.stat_in_dim = int(stat_in_dim) if use_static else 0
        self.use_static = bool(use_static)
        self.past_len = int(past_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.src_len = self.past_len + self.pred_len
        self.tgt_len = self.past_len + self.pred_len

        enc_in_dim = self.met_in_dim + self.stat_in_dim
        self.enc_proj = nn.Linear(enc_in_dim, self.d_model)
        self.dec_proj = nn.Linear(1, self.d_model)
        self.src_pos_emb = nn.Embedding(self.src_len, self.d_model)
        self.tgt_pos_emb = nn.Embedding(self.tgt_len, self.d_model)

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

        self.register_buffer(
            "src_pos_idx",
            torch.arange(self.src_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "tgt_pos_idx",
            torch.arange(self.tgt_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x_met: torch.Tensor,
        x_q_hist: torch.Tensor,
        x_stat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict `(batch, pred_len)` standardized runoff from `(batch, src_len, met)` and `(batch, past_len, 1)`."""
        bsz, src_len, _ = x_met.shape
        if self.use_static and x_stat is not None and self.stat_in_dim > 0:
            xs = x_stat.unsqueeze(1).expand(-1, src_len, -1)
            enc_in = torch.cat([x_met, xs], dim=-1)
        else:
            enc_in = x_met

        memory = self.enc_proj(enc_in) + self.src_pos_emb(self.src_pos_idx)
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory)

        q_in = x_q_hist.unsqueeze(-1) if x_q_hist.ndim == 2 else x_q_hist
        dec_tokens = torch.zeros((bsz, self.tgt_len, 1), dtype=x_met.dtype, device=x_met.device)
        dec_tokens[:, : self.past_len, :] = q_in[:, : self.past_len, :1]
        dec_out = self.dec_proj(dec_tokens) + self.tgt_pos_emb(self.tgt_pos_idx)
        for dec_layer in self.decoder_layers:
            dec_out = dec_layer(dec_out, memory, self.past_len)
        return self.head(dec_out[:, -self.pred_len :, :]).squeeze(-1)


class RRFormerTrainer:
    """Global NSELoss pretraining + per-basin-normalized full-network Adam MSE fine-tuning."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        past_len: int = 15,
        pred_len: int = 1,
        use_static: bool = False,
        use_future_fea: bool = True,
        decode_mode: str = "NAR",
        dropout: float = 0.1,
        batch_size: int = 512,
        epochs: int = 200,
        finetune_epochs: int = 200,
        finetune_patience: int = 0,
        pretrain_patience: int = 0,
        lr: float = 1e-3,
        finetune_lr: float = 1e-3,
        n_batches_per_epoch: int | None = None,
        clip_grad_norm: float = 0.0,
        eval_batch_size: int = 4096,
        lag_dropout_prob: float = 0.0,
        nse_eps: float = 0.1,
        use_amp: bool = False,
        use_compile: bool = False,
        clip_nonneg: bool = True,
        checkpoint_dir: str | Path | None = None,
        load_pretrain_checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if past_len < 1 or pred_len < 1:
            raise ValueError(f"past_len ({past_len}) and pred_len ({pred_len}) must be >= 1")
        if decode_mode != "NAR":
            raise ValueError(f"Only decode_mode='NAR' is supported, got {decode_mode!r}")
        if not use_future_fea:
            raise ValueError("use_future_fea=True is required for RRFormer.")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.past_len = int(past_len)
        self.pred_len = int(pred_len)
        self.src_len = self.past_len + self.pred_len
        self.use_static = bool(use_static)
        self.use_future_fea = bool(use_future_fea)
        self.decode_mode = decode_mode
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_patience = int(finetune_patience)
        self.pretrain_patience = int(pretrain_patience)
        self.lr = float(lr)
        self.finetune_lr = float(finetune_lr)
        self.n_batches_per_epoch = (
            None if n_batches_per_epoch is None else int(n_batches_per_epoch)
        )
        self.clip_grad_norm = float(clip_grad_norm)
        self.eval_batch_size = int(eval_batch_size)
        self.lag_dropout_prob = float(lag_dropout_prob)
        self.nse_eps = float(nse_eps)
        self.use_amp = bool(use_amp)
        self.use_compile = bool(use_compile)
        self.clip_nonneg = bool(clip_nonneg)
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.load_pretrain_checkpoint = (
            None if load_pretrain_checkpoint is None else Path(load_pretrain_checkpoint)
        )
        self.device = resolve_device(device)
        self.last_pretrain_metrics: dict[int, dict[str, float]] = {}

    def _build_model(self, met_in_dim: int, stat_in_dim: int, dev: torch.device) -> RRFormer:
        return RRFormer(
            met_in_dim=met_in_dim,
            stat_in_dim=stat_in_dim,
            past_len=self.past_len,
            pred_len=self.pred_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            use_static=self.use_static,
        ).to(dev)

    def _gather_flat(
        self,
        xd_flat: torch.Tensor,
        xq_flat: torch.Tensor,
        y_flat: torch.Tensor,
        n_days: int,
        n_feat: int,
        b_idx: torch.Tensor,
        t0_idx: torch.Tensor,
        src_offs: torch.Tensor,
        past_offs: torch.Tensor,
        pred_offs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = b_idx.size(0)
        base = b_idx.unsqueeze(1) * n_days + t0_idx.unsqueeze(1)
        x_met = xd_flat[(base + src_offs.unsqueeze(0)).reshape(-1)].view(bsz, self.src_len, n_feat)
        x_q = xq_flat[(base + past_offs.unsqueeze(0)).reshape(-1)].view(bsz, self.past_len, 1)
        y_tgt = y_flat[(base + pred_offs.unsqueeze(0)).reshape(-1)].view(bsz, self.pred_len)
        return x_met, x_q, y_tgt

    @torch.no_grad()
    def _eval_windows_nse(
        self,
        model: RRFormer,
        xd_flat: torch.Tensor,
        xq_flat: torch.Tensor,
        n_days: int,
        n_feat: int,
        xs_t: torch.Tensor | None,
        y_mm_raw: np.ndarray,
        pairs_np: np.ndarray,
        y_mean: float,
        y_std: float,
        dev: torch.device,
        use_amp: bool,
    ) -> tuple[float, float, list[float]]:
        """Return `(median_basin_nse, mean_basin_nse, per_lead_medians)` over validation windows."""
        if pairs_np.shape[0] == 0:
            return -float("inf"), -float("inf"), [float("nan")] * self.pred_len
        model.eval()
        src_offs = torch.arange(-self.past_len, self.pred_len, device=dev, dtype=torch.long)
        past_offs = torch.arange(-self.past_len, 0, device=dev, dtype=torch.long)
        pred_offs = torch.arange(0, self.pred_len, device=dev, dtype=torch.long)
        y_dummy = xq_flat.squeeze(-1)

        preds = np.empty((pairs_np.shape[0], self.pred_len), dtype=np.float32)
        pairs_t = torch.from_numpy(pairs_np).to(device=dev, dtype=torch.long)
        for s0 in range(0, pairs_np.shape[0], self.eval_batch_size):
            s1 = min(s0 + self.eval_batch_size, pairs_np.shape[0])
            rows = pairs_t[s0:s1]
            b_idx = rows[:, 0]
            t0_idx = rows[:, 1]
            x_met, x_q, _ = self._gather_flat(
                xd_flat, xq_flat, y_dummy, n_days, n_feat, b_idx, t0_idx, src_offs, past_offs, pred_offs
            )
            xs_b = xs_t[b_idx] if (self.use_static and xs_t is not None) else None
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x_met, x_q, xs_b).float()
            out_mm = out.cpu().numpy() * y_std + y_mean
            if self.clip_nonneg:
                out_mm = np.clip(out_mm, 0.0, None)
            preds[s0:s1] = out_mm

        t0_all = pairs_np[:, 1]
        b_all = pairs_np[:, 0]
        obs_mat = np.stack([y_mm_raw[b_all, t0_all + h] for h in range(self.pred_len)], axis=1)

        # Basins appear in contiguous blocks from `np.argwhere`; compute boundaries in O(N).
        uniq_b, start_idx = np.unique(b_all, return_index=True)
        order = np.argsort(start_idx)
        uniq_b = uniq_b[order]
        start_idx = start_idx[order]
        end_idx = np.append(start_idx[1:], b_all.shape[0])

        per_lead_mat = np.full((uniq_b.shape[0], self.pred_len), np.nan, dtype=np.float64)
        basin_means: list[float] = []
        for k, (lo, hi) in enumerate(zip(start_idx, end_idx)):
            if np.all(b_all[lo:hi] == uniq_b[k]):
                p_b = preds[lo:hi]
                o_b = obs_mat[lo:hi]
            else:
                mask_b = b_all == uniq_b[k]
                p_b = preds[mask_b]
                o_b = obs_mat[mask_b]
            leads = _compute_per_lead_nse(p_b, o_b)
            per_lead_mat[k] = leads
            fin = leads[np.isfinite(leads)]
            if fin.size > 0:
                basin_means.append(float(np.mean(fin)))

        if not basin_means:
            return -float("inf"), -float("inf"), [float("nan")] * self.pred_len
        per_lead_med = [
            float(np.nanmedian(per_lead_mat[:, h]))
            if np.any(np.isfinite(per_lead_mat[:, h]))
            else float("nan")
            for h in range(self.pred_len)
        ]
        return float(np.median(basin_means)), float(np.mean(basin_means)), per_lead_med

    def _finetune_single_basin(
        self,
        *,
        basin_i: int,
        global_state: dict[str, torch.Tensor],
        met_in_dim: int,
        stat_in_dim: int,
        xd_tr_b_raw: np.ndarray,
        q_tr_b_raw: np.ndarray,
        spinup_days_train: int,
        xd_va_b_raw: np.ndarray | None,
        q_va_b_raw: np.ndarray | None,
        spinup_days_val: int,
        xs_b_norm: np.ndarray | None,
        gpu_id: int | None,
        seed: int,
        reuse_model: RRFormer | None = None,
        compiled_model: nn.Module | None = None,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, float, float, float] | None:
        """Fine-tune all weights on one basin with per-basin z-score normalization and Adam MSE."""
        x_mean_b, x_std_b, y_mean_b, y_std_b = _compute_basin_stats(
            xd_tr_b_raw[spinup_days_train:], q_tr_b_raw[spinup_days_train:]
        )

        xd_tr_b_norm = np.nan_to_num(
            ((xd_tr_b_raw - x_mean_b) / x_std_b).astype(np.float32), nan=0.0
        )
        valid_q_tr = np.isfinite(q_tr_b_raw) & (q_tr_b_raw >= 0.0)
        q_tr_b_norm = np.where(
            valid_q_tr,
            ((np.where(valid_q_tr, q_tr_b_raw, y_mean_b) - y_mean_b) / y_std_b),
            0.0,
        ).astype(np.float32)

        tr_pairs_b = build_window_starts(
            valid_q=valid_q_tr[None, :],
            forcing_finite=np.all(np.isfinite(xd_tr_b_norm), axis=1)[None, :],
            past_len=self.past_len,
            pred_len=self.pred_len,
            t_first=spinup_days_train,
            require_full_past=True,
            require_full_future=True,
        )
        if tr_pairs_b.shape[0] < 10:
            return None

        dev = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else self.device
        use_amp = self.use_amp and dev.type == "cuda"

        model = (
            reuse_model
            if reuse_model is not None
            else self._build_model(met_in_dim, stat_in_dim, dev)
        )
        model.load_state_dict({k: v.to(dev) for k, v in global_state.items()})

        src_offs = np.arange(-self.past_len, self.pred_len, dtype=np.int64)
        past_offs = np.arange(-self.past_len, 0, dtype=np.int64)
        pred_offs = np.arange(0, self.pred_len, dtype=np.int64)

        tr_t0 = tr_pairs_b[:, 1]
        xm_tr_b = torch.from_numpy(xd_tr_b_norm[tr_t0[:, None] + src_offs[None, :]]).to(dev)
        xq_tr_b = torch.from_numpy(
            q_tr_b_norm[tr_t0[:, None] + past_offs[None, :]][:, :, None]
        ).to(dev)
        yt_tr_b = torch.from_numpy(q_tr_b_norm[tr_t0[:, None] + pred_offs[None, :]]).to(dev)
        xs_b_t = (
            torch.from_numpy(xs_b_norm[None, :]).to(dev)
            if (self.use_static and xs_b_norm is not None)
            else None
        )

        has_val = False
        xm_va_b: torch.Tensor | None = None
        xq_va_b: torch.Tensor | None = None
        va_obs_mm: np.ndarray | None = None
        if xd_va_b_raw is not None and q_va_b_raw is not None:
            xd_va_b_norm = np.nan_to_num(
                ((xd_va_b_raw - x_mean_b) / x_std_b).astype(np.float32), nan=0.0
            )
            valid_q_va = np.isfinite(q_va_b_raw) & (q_va_b_raw >= 0.0)
            q_va_b_norm = np.where(
                valid_q_va,
                ((np.where(valid_q_va, q_va_b_raw, y_mean_b) - y_mean_b) / y_std_b),
                0.0,
            ).astype(np.float32)
            va_pairs_b = build_window_starts(
                valid_q=valid_q_va[None, :],
                forcing_finite=np.all(np.isfinite(xd_va_b_norm), axis=1)[None, :],
                past_len=self.past_len,
                pred_len=self.pred_len,
                t_first=spinup_days_val,
                require_full_past=True,
                require_full_future=True,
            )
            if va_pairs_b.shape[0] >= 10:
                has_val = True
                va_t0 = va_pairs_b[:, 1]
                xm_va_b = torch.from_numpy(xd_va_b_norm[va_t0[:, None] + src_offs[None, :]]).to(dev)
                xq_va_b = torch.from_numpy(
                    q_va_b_norm[va_t0[:, None] + past_offs[None, :]][:, :, None]
                ).to(dev)
                va_obs_mm = q_va_b_raw[va_t0[:, None] + pred_offs[None, :]]

        ft_epochs = max(1, self.finetune_epochs)
        n_tr = xm_tr_b.size(0)
        ft_bs = min(self.batch_size, max(1, n_tr))
        full_batches = (n_tr + ft_bs - 1) // ft_bs
        n_ft_batches = (
            min(self.n_batches_per_epoch, full_batches)
            if (self.n_batches_per_epoch is not None and self.n_batches_per_epoch > 0)
            else full_batches
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.finetune_lr, fused=(dev.type == "cuda")
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + 10000 + basin_i)

        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_val_nse = -float("inf")
        warmup_ep = int(round(0.25 * ft_epochs))
        no_improve = 0

        for ep in range(ft_epochs):
            cur_lr = _lr_warmup_exp(ep, ft_epochs, self.finetune_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            model.train()
            perm = torch.randperm(n_tr, generator=gen, device=dev)
            for b in range(n_ft_batches):
                idx = perm[b * ft_bs : (b + 1) * ft_bs]
                if idx.numel() == 0:
                    continue
                xs_sl = xs_b_t.expand(idx.numel(), -1) if xs_b_t is not None else None
                fwd_fn = (
                    compiled_model
                    if (compiled_model is not None and idx.numel() == self.batch_size)
                    else model
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = fwd_fn(xm_tr_b[idx], xq_tr_b[idx], xs_sl).float()
                loss = F.mse_loss(out, yt_tr_b[idx])
                scaler.scale(loss).backward()
                if self.clip_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if has_val and xm_va_b is not None and xq_va_b is not None and va_obs_mm is not None:
                model.eval()
                with torch.no_grad():
                    v_chunks: list[np.ndarray] = []
                    for s0 in range(0, xm_va_b.size(0), self.eval_batch_size):
                        s1 = min(s0 + self.eval_batch_size, xm_va_b.size(0))
                        xs_sl = xs_b_t.expand(s1 - s0, -1) if xs_b_t is not None else None
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            vo = model(xm_va_b[s0:s1], xq_va_b[s0:s1], xs_sl).float()
                        v_mm = vo.cpu().numpy() * y_std_b + y_mean_b
                        if self.clip_nonneg:
                            v_mm = np.clip(v_mm, 0.0, None)
                        v_chunks.append(v_mm)
                    v_pred = np.concatenate(v_chunks, axis=0)
                    v_nse = _compute_per_lead_mean_nse(v_pred, va_obs_mm)
                if v_nse > best_val_nse:
                    best_val_nse = v_nse
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.finetune_patience > 0 and ep >= warmup_ep and no_improve >= self.finetune_patience:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        return best_state, x_mean_b, x_std_b, y_mean_b, y_std_b, best_val_nse

    def _build_q_hist_channel(
        self,
        q_masked: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> np.ndarray:
        """Build `(n_basins, n_days, 1)` standardized historical runoff with NaN/negative mapped to 0.0."""
        valid = np.isfinite(q_masked) & (q_masked >= 0.0)
        safe_q = np.where(valid, q_masked, y_mean)
        q_std = np.where(valid, (safe_q - y_mean) / max(float(y_std), 1e-6), 0.0).astype(
            np.float32
        )
        return q_std[:, :, None]

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
        x_dyn_val: np.ndarray | None = None,
        x_stat_val: np.ndarray | None = None,
        qobs_val: np.ndarray | None = None,
        spinup_days_val: int = 0,
        train_basins: Sequence[str] | None = None,
        test_basins: Sequence[str] | None = None,
        seeds: Sequence[int] = (601, 602, 603),
        return_lead_predictions: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global NSELoss pretraining + per-basin-normalized Adam MSE fine-tuning + inference."""
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds must be non-empty.")

        xd_tr_raw = np.asarray(x_dyn_train, dtype=np.float32)
        xs_tr_raw = np.asarray(x_stat_train, dtype=np.float32)
        xd_te_raw = np.asarray(x_dyn_test, dtype=np.float32)
        xs_te_raw = np.asarray(x_stat_test, dtype=np.float32)

        q_tr_all = mask_spinup_discharge(qobs_train, spinup_days_train)
        q_te_all = mask_spinup_discharge(qobs_test, spinup_days_test)

        # If no external validation slice is passed (Stage 2 MIP protocol), carve chronological
        # validation strictly from the tail of `train_period` so zero test data is ever touched.
        if x_dyn_val is None or qobs_val is None:
            n_tr_tot = xd_tr_raw.shape[1]
            n_tr_target = n_tr_tot - int(spinup_days_train)
            val_days = 731 if n_tr_target >= 2500 else min(730, max(60, n_tr_target // 5))
            if n_tr_target - val_days < self.src_len + self.past_len + 30:
                val_days = 0
            if val_days > 0:
                split_t = n_tr_tot - val_days
                val_start_with_spin = max(int(spinup_days_train), split_t - self.past_len)
                xd_va_raw = xd_tr_raw[:, val_start_with_spin:].copy()
                xs_va_raw = xs_tr_raw.copy()
                # Note: `val_start_with_spin >= spinup_days_train`, so all days in `q_va_all`
                # lie strictly inside `train_period` (zero eval-period leakage).
                q_va_all = q_tr_all[:, val_start_with_spin:].copy()
                spin_va = split_t - val_start_with_spin

                xd_tr_fit_raw = xd_tr_raw[:, :split_t].copy()
                q_tr_fit_all = q_tr_all[:, :split_t].copy()
            else:
                xd_va_raw = None
                xs_va_raw = None
                q_va_all = None
                spin_va = 0
                xd_tr_fit_raw = xd_tr_raw
                q_tr_fit_all = q_tr_all
        else:
            xd_va_raw = np.asarray(x_dyn_val, dtype=np.float32)
            xs_va_raw = np.asarray(
                x_stat_val if x_stat_val is not None else xs_tr_raw, dtype=np.float32
            )
            q_va_all = mask_spinup_discharge(qobs_val, spinup_days_val)
            spin_va = int(spinup_days_val)
            xd_tr_fit_raw = xd_tr_raw
            q_tr_fit_all = q_tr_all

        gstats = _compute_global_stats(
            xd_tr_fit_raw[:, spinup_days_train:, :],
            xs_tr_raw,
            q_tr_fit_all[:, spinup_days_train:],
        )
        xd_mean_g = np.asarray(gstats["xd_mean"], dtype=np.float32)
        xd_std_g = np.asarray(gstats["xd_std"], dtype=np.float32)
        xs_mean_g = np.asarray(gstats["xs_mean"], dtype=np.float32)
        xs_std_g = np.asarray(gstats["xs_std"], dtype=np.float32)
        y_mean = float(gstats["y_mean"])
        y_std = float(gstats["y_std"])
        weights_np = nse_star_weights(
            np.asarray(gstats["basin_std_mm"], dtype=np.float32), eps=self.nse_eps
        )

        xd_tr_norm = np.nan_to_num(
            ((xd_tr_fit_raw - xd_mean_g) / xd_std_g).astype(np.float32), nan=0.0
        )
        xd_te_norm = np.nan_to_num(
            ((xd_te_raw - xd_mean_g) / xd_std_g).astype(np.float32), nan=0.0
        )
        xs_tr_norm = (
            ((xs_tr_raw - xs_mean_g) / xs_std_g).astype(np.float32) if self.use_static else None
        )
        xs_te_norm = (
            ((xs_te_raw - xs_mean_g) / xs_std_g).astype(np.float32) if self.use_static else None
        )

        xq_tr_norm = self._build_q_hist_channel(q_tr_fit_all, y_mean, y_std)
        xq_te_norm = self._build_q_hist_channel(q_te_all, y_mean, y_std)
        if self.lag_dropout_prob > 0.0:
            drop_rng = np.random.RandomState(seed_list[0] + 6151)
            dmask = drop_rng.uniform(0.0, 1.0, size=xq_tr_norm.shape[:2]) < self.lag_dropout_prob
            xq_tr_norm = xq_tr_norm.copy()
            xq_tr_norm[dmask] = 0.0

        y_tr_norm = np.nan_to_num(((q_tr_fit_all - y_mean) / y_std).astype(np.float32), nan=0.0)
        valid_q_tr = np.isfinite(q_tr_fit_all) & (q_tr_fit_all >= 0.0)
        f_tr_ok = np.all(np.isfinite(xd_tr_norm), axis=2)

        tr_pairs = build_window_starts(
            valid_q=valid_q_tr,
            forcing_finite=f_tr_ok,
            past_len=self.past_len,
            pred_len=self.pred_len,
            t_first=spinup_days_train,
            require_full_past=True,
            require_full_future=True,
        )
        fin_w = np.isfinite(weights_np)
        tr_pairs = tr_pairs[fin_w[tr_pairs[:, 0]]]
        if tr_pairs.shape[0] == 0:
            raise ValueError("No valid training windows in RRFormerTrainer.")

        if xd_va_raw is not None and q_va_all is not None:
            xd_va_norm = np.nan_to_num(
                ((xd_va_raw - xd_mean_g) / xd_std_g).astype(np.float32), nan=0.0
            )
            xs_va_norm = (
                ((xs_va_raw - xs_mean_g) / xs_std_g).astype(np.float32)
                if (self.use_static and xs_va_raw is not None)
                else None
            )
            xq_va_norm = self._build_q_hist_channel(q_va_all, y_mean, y_std)
            valid_q_va = np.isfinite(q_va_all) & (q_va_all >= 0.0)
            va_pairs = build_window_starts(
                valid_q=valid_q_va,
                forcing_finite=np.all(np.isfinite(xd_va_norm), axis=2),
                past_len=self.past_len,
                pred_len=self.pred_len,
                t_first=spin_va,
                require_full_past=True,
                require_full_future=True,
            )
        else:
            xd_va_norm = xd_tr_norm
            xs_va_norm = xs_tr_norm
            xq_va_norm = xq_tr_norm
            q_va_all = q_tr_fit_all
            va_pairs = np.zeros((0, 2), dtype=np.int64)

        full_batches = (tr_pairs.shape[0] + self.batch_size - 1) // self.batch_size
        n_batches = (
            min(int(self.n_batches_per_epoch), full_batches)
            if (self.n_batches_per_epoch is not None and self.n_batches_per_epoch > 0)
            else full_batches
        )

        n_te = xd_te_norm.shape[0]
        n_te_total_days = xd_te_norm.shape[1]
        n_target_days = n_te_total_days - int(spinup_days_test)
        n_eval_windows = n_target_days - self.pred_len + 1
        if n_eval_windows <= 0:
            raise ValueError(
                f"x_dyn_test target length {n_target_days} < pred_len {self.pred_len}."
            )
        if int(spinup_days_test) < self.past_len:
            raise ValueError(
                f"spinup_days_test={spinup_days_test} < past_len={self.past_len}."
            )

        dev = self.device
        n_gpus = torch.cuda.device_count() if dev.type == "cuda" else 0
        use_amp = self.use_amp and dev.type == "cuda"

        n_tr_basins, n_tr_days, met_in_dim = xd_tr_norm.shape
        n_va_days = xd_va_norm.shape[1]
        stat_in_dim = xs_tr_norm.shape[1] if xs_tr_norm is not None else 0

        xd_tr_flat = torch.from_numpy(xd_tr_norm).to(dev).reshape(-1, met_in_dim)
        xq_tr_flat = torch.from_numpy(xq_tr_norm).to(dev).reshape(-1, 1)
        y_tr_flat = torch.from_numpy(y_tr_norm).to(dev).reshape(-1)
        xs_tr_t = torch.from_numpy(xs_tr_norm).to(dev) if xs_tr_norm is not None else None
        w_tr_t = torch.from_numpy(weights_np).to(dev)

        xd_va_flat = torch.from_numpy(xd_va_norm).to(dev).reshape(-1, met_in_dim)
        xq_va_flat = torch.from_numpy(xq_va_norm).to(dev).reshape(-1, 1)
        xs_va_t = torch.from_numpy(xs_va_norm).to(dev) if xs_va_norm is not None else None

        xd_te_t = torch.from_numpy(xd_te_norm).to(dev)
        xq_te_t = torch.from_numpy(xq_te_norm).to(dev)
        xs_te_t = torch.from_numpy(xs_te_norm).to(dev) if xs_te_norm is not None else None

        tr_pairs_t = torch.from_numpy(tr_pairs).to(device=dev, dtype=torch.long)
        src_offs_t = torch.arange(-self.past_len, self.pred_len, device=dev, dtype=torch.long)
        past_offs_t = torch.arange(-self.past_len, 0, device=dev, dtype=torch.long)
        pred_offs_t = torch.arange(0, self.pred_len, device=dev, dtype=torch.long)

        tr_index_map: dict[str, int] = {}
        if train_basins is not None:
            for idx, b_id in enumerate(train_basins):
                tr_index_map[str(b_id)] = idx

        self.last_pretrain_metrics = {}
        member_preds: list[np.ndarray] = []
        for seed in seed_list:
            torch.manual_seed(seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            base_model = self._build_model(met_in_dim, stat_in_dim, dev)
            best_val_nse = -float("inf")
            best_val_mean = -float("inf")
            best_epoch = -1

            if self.load_pretrain_checkpoint is not None:
                ckpt = torch.load(
                    self.load_pretrain_checkpoint, map_location="cpu", weights_only=True
                )
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    best_global_state = ckpt["state_dict"]
                    best_val_nse = float(ckpt.get("best_val_nse_median", -float("inf")))
                    best_val_mean = float(ckpt.get("best_val_nse_mean", -float("inf")))
                    best_epoch = int(ckpt.get("best_epoch", -1))
                else:
                    best_global_state = ckpt
                print(
                    f"[RR-Former pretrain seed={seed}] loaded checkpoint {self.load_pretrain_checkpoint} "
                    f"(best_ep={best_epoch} val_med={best_val_nse:.4f} val_mean={best_val_mean:.4f})",
                    flush=True,
                )
            else:
                compiled_model: nn.Module = (
                    torch.compile(base_model, mode="reduce-overhead")
                    if (self.use_compile and dev.type == "cuda")
                    else base_model
                )
                optimizer = torch.optim.Adam(
                    base_model.parameters(), lr=self.lr, fused=(dev.type == "cuda")
                )
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
                gen = torch.Generator(device=dev)
                gen.manual_seed(seed + 4919)

                best_global_state = {
                    k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
                }
                warmup_ep = int(round(0.25 * self.epochs))
                no_improve = 0
                t_pre0 = time.time()

                for ep in range(self.epochs):
                    ep_t0 = time.time()
                    base_model.train()
                    cur_lr = _lr_warmup_exp(ep, self.epochs, self.lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = cur_lr

                    perm = torch.randperm(tr_pairs_t.size(0), generator=gen, device=dev)
                    loss_accum = torch.zeros((), dtype=torch.float64, device=dev)
                    loss_cnt = 0
                    for b in range(n_batches):
                        s0 = b * self.batch_size
                        idx = perm[s0 : s0 + self.batch_size]
                        if idx.numel() == 0:
                            continue
                        rows = tr_pairs_t[idx]
                        b_idx = rows[:, 0]
                        t0_idx = rows[:, 1]
                        x_met, x_q, y_tgt = self._gather_flat(
                            xd_tr_flat,
                            xq_tr_flat,
                            y_tr_flat,
                            n_tr_days,
                            met_in_dim,
                            b_idx,
                            t0_idx,
                            src_offs_t,
                            past_offs_t,
                            pred_offs_t,
                        )
                        xs_b = xs_tr_t[b_idx] if (self.use_static and xs_tr_t is not None) else None
                        w_b = w_tr_t[b_idx].unsqueeze(1)

                        fwd_fn = (
                            compiled_model
                            if (self.use_compile and idx.numel() == self.batch_size)
                            else base_model
                        )
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            y_hat = fwd_fn(x_met, x_q, xs_b).float()
                        loss = torch.mean(w_b * ((y_hat - y_tgt) ** 2))
                        scaler.scale(loss).backward()
                        if self.clip_grad_norm > 0.0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                base_model.parameters(), max_norm=self.clip_grad_norm
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        loss_accum += loss.detach().to(torch.float64)
                        loss_cnt += 1

                    ep_sec = time.time() - ep_t0
                    train_loss = float(loss_accum.item()) / max(1, loss_cnt)

                    if va_pairs.shape[0] > 0:
                        val_med, val_mean, val_leads = self._eval_windows_nse(
                            base_model,
                            xd_va_flat,
                            xq_va_flat,
                            n_va_days,
                            met_in_dim,
                            xs_va_t,
                            q_va_all,
                            va_pairs,
                            y_mean,
                            y_std,
                            dev,
                            use_amp,
                        )
                        if val_med > best_val_nse:
                            best_val_nse = val_med
                            best_val_mean = val_mean
                            best_epoch = ep + 1
                            best_global_state = {
                                k: v.detach().cpu().clone()
                                for k, v in base_model.state_dict().items()
                            }
                            no_improve = 0
                            if self.checkpoint_dir is not None:
                                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                                torch.save(
                                    {
                                        "state_dict": best_global_state,
                                        "seed": seed,
                                        "best_epoch": best_epoch,
                                        "best_val_nse_median": best_val_nse,
                                        "best_val_nse_mean": best_val_mean,
                                        "best_val_per_lead_median": val_leads,
                                        "xd_mean": torch.from_numpy(xd_mean_g.copy()),
                                        "xd_std": torch.from_numpy(xd_std_g.copy()),
                                        "y_mean": y_mean,
                                        "y_std": y_std,
                                    },
                                    self.checkpoint_dir / f"pretrain_seed{seed}_best.pt",
                                )
                        else:
                            no_improve += 1
                            if (
                                self.pretrain_patience > 0
                                and ep >= warmup_ep
                                and no_improve >= self.pretrain_patience
                            ):
                                print(
                                    f"[RR-Former pretrain seed={seed}] early stop ep={ep + 1} "
                                    f"best_val_med={best_val_nse:.4f} best_val_mean={best_val_mean:.4f}",
                                    flush=True,
                                )
                                break
                        if (
                            self.epochs >= 5
                            and (ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == self.epochs)
                        ):
                            lead_str = ",".join(f"{v:.4f}" for v in val_leads)
                            print(
                                f"[RR-Former pretrain seed={seed}] ep={ep + 1}/{self.epochs} "
                                f"lr={cur_lr:.2e} loss={train_loss:.5f} val_med={val_med:.4f} "
                                f"val_mean={val_mean:.4f} best={best_val_nse:.4f}@{best_epoch} "
                                f"leads=[{lead_str}] ({ep_sec:.1f}s/ep, tot={(time.time()-t_pre0)/60:.1f}m)",
                                flush=True,
                            )
                    else:
                        best_global_state = {
                            k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
                        }

            self.last_pretrain_metrics[seed] = {
                "best_epoch": float(best_epoch),
                "best_val_nse_median": float(best_val_nse),
                "best_val_nse_mean": float(best_val_mean),
            }

            needed_tr_indices: set[int] = set()
            te_to_tr: list[int | None] = []
            if test_basins is not None and tr_index_map:
                for b_id in test_basins:
                    tr_i = tr_index_map.get(str(b_id))
                    te_to_tr.append(tr_i)
                    if tr_i is not None:
                        needed_tr_indices.add(tr_i)
            elif n_te == n_tr_basins and test_basins is None:
                for i in range(n_te):
                    te_to_tr.append(i)
                    needed_tr_indices.add(i)
            else:
                te_to_tr = [None] * n_te

            ft_results: dict[
                int, tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, float, float, float]
            ] = {}
            sorted_tr = sorted(needed_tr_indices)
            if sorted_tr and self.finetune_epochs > 0:
                t_ft0 = time.time()
                if n_gpus > 1 and len(sorted_tr) > 1:
                    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                        futs = [
                            (
                                bi,
                                pool.submit(
                                    self._finetune_single_basin,
                                    basin_i=bi,
                                    global_state=best_global_state,
                                    met_in_dim=met_in_dim,
                                    stat_in_dim=stat_in_dim,
                                    xd_tr_b_raw=xd_tr_fit_raw[bi],
                                    q_tr_b_raw=q_tr_fit_all[bi],
                                    spinup_days_train=int(spinup_days_train),
                                    xd_va_b_raw=xd_va_raw[bi] if xd_va_raw is not None else None,
                                    q_va_b_raw=q_va_all[bi] if xd_va_raw is not None else None,
                                    spinup_days_val=int(spin_va),
                                    xs_b_norm=xs_tr_norm[bi] if xs_tr_norm is not None else None,
                                    gpu_id=idx % n_gpus,
                                    seed=seed,
                                ),
                            )
                            for idx, bi in enumerate(sorted_tr)
                        ]
                        done_cnt = 0
                        for bi, fut in futs:
                            res = fut.result()
                            if res is not None:
                                ft_results[bi] = res
                            done_cnt += 1
                            if len(sorted_tr) >= 50 and (
                                done_cnt % 50 == 0 or done_cnt == len(sorted_tr)
                            ):
                                ft_vals = [r[5] for r in ft_results.values() if np.isfinite(r[5])]
                                med_ft = float(np.median(ft_vals)) if ft_vals else float("nan")
                                print(
                                    f"[RR-Former finetune seed={seed}] {done_cnt}/{len(sorted_tr)} "
                                    f"val_med={med_ft:.4f} elapsed={(time.time()-t_ft0)/60:.1f}m",
                                    flush=True,
                                )
                else:
                    single_gpu = dev.index if (dev.type == "cuda" and dev.index is not None) else (
                        0 if (n_gpus >= 1 and dev.type == "cuda") else None
                    )
                    ft_dev = torch.device(f"cuda:{single_gpu}") if single_gpu is not None else dev
                    ft_reuse = self._build_model(met_in_dim, stat_in_dim, ft_dev)
                    ft_compiled: nn.Module | None = (
                        torch.compile(ft_reuse, mode="reduce-overhead")
                        if (self.use_compile and ft_dev.type == "cuda")
                        else None
                    )
                    for idx, bi in enumerate(sorted_tr):
                        res = self._finetune_single_basin(
                            basin_i=bi,
                            global_state=best_global_state,
                            met_in_dim=met_in_dim,
                            stat_in_dim=stat_in_dim,
                            xd_tr_b_raw=xd_tr_fit_raw[bi],
                            q_tr_b_raw=q_tr_fit_all[bi],
                            spinup_days_train=int(spinup_days_train),
                            xd_va_b_raw=xd_va_raw[bi] if xd_va_raw is not None else None,
                            q_va_b_raw=q_va_all[bi] if xd_va_raw is not None else None,
                            spinup_days_val=int(spin_va),
                            xs_b_norm=xs_tr_norm[bi] if xs_tr_norm is not None else None,
                            gpu_id=single_gpu,
                            seed=seed,
                            reuse_model=ft_reuse,
                            compiled_model=ft_compiled,
                        )
                        if res is not None:
                            ft_results[bi] = res
                        if len(sorted_tr) >= 50 and (
                            (idx + 1) % 50 == 0 or (idx + 1) == len(sorted_tr)
                        ):
                            ft_vals = [r[5] for r in ft_results.values() if np.isfinite(r[5])]
                            med_ft = float(np.median(ft_vals)) if ft_vals else float("nan")
                            print(
                                f"[RR-Former finetune seed={seed}] {idx + 1}/{len(sorted_tr)} "
                                f"val_med={med_ft:.4f} elapsed={(time.time()-t_ft0)/60:.1f}m",
                                flush=True,
                            )

            pred_seed = np.empty((n_te, n_eval_windows, self.pred_len), dtype=np.float32)
            t0_seq = torch.arange(
                int(spinup_days_test),
                int(spinup_days_test) + n_eval_windows,
                device=dev,
                dtype=torch.long,
            )
            src_idx = t0_seq.unsqueeze(1) + src_offs_t.unsqueeze(0)
            past_idx = t0_seq.unsqueeze(1) + past_offs_t.unsqueeze(0)

            base_model.eval()
            with torch.no_grad():
                for i_te in range(n_te):
                    tr_i = te_to_tr[i_te]
                    ft_entry = ft_results.get(tr_i) if tr_i is not None else None
                    if ft_entry is not None:
                        state_use, xm_b_mean, xm_b_std, ym_b, ys_b, _ = ft_entry
                        base_model.load_state_dict({k: v.to(dev) for k, v in state_use.items()})
                        xd_te_b_norm = np.nan_to_num(
                            ((xd_te_raw[i_te] - xm_b_mean) / xm_b_std).astype(np.float32), nan=0.0
                        )
                        valid_q_te = np.isfinite(q_te_all[i_te]) & (q_te_all[i_te] >= 0.0)
                        q_te_b_norm = np.where(
                            valid_q_te,
                            ((np.where(valid_q_te, q_te_all[i_te], ym_b) - ym_b) / ys_b),
                            0.0,
                        ).astype(np.float32)[:, None]
                        xm_b = torch.from_numpy(xd_te_b_norm).to(dev)[src_idx]
                        xq_b = torch.from_numpy(q_te_b_norm).to(dev)[past_idx]
                        scale_mean, scale_std = ym_b, ys_b
                    else:
                        base_model.load_state_dict(
                            {k: v.to(dev) for k, v in copy.deepcopy(best_global_state).items()}
                        )
                        xm_b = xd_te_t[i_te][src_idx]
                        xq_b = xq_te_t[i_te][past_idx]
                        scale_mean, scale_std = y_mean, y_std

                    out_chunks: list[np.ndarray] = []
                    for s0 in range(0, n_eval_windows, self.eval_batch_size):
                        s1 = min(s0 + self.eval_batch_size, n_eval_windows)
                        xs_sl = (
                            xs_te_t[i_te : i_te + 1].expand(s1 - s0, -1)
                            if (self.use_static and xs_te_t is not None)
                            else None
                        )
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            o = base_model(xm_b[s0:s1], xq_b[s0:s1], xs_sl).float()
                        o_mm = o.cpu().numpy() * scale_std + scale_mean
                        if self.clip_nonneg:
                            o_mm = np.clip(o_mm, 0.0, None)
                        out_chunks.append(o_mm)
                    pred_seed[i_te] = np.concatenate(out_chunks, axis=0)

            if return_lead_predictions:
                member_preds.append(pred_seed)
            else:
                member_preds.append(pred_seed[:, :, 0])

        members = np.stack(member_preds, axis=0).astype(np.float32)
        return np.mean(members, axis=0, dtype=np.float32), members
