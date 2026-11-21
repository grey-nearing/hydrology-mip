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

Faithful architecture & two-stage protocol:
  - Encoder-decoder Transformer (`d_model=64`, `nhead=4`, `num_encoder_layers=4`,
    `num_decoder_layers=4`, `dim_feedforward=256`, `dropout=0.1`, `decode_mode="NAR"`,
    `use_static=False` dynamic forcings only, `use_future_fea=True`) with learnable
    positional embeddings (`src_pos_emb`, `tgt_pos_emb`) and causal decoder self-attention mask.
  - Window (`src_len = past_len + pred_len`, `tgt_len = past_len + pred_len`):
      * Encoder input: `src_len` days `[t_0 - past_len .. t_0 + pred_len - 1]` of
        standardized meteorological forcings (`prcp, srad, tmax, tmin, vp`).
      * Decoder input: positions `0 .. past_len - 1` carry standardized historical
        observed runoff `[q_std, q_valid]` on days `t_0 - past_len .. t_0 - 1`, and
        positions `past_len .. tgt_len - 1` are zero-padded target queries.
      * Causal + past-only key mask prevents attending to zero-padded future tokens or
        future runoff within the historical sequence.
      * Readout: `Linear(d_model, 1)` on the trailing `pred_len` decoder positions.
  - Training:
      1. Global pretraining across all training basins (`NSELoss`, `Adam lr=1e-3`,
         `batch_size=512`, up to `200` epochs, linear warmup `0.25 * epochs` + `0.99`
         exponential decay) selecting best checkpoint by validation NSE.
      2. Full-network per-basin fine-tuning initialized from the best global checkpoint,
         updating ALL Transformer weights per basin (`Adam`, `MSELoss`, linear warmup +
         `0.99` exponential decay, early stopping by per-basin validation NSE).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.kratzert_lstm.model import nse_star_loss
from models.kratzert_lstm.windows import (
    feature_stats,
    mask_spinup_discharge,
    nse_star_weights,
    resolve_device,
    target_stats,
)


def _compute_basin_nse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute scalar NSE ignoring NaN/negative targets; returns `-inf` if < 2 valid points."""
    valid = np.isfinite(y_true) & (y_true >= 0.0) & np.isfinite(y_pred)
    if int(valid.sum()) < 2:
        return -float("inf")
    o = y_true[valid].astype(np.float64)
    s = y_pred[valid].astype(np.float64)
    denom = float(np.sum((o - np.mean(o)) ** 2))
    if denom <= 1e-12:
        return -float("inf")
    num = float(np.sum((s - o) ** 2))
    return 1.0 - num / denom


def _lr_warmup_exp(epoch: int, total_epochs: int, base_lr: float, decay: float = 0.99) -> float:
    """Linear warmup over first `0.25 * total_epochs` epochs followed by `decay^k` schedule."""
    if total_epochs <= 1:
        return float(base_lr)
    warmup_epochs = max(1, int(round(0.25 * total_epochs)))
    if epoch < warmup_epochs:
        return float(base_lr) * float(epoch + 1) / float(warmup_epochs)
    return float(base_lr) * (float(decay) ** float(epoch - warmup_epochs + 1))


def build_window_starts(
    valid_target: np.ndarray,
    forcing_finite: np.ndarray,
    past_len: int,
    pred_len: int,
    t_first: int,
) -> np.ndarray:
    """Return all `(basin, t_0)` pairs where forcing `[t_0-past_len, t_0+pred_len)` is finite and >=1 target in `[t_0, t_0+pred_len)` is valid."""
    n_b, n_t = valid_target.shape
    min_t0 = max(int(past_len), int(t_first))
    max_t0 = n_t - int(pred_len)
    if max_t0 < min_t0:
        return np.zeros((0, 2), dtype=np.int64)

    src_len = int(past_len) + int(pred_len)
    inv_f = (~forcing_finite).astype(np.int64)
    cum_f = np.concatenate([np.zeros((n_b, 1), dtype=np.int64), np.cumsum(inv_f, axis=1)], axis=1)
    cum_v = np.concatenate(
        [np.zeros((n_b, 1), dtype=np.int64), np.cumsum(valid_target.astype(np.int64), axis=1)],
        axis=1,
    )

    t0_arr = np.arange(min_t0, max_t0 + 1, dtype=np.int64)
    f_bad = cum_f[:, t0_arr + int(pred_len)] - cum_f[:, t0_arr - int(past_len)]
    v_cnt = cum_v[:, t0_arr + int(pred_len)] - cum_v[:, t0_arr]
    _ = src_len
    ok = (f_bad == 0) & (v_cnt > 0)
    rows = np.argwhere(ok).astype(np.int64)
    if rows.shape[0] > 0:
        rows[:, 1] += min_t0
    return rows


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
        self.dropout_p = float(dropout)
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
        attn = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(bsz, seq_len, d_model)
        )
        x = self.norm1(x + self.dropout(self.fc(attn)))
        ff = self.dropout(self.ffn2(self.dropout(F.relu(self.ffn1(x)))))
        return self.norm2(x + ff)


class RRFormerDecoderLayer(nn.Module):
    """Post-LayerNorm NAR Transformer decoder layer with causal runoff self-attention mask."""

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
        self.dropout_p = float(dropout)
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
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, tgt_len, d_model = tgt.shape
        src_len = memory.size(1)

        q = self.self_q(tgt).view(bsz, tgt_len, self.nhead, self.d_head).transpose(1, 2)
        k, v = (
            self.self_kv(tgt)
            .view(bsz, tgt_len, 2, self.nhead, self.d_head)
            .unbind(dim=2)
        )
        attn1 = (
            F.scaled_dot_product_attention(
                q,
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(bsz, tgt_len, d_model)
        )
        x = self.norm1(tgt + self.dropout(self.self_fc(attn1)))

        cq = self.cross_q(x).view(bsz, tgt_len, self.nhead, self.d_head).transpose(1, 2)
        ck, cv = (
            self.cross_kv(memory)
            .view(bsz, src_len, 2, self.nhead, self.d_head)
            .unbind(dim=2)
        )
        attn2 = (
            F.scaled_dot_product_attention(
                cq,
                ck.transpose(1, 2),
                cv.transpose(1, 2),
                dropout_p=self.dropout_p if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(bsz, tgt_len, d_model)
        )
        x = self.norm2(x + self.dropout(self.cross_fc(attn2)))
        ff = self.dropout(self.ffn2(self.dropout(F.relu(self.ffn1(x)))))
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
        max_len: int = 64,
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
        self.dec_proj = nn.Linear(2, self.d_model)
        self.src_pos_emb = nn.Embedding(max(max_len, self.src_len + 4), self.d_model)
        self.tgt_pos_emb = nn.Embedding(max(max_len, self.tgt_len + 4), self.d_model)
        self.emb_drop = nn.Dropout(p=dropout)

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

        # Causal + historical-only self-attention mask: query i attends to key j iff j <= i and j < past_len.
        q_pos = torch.arange(self.tgt_len, dtype=torch.long).unsqueeze(1)
        k_pos = torch.arange(self.tgt_len, dtype=torch.long).unsqueeze(0)
        allowed = (k_pos <= q_pos) & (k_pos < self.past_len)
        self.register_buffer("tgt_attn_mask", allowed, persistent=False)

    def forward(
        self,
        x_met: torch.Tensor,
        x_q_hist: torch.Tensor,
        x_stat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict `(batch, pred_len)` standardized runoff from `(batch, src_len, met)` and `(batch, past_len, 2)`."""
        bsz, src_len, _ = x_met.shape
        if self.use_static and x_stat is not None and self.stat_in_dim > 0:
            xs = x_stat.unsqueeze(1).expand(-1, src_len, -1)
            enc_in = torch.cat([x_met, xs], dim=-1)
        else:
            enc_in = x_met

        src_pos = torch.arange(src_len, device=x_met.device, dtype=torch.long).unsqueeze(0)
        memory = self.emb_drop(self.enc_proj(enc_in) + self.src_pos_emb(src_pos))
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory)

        dec_tokens = torch.zeros(
            (bsz, self.tgt_len, 2), dtype=x_met.dtype, device=x_met.device
        )
        dec_tokens[:, : self.past_len, :] = x_q_hist[:, : self.past_len, :]
        tgt_pos = torch.arange(self.tgt_len, device=x_met.device, dtype=torch.long).unsqueeze(0)
        dec_out = self.emb_drop(self.dec_proj(dec_tokens) + self.tgt_pos_emb(tgt_pos))
        mask = self.tgt_attn_mask
        for dec_layer in self.decoder_layers:
            dec_out = dec_layer(dec_out, memory, attn_mask=mask)
        return self.head(dec_out[:, -self.pred_len :, :]).squeeze(-1)


class RRFormerTrainer:
    """Global pretraining + full-network per-basin fine-tuning trainer for Yin et al. (2022) RR-Former."""

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
        finetune_patience: int = 25,
        pretrain_patience: int = 30,
        lr: float = 1e-3,
        finetune_lr: float = 1e-3,
        n_batches_per_epoch: int | None = None,
        clip_grad_norm: float = 1.0,
        eval_batch_size: int = 4096,
        lag_dropout_prob: float = 0.0,
        nse_eps: float = 0.1,
        use_amp: bool = True,
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
        self.device = resolve_device(device)

    def _gather_src_hist_tgt(
        self,
        xd_t: torch.Tensor,
        xq_t: torch.Tensor,
        y_t: torch.Tensor,
        m_t: torch.Tensor,
        b_idx: torch.Tensor,
        t0_idx: torch.Tensor,
        src_offs: torch.Tensor,
        past_offs: torch.Tensor,
        pred_offs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b_src = b_idx.unsqueeze(1).expand(-1, self.src_len)
        t_src = t0_idx.unsqueeze(1) + src_offs.unsqueeze(0)
        x_met = xd_t[b_src, t_src]

        b_hist = b_idx.unsqueeze(1).expand(-1, self.past_len)
        t_hist = t0_idx.unsqueeze(1) + past_offs.unsqueeze(0)
        x_q = xq_t[b_hist, t_hist]

        b_pred = b_idx.unsqueeze(1).expand(-1, self.pred_len)
        t_pred = t0_idx.unsqueeze(1) + pred_offs.unsqueeze(0)
        y_tgt = y_t[b_pred, t_pred]
        m_tgt = m_t[b_pred, t_pred]
        return x_met, x_q, y_tgt, m_tgt

    @torch.no_grad()
    def _eval_windows_nse(
        self,
        model: RRFormer,
        xd_t: torch.Tensor,
        xq_t: torch.Tensor,
        xs_t: torch.Tensor | None,
        y_mm_raw: np.ndarray,
        pairs_np: np.ndarray,
        y_mean: float,
        y_std: float,
        dev: torch.device,
        use_amp: bool,
    ) -> float:
        """Compute mean per-basin validation NSE across all basins with >=2 valid points."""
        if pairs_np.shape[0] == 0:
            return -float("inf")
        model.eval()
        src_offs = torch.arange(-self.past_len, self.pred_len, device=dev, dtype=torch.long)
        past_offs = torch.arange(-self.past_len, 0, device=dev, dtype=torch.long)
        pred_offs = torch.arange(0, self.pred_len, device=dev, dtype=torch.long)
        y_dummy = xq_t[:, :, 0]
        preds = np.empty((pairs_np.shape[0], self.pred_len), dtype=np.float32)
        pairs_t = torch.from_numpy(pairs_np).to(device=dev, dtype=torch.long)
        for s0 in range(0, pairs_np.shape[0], self.eval_batch_size):
            s1 = min(s0 + self.eval_batch_size, pairs_np.shape[0])
            rows = pairs_t[s0:s1]
            b_idx = rows[:, 0]
            t0_idx = rows[:, 1]
            x_met, x_q, _, _ = self._gather_src_hist_tgt(
                xd_t, xq_t, y_dummy, y_dummy, b_idx, t0_idx, src_offs, past_offs, pred_offs
            )
            xs_b = xs_t[b_idx] if (self.use_static and xs_t is not None) else None
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x_met, x_q, xs_b).float()
            preds[s0:s1] = np.clip(out.cpu().numpy() * y_std + y_mean, 0.0, None)

        # Group by basin using lead 0 (or all leads for per-basin validation NSE)
        t0_all = pairs_np[:, 1]
        b_all = pairs_np[:, 0]
        obs_mat = np.stack(
            [y_mm_raw[b_all, t0_all + h] for h in range(self.pred_len)], axis=1
        )
        uniq_b = np.unique(b_all)
        basin_nses: list[float] = []
        for b in uniq_b:
            mask_b = b_all == b
            score = _compute_basin_nse(preds[mask_b].reshape(-1), obs_mat[mask_b].reshape(-1))
            if np.isfinite(score):
                basin_nses.append(score)
        return float(np.median(basin_nses)) if basin_nses else -float("inf")

    def _finetune_single_basin(
        self,
        *,
        basin_i: int,
        global_state: dict[str, torch.Tensor],
        met_in_dim: int,
        stat_in_dim: int,
        tr_pairs_b: np.ndarray,
        val_pairs_b: np.ndarray,
        xd_tr_cpu: torch.Tensor,
        xq_tr_cpu: torch.Tensor,
        xs_tr_cpu: torch.Tensor | None,
        y_tr_cpu: torch.Tensor,
        m_tr_cpu: torch.Tensor,
        q_tr_raw_b: np.ndarray,
        xd_val_cpu: torch.Tensor,
        xq_val_cpu: torch.Tensor,
        xs_val_cpu: torch.Tensor | None,
        q_val_raw_b: np.ndarray,
        y_mean: float,
        y_std: float,
        gpu_id: int | None,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        """Fine-tune all Transformer weights on a single basin with Adam MSE + val NSE early stopping."""
        if tr_pairs_b.shape[0] < 30:
            return copy.deepcopy(global_state)

        dev = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else self.device
        use_amp = self.use_amp and dev.type == "cuda"

        model = RRFormer(
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
            max_len=max(64, self.src_len + 8),
        ).to(dev)
        model.load_state_dict({k: v.to(dev) for k, v in global_state.items()})

        # Pre-gather this basin's entire window dataset once onto `dev` (<3 MB per basin).
        src_offs = np.arange(-self.past_len, self.pred_len, dtype=np.int64)
        past_offs = np.arange(-self.past_len, 0, dtype=np.int64)
        pred_offs = np.arange(0, self.pred_len, dtype=np.int64)

        tr_t0 = tr_pairs_b[:, 1]
        xm_tr_b = xd_tr_cpu[basin_i][tr_t0[:, None] + src_offs[None, :]].to(dev)
        xq_tr_b = xq_tr_cpu[basin_i][tr_t0[:, None] + past_offs[None, :]].to(dev)
        yt_tr_b = y_tr_cpu[basin_i][tr_t0[:, None] + pred_offs[None, :]].to(dev)
        mt_tr_b = m_tr_cpu[basin_i][tr_t0[:, None] + pred_offs[None, :]].to(dev)
        xs_b = (
            xs_tr_cpu[basin_i : basin_i + 1].to(dev)
            if (self.use_static and xs_tr_cpu is not None)
            else None
        )

        has_val = val_pairs_b.shape[0] >= 10
        if has_val:
            va_t0 = val_pairs_b[:, 1]
            xm_va_b = xd_val_cpu[basin_i][va_t0[:, None] + src_offs[None, :]].to(dev)
            xq_va_b = xq_val_cpu[basin_i][va_t0[:, None] + past_offs[None, :]].to(dev)
            va_obs_mm = q_val_raw_b[va_t0[:, None] + pred_offs[None, :]]
            xs_va_b = (
                xs_val_cpu[basin_i : basin_i + 1].to(dev)
                if (self.use_static and xs_val_cpu is not None)
                else None
            )

        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_val_nse = -float("inf")
        if has_val:
            model.eval()
            with torch.no_grad():
                v_chunks: list[np.ndarray] = []
                for s0 in range(0, xm_va_b.size(0), self.eval_batch_size):
                    s1 = min(s0 + self.eval_batch_size, xm_va_b.size(0))
                    xs_sl = xs_va_b.expand(s1 - s0, -1) if xs_va_b is not None else None
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        vo = model(xm_va_b[s0:s1], xq_va_b[s0:s1], xs_sl).float()
                    v_chunks.append(np.clip(vo.cpu().numpy() * y_std + y_mean, 0.0, None))
                v_pred = np.concatenate(v_chunks, axis=0)
                init_nse = _compute_basin_nse(v_pred.reshape(-1), va_obs_mm.reshape(-1))
                if np.isfinite(init_nse):
                    best_val_nse = init_nse

        ft_epochs = max(1, self.finetune_epochs)
        ft_bs = min(self.batch_size, max(1, xm_tr_b.size(0)))
        n_ft_batches = (
            min(self.n_batches_per_epoch, max(1, xm_tr_b.size(0) // ft_bs))
            if self.n_batches_per_epoch is not None and self.n_batches_per_epoch > 0
            else max(1, (xm_tr_b.size(0) + ft_bs - 1) // ft_bs)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.finetune_lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + 10000 + basin_i)

        warmup_ep = max(1, int(round(0.25 * ft_epochs)))
        no_improve = 0
        for ep in range(ft_epochs):
            cur_lr = _lr_warmup_exp(ep, ft_epochs, self.finetune_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            model.train()
            perm = torch.randperm(xm_tr_b.size(0), generator=gen, device=dev)
            for b in range(n_ft_batches):
                idx = perm[b * ft_bs : (b + 1) * ft_bs]
                if idx.numel() == 0:
                    continue
                xs_sl = xs_b.expand(idx.numel(), -1) if xs_b is not None else None
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(xm_tr_b[idx], xq_tr_b[idx], xs_sl).float()
                sq = ((out - yt_tr_b[idx]) ** 2) * mt_tr_b[idx]
                loss = sq.sum() / torch.clamp(mt_tr_b[idx].sum(), min=1.0)
                scaler.scale(loss).backward()
                if self.clip_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if has_val:
                model.eval()
                with torch.no_grad():
                    v_chunks = []
                    for s0 in range(0, xm_va_b.size(0), self.eval_batch_size):
                        s1 = min(s0 + self.eval_batch_size, xm_va_b.size(0))
                        xs_sl = xs_va_b.expand(s1 - s0, -1) if xs_va_b is not None else None
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            vo = model(xm_va_b[s0:s1], xq_va_b[s0:s1], xs_sl).float()
                        v_chunks.append(np.clip(vo.cpu().numpy() * y_std + y_mean, 0.0, None))
                    v_pred = np.concatenate(v_chunks, axis=0)
                    v_nse = _compute_basin_nse(v_pred.reshape(-1), va_obs_mm.reshape(-1))
                if v_nse > best_val_nse + 1e-4:
                    best_val_nse = v_nse
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if ep >= warmup_ep and no_improve >= self.finetune_patience:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        return best_state

    def _build_q_hist_features(
        self,
        q_masked: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> np.ndarray:
        """Build `(n_basins, n_days, 2)` `[q_std, q_valid]` aligned on day `d` (strictly past when sliced before `t_0`)."""
        valid = np.isfinite(q_masked) & (q_masked >= 0.0)
        safe_q = np.where(valid, q_masked, y_mean)
        q_std = np.where(valid, (safe_q - y_mean) / max(float(y_std), 1e-6), 0.0).astype(
            np.float32
        )
        return np.stack([q_std, valid.astype(np.float32)], axis=-1)

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
        """Global NSELoss pretraining + full-network per-basin Adam MSE fine-tuning + inference."""
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds must be non-empty.")

        xd_tr_raw = np.asarray(x_dyn_train, dtype=np.float32)
        xs_tr_raw = np.asarray(x_stat_train, dtype=np.float32)
        xd_te_raw = np.asarray(x_dyn_test, dtype=np.float32)
        xs_te_raw = np.asarray(x_stat_test, dtype=np.float32)

        q_tr_all = mask_spinup_discharge(qobs_train, spinup_days_train)
        q_te_all = mask_spinup_discharge(qobs_test, spinup_days_test)

        # If no external val slice is passed (Stage 2 MIP protocol), carve chronological validation
        # strictly from the tail of `train_period` so zero test data is ever touched.
        if x_dyn_val is None or qobs_val is None:
            n_tr_tot = xd_tr_raw.shape[1]
            n_tr_target = n_tr_tot - int(spinup_days_train)
            val_days = min(730, max(60, n_tr_target // 5))
            if n_tr_target - val_days < self.src_len + 30:
                val_days = 0
            if val_days > 0:
                split_t = n_tr_tot - val_days
                val_start_with_spin = max(0, split_t - self.past_len)
                xd_va_raw = xd_tr_raw[:, val_start_with_spin:].copy()
                xs_va_raw = xs_tr_raw.copy()
                q_va_all = q_tr_all[:, val_start_with_spin:].copy()
                spin_va = split_t - val_start_with_spin
                q_va_all[:, :spin_va] = np.nan

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
            xs_va_raw = np.asarray(x_stat_val if x_stat_val is not None else xs_tr_raw, dtype=np.float32)
            q_va_all = mask_spinup_discharge(qobs_val, spinup_days_val)
            spin_va = int(spinup_days_val)
            xd_tr_fit_raw = xd_tr_raw
            q_tr_fit_all = q_tr_all

        fstats = feature_stats(xd_tr_fit_raw[:, spinup_days_train:, :], xs_tr_raw)
        tstats = target_stats(q_tr_fit_all[:, spinup_days_train:])
        y_mean = float(tstats["y_mean"])
        y_std = float(tstats["y_std"])
        weights_np = nse_star_weights(tstats["basin_std_mm"], eps=self.nse_eps)

        xd_tr_norm = np.nan_to_num(
            ((xd_tr_fit_raw - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
        )
        xd_te_norm = np.nan_to_num(
            ((xd_te_raw - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
        )
        xs_tr_norm = (
            ((xs_tr_raw - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
            if self.use_static
            else None
        )
        xs_te_norm = (
            ((xs_te_raw - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
            if self.use_static
            else None
        )

        xq_tr_norm = self._build_q_hist_features(q_tr_fit_all, y_mean, y_std)
        xq_te_norm = self._build_q_hist_features(q_te_all, y_mean, y_std)
        if self.lag_dropout_prob > 0.0:
            drop_rng = np.random.RandomState(seed_list[0] + 6151)
            dmask = drop_rng.uniform(0.0, 1.0, size=xq_tr_norm.shape[:2]) < self.lag_dropout_prob
            xq_tr_norm = xq_tr_norm.copy()
            xq_tr_norm[dmask] = 0.0

        y_tr_norm = np.nan_to_num(((q_tr_fit_all - y_mean) / y_std).astype(np.float32), nan=0.0)
        m_tr_valid = (np.isfinite(q_tr_fit_all) & (q_tr_fit_all >= 0.0)).astype(np.float32)
        f_tr_ok = np.all(np.isfinite(xd_tr_norm), axis=2)

        tr_pairs = build_window_starts(
            valid_target=m_tr_valid > 0.0,
            forcing_finite=f_tr_ok,
            past_len=self.past_len,
            pred_len=self.pred_len,
            t_first=spinup_days_train,
        )
        # Filter basins with finite NSE* weight
        fin_w = np.isfinite(weights_np)
        tr_pairs = tr_pairs[fin_w[tr_pairs[:, 0]]]
        if tr_pairs.shape[0] == 0:
            raise ValueError("No valid training windows in RRFormerTrainer.")

        if xd_va_raw is not None and q_va_all is not None:
            xd_va_norm = np.nan_to_num(
                ((xd_va_raw - fstats["xd_mean"]) / fstats["xd_std"]).astype(np.float32), nan=0.0
            )
            xs_va_norm = (
                ((xs_va_raw - fstats["xs_mean"]) / fstats["xs_std"]).astype(np.float32)
                if self.use_static and xs_va_raw is not None
                else None
            )
            xq_va_norm = self._build_q_hist_features(q_va_all, y_mean, y_std)
            m_va_valid = (np.isfinite(q_va_all) & (q_va_all >= 0.0)).astype(np.float32)
            va_pairs = build_window_starts(
                valid_target=m_va_valid > 0.0,
                forcing_finite=np.all(np.isfinite(xd_va_norm), axis=2),
                past_len=self.past_len,
                pred_len=self.pred_len,
                t_first=spin_va,
            )
        else:
            xd_va_norm = xd_tr_norm
            xs_va_norm = xs_tr_norm
            xq_va_norm = xq_tr_norm
            q_va_all = q_tr_fit_all
            va_pairs = np.zeros((0, 2), dtype=np.int64)

        n_batches = (
            int(self.n_batches_per_epoch)
            if self.n_batches_per_epoch is not None and self.n_batches_per_epoch > 0
            else max(1, tr_pairs.shape[0] // self.batch_size)
        )

        # Evaluation windows over test target days
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

        xd_tr_cpu = torch.from_numpy(xd_tr_norm)
        xq_tr_cpu = torch.from_numpy(xq_tr_norm)
        xs_tr_cpu = torch.from_numpy(xs_tr_norm) if xs_tr_norm is not None else None
        y_tr_cpu = torch.from_numpy(y_tr_norm)
        m_tr_cpu = torch.from_numpy(m_tr_valid)

        xd_va_cpu = torch.from_numpy(xd_va_norm)
        xq_va_cpu = torch.from_numpy(xq_va_norm)
        xs_va_cpu = torch.from_numpy(xs_va_norm) if xs_va_norm is not None else None

        xd_tr_t = xd_tr_cpu.to(dev)
        xq_tr_t = xq_tr_cpu.to(dev)
        xs_tr_t = xs_tr_cpu.to(dev) if xs_tr_cpu is not None else None
        y_tr_t = y_tr_cpu.to(dev)
        m_tr_t = m_tr_cpu.to(dev)
        w_tr_t = torch.from_numpy(weights_np).to(dev)

        xd_va_t = xd_va_cpu.to(dev)
        xq_va_t = xq_va_cpu.to(dev)
        xs_va_t = xs_va_cpu.to(dev) if xs_va_cpu is not None else None

        xd_te_t = torch.from_numpy(xd_te_norm).to(dev)
        xq_te_t = torch.from_numpy(xq_te_norm).to(dev)
        xs_te_t = torch.from_numpy(xs_te_norm).to(dev) if xs_te_norm is not None else None
        y_std_t = torch.tensor(y_std, dtype=torch.float32, device=dev)

        tr_pairs_t = torch.from_numpy(tr_pairs).to(device=dev, dtype=torch.long)
        src_offs_t = torch.arange(-self.past_len, self.pred_len, device=dev, dtype=torch.long)
        past_offs_t = torch.arange(-self.past_len, 0, device=dev, dtype=torch.long)
        pred_offs_t = torch.arange(0, self.pred_len, device=dev, dtype=torch.long)

        # Pre-index per-basin training and validation rows for fast fine-tuning
        n_tr_basins = xd_tr_norm.shape[0]
        tr_by_basin = [tr_pairs[tr_pairs[:, 0] == b] for b in range(n_tr_basins)]
        va_by_basin = [
            va_pairs[va_pairs[:, 0] == b] if va_pairs.shape[0] > 0 else np.zeros((0, 2), dtype=np.int64)
            for b in range(n_tr_basins)
        ]
        tr_index_map: dict[str, int] = {}
        if train_basins is not None:
            for idx, b_id in enumerate(train_basins):
                tr_index_map[str(b_id)] = idx

        met_in_dim = xd_tr_norm.shape[2]
        stat_in_dim = xs_tr_norm.shape[1] if xs_tr_norm is not None else 0

        member_preds: list[np.ndarray] = []
        for seed in seed_list:
            torch.manual_seed(seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            base_model = RRFormer(
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
                max_len=max(64, self.src_len + 8),
            ).to(dev)
            fwd_model: nn.Module = base_model

            optimizer = torch.optim.Adam(base_model.parameters(), lr=self.lr)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed + 4919)

            best_val_nse = -float("inf")
            best_global_state = {
                k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
            }
            warmup_ep = max(1, int(round(0.25 * self.epochs)))
            no_improve = 0

            for ep in range(self.epochs):
                fwd_model.train()
                cur_lr = _lr_warmup_exp(ep, self.epochs, self.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                perm = torch.randperm(tr_pairs_t.size(0), generator=gen, device=dev)
                for b in range(n_batches):
                    s0 = (b * self.batch_size) % tr_pairs_t.size(0)
                    idx = perm[s0 : s0 + self.batch_size]
                    if idx.numel() == 0:
                        continue
                    rows = tr_pairs_t[idx]
                    b_idx = rows[:, 0]
                    t0_idx = rows[:, 1]
                    x_met, x_q, y_tgt, m_tgt = self._gather_src_hist_tgt(
                        xd_tr_t,
                        xq_tr_t,
                        y_tr_t,
                        m_tr_t,
                        b_idx,
                        t0_idx,
                        src_offs_t,
                        past_offs_t,
                        pred_offs_t,
                    )
                    xs_b = xs_tr_t[b_idx] if (self.use_static and xs_tr_t is not None) else None
                    w_b = w_tr_t[b_idx].unsqueeze(1).expand(-1, self.pred_len)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        y_hat = fwd_model(x_met, x_q, xs_b).float()
                    active = m_tgt > 0.0
                    if active.any():
                        loss = nse_star_loss(
                            ((y_hat - y_tgt) * y_std_t)[active], w_b[active]
                        )
                        scaler.scale(loss).backward()
                        if self.clip_grad_norm > 0.0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                base_model.parameters(), max_norm=self.clip_grad_norm
                            )
                        scaler.step(optimizer)
                        scaler.update()

                if va_pairs.shape[0] > 0:
                    va_eval_pairs = (
                        va_pairs[:: max(1, self.pred_len)]
                        if va_pairs.shape[0] > 100000
                        else va_pairs
                    )
                    val_nse = self._eval_windows_nse(
                        base_model,
                        xd_va_t,
                        xq_va_t,
                        xs_va_t,
                        q_va_all,
                        va_eval_pairs,
                        y_mean,
                        y_std,
                        dev,
                        use_amp,
                    )
                    if val_nse > best_val_nse + 1e-4:
                        best_val_nse = val_nse
                        best_global_state = {
                            k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
                        }
                        no_improve = 0
                    else:
                        no_improve += 1
                        if ep >= warmup_ep and no_improve >= self.pretrain_patience:
                            print(
                                f"[RR-Former pretrain seed={seed}] early stop ep={ep + 1} best_val_nse={best_val_nse:.4f}",
                                flush=True,
                            )
                            break
                    if self.epochs >= 10 and (ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == self.epochs):
                        print(
                            f"[RR-Former pretrain seed={seed}] ep={ep + 1}/{self.epochs} lr={cur_lr:.2e} val_nse={val_nse:.4f} best={best_val_nse:.4f}",
                            flush=True,
                        )
                else:
                    best_global_state = {
                        k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
                    }

            # Determine which test basins need per-basin fine-tuning
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

            ft_states: dict[int, dict[str, torch.Tensor]] = {}
            sorted_tr = sorted(needed_tr_indices)
            if sorted_tr and self.finetune_epochs > 0:
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
                                    tr_pairs_b=tr_by_basin[bi],
                                    val_pairs_b=va_by_basin[bi],
                                    xd_tr_cpu=xd_tr_cpu,
                                    xq_tr_cpu=xq_tr_cpu,
                                    xs_tr_cpu=xs_tr_cpu,
                                    y_tr_cpu=y_tr_cpu,
                                    m_tr_cpu=m_tr_cpu,
                                    q_tr_raw_b=q_tr_fit_all[bi],
                                    xd_val_cpu=xd_va_cpu,
                                    xq_val_cpu=xq_va_cpu,
                                    xs_val_cpu=xs_va_cpu,
                                    q_val_raw_b=q_va_all[bi],
                                    y_mean=y_mean,
                                    y_std=y_std,
                                    gpu_id=idx % n_gpus,
                                    seed=seed,
                                ),
                            )
                            for idx, bi in enumerate(sorted_tr)
                        ]
                        for bi, fut in futs:
                            ft_states[bi] = fut.result()
                else:
                    single_gpu = 0 if n_gpus >= 1 and dev.type == "cuda" else None
                    for bi in sorted_tr:
                        ft_states[bi] = self._finetune_single_basin(
                            basin_i=bi,
                            global_state=best_global_state,
                            met_in_dim=met_in_dim,
                            stat_in_dim=stat_in_dim,
                            tr_pairs_b=tr_by_basin[bi],
                            val_pairs_b=va_by_basin[bi],
                            xd_tr_cpu=xd_tr_cpu,
                            xq_tr_cpu=xq_tr_cpu,
                            xs_tr_cpu=xs_tr_cpu,
                            y_tr_cpu=y_tr_cpu,
                            m_tr_cpu=m_tr_cpu,
                            q_tr_raw_b=q_tr_fit_all[bi],
                            xd_val_cpu=xd_va_cpu,
                            xq_val_cpu=xq_va_cpu,
                            xs_val_cpu=xs_va_cpu,
                            q_val_raw_b=q_va_all[bi],
                            y_mean=y_mean,
                            y_std=y_std,
                            gpu_id=single_gpu,
                            seed=seed,
                        )

            # Inference per test basin across all evaluation windows `[t0, t0 + pred_len)`
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
                    state_use = ft_states.get(tr_i, best_global_state) if tr_i is not None else best_global_state
                    base_model.load_state_dict({k: v.to(dev) for k, v in state_use.items()})
                    xm_b = xd_te_t[i_te][src_idx]
                    xq_b = xq_te_t[i_te][past_idx]
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
                        out_chunks.append(np.clip(o.cpu().numpy() * y_std + y_mean, 0.0, None))
                    pred_seed[i_te] = np.concatenate(out_chunks, axis=0)

            if return_lead_predictions:
                member_preds.append(pred_seed)
            else:
                member_preds.append(pred_seed[:, :, 0])

        members = np.stack(member_preds, axis=0).astype(np.float32)
        return np.mean(members, axis=0, dtype=np.float32), members
