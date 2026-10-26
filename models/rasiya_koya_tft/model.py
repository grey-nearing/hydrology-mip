"""Rasiya Koya et al. (2023) Temporal Fusion Transformer (TFT) architecture for rainfall-runoff modeling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(self.drop(x)).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ctx_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, out_dim, dropout=dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        res = self.skip(x)
        h = self.fc1(x)
        if self.ctx_fc is not None and ctx is not None:
            h = h + self.ctx_fc(ctx)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.norm(res + h)


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.var_projs = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_vars)])
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_vars)
            ]
        )
        self.weight_grn = GatedResidualNetwork(
            num_vars * hidden_dim,
            hidden_dim,
            num_vars,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        if torch.is_autocast_enabled("cuda"):
            dt = torch.get_autocast_dtype("cuda")
            x = x.to(dt)
            if ctx is not None:
                ctx = ctx.to(dt)
        orig_shape = x.shape[:-1]
        v = self.num_vars
        h = self.var_projs[0].out_features
        dt = x.dtype
        w_proj = torch.stack([p.weight.squeeze(-1) for p in self.var_projs], dim=0).to(dt)
        b_proj = torch.stack([p.bias for p in self.var_projs], dim=0).to(dt)
        h0 = x.unsqueeze(-1) * w_proj + b_proj
        h_flat = h0.reshape(-1, v, h).transpose(0, 1).contiguous()

        w1 = torch.stack([g.fc1.weight for g in self.var_grns], dim=0).to(dt)
        b1 = torch.stack([g.fc1.bias for g in self.var_grns], dim=0).unsqueeze(1).to(dt)
        w2 = torch.stack([g.fc2.weight for g in self.var_grns], dim=0).to(dt)
        b2 = torch.stack([g.fc2.bias for g in self.var_grns], dim=0).unsqueeze(1).to(dt)
        wg = torch.stack([g.glu.fc.weight for g in self.var_grns], dim=0).to(dt)
        bg = torch.stack([g.glu.fc.bias for g in self.var_grns], dim=0).unsqueeze(1).to(dt)
        ln_w = torch.stack([g.norm.weight for g in self.var_grns], dim=0).unsqueeze(1).to(dt)
        ln_b = torch.stack([g.norm.bias for g in self.var_grns], dim=0).unsqueeze(1).to(dt)

        z = F.elu(torch.bmm(h_flat, w1.transpose(1, 2)) + b1)
        z = torch.bmm(z, w2.transpose(1, 2)) + b2
        z = self.var_grns[0].glu.drop(z)
        a, b = (torch.bmm(z, wg.transpose(1, 2)) + bg).chunk(2, dim=-1)
        u = F.layer_norm(h_flat + a * torch.sigmoid(b), (h,)).to(dt) * ln_w + ln_b

        stacked = u.transpose(0, 1).reshape(*orig_shape, v, h)
        flat = stacked.reshape(*orig_shape, v * h)
        weights = torch.softmax(self.weight_grn(flat, ctx), dim=-1).unsqueeze(-1)
        return (stacked * weights.to(dt)).sum(dim=-2)


class TemporalFusionTransformerHydro(nn.Module):
    """Temporal Fusion Transformer (Rasiya Koya et al., 2023) for CAMELS-US daily streamflow.

    Encoder-only TFT: static variable selection and context GRNs, temporal variable selection,
    LSTM encoder seeded with static context, gated skip/LayerNorm, static enrichment, causal
    multi-head self-attention, position-wise GRN, and a linear head predicting the standardized
    discharge at every time step.  Trainers use the last time step of a `seq_len` window.
    """

    def __init__(
        self,
        dyn_in_dim: int,
        stat_in_dim: int = 27,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.15,
        initial_forget_bias: float = 3.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.static_vsn = VariableSelectionNetwork(stat_in_dim, hidden_size, dropout=dropout)
        self.ctx_var_sel = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.ctx_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        self.ctx_h = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.ctx_c = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)

        self.temporal_vsn = VariableSelectionNetwork(
            dyn_in_dim,
            hidden_size,
            context_dim=hidden_size,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm.bias_hh_l0.data[hidden_size : 2 * hidden_size].fill_(float(initial_forget_bias))
        self.post_lstm_glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        self.static_enrich_grn = GatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            context_dim=hidden_size,
            dropout=dropout,
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        self.pos_wise_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.out_glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    supports_last_only: bool = True

    def forward(
        self, x_d: torch.Tensor, x_s: torch.Tensor, *, last_only: bool = False
    ) -> torch.Tensor:
        s_emb = self.static_vsn(x_s)
        c_s = self.ctx_var_sel(s_emb).unsqueeze(1)
        c_e = self.ctx_enrichment(s_emb).unsqueeze(1)
        h0 = self.ctx_h(s_emb).unsqueeze(0).contiguous()
        c0 = self.ctx_c(s_emb).unsqueeze(0).contiguous()

        xi = self.temporal_vsn(x_d, c_s)
        lstm_out, _ = self.lstm(xi, (h0, c0))
        phi = self.post_lstm_norm(xi + self.post_lstm_glu(lstm_out))
        theta = self.static_enrich_grn(phi, c_e)

        bsz, slen, edim = theta.shape
        n_heads = self.mha.num_heads
        head_dim = edim // n_heads
        if last_only:
            w_q, w_k, w_v = self.mha.in_proj_weight.chunk(3, dim=0)
            if self.mha.in_proj_bias is not None:
                b_q, b_k, b_v = self.mha.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None
            theta_last = theta[:, -1:, :]
            q = F.linear(theta_last, w_q, b_q).view(bsz, 1, n_heads, head_dim).transpose(1, 2)
            k = F.linear(theta, w_k, b_k).view(bsz, slen, n_heads, head_dim).transpose(1, 2)
            v = F.linear(theta, w_v, b_v).view(bsz, slen, n_heads, head_dim).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.mha.dropout if self.training else 0.0, is_causal=False
            )
            attn_out = self.mha.out_proj(attn_out.transpose(1, 2).contiguous().view(bsz, 1, edim))
            delta = self.post_attn_norm(theta_last + self.post_attn_glu(attn_out))
            psi = self.pos_wise_grn(delta)
            out = self.out_norm(phi[:, -1:, :] + self.out_glu(psi))
            return self.head(out).squeeze(-1)

        qkv = F.linear(theta, self.mha.in_proj_weight, self.mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, slen, n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, slen, n_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, slen, n_heads, head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.mha.dropout if self.training else 0.0, is_causal=True
        )
        attn_out = self.mha.out_proj(attn_out.transpose(1, 2).contiguous().view(bsz, slen, edim))
        delta = self.post_attn_norm(theta + self.post_attn_glu(attn_out))
        psi = self.pos_wise_grn(delta)
        out = self.out_norm(phi + self.out_glu(psi))
        return self.head(out).squeeze(-1)

    @torch.no_grad()
    def predict_eval_series(
        self,
        xd_te_t: torch.Tensor,
        xs_te_t: torch.Tensor,
        seq_len: int,
        spinup_days_test: int,
        n_target_days: int,
        batch_size: int = 2048,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Exact fast evaluation precomputing basin static contexts and pointwise temporal VSN."""
        n_basins, total_days, _ = xd_te_t.shape
        device = xd_te_t.device
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            s_emb = self.static_vsn(xs_te_t)
            c_s = self.ctx_var_sel(s_emb).unsqueeze(1)
            c_e = self.ctx_enrichment(s_emb).unsqueeze(1)
            h0_all = self.ctx_h(s_emb)
            c0_all = self.ctx_c(s_emb)
            xi_chunks = []
            for b0 in range(0, n_basins, 64):
                b1 = min(b0 + 64, n_basins)
                xi_chunks.append(self.temporal_vsn(xd_te_t[b0:b1], c_s[b0:b1]))
            xi_full = torch.cat(xi_chunks, dim=0)

            w_q, w_k, w_v = self.mha.in_proj_weight.chunk(3, dim=0)
            if self.mha.in_proj_bias is not None:
                b_q, b_k, b_v = self.mha.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None

        n_heads = self.mha.num_heads
        edim = self.hidden_size
        head_dim = edim // n_heads
        offsets = torch.arange(-seq_len + 1, 1, device=device, dtype=torch.long)

        b_grid = torch.arange(n_basins, device=device, dtype=torch.long).unsqueeze(1).expand(n_basins, n_target_days).reshape(-1)
        t_grid = (torch.arange(n_target_days, device=device, dtype=torch.long) + spinup_days_test).unsqueeze(0).expand(n_basins, n_target_days).reshape(-1)
        total_w = b_grid.numel()
        preds_flat = torch.empty(total_w, dtype=torch.float32, device=device)

        eff_bsz = max(batch_size, 2048)
        for i in range(0, total_w, eff_bsz):
            j = min(i + eff_bsz, total_w)
            b_idx = b_grid[i:j]
            t_end = t_grid[i:j]
            bsz = b_idx.numel()
            t_win = t_end.unsqueeze(1) + offsets.unsqueeze(0)
            xi = xi_full[b_idx.unsqueeze(1), t_win]
            h0 = h0_all[b_idx].unsqueeze(0).contiguous()
            c0 = c0_all[b_idx].unsqueeze(0).contiguous()
            c_e_b = c_e[b_idx]
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                lstm_out, _ = self.lstm(xi, (h0, c0))
                phi = self.post_lstm_norm(xi + self.post_lstm_glu(lstm_out))
                theta = self.static_enrich_grn(phi, c_e_b)
                theta_last = theta[:, -1:, :]
                q = F.linear(theta_last, w_q, b_q).view(bsz, 1, n_heads, head_dim).transpose(1, 2)
                k = F.linear(theta, w_k, b_k).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
                v = F.linear(theta, w_v, b_v).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
                attn_out = self.mha.out_proj(attn_out.transpose(1, 2).contiguous().view(bsz, 1, edim))
                delta = self.post_attn_norm(theta_last + self.post_attn_glu(attn_out))
                psi = self.pos_wise_grn(delta)
                out = self.out_norm(phi[:, -1:, :] + self.out_glu(psi))
                preds_flat[i:j] = self.head(out).squeeze(-1).squeeze(-1).float()

        return preds_flat.view(n_basins, n_target_days)
