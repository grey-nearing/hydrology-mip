# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""ERA Physics-Guided Gauged winning architecture (Node 187 — PhysicsHybridEALSTM / d-EA-LSTM)."""

from __future__ import annotations

import torch
from torch import nn


@torch.jit.script
def _snowpack_scan(
    p: torch.Tensor,
    t: torch.Tensor,
    mf: torch.Tensor,
    tc: torch.Tensor,
) -> torch.Tensor:
    b = p.size(0)
    seq_len = p.size(1)
    snow_pack = torch.zeros(b, device=p.device, dtype=p.dtype)
    liq_out: list[torch.Tensor] = []
    for i in range(seq_len):
        temp = t[:, i]
        precip = p[:, i]
        is_snow = (temp < tc).to(p.dtype)
        snowfall = precip * is_snow
        rainfall = precip * (1.0 - is_snow)
        melt = torch.clamp(mf * (temp - tc), min=0.0)
        actual_melt = torch.min(snow_pack, melt)
        snow_pack = snow_pack + snowfall - actual_melt
        liq_out.append(rainfall + actual_melt)
    return torch.stack(liq_out, dim=1)


@torch.jit.script
def _ealstm_scan(
    dyn_proj: torch.Tensor,
    i_gate: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    w_hh_t: torch.Tensor,
) -> torch.Tensor:
    seq_len = dyn_proj.size(1)
    h_out: list[torch.Tensor] = []
    for t in range(seq_len):
        gates = dyn_proj[:, t, :] + torch.mm(h, w_hh_t)
        f, g, o = gates.chunk(3, dim=-1)
        c = torch.sigmoid(f + 2.0) * c + i_gate * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        h_out.append(h)
    return torch.stack(h_out, dim=1)


class DifferentiableSnowPack(nn.Module):
    """Differentiable degree-day snow accumulation and melt module."""

    def forward(
        self,
        p: torch.Tensor,
        t: torch.Tensor,
        melt_factor: torch.Tensor,
        t_crit: torch.Tensor,
    ) -> torch.Tensor:
        mf = melt_factor.squeeze(-1)
        tc = t_crit.squeeze(-1)
        return _snowpack_scan(p, t, mf, tc)



class PhysicsHybridEALSTM(nn.Module):
    """Differentiable degree-day snowpack module coupled with an Entity-Aware LSTM (d-EA-LSTM)."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        num_basins: int = 531,
        hidden_size: int = 256,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_basins = int(num_basins)
        self.basin_embed = nn.Embedding(self.num_basins, int(embed_dim))

        self.static_net = nn.Sequential(
            nn.Linear(int(stat_dim) + int(embed_dim), self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
        )

        self.weight_dyn = nn.Linear(int(dyn_dim) + 1, self.hidden_size * 3)
        self.recurrent_dyn = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)

        self.phys_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

        self.snow_module = DifferentiableSnowPack()
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x_d: torch.Tensor,
        x_s: torch.Tensor,
        b_idx: torch.Tensor,
        raw_p: torch.Tensor,
        raw_t: torch.Tensor,
    ) -> torch.Tensor:
        b, seq_len, _ = x_d.shape
        if b_idx.numel() > 0 and (int(b_idx.min()) < 0 or int(b_idx.max()) >= self.num_basins):
            raise ValueError(
                f"basin index out of range [0, {self.num_basins}): "
                f"min={int(b_idx.min())}, max={int(b_idx.max())}"
            )
        s_feat = torch.cat([x_s, self.basin_embed(b_idx)], dim=-1)
        static_hidden = self.static_net(s_feat)
        i_gate_feat, phys_feat = torch.chunk(static_hidden, 2, dim=-1)

        p_params = self.phys_head(phys_feat)
        melt_factor = p_params[:, 0:1] * 8.0
        t_crit = (p_params[:, 1:2] - 0.5) * 4.0

        liq = self.snow_module(raw_p, raw_t, melt_factor, t_crit).unsqueeze(-1)

        i_gate = torch.sigmoid(i_gate_feat)
        h = torch.zeros(b, self.hidden_size, device=x_d.device, dtype=x_d.dtype)
        c = torch.zeros(b, self.hidden_size, device=x_d.device, dtype=x_d.dtype)

        x_combined = torch.cat([x_d, liq], dim=-1)
        dyn_proj = self.weight_dyn(x_combined)
        w_hh_t = self.recurrent_dyn.weight.t().contiguous().to(dtype=dyn_proj.dtype)
        h_seq = _ealstm_scan(
            dyn_proj,
            i_gate.to(dtype=dyn_proj.dtype),
            h.to(dtype=dyn_proj.dtype),
            c.to(dtype=dyn_proj.dtype),
            w_hh_t,
        )
        return self.output_head(h_seq)
