# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Neural hydrological architectures for ERA Gauged Multi-Paradigm (Newman CV)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvGRULSTM(nn.Module):
    """Multi-scale causal dilated Conv1D + coupled GRU-LSTM with basin conditioning."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        num_basins: int = 531,
        hidden_size: int = 288,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.basin_emb = nn.Embedding(num_basins, 32)
        self.stat_net = nn.Sequential(
            nn.Linear(stat_dim + 32, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.conv_d1 = nn.Conv1d(dyn_dim, 64, kernel_size=3, dilation=1)
        self.conv_d3 = nn.Conv1d(dyn_dim, 64, kernel_size=3, dilation=3)
        self.conv_d9 = nn.Conv1d(dyn_dim, 64, kernel_size=3, dilation=9)

        in_rec = dyn_dim + 192 + hidden_size
        self.gru = nn.GRU(in_rec, hidden_size, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(hidden_size + hidden_size, hidden_size, num_layers=1, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(3.0)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
        _, t_steps, _ = x_d.shape
        s_emb = self.stat_net(torch.cat([x_s, self.basin_emb(b_idx)], dim=-1))
        xt = x_d.transpose(1, 2)
        c1 = F.silu(self.conv_d1(F.pad(xt, (2, 0)))).transpose(1, 2)
        c3 = F.silu(self.conv_d3(F.pad(xt, (6, 0)))).transpose(1, 2)
        c9 = F.silu(self.conv_d9(F.pad(xt, (18, 0)))).transpose(1, 2)
        s_seq = s_emb.unsqueeze(1).expand(-1, t_steps, -1)
        g_out, _ = self.gru(torch.cat([x_d, c1, c3, c9, s_seq], dim=-1))
        l_out, _ = self.lstm(torch.cat([g_out, s_seq], dim=-1))
        return self.head(g_out + l_out).squeeze(-1)


class MultiScaleFiLMEALSTM(nn.Module):
    """Two-layer FiLM-modulated Entity-Aware LSTM with static + basin conditioning."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        num_basins: int = 531,
        hidden_size: int = 320,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.basin_emb = nn.Embedding(num_basins, 32)
        in_stat = stat_dim + 32

        self.stat_net = nn.Sequential(
            nn.Linear(in_stat, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.init_h = nn.Linear(hidden_size, self.num_layers * hidden_size)
        self.init_c = nn.Linear(hidden_size, self.num_layers * hidden_size)
        self.in_film = nn.Linear(hidden_size, dyn_dim * 2)
        self.out_film = nn.Linear(hidden_size, hidden_size * 2)

        self.lstm = nn.LSTM(
            dyn_dim + hidden_size,
            hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(3.5)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
        b_sz, t_steps, _ = x_d.shape
        s_emb = self.stat_net(torch.cat([x_s, self.basin_emb(b_idx)], dim=-1))
        gamma_in, beta_in = self.in_film(s_emb).chunk(2, dim=-1)
        x_d_mod = x_d * (1.0 + 0.2 * torch.tanh(gamma_in.unsqueeze(1))) + 0.2 * beta_in.unsqueeze(1)

        h0 = torch.tanh(self.init_h(s_emb)).view(self.num_layers, b_sz, self.hidden_size).contiguous()
        c0 = torch.tanh(self.init_c(s_emb)).view(self.num_layers, b_sz, self.hidden_size).contiguous()

        s_seq = s_emb.unsqueeze(1).expand(-1, t_steps, -1)
        out, _ = self.lstm(torch.cat([x_d_mod, s_seq], dim=-1), (h0, c0))

        gamma_out, beta_out = self.out_film(s_emb).chunk(2, dim=-1)
        out = out * (1.0 + 0.2 * torch.tanh(gamma_out.unsqueeze(1))) + 0.2 * beta_out.unsqueeze(1)
        return self.head(out).squeeze(-1)


class EntityAwareHydroLSTM(nn.Module):
    """Pure static-conditioned two-layer Entity-Aware HydroLSTM on snow-augmented features."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        hidden_size: int = 320,
        num_layers: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.stat_embed = nn.Sequential(
            nn.Linear(stat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.init_h = nn.Linear(hidden_size, num_layers * hidden_size)
        self.init_c = nn.Linear(hidden_size, num_layers * hidden_size)
        self.lstm = nn.LSTM(
            dyn_dim + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(4.0)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        b_sz, t_steps, _ = x_dyn.shape
        s_emb = self.stat_embed(x_stat)
        h0 = self.init_h(s_emb).view(self.num_layers, b_sz, self.hidden_size).contiguous()
        c0 = self.init_c(s_emb).view(self.num_layers, b_sz, self.hidden_size).contiguous()
        s_seq = s_emb.unsqueeze(1).expand(-1, t_steps, -1)
        out, _ = self.lstm(torch.cat([x_dyn, s_seq], dim=-1), (h0, c0))
        return self.head(out).squeeze(-1)


class DualPathwayFiLMGRULSTM(nn.Module):
    """Pure static-conditioned Dual-Pathway FiLM GRU + LSTM on snow-augmented features."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        hidden_size: int = 320,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.gru_hidden = 256
        self.stat_embed = nn.Sequential(
            nn.Linear(stat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.in_film = nn.Linear(hidden_size, dyn_dim * 2)
        self.init_h = nn.Linear(hidden_size, self.num_layers * hidden_size)
        self.init_c = nn.Linear(hidden_size, self.num_layers * hidden_size)
        self.init_gru = nn.Linear(hidden_size, self.num_layers * self.gru_hidden)

        self.lstm = nn.LSTM(
            dyn_dim + hidden_size,
            hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            dyn_dim + hidden_size,
            self.gru_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.gru_proj = nn.Linear(self.gru_hidden, hidden_size)
        self.out_film = nn.Linear(hidden_size, hidden_size * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(4.0)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        b_sz, t_steps, _ = x_dyn.shape
        s_emb = self.stat_embed(x_stat)
        gamma_in, beta_in = self.in_film(s_emb).chunk(2, dim=-1)
        x_mod = x_dyn * (1.0 + 0.15 * torch.tanh(gamma_in.unsqueeze(1))) + 0.15 * beta_in.unsqueeze(1)

        h0 = self.init_h(s_emb).view(self.num_layers, b_sz, self.hidden_size).contiguous()
        c0 = self.init_c(s_emb).view(self.num_layers, b_sz, self.hidden_size).contiguous()
        g0 = self.init_gru(s_emb).view(self.num_layers, b_sz, self.gru_hidden).contiguous()

        s_seq = s_emb.unsqueeze(1).expand(-1, t_steps, -1)
        inp = torch.cat([x_mod, s_seq], dim=-1)
        l_out, _ = self.lstm(inp, (h0, c0))
        g_out, _ = self.gru(inp, g0)
        out = l_out + 0.4 * self.gru_proj(g_out)

        gamma_out, beta_out = self.out_film(s_emb).chunk(2, dim=-1)
        out = out * (1.0 + 0.15 * torch.tanh(gamma_out.unsqueeze(1))) + 0.15 * beta_out.unsqueeze(1)
        return self.head(out).squeeze(-1)


class ResidualAtrousGRU(nn.Module):
    """5th family (Candidate 3): Multi-scale atrous Conv1D + GRU."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        num_basins: int = 531,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.basin_emb = nn.Embedding(num_basins, 32)
        self.stat_net = nn.Sequential(
            nn.Linear(stat_dim + 32, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.dilations = (1, 2, 4)
        self.convs = nn.ModuleList(
            [nn.Conv1d(dyn_dim, 64, kernel_size=3, dilation=d) for d in self.dilations]
        )
        self.gru = nn.GRU(64 * 3 + dyn_dim + hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
        _, t_steps, _ = x_d.shape
        s_feat = self.stat_net(torch.cat([x_s, self.basin_emb(b_idx)], dim=-1))
        xt = x_d.transpose(1, 2)
        c_feat = torch.cat(
            [F.silu(c(F.pad(xt, (2 * d, 0)))) for c, d in zip(self.convs, self.dilations)],
            dim=1,
        ).transpose(1, 2)
        inp = torch.cat([x_d, c_feat, s_feat.unsqueeze(1).expand(-1, t_steps, -1)], dim=-1)
        out, _ = self.gru(inp)
        return self.head(out).squeeze(-1)


class CausalAtrousGRU(nn.Module):
    """Lightweight causal depthwise atrous GRU."""

    def __init__(self, d_in: int, d_stat: int = 27, hidden: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_in, 3, dilation=2, groups=d_in)
        self.gru = nn.GRU(d_in + d_stat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, s: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
        del b
        xt = x.transpose(1, 2)
        x_c = F.silu(self.conv(F.pad(xt, (4, 0)))).transpose(1, 2)
        s_seq = s.unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.gru(torch.cat([x_c, s_seq], dim=-1))
        return self.head(out).squeeze(-1)
