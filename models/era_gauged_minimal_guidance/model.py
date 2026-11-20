# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""ERA Gauged Minimal Guidance winning architecture (Node 73 — EntityAwareLSTM)."""

from __future__ import annotations

import torch
from torch import nn


class EntityAwareLSTM(nn.Module):
    """2-layer Entity-Aware LSTM with 3-layer SiLU+LayerNorm static context embedding and initial state conditioning."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        hidden_size: int = 320,
        num_layers: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)

        self.stat_embed = nn.Sequential(
            nn.Linear(stat_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout),
        )

        self.init_h = nn.Linear(self.hidden_size, self.num_layers * self.hidden_size)
        self.init_c = nn.Linear(self.hidden_size, self.num_layers * self.hidden_size)

        self.lstm = nn.LSTM(
            dyn_dim + self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, 1),
        )

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(4.0)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_dyn.shape
        s_emb = self.stat_embed(x_stat)
        h0 = self.init_h(s_emb).view(self.num_layers, batch_size, self.hidden_size).contiguous()
        c0 = self.init_c(s_emb).view(self.num_layers, batch_size, self.hidden_size).contiguous()

        s_emb_seq = s_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x_combined = torch.cat([x_dyn, s_emb_seq], dim=-1)

        out, _ = self.lstm(x_combined, (h0, c0))
        return self.head(out)
