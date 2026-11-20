# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""ERA PUR Minimal Guidance winning architecture (Node 135 — DualFiLMLSTM)."""

from __future__ import annotations

import torch
from torch import nn


class StatEncoder(nn.Module):
    """Static catchment identity encoder with LayerNorm, GELU, and Dropout."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualFiLMLSTM(nn.Module):
    """Dual Feature-wise Linear Modulation (FiLM) LSTM for geographic extrapolation (PUR)."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 33,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.h = int(hidden_size)
        self.stat_enc = StatEncoder(stat_dim, self.h)

        self.film_in = nn.Linear(self.h, dyn_dim * 2)
        self.film_out = nn.Linear(self.h, self.h * 2)

        self.init_h = nn.Linear(self.h, self.h)
        self.init_c = nn.Linear(self.h, self.h)

        self.lstm = nn.LSTM(dyn_dim, self.h, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.h, self.h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.h // 2, 1),
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        stat_emb = self.stat_enc(x_s)

        params_in = self.film_in(stat_emb).unsqueeze(1)
        gamma1, beta1 = params_in.chunk(2, dim=-1)
        x_d = x_d * (1.0 + gamma1) + beta1

        h0 = self.init_h(stat_emb).unsqueeze(0).contiguous()
        c0 = self.init_c(stat_emb).unsqueeze(0).contiguous()

        out, _ = self.lstm(x_d, (h0, c0))

        params_out = self.film_out(stat_emb).unsqueeze(1)
        gamma2, beta2 = params_out.chunk(2, dim=-1)
        out = out * (1.0 + gamma2) + beta2

        if not return_seq:
            out = out[:, -1, :]
        return self.head(out).squeeze(-1)
