"""ERA Physics-Guided PUR winning architecture (Node 206 — FiLM-Modulated API & PET LSTM)."""

from __future__ import annotations

import torch
from torch import nn


class PURPhysicsModel(nn.Module):
    """Physics-Informed FiLM-Modulated LSTM with static state initialization and static noise augmentation (Node 206)."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 30,
        hidden_size: int = 128,
        stat_emb_size: int = 64,
        dropout: float = 0.4,
        static_noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.static_noise_std = static_noise_std
        self.static_encoder = nn.Sequential(
            nn.Linear(stat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, stat_emb_size),
            nn.ReLU(),
        )
        self.input_weight = nn.Linear(stat_emb_size, dyn_dim)
        self.input_bias = nn.Linear(stat_emb_size, dyn_dim)

        self.h0_gen = nn.Linear(stat_emb_size, hidden_size)
        self.c0_gen = nn.Linear(stat_emb_size, hidden_size)

        self.lstm = nn.LSTM(dyn_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor,
        augment: bool = False,
        return_seq: bool = False,
    ) -> torch.Tensor:
        if augment and self.training and self.static_noise_std > 0.0:
            x_stat = x_stat + torch.randn_like(x_stat) * self.static_noise_std

        emb = self.static_encoder(x_stat)
        gamma = self.input_weight(emb).unsqueeze(1)
        beta = self.input_bias(emb).unsqueeze(1)
        x_gated = x_dyn * (1.0 + gamma) + beta

        h0 = self.h0_gen(emb).unsqueeze(0).contiguous()
        c0 = self.c0_gen(emb).unsqueeze(0).contiguous()

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x_gated, (h0, c0))
        if return_seq:
            return self.head(self.dropout(out)).squeeze(-1)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def scaled_mse_loss(y_hat: torch.Tensor, y_true: torch.Tensor, basin_scales: torch.Tensor) -> torch.Tensor:
    """Basin-std-scaled MSE loss (Node 206)."""
    yh = y_hat.float()
    yt = y_true.float()
    sc = basin_scales.float()
    sq_err = (yh - yt) ** 2
    return torch.mean(sq_err / (sc + 1e-3))
