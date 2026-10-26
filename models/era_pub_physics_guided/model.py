"""ERA Physics-Guided PUB winning architecture (Node 203 — PhysicsEALSTM with Static State Init & Softplus Head)."""

from __future__ import annotations

import torch
from torch import nn


class PhysicsEALSTM(nn.Module):
    """Physics-Informed Entity-Aware LSTM with static hidden/cell initialization and Softplus discharge head (Node 203)."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 27,
        hidden_size: int = 160,
        stat_emb_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stat_enc = nn.Sequential(
            nn.Linear(stat_dim, stat_emb_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(stat_emb_size, stat_emb_size),
        )
        self.h_init = nn.Linear(stat_emb_size, hidden_size)
        self.c_init = nn.Linear(stat_emb_size, hidden_size)
        self.lstm = nn.LSTM(dyn_dim + stat_emb_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        s_emb = self.stat_enc(x_s)
        s_rep = s_emb.unsqueeze(1).expand(-1, x_d.size(1), -1)
        x_combined = torch.cat([x_d, s_rep], dim=-1)

        h0 = self.h_init(s_emb).unsqueeze(0).contiguous()
        c0 = self.c_init(s_emb).unsqueeze(0).contiguous()

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x_combined, (h0, c0))
        q = self.head(out).squeeze(-1)
        return q if return_seq else q[:, -1]


def hybrid_nse_log_loss(y_hat: torch.Tensor, y_true: torch.Tensor, b_std: torch.Tensor) -> torch.Tensor:
    """Combines basin-std-scaled MSE and log1p-MSE loss (Node 203)."""
    yh = y_hat.float()
    yt = y_true.float()
    bs = b_std.float()
    mse_loss = torch.mean(((yh - yt) / (bs + 0.1)) ** 2)
    log_loss = torch.mean((torch.log1p(torch.clamp(yh, min=0.0)) - torch.log1p(torch.clamp(yt, min=0.0))) ** 2)
    return mse_loss + 0.2 * log_loss
