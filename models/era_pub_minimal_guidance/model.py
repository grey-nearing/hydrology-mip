"""ERA PUB Minimal Guidance winning architecture (Node 82 — Enhanced EALSTM with EMA & Hybrid Huber-LogMSE Loss)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EALSTM(nn.Module):
    """Enhanced Entity-Aware LSTM with 2-layer LayerNorm+SiLU static bottleneck embedding for PUB."""

    def __init__(
        self,
        dyn_dim: int,
        stat_dim: int = 30,
        hidden_size: int = 256,
        stat_emb_size: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stat_embed = nn.Sequential(
            nn.Linear(stat_dim, stat_emb_size),
            nn.LayerNorm(stat_emb_size),
            nn.SiLU(),
            nn.Linear(stat_emb_size, stat_emb_size),
            nn.LayerNorm(stat_emb_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(dyn_dim + stat_emb_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        s_emb = self.stat_embed(x_stat)
        s_rep = s_emb.unsqueeze(1).expand(-1, x_dyn.size(1), -1)
        x_combined = torch.cat([x_dyn, s_rep], dim=-1)

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x_combined)

        if return_seq:
            return self.head(out).squeeze(-1)
        return self.head(out[:, -1, :]).squeeze(-1)


def hybrid_nse_loss(y_hat: torch.Tensor, y_true: torch.Tensor, b_std: torch.Tensor) -> torch.Tensor:
    """Combines basin-scaled Huber loss and log-MSE for PUB robustness (Node 82)."""
    scaled_err = (y_hat - y_true) / (b_std + 0.1)
    huber = F.huber_loss(scaled_err, torch.zeros_like(scaled_err), delta=1.0)
    log_mse = torch.mean(torch.square(torch.log1p(torch.relu(y_hat)) - torch.log1p(torch.relu(y_true))))
    return huber + 0.1 * log_mse
