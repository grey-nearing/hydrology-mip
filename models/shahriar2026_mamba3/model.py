# Copyright 2026 Google LLC, Grey Nearing, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shahriar (2026) Mamba-3 Static-Dynamic Adapter architecture.

Reference:
  Shahriar, M. (2026). Can Mamba match LSTM for daily streamflow prediction?
  A parameter- and depth-matched benchmark on 671 CAMELS basins. SSRN.
  https://doi.org/10.2139/ssrn.7552815
  Code & Checkpoints: https://doi.org/10.5281/zenodo.23087719

Architecture (Tier-1 3-layer Mamba-3 Adapter, ~210K parameters):
  1. Dynamic input projection `Linear(n_dyn, H)` followed by `RMSNorm(H)`.
  2. `L = 3` layers, each applying **first** a residual Mamba-3 selective SSM
     block (`x = x + scale * Dropout(Mamba3(RMSNorm(x)))`, initialized with
     layer-scale `0.1`) and **second** a static-gated bottleneck adapter
     (`x = x + Adapter(x) * Sigmoid(StaticGate(x_stat))`).
  3. Linear readout `Linear(H, 1)` producing standardized daily runoff at every
     time step.
"""

from __future__ import annotations

import torch
from torch import nn

from models.shahriar2026_mamba3.mamba3 import Mamba3


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization computed in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(hidden_size)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


class MambaResidualBlock(nn.Module):
    """Pre-RMSNorm Mamba-3 residual mixer with learnable residual layer scale."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 32,
        mamba_dropout: float = 0.11,
        mamba_residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(mamba_residual_scale_init)))
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba3(
            d_model=int(d_model),
            d_state=int(d_state),
            expand=int(expand),
            headdim=int(headdim),
        )
        self.dropout = nn.Dropout(float(mamba_dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x))) * self.scale


class StaticAdapter(nn.Module):
    """Static catchment-attribute gated bottleneck adapter applied after each Mamba block."""

    def __init__(
        self,
        hidden_size: int,
        num_attrs: int,
        bottleneck: int = 64,
        dropout: float = 0.11,
    ) -> None:
        super().__init__()
        self.static_gate = nn.Sequential(
            nn.Linear(int(num_attrs), int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), int(hidden_size)),
            nn.Sigmoid(),
        )
        self.adapter = nn.Sequential(
            nn.Linear(int(hidden_size), int(bottleneck)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck), int(hidden_size)),
        )

    def forward(self, x: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        gate = self.static_gate(statics).unsqueeze(1)
        return x + self.adapter(x) * gate


class MambaStreamflowModelAdapter(nn.Module):
    """Shahriar (2026) Mamba-3 Static-Dynamic Adapter regional streamflow model."""

    def __init__(
        self,
        dim_dyn_in: int = 5,
        hidden_size: int = 80,
        num_attrs: int = 27,
        mamba_layers: int = 3,
        mamba_version: int = 3,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        mamba_adapter_bottleneck: int = 64,
        mamba_dropout: float = 0.11,
        mamba_residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if int(mamba_version) != 3:
            raise ValueError(f"Only Mamba-3 (mamba_version=3) is supported, got {mamba_version}.")
        del d_conv  # Mamba-3 uses trapezoidal discretization + rotary SSM in place of causal conv1d.
        self.dim_dyn_in = int(dim_dyn_in)
        self.hidden_size = int(hidden_size)
        self.num_attrs = int(num_attrs)
        self.mamba_layers = int(mamba_layers)

        self.dyn_in = nn.Linear(self.dim_dyn_in, self.hidden_size)
        self.norm_in = RMSNorm(self.hidden_size)

        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=self.hidden_size,
                    d_state=int(d_state),
                    expand=int(expand),
                    headdim=int(headdim),
                    mamba_dropout=float(mamba_dropout),
                    mamba_residual_scale_init=float(mamba_residual_scale_init),
                )
                for _ in range(self.mamba_layers)
            ]
        )
        self.adapters = nn.ModuleList(
            [
                StaticAdapter(
                    hidden_size=self.hidden_size,
                    num_attrs=self.num_attrs,
                    bottleneck=int(mamba_adapter_bottleneck),
                    dropout=float(mamba_dropout),
                )
                for _ in range(self.mamba_layers)
            ]
        )
        self.linearOut = nn.Linear(self.hidden_size, 1)

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        """Run Mamba-3 blocks + post-block static adapters on `(B, L, C_dyn)` and `(B, C_stat)`."""
        x = self.norm_in(self.dyn_in(x_dyn))
        # Verified directly against Shahriar (2026) Zenodo `model_Ep30.pt` + `pred_daily_m3s.npy`:
        # each layer applies `MambaResidualBlock` first, followed by `StaticAdapter`.
        for block, adapter in zip(self.blocks, self.adapters, strict=True):
            x = block(x)
            x = adapter(x, x_stat)
        return self.linearOut(x)


def build_optimizer_param_groups(
    model: nn.Module, weight_decay: float = 1e-3
) -> list[dict[str, object]]:
    """Split parameters into decayed 2-D projections and zero-weight-decay SSM/norm/bias vectors."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    no_decay_tokens = ("bias", "norm", "scale", "a_log", ".d", "b_bias", "c_bias", "dt_bias")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim < 2 or any(tok in lname for tok in no_decay_tokens):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
