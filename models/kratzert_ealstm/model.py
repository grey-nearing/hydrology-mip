# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# The EA-LSTM gate parameterization and initialization in this file follow
# papercode/ealstm.py from https://github.com/kratzert/ealstm_regional_modeling,
# Copyright 2019 Frederik Kratzert (Apache License 2.0), and
# neuralhydrology/modelzoo/ealstm.py, Copyright (c) 2021, NeuralHydrology
# (BSD 3-Clause License). See THIRD_PARTY_NOTICES.md.

"""Kratzert et al. (2019) Entity-Aware LSTM (EA-LSTM) architecture."""

from __future__ import annotations

import torch
from torch import nn


class _DynamicGates(nn.Module):
    """Dynamic forget, output, and cell update gate linear transformation for EA-LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        initial_forget_bias: float = 3.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
        self.bias = nn.Parameter(torch.empty(3 * hidden_size))
        self._reset_parameters(initial_forget_bias)

    def _reset_parameters(self, initial_forget_bias: float) -> None:
        nn.init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size).repeat(1, 3)
        self.weight_hh.data.copy_(weight_hh_data)
        nn.init.constant_(self.bias.data, val=0.0)
        self.bias.data[: self.hidden_size].fill_(float(initial_forget_bias))

    def forward(self, h: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        return h @ self.weight_hh + x_d @ self.weight_ih + self.bias


class EALSTM(nn.Module):
    """Entity-Aware LSTM (Kratzert et al., 2019) where static attributes modulate only the input gate."""

    def __init__(
        self,
        dyn_in_dim: int,
        stat_in_dim: int = 27,
        hidden_size: int = 256,
        dropout: float = 0.4,
        initial_forget_bias: float = 3.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gate = nn.Linear(stat_in_dim, hidden_size)
        self.dynamic_gates = _DynamicGates(
            input_size=dyn_in_dim,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def _forward_cudnn(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        """Exact fused cuDNN execution of EA-LSTM by zeroing dynamic/recurrent weights on `i`."""
        batch_size, seq_len, dyn_dim = x_d.shape
        stat_dim = x_s.size(1)
        h = self.hidden_size
        w_ih_f, w_ih_o, w_ih_g = self.dynamic_gates.weight_ih.t().chunk(3, dim=0)
        w_hh_f, w_hh_o, w_hh_g = self.dynamic_gates.weight_hh.t().chunk(3, dim=0)
        b_f, b_o, b_g = self.dynamic_gates.bias.chunk(3, dim=0)

        zero_id = x_d.new_zeros((h, dyn_dim))
        zero_s = x_d.new_zeros((h, stat_dim))
        zero_hh = x_d.new_zeros((h, h))
        w_ih = torch.cat(
            [
                torch.cat([zero_id, self.input_gate.weight], dim=1),
                torch.cat([w_ih_f, zero_s], dim=1),
                torch.cat([w_ih_g, zero_s], dim=1),
                torch.cat([w_ih_o, zero_s], dim=1),
            ],
            dim=0,
        ).contiguous()
        w_hh = torch.cat([zero_hh, w_hh_f, w_hh_g, w_hh_o], dim=0).contiguous()
        b_ih = torch.cat([self.input_gate.bias, b_f, b_g, b_o], dim=0).contiguous()
        b_hh = x_d.new_zeros((4 * h,))
        h0 = x_d.new_zeros((1, batch_size, h))
        c0 = x_d.new_zeros((1, batch_size, h))
        x_in = torch.cat([x_d, x_s.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        if torch.is_autocast_enabled("cuda"):
            amp_dtype = torch.get_autocast_dtype("cuda")
            x_in = x_in.to(amp_dtype)
            h0 = h0.to(amp_dtype)
            c0 = c0.to(amp_dtype)
            w_ih = w_ih.to(amp_dtype)
            w_hh = w_hh.to(amp_dtype)
            b_ih = b_ih.to(amp_dtype)
            b_hh = b_hh.to(amp_dtype)
        return torch._VF.lstm(
            x_in, (h0, c0), [w_ih, w_hh, b_ih, b_hh], True, 1, 0.0, self.training, False, True
        )[0]

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        if x_d.is_cuda:
            h_out = self._forward_cudnn(x_d, x_s)
        else:
            i_gate = torch.sigmoid(self.input_gate(x_s))
            x_pre = x_d @ self.dynamic_gates.weight_ih + self.dynamic_gates.bias
            h_out = _ealstm_recurrence(x_pre, i_gate, self.dynamic_gates.weight_hh)
        return self.head(self.dropout(h_out)).squeeze(-1)


@torch.jit.script
def _ealstm_recurrence(x_pre: torch.Tensor, i_gate: torch.Tensor, w_hh: torch.Tensor) -> torch.Tensor:
    """EA-LSTM time loop (TorchScript to remove per-step Python overhead on CPU).

    `x_pre` is the pre-computed input projection `(batch, seq_len, 3*hidden)`, `i_gate` the static
    input gate `(batch, hidden)`, and `w_hh` the recurrent weight `(hidden, 3*hidden)`; gate order
    is forget, output, cell-update.  Numerically identical to the eager loop.
    """
    batch_size, seq_len, _ = x_pre.shape
    hidden_size = w_hh.shape[0]
    h_t = torch.zeros(batch_size, hidden_size, dtype=x_pre.dtype, device=x_pre.device)
    c_t = torch.zeros(batch_size, hidden_size, dtype=x_pre.dtype, device=x_pre.device)
    outs: list[torch.Tensor] = []
    for t in range(seq_len):
        gates = torch.mm(h_t, w_hh) + x_pre[:, t, :]
        f, o, g = gates.chunk(3, dim=1)
        c_t = torch.sigmoid(f) * c_t + i_gate * torch.tanh(g)
        h_t = torch.sigmoid(o) * torch.tanh(c_t)
        outs.append(h_t)
    return torch.stack(outs, dim=1)
