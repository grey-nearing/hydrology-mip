# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mamba-3 sequence mixer layer for benchmarks.

Reference:
  Mamba-3: Improved Sequence Modeling using State Space Principles
  (arXiv:2603.15569).

Mamba-3 Architecture Overview:
------------------------------
State Space Models (SSMs) map a 1D continuous-time input signal x(t) to an
output signal y(t) through an N-dimensional latent state h(t):
    h'(t) = A * h(t) + B * x(t)
    y(t)  = C * h(t) + D * x(t)

Mamba-3 introduces three foundational architectural advancements over prior SSMs
(Mamba-1 and Mamba-2):

1. Exponential-Trapezoidal Discretization:
   Traditional SSMs rely exclusively on Zero-Order Hold (ZOH) discretization,
   which assumes the input x(t) is constant across the interval [t_{k-1}, t_k):
     h_k = exp(A * dt_k) * h_{k-1} + dt_k * (B * x_k)
   While computationally simple, ZOH introduces numerical errors when tracking
   high-frequency oscillations and rapidly varying inputs. Trapezoidal
   (bilinear / Tustin) integration instead approximates the input integral using
   the average of the endpoints:
     0.5 * (B * x_k + B * x_{k-1})
   Mamba-3 introduces a data-dependent, learned trapezoidal blending gate
   `trap_k` in (0, 1) that dynamically interpolates between ZOH and trapezoidal:
     u_k = dt_k * [(1 - trap_k) * (B * x_k) + trap_k * 0.5 * (B * x_k + B *
     x_{k-1})]
   This combines the bounded stability of ZOH with the higher-order numerical
   accuracy of trapezoidal integration.

2. Complex-Valued / Rotary State Space (RoPE in SSMs):
   Standard Rotary Position Embeddings (RoPE) in Transformers rotate paired
   query
   and key coordinates by fixed position frequencies: theta = position * omega.
   In continuous-time SSMs, the time step dt_k varies dynamically per token.
   Mamba-3 integrates angular velocity omega over continuous time:
     theta_k = sum_{s=0}^k dt_s * omega  =  integral_0^{t_k} omega * dt(s) ds
   The accumulated angle theta_k rotates paired coordinates of the state
   projection vectors B and C via 2D Givens rotation matrices:
     [B'_{2j}  ] = [cos(theta_j)  -sin(theta_j)] [B_{2j}  ]
     [B'_{2j+1}]   [sin(theta_j)   cos(theta_j)] [B_{2j+1}]
   This endows the recurrent memory with continuous, input-dependent phase
   dynamics.

3. Multi-Input Multi-Output (MIMO) Recurrence:
   In Single-Input Single-Output (SISO) state spaces, each head processes a
   scalar channel per step through an outer-product state update.
   MIMO generalizes this by projecting each head's channel into a rank-R
   subspace (via `mimo_x`), tracking R interacting state components per head,
   and contracting them back to output space (via `mimo_o`):
     x_r = x * mimo_x   (R sub-channels per head)
     y   = sum_r y_r * mimo_o
   Setting `is_mimo=False` seamlessly recovers standard SISO recurrence (rank
   1).

Computational Flow Diagram:
---------------------------
  u (batch, seq_len, d_model)
   │
   ├─► z_proj ────────────────────────────────────────────► z (gate)
   ├─► x_proj ────────────────────────────────────────────► x
   ├─► b_proj ─────► RMSNorm ─► + b_bias ─► RoPE(theta) ──► B
   ├─► c_proj ─────► RMSNorm ─► + c_bias ─► RoPE(theta) ──► C
   ├─► dt_proj ────► softplus(. + dt_bias) ───────────────► dt
   ├─► a_proj ─────► -max(softplus(.), a_floor) ──────────► A
   ├─► trap_proj ──► sigmoid ─────────────────────────────► trap
   └─► angle_proj ─► omega, theta = cumsum(dt * omega) over seq_len

  y   = mamba3_scan(x, B, C, A * dt, dt, trap, D, z)
      = (C h + D x) * silu(z), with the exponential-trapezoidal SSM state h
  out = out_proj(y)  (batch, seq_len, d_model)

Portability:
------------
This implementation is written in pure PyTorch without custom CUDA or Triton
kernels, ensuring portable, zero-overhead execution across TorchTPU (XLA),
TorchAX (JAX), and GPU platforms.
"""

from collections.abc import Callable
import math
from typing import Any, NamedTuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default chunk size of the chunked scan. Tensors whose minor dimension is a
# SISO chunk of 128 tokens fill the 128 lanes of TPU vector registers. MIMO
# chunks hold at most 128 // R tokens; see mamba3_scan.
_CHUNK_SIZE = 128

# Eager-mode cache of constant helper matrices, keyed by name, sizes, device,
# and dtype.
_ConstantCache = dict[tuple[object, ...], Any]

# Maximum number of entries of a _ConstantCache. A layer call adds at most four
# entries for a new sequence length, so only eager workloads with many
# distinct sequence lengths reach the limit, at which the cache is cleared.
_MAX_CACHE_ENTRIES = 64

_T = TypeVar("_T")


class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization (RMSNorm).

  Normalizes inputs across the last dimension by their root-mean-square:
    y = (x / sqrt(mean(x^2) + eps)) * weight

  RMSNorm is applied to B and C state projections to bound recurrence state
  magnitudes and stabilize gradients across deep unrolled recurrent steps.
  Calculations are executed in float32 for numerical stability across
  lower-precision modes (bfloat16, float16).
  """

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute reciprocal root-mean-square in float32 to prevent half-precision
    # overflow/underflow and use hardware rsqrt instead of sqrt + div.
    x_f = x.float()
    rrms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return (x_f * rrms * self.weight.float()).to(x.dtype)


def _cached(
    cache: _ConstantCache | None,
    key: tuple[object, ...],
    build: Callable[[], _T],
) -> _T:
  """Returns build(), memoized in cache[key] if a cache is given."""
  if cache is None:
    return build()
  # Another thread may clear the cache at any time, so the value is returned
  # from a local rather than read back from the cache. Concurrent misses at
  # worst build a value twice.
  value = cache.get(key)
  if value is None:
    if len(cache) >= _MAX_CACHE_ENTRIES:
      cache.clear()
    # Inference tensors cannot be saved for backward, so constants created
    # during a torch.inference_mode() call would break later training steps.
    with torch.inference_mode(False):
      value = build()
    cache[key] = value
  return value


def _additive_mask(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
  """Returns 0 where keep is True and -1e30 elsewhere.

  Adding the mask to a finite log-weight before exp() zeroes the masked
  weights, and the finite fill value keeps the sum finite.

  Args:
    keep: Boolean tensor.
    dtype: Floating-point dtype of the mask.
  """
  return torch.where(
      keep,
      torch.zeros((), device=keep.device, dtype=dtype),
      torch.full((), -1e30, device=keep.device, dtype=dtype),
  )


def _pad_seq(t: torch.Tensor, pad_len: int) -> torch.Tensor:
  """Zero-pads dim 1 (the sequence dimension) of t at the end by pad_len."""
  if pad_len == 0:
    return t
  return F.pad(t, (0, 0) * (t.ndim - 2) + (0, pad_len))


def _chunk_heads_first(
    t: torch.Tensor, num_chunks: int, q_len: int
) -> torch.Tensor:
  """Reshapes t of shape (B, C * Q, H, ...) to contiguous (B, H, C, Q, ...)."""
  t = t.reshape(t.shape[0], num_chunks, q_len, *t.shape[2:])
  return t.movedim(3, 1).contiguous()


class _SegsumMats(NamedTuple):
  """Constant (T, T) matrices for _segsum_exp.

  Attributes:
    prefix: (x @ prefix)[t] = x[1] + ... + x[t].
    strict_tril: 1 where tau < t, indexed [t, tau].
    causal_mask: Additive mask keeping tau <= t, indexed [t, tau].
  """

  prefix: torch.Tensor
  strict_tril: torch.Tensor
  causal_mask: torch.Tensor


def _get_segsum_mats(
    t_len: int,
    device: torch.device,
    dtype: torch.dtype,
    cache: _ConstantCache | None = None,
) -> _SegsumMats:
  """Returns the (cached) constant matrices for _segsum_exp."""

  def build() -> _SegsumMats:
    idx = torch.arange(t_len, device=device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)
    return _SegsumMats(
        prefix=((row >= 1) & (row <= col)).to(dtype),
        strict_tril=(row > col).to(dtype),
        causal_mask=_additive_mask(row >= col, dtype),
    )

  return _cached(cache, ("segsum", t_len, device, dtype), build)


def _segsum_exp(
    x: torch.Tensor, cache: _ConstantCache | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes lower-triangular segment sums and their exponentials.

  For an input tensor x of shape (..., T), computes for 0 <= tau <= t < T:
    log_m[..., t, tau] = sum_{k=tau+1}^t x[..., k]
    m[..., t, tau]     = exp(log_m[..., t, tau])  (and 0 for tau > t)

  The segment sums are differences of prefix sums that start at k = 1, so
  x[..., 0] never enters them. The diagonal (log_m = 0, m = 1) is masked
  instead of computed as s_t - s_t, whose gradient terms would only cancel in
  exact arithmetic.

  Args:
    x: Log-decay tensor of shape (..., T).
    cache: Optional dictionary for caching constant helper matrices in eager
      mode.

  Returns:
    A tuple (log_m, m) of shape (..., T, T).
  """
  mats = _get_segsum_mats(x.shape[-1], x.device, x.dtype, cache)
  s = torch.matmul(x, mats.prefix)
  log_m = (s.unsqueeze(-1) - s.unsqueeze(-2)) * mats.strict_tril
  m = torch.exp(log_m + mats.causal_mask)
  return log_m, m


class _CumsumMats(NamedTuple):
  """Constant matrices for chunked prefix sums over chunks of Q positions.

  Attributes:
    tril_q: (Q, Q) lower-triangular ones, tril_q @ v is the inclusive prefix sum
      of v over Q.
    ones_1xq: (1, Q) ones, for chunk totals and broadcasts over Q.
  """

  tril_q: torch.Tensor
  ones_1xq: torch.Tensor


def _get_cumsum_mats(
    q_len: int,
    device: torch.device,
    dtype: torch.dtype,
    cache: _ConstantCache | None = None,
) -> _CumsumMats:
  """Returns the (cached) constant matrices for _rope_angles_chunked."""

  def build() -> _CumsumMats:
    return _CumsumMats(
        tril_q=torch.tril(torch.ones(q_len, q_len, device=device, dtype=dtype)),
        ones_1xq=torch.ones(1, q_len, device=device, dtype=dtype),
    )

  return _cached(cache, ("cumsum", q_len, device, dtype), build)


def _rope_angles_chunked(
    increments: torch.Tensor, cache: _ConstantCache | None = None
) -> torch.Tensor:
  """Returns RoPE angles theta = cumsum(increments) over chunks, modulo 2 pi.

  The prefix sum runs over the flattened (C, Q) dimensions: a lower-triangular
  matmul sums within each chunk, and the exclusive prefix sum of the chunk
  totals is added to every position of the chunk. These chunk offsets are
  reduced to [-pi, pi), which leaves cos(theta) and sin(theta) unchanged.

  On TPU, float32 matmuls round their operands to bfloat16 unless
  torch.set_float32_matmul_precision("high") or "highest" is set. The operands
  here are the per-token increments, which then have a relative error of at
  most 2^-8 each, and the offsets reduced to [-pi, pi), which then have an
  absolute error of at most 2^-7. The angles thus keep the precision of
  bfloat16 increments, whereas rounding the accumulated angles would lose it
  entirely: bfloat16 rounds an angle of 400 rad to a multiple of 2 rad.

  Args:
    increments: Float32 angle increments dt * omega of shape (..., C, Q, N).
    cache: Optional dictionary for caching constant helper matrices in eager
      mode.

  Returns:
    Angles of shape (..., C, Q, N), congruent modulo 2 pi to the inclusive
    prefix sum of the increments over (C, Q).
  """
  num_chunks, q_len = increments.shape[-3:-1]
  mats = _get_cumsum_mats(q_len, increments.device, increments.dtype, cache)
  angles = torch.matmul(mats.tril_q, increments)
  if num_chunks > 1:
    totals = torch.matmul(mats.ones_1xq, increments)  # (..., C, 1, N)
    offsets = torch.cumsum(totals, dim=-3) - totals
    offsets = torch.remainder(offsets + math.pi, 2 * math.pi) - math.pi
    # Broadcast the offsets over Q with a matmul, so that the backward pass
    # contracts over Q on the MXU instead of reducing across TPU lanes.
    angles = angles + torch.matmul(mats.ones_1xq.mT, offsets)
  return angles


def _rope_angles_dim1(
    increments: torch.Tensor, cache: _ConstantCache | None = None
) -> torch.Tensor:
  """Returns RoPE angles cumsum(increments, dim=1), modulo 2 pi.

  Args:
    increments: Float32 angle increments of shape (B, L, ...).
    cache: Optional dictionary for caching constant helper matrices in eager
      mode.

  Returns:
    Angles of the same shape, congruent modulo 2 pi to
    torch.cumsum(increments, dim=1). See _rope_angles_chunked.
  """
  batch_size, seq_len = increments.shape[:2]
  q_len = min(_CHUNK_SIZE, seq_len)
  pad_len = (-seq_len) % q_len
  num_chunks = (seq_len + pad_len) // q_len
  flat = _pad_seq(increments.reshape(batch_size, seq_len, -1), pad_len)
  angles = _rope_angles_chunked(
      flat.view(batch_size, num_chunks, q_len, -1), cache
  )
  angles = angles.view(batch_size, num_chunks * q_len, *increments.shape[2:])
  return angles[:, :seq_len] if pad_len > 0 else angles


class _RopeMats(NamedTuple):
  """Constant matrices that apply RoPE with matmuls instead of strided slices.

  For t of shape (..., D) and per-angle cos/sin of shape (..., N):
    cos_full = cos @ expand (+ unrot_mask),  sin_full = sin @ expand,
    rope(t)  = t * cos_full + (t @ perm) * sin_full.

  Attributes:
    expand: (N, D), repeats angle k at coordinates 2k and 2k + 1.
    perm: (D, D), (t @ perm)[2k] = -t[2k + 1] and (t @ perm)[2k + 1] = t[2k].
    unrot_mask: (D,) ones at the unrotated coordinates d >= 2N, or None if all
      coordinates are rotated.
  """

  expand: torch.Tensor
  perm: torch.Tensor
  unrot_mask: torch.Tensor | None


def _get_rope_mats(
    d_state: int,
    num_angles: int,
    device: torch.device,
    dtype: torch.dtype,
    cache: _ConstantCache | None = None,
) -> _RopeMats:
  """Returns the (cached) constant RoPE matrices."""

  def build() -> _RopeMats:
    k_row = torch.arange(num_angles, device=device).unsqueeze(1)
    d_idx = torch.arange(d_state, device=device)
    d_row = d_idx.unsqueeze(1)
    d_col = d_idx.unsqueeze(0)
    expand = (d_col.eq(k_row * 2) | d_col.eq(k_row * 2 + 1)).to(dtype)
    rotated = d_idx < 2 * num_angles
    even = rotated & (d_idx % 2 == 0)
    odd = rotated & (d_idx % 2 == 1)
    pos = (even.unsqueeze(1) & (d_col == d_row + 1)).to(dtype)
    neg = (odd.unsqueeze(1) & (d_col == d_row - 1)).to(dtype)
    unrot_mask = (~rotated).to(dtype) if 2 * num_angles < d_state else None
    return _RopeMats(expand=expand, perm=pos - neg, unrot_mask=unrot_mask)

  return _cached(cache, ("rope", d_state, num_angles, device, dtype), build)


class _SsdChunkMats(NamedTuple):
  """Constant matrices for _mamba3_scan_prechunked.

  Per-position vectors v have the Q positions of a chunk as their last
  dimension, so v @ m applies m. Per-chunk tensors (..., C, Q) are mixed
  across chunks by m @ v. Matrices over pairs of positions (or chunks) are
  indexed [t, tau] (or [c, c']).

  Attributes:
    prefix_q: (Q, Q), (v @ prefix_q)[t] = v[1] + ... + v[t].
    shift_q: (Q, Q), (v @ shift_q)[s] = v[s + 1], and 0 for s = Q - 1.
    suffix_q: (Q, Q), (v @ suffix_q)[s] = v[s + 1] + ... + v[Q - 1].
    first_q: (Q, 1), v @ first_q = v[0].
    first_to_last_q: (Q, Q), v @ first_to_last_q is v[0] at Q - 1, else 0.
    e0_q: (Q,), one-hot at position 0.
    eye_q: (Q, Q) identity.
    below_sub_q: (Q, Q), 1 where tau <= t - 2.
    strict_causal_q: (Q, Q), additive mask keeping tau < t.
    shift_c: (C, C), (shift_c @ v)[c] = v[c + 1], and 0 for c = C - 1.
    excl_prefix_c: (C, C), (v @ excl_prefix_c)[c] = v[1] + ... + v[c - 1].
    incl_prefix_c: (C, C), (v @ incl_prefix_c)[c] = v[1] + ... + v[c].
    below_sub_c: (C, C), 1 where c' <= c - 2.
    strict_causal_c: (C, C), additive mask keeping c' < c.
  """

  prefix_q: torch.Tensor
  shift_q: torch.Tensor
  suffix_q: torch.Tensor
  first_q: torch.Tensor
  first_to_last_q: torch.Tensor
  e0_q: torch.Tensor
  eye_q: torch.Tensor
  below_sub_q: torch.Tensor
  strict_causal_q: torch.Tensor
  shift_c: torch.Tensor
  excl_prefix_c: torch.Tensor
  incl_prefix_c: torch.Tensor
  below_sub_c: torch.Tensor
  strict_causal_c: torch.Tensor


def _get_ssd_chunk_mats(
    q_len: int,
    num_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
    cache: _ConstantCache | None = None,
) -> _SsdChunkMats:
  """Returns the (cached) constant matrices for _mamba3_scan_prechunked."""

  def build() -> _SsdChunkMats:
    q_idx = torch.arange(q_len, device=device)
    q_row = q_idx.unsqueeze(1)
    q_col = q_idx.unsqueeze(0)
    c_idx = torch.arange(num_chunks, device=device)
    c_row = c_idx.unsqueeze(1)
    c_col = c_idx.unsqueeze(0)
    first_q = (q_row == 0).to(dtype)
    return _SsdChunkMats(
        prefix_q=((q_row >= 1) & (q_row <= q_col)).to(dtype),
        shift_q=(q_row == q_col + 1).to(dtype),
        suffix_q=(q_row > q_col).to(dtype),
        first_q=first_q,
        first_to_last_q=((q_row == 0) & (q_col == q_len - 1)).to(dtype),
        e0_q=first_q.squeeze(1),
        eye_q=(q_row == q_col).to(dtype),
        below_sub_q=(q_col <= q_row - 2).to(dtype),
        strict_causal_q=_additive_mask(q_col < q_row, dtype),
        shift_c=(c_col == c_row + 1).to(dtype),
        excl_prefix_c=((c_row >= 1) & (c_row <= c_col - 1)).to(dtype),
        incl_prefix_c=((c_row >= 1) & (c_row <= c_col)).to(dtype),
        below_sub_c=(c_col <= c_row - 2).to(dtype),
        strict_causal_c=_additive_mask(c_col < c_row, dtype),
    )

  return _cached(cache, ("ssd", q_len, num_chunks, device, dtype), build)


def _mamba3_scan_prechunked(
    x_c: torch.Tensor,
    b_c: torch.Tensor,
    c_c: torch.Tensor,
    adt_c: torch.Tensor,
    dt_c: torch.Tensor,
    trap_c: torch.Tensor,
    d_skip: torch.Tensor,
    z_c: torch.Tensor | None = None,
    cache: _ConstantCache | None = None,
) -> torch.Tensor:
  """Runs the chunked SISO Mamba-3 scan on (B, H, C, Q, ...) tensors.

  With beta_t = dt_t * trap_t / 2 and alpha_t = dt_t - beta_t, the recurrence
    h_t = exp(adt_t) h_{t-1} + alpha_t B_t x_t + beta_t B_{t-1} x_{t-1}
  unrolls to h_t = sum_{tau <= t} W[t, tau] B_tau x_tau with
    W[t, t]   = alpha_t,
    W[t, tau] = M[t, tau] alpha_tau + M[t, tau + 1] beta_{tau + 1}
              = M[t, tau + 1] alpha_hat_tau                     (tau < t),
    alpha_hat_tau = exp(adt_{tau + 1}) alpha_tau + beta_{tau + 1},
  where M[t, s] = exp(adt_{s + 1} + ... + adt_t) is the decay from s to t.
  Every exponent is a sum of log-decays adt <= 0, so nothing overflows even
  under strong decay. The unit decays M[s, s], in W[s, s - 1] and in the
  weight of U_{c - 1} in G_c below, come from masks rather than from the
  difference of two equal prefix sums, whose gradient terms would only cancel
  in exact arithmetic.

  Within a chunk, y = (C B^T * W) x. Across chunks, chunk c' contributes
    U_c' = sum_{tau in c'} W[f(c' + 1), tau] x_tau B_tau^T
  to the state at the first position f(c' + 1) of the next chunk (excluding
  the input at that position), chunk c starts from
    G_c = sum_{c' < c} M[f(c), f(c' + 1)] U_c',
  and position t of chunk c adds M[t, f(c)] G_c C_t.

  Decay weights are computed in float32 and cast to the dtype of x_c for the
  contractions. On TPU, the float32 matmuls that form prefix sums and shifts
  round their operands to bfloat16 unless a higher float32 matmul precision is
  set (see _rope_angles_chunked). That changes each log M by a relative error
  of at most about 2^-7 (2^-8 per rounding, and chunk totals are rounded
  twice), and hence each M by at most about 2^-7 / e, since -M log M <= 1 / e.

  Args:
    x_c: Input values of shape (B, H, C, Q, P).
    b_c: B projection of shape (B, H, C, Q, D) in the dtype of x_c.
    c_c: C projection of shape (B, H, C, Q, D) in the dtype of x_c.
    adt_c: Float32 log-decay A * dt of shape (B, H, C, Q).
    dt_c: Float32 time step of shape (B, H, C, Q).
    trap_c: Float32 trapezoidal gate of shape (B, H, C, Q).
    d_skip: Skip weight of shape (H,).
    z_c: Optional gate of shape (B, H, C, Q, P).
    cache: Optional dictionary for caching constant helper matrices in eager
      mode.

  Returns:
    Output of shape (B, C * Q, H, P), gated by silu(z_c) if z_c is given.
  """
  batch_size, num_heads, num_chunks, q_len, head_dim = x_c.shape
  d_state = b_c.shape[-1]
  dtype = x_c.dtype
  mats = _get_ssd_chunk_mats(
      q_len, num_chunks, adt_c.device, torch.float32, cache
  )

  beta_c = dt_c * (0.5 * trap_c)
  alpha_c = dt_c - beta_c

  # Intra-chunk weights W of shape (B, H, C, Q, Q). With
  # cum[t] = adt[1] + ... + adt[t], log M[t, tau + 1] = cum[t] - cum_next[tau]
  # where cum_next[tau] = cum[tau] + adt[tau + 1] = cum[tau + 1].
  cum = torch.matmul(adt_c, mats.prefix_q)
  adt_next = torch.matmul(adt_c, mats.shift_q)
  beta_next = torch.matmul(beta_c, mats.shift_q)
  alpha_hat = torch.exp(adt_next) * alpha_c + beta_next
  log_m_next = (
      cum.unsqueeze(-1) - (cum + adt_next).unsqueeze(-2)
  ) * mats.below_sub_q
  m_next = torch.exp(log_m_next + mats.strict_causal_q)
  w = (
      m_next * alpha_hat.unsqueeze(-2) + mats.eye_q * alpha_c.unsqueeze(-2)
  ).to(dtype)

  cb = torch.matmul(c_c, b_c.transpose(-1, -2))  # (B, H, C, Q, Q)
  y_out = torch.matmul(cb * w, x_c)  # (B, H, C, Q, P)

  if num_chunks > 1:
    # w_next[c', s] = W[f(c' + 1), s]. log_to_next[s] = log M[f(c' + 1), s]
    # = adt[s + 1] + ... + adt[Q - 1] + adt_next_chunk[0], so for s < Q - 1,
    # W = exp(log_to_next[s] - adt[s + 1]) alpha_hat[s]. For s = Q - 1,
    # adt_next and beta_next are 0, and the same expression plus
    # beta_next_chunk[0] gives W = M[f(c' + 1), s] alpha_s + beta_{f(c' + 1)}.
    adt_next_chunk = torch.matmul(mats.shift_c, adt_c)
    beta_next_chunk = torch.matmul(mats.shift_c, beta_c)
    log_to_next = torch.matmul(adt_c, mats.suffix_q) + torch.matmul(
        adt_next_chunk, mats.first_q
    )
    w_next = (
        torch.exp(log_to_next - adt_next) * alpha_hat
        + torch.matmul(beta_next_chunk, mats.first_to_last_q)
    ).to(dtype)
    m_first = torch.exp(cum).to(dtype)  # M[t, f(c)]

    # Keep d_state (128) as the minor-most dimension of u_chunk_t so flatten(-2)
    # and view(..., head_dim, d_state) are zero-copy bitcasts without tile
    # padding.
    u_chunk_t = torch.matmul(
        x_c.transpose(-1, -2), b_c * w_next.unsqueeze(-1)
    )  # (B, H, C, P, D)
    # log_gamma[c'] = log M[f(c' + 1), f(c')], so that
    # log M[f(c), f(c' + 1)] = log_gamma[c' + 1] + ... + log_gamma[c - 1].
    log_gamma = (log_to_next * mats.e0_q).sum(dim=-1)  # (B, H, C)
    cum_dst = torch.matmul(log_gamma, mats.excl_prefix_c)
    cum_src = torch.matmul(log_gamma, mats.incl_prefix_c)
    log_m_inter = (
        cum_dst.unsqueeze(-1) - cum_src.unsqueeze(-2)
    ) * mats.below_sub_c
    m_inter = torch.exp(log_m_inter + mats.strict_causal_c).to(dtype)
    g_t = torch.matmul(m_inter, u_chunk_t.flatten(-2)).view(
        batch_size, num_heads, num_chunks, head_dim, d_state
    )
    y_inter = torch.matmul(c_c, g_t.transpose(-1, -2)) * m_first.unsqueeze(-1)
    y_out = y_out + y_inter

  y_out = y_out + d_skip.to(dtype).view(1, num_heads, 1, 1, 1) * x_c
  if z_c is not None:
    y_out = y_out * F.silu(z_c.to(dtype))
  return (
      y_out.permute(0, 2, 3, 1, 4)
      .contiguous()
      .view(batch_size, num_chunks * q_len, num_heads, head_dim)
  )


def _mamba3_scan_mimo(
    x: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    adt_c: torch.Tensor,
    dt_c: torch.Tensor,
    trap_c: torch.Tensor,
    d_skip: torch.Tensor,
    z: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_z: torch.Tensor,
    mimo_o: torch.Tensor,
    cache: _ConstantCache | None = None,
) -> torch.Tensor:
  """Runs the chunked MIMO Mamba-3 scan.

  Uses the weights W of _mamba3_scan_prechunked, computed with _segsum_exp,
  and contracts the rank-R inputs x_r = x * mimo_x against rank-R B and C.

  Args:
    x: Input values of shape (B, L, H, P).
    b_proj: B projection of shape (B, L, R, H, D).
    c_proj: C projection of shape (B, L, R, H, D).
    adt_c: Float32 log-decay of shape (B, H, C, Q), zero-padded to C * Q >= L.
    dt_c: Float32 time step of shape (B, H, C, Q), zero-padded.
    trap_c: Float32 trapezoidal gate of shape (B, H, C, Q), zero-padded.
    d_skip: Skip weight of shape (H,).
    z: Gate of shape (B, L, H, P).
    mimo_x: Rank expansion for x of shape (H, R, P).
    mimo_z: Rank expansion for z of shape (H, R, P).
    mimo_o: Rank contraction of shape (H, R, P).
    cache: Optional dictionary for caching constant helper matrices in eager
      mode.

  Returns:
    Output of shape (B, L, H, P).
  """
  batch_size, seq_len, num_heads, head_dim = x.shape
  mimo_rank = b_proj.shape[2]
  d_state = b_proj.shape[-1]
  num_chunks, q_len = adt_c.shape[-2:]
  pad_len = num_chunks * q_len - seq_len
  dtype = x.dtype

  # W[t, tau] = M[t, tau] alpha_tau + M[t, tau + 1] beta_{tau + 1}.
  beta_c = dt_c * (0.5 * trap_c)
  alpha_c = dt_c - beta_c
  log_m, m = _segsum_exp(adt_c, cache)
  w = m * alpha_c.unsqueeze(-2)
  if q_len > 1:
    w = w + F.pad(m[..., 1:] * beta_c[..., 1:].unsqueeze(-2), (0, 1))

  x_r = torch.einsum("blhp,hrp->blrhp", x, mimo_x.to(dtype))
  chunk_shape = (batch_size, num_chunks, q_len, mimo_rank, num_heads)
  x_r_c = _pad_seq(x_r, pad_len).view(*chunk_shape, head_dim)
  b_c = _pad_seq(b_proj.to(dtype), pad_len).view(*chunk_shape, d_state)
  c_c = _pad_seq(c_proj.to(dtype), pad_len).view(*chunk_shape, d_state)

  cb = torch.einsum("bctrhd,bcsqhd->bhctrsq", c_c, b_c)
  attn = (cb.float() * w.unsqueeze(4).unsqueeze(6)).to(dtype)
  y_out = torch.einsum("bhctrsq,bcsqhp->bctrhp", attn, x_r_c)

  if num_chunks > 1:
    # W[f(c' + 1), s] = exp(adt[f(c' + 1)]) W[Q - 1, s] (+ beta[f(c' + 1)] for
    # s = Q - 1), for chunks c' = 0, ..., C - 2.
    w_last_src = w[:, :, :-1, -1, :]  # (B, H, C - 1, Q)
    next_decay_0 = torch.exp(adt_c[:, :, 1:, 0]).unsqueeze(-1)
    next_beta_0 = beta_c[:, :, 1:, 0].unsqueeze(-1)
    w_tilde_t = (
        (w_last_src * next_decay_0 + F.pad(next_beta_0, (q_len - 1, 0)))
        .permute(0, 2, 3, 1)
        .to(dtype)
    )  # (B, C - 1, Q, H)
    m_col0 = m[:, :, 1:, :, 0].permute(0, 2, 3, 1).to(dtype)

    b_weighted = b_c[:, :-1] * w_tilde_t.unsqueeze(3).unsqueeze(-1)
    u_chunk = torch.einsum("bcsrhp,bcsrhd->bhcpd", x_r_c[:, :-1], b_weighted)
    if num_chunks > 2:
      # log_gamma[c'] = log M[f(c' + 1), f(c')].
      log_gamma = log_m[:, :, :-1, -1, 0] + adt_c[:, :, 1:, 0]
      _, m_inter = _segsum_exp(log_gamma, cache)  # (B, H, C - 1, C - 1)
      h_tilde = torch.matmul(m_inter.to(dtype), u_chunk.flatten(-2)).view(
          batch_size, num_heads, num_chunks - 1, head_dim, d_state
      )
    else:
      h_tilde = u_chunk
    c_scaled = c_c[:, 1:] * m_col0.unsqueeze(3).unsqueeze(-1)
    y_inter = torch.einsum("bctrhd,bhcpd->bctrhp", c_scaled, h_tilde)
    y_out = torch.cat([y_out[:, :1], y_out[:, 1:] + y_inter], dim=1)

  y_r = y_out.reshape(
      batch_size, num_chunks * q_len, mimo_rank, num_heads, head_dim
  )[:, :seq_len]
  skip = d_skip.to(dtype).view(1, 1, 1, num_heads, 1) * x_r
  z_r = torch.einsum("blhp,hrp->blrhp", z.to(dtype), mimo_z.to(dtype))
  y_r = (y_r.float() + skip.float()) * F.silu(z_r.float())
  return torch.einsum("blrhp,hrp->blhp", y_r.to(dtype), mimo_o.to(dtype))


def mamba3_scan(
    x: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    d_skip: torch.Tensor,
    z: torch.Tensor | None = None,
    mimo_x: torch.Tensor | None = None,
    mimo_z: torch.Tensor | None = None,
    mimo_o: torch.Tensor | None = None,
    is_mimo: bool = False,
    chunk_size: int = _CHUNK_SIZE,
    cache: _ConstantCache | None = None,
) -> torch.Tensor:
  """Executes the Mamba-3 SSM scan recurrence using State Space Duality.

  SSM Continuous-Time State Formulation:
    h'(t) = A * h(t) + B * x(t)
    y(t)  = C * h(t) + D * x(t)

  Discretization with Exponential Decay and Trapezoidal Blending:
    decay_t = exp(A * dt_t)          # Multiplicative state decay.
    (Bx)_t  = B_t * x_t              # Current token state input.
    (Bx)_blended = (1 - trap_t) * (Bx)_t + trap_t * 0.5 * ((Bx)_t + (Bx)_{t-1})
    u_t     = dt_t * (Bx)_blended    # Discretized input injection.
    h_t     = decay_t * h_{t-1} + u_t
    y_t     = (C_t * h_t + D * x_t) * silu(z_t)

  Tensor Dimensions:
    B: batch_size
    L: seq_len
    H: num_heads
    P: head_dim
    D: d_state (SSM latent memory dimension per head)
    R: mimo_rank (number of interacting sub-channels per head)
    Q: chunk length, min(chunk_size, L), or min(max(chunk_size // R, 1), L)
      in MIMO mode
    C: number of chunks, ceil(L / Q)

  Chunked State Space Duality (SSD) Formulation:
    The sequence is zero-padded to C chunks of Q tokens. Within each chunk,
    the outputs are a masked, decay-weighted attention-like contraction
    (C B^T * W) x. The chunk states are carried across chunks through a
    (C, C) lower-triangular decay matrix. All chunks are processed by batched
    matmuls without a sequential loop over tokens or chunks. See
    _mamba3_scan_prechunked for the derivation. Decay weights are computed in
    float32 and the contractions in the dtype of x.

  Args:
    x: Input values tensor of shape (B, L, H, P).
    b_proj: Key projection B tensor of shape (B, L, R, H, D). SISO mode uses
      only rank 0.
    c_proj: Query projection C tensor of shape (B, L, R, H, D). SISO mode uses
      only rank 0.
    adt: Log decay tensor A * dt of shape (B, L, H).
    dt: Time step dt tensor of shape (B, L, H).
    trap: Trapezoidal blending gate tensor of shape (B, L, H) in (0, 1).
    d_skip: Direct skip connection weight of shape (H,).
    z: Output gate tensor of shape (B, L, H, P). Required if is_mimo.
    mimo_x: MIMO rank expansion parameter for x of shape (H, R, P).
    mimo_z: MIMO rank expansion parameter for z of shape (H, R, P).
    mimo_o: MIMO rank contraction parameter of shape (H, R, P).
    is_mimo: Whether to use Multi-Input Multi-Output recurrence.
    chunk_size: Maximum chunk length in SISO mode. MIMO chunks hold at most
      max(chunk_size // R, 1) tokens.
    cache: Optional dictionary for caching constant helper matrices in eager
      mode.

  Returns:
    Output tensor y of shape (B, L, H, P).

  Raises:
    ValueError: If chunk_size is not positive, or if is_mimo is set and any of
      mimo_x, mimo_z, mimo_o, and z is None.
  """
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")
  seq_len = x.shape[1]
  dtype = x.dtype
  max_q_len = chunk_size
  if is_mimo:
    # MIMO chunks have (Q R, Q R) intra-chunk matrices. Chunks of
    # chunk_size // R tokens keep them within the (chunk_size, chunk_size)
    # size of SISO chunks, so that their total memory and FLOPs grow linearly
    # rather than quadratically in R.
    mimo_rank = b_proj.shape[2]
    max_q_len = max(chunk_size // mimo_rank, 1)
  q_len = min(max_q_len, seq_len)
  pad_len = (-seq_len) % q_len
  num_chunks = (seq_len + pad_len) // q_len

  def to_chunks(t: torch.Tensor) -> torch.Tensor:
    # (B, L, H, ...) -> (B, H, C, Q, ...), so that Q is the minor dimension of
    # the per-token scalars adt, dt, and trap.
    return _chunk_heads_first(_pad_seq(t, pad_len), num_chunks, q_len)

  if is_mimo:
    if mimo_x is None or mimo_z is None or mimo_o is None or z is None:
      raise ValueError("is_mimo requires mimo_x, mimo_z, mimo_o, and z")
    return _mamba3_scan_mimo(
        x=x,
        b_proj=b_proj,
        c_proj=c_proj,
        adt_c=to_chunks(adt.float()),
        dt_c=to_chunks(dt.float()),
        trap_c=to_chunks(trap.float()),
        d_skip=d_skip,
        z=z,
        mimo_x=mimo_x,
        mimo_z=mimo_z,
        mimo_o=mimo_o,
        cache=cache,
    )

  y = _mamba3_scan_prechunked(
      x_c=to_chunks(x),
      b_c=to_chunks(b_proj[:, :, 0].to(dtype)),
      c_c=to_chunks(c_proj[:, :, 0].to(dtype)),
      adt_c=to_chunks(adt.float()),
      dt_c=to_chunks(dt.float()),
      trap_c=to_chunks(trap.float()),
      d_skip=d_skip,
      cache=cache,
  )
  if pad_len > 0:
    y = y[:, :seq_len]
  if z is not None:
    y = y * F.silu(z.to(dtype))
  return y


class Mamba3(nn.Module):
  """Mamba-3 sequence mixer layer.

  Implements exponential-trapezoidal discretization, complex-valued / rotary
  state space (RoPE), and Multi-Input Multi-Output (MIMO) SSM recurrence.

  Attributes:
    d_model: Input and output embedding dimension.
    d_state: Latent state space dimension per head (D).
    expand: Hidden dimension expansion factor (d_inner = expand * d_model).
    headdim: Dimension per head (P).
    nheads: Number of attention/SSM heads (H = d_inner // headdim).
    ngroups: Number of key/value (B/C) head groups (grouped-query SSM).
    num_bc_heads: Number of head groups for B and C projections (= ngroups).
    rope_fraction: Fraction of state space coordinates rotated via RoPE.
    split_tensor_size: Number of coordinates rotated via RoPE.
    num_rope_angles: Number of unique 2D rotation angles (= split_tensor_size //
      2).
    is_mimo: Boolean flag indicating whether MIMO recurrence is active.
    mimo_rank: Rank R for MIMO projection expansions and contractions.
  """

  def __init__(
      self,
      d_model: int,
      d_state: int = 128,
      expand: int = 2,
      headdim: int = 64,
      ngroups: int = 1,
      rope_fraction: float = 0.5,
      dt_min: float = 0.001,
      dt_max: float = 0.1,
      dt_init_floor: float = 1e-4,
      a_floor: float = 1e-4,
      is_mimo: bool = False,
      mimo_rank: int = 4,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Mamba-3 layer.

    Args:
      d_model: Input and output dimension of token embeddings.
      d_state: SSM latent state dimension per head (D).
      expand: Expansion ratio from d_model to d_inner.
      headdim: Dimension of each attention/SSM head (P).
      ngroups: Number of B and C projection head groups (grouped-query SSM).
      rope_fraction: Fraction of d_state coordinates rotated via RoPE (0.5 or
        1.0).
      dt_min: Minimum initial step size dt.
      dt_max: Maximum initial step size dt.
      dt_init_floor: Minimum floor for step size initialization.
      a_floor: Minimum magnitude floor for the negative continuous decay rate A.
      is_mimo: Whether to enable Multi-Input Multi-Output recurrence.
      mimo_rank: Rank dimension R for MIMO state projections.
      device: Optional torch device.
      dtype: Optional torch dtype for weights.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.expand = expand
    self.headdim = headdim
    self.a_floor = a_floor
    self.is_mimo = is_mimo
    self.mimo_rank = mimo_rank if is_mimo else 1
    self.num_bc_heads = ngroups
    # Constant helper matrices reused across eager calls (see _constant_cache).
    self._eager_cache: _ConstantCache = {}

    # Compute inner dimension and verify divisibility by head dimension.
    self.d_inner = int(expand * d_model)
    if self.d_inner % headdim != 0:
      raise ValueError("d_inner must be divisible by headdim")
    self.nheads = self.d_inner // headdim

    # Ensure head count is divisible by the number of B/C groups
    # (grouped-query SSM).
    if self.nheads % self.num_bc_heads != 0:
      raise ValueError("nheads must be divisible by ngroups")

    # Validate RoPE fraction and calculate coordinate split sizes.
    if rope_fraction not in (0.5, 1.0):
      raise ValueError("rope_fraction must be 0.5 or 1.0")
    self.split_tensor_size = int(d_state * rope_fraction)
    # Ensure split size is even so coordinates can be paired for 2D rotation.
    if self.split_tensor_size % 2 != 0:
      self.split_tensor_size -= 1
    self.num_rope_angles = self.split_tensor_size // 2

    # -------------------------------------------------------------------------
    # Input projections:
    # Separate projections map d_model directly to each branch, so that no
    # branch is a slice of a fused projection output along the feature
    # dimension:
    # - z_proj (gate branch): d_inner
    # - x_proj (SSM input branch): d_inner
    # - b_proj (Key): d_state * ngroups * mimo_rank
    # - c_proj (Query): d_state * ngroups * mimo_rank
    # - dt_proj (dt delta), a_proj (A parameter delta), trap_proj (trapezoidal
    #   blending gate): nheads each
    # - angle_proj (RoPE angular frequencies): num_rope_angles
    # -------------------------------------------------------------------------
    bc_size = d_state * ngroups * self.mimo_rank
    d_in_proj = self.d_inner * 2 + bc_size * 2 + self.nheads * 3 + self.num_rope_angles
    self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

    # -------------------------------------------------------------------------
    # Time step dt bias initialization (Inverse Softplus):
    # During forward pass, dt = softplus(dd_dt + dt_bias).
    # To initialize dt to dt_init ~ Exp(Uniform(log(dt_min), log(dt_max))),
    # we solve softplus(dt_bias) = dt_init:
    #   dt_init = log(1 + exp(dt_bias))
    #   exp(dt_bias) = exp(dt_init) - 1
    #   dt_bias = log(exp(dt_init) - 1)
    #           = dt_init + log(1 - exp(-dt_init))
    #           = dt_init + log(-expm1(-dt_init))
    # This stable form prevents floating-point overflow for large dt_init.
    # -------------------------------------------------------------------------
    dt_init = torch.exp(
        torch.rand(self.nheads, dtype=torch.float32)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    dt_bias_init = dt_init + torch.log(-torch.expm1(-dt_init))
    self.dt_bias = nn.Parameter(dt_bias_init)

    # State projection biases for B and C of shape (mimo_rank, nheads,
    # d_state), added per head after RMSNorm and the group-to-head broadcast.
    self.b_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32))
    self.c_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32))

    # RMSNorm applied to B and C prior to head expansion to stabilize state
    # norms.
    self.b_norm = RMSNorm(d_state)
    self.c_norm = RMSNorm(d_state)

    # MIMO projection parameters (rank expansion and contraction).
    # Scaled by 1.0 / mimo_rank to preserve activation variance across ranks.
    if self.is_mimo:
      self.mimo_x = nn.Parameter(
          torch.ones(
              self.nheads, self.mimo_rank, self.headdim, **factory_kwargs
          )
          / self.mimo_rank
      )
      self.mimo_z = nn.Parameter(
          torch.ones(
              self.nheads, self.mimo_rank, self.headdim, **factory_kwargs
          )
      )
      self.mimo_o = nn.Parameter(
          torch.ones(
              self.nheads, self.mimo_rank, self.headdim, **factory_kwargs
          )
          / self.mimo_rank
      )
    else:
      self.mimo_x = None
      self.mimo_z = None
      self.mimo_o = None

    # D skip connection weight per head (continuous feedthrough matrix D).
    self.d = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))

    # Final linear projection from inner dimension back to d_model.
    self.out_proj = nn.Linear(
        self.d_inner, d_model, bias=False, **factory_kwargs
    )

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    """Runs forward pass of Mamba-3 sequence mixer.

    Args:
      u: Input tensor of shape (batch, seq_len, d_model).

    Returns:
      Output tensor of shape (batch, seq_len, d_model).
    """
    seq_len = u.shape[1]
    cache = self._constant_cache(u)
    q_len = min(_CHUNK_SIZE, seq_len)
    if not self.is_mimo and seq_len % q_len == 0:
      return self._forward_chunked_siso(u, q_len, cache)
    return self._forward_general(u, cache)

  def _constant_cache(self, u: torch.Tensor) -> _ConstantCache | None:
    """Returns the cache for constant helper matrices, or None to not cache.

    Only plain eager tensors use the cache: tensors created while tracing
    (torch.compile, torch.jit) or from tensor subclasses such as TorchAX
    tensors, which may hold JAX tracers, must not outlive the call.

    Args:
      u: Input tensor of the layer.
    """
    if (
        type(u) is not torch.Tensor  # pylint: disable=unidiomatic-typecheck
        or torch.compiler.is_compiling()
        or torch.jit.is_tracing()
    ):
      return None
    return self._eager_cache

  def _forward_chunked_siso(
      self, u: torch.Tensor, q_len: int, cache: _ConstantCache | None
  ) -> torch.Tensor:
    """SISO forward pass in the chunked layout of _mamba3_scan_prechunked.

    Computes the same function as _forward_general for seq_len % q_len == 0.
    The projections are permuted once into the (B, H or G, C, Q, ...) layout
    of the scan, so RMSNorm, RoPE, and the scan run without intermediate
    transposes.

    Args:
      u: Input tensor of shape (batch, seq_len, d_model).
      q_len: Chunk length Q, a divisor of seq_len.
      cache: Optional dictionary for caching constant helper matrices.

    Returns:
      Output tensor of shape (batch, seq_len, d_model).
    """
    batch_size, seq_len, _ = u.shape
    num_chunks = seq_len // q_len
    u_2d = u.reshape(batch_size * seq_len, self.d_model)

    def project(linear: nn.Linear, *head_shape: int) -> torch.Tensor:
      # (B * L, d_model) -> (B, C, Q, H or G, ...) -> (B, H or G, C, Q, ...).
      t = linear(u_2d).view(batch_size, num_chunks, q_len, *head_shape)
      return t.movedim(3, 1).contiguous()

    # Linear input projections. Measured on TPU v7x (compiled, batch 4,
    # seq_len 1024): in eval, gating with z in the chunked layout inside the
    # scan, before its output permute, saved ~170 us/step by keeping XLA from
    # fusing that permute into z_proj. In training, the same change added
    # ~150 us/step of reshapes, so there z stays (B, L, H, P).
    proj_2d = self.in_proj(u_2d)
    z_raw, x_raw, b_raw, c_raw, dt_raw, a_raw, trap_raw, angle_raw = torch.split(proj_2d, [self.d_inner, self.d_inner, self.d_state * self.num_bc_heads * self.mimo_rank, self.d_state * self.num_bc_heads * self.mimo_rank, self.nheads, self.nheads, self.nheads, self.num_rope_angles], dim=-1)

    def project_raw(tensor, *head_shape):
      t = tensor.view(batch_size, num_chunks, q_len, *head_shape)
      return t.movedim(3, 1).contiguous()

    z = z_raw.view(batch_size, seq_len, self.nheads, self.headdim) if torch.is_grad_enabled() else None
    z_c = None if torch.is_grad_enabled() else project_raw(z_raw, self.nheads, self.headdim)
    x_c = project_raw(x_raw, self.nheads, self.headdim)
    b_raw_c = project_raw(b_raw, self.num_bc_heads, self.d_state)
    c_raw_c = project_raw(c_raw, self.num_bc_heads, self.d_state)
    dd_dt_c = project_raw(dt_raw, self.nheads).float()
    dd_a_c = project_raw(a_raw, self.nheads).float()
    trap_raw_c = project_raw(trap_raw, self.nheads).float()
    # The RoPE frequencies are shared by all heads: (B, 1, C, Q, N).
    angle_c = (angle_raw.view(batch_size, 1, num_chunks, q_len, self.num_rope_angles)
        .float()
    )

    # Parameter activations, as in _forward_general.
    a_c = -(F.softplus(dd_a_c).clamp(min=self.a_floor))
    dt_c = F.softplus(dd_dt_c + self.dt_bias.view(1, self.nheads, 1, 1))
    adt_c = a_c * dt_c
    trap_c = torch.sigmoid(trap_raw_c)

    # RoPE angles theta = cumsum(dt * omega), accumulated in float32, expanded
    # to per-coordinate cos/sin of shape (B, H, C, Q, D).
    rope = _get_rope_mats(
        self.d_state, self.num_rope_angles, u.device, u.dtype, cache
    )
    angles = _rope_angles_chunked(dt_c.unsqueeze(-1) * angle_c, cache)
    cos_full = torch.matmul(torch.cos(angles).to(u.dtype), rope.expand)
    if rope.unrot_mask is not None:
      cos_full = cos_full + rope.unrot_mask
    sin_full = torch.matmul(torch.sin(angles).to(u.dtype), rope.expand)

    # RoPE is linear, so the normalized projections and the per-head biases
    # are rotated separately. Rotate B and C at G = num_bc_heads groups and
    # evaluate the RoPE linear combination in 6D before collapsing
    # (G, repeat_factor) -> H so XLA fuses the G -> H broadcasts directly into
    # the RoPE loop fusion without standalone broadcast/reduce kernels.
    b_normed = self.b_norm(b_raw_c)
    c_normed = self.c_norm(c_raw_c)
    b_normed_rot = torch.matmul(b_normed, rope.perm)
    c_normed_rot = torch.matmul(c_normed, rope.perm)
    b_bias_d = self.b_bias.to(b_normed.dtype)
    c_bias_d = self.c_bias.to(c_normed.dtype)
    b_bias_rot = torch.matmul(b_bias_d, rope.perm)
    c_bias_rot = torch.matmul(c_bias_d, rope.perm)

    if self.num_bc_heads in (1, self.nheads):
      bias_shape_1 = (1, self.nheads, 1, 1, self.d_state)
      b_bias_d = b_bias_d.transpose(0, 1)
      c_bias_d = c_bias_d.transpose(0, 1)
      b_bias_rot = b_bias_rot.transpose(0, 1)
      c_bias_rot = c_bias_rot.transpose(0, 1)
      b_c = (b_normed + b_bias_d.view(bias_shape_1)) * cos_full + (
          b_normed_rot + b_bias_rot.view(bias_shape_1)
      ) * sin_full
      c_c = (c_normed + c_bias_d.view(bias_shape_1)) * cos_full + (
          c_normed_rot + c_bias_rot.view(bias_shape_1)
      ) * sin_full
    else:
      # Head h uses group h // repeat_factor, as with repeat_interleave.
      repeat_factor = self.nheads // self.num_bc_heads
      bias_shape_g = (
          1,
          self.num_bc_heads,
          repeat_factor,
          1,
          1,
          self.d_state,
      )
      group_shape = (
          batch_size,
          self.num_bc_heads,
          repeat_factor,
          num_chunks,
          q_len,
          self.d_state,
      )
      out_shape = (batch_size, self.nheads, num_chunks, q_len, self.d_state)
      cos_g = cos_full.view(group_shape)
      sin_g = sin_full.view(group_shape)
      b_c = (
          (b_normed.unsqueeze(2) + b_bias_d.view(bias_shape_g)) * cos_g
          + (b_normed_rot.unsqueeze(2) + b_bias_rot.view(bias_shape_g)) * sin_g
      ).view(out_shape)
      c_c = (
          (c_normed.unsqueeze(2) + c_bias_d.view(bias_shape_g)) * cos_g
          + (c_normed_rot.unsqueeze(2) + c_bias_rot.view(bias_shape_g)) * sin_g
      ).view(out_shape)

    # Core SSM scan with skip connection and gating, then output projection.
    y = _mamba3_scan_prechunked(
        x_c=x_c,
        b_c=b_c,
        c_c=c_c,
        adt_c=adt_c,
        dt_c=dt_c,
        trap_c=trap_c,
        d_skip=self.d,
        z_c=z_c,
        cache=cache,
    )
    if z is not None:
      y = y * F.silu(z.to(y.dtype))
    y_2d = y.view(batch_size * seq_len, self.d_inner)
    return self.out_proj(y_2d.to(u.dtype)).view(
        batch_size, seq_len, self.d_model
    )

  def _forward_general(
      self, u: torch.Tensor, cache: _ConstantCache | None
  ) -> torch.Tensor:
    """Forward pass for MIMO, or for seq_len % min(128, seq_len) != 0.

    Args:
      u: Input tensor of shape (batch, seq_len, d_model).
      cache: Optional dictionary for caching constant helper matrices.

    Returns:
      Output tensor of shape (batch, seq_len, d_model).
    """
    # -------------------------------------------------------------------------
    # Step 1: Linear input projections into all parallel branches: gate z, SSM
    # input x, B, C, dt, A, trap, and RoPE angles.
    # -------------------------------------------------------------------------
    batch_size, seq_len, _ = u.shape
    # Reshape input and gate tensors to multi-head layout: (B, L, H, P).
    proj_u = self.in_proj(u)
    z_raw, x_raw, b_raw, c_raw, dt_raw, a_raw, trap_raw, angle_raw = torch.split(proj_u, [self.d_inner, self.d_inner, self.d_state * self.num_bc_heads * self.mimo_rank, self.d_state * self.num_bc_heads * self.mimo_rank, self.nheads, self.nheads, self.nheads, self.num_rope_angles], dim=-1)
    z = z_raw.view(batch_size, seq_len, self.nheads, self.headdim)
    x = x_raw.view(batch_size, seq_len, self.nheads, self.headdim)
    b_raw = b_raw.view(batch_size, seq_len, self.num_bc_heads, self.d_state)
    c_raw = c_raw.view(batch_size, seq_len, self.num_bc_heads, self.d_state)
    dd_dt = dt_raw
    dd_a = a_raw
    trap_raw = trap_raw.float()

    # -------------------------------------------------------------------------
    # Step 2: Parameter activations (decay rate A, step size dt, trap gate).
    # - Decay rate A = -max(softplus(dd_a), a_floor) <= -a_floor.
    # - Time step dt is strictly positive: softplus(dd_dt + dt_bias).
    # - Log decay factor adt = A * dt.
    # - Trapezoidal blending gate trap is squashed to (0, 1) via sigmoid.
    # -------------------------------------------------------------------------
    a = -(F.softplus(dd_a).clamp(min=self.a_floor))
    dt = F.softplus(dd_dt + self.dt_bias)
    adt = a * dt
    trap = torch.sigmoid(trap_raw)

    # -------------------------------------------------------------------------
    # Step 3: Normalize B and C projections and broadcast across head groups.
    # Applies RMSNorm per group to stabilize state magnitudes, then broadcasts
    # group heads G to total heads H while adding the per-head biases. Head h
    # uses group h // repeat_factor, as with repeat_interleave.
    # -------------------------------------------------------------------------
    repeat_factor = self.nheads // self.num_bc_heads
    head_shape = (
        batch_size,
        seq_len,
        self.mimo_rank,
        self.nheads,
        self.d_state,
    )

    def add_head_bias(normed: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
      # (B, L, R, G, 1, D) + (R, G, repeat_factor, D) -> (B, L, R, H, D).
      bias = bias.to(normed.dtype).view(self.nheads, self.mimo_rank, self.d_state).transpose(0, 1).view(self.mimo_rank, self.num_bc_heads, repeat_factor, self.d_state)
      return (normed.unsqueeze(-2) + bias).view(head_shape)

    b_exp = add_head_bias(self.b_norm(b_raw), self.b_bias)
    c_exp = add_head_bias(self.c_norm(c_raw), self.c_bias)

    # -------------------------------------------------------------------------
    # Step 4: Apply continuous-time Rotary Position Embedding (RoPE) to state.
    # In continuous SSMs, time step dt varies per token. Angular increments
    # dt * omega are accumulated along sequence dimension using parallel prefix
    # sum in float32: theta_t = sum_{s=0}^t dt_s * omega.
    # -------------------------------------------------------------------------
    # Angular increments: (B, L, H, num_rope_angles).
    increments = angle_raw.unsqueeze(2) * dt.unsqueeze(-1)
    angles = _rope_angles_dim1(increments, cache)
    rope = _get_rope_mats(
        self.d_state, self.num_rope_angles, u.device, b_exp.dtype, cache
    )
    cos_full = torch.matmul(torch.cos(angles).to(b_exp.dtype), rope.expand)
    if rope.unrot_mask is not None:
      cos_full = cos_full + rope.unrot_mask
    sin_full = torch.matmul(torch.sin(angles).to(b_exp.dtype), rope.expand)
    # Share the angles across MIMO ranks: (B, L, 1, H, D).
    cos_full = cos_full.unsqueeze(2)
    sin_full = sin_full.unsqueeze(2)
    b_proj = b_exp * cos_full + torch.matmul(b_exp, rope.perm) * sin_full
    c_proj = c_exp * cos_full + torch.matmul(c_exp, rope.perm) * sin_full

    # -------------------------------------------------------------------------
    # Step 5: Core SSM scan recurrence.
    # Executes the exponential-trapezoidal discretization recurrence, applies
    # skip connection D * x, and gates with silu(z).
    # -------------------------------------------------------------------------
    y = mamba3_scan(
        x=x,
        b_proj=b_proj,
        c_proj=c_proj,
        adt=adt,
        dt=dt,
        trap=trap,
        d_skip=self.d,
        z=z,
        mimo_x=self.mimo_x,
        mimo_z=self.mimo_z,
        mimo_o=self.mimo_o,
        is_mimo=self.is_mimo,
        cache=cache,
    )

    # -------------------------------------------------------------------------
    # Step 6: Output linear projection.
    # Flatten multi-head layout (B, L, H, P) back to (B, L, d_inner) and
    # project back to token dimension (B, L, d_model).
    # -------------------------------------------------------------------------
    y = y.reshape(batch_size, seq_len, self.d_inner)
    return self.out_proj(y.to(u.dtype))
