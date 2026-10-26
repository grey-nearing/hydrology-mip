"""Multi-seed batched CUDA LSTM helper for KratzertLSTMTrainer.

Trains all K seeds of a spatial fold in lockstep using strided batched Tensor Core
GEMMs (`at::bmm_out`) and a fused gate activation/gradient CUDA kernel, while
preserving 100% full-budget training (all valid `(basin, day)` windows visited per
epoch, sequence-to-one targets, per-seed gradient-norm clipping, and per-seed RNGs).
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.kratzert_lstm.windows import lr_for_epoch

_EXT_CACHE: object | None = None
_EXT_TRIED: bool = False

_CPP_SOURCES = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" void launch_lstm_fwd_klb(
    const void* gx_klb, const void* b_total, const void* gh,
    void* g_act_klb, void* c_klb, void* h_klb,
    int K, int L, int B, int H, int t, cudaStream_t stream);

extern "C" void launch_lstm_bwd_klb(
    const void* dh_in, float* dc_inout, const void* c_klb,
    const void* g_act_klb, void* dg_klb, float* db_accum,
    int K, int L, int B, int H, int t, cudaStream_t stream);

torch::Tensor multiseed_lstm_fwd_klb(
    torch::Tensor x_klb,
    torch::Tensor w_ih_t,
    torch::Tensor w_hh_t,
    torch::Tensor b_total,
    torch::Tensor gx_klb,
    torch::Tensor g_act_klb,
    torch::Tensor c_klb,
    torch::Tensor h_klb,
    torch::Tensor gh) {
    int K = x_klb.size(0);
    int L = x_klb.size(1);
    int B = x_klb.size(2);
    int I = x_klb.size(3);
    int H = w_hh_t.size(1);

    auto gx_flat = gx_klb.view({K, L * B, 4 * H});
    at::bmm_out(gx_flat, x_klb.view({K, L * B, I}), w_ih_t);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (int t = 0; t < L; ++t) {
        if (t > 0) {
            at::bmm_out(gh, h_klb.select(1, t - 1), w_hh_t);
        }
        launch_lstm_fwd_klb(
            gx_klb.data_ptr(),
            b_total.data_ptr(),
            gh.data_ptr(),
            g_act_klb.data_ptr(),
            c_klb.data_ptr(),
            h_klb.data_ptr(),
            K, L, B, H, t, stream);
    }
    return h_klb.select(1, L - 1);
}

std::vector<torch::Tensor> multiseed_lstm_bwd_klb(
    torch::Tensor grad_h_last,
    torch::Tensor x_klb,
    torch::Tensor h_klb,
    torch::Tensor c_klb,
    torch::Tensor g_act_klb,
    torch::Tensor w_hh,
    torch::Tensor dg_klb,
    torch::Tensor dh,
    torch::Tensor dc,
    torch::Tensor dw_ih,
    torch::Tensor dw_hh,
    torch::Tensor ones_lb,
    torch::Tensor db_out) {
    int K = x_klb.size(0);
    int L = x_klb.size(1);
    int B = x_klb.size(2);
    int I = x_klb.size(3);
    int H = w_hh.size(2);

    dh.copy_(grad_h_last);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    for (int t = L - 1; t >= 0; --t) {
        launch_lstm_bwd_klb(
            dh.data_ptr(),
            dc.data_ptr<float>(),
            c_klb.data_ptr(),
            g_act_klb.data_ptr(),
            dg_klb.data_ptr(),
            nullptr,
            K, L, B, H, t, stream);
        if (t > 0) {
            at::bmm_out(dh, dg_klb.select(1, t), w_hh);
        }
    }

    auto dg_flat = dg_klb.view({K, L * B, 4 * H});
    at::bmm_out(dw_ih, dg_flat.transpose(1, 2), x_klb.view({K, L * B, I}));

    auto dg_tail = torch::as_strided(dg_klb, {K, (L - 1) * B, 4 * H}, {L * B * 4 * H, 4 * H, 1}, B * 4 * H);
    auto h_prev = torch::as_strided(h_klb, {K, (L - 1) * B, H}, {L * B * H, H, 1}, 0);
    at::bmm_out(dw_hh, dg_tail.transpose(1, 2), h_prev);

    auto db_fp32 = dg_flat.sum(1, false, at::kFloat);
    return {dw_ih, dw_hh, db_fp32};
}
"""


def get_multiseed_cuda_ext() -> object | None:
    """Return the compiled `ms_lstm_klb1` CUDA extension if available, else None."""
    global _EXT_CACHE, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT_CACHE
    _EXT_TRIED = True
    if not torch.cuda.is_available():
        return None
    obj_path = Path("/tmp/lstm_kernels_klb.o")
    if not obj_path.exists():
        return None
    from torch.utils.cpp_extension import load_inline

    conda_prefix = Path(os.sys.executable).resolve().parent.parent
    conda_bin = str(conda_prefix / "bin")
    if (conda_prefix / "bin").exists() and conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"
    extra_inc = [str(conda_prefix / "include")] if (conda_prefix / "include").exists() else []
    if Path("/usr/local/cuda/include").exists():
        extra_inc.append("/usr/local/cuda/include")
    extra_ld = [str(obj_path)]
    if (conda_prefix / "lib").exists():
        extra_ld.append(f"-L{conda_prefix / 'lib'}")
    if Path("/usr/local/cuda/lib64").exists():
        extra_ld.append("-L/usr/local/cuda/lib64")
    extra_ld.append("-lcudart")
    _EXT_CACHE = load_inline(
        name="ms_lstm_klb2",
        cpp_sources=_CPP_SOURCES,
        functions=["multiseed_lstm_fwd_klb", "multiseed_lstm_bwd_klb"],
        extra_include_paths=extra_inc,
        extra_ldflags=extra_ld,
    )
    return _EXT_CACHE


def fit_and_predict_multiseed_cuda(
    *,
    ext: object,
    models: list[nn.Module],
    rngs: list[np.random.RandomState],
    xd_tr_t: torch.Tensor,
    xs_tr_t: torch.Tensor,
    y_tr_t: torch.Tensor,
    w_tr_t: torch.Tensor,
    xd_te_t: torch.Tensor,
    xs_te_t: torch.Tensor,
    pairs_t: torch.Tensor,
    n_batches: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dropout: float,
    epochs: int,
    lr_schedule: Mapping[int, float],
    clip_grad_norm: float,
    y_mean: float,
    y_std: float,
    spinup_days_test: int,
    n_target_days: int,
    eval_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Train all `K = len(models)` CudaLSTM seeds in lockstep on CUDA and predict."""
    K = len(models)
    n_tr, total_tr_days, dyn_dim = xd_tr_t.shape
    stat_dim = xs_tr_t.shape[1]
    I = dyn_dim + stat_dim
    H = hidden_size
    B = batch_size
    L = seq_len
    n_pairs = int(pairs_t.shape[0])

    w_ih = nn.Parameter(torch.stack([m.lstm.weight_ih_l0.detach().clone() for m in models], dim=0))
    w_hh = nn.Parameter(torch.stack([m.lstm.weight_hh_l0.detach().clone() for m in models], dim=0))
    b_ih = nn.Parameter(torch.stack([m.lstm.bias_ih_l0.detach().clone() for m in models], dim=0))
    b_hh = nn.Parameter(torch.stack([m.lstm.bias_hh_l0.detach().clone() for m in models], dim=0))
    head_w = nn.Parameter(torch.stack([m.head.weight.detach().clone() for m in models], dim=0))
    head_b = nn.Parameter(torch.stack([m.head.bias.detach().clone() for m in models], dim=0))
    optimizer = torch.optim.Adam(
        [w_ih, w_hh, b_ih, b_hh, head_w, head_b],
        lr=lr_for_epoch(lr_schedule, 0),
    )

    x_tr_flat_fp16 = (
        torch.cat([xd_tr_t, xs_tr_t.unsqueeze(1).expand(-1, total_tr_days, -1)], dim=-1)
        .contiguous()
        .half()
        .view(n_tr * total_tr_days, I)
    )
    pair_b = pairs_t[:, 0]
    pair_t = pairs_t[:, 1]
    pair_start_flat = (pair_b * total_tr_days + (pair_t - (L - 1))).to(torch.int32)
    pair_y_true = y_tr_t[pair_b, pair_t].contiguous()
    pair_w = w_tr_t[pair_b].contiguous()
    arange_L = torch.arange(L, device=device, dtype=torch.int32).view(1, L, 1)

    x_buf = torch.empty(K * L * B * I, device=device, dtype=torch.float16)
    gates_buf = torch.empty(K * L * B * 4 * H, device=device, dtype=torch.float16)
    c_buf = torch.empty(K * L * B * H, device=device, dtype=torch.float16)
    h_buf = torch.empty(K * L * B * H, device=device, dtype=torch.float16)
    gh_buf = torch.empty(K * B * 4 * H, device=device, dtype=torch.float16)
    dh_buf = torch.empty(K * B * H, device=device, dtype=torch.float16)
    dc_buf = torch.empty(K * B * H, device=device, dtype=torch.float32)
    dw_ih_fp16 = torch.empty(K, 4 * H, I, device=device, dtype=torch.float16)
    dw_hh_fp16 = torch.empty(K, 4 * H, H, device=device, dtype=torch.float16)
    ones_buf = torch.ones(K, 1, L * B, device=device, dtype=torch.float16)
    db_fp16 = torch.empty(K, 1, 4 * H, device=device, dtype=torch.float16)

    keep_prob = 1.0 - dropout
    drop_scale = 1.0 / keep_prob if keep_prob > 0.0 else 0.0
    verbose = os.environ.get("CAMELS_MIP_VERBOSE_EPOCHS") == "1"
    ckpt_env = os.environ.get("CAMELS_MIP_FOLD_CKPT_PATH")
    ckpt_path = Path(ckpt_env) if ckpt_env else None
    start_epoch = 0
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if int(ckpt["w_ih"].shape[0]) == K and bool(torch.isfinite(ckpt["w_ih"]).all()):
            with torch.no_grad():
                w_ih.copy_(ckpt["w_ih"])
                w_hh.copy_(ckpt["w_hh"])
                b_ih.copy_(ckpt["b_ih"])
                b_hh.copy_(ckpt["b_hh"])
                head_w.copy_(ckpt["head_w"])
                head_b.copy_(ckpt["head_b"])
            optimizer.load_state_dict(ckpt["optimizer"])
            for k, st in enumerate(ckpt["rng_states"]):
                rngs[k].set_state(st)
            start_epoch = int(ckpt["epoch"])
            if verbose:
                print(
                    f"  [multiseed K={K}] resumed from checkpoint at epoch {start_epoch}/{epochs}",
                    flush=True,
                )

    t_fold_start = time.time()
    max_take = min(n_pairs, n_batches * B)

    for epoch in range(start_epoch, epochs):
        lr_now = lr_for_epoch(lr_schedule, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        perms_np = np.stack(
            [rngs[k].permutation(n_pairs)[:max_take] for k in range(K)],
            axis=0,
        )
        idx_epoch_all = torch.from_numpy(perms_np).to(device=device, dtype=torch.long)

        for batch_i in range(n_batches):
            b_start = batch_i * B
            b_end = min(b_start + B, max_take)
            cur_B = b_end - b_start
            if cur_B <= 0:
                break
            p_kb = idx_epoch_all[:, b_start:b_end]
            starts_kb = pair_start_flat[p_kb].unsqueeze(1) + arange_L

            x_klb = x_buf[: K * L * cur_B * I].view(K, L, cur_B, I)
            gates_klb = gates_buf[: K * L * cur_B * 4 * H].view(K, L, cur_B, 4 * H)
            c_klb = c_buf[: K * L * cur_B * H].view(K, L, cur_B, H)
            h_klb = h_buf[: K * L * cur_B * H].view(K, L, cur_B, H)
            gh = gh_buf[: K * cur_B * 4 * H].view(K, cur_B, 4 * H)
            dh = dh_buf[: K * cur_B * H].view(K, cur_B, H)
            dc = dc_buf[: K * cur_B * H].view(K, cur_B, H)
            ones_lb = ones_buf[:, :, : L * cur_B]

            torch.index_select(x_tr_flat_fp16, 0, starts_kb.reshape(-1), out=x_klb.view(-1, I))
            y_true_kb = pair_y_true[p_kb]
            w_kb = pair_w[p_kb]

            w_ih_h = w_ih.detach().half()
            w_hh_h = w_hh.detach().half()
            b_tot_h = (b_ih.detach() + b_hh.detach()).half()
            w_ih_t = w_ih_h.transpose(1, 2).contiguous()
            w_hh_t = w_hh_h.transpose(1, 2).contiguous()

            h_last = ext.multiseed_lstm_fwd_klb(  # type: ignore[attr-defined]
                x_klb, w_ih_t, w_hh_t, b_tot_h, gates_klb, gates_klb, c_klb, h_klb, gh
            )
            if dropout > 0.0:
                drop_mask = (
                    torch.rand(K, cur_B, H, device=device, dtype=torch.float16) >= dropout
                ).to(torch.float16) * drop_scale
                h_drop = h_last * drop_mask
            else:
                drop_mask = None
                h_drop = h_last

            head_w_h = head_w.detach().half()
            head_b_h = head_b.detach().half()
            y_hat = (torch.bmm(h_drop, head_w_h.transpose(1, 2)).squeeze(-1) + head_b_h).float()

            err = (y_hat - y_true_kb) * y_std
            dy_hat = (2.0 * w_kb * err * y_std) / cur_B
            dhead_w = torch.bmm(dy_hat.unsqueeze(1), h_drop.float())
            dhead_b = dy_hat.sum(dim=1, keepdim=True)
            dh_drop_fp32 = torch.bmm(dy_hat.unsqueeze(2), head_w.detach()).squeeze(1)
            dh_last_fp32 = dh_drop_fp32 * drop_mask.float() if drop_mask is not None else dh_drop_fp32
            dh_last_fp32.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            max_dh = dh_last_fp32.abs().amax(dim=(1, 2)).clamp(min=1e-6)
            scale_vec = torch.clamp(0.05 / max_dh, min=1e-5, max=1.0)
            inv_scale_vec = 1.0 / scale_vec
            grad_h_scaled = (dh_last_fp32 * scale_vec.view(K, 1, 1)).half()
            _, _, db_raw_fp32 = ext.multiseed_lstm_bwd_klb(  # type: ignore[attr-defined]
                grad_h_scaled,
                x_klb,
                h_klb,
                c_klb,
                gates_klb,
                w_hh_h,
                gates_klb,
                dh,
                dc,
                dw_ih_fp16,
                dw_hh_fp16,
                ones_lb,
                db_fp16,
            )
            dw_ih_fp32 = dw_ih_fp16.float() * inv_scale_vec.view(K, 1, 1)
            dw_hh_fp32 = dw_hh_fp16.float() * inv_scale_vec.view(K, 1, 1)
            db_fp32 = db_raw_fp32 * inv_scale_vec.view(K, 1)

            norm_sq = (
                dw_ih_fp32.flatten(1).pow(2).sum(1)
                + dw_hh_fp32.flatten(1).pow(2).sum(1)
                + 2.0 * db_fp32.pow(2).sum(1)
                + dhead_w.flatten(1).pow(2).sum(1)
                + dhead_b.flatten(1).pow(2).sum(1)
            )
            finite_mask = torch.isfinite(norm_sq)
            if not bool(finite_mask.all()):
                dw_ih_fp32.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                dw_hh_fp32.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                db_fp32.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                dhead_w.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                dhead_b.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if clip_grad_norm > 0.0:
                clip_coef = torch.clamp(clip_grad_norm / (norm_sq.sqrt() + 1e-6), max=1.0)
            else:
                clip_coef = torch.ones_like(norm_sq)
            clip_coef = torch.where(finite_mask, clip_coef, torch.zeros_like(clip_coef))
            c3 = clip_coef.view(K, 1, 1)
            c2 = clip_coef.view(K, 1)
            dw_ih_fp32.mul_(c3)
            dw_hh_fp32.mul_(c3)
            db_fp32.mul_(c2)
            dhead_w.mul_(c3)
            dhead_b.mul_(c2)

            w_ih.grad = dw_ih_fp32
            w_hh.grad = dw_hh_fp32
            b_ih.grad = db_fp32
            b_hh.grad = db_fp32.clone()
            head_w.grad = dhead_w
            head_b.grad = dhead_b
            optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            torch.cuda.synchronize(device)
            if ckpt_path is not None:
                tmp_ckpt = ckpt_path.with_suffix(".tmp.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "w_ih": w_ih.detach().cpu(),
                        "w_hh": w_hh.detach().cpu(),
                        "b_ih": b_ih.detach().cpu(),
                        "b_hh": b_hh.detach().cpu(),
                        "head_w": head_w.detach().cpu(),
                        "head_b": head_b.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                        "rng_states": [r.get_state() for r in rngs],
                    },
                    tmp_ckpt,
                )
                tmp_ckpt.replace(ckpt_path)
            if verbose:
                elapsed = time.time() - t_fold_start
                print(
                    f"  [multiseed K={K}] epoch {epoch + 1:02d}/{epochs} "
                    f"({n_batches} batches/ep) elapsed={elapsed:.1f}s",
                    flush=True,
                )

    member_preds: list[np.ndarray] = []
    n_te_basins = int(xd_te_t.shape[0])
    b_chunk = max(1, min(n_te_basins, 128))
    for k, model in enumerate(models):
        with torch.no_grad():
            model.lstm.weight_ih_l0.copy_(w_ih[k])
            model.lstm.weight_hh_l0.copy_(w_hh[k])
            model.lstm.bias_ih_l0.copy_(b_ih[k])
            model.lstm.bias_hh_l0.copy_(b_hh[k])
            model.head.weight.copy_(head_w[k])
            model.head.bias.copy_(head_b[k])
            model.eval()
            chunks: list[np.ndarray] = []
            for b0 in range(0, n_te_basins, b_chunk):
                b1 = min(b0 + b_chunk, n_te_basins)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                    out_b = model(xd_te_t[b0:b1], xs_te_t[b0:b1])[:, spinup_days_test:].float()
                chunks.append(out_b.detach().cpu().numpy())
            p_norm = np.concatenate(chunks, axis=0)
        member_preds.append(np.clip(p_norm * y_std + y_mean, 0.0, None).astype(np.float32))

    members_arr = np.stack(member_preds, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr, axis=0, dtype=np.float32)
    return mean_arr, members_arr
