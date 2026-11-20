# Copyright 2026 Google LLC, Grey Nearing, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold runner for Shahriar (2026) Mamba-3 Static-Dynamic Adapter.

Follows the exact `hydroDL` training and inference protocol recorded in
`master.json` of Shahriar (2026, Zenodo `10.5281/zenodo.23087719`):
  - Dynamic inputs (`varT`): `["dayl", "prcp", "srad", "tmean", "vp"]` per CAMELS
    forcing product (`tmean = 0.5 * (tmax + tmin)`, `dayl` from `x_aux[:, :, 0]`).
  - Static attributes: 27 standard CAMELS catchment attributes in Stage-2 MIP runs.
  - Standardization: linear Gaussian z-score `(v - mu) / sigma` on dynamic inputs,
    static attributes, and daily specific discharge `qobs` (`mm/day`), fit strictly
    on the training target period (`[:, spinup_days:]`) of `train_basins`.
  - Sequence sampling: `miniBatch = [128, 365]` with `trainBuff = 365` warm-up days
    (`730` consecutive days per sample; loss evaluated strictly on the trailing `365`
    target days inside `train_period`), `nIterEp = ceil(ln(0.01) / ln(1 - B*rho / (N*T)))`
    random mini-batches per epoch across `epochs = 30`.
  - Loss (`hydroDL.model.crit.NSELossBatch`): per-basin variance-weighted MSE
    `mean(((y_hat - y_norm) / (std_b + 0.1)) ** 2)` over valid target observations.
  - Optimizer & schedule: `AdamW(lr=3e-3, betas=(0.9, 0.95), weight_decay=1e-3)` with
    zero decay on norms/biases/SSM state parameters, linear warmup over 5 epochs +
    cosine decay to `min_lr = 3e-4`, gradient norm clipping at `0.5`.
  - Evaluation: full-sequence continuous rollout over `[eval_start - 365d, eval_end]`
    sliced after spin-up (`[:, spinup_days:]`), inverse Gaussian scaling, and
    non-negative clipping (`doReLU=true`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import random

import numpy as np
import torch

from models.camels_data import load_camels_slice
from models.kratzert_lstm.windows import resolve_device, resolve_runner_kwargs
from models.shahriar2026_mamba3.model import (
    MambaStreamflowModelAdapter,
    build_optimizer_param_groups,
)

DEFAULT_SEEDS: tuple[int, ...] = (111111,)

RUNNER_DEFAULTS: dict[str, object] = {
    "hidden_size": 80,
    "mamba_layers": 3,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "headdim": 32,
    "adapter_bottleneck": 64,
    "dropout": 0.11,
    "residual_scale_init": 0.1,
    "rho": 365,
    "spinup_days": 365,
    "batch_size": 128,
    "epochs": 30,
    "n_batches_per_epoch": None,
    "lr": 3e-3,
    "min_lr": 3e-4,
    "weight_decay": 1e-3,
    "betas": (0.9, 0.95),
    "warmup_epochs": 5,
    "grad_clip": 0.5,
    "nse_eps": 0.1,
    "eval_batch_size": 64,
    "device": None,
    "return_members": False,
    "spatial_regime": None,
    "forcing_regime": None,
}


def build_shahriar_dynamic_inputs(slice_dict: Mapping[str, np.ndarray]) -> np.ndarray:
    """Assemble `['dayl', 'prcp', 'srad', 'tmean', 'vp']` per forcing product from a CAMELS slice."""
    x_dyn = np.asarray(slice_dict["x_dyn"], dtype=np.float32)
    n_channels = x_dyn.shape[-1]
    if n_channels % 5 != 0 or "x_aux" not in slice_dict:
        return x_dyn
    x_aux = np.asarray(slice_dict["x_aux"], dtype=np.float32)
    n_products = n_channels // 5
    if x_aux.shape[:2] != x_dyn.shape[:2] or x_aux.shape[-1] < 2 * n_products:
        return x_dyn

    blocks: list[np.ndarray] = []
    for p in range(n_products):
        base = 5 * p
        dayl = x_aux[:, :, 2 * p : 2 * p + 1]
        prcp = x_dyn[:, :, base + 0 : base + 1]
        srad = x_dyn[:, :, base + 1 : base + 2]
        tmax = x_dyn[:, :, base + 2 : base + 3]
        tmin = x_dyn[:, :, base + 3 : base + 4]
        vp = x_dyn[:, :, base + 4 : base + 5]
        tmean = 0.5 * (tmax + tmin)
        blocks.append(np.concatenate([dayl, prcp, srad, tmean, vp], axis=-1))
    return np.ascontiguousarray(np.concatenate(blocks, axis=-1), dtype=np.float32)


def compute_hydrodl_iters_per_epoch(
    n_basins: int, n_target_days: int, batch_size: int, rho: int
) -> int:
    """Exact `hydroDL` iteration count covering 99% of basin-days in expectation per epoch."""
    total_pts = max(1, int(n_basins) * int(n_target_days))
    batch_pts = max(1, int(batch_size) * int(rho))
    if batch_pts >= total_pts:
        return max(1, math.ceil(4.605170185988091 / 1.0))
    frac = batch_pts / total_pts
    return max(1, int(math.ceil(math.log(0.01) / math.log(1.0 - frac))))


def nse_loss_batch(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    std_b: torch.Tensor,
    eps: float = 0.1,
) -> torch.Tensor:
    """`hydroDL.model.crit.NSELossBatch` over valid standardized target steps."""
    valid = torch.isfinite(y_true)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return y_hat.sum() * 0.0
    denom = (std_b.view(-1, 1, 1) + float(eps)).clamp_min(1e-3)
    sq_err = ((y_hat - torch.nan_to_num(y_true, nan=0.0)) / denom) ** 2
    return sq_err[valid].mean()


def cosine_warmup_lr(
    epoch: int, total_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int
) -> float:
    """Epoch-level linear warmup followed by cosine decay to `min_lr`."""
    if total_epochs <= 1:
        return float(base_lr)
    warm = min(max(0, int(warmup_epochs)), total_epochs - 1)
    if warm > 0 and epoch < warm:
        return float(base_lr) * float(epoch + 1) / float(warm)
    denom = max(1, total_epochs - warm - 1)
    progress = min(1.0, max(0.0, float(epoch - warm) / float(denom)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train Shahriar (2026) Mamba-3 Adapter on `train_basins` and predict `eval_period`."""
    cfg = resolve_runner_kwargs(kwargs, RUNNER_DEFAULTS)
    req_spinup = max(0, int(cfg["spinup_days"]))
    req_rho = max(1, int(cfg["rho"]))

    train_slice = load_camels_slice(
        basins=train_basins,
        period=train_period,
        forcings=forcings,
        data_dir=data_dir,
        spinup_days=req_spinup,
        require_valid_targets=True,
    )
    eval_slice = load_camels_slice(
        basins=test_basins,
        period=eval_period,
        forcings=forcings,
        data_dir=data_dir,
        spinup_days=req_spinup,
        require_valid_targets=False,
    )

    spin_tr = int(train_slice["spinup_days"])
    spin_ev = int(eval_slice["spinup_days"])
    x_dyn_tr_raw = build_shahriar_dynamic_inputs(train_slice)
    x_dyn_ev_raw = build_shahriar_dynamic_inputs(eval_slice)
    x_stat_tr_raw = np.asarray(train_slice["x_stat"], dtype=np.float32)
    x_stat_ev_raw = np.asarray(eval_slice["x_stat"], dtype=np.float32)
    qobs_tr = np.asarray(train_slice["qobs"], dtype=np.float32)  # [:, :spin_tr] is NaN

    # Fit Gaussian z-score scalers strictly on training target days of train_basins.
    x_dyn_tr_target = x_dyn_tr_raw[:, spin_tr:, :]
    dyn_mean = np.mean(x_dyn_tr_target, axis=(0, 1), keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    dyn_std = np.std(x_dyn_tr_target, axis=(0, 1), keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    dyn_std = np.where(dyn_std < 1e-6, 1.0, dyn_std).astype(np.float32)

    stat_mean = np.mean(x_stat_tr_raw, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    stat_std = np.std(x_stat_tr_raw, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    stat_std = np.where(stat_std < 1e-6, 1.0, stat_std).astype(np.float32)

    q_tr_target = qobs_tr[:, spin_tr:]
    valid_q = np.isfinite(q_tr_target) & (q_tr_target >= 0.0)
    if not np.any(valid_q):
        raise ValueError("Zero valid target streamflow observations in training period.")
    y_vals = q_tr_target[valid_q].astype(np.float64)
    y_mean = float(np.mean(y_vals))
    y_std = float(max(np.std(y_vals), 1e-6))

    x_dyn_tr = ((x_dyn_tr_raw - dyn_mean) / dyn_std).astype(np.float32)
    x_stat_tr = ((x_stat_tr_raw - stat_mean) / stat_std).astype(np.float32)
    x_dyn_ev = ((x_dyn_ev_raw - dyn_mean) / dyn_std).astype(np.float32)
    x_stat_ev = ((x_stat_ev_raw - stat_mean) / stat_std).astype(np.float32)

    y_norm_tr = ((qobs_tr - y_mean) / y_std).astype(np.float32)
    y_norm_target = y_norm_tr[:, spin_tr:]

    n_train_basins, n_target_days = y_norm_target.shape
    basin_std = np.zeros(n_train_basins, dtype=np.float32)
    basin_valid_counts = np.zeros(n_train_basins, dtype=np.int64)
    for b in range(n_train_basins):
        vb = np.isfinite(y_norm_target[b])
        cnt = int(vb.sum())
        basin_valid_counts[b] = cnt
        basin_std[b] = float(np.std(y_norm_target[b, vb], dtype=np.float64)) if cnt >= 2 else 0.0

    eligible_basins = np.where(basin_valid_counts > 0)[0]
    if eligible_basins.size == 0:
        raise ValueError("No training basins have valid streamflow observations.")

    eff_rho = min(req_rho, n_target_days)
    eff_buff = spin_tr
    max_target_offset = max(0, n_target_days - eff_rho)

    batch_size = max(1, int(cfg["batch_size"]))
    epochs = max(1, int(cfg["epochs"]))
    n_iters = (
        int(cfg["n_batches_per_epoch"])
        if cfg["n_batches_per_epoch"] is not None
        else compute_hydrodl_iters_per_epoch(n_train_basins, n_target_days, batch_size, eff_rho)
    )
    base_lr = float(cfg["lr"])
    min_lr = float(cfg["min_lr"])
    weight_decay = float(cfg["weight_decay"])
    betas = tuple(float(v) for v in cfg["betas"])  # type: ignore[arg-type]
    warmup_epochs = int(cfg["warmup_epochs"])
    grad_clip = float(cfg["grad_clip"])
    nse_eps = float(cfg["nse_eps"])
    eval_bs = max(1, int(cfg["eval_batch_size"]))

    dev = resolve_device(cfg["device"])  # type: ignore[arg-type]
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    xd_tr_t = torch.from_numpy(x_dyn_tr).to(dev)
    xs_tr_t = torch.from_numpy(x_stat_tr).to(dev)
    yn_tr_t = torch.from_numpy(y_norm_tr).to(dev)
    bstd_t = torch.from_numpy(basin_std).to(dev)

    xd_ev_t = torch.from_numpy(x_dyn_ev).to(dev)
    xs_ev_t = torch.from_numpy(x_stat_ev).to(dev)

    win_len = eff_buff + eff_rho
    time_arange = torch.arange(win_len, device=dev, dtype=torch.long).unsqueeze(0)
    targ_arange = torch.arange(eff_rho, device=dev, dtype=torch.long).unsqueeze(0)

    members: list[np.ndarray] = []
    for seed in seeds:
        isd = int(seed)
        random.seed(isd)
        np.random.seed(isd)
        torch.manual_seed(isd)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(isd)
        rng = np.random.RandomState(isd)

        model = MambaStreamflowModelAdapter(
            dim_dyn_in=x_dyn_tr.shape[-1],
            hidden_size=int(cfg["hidden_size"]),
            num_attrs=x_stat_tr.shape[-1],
            mamba_layers=int(cfg["mamba_layers"]),
            mamba_version=3,
            d_state=int(cfg["d_state"]),
            d_conv=int(cfg["d_conv"]),
            expand=int(cfg["expand"]),
            headdim=int(cfg["headdim"]),
            mamba_adapter_bottleneck=int(cfg["adapter_bottleneck"]),
            mamba_dropout=float(cfg["dropout"]),
            mamba_residual_scale_init=float(cfg["residual_scale_init"]),
        ).to(dev)

        param_groups = build_optimizer_param_groups(model, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas)  # type: ignore[arg-type]

        for ep in range(epochs):
            cur_lr = cosine_warmup_lr(ep, epochs, base_lr, min_lr, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            model.train()
            ep_loss = 0.0
            n_steps = 0
            for _ in range(n_iters):
                b_np = rng.choice(eligible_basins, size=batch_size, replace=True)
                if max_target_offset > 0:
                    t0_np = rng.randint(0, max_target_offset + 1, size=batch_size)
                else:
                    t0_np = np.zeros(batch_size, dtype=np.int64)

                b_idx = torch.from_numpy(b_np).to(device=dev, dtype=torch.long)
                # Window spans [spin_tr + t0 - eff_buff, spin_tr + t0 + eff_rho):
                # first `eff_buff` steps warm up Mamba-3 state; trailing `eff_rho` steps
                # lie strictly within `[:, spin_tr:]` (the training period).
                win_start = torch.from_numpy(spin_tr - eff_buff + t0_np).to(
                    device=dev, dtype=torch.long
                )
                targ_start = torch.from_numpy(spin_tr + t0_np).to(device=dev, dtype=torch.long)

                t_grid = win_start.unsqueeze(1) + time_arange  # (B, win_len)
                x_batch = xd_tr_t[b_idx.unsqueeze(1), t_grid]  # (B, win_len, C)
                s_batch = xs_tr_t[b_idx]  # (B, S)
                y_batch = yn_tr_t[b_idx.unsqueeze(1), targ_start.unsqueeze(1) + targ_arange].unsqueeze(
                    -1
                )

                optimizer.zero_grad(set_to_none=True)
                y_pred_full = model(x_batch, s_batch)
                y_pred_targ = y_pred_full[:, eff_buff:, :]
                loss = nse_loss_batch(y_pred_targ, y_batch, bstd_t[b_idx], eps=nse_eps)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                ep_loss += float(loss.detach().item())
                n_steps += 1

            mean_loss = ep_loss / max(1, n_steps)
            print(
                f"[shahriar2026_mamba3 seed={isd}] epoch {ep + 1:02d}/{epochs} "
                f"iters={n_steps} lr={cur_lr:.5f} loss={mean_loss:.6f}",
                flush=True,
            )

        model.eval()
        pred_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(test_basins), eval_bs):
                end = min(start + eval_bs, len(test_basins))
                out = model(xd_ev_t[start:end], xs_ev_t[start:end])
                pred_chunks.append(out[:, spin_ev:, 0].detach().cpu().numpy())

        pred_norm = np.concatenate(pred_chunks, axis=0).astype(np.float64)
        pred_mm = np.clip(pred_norm * y_std + y_mean, 0.0, None).astype(np.float32)
        pred_mm = np.nan_to_num(pred_mm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        members.append(pred_mm)

    qsim_members = np.stack(members, axis=0).astype(np.float32)
    qsim_mean = np.mean(qsim_members.astype(np.float64), axis=0).astype(np.float32)
    if bool(cfg["return_members"]):
        return qsim_mean, qsim_members
    return qsim_mean
