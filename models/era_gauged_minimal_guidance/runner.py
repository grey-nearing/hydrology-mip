# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold execution interface for the ERA Gauged Minimal Guidance winning model (Node 73)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.camels_data import load_camels_slice
from models.era_gauged_minimal_guidance.model import EntityAwareLSTM

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "hidden_size": 320,
    "num_layers": 2,
    "dropout": 0.4,
    "batch_size": 128,
    "lr": 1.2e-3,
    "weight_decay": 0.1,
    "epochs": 70,
    "samples_per_basin_per_epoch": 25,
    "nse_stability_eps": 0.05,
    "log_loss_weight": 0.1,
    "n_batches_per_epoch": None,
    "eval_batch_size": 64,
    "device": None,
    "return_members": False,
    "fold_id": None,
    "fold_index": None,
}


def _get_rolling(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean along axis 1 matching Node 73's feature engineering."""
    res = np.zeros_like(x)
    cumsum = np.cumsum(x, axis=1, dtype=np.float64)
    res[:, w:] = ((cumsum[:, w:] - cumsum[:, :-w]) / float(w)).astype(np.float32)
    res[:, :w] = (cumsum[:, :w] / np.arange(1, w + 1, dtype=np.float64)[None, :]).astype(np.float32)
    return res


def _build_dynamic_features(x_dyn: np.ndarray, x_aux: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Assemble the 30-channel dynamic input tensor from Node 73."""
    n_basins, _, n_dyn = x_dyn.shape
    n_prods = max(1, n_dyn // 5)
    p_idx = [5 * k for k in range(n_prods)]
    sr_idx = [5 * k + 1 for k in range(n_prods)]
    t_idx = [idx for k in range(n_prods) for idx in (5 * k + 2, 5 * k + 3)]
    vp_idx = [5 * k + 4 for k in range(n_prods)]

    dates_dt = pd.to_datetime(dates)
    d_sin = np.sin(2.0 * np.pi * dates_dt.dayofyear.values / 365.25).astype(np.float32)
    d_cos = np.cos(2.0 * np.pi * dates_dt.dayofyear.values / 365.25).astype(np.float32)
    doy_feats = np.stack([np.tile(d_sin, (n_basins, 1)), np.tile(d_cos, (n_basins, 1))], axis=-1)

    p_mean = x_dyn[..., p_idx].mean(axis=-1)
    p_7d = _get_rolling(p_mean, 7)
    p_30d = _get_rolling(p_mean, 30)

    return np.concatenate(
        [
            x_dyn,
            x_aux,
            doy_feats,
            np.expand_dims(p_mean, -1),
            np.expand_dims(p_7d, -1),
            np.expand_dims(p_30d, -1),
            x_dyn[..., t_idx].mean(axis=-1, keepdims=True),
            x_dyn[..., sr_idx].mean(axis=-1, keepdims=True),
            x_dyn[..., vp_idx].mean(axis=-1, keepdims=True),
            np.log1p(np.maximum(p_mean[..., None], 0.0)),
        ],
        axis=-1,
    ).astype(np.float32)


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (42, 88, 2024, 101, 102, 103, 104, 105, 106, 107),
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train ERA Gauged Minimal Guidance (Node 73) on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_gauged_minimal_guidance: {unknown}")
    cfg = dict(RUNNER_DEFAULTS)
    cfg.update(kwargs)

    seq_len = int(cfg["seq_len"])
    spinup_days = int(cfg["spinup_days"])
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_layers"])
    dropout = float(cfg["dropout"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    samples_per_basin = int(cfg["samples_per_basin_per_epoch"])
    nse_eps = float(cfg["nse_stability_eps"])
    log_loss_weight = float(cfg["log_loss_weight"])
    n_batches_override = cfg["n_batches_per_epoch"]
    eval_batch_size = int(cfg["eval_batch_size"])
    return_members = bool(cfg["return_members"])

    dev_str = str(cfg["device"]) if cfg["device"] is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    use_amp = device.type == "cuda"

    train_slice = load_camels_slice(
        basins=list(train_basins),
        period=train_period,
        forcings=list(forcings),
        data_dir=data_dir,
        spinup_days=spinup_days,
        require_valid_targets=True,
    )
    eval_slice = load_camels_slice(
        basins=list(test_basins),
        period=eval_period,
        forcings=list(forcings),
        data_dir=data_dir,
        spinup_days=spinup_days,
        require_valid_targets=False,
    )

    stat_order = np.argsort(np.asarray(train_slice["static_attribute_names"], dtype=str))
    x_stat_train_raw = np.nan_to_num(np.asarray(train_slice["x_stat"], dtype=np.float32)[:, stat_order], nan=0.0)
    x_stat_eval_raw = np.nan_to_num(np.asarray(eval_slice["x_stat"], dtype=np.float32)[:, stat_order], nan=0.0)

    x_dyn_train_raw = _build_dynamic_features(
        np.asarray(train_slice["x_dyn"], dtype=np.float32),
        np.asarray(train_slice["x_aux"], dtype=np.float32),
        np.asarray(train_slice["dates"]),
    )
    x_dyn_eval_raw = _build_dynamic_features(
        np.asarray(eval_slice["x_dyn"], dtype=np.float32),
        np.asarray(eval_slice["x_aux"], dtype=np.float32),
        np.asarray(eval_slice["dates"]),
    )

    d_mean = x_dyn_train_raw[:, spinup_days:, :].mean(axis=(0, 1), keepdims=True)
    d_std = x_dyn_train_raw[:, spinup_days:, :].std(axis=(0, 1), keepdims=True) + 1e-6
    x_dyn_train_norm = ((x_dyn_train_raw - d_mean) / d_std).astype(np.float32)
    x_dyn_eval_norm = ((x_dyn_eval_raw - d_mean) / d_std).astype(np.float32)

    s_mean = x_stat_train_raw.mean(axis=0, keepdims=True)
    s_std = x_stat_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_stat_train_norm = ((x_stat_train_raw - s_mean) / s_std).astype(np.float32)
    x_stat_eval_norm = ((x_stat_eval_raw - s_mean) / s_std).astype(np.float32)

    qobs_train = np.asarray(train_slice["qobs"], dtype=np.float32)
    target_qobs = np.asarray(train_slice["target_qobs"], dtype=np.float32)
    y_mean = float(np.nanmean(target_qobs))
    y_std = float(np.nanstd(target_qobs)) + 1e-6
    y_norm = ((qobs_train - y_mean) / y_std).astype(np.float32)

    n_train_basins, n_train_days, dyn_dim = x_dyn_train_norm.shape
    stat_dim = x_stat_train_norm.shape[-1]
    per_basin_vars = np.ones(n_train_basins, dtype=np.float32)
    for b in range(n_train_basins):
        vb = target_qobs[b]
        vb = vb[~np.isnan(vb)]
        if len(vb) > 10:
            per_basin_vars[b] = float(np.var(vb / y_std))

    train_mask_np = np.zeros(n_train_days, dtype=bool)
    train_mask_np[spinup_days:] = True

    x_d_train_gpu = torch.from_numpy(x_dyn_train_norm).to(device)
    x_s_train_gpu = torch.from_numpy(x_stat_train_norm).to(device)
    y_gpu = torch.from_numpy(y_norm).to(device)
    b_var_gpu = torch.from_numpy(per_basin_vars).to(device)
    train_mask_gpu = torch.from_numpy(train_mask_np).to(device)

    x_d_eval_gpu = torch.from_numpy(x_dyn_eval_norm).to(device)
    x_s_eval_gpu = torch.from_numpy(x_stat_eval_norm).to(device)
    seq_offsets = torch.arange(seq_len, device=device).unsqueeze(0)

    train_idx = np.where(train_mask_np)[0]
    t_min = max(0, int(train_idx[0]) - seq_len + min(32, seq_len))
    t_max = max(t_min, int(train_idx[-1]) - seq_len)

    n_eval_days = int(eval_slice["n_target_days"])
    n_test_basins = len(test_basins)
    members: list[np.ndarray] = []

    for seed in seeds:
        s = int(seed)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        model = EntityAwareLSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_samples = n_train_basins * samples_per_basin
        steps_per_epoch = math.ceil(total_samples / batch_size)
        if n_batches_override is not None:
            steps_per_epoch = min(steps_per_epoch, max(1, int(n_batches_override)))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for _epoch in range(epochs):
            model.train()
            p_b = np.repeat(np.arange(n_train_basins), samples_per_basin)
            p_t = np.random.randint(t_min, t_max + 1, size=len(p_b))
            perm = np.random.permutation(len(p_b))
            p_b, p_t = p_b[perm], p_t[perm]
            if n_batches_override is not None:
                p_b = p_b[: steps_per_epoch * batch_size]
                p_t = p_t[: steps_per_epoch * batch_size]

            for i in range(0, len(p_b), batch_size):
                idx_b = torch.from_numpy(p_b[i : i + batch_size]).to(device)
                idx_t = torch.from_numpy(p_t[i : i + batch_size]).to(device)
                win_t = idx_t.unsqueeze(1) + seq_offsets
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    y_hat = model(x_d_train_gpu[idx_b.unsqueeze(1), win_t], x_s_train_gpu[idx_b]).squeeze(-1)
                    y_true = y_gpu[idx_b.unsqueeze(1), win_t]
                    m = train_mask_gpu[win_t] & (~torch.isnan(y_true))
                    if bool(m.any()):
                        nse_loss = torch.mean(
                            ((y_hat[m] - y_true[m]) ** 2)
                            / (b_var_gpu[idx_b].unsqueeze(1).expand(-1, seq_len)[m] + nse_eps)
                        )
                        y_true_unnorm = torch.clamp(y_true[m] * y_std + y_mean, min=0.0)
                        y_hat_unnorm = torch.clamp(y_hat[m] * y_std + y_mean, min=0.0)
                        log_loss = torch.mean((torch.log1p(y_hat_unnorm) - torch.log1p(y_true_unnorm)) ** 2)
                        loss = nse_loss + log_loss_weight * log_loss

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                scheduler.step()

        model.eval()
        pred_seed = np.zeros((n_test_basins, n_eval_days), dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            for b_start in range(0, n_test_basins, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_test_basins)
                y_hat = model(x_d_eval_gpu[b_start:b_end], x_s_eval_gpu[b_start:b_end]).squeeze(-1)
                y_mm = torch.clamp(y_hat[:, spinup_days:] * y_std + y_mean, min=0.0)
                pred_seed[b_start:b_end] = y_mm.cpu().numpy().astype(np.float32)
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
