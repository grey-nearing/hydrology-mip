"""Fold execution interface for the ERA Physics-Guided PUB winning model (Node 203 — PhysicsEALSTM)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.camels_data import load_camels_slice
from models.era_pub_physics_guided.model import PhysicsEALSTM, hybrid_nse_log_loss

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "hidden_size": 160,
    "stat_emb_size": 64,
    "dropout": 0.2,
    "batch_size": 256,
    "lr": 3.0e-3,
    "weight_decay": 0.05,
    "epochs": 40,
    "samples_per_basin_per_epoch": 64,
    "ema_alpha": 0.99,
    "static_noise_std": 0.05,
    "n_batches_per_epoch": None,
    "eval_batch_size": 32,
    "device": None,
    "return_members": False,
    "fold_id": None,
    "fold_index": None,
}


def _prepare_features(
    x_dyn: np.ndarray,
    x_aux: np.ndarray,
    dates: np.ndarray,
    x_stat_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 14-channel physics-informed dynamic features and log-transformed 27-channel static features (Node 203)."""
    n_basins, n_days, n_dyn = x_dyn.shape
    n_prods = max(1, n_dyn // 5)
    n_aux_prods = max(1, x_aux.shape[-1] // 2)

    p_idx = [5 * k + 0 for k in range(n_prods)]
    sr_idx = [5 * k + 1 for k in range(n_prods)]
    tmax_idx = [5 * k + 2 for k in range(n_prods)]
    tmin_idx = [5 * k + 3 for k in range(n_prods)]
    vp_idx = [5 * k + 4 for k in range(n_prods)]

    dayl_idx = [2 * k + 0 for k in range(n_aux_prods)]
    swe_idx = [2 * k + 1 for k in range(n_aux_prods)]

    p_all = x_dyn[:, :, p_idx]
    p_mean = np.mean(p_all, axis=-1)
    p_std = np.std(p_all, axis=-1)

    t_max_all = x_dyn[:, :, tmax_idx]
    t_min_all = x_dyn[:, :, tmin_idx]
    t_mean = (np.mean(t_max_all, axis=-1) + np.mean(t_min_all, axis=-1)) / 2.0
    t_range = np.mean(t_max_all - t_min_all, axis=-1)

    sr_mean = np.mean(x_dyn[:, :, sr_idx], axis=-1)
    vp_mean = np.mean(x_dyn[:, :, vp_idx], axis=-1)
    swe_mean = np.mean(x_aux[:, :, swe_idx], axis=-1)
    dayl_mean = np.mean(x_aux[:, :, dayl_idx], axis=-1)

    pet = 0.0023 * sr_mean * (t_mean + 17.8) * np.sqrt(np.maximum(0.0, t_range))
    is_snow = (t_mean < 0.0).astype(np.float32)
    p_liq = p_mean * (1.0 - is_snow)
    p_snow = p_mean * is_snow
    melt_pot = np.maximum(0.0, t_mean * 2.0)

    doy = pd.to_datetime(dates).dayofyear.values
    sn = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    cs = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)

    feats = [
        np.log1p(np.maximum(0.0, p_mean)),
        p_mean,
        p_std,
        t_mean,
        t_range,
        vp_mean,
        pet,
        p_liq,
        p_snow,
        melt_pot,
        swe_mean,
        dayl_mean,
        np.tile(sn, (n_basins, 1)),
        np.tile(cs, (n_basins, 1)),
    ]
    x_dyn_proc = np.stack(feats, axis=-1).astype(np.float32)

    x_stat_proc = x_stat_sorted.copy().astype(np.float32)
    for i in (0, 1, 2, 3, 4):
        if i < x_stat_proc.shape[1]:
            x_stat_proc[:, i] = np.log1p(np.maximum(0.0, x_stat_proc[:, i]))
    x_stat_proc = np.nan_to_num(x_stat_proc, nan=0.0).astype(np.float32)

    return x_dyn_proc, x_stat_proc


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train ERA Physics-Guided PUB (Node 203) on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_pub_physics_guided: {unknown}")
    cfg = dict(RUNNER_DEFAULTS)
    cfg.update(kwargs)

    seq_len = int(cfg["seq_len"])
    spinup_days = int(cfg["spinup_days"])
    hidden_size = int(cfg["hidden_size"])
    stat_emb_size = int(cfg["stat_emb_size"])
    dropout = float(cfg["dropout"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    samples_per_basin = int(cfg["samples_per_basin_per_epoch"])
    ema_alpha = float(cfg["ema_alpha"])
    static_noise_std = float(cfg["static_noise_std"])
    n_batches_override = cfg["n_batches_per_epoch"]
    eval_batch_size = int(cfg["eval_batch_size"])
    return_members = bool(cfg["return_members"])
    fold_index = int(cfg["fold_index"]) if cfg["fold_index"] is not None else 0

    dev_str = str(cfg["device"]) if cfg["device"] is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)

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
    x_stat_train_sorted = np.asarray(train_slice["x_stat"], dtype=np.float32)[:, stat_order]
    x_stat_eval_sorted = np.asarray(eval_slice["x_stat"], dtype=np.float32)[:, stat_order]

    x_dyn_train_raw, x_stat_train_raw = _prepare_features(
        x_dyn=np.asarray(train_slice["x_dyn"], dtype=np.float32),
        x_aux=np.asarray(train_slice["x_aux"], dtype=np.float32),
        dates=np.asarray(train_slice["dates"]),
        x_stat_sorted=x_stat_train_sorted,
    )
    x_dyn_eval_raw, x_stat_eval_raw = _prepare_features(
        x_dyn=np.asarray(eval_slice["x_dyn"], dtype=np.float32),
        x_aux=np.asarray(eval_slice["x_aux"], dtype=np.float32),
        dates=np.asarray(eval_slice["dates"]),
        x_stat_sorted=x_stat_eval_sorted,
    )

    d_mean = x_dyn_train_raw.mean(axis=(0, 1), keepdims=True)
    d_std = x_dyn_train_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    x_d_train_norm = ((x_dyn_train_raw - d_mean) / d_std).astype(np.float32)
    x_d_eval_norm = ((x_dyn_eval_raw - d_mean) / d_std).astype(np.float32)

    s_mean = x_stat_train_raw.mean(axis=0, keepdims=True)
    s_std = x_stat_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_s_train_norm = ((x_stat_train_raw - s_mean) / s_std).astype(np.float32)
    x_s_eval_norm = ((x_stat_eval_raw - s_mean) / s_std).astype(np.float32)

    qobs_train = np.asarray(train_slice["qobs"], dtype=np.float32)
    n_train_basins, n_train_days, dyn_dim = x_d_train_norm.shape
    stat_dim = x_s_train_norm.shape[-1]

    train_mask_np = np.zeros(n_train_days, dtype=bool)
    train_mask_np[spinup_days:] = True

    b_stds = np.ones(n_train_basins, dtype=np.float32)
    train_meta: list[tuple[int, np.ndarray]] = []
    for b in range(n_train_basins):
        std = float(np.nanstd(qobs_train[b, train_mask_np]))
        if not np.isnan(std) and std > 1e-3:
            b_stds[b] = std
        valid_idx = np.where(train_mask_np & (~np.isnan(qobs_train[b])))[0]
        valid_idx = valid_idx[valid_idx >= (seq_len - 1)]
        if len(valid_idx) > 0:
            train_meta.append((b, valid_idx))

    if not train_meta:
        raise ValueError("No valid training timesteps after spin-up in era_pub_physics_guided.")

    xd_g = torch.from_numpy(x_d_train_norm).to(device)
    xs_g = torch.from_numpy(x_s_train_norm).to(device)
    y_g = torch.from_numpy(qobs_train).to(device)
    std_g = torch.from_numpy(b_stds).to(device)
    xd_eval_g = torch.from_numpy(x_d_eval_norm).to(device)
    xs_eval_g = torch.from_numpy(x_s_eval_norm).to(device)

    offsets = torch.arange(-seq_len + 1, 1, device=device).unsqueeze(0)
    n_eval_days = int(eval_slice["n_target_days"])
    n_test_basins = len(test_basins)
    members: list[np.ndarray] = []

    for seed in seeds:
        s = int(seed)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        model = PhysicsEALSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_size=hidden_size,
            stat_emb_size=stat_emb_size,
            dropout=dropout,
        ).to(device)
        ema_model = copy.deepcopy(model).eval()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=1,
            epochs=max(1, epochs),
        )
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        model_params = list(model.parameters())
        ema_params = list(ema_model.parameters())
        rng = np.random.RandomState(s + fold_index)

        for _epoch in range(epochs):
            model.train()
            sb_list: list[np.ndarray] = []
            st_list: list[np.ndarray] = []
            for b, v_idx in train_meta:
                sel = rng.choice(v_idx, size=min(len(v_idx), samples_per_basin))
                sb_list.append(np.full(len(sel), b, dtype=np.int64))
                st_list.append(sel.astype(np.int64))

            sb = np.concatenate(sb_list)
            st = np.concatenate(st_list)
            perm = rng.permutation(len(sb))
            if n_batches_override is not None:
                perm = perm[: max(1, int(n_batches_override)) * batch_size]

            sb_gpu = torch.from_numpy(sb[perm]).to(device)
            st_gpu = torch.from_numpy(st[perm]).to(device)

            for i in range(0, len(perm), batch_size):
                bb = sb_gpu[i : i + batch_size]
                bt = st_gpu[i : i + batch_size]

                stat_in = xs_g[bb]
                if static_noise_std > 0.0:
                    stat_in = stat_in + torch.randn_like(stat_in) * static_noise_std

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    yh = model(xd_g[bb.unsqueeze(1), bt.unsqueeze(1) + offsets], stat_in)
                    yt = y_g[bb, bt]
                    loss = hybrid_nse_log_loss(yh, yt, std_g[bb])

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model_params, 1.0)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    torch._foreach_lerp_(ema_params, model_params, 1.0 - ema_alpha)

            scheduler.step()

        ema_model.eval()
        pred_seed = np.zeros((n_test_basins, n_eval_days), dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for b_start in range(0, n_test_basins, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_test_basins)
                p = ema_model(xd_eval_g[b_start:b_end], xs_eval_g[b_start:b_end], return_seq=True)
                p_mm = torch.clamp(p[:, spinup_days:], min=0.0).float().cpu().numpy()
                pred_seed[b_start:b_end] = p_mm.astype(np.float32)
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
