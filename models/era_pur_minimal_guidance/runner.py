"""Fold execution interface for the ERA PUR Minimal Guidance winning model (Node 135)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.camels_data import load_camels_slice
from models.era_pur_minimal_guidance.model import DualFiLMLSTM

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "hidden_size": 256,
    "dropout": 0.3,
    "batch_size": 128,
    "lr": 1.0e-3,
    "weight_decay": 0.05,
    "epochs": 50,
    "samples_per_basin": 64,
    "static_noise_std": 0.015,
    "n_batches_per_epoch": None,
    "eval_batch_size": 64,
    "device": None,
    "return_members": False,
    "fold_id": None,
    "fold_index": None,
}


def _prepare_features(
    x_dyn: np.ndarray,
    x_aux: np.ndarray,
    x_stat_sorted: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    area_km2: np.ndarray,
    clim_x_dyn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 23-channel dynamic and 33-channel static inputs from Node 135."""
    n_basins, n_days, n_dyn = x_dyn.shape
    n_prods = max(1, n_dyn // 5)
    p_idx = [5 * k for k in range(n_prods)]
    t_idx = [idx for k in range(n_prods) for idx in (5 * k + 2, 5 * k + 3)]

    area_log = np.log1p(np.maximum(area_km2.astype(np.float32), 0.0)).reshape(-1, 1)
    p_mean = clim_x_dyn[:, :, p_idx].mean(axis=(1, 2)).reshape(-1, 1)
    p_mean_log = np.log1p(np.maximum(p_mean, 0.0))
    t_mean = clim_x_dyn[:, :, t_idx].mean(axis=(1, 2)).reshape(-1, 1)
    aridity = (p_mean / (np.maximum(t_mean, -5.0) + 10.0)).reshape(-1, 1)

    x_s = np.column_stack(
        [
            x_stat_sorted,
            lat.astype(np.float32).reshape(-1, 1),
            lon.astype(np.float32).reshape(-1, 1),
            area_log,
            p_mean_log,
            t_mean,
            aridity,
        ]
    ).astype(np.float32)

    col_means = np.nanmean(x_s, axis=0)
    inds = np.where(np.isnan(x_s))
    if inds[0].size > 0:
        x_s[inds] = np.take(col_means, inds[1])

    d_idx = np.arange(n_days, dtype=np.float32)
    sin_d = np.sin(2.0 * np.pi * d_idx / 365.25)
    cos_d = np.cos(2.0 * np.pi * d_idx / 365.25)
    seasonal = np.stack([np.tile(sin_d, (n_basins, 1)), np.tile(cos_d, (n_basins, 1))], axis=-1).astype(np.float32)

    x_d = np.concatenate([x_dyn, x_aux, seasonal], axis=-1).astype(np.float32)
    return x_d, x_s


def run_fold(
    train_basins: Sequence[str],
    test_basins: Sequence[str],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    **kwargs: object,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Train ERA PUR Minimal Guidance (Node 135) on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_pur_minimal_guidance: {unknown}")
    cfg = dict(RUNNER_DEFAULTS)
    cfg.update(kwargs)

    seq_len = int(cfg["seq_len"])
    spinup_days = int(cfg["spinup_days"])
    hidden_size = int(cfg["hidden_size"])
    dropout = float(cfg["dropout"])
    batch_size = int(cfg["batch_size"])
    lr_max = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    samples_per_basin = int(cfg["samples_per_basin"])
    static_noise_std = float(cfg["static_noise_std"])
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
    test_clim_slice = load_camels_slice(
        basins=list(test_basins),
        period=train_period,
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
        x_stat_sorted=x_stat_train_sorted,
        lat=np.asarray(train_slice["lat"], dtype=np.float32),
        lon=np.asarray(train_slice["lon"], dtype=np.float32),
        area_km2=np.asarray(train_slice["area_km2"], dtype=np.float32),
        clim_x_dyn=np.asarray(train_slice["x_dyn"], dtype=np.float32),
    )
    x_dyn_eval_raw, x_stat_eval_raw = _prepare_features(
        x_dyn=np.asarray(eval_slice["x_dyn"], dtype=np.float32),
        x_aux=np.asarray(eval_slice["x_aux"], dtype=np.float32),
        x_stat_sorted=x_stat_eval_sorted,
        lat=np.asarray(eval_slice["lat"], dtype=np.float32),
        lon=np.asarray(eval_slice["lon"], dtype=np.float32),
        area_km2=np.asarray(eval_slice["area_km2"], dtype=np.float32),
        clim_x_dyn=np.asarray(test_clim_slice["x_dyn"], dtype=np.float32),
    )

    xd_m = x_dyn_train_raw[:, spinup_days:, :].mean(axis=(0, 1), keepdims=True)
    xd_s = x_dyn_train_raw[:, spinup_days:, :].std(axis=(0, 1), keepdims=True) + 1e-6
    x_d_train_gpu = torch.from_numpy(((x_dyn_train_raw - xd_m) / xd_s).astype(np.float32)).to(device)
    x_d_eval_gpu = torch.from_numpy(((x_dyn_eval_raw - xd_m) / xd_s).astype(np.float32)).to(device)

    xs_m = x_stat_train_raw.mean(axis=0, keepdims=True)
    xs_s = x_stat_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_s_train_gpu = torch.from_numpy(((x_stat_train_raw - xs_m) / xs_s).astype(np.float32)).to(device)
    x_s_eval_gpu = torch.from_numpy(((x_stat_eval_raw - xs_m) / xs_s).astype(np.float32)).to(device)

    qobs_train = np.asarray(train_slice["qobs"], dtype=np.float32)
    q_target = np.sqrt(np.maximum(qobs_train, 0.0)).astype(np.float32)
    y_m = float(np.nanmean(q_target))
    y_s = float(np.nanstd(q_target)) + 1e-6
    y_gpu = torch.from_numpy(((q_target - y_m) / y_s).astype(np.float32)).to(device)

    n_train_basins, n_train_days, dyn_dim = x_dyn_train_raw.shape
    stat_dim = x_stat_train_raw.shape[-1]
    train_mask_np = np.zeros(n_train_days, dtype=bool)
    train_mask_np[spinup_days:] = True

    valid_t = {
        b: np.where(train_mask_np & (~np.isnan(q_target[b])))[0]
        for b in range(n_train_basins)
    }
    valid_t = {b: t[t >= seq_len - 1] for b, t in valid_t.items() if int(np.sum(t >= seq_len - 1)) > 0}
    if not valid_t:
        raise ValueError("No valid training timesteps after spin-up in era_pur_minimal_guidance.")

    seq_offs = torch.arange(-seq_len + 1, 1, device=device).unsqueeze(0)
    n_eval_days = int(eval_slice["n_target_days"])
    n_test_basins = len(test_basins)
    members: list[np.ndarray] = []

    for seed in seeds:
        s = int(seed)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        model = DualFiLMLSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

        steps = math.ceil((len(valid_t) * samples_per_basin) / batch_size)
        if n_batches_override is not None:
            steps = min(steps, max(1, int(n_batches_override)))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr_max,
            steps_per_epoch=steps,
            epochs=epochs,
        )
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.RandomState(s)

        for _epoch in range(epochs):
            model.train()
            b_list: list[np.ndarray] = []
            t_list: list[np.ndarray] = []
            for b, t_arr in valid_t.items():
                idx = rng.choice(t_arr, size=samples_per_basin, replace=(len(t_arr) < samples_per_basin))
                b_list.append(np.full(len(idx), b, dtype=np.int64))
                t_list.append(idx.astype(np.int64))

            b_all = np.concatenate(b_list)
            t_all = np.concatenate(t_list)
            perm = rng.permutation(len(b_all))
            b_all, t_all = b_all[perm], t_all[perm]
            if n_batches_override is not None:
                b_all = b_all[: steps * batch_size]
                t_all = t_all[: steps * batch_size]

            for i in range(0, len(b_all), batch_size):
                bb = torch.from_numpy(b_all[i : i + batch_size]).to(device)
                bt = torch.from_numpy(t_all[i : i + batch_size]).to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    s_feat = x_s_train_gpu[bb]
                    if model.training and static_noise_std > 0.0:
                        s_feat = s_feat + torch.randn_like(s_feat) * static_noise_std
                    x_d_seq = x_d_train_gpu[bb.unsqueeze(1), bt.unsqueeze(1) + seq_offs]
                    y_hat = model(x_d_seq, s_feat)
                    loss = criterion(y_hat, y_gpu[bb, bt])

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        model.eval()
        pred_seed = np.zeros((n_test_basins, n_eval_days), dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            for b_start in range(0, n_test_basins, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_test_basins)
                p_norm = model(x_d_eval_gpu[b_start:b_end], x_s_eval_gpu[b_start:b_end], return_seq=True)
                p_mm = (torch.clamp(p_norm[:, spinup_days:] * y_s + y_m, min=0.0) ** 2).cpu().numpy()
                pred_seed[b_start:b_end] = p_mm.astype(np.float32)
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
