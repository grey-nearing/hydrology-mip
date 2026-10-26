"""Fold execution interface for the ERA PUB Minimal Guidance winning model (Node 82)."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.camels_data import load_camels_slice
from models.era_pub_minimal_guidance.model import EALSTM, hybrid_nse_loss

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "hidden_size": 256,
    "stat_emb_size": 128,
    "dropout": 0.3,
    "batch_size": 256,
    "lr": 2.5e-3,
    "weight_decay": 1.0e-2,
    "epochs": 45,
    "samples_per_basin_per_epoch": 64,
    "ema_alpha": 0.999,
    "input_noise_std": 0.005,
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
    dates: np.ndarray,
    x_stat_sorted: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    area_km2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 26-channel dynamic and 30-channel static inputs from Node 82."""
    n_basins, _, n_dyn = x_dyn.shape
    if n_dyn >= 12:
        t_range = x_dyn[:, :, 6:9] - x_dyn[:, :, 9:12]
    else:
        t_range = x_dyn[:, :, 2:3] - x_dyn[:, :, 3:4]

    d_dates = pd.to_datetime(dates)
    doy = d_dates.dayofyear.values
    doy_sin = np.tile(np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)[None, :, None], (n_basins, 1, 1))
    doy_cos = np.tile(np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)[None, :, None], (n_basins, 1, 1))

    x_dyn_raw = np.concatenate([x_dyn, x_aux, t_range, doy_sin, doy_cos], axis=-1).astype(np.float32)
    log_area = np.log1p(np.maximum(area_km2.astype(np.float32), 0.0))[:, None]
    x_stat_raw = np.concatenate(
        [
            x_stat_sorted,
            lat.astype(np.float32)[:, None],
            lon.astype(np.float32)[:, None],
            log_area,
        ],
        axis=-1,
    ).astype(np.float32)
    return x_dyn_raw, x_stat_raw


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
    """Train ERA PUB Minimal Guidance (Node 82) on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_pub_minimal_guidance: {unknown}")
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
    input_noise_std = float(cfg["input_noise_std"])
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
        lat=np.asarray(train_slice["lat"], dtype=np.float32),
        lon=np.asarray(train_slice["lon"], dtype=np.float32),
        area_km2=np.asarray(train_slice["area_km2"], dtype=np.float32),
    )
    x_dyn_eval_raw, x_stat_eval_raw = _prepare_features(
        x_dyn=np.asarray(eval_slice["x_dyn"], dtype=np.float32),
        x_aux=np.asarray(eval_slice["x_aux"], dtype=np.float32),
        dates=np.asarray(eval_slice["dates"]),
        x_stat_sorted=x_stat_eval_sorted,
        lat=np.asarray(eval_slice["lat"], dtype=np.float32),
        lon=np.asarray(eval_slice["lon"], dtype=np.float32),
        area_km2=np.asarray(eval_slice["area_km2"], dtype=np.float32),
    )

    d_mean = x_dyn_train_raw[:, spinup_days:, :].mean(axis=(0, 1), keepdims=True)
    d_std = x_dyn_train_raw[:, spinup_days:, :].std(axis=(0, 1), keepdims=True) + 1e-6
    x_d_train_norm = ((x_dyn_train_raw - d_mean) / d_std).astype(np.float32)
    x_d_eval_norm = ((x_dyn_eval_raw - d_mean) / d_std).astype(np.float32)

    s_mean = np.nanmean(x_stat_train_raw, axis=0, keepdims=True)
    s_std = np.nanstd(x_stat_train_raw, axis=0, keepdims=True) + 1e-6
    x_s_train_proc = np.where(np.isnan(x_stat_train_raw), s_mean, x_stat_train_raw)
    x_s_eval_proc = np.where(np.isnan(x_stat_eval_raw), s_mean, x_stat_eval_raw)
    x_s_train_norm = ((x_s_train_proc - s_mean) / s_std).astype(np.float32)
    x_s_eval_norm = ((x_s_eval_proc - s_mean) / s_std).astype(np.float32)

    qobs_train = np.asarray(train_slice["qobs"], dtype=np.float32)
    y_train_vals = qobs_train[:, spinup_days:]
    y_m = float(np.nanmean(y_train_vals))
    y_s = float(np.nanstd(y_train_vals)) + 1e-6
    y_norm = ((qobs_train - y_m) / y_s).astype(np.float32)

    n_train_basins, n_train_days, dyn_dim = x_d_train_norm.shape
    stat_dim = x_s_train_norm.shape[-1]
    train_mask_np = np.zeros(n_train_days, dtype=bool)
    train_mask_np[spinup_days:] = True

    b_stds = np.ones(n_train_basins, dtype=np.float32)
    valid_t_map: dict[int, np.ndarray] = {}
    min_valid_days = min(90, max(1, (n_train_days - spinup_days) // 2))
    for b in range(n_train_basins):
        q_b = qobs_train[b, train_mask_np]
        q_b_valid = q_b[~np.isnan(q_b)]
        if len(q_b_valid) >= min_valid_days:
            b_stds[b] = max(float(np.std(q_b_valid)), 0.02) / y_s
            t_idx = np.where(train_mask_np & (~np.isnan(qobs_train[b])))[0]
            t_idx = t_idx[t_idx >= seq_len - 1]
            if len(t_idx) > 0:
                valid_t_map[b] = t_idx

    if not valid_t_map:
        raise ValueError("No valid training timesteps after spin-up in era_pub_minimal_guidance.")

    xd_g = torch.from_numpy(x_d_train_norm).to(device)
    xs_g = torch.from_numpy(x_s_train_norm).to(device)
    y_g = torch.from_numpy(y_norm).to(device)
    bstd_g = torch.from_numpy(b_stds).to(device)
    xd_eval_g = torch.from_numpy(x_d_eval_norm).to(device)
    xs_eval_g = torch.from_numpy(x_s_eval_norm).to(device)

    offsets = torch.arange(-seq_len + 1, 1, device=device).unsqueeze(0)
    n_eval_days = int(eval_slice["n_target_days"])
    n_test_basins = len(test_basins)
    members: list[np.ndarray] = []

    for seed in seeds:
        s = int(seed)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        model = EALSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_size=hidden_size,
            stat_emb_size=stat_emb_size,
            dropout=dropout,
        ).to(device)
        ema_model = copy.deepcopy(model).eval()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_samples = len(valid_t_map) * samples_per_basin
        steps_per_epoch = math.ceil(num_samples / batch_size)
        if n_batches_override is not None:
            steps_per_epoch = min(steps_per_epoch, max(1, int(n_batches_override)))

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        rng = np.random.RandomState(s + fold_index)

        for _epoch in range(epochs):
            model.train()
            pb: list[np.ndarray] = []
            pt: list[np.ndarray] = []
            for b, t_arr in valid_t_map.items():
                idx = rng.choice(t_arr, size=samples_per_basin, replace=True)
                pb.append(np.full(len(idx), b, dtype=np.int64))
                pt.append(idx.astype(np.int64))

            pb_all = np.concatenate(pb)
            pt_all = np.concatenate(pt)
            indices = rng.permutation(len(pb_all))
            if n_batches_override is not None:
                indices = indices[: steps_per_epoch * batch_size]

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                ib = torch.from_numpy(pb_all[batch_idx]).to(device)
                it = torch.from_numpy(pt_all[batch_idx]).to(device)

                x_d_batch = xd_g[ib.unsqueeze(1), it.unsqueeze(1) + offsets]
                if input_noise_std > 0.0:
                    x_d_batch = x_d_batch + torch.randn_like(x_d_batch) * input_noise_std

                opt.zero_grad()
                pred = model(x_d_batch, xs_g[ib])
                loss = hybrid_nse_loss(pred, y_g[ib, it], bstd_g[ib])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                with torch.no_grad():
                    for p, ep in zip(model.parameters(), ema_model.parameters()):
                        ep.data.mul_(ema_alpha).add_(p.data, alpha=1.0 - ema_alpha)

        ema_model.eval()
        pred_seed = np.zeros((n_test_basins, n_eval_days), dtype=np.float32)
        with torch.no_grad():
            for b_start in range(0, n_test_basins, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_test_basins)
                p_norm = ema_model(xd_eval_g[b_start:b_end], xs_eval_g[b_start:b_end], return_seq=True)
                p_mm = torch.clamp(p_norm[:, spinup_days:] * y_s + y_m, min=0.0).cpu().numpy()
                pred_seed[b_start:b_end] = p_mm.astype(np.float32)
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
