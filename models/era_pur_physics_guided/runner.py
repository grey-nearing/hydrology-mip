"""Fold execution interface for the ERA Physics-Guided PUR winning model (Node 206 — PURPhysicsModel)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.camels_data import load_camels_slice
from models.era_pur_physics_guided.model import PURPhysicsModel, scaled_mse_loss

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "hidden_size": 128,
    "stat_emb_size": 64,
    "dropout": 0.4,
    "batch_size": 512,
    "lr": 1.0e-3,
    "weight_decay": 0.05,
    "epochs": 40,
    "samples_per_basin": 128,
    "static_noise_std": 0.02,
    "n_batches_per_epoch": None,
    "eval_batch_size": 64,
    "device": None,
    "return_members": False,
    "fold_id": None,
    "fold_index": None,
}


def _compute_api(p_mean: np.ndarray, decays: Sequence[float] = (7.0, 30.0, 90.0, 180.0)) -> np.ndarray:
    """Vectorized exponential-decay Antecedent Precipitation Index (API) across basins."""
    b_cnt, t_cnt, _ = p_mean.shape
    apis: list[np.ndarray] = []
    for d in decays:
        api = np.zeros((b_cnt, t_cnt, 1), dtype=np.float32)
        current = np.zeros((b_cnt, 1), dtype=np.float32)
        alpha = float(np.exp(-1.0 / float(d)))
        for t in range(t_cnt):
            current = current * alpha + p_mean[:, t, :]
            api[:, t, :] = current
        apis.append(api)
    return np.concatenate(apis, axis=-1)


def _prepare_features(
    x_dyn: np.ndarray,
    x_aux: np.ndarray,
    dates: np.ndarray,
    x_stat_sorted: np.ndarray,
    area_km2: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 26-channel physics-engineered dynamic features and 30-channel static features (Node 206)."""
    n_basins, _, n_dyn = x_dyn.shape
    n_prods = max(1, n_dyn // 5)
    n_aux_prods = max(1, x_aux.shape[-1] // 2)

    p_idx = [5 * k + 0 for k in range(n_prods)]
    sr_idx = [5 * k + 1 for k in range(n_prods)]
    tmax_idx = [5 * k + 2 for k in range(n_prods)]
    tmin_idx = [5 * k + 3 for k in range(n_prods)]

    dayl_idx = [2 * k + 0 for k in range(n_aux_prods)]
    swe_idx = [2 * k + 1 for k in range(n_aux_prods)]

    p_mean = np.mean(x_dyn[..., p_idx], axis=-1, keepdims=True)
    t_max = np.mean(x_dyn[..., tmax_idx], axis=-1, keepdims=True)
    t_min = np.mean(x_dyn[..., tmin_idx], axis=-1, keepdims=True)
    t_mean = (t_max + t_min) / 2.0
    srad = np.mean(x_dyn[..., sr_idx], axis=-1, keepdims=True)

    pet = 0.0023 * srad * (t_mean + 17.8) * np.sqrt(np.maximum(0.0, t_max - t_min))
    p_api = _compute_api(p_mean, (7.0, 30.0, 90.0, 180.0))

    doy = pd.to_datetime(dates).dayofyear.values
    sin_doy = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    cos_doy = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)
    season = np.tile(np.stack([sin_doy, cos_doy], axis=-1)[None, :, :], (n_basins, 1, 1))

    dayl = np.mean(x_aux[..., dayl_idx], axis=-1, keepdims=True)
    swe = np.mean(x_aux[..., swe_idx], axis=-1, keepdims=True)

    x_dyn_eng = np.concatenate([x_dyn, p_mean, t_mean, pet, p_api, dayl, swe, season], axis=-1).astype(np.float32)

    log_area = np.log1p(np.maximum(0.0, area_km2.astype(np.float32))).reshape(-1, 1)
    x_stat_plus = np.concatenate(
        [
            x_stat_sorted.astype(np.float32),
            log_area,
            lat.astype(np.float32).reshape(-1, 1),
            lon.astype(np.float32).reshape(-1, 1),
        ],
        axis=-1,
    )
    x_stat_plus = np.nan_to_num(x_stat_plus, nan=0.0).astype(np.float32)
    return x_dyn_eng, x_stat_plus


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
    """Train ERA Physics-Guided PUR (Node 206) on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_pur_physics_guided: {unknown}")
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
    samples_per_basin = int(cfg["samples_per_basin"])
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
        area_km2=np.asarray(train_slice["area_km2"], dtype=np.float32),
        lat=np.asarray(train_slice["lat"], dtype=np.float32),
        lon=np.asarray(train_slice["lon"], dtype=np.float32),
    )
    x_dyn_eval_raw, x_stat_eval_raw = _prepare_features(
        x_dyn=np.asarray(eval_slice["x_dyn"], dtype=np.float32),
        x_aux=np.asarray(eval_slice["x_aux"], dtype=np.float32),
        dates=np.asarray(eval_slice["dates"]),
        x_stat_sorted=x_stat_eval_sorted,
        area_km2=np.asarray(eval_slice["area_km2"], dtype=np.float32),
        lat=np.asarray(eval_slice["lat"], dtype=np.float32),
        lon=np.asarray(eval_slice["lon"], dtype=np.float32),
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

    basin_scales = (np.nanstd(qobs_train[:, train_mask_np], axis=1) + 1e-2).astype(np.float32)
    basin_scales = np.nan_to_num(basin_scales, nan=1.0).astype(np.float32)

    valid_by_basin: list[tuple[int, np.ndarray]] = []
    for b in range(n_train_basins):
        valid_t = np.where(train_mask_np & (~np.isnan(qobs_train[b])))[0]
        valid_t = valid_t[valid_t >= (seq_len - 1)]
        if len(valid_t) > 0:
            valid_by_basin.append((b, valid_t))

    if not valid_by_basin:
        raise ValueError("No valid training timesteps after spin-up in era_pur_physics_guided.")

    xd_g = torch.from_numpy(x_d_train_norm).to(device)
    xs_g = torch.from_numpy(x_s_train_norm).to(device)
    y_g = torch.from_numpy(qobs_train).to(device)
    scales_g = torch.from_numpy(basin_scales).to(device)
    xd_eval_g = torch.from_numpy(x_d_eval_norm).to(device)
    xs_eval_g = torch.from_numpy(x_s_eval_norm).to(device)

    offsets = torch.arange(-seq_len + 1, 1, device=device).unsqueeze(0)
    n_eval_days = int(eval_slice["n_target_days"])
    n_test_basins = len(test_basins)
    use_amp = device.type == "cuda"
    members: list[np.ndarray] = []

    for seed in seeds:
        s = int(seed)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        model = PURPhysicsModel(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            hidden_size=hidden_size,
            stat_emb_size=stat_emb_size,
            dropout=dropout,
            static_noise_std=static_noise_std,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=1,
            epochs=max(1, epochs),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        model_params = list(model.parameters())
        rng = np.random.RandomState(s + fold_index)

        # Sample fixed pool of 400 sequences per basin per seed (exact Node 206 protocol)
        sb_list: list[np.ndarray] = []
        st_list: list[np.ndarray] = []
        for b, valid_t in valid_by_basin:
            sel = rng.choice(valid_t, size=min(len(valid_t), samples_per_basin), replace=False)
            sb_list.append(np.full(len(sel), b, dtype=np.int64))
            st_list.append(sel.astype(np.int64))
        sb_all = np.concatenate(sb_list)
        st_all = np.concatenate(st_list)

        for _epoch in range(epochs):
            model.train()
            perm = rng.permutation(len(sb_all))
            if n_batches_override is not None:
                perm = perm[: max(1, int(n_batches_override)) * batch_size]

            sb_gpu = torch.from_numpy(sb_all[perm]).to(device)
            st_gpu = torch.from_numpy(st_all[perm]).to(device)

            for i in range(0, len(perm), batch_size):
                bb = sb_gpu[i : i + batch_size]
                tt = st_gpu[i : i + batch_size]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    y_hat = model(xd_g[bb.unsqueeze(1), tt.unsqueeze(1) + offsets], xs_g[bb], augment=True)
                    loss = scaled_mse_loss(y_hat, y_g[bb, tt], scales_g[bb])

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model_params, 1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

        model.eval()
        pred_seed = np.zeros((n_test_basins, n_eval_days), dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for b_start in range(0, n_test_basins, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_test_basins)
                p = model(xd_eval_g[b_start:b_end], xs_eval_g[b_start:b_end], augment=False, return_seq=True)
                p_mm = torch.clamp(p[:, spinup_days:], min=0.0).float().cpu().numpy()
                pred_seed[b_start:b_end] = p_mm.astype(np.float32)
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
