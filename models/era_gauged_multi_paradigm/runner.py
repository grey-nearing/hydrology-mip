"""Fold execution interface for the ERA Gauged Multi-Paradigm (Newman CV) winning model."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.camels_data import load_camels_slice
from models.era_gauged_multi_paradigm.model import (
    CausalAtrousGRU,
    CausalConvGRULSTM,
    DualPathwayFiLMGRULSTM,
    EntityAwareHydroLSTM,
    MultiScaleFiLMEALSTM,
    ResidualAtrousGRU,
)

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "batch_size": 128,
    "lr": 1.2e-3,
    "samples_per_basin": 25,
    "epochs_paradigm_a": 45,
    "epochs_paradigm_b": 60,
    "n_batches_per_epoch": None,
    "eval_batch_size": 64,
    "family_weights": (0.38, 0.12, 0.36, 0.14),
    "device": None,
    "return_members": False,
    "fold_id": None,
    "fold_index": None,
}


def _causal_rolling(x: np.ndarray, w: int) -> np.ndarray:
    res = np.zeros_like(x, dtype=np.float32)
    cs = np.cumsum(x, axis=1, dtype=np.float64)
    res[:, w:] = ((cs[:, w:] - cs[:, :-w]) / float(w)).astype(np.float32)
    res[:, :w] = (cs[:, :w] / np.arange(1, w + 1, dtype=np.float64)[None, :]).astype(np.float32)
    return res


def _compute_3forcing_snowpack(
    x_dyn: np.ndarray, tc: float = 0.5, mf: float = 3.5
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous degree-day snowpack & liquid water delivery across all forcing products."""
    n_b, n_t, n_d = x_dyn.shape
    n_prods = max(1, n_d // 5)
    p_idx = [5 * k for k in range(n_prods)]
    tmax_idx = [5 * k + 2 for k in range(n_prods)]
    tmin_idx = [5 * k + 3 for k in range(n_prods)]
    p_3f = x_dyn[:, :, p_idx].astype(np.float32)
    t_3f = (0.5 * (x_dyn[:, :, tmax_idx] + x_dyn[:, :, tmin_idx])).astype(np.float32)
    swe = np.zeros((n_b, n_prods), dtype=np.float32)
    liq_out = np.zeros((n_b, n_t, n_prods), dtype=np.float32)
    swe_out = np.zeros((n_b, n_t, n_prods), dtype=np.float32)
    for t in range(n_t):
        pt = p_3f[:, t, :]
        tt = t_3f[:, t, :]
        snow_frac = 1.0 / (1.0 + np.exp((tt - tc) * 1.5))
        snowfall = pt * snow_frac
        rainfall = pt * (1.0 - snow_frac)
        pot_melt = np.maximum(0.0, mf * (tt - tc))
        act_melt = np.minimum(swe, pot_melt)
        swe = swe + snowfall - act_melt
        liq_out[:, t, :] = rainfall + act_melt
        swe_out[:, t, :] = np.log1p(np.maximum(0.0, swe))
    return liq_out, swe_out


def build_rich_60_features(x_dyn: np.ndarray, x_aux: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Assemble 60 multi-scale hydroclimatic, rolling water-balance, and cross-forcing features."""
    n_b, _, n_d = x_dyn.shape
    n_prods = max(1, n_d // 5)
    p_all = x_dyn[:, :, [5 * k for k in range(n_prods)]]
    sr_all = x_dyn[:, :, [5 * k + 1 for k in range(n_prods)]]
    tmax_all = x_dyn[:, :, [5 * k + 2 for k in range(n_prods)]]
    tmin_all = x_dyn[:, :, [5 * k + 3 for k in range(n_prods)]]
    vp_all = x_dyn[:, :, [5 * k + 4 for k in range(n_prods)]]
    tmean_all = 0.5 * (tmax_all + tmin_all)
    tdiurnal_all = tmax_all - tmin_all

    dayl_all = x_aux[:, :, [2 * k for k in range(n_prods)]]
    swe_all = x_aux[:, :, [2 * k + 1 for k in range(n_prods)]]

    p_mean = p_all.mean(axis=-1)
    p_std = p_all.std(axis=-1)
    t_mean = tmean_all.mean(axis=-1)
    t_std = tmean_all.std(axis=-1)
    sr_mean = sr_all.mean(axis=-1)
    vp_mean = vp_all.mean(axis=-1)
    swe_mean = swe_all.mean(axis=-1)
    dayl_mean = dayl_all.mean(axis=-1)

    pet_proxy = np.maximum(
        0.0, sr_mean * np.maximum(0.0, t_mean + 5.0) * (dayl_mean / 43200.0) * 0.002
    )
    wb_daily = p_mean - pet_proxy

    feats = [
        x_dyn,
        x_aux,
        tdiurnal_all,
        p_mean[..., None],
        p_std[..., None],
        t_mean[..., None],
        t_std[..., None],
        sr_mean[..., None],
        vp_mean[..., None],
        swe_mean[..., None],
        pet_proxy[..., None],
        np.log1p(np.maximum(0.0, p_mean))[..., None],
        np.log1p(np.maximum(0.0, swe_mean))[..., None],
    ]
    for w in (3, 7, 15, 30, 60, 120, 365):
        feats.append(_causal_rolling(p_mean, w)[..., None])
    for w in (7, 30, 90):
        feats.append(_causal_rolling(t_mean, w)[..., None])
        feats.append(_causal_rolling(wb_daily, w)[..., None])
    for k in range(n_prods):
        pk = p_all[:, :, k]
        for w in (7, 30, 90):
            feats.append(_causal_rolling(pk, w)[..., None])

    dates_dt = pd.to_datetime(dates)
    doy = dates_dt.dayofyear.values.astype(np.float32)
    for mult in (1.0, 2.0):
        s = np.sin(2.0 * np.pi * mult * doy / 365.25).astype(np.float32)
        c = np.cos(2.0 * np.pi * mult * doy / 365.25).astype(np.float32)
        feats.append(np.tile(s, (n_b, 1))[..., None])
        feats.append(np.tile(c, (n_b, 1))[..., None])
    return np.concatenate(feats, axis=-1).astype(np.float32)


def build_clean_36_features(x_dyn: np.ndarray, x_aux: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Assemble 30 canonical features + 6 continuous 3-forcing degree-day snowpack features = 36 features."""
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
    p_7d = _causal_rolling(p_mean, 7)
    p_30d = _causal_rolling(p_mean, 30)
    liq_3f, log_swe_3f = _compute_3forcing_snowpack(x_dyn)

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
            liq_3f,
            log_swe_3f,
        ],
        axis=-1,
    ).astype(np.float32)


def _infer_cont_and_win(
    eval_fn: Callable[[int, int, int, int], np.ndarray],
    n_basins: int,
    n_total_days: int,
    eval_batch_size: int = 64,
) -> np.ndarray:
    pred_cont = np.zeros((n_basins, n_total_days), dtype=np.float32)
    for b0 in range(0, n_basins, eval_batch_size):
        b1 = min(b0 + eval_batch_size, n_basins)
        pred_cont[b0:b1] = eval_fn(b0, b1, 0, n_total_days)

    pred_win_sum = np.zeros((n_basins, n_total_days), dtype=np.float64)
    pred_win_cnt = np.zeros((n_basins, n_total_days), dtype=np.float64)
    win_len = min(730, n_total_days)
    spin_len = min(365, win_len // 2)
    starts = list(range(0, max(1, n_total_days - win_len + 1), 182))
    if starts[-1] != max(0, n_total_days - win_len):
        starts.append(max(0, n_total_days - win_len))
    for t0 in starts:
        t1 = min(n_total_days, t0 + win_len)
        for b0 in range(0, n_basins, eval_batch_size):
            b1 = min(b0 + eval_batch_size, n_basins)
            ymm = eval_fn(b0, b1, t0, t1)
            if t0 == 0:
                pred_win_sum[b0:b1, :t1] += ymm
                pred_win_cnt[b0:b1, :t1] += 1.0
            else:
                pred_win_sum[b0:b1, t0 + spin_len : t1] += ymm[:, spin_len:]
                pred_win_cnt[b0:b1, t0 + spin_len : t1] += 1.0
    pred_win = (pred_win_sum / np.maximum(pred_win_cnt, 1.0)).astype(np.float32)
    return (0.5 * pred_cont + 0.5 * pred_win).astype(np.float32)


def train_and_predict_family(
    family: str,
    seed: int,
    device: torch.device,
    x_rich_train: np.ndarray,
    x_stat_train: np.ndarray,
    x_rich_eval: np.ndarray,
    x_stat_eval: np.ndarray,
    x_clean_train: np.ndarray,
    x_stat_sorted_train: np.ndarray,
    x_clean_eval: np.ndarray,
    x_stat_sorted_eval: np.ndarray,
    y_norm_train: np.ndarray,
    y_mean: float,
    y_std: float,
    per_basin_vars: np.ndarray,
    train_mask_np: np.ndarray,
    all_b_eval: torch.Tensor,
    seq_len: int = 365,
    batch_size: int = 128,
    samples_per_basin: int = 25,
    epochs_a: int = 45,
    epochs_b: int = 60,
    lr: float = 1.2e-3,
    n_batches_override: int | None = None,
    eval_batch_size: int = 64,
    spinup_days: int = 365,
) -> np.ndarray:
    """Train a single model family for one seed and return evaluation target predictions `(n_test_basins, n_eval_target_days)`."""
    use_amp = device.type == "cuda"
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_train_basins, n_train_days, _ = x_rich_train.shape
    n_test_basins, n_eval_total_days, _ = x_rich_eval.shape

    y_gpu = torch.from_numpy(y_norm_train).to(device)
    b_var_gpu = torch.from_numpy(per_basin_vars).to(device)
    train_mask_gpu = torch.from_numpy(train_mask_np).to(device)
    seq_offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    steps_per_epoch = math.ceil((n_train_basins * samples_per_basin) / batch_size)
    if n_batches_override is not None:
        steps_per_epoch = min(steps_per_epoch, max(1, int(n_batches_override)))

    valid_starts_a = np.asarray(
        [
            t0
            for t0 in range(0, max(1, n_train_days - seq_len + 1))
            if np.sum(train_mask_np[min(n_train_days, t0 + 90) : min(n_train_days, t0 + seq_len)]) >= 1
        ],
        dtype=np.int64,
    )
    if len(valid_starts_a) == 0:
        valid_starts_a = np.array([0], dtype=np.int64)

    train_idx = np.where(train_mask_np)[0]
    t_min_b = max(0, int(train_idx[0]) - seq_len + 32) if len(train_idx) > 0 else 0
    t_max_b = max(t_min_b, int(train_idx[-1]) - seq_len) if len(train_idx) > 0 else 0

    if family in ("conv_gru_lstm", "film_ealstm", "residual_atrous_gru", "causal_atrous_gru"):
        x_d_gpu = torch.from_numpy(x_rich_train).to(device)
        x_s_gpu = torch.from_numpy(x_stat_train).to(device)
        x_d_eval_gpu = torch.from_numpy(x_rich_eval).to(device)
        x_s_eval_gpu = torch.from_numpy(x_stat_eval).to(device)
        epochs = epochs_a
        if family == "conv_gru_lstm":
            model = CausalConvGRULSTM(
                dyn_dim=x_rich_train.shape[-1],
                stat_dim=x_stat_train.shape[-1],
                num_basins=n_train_basins,
            ).to(device)
        elif family == "film_ealstm":
            model = MultiScaleFiLMEALSTM(
                dyn_dim=x_rich_train.shape[-1],
                stat_dim=x_stat_train.shape[-1],
                num_basins=n_train_basins,
            ).to(device)
        elif family == "residual_atrous_gru":
            model = ResidualAtrousGRU(
                dyn_dim=x_rich_train.shape[-1],
                stat_dim=x_stat_train.shape[-1],
                num_basins=n_train_basins,
            ).to(device)
        else:
            model = CausalAtrousGRU(
                d_in=x_rich_train.shape[-1],
                d_stat=x_stat_train.shape[-1],
            ).to(device)

        ema_model = copy.deepcopy(model)
        ema_decay = 0.992
        ema_start_epoch = max(1, int(epochs * 0.45))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=max(1, steps_per_epoch), epochs=max(1, epochs)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for ep in range(1, epochs + 1):
            model.train()
            p_b = np.repeat(np.arange(n_train_basins), samples_per_basin)
            p_t = rng.choice(valid_starts_a, size=len(p_b), replace=True)
            perm = rng.permutation(len(p_b))
            if n_batches_override is not None:
                perm = perm[: max(1, int(n_batches_override)) * batch_size]
            p_b, p_t = p_b[perm], p_t[perm]

            for i in range(0, len(p_b), batch_size):
                idx_b = torch.from_numpy(p_b[i : i + batch_size]).long().to(device)
                idx_t = torch.from_numpy(p_t[i : i + batch_size]).long().to(device)
                win_t = idx_t.unsqueeze(1) + seq_offsets

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    y_hat = model(x_d_gpu[idx_b.unsqueeze(1), win_t], x_s_gpu[idx_b], idx_b)
                    y_true = y_gpu[idx_b.unsqueeze(1), win_t]
                    m = train_mask_gpu[win_t] & (~torch.isnan(y_true))
                    warmup_cut = min(90, seq_len - 1)
                    m[:, :warmup_cut] = False
                    if bool(m.any()):
                        diff = torch.where(m, y_hat - y_true, torch.zeros_like(y_hat))
                        sq_err = diff ** 2
                        denom = (b_var_gpu[idx_b] + 0.05).unsqueeze(1)
                        norm_sq_err = sq_err / denom
                        cnt = m.float().sum(dim=1).clamp(min=1.0)
                        valid_row = m.any(dim=1)
                        traj_nse_loss = (norm_sq_err.sum(dim=1) / cnt)[valid_row]
                        delta = 2.0
                        huber_nse = (delta ** 2) * (torch.sqrt(1.0 + (traj_nse_loss / delta) ** 2) - 1.0)
                        nse_term = torch.mean(huber_nse)

                        y_true_mm = torch.clamp(y_true[m] * y_std + y_mean, min=0.0)
                        y_hat_mm = torch.clamp(y_hat[m] * y_std + y_mean, min=0.0)
                        log_term = torch.mean((torch.log1p(y_hat_mm) - torch.log1p(y_true_mm)) ** 2)
                        loss = nse_term + 0.12 * log_term

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                scheduler.step()

                if ep >= ema_start_epoch:
                    with torch.no_grad():
                        for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_cur.data, alpha=1.0 - ema_decay)
                else:
                    with torch.no_grad():
                        for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                            p_ema.data.copy_(p_cur.data)

        ema_model.eval()

        def _eval_a(b0: int, b1: int, t0: int, t1: int) -> np.ndarray:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                yh = ema_model(x_d_eval_gpu[b0:b1, t0:t1], x_s_eval_gpu[b0:b1], all_b_eval[b0:b1])
                return torch.clamp(yh * y_std + y_mean, min=0.0).float().cpu().numpy()

        full_pred = _infer_cont_and_win(_eval_a, n_test_basins, n_eval_total_days, eval_batch_size)
        return full_pred[:, spinup_days:].astype(np.float32)

    else:
        x_d_gpu = torch.from_numpy(x_clean_train).to(device)
        x_s_gpu = torch.from_numpy(x_stat_sorted_train).to(device)
        x_d_eval_gpu = torch.from_numpy(x_clean_eval).to(device)
        x_s_eval_gpu = torch.from_numpy(x_stat_sorted_eval).to(device)
        epochs = epochs_b
        if family == "snow_film_grulstm":
            model = DualPathwayFiLMGRULSTM(
                dyn_dim=x_clean_train.shape[-1], stat_dim=x_stat_sorted_train.shape[-1]
            ).to(device)
        else:
            model = EntityAwareHydroLSTM(
                dyn_dim=x_clean_train.shape[-1], stat_dim=x_stat_sorted_train.shape[-1]
            ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=max(1, steps_per_epoch), epochs=max(1, epochs)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for _ep in range(epochs):
            model.train()
            p_b = np.repeat(np.arange(n_train_basins), samples_per_basin)
            p_t = rng.integers(t_min_b, t_max_b + 1, size=len(p_b))
            perm = rng.permutation(len(p_b))
            if n_batches_override is not None:
                perm = perm[: max(1, int(n_batches_override)) * batch_size]
            p_b, p_t = p_b[perm], p_t[perm]

            for i in range(0, len(p_b), batch_size):
                idx_b = torch.from_numpy(p_b[i : i + batch_size]).long().to(device)
                idx_t = torch.from_numpy(p_t[i : i + batch_size]).long().to(device)
                win_t = idx_t.unsqueeze(1) + seq_offsets

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    y_hat = model(x_d_gpu[idx_b.unsqueeze(1), win_t], x_s_gpu[idx_b])
                    y_true = y_gpu[idx_b.unsqueeze(1), win_t]
                    m = train_mask_gpu[win_t] & (~torch.isnan(y_true))
                    if bool(m.any()):
                        nse_loss = torch.mean(
                            ((y_hat[m] - y_true[m]) ** 2)
                            / (b_var_gpu[idx_b].unsqueeze(1).expand(-1, seq_len)[m] + 0.05)
                        )
                        y_true_unnorm = torch.clamp(y_true[m] * y_std + y_mean, min=0.0)
                        y_hat_unnorm = torch.clamp(y_hat[m] * y_std + y_mean, min=0.0)
                        log_loss = torch.mean(
                            (torch.log1p(y_hat_unnorm) - torch.log1p(y_true_unnorm)) ** 2
                        )
                        loss = nse_loss + 0.10 * log_loss

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                scheduler.step()

        model.eval()

        def _eval_b(b0: int, b1: int, t0: int, t1: int) -> np.ndarray:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                yh = model(x_d_eval_gpu[b0:b1, t0:t1], x_s_eval_gpu[b0:b1])
                return torch.clamp(yh * y_std + y_mean, min=0.0).float().cpu().numpy()

        full_pred = _infer_cont_and_win(_eval_b, n_test_basins, n_eval_total_days, eval_batch_size)
        return full_pred[:, spinup_days:].astype(np.float32)


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
    """Train ERA Gauged Multi-Paradigm on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_gauged_multi_paradigm: {unknown}")
    cfg = dict(RUNNER_DEFAULTS)
    cfg.update(kwargs)

    seq_len = int(cfg["seq_len"])
    spinup_days = int(cfg["spinup_days"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    samples_per_basin = int(cfg["samples_per_basin"])
    epochs_a = int(cfg["epochs_paradigm_a"])
    epochs_b = int(cfg["epochs_paradigm_b"])
    n_batches_override = cfg["n_batches_per_epoch"]
    eval_batch_size = int(cfg["eval_batch_size"])
    w_conv, w_film, w_sfilm, w_sea = [float(x) for x in cfg["family_weights"]]  # type: ignore[misc]
    return_members = bool(cfg["return_members"])

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

    x_dyn_train = np.asarray(train_slice["x_dyn"], dtype=np.float32)
    x_aux_train = np.asarray(train_slice["x_aux"], dtype=np.float32)
    dates_train = np.asarray(train_slice["dates"])

    x_dyn_eval = np.asarray(eval_slice["x_dyn"], dtype=np.float32)
    x_aux_eval = np.asarray(eval_slice["x_aux"], dtype=np.float32)
    dates_eval = np.asarray(eval_slice["dates"])

    x_rich_raw_train = build_rich_60_features(x_dyn_train, x_aux_train, dates_train)
    x_rich_raw_eval = build_rich_60_features(x_dyn_eval, x_aux_eval, dates_eval)

    x_clean_raw_train = build_clean_36_features(x_dyn_train, x_aux_train, dates_train)
    x_clean_raw_eval = build_clean_36_features(x_dyn_eval, x_aux_eval, dates_eval)

    stat_names = np.asarray(train_slice["static_attribute_names"], dtype=str)
    stat_order = np.argsort(stat_names)

    x_stat_raw_train = np.nan_to_num(np.asarray(train_slice["x_stat"], dtype=np.float32), nan=0.0)
    x_stat_raw_eval = np.nan_to_num(np.asarray(eval_slice["x_stat"], dtype=np.float32), nan=0.0)
    x_stat_sorted_raw_train = x_stat_raw_train[:, stat_order]
    x_stat_sorted_raw_eval = x_stat_raw_eval[:, stat_order]

    r_mean = x_rich_raw_train[:, spinup_days:, :].mean(axis=(0, 1), keepdims=True)
    r_std = x_rich_raw_train[:, spinup_days:, :].std(axis=(0, 1), keepdims=True) + 1e-6
    x_rich_train = ((x_rich_raw_train - r_mean) / r_std).astype(np.float32)
    x_rich_eval = ((x_rich_raw_eval - r_mean) / r_std).astype(np.float32)

    c_mean = x_clean_raw_train[:, spinup_days:, :].mean(axis=(0, 1), keepdims=True)
    c_std = x_clean_raw_train[:, spinup_days:, :].std(axis=(0, 1), keepdims=True) + 1e-6
    x_clean_train = ((x_clean_raw_train - c_mean) / c_std).astype(np.float32)
    x_clean_eval = ((x_clean_raw_eval - c_mean) / c_std).astype(np.float32)

    s_mean = x_stat_raw_train.mean(axis=0, keepdims=True)
    s_std = x_stat_raw_train.std(axis=0, keepdims=True) + 1e-6
    x_stat_train = ((x_stat_raw_train - s_mean) / s_std).astype(np.float32)
    x_stat_eval = ((x_stat_raw_eval - s_mean) / s_std).astype(np.float32)

    ss_mean = x_stat_sorted_raw_train.mean(axis=0, keepdims=True)
    ss_std = x_stat_sorted_raw_train.std(axis=0, keepdims=True) + 1e-6
    x_stat_sorted_train = ((x_stat_sorted_raw_train - ss_mean) / ss_std).astype(np.float32)
    x_stat_sorted_eval = ((x_stat_sorted_raw_eval - ss_mean) / ss_std).astype(np.float32)

    qobs_train = np.asarray(train_slice["qobs"], dtype=np.float32)
    target_qobs = np.asarray(train_slice["target_qobs"], dtype=np.float32)
    y_mean = float(np.nanmean(target_qobs))
    y_std = float(np.nanstd(target_qobs)) + 1e-6
    y_norm_train = ((qobs_train - y_mean) / y_std).astype(np.float32)

    n_train_basins, n_train_days, _ = x_rich_train.shape
    per_basin_vars = np.ones(n_train_basins, dtype=np.float32)
    for b in range(n_train_basins):
        vb = target_qobs[b, ~np.isnan(target_qobs[b])]
        if len(vb) > 10:
            per_basin_vars[b] = float(np.var(vb / y_std))

    train_mask_np = np.zeros(n_train_days, dtype=bool)
    train_mask_np[spinup_days:] = True

    train_basin_to_idx = {str(b).zfill(8): i for i, b in enumerate(train_basins)}
    eval_basin_indices = [train_basin_to_idx.get(str(b).zfill(8), 0) for b in test_basins]
    all_b_eval = torch.tensor(eval_basin_indices, dtype=torch.long, device=device)

    members: list[np.ndarray] = []
    for seed in seeds:
        s = int(seed)
        p_conv = train_and_predict_family(
            "conv_gru_lstm",
            s,
            device,
            x_rich_train,
            x_stat_train,
            x_rich_eval,
            x_stat_eval,
            x_clean_train,
            x_stat_sorted_train,
            x_clean_eval,
            x_stat_sorted_eval,
            y_norm_train,
            y_mean,
            y_std,
            per_basin_vars,
            train_mask_np,
            all_b_eval,
            seq_len=seq_len,
            batch_size=batch_size,
            samples_per_basin=samples_per_basin,
            epochs_a=epochs_a,
            epochs_b=epochs_b,
            lr=lr,
            n_batches_override=int(n_batches_override) if n_batches_override is not None else None,
            eval_batch_size=eval_batch_size,
            spinup_days=spinup_days,
        )
        p_film = train_and_predict_family(
            "film_ealstm",
            s,
            device,
            x_rich_train,
            x_stat_train,
            x_rich_eval,
            x_stat_eval,
            x_clean_train,
            x_stat_sorted_train,
            x_clean_eval,
            x_stat_sorted_eval,
            y_norm_train,
            y_mean,
            y_std,
            per_basin_vars,
            train_mask_np,
            all_b_eval,
            seq_len=seq_len,
            batch_size=batch_size,
            samples_per_basin=samples_per_basin,
            epochs_a=epochs_a,
            epochs_b=epochs_b,
            lr=lr,
            n_batches_override=int(n_batches_override) if n_batches_override is not None else None,
            eval_batch_size=eval_batch_size,
            spinup_days=spinup_days,
        )
        p_sfilm = train_and_predict_family(
            "snow_film_grulstm",
            s,
            device,
            x_rich_train,
            x_stat_train,
            x_rich_eval,
            x_stat_eval,
            x_clean_train,
            x_stat_sorted_train,
            x_clean_eval,
            x_stat_sorted_eval,
            y_norm_train,
            y_mean,
            y_std,
            per_basin_vars,
            train_mask_np,
            all_b_eval,
            seq_len=seq_len,
            batch_size=batch_size,
            samples_per_basin=samples_per_basin,
            epochs_a=epochs_a,
            epochs_b=epochs_b,
            lr=lr,
            n_batches_override=int(n_batches_override) if n_batches_override is not None else None,
            eval_batch_size=eval_batch_size,
            spinup_days=spinup_days,
        )
        p_sea = train_and_predict_family(
            "snow_ealstm",
            s,
            device,
            x_rich_train,
            x_stat_train,
            x_rich_eval,
            x_stat_eval,
            x_clean_train,
            x_stat_sorted_train,
            x_clean_eval,
            x_stat_sorted_eval,
            y_norm_train,
            y_mean,
            y_std,
            per_basin_vars,
            train_mask_np,
            all_b_eval,
            seq_len=seq_len,
            batch_size=batch_size,
            samples_per_basin=samples_per_basin,
            epochs_a=epochs_a,
            epochs_b=epochs_b,
            lr=lr,
            n_batches_override=int(n_batches_override) if n_batches_override is not None else None,
            eval_batch_size=eval_batch_size,
            spinup_days=spinup_days,
        )
        pred_seed = (w_conv * p_conv + w_film * p_film + w_sfilm * p_sfilm + w_sea * p_sea).astype(
            np.float32
        )
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
