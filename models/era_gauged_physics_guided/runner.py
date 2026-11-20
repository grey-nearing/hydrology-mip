# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fold execution interface for the ERA Physics-Guided Gauged winning model (Node 187 — PhysicsHybridEALSTM)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.camels_data import load_camels_slice
from models.era_gauged_physics_guided.model import PhysicsHybridEALSTM

RUNNER_DEFAULTS: dict[str, object] = {
    "seq_len": 365,
    "spinup_days": 365,
    "warmup_len": 180,
    "hidden_size": 256,
    "embed_dim": 64,
    "dropout": 0.2,
    "batch_size": 128,
    "lr": 1.5e-3,
    "weight_decay": 0.1,
    "epochs": 50,
    "samples_per_basin": 25,
    "n_batches_per_epoch": None,
    "eval_batch_size": 32,
    "device": None,
    "return_members": False,
    "fold_id": None,
    "fold_index": None,
}


def _prepare_features(x_dyn: np.ndarray, x_aux: np.ndarray) -> np.ndarray:
    """Assemble multi-scale rolling precipitation features matching Node 187."""
    n_basins, n_days, n_dyn = x_dyn.shape
    n_prods = max(1, n_dyn // 5)
    p_idx = [5 * k for k in range(n_prods)]
    feats = [x_dyn, x_aux]
    for w in (7, 30, 90):
        w_eff = min(w, n_days)
        p_roll = np.zeros((n_basins, n_days, len(p_idx)), dtype=np.float32)
        for i, idx in enumerate(p_idx):
            p_series = x_dyn[:, :, idx]
            cs = np.cumsum(np.insert(p_series, 0, 0.0, axis=1), axis=1, dtype=np.float64)
            p_roll[:, w_eff - 1 :, i] = ((cs[:, w_eff:] - cs[:, :-w_eff]) / float(w_eff)).astype(np.float32)
        feats.append(p_roll)
    return np.concatenate(feats, axis=-1).astype(np.float32)


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
    """Train ERA Physics-Guided Gauged (Node 187) on `train_basins` over `train_period` and predict `test_basins` over `eval_period`."""
    unknown = sorted(set(kwargs) - set(RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown runner kwargs for era_gauged_physics_guided: {unknown}")
    cfg = dict(RUNNER_DEFAULTS)
    cfg.update(kwargs)

    seq_len = int(cfg["seq_len"])
    spinup_days = int(cfg["spinup_days"])
    warmup_len = min(int(cfg["warmup_len"]), max(0, seq_len - 1))
    hidden_size = int(cfg["hidden_size"])
    embed_dim = int(cfg["embed_dim"])
    dropout = float(cfg["dropout"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    samples_per_basin = int(cfg["samples_per_basin"])
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

    x_dyn_train = np.asarray(train_slice["x_dyn"], dtype=np.float32)
    x_aux_train = np.asarray(train_slice["x_aux"], dtype=np.float32)
    x_dyn_eval = np.asarray(eval_slice["x_dyn"], dtype=np.float32)
    x_aux_eval = np.asarray(eval_slice["x_aux"], dtype=np.float32)

    x_d_train_raw = _prepare_features(x_dyn_train, x_aux_train)
    x_d_eval_raw = _prepare_features(x_dyn_eval, x_aux_eval)

    x_s_train_raw = np.nan_to_num(np.asarray(train_slice["x_stat"], dtype=np.float32), nan=0.0)
    x_s_eval_raw = np.nan_to_num(np.asarray(eval_slice["x_stat"], dtype=np.float32), nan=0.0)

    d_mean = x_d_train_raw[:, spinup_days:, :].mean(axis=(0, 1), keepdims=True)
    d_std = x_d_train_raw[:, spinup_days:, :].std(axis=(0, 1), keepdims=True) + 1e-6
    x_d_train_norm = torch.from_numpy(((x_d_train_raw - d_mean) / d_std).astype(np.float32)).to(device)
    x_d_eval_norm = torch.from_numpy(((x_d_eval_raw - d_mean) / d_std).astype(np.float32)).to(device)

    s_mean = x_s_train_raw.mean(axis=0, keepdims=True)
    s_std = x_s_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_s_train_norm = torch.from_numpy(((x_s_train_raw - s_mean) / s_std).astype(np.float32)).to(device)
    x_s_eval_norm = torch.from_numpy(((x_s_eval_raw - s_mean) / s_std).astype(np.float32)).to(device)

    raw_p_train = torch.from_numpy(x_dyn_train[:, :, 0].astype(np.float32)).to(device)
    raw_t_train = torch.from_numpy(((x_dyn_train[:, :, 2] + x_dyn_train[:, :, 3]) / 2.0).astype(np.float32)).to(device)
    raw_p_eval = torch.from_numpy(x_dyn_eval[:, :, 0].astype(np.float32)).to(device)
    raw_t_eval = torch.from_numpy(((x_dyn_eval[:, :, 2] + x_dyn_eval[:, :, 3]) / 2.0).astype(np.float32)).to(device)

    qobs_train = np.asarray(train_slice["qobs"], dtype=np.float32)
    target_qobs = np.asarray(train_slice["target_qobs"], dtype=np.float32)
    y_gpu = torch.from_numpy(qobs_train).to(device)

    n_train_basins, n_train_days, dyn_dim = x_d_train_norm.shape
    stat_dim = x_s_train_norm.shape[-1]
    y_std_basin = np.array(
        [float(np.nanstd(target_qobs[b])) + 0.05 for b in range(n_train_basins)],
        dtype=np.float32,
    )
    y_std_gpu = torch.from_numpy(y_std_basin).to(device)

    train_mask_np = np.zeros(n_train_days, dtype=bool)
    train_mask_np[spinup_days:] = True
    train_indices = np.where(train_mask_np)[0]
    valid_starts = train_indices[train_indices >= (seq_len - 1)]
    if len(valid_starts) == 0:
        raise ValueError(
            f"No valid training window end days: seq_len={seq_len} exceeds the post-spin-up "
            f"training slice ({n_train_days} days, spinup_days={spinup_days})."
        )

    # PhysicsHybridEALSTM learns one embedding per training basin, so it is only defined for
    # basins seen in training (gauged regime).  Unseen basins must not borrow another basin's
    # embedding.
    train_basin_to_idx = {str(b).zfill(8): i for i, b in enumerate(train_basins)}
    unseen_basins = [str(b).zfill(8) for b in test_basins if str(b).zfill(8) not in train_basin_to_idx]
    if unseen_basins:
        raise ValueError(
            "era_gauged_physics_guided uses per-training-basin embeddings and supports only the "
            f"gauged spatial regime; {len(unseen_basins)} evaluation basin(s) were not in the "
            f"training set: {unseen_basins}"
        )
    eval_basin_indices = [train_basin_to_idx[str(b).zfill(8)] for b in test_basins]
    all_b_eval = torch.tensor(eval_basin_indices, dtype=torch.long, device=device)

    n_eval_days = int(eval_slice["n_target_days"])
    n_test_basins = len(test_basins)
    seq_offsets = torch.arange(seq_len, device=device) - (seq_len - 1)
    members: list[np.ndarray] = []

    for seed in seeds:
        s = int(seed)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        model = PhysicsHybridEALSTM(
            dyn_dim=dyn_dim,
            stat_dim=stat_dim,
            num_basins=n_train_basins,
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for _epoch in range(1, epochs + 1):
            model.train()
            b_idx_arr = np.repeat(np.arange(n_train_basins), samples_per_basin)
            t_idx_arr = np.concatenate(
                [np.random.choice(valid_starts, samples_per_basin) for _ in range(n_train_basins)]
            )
            perm = np.random.permutation(len(b_idx_arr))
            if n_batches_override is not None:
                max_items = max(1, int(n_batches_override)) * batch_size
                perm = perm[:max_items]

            for i in range(0, len(perm), batch_size):
                idx = perm[i : i + batch_size]
                batch_b = torch.from_numpy(b_idx_arr[idx]).long().to(device)
                batch_t = torch.from_numpy(t_idx_arr[idx]).long().to(device)

                time_cols = batch_t.unsqueeze(1) + seq_offsets.unsqueeze(0)
                bx_d = x_d_train_norm[batch_b.unsqueeze(1), time_cols]
                bx_s = x_s_train_norm[batch_b]
                by = y_gpu[batch_b.unsqueeze(1), time_cols]
                brp = raw_p_train[batch_b.unsqueeze(1), time_cols]
                brt = raw_t_train[batch_b.unsqueeze(1), time_cols]
                bstd = y_std_gpu[batch_b].unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds = model(bx_d, bx_s, batch_b, brp, brt).squeeze(-1)
                    mask = ~torch.isnan(by)
                    mask[:, :warmup_len] = False
                    if bool(mask.any()):
                        diff = (preds[mask] - by[mask]) / bstd.expand_as(preds)[mask]
                        loss = F.mse_loss(diff, torch.zeros_like(diff))
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

            scheduler.step()

        model.eval()
        pred_seed = np.zeros((n_test_basins, n_eval_days), dtype=np.float32)
        with torch.no_grad():
            for b_start in range(0, n_test_basins, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_test_basins)
                p_out = model(
                    x_d_eval_norm[b_start:b_end],
                    x_s_eval_norm[b_start:b_end],
                    all_b_eval[b_start:b_end],
                    raw_p_eval[b_start:b_end],
                    raw_t_eval[b_start:b_end],
                ).squeeze(-1)
                pred_seed[b_start:b_end] = np.maximum(
                    p_out[:, spinup_days:].cpu().numpy().astype(np.float32),
                    0.0,
                )
        members.append(pred_seed)

    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    if return_members:
        return mean_arr, members_arr
    return mean_arr
