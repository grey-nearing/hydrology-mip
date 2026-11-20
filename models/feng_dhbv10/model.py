# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Feng et al. (2022, 2023) Differentiable HBV 1.0 (dHBV 1.0) Architecture & Trainer.

References:
- Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable,
  regionalized process-based models with multiphysical outputs can approach
  state-of-the-art hydrologic prediction accuracy. Water Resources Research,
  58(10), e2022WR032404.
- Feng, D., Beck, H., Lawson, K., & Shen, C. (2023). The suitability of
  differentiable, physics-informed machine learning hydrologic models for
  ungauged regions and climate change impact assessment. Hydrology and Earth
  System Sciences, 27(12), 2357-2373.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models._dhbv_common.core import (
    CudnnLstm,
    HbvCoreFunction,
    StandaloneDHBV,
    uh_conv_fast,
    uh_gamma_fast,
)

__all__ = [
    "CudnnLstm",
    "DHBV10Trainer",
    "DHBVTrainer",
    "HbvCoreFunction",
    "NseBatchLoss",
    "RmseCombLoss",
    "StandaloneDHBV",
    "compute_hargreaves_pet",
    "uh_conv_fast",
    "uh_gamma_fast",
]

LOGGER = logging.getLogger(__name__)

# Training raises if more than this fraction of minibatches produce a non-finite loss.
MAX_NONFINITE_LOSS_FRACTION = 0.01


def _require_finite(name: str, arr: np.ndarray) -> None:
    """Raise if `arr` contains NaN/Inf (``camels_data`` guarantees gap-free forcings and attributes)."""
    n_bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    if n_bad > 0:
        raise ValueError(f"{name} contains {n_bad} non-finite values; inputs must be gap-free.")


def compute_hargreaves_pet(
    tmax: np.ndarray,
    tmin: np.ndarray,
    lat_deg: np.ndarray,
    doy: np.ndarray,
) -> np.ndarray:
    """Compute daily Potential Evapotranspiration (mm/day) via Hargreaves (1985).

    Args:
        tmax: Daily maximum temperature in degC, shape (N, T).
        tmin: Daily minimum temperature in degC, shape (N, T).
        lat_deg: Basin gauge latitude in decimal degrees, shape (N,).
        doy: Day of year (1..366), shape (T,).
    """
    tmean = 0.5 * (tmax + tmin)
    trange = np.maximum(0.0, tmax - tmin)
    lat_clipped = np.clip(lat_deg.astype(np.float64), -66.5, 66.5)
    lat_rad = np.deg2rad(lat_clipped)[:, None]
    j = doy.astype(np.float64)[None, :]
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * j)
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * j - 1.39)
    arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(arg)
    ra = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    )
    ra = np.maximum(0.0, ra)
    pet = 0.408 * 0.0023 * ra * np.maximum(0.0, tmean + 17.8) * np.sqrt(trange)
    return np.maximum(0.0, pet).astype(np.float32)


def _derive_physical_p_t_pet(
    x_dyn: np.ndarray,
    lat_deg: Optional[np.ndarray] = None,
    doy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive physical (P, Tmean, PET) in mm/day and degC from CAMELS forcing channels.

    Supports single-forcing (D=5) and multi-forcing (D=5*k) inputs where each
    5-channel block is [prcp(mm/day), srad(W/m2), tmax(C), tmin(C), vp(Pa)].
    """
    n_channels = x_dyn.shape[2]
    if n_channels < 5 or n_channels % 5 != 0:
        raise ValueError(f"Expected 5 channels per forcing product, got {n_channels} channels.")
    if lat_deg is None or doy is None or len(doy) != x_dyn.shape[1]:
        raise ValueError("lat_deg and a doy array matching the time axis are required for Hargreaves PET.")
    _require_finite("x_dyn", x_dyn)
    prcp = np.maximum(0.0, np.mean(x_dyn[:, :, 0::5], axis=2))
    tmax = np.mean(x_dyn[:, :, 2::5], axis=2)
    tmin = np.mean(x_dyn[:, :, 3::5], axis=2)
    tmean = 0.5 * (tmax + tmin)
    pet = compute_hargreaves_pet(tmax, tmin, lat_deg, doy)

    return prcp.astype(np.float32), tmean.astype(np.float32), pet.astype(np.float32)


def _transform_lstm_forcings(x_dyn: np.ndarray) -> np.ndarray:
    """Apply Feng et al. (2022) log10(sqrt(prcp) + 0.1) transform to precip channels."""
    _require_finite("x_dyn", x_dyn)
    x_nn = x_dyn.astype(np.float32, copy=True)
    for ch in range(0, x_nn.shape[2], 5):
        x_nn[:, :, ch] = np.log10(np.sqrt(np.maximum(0.0, x_nn[:, :, ch])) + 0.1)
    return x_nn


class RmseCombLoss(nn.Module):
    """Combined RMSE and root-transformed RMSE loss from Feng et al. (2022)."""

    def __init__(self, alpha: float = 0.25, beta: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_sim: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(y_obs) & (y_obs >= 0.0)
        if not mask.any():
            return (y_sim * 0.0).sum()
        yp = y_sim[mask]
        yt = y_obs[mask]
        loss1 = torch.sqrt(torch.mean((yp - yt) ** 2) + 1e-12)
        yp_trans = torch.log10(torch.sqrt(torch.clamp(yp, min=0.0) + self.beta) + 0.1)
        yt_trans = torch.log10(torch.sqrt(torch.clamp(yt, min=0.0) + self.beta) + 0.1)
        loss2 = torch.sqrt(torch.mean((yp_trans - yt_trans) ** 2) + 1e-12)
        return (1.0 - self.alpha) * loss1 + self.alpha * loss2


class NseBatchLoss(nn.Module):
    """Basin-variance normalized squared error loss (Kratzert 2019 / Song 2024)."""

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_sim: torch.Tensor,
        y_obs: torch.Tensor,
        std_batch: torch.Tensor,
    ) -> torch.Tensor:
        mask = ~torch.isnan(y_obs) & (y_obs >= 0.0)
        if not mask.any():
            return (y_sim * 0.0).sum()
        sq_res = (y_sim[mask] - y_obs[mask]) ** 2
        denom = (std_batch[mask] + self.eps) ** 2
        return torch.mean(sq_res / denom)


class DHBV10Trainer:
    """End-to-end trainer and evaluator for Feng et al. (2022, 2023) dHBV 1.0."""

    def __init__(self, device: torch.device, seed: int = 42) -> None:
        self.device = device
        self.seed = seed

    def fit_and_predict(
        self,
        x_dyn_train: np.ndarray,
        x_stat_train: np.ndarray,
        y_obs_train: np.ndarray,
        x_dyn_test: np.ndarray,
        x_stat_test: np.ndarray,
        eval_start_idx: int,
        epochs: int = 50,
        batch_size: int = 100,
        rho: int = 365,
        warmup: int = 365,
        hidden_size: int = 256,
        nmul: int = 16,
        dropout: float = 0.5,
        lr: float = 1.0,
        spinup_days: Optional[int] = None,
        train_doy: Optional[np.ndarray] = None,
        test_doy: Optional[np.ndarray] = None,
        lat_train: Optional[np.ndarray] = None,
        lat_test: Optional[np.ndarray] = None,
        loss_type: str = "rmse_comb",
        n_batches_per_epoch: Optional[int] = None,
    ) -> np.ndarray:
        """Train dHBV 1.0 strictly on post-spinup training data and predict test period."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        n_train, t_train, _ = x_dyn_train.shape
        n_test, t_test, _ = x_dyn_test.shape
        s_dim = x_stat_train.shape[1]

        if spinup_days is None:
            spinup_days = warmup
        eff_spinup = spinup_days if t_train > spinup_days else 0

        # Mask pre-train_start warmup observations so they can never leak into loss or stats
        y_train_clean = y_obs_train.astype(np.float32, copy=True)
        if eff_spinup > 0:
            y_train_clean[:, :eff_spinup] = np.nan

        p_tr, t_tr, pet_tr = _derive_physical_p_t_pet(
            x_dyn_train, lat_deg=lat_train, doy=train_doy
        )
        p_te, t_te, pet_te = _derive_physical_p_t_pet(
            x_dyn_test, lat_deg=lat_test, doy=test_doy
        )
        x_phy_tr = np.stack([p_tr, t_tr, pet_tr], axis=-1)
        x_phy_te = np.stack([p_te, t_te, pet_te], axis=-1)

        x_nn_tr_raw = np.concatenate(
            [_transform_lstm_forcings(x_dyn_train), x_phy_tr[:, :, 1:3]], axis=2
        )
        x_nn_te_raw = np.concatenate(
            [_transform_lstm_forcings(x_dyn_test), x_phy_te[:, :, 1:3]], axis=2
        )
        d_nn_dim = x_nn_tr_raw.shape[2]

        # Normalization statistics computed strictly on post-spinup training period
        xd_mean = np.nanmean(x_nn_tr_raw[:, eff_spinup:, :], axis=(0, 1), keepdims=True)
        xd_std = np.maximum(
            np.nanstd(x_nn_tr_raw[:, eff_spinup:, :], axis=(0, 1), keepdims=True), 1e-3
        )
        xs_mean = np.nanmean(x_stat_train, axis=0, keepdims=True)
        xs_std = np.maximum(np.nanstd(x_stat_train, axis=0, keepdims=True), 1e-3)

        _require_finite("x_stat_train", x_stat_train)
        _require_finite("x_stat_test", x_stat_test)
        xd_tr = ((x_nn_tr_raw - xd_mean) / xd_std).astype(np.float32)
        xd_te = ((x_nn_te_raw - xd_mean) / xd_std).astype(np.float32)
        xs_tr = ((x_stat_train - xs_mean) / xs_std).astype(np.float32)
        xs_te = ((x_stat_test - xs_mean) / xs_std).astype(np.float32)

        std_tr = np.zeros(n_train, dtype=np.float32)
        for i in range(n_train):
            yq = y_train_clean[i, eff_spinup:]
            valid = ~np.isnan(yq) & (yq >= 0.0)
            std_tr[i] = max(float(np.std(yq[valid])), 1e-3) if np.sum(valid) > 1 else 1.0

        eff_warmup = min(warmup, eff_spinup if eff_spinup > 0 else max(1, t_train // 4))
        min_target_idx = max(eff_warmup, eff_spinup)
        eff_rho = min(rho, max(2, t_train - min_target_idx))
        if min_target_idx + eff_rho > t_train:
            min_target_idx = eff_warmup
            eff_rho = max(2, t_train - min_target_idx)

        model = StandaloneDHBV(
            nx=d_nn_dim + s_dim,
            version="1_0",
            hidden_size=hidden_size,
            dr=dropout,
            nmul=nmul,
            warmup=eff_warmup,
        ).to(self.device)
        del lr
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
        rmse_criterion = RmseCombLoss(alpha=0.25, beta=1e-6)
        nse_criterion = NseBatchLoss(eps=0.1)

        valid_basins = [
            i
            for i in range(n_train)
            if np.any(~np.isnan(y_train_clean[i, min_target_idx:]))
        ]
        if not valid_basins:
            valid_basins = list(range(n_train))

        if n_batches_per_epoch is not None and int(n_batches_per_epoch) > 0:
            n_iter = int(n_batches_per_epoch)
        else:
            n_eff_days = max(1, t_train - min_target_idx)
            ratio = (batch_size * eff_rho) / max(1.0, float(len(valid_basins) * n_eff_days))
            if ratio < 0.99:
                n_iter = max(1, int(math.ceil(math.log(0.01) / math.log(1.0 - ratio))))
            else:
                n_iter = max(1, len(valid_basins) // max(1, batch_size))
            if len(valid_basins) <= 4:
                n_iter = min(n_iter, 4)

        xd_tr_g = torch.from_numpy(xd_tr).to(self.device)
        xs_tr_g = torch.from_numpy(xs_tr).to(self.device)
        xp_tr_g = torch.from_numpy(x_phy_tr).to(self.device)
        yo_tr_g = torch.from_numpy(y_train_clean).to(self.device)
        st_tr_g = torch.from_numpy(std_tr).to(self.device)
        seq_tot = eff_warmup + eff_rho
        seq_offs = torch.arange(seq_tot, device=self.device, dtype=torch.long)
        rho_offs = torch.arange(eff_rho, device=self.device, dtype=torch.long)
        max_start = max(min_target_idx, t_train - eff_rho)

        model.train()
        n_nonfinite_loss = 0
        n_loss_total = 0
        for _ in range(epochs):
            for _ in range(n_iter):
                b_idx = np.random.choice(
                    valid_basins,
                    size=min(batch_size, len(valid_basins)),
                    replace=False,
                )
                b_act = len(b_idx)
                it_list = [
                    np.random.randint(min_target_idx, max_start + 1)
                    if max_start > min_target_idx
                    else min_target_idx
                    for _ in range(b_act)
                ]
                t0_list = [i_t - eff_warmup for i_t in it_list]
                b_idx_t = torch.from_numpy(np.asarray(b_idx, dtype=np.int64)).to(self.device)
                t0_t = torch.from_numpy(np.asarray(t0_list, dtype=np.int64)).to(self.device)
                it_t = torch.from_numpy(np.asarray(it_list, dtype=np.int64)).to(self.device)
                t_full_idx = t0_t[None, :] + seq_offs[:, None]
                t_rho_idx = it_t[None, :] + rho_offs[:, None]
                b_grid_full = b_idx_t[None, :].expand(seq_tot, b_act)
                b_grid_rho = b_idx_t[None, :].expand(eff_rho, b_act)

                xd_sub = xd_tr_g[b_grid_full, t_full_idx]
                xs_sub = xs_tr_g[b_idx_t].unsqueeze(0).expand(seq_tot, b_act, s_dim)
                xc_t = torch.cat([xd_sub, xs_sub], dim=-1).contiguous()
                xp_t = xp_tr_g[b_grid_full, t_full_idx].contiguous()
                yo_t = yo_tr_g[b_grid_rho, t_rho_idx].contiguous()
                st_t = st_tr_g[b_idx_t].unsqueeze(0).expand(eff_rho, b_act).contiguous()

                q_sim = model(xc_t, xp_t)
                if loss_type == "nse_batch":
                    loss = nse_criterion(q_sim, yo_t, st_t)
                else:
                    loss = rmse_criterion(q_sim, yo_t)
                n_loss_total += 1
                if not torch.isfinite(loss):
                    n_nonfinite_loss += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        if n_nonfinite_loss > 0:
            LOGGER.warning(
                "Skipped %d of %d minibatches with a non-finite loss (seed %d).",
                n_nonfinite_loss,
                n_loss_total,
                self.seed,
            )
        if n_nonfinite_loss > MAX_NONFINITE_LOSS_FRACTION * n_loss_total:
            raise RuntimeError(
                f"{n_nonfinite_loss}/{n_loss_total} minibatches produced a non-finite loss "
                f"(> {MAX_NONFINITE_LOSS_FRACTION:.0%}); training diverged."
            )

        model.eval()
        model.warmup = eval_start_idx if t_test > eval_start_idx else 0
        preds = np.zeros((n_test, t_test - eval_start_idx), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                sl = slice(i, min(i + batch_size, n_test))
                b_sz = sl.stop - sl.start
                xd_ev = torch.from_numpy(np.transpose(xd_te[sl], (1, 0, 2))).to(self.device)
                xs_ev = (
                    torch.from_numpy(xs_te[sl])
                    .to(self.device)
                    .unsqueeze(0)
                    .expand(t_test, b_sz, s_dim)
                )
                xc_ev = torch.cat([xd_ev, xs_ev], dim=-1).contiguous()
                xp_ev = torch.from_numpy(np.transpose(x_phy_te[sl], (1, 0, 2))).to(self.device).contiguous()
                q_out = model(xc_ev, xp_ev)
                q_np = q_out.cpu().numpy().T
                # NaN/Inf propagate so run_model's non-finite guard rejects a diverged model.
                preds[sl] = np.maximum(0.0, q_np)
        return preds


DHBVTrainer = DHBV10Trainer

