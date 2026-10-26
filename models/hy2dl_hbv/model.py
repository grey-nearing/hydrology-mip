"""Acuña Espinoza et al. (2024) Hy2DL Hybrid LSTM + 16-HBV Conceptual-DL Architecture.

Reference:
- Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., & Ehret, U.
  (2024). To bucket or not to bucket? Analyzing the performance and
  interpretability of hybrid hydrological models with dynamic parameterization.
  Hydrology and Earth System Sciences, 28(12), 2705-2719.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feng_dhbv10.dhbv_standalone_core import HbvCoreFunction, get_hbv_cpu
from models.poudel_mlp_hbv.model import _derive_physical_p_t_pet

HY2DL_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "BETA": (1.0, 6.0),
    "FC": (50.0, 1000.0),
    "K0": (0.05, 0.9),
    "K1": (0.01, 0.5),
    "K2": (0.001, 0.2),
    "LP": (0.2, 1.0),
    "PERC": (0.0, 10.0),
    "UZL": (0.0, 100.0),
    "TT": (-2.5, 2.5),
    "CFMAX": (0.5, 10.0),
    "CFR": (0.0, 0.1),
    "CWH": (0.0, 0.2),
    "BETAET": (0.3, 5.0),
}
HY2DL_DYNAMIC_PARAMS = ("BETA", "BETAET")


@torch.jit.script
def _hy2dl_hbv_step_loop(
    p: torch.Tensor,
    temp: torch.Tensor,
    pet: torch.Tensor,
    beta: torch.Tensor,
    fc: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    lp: torch.Tensor,
    perc_max: torch.Tensor,
    uzl: torch.Tensor,
    tt: torch.Tensor,
    cfmax: torch.Tensor,
    cfr: torch.Tensor,
    cwh: torch.Tensor,
    betaet: torch.Tensor,
    init_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable 16-bucket HBV step loop for Hy2DL (Acuña Espinoza et al. 2024).

    Args:
        p, temp, pet: (B, T)
        beta .. betaet: (B, T, M)
        init_states: (B, M, 5) -> [snow, liquid_w, sm, suz, slz]

    Returns:
        q_raw: (B, T) unrouted mean discharge across M buckets
        final_states: (B, M, 5)
    """
    t_steps = p.shape[1]
    snow = init_states[:, :, 0]
    liquid_w = init_states[:, :, 1]
    sm = init_states[:, :, 2]
    suz = init_states[:, :, 3]
    slz = init_states[:, :, 4]
    nearzero = 1e-5

    q_list: List[torch.Tensor] = []
    for t in range(t_steps):
        pt = p[:, t : t + 1]
        tempt = temp[:, t : t + 1]
        pett = pet[:, t : t + 1]

        tt_t = tt[:, t, :]
        cfmax_t = cfmax[:, t, :]
        cfr_t = cfr[:, t, :]
        cwh_t = cwh[:, t, :]
        fc_t = fc[:, t, :]
        lp_t = lp[:, t, :]
        beta_t = beta[:, t, :]
        betaet_t = betaet[:, t, :]
        perc_t = perc_max[:, t, :]
        uzl_t = uzl[:, t, :]
        k0_t = k0[:, t, :]
        k1_t = k1[:, t, :]
        k2_t = k2[:, t, :]

        is_snow = (tempt < tt_t).to(pt.dtype)
        p_snow = pt * is_snow
        p_rain = pt * (1.0 - is_snow)

        snow = snow + p_snow
        melt = torch.minimum(snow, cfmax_t * F.relu(tempt - tt_t))
        snow = snow - melt
        liquid_w = liquid_w + melt

        refreeze = torch.minimum(liquid_w, cfr_t * cfmax_t * F.relu(tt_t - tempt))
        snow = snow + refreeze
        liquid_w = liquid_w - refreeze

        tosoil = F.relu(liquid_w - cwh_t * snow)
        liquid_w = liquid_w - tosoil

        infil = p_rain + tosoil
        sm_ratio = torch.clamp(sm / (fc_t + nearzero), min=nearzero, max=1.0)
        wet = torch.clamp(sm_ratio.pow(beta_t), min=0.0, max=1.0)
        recharge = infil * wet
        sm1 = sm + infil - recharge
        excess = F.relu(sm1 - fc_t)
        sm2 = sm1 - excess

        et_ratio = torch.clamp(sm2 / (lp_t * fc_t + nearzero), min=nearzero, max=1.0)
        evapf = torch.clamp(et_ratio.pow(betaet_t), min=0.0, max=1.0)
        aet = torch.minimum(sm2, pett * evapf)
        sm = torch.clamp(sm2 - aet, min=nearzero)

        suz = suz + recharge + excess
        perc = torch.minimum(suz, perc_t)
        suz = suz - perc
        slz = slz + perc

        q0 = k0_t * F.relu(suz - uzl_t)
        suz = F.relu(suz - q0)
        q1 = k1_t * suz
        suz = F.relu(suz - q1)
        q2 = k2_t * slz
        slz = F.relu(slz - q2)

        q_list.append((q0 + q1 + q2).mean(dim=-1))

    q_raw = torch.stack(q_list, dim=1)
    final_states = torch.stack([snow, liquid_w, sm, suz, slz], dim=-1)
    return q_raw, final_states


def _hy2dl_gamma_uh(route_a: torch.Tensor, route_b: torch.Tensor, len_f: int = 15) -> torch.Tensor:
    """Compute Gamma unit hydrograph routing weights of shape (B, 1, len_f)."""
    aa = F.relu(route_a).unsqueeze(-1) + 0.1
    theta = F.relu(route_b).unsqueeze(-1) + 0.5
    t = torch.arange(0.5, len_f * 1.0, device=route_a.device, dtype=route_a.dtype).view(1, len_f)
    denom = aa.lgamma().exp() * (theta.pow(aa))
    w = (t.pow(aa - 1.0)) * torch.exp(-t / theta) / (denom + 1e-12)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
    return w.unsqueeze(1)


class Hy2DLHBVModel(nn.Module):
    """Acuña Espinoza et al. (2024) Hy2DL LSTM + 16-HBV with dynamic parameterization."""

    def __init__(
        self,
        dyn_in_dim: int = 5,
        stat_in_dim: int = 27,
        hidden_size: int = 256,
        n_conceptual_models: int = 16,
        dropout: float = 0.4,
        routing_len: int = 15,
    ) -> None:
        super().__init__()
        self.n_models = n_conceptual_models
        self.routing_len = routing_len
        self.param_names = list(HY2DL_PARAM_BOUNDS.keys())
        self.n_params = len(self.param_names)
        del dropout

        self.lstm = nn.LSTM(
            input_size=dyn_in_dim + stat_in_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        with torch.no_grad():
            self.lstm.bias_hh_l0[hidden_size : 2 * hidden_size] = 3.0
        self.linear_out = nn.Linear(hidden_size, self.n_models * self.n_params + 2)

    def _map_params(
        self, raw_phy: torch.Tensor, warmup_period: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        t_steps = raw_phy.shape[1]
        t_sim = t_steps - warmup_period
        pw: Dict[str, torch.Tensor] = {}
        ps: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.param_names):
            lo, hi = HY2DL_PARAM_BOUNDS[name]
            scale = hi - lo
            if name in HY2DL_DYNAMIC_PARAMS:
                if warmup_period > 0:
                    w_val = raw_phy[:, warmup_period - 1 : warmup_period, :, i].expand(
                        -1, warmup_period, -1
                    )
                    pw[name] = lo + scale * w_val
                ps[name] = lo + scale * raw_phy[:, warmup_period:, :, i]
            else:
                sta = lo + scale * raw_phy[:, -1:, :, i]
                if warmup_period > 0:
                    pw[name] = sta.expand(-1, warmup_period, -1)
                ps[name] = sta.expand(-1, t_sim, -1)
        return pw, ps

    def _run_hbv(
        self,
        p: torch.Tensor,
        temp: torch.Tensor,
        pet: torch.Tensor,
        pdict: Dict[str, torch.Tensor],
        init_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if p.is_cuda or get_hbv_cpu():
            pm = p.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.n_models).contiguous()
            tm = temp.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.n_models).contiguous()
            petm = pet.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.n_models).contiguous()
            init_tbm = init_states.permute(2, 0, 1).contiguous()
            zero_c = torch.zeros(1, dtype=p.dtype, device=p.device)
            q0, q1, q2, fin_tbm = HbvCoreFunction.apply(
                pm,
                tm,
                petm,
                init_tbm,
                pdict["BETA"].transpose(0, 1).contiguous(),
                pdict["FC"].transpose(0, 1).contiguous(),
                pdict["K0"].transpose(0, 1).contiguous(),
                pdict["K1"].transpose(0, 1).contiguous(),
                pdict["K2"].transpose(0, 1).contiguous(),
                pdict["LP"].transpose(0, 1).contiguous(),
                pdict["PERC"].transpose(0, 1).contiguous(),
                pdict["UZL"].transpose(0, 1).contiguous(),
                pdict["TT"].transpose(0, 1).contiguous(),
                pdict["CFMAX"].transpose(0, 1).contiguous(),
                pdict["CFR"].transpose(0, 1).contiguous(),
                pdict["CWH"].transpose(0, 1).contiguous(),
                pdict["BETAET"].transpose(0, 1).contiguous(),
                zero_c,
                1e-5,
                False,
            )
            q_raw = (q0 + q1 + q2).mean(dim=-1).transpose(0, 1)
            return q_raw, fin_tbm.permute(1, 2, 0)
        return _hy2dl_hbv_step_loop(
            p,
            temp,
            pet,
            pdict["BETA"],
            pdict["FC"],
            pdict["K0"],
            pdict["K1"],
            pdict["K2"],
            pdict["LP"],
            pdict["PERC"],
            pdict["UZL"],
            pdict["TT"],
            pdict["CFMAX"],
            pdict["CFR"],
            pdict["CWH"],
            pdict["BETAET"],
            init_states,
        )

    def forward(
        self,
        x_d: torch.Tensor,
        x_s: torch.Tensor,
        p: torch.Tensor,
        temp: torch.Tensor,
        pet: torch.Tensor,
        warmup_period: int = 0,
    ) -> torch.Tensor:
        """Run LSTM dynamic parameter generator and 16-bucket HBV + Gamma routing."""
        bsz, t_steps, _ = x_d.shape
        n_phy = self.n_models * self.n_params
        init_states = torch.full(
            (bsz, self.n_models, 5), 0.001, dtype=p.dtype, device=p.device
        )

        xs_exp = x_s.unsqueeze(1).expand(-1, t_steps, -1)
        x_in = torch.cat([x_d, xs_exp], dim=-1)
        if x_in.is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                h_out, _ = self.lstm(x_in)
                raw_lin = self.linear_out(h_out)
            raw_all = torch.sigmoid(raw_lin.float())
        else:
            h_out, _ = self.lstm(x_in)
            raw_all = torch.sigmoid(self.linear_out(h_out).float())

        raw_phy = raw_all[:, :, :n_phy].view(bsz, t_steps, self.n_models, self.n_params)
        raw_route = raw_all[:, -1, n_phy:]
        route_a = raw_route[:, 0] * 2.9
        route_b = raw_route[:, 1] * 6.5

        wp = warmup_period if (warmup_period > 0 and t_steps > warmup_period) else 0
        pdict_w, pdict_s = self._map_params(raw_phy, wp)
        if wp > 0:
            with torch.no_grad():
                _, init_states = self._run_hbv(
                    p[:, :wp],
                    temp[:, :wp],
                    pet[:, :wp],
                    pdict_w,
                    init_states,
                )
            init_states = init_states.detach()
            q_raw, _ = self._run_hbv(
                p[:, wp:],
                temp[:, wp:],
                pet[:, wp:],
                pdict_s,
                init_states,
            )
        else:
            q_raw, _ = self._run_hbv(p, temp, pet, pdict_s, init_states)

        t_sim = q_raw.shape[1]
        uh = _hy2dl_gamma_uh(route_a, route_b, len_f=self.routing_len)
        w_flip = torch.flip(uh, dims=[2])
        pad = self.routing_len - 1
        q_in = q_raw.contiguous().view(1, bsz, t_sim)
        q_routed = F.conv1d(q_in, w_flip, groups=bsz, padding=pad)
        if pad > 0:
            q_routed = q_routed[:, :, :-pad]
        return q_routed.view(bsz, t_sim)


class Hy2DLHBVTrainer:
    """Trainer and evaluator for Acuña Espinoza et al. (2024) Hy2DL LSTM + 16-HBV."""

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
        epochs: int = 20,
        batch_size: int = 256,
        rho: int = 365,
        warmup: int = 365,
        hidden_size: int = 256,
        n_conceptual_models: int = 16,
        dropout: float = 0.4,
        lr: float = 1e-3,
        spinup_days: Optional[int] = None,
        train_doy: Optional[np.ndarray] = None,
        test_doy: Optional[np.ndarray] = None,
        lat_train: Optional[np.ndarray] = None,
        lat_test: Optional[np.ndarray] = None,
        n_batches_per_epoch: Optional[int] = None,
    ) -> np.ndarray:
        """Train Hy2DL strictly on post-spinup training data and predict test period."""
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

        x_nn_tr_raw = np.concatenate(
            [x_dyn_train.astype(np.float32), t_tr[..., None], pet_tr[..., None]], axis=2
        )
        x_nn_te_raw = np.concatenate(
            [x_dyn_test.astype(np.float32), t_te[..., None], pet_te[..., None]], axis=2
        )
        d_nn_dim = x_nn_tr_raw.shape[2]

        # Normalization statistics computed strictly on post-spinup training period
        xd_mean = np.nanmean(x_nn_tr_raw[:, eff_spinup:, :], axis=(0, 1), keepdims=True)
        xd_std = np.nanstd(x_nn_tr_raw[:, eff_spinup:, :], axis=(0, 1), keepdims=True) + 1e-6
        xs_mean = np.nanmean(x_stat_train, axis=0, keepdims=True)
        xs_std = np.nanstd(x_stat_train, axis=0, keepdims=True) + 1e-6

        xd_tr = np.nan_to_num((x_nn_tr_raw - xd_mean) / xd_std, nan=0.0).astype(np.float32)
        xd_te = np.nan_to_num((x_nn_te_raw - xd_mean) / xd_std, nan=0.0).astype(np.float32)
        xs_tr = np.nan_to_num((x_stat_train - xs_mean) / xs_std, nan=0.0).astype(np.float32)
        xs_te = np.nan_to_num((x_stat_test - xs_mean) / xs_std, nan=0.0).astype(np.float32)

        eff_warmup = min(warmup, eff_spinup if eff_spinup > 0 else max(1, t_train // 4))
        min_target_idx = max(eff_warmup, eff_spinup)
        eff_rho = min(rho, max(2, t_train - min_target_idx))
        if min_target_idx + eff_rho > t_train:
            min_target_idx = eff_warmup
            eff_rho = max(2, t_train - min_target_idx)

        model = Hy2DLHBVModel(
            dyn_in_dim=d_nn_dim,
            stat_in_dim=s_dim,
            hidden_size=hidden_size,
            n_conceptual_models=n_conceptual_models,
            dropout=dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        valid_basins = [
            i
            for i in range(n_train)
            if np.any(~np.isnan(y_train_clean[i, min_target_idx:]))
        ]
        if not valid_basins:
            valid_basins = list(range(n_train))
        if n_batches_per_epoch is not None and int(n_batches_per_epoch) > 0:
            n_batches = int(n_batches_per_epoch)
        else:
            n_batches = 450
            if len(valid_basins) <= 4:
                n_batches = min(n_batches, 4)

        max_target_start = max(min_target_idx, t_train - eff_rho)
        xd_tr_g = torch.from_numpy(xd_tr).to(self.device)
        xs_tr_g = torch.from_numpy(xs_tr).to(self.device)
        p_tr_g = torch.from_numpy(p_tr).to(self.device)
        t_tr_g = torch.from_numpy(t_tr).to(self.device)
        pet_tr_g = torch.from_numpy(pet_tr).to(self.device)
        yo_tr_g = torch.from_numpy(y_train_clean).to(self.device)
        seq_tot = eff_warmup + eff_rho
        seq_offs = torch.arange(seq_tot, device=self.device, dtype=torch.long)
        rho_offs = torch.arange(eff_rho, device=self.device, dtype=torch.long)
        b_act = batch_size if len(valid_basins) > 4 else len(valid_basins)

        model.train()
        for _ in range(epochs):
            for _ in range(n_batches):
                b_idx = np.random.choice(
                    valid_basins,
                    size=b_act,
                    replace=(len(valid_basins) < b_act),
                )
                if max_target_start > min_target_idx:
                    it_arr = np.random.randint(
                        min_target_idx, max_target_start + 1, size=b_act
                    )
                else:
                    it_arr = np.full(b_act, min_target_idx, dtype=np.int64)
                t0_arr = it_arr - eff_warmup
                b_idx_t = torch.from_numpy(np.asarray(b_idx, dtype=np.int64)).to(self.device)
                t0_t = torch.from_numpy(np.asarray(t0_arr, dtype=np.int64)).to(self.device)
                it_t = torch.from_numpy(np.asarray(it_arr, dtype=np.int64)).to(self.device)

                t_full_idx = t0_t[:, None] + seq_offs[None, :]
                t_rho_idx = it_t[:, None] + rho_offs[None, :]
                b_grid_full = b_idx_t[:, None].expand(b_act, seq_tot)
                b_grid_rho = b_idx_t[:, None].expand(b_act, eff_rho)

                xd_b = xd_tr_g[b_grid_full, t_full_idx]
                xs_b = xs_tr_g[b_idx_t]
                p_b = p_tr_g[b_grid_full, t_full_idx]
                t_b = t_tr_g[b_grid_full, t_full_idx]
                pet_b = pet_tr_g[b_grid_full, t_full_idx]
                yo_b = yo_tr_g[b_grid_rho, t_rho_idx]

                optimizer.zero_grad(set_to_none=True)
                q_sim = model(
                    xd_b, xs_b, p_b, t_b, pet_b, warmup_period=eff_warmup
                )
                valid_mask = ~torch.isnan(yo_b) & (yo_b >= 0.0)
                if valid_mask.any():
                    yp = q_sim[valid_mask]
                    yt = yo_b[valid_mask]
                    loss1 = torch.sqrt(torch.mean((yp - yt) ** 2) + 1e-12)
                    yp_t = torch.log10(torch.sqrt(torch.clamp(yp, min=0.0) + 1e-6) + 0.1)
                    yt_t = torch.log10(torch.sqrt(torch.clamp(yt, min=0.0) + 1e-6) + 0.1)
                    loss2 = torch.sqrt(torch.mean((yp_t - yt_t) ** 2) + 1e-12)
                    loss = 0.75 * loss1 + 0.25 * loss2
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

        model.eval()
        eval_w = eval_start_idx if t_test > eval_start_idx else 0
        preds = np.zeros((n_test, t_test - eval_start_idx), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_test, 100):
                sl = slice(i, min(i + 100, n_test))
                q_eval = model(
                    torch.from_numpy(xd_te[sl]).to(self.device),
                    torch.from_numpy(xs_te[sl]).to(self.device),
                    torch.from_numpy(p_te[sl]).to(self.device),
                    torch.from_numpy(t_te[sl]).to(self.device),
                    torch.from_numpy(pet_te[sl]).to(self.device),
                    warmup_period=eval_w,
                ).cpu().numpy()
                preds[sl] = np.maximum(0.0, np.nan_to_num(q_eval, nan=0.0))
        return preds
