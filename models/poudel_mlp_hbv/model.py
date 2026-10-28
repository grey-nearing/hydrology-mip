# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Independent first-party C++/OpenMP, CUDA, and PyTorch implementation of the
# 16-component differentiable HBV architecture and 20 parameter bounds published
# in Poudel & Steinschneider (2026, HESS, CC-BY-4.0, doi:10.5194/hess-30-5173-2026).

"""Poudel & Steinschneider (2026) Hybrid MLP + 16-Component Differentiable HBV.

Reference:
- Poudel, S., & Steinschneider, S. (2026). An argument for parsimony in
  differentiable hydrologic models. Hydrology and Earth System Sciences,
  30(15), 5173-5193. https://doi.org/10.5194/hess-30-5173-2026 (CC-BY-4.0)
- Reference Code: https://github.com/snpoudel/diff-hydro

Independent C++/OpenMP, CUDA, and PyTorch implementation of the published
Poudel & Steinschneider (2026) formulation (`MLPParameterNet` +
16-unit differentiable HBV with the 20 published parameter bounds, full-window
Gamma unit-hydrograph routing, 730-day spinup + 365-day target windows with
60-day stride, unnormalized masked MSE loss, and Adam lr=1e-4 with
ReduceLROnPlateau & validation early stopping), accelerated via a custom
OpenMP C++ forward/backward kernel (`_POUDEL_OMP_CPP`).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 20 HBV parameter bounds published in Poudel & Steinschneider (2026, HESS, CC-BY-4.0)
POUDEL_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "fc": (50.0, 1000.0),
    "beta": (1.0, 6.0),
    "pwp": (0.2, 1.0),
    "l": (0.0, 100.0),
    "ks": (0.05, 0.9),
    "ki": (0.01, 0.5),
    "kb": (0.01, 0.2),
    "kperc": (0.0001, 0.99),
    "coeff_pet": (0.5, 3.0),
    "ddf": (0.5, 10.0),
    "scf": (0.5, 2.0),
    "ts": (-4.0, 4.0),
    "tm": (-2.5, 2.5),
    "tti": (0.1, 4.0),
    "whc": (0.0, 0.2),
    "crf": (0.0, 0.1),
    "d_shape": (1.0, 4.0),
    "d_scale": (0.5, 4.0),
    "b_shape": (1.0, 6.0),
    "b_scale": (1.0, 7.0),
}

HBV_BOUNDS_LIST: List[List[float]] = [
    list(POUDEL_PARAM_BOUNDS[k])
    for k in [
        "fc",
        "beta",
        "pwp",
        "l",
        "ks",
        "ki",
        "kb",
        "kperc",
        "coeff_pet",
        "ddf",
        "scf",
        "ts",
        "tm",
        "tti",
        "whc",
        "crf",
        "d_shape",
        "d_scale",
        "b_shape",
        "b_scale",
    ]
]


def compute_daylength_hours(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Compute daily daylight duration in hours from latitude (deg) and day-of-year."""
    lat_clipped = np.clip(lat_deg.astype(np.float64), -66.5, 66.5)
    lat_rad = np.deg2rad(lat_clipped)[:, None]
    j = doy.astype(np.float64)[None, :]
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * j - 1.39)
    arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(arg)
    daylen = (24.0 / np.pi) * ws
    return np.clip(daylen, 0.0, 24.0).astype(np.float32)


def _derive_physical_p_t_daylen(
    x_dyn: np.ndarray,
    lat_deg: Optional[np.ndarray] = None,
    doy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (precip mm/d, tmean degC, daylen hours) from CAMELS forcing channels."""
    n_channels = x_dyn.shape[2]
    if n_channels >= 5 and n_channels % 5 == 0:
        prcp = np.maximum(0.0, np.nanmean(x_dyn[:, :, 0::5], axis=2))
        tmax = np.nanmean(x_dyn[:, :, 2::5], axis=2)
        tmin = np.nanmean(x_dyn[:, :, 3::5], axis=2)
    else:
        prcp = np.maximum(0.0, x_dyn[:, :, 0])
        tmax = x_dyn[:, :, 2]
        tmin = x_dyn[:, :, 3]

    prcp = np.nan_to_num(prcp, nan=0.0).astype(np.float32)
    tmax = np.nan_to_num(tmax, nan=0.0).astype(np.float32)
    tmin = np.nan_to_num(tmin, nan=0.0).astype(np.float32)
    tmean = (0.5 * (tmax + tmin)).astype(np.float32)

    if lat_deg is not None and doy is not None and len(doy) == x_dyn.shape[1]:
        daylen = compute_daylength_hours(lat_deg, doy)
    else:
        daylen = np.full(prcp.shape, 12.0, dtype=np.float32)

    return prcp, tmean, daylen


def compute_hamon_pet_base(tmean: np.ndarray, daylen: np.ndarray) -> np.ndarray:
    """Compute unscaled Hamon (1961) potential evapotranspiration in mm/day."""
    es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3 + 1e-5))
    rhosat = 216.7 * es / (tmean + 273.15 + 1e-5)
    pet = 0.1651 * (daylen / 12.0) * rhosat
    return np.maximum(0.0, pet).astype(np.float32)


def _derive_physical_p_t_pet(
    x_dyn: np.ndarray,
    lat_deg: Optional[np.ndarray] = None,
    doy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive physical (P, Tmean, PET) in mm/day and degC from CAMELS forcing channels."""
    n_channels = x_dyn.shape[2]
    if n_channels >= 5 and n_channels % 5 == 0:
        prcp = np.maximum(0.0, np.nanmean(x_dyn[:, :, 0::5], axis=2))
        srad = np.maximum(0.0, np.nanmean(x_dyn[:, :, 1::5], axis=2))
        tmax = np.nanmean(x_dyn[:, :, 2::5], axis=2)
        tmin = np.nanmean(x_dyn[:, :, 3::5], axis=2)
    else:
        prcp = np.maximum(0.0, x_dyn[:, :, 0])
        srad = np.maximum(0.0, x_dyn[:, :, 1])
        tmax = x_dyn[:, :, 2]
        tmin = x_dyn[:, :, 3]

    prcp = np.nan_to_num(prcp, nan=0.0)
    srad = np.nan_to_num(srad, nan=0.0)
    tmax = np.nan_to_num(tmax, nan=0.0)
    tmin = np.nan_to_num(tmin, nan=0.0)
    tmean = 0.5 * (tmax + tmin)

    if lat_deg is not None and doy is not None and len(doy) == x_dyn.shape[1]:
        daylen = compute_daylength_hours(lat_deg, doy)
        pet = compute_hamon_pet_base(tmean, daylen)
    else:
        rn_mm = srad * 0.0864 * 0.408 * 0.55
        es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3 + 1e-6))
        delta = (4098.0 * es) / ((tmean + 237.3 + 1e-6) ** 2)
        gamma = 0.066
        pet = np.maximum(0.0, 1.26 * (delta / (delta + gamma)) * rn_mm)

    return prcp.astype(np.float32), tmean.astype(np.float32), pet.astype(np.float32)


_POUDEL_OMP_CPP = r"""
#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <vector>

// Clean-room C++/OpenMP implementation of the 16-zone HBV equations in Poudel & Steinschneider (2026, HESS, CC-BY-4.0)
// States order: [0]=s_up, [1]=s_low, [2]=s_snow, [3]=s_sliq, [4]=s_sma
// Params order (16): [0]=fc, [1]=beta, [2]=pwp, [3]=l, [4]=ks, [5]=ki, [6]=kb, [7]=kperc,
//                    [8]=coeff_pet, [9]=ddf, [10]=scf, [11]=ts, [12]=tm, [13]=tti, [14]=whc, [15]=crf
std::vector<torch::Tensor> poudel_exact_forward_cpu(
    torch::Tensor p,          // [B, T]
    torch::Tensor temp,       // [B, T]
    torch::Tensor daylen,     // [B, T]
    torch::Tensor params,     // [B, M, 16]
    torch::Tensor init_states // [B, M, 5]
) {
    int B = p.size(0);
    int T = p.size(1);
    int M = params.size(1);
    int BM = B * M;

    auto opts = p.options();
    auto q_di = torch::empty({B, M, T}, opts);
    auto q_b = torch::empty({B, M, T}, opts);
    auto states_hist = torch::empty({B, M, T, 5}, opts);
    auto final_states = torch::empty({B, M, 5}, opts);

    const float* p_p = p.data_ptr<float>();
    const float* temp_p = temp.data_ptr<float>();
    const float* dayl_p = daylen.data_ptr<float>();
    const float* par_p = params.data_ptr<float>();
    const float* init_p = init_states.data_ptr<float>();

    float* q_di_p = q_di.data_ptr<float>();
    float* q_b_p = q_b.data_ptr<float>();
    float* hist_p = states_hist.data_ptr<float>();
    float* fin_p = final_states.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < BM; ++idx) {
        int b = idx / M;
        const float* pm = par_p + idx * 16;
        float fc = pm[0], beta = pm[1], pwp = pm[2], l_th = pm[3];
        float ks = pm[4], ki = pm[5], kb = pm[6], kperc = pm[7];
        float coeff_pet = pm[8], ddf = pm[9], scf = pm[10], ts = pm[11];
        float tm = pm[12], tti = pm[13], whc = pm[14], crf = pm[15];

        const float* in_s = init_p + idx * 5;
        float s_up = in_s[0], s_low = in_s[1], s_snow = in_s[2], s_sliq = in_s[3], s_sma = in_s[4];

        const float* p_row = p_p + b * T;
        const float* t_row = temp_p + b * T;
        const float* d_row = dayl_p + b * T;
        float* q_di_bm = q_di_p + idx * T;
        float* q_b_bm = q_b_p + idx * T;
        float* hist_bm = hist_p + idx * (T * 5);

        for (int t = 0; t < T; ++t) {
            s_up = std::fmin(std::fmax(s_up, 0.0f), 2000.0f);
            s_low = std::fmin(std::fmax(s_low, 0.0f), 2000.0f);
            s_snow = std::fmin(std::fmax(s_snow, 0.0f), 2000.0f);
            s_sliq = std::fmin(std::fmax(s_sliq, 0.0f), 2000.0f);
            s_sma = std::fmin(std::fmax(s_sma, 0.0f), 2000.0f);

            float* h = hist_bm + t * 5;
            h[0] = s_up; h[1] = s_low; h[2] = s_snow; h[3] = s_sliq; h[4] = s_sma;

            float ct = t_row[t];
            float cp = p_row[t];
            float dl = d_row[t];
            float esat = 0.611f * std::exp(17.27f * ct / (ct + 237.3f));
            float pe_t = coeff_pet * 29.8f * dl * (esat / (ct + 273.2f));

            float raw_sf = -1.0f / (tti + 1e-3f) * (ct - ts) + 1.0f;
            float snowfrac = std::fmin(std::fmax(raw_sf, 0.0f), 1.0f);
            float snow = cp * snowfrac;
            float rain = cp * (1.0f - snowfrac);

            float pot_melt = (ct > tm) ? (ddf * (ct - tm)) : 0.0f;
            float melt = std::fmin(pot_melt, s_snow);
            float s_snow1 = std::fmax(s_snow - melt, 0.0f);
            float s_sliq1 = std::fmax(s_sliq + melt + rain, 0.0f);
            float liqmax = std::fmax(s_snow1 * whc, 0.0f);
            float pr_eff = std::fmax(s_sliq1 - liqmax, 0.0f);
            float s_sliq2 = std::fmin(s_sliq1, liqmax);

            float pot_refreeze = (ct < tm) ? ((tm - ct) * ddf * crf) : 0.0f;
            float refreeze = std::fmin(pot_refreeze, s_sliq2);
            float s_snow2 = std::fmax(s_snow1 + refreeze + snow * scf, 0.0f);
            float s_sliq3 = std::fmax(s_sliq2 - refreeze, 0.0f);

            float eff_ratio_base = std::fmin(std::fmax(s_sma / fc, 1e-4f), 1.0f);
            float effratio = std::pow(eff_ratio_base, beta);
            float remainwater = pr_eff * (1.0f - effratio);
            float added = std::fmin(remainwater + s_sma, fc) - s_sma;
            float peff = pr_eff - added;
            float s_sma1 = std::fmax(s_sma + added, 0.0f);

            float potevap_factor = std::fmin(std::fmax(s_sma1 / (pwp * fc + 1e-3f), 0.0f), 1.0f);
            float pet = (s_sma1 > pwp * fc) ? pe_t : (pe_t * potevap_factor);
            float et = std::fmin(pet, s_sma1);
            float s_sma2 = std::fmax(s_sma1 - et, 0.0f);

            float s_up1 = std::fmax(s_up + peff, 0.0f);
            float qs = std::fmax(s_up1 - l_th, 0.0f) * ks;
            float qi = std::fmin(l_th, s_up1) * ki;
            float qperc = std::fmax(s_up1 - qs - qi, 0.0f) * kperc;
            float s_up2 = std::fmax(s_up1 - (qs + qi + qperc), 0.0f);
            float qq = qs + qi;

            float s_low1 = std::fmax(s_low + qperc, 0.0f);
            float qb = s_low1 * kb;
            float s_low2 = std::fmax(s_low1 - qb, 0.0f);

            q_di_bm[t] = qq;
            q_b_bm[t] = qb;

            s_up = s_up2; s_low = s_low2; s_snow = s_snow2; s_sliq = s_sliq3; s_sma = s_sma2;
        }
        float* f_s = fin_p + idx * 5;
        f_s[0] = s_up; f_s[1] = s_low; f_s[2] = s_snow; f_s[3] = s_sliq; f_s[4] = s_sma;
    }
    return {q_di, q_b, states_hist, final_states};
}

torch::Tensor poudel_exact_backward_cpu(
    torch::Tensor p,
    torch::Tensor temp,
    torch::Tensor daylen,
    torch::Tensor params,
    torch::Tensor states_hist,
    torch::Tensor d_q_di,
    torch::Tensor d_q_b
) {
    int B = p.size(0);
    int T = p.size(1);
    int M = params.size(1);
    int BM = B * M;

    auto d_params = torch::zeros_like(params);

    const float* p_p = p.data_ptr<float>();
    const float* temp_p = temp.data_ptr<float>();
    const float* dayl_p = daylen.data_ptr<float>();
    const float* par_p = params.data_ptr<float>();
    const float* hist_p = states_hist.data_ptr<float>();
    const float* dq_di_p = d_q_di.data_ptr<float>();
    const float* dq_b_p = d_q_b.data_ptr<float>();
    float* dpar_p = d_params.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < BM; ++idx) {
        int b = idx / M;
        const float* pm = par_p + idx * 16;
        float fc = pm[0], beta = pm[1], pwp = pm[2], l_th = pm[3];
        float ks = pm[4], ki = pm[5], kb = pm[6], kperc = pm[7];
        float coeff_pet = pm[8], ddf = pm[9], scf = pm[10], ts = pm[11];
        float tm = pm[12], tti = pm[13], whc = pm[14], crf = pm[15];

        float g_fc = 0.0f, g_beta = 0.0f, g_pwp = 0.0f, g_l = 0.0f;
        float g_ks = 0.0f, g_ki = 0.0f, g_kb = 0.0f, g_kperc = 0.0f;
        float g_coeff_pet = 0.0f, g_ddf = 0.0f, g_scf = 0.0f, g_ts = 0.0f;
        float g_tm = 0.0f, g_tti = 0.0f, g_whc = 0.0f, g_crf = 0.0f;

        float d_sup = 0.0f, d_slow = 0.0f, d_ssnow = 0.0f, d_ssliq = 0.0f, d_ssma = 0.0f;

        const float* p_row = p_p + b * T;
        const float* t_row = temp_p + b * T;
        const float* d_row = dayl_p + b * T;
        const float* dq_di_bm = dq_di_p + idx * T;
        const float* dq_b_bm = dq_b_p + idx * T;
        const float* hist_bm = hist_p + idx * (T * 5);

        for (int t = T - 1; t >= 0; --t) {
            const float* h = hist_bm + t * 5;
            float s_up = h[0], s_low = h[1], s_snow = h[2], s_sliq = h[3], s_sma = h[4];

            float ct = t_row[t];
            float cp = p_row[t];
            float dl = d_row[t];
            float esat = 0.611f * std::exp(17.27f * ct / (ct + 237.3f));
            float pe_base = 29.8f * dl * (esat / (ct + 273.2f));
            float pe_t = coeff_pet * pe_base;

            float inv_tti = 1.0f / (tti + 1e-3f);
            float raw_sf = -inv_tti * (ct - ts) + 1.0f;
            float snowfrac = std::fmin(std::fmax(raw_sf, 0.0f), 1.0f);
            float snow = cp * snowfrac;
            float rain = cp * (1.0f - snowfrac);

            float pot_melt = (ct > tm) ? (ddf * (ct - tm)) : 0.0f;
            float melt = std::fmin(pot_melt, s_snow);
            float raw_ssnow1 = s_snow - melt;
            float s_snow1 = std::fmax(raw_ssnow1, 0.0f);
            float raw_ssliq1 = s_sliq + melt + rain;
            float s_sliq1 = std::fmax(raw_ssliq1, 0.0f);
            float raw_liqmax = s_snow1 * whc;
            float liqmax = std::fmax(raw_liqmax, 0.0f);
            float raw_preff = s_sliq1 - liqmax;
            float pr_eff = std::fmax(raw_preff, 0.0f);
            float s_sliq2 = std::fmin(s_sliq1, liqmax);

            float pot_refreeze = (ct < tm) ? ((tm - ct) * ddf * crf) : 0.0f;
            float refreeze = std::fmin(pot_refreeze, s_sliq2);
            float raw_ssnow2 = s_snow1 + refreeze + snow * scf;
            float raw_ssliq3 = s_sliq2 - refreeze;

            float raw_eff_base = s_sma / fc;
            float eff_ratio_base = std::fmin(std::fmax(raw_eff_base, 1e-4f), 1.0f);
            float effratio = std::pow(eff_ratio_base, beta);
            float remainwater = pr_eff * (1.0f - effratio);
            float sum_rw_sma = remainwater + s_sma;
            float min_rw_fc = std::fmin(sum_rw_sma, fc);
            float added = min_rw_fc - s_sma;
            float peff = pr_eff - added;
            float raw_ssma1 = s_sma + added;
            float s_sma1 = std::fmax(raw_ssma1, 0.0f);

            float denom_pwp = pwp * fc + 1e-3f;
            float raw_pef = s_sma1 / denom_pwp;
            float potevap_factor = std::fmin(std::fmax(raw_pef, 0.0f), 1.0f);
            bool sma_gt_pwp = (s_sma1 > pwp * fc);
            float pet = sma_gt_pwp ? pe_t : (pe_t * potevap_factor);
            float et = std::fmin(pet, s_sma1);
            float raw_ssma2 = s_sma1 - et;

            float raw_sup1 = s_up + peff;
            float s_up1 = std::fmax(raw_sup1, 0.0f);
            float raw_sup_l = s_up1 - l_th;
            float rel_sup_l = std::fmax(raw_sup_l, 0.0f);
            float qs = rel_sup_l * ks;
            float min_l_sup1 = std::fmin(l_th, s_up1);
            float qi = min_l_sup1 * ki;
            float raw_rem_up = s_up1 - qs - qi;
            float rel_rem_up = std::fmax(raw_rem_up, 0.0f);
            float qperc = rel_rem_up * kperc;
            float raw_sup2 = s_up1 - (qs + qi + qperc);

            float raw_slow1 = s_low + qperc;
            float s_low1 = std::fmax(raw_slow1, 0.0f);
            float qb = s_low1 * kb;
            float raw_slow2 = s_low1 - qb;

            float d_sup2 = d_sup;
            float d_slow2 = d_slow;
            float d_ssnow2 = d_ssnow;
            float d_ssliq3 = d_ssliq;
            float d_ssma2 = d_ssma;

            float g_qq = dq_di_bm[t];
            float g_qb_out = dq_b_bm[t];

            // Lower zone
            float d_raw_slow2 = (raw_slow2 > 0.0f) ? d_slow2 : 0.0f;
            float g_qb = g_qb_out - d_raw_slow2;
            float d_slow1 = d_raw_slow2 + g_qb * kb;
            g_kb += g_qb * s_low1;
            float d_raw_slow1 = (raw_slow1 > 0.0f) ? d_slow1 : 0.0f;
            float d_slow_prev = d_raw_slow1;
            float g_qperc = d_raw_slow1;

            // Upper zone
            float d_raw_sup2 = (raw_sup2 > 0.0f) ? d_sup2 : 0.0f;
            float d_sup1 = d_raw_sup2;
            float g_qs = g_qq - d_raw_sup2;
            float g_qi = g_qq - d_raw_sup2;
            g_qperc -= d_raw_sup2;

            g_kperc += g_qperc * rel_rem_up;
            float d_rel_rem_up = g_qperc * kperc;
            if (raw_rem_up > 0.0f) {
                d_sup1 += d_rel_rem_up;
                g_qs -= d_rel_rem_up;
                g_qi -= d_rel_rem_up;
            }

            g_ki += g_qi * min_l_sup1;
            float d_min_l_sup1 = g_qi * ki;
            if (l_th < s_up1) {
                g_l += d_min_l_sup1;
            } else {
                d_sup1 += d_min_l_sup1;
            }

            g_ks += g_qs * rel_sup_l;
            float d_rel_sup_l = g_qs * ks;
            if (raw_sup_l > 0.0f) {
                d_sup1 += d_rel_sup_l;
                g_l -= d_rel_sup_l;
            }

            float d_raw_sup1 = (raw_sup1 > 0.0f) ? d_sup1 : 0.0f;
            float d_sup_prev = d_raw_sup1;
            float g_peff = d_raw_sup1;

            // Soil moisture & ET
            float d_raw_ssma2 = (raw_ssma2 > 0.0f) ? d_ssma2 : 0.0f;
            float d_ssma1 = d_raw_ssma2;
            float g_et = -d_raw_ssma2;
            float g_pet = 0.0f;
            if (pet < s_sma1) {
                g_pet = g_et;
            } else {
                d_ssma1 += g_et;
            }

            float g_pe_t = 0.0f;
            if (sma_gt_pwp) {
                g_pe_t = g_pet;
            } else {
                g_pe_t = g_pet * potevap_factor;
                float g_pef = g_pet * pe_t;
                if (raw_pef > 0.0f && raw_pef < 1.0f) {
                    float d_raw_pef = g_pef / denom_pwp;
                    d_ssma1 += d_raw_pef;
                    float d_denom_pwp = -d_raw_pef * raw_pef;
                    g_pwp += d_denom_pwp * fc;
                    g_fc += d_denom_pwp * pwp;
                }
            }
            g_coeff_pet += g_pe_t * pe_base;

            float d_raw_ssma1 = (raw_ssma1 > 0.0f) ? d_ssma1 : 0.0f;
            float d_ssma_prev = d_raw_ssma1;
            float g_added = d_raw_ssma1 - g_peff;
            float g_preff = g_peff;

            float d_min_rw_fc = g_added;
            d_ssma_prev -= g_added;
            float g_remainwater = 0.0f;
            if (sum_rw_sma < fc) {
                g_remainwater = d_min_rw_fc;
                d_ssma_prev += d_min_rw_fc;
            } else {
                g_fc += d_min_rw_fc;
            }

            g_preff += g_remainwater * (1.0f - effratio);
            float g_effratio = -g_remainwater * pr_eff;
            g_beta += g_effratio * effratio * std::log(eff_ratio_base);
            float d_eff_base = g_effratio * beta * std::pow(eff_ratio_base, beta - 1.0f);
            if (raw_eff_base > 1e-4f && raw_eff_base < 1.0f) {
                d_ssma_prev += d_eff_base / fc;
                g_fc -= d_eff_base * raw_eff_base / fc;
            }

            // Snow & liquid water
            float d_raw_ssliq3 = (raw_ssliq3 > 0.0f) ? d_ssliq3 : 0.0f;
            float d_raw_ssnow2 = (raw_ssnow2 > 0.0f) ? d_ssnow2 : 0.0f;
            float d_ssnow1 = d_raw_ssnow2;
            float d_ssliq2 = d_raw_ssliq3;
            float g_refreeze = d_raw_ssnow2 - d_raw_ssliq3;
            float g_snow = d_raw_ssnow2 * scf;
            g_scf += d_raw_ssnow2 * snow;

            if (pot_refreeze < s_sliq2) {
                if (ct < tm) {
                    g_tm += g_refreeze * ddf * crf;
                    g_ddf += g_refreeze * (tm - ct) * crf;
                    g_crf += g_refreeze * (tm - ct) * ddf;
                }
            } else {
                d_ssliq2 += g_refreeze;
            }

            float d_ssliq1 = 0.0f;
            float g_liqmax = 0.0f;
            if (s_sliq1 < liqmax) {
                d_ssliq1 += d_ssliq2;
            } else {
                g_liqmax += d_ssliq2;
            }
            if (raw_preff > 0.0f) {
                d_ssliq1 += g_preff;
                g_liqmax -= g_preff;
            }
            if (raw_liqmax > 0.0f) {
                d_ssnow1 += g_liqmax * whc;
                g_whc += g_liqmax * s_snow1;
            }

            float d_raw_ssliq1 = (raw_ssliq1 > 0.0f) ? d_ssliq1 : 0.0f;
            float d_raw_ssnow1 = (raw_ssnow1 > 0.0f) ? d_ssnow1 : 0.0f;
            float d_ssliq_prev = d_raw_ssliq1;
            float g_rain = d_raw_ssliq1;
            float g_melt = d_raw_ssliq1 - d_raw_ssnow1;
            float d_ssnow_prev = d_raw_ssnow1;

            if (pot_melt < s_snow) {
                if (ct > tm) {
                    g_ddf += g_melt * (ct - tm);
                    g_tm -= g_melt * ddf;
                }
            } else {
                d_ssnow_prev += g_melt;
            }

            float g_snowfrac = (g_snow - g_rain) * cp;
            if (raw_sf > 0.0f && raw_sf < 1.0f) {
                g_ts += g_snowfrac * inv_tti;
                g_tti += g_snowfrac * (ct - ts) * inv_tti * inv_tti;
            }

            d_sup = (s_up >= 0.0f && s_up <= 2000.0f) ? d_sup_prev : 0.0f;
            d_slow = (s_low >= 0.0f && s_low <= 2000.0f) ? d_slow_prev : 0.0f;
            d_ssnow = (s_snow >= 0.0f && s_snow <= 2000.0f) ? d_ssnow_prev : 0.0f;
            d_ssliq = (s_sliq >= 0.0f && s_sliq <= 2000.0f) ? d_ssliq_prev : 0.0f;
            d_ssma = (s_sma >= 0.0f && s_sma <= 2000.0f) ? d_ssma_prev : 0.0f;
        }

        float* dp = dpar_p + idx * 16;
        dp[0] = g_fc; dp[1] = g_beta; dp[2] = g_pwp; dp[3] = g_l;
        dp[4] = g_ks; dp[5] = g_ki; dp[6] = g_kb; dp[7] = g_kperc;
        dp[8] = g_coeff_pet; dp[9] = g_ddf; dp[10] = g_scf; dp[11] = g_ts;
        dp[12] = g_tm; dp[13] = g_tti; dp[14] = g_whc; dp[15] = g_crf;
    }
    return d_params;
}

// Zero-copy windowed forward: runs `warmup` spinup steps in registers (no memory traffic)
// and records `rho` target steps into q_di, q_b, states_hist.
std::vector<torch::Tensor> poudel_window_forward_cpu(
    torch::Tensor p_full,     // [N_basins, T_total]
    torch::Tensor temp_full,  // [N_basins, T_total]
    torch::Tensor dayl_full,  // [N_basins, T_total]
    torch::Tensor b_idx,      // [B] int64
    torch::Tensor s_idx,      // [B] int64
    int warmup,
    int rho,
    torch::Tensor params      // [B, M, 16]
) {
    int B = b_idx.size(0);
    int T_total = p_full.size(1);
    int M = params.size(1);
    int BM = B * M;

    auto opts = p_full.options();
    auto q_di = torch::empty({B, M, rho}, opts);
    auto q_b = torch::empty({B, M, rho}, opts);
    auto states_hist = torch::empty({B, M, rho, 5}, opts);

    const float* p_p = p_full.data_ptr<float>();
    const float* temp_p = temp_full.data_ptr<float>();
    const float* dayl_p = dayl_full.data_ptr<float>();
    const int64_t* b_p = b_idx.data_ptr<int64_t>();
    const int64_t* s_p = s_idx.data_ptr<int64_t>();
    const float* par_p = params.data_ptr<float>();

    float* q_di_p = q_di.data_ptr<float>();
    float* q_b_p = q_b.data_ptr<float>();
    float* hist_p = states_hist.data_ptr<float>();

    int total_win = warmup + rho;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < BM; ++idx) {
        int b = idx / M;
        int64_t basin = b_p[b];
        int64_t s0 = s_p[b];
        const float* pm = par_p + idx * 16;
        float fc = pm[0], beta = pm[1], pwp = pm[2], l_th = pm[3];
        float ks = pm[4], ki = pm[5], kb = pm[6], kperc = pm[7];
        float coeff_pet = pm[8], ddf = pm[9], scf = pm[10], ts = pm[11];
        float tm = pm[12], tti = pm[13], whc = pm[14], crf = pm[15];

        float s_up = 0.0f, s_low = 0.0f, s_snow = 0.0f, s_sliq = 0.0f, s_sma = 0.0f;

        const float* p_row = p_p + basin * T_total + s0;
        const float* t_row = temp_p + basin * T_total + s0;
        const float* d_row = dayl_p + basin * T_total + s0;
        float* q_di_bm = q_di_p + idx * rho;
        float* q_b_bm = q_b_p + idx * rho;
        float* hist_bm = hist_p + idx * (rho * 5);

        for (int t = 0; t < total_win; ++t) {
            s_up = std::fmin(std::fmax(s_up, 0.0f), 2000.0f);
            s_low = std::fmin(std::fmax(s_low, 0.0f), 2000.0f);
            s_snow = std::fmin(std::fmax(s_snow, 0.0f), 2000.0f);
            s_sliq = std::fmin(std::fmax(s_sliq, 0.0f), 2000.0f);
            s_sma = std::fmin(std::fmax(s_sma, 0.0f), 2000.0f);

            int trel = t - warmup;
            if (trel >= 0) {
                float* h = hist_bm + trel * 5;
                h[0] = s_up; h[1] = s_low; h[2] = s_snow; h[3] = s_sliq; h[4] = s_sma;
            }

            float ct = t_row[t];
            float cp = p_row[t];
            float dl = d_row[t];
            float esat = 0.611f * std::exp(17.27f * ct / (ct + 237.3f));
            float pe_t = coeff_pet * 29.8f * dl * (esat / (ct + 273.2f));

            float raw_sf = -1.0f / (tti + 1e-3f) * (ct - ts) + 1.0f;
            float snowfrac = std::fmin(std::fmax(raw_sf, 0.0f), 1.0f);
            float snow = cp * snowfrac;
            float rain = cp * (1.0f - snowfrac);

            float pot_melt = (ct > tm) ? (ddf * (ct - tm)) : 0.0f;
            float melt = std::fmin(pot_melt, s_snow);
            float s_snow1 = std::fmax(s_snow - melt, 0.0f);
            float s_sliq1 = std::fmax(s_sliq + melt + rain, 0.0f);
            float liqmax = std::fmax(s_snow1 * whc, 0.0f);
            float pr_eff = std::fmax(s_sliq1 - liqmax, 0.0f);
            float s_sliq2 = std::fmin(s_sliq1, liqmax);

            float pot_refreeze = (ct < tm) ? ((tm - ct) * ddf * crf) : 0.0f;
            float refreeze = std::fmin(pot_refreeze, s_sliq2);
            float s_snow2 = std::fmax(s_snow1 + refreeze + snow * scf, 0.0f);
            float s_sliq3 = std::fmax(s_sliq2 - refreeze, 0.0f);

            float eff_ratio_base = std::fmin(std::fmax(s_sma / fc, 1e-4f), 1.0f);
            float effratio = std::pow(eff_ratio_base, beta);
            float remainwater = pr_eff * (1.0f - effratio);
            float added = std::fmin(remainwater + s_sma, fc) - s_sma;
            float peff = pr_eff - added;
            float s_sma1 = std::fmax(s_sma + added, 0.0f);

            float potevap_factor = std::fmin(std::fmax(s_sma1 / (pwp * fc + 1e-3f), 0.0f), 1.0f);
            float pet = (s_sma1 > pwp * fc) ? pe_t : (pe_t * potevap_factor);
            float et = std::fmin(pet, s_sma1);
            float s_sma2 = std::fmax(s_sma1 - et, 0.0f);

            float s_up1 = std::fmax(s_up + peff, 0.0f);
            float qs = std::fmax(s_up1 - l_th, 0.0f) * ks;
            float qi = std::fmin(l_th, s_up1) * ki;
            float qperc = std::fmax(s_up1 - qs - qi, 0.0f) * kperc;
            float s_up2 = std::fmax(s_up1 - (qs + qi + qperc), 0.0f);
            float qq = qs + qi;

            float s_low1 = std::fmax(s_low + qperc, 0.0f);
            float qb = s_low1 * kb;
            float s_low2 = std::fmax(s_low1 - qb, 0.0f);

            if (trel >= 0) {
                q_di_bm[trel] = qq;
                q_b_bm[trel] = qb;
            }

            s_up = s_up2; s_low = s_low2; s_snow = s_snow2; s_sliq = s_sliq3; s_sma = s_sma2;
        }
    }
    return {q_di, q_b, states_hist};
}

torch::Tensor poudel_window_backward_cpu(
    torch::Tensor p_full,
    torch::Tensor temp_full,
    torch::Tensor dayl_full,
    torch::Tensor b_idx,
    torch::Tensor s_idx,
    int warmup,
    int rho,
    torch::Tensor params,
    torch::Tensor states_hist,
    torch::Tensor d_q_di,
    torch::Tensor d_q_b
) {
    int B = b_idx.size(0);
    int T_total = p_full.size(1);
    int M = params.size(1);
    int BM = B * M;

    auto d_params = torch::zeros_like(params);

    const float* p_p = p_full.data_ptr<float>();
    const float* temp_p = temp_full.data_ptr<float>();
    const float* dayl_p = dayl_full.data_ptr<float>();
    const int64_t* b_p = b_idx.data_ptr<int64_t>();
    const int64_t* s_p = s_idx.data_ptr<int64_t>();
    const float* par_p = params.data_ptr<float>();
    const float* hist_p = states_hist.data_ptr<float>();
    const float* dq_di_p = d_q_di.data_ptr<float>();
    const float* dq_b_p = d_q_b.data_ptr<float>();
    float* dpar_p = d_params.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < BM; ++idx) {
        int b = idx / M;
        int64_t basin = b_p[b];
        int64_t s0 = s_p[b] + warmup;
        const float* pm = par_p + idx * 16;
        float fc = pm[0], beta = pm[1], pwp = pm[2], l_th = pm[3];
        float ks = pm[4], ki = pm[5], kb = pm[6], kperc = pm[7];
        float coeff_pet = pm[8], ddf = pm[9], scf = pm[10], ts = pm[11];
        float tm = pm[12], tti = pm[13], whc = pm[14], crf = pm[15];

        float g_fc = 0.0f, g_beta = 0.0f, g_pwp = 0.0f, g_l = 0.0f;
        float g_ks = 0.0f, g_ki = 0.0f, g_kb = 0.0f, g_kperc = 0.0f;
        float g_coeff_pet = 0.0f, g_ddf = 0.0f, g_scf = 0.0f, g_ts = 0.0f;
        float g_tm = 0.0f, g_tti = 0.0f, g_whc = 0.0f, g_crf = 0.0f;

        float d_sup = 0.0f, d_slow = 0.0f, d_ssnow = 0.0f, d_ssliq = 0.0f, d_ssma = 0.0f;

        const float* p_row = p_p + basin * T_total + s0;
        const float* t_row = temp_p + basin * T_total + s0;
        const float* d_row = dayl_p + basin * T_total + s0;
        const float* dq_di_bm = dq_di_p + idx * rho;
        const float* dq_b_bm = dq_b_p + idx * rho;
        const float* hist_bm = hist_p + idx * (rho * 5);

        for (int t = rho - 1; t >= 0; --t) {
            const float* h = hist_bm + t * 5;
            float s_up = h[0], s_low = h[1], s_snow = h[2], s_sliq = h[3], s_sma = h[4];

            float ct = t_row[t];
            float cp = p_row[t];
            float dl = d_row[t];
            float esat = 0.611f * std::exp(17.27f * ct / (ct + 237.3f));
            float pe_base = 29.8f * dl * (esat / (ct + 273.2f));
            float pe_t = coeff_pet * pe_base;

            float inv_tti = 1.0f / (tti + 1e-3f);
            float raw_sf = -inv_tti * (ct - ts) + 1.0f;
            float snowfrac = std::fmin(std::fmax(raw_sf, 0.0f), 1.0f);
            float snow = cp * snowfrac;
            float rain = cp * (1.0f - snowfrac);

            float pot_melt = (ct > tm) ? (ddf * (ct - tm)) : 0.0f;
            float melt = std::fmin(pot_melt, s_snow);
            float raw_ssnow1 = s_snow - melt;
            float s_snow1 = std::fmax(raw_ssnow1, 0.0f);
            float raw_ssliq1 = s_sliq + melt + rain;
            float s_sliq1 = std::fmax(raw_ssliq1, 0.0f);
            float raw_liqmax = s_snow1 * whc;
            float liqmax = std::fmax(raw_liqmax, 0.0f);
            float raw_preff = s_sliq1 - liqmax;
            float pr_eff = std::fmax(raw_preff, 0.0f);
            float s_sliq2 = std::fmin(s_sliq1, liqmax);

            float pot_refreeze = (ct < tm) ? ((tm - ct) * ddf * crf) : 0.0f;
            float refreeze = std::fmin(pot_refreeze, s_sliq2);
            float raw_ssnow2 = s_snow1 + refreeze + snow * scf;
            float raw_ssliq3 = s_sliq2 - refreeze;

            float raw_eff_base = s_sma / fc;
            float eff_ratio_base = std::fmin(std::fmax(raw_eff_base, 1e-4f), 1.0f);
            float effratio = std::pow(eff_ratio_base, beta);
            float remainwater = pr_eff * (1.0f - effratio);
            float sum_rw_sma = remainwater + s_sma;
            float min_rw_fc = std::fmin(sum_rw_sma, fc);
            float added = min_rw_fc - s_sma;
            float peff = pr_eff - added;
            float raw_ssma1 = s_sma + added;
            float s_sma1 = std::fmax(raw_ssma1, 0.0f);

            float denom_pwp = pwp * fc + 1e-3f;
            float raw_pef = s_sma1 / denom_pwp;
            float potevap_factor = std::fmin(std::fmax(raw_pef, 0.0f), 1.0f);
            bool sma_gt_pwp = (s_sma1 > pwp * fc);
            float pet = sma_gt_pwp ? pe_t : (pe_t * potevap_factor);
            float et = std::fmin(pet, s_sma1);
            float raw_ssma2 = s_sma1 - et;

            float raw_sup1 = s_up + peff;
            float s_up1 = std::fmax(raw_sup1, 0.0f);
            float raw_sup_l = s_up1 - l_th;
            float rel_sup_l = std::fmax(raw_sup_l, 0.0f);
            float qs = rel_sup_l * ks;
            float min_l_sup1 = std::fmin(l_th, s_up1);
            float qi = min_l_sup1 * ki;
            float raw_rem_up = s_up1 - qs - qi;
            float rel_rem_up = std::fmax(raw_rem_up, 0.0f);
            float qperc = rel_rem_up * kperc;
            float raw_sup2 = s_up1 - (qs + qi + qperc);

            float raw_slow1 = s_low + qperc;
            float s_low1 = std::fmax(raw_slow1, 0.0f);
            float qb = s_low1 * kb;
            float raw_slow2 = s_low1 - qb;

            float d_sup2 = d_sup;
            float d_slow2 = d_slow;
            float d_ssnow2 = d_ssnow;
            float d_ssliq3 = d_ssliq;
            float d_ssma2 = d_ssma;

            float g_qq = dq_di_bm[t];
            float g_qb_out = dq_b_bm[t];

            float d_raw_slow2 = (raw_slow2 > 0.0f) ? d_slow2 : 0.0f;
            float g_qb = g_qb_out - d_raw_slow2;
            float d_slow1 = d_raw_slow2 + g_qb * kb;
            g_kb += g_qb * s_low1;
            float d_raw_slow1 = (raw_slow1 > 0.0f) ? d_slow1 : 0.0f;
            float d_slow_prev = d_raw_slow1;
            float g_qperc = d_raw_slow1;

            float d_raw_sup2 = (raw_sup2 > 0.0f) ? d_sup2 : 0.0f;
            float d_sup1 = d_raw_sup2;
            float g_qs = g_qq - d_raw_sup2;
            float g_qi = g_qq - d_raw_sup2;
            g_qperc -= d_raw_sup2;

            g_kperc += g_qperc * rel_rem_up;
            float d_rel_rem_up = g_qperc * kperc;
            if (raw_rem_up > 0.0f) {
                d_sup1 += d_rel_rem_up;
                g_qs -= d_rel_rem_up;
                g_qi -= d_rel_rem_up;
            }

            g_ki += g_qi * min_l_sup1;
            float d_min_l_sup1 = g_qi * ki;
            if (l_th < s_up1) {
                g_l += d_min_l_sup1;
            } else {
                d_sup1 += d_min_l_sup1;
            }

            g_ks += g_qs * rel_sup_l;
            float d_rel_sup_l = g_qs * ks;
            if (raw_sup_l > 0.0f) {
                d_sup1 += d_rel_sup_l;
                g_l -= d_rel_sup_l;
            }

            float d_raw_sup1 = (raw_sup1 > 0.0f) ? d_sup1 : 0.0f;
            float d_sup_prev = d_raw_sup1;
            float g_peff = d_raw_sup1;

            float d_raw_ssma2 = (raw_ssma2 > 0.0f) ? d_ssma2 : 0.0f;
            float d_ssma1 = d_raw_ssma2;
            float g_et = -d_raw_ssma2;
            float g_pet = 0.0f;
            if (pet < s_sma1) {
                g_pet = g_et;
            } else {
                d_ssma1 += g_et;
            }

            float g_pe_t = 0.0f;
            if (sma_gt_pwp) {
                g_pe_t = g_pet;
            } else {
                g_pe_t = g_pet * potevap_factor;
                float g_pef = g_pet * pe_t;
                if (raw_pef > 0.0f && raw_pef < 1.0f) {
                    float d_raw_pef = g_pef / denom_pwp;
                    d_ssma1 += d_raw_pef;
                    float d_denom_pwp = -d_raw_pef * raw_pef;
                    g_pwp += d_denom_pwp * fc;
                    g_fc += d_denom_pwp * pwp;
                }
            }
            g_coeff_pet += g_pe_t * pe_base;

            float d_raw_ssma1 = (raw_ssma1 > 0.0f) ? d_ssma1 : 0.0f;
            float d_ssma_prev = d_raw_ssma1;
            float g_added = d_raw_ssma1 - g_peff;
            float g_preff = g_peff;

            float d_min_rw_fc = g_added;
            d_ssma_prev -= g_added;
            float g_remainwater = 0.0f;
            if (sum_rw_sma < fc) {
                g_remainwater = d_min_rw_fc;
                d_ssma_prev += d_min_rw_fc;
            } else {
                g_fc += d_min_rw_fc;
            }

            g_preff += g_remainwater * (1.0f - effratio);
            float g_effratio = -g_remainwater * pr_eff;
            g_beta += g_effratio * effratio * std::log(eff_ratio_base);
            float d_eff_base = g_effratio * beta * std::pow(eff_ratio_base, beta - 1.0f);
            if (raw_eff_base > 1e-4f && raw_eff_base < 1.0f) {
                d_ssma_prev += d_eff_base / fc;
                g_fc -= d_eff_base * raw_eff_base / fc;
            }

            float d_raw_ssliq3 = (raw_ssliq3 > 0.0f) ? d_ssliq3 : 0.0f;
            float d_raw_ssnow2 = (raw_ssnow2 > 0.0f) ? d_ssnow2 : 0.0f;
            float d_ssnow1 = d_raw_ssnow2;
            float d_ssliq2 = d_raw_ssliq3;
            float g_refreeze = d_raw_ssnow2 - d_raw_ssliq3;
            float g_snow = d_raw_ssnow2 * scf;
            g_scf += d_raw_ssnow2 * snow;

            if (pot_refreeze < s_sliq2) {
                if (ct < tm) {
                    g_tm += g_refreeze * ddf * crf;
                    g_ddf += g_refreeze * (tm - ct) * crf;
                    g_crf += g_refreeze * (tm - ct) * ddf;
                }
            } else {
                d_ssliq2 += g_refreeze;
            }

            float d_ssliq1 = 0.0f;
            float g_liqmax = 0.0f;
            if (s_sliq1 < liqmax) {
                d_ssliq1 += d_ssliq2;
            } else {
                g_liqmax += d_ssliq2;
            }
            if (raw_preff > 0.0f) {
                d_ssliq1 += g_preff;
                g_liqmax -= g_preff;
            }
            if (raw_liqmax > 0.0f) {
                d_ssnow1 += g_liqmax * whc;
                g_whc += g_liqmax * s_snow1;
            }

            float d_raw_ssliq1 = (raw_ssliq1 > 0.0f) ? d_ssliq1 : 0.0f;
            float d_raw_ssnow1 = (raw_ssnow1 > 0.0f) ? d_ssnow1 : 0.0f;
            float d_ssliq_prev = d_raw_ssliq1;
            float g_rain = d_raw_ssliq1;
            float g_melt = d_raw_ssliq1 - d_raw_ssnow1;
            float d_ssnow_prev = d_raw_ssnow1;

            if (pot_melt < s_snow) {
                if (ct > tm) {
                    g_ddf += g_melt * (ct - tm);
                    g_tm -= g_melt * ddf;
                }
            } else {
                d_ssnow_prev += g_melt;
            }

            float g_snowfrac = (g_snow - g_rain) * cp;
            if (raw_sf > 0.0f && raw_sf < 1.0f) {
                g_ts += g_snowfrac * inv_tti;
                g_tti += g_snowfrac * (ct - ts) * inv_tti * inv_tti;
            }

            d_sup = (s_up >= 0.0f && s_up <= 2000.0f) ? d_sup_prev : 0.0f;
            d_slow = (s_low >= 0.0f && s_low <= 2000.0f) ? d_slow_prev : 0.0f;
            d_ssnow = (s_snow >= 0.0f && s_snow <= 2000.0f) ? d_ssnow_prev : 0.0f;
            d_ssliq = (s_sliq >= 0.0f && s_sliq <= 2000.0f) ? d_ssliq_prev : 0.0f;
            d_ssma = (s_sma >= 0.0f && s_sma <= 2000.0f) ? d_ssma_prev : 0.0f;
        }

        float* dp = dpar_p + idx * 16;
        dp[0] = g_fc; dp[1] = g_beta; dp[2] = g_pwp; dp[3] = g_l;
        dp[4] = g_ks; dp[5] = g_ki; dp[6] = g_kb; dp[7] = g_kperc;
        dp[8] = g_coeff_pet; dp[9] = g_ddf; dp[10] = g_scf; dp[11] = g_ts;
        dp[12] = g_tm; dp[13] = g_tti; dp[14] = g_whc; dp[15] = g_crf;
    }
    return d_params;
}
"""

_POUDEL_OMP_EXT = None
_POUDEL_CUDA_EXT = None

_POUDEL_CUDA_CPP = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> poudel_window_forward_cuda(
    torch::Tensor p_full,
    torch::Tensor temp_full,
    torch::Tensor dayl_full,
    torch::Tensor b_idx,
    torch::Tensor s_idx,
    int warmup,
    int rho,
    torch::Tensor params
);

torch::Tensor poudel_window_backward_cuda(
    torch::Tensor p_full,
    torch::Tensor temp_full,
    torch::Tensor dayl_full,
    torch::Tensor b_idx,
    torch::Tensor s_idx,
    int warmup,
    int rho,
    torch::Tensor params,
    torch::Tensor states_hist,
    torch::Tensor d_q_di,
    torch::Tensor d_q_b
);
"""

_POUDEL_CUDA_CU = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__global__ void poudel_window_forward_kernel(
    int BM, int M, int T_total, int warmup, int rho, bool save_hist,
    const float* __restrict__ p_p,
    const float* __restrict__ temp_p,
    const float* __restrict__ dayl_p,
    const int64_t* __restrict__ b_p,
    const int64_t* __restrict__ s_p,
    const float* __restrict__ par_p,
    float* __restrict__ q_di_p,
    float* __restrict__ q_b_p,
    float* __restrict__ hist_p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BM) return;

    int b = idx / M;
    int64_t basin = b_p[b];
    int64_t s0 = s_p[b];
    const float* pm = par_p + idx * 16;
    float fc = pm[0], beta = pm[1], pwp = pm[2], l_th = pm[3];
    float ks = pm[4], ki = pm[5], kb = pm[6], kperc = pm[7];
    float coeff_pet = pm[8], ddf = pm[9], scf = pm[10], ts = pm[11];
    float tm = pm[12], tti = pm[13], whc = pm[14], crf = pm[15];

    float s_up = 0.0f, s_low = 0.0f, s_snow = 0.0f, s_sliq = 0.0f, s_sma = 0.0f;

    const float* p_row = p_p + basin * T_total + s0;
    const float* t_row = temp_p + basin * T_total + s0;
    const float* d_row = dayl_p + basin * T_total + s0;
    float* q_di_bm = q_di_p + idx * rho;
    float* q_b_bm = q_b_p + idx * rho;

    int total_win = warmup + rho;
    for (int t = 0; t < total_win; ++t) {
        s_up = fminf(fmaxf(s_up, 0.0f), 2000.0f);
        s_low = fminf(fmaxf(s_low, 0.0f), 2000.0f);
        s_snow = fminf(fmaxf(s_snow, 0.0f), 2000.0f);
        s_sliq = fminf(fmaxf(s_sliq, 0.0f), 2000.0f);
        s_sma = fminf(fmaxf(s_sma, 0.0f), 2000.0f);

        int trel = t - warmup;
        if (trel >= 0 && save_hist) {
            size_t hbase = (size_t)trel * 5 * BM + idx;
            hist_p[hbase] = s_up;
            hist_p[hbase + BM] = s_low;
            hist_p[hbase + 2 * BM] = s_snow;
            hist_p[hbase + 3 * BM] = s_sliq;
            hist_p[hbase + 4 * BM] = s_sma;
        }

        float ct = t_row[t];
        float cp = p_row[t];
        float dl = d_row[t];
        float esat = 0.611f * expf(17.27f * ct / (ct + 237.3f));
        float pe_t = coeff_pet * 29.8f * dl * (esat / (ct + 273.2f));

        float raw_sf = -1.0f / (tti + 1e-3f) * (ct - ts) + 1.0f;
        float snowfrac = fminf(fmaxf(raw_sf, 0.0f), 1.0f);
        float snow = cp * snowfrac;
        float rain = cp * (1.0f - snowfrac);

        float pot_melt = (ct > tm) ? (ddf * (ct - tm)) : 0.0f;
        float melt = fminf(pot_melt, s_snow);
        float s_snow1 = fmaxf(s_snow - melt, 0.0f);
        float s_sliq1 = fmaxf(s_sliq + melt + rain, 0.0f);
        float liqmax = fmaxf(s_snow1 * whc, 0.0f);
        float pr_eff = fmaxf(s_sliq1 - liqmax, 0.0f);
        float s_sliq2 = fminf(s_sliq1, liqmax);

        float pot_refreeze = (ct < tm) ? ((tm - ct) * ddf * crf) : 0.0f;
        float refreeze = fminf(pot_refreeze, s_sliq2);
        float s_snow2 = fmaxf(s_snow1 + refreeze + snow * scf, 0.0f);
        float s_sliq3 = fmaxf(s_sliq2 - refreeze, 0.0f);

        float eff_ratio_base = fminf(fmaxf(s_sma / fc, 1e-4f), 1.0f);
        float effratio = powf(eff_ratio_base, beta);
        float remainwater = pr_eff * (1.0f - effratio);
        float added = fminf(remainwater + s_sma, fc) - s_sma;
        float peff = pr_eff - added;
        float s_sma1 = fmaxf(s_sma + added, 0.0f);

        float potevap_factor = fminf(fmaxf(s_sma1 / (pwp * fc + 1e-3f), 0.0f), 1.0f);
        float pet = (s_sma1 > pwp * fc) ? pe_t : (pe_t * potevap_factor);
        float et = fminf(pet, s_sma1);
        float s_sma2 = fmaxf(s_sma1 - et, 0.0f);

        float s_up1 = fmaxf(s_up + peff, 0.0f);
        float qs = fmaxf(s_up1 - l_th, 0.0f) * ks;
        float qi = fminf(l_th, s_up1) * ki;
        float qperc = fmaxf(s_up1 - qs - qi, 0.0f) * kperc;
        float s_up2 = fmaxf(s_up1 - (qs + qi + qperc), 0.0f);
        float qq = qs + qi;

        float s_low1 = fmaxf(s_low + qperc, 0.0f);
        float qb = s_low1 * kb;
        float s_low2 = fmaxf(s_low1 - qb, 0.0f);

        if (trel >= 0) {
            q_di_bm[trel] = qq;
            q_b_bm[trel] = qb;
        }

        s_up = s_up2; s_low = s_low2; s_snow = s_snow2; s_sliq = s_sliq3; s_sma = s_sma2;
    }
}

__global__ void poudel_window_backward_kernel(
    int BM, int M, int T_total, int warmup, int rho,
    const float* __restrict__ p_p,
    const float* __restrict__ temp_p,
    const float* __restrict__ dayl_p,
    const int64_t* __restrict__ b_p,
    const int64_t* __restrict__ s_p,
    const float* __restrict__ par_p,
    const float* __restrict__ hist_p,
    const float* __restrict__ dq_di_p,
    const float* __restrict__ dq_b_p,
    float* __restrict__ dpar_p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BM) return;

    int b = idx / M;
    int64_t basin = b_p[b];
    int64_t s0 = s_p[b] + warmup;
    const float* pm = par_p + idx * 16;
    float fc = pm[0], beta = pm[1], pwp = pm[2], l_th = pm[3];
    float ks = pm[4], ki = pm[5], kb = pm[6], kperc = pm[7];
    float coeff_pet = pm[8], ddf = pm[9], scf = pm[10], ts = pm[11];
    float tm = pm[12], tti = pm[13], whc = pm[14], crf = pm[15];

    float g_fc = 0.0f, g_beta = 0.0f, g_pwp = 0.0f, g_l = 0.0f;
    float g_ks = 0.0f, g_ki = 0.0f, g_kb = 0.0f, g_kperc = 0.0f;
    float g_coeff_pet = 0.0f, g_ddf = 0.0f, g_scf = 0.0f, g_ts = 0.0f;
    float g_tm = 0.0f, g_tti = 0.0f, g_whc = 0.0f, g_crf = 0.0f;

    float d_sup = 0.0f, d_slow = 0.0f, d_ssnow = 0.0f, d_ssliq = 0.0f, d_ssma = 0.0f;

    const float* p_row = p_p + basin * T_total + s0;
    const float* t_row = temp_p + basin * T_total + s0;
    const float* d_row = dayl_p + basin * T_total + s0;
    const float* dq_di_bm = dq_di_p + idx * rho;
    const float* dq_b_bm = dq_b_p + idx * rho;

    for (int t = rho - 1; t >= 0; --t) {
        size_t hbase = (size_t)t * 5 * BM + idx;
        float s_up = hist_p[hbase];
        float s_low = hist_p[hbase + BM];
        float s_snow = hist_p[hbase + 2 * BM];
        float s_sliq = hist_p[hbase + 3 * BM];
        float s_sma = hist_p[hbase + 4 * BM];

        float ct = t_row[t];
        float cp = p_row[t];
        float dl = d_row[t];
        float esat = 0.611f * expf(17.27f * ct / (ct + 237.3f));
        float pe_base = 29.8f * dl * (esat / (ct + 273.2f));
        float pe_t = coeff_pet * pe_base;

        float inv_tti = 1.0f / (tti + 1e-3f);
        float raw_sf = -inv_tti * (ct - ts) + 1.0f;
        float snowfrac = fminf(fmaxf(raw_sf, 0.0f), 1.0f);
        float snow = cp * snowfrac;
        float rain = cp * (1.0f - snowfrac);

        float pot_melt = (ct > tm) ? (ddf * (ct - tm)) : 0.0f;
        float melt = fminf(pot_melt, s_snow);
        float raw_ssnow1 = s_snow - melt;
        float s_snow1 = fmaxf(raw_ssnow1, 0.0f);
        float raw_ssliq1 = s_sliq + melt + rain;
        float s_sliq1 = fmaxf(raw_ssliq1, 0.0f);
        float raw_liqmax = s_snow1 * whc;
        float liqmax = fmaxf(raw_liqmax, 0.0f);
        float raw_preff = s_sliq1 - liqmax;
        float pr_eff = fmaxf(raw_preff, 0.0f);
        float s_sliq2 = fminf(s_sliq1, liqmax);

        float pot_refreeze = (ct < tm) ? ((tm - ct) * ddf * crf) : 0.0f;
        float refreeze = fminf(pot_refreeze, s_sliq2);
        float raw_ssnow2 = s_snow1 + refreeze + snow * scf;
        float raw_ssliq3 = s_sliq2 - refreeze;

        float raw_eff_base = s_sma / fc;
        float eff_ratio_base = fminf(fmaxf(raw_eff_base, 1e-4f), 1.0f);
        float effratio = powf(eff_ratio_base, beta);
        float remainwater = pr_eff * (1.0f - effratio);
        float sum_rw_sma = remainwater + s_sma;
        float min_rw_fc = fminf(sum_rw_sma, fc);
        float added = min_rw_fc - s_sma;
        float peff = pr_eff - added;
        float raw_ssma1 = s_sma + added;
        float s_sma1 = fmaxf(raw_ssma1, 0.0f);

        float denom_pwp = pwp * fc + 1e-3f;
        float raw_pef = s_sma1 / denom_pwp;
        float potevap_factor = fminf(fmaxf(raw_pef, 0.0f), 1.0f);
        bool sma_gt_pwp = (s_sma1 > pwp * fc);
        float pet = sma_gt_pwp ? pe_t : (pe_t * potevap_factor);
        float et = fminf(pet, s_sma1);
        float raw_ssma2 = s_sma1 - et;

        float raw_sup1 = s_up + peff;
        float s_up1 = fmaxf(raw_sup1, 0.0f);
        float raw_sup_l = s_up1 - l_th;
        float rel_sup_l = fmaxf(raw_sup_l, 0.0f);
        float qs = rel_sup_l * ks;
        float min_l_sup1 = fminf(l_th, s_up1);
        float qi = min_l_sup1 * ki;
        float raw_rem_up = s_up1 - qs - qi;
        float rel_rem_up = fmaxf(raw_rem_up, 0.0f);
        float qperc = rel_rem_up * kperc;
        float raw_sup2 = s_up1 - (qs + qi + qperc);

        float raw_slow1 = s_low + qperc;
        float s_low1 = fmaxf(raw_slow1, 0.0f);
        float qb = s_low1 * kb;
        float raw_slow2 = s_low1 - qb;

        float d_sup2 = d_sup;
        float d_slow2 = d_slow;
        float d_ssnow2 = d_ssnow;
        float d_ssliq3 = d_ssliq;
        float d_ssma2 = d_ssma;

        float g_qq = dq_di_bm[t];
        float g_qb_out = dq_b_bm[t];

        float d_raw_slow2 = (raw_slow2 > 0.0f) ? d_slow2 : 0.0f;
        float g_qb = g_qb_out - d_raw_slow2;
        float d_slow1 = d_raw_slow2 + g_qb * kb;
        g_kb += g_qb * s_low1;
        float d_raw_slow1 = (raw_slow1 > 0.0f) ? d_slow1 : 0.0f;
        float d_slow_prev = d_raw_slow1;
        float g_qperc = d_raw_slow1;

        float d_raw_sup2 = (raw_sup2 > 0.0f) ? d_sup2 : 0.0f;
        float d_sup1 = d_raw_sup2;
        float g_qs = g_qq - d_raw_sup2;
        float g_qi = g_qq - d_raw_sup2;
        g_qperc -= d_raw_sup2;

        g_kperc += g_qperc * rel_rem_up;
        float d_rel_rem_up = g_qperc * kperc;
        if (raw_rem_up > 0.0f) {
            d_sup1 += d_rel_rem_up;
            g_qs -= d_rel_rem_up;
            g_qi -= d_rel_rem_up;
        }

        g_ki += g_qi * min_l_sup1;
        float d_min_l_sup1 = g_qi * ki;
        if (l_th < s_up1) {
            g_l += d_min_l_sup1;
        } else {
            d_sup1 += d_min_l_sup1;
        }

        g_ks += g_qs * rel_sup_l;
        float d_rel_sup_l = g_qs * ks;
        if (raw_sup_l > 0.0f) {
            d_sup1 += d_rel_sup_l;
            g_l -= d_rel_sup_l;
        }

        float d_raw_sup1 = (raw_sup1 > 0.0f) ? d_sup1 : 0.0f;
        float d_sup_prev = d_raw_sup1;
        float g_peff = d_raw_sup1;

        float d_raw_ssma2 = (raw_ssma2 > 0.0f) ? d_ssma2 : 0.0f;
        float d_ssma1 = d_raw_ssma2;
        float g_et = -d_raw_ssma2;
        float g_pet = 0.0f;
        if (pet < s_sma1) {
            g_pet = g_et;
        } else {
            d_ssma1 += g_et;
        }

        float g_pe_t = 0.0f;
        if (sma_gt_pwp) {
            g_pe_t = g_pet;
        } else {
            g_pe_t = g_pet * potevap_factor;
            float g_pef = g_pet * pe_t;
            if (raw_pef > 0.0f && raw_pef < 1.0f) {
                float d_raw_pef = g_pef / denom_pwp;
                d_ssma1 += d_raw_pef;
                float d_denom_pwp = -d_raw_pef * raw_pef;
                g_pwp += d_denom_pwp * fc;
                g_fc += d_denom_pwp * pwp;
            }
        }
        g_coeff_pet += g_pe_t * pe_base;

        float d_raw_ssma1 = (raw_ssma1 > 0.0f) ? d_ssma1 : 0.0f;
        float d_ssma_prev = d_raw_ssma1;
        float g_added = d_raw_ssma1 - g_peff;
        float g_preff = g_peff;

        float d_min_rw_fc = g_added;
        d_ssma_prev -= g_added;
        float g_remainwater = 0.0f;
        if (sum_rw_sma < fc) {
            g_remainwater = d_min_rw_fc;
            d_ssma_prev += d_min_rw_fc;
        } else {
            g_fc += d_min_rw_fc;
        }

        g_preff += g_remainwater * (1.0f - effratio);
        float g_effratio = -g_remainwater * pr_eff;
        g_beta += g_effratio * effratio * logf(eff_ratio_base);
        float d_eff_base = g_effratio * beta * powf(eff_ratio_base, beta - 1.0f);
        if (raw_eff_base > 1e-4f && raw_eff_base < 1.0f) {
            d_ssma_prev += d_eff_base / fc;
            g_fc -= d_eff_base * raw_eff_base / fc;
        }

        float d_raw_ssliq3 = (raw_ssliq3 > 0.0f) ? d_ssliq3 : 0.0f;
        float d_raw_ssnow2 = (raw_ssnow2 > 0.0f) ? d_ssnow2 : 0.0f;
        float d_ssnow1 = d_raw_ssnow2;
        float d_ssliq2 = d_raw_ssliq3;
        float g_refreeze = d_raw_ssnow2 - d_raw_ssliq3;
        float g_snow = d_raw_ssnow2 * scf;
        g_scf += d_raw_ssnow2 * snow;

        if (pot_refreeze < s_sliq2) {
            if (ct < tm) {
                g_tm += g_refreeze * ddf * crf;
                g_ddf += g_refreeze * (tm - ct) * crf;
                g_crf += g_refreeze * (tm - ct) * ddf;
            }
        } else {
            d_ssliq2 += g_refreeze;
        }

        float d_ssliq1 = 0.0f;
        float g_liqmax = 0.0f;
        if (s_sliq1 < liqmax) {
            d_ssliq1 += d_ssliq2;
        } else {
            g_liqmax += d_ssliq2;
        }
        if (raw_preff > 0.0f) {
            d_ssliq1 += g_preff;
            g_liqmax -= g_preff;
        }
        if (raw_liqmax > 0.0f) {
            d_ssnow1 += g_liqmax * whc;
            g_whc += g_liqmax * s_snow1;
        }

        float d_raw_ssliq1 = (raw_ssliq1 > 0.0f) ? d_ssliq1 : 0.0f;
        float d_raw_ssnow1 = (raw_ssnow1 > 0.0f) ? d_ssnow1 : 0.0f;
        float d_ssliq_prev = d_raw_ssliq1;
        float g_rain = d_raw_ssliq1;
        float g_melt = d_raw_ssliq1 - d_raw_ssnow1;
        float d_ssnow_prev = d_raw_ssnow1;

        if (pot_melt < s_snow) {
            if (ct > tm) {
                g_ddf += g_melt * (ct - tm);
                g_tm -= g_melt * ddf;
            }
        } else {
            d_ssnow_prev += g_melt;
        }

        float g_snowfrac = (g_snow - g_rain) * cp;
        if (raw_sf > 0.0f && raw_sf < 1.0f) {
            g_ts += g_snowfrac * inv_tti;
            g_tti += g_snowfrac * (ct - ts) * inv_tti * inv_tti;
        }

        d_sup = (s_up >= 0.0f && s_up <= 2000.0f) ? d_sup_prev : 0.0f;
        d_slow = (s_low >= 0.0f && s_low <= 2000.0f) ? d_slow_prev : 0.0f;
        d_ssnow = (s_snow >= 0.0f && s_snow <= 2000.0f) ? d_ssnow_prev : 0.0f;
        d_ssliq = (s_sliq >= 0.0f && s_sliq <= 2000.0f) ? d_ssliq_prev : 0.0f;
        d_ssma = (s_sma >= 0.0f && s_sma <= 2000.0f) ? d_ssma_prev : 0.0f;
    }

    float* dp = dpar_p + idx * 16;
    dp[0] = g_fc; dp[1] = g_beta; dp[2] = g_pwp; dp[3] = g_l;
    dp[4] = g_ks; dp[5] = g_ki; dp[6] = g_kb; dp[7] = g_kperc;
    dp[8] = g_coeff_pet; dp[9] = g_ddf; dp[10] = g_scf; dp[11] = g_ts;
    dp[12] = g_tm; dp[13] = g_tti; dp[14] = g_whc; dp[15] = g_crf;
}

std::vector<torch::Tensor> poudel_window_forward_cuda(
    torch::Tensor p_full,
    torch::Tensor temp_full,
    torch::Tensor dayl_full,
    torch::Tensor b_idx,
    torch::Tensor s_idx,
    int warmup,
    int rho,
    torch::Tensor params
) {
    int B = b_idx.size(0);
    int T_total = p_full.size(1);
    int M = params.size(1);
    int BM = B * M;

    bool save_hist = true;
    auto opts = params.options();
    auto q_di = torch::empty({B, M, rho}, opts);
    auto q_b = torch::empty({B, M, rho}, opts);
    auto states_hist = torch::empty({rho, 5, BM}, opts);

    int threads = 256;
    int blocks = (BM + threads - 1) / threads;
    poudel_window_forward_kernel<<<blocks, threads>>>(
        BM, M, T_total, warmup, rho, save_hist,
        p_full.data_ptr<float>(),
        temp_full.data_ptr<float>(),
        dayl_full.data_ptr<float>(),
        b_idx.data_ptr<int64_t>(),
        s_idx.data_ptr<int64_t>(),
        params.data_ptr<float>(),
        q_di.data_ptr<float>(),
        q_b.data_ptr<float>(),
        states_hist.data_ptr<float>()
    );
    return {q_di, q_b, states_hist};
}

torch::Tensor poudel_window_backward_cuda(
    torch::Tensor p_full,
    torch::Tensor temp_full,
    torch::Tensor dayl_full,
    torch::Tensor b_idx,
    torch::Tensor s_idx,
    int warmup,
    int rho,
    torch::Tensor params,
    torch::Tensor states_hist,
    torch::Tensor d_q_di,
    torch::Tensor d_q_b
) {
    int B = b_idx.size(0);
    int T_total = p_full.size(1);
    int M = params.size(1);
    int BM = B * M;

    auto d_params = torch::zeros_like(params);
    int threads = 256;
    int blocks = (BM + threads - 1) / threads;
    poudel_window_backward_kernel<<<blocks, threads>>>(
        BM, M, T_total, warmup, rho,
        p_full.data_ptr<float>(),
        temp_full.data_ptr<float>(),
        dayl_full.data_ptr<float>(),
        b_idx.data_ptr<int64_t>(),
        s_idx.data_ptr<int64_t>(),
        params.data_ptr<float>(),
        states_hist.data_ptr<float>(),
        d_q_di.data_ptr<float>(),
        d_q_b.data_ptr<float>(),
        d_params.data_ptr<float>()
    );
    return d_params;
}
"""


def _get_poudel_cuda_ext():
    global _POUDEL_CUDA_EXT
    if _POUDEL_CUDA_EXT is not None:
        return _POUDEL_CUDA_EXT
    from torch.utils.cpp_extension import load_inline

    _POUDEL_CUDA_EXT = load_inline(
        name="poudel_window_cuda_v3",
        cpp_sources=_POUDEL_CUDA_CPP,
        cuda_sources=_POUDEL_CUDA_CU,
        functions=[
            "poudel_window_forward_cuda",
            "poudel_window_backward_cuda",
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _POUDEL_CUDA_EXT


def _get_poudel_omp_ext():
    global _POUDEL_OMP_EXT
    if _POUDEL_OMP_EXT is not None:
        return _POUDEL_OMP_EXT
    from torch.utils.cpp_extension import load_inline

    _POUDEL_OMP_EXT = load_inline(
        name="poudel_exact_omp_v3",
        cpp_sources=_POUDEL_OMP_CPP,
        functions=[
            "poudel_exact_forward_cpu",
            "poudel_exact_backward_cpu",
            "poudel_window_forward_cpu",
            "poudel_window_backward_cpu",
        ],
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )
    return _POUDEL_OMP_EXT


class PoudelWindowMhbvFunction(torch.autograd.Function):
    """Zero-copy CUDA/OpenMP C++ autograd wrapper combining spinup + target window."""

    @staticmethod
    def forward(ctx, p_full, temp_full, dayl_full, b_idx, s_idx, warmup, rho, params_16):
        dev = params_16.device
        ctx.dev = dev
        ctx.warmup = int(warmup)
        ctx.rho = int(rho)
        if dev.type == "cuda" and p_full.is_cuda:
            ext = _get_poudel_cuda_ext()
            par_c = params_16.contiguous()
            q_di, q_b, states_hist = ext.poudel_window_forward_cuda(
                p_full, temp_full, dayl_full, b_idx, s_idx, int(warmup), int(rho), par_c
            )
            ctx.save_for_backward(p_full, temp_full, dayl_full, b_idx, s_idx, par_c, states_hist)
            return q_di, q_b
        ext = _get_poudel_omp_ext()
        par_c = params_16.detach().cpu().contiguous()
        q_di, q_b, states_hist = ext.poudel_window_forward_cpu(
            p_full, temp_full, dayl_full, b_idx, s_idx, int(warmup), int(rho), par_c
        )
        ctx.save_for_backward(p_full, temp_full, dayl_full, b_idx, s_idx, par_c, states_hist)
        if dev.type != "cpu":
            return q_di.to(dev), q_b.to(dev)
        return q_di, q_b

    @staticmethod
    def backward(ctx, d_q_di, d_q_b):
        p_full, temp_full, dayl_full, b_idx, s_idx, par_c, states_hist = ctx.saved_tensors
        if ctx.dev.type == "cuda" and p_full.is_cuda:
            ext = _get_poudel_cuda_ext()
            d_params = ext.poudel_window_backward_cuda(
                p_full,
                temp_full,
                dayl_full,
                b_idx,
                s_idx,
                ctx.warmup,
                ctx.rho,
                par_c,
                states_hist,
                d_q_di.contiguous(),
                d_q_b.contiguous(),
            )
            return None, None, None, None, None, None, None, d_params
        ext = _get_poudel_omp_ext()
        d_params = ext.poudel_window_backward_cpu(
            p_full,
            temp_full,
            dayl_full,
            b_idx,
            s_idx,
            ctx.warmup,
            ctx.rho,
            par_c,
            states_hist,
            d_q_di.detach().cpu().contiguous(),
            d_q_b.detach().cpu().contiguous(),
        )
        if ctx.dev.type != "cpu":
            d_params = d_params.to(ctx.dev)
        return None, None, None, None, None, None, None, d_params


class PoudelExactMhbvFunction(torch.autograd.Function):
    """Exact OpenMP C++ autograd wrapper for snpoudel/diff-hydro 16-unit MHBV."""

    @staticmethod
    def forward(ctx, p, temp, daylen, params_16, init_states):
        ext = _get_poudel_omp_ext()
        dev = params_16.device
        p_c = p.detach().cpu().contiguous()
        temp_c = temp.detach().cpu().contiguous()
        dayl_c = daylen.detach().cpu().contiguous()
        par_c = params_16.detach().cpu().contiguous()
        init_c = init_states.detach().cpu().contiguous()
        q_di, q_b, states_hist, final_states = ext.poudel_exact_forward_cpu(
            p_c, temp_c, dayl_c, par_c, init_c
        )
        ctx.dev = dev
        ctx.save_for_backward(p_c, temp_c, dayl_c, par_c, states_hist)
        ctx.mark_non_differentiable(final_states)
        if dev.type != "cpu":
            return q_di.to(dev), q_b.to(dev), final_states.to(dev)
        return q_di, q_b, final_states

    @staticmethod
    def backward(ctx, d_q_di, d_q_b, _d_final_states):
        ext = _get_poudel_omp_ext()
        p_c, temp_c, dayl_c, par_c, states_hist = ctx.saved_tensors
        d_params = ext.poudel_exact_backward_cpu(
            p_c,
            temp_c,
            dayl_c,
            par_c,
            states_hist,
            d_q_di.detach().cpu().contiguous(),
            d_q_b.detach().cpu().contiguous(),
        )
        if ctx.dev.type != "cpu":
            d_params = d_params.to(ctx.dev)
        return None, None, None, d_params, None


def gamma_unit_hydrograph(
    t: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Two-parameter Gamma unit hydrograph impulse response (Poudel & Steinschneider, 2026)."""
    eps = 1e-6
    return (
        (t + eps).pow(alpha - 1.0)
        * torch.exp(-(t + eps) / theta)
        / (theta.pow(alpha) * torch.lgamma(alpha).exp())
    )


def apply_multi_gamma_routing(
    qin: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Full-sequence causal 1D convolution matching snpoudel/diff-hydro via exact FFT."""
    bsz, units, t_len = qin.shape
    t = torch.arange(t_len, device=qin.device, dtype=qin.dtype)
    uh = gamma_unit_hydrograph(t, alpha, theta)
    uh = uh / (uh.sum(dim=-1, keepdim=True) + 1e-12)
    n_fft = 1 << (2 * t_len - 1).bit_length()
    q_f = torch.fft.rfft(qin, n=n_fft, dim=-1)
    k_f = torch.fft.rfft(uh, n=n_fft, dim=-1)
    return torch.fft.irfft(q_f * k_f, n=n_fft, dim=-1)[..., :t_len]


class PoudelHBVModel(nn.Module):
    """Poudel & Steinschneider (2026) MLPParameterNet + 16-Unit DifferentiableMHBV."""

    def __init__(
        self,
        stat_in_dim: int = 27,
        nmul: int = 16,
        hidden_dim: int = 2048,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.nmul = nmul
        self.n_params = 20
        self.register_buffer(
            "bounds",
            torch.tensor(HBV_BOUNDS_LIST, dtype=torch.float32).repeat(nmul, 1),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(stat_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, nmul * self.n_params),
        )

    def constrain_parameters(self, x_s: torch.Tensor) -> torch.Tensor:
        raw_pars = self.mlp(x_s)
        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]
        constrained = mins + (maxs - mins) * torch.sigmoid(raw_pars)
        return constrained.view(raw_pars.shape[0], self.nmul, self.n_params)

    @torch.no_grad()
    def run_spinup(
        self,
        pars_all: torch.Tensor,
        p: torch.Tensor,
        temp: torch.Tensor,
        daylen: torch.Tensor,
    ) -> torch.Tensor:
        ext = _get_poudel_omp_ext()
        bsz = p.shape[0]
        init_c = torch.zeros((bsz, self.nmul, 5), dtype=torch.float32)
        _, _, _, final_states = ext.poudel_exact_forward_cpu(
            p.detach().cpu().contiguous(),
            temp.detach().cpu().contiguous(),
            daylen.detach().cpu().contiguous(),
            pars_all[:, :, :16].detach().cpu().contiguous(),
            init_c,
        )
        return final_states.to(pars_all.device)

    def forward_from_pars(
        self,
        pars_all: torch.Tensor,
        p: torch.Tensor,
        temp: torch.Tensor,
        daylen: torch.Tensor,
        init_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = p.shape[0]
        if init_states is None:
            init_states = torch.zeros(
                (bsz, self.nmul, 5), dtype=p.dtype, device=pars_all.device
            )
        params_16 = pars_all[:, :, :16].contiguous()
        d_shape = pars_all[:, :, 16:17]
        d_scale = pars_all[:, :, 17:18]
        b_shape = pars_all[:, :, 18:19]
        b_scale = pars_all[:, :, 19:20]

        q_di, q_b, _ = PoudelExactMhbvFunction.apply(
            p, temp, daylen, params_16, init_states
        )
        qdirect = apply_multi_gamma_routing(q_di, d_shape, d_scale)
        qbase = apply_multi_gamma_routing(q_b, b_shape, b_scale)
        return (qdirect + qbase).mean(dim=1)

    def forward_window_from_pars(
        self,
        pars_all: torch.Tensor,
        p_full: torch.Tensor,
        temp_full: torch.Tensor,
        dayl_full: torch.Tensor,
        b_idx: torch.Tensor,
        s_idx: torch.Tensor,
        warmup: int,
        rho: int,
    ) -> torch.Tensor:
        params_16 = pars_all[:, :, :16].contiguous()
        d_shape = pars_all[:, :, 16:17]
        d_scale = pars_all[:, :, 17:18]
        b_shape = pars_all[:, :, 18:19]
        b_scale = pars_all[:, :, 19:20]

        q_di, q_b = PoudelWindowMhbvFunction.apply(
            p_full, temp_full, dayl_full, b_idx, s_idx, warmup, rho, params_16
        )
        qdirect = apply_multi_gamma_routing(q_di, d_shape, d_scale)
        qbase = apply_multi_gamma_routing(q_b, b_shape, b_scale)
        return (qdirect + qbase).mean(dim=1)

    def forward(
        self,
        p: torch.Tensor,
        temp: torch.Tensor,
        daylen: torch.Tensor,
        x_s: torch.Tensor,
        init_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pars_all = self.constrain_parameters(x_s)
        return self.forward_from_pars(pars_all, p, temp, daylen, init_states=init_states)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unnormalized masked MSE in mm^2/d^2 matching 01best_mlp_hbv.py."""
    m = torch.isfinite(target) & (target >= 0.0)
    if int(m.sum().item()) == 0:
        return pred.sum() * 0.0
    return ((pred[m] - target[m]) ** 2).mean()


class PoudelHBVTrainer:
    """Faithful trainer for Poudel & Steinschneider (2026) MLP + 16-HBV."""

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
        epochs: int = 30,
        batch_size: int = 128,
        rho: int = 365,
        warmup: int = 730,
        stride: int = 60,
        nmul: int = 16,
        hidden_dim: int = 2048,
        dropout: float = 0.4,
        lr: float = 1e-4,
        lr_patience: int = 5,
        early_stop_patience: int = 10,
        spinup_days: Optional[int] = None,
        train_doy: Optional[np.ndarray] = None,
        test_doy: Optional[np.ndarray] = None,
        lat_train: Optional[np.ndarray] = None,
        lat_test: Optional[np.ndarray] = None,
        n_batches_per_epoch: Optional[int] = None,
    ) -> np.ndarray:
        """Train MLP+16HBV with Poudel & Steinschneider (2026) windowing and loss."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        n_train, t_train, _ = x_dyn_train.shape
        n_test, t_test, _ = x_dyn_test.shape
        s_dim = x_stat_train.shape[1]

        if spinup_days is None:
            spinup_days = warmup
        eff_spinup = min(int(spinup_days), max(0, t_train - 2))
        eff_warmup = min(int(warmup), eff_spinup if eff_spinup > 0 else max(1, t_train // 4))
        eff_rho = min(int(rho), max(2, t_train - eff_warmup))
        seq_total = eff_warmup + eff_rho

        # Mask pre-train_start spinup discharge so pre-period targets never enter loss
        y_train_clean = y_obs_train.astype(np.float32, copy=True)
        if eff_spinup > 0:
            y_train_clean[:, :eff_spinup] = np.nan

        p_tr, t_tr, d_tr = _derive_physical_p_t_daylen(
            x_dyn_train, lat_deg=lat_train, doy=train_doy
        )
        p_te, t_te, d_te = _derive_physical_p_t_daylen(
            x_dyn_test, lat_deg=lat_test, doy=test_doy
        )

        xs_mean = np.nanmean(x_stat_train, axis=0, keepdims=True)
        xs_std = np.nanstd(x_stat_train, axis=0, keepdims=True) + 1e-8
        xs_tr = np.nan_to_num((x_stat_train - xs_mean) / xs_std, nan=0.0).astype(np.float32)
        xs_te = np.nan_to_num((x_stat_test - xs_mean) / xs_std, nan=0.0).astype(np.float32)

        # Build deterministic sliding windows with stride=60 matching 01best_mlp_hbv.py
        first_win_start = max(0, eff_spinup - eff_warmup)
        n_target_train = t_train - eff_spinup
        if n_target_train >= 5000:
            # 15-year Stage-2 period (1980-1995): 10y cal (1980-1990) + 5y val (1990-1995)
            val_days = 1826
            cal_end = t_train - val_days
            cal_starts = list(range(first_win_start, cal_end - seq_total + 1, max(1, stride)))
            val_first_start = max(first_win_start, cal_end - eff_warmup)
            val_starts = list(range(val_first_start, t_train - seq_total + 1, max(1, stride)))
            tr_wins = [
                (b, s)
                for b in range(n_train)
                for s in cal_starts
                if np.any(np.isfinite(y_train_clean[b, s + eff_warmup : s + seq_total]))
            ]
            va_wins = [
                (b, s)
                for b in range(n_train)
                for s in val_starts
                if np.any(np.isfinite(y_train_clean[b, s + eff_warmup : s + seq_total]))
            ]
            if not tr_wins:
                tr_wins = [(b, first_win_start) for b in range(n_train)]
            if not va_wins:
                va_wins = tr_wins
        else:
            starts = list(range(first_win_start, t_train - seq_total + 1, max(1, stride)))
            if not starts:
                starts = [first_win_start]
            all_wins = [
                (b, s)
                for b in range(n_train)
                for s in starts
                if np.any(np.isfinite(y_train_clean[b, s + eff_warmup : s + seq_total]))
            ]
            if not all_wins:
                all_wins = [(b, first_win_start) for b in range(n_train)]
            split_rng = np.random.RandomState(self.seed + 1000)
            perm_all = split_rng.permutation(len(all_wins))
            if len(all_wins) >= 20:
                n_val = max(1, len(all_wins) // 10)
                va_wins = [all_wins[i] for i in perm_all[:n_val]]
                tr_wins = [all_wins[i] for i in perm_all[n_val:]]
            else:
                tr_wins = all_wins
                va_wins = all_wins

        p_tr_dev = torch.from_numpy(np.ascontiguousarray(p_tr, dtype=np.float32)).to(self.device)
        t_tr_dev = torch.from_numpy(np.ascontiguousarray(t_tr, dtype=np.float32)).to(self.device)
        d_tr_dev = torch.from_numpy(np.ascontiguousarray(d_tr, dtype=np.float32)).to(self.device)

        tr_b_dev = torch.tensor([w[0] for w in tr_wins], dtype=torch.int64, device=self.device)
        tr_s_dev = torch.tensor([w[1] for w in tr_wins], dtype=torch.int64, device=self.device)
        tr_y_dev = torch.from_numpy(
            np.stack([y_train_clean[b, s + eff_warmup : s + seq_total] for b, s in tr_wins], axis=0)
        ).to(self.device)

        va_b_dev = torch.tensor([w[0] for w in va_wins], dtype=torch.int64, device=self.device)
        va_s_dev = torch.tensor([w[1] for w in va_wins], dtype=torch.int64, device=self.device)
        va_y_dev = torch.from_numpy(
            np.stack([y_train_clean[b, s + eff_warmup : s + seq_total] for b, s in va_wins], axis=0)
        ).to(self.device)

        model = PoudelHBVModel(
            stat_in_dim=s_dim,
            nmul=nmul,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=lr_patience, min_lr=1e-6
        )
        rng = np.random.RandomState(self.seed)

        xs_tr_t = torch.from_numpy(xs_tr).to(self.device)
        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0

        for _ep in range(1, epochs + 1):
            model.train()
            order_np = rng.permutation(len(tr_wins))
            if n_batches_per_epoch is not None and int(n_batches_per_epoch) > 0:
                order_np = order_np[: int(n_batches_per_epoch) * batch_size]
            order_t = torch.from_numpy(order_np).to(self.device)

            for i in range(0, len(order_t), batch_size):
                sel = order_t[i : i + batch_size]
                b_idx_d = tr_b_dev[sel].contiguous()
                s_idx_d = tr_s_dev[sel].contiguous()
                q_targ = tr_y_dev[sel]
                s_b = xs_tr_t[b_idx_d]

                pars = model.constrain_parameters(s_b)
                q_sim = model.forward_window_from_pars(
                    pars,
                    p_tr_dev,
                    t_tr_dev,
                    d_tr_dev,
                    b_idx_d,
                    s_idx_d,
                    eff_warmup,
                    eff_rho,
                )
                loss = masked_mse(q_sim, q_targ)
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pars_eval_all = model.constrain_parameters(xs_tr_t)
                val_losses: list[float] = []
                val_bsz = batch_size * 4
                for i in range(0, len(va_wins), val_bsz):
                    b_idx_d = va_b_dev[i : i + val_bsz].contiguous()
                    s_idx_d = va_s_dev[i : i + val_bsz].contiguous()
                    q_targ = va_y_dev[i : i + val_bsz]
                    pars = pars_eval_all[b_idx_d]
                    q_sim = model.forward_window_from_pars(
                        pars,
                        p_tr_dev,
                        t_tr_dev,
                        d_tr_dev,
                        b_idx_d,
                        s_idx_d,
                        eff_warmup,
                        eff_rho,
                    )
                    v_l = masked_mse(q_sim, q_targ)
                    if torch.isfinite(v_l):
                        val_losses.append(float(v_l.item()))
                va_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            scheduler.step(va_loss)
            if va_loss < best_val:
                best_val = va_loss
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        preds = np.zeros((n_test, t_test - eval_start_idx), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_test, max(1, batch_size)):
                sl = slice(i, min(i + max(1, batch_size), n_test))
                q_full = model(
                    torch.from_numpy(p_te[sl]),
                    torch.from_numpy(t_te[sl]),
                    torch.from_numpy(d_te[sl]),
                    torch.from_numpy(xs_te[sl]).to(self.device),
                ).cpu().numpy()
                preds[sl] = np.clip(
                    np.nan_to_num(q_full[:, eval_start_idx:], nan=0.0), 0.0, None
                )
        return preds
