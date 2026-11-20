# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Independent first-party CUDA C++ and PyTorch implementation of the 16-bucket
# differentiable HBV (dHBV 1.0 and dHBV 1.1p) equations published in
# Feng et al. (2022, 2023) and Song et al. (2024, 2025).

"""dHBV 1.0 & dHBV 1.1p Standalone Differentiable HBV Core (CUDA + CPU Autograd).

References:
- Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable,
  regionalized process-based models with multiphysical outputs can approach
  state-of-the-art hydrologic prediction accuracy. Water Resources Research,
  58(10), e2022WR032404.
- Song, Y., Bindas, T., Shen, C., Ji, H., Knoben, W. J. M., Lonzarich, L., et al.
  (2025). High-resolution national-scale water modeling is enhanced by
  multiscale differentiable physics-informed machine learning. Water Resources
  Research, 61(4), e2024WR038928.

Shared by ``models.feng_dhbv10`` (dHBV 1.0) and ``models.song_dhbv11p`` (dHBV 1.1p).
The CPU path always runs the C++ kernel (``get_hbv_cpu``); OpenMP is enabled
where the compiler supports it (``-fopenmp`` is skipped on macOS / Apple clang,
in which case the ``#pragma omp`` loops run serially with identical results).
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

__global__ void hbv_forward_kernel(
    const float* __restrict__ Pm,
    const float* __restrict__ Tm,
    const float* __restrict__ PETm,
    const float* __restrict__ init_states, // [5, B, M]
    const float* __restrict__ parBETA,     // [T or 1, B, M]
    const float* __restrict__ parFC,       // [T or 1, B, M]
    const float* __restrict__ parK0,
    const float* __restrict__ parK1,
    const float* __restrict__ parK2,
    const float* __restrict__ parLP,
    const float* __restrict__ parPERC,
    const float* __restrict__ parUZL,
    const float* __restrict__ parTT,
    const float* __restrict__ parCFMAX,
    const float* __restrict__ parCFR,
    const float* __restrict__ parCWH,
    const float* __restrict__ parBETAET,
    const float* __restrict__ parC,
    int s_P, int s_BETA, int s_FC, int s_K0, int s_K1, int s_K2,
    int s_LP, int s_PERC, int s_UZL, int s_TT, int s_CFMAX, int s_CFR,
    int s_CWH, int s_BETAET, int s_C,
    float* __restrict__ Q0_out,            // [T, B, M]
    float* __restrict__ Q1_out,
    float* __restrict__ Q2_out,
    float* __restrict__ states_hist,       // [5, T, B, M] if save_states
    float* __restrict__ final_states,      // [5, B, M]
    int T, int B, int M, int BM, float nearzero, bool has_capillary, bool save_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BM) return;
    int b = idx / M;
    int p_base = (s_P == 1) ? b : idx;
    int p_step = (s_P == 1) ? B : BM;

    float sp = init_states[0 * BM + idx];
    float mw = init_states[1 * BM + idx];
    float sm = init_states[2 * BM + idx];
    float suz = init_states[3 * BM + idx];
    float slz = init_states[4 * BM + idx];

    float FC_reg = (s_FC == 0) ? parFC[idx] : 0.0f;
    float K0_reg = (s_K0 == 0) ? parK0[idx] : 0.0f;
    float K1_reg = (s_K1 == 0) ? parK1[idx] : 0.0f;
    float K2_reg = (s_K2 == 0) ? parK2[idx] : 0.0f;
    float LP_reg = (s_LP == 0) ? parLP[idx] : 0.0f;
    float PERC_reg = (s_PERC == 0) ? parPERC[idx] : 0.0f;
    float UZL_reg = (s_UZL == 0) ? parUZL[idx] : 0.0f;
    float TT_reg = (s_TT == 0) ? parTT[idx] : 0.0f;
    float CFMAX_reg = (s_CFMAX == 0) ? parCFMAX[idx] : 0.0f;
    float CFR_reg = (s_CFR == 0) ? parCFR[idx] : 0.0f;
    float CWH_reg = (s_CWH == 0) ? parCWH[idx] : 0.0f;
    float BETA_reg = (s_BETA == 0) ? parBETA[idx] : 0.0f;
    float BETAET_reg = (s_BETAET == 0) ? parBETAET[idx] : 0.0f;
    float C_reg = (has_capillary && s_C == 0) ? parC[idx] : 0.0f;

    int TBM = T * BM;

    for (int t = 0; t < T; ++t) {
        int tidx = t * BM + idx;
        if (save_states) {
            states_hist[0 * TBM + tidx] = sp;
            states_hist[1 * TBM + tidx] = mw;
            states_hist[2 * TBM + tidx] = sm;
            states_hist[3 * TBM + tidx] = suz;
            states_hist[4 * TBM + tidx] = slz;
        }

        int pidx = t * p_step + p_base;
        float P = Pm[pidx];
        float Tt = Tm[pidx];
        float PETt = PETm[pidx];

        float BETA = (s_BETA == 0) ? BETA_reg : parBETA[tidx];
        float FC = (s_FC == 0) ? FC_reg : parFC[tidx];
        float K0 = (s_K0 == 0) ? K0_reg : parK0[tidx];
        float K1 = (s_K1 == 0) ? K1_reg : parK1[tidx];
        float K2 = (s_K2 == 0) ? K2_reg : parK2[tidx];
        float LP = (s_LP == 0) ? LP_reg : parLP[tidx];
        float PERC_p = (s_PERC == 0) ? PERC_reg : parPERC[tidx];
        float UZL = (s_UZL == 0) ? UZL_reg : parUZL[tidx];
        float TT = (s_TT == 0) ? TT_reg : parTT[tidx];
        float CFMAX = (s_CFMAX == 0) ? CFMAX_reg : parCFMAX[tidx];
        float CFR = (s_CFR == 0) ? CFR_reg : parCFR[tidx];
        float CWH = (s_CWH == 0) ? CWH_reg : parCWH[tidx];
        float BETAET = (s_BETAET == 0) ? BETAET_reg : parBETAET[tidx];
        float C = has_capillary ? ((s_C == 0) ? C_reg : parC[tidx]) : 0.0f;

        float RAIN = (Tt >= TT) ? P : 0.0f;
        float SNOW = (Tt < TT) ? P : 0.0f;

        float sp1 = sp + SNOW;
        float melt_pos = fmaxf(CFMAX * (Tt - TT), 0.0f);
        float melt = fminf(melt_pos, sp1);
        float mw1 = mw + melt;
        float sp2 = sp1 - melt;
        float refrz_pos = fmaxf(CFR * CFMAX * (TT - Tt), 0.0f);
        float refrz = fminf(refrz_pos, mw1);
        float sp3 = sp2 + refrz;
        float mw2 = mw1 - refrz;
        float tosoil = fmaxf(mw2 - CWH * sp3, 0.0f);
        float mw3 = mw2 - tosoil;

        float infil = RAIN + tosoil;
        float wet_raw = powf(fmaxf(sm / FC, 0.0f), BETA);
        float wet = fminf(fmaxf(wet_raw, 0.0f), 1.0f);
        float recharge = infil * wet;
        float sm1 = sm + infil - recharge;
        float excess = fmaxf(sm1 - FC, 0.0f);
        float sm2 = sm1 - excess;
        float evapf_raw = powf(fmaxf(sm2 / (LP * FC), 0.0f), BETAET);
        float evapf = fminf(fmaxf(evapf_raw, 0.0f), 1.0f);
        float ETact = fminf(sm2, PETt * evapf);
        float sm3 = fmaxf(sm2 - ETact, nearzero);

        float sm4 = sm3;
        float slz1 = slz;
        if (has_capillary) {
            float sm3_clamped = fminf(sm3 / FC, 1.0f);
            float cap_raw = C * slz * (1.0f - sm3_clamped);
            float cap = fminf(slz, cap_raw);
            sm4 = fmaxf(sm3 + cap, nearzero);
            slz1 = fmaxf(slz - cap, nearzero);
        }

        float suz1 = suz + recharge + excess;
        float PERC = fminf(suz1, PERC_p);
        float suz2 = suz1 - PERC;
        float suz_ex = fmaxf(suz2 - UZL, 0.0f);
        float Q0 = K0 * suz_ex;
        float suz3 = suz2 - Q0;
        float Q1 = K1 * suz3;
        float suz4 = suz3 - Q1;
        float slz2 = slz1 + PERC;
        float Q2 = K2 * slz2;
        float slz3 = slz2 - Q2;

        Q0_out[tidx] = Q0;
        Q1_out[tidx] = Q1;
        Q2_out[tidx] = Q2;

        sp = sp3;
        mw = mw3;
        sm = sm4;
        suz = suz4;
        slz = slz3;
    }

    final_states[0 * BM + idx] = sp;
    final_states[1 * BM + idx] = mw;
    final_states[2 * BM + idx] = sm;
    final_states[3 * BM + idx] = suz;
    final_states[4 * BM + idx] = slz;
}

__global__ void hbv_backward_kernel(
    const float* __restrict__ Pm,
    const float* __restrict__ Tm,
    const float* __restrict__ PETm,
    const float* __restrict__ states_hist, // [5, T, B, M]
    const float* __restrict__ parBETA,
    const float* __restrict__ parFC,
    const float* __restrict__ parK0,
    const float* __restrict__ parK1,
    const float* __restrict__ parK2,
    const float* __restrict__ parLP,
    const float* __restrict__ parPERC,
    const float* __restrict__ parUZL,
    const float* __restrict__ parTT,
    const float* __restrict__ parCFMAX,
    const float* __restrict__ parCFR,
    const float* __restrict__ parCWH,
    const float* __restrict__ parBETAET,
    const float* __restrict__ parC,
    int s_P, int s_BETA, int s_FC, int s_K0, int s_K1, int s_K2,
    int s_LP, int s_PERC, int s_UZL, int s_TT, int s_CFMAX, int s_CFR,
    int s_CWH, int s_BETAET, int s_C,
    const float* __restrict__ dQ0_out,
    const float* __restrict__ dQ1_out,
    const float* __restrict__ dQ2_out,
    float* __restrict__ d_parBETA,
    float* __restrict__ d_parFC,
    float* __restrict__ d_parK0,
    float* __restrict__ d_parK1,
    float* __restrict__ d_parK2,
    float* __restrict__ d_parLP,
    float* __restrict__ d_parPERC,
    float* __restrict__ d_parUZL,
    float* __restrict__ d_parTT,
    float* __restrict__ d_parCFMAX,
    float* __restrict__ d_parCFR,
    float* __restrict__ d_parCWH,
    float* __restrict__ d_parBETAET,
    float* __restrict__ d_parC,
    int T, int B, int M, int BM, float nearzero, bool has_capillary
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BM) return;
    int b = idx / M;
    int p_base = (s_P == 1) ? b : idx;
    int p_step = (s_P == 1) ? B : BM;
    int TBM = T * BM;

    float d_sp = 0.0f;
    float d_mw = 0.0f;
    float d_sm = 0.0f;
    float d_suz = 0.0f;
    float d_slz = 0.0f;

    float FC_reg = (s_FC == 0) ? parFC[idx] : 0.0f;
    float K0_reg = (s_K0 == 0) ? parK0[idx] : 0.0f;
    float K1_reg = (s_K1 == 0) ? parK1[idx] : 0.0f;
    float K2_reg = (s_K2 == 0) ? parK2[idx] : 0.0f;
    float LP_reg = (s_LP == 0) ? parLP[idx] : 0.0f;
    float PERC_reg = (s_PERC == 0) ? parPERC[idx] : 0.0f;
    float UZL_reg = (s_UZL == 0) ? parUZL[idx] : 0.0f;
    float TT_reg = (s_TT == 0) ? parTT[idx] : 0.0f;
    float CFMAX_reg = (s_CFMAX == 0) ? parCFMAX[idx] : 0.0f;
    float CFR_reg = (s_CFR == 0) ? parCFR[idx] : 0.0f;
    float CWH_reg = (s_CWH == 0) ? parCWH[idx] : 0.0f;
    float BETA_reg = (s_BETA == 0) ? parBETA[idx] : 0.0f;
    float BETAET_reg = (s_BETAET == 0) ? parBETAET[idx] : 0.0f;
    float C_reg = (has_capillary && s_C == 0) ? parC[idx] : 0.0f;

    float acc_BETA = 0.0f, acc_FC = 0.0f, acc_K0 = 0.0f, acc_K1 = 0.0f, acc_K2 = 0.0f;
    float acc_LP = 0.0f, acc_PERC = 0.0f, acc_UZL = 0.0f, acc_TT = 0.0f;
    float acc_CFMAX = 0.0f, acc_CFR = 0.0f, acc_CWH = 0.0f, acc_BETAET = 0.0f, acc_C = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        int tidx = t * BM + idx;
        float sp0 = states_hist[0 * TBM + tidx];
        float mw0 = states_hist[1 * TBM + tidx];
        float sm0 = states_hist[2 * TBM + tidx];
        float suz0 = states_hist[3 * TBM + tidx];
        float slz0 = states_hist[4 * TBM + tidx];

        int pidx = t * p_step + p_base;
        float P = Pm[pidx];
        float Tt = Tm[pidx];
        float PETt = PETm[pidx];

        float BETA = (s_BETA == 0) ? BETA_reg : parBETA[tidx];
        float FC = (s_FC == 0) ? FC_reg : parFC[tidx];
        float K0 = (s_K0 == 0) ? K0_reg : parK0[tidx];
        float K1 = (s_K1 == 0) ? K1_reg : parK1[tidx];
        float K2 = (s_K2 == 0) ? K2_reg : parK2[tidx];
        float LP = (s_LP == 0) ? LP_reg : parLP[tidx];
        float PERC_p = (s_PERC == 0) ? PERC_reg : parPERC[tidx];
        float UZL = (s_UZL == 0) ? UZL_reg : parUZL[tidx];
        float TT = (s_TT == 0) ? TT_reg : parTT[tidx];
        float CFMAX = (s_CFMAX == 0) ? CFMAX_reg : parCFMAX[tidx];
        float CFR = (s_CFR == 0) ? CFR_reg : parCFR[tidx];
        float CWH = (s_CWH == 0) ? CWH_reg : parCWH[tidx];
        float BETAET = (s_BETAET == 0) ? BETAET_reg : parBETAET[tidx];
        float C = has_capillary ? ((s_C == 0) ? C_reg : parC[tidx]) : 0.0f;

        float RAIN = (Tt >= TT) ? P : 0.0f;
        float SNOW = (Tt < TT) ? P : 0.0f;

        float sp1 = sp0 + SNOW;
        float melt_raw = CFMAX * (Tt - TT);
        float melt_pos = fmaxf(melt_raw, 0.0f);
        float melt = fminf(melt_pos, sp1);
        float mw1 = mw0 + melt;
        float sp2 = sp1 - melt;
        float refrz_raw = CFR * CFMAX * (TT - Tt);
        float refrz_pos = fmaxf(refrz_raw, 0.0f);
        float refrz = fminf(refrz_pos, mw1);
        float sp3 = sp2 + refrz;
        float mw2 = mw1 - refrz;
        float tosoil_raw = mw2 - CWH * sp3;
        float tosoil = fmaxf(tosoil_raw, 0.0f);

        float infil = RAIN + tosoil;
        float ratio_fc = sm0 / FC;
        float wet_raw = (ratio_fc > 0.0f) ? powf(ratio_fc, BETA) : 0.0f;
        float wet = fminf(fmaxf(wet_raw, 0.0f), 1.0f);
        float recharge = infil * wet;
        float sm1 = sm0 + infil - recharge;
        float excs_raw = sm1 - FC;
        float excess = fmaxf(excs_raw, 0.0f);
        float sm2 = sm1 - excess;
        float denom_et = LP * FC;
        float ratio_et = sm2 / denom_et;
        float evapf_raw = (ratio_et > 0.0f) ? powf(ratio_et, BETAET) : 0.0f;
        float evapf = fminf(fmaxf(evapf_raw, 0.0f), 1.0f);
        float et_pot = PETt * evapf;
        float ETact = fminf(sm2, et_pot);
        float sm3_raw = sm2 - ETact;
        float sm3 = fmaxf(sm3_raw, nearzero);

        float sm3_ratio = sm3 / FC;
        float sm3_clamped = fminf(sm3_ratio, 1.0f);
        float cap_raw = has_capillary ? (C * slz0 * (1.0f - sm3_clamped)) : 0.0f;
        float cap = has_capillary ? fminf(slz0, cap_raw) : 0.0f;
        float sm4_raw = sm3 + cap;
        float slz1_raw = slz0 - cap;
        float slz1 = has_capillary ? fmaxf(slz1_raw, nearzero) : slz0;

        float suz1 = suz0 + recharge + excess;
        float PERC = fminf(suz1, PERC_p);
        float suz2 = suz1 - PERC;
        float suz_ex_raw = suz2 - UZL;
        float suz_ex = fmaxf(suz_ex_raw, 0.0f);
        float Q0 = K0 * suz_ex;
        float suz3 = suz2 - Q0;
        float slz2 = slz1 + PERC;

        float g_Q0 = dQ0_out[tidx];
        float g_Q1 = dQ1_out[tidx];
        float g_Q2 = dQ2_out[tidx];

        g_Q2 += -d_slz;
        float d_slz2 = d_slz + g_Q2 * K2;
        float g_K2 = g_Q2 * slz2;

        float d_slz1 = d_slz2;
        float g_PERC = d_slz2;

        g_Q1 += -d_suz;
        float d_suz3 = d_suz + g_Q1 * K1;
        float g_K1 = g_Q1 * suz3;

        g_Q0 += -d_suz3;
        float d_suz2 = d_suz3;
        float g_K0 = g_Q0 * suz_ex;
        float d_suz_ex = g_Q0 * K0;
        float d_suz_ex_raw = (suz_ex_raw >= 0.0f) ? d_suz_ex : 0.0f;
        d_suz2 += d_suz_ex_raw;
        float g_UZL = -d_suz_ex_raw;

        float d_suz1 = d_suz2;
        g_PERC += -d_suz2;
        float g_PERC_p = 0.0f;
        if (suz1 < PERC_p) {
            d_suz1 += g_PERC;
        } else if (PERC_p < suz1) {
            g_PERC_p = g_PERC;
        } else {
            d_suz1 += 0.5f * g_PERC;
            g_PERC_p = 0.5f * g_PERC;
        }

        float d_suz0 = d_suz1;
        float g_recharge = d_suz1;
        float g_excess = d_suz1;

        float d_sm3 = 0.0f;
        float d_slz0 = 0.0f;
        float g_FC = 0.0f;
        float g_C = 0.0f;
        if (has_capillary) {
            float d_sm4_raw = (sm4_raw >= nearzero) ? d_sm : 0.0f;
            float d_slz1_raw = (slz1_raw >= nearzero) ? d_slz1 : 0.0f;
            d_sm3 += d_sm4_raw;
            float g_cap = d_sm4_raw - d_slz1_raw;
            d_slz0 += d_slz1_raw;

            float g_cap_raw = 0.0f;
            if (slz0 < cap_raw) {
                d_slz0 += g_cap;
            } else if (cap_raw < slz0) {
                g_cap_raw = g_cap;
            } else {
                d_slz0 += 0.5f * g_cap;
                g_cap_raw = 0.5f * g_cap;
            }
            g_C = g_cap_raw * slz0 * (1.0f - sm3_clamped);
            d_slz0 += g_cap_raw * C * (1.0f - sm3_clamped);
            float d_sm3_clamped = -g_cap_raw * C * slz0;
            float d_sm3_ratio = (sm3_ratio <= 1.0f) ? d_sm3_clamped : 0.0f;
            d_sm3 += d_sm3_ratio / FC;
            g_FC += -d_sm3_ratio * sm3 / (FC * FC);
        } else {
            d_sm3 = d_sm;
            d_slz0 = d_slz1;
        }

        float d_sm3_raw = (sm3_raw >= nearzero) ? d_sm3 : 0.0f;
        float d_sm2 = d_sm3_raw;
        float g_ETact = -d_sm3_raw;

        float g_et_pot = 0.0f;
        if (sm2 < et_pot) {
            d_sm2 += g_ETact;
        } else if (et_pot < sm2) {
            g_et_pot = g_ETact;
        } else {
            d_sm2 += 0.5f * g_ETact;
            g_et_pot = 0.5f * g_ETact;
        }
        float g_evapf = g_et_pot * PETt;
        float g_evapf_raw = (evapf_raw >= 0.0f && evapf_raw <= 1.0f) ? g_evapf : 0.0f;
        float g_BETAET = 0.0f;
        float d_ratio_et = 0.0f;
        if (ratio_et > 0.0f) {
            d_ratio_et = g_evapf_raw * BETAET * powf(ratio_et, BETAET - 1.0f);
            g_BETAET = g_evapf_raw * evapf_raw * logf(ratio_et);
        }
        d_sm2 += d_ratio_et / denom_et;
        float d_denom_et = -d_ratio_et * sm2 / (denom_et * denom_et);
        float g_LP = d_denom_et * FC;
        g_FC += d_denom_et * LP;

        float d_sm1 = d_sm2;
        g_excess += -d_sm2;
        float d_excs_raw = (excs_raw >= 0.0f) ? g_excess : 0.0f;
        d_sm1 += d_excs_raw;
        g_FC += -d_excs_raw;

        float d_sm0 = d_sm1;
        float g_infil = d_sm1;
        g_recharge += -d_sm1;
        g_infil += g_recharge * wet;
        float g_wet = g_recharge * infil;
        float g_wet_raw = (wet_raw >= 0.0f && wet_raw <= 1.0f) ? g_wet : 0.0f;
        float g_BETA = 0.0f;
        float d_ratio_fc = 0.0f;
        if (ratio_fc > 0.0f) {
            d_ratio_fc = g_wet_raw * BETA * powf(ratio_fc, BETA - 1.0f);
            g_BETA = g_wet_raw * wet_raw * logf(ratio_fc);
        }
        d_sm0 += d_ratio_fc / FC;
        g_FC += -d_ratio_fc * sm0 / (FC * FC);

        float g_tosoil = g_infil - d_mw;
        float d_mw2 = d_mw;
        float d_sp3 = d_sp;
        float d_tosoil_raw = (tosoil_raw >= 0.0f) ? g_tosoil : 0.0f;
        d_mw2 += d_tosoil_raw;
        float g_CWH = -d_tosoil_raw * sp3;
        d_sp3 += -d_tosoil_raw * CWH;

        float d_sp2 = d_sp3;
        float d_mw1 = d_mw2;
        float g_refrz = d_sp3 - d_mw2;
        float g_refrz_pos = 0.0f;
        if (refrz_pos < mw1) {
            g_refrz_pos = g_refrz;
        } else if (mw1 < refrz_pos) {
            d_mw1 += g_refrz;
        } else {
            g_refrz_pos = 0.5f * g_refrz;
            d_mw1 += 0.5f * g_refrz;
        }
        float g_refrz_raw = (refrz_raw >= 0.0f) ? g_refrz_pos : 0.0f;
        float g_CFR = g_refrz_raw * CFMAX * (TT - Tt);
        float g_CFMAX = g_refrz_raw * CFR * (TT - Tt);
        float g_TT = g_refrz_raw * CFR * CFMAX;

        float d_mw0 = d_mw1;
        float d_sp1 = d_sp2;
        float g_melt = d_mw1 - d_sp2;
        float g_melt_pos = 0.0f;
        if (melt_pos < sp1) {
            g_melt_pos = g_melt;
        } else if (sp1 < melt_pos) {
            d_sp1 += g_melt;
        } else {
            g_melt_pos = 0.5f * g_melt;
            d_sp1 += 0.5f * g_melt;
        }
        float g_melt_raw = (melt_raw >= 0.0f) ? g_melt_pos : 0.0f;
        g_CFMAX += g_melt_raw * (Tt - TT);
        g_TT += -g_melt_raw * CFMAX;

        float d_sp0 = d_sp1;

        if (s_BETA == 0) acc_BETA += g_BETA; else d_parBETA[tidx] = g_BETA;
        if (s_FC == 0) acc_FC += g_FC; else d_parFC[tidx] = g_FC;
        if (s_K0 == 0) acc_K0 += g_K0; else d_parK0[tidx] = g_K0;
        if (s_K1 == 0) acc_K1 += g_K1; else d_parK1[tidx] = g_K1;
        if (s_K2 == 0) acc_K2 += g_K2; else d_parK2[tidx] = g_K2;
        if (s_LP == 0) acc_LP += g_LP; else d_parLP[tidx] = g_LP;
        if (s_PERC == 0) acc_PERC += g_PERC_p; else d_parPERC[tidx] = g_PERC_p;
        if (s_UZL == 0) acc_UZL += g_UZL; else d_parUZL[tidx] = g_UZL;
        if (s_TT == 0) acc_TT += g_TT; else d_parTT[tidx] = g_TT;
        if (s_CFMAX == 0) acc_CFMAX += g_CFMAX; else d_parCFMAX[tidx] = g_CFMAX;
        if (s_CFR == 0) acc_CFR += g_CFR; else d_parCFR[tidx] = g_CFR;
        if (s_CWH == 0) acc_CWH += g_CWH; else d_parCWH[tidx] = g_CWH;
        if (s_BETAET == 0) acc_BETAET += g_BETAET; else d_parBETAET[tidx] = g_BETAET;
        if (has_capillary) {
            if (s_C == 0) acc_C += g_C; else d_parC[tidx] = g_C;
        }

        d_sp = d_sp0;
        d_mw = d_mw0;
        d_sm = d_sm0;
        d_suz = d_suz0;
        d_slz = d_slz0;
    }

    if (s_BETA == 0) d_parBETA[idx] = acc_BETA;
    if (s_FC == 0) d_parFC[idx] = acc_FC;
    if (s_K0 == 0) d_parK0[idx] = acc_K0;
    if (s_K1 == 0) d_parK1[idx] = acc_K1;
    if (s_K2 == 0) d_parK2[idx] = acc_K2;
    if (s_LP == 0) d_parLP[idx] = acc_LP;
    if (s_PERC == 0) d_parPERC[idx] = acc_PERC;
    if (s_UZL == 0) d_parUZL[idx] = acc_UZL;
    if (s_TT == 0) d_parTT[idx] = acc_TT;
    if (s_CFMAX == 0) d_parCFMAX[idx] = acc_CFMAX;
    if (s_CFR == 0) d_parCFR[idx] = acc_CFR;
    if (s_CWH == 0) d_parCWH[idx] = acc_CWH;
    if (s_BETAET == 0) d_parBETAET[idx] = acc_BETAET;
    if (has_capillary && s_C == 0) d_parC[idx] = acc_C;
}

std::vector<torch::Tensor> hbv_forward_cuda(
    torch::Tensor Pm, torch::Tensor Tm, torch::Tensor PETm,
    torch::Tensor init_states,
    torch::Tensor parBETA, torch::Tensor parFC, torch::Tensor parK0,
    torch::Tensor parK1, torch::Tensor parK2, torch::Tensor parLP,
    torch::Tensor parPERC, torch::Tensor parUZL, torch::Tensor parTT,
    torch::Tensor parCFMAX, torch::Tensor parCFR, torch::Tensor parCWH,
    torch::Tensor parBETAET, torch::Tensor parC,
    float nearzero, bool has_capillary, bool save_states
) {
    int T = Pm.size(0);
    int B = init_states.size(1);
    int M = init_states.size(2);
    int BM = B * M;
    int s_P = (Pm.dim() == 2 || Pm.size(2) == 1) ? 1 : 0;

    auto opts = Pm.options();
    auto Q0_out = torch::empty({T, B, M}, opts);
    auto Q1_out = torch::empty({T, B, M}, opts);
    auto Q2_out = torch::empty({T, B, M}, opts);
    auto states_hist = save_states ? torch::empty({5, T, B, M}, opts) : torch::empty({1}, opts);
    auto final_states = torch::empty({5, B, M}, opts);

    int threads = 256;
    int blocks = (BM + threads - 1) / threads;

    hbv_forward_kernel<<<blocks, threads>>>(
        Pm.data_ptr<float>(), Tm.data_ptr<float>(), PETm.data_ptr<float>(),
        init_states.data_ptr<float>(),
        parBETA.data_ptr<float>(), parFC.data_ptr<float>(), parK0.data_ptr<float>(),
        parK1.data_ptr<float>(), parK2.data_ptr<float>(), parLP.data_ptr<float>(),
        parPERC.data_ptr<float>(), parUZL.data_ptr<float>(), parTT.data_ptr<float>(),
        parCFMAX.data_ptr<float>(), parCFR.data_ptr<float>(), parCWH.data_ptr<float>(),
        parBETAET.data_ptr<float>(), parC.data_ptr<float>(),
        s_P,
        (parBETA.size(0) == 1) ? 0 : 1,
        (parFC.size(0) == 1) ? 0 : 1,
        (parK0.size(0) == 1) ? 0 : 1,
        (parK1.size(0) == 1) ? 0 : 1,
        (parK2.size(0) == 1) ? 0 : 1,
        (parLP.size(0) == 1) ? 0 : 1,
        (parPERC.size(0) == 1) ? 0 : 1,
        (parUZL.size(0) == 1) ? 0 : 1,
        (parTT.size(0) == 1) ? 0 : 1,
        (parCFMAX.size(0) == 1) ? 0 : 1,
        (parCFR.size(0) == 1) ? 0 : 1,
        (parCWH.size(0) == 1) ? 0 : 1,
        (parBETAET.size(0) == 1) ? 0 : 1,
        (has_capillary && parC.size(0) == 1) ? 0 : 1,
        Q0_out.data_ptr<float>(), Q1_out.data_ptr<float>(), Q2_out.data_ptr<float>(),
        states_hist.data_ptr<float>(), final_states.data_ptr<float>(),
        T, B, M, BM, nearzero, has_capillary, save_states
    );

    return {Q0_out, Q1_out, Q2_out, states_hist, final_states};
}

std::vector<torch::Tensor> hbv_backward_cuda(
    torch::Tensor Pm, torch::Tensor Tm, torch::Tensor PETm,
    torch::Tensor states_hist,
    torch::Tensor parBETA, torch::Tensor parFC, torch::Tensor parK0,
    torch::Tensor parK1, torch::Tensor parK2, torch::Tensor parLP,
    torch::Tensor parPERC, torch::Tensor parUZL, torch::Tensor parTT,
    torch::Tensor parCFMAX, torch::Tensor parCFR, torch::Tensor parCWH,
    torch::Tensor parBETAET, torch::Tensor parC,
    torch::Tensor dQ0_out, torch::Tensor dQ1_out, torch::Tensor dQ2_out,
    float nearzero, bool has_capillary
) {
    int T = Pm.size(0);
    int B = states_hist.size(2);
    int M = states_hist.size(3);
    int BM = B * M;
    int s_P = (Pm.dim() == 2 || Pm.size(2) == 1) ? 1 : 0;

    auto opts = Pm.options();
    auto d_parBETA = torch::empty_like(parBETA);
    auto d_parFC = torch::empty_like(parFC);
    auto d_parK0 = torch::empty_like(parK0);
    auto d_parK1 = torch::empty_like(parK1);
    auto d_parK2 = torch::empty_like(parK2);
    auto d_parLP = torch::empty_like(parLP);
    auto d_parPERC = torch::empty_like(parPERC);
    auto d_parUZL = torch::empty_like(parUZL);
    auto d_parTT = torch::empty_like(parTT);
    auto d_parCFMAX = torch::empty_like(parCFMAX);
    auto d_parCFR = torch::empty_like(parCFR);
    auto d_parCWH = torch::empty_like(parCWH);
    auto d_parBETAET = torch::empty_like(parBETAET);
    auto d_parC = has_capillary ? torch::empty_like(parC) : torch::empty({1}, opts);

    int threads = 256;
    int blocks = (BM + threads - 1) / threads;

    hbv_backward_kernel<<<blocks, threads>>>(
        Pm.data_ptr<float>(), Tm.data_ptr<float>(), PETm.data_ptr<float>(),
        states_hist.data_ptr<float>(),
        parBETA.data_ptr<float>(), parFC.data_ptr<float>(), parK0.data_ptr<float>(),
        parK1.data_ptr<float>(), parK2.data_ptr<float>(), parLP.data_ptr<float>(),
        parPERC.data_ptr<float>(), parUZL.data_ptr<float>(), parTT.data_ptr<float>(),
        parCFMAX.data_ptr<float>(), parCFR.data_ptr<float>(), parCWH.data_ptr<float>(),
        parBETAET.data_ptr<float>(), parC.data_ptr<float>(),
        s_P,
        (parBETA.size(0) == 1) ? 0 : 1,
        (parFC.size(0) == 1) ? 0 : 1,
        (parK0.size(0) == 1) ? 0 : 1,
        (parK1.size(0) == 1) ? 0 : 1,
        (parK2.size(0) == 1) ? 0 : 1,
        (parLP.size(0) == 1) ? 0 : 1,
        (parPERC.size(0) == 1) ? 0 : 1,
        (parUZL.size(0) == 1) ? 0 : 1,
        (parTT.size(0) == 1) ? 0 : 1,
        (parCFMAX.size(0) == 1) ? 0 : 1,
        (parCFR.size(0) == 1) ? 0 : 1,
        (parCWH.size(0) == 1) ? 0 : 1,
        (parBETAET.size(0) == 1) ? 0 : 1,
        (has_capillary && parC.size(0) == 1) ? 0 : 1,
        dQ0_out.data_ptr<float>(), dQ1_out.data_ptr<float>(), dQ2_out.data_ptr<float>(),
        d_parBETA.data_ptr<float>(), d_parFC.data_ptr<float>(), d_parK0.data_ptr<float>(),
        d_parK1.data_ptr<float>(), d_parK2.data_ptr<float>(), d_parLP.data_ptr<float>(),
        d_parPERC.data_ptr<float>(), d_parUZL.data_ptr<float>(), d_parTT.data_ptr<float>(),
        d_parCFMAX.data_ptr<float>(), d_parCFR.data_ptr<float>(), d_parCWH.data_ptr<float>(),
        d_parBETAET.data_ptr<float>(), d_parC.data_ptr<float>(),
        T, B, M, BM, nearzero, has_capillary
    );

    return {
        d_parBETA, d_parFC, d_parK0, d_parK1, d_parK2, d_parLP,
        d_parPERC, d_parUZL, d_parTT, d_parCFMAX, d_parCFR, d_parCWH,
        d_parBETAET, d_parC
    };
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> hbv_forward_cuda(
    torch::Tensor Pm, torch::Tensor Tm, torch::Tensor PETm,
    torch::Tensor init_states,
    torch::Tensor parBETA, torch::Tensor parFC, torch::Tensor parK0,
    torch::Tensor parK1, torch::Tensor parK2, torch::Tensor parLP,
    torch::Tensor parPERC, torch::Tensor parUZL, torch::Tensor parTT,
    torch::Tensor parCFMAX, torch::Tensor parCFR, torch::Tensor parCWH,
    torch::Tensor parBETAET, torch::Tensor parC,
    float nearzero, bool has_capillary, bool save_states
);
std::vector<torch::Tensor> hbv_backward_cuda(
    torch::Tensor Pm, torch::Tensor Tm, torch::Tensor PETm,
    torch::Tensor states_hist,
    torch::Tensor parBETA, torch::Tensor parFC, torch::Tensor parK0,
    torch::Tensor parK1, torch::Tensor parK2, torch::Tensor parLP,
    torch::Tensor parPERC, torch::Tensor parUZL, torch::Tensor parTT,
    torch::Tensor parCFMAX, torch::Tensor parCFR, torch::Tensor parCWH,
    torch::Tensor parBETAET, torch::Tensor parC,
    torch::Tensor dQ0_out, torch::Tensor dQ1_out, torch::Tensor dQ2_out,
    float nearzero, bool has_capillary
);
"""

_HBV_CPU_OMP_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>
#include <vector>

std::vector<torch::Tensor> hbv_forward_cpu(
    torch::Tensor Pm, torch::Tensor Tm, torch::Tensor PETm,
    torch::Tensor init_states,
    torch::Tensor parBETA, torch::Tensor parFC, torch::Tensor parK0,
    torch::Tensor parK1, torch::Tensor parK2, torch::Tensor parLP,
    torch::Tensor parPERC, torch::Tensor parUZL, torch::Tensor parTT,
    torch::Tensor parCFMAX, torch::Tensor parCFR, torch::Tensor parCWH,
    torch::Tensor parBETAET, torch::Tensor parC,
    double nearzero_d, bool has_capillary
) {
    int T = Pm.size(0);
    int B = Pm.size(1);
    int M = Pm.size(2);
    int BM = B * M;
    float nearzero = static_cast<float>(nearzero_d);

    auto opts = Pm.options();
    auto Q0_out = torch::empty({T, B, M}, opts);
    auto Q1_out = torch::empty({T, B, M}, opts);
    auto Q2_out = torch::empty({T, B, M}, opts);
    auto states_hist = torch::empty({T, B, M, 5}, opts);
    auto final_states = torch::empty({5, B, M}, opts);

    const float* Pm_p = Pm.data_ptr<float>();
    const float* Tm_p = Tm.data_ptr<float>();
    const float* PETm_p = PETm.data_ptr<float>();
    const float* init_p = init_states.data_ptr<float>();
    const float* parBETA_p = parBETA.data_ptr<float>();
    const float* parFC_p = parFC.data_ptr<float>();
    const float* parK0_p = parK0.data_ptr<float>();
    const float* parK1_p = parK1.data_ptr<float>();
    const float* parK2_p = parK2.data_ptr<float>();
    const float* parLP_p = parLP.data_ptr<float>();
    const float* parPERC_p = parPERC.data_ptr<float>();
    const float* parUZL_p = parUZL.data_ptr<float>();
    const float* parTT_p = parTT.data_ptr<float>();
    const float* parCFMAX_p = parCFMAX.data_ptr<float>();
    const float* parCFR_p = parCFR.data_ptr<float>();
    const float* parCWH_p = parCWH.data_ptr<float>();
    const float* parBETAET_p = parBETAET.data_ptr<float>();
    const float* parC_p = parC.data_ptr<float>();

    float* Q0_p = Q0_out.data_ptr<float>();
    float* Q1_p = Q1_out.data_ptr<float>();
    float* Q2_p = Q2_out.data_ptr<float>();
    float* hist_p = states_hist.data_ptr<float>();
    float* fin_p = final_states.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < BM; ++idx) {
        float sp = init_p[0 * BM + idx];
        float mw = init_p[1 * BM + idx];
        float sm = init_p[2 * BM + idx];
        float suz = init_p[3 * BM + idx];
        float slz = init_p[4 * BM + idx];

        for (int t = 0; t < T; ++t) {
            int tidx = t * BM + idx;
            int hidx = tidx * 5;
            hist_p[hidx + 0] = sp;
            hist_p[hidx + 1] = mw;
            hist_p[hidx + 2] = sm;
            hist_p[hidx + 3] = suz;
            hist_p[hidx + 4] = slz;

            float P = Pm_p[tidx];
            float Tt = Tm_p[tidx];
            float PETt = PETm_p[tidx];

            float BETA = parBETA_p[tidx];
            float FC = parFC_p[tidx];
            float K0 = parK0_p[tidx];
            float K1 = parK1_p[tidx];
            float K2 = parK2_p[tidx];
            float LP = parLP_p[tidx];
            float PERC_v = parPERC_p[tidx];
            float UZL = parUZL_p[tidx];
            float TT = parTT_p[tidx];
            float CFMAX = parCFMAX_p[tidx];
            float CFR = parCFR_p[tidx];
            float CWH = parCWH_p[tidx];
            float BETAET = parBETAET_p[tidx];
            float C = has_capillary ? parC_p[tidx] : 0.0f;

            float RAIN = (Tt >= TT) ? P : 0.0f;
            float SNOW = (Tt < TT) ? P : 0.0f;

            float sp1 = sp + SNOW;
            float melt_pos = std::fmax(CFMAX * (Tt - TT), 0.0f);
            float melt = std::fmin(melt_pos, sp1);
            float mw1 = mw + melt;
            float sp2 = sp1 - melt;
            float refrz_pos = std::fmax(CFR * CFMAX * (TT - Tt), 0.0f);
            float refrz = std::fmin(refrz_pos, mw1);
            float sp3 = sp2 + refrz;
            float mw2 = mw1 - refrz;
            float tosoil = std::fmax(mw2 - CWH * sp3, 0.0f);
            float mw3 = mw2 - tosoil;

            float infil = RAIN + tosoil;
            float wet_raw = std::pow(std::fmax(sm / FC, nearzero), BETA);
            float wet = std::fmin(std::fmax(wet_raw, 0.0f), 1.0f);
            float recharge = infil * wet;
            float sm1 = sm + infil - recharge;
            float excess = std::fmax(sm1 - FC, 0.0f);
            float sm2 = sm1 - excess;
            float evapf_raw = std::pow(std::fmax(sm2 / (LP * FC), nearzero), BETAET);
            float evapf = std::fmin(std::fmax(evapf_raw, 0.0f), 1.0f);
            float ETact = std::fmin(sm2, PETt * evapf);
            float sm3 = std::fmax(sm2 - ETact, nearzero);

            float sm4 = sm3;
            float slz1 = slz;
            if (has_capillary) {
                float sm3_clamped = std::fmin(sm3 / FC, 1.0f);
                float cap_raw = C * slz * (1.0f - sm3_clamped);
                float cap = std::fmin(slz, cap_raw);
                sm4 = std::fmax(sm3 + cap, nearzero);
                slz1 = std::fmax(slz - cap, nearzero);
            }

            float suz1 = suz + recharge + excess;
            float PERC = std::fmin(suz1, PERC_v);
            float suz2 = suz1 - PERC;
            float suz_ex = std::fmax(suz2 - UZL, 0.0f);
            float Q0 = K0 * suz_ex;
            float suz3 = suz2 - Q0;
            float Q1 = K1 * suz3;
            float suz4 = suz3 - Q1;
            float slz2 = slz1 + PERC;
            float Q2 = K2 * slz2;
            float slz3 = slz2 - Q2;

            Q0_p[tidx] = Q0;
            Q1_p[tidx] = Q1;
            Q2_p[tidx] = Q2;

            sp = sp3;
            mw = mw3;
            sm = sm4;
            suz = suz4;
            slz = slz3;
        }

        fin_p[0 * BM + idx] = sp;
        fin_p[1 * BM + idx] = mw;
        fin_p[2 * BM + idx] = sm;
        fin_p[3 * BM + idx] = suz;
        fin_p[4 * BM + idx] = slz;
    }

    return {Q0_out, Q1_out, Q2_out, states_hist, final_states};
}

std::vector<torch::Tensor> hbv_backward_cpu(
    torch::Tensor Pm, torch::Tensor Tm, torch::Tensor PETm,
    torch::Tensor states_hist,
    torch::Tensor parBETA, torch::Tensor parFC, torch::Tensor parK0,
    torch::Tensor parK1, torch::Tensor parK2, torch::Tensor parLP,
    torch::Tensor parPERC, torch::Tensor parUZL, torch::Tensor parTT,
    torch::Tensor parCFMAX, torch::Tensor parCFR, torch::Tensor parCWH,
    torch::Tensor parBETAET, torch::Tensor parC,
    torch::Tensor dQ0_out, torch::Tensor dQ1_out, torch::Tensor dQ2_out,
    double nearzero_d, bool has_capillary
) {
    int T = Pm.size(0);
    int B = Pm.size(1);
    int M = Pm.size(2);
    int BM = B * M;
    float nearzero = static_cast<float>(nearzero_d);

    auto opts = Pm.options();
    auto d_parBETA = torch::empty({T, B, M}, opts);
    auto d_parFC = torch::empty({T, B, M}, opts);
    auto d_parK0 = torch::empty({T, B, M}, opts);
    auto d_parK1 = torch::empty({T, B, M}, opts);
    auto d_parK2 = torch::empty({T, B, M}, opts);
    auto d_parLP = torch::empty({T, B, M}, opts);
    auto d_parPERC = torch::empty({T, B, M}, opts);
    auto d_parUZL = torch::empty({T, B, M}, opts);
    auto d_parTT = torch::empty({T, B, M}, opts);
    auto d_parCFMAX = torch::empty({T, B, M}, opts);
    auto d_parCFR = torch::empty({T, B, M}, opts);
    auto d_parCWH = torch::empty({T, B, M}, opts);
    auto d_parBETAET = torch::empty({T, B, M}, opts);
    auto d_parC = has_capillary ? torch::empty({T, B, M}, opts) : torch::zeros({1}, opts);

    const float* Pm_p = Pm.data_ptr<float>();
    const float* Tm_p = Tm.data_ptr<float>();
    const float* PETm_p = PETm.data_ptr<float>();
    const float* hist_p = states_hist.data_ptr<float>();
    const float* parBETA_p = parBETA.data_ptr<float>();
    const float* parFC_p = parFC.data_ptr<float>();
    const float* parK0_p = parK0.data_ptr<float>();
    const float* parK1_p = parK1.data_ptr<float>();
    const float* parK2_p = parK2.data_ptr<float>();
    const float* parLP_p = parLP.data_ptr<float>();
    const float* parPERC_p = parPERC.data_ptr<float>();
    const float* parUZL_p = parUZL.data_ptr<float>();
    const float* parTT_p = parTT.data_ptr<float>();
    const float* parCFMAX_p = parCFMAX.data_ptr<float>();
    const float* parCFR_p = parCFR.data_ptr<float>();
    const float* parCWH_p = parCWH.data_ptr<float>();
    const float* parBETAET_p = parBETAET.data_ptr<float>();
    const float* parC_p = parC.data_ptr<float>();
    const float* dQ0_p = dQ0_out.data_ptr<float>();
    const float* dQ1_p = dQ1_out.data_ptr<float>();
    const float* dQ2_p = dQ2_out.data_ptr<float>();

    float* d_parBETA_p = d_parBETA.data_ptr<float>();
    float* d_parFC_p = d_parFC.data_ptr<float>();
    float* d_parK0_p = d_parK0.data_ptr<float>();
    float* d_parK1_p = d_parK1.data_ptr<float>();
    float* d_parK2_p = d_parK2.data_ptr<float>();
    float* d_parLP_p = d_parLP.data_ptr<float>();
    float* d_parPERC_p = d_parPERC.data_ptr<float>();
    float* d_parUZL_p = d_parUZL.data_ptr<float>();
    float* d_parTT_p = d_parTT.data_ptr<float>();
    float* d_parCFMAX_p = d_parCFMAX.data_ptr<float>();
    float* d_parCFR_p = d_parCFR.data_ptr<float>();
    float* d_parCWH_p = d_parCWH.data_ptr<float>();
    float* d_parBETAET_p = d_parBETAET.data_ptr<float>();
    float* d_parC_p = d_parC.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < BM; ++idx) {
        float d_sp = 0.0f, d_mw = 0.0f, d_sm = 0.0f, d_suz = 0.0f, d_slz = 0.0f;

        for (int t = T - 1; t >= 0; --t) {
            int tidx = t * BM + idx;
            int hidx = tidx * 5;
            float sp0 = hist_p[hidx + 0];
            float mw0 = hist_p[hidx + 1];
            float sm0 = hist_p[hidx + 2];
            float suz0 = hist_p[hidx + 3];
            float slz0 = hist_p[hidx + 4];

            float P = Pm_p[tidx];
            float Tt = Tm_p[tidx];
            float PETt = PETm_p[tidx];

            float BETA = parBETA_p[tidx];
            float FC = parFC_p[tidx];
            float K0 = parK0_p[tidx];
            float K1 = parK1_p[tidx];
            float K2 = parK2_p[tidx];
            float LP = parLP_p[tidx];
            float PERC_v = parPERC_p[tidx];
            float UZL = parUZL_p[tidx];
            float TT = parTT_p[tidx];
            float CFMAX = parCFMAX_p[tidx];
            float CFR = parCFR_p[tidx];
            float CWH = parCWH_p[tidx];
            float BETAET = parBETAET_p[tidx];
            float C = has_capillary ? parC_p[tidx] : 0.0f;

            float RAIN = (Tt >= TT) ? P : 0.0f;
            float SNOW = (Tt < TT) ? P : 0.0f;

            float sp1 = sp0 + SNOW;
            float melt_raw = CFMAX * (Tt - TT);
            float melt_pos = std::fmax(melt_raw, 0.0f);
            float melt = std::fmin(melt_pos, sp1);
            float mw1 = mw0 + melt;
            float sp2 = sp1 - melt;
            float refrz_raw = CFR * CFMAX * (TT - Tt);
            float refrz_pos = std::fmax(refrz_raw, 0.0f);
            float refrz = std::fmin(refrz_pos, mw1);
            float sp3 = sp2 + refrz;
            float mw2 = mw1 - refrz;
            float tosoil_raw = mw2 - CWH * sp3;
            float tosoil = std::fmax(tosoil_raw, 0.0f);

            float infil = RAIN + tosoil;
            float ratio_fc_raw = sm0 / FC;
            float ratio_fc = std::fmax(ratio_fc_raw, nearzero);
            float wet_raw = std::pow(ratio_fc, BETA);
            float wet = std::fmin(std::fmax(wet_raw, 0.0f), 1.0f);
            float recharge = infil * wet;
            float sm1 = sm0 + infil - recharge;
            float excs_raw = sm1 - FC;
            float excess = std::fmax(excs_raw, 0.0f);
            float sm2 = sm1 - excess;
            float denom_et = LP * FC;
            float ratio_et_raw = sm2 / denom_et;
            float ratio_et = std::fmax(ratio_et_raw, nearzero);
            float evapf_raw = std::pow(ratio_et, BETAET);
            float evapf = std::fmin(std::fmax(evapf_raw, 0.0f), 1.0f);
            float et_pot = PETt * evapf;
            float ETact = std::fmin(sm2, et_pot);
            float sm3_raw = sm2 - ETact;
            float sm3 = std::fmax(sm3_raw, nearzero);

            float sm3_ratio = sm3 / FC;
            float sm3_clamped = std::fmin(sm3_ratio, 1.0f);
            float cap_raw = has_capillary ? (C * slz0 * (1.0f - sm3_clamped)) : 0.0f;
            float cap = has_capillary ? std::fmin(slz0, cap_raw) : 0.0f;
            float sm4_raw = sm3 + cap;
            float slz1_raw = slz0 - cap;
            float slz1 = has_capillary ? std::fmax(slz1_raw, nearzero) : slz0;

            float suz1 = suz0 + recharge + excess;
            float PERC = std::fmin(suz1, PERC_v);
            float suz2 = suz1 - PERC;
            float suz_ex_raw = suz2 - UZL;
            float suz_ex = std::fmax(suz_ex_raw, 0.0f);
            float Q0 = K0 * suz_ex;
            float suz3 = suz2 - Q0;
            float slz2 = slz1 + PERC;

            float g_Q0 = dQ0_p[tidx];
            float g_Q1 = dQ1_p[tidx];
            float g_Q2 = dQ2_p[tidx];

            g_Q2 += -d_slz;
            float d_slz2 = d_slz + g_Q2 * K2;
            float g_K2 = g_Q2 * slz2;

            float d_slz1 = d_slz2;
            float g_PERC = d_slz2;

            g_Q1 += -d_suz;
            float d_suz3 = d_suz + g_Q1 * K1;
            float g_K1 = g_Q1 * suz3;

            g_Q0 += -d_suz3;
            float d_suz2 = d_suz3;
            float g_K0 = g_Q0 * suz_ex;
            float d_suz_ex = g_Q0 * K0;
            float d_suz_ex_raw = (suz_ex_raw > 0.0f) ? d_suz_ex : 0.0f;
            d_suz2 += d_suz_ex_raw;
            float g_UZL = -d_suz_ex_raw;

            float d_suz1 = d_suz2;
            g_PERC += -d_suz2;
            float g_PERC_p = 0.0f;
            if (suz1 < PERC_v) {
                d_suz1 += g_PERC;
            } else {
                g_PERC_p = g_PERC;
            }

            float d_suz0 = d_suz1;
            float g_recharge = d_suz1;
            float g_excess = d_suz1;

            float d_sm3 = 0.0f;
            float d_slz0 = 0.0f;
            float g_FC = 0.0f;
            float g_C = 0.0f;
            if (has_capillary) {
                float d_sm4_raw = (sm4_raw > nearzero) ? d_sm : 0.0f;
                float d_slz1_raw = (slz1_raw > nearzero) ? d_slz1 : 0.0f;
                d_sm3 += d_sm4_raw;
                float g_cap = d_sm4_raw - d_slz1_raw;
                d_slz0 += d_slz1_raw;

                float g_cap_raw = 0.0f;
                if (slz0 < cap_raw) {
                    d_slz0 += g_cap;
                } else {
                    g_cap_raw = g_cap;
                }
                g_C = g_cap_raw * slz0 * (1.0f - sm3_clamped);
                d_slz0 += g_cap_raw * C * (1.0f - sm3_clamped);
                float d_sm3_clamped = -g_cap_raw * C * slz0;
                float d_sm3_ratio = (sm3_ratio < 1.0f) ? d_sm3_clamped : 0.0f;
                d_sm3 += d_sm3_ratio / FC;
                g_FC += -d_sm3_ratio * sm3 / (FC * FC);
            } else {
                d_sm3 = d_sm;
                d_slz0 = d_slz1;
            }

            float d_sm3_raw = (sm3_raw > nearzero) ? d_sm3 : 0.0f;
            float d_sm2 = d_sm3_raw;
            float g_ETact = -d_sm3_raw;

            float g_et_pot = 0.0f;
            if (sm2 < et_pot) {
                d_sm2 += g_ETact;
            } else {
                g_et_pot = g_ETact;
            }
            float g_evapf = g_et_pot * PETt;
            float g_evapf_raw = (evapf_raw > 0.0f && evapf_raw < 1.0f) ? g_evapf : 0.0f;
            float g_BETAET = g_evapf_raw * evapf_raw * std::log(ratio_et);
            float d_ratio_et = (ratio_et_raw > nearzero) ? (g_evapf_raw * BETAET * std::pow(ratio_et, BETAET - 1.0f)) : 0.0f;
            d_sm2 += d_ratio_et / denom_et;
            float d_denom_et = -d_ratio_et * sm2 / (denom_et * denom_et);
            float g_LP = d_denom_et * FC;
            g_FC += d_denom_et * LP;

            float d_sm1 = d_sm2;
            g_excess += -d_sm2;
            float d_excs_raw = (excs_raw > 0.0f) ? g_excess : 0.0f;
            d_sm1 += d_excs_raw;
            g_FC += -d_excs_raw;

            float d_sm0 = d_sm1;
            float g_infil = d_sm1;
            g_recharge += -d_sm1;
            g_infil += g_recharge * wet;
            float g_wet = g_recharge * infil;
            float g_wet_raw = (wet_raw > 0.0f && wet_raw < 1.0f) ? g_wet : 0.0f;
            float g_BETA = g_wet_raw * wet_raw * std::log(ratio_fc);
            float d_ratio_fc = (ratio_fc_raw > nearzero) ? (g_wet_raw * BETA * std::pow(ratio_fc, BETA - 1.0f)) : 0.0f;
            d_sm0 += d_ratio_fc / FC;
            g_FC += -d_ratio_fc * sm0 / (FC * FC);

            float g_tosoil = g_infil - d_mw;
            float d_mw2 = d_mw;
            float d_sp3 = d_sp;
            float d_tosoil_raw = (tosoil_raw > 0.0f) ? g_tosoil : 0.0f;
            d_mw2 += d_tosoil_raw;
            float g_CWH = -d_tosoil_raw * sp3;
            d_sp3 += -d_tosoil_raw * CWH;

            float d_sp2 = d_sp3;
            float d_mw1 = d_mw2;
            float g_refrz = d_sp3 - d_mw2;
            float g_refrz_pos = 0.0f;
            if (refrz_pos < mw1) {
                g_refrz_pos = g_refrz;
            } else {
                d_mw1 += g_refrz;
            }
            float g_refrz_raw = (refrz_raw > 0.0f) ? g_refrz_pos : 0.0f;
            float g_CFR = g_refrz_raw * CFMAX * (TT - Tt);
            float g_CFMAX = g_refrz_raw * CFR * (TT - Tt);
            float g_TT = g_refrz_raw * CFR * CFMAX;

            float d_mw0 = d_mw1;
            float d_sp1 = d_sp2;
            float g_melt = d_mw1 - d_sp2;
            float g_melt_pos = 0.0f;
            if (melt_pos < sp1) {
                g_melt_pos = g_melt;
            } else {
                d_sp1 += g_melt;
            }
            float g_melt_raw = (melt_raw > 0.0f) ? g_melt_pos : 0.0f;
            g_CFMAX += g_melt_raw * (Tt - TT);
            g_TT += -g_melt_raw * CFMAX;

            float d_sp0 = d_sp1;

            d_parBETA_p[tidx] = g_BETA;
            d_parFC_p[tidx] = g_FC;
            d_parK0_p[tidx] = g_K0;
            d_parK1_p[tidx] = g_K1;
            d_parK2_p[tidx] = g_K2;
            d_parLP_p[tidx] = g_LP;
            d_parPERC_p[tidx] = g_PERC_p;
            d_parUZL_p[tidx] = g_UZL;
            d_parTT_p[tidx] = g_TT;
            d_parCFMAX_p[tidx] = g_CFMAX;
            d_parCFR_p[tidx] = g_CFR;
            d_parCWH_p[tidx] = g_CWH;
            d_parBETAET_p[tidx] = g_BETAET;
            if (has_capillary) {
                d_parC_p[tidx] = g_C;
            }

            d_sp = d_sp0;
            d_mw = d_mw0;
            d_sm = d_sm0;
            d_suz = d_suz0;
            d_slz = d_slz0;
        }
    }

    return {
        d_parBETA, d_parFC, d_parK0, d_parK1, d_parK2, d_parLP,
        d_parPERC, d_parUZL, d_parTT, d_parCFMAX, d_parCFR, d_parCWH,
        d_parBETAET, d_parC
    };
}
"""

_hbv_cuda_mod = None
_hbv_cpu_mod = None


def openmp_flags() -> list[str]:
    """Return the OpenMP compiler/linker flags for this platform.

    Apple clang does not accept ``-fopenmp``; without it the ``#pragma omp`` loops are
    ignored and the kernels run serially with identical numerics.
    """
    return [] if sys.platform == "darwin" else ["-fopenmp"]


def get_hbv_cuda():
    """Compile or return cached inline CUDA HBV extension."""
    global _hbv_cuda_mod
    if _hbv_cuda_mod is None:
        _hbv_cuda_mod = load_inline(
            name="hbv_cuda_v4",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["hbv_forward_cuda", "hbv_backward_cuda"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _hbv_cuda_mod


def get_hbv_cpu():
    """Compile or return cached inline OpenMP C++ CPU HBV extension."""
    global _hbv_cpu_mod
    if _hbv_cpu_mod is None:
        _hbv_cpu_mod = load_inline(
            name="dhbv_cpu_omp_v1",
            cpp_sources=_HBV_CPU_OMP_CPP,
            functions=["hbv_forward_cpu", "hbv_backward_cpu"],
            extra_cflags=["-O3", *openmp_flags()],
            extra_ldflags=openmp_flags(),
            verbose=False,
        )
    return _hbv_cpu_mod


class HbvCoreFunction(torch.autograd.Function):
    """Autograd function for multi-bucket differentiable HBV (CUDA kernel or C++ CPU kernel)."""

    @staticmethod
    def forward(
        ctx,
        Pm,
        Tm,
        PETm,
        init_states,
        parBETA,
        parFC,
        parK0,
        parK1,
        parK2,
        parLP,
        parPERC,
        parUZL,
        parTT,
        parCFMAX,
        parCFR,
        parCWH,
        parBETAET,
        parC,
        nearzero: float,
        has_capillary: bool,
    ):
        Pm = Pm.contiguous()
        Tm = Tm.contiguous()
        PETm = PETm.contiguous()
        init_states = init_states.contiguous()
        parBETA = parBETA.contiguous()
        parFC = parFC.contiguous()
        parK0 = parK0.contiguous()
        parK1 = parK1.contiguous()
        parK2 = parK2.contiguous()
        parLP = parLP.contiguous()
        parPERC = parPERC.contiguous()
        parUZL = parUZL.contiguous()
        parTT = parTT.contiguous()
        parCFMAX = parCFMAX.contiguous()
        parCFR = parCFR.contiguous()
        parCWH = parCWH.contiguous()
        parBETAET = parBETAET.contiguous()
        parC = parC.contiguous()

        save_states = any(ctx.needs_input_grad)
        if Pm.is_cuda:
            Q0_out, Q1_out, Q2_out, states_hist, final_states = get_hbv_cuda().hbv_forward_cuda(
                Pm,
                Tm,
                PETm,
                init_states,
                parBETA,
                parFC,
                parK0,
                parK1,
                parK2,
                parLP,
                parPERC,
                parUZL,
                parTT,
                parCFMAX,
                parCFR,
                parCWH,
                parBETAET,
                parC,
                float(nearzero),
                bool(has_capillary),
                bool(save_states),
            )
        else:
            Q0_out, Q1_out, Q2_out, states_hist, final_states = get_hbv_cpu().hbv_forward_cpu(
                Pm,
                Tm,
                PETm,
                init_states,
                parBETA,
                parFC,
                parK0,
                parK1,
                parK2,
                parLP,
                parPERC,
                parUZL,
                parTT,
                parCFMAX,
                parCFR,
                parCWH,
                parBETAET,
                parC,
                float(nearzero),
                bool(has_capillary),
            )
        if save_states:
            ctx.save_for_backward(
            Pm,
            Tm,
            PETm,
            states_hist,
            parBETA,
            parFC,
            parK0,
            parK1,
            parK2,
            parLP,
            parPERC,
            parUZL,
            parTT,
            parCFMAX,
            parCFR,
            parCWH,
            parBETAET,
            parC,
        )
        ctx.nearzero = float(nearzero)
        ctx.has_capillary = bool(has_capillary)
        ctx.mark_non_differentiable(final_states)
        return Q0_out, Q1_out, Q2_out, final_states

    @staticmethod
    def backward(ctx, dQ0_out, dQ1_out, dQ2_out, _d_final_states):
        (
            Pm,
            Tm,
            PETm,
            states_hist,
            parBETA,
            parFC,
            parK0,
            parK1,
            parK2,
            parLP,
            parPERC,
            parUZL,
            parTT,
            parCFMAX,
            parCFR,
            parCWH,
            parBETAET,
            parC,
        ) = ctx.saved_tensors
        if Pm.is_cuda:
            grads = get_hbv_cuda().hbv_backward_cuda(
                Pm,
                Tm,
                PETm,
                states_hist,
                parBETA,
                parFC,
                parK0,
                parK1,
                parK2,
                parLP,
                parPERC,
                parUZL,
                parTT,
                parCFMAX,
                parCFR,
                parCWH,
                parBETAET,
                parC,
                dQ0_out.contiguous(),
                dQ1_out.contiguous(),
                dQ2_out.contiguous(),
                ctx.nearzero,
                ctx.has_capillary,
            )
        else:
            grads = get_hbv_cpu().hbv_backward_cpu(
                Pm,
                Tm,
                PETm,
                states_hist,
                parBETA,
                parFC,
                parK0,
                parK1,
                parK2,
                parLP,
                parPERC,
                parUZL,
                parTT,
                parCFMAX,
                parCFR,
                parCWH,
                parBETAET,
                parC,
                dQ0_out.contiguous(),
                dQ1_out.contiguous(),
                dQ2_out.contiguous(),
                ctx.nearzero,
                ctx.has_capillary,
            )
        return (
            None,
            None,
            None,
            None,
            grads[0],
            grads[1],
            grads[2],
            grads[3],
            grads[4],
            grads[5],
            grads[6],
            grads[7],
            grads[8],
            grads[9],
            grads[10],
            grads[11],
            grads[12],
            grads[13] if ctx.has_capillary else None,
            None,
            None,
        )


def uh_gamma_fast(route_a: torch.Tensor, route_b: torch.Tensor, lenF: int = 15) -> torch.Tensor:
    """Evaluate a normalized discrete Gamma unit-hydrograph impulse response at midpoint lags."""
    shape = F.relu(route_a).unsqueeze(0).unsqueeze(-1) + 0.1
    scale = F.relu(route_b).unsqueeze(0).unsqueeze(-1) + 0.5
    lags = (
        torch.arange(lenF, device=route_a.device, dtype=route_a.dtype) + 0.5
    ).view(lenF, 1, 1)
    log_pdf = (
        (shape - 1.0) * torch.log(lags)
        - (lags / scale)
        - shape * torch.log(scale)
        - torch.lgamma(shape)
    )
    unnorm = torch.exp(log_pdf)
    return unnorm / (unnorm.sum(dim=0, keepdim=True) + 1e-12)


def uh_conv_fast(x: torch.Tensor, UH: torch.Tensor) -> torch.Tensor:
    """Apply causal 1D per-basin convolution with unit-hydrograph weights via left zero-padding."""
    n_basins, _, n_steps = x.shape
    kernel_len = UH.shape[-1]
    padded = F.pad(x.reshape(1, n_basins, n_steps), (kernel_len - 1, 0))
    flipped_kernel = torch.flip(UH.reshape(n_basins, 1, kernel_len), dims=[2])
    routed = F.conv1d(padded, flipped_kernel, groups=n_basins)
    return routed.reshape(n_basins, 1, n_steps)


class CudnnLstm(nn.Module):
    """Single-layer cuDNN LSTM with DropConnect weight regularization (Wan et al., 2013)."""

    def __init__(self, nx: int, hidden_size: int = 256, dr: float = 0.5):
        super().__init__()
        self.nx = int(nx)
        self.hidden_size = int(hidden_size)
        self.dr = float(dr)
        self.w_ih = Parameter(torch.empty(self.hidden_size * 4, self.nx))
        self.w_hh = Parameter(torch.empty(self.hidden_size * 4, self.hidden_size))
        self.b_ih = Parameter(torch.empty(self.hidden_size * 4))
        self.b_hh = Parameter(torch.empty(self.hidden_size * 4))
        self._init_parameters()

    def _init_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)

    def forward(
        self,
        input: torch.Tensor,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        batch_size = input.size(1)
        if init_state is not None:
            hx, cx = init_state
        else:
            zeros = input.new_zeros(1, batch_size, self.hidden_size)
            hx, cx = zeros, zeros.clone()
        if self.training and self.dr > 0.0:
            w_ih_eff = F.dropout(self.w_ih, p=self.dr, training=True)
            w_hh_eff = F.dropout(self.w_hh, p=self.dr, training=True)
            params = [w_ih_eff, w_hh_eff, self.b_ih, self.b_hh]
        else:
            params = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        output, hy, cy = torch._VF.lstm(
            input,
            (hx, cx),
            params,
            True,
            1,
            0.0,
            self.training,
            False,
            False,
        )
        if return_state:
            return output, (hy, cy)
        return output


class StandaloneDHBV(nn.Module):
    """Differentiable 16-bucket HBV (dHBV 1.0 and dHBV 1.1p) coupled with LSTM."""

    def __init__(
        self,
        nx: int,
        version: str = "1_0",
        hidden_size: int = 256,
        dr: float = 0.5,
        nmul: int = 16,
        warmup: int = 365,
    ):
        super().__init__()
        self.version = version
        self.nmul = nmul
        self.warmup = warmup
        self.nearzero = 1e-5
        self.routing_len = 15
        if version == "1_1p":
            self.has_capillary = True
            self.warmup_states = False
            self.dynamic_params = ["parBETA", "parK0", "parBETAET"]
            self.param_bounds = {
                "parBETA": [1.0, 6.0],
                "parFC": [50.0, 1000.0],
                "parK0": [0.05, 0.9],
                "parK1": [0.01, 0.5],
                "parK2": [0.001, 0.2],
                "parLP": [0.2, 1.0],
                "parPERC": [0.0, 10.0],
                "parUZL": [0.0, 100.0],
                "parTT": [-2.5, 2.5],
                "parCFMAX": [0.5, 10.0],
                "parCFR": [0.0, 0.1],
                "parCWH": [0.0, 0.2],
                "parBETAET": [0.3, 5.0],
                "parC": [0.0, 1.0],
            }
        else:
            self.has_capillary = False
            self.warmup_states = True
            self.dynamic_params = ["parBETA", "parBETAET"]
            self.param_bounds = {
                "parBETA": [1.0, 6.0],
                "parFC": [50.0, 1000.0],
                "parK0": [0.05, 0.9],
                "parK1": [0.01, 0.5],
                "parK2": [0.001, 0.2],
                "parLP": [0.2, 1.0],
                "parPERC": [0.0, 10.0],
                "parUZL": [0.0, 100.0],
                "parTT": [-2.5, 2.5],
                "parCFMAX": [0.5, 10.0],
                "parCFR": [0.0, 0.1],
                "parCWH": [0.0, 0.2],
                "parBETAET": [0.3, 5.0],
            }
        self.route_bounds = {"route_a": [0.0, 2.9], "route_b": [0.0, 6.5]}
        self.phy_param_count = len(self.param_bounds)
        self.ny = self.phy_param_count * self.nmul + 2

        self.linear_in = nn.Linear(nx, hidden_size)
        self.lstm = CudnnLstm(nx=hidden_size, hidden_size=hidden_size, dr=dr)
        self.linear_out = nn.Linear(hidden_size, self.ny)

    def _descale_phy(
        self, phy_params: torch.Tensor, dy_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        nsteps = phy_params.shape[0]
        pdict: Dict[str, torch.Tensor] = {}
        for i, (name, bnds) in enumerate(self.param_bounds.items()):
            scale = bnds[1] - bnds[0]
            offset = bnds[0]
            if name in dy_list:
                pdict[name] = phy_params[:, :, i, :] * scale + offset
            else:
                sta = phy_params[-1, :, i, :] * scale + offset
                if phy_params.is_cuda:
                    pdict[name] = sta.unsqueeze(0)
                else:
                    pdict[name] = sta.unsqueeze(0).expand(nsteps, -1, -1)
        return pdict

    def _run_pbm(
        self,
        x_phy: torch.Tensor,
        init_states: torch.Tensor,
        pdict: Dict[str, torch.Tensor],
        route_a: torch.Tensor,
        route_b: torch.Tensor,
        do_routing: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        P = x_phy[:, :, 0]
        T = x_phy[:, :, 1]
        PET = x_phy[:, :, 2]
        if x_phy.is_cuda:
            Pm = P.unsqueeze(2).contiguous()
            Tm = T.unsqueeze(2).contiguous()
            PETm = PET.unsqueeze(2).contiguous()
            parC = (
                pdict["parC"]
                if self.has_capillary
                else torch.zeros(1, dtype=x_phy.dtype, device=x_phy.device)
            )
        else:
            Pm = P.unsqueeze(2).expand(-1, -1, self.nmul).contiguous()
            Tm = T.unsqueeze(2).expand(-1, -1, self.nmul).contiguous()
            PETm = PET.unsqueeze(2).expand(-1, -1, self.nmul).contiguous()
            parC = pdict["parC"] if self.has_capillary else torch.zeros_like(pdict["parBETA"])

        # HbvCoreFunction dispatches on device: CUDA kernel for CUDA tensors, the C++ CPU
        # kernel (OpenMP where available) otherwise.
        Q0_sim, Q1_sim, Q2_sim, final_states = HbvCoreFunction.apply(
            Pm,
            Tm,
            PETm,
            init_states,
            pdict["parBETA"],
            pdict["parFC"],
            pdict["parK0"],
            pdict["parK1"],
            pdict["parK2"],
            pdict["parLP"],
            pdict["parPERC"],
            pdict["parUZL"],
            pdict["parTT"],
            pdict["parCFMAX"],
            pdict["parCFR"],
            pdict["parCWH"],
            pdict["parBETAET"],
            parC,
            float(self.nearzero),
            bool(self.has_capillary),
        )

        if not do_routing:
            return None, final_states
        Qsimavg = (Q0_sim + Q1_sim + Q2_sim).mean(-1)
        UH = uh_gamma_fast(route_a, route_b, lenF=self.routing_len)
        rf = Qsimavg.unsqueeze(-1).permute(1, 2, 0)
        UH_p = UH.permute(1, 2, 0)
        Qsrout = uh_conv_fast(rf, UH_p).permute(2, 0, 1)
        return Qsrout[:, :, 0], final_states

    def forward(self, xc_nn_norm: torch.Tensor, x_phy: torch.Tensor) -> torch.Tensor:
        if xc_nn_norm.is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                x0 = F.relu(self.linear_in(xc_nn_norm))
                lstm_out = self.lstm(x0)
                raw_params = self.linear_out(lstm_out)
            parameters = raw_params.float()
        else:
            x0 = F.relu(self.linear_in(xc_nn_norm))
            lstm_out = self.lstm(x0)
            parameters = self.linear_out(lstm_out).float()

        T_tot, ngrid, _ = x_phy.shape
        phy_params = torch.sigmoid(parameters[:, :, : self.phy_param_count * self.nmul]).view(
            T_tot, ngrid, self.phy_param_count, self.nmul
        )
        routing_params = torch.sigmoid(parameters[-1, :, self.phy_param_count * self.nmul :])
        route_a = routing_params[:, 0] * 2.9
        route_b = routing_params[:, 1] * 6.5

        init_states = torch.full(
            (5, ngrid, self.nmul), 0.001, dtype=torch.float32, device=x_phy.device
        )
        if self.warmup_states and self.warmup > 0 and T_tot > self.warmup:
            with torch.no_grad():
                pdict_w = self._descale_phy(phy_params[: self.warmup], dy_list=[])
                _, init_states = self._run_pbm(
                    x_phy[: self.warmup],
                    init_states,
                    pdict_w,
                    route_a,
                    route_b,
                    do_routing=False,
                )
            pdict_sim = self._descale_phy(phy_params[self.warmup :], dy_list=self.dynamic_params)
            qs, _ = self._run_pbm(
                x_phy[self.warmup :],
                init_states,
                pdict_sim,
                route_a,
                route_b,
                do_routing=True,
            )
            if qs is None:
                raise RuntimeError("_run_pbm returned no routed discharge with do_routing=True.")
            return qs

        pdict_sim = self._descale_phy(phy_params, dy_list=self.dynamic_params)
        qs, _ = self._run_pbm(
            x_phy,
            init_states,
            pdict_sim,
            route_a,
            route_b,
            do_routing=True,
        )
        if qs is None:
            raise RuntimeError("_run_pbm returned no routed discharge with do_routing=True.")
        return qs[self.warmup :] if T_tot > self.warmup else qs


class StandaloneDHBV11p(StandaloneDHBV):
    """Song et al. (2024, 2025) dHBV 1.1p with capillary rise and dynamic parK0."""

    def __init__(
        self,
        nx: int,
        hidden_size: int = 256,
        dr: float = 0.5,
        nmul: int = 16,
        warmup: int = 365,
    ) -> None:
        super().__init__(
            nx=nx,
            version="1_1p",
            hidden_size=hidden_size,
            dr=dr,
            nmul=nmul,
            warmup=warmup,
        )
