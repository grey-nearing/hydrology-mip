# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Nearing et al. (2020) & Newman et al. (2017) Snow-17 + SAC-SMA + Duamel UH Model.

References:
- Nearing, G., Sampson, A., Kratzert, F., & Frame, J. (2020). Post-Processing a
  Conceptual Rainfall-Runoff Model with an LSTM. EarthArXiv Preprint,
  https://doi.org/10.31223/osf.io/53te4 (https://github.com/Upstream-Tech/SACSMA-SNOW17).
- Newman, A. J., Mizukami, N., Clark, M. P., Wood, A. W., Nijssen, B., &
  Nearing, G. (2017). Benchmarking of a physically based hydrologic model.
  Journal of Hydrometeorology, 18(8), 2215-2225,
  https://doi.org/10.1175/JHM-D-16-0284.1 (https://github.com/NCAR/camels).
- Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A.,
  Viger, R. J., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T., & Duan, Q.
  (2015). Development of a large-sample watershed-scale hydrometeorological
  data set for the contiguous USA: data set characteristics and assessment of
  regional variability in hydrologic model performance. Hydrology and Earth
  System Sciences, 19(1), 209-223, https://doi.org/10.5194/hess-19-209-2015.
- Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search
  algorithm for computationally efficient watershed model calibration.
  Water Resources Research, 43(1), W01413.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_SACSMA_OMP_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>
#include <vector>

torch::Tensor simulate_sacsma_batch_cpu(
    torch::Tensor prcp_t,   // [N, T] float64
    torch::Tensor tmean_t,  // [N, T] float64
    torch::Tensor pet_t,    // [N, T] float64
    torch::Tensor doy_t,    // [T] float64
    torch::Tensor params_t, // [N, 27] float64
    torch::Tensor elev_t    // [N] float64
) {
    int N = prcp_t.size(0);
    int T = prcp_t.size(1);
    auto q_out = torch::empty({N, T}, torch::TensorOptions().dtype(torch::kFloat32));

    const double* prcp_p = prcp_t.data_ptr<double>();
    const double* tmean_p = tmean_t.data_ptr<double>();
    const double* pet_p = pet_t.data_ptr<double>();
    const double* doy_p = doy_t.data_ptr<double>();
    const double* par_p = params_t.data_ptr<double>();
    const double* elev_p = elev_t.data_ptr<double>();
    float* q_out_p = q_out.data_ptr<float>();

    std::vector<double> seasonal(T);
    for (int t = 0; t < T; ++t) {
        seasonal[t] = 0.5 * (std::sin(2.0 * M_PI * (doy_p[t] - 81.0) / 365.25) + 1.0);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const double* p_row = prcp_p + i * T;
        const double* ta_row = tmean_p + i * T;
        const double* pet_row = pet_p + i * T;
        const double* pm = par_p + i * 27;
        float* qo = q_out_p + i * T;

        double uztwm = pm[0], uzfwm = pm[1], lztwm = pm[2], lzfpm = pm[3], lzfsm = pm[4];
        double adimp = pm[5], uzk = pm[6], lzpk = pm[7], lzsk = pm[8], zperc = pm[9];
        double rexp = pm[10], pctim = pm[11], pfree = pm[12], riva = pm[13], side = pm[14];
        double scf = pm[15], mfmax = pm[16], mfmin = pm[17], uadj = pm[18], si = pm[19];
        double pxtemp = pm[20], nmf = pm[21], tipm = pm[22], mbase = pm[23], plwhc = pm[24];
        double uh_shape = pm[25], uh_scale = pm[26];

        double swe = 0.0, liq_w = 0.0, ati = 0.0, deficit = 0.0;
        double uztwc = 0.5 * uztwm, uzfwc = 0.1 * uzfwm, lztwc = 0.5 * lztwm;
        double lzfpc = 0.3 * lzfpm, lzfsc = 0.2 * lzfsm, adimc = uztwc + lztwc;

        double elev_m = std::max(0.0, elev_p[i]);
        double pa_mb = 1013.25 * std::pow(1.0 - 2.25577e-5 * elev_m, 5.25588);
        double parea = std::max(0.0, 1.0 - pctim - adimp);

        // Precompute 15-lag Duamel Gamma UH weights
        double a = std::max(0.25, uh_shape);
        double b = std::max(0.25, uh_scale);
        double uh_w[15];
        double max_lw = -1e30;
        for (int k = 0; k < 15; ++k) {
            double tk = 0.5 + k;
            uh_w[k] = (a - 1.0) * std::log(tk) - (tk / b);
            if (uh_w[k] > max_lw) max_lw = uh_w[k];
        }
        double w_sum = 0.0;
        for (int k = 0; k < 15; ++k) {
            uh_w[k] = std::exp(uh_w[k] - max_lw);
            w_sum += uh_w[k];
        }
        for (int k = 0; k < 15; ++k) {
            uh_w[k] /= (w_sum > 1e-12 ? w_sum : 1.0);
        }

        double q_fast_ring[15] = {0.0};

        for (int t = 0; t < T; ++t) {
            double p = p_row[t];
            double ta = ta_row[t];
            double pet_d = pet_row[t];
            double sv = seasonal[t];

            double mf = 4.0 * (mfmin + (mfmax - mfmin) * sv);
            double snow_in = (ta <= pxtemp) ? (p * scf) : 0.0;
            double rain_in = (ta <= pxtemp) ? 0.0 : p;

            if (snow_in > 0.0) {
                swe += snow_in;
                if (ta < 0.0) deficit += -ta * snow_in / 160.0;
                ati = (swe <= snow_in * 1.01) ? ta : (ati + tipm * (ta - ati));
            } else if (swe > 0.0) {
                ati = std::min(0.0, ati + tipm * (ta - ati));
            } else {
                ati = 0.0;
                deficit = 0.0;
            }

            double melt = 0.0;
            if (swe > 0.0) {
                double adc = (swe >= si) ? 1.0 : std::min(1.0, std::max(0.05, std::pow(swe / std::max(1.0, si), 0.75)));
                if (ta < mbase) {
                    double delta_def = 4.0 * nmf * (mf / std::max(1e-4, 4.0 * mfmax)) * std::max(0.0, ati - ta);
                    deficit = std::min(swe * 0.33, std::max(0.0, deficit + delta_def));
                    melt = 0.0;
                } else {
                    double pot_melt = 0.0;
                    if (rain_in > 1.5) {
                        double ea_mb = 6.11 * std::exp((17.27 * ta) / (ta + 237.3));
                        double wind_term = 4.0 * uadj * std::max(0.0, (ea_mb - 6.11) + 0.00057 * pa_mb * ta);
                        double adv_term = 0.0125 * rain_in * std::max(0.0, ta - mbase);
                        double rad_term = 1.2 * std::max(0.0, ta - mbase);
                        pot_melt = (rad_term + adv_term + wind_term) * adc;
                    } else {
                        pot_melt = mf * std::max(0.0, ta - mbase) * adc;
                    }
                    if (deficit > 0.0) {
                        double used = std::min(deficit, pot_melt);
                        deficit -= used;
                        pot_melt -= used;
                    }
                    melt = std::min(swe, std::max(0.0, pot_melt));
                    swe -= melt;
                }
            } else {
                melt = 0.0;
                deficit = 0.0;
            }

            liq_w += rain_in + melt;
            if (ta < mbase && liq_w > 0.0) {
                double refrz = std::min(liq_w, 4.0 * nmf * (mbase - ta));
                swe += refrz;
                liq_w -= refrz;
            }

            double max_liq = plwhc * swe;
            double pxv = std::max(0.0, liq_w - max_liq);
            liq_w -= pxv;

            double e1 = std::min(uztwc, pet_d * (uztwc / std::max(1e-4, uztwm)));
            uztwc -= e1;
            double rem_pet = std::max(0.0, pet_d - e1);

            double e2 = std::min(uzfwc, rem_pet);
            uzfwc -= e2;
            rem_pet = std::max(0.0, rem_pet - e2);

            if ((uztwc / std::max(1e-4, uztwm)) < (uzfwc / std::max(1e-4, uzfwm))) {
                double uzrat = (uztwc + uzfwc) / std::max(1e-4, uztwm + uzfwm);
                uztwc = uztwm * uzrat;
                uzfwc = uzfwm * uzrat;
            }

            double e3 = std::min(lztwc, rem_pet * (lztwc / std::max(1e-4, uztwm + lztwm)));
            lztwc -= e3;

            double e5 = std::min(adimc, e1 + rem_pet * ((adimc - e1 - uztwc) / std::max(1e-4, uztwm + lztwm)));
            e5 = std::max(0.0, e5);
            adimc = std::max(uztwc, std::min(uztwm + lztwm, adimc - e5));

            double saved = side / (1.0 + side);
            double ratlzt = lztwc / std::max(1e-4, lztwm);
            double ratlz = (lztwc + lzfpc * (1.0 - saved) + lzfsc) / std::max(1e-4, lztwm + lzfpm * (1.0 - saved) + lzfsm);
            if (ratlzt < ratlz) {
                double del_lz = std::min((ratlz - ratlzt) * lztwm, lzfsc + lzfpc);
                lztwc += del_lz;
                if (del_lz <= lzfsc) {
                    lzfsc -= del_lz;
                } else {
                    double rem_del = del_lz - lzfsc;
                    lzfsc = 0.0;
                    lzfpc = std::max(0.0, lzfpc - rem_del);
                }
            }

            double roimp = pxv * pctim;
            double pav = std::max(0.0, pxv - uztwm + uztwc);
            adimc += pxv - pav;
            double roimp_ad = pav * ((adimc - uztwc) / std::max(1e-4, lztwm)) * adimp;
            roimp_ad = std::max(0.0, std::min(pav * adimp, roimp_ad));
            adimc = (adimp > 1e-6) ? std::min(uztwm + lztwm, adimc + (pav - roimp_ad / adimp)) : (uztwc + lztwc);

            uztwc = std::min(uztwm, uztwc + pxv);

            double pbase = lzfpm * lzpk + lzfsm * lzsk;
            double def_ratio = std::max(0.0, 1.0 - (lztwc + lzfpc + lzfsc) / std::max(1e-4, lztwm + lzfpm + lzfsm));
            double perc_demand = pbase * (1.0 + zperc * std::pow(def_ratio, rexp));
            double perc = std::min(uzfwc, perc_demand * (uzfwc / std::max(1e-4, uzfwm)));
            uzfwc -= perc;

            double perc_free = perc * pfree;
            double perc_tens = perc - perc_free;
            if (lztwc + perc_tens > lztwm) {
                perc_free += (lztwc + perc_tens) - lztwm;
                lztwc = lztwm;
            } else {
                lztwc += perc_tens;
            }

            if (perc_free > 0.0) {
                double f_sup = std::min(lzfsm - lzfsc, perc_free * 0.5);
                lzfsc += f_sup;
                lzfpc = std::min(lzfpm, lzfpc + (perc_free - f_sup));
            }

            double ssur = 0.0;
            if (uzfwc + pav > uzfwm) {
                ssur = (uzfwc + pav - uzfwm) * parea;
                uzfwc = uzfwm;
            } else {
                uzfwc += pav;
            }

            double sif = uzfwc * uzk * parea;
            uzfwc -= uzfwc * uzk;

            double bfp = lzfpc * lzpk * parea;
            lzfpc -= lzfpc * lzpk;
            double bfs = lzfsc * lzsk * parea;
            lzfsc -= lzfsc * lzsk;

            double bf_channel = (bfp + bfs) / (1.0 + side);
            double e4 = riva * pet_d;

            double qf = std::max(0.0, roimp + roimp_ad + ssur + sif);
            double qb = std::max(0.0, bf_channel - e4);

            // Causal 15-lag UH convolution via ring buffer
            int r_idx = t % 15;
            q_fast_ring[r_idx] = qf;
            double qf_routed = 0.0;
            for (int k = 0; k < 15; ++k) {
                int past_idx = (t - k);
                if (past_idx >= 0) {
                    qf_routed += q_fast_ring[past_idx % 15] * uh_w[k];
                }
            }
            qo[t] = static_cast<float>(std::max(0.0, qf_routed + qb));
        }
    }
    return q_out;
}
"""

_SACSMA_OMP_EXT = None
# Pure memo of calibrated parameter vectors; see SacSmaCalibratorAndDonorTransfer._calibration_cache_key.
_BASIN_CALIBRATION_CACHE: dict[tuple, np.ndarray] = {}

# Simulator backends: the batched C++ kernel (OpenMP where available) or the NumPy reference.
SIMULATOR_BACKENDS: Tuple[str, ...] = ("cpp", "numpy")
# Basins with fewer valid training observations keep the mid-range parameter vector.
MIN_VALID_CALIBRATION_DAYS: int = 10
# Each basin's DDS random numbers are drawn from its own generator in blocks of this many steps.
DDS_DRAW_BLOCK_STEPS: int = 64


def _openmp_flags() -> List[str]:
    """Return ``-fopenmp`` except on macOS (Apple clang), where the kernel builds serially."""
    return [] if sys.platform == "darwin" else ["-fopenmp"]


def _require_finite(name: str, arr: np.ndarray) -> None:
    """Raise if `arr` contains NaN/Inf (``camels_data`` guarantees gap-free forcings and attributes)."""
    n_bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    if n_bad > 0:
        raise ValueError(f"{name} contains {n_bad} non-finite values; inputs must be gap-free.")


def _get_sacsma_omp_ext():
    global _SACSMA_OMP_EXT
    if _SACSMA_OMP_EXT is not None:
        return _SACSMA_OMP_EXT
    from torch.utils.cpp_extension import load_inline

    _SACSMA_OMP_EXT = load_inline(
        name="sacsma_omp_v1",
        cpp_sources=_SACSMA_OMP_CPP,
        functions=["simulate_sacsma_batch_cpu"],
        extra_cflags=["-O3", *_openmp_flags()],
        extra_ldflags=_openmp_flags(),
        verbose=False,
    )
    return _SACSMA_OMP_EXT


def compute_hargreaves_pet(
    tmax: np.ndarray,
    tmin: np.ndarray,
    lat_deg: np.ndarray,
    doy: np.ndarray,
) -> np.ndarray:
    """Compute Hargreaves & Samani (1985) potential evapotranspiration in mm/day."""
    tmean = 0.5 * (tmax + tmin)
    tdiff = np.maximum(0.5, tmax - tmin)
    lat_rad = np.deg2rad(np.clip(lat_deg.astype(np.float64), -66.5, 66.5))[:, None]
    j = doy.astype(np.float64)[None, :]
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * j)
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * j - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))
    ra = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    )
    ra_mm = np.maximum(0.0, ra) * 0.408
    pet = 0.0023 * ra_mm * np.maximum(0.0, tmean + 17.8) * np.sqrt(tdiff)
    return np.maximum(0.0, pet).astype(np.float32)


def _derive_sacsma_p_t_pet(
    x_dyn: np.ndarray,
    lat_deg: Optional[np.ndarray] = None,
    doy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive daily (P, Tmean, PET) for Snow-17 + SAC-SMA from one 5-channel forcing block."""
    if lat_deg is None or doy is None or len(doy) != x_dyn.shape[1]:
        raise ValueError("lat_deg and a doy array matching the time axis are required for Hargreaves PET.")
    if x_dyn.shape[2] != 5:
        raise ValueError(f"Expected one 5-channel forcing block, got {x_dyn.shape[2]} channels.")
    _require_finite("x_dyn", x_dyn)
    prcp = np.maximum(0.0, x_dyn[:, :, 0]).astype(np.float32)
    tmax = x_dyn[:, :, 2].astype(np.float32)
    tmin = x_dyn[:, :, 3].astype(np.float32)
    tmean = 0.5 * (tmax + tmin)
    pet = compute_hargreaves_pet(tmax, tmin, lat_deg, doy) * 1.15
    return prcp, tmean.astype(np.float32), pet.astype(np.float32)

SACSMA_BOUNDS: Dict[str, Tuple[float, float]] = {
    "uztwm": (1.0, 150.0),
    "uzfwm": (1.0, 150.0),
    "lztwm": (1.0, 500.0),
    "lzfpm": (1.0, 1000.0),
    "lzfsm": (1.0, 1000.0),
    "adimp": (0.0, 0.4),
    "uzk": (0.1, 0.5),
    "lzpk": (0.0001, 0.025),
    "lzsk": (0.001, 0.25),
    "zperc": (1.0, 250.0),
    "rexp": (1.0, 5.0),
    "pctim": (0.0, 0.1),
    "pfree": (0.0, 0.6),
    "riva": (0.0, 0.2),
    "side": (0.0, 0.5),
    "scf": (0.7, 1.4),
    "mfmax": (0.5, 2.0),
    "mfmin": (0.05, 0.99),
    "uadj": (0.01, 0.40),
    "si": (1.0, 500.0),
    "pxtemp": (-2.0, 2.0),
    "nmf": (0.05, 0.50),
    "tipm": (0.05, 0.95),
    "mbase": (-1.0, 1.0),
    "plwhc": (0.01, 0.30),
    "uh_shape": (0.5, 5.0),
    "uh_scale": (0.5, 10.0),
}


def calc_surface_pressure(elev_m: float | np.ndarray) -> float | np.ndarray:
    """Compute standard atmospheric surface pressure (mb / hPa) from elevation (m)."""
    return 1013.25 * ((1.0 - 2.25577e-5 * np.maximum(0.0, elev_m)) ** 5.25588)


def priestley_taylor_pet(
    srad: np.ndarray,
    tmax: np.ndarray,
    tmin: np.ndarray,
    doy: Optional[np.ndarray] = None,
    elev_m: float = 500.0,
) -> np.ndarray:
    """Compute Priestley-Taylor potential evapotranspiration in mm/day (Newman et al. 2015)."""
    del doy
    tmean = 0.5 * (tmax + tmin)
    pa_kpa = 0.1 * float(calc_surface_pressure(elev_m))
    gamma = 0.000665 * pa_kpa
    es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3 + 1e-6))
    delta = (4098.0 * es) / ((tmean + 237.3 + 1e-6) ** 2)
    rn_mm = np.maximum(0.0, srad) * 0.0864 * 0.408 * 0.55
    pet = 1.26 * (delta / (delta + gamma)) * rn_mm
    return np.maximum(0.0, pet).astype(np.float32)


def _duamel_gamma_uh(shape: float, scale: float, max_lag: int = 15) -> np.ndarray:
    """Compute discrete Duamel / Gamma unit hydrograph ordinate weights of length max_lag."""
    a = max(0.25, float(shape))
    b = max(0.25, float(scale))
    t = np.arange(0.5, max_lag * 1.0, dtype=np.float64)
    log_w = (a - 1.0) * np.log(t) - (t / b)
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    w_sum = float(np.sum(w))
    if w_sum > 1e-12:
        w /= w_sum
    else:
        w = np.zeros(max_lag, dtype=np.float64)
        w[0] = 1.0
    return w


class SacSmaSnow17Simulator:
    """Coupled Snow-17 + SAC-SMA + Duamel Unit Hydrograph daily hydrological model."""

    @staticmethod
    def simulate(
        prcp: np.ndarray,
        tmean: np.ndarray,
        pet: np.ndarray,
        doy: np.ndarray,
        params: np.ndarray,
        elev_m: float = 500.0,
    ) -> np.ndarray:
        """Run coupled Snow-17 + SAC-SMA + Duamel UH over T daily timesteps.

        All 27 parameters in SACSMA_BOUNDS are physically active:
        - Snow-17: scf, mfmax, mfmin, uadj (rain-on-snow turbulent melt with elevation
          pressure), si (areal depletion curve), pxtemp, nmf, tipm (antecedent temperature
          index and cold content deficit), mbase, plwhc.
        - SAC-SMA: uztwm, uzfwm, lztwm, lzfpm, lzfsm, adimp (variable impervious area
          runoff), uzk, lzpk, lzsk, zperc, rexp, pctim, pfree, riva (riparian ET), side.
        - Routing: uh_shape, uh_scale (Duamel Gamma unit hydrograph routing on surface,
          impervious, direct, and interflow runoff).
        """
        (
            uztwm,
            uzfwm,
            lztwm,
            lzfpm,
            lzfsm,
            adimp,
            uzk,
            lzpk,
            lzsk,
            zperc,
            rexp,
            pctim,
            pfree,
            riva,
            side,
            scf,
            mfmax,
            mfmin,
            uadj,
            si,
            pxtemp,
            nmf,
            tipm,
            mbase,
            plwhc,
            uh_shape,
            uh_scale,
        ) = [float(x) for x in params]

        t_steps = len(prcp)
        q_fast = np.zeros(t_steps, dtype=np.float64)
        q_base = np.zeros(t_steps, dtype=np.float64)

        # Snow-17 state variables
        swe = 0.0
        liq_w = 0.0
        ati = 0.0
        deficit = 0.0

        # SAC-SMA state variables (initialized at 50% capacity as in Newman et al. 2015)
        uztwc = 0.5 * uztwm
        uzfwc = 0.1 * uzfwm
        lztwc = 0.5 * lztwm
        lzfpc = 0.3 * lzfpm
        lzfsc = 0.2 * lzfsm
        adimc = uztwc + lztwc

        # Atmospheric pressure (hPa/mb) from elevation for rain-on-snow psychrometric constant
        pa_mb = float(calc_surface_pressure(elev_m))
        parea = max(0.0, 1.0 - pctim - adimp)

        seasonal = 0.5 * (
            np.sin(2.0 * np.pi * (doy.astype(np.float64) - 81.0) / 365.25) + 1.0
        )

        for t in range(t_steps):
            p = float(prcp[t])
            ta = float(tmean[t])
            pet_t = float(pet[t])
            sv = float(seasonal[t])

            # 1. Snow-17 accumulation, cold content (tipm, nmf), areal depletion (si), and melt
            mf = 4.0 * (mfmin + (mfmax - mfmin) * sv)
            if ta <= pxtemp:
                snow_in = p * scf
                rain_in = 0.0
            else:
                snow_in = 0.0
                rain_in = p

            if snow_in > 0.0:
                swe += snow_in
                if ta < 0.0:
                    deficit += -ta * snow_in / 160.0
                ati = ta if swe <= snow_in * 1.01 else (ati + tipm * (ta - ati))
            elif swe > 0.0:
                ati = min(0.0, ati + tipm * (ta - ati))
            else:
                ati = 0.0
                deficit = 0.0

            if swe > 0.0:
                adc = 1.0 if swe >= si else min(1.0, max(0.05, (swe / max(1.0, si)) ** 0.75))
                if ta < mbase:
                    # Negative heat exchange increases cold-content deficit
                    delta_def = 4.0 * nmf * (mf / max(1e-4, 4.0 * mfmax)) * max(0.0, ati - ta)
                    deficit = min(swe * 0.33, max(0.0, deficit + delta_def))
                    melt = 0.0
                else:
                    if rain_in > 1.5:
                        # Rain-on-snow energy balance melt using uadj and atmospheric pressure
                        ea_mb = 6.11 * np.exp((17.27 * ta) / (ta + 237.3))
                        wind_term = 4.0 * uadj * max(0.0, (ea_mb - 6.11) + 0.00057 * pa_mb * ta)
                        adv_term = 0.0125 * rain_in * max(0.0, ta - mbase)
                        rad_term = 1.2 * max(0.0, ta - mbase)
                        pot_melt = (rad_term + adv_term + wind_term) * adc
                    else:
                        pot_melt = mf * max(0.0, ta - mbase) * adc

                    # Satisfy snowpack heat deficit first before releasing liquid melt
                    if deficit > 0.0:
                        used_for_def = min(deficit, pot_melt)
                        deficit -= used_for_def
                        pot_melt -= used_for_def

                    melt = min(swe, max(0.0, pot_melt))
                    swe -= melt
            else:
                melt = 0.0
                deficit = 0.0

            liq_w += rain_in + melt
            if ta < mbase and liq_w > 0.0:
                refrz = min(liq_w, 4.0 * nmf * (mbase - ta))
                swe += refrz
                liq_w -= refrz

            max_liq = plwhc * swe
            pxv = max(0.0, liq_w - max_liq)
            liq_w -= pxv

            # 2. SAC-SMA soil moisture accounting
            # Evapotranspiration from upper zone tension (e1), free water (e2),
            # lower zone tension (e3), additional impervious area (e5), and riparian (e4)
            e1 = min(uztwc, pet_t * (uztwc / max(1e-4, uztwm)))
            uztwc -= e1
            rem_pet = max(0.0, pet_t - e1)

            e2 = min(uzfwc, rem_pet)
            uzfwc -= e2
            rem_pet = max(0.0, rem_pet - e2)

            if (uztwc / max(1e-4, uztwm)) < (uzfwc / max(1e-4, uzfwm)):
                uzrat = (uztwc + uzfwc) / max(1e-4, uztwm + uzfwm)
                uztwc = uztwm * uzrat
                uzfwc = uzfwm * uzrat

            e3 = min(lztwc, rem_pet * (lztwc / max(1e-4, uztwm + lztwm)))
            lztwc -= e3

            e5 = min(adimc, e1 + (rem_pet) * ((adimc - e1 - uztwc) / max(1e-4, uztwm + lztwm)))
            e5 = max(0.0, e5)
            adimc = max(uztwc, min(uztwm + lztwm, adimc - e5))

            saved = side / (1.0 + side)
            ratlzt = lztwc / max(1e-4, lztwm)
            ratlz = (lztwc + lzfpc * (1.0 - saved) + lzfsc) / max(
                1e-4, lztwm + lzfpm * (1.0 - saved) + lzfsm
            )
            if ratlzt < ratlz:
                del_lz = (ratlz - ratlzt) * lztwm
                del_lz = min(del_lz, lzfsc + lzfpc)
                lztwc += del_lz
                if del_lz <= lzfsc:
                    lzfsc -= del_lz
                else:
                    rem_del = del_lz - lzfsc
                    lzfsc = 0.0
                    lzfpc = max(0.0, lzfpc - rem_del)

            # Impervious runoff (pctim) and additional impervious area runoff (adimp)
            roimp = pxv * pctim
            pav = max(0.0, pxv - uztwm + uztwc)
            adimc += pxv - pav
            roimp_ad = pav * ((adimc - uztwc) / max(1e-4, lztwm)) * adimp
            roimp_ad = max(0.0, min(pav * adimp, roimp_ad))
            adimc = min(uztwm + lztwm, adimc + (pav - roimp_ad / max(1e-6, adimp)) if adimp > 1e-6 else uztwc + lztwc)

            uztwc = min(uztwm, uztwc + pxv)

            # Percolation from upper zone free water to lower zone
            pbase = lzfpm * lzpk + lzfsm * lzsk
            def_ratio = max(
                0.0,
                1.0 - (lztwc + lzfpc + lzfsc) / max(1e-4, lztwm + lzfpm + lzfsm),
            )
            perc_demand = pbase * (1.0 + zperc * (def_ratio**rexp))
            perc = min(uzfwc, perc_demand * (uzfwc / max(1e-4, uzfwm)))
            uzfwc -= perc

            # Split percolation between lower zone tension and free water
            perc_free = perc * pfree
            perc_tens = perc - perc_free
            if lztwc + perc_tens > lztwm:
                perc_free += (lztwc + perc_tens) - lztwm
                lztwc = lztwm
            else:
                lztwc += perc_tens

            # Distribute free percolation between supplemental and primary lower zones
            if perc_free > 0.0:
                f_sup = min(lzfsm - lzfsc, perc_free * 0.5)
                lzfsc += f_sup
                lzfpc = min(lzfpm, lzfpc + (perc_free - f_sup))

            # Add remaining infiltration pav to upper zone free water and generate surface runoff
            if uzfwc + pav > uzfwm:
                ssur = (uzfwc + pav - uzfwm) * parea
                uzfwc = uzfwm
            else:
                ssur = 0.0
                uzfwc += pav

            # Interflow and baseflow
            sif = uzfwc * uzk * parea
            uzfwc -= uzfwc * uzk

            bfp = lzfpc * lzpk * parea
            lzfpc -= lzfpc * lzpk
            bfs = lzfsc * lzsk * parea
            lzfsc -= lzfsc * lzsk

            bf_channel = (bfp + bfs) / (1.0 + side)
            e4 = riva * pet_t

            q_fast[t] = max(0.0, roimp + roimp_ad + ssur + sif)
            q_base[t] = max(0.0, bf_channel - e4)

        # 3. Duamel Gamma Unit Hydrograph routing of fast runoff + baseflow
        uh_weights = _duamel_gamma_uh(uh_shape, uh_scale, max_lag=15)
        q_fast_routed = np.convolve(q_fast, uh_weights, mode="full")[:t_steps]
        q_total = np.maximum(0.0, q_fast_routed + q_base)
        return q_total.astype(np.float32)


class SacSmaCalibratorAndDonorTransfer:
    """Per-basin DDS calibrator with spatial + attribute donor transfer for PUB/PUR.

    Calibration of a basin is a pure function of (basin id, its own training data, seed,
    ``n_iter``, ``r_dds``, backend): every basin draws its DDS random numbers from its own
    generator ``np.random.default_rng([seed, int(basin_id), product_index])``, so the result
    does not depend on which other basins are calibrated in the same call, on fold order, or
    on the executor. ``_BASIN_CALIBRATION_CACHE`` is therefore a pure memo keyed on a content
    hash of the basin's inputs (forcing series, day-of-year axis, masked training discharge,
    elevation) plus every calibration setting.
    """

    def __init__(
        self,
        n_iter: int = 200,
        seed: int = 42,
        r_dds: float = 0.2,
        k_donors: int = 1,
        backend: str = "cpp",
    ) -> None:
        if backend not in SIMULATOR_BACKENDS:
            raise ValueError(f"backend must be one of {SIMULATOR_BACKENDS}, got {backend!r}.")
        self.n_iter = n_iter
        self.seed = seed
        self.r_dds = r_dds
        self.k_donors = k_donors
        self.backend = backend
        self.bounds = np.asarray(list(SACSMA_BOUNDS.values()), dtype=np.float64)

    def _make_batch_simulator(
        self,
        prcp: np.ndarray,
        tmean: np.ndarray,
        pet: np.ndarray,
        doy: np.ndarray,
        elev: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return ``simulate(params[N, 27]) -> q[N, T]`` for N basins on the selected backend."""
        if self.backend == "cpp":
            ext = _get_sacsma_omp_ext()
            p_t = torch.from_numpy(np.ascontiguousarray(prcp, dtype=np.float64))
            t_t = torch.from_numpy(np.ascontiguousarray(tmean, dtype=np.float64))
            pet_t = torch.from_numpy(np.ascontiguousarray(pet, dtype=np.float64))
            doy_t = torch.from_numpy(np.ascontiguousarray(doy, dtype=np.float64))
            elev_t = torch.from_numpy(np.ascontiguousarray(elev, dtype=np.float64))

            def simulate_cpp(params: np.ndarray) -> np.ndarray:
                par_t = torch.from_numpy(np.ascontiguousarray(params, dtype=np.float64))
                return ext.simulate_sacsma_batch_cpu(p_t, t_t, pet_t, doy_t, par_t, elev_t).numpy()

            return simulate_cpp

        def simulate_numpy(params: np.ndarray) -> np.ndarray:
            return np.stack(
                [
                    SacSmaSnow17Simulator.simulate(
                        prcp[k], tmean[k], pet[k], doy, params[k], elev_m=float(elev[k])
                    )
                    for k in range(params.shape[0])
                ],
                axis=0,
            )

        return simulate_numpy

    def _dds_calibrate_batch(
        self,
        simulate: Callable[[np.ndarray], np.ndarray],
        y_eval: np.ndarray,
        valid: np.ndarray,
        rngs: List[np.random.Generator],
    ) -> np.ndarray:
        """Run Tolson & Shoemaker (2007) Dynamically Dimensioned Search independently per basin.

        Args:
            simulate: maps candidate parameters ``[N, 27]`` to post-spinup discharge ``[N, T]``.
            y_eval: post-spinup observed discharge ``[N, T]`` (NaN where missing).
            valid: boolean mask of usable observations ``[N, T]``.
            rngs: one generator per basin; basin ``k`` only ever reads ``rngs[k]``.

        Basins with fewer than ``MIN_VALID_CALIBRATION_DAYS`` observations keep the mid-range
        parameter vector.
        """
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        span = hi - lo
        n_u = y_eval.shape[0]
        n_params = len(lo)
        n_steps = max(1, int(self.n_iter))
        n_init = min(max(2, n_steps // 10), 25)
        log_n = np.log(float(max(2, n_steps)))

        n_valid = np.sum(valid, axis=1)
        valid_counts = np.maximum(1, n_valid)
        calibratable = n_valid >= MIN_VALID_CALIBRATION_DAYS
        y_zero = np.where(valid, y_eval, 0.0)

        def rmse_of(q_sim: np.ndarray) -> np.ndarray:
            diff = np.where(valid, q_sim - y_zero, 0.0)
            return np.sqrt(np.sum(diff**2, axis=1) / valid_counts)

        best_x = np.tile(lo + 0.5 * span, (n_u, 1))
        best_rmse = rmse_of(simulate(best_x))
        for blk_start in range(1, n_steps + 1, DDS_DRAW_BLOCK_STEPS):
            blk = min(DDS_DRAW_BLOCK_STEPS, n_steps - blk_start + 1)
            u_draw = np.empty((n_u, blk, n_params), dtype=np.float64)
            pick_draw = np.empty((n_u, blk), dtype=np.float64)
            z_draw = np.empty((n_u, blk, n_params), dtype=np.float64)
            for k, rng in enumerate(rngs):
                u_draw[k] = rng.random((blk, n_params))
                pick_draw[k] = rng.random(blk)
                z_draw[k] = rng.standard_normal((blk, n_params))
            for off in range(blk):
                step = blk_start + off
                if step <= n_init:
                    cand = lo[None, :] + u_draw[:, off] * span[None, :]
                else:
                    prob = max(1.0 / n_params, 1.0 - np.log(float(step)) / log_n)
                    mask_p = u_draw[:, off] < prob
                    rows = np.flatnonzero(~np.any(mask_p, axis=1))
                    cols = np.minimum((pick_draw[rows, off] * n_params).astype(np.int64), n_params - 1)
                    mask_p[rows, cols] = True
                    perturb = z_draw[:, off] * self.r_dds * span[None, :]
                    cand_p = best_x + np.where(mask_p, perturb, 0.0)
                    cand_p = np.where(cand_p < lo[None, :], lo[None, :] + (lo[None, :] - cand_p), cand_p)
                    cand_p = np.where(cand_p > hi[None, :], hi[None, :] - (cand_p - hi[None, :]), cand_p)
                    cand = np.clip(cand_p, lo[None, :], hi[None, :])
                rmse = rmse_of(simulate(cand))
                improved = (rmse <= best_rmse) & calibratable
                best_rmse = np.where(improved, rmse, best_rmse)
                best_x = np.where(improved[:, None], cand, best_x)
        return best_x

    def _calibration_cache_key(
        self,
        basin_id: str,
        product_index: int,
        eff_spinup: int,
        arrays: Tuple[np.ndarray, ...],
    ) -> tuple:
        """Content-addressed memo key: identical inputs and settings imply identical parameters."""
        digest = hashlib.sha256()
        for arr in arrays:
            a = np.ascontiguousarray(arr)
            digest.update(f"{a.dtype.str}{a.shape}".encode())
            digest.update(a.tobytes())
        return (
            basin_id,
            int(product_index),
            int(self.seed),
            int(self.n_iter),
            float(self.r_dds),
            int(eff_spinup),
            self.backend,
            digest.hexdigest(),
        )

    def fit_and_predict(
        self,
        train_basins: List[str],
        test_basins: List[str],
        x_dyn_train: np.ndarray,
        x_stat_train: np.ndarray,
        y_obs_train: np.ndarray,
        x_dyn_test: np.ndarray,
        x_stat_test: np.ndarray,
        train_doy: np.ndarray,
        test_doy: np.ndarray,
        eval_start_idx: int,
        spinup_days: int = 365,
        lat_train: Optional[np.ndarray] = None,
        lon_train: Optional[np.ndarray] = None,
        elev_train: Optional[np.ndarray] = None,
        lat_test: Optional[np.ndarray] = None,
        lon_test: Optional[np.ndarray] = None,
        elev_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calibrate SAC-SMA + Snow-17 on train_basins and predict test_basins.

        In gauged mode (test basin in train_basins), uses the basin's own calibrated
        parameter vector. In PUB/PUR mode (test basin not in train_basins), transfers
        parameters from the nearest donor basin(s) in train_basins using spatial
        (lat, lon) and standardized catchment attributes with zero access to test qobs.
        Supports multi-forcing (3f) by calibrating and simulating each forcing product
        independently and averaging the resulting hydrographs.
        """
        for name, arr in (
            ("lat_train", lat_train),
            ("lon_train", lon_train),
            ("elev_train", elev_train),
            ("lat_test", lat_test),
            ("lon_test", lon_test),
            ("elev_test", elev_test),
        ):
            if arr is None:
                raise ValueError(f"{name} is required (camels_data always provides it).")
        _require_finite("x_stat_train", x_stat_train)
        _require_finite("x_stat_test", x_stat_test)
        n_train, t_train, d_dim = x_dyn_train.shape
        n_test, t_test, _ = x_dyn_test.shape
        if d_dim < 5 or d_dim % 5 != 0:
            raise ValueError(f"Expected 5 channels per forcing product, got {d_dim} channels.")
        eff_spinup = spinup_days if t_train > spinup_days else 0
        elev_tr_arr = np.asarray(elev_train, dtype=np.float64)
        elev_te_arr = np.asarray(elev_test, dtype=np.float64)

        # Mask pre-train_start warmup observations so they can never leak into calibration
        y_train_clean = y_obs_train.astype(np.float32, copy=True)
        if eff_spinup > 0:
            y_train_clean[:, :eff_spinup] = np.nan

        n_products = d_dim // 5

        # Standardize static attributes strictly on train_basins for donor similarity
        xs_mean = np.mean(x_stat_train, axis=0, keepdims=True)
        xs_std = np.std(x_stat_train, axis=0, keepdims=True) + 1e-6
        xs_tr_norm = (x_stat_train - xs_mean) / xs_std
        xs_te_norm = (x_stat_test - xs_mean) / xs_std

        coord_tr = np.stack([lat_train / 10.0, lon_train / 10.0], axis=1)
        coord_te = np.stack([lat_test / 10.0, lon_test / 10.0], axis=1)
        feat_tr = np.concatenate([coord_tr * 2.0, xs_tr_norm], axis=1)
        feat_te = np.concatenate([coord_te * 2.0, xs_te_norm], axis=1)

        train_ids = [str(b).zfill(8) for b in train_basins]
        test_ids = [str(b).zfill(8) for b in test_basins]
        train_basin_to_idx = {b: idx for idx, b in enumerate(train_ids)}

        # Donor indices depend only on static features, not on the forcing product.
        k = min(max(1, self.k_donors), n_train)
        donor_indices = np.zeros((n_test, k), dtype=np.int64)
        for j, tb in enumerate(test_ids):
            if tb in train_basin_to_idx:
                donor_indices[j, :] = train_basin_to_idx[tb]
            else:
                dists = np.linalg.norm(feat_tr - feat_te[j : j + 1], axis=1)
                donor_indices[j, :] = np.argsort(dists)[:k]
        all_gauged = all(tb in train_basin_to_idx for tb in test_ids)
        eff_k = 1 if all_gauged else k

        product_preds: List[np.ndarray] = []
        for p_idx in range(n_products):
            xd_tr_p = x_dyn_train[:, :, 5 * p_idx : 5 * (p_idx + 1)]
            xd_te_p = x_dyn_test[:, :, 5 * p_idx : 5 * (p_idx + 1)]
            p_tr, t_tr, pet_tr = _derive_sacsma_p_t_pet(xd_tr_p, lat_deg=lat_train, doy=train_doy)
            p_te, t_te, pet_te = _derive_sacsma_p_t_pet(xd_te_p, lat_deg=lat_test, doy=test_doy)

            calibrated_params = np.zeros((n_train, self.bounds.shape[0]), dtype=np.float64)
            cache_keys = []
            uncal_idx = []
            for i, b_id in enumerate(train_ids):
                ck = self._calibration_cache_key(
                    b_id,
                    p_idx,
                    eff_spinup,
                    (p_tr[i], t_tr[i], pet_tr[i], train_doy, y_train_clean[i], elev_tr_arr[i : i + 1]),
                )
                cache_keys.append(ck)
                if ck in _BASIN_CALIBRATION_CACHE:
                    calibrated_params[i] = _BASIN_CALIBRATION_CACHE[ck]
                else:
                    uncal_idx.append(i)

            if uncal_idx:
                u_idx = np.asarray(uncal_idx, dtype=np.int64)
                simulate_train = self._make_batch_simulator(
                    p_tr[u_idx], t_tr[u_idx], pet_tr[u_idx], train_doy, elev_tr_arr[u_idx]
                )
                y_eval_all = y_train_clean[u_idx, eff_spinup:]
                best_x = self._dds_calibrate_batch(
                    simulate=lambda params, sim=simulate_train: sim(params)[:, eff_spinup:],
                    y_eval=y_eval_all,
                    valid=~np.isnan(y_eval_all) & (y_eval_all >= 0.0),
                    rngs=[
                        np.random.default_rng([int(self.seed), int(train_ids[i]), int(p_idx)])
                        for i in uncal_idx
                    ],
                )
                for local_k, orig_i in enumerate(uncal_idx):
                    calibrated_params[orig_i] = best_x[local_k]
                    _BASIN_CALIBRATION_CACHE[cache_keys[orig_i]] = best_x[local_k].copy()

            simulate_test = self._make_batch_simulator(p_te, t_te, pet_te, test_doy, elev_te_arr)
            donor_acc = np.zeros((n_test, t_test - eval_start_idx), dtype=np.float64)
            for rank_d in range(eff_k):
                q_te_all = simulate_test(calibrated_params[donor_indices[:, rank_d]])
                donor_acc += q_te_all[:, eval_start_idx:]
            # NaN/Inf propagate so run_model's non-finite guard rejects a broken simulation.
            product_preds.append((donor_acc / float(eff_k)).astype(np.float32))

        if len(product_preds) == 1:
            return product_preds[0]
        return np.mean(np.stack(product_preds, axis=0), axis=0).astype(np.float32)
