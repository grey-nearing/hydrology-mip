# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible re-export of the canonical Gauch et al. (2023) daily metrics suite.

The single source of truth lives in ``hydromip.metrics.deterministic_daily`` so future
daily streamflow MIPs share the exact same metric definitions without duplication.
"""

from hydromip.metrics.deterministic_daily import (
    AllNaNError,
    alpha_nse,
    beta_kge,
    beta_nse,
    calculate_all_metrics,
    calculate_metrics,
    fdc_fhv,
    fdc_flv,
    fdc_fms,
    get_available_metrics,
    get_frequency_factor,
    infer_datetime_coord,
    kge,
    log_kge,
    log_nse,
    mean_peak_timing,
    mse,
    nse,
    pearsonr,
    rmse,
)

__all__ = [
    "AllNaNError",
    "alpha_nse",
    "beta_kge",
    "beta_nse",
    "calculate_all_metrics",
    "calculate_metrics",
    "fdc_fhv",
    "fdc_flv",
    "fdc_fms",
    "get_available_metrics",
    "get_frequency_factor",
    "infer_datetime_coord",
    "kge",
    "log_kge",
    "log_nse",
    "mean_peak_timing",
    "mse",
    "nse",
    "pearsonr",
    "rmse",
]
