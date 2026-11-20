# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared pluggable metric suites for HydroMIP."""

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
    mean_peak_timing,
    mse,
    nse,
    pearsonr,
    rmse,
)
from hydromip.metrics.registry import (
    GAUCH_2023_DETERMINISTIC_SUITE,
    METRIC_SUITES,
    MetricSuiteSpec,
    get_metric_suite,
    list_metric_suites,
    register_metric_suite,
)

__all__ = [
    "GAUCH_2023_DETERMINISTIC_SUITE",
    "METRIC_SUITES",
    "AllNaNError",
    "MetricSuiteSpec",
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
    "get_metric_suite",
    "infer_datetime_coord",
    "kge",
    "list_metric_suites",
    "mean_peak_timing",
    "mse",
    "nse",
    "pearsonr",
    "register_metric_suite",
    "rmse",
]
