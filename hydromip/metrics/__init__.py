# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared pluggable metric suites for HydrologyMIP."""

from typing import Any

from hydromip.metrics.registry import (
    GAUCH_2023_DETERMINISTIC_SUITE,
    METRIC_SUITES,
    MetricSuiteSpec,
    get_metric_suite,
    list_metric_suites,
    register_metric_suite,
)

_DETERMINISTIC_DAILY_EXPORTS: frozenset[str] = frozenset(
    {
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
    }
)


def __getattr__(name: str) -> Any:
    """Defer importing scientific dependencies until daily metric callables are accessed."""
    if name in _DETERMINISTIC_DAILY_EXPORTS:
        from hydromip.metrics import deterministic_daily as _daily

        return getattr(_daily, name)
    raise AttributeError(f"module 'hydromip.metrics' has no attribute {name!r}")

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
    "log_kge",
    "log_nse",
    "list_metric_suites",
    "mean_peak_timing",
    "mse",
    "nse",
    "pearsonr",
    "register_metric_suite",
    "rmse",
]
