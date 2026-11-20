# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pluggable metric-suite registry for multi-MIP HydrologyMIP evaluation.

Each MIP declares its ``metric_suite`` in ``mips/<mip_id>/protocol.json`` (for example,
``gauch_2023_deterministic`` for CAMELS-US and Caravan daily streamflow). Future MIPs
(probabilistic CRPS/PIT, 2D flood inundation CSI/IoU, soil moisture ubRMSE/R, forecast
lead-time skill) register their metric suites here without modifying existing suites.

Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hydromip.metrics.deterministic_daily import (
    calculate_all_metrics as calculate_gauch_2023_metrics,
)


@dataclass(frozen=True)
class MetricSuiteSpec:
    """Declarative specification and compute entrypoint for a named HydrologyMIP metric suite."""

    suite_id: str
    description: str
    metric_names: tuple[str, ...]
    primary_metric: str
    higher_is_better: bool
    compute_fn: Callable[..., dict[str, float]]
    citation: str


GAUCH_2023_DETERMINISTIC_SUITE = MetricSuiteSpec(
    suite_id="gauch_2023_deterministic",
    description=(
        "14 canonical daily deterministic streamflow metrics from Gauch et al. (2023) "
        "'In Defense of Metrics' (Water Resources Research), including the log-space "
        "efficiencies logNSE and logKGE."
    ),
    metric_names=(
        "NSE",
        "MSE",
        "RMSE",
        "KGE",
        "logNSE",
        "logKGE",
        "Alpha-NSE",
        "Pearson-r",
        "Beta-KGE",
        "Beta-NSE",
        "FHV",
        "FMS",
        "FLV",
        "Peak-Timing",
    ),
    primary_metric="NSE",
    higher_is_better=True,
    compute_fn=calculate_gauch_2023_metrics,
    citation="Gauch et al. (2023), Water Resources Research, 10.1029/2022WR033918",
)

METRIC_SUITES: dict[str, MetricSuiteSpec] = {
    GAUCH_2023_DETERMINISTIC_SUITE.suite_id: GAUCH_2023_DETERMINISTIC_SUITE,
}


def register_metric_suite(spec: MetricSuiteSpec, *, overwrite: bool = False) -> None:
    """Register a new metric suite in the global ``METRIC_SUITES`` registry."""
    if not spec.suite_id.strip():
        raise ValueError("MetricSuiteSpec.suite_id must be non-empty.")
    if spec.primary_metric not in spec.metric_names:
        raise ValueError(
            f"primary_metric {spec.primary_metric!r} must be listed in metric_names {spec.metric_names}."
        )
    if spec.suite_id in METRIC_SUITES and not overwrite:
        raise ValueError(
            f"Metric suite {spec.suite_id!r} is already registered; pass overwrite=True to replace."
        )
    METRIC_SUITES[spec.suite_id] = spec


def get_metric_suite(suite_id: str) -> MetricSuiteSpec:
    """Return a registered ``MetricSuiteSpec`` by ``suite_id``."""
    if suite_id not in METRIC_SUITES:
        raise KeyError(
            f"Unknown metric suite {suite_id!r}; registered suites: {sorted(METRIC_SUITES.keys())}"
        )
    return METRIC_SUITES[suite_id]


def list_metric_suites() -> list[dict[str, Any]]:
    """Return JSON-serializable metadata for every registered metric suite."""
    return [
        {
            "suite_id": spec.suite_id,
            "description": spec.description,
            "metric_names": list(spec.metric_names),
            "primary_metric": spec.primary_metric,
            "higher_is_better": spec.higher_is_better,
            "citation": spec.citation,
        }
        for _, spec in sorted(METRIC_SUITES.items())
    ]
