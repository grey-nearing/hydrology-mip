# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""HydrologyMIP — Shared Multi-MIP Core Framework.

Provides MIP-agnostic protocol discovery (``mips/*/protocol.json``), canonical path
resolution, cross-validation anti-leakage hygiene verification, pluggable metric suites,
and shared cloud Zarr I/O so new MIPs (global Caravan, probabilistic streamflow, soil
moisture, 2D flood inundation, real-time, forecasting) plug in cleanly under ``mips/``.
"""

from hydromip.metrics.registry import (
    GAUCH_2023_DETERMINISTIC_SUITE,
    METRIC_SUITES,
    MetricSuiteSpec,
    get_metric_suite,
    list_metric_suites,
    register_metric_suite,
)
from hydromip.paths import (
    DEFAULT_MIP_ID,
    MIPS_DIR,
    MODELS_DIR,
    REPO_ROOT,
    MipPaths,
    get_repo_root,
    resolve_mip_paths,
    resolve_mip_subdir,
)
from hydromip.protocol import (
    SCHEMA_VERSION,
    SUPPORTED_TASK_TYPES,
    ForcingRegimeSpec,
    LeaderboardSpec,
    MipProtocol,
    SpatialRegimeSpec,
    discover_mips,
    get_mip,
    load_mip_protocol,
    validate_protocol_dict,
)
from hydromip.splits import CvFold, load_cv_splits_json, validate_cv_folds

__all__ = [
    "DEFAULT_MIP_ID",
    "GAUCH_2023_DETERMINISTIC_SUITE",
    "METRIC_SUITES",
    "MIPS_DIR",
    "MODELS_DIR",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "SUPPORTED_TASK_TYPES",
    "CvFold",
    "ForcingRegimeSpec",
    "LeaderboardSpec",
    "MetricSuiteSpec",
    "MipPaths",
    "MipProtocol",
    "SpatialRegimeSpec",
    "discover_mips",
    "get_metric_suite",
    "get_mip",
    "get_repo_root",
    "list_metric_suites",
    "load_cv_splits_json",
    "load_mip_protocol",
    "register_metric_suite",
    "resolve_mip_paths",
    "resolve_mip_subdir",
    "validate_cv_folds",
    "validate_protocol_dict",
]
