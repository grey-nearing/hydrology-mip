# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation utilities for HydrologyMIP."""

from hydromip.evaluation.zarr_io import (
    STORE_ID_PATTERN,
    deterministic_io,
    get_gcs_storage_options,
    open_zarr_store,
    running_on_gce,
    validate_store_id,
    zarr_store_exists,
)

__all__ = [
    "STORE_ID_PATTERN",
    "deterministic_io",
    "get_gcs_storage_options",
    "open_zarr_store",
    "running_on_gce",
    "validate_store_id",
    "zarr_store_exists",
]
