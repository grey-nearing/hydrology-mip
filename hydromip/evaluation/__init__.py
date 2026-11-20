# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation utilities for HydroMIP."""

from hydromip.evaluation.zarr_io import (
    get_gcs_storage_options,
    open_zarr_store,
    zarr_store_exists,
)

__all__ = [
    "get_gcs_storage_options",
    "open_zarr_store",
    "zarr_store_exists",
]
