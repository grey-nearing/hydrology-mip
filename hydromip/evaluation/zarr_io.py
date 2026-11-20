# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared cloud/local Zarr store I/O utilities for HydroMIP evaluators.

Zero ``try``/``except`` blocks. All store existence checks are explicit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gcsfs
import xarray as xr


def get_gcs_storage_options() -> dict[str, Any]:
    """Return Google Cloud Storage authentication options for ``gcsfs`` and ``xarray``."""
    adc_override = (
        os.environ["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"].strip()
        if "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE" in os.environ
        else ""
    )
    if adc_override and os.path.exists(adc_override):
        return {"token": adc_override}
    default_adc = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    if os.path.exists(default_adc):
        return {"token": default_adc}
    return {"token": "google_default"}


def zarr_store_exists(
    store_uri: str,
    storage_options: dict[str, Any] | None = None,
) -> bool:
    """Explicitly check whether a GCS or local Zarr store exists."""
    if store_uri.startswith("gs://"):
        opts = (
            storage_options
            if storage_options is not None
            else get_gcs_storage_options()
        )
        fs = gcsfs.GCSFileSystem(**opts)
        return bool(
            fs.exists(store_uri)
            or fs.exists(f"{store_uri}/.zmetadata")
            or fs.exists(f"{store_uri}/zarr.json")
            or fs.exists(f"{store_uri}/.zgroup")
        )
    local_path = Path(store_uri.removeprefix("file://"))
    if not local_path.exists() or not local_path.is_dir():
        return False
    return (
        (local_path / ".zmetadata").exists()
        or (local_path / "zarr.json").exists()
        or (local_path / ".zgroup").exists()
    )


def open_zarr_store(store_uri: str) -> xr.Dataset:
    """Verify store existence and open a GCS or local Zarr store with ``xarray``."""
    if store_uri.startswith("gs://"):
        storage_options = get_gcs_storage_options()
        fs = gcsfs.GCSFileSystem(**storage_options)
        has_consolidated = bool(fs.exists(f"{store_uri}/.zmetadata"))
        if not (
            has_consolidated
            or fs.exists(store_uri)
            or fs.exists(f"{store_uri}/zarr.json")
            or fs.exists(f"{store_uri}/.zgroup")
        ):
            raise FileNotFoundError(
                f"Required cloud Zarr store not found on GCS: {store_uri}"
            )
        return xr.open_zarr(
            store_uri,
            consolidated=has_consolidated,
            storage_options=storage_options,
        )
    if not zarr_store_exists(store_uri):
        raise FileNotFoundError(
            f"Required Zarr store not found at path: {store_uri}"
        )
    local_path = Path(store_uri.removeprefix("file://"))
    has_consolidated = (local_path / ".zmetadata").exists()
    return xr.open_zarr(str(local_path), consolidated=has_consolidated)
