# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared cloud/local Zarr store I/O utilities for HydrologyMIP evaluators.

This module is the single implementation of GCS credential resolution, store existence
checks, store opening, and store-identifier validation used by `metrics/`,
`leaderboards/`, and `benchmarks/`.

Zero ``try``/``except`` blocks. All store existence checks are explicit.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dask
import fsspec
import gcsfs
import xarray as xr
import zarr

#: Allowed store / entry identifier: one alphanumeric character followed by up to 127
#: alphanumerics, ``.``, ``_`` or ``-`` (and never ``..``), so identifiers interpolated into
#: ``<bucket>/hydrographs/<leaderboard_id>/<store_id>.zarr`` can never escape that prefix.
STORE_ID_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

#: Environment variables naming an explicit credentials file, in precedence order.
CREDENTIAL_FILE_ENV_VARS: tuple[str, ...] = (
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
    "GOOGLE_APPLICATION_CREDENTIALS",
)
#: Set to ``1`` / ``true`` / ``yes`` to force anonymous (public-bucket) access.
ANONYMOUS_ENV_VAR: str = "HYDROMIP_GCS_ANON"
DEFAULT_ADC_PATH: Path = Path("~/.config/gcloud/application_default_credentials.json")
_GCE_PRODUCT_NAME_PATH: Path = Path("/sys/class/dmi/id/product_name")
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes"})


def validate_store_id(value: str, field: str = "model_id") -> str:
    """Return ``value`` if it is a safe store/entry identifier, otherwise raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}.")
    if STORE_ID_PATTERN.fullmatch(value) is None or ".." in value:
        raise ValueError(
            f"{field} {value!r} is not a valid identifier: it must match "
            f"^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ and must not contain '..'."
        )
    return value


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _credential_file_from_env(var: str) -> str | None:
    """Return the credentials file named by ``var``; raise if it is set but missing."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{var} is set to {raw!r}, but no credentials file exists at that path."
        )
    return str(path)


def running_on_gce() -> bool:
    """Return True on a Google Compute Engine VM (local DMI check; no network access).

    ``NO_GCE_CHECK=1`` (the switch honored by ``google-auth``) forces False.
    """
    if _env_flag("NO_GCE_CHECK"):
        return False
    if not _GCE_PRODUCT_NAME_PATH.is_file():
        return False
    product = _GCE_PRODUCT_NAME_PATH.read_text(encoding="utf-8", errors="replace")
    return product.strip().startswith("Google")


def get_gcs_storage_options() -> dict[str, Any]:
    """Return ``gcsfs`` / ``xarray`` storage options for Google Cloud Storage.

    Resolution order (the first match wins):

    1. ``HYDROMIP_GCS_ANON=1``: anonymous access (``{"token": "anon"}``).
    2. ``CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE``, then ``GOOGLE_APPLICATION_CREDENTIALS``:
       that credentials file. A variable that is set to a missing file raises
       ``FileNotFoundError`` instead of silently trying other credentials.
    3. The gcloud application-default credentials file
       (``~/.config/gcloud/application_default_credentials.json``).
    4. On a GCE VM: the metadata-server credentials (``{"token": "google_default"}``).
    5. Otherwise: anonymous access, which can read public buckets only.
    """
    if _env_flag(ANONYMOUS_ENV_VAR):
        return {"token": "anon"}
    for var in CREDENTIAL_FILE_ENV_VARS:
        credential_file = _credential_file_from_env(var)
        if credential_file is not None:
            return {"token": credential_file}
    default_adc = DEFAULT_ADC_PATH.expanduser()
    if default_adc.is_file():
        return {"token": str(default_adc)}
    if running_on_gce():
        return {"token": "google_default"}
    return {"token": "anon"}


@contextlib.contextmanager
def deterministic_io() -> Iterator[None]:
    """Scope single-threaded Zarr/dask I/O to a block instead of mutating global config.

    Inside the block ``zarr`` async concurrency is 1 and dask uses the synchronous
    scheduler; both settings are restored on exit.
    """
    with zarr.config.set({"async.concurrency": 1}), dask.config.set(scheduler="synchronous"):
        yield


def zarr_store_exists(
    store_uri: str,
    storage_options: dict[str, Any] | None = None,
) -> bool:
    """Explicitly check whether a GCS, public HTTPS/HTTP, or local Zarr store exists."""
    clean_uri = store_uri.rstrip("/")
    if clean_uri.startswith(("https://", "http://")):
        scheme = "https" if clean_uri.startswith("https://") else "http"
        fs = fsspec.filesystem(scheme)
        return bool(
            fs.exists(f"{clean_uri}/.zmetadata")
            or fs.exists(f"{clean_uri}/zarr.json")
            or fs.exists(f"{clean_uri}/.zgroup")
        )
    if clean_uri.startswith("gs://"):
        opts = (
            storage_options
            if storage_options is not None
            else get_gcs_storage_options()
        )
        fs = gcsfs.GCSFileSystem(**opts)
        return bool(
            fs.exists(clean_uri)
            or fs.exists(f"{clean_uri}/.zmetadata")
            or fs.exists(f"{clean_uri}/zarr.json")
            or fs.exists(f"{clean_uri}/.zgroup")
        )
    local_path = Path(clean_uri.removeprefix("file://"))
    if not local_path.exists() or not local_path.is_dir():
        return False
    return (
        (local_path / ".zmetadata").exists()
        or (local_path / "zarr.json").exists()
        or (local_path / ".zgroup").exists()
    )


def open_zarr_store(store_uri: str) -> xr.Dataset:
    """Verify store existence and open a GCS, public HTTPS/HTTP, or local Zarr store."""
    clean_uri = store_uri.rstrip("/")
    if clean_uri.startswith(("https://", "http://")):
        scheme = "https" if clean_uri.startswith("https://") else "http"
        fs = fsspec.filesystem(scheme)
        has_consolidated = bool(fs.exists(f"{clean_uri}/.zmetadata"))
        if not (
            has_consolidated
            or fs.exists(f"{clean_uri}/zarr.json")
            or fs.exists(f"{clean_uri}/.zgroup")
        ):
            raise FileNotFoundError(
                f"Required public HTTP(S) Zarr store not found: {clean_uri}"
            )
        store = zarr.storage.FsspecStore.from_url(clean_uri)
        return xr.open_zarr(store, consolidated=has_consolidated)
    if clean_uri.startswith("gs://"):
        storage_options = get_gcs_storage_options()
        fs = gcsfs.GCSFileSystem(**storage_options)
        has_consolidated = bool(fs.exists(f"{clean_uri}/.zmetadata"))
        if not (
            has_consolidated
            or fs.exists(clean_uri)
            or fs.exists(f"{clean_uri}/zarr.json")
            or fs.exists(f"{clean_uri}/.zgroup")
        ):
            raise FileNotFoundError(
                f"Required cloud Zarr store not found on GCS: {clean_uri}"
            )
        return xr.open_zarr(
            clean_uri,
            consolidated=has_consolidated,
            storage_options=storage_options,
        )
    if not zarr_store_exists(clean_uri):
        raise FileNotFoundError(
            f"Required Zarr store not found at path: {clean_uri}"
        )
    local_path = Path(clean_uri.removeprefix("file://"))
    has_consolidated = (local_path / ".zmetadata").exists()
    return xr.open_zarr(str(local_path), consolidated=has_consolidated)
