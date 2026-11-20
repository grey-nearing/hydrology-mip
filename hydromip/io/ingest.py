# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Safe multi-format hydrograph inspection and normalization to CAMELS-US MIP (531, 3652).

Designed to drop cleanly into ``hydromip/io/ingest.py`` and pass repository AST guardrails:
explicit input validation without ``try``/``except`` error swallowing, strict ``allow_pickle=False``
on NumPy loads, strict pickle extension rejection, ZIP traversal/symlink/size guards, and exact
delegation of per-basin metric checks to ``hydromip.metrics.deterministic_daily``.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import re
import shutil
import stat
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
from pydantic import BaseModel, Field

EXPECTED_NUM_BASINS: int = 531
EXPECTED_NUM_DAYS: int = 3652
EVAL_START_DATE: str = "1989-10-01"
EVAL_END_DATE: str = "1999-09-30"
SAMPLE_WATER_YEAR_START: str = "1994-10-01"
SAMPLE_WATER_YEAR_END: str = "1995-09-30"

MAX_ZIP_UNCOMPRESSED_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GiB
MAX_ZIP_MEMBERS: int = 20_000
MIN_VALID_OBS_DAYS: int = 30
MIN_SST_THRESHOLD: float = 1e-12

FORBIDDEN_PICKLE_EXTS: tuple[str, ...] = (".pkl", ".pickle", ".p")
DEFAULT_PREFIX_STRIPS: tuple[str, ...] = ("usgs_", "usgs-", "camels_", "basin_", "gauge_", "gage_")

DISCHARGE_VAR_CANDIDATES: tuple[str, ...] = (
    "qsim",
    "q_sim",
    "qsim_members",
    "streamflow",
    "discharge",
    "runoff",
    "q",
    "pred",
    "y_hat",
    "yhat",
    "flow",
    "sim",
    "q_pred",
    "predicted_streamflow",
)
EXCLUDED_OBS_NAMES: frozenset[str] = frozenset(
    {"qobs", "q_obs", "obs", "observed", "target", "y_true", "huc_02", "lat", "lon", "elev", "area"}
)
BASIN_DIM_CANDIDATES: tuple[str, ...] = (
    "basin",
    "basins",
    "basin_id",
    "gauge_id",
    "gage_id",
    "station_id",
    "station",
    "gauge",
    "gage",
    "catchment",
    "catchment_id",
    "hru",
    "hru_id",
    "site_no",
    "site_id",
    "id",
)
TIME_DIM_CANDIDATES: tuple[str, ...] = (
    "date",
    "dates",
    "time",
    "datetime",
    "timestamp",
    "day",
    "valid_time",
    "t",
)
MEMBER_DIM_CANDIDATES: tuple[str, ...] = (
    "member",
    "members",
    "seed",
    "seeds",
    "ensemble",
    "ensemble_member",
    "realization",
    "sample",
    "run",
)


def ensure_hydromip_on_path() -> Path | None:
    """Locate the ``camels-us-mip`` repository root and ensure ``hydromip`` is importable."""
    if importlib.util.find_spec("hydromip") is not None:
        return None
    candidates: list[Path] = []
    env_root = os.environ.get("HYDROMIP_REPO_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent)
        candidates.append(parent / "camels-us-mip")
    for cand in candidates:
        if (cand / "hydromip" / "metrics" / "deterministic_daily.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    return None


ensure_hydromip_on_path()
from hydromip.metrics.deterministic_daily import nse as canonical_nse


class IngestionRecipe(BaseModel):
    """Validated structural recipe describing how to extract canonical hydrographs."""

    format: Literal["netcdf", "zarr", "zarr.zip", "csv", "parquet", "npz"]
    var: str | None = None
    basin_dim: str | None = None
    time_dim: str | None = None
    member_dim: str | None = None
    basin_col: str | None = None
    date_col: str | None = None
    value_col: str | None = None
    member_col: str | None = None
    id_prefix_strip: list[str] = Field(default_factory=lambda: list(DEFAULT_PREFIX_STRIPS))
    zero_pad_width: int = 8
    allow_positional_order: bool = False
    unit_scale: float = 1.0
    recipe_source: Literal["deterministic_tier1", "llm_tier2", "user_override"] = "deterministic_tier1"
    confidence: float = 1.0
    notes: str = ""


def _check_not_pickle(path: Path) -> None:
    name_low = path.name.lower()
    for ext in FORBIDDEN_PICKLE_EXTS:
        if name_low.endswith(ext):
            raise ValueError(
                "Pickle files are rejected for security reasons (.pkl/.pickle/.p). "
                "Please provide NetCDF (.nc), Zarr (.zarr/.zarr.zip), CSV, Parquet, or array-only NPZ."
            )


def detect_file_format(path: Path) -> Literal["netcdf", "zarr", "zarr.zip", "csv", "parquet", "npz"]:
    """Determine supported submission file format and reject unsafe or unknown types."""
    _check_not_pickle(path)
    name_low = path.name.lower()
    if name_low.endswith(".zarr.zip"):
        return "zarr.zip"
    if path.is_dir() and (
        name_low.endswith(".zarr")
        or (path / ".zmetadata").exists()
        or (path / ".zgroup").exists()
        or (path / "zarr.json").exists()
    ):
        return "zarr"
    if name_low.endswith((".nc", ".nc4")):
        return "netcdf"
    if name_low.endswith((".csv", ".tsv")):
        return "csv"
    if name_low.endswith(".parquet"):
        return "parquet"
    if name_low.endswith(".npz"):
        return "npz"
    if name_low.endswith(".zip"):
        return "zarr.zip"
    raise ValueError(
        f"Unsupported hydrograph file extension for '{path.name}'. "
        "Supported formats: NetCDF (.nc/.nc4), Zarr (.zarr/.zarr.zip), CSV/TSV, Parquet, and NPZ."
    )


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Safely extract a ZIP archive after rejecting path traversal, symlinks, pickles, and bombs."""
    _check_not_pickle(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Archive '{zip_path.name}' is not a valid ZIP file.")
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_MEMBERS:
            raise ValueError(
                f"ZIP archive contains {len(infos)} entries, exceeding limit of {MAX_ZIP_MEMBERS}."
            )
        total_declared = 0
        for info in infos:
            raw_name = info.filename
            if not raw_name or raw_name.startswith(("/", "\\")) or "\\" in raw_name:
                raise ValueError(f"Unsafe ZIP member path rejected: {raw_name!r}")
            posix = PurePosixPath(raw_name)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"ZIP path traversal attempt rejected: {raw_name!r}")
            target = (dest_resolved / posix).resolve()
            if target != dest_resolved and not target.is_relative_to(dest_resolved):
                raise ValueError(f"ZIP member escapes destination root: {raw_name!r}")

            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink entry rejected in ZIP archive: {raw_name!r}")

            low_name = raw_name.lower()
            for ext in FORBIDDEN_PICKLE_EXTS:
                if low_name.endswith(ext):
                    raise ValueError(
                        "Pickle files are rejected for security reasons inside ZIP archives."
                    )

            total_declared += int(info.file_size)
            if total_declared > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"ZIP uncompressed size exceeds 1 GiB cap ({total_declared} bytes declared)."
                )

        # Stream entry bytes explicitly to enforce true 1 GiB cap even if headers lie.
        actual_written = 0
        chunk_bytes = 1024 * 1024
        for info in infos:
            target = (dest_resolved / PurePosixPath(info.filename)).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                while True:
                    buf = src.read(chunk_bytes)
                    if not buf:
                        break
                    actual_written += len(buf)
                    if actual_written > MAX_ZIP_UNCOMPRESSED_BYTES:
                        raise ValueError("ZIP decompressed stream exceeds 1 GiB security limit.")
                    dst.write(buf)
    return dest_resolved


def _locate_zarr_root(extracted_dir: Path) -> Path:
    """Locate the Zarr group root inside an extracted directory."""
    if (
        (extracted_dir / ".zgroup").exists()
        or (extracted_dir / ".zmetadata").exists()
        or (extracted_dir / "zarr.json").exists()
    ):
        return extracted_dir
    child_dirs = sorted(
        p for p in extracted_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "__MACOSX"))
    )
    for child in child_dirs:
        if (
            (child / ".zgroup").exists()
            or (child / ".zmetadata").exists()
            or (child / "zarr.json").exists()
        ):
            return child
    if len(child_dirs) == 1:
        return child_dirs[0]
    raise ValueError(
        "Could not locate a valid Zarr store root (.zgroup / .zmetadata / zarr.json) inside archive."
    )


def _open_zarr_dir(zarr_dir: Path) -> xr.Dataset:
    consolidated = (zarr_dir / ".zmetadata").exists()
    return xr.open_zarr(zarr_dir, consolidated=consolidated)


def _detect_csv_sep(path: Path) -> str:
    if path.name.lower().endswith(".tsv"):
        return "\t"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if "\t" in header and header.count("\t") >= header.count(","):
        return "\t"
    return ","


def _json_safe_value(val: Any) -> Any:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, (np.datetime64, pd.Timestamp, dt.date, dt.datetime)):
        return str(pd.Timestamp(val).date())
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        fval = float(val)
        return None if not np.isfinite(fval) else round(fval, 6)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [_json_safe_value(x) for x in val.tolist()]
    if isinstance(val, (str, int, float, bool)) or val is None:
        if isinstance(val, float) and not np.isfinite(val):
            return None
        return val
    return str(val)


def _sample_array_values(arr: np.ndarray, limit: int = 5) -> list[Any]:
    flat = np.asarray(arr).ravel()[:limit]
    return [_json_safe_value(x) for x in flat]


def _extract_xarray_metadata(ds: xr.Dataset, path: Path, fmt: str) -> dict[str, Any]:
    dims = list(ds.dims)
    dim_sizes = {str(k): int(v) for k, v in ds.sizes.items()}
    data_vars = list(ds.data_vars)
    coords = list(ds.coords)
    var_shapes = {str(v): [int(s) for s in ds[v].shape] for v in ds.variables}
    var_dims = {str(v): [str(d) for d in ds[v].dims] for v in ds.variables}
    var_dtypes = {str(v): str(ds[v].dtype) for v in ds.variables}
    var_attrs = {
        str(v): {str(ak): _json_safe_value(av) for ak, av in ds[v].attrs.items()}
        for v in ds.variables
    }
    sample_coords: dict[str, list[Any]] = {}
    for c_name in list(coords) + [v for v in data_vars if ds[v].ndim == 1]:
        da = ds[c_name]
        if da.ndim == 1:
            sample_coords[str(c_name)] = _sample_array_values(da.isel({da.dims[0]: slice(0, 5)}).values)

    return {
        "filename": path.name,
        "format": fmt,
        "file_size_bytes": int(path.stat().st_size) if path.is_file() else 0,
        "dims": [str(d) for d in dims],
        "dim_sizes": dim_sizes,
        "coords": [str(c) for c in coords],
        "vars": [str(v) for v in data_vars],
        "var_shapes": var_shapes,
        "var_dims": var_dims,
        "var_dtypes": var_dtypes,
        "var_attrs": var_attrs,
        "global_attrs": {str(k): _json_safe_value(v) for k, v in ds.attrs.items()},
        "sample_coords": sample_coords,
        "columns": [],
        "sample_rows": [],
    }


def extract_file_metadata(path: Path | str, work_dir: Path | None = None) -> dict[str, Any]:
    """Extract purely structural JSON metadata from a supported archive without code execution."""
    src_path = Path(path)
    fmt = detect_file_format(src_path)

    if fmt == "netcdf":
        with xr.open_dataset(src_path) as ds:
            return _extract_xarray_metadata(ds, src_path, "netcdf")

    if fmt == "zarr":
        with _open_zarr_dir(src_path) as ds:
            return _extract_xarray_metadata(ds, src_path, "zarr")

    if fmt == "zarr.zip":
        base = work_dir if work_dir is not None else src_path.parent / f".inspect_{src_path.stem}"
        cleanup = work_dir is None
        extracted = safe_extract_zip(src_path, base)
        zarr_root = _locate_zarr_root(extracted)
        with _open_zarr_dir(zarr_root) as ds:
            meta = _extract_xarray_metadata(ds, src_path, "zarr.zip")
        if cleanup and extracted.exists():
            shutil.rmtree(extracted)
        return meta

    if fmt == "csv":
        sep = _detect_csv_sep(src_path)
        df_head = pd.read_csv(src_path, sep=sep, nrows=6)
        cols = [str(c) for c in df_head.columns]
        sample_rows = [
            {str(k): _json_safe_value(v) for k, v in row.items()}
            for row in df_head.head(5).to_dict(orient="records")
        ]
        sample_cols = {c: [_json_safe_value(v) for v in df_head[c].head(5).tolist()] for c in cols[:25]}
        return {
            "filename": src_path.name,
            "format": "csv",
            "file_size_bytes": int(src_path.stat().st_size),
            "sep": sep,
            "dims": cols[:8],
            "dim_sizes": {"sample_rows": len(df_head), "n_cols": len(cols)},
            "coords": cols[:8],
            "vars": cols[:20],
            "var_shapes": {c: [len(df_head)] for c in cols[:20]},
            "var_dtypes": {str(c): str(df_head[c].dtype) for c in cols[:20]},
            "var_attrs": {},
            "global_attrs": {},
            "sample_coords": sample_cols,
            "columns": cols,
            "sample_rows": sample_rows,
        }

    if fmt == "parquet":
        pf = pq.ParquetFile(src_path)
        schema = pf.schema_arrow
        cols = [str(field.name) for field in schema]
        dtypes = {str(field.name): str(field.type) for field in schema}
        n_rows = int(pf.metadata.num_rows)
        head_tbl = pf.read_row_group(0).slice(0, 5) if pf.num_row_groups > 0 else pf.read()
        df_head = head_tbl.to_pandas().head(5)
        sample_rows = [
            {str(k): _json_safe_value(v) for k, v in row.items()}
            for row in df_head.to_dict(orient="records")
        ]
        sample_cols = {c: [_json_safe_value(v) for v in df_head[c].head(5).tolist()] for c in cols[:25]}
        return {
            "filename": src_path.name,
            "format": "parquet",
            "file_size_bytes": int(src_path.stat().st_size),
            "dims": cols[:8],
            "dim_sizes": {"rows": n_rows, "n_cols": len(cols)},
            "coords": cols[:8],
            "vars": cols[:20],
            "var_shapes": {c: [n_rows] for c in cols[:20]},
            "var_dtypes": dtypes,
            "var_attrs": {},
            "global_attrs": {},
            "sample_coords": sample_cols,
            "columns": cols,
            "sample_rows": sample_rows,
        }

    # NPZ (strictly allow_pickle=False)
    with np.load(src_path, allow_pickle=False) as npz:
        keys = [str(k) for k in npz.files]
        var_shapes = {k: [int(s) for s in npz[k].shape] for k in keys}
        var_dtypes = {k: str(npz[k].dtype) for k in keys}
        sample_coords = {
            k: _sample_array_values(npz[k][:5])
            for k in keys
            if npz[k].ndim == 1
        }
        array_vars = [k for k in keys if npz[k].ndim in (2, 3)]
        coord_keys = [k for k in keys if npz[k].ndim == 1]
    return {
        "filename": src_path.name,
        "format": "npz",
        "file_size_bytes": int(src_path.stat().st_size),
        "dims": coord_keys,
        "dim_sizes": {k: var_shapes[k][0] for k in coord_keys},
        "coords": coord_keys,
        "vars": array_vars or keys,
        "var_shapes": var_shapes,
        "var_dtypes": var_dtypes,
        "var_attrs": {},
        "global_attrs": {},
        "sample_coords": sample_coords,
        "columns": [],
        "sample_rows": [],
    }


def normalize_basin_id(
    raw: Any,
    prefix_strip: Sequence[str] = DEFAULT_PREFIX_STRIPS,
    zero_pad_width: int = 8,
) -> str:
    """Normalize a USGS gauge identifier by stripping common prefixes and zero-padding to 8 chars."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, (int, np.integer)):
        text = str(int(raw))
    elif isinstance(raw, (float, np.floating)):
        text = str(int(raw)) if float(raw).is_integer() else str(raw)
    else:
        text = str(raw)
    text = text.strip().strip("\"'")
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    low = text.lower()
    for pfx in prefix_strip:
        pfx_low = str(pfx).lower()
        if pfx_low and low.startswith(pfx_low):
            text = text[len(pfx_low) :]
            low = text.lower()
    text = text.strip()
    if text.isdigit() and len(text) <= zero_pad_width:
        return text.zfill(zero_pad_width)
    return text


def _looks_like_basin_values(
    values: Sequence[Any],
    prefix_strip: Sequence[str] = DEFAULT_PREFIX_STRIPS,
) -> bool:
    if not values:
        return False
    hits = 0
    for v in values:
        norm = normalize_basin_id(v, prefix_strip=prefix_strip)
        if len(norm) == 8 and norm.isdigit():
            hits += 1
    return hits >= max(1, (len(values) + 1) // 2)


def _looks_like_date_values(values: Sequence[Any]) -> bool:
    if not values:
        return False
    strs = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not strs:
        return False
    # Reject plain integers like gauge IDs (`01022500`) from masquerading as YYYYMMDD timestamps
    digit_only = sum(1 for s in strs if s.isdigit() and len(s) != 8)
    if digit_only == len(strs):
        return False
    if all(s.isdigit() and len(s) == 8 and not s.startswith(("19", "20")) for s in strs):
        return False
    parsed = pd.to_datetime(strs, errors="coerce")
    return bool(parsed.notna().sum() >= max(1, (len(strs) + 1) // 2))


def _pick_first_matching(names: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_map = {str(n).lower(): str(n) for n in names}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for cand in candidates:
        if len(cand) < 3:
            continue
        for n in names:
            nl = str(n).lower()
            if nl in EXCLUDED_OBS_NAMES:
                continue
            if cand in nl:
                return str(n)
    return None


def _pick_xarray_var(meta: Mapping[str, Any]) -> str | None:
    vars_list = [str(v) for v in meta.get("vars", []) if str(v).lower() not in EXCLUDED_OBS_NAMES]
    var_shapes: Mapping[str, list[int]] = meta.get("var_shapes", {})
    nd_vars = [v for v in vars_list if len(var_shapes.get(v, [])) in (2, 3)]
    pool = nd_vars or vars_list
    if not pool:
        return None
    picked = _pick_first_matching(pool, DISCHARGE_VAR_CANDIDATES)
    if picked is not None:
        return picked
    if len(nd_vars) == 1:
        return nd_vars[0]
    return None


def resolve_deterministic_recipe(
    meta: Mapping[str, Any],
    recipe_override: Mapping[str, Any] | IngestionRecipe | None = None,
) -> tuple[IngestionRecipe | None, list[str]]:
    """Resolve an ``IngestionRecipe`` deterministically from metadata (Tier 1) or user override."""
    fmt = str(meta.get("format", "")).lower()
    if recipe_override is not None:
        if isinstance(recipe_override, IngestionRecipe):
            return recipe_override.model_copy(update={"recipe_source": "user_override"}), []
        merged = {
            "format": fmt,
            **dict(recipe_override),
            "recipe_source": "user_override",
        }
        return IngestionRecipe.model_validate(merged), []

    issues: list[str] = []

    if fmt in ("netcdf", "zarr", "zarr.zip"):
        var = _pick_xarray_var(meta)
        if var is None:
            issues.append(
                f"Ambiguous or missing discharge variable in data_vars {meta.get('vars', [])}."
            )
            return None, issues
        var_dims = list(meta.get("var_dims", {}).get(var, []))
        sample_coords: Mapping[str, list[Any]] = meta.get("sample_coords", {})
        coords = list(meta.get("coords", [])) + [
            v for v, shp in meta.get("var_shapes", {}).items() if len(shp) == 1
        ]
        basin_dim = _pick_first_matching(var_dims, BASIN_DIM_CANDIDATES)
        if basin_dim is None:
            for d in var_dims:
                if _looks_like_basin_values(sample_coords.get(d, [])):
                    basin_dim = d
                    break
        remaining_for_time = [d for d in var_dims if d != basin_dim]
        time_dim = _pick_first_matching(remaining_for_time, TIME_DIM_CANDIDATES)
        if time_dim is None:
            for d in remaining_for_time:
                if _looks_like_date_values(sample_coords.get(d, [])):
                    time_dim = d
                    break
        member_dim = None
        if len(var_dims) == 3:
            remaining = [d for d in var_dims if d not in (basin_dim, time_dim)]
            member_dim = _pick_first_matching(remaining, MEMBER_DIM_CANDIDATES) or (
                remaining[0] if len(remaining) == 1 else None
            )

        # Verify that the basin dimension is identifiable via a coordinate or 1D variable
        has_basin_coord = basin_dim is not None and (
            basin_dim in sample_coords
            or any(
                c in BASIN_DIM_CANDIDATES
                and meta.get("var_dims", {}).get(c, []) == [basin_dim]
                for c in coords
            )
        )
        if basin_dim is None or not has_basin_coord:
            issues.append(
                f"Cannot identify labeled basin dimension/coordinate from dims {var_dims} and coords {coords}."
            )
        if time_dim is None or (time_dim not in sample_coords and not any(
            meta.get("var_dims", {}).get(c, []) == [time_dim] and c.lower() in TIME_DIM_CANDIDATES
            for c in coords
        )):
            issues.append(
                f"Cannot identify labeled date/time coordinate from dims {var_dims} and coords {coords}."
            )
        if len(var_dims) == 3 and member_dim is None:
            issues.append(f"Cannot identify ensemble member dimension among dims {var_dims}.")

        if issues:
            return None, issues
        return (
            IngestionRecipe(
                format=fmt,  # type: ignore[arg-type]
                var=var,
                basin_dim=basin_dim,
                time_dim=time_dim,
                member_dim=member_dim,
                recipe_source="deterministic_tier1",
                confidence=1.0,
                notes="Resolved deterministically from variable and coordinate metadata.",
            ),
            [],
        )

    if fmt in ("csv", "parquet"):
        cols: list[str] = [str(c) for c in meta.get("columns", [])]
        sample_cols: Mapping[str, list[Any]] = meta.get("sample_coords", {})
        date_col = _pick_first_matching(cols, TIME_DIM_CANDIDATES)
        basin_col = _pick_first_matching(
            [c for c in cols if c != date_col], BASIN_DIM_CANDIDATES
        )
        member_col = _pick_first_matching(
            [c for c in cols if c not in (date_col, basin_col)], MEMBER_DIM_CANDIDATES
        )

        basin_header_cols = [c for c in cols if c != date_col and _looks_like_basin_values([c])]
        if len(basin_header_cols) >= 10:
            # Wide table: rows = dates, columns = basin IDs
            if date_col is None and cols:
                # Check if first column holds dates
                first_col = cols[0]
                if _looks_like_date_values(sample_cols.get(first_col, [])):
                    date_col = first_col
            if date_col is None:
                issues.append("Wide table has basin columns but no recognizable date column.")
                return None, issues
            return (
                IngestionRecipe(
                    format=fmt,  # type: ignore[arg-type]
                    date_col=date_col,
                    basin_dim="columns",
                    time_dim=date_col,
                    member_col=member_col,
                    recipe_source="deterministic_tier1",
                    confidence=0.98,
                    notes=f"Wide {fmt.upper()} layout with {len(basin_header_cols)} basin columns.",
                ),
                [],
            )

        # Also support transposed wide table (rows = basins, columns = dates)
        date_header_cols = [c for c in cols if c != basin_col and _looks_like_date_values([c])]
        if len(date_header_cols) >= 30 and basin_col is not None:
            return (
                IngestionRecipe(
                    format=fmt,  # type: ignore[arg-type]
                    basin_col=basin_col,
                    basin_dim=basin_col,
                    time_dim="columns",
                    member_col=member_col,
                    recipe_source="deterministic_tier1",
                    confidence=0.95,
                    notes=f"Transposed wide {fmt.upper()} layout with {len(date_header_cols)} date columns.",
                ),
                [],
            )

        # Long/tidy table: requires basin_col, date_col, value_col
        exclude_cols = {c for c in (basin_col, date_col, member_col) if c} | EXCLUDED_OBS_NAMES
        value_candidates = [c for c in cols if c.lower() not in {x.lower() for x in exclude_cols}]
        value_col = _pick_first_matching(value_candidates, DISCHARGE_VAR_CANDIDATES)
        if value_col is None and len(value_candidates) == 1:
            value_col = value_candidates[0]

        if basin_col is None or not _looks_like_basin_values(sample_cols.get(basin_col, [])):
            issues.append(f"Missing or unlabeled basin column in {cols[:15]}.")
        if date_col is None or not _looks_like_date_values(sample_cols.get(date_col, [])):
            issues.append(f"Missing or unparseable date column in {cols[:15]}.")
        if value_col is None:
            issues.append(f"Missing or ambiguous discharge value column in {cols[:15]}.")
        if issues:
            return None, issues
        return (
            IngestionRecipe(
                format=fmt,  # type: ignore[arg-type]
                var=value_col,
                basin_col=basin_col,
                date_col=date_col,
                value_col=value_col,
                member_col=member_col,
                basin_dim=basin_col,
                time_dim=date_col,
                member_dim=member_col,
                recipe_source="deterministic_tier1",
                confidence=0.98,
                notes=f"Long/tidy {fmt.upper()} layout.",
            ),
            [],
        )

    if fmt == "npz":
        var_shapes: Mapping[str, list[int]] = meta.get("var_shapes", {})
        sample_coords: Mapping[str, list[Any]] = meta.get("sample_coords", {})
        coord_keys = list(meta.get("coords", []))
        array_vars = [k for k in meta.get("vars", []) if str(k).lower() not in EXCLUDED_OBS_NAMES]
        var = _pick_first_matching(array_vars, DISCHARGE_VAR_CANDIDATES)
        if var is None and len(array_vars) == 1:
            var = array_vars[0]
        basin_key = _pick_first_matching(coord_keys, BASIN_DIM_CANDIDATES)
        date_key = _pick_first_matching(coord_keys, TIME_DIM_CANDIDATES)
        if var is None:
            issues.append(f"Missing or ambiguous prediction array key among {list(var_shapes)}.")
        if basin_key is None or not _looks_like_basin_values(sample_coords.get(basin_key, [])):
            issues.append(
                f"NPZ archive lacks an explicit basin coordinate key among {coord_keys}; "
                "positional basin order is disabled by default."
            )
        if date_key is None or not _looks_like_date_values(sample_coords.get(date_key, [])):
            issues.append(f"NPZ archive lacks an explicit date coordinate key among {coord_keys}.")
        if issues:
            return None, issues
        shp = var_shapes.get(var or "", [])
        return (
            IngestionRecipe(
                format="npz",
                var=var,
                basin_dim=basin_key,
                time_dim=date_key,
                member_dim="member" if len(shp) == 3 else None,
                recipe_source="deterministic_tier1",
                confidence=0.99,
                notes="NPZ archive with explicit prediction, basin, and date arrays.",
            ),
            [],
        )

    issues.append(f"Unsupported format {fmt!r}.")
    return None, issues


def _align_axes_and_slice(
    raw_values: np.ndarray,
    raw_basins: Sequence[Any] | None,
    raw_dates: Sequence[Any] | None,
    axis_roles: tuple[str, ...],
    canonical_basins: Sequence[str],
    canonical_dates: Sequence[str],
    recipe: IngestionRecipe,
    *,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, list[str]]:
    """Transpose, normalize IDs/dates, align basins & slice the canonical evaluation window."""
    if len(axis_roles) not in (2, 3) or set(axis_roles) - {"member", "basin", "date"}:
        raise ValueError(f"Unsupported axis roles {axis_roles}.")

    target_order = ("member", "basin", "date") if "member" in axis_roles else ("basin", "date")
    perm = [axis_roles.index(role) for role in target_order]
    arr = np.transpose(np.asarray(raw_values, dtype=np.float32), perm)
    if recipe.unit_scale != 1.0:
        arr = (arr * np.float32(recipe.unit_scale)).astype(np.float32)

    basin_count = arr.shape[-2]
    date_count = arr.shape[-1]
    canon_basins = [normalize_basin_id(b) for b in canonical_basins]
    canon_dates = np.asarray(canonical_dates, dtype="datetime64[D]")

    if raw_basins is None:
        if not recipe.allow_positional_order:
            raise ValueError(
                "Archive omits explicit basin ID labels and allow_positional_order=False; "
                "refusing to guess basin order."
            )
        if basin_count != len(canon_basins):
            raise ValueError(
                f"Positional basin count {basin_count} does not equal expected {len(canon_basins)}."
            )
        norm_basins = list(canon_basins)
    else:
        if len(raw_basins) != basin_count:
            raise ValueError(f"Basin label length {len(raw_basins)} != array axis size {basin_count}.")
        norm_basins = [
            normalize_basin_id(
                b, prefix_strip=recipe.id_prefix_strip, zero_pad_width=recipe.zero_pad_width
            )
            for b in raw_basins
        ]
        if len(set(norm_basins)) != len(norm_basins):
            raise ValueError("Duplicate basin identifiers detected after ID normalization.")

    if raw_dates is None:
        if not recipe.allow_positional_order:
            raise ValueError(
                "Archive omits explicit date coordinates and allow_positional_order=False; "
                "refusing to guess date alignment."
            )
        if date_count != len(canon_dates):
            raise ValueError(
                f"Positional time step count {date_count} does not equal expected {len(canon_dates)}."
            )
        norm_dates = canon_dates.copy()
    else:
        if len(raw_dates) != date_count:
            raise ValueError(f"Date coordinate length {len(raw_dates)} != array axis size {date_count}.")
        parsed = pd.to_datetime(list(raw_dates), errors="coerce")
        if parsed.isna().any():
            bad_sample = [str(raw_dates[i]) for i in np.where(parsed.isna())[0][:3]]
            raise ValueError(f"Unparseable date coordinates detected (e.g. {bad_sample}).")
        norm_dates = parsed.values.astype("datetime64[D]")
        if len(np.unique(norm_dates)) != len(norm_dates):
            raise ValueError("Duplicate timestamps detected in date coordinate.")

    # Map dates onto canonical 3652-day Newman window
    date_to_idx = {d: i for i, d in enumerate(norm_dates)}
    matched_date_mask = np.asarray([d in date_to_idx for d in canon_dates], dtype=bool)
    n_dates_matched = int(np.sum(matched_date_mask))
    if strict and n_dates_matched != len(canon_dates):
        missing_dates = [str(canon_dates[i]) for i in np.where(~matched_date_mask)[0][:3]]
        raise ValueError(
            f"Archive covers only {n_dates_matched}/{len(canon_dates)} canonical evaluation days "
            f"({EVAL_START_DATE}..{EVAL_END_DATE}); first missing: {missing_dates}."
        )

    basin_to_idx = {b: i for i, b in enumerate(norm_basins)}
    matched_basin_mask = np.asarray([b in basin_to_idx for b in canon_basins], dtype=bool)
    n_basins_matched = int(np.sum(matched_basin_mask))
    if strict and n_basins_matched != len(canon_basins):
        missing_basins = [canon_basins[i] for i in np.where(~matched_basin_mask)[0][:5]]
        raise ValueError(
            f"Archive matched only {n_basins_matched}/{len(canon_basins)} canonical CAMELS-US basins "
            f"(first missing: {missing_basins})."
        )

    canon_b_idx = np.where(matched_basin_mask)[0]
    src_b_idx = np.asarray([basin_to_idx[canon_basins[i]] for i in canon_b_idx], dtype=np.intp)
    canon_d_idx = np.where(matched_date_mask)[0]
    src_d_idx = np.asarray([date_to_idx[canon_dates[i]] for i in canon_d_idx], dtype=np.intp)

    diagnostics: list[str] = []
    if len(norm_dates) > len(canon_dates) and n_dates_matched == len(canon_dates):
        diagnostics.append(
            f"Sliced {len(norm_dates)}-day input down to canonical {len(canon_dates)}-day "
            f"Newman window ({EVAL_START_DATE} to {EVAL_END_DATE})."
        )

    if arr.ndim == 3:
        n_members = int(arr.shape[0])
        members_out = np.full(
            (n_members, len(canon_basins), len(canon_dates)), np.nan, dtype=np.float32
        )
        sliced = arr[:, src_b_idx[:, None], src_d_idx[None, :]]
        members_out[:, canon_b_idx[:, None], canon_d_idx[None, :]] = sliced
        qsim_out = np.mean(members_out.astype(np.float64), axis=0).astype(np.float32)
        return qsim_out, members_out, matched_basin_mask, matched_date_mask, diagnostics

    qsim_out = np.full((len(canon_basins), len(canon_dates)), np.nan, dtype=np.float32)
    qsim_out[np.ix_(canon_b_idx, canon_d_idx)] = arr[np.ix_(src_b_idx, src_d_idx)]
    return qsim_out, None, matched_basin_mask, matched_date_mask, diagnostics


def _find_xarray_coord_values(ds: xr.Dataset, dim_name: str, role_candidates: Sequence[str]) -> np.ndarray | None:
    if dim_name in ds.coords:
        return np.asarray(ds.coords[dim_name].values)
    if dim_name in ds.variables and tuple(ds[dim_name].dims) == (dim_name,):
        return np.asarray(ds[dim_name].values)
    for name in list(ds.coords) + list(ds.data_vars):
        da = ds[name]
        if tuple(da.dims) == (dim_name,) and str(name).lower() in role_candidates:
            return np.asarray(da.values)
    return None


def _load_xarray_to_canonical(
    ds: xr.Dataset,
    recipe: IngestionRecipe,
    canonical_basins: Sequence[str],
    canonical_dates: Sequence[str],
    *,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, list[str]]:
    if not recipe.var or recipe.var not in ds.variables:
        raise ValueError(f"Variable {recipe.var!r} not found in dataset variables {list(ds.variables)}.")
    da = ds[recipe.var]
    dims = list(da.dims)
    if not recipe.basin_dim or recipe.basin_dim not in dims:
        raise ValueError(f"Basin dim {recipe.basin_dim!r} not in {recipe.var} dims {dims}.")
    if not recipe.time_dim or recipe.time_dim not in dims:
        raise ValueError(f"Time dim {recipe.time_dim!r} not in {recipe.var} dims {dims}.")

    roles: list[str] = []
    for d in dims:
        if d == recipe.basin_dim:
            roles.append("basin")
        elif d == recipe.time_dim:
            roles.append("date")
        elif recipe.member_dim and d == recipe.member_dim:
            roles.append("member")
        else:
            raise ValueError(f"Unexpected unmapped dimension {d!r} on {recipe.var!r}.")

    raw_basins = _find_xarray_coord_values(ds, recipe.basin_dim, BASIN_DIM_CANDIDATES)
    raw_dates = _find_xarray_coord_values(ds, recipe.time_dim, TIME_DIM_CANDIDATES)
    qsim, members, b_mask, d_mask, diags = _align_axes_and_slice(
        da.values,
        raw_basins,
        raw_dates,
        tuple(roles),
        canonical_basins,
        canonical_dates,
        recipe,
        strict=strict,
    )

    # Preserve sibling `qsim_members` if caller selected a 2D `qsim` inside a Zarr/NetCDF multi-seed store
    if members is None and "qsim_members" in ds.data_vars and recipe.var == "qsim":
        mem_da = ds["qsim_members"]
        if mem_da.ndim == 3 and recipe.basin_dim in mem_da.dims and recipe.time_dim in mem_da.dims:
            m_roles = tuple(
                "basin" if d == recipe.basin_dim else ("date" if d == recipe.time_dim else "member")
                for d in mem_da.dims
            )
            _, members_loaded, _, _, _ = _align_axes_and_slice(
                mem_da.values,
                raw_basins,
                raw_dates,
                m_roles,
                canonical_basins,
                canonical_dates,
                recipe,
                strict=strict,
            )
            members = members_loaded
    return qsim, members, b_mask, d_mask, diags


def _load_tabular_to_canonical(
    df: pd.DataFrame,
    recipe: IngestionRecipe,
    canonical_basins: Sequence[str],
    canonical_dates: Sequence[str],
    *,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, list[str]]:
    cols = [str(c) for c in df.columns]
    df = df.copy()
    df.columns = cols

    # Wide layout: rows = dates, columns = basins
    if recipe.basin_dim == "columns" or (recipe.date_col and not recipe.basin_col and not recipe.value_col):
        date_col = recipe.date_col or recipe.time_dim
        if not date_col or date_col not in df.columns:
            raise ValueError(f"Date column {date_col!r} not found in tabular columns.")
        raw_dates = df[date_col].tolist()
        basin_cols = [
            c for c in cols if c != date_col and _looks_like_basin_values([c], recipe.id_prefix_strip)
        ]
        if not basin_cols:
            raise ValueError("No valid basin columns detected in wide tabular dataset.")
        vals = df[basin_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        return _align_axes_and_slice(
            vals,
            basin_cols,
            raw_dates,
            ("date", "basin"),
            canonical_basins,
            canonical_dates,
            recipe,
            strict=strict,
        )

    # Transposed wide layout: rows = basins, columns = dates
    if recipe.time_dim == "columns" and recipe.basin_col:
        if recipe.basin_col not in df.columns:
            raise ValueError(f"Basin column {recipe.basin_col!r} not found in tabular columns.")
        raw_basins = df[recipe.basin_col].tolist()
        date_cols = [c for c in cols if c != recipe.basin_col and _looks_like_date_values([c])]
        if not date_cols:
            raise ValueError("No valid date columns detected in transposed wide tabular dataset.")
        vals = df[date_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        return _align_axes_and_slice(
            vals,
            raw_basins,
            date_cols,
            ("basin", "date"),
            canonical_basins,
            canonical_dates,
            recipe,
            strict=strict,
        )

    # Long/tidy layout
    basin_col = recipe.basin_col or recipe.basin_dim
    date_col = recipe.date_col or recipe.time_dim
    value_col = recipe.value_col or recipe.var
    member_col = recipe.member_col or recipe.member_dim
    for label, col_name in (("basin_col", basin_col), ("date_col", date_col), ("value_col", value_col)):
        if not col_name or col_name not in df.columns:
            raise ValueError(f"Required {label}={col_name!r} not present in columns {cols[:15]}.")

    canon_basins = [normalize_basin_id(b) for b in canonical_basins]
    canon_dates = np.asarray(canonical_dates, dtype="datetime64[D]")
    basin_to_idx = {b: i for i, b in enumerate(canon_basins)}
    date_to_idx = {d: i for i, d in enumerate(canon_dates)}

    norm_b_series = [
        normalize_basin_id(
            v, prefix_strip=recipe.id_prefix_strip, zero_pad_width=recipe.zero_pad_width
        )
        for v in df[basin_col].tolist()
    ]
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    if parsed_dates.isna().all():
        raise ValueError(f"Column {date_col!r} could not be parsed as dates.")
    norm_d_arr = parsed_dates.values.astype("datetime64[D]")
    vals_arr = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float32)
    if recipe.unit_scale != 1.0:
        vals_arr = (vals_arr * np.float32(recipe.unit_scale)).astype(np.float32)

    b_indices = np.asarray([basin_to_idx.get(b, -1) for b in norm_b_series], dtype=np.int64)
    d_indices = np.asarray(
        [date_to_idx.get(d, -1) if not np.isnat(d) else -1 for d in norm_d_arr], dtype=np.int64
    )
    matched_b = np.zeros(len(canon_basins), dtype=bool)
    matched_d = np.zeros(len(canon_dates), dtype=bool)
    valid_b_mask = b_indices >= 0
    valid_d_mask = d_indices >= 0
    if np.any(valid_b_mask):
        matched_b[b_indices[valid_b_mask]] = True
    if np.any(valid_d_mask):
        matched_d[d_indices[valid_d_mask]] = True

    if strict and int(np.sum(matched_b)) != len(canon_basins):
        raise ValueError(
            f"Long table matched only {int(np.sum(matched_b))}/{len(canon_basins)} canonical basins."
        )
    if strict and int(np.sum(matched_d)) != len(canon_dates):
        raise ValueError(
            f"Long table matched only {int(np.sum(matched_d))}/{len(canon_dates)} canonical evaluation days."
        )

    keep = valid_b_mask & valid_d_mask
    if member_col and member_col in df.columns:
        unique_members = sorted(df[member_col].dropna().unique().tolist(), key=str)
        mem_to_idx = {m: i for i, m in enumerate(unique_members)}
        m_indices = np.asarray([mem_to_idx.get(v, -1) for v in df[member_col].tolist()], dtype=np.int64)
        keep = keep & (m_indices >= 0)
        members_out = np.full(
            (len(unique_members), len(canon_basins), len(canon_dates)), np.nan, dtype=np.float32
        )
        members_out[m_indices[keep], b_indices[keep], d_indices[keep]] = vals_arr[keep]
        qsim_out = np.mean(members_out.astype(np.float64), axis=0).astype(np.float32)
        return qsim_out, members_out, matched_b, matched_d, []

    qsim_out = np.full((len(canon_basins), len(canon_dates)), np.nan, dtype=np.float32)
    qsim_out[b_indices[keep], d_indices[keep]] = vals_arr[keep]
    return qsim_out, None, matched_b, matched_d, []


def _infer_3d_axis_roles(
    shape: tuple[int, ...],
    raw_basins: np.ndarray | None,
    raw_dates: np.ndarray | None,
) -> tuple[str, str, str]:
    b_len = int(len(raw_basins)) if raw_basins is not None else EXPECTED_NUM_BASINS
    d_len = int(len(raw_dates)) if raw_dates is not None else EXPECTED_NUM_DAYS
    s0, s1, s2 = (int(shape[0]), int(shape[1]), int(shape[2]))
    if (s1, s2) == (d_len, b_len) and d_len != b_len:
        return ("member", "date", "basin")
    if (s1, s2) == (b_len, d_len):
        return ("member", "basin", "date")
    if (s0, s1) == (b_len, d_len):
        return ("basin", "date", "member")
    if (s0, s1) == (d_len, b_len):
        return ("date", "basin", "member")
    if (s0, s2) == (b_len, d_len):
        return ("basin", "member", "date")
    if (s0, s2) == (d_len, b_len):
        return ("date", "member", "basin")
    if s1 > s2 and d_len > b_len and s1 == d_len:
        return ("member", "date", "basin")
    return ("member", "basin", "date")


def _load_npz_to_canonical(
    path: Path,
    recipe: IngestionRecipe,
    canonical_basins: Sequence[str],
    canonical_dates: Sequence[str],
    *,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as npz:
        if not recipe.var or recipe.var not in npz.files:
            raise ValueError(f"Array key {recipe.var!r} not found in NPZ keys {npz.files}.")
        raw_arr = np.asarray(npz[recipe.var])
        raw_basins = (
            npz[recipe.basin_dim]
            if recipe.basin_dim and recipe.basin_dim in npz.files
            else None
        )
        raw_dates = (
            npz[recipe.time_dim]
            if recipe.time_dim and recipe.time_dim in npz.files
            else None
        )
        extra_members = (
            np.asarray(npz["qsim_members"])
            if "qsim_members" in npz.files and recipe.var != "qsim_members"
            else None
        )

    if raw_arr.ndim == 3:
        roles = _infer_3d_axis_roles(raw_arr.shape, raw_basins, raw_dates)
    elif raw_arr.ndim == 2:
        roles = ("basin", "date")
        if (
            raw_basins is not None
            and raw_dates is not None
            and raw_arr.shape == (len(raw_dates), len(raw_basins))
            and len(raw_dates) != len(raw_basins)
        ) or (
            raw_basins is None
            and raw_dates is None
            and raw_arr.shape == (EXPECTED_NUM_DAYS, EXPECTED_NUM_BASINS)
        ):
            roles = ("date", "basin")
    else:
        raise ValueError(f"NPZ array {recipe.var!r} must be 2D or 3D, got shape {raw_arr.shape}.")

    qsim, members, b_mask, d_mask, diags = _align_axes_and_slice(
        raw_arr, raw_basins, raw_dates, roles, canonical_basins, canonical_dates, recipe, strict=strict
    )
    if members is None and extra_members is not None and extra_members.ndim == 3:
        extra_roles = _infer_3d_axis_roles(extra_members.shape, raw_basins, raw_dates)
        _, members, _, _, _ = _align_axes_and_slice(
            extra_members,
            raw_basins,
            raw_dates,
            extra_roles,
            canonical_basins,
            canonical_dates,
            recipe,
            strict=strict,
        )
    return qsim, members, b_mask, d_mask, diags


def ingest_hydrograph(
    path: Path | str,
    recipe: IngestionRecipe,
    canonical_basins: Sequence[str],
    canonical_dates: Sequence[str],
    *,
    strict: bool = True,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Load a hydrograph archive via ``recipe`` and normalize it to canonical ``(531, 3652)``."""
    src_path = Path(path)
    fmt = detect_file_format(src_path)

    if fmt == "netcdf":
        with xr.open_dataset(src_path) as ds:
            qsim, members, b_mask, d_mask, diags = _load_xarray_to_canonical(
                ds, recipe, canonical_basins, canonical_dates, strict=strict
            )
    elif fmt == "zarr":
        with _open_zarr_dir(src_path) as ds:
            qsim, members, b_mask, d_mask, diags = _load_xarray_to_canonical(
                ds, recipe, canonical_basins, canonical_dates, strict=strict
            )
    elif fmt == "zarr.zip":
        base = work_dir if work_dir is not None else src_path.parent / f".ingest_{src_path.stem}"
        cleanup = work_dir is None
        extracted = safe_extract_zip(src_path, base)
        zarr_root = _locate_zarr_root(extracted)
        with _open_zarr_dir(zarr_root) as ds:
            qsim, members, b_mask, d_mask, diags = _load_xarray_to_canonical(
                ds, recipe, canonical_basins, canonical_dates, strict=strict
            )
        if cleanup and extracted.exists():
            shutil.rmtree(extracted)
    elif fmt == "csv":
        sep = _detect_csv_sep(src_path)
        df = pd.read_csv(src_path, sep=sep, low_memory=False)
        qsim, members, b_mask, d_mask, diags = _load_tabular_to_canonical(
            df, recipe, canonical_basins, canonical_dates, strict=strict
        )
    elif fmt == "parquet":
        df = pd.read_parquet(src_path)
        qsim, members, b_mask, d_mask, diags = _load_tabular_to_canonical(
            df, recipe, canonical_basins, canonical_dates, strict=strict
        )
    else:
        qsim, members, b_mask, d_mask, diags = _load_npz_to_canonical(
            src_path, recipe, canonical_basins, canonical_dates, strict=strict
        )

    matched_indices = np.where(d_mask)[0]
    date_start = str(canonical_dates[int(matched_indices[0])]) if len(matched_indices) else ""
    date_end = str(canonical_dates[int(matched_indices[-1])]) if len(matched_indices) else ""
    return {
        "qsim": qsim,
        "qsim_members": members,
        "matched_basin_mask": b_mask,
        "matched_date_mask": d_mask,
        "date_start": date_start,
        "date_end": date_end,
        "diagnostics": diags,
    }


def _select_representative_basin_indices(
    qobs: np.ndarray,
    qsim: np.ndarray,
    matched_basin_mask: np.ndarray,
) -> list[int]:
    """Choose 3 matched basins spanning dry/moderate/wet flow regimes deterministically."""
    candidates = [
        int(i)
        for i in np.where(matched_basin_mask)[0]
        if np.any(np.isfinite(qsim[i])) and np.any(np.isfinite(qobs[i]))
    ]
    if not candidates:
        return []
    mean_flows = [float(np.nanmean(qobs[i])) for i in candidates]
    order = [candidates[idx] for idx in np.argsort(mean_flows)]
    if len(order) <= 3:
        return order
    picks = [order[max(0, round(q * (len(order) - 1)))] for q in (0.15, 0.50, 0.85)]
    deduped: list[int] = []
    for p in picks:
        if p not in deduped:
            deduped.append(p)
    return deduped[:3]


def run_physical_sanity_checks(
    qsim: np.ndarray,
    qobs: np.ndarray,
    basins: Sequence[str],
    dates: Sequence[str],
    *,
    matched_basin_mask: np.ndarray | None = None,
    matched_date_mask: np.ndarray | None = None,
    gauge_names: Sequence[str] | None = None,
    initial_diagnostics: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate 531-basin alignment, NaNs, dates, unit plausibility, and sample hydrographs."""
    qsim_64 = np.asarray(qsim, dtype=np.float64)
    qobs_64 = np.asarray(qobs, dtype=np.float64)
    b_mask = (
        np.ones(len(basins), dtype=bool)
        if matched_basin_mask is None
        else np.asarray(matched_basin_mask, dtype=bool)
    )
    d_mask = (
        np.ones(len(dates), dtype=bool)
        if matched_date_mask is None
        else np.asarray(matched_date_mask, dtype=bool)
    )

    n_basins_matched = int(np.sum(b_mask))
    n_basins_expected = len(basins)
    obs_valid = (~np.isnan(qobs_64)) & (qobs_64 >= 0.0) & b_mask[:, None] & d_mask[None, :]
    nan_count = int(np.sum(obs_valid & np.isnan(qsim_64)))
    inf_count = int(np.sum(b_mask[:, None] & d_mask[None, :] & np.isinf(qsim_64)))

    matched_d_idx = np.where(d_mask)[0]
    date_start = str(dates[int(matched_d_idx[0])]) if len(matched_d_idx) else ""
    date_end = str(dates[int(matched_d_idx[-1])]) if len(matched_d_idx) else ""

    # Per-basin mean ratio median robustly detects cfs/L-s vs mm/day or m/day vs mm/day.
    basin_ratios: list[float] = []
    for i in np.where(b_mask)[0]:
        v = obs_valid[i] & np.isfinite(qsim_64[i])
        if np.any(v):
            obs_m = float(np.mean(qobs_64[i, v]))
            if obs_m > 1e-4:
                sim_m = float(np.mean(qsim_64[i, v]))
                basin_ratios.append(sim_m / obs_m)
    median_ratio = round(float(np.median(basin_ratios)), 4) if basin_ratios else None

    unit_warning: str | None = None
    diagnostics = list(initial_diagnostics or [])
    if median_ratio is not None:
        if median_ratio > 5.0:
            unit_warning = (
                f"Predicted/observed runoff ratio is {median_ratio:.2f}x (> 5.0x) — predictions "
                "may be in cfs or volume units rather than mm/day."
            )
            diagnostics.append(unit_warning)
        elif median_ratio < 0.2:
            unit_warning = (
                f"Predicted/observed runoff ratio is {median_ratio:.3f}x (< 0.20x) — predictions "
                "may be in m/day or m³/s rather than mm/day."
            )
            diagnostics.append(unit_warning)

    finite_mask = b_mask[:, None] & d_mask[None, :] & np.isfinite(qsim_64)
    neg_frac = float(np.mean(qsim_64[finite_mask] < 0.0)) if np.any(finite_mask) else 0.0
    if neg_frac > 0.05:
        diagnostics.append(
            f"{neg_frac * 100:.1f}% of predicted daily flows are negative (verify sign / lower bound)."
        )
    if n_basins_matched != n_basins_expected:
        diagnostics.append(
            f"Matched {n_basins_matched}/{n_basins_expected} canonical CAMELS-US basins."
        )
    if nan_count > 0:
        diagnostics.append(
            f"Detected {nan_count} NaN predictions on valid observation days (anti-masking rule requires 0)."
        )
    if inf_count > 0:
        diagnostics.append(f"Detected {inf_count} +/-Inf predictions.")
    if date_start != EVAL_START_DATE or date_end != EVAL_END_DATE or int(np.sum(d_mask)) != EXPECTED_NUM_DAYS:
        diagnostics.append(
            f"Matched {int(np.sum(d_mask))}/{EXPECTED_NUM_DAYS} evaluation days "
            f"({date_start or '?'} to {date_end or '?'} vs required {EVAL_START_DATE} to {EVAL_END_DATE})."
        )

    checks_passed = (
        n_basins_matched == EXPECTED_NUM_BASINS
        and int(np.sum(d_mask)) == EXPECTED_NUM_DAYS
        and date_start == EVAL_START_DATE
        and date_end == EVAL_END_DATE
        and nan_count == 0
        and inf_count == 0
        and unit_warning is None
        and neg_frac <= 0.50
    )

    sanity = {
        "n_basins_matched": n_basins_matched,
        "n_basins_expected": n_basins_expected,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "negative_flow_fraction": round(neg_frac, 4),
        "date_start": date_start,
        "date_end": date_end,
        "median_sim_obs_ratio": median_ratio,
        "unit_warning": unit_warning,
        "checks_passed": bool(checks_passed),
        "diagnostics": diagnostics,
    }

    # Build 3-basin water-year slice (1994-10-01..1995-09-30) + full-decade NSE via hydromip
    dates_str = [str(d) for d in dates]
    dates_ns = pd.to_datetime(dates_str).values.astype("datetime64[ns]")
    wy_idx = np.asarray(
        [
            i
            for i, d_str in enumerate(dates_str)
            if SAMPLE_WATER_YEAR_START <= d_str <= SAMPLE_WATER_YEAR_END
        ],
        dtype=np.intp,
    )
    if len(wy_idx) == 0:
        wy_idx = np.arange(min(365, len(dates_str)), dtype=np.intp)
    wy_dates = [dates_str[i] for i in wy_idx]

    sample_basins: list[dict[str, Any]] = []
    for b_idx in _select_representative_basin_indices(qobs_64, qsim_64, b_mask):
        obs_b = np.where(obs_valid[b_idx], qobs_64[b_idx], np.nan)
        sim_b = qsim_64[b_idx]
        obs_da = xr.DataArray(obs_b, coords={"date": dates_ns}, dims=("date",))
        sim_da = xr.DataArray(sim_b, coords={"date": dates_ns}, dims=("date",))
        nse_val = float("nan")
        if int(np.sum(np.isfinite(obs_b) & np.isfinite(sim_b))) >= MIN_VALID_OBS_DAYS:
            o_sub = obs_b[np.isfinite(obs_b)]
            sst = float(np.sum((o_sub - float(np.mean(o_sub))) ** 2))
            if sst > MIN_SST_THRESHOLD and np.any(np.isfinite(sim_b)):
                nse_val = float(canonical_nse(obs_da, sim_da))
        sample_basins.append(
            {
                "basin_id": str(basins[b_idx]),
                "gauge_name": str(gauge_names[b_idx]) if gauge_names is not None else str(basins[b_idx]),
                "dates": wy_dates,
                "qobs": [
                    None if not np.isfinite(v) else round(float(v), 3)
                    for v in qobs_64[b_idx, wy_idx]
                ],
                "qsim": [
                    None if not np.isfinite(v) else round(float(v), 3)
                    for v in qsim_64[b_idx, wy_idx]
                ],
                "nse": round(nse_val, 4) if np.isfinite(nse_val) else None,
            }
        )

    return sanity, sample_basins


def write_canonical_zarr_v2(
    qsim: np.ndarray,
    basins: Sequence[str],
    output_path: Path | str,
    metadata: Mapping[str, Any],
    qsim_members: np.ndarray | None = None,
    huc_02: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
) -> Path:
    """Write a validated, consolidated Zarr v2 CAMELS-US MIP hydrograph store."""
    basin_list = [normalize_basin_id(b) for b in basins]
    if len(basin_list) != EXPECTED_NUM_BASINS or len(set(basin_list)) != EXPECTED_NUM_BASINS:
        raise ValueError(f"Expected {EXPECTED_NUM_BASINS} unique canonical basins.")

    qsim_arr = np.asarray(qsim, dtype=np.float32)
    if qsim_arr.shape != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
        raise ValueError(
            f"qsim must have shape ({EXPECTED_NUM_BASINS}, {EXPECTED_NUM_DAYS}), got {qsim_arr.shape}."
        )

    sort_order = np.argsort(np.asarray(basin_list, dtype="<U8"))
    sorted_basins = np.asarray([basin_list[i] for i in sort_order], dtype="<U8")
    qsim_sorted = np.ascontiguousarray(qsim_arr[sort_order, :], dtype=np.float32)

    if dates is None:
        date_arr = pd.date_range(EVAL_START_DATE, EVAL_END_DATE, freq="D").values.astype("datetime64[ns]")
    else:
        date_arr = pd.to_datetime(list(dates)).values.astype("datetime64[ns]")
        if date_arr.shape != (EXPECTED_NUM_DAYS,):
            raise ValueError(f"Expected {EXPECTED_NUM_DAYS} dates, got {date_arr.shape}.")

    coords: dict[str, tuple[str, np.ndarray]] = {
        "basin": ("basin", sorted_basins),
        "date": ("date", date_arr),
    }
    if huc_02 is not None:
        huc_sorted = np.asarray([str(huc_02[i]).zfill(2) for i in sort_order], dtype="<U2")
        coords["huc_02"] = ("basin", huc_sorted)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]] = {
        "qsim": (
            ("basin", "date"),
            qsim_sorted,
            {"units": "mm/day", "long_name": "Simulated daily streamflow"},
        ),
    }
    encoding: dict[str, dict[str, object]] = {
        "qsim": {"chunks": (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS), "dtype": "float32", "_FillValue": np.nan},
        "basin": {"dtype": "<U8"},
    }
    if huc_02 is not None:
        encoding["huc_02"] = {"dtype": "<U2"}

    num_members = 1
    is_ensemble = bool(metadata.get("is_ensemble", False))
    if qsim_members is not None:
        mem_arr = np.asarray(qsim_members, dtype=np.float32)
        if mem_arr.ndim != 3 or mem_arr.shape[1:] != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
            raise ValueError(
                f"qsim_members must have shape (N, {EXPECTED_NUM_BASINS}, {EXPECTED_NUM_DAYS}), got {mem_arr.shape}."
            )
        num_members = int(mem_arr.shape[0])
        is_ensemble = num_members > 1
        mem_sorted = np.ascontiguousarray(mem_arr[:, sort_order, :], dtype=np.float32)
        data_vars["qsim_members"] = (
            ("member", "basin", "date"),
            mem_sorted,
            {"units": "mm/day", "long_name": "Simulated daily streamflow per ensemble member"},
        )
        coords["member"] = ("member", np.arange(num_members, dtype=np.int64))
        encoding["qsim_members"] = {
            "chunks": (1, EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS),
            "dtype": "float32",
            "_FillValue": np.nan,
        }

    attrs: dict[str, Any] = {
        **{str(k): _json_safe_value(v) for k, v in metadata.items() if v is not None},
        "is_ensemble": is_ensemble,
        "num_members": int(metadata.get("num_members", num_members)),
        "ensemble_reduction": str(metadata.get("ensemble_reduction", "mean")),
        "eval_period": f"{EVAL_START_DATE}/{EVAL_END_DATE}",
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(out), mode="w", consolidated=True, zarr_format=2, encoding=encoding)
    return out
