# Copyright 2026 HydroMIP Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FastAPI backend for hydromip.org (`camels-us-mip/frontend/`).

Serves the 3 x 4 = 12 CAMELS-531 Newman-period leaderboards, the model-family
registry (Table 2; the family count is always read from `table_2_registry.json`,
never hard-coded), canonical spatial/temporal split protocol metadata, static UI
assets, and the cloud-native Zarr / (531, 3652) community hydrograph scoring and
submission endpoints.

Scoring parity: `/api/score` and `/api/submit` score uploaded hydrographs with exactly
the same rules as `metrics.evaluator.calculate_model_leaderboard_metrics` (float64,
no clipping of negative predictions, `obs_valid = ~isnan(qobs) & (qobs >= 0)`,
evaluable basins `cnt >= 30 & SST > 1e-12`, per-basin
`metrics.metrics.calculate_all_metrics(..., resolution="1D", datetime_coord="date")`,
summaries as median/mean over finite per-basin values).
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_subdir
from hydromip.protocol import discover_mips, get_mip
from leaderboards.update_leaderboard import (
    load_all_leaderboards,
    rank_leaderboard_rows,
    upsert_leaderboard_entry,
)
from metrics.metrics import (
    calculate_all_metrics,
    get_available_metrics,
    nse as gauch_nse,
)

NEWMAN_START_DATE = "1989-10-01"
NEWMAN_END_DATE = "1999-09-30"
NEWMAN_N_DAYS = 3652
NEWMAN_N_BASINS = 531

# Evaluable-basin rule shared with `metrics/evaluator.py` (MIN_VALID_OBS_DAYS, MIN_SST_THRESHOLD).
MIN_VALID_OBS_DAYS = 30
MIN_SST_THRESHOLD = 1e-12

# Tolerances shared with `metrics/evaluator.py`: an ensemble store's `qsim` must equal the declared
# member reduction within `rtol=atol=1e-4`; a single-model store that ships its sibling seeds must
# have `qsim` equal to exactly one member within `rtol=atol=1e-5`.
MEMBER_MEAN_RTOL = 1e-4
MEMBER_MEAN_ATOL = 1e-4
SINGLE_MEMBER_RTOL = 1e-5
SINGLE_MEMBER_ATOL = 1e-5

# Member reductions accepted for the Zarr attribute `ensemble_reduction` (default "mean"), identical
# to `metrics.evaluator.SUPPORTED_ENSEMBLE_REDUCTIONS` + `PER_BASIN_ORACLE_REDUCTION`.
SUPPORTED_ENSEMBLE_REDUCTIONS: tuple[str, ...] = ("mean", "median")
PER_BASIN_ORACLE_REDUCTION = "per_basin_oracle"

# `ensemble_source` vocabulary (identical to the evaluator's, plus one value for members-only NumPy
# uploads that carry no stored `qsim` at all):
#   "mean_of_qsim_members" | "median_of_qsim_members"  stored qsim verified against the member reduction
#   "single_member"                                     is_ensemble=False, qsim equals exactly one member
#   "per_basin_oracle"                                  declared per-basin best-member bound (pass-through)
#   "stored_qsim_without_members"                       plain 2-D upload, nothing to verify against
#   "mean_of_members_no_stored_qsim"                    3-D .npz/.npy: ensemble = mean(members) computed here
ENSEMBLE_SOURCE_MEMBERS_ONLY = "mean_of_members_no_stored_qsim"

# 101-point quantile grid of the per-basin NSE empirical CDF served by `/api/entry/{id}/basins`.
NSE_CDF_QUANTILES: np.ndarray = np.linspace(0.0, 1.0, 101)

# Reference CDF curves shown in the basin inspector. These are real leaderboard rows (the
# Kratzert et al. (2021) 3-forcing and Daymet 10-model ensemble LSTMs); their curves are only
# ever computed from stored per-basin NSE values of those rows, never synthesized.
REFERENCE_CDF_ENTRY_IDS: dict[str, str] = {
    "ref_3f_lstm": "t4_05",
    "ref_1f_daymet": "t3_02",
}

GAUCH_2023_METRICS: tuple[str, ...] = tuple(get_available_metrics())

DEFAULT_BUCKET_URI = "gs://camels-us-mip"
DEFAULT_TARGETS_ZARR_URI = "gs://camels-us-mip/observations/newman_eval_targets.zarr"

FRONTEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = FRONTEND_ROOT / "static"

CANONICAL_LEADERBOARD_IDS: tuple[str, ...] = (
    "gauged__single_camels",
    "gauged__multi_camels",
    "gauged__lagged_streamflow",
    "gauged__freeform",
    "pub__single_camels",
    "pub__multi_camels",
    "pub__lagged_streamflow",
    "pub__freeform",
    "pur__single_camels",
    "pur__multi_camels",
    "pur__lagged_streamflow",
    "pur__freeform",
)

app = FastAPI(
    title="HydroMIP — CAMELS-US Model Intercomparison Project",
    description=(
        "Public 3x4 = 12 leaderboards, reproducible model-family registry (Table 2), "
        "protocol metadata, and deterministic 531x3652 hydrograph scoring engine "
        "for hydromip.org."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_EVAL_TARGETS_CACHE: dict[str, dict[str, Any]] = {}


def get_data_root() -> Path:
    """Resolve the repository or override data directory (`HYDROMIP_DATA_DIR` / `CAMELS_MIP_DATA_DIR`)."""
    for env_var in ("CAMELS_MIP_DATA_DIR", "HYDROMIP_DATA_DIR"):
        env_dir = os.environ.get(env_var, "").strip()
        if env_dir:
            root = Path(env_dir).resolve()
            if not root.exists():
                raise FileNotFoundError(f"{env_var} does not exist: {root}")
            return root
    return REPO_ROOT


def get_leaderboards_dir(mip_id: str = DEFAULT_MIP_ID) -> Path:
    """Return the directory containing `tables_3_to_14.json`, `table_2_registry.json`, and `contributions_index.json`."""
    return resolve_mip_subdir(
        get_data_root(),
        "leaderboards",
        mip_id=mip_id,
        marker_filename="tables_3_to_14.json",
    )


def get_splits_dir(mip_id: str = DEFAULT_MIP_ID) -> Path:
    """Return the directory containing `splits_and_regimes.json`."""
    return resolve_mip_subdir(
        get_data_root(),
        "splits",
        mip_id=mip_id,
        marker_filename="splits_and_regimes.json",
    )


def get_figures_dir(mip_id: str = DEFAULT_MIP_ID) -> Path:
    """Return the directory containing generated figures for `mip_id`."""
    return resolve_mip_subdir(
        get_data_root(),
        "figures",
        mip_id=mip_id,
        marker_filename="figures_manifest.json",
    )


def get_submissions_dir() -> Path:
    """Resolve the writable root directory where community submission Zarr stores are persisted.

    `CAMELS_MIP_SUBMISSIONS_DIR` takes precedence; otherwise `<data_root>/submissions` is used.
    Raises HTTP 503 when the directory cannot be created or written, so that `/api/submit`
    refuses a submission instead of recording a hydrograph path that does not exist.
    """
    env_dir = os.environ.get("CAMELS_MIP_SUBMISSIONS_DIR", "").strip()
    root = Path(env_dir).resolve() if env_dir else (get_data_root() / "submissions").resolve()

    if not root.exists():
        nearest_parent = root
        while not nearest_parent.exists() and nearest_parent.parent != nearest_parent:
            nearest_parent = nearest_parent.parent
        if not os.access(nearest_parent, os.W_OK):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Submission storage is not configured: cannot create '{root}'. "
                    "Set CAMELS_MIP_SUBMISSIONS_DIR to a writable directory."
                ),
            )
        root.mkdir(parents=True, exist_ok=True)

    if not root.is_dir() or not os.access(root, os.W_OK):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Submission storage is not writable: '{root}'. "
                "Set CAMELS_MIP_SUBMISSIONS_DIR to a writable directory."
            ),
        )
    return root


def _read_json_file(path: Path) -> dict[str, Any]:
    """Validate existence and load a JSON dictionary from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON dictionary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_tables_data() -> dict[str, Any]:
    """Load all 12 per-leaderboard JSON files (`leaderboards/<leaderboard_id>.json`) from `leaderboards/`."""
    lb_dir = get_leaderboards_dir()
    return load_all_leaderboards(lb_dir)


def load_registry_data() -> dict[str, Any]:
    """Load `table_2_registry.json` from `leaderboards/`."""
    lb_dir = get_leaderboards_dir()
    data = _read_json_file(lb_dir / "table_2_registry.json")
    if "families" not in data:
        raise KeyError("table_2_registry.json must contain 'families' key.")
    return data


def load_contributions_index() -> dict[str, Any]:
    """Load `contributions_index.json` from `leaderboards/`."""
    lb_dir = get_leaderboards_dir()
    data = _read_json_file(lb_dir / "contributions_index.json")
    if "entries" not in data:
        raise KeyError("contributions_index.json must contain 'entries' key.")
    return data


def load_splits_protocol() -> dict[str, Any]:
    """Load `splits_and_regimes.json` from `splits/`."""
    splits_dir = get_splits_dir()
    data = _read_json_file(splits_dir / "splits_and_regimes.json")
    required_keys = (
        "protocol_name",
        "num_basins",
        "training_period",
        "evaluation_period",
        "spatial_regimes",
        "basin_to_huc02",
        "pub_12fold_test_basins",
        "pur_18huc02_test_basins",
    )
    for key in required_keys:
        if key not in data:
            raise KeyError(f"splits_and_regimes.json missing required key '{key}'.")
    return data


def resolve_zarr_uri(uri: str = DEFAULT_TARGETS_ZARR_URI) -> str | Path:
    """Resolve a canonical `gs://camels-us-mip/...` Zarr URI or configured local/test store."""
    env_targets = os.environ.get("CAMELS_MIP_TARGETS_ZARR", "").strip()
    if env_targets and uri == DEFAULT_TARGETS_ZARR_URI:
        return env_targets

    env_gcs_root = os.environ.get("CAMELS_MIP_GCS_ROOT", "").strip()
    prefix = f"{DEFAULT_BUCKET_URI}/"
    if env_gcs_root and uri.startswith(prefix):
        rel_path = uri[len(prefix) :]
        return Path(env_gcs_root).resolve() / rel_path

    data_root_env = os.environ.get("CAMELS_MIP_DATA_DIR", "").strip()
    if data_root_env and uri.startswith(prefix):
        rel_path = uri[len(prefix) :]
        local_candidate = Path(data_root_env).resolve() / rel_path
        if local_candidate.exists():
            return local_candidate

    return uri


def load_eval_targets(targets_uri: str = DEFAULT_TARGETS_ZARR_URI) -> dict[str, Any]:
    """Load and validate canonical Newman evaluation targets from `newman_eval_targets.zarr`."""
    resolved = resolve_zarr_uri(targets_uri)
    cache_key = str(resolved)
    if cache_key in _EVAL_TARGETS_CACHE:
        return _EVAL_TARGETS_CACHE[cache_key]

    if isinstance(resolved, Path) or (isinstance(resolved, str) and not resolved.startswith("gs://")):
        store_path = Path(resolved)
        if not store_path.exists():
            raise FileNotFoundError(f"Canonical evaluation target Zarr store not found: {store_path}")
        consolidated = (store_path / ".zmetadata").exists()
        ds = xr.open_zarr(store_path, consolidated=consolidated)
    else:
        ds = xr.open_zarr(resolved, consolidated=True)

    if "qobs" not in ds.data_vars:
        raise KeyError("newman_eval_targets.zarr missing required 'qobs' data variable.")
    if "basin" not in ds.coords or "date" not in ds.coords:
        raise KeyError("newman_eval_targets.zarr missing required 'basin' or 'date' coordinates.")
    if "huc_02" not in ds.coords and "huc_02" not in ds.data_vars:
        raise KeyError("newman_eval_targets.zarr missing required 'huc_02' coordinate.")

    qobs_da = ds["qobs"]
    if tuple(qobs_da.dims) != ("basin", "date") or qobs_da.shape != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
        raise ValueError(
            f"Expected newman_eval_targets.zarr 'qobs' dims ('basin', 'date') and shape "
            f"({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS}), got dims {qobs_da.dims} and shape {qobs_da.shape}."
        )

    basins = [str(b).zfill(8) for b in ds.coords["basin"].values]
    if basins != sorted(basins) or len(basins) != NEWMAN_N_BASINS:
        raise ValueError("newman_eval_targets.zarr 'basin' coordinate must be 531 sorted 8-digit USGS IDs.")

    dates_dt = pd.to_datetime(ds.coords["date"].values)
    date_strings = dates_dt.strftime("%Y-%m-%d").tolist()
    expected_dates = pd.date_range(NEWMAN_START_DATE, NEWMAN_END_DATE, freq="D").strftime("%Y-%m-%d").tolist()
    if date_strings != expected_dates:
        raise ValueError(
            f"newman_eval_targets.zarr 'date' coordinate does not match {NEWMAN_START_DATE}..{NEWMAN_END_DATE}."
        )

    huc_02 = [str(h).zfill(2) for h in ds["huc_02"].values]
    # float64, exactly like `metrics.evaluator._validate_targets_dataset`, so that `/api/score`
    # and the cloud evaluator see bit-identical observations.
    qobs = np.asarray(qobs_da.values, dtype=np.float64)

    targets = {
        "uri": targets_uri,
        "basins": basins,
        "dates": date_strings,
        "dates_np": dates_dt.values.astype("datetime64[D]"),
        "huc_02": huc_02,
        "qobs": qobs,
    }
    _EVAL_TARGETS_CACHE[cache_key] = targets
    return targets


def normalize_github_url(raw: str | None) -> str:
    """Normalize a GitHub repository slug or URL."""
    if not raw:
        return ""
    s = raw.strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s.rstrip("/")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", s):
        return f"https://github.com/{s}"
    return s


def normalize_zenodo(raw: str | None) -> tuple[str, str]:
    """Return `(zenodo_doi, zenodo_url)` from either a DOI or a Zenodo URL."""
    if not raw:
        return "", ""
    s = raw.strip()
    if not s:
        return "", ""
    m_doi = re.search(r"(10\.5281/zenodo\.\d+)", s, re.IGNORECASE)
    if m_doi:
        doi = m_doi.group(1).lower()
        rec_id = doi.split(".")[-1]
        return doi, f"https://zenodo.org/records/{rec_id}"
    m_rec = re.search(r"zenodo\.org/(?:records?|record|doi)/(\d+)", s, re.IGNORECASE)
    if m_rec:
        rec_id = m_rec.group(1)
        return f"10.5281/zenodo.{rec_id}", f"https://zenodo.org/records/{rec_id}"
    if s.isdigit():
        return f"10.5281/zenodo.{s}", f"https://zenodo.org/records/{s}"
    if s.startswith("http://") or s.startswith("https://"):
        return s, s
    return s, f"https://doi.org/{s}"


def _clean_float(val: Any, digits: int = 4) -> float | None:
    """Round a finite numeric value or return None for NaN/Inf/None."""
    if val is None or isinstance(val, bool):
        return None
    if not isinstance(val, (int, float, np.number)):
        return None
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def _finite_or_none(val: Any) -> float | None:
    """Return `float(val)` when it is a finite number, otherwise `None` (no sentinel values)."""
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        return None
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def leaderboards_updated_at(lb_dir: Path | None = None) -> str:
    """Return the UTC ISO-8601 timestamp of the most recently modified of the 12 leaderboard JSON files."""
    directory = lb_dir if lb_dir is not None else get_leaderboards_dir()
    paths = [directory / f"{lb_id}.json" for lb_id in CANONICAL_LEADERBOARD_IDS]
    mtimes = [p.stat().st_mtime for p in paths if p.is_file()]
    if not mtimes:
        raise FileNotFoundError(f"No per-leaderboard JSON files found under {directory}")
    return dt.datetime.fromtimestamp(max(mtimes), tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _per_basin_nse_array(raw: Any) -> np.ndarray | None:
    """Coerce a stored `per_basin_nse` list (531 x float|None|NaN) into a float64 vector, else None."""
    if not isinstance(raw, list) or len(raw) != NEWMAN_N_BASINS:
        return None
    arr = np.array([np.nan if _finite_or_none(v) is None else float(v) for v in raw], dtype=np.float64)
    if not np.isfinite(arr).any():
        return None
    return arr


def _empirical_cdf_101(values: np.ndarray) -> list[float]:
    """Return the 101-point empirical CDF (`np.quantile` at `NSE_CDF_QUANTILES`) of the finite values."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return [round(float(v), 4) for v in np.quantile(finite, NSE_CDF_QUANTILES)]


def _attach_flat_gauch_metrics(row: dict[str, Any]) -> None:
    """Attach flat numeric fields for all 14 Gauch et al. (2023) metrics so every column is sortable."""
    gauch = row.get("gauch_2023_metrics")
    g = gauch if isinstance(gauch, dict) else {}

    def _gmed(metric_key: str) -> float | None:
        m_obj = g.get(metric_key)
        if isinstance(m_obj, dict):
            return _finite_or_none(m_obj.get("median"))
        return None

    def _gmean(metric_key: str) -> float | None:
        m_obj = g.get(metric_key)
        if isinstance(m_obj, dict):
            return _finite_or_none(m_obj.get("mean"))
        return None

    nse_med = _gmed("NSE")
    if nse_med is None:
        nse_med = _finite_or_none(row.get("primary_ranking_nse"))
    row["gauch_nse"] = nse_med
    if _finite_or_none(row.get("mean_nse")) is None and _gmean("NSE") is not None:
        row["mean_nse"] = _gmean("NSE")

    kge_med = _gmed("KGE")
    if kge_med is None:
        kge_med = _finite_or_none(row.get("med_kge"))
    row["gauch_kge"] = kge_med
    if _finite_or_none(row.get("med_kge")) is None and kge_med is not None:
        row["med_kge"] = kge_med

    row["gauch_log_nse"] = _gmed("logNSE")
    row["gauch_log_kge"] = _gmed("logKGE")
    row["gauch_pearson_r"] = _gmed("Pearson-r")
    row["gauch_alpha_nse"] = _gmed("Alpha-NSE")
    row["gauch_beta_nse"] = _gmed("Beta-NSE")
    row["gauch_beta_kge"] = _gmed("Beta-KGE")
    row["gauch_mse"] = _gmed("MSE")
    row["gauch_rmse"] = _gmed("RMSE")
    row["gauch_fhv"] = _gmed("FHV")
    if _finite_or_none(row.get("med_fhv_abs")) is None and row["gauch_fhv"] is not None:
        row["med_fhv_abs"] = round(abs(float(row["gauch_fhv"])), 2)
    row["gauch_fms"] = _gmed("FMS")
    row["gauch_flv"] = _gmed("FLV")
    row["gauch_peak_timing"] = _gmed("Peak-Timing")


def build_enriched_leaderboards(
    spatial_regime: str | None = None,
    forcing_regime: str | None = None,
    paradigm: str | None = None,
    category: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Assemble the 12 leaderboards, the Table 2 model families, and contributions metadata.

    Rows are ordered with `leaderboards.update_leaderboard.rank_leaderboard_rows` (the same
    ranking used when the JSON files are written); rows without a numeric
    `primary_ranking_nse` sort last and are reported as `None`, never as a sentinel value.
    Heavy per-basin vectors (`per_basin_nse`) are stripped from the returned rows and
    replaced by the boolean `has_per_basin_nse`; `/api/entry/{entry_id}/basins` serves them.
    """
    tables_data = load_tables_data()
    reg_data = load_registry_data()
    contrib_data = load_contributions_index()

    families = copy.deepcopy(reg_data["families"])
    families_by_id = {f["family_id"]: f for f in families}
    contrib_entries = contrib_data.get("entries", {})
    q = (search or "").strip().lower()

    boards_out: list[dict[str, Any]] = []
    total_entries_count = 0
    all_entries_flat: list[dict[str, Any]] = []

    for lb in tables_data["leaderboards"]:
        for r in lb.get("rows", []):
            all_entries_flat.append(r)

        if spatial_regime and lb["spatial_regime"] != spatial_regime:
            continue
        if forcing_regime and lb["forcing_regime"] != forcing_regime:
            continue

        raw_rows = rank_leaderboard_rows(lb.get("rows", []))

        enriched_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            fam = families_by_id.get(row.get("family_id", ""), {})
            contrib = contrib_entries.get(row.get("entry_id", ""), {})
            row["primary_ranking_nse"] = _finite_or_none(row.get("primary_ranking_nse"))
            row["has_per_basin_nse"] = _per_basin_nse_array(row.pop("per_basin_nse", None)) is not None
            row["family_name"] = row.get("family_name") or fam.get("name", row.get("family_id", ""))
            row["paradigm"] = row.get("paradigm") or fam.get("paradigm", "")
            row["citation"] = row.get("citation") or fam.get("citation", "")
            row["paper_url"] = row.get("paper_url") or fam.get("paper_url", "")
            row["github_url"] = row.get("github_url") or contrib.get("code_url") or fam.get("github_url", "")
            row["zenodo_doi"] = row.get("zenodo_doi") or fam.get("zenodo_doi", "")
            row["zenodo_url"] = row.get("zenodo_url") or contrib.get("data_url") or fam.get("zenodo_url", "")
            row["hydroshare_url"] = row.get("hydroshare_url") or fam.get("hydroshare_url", "")
            row["family_description"] = row.get("family_description") or fam.get("description", "")

            data_url = (
                row.get("data_archive_url")
                or row["zenodo_url"]
                or row["hydroshare_url"]
                or (f"https://doi.org/{row['zenodo_doi']}" if row["zenodo_doi"] else "")
            )
            row["data_archive_url"] = data_url
            row["code_url"] = row.get("code_url") or row["github_url"] or ""
            cat_str = str(row.get("category") or row.get("code_provenance") or "").strip().lower()
            row["is_open_source"] = bool(row["code_url"])
            row["is_closed_source"] = (not row["is_open_source"]) and (
                cat_str.startswith("contrib") or cat_str == "community" or bool(row.get("is_user_submission"))
            )
            _attach_flat_gauch_metrics(row)

            if paradigm and paradigm.lower() not in str(row.get("paradigm", "")).lower():
                continue
            if category and category.lower() not in str(row.get("category", "")).lower():
                continue
            if q:
                haystack = " ".join(
                    [
                        str(row.get("model_name", "")),
                        str(row.get("family_name", "")),
                        str(row.get("paradigm", "")),
                        str(row.get("citation", "")),
                        str(row.get("forcing", "")),
                        str(row.get("zenodo_doi", "")),
                        str(row.get("github_url", "")),
                    ]
                ).lower()
                if q not in haystack:
                    continue

            enriched_rows.append(row)

        total_entries_count += len(enriched_rows)
        top_entry = enriched_rows[0] if enriched_rows else None
        top_nse = top_entry["primary_ranking_nse"] if top_entry is not None else None
        boards_out.append(
            {
                **{k: v for k, v in lb.items() if k != "rows"},
                "n_entries": len(enriched_rows),
                "top_model": top_entry["model_name"] if top_entry else None,
                "top_nse": top_nse,
                "rows": enriched_rows,
            }
        )

    families_out: list[dict[str, Any]] = []
    for fam in families:
        f_copy = copy.deepcopy(fam)
        member_entries = [e for e in all_entries_flat if e.get("family_id") == fam["family_id"]]
        f_copy["n_entries"] = len(member_entries)
        f_copy["leaderboards_covered"] = sorted(list({e["leaderboard_id"] for e in member_entries}))
        valid_nses = [
            v for v in (_finite_or_none(e.get("primary_ranking_nse")) for e in member_entries) if v is not None
        ]
        f_copy["best_nse"] = max(valid_nses) if valid_nses else fam.get("best_nse")
        families_out.append(f_copy)

    protocol = copy.deepcopy(tables_data["protocol"])
    return {
        "updated_at": leaderboards_updated_at(),
        "protocol": protocol,
        "total_entries": total_entries_count,
        "total_families": len(families_out),
        "total_contributions": int(contrib_data.get("total_contributions", len(contrib_entries))),
        "leaderboards": boards_out,
        "families": families_out,
    }


def _find_zarr_root_in_directory(extracted_dir: Path) -> Path | None:
    """Locate a valid Zarr group root inside an extracted zip directory."""
    markers = (".zgroup", ".zmetadata", "zarr.json")
    if any((extracted_dir / m).exists() for m in markers) or (extracted_dir / "qsim").is_dir():
        return extracted_dir
    for child in sorted(extracted_dir.iterdir()):
        if child.is_dir():
            if any((child / m).exists() for m in markers) or (child / "qsim").is_dir():
                return child
    return None


@dataclass(frozen=True)
class ParsedHydrograph:
    """A validated hydrograph upload, split into the hydrograph that is scored and its members.

    Attributes:
        ensemble: The `(531, 3652)` float64 hydrograph that is scored on the leaderboard. For a
            Zarr upload this is always the stored `qsim` exactly as submitted (never recomputed);
            for a 3-D `.npz`/`.npy` upload (members only) it is `mean(members, axis=0)`.
        members: Optional `(N_members, 531, 3652)` float64 per-member hydrographs, or `None`.
        has_stored_qsim: True when `ensemble` is the submitted array (Zarr `qsim`, 2-D NumPy),
            False when it had to be derived from a members-only NumPy upload.
        declared_is_ensemble: Value of the optional Zarr global attribute `is_ensemble`
            (authoritative when present, like in `metrics/evaluator.py`), else `None`.
        ensemble_reduction: Declared member reduction (Zarr attribute `ensemble_reduction`,
            default `"mean"`): `"mean"`, `"median"`, or `"per_basin_oracle"`.
        meta: JSON-serialisable file metadata (`filename`, `size_bytes`, `sha256`, `format`, ...).
    """

    ensemble: np.ndarray
    members: np.ndarray | None
    has_stored_qsim: bool
    declared_is_ensemble: bool | None
    ensemble_reduction: str
    meta: dict[str, Any]


def _parse_is_ensemble_attr(raw: Any, filename: str) -> bool | None:
    """Coerce the optional Zarr `is_ensemble` attribute into a bool (HTTP 400 if not boolean-like)."""
    if raw is None:
        return None
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, (int, np.integer)) and int(raw) in (0, 1):
        return bool(raw)
    if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        return raw.strip().lower() == "true"
    raise HTTPException(
        status_code=400,
        detail=f"Zarr attribute 'is_ensemble' in '{filename}' must be boolean, got {raw!r}.",
    )


def _parse_ensemble_reduction_attr(raw: Any, filename: str) -> str:
    """Read the declared member reduction (`ensemble_reduction` attribute; default `"mean"`), HTTP 400 if unsupported."""
    if raw is None:
        return "mean"
    if not isinstance(raw, str):
        raise HTTPException(
            status_code=400,
            detail=f"Zarr attribute 'ensemble_reduction' in '{filename}' must be a string, got {raw!r}.",
        )
    reduction = raw.strip().lower()
    allowed = SUPPORTED_ENSEMBLE_REDUCTIONS + (PER_BASIN_ORACLE_REDUCTION,)
    if reduction not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported 'ensemble_reduction' {raw!r} in '{filename}'; expected one of {allowed}.",
        )
    return reduction


def resolve_member_info(parsed: ParsedHydrograph) -> dict[str, Any]:
    """Resolve ensemble/single-seed bookkeeping exactly like `metrics.evaluator._check_members_against_qsim`.

    - No members: `is_ensemble` follows the attribute (default False);
      `ensemble_source="stored_qsim_without_members"` (or `"mean_of_members_no_stored_qsim"`
      for a members-only NumPy upload, whose members are kept).
    - `is_ensemble=True` (attribute, or default when N > 1): `qsim` must equal the declared
      reduction of the members (`mean`/`median`, `rtol=atol=1e-4` on jointly-finite cells) or
      HTTP 400; `ensemble_reduction="per_basin_oracle"` with `is_ensemble=True` is HTTP 400.
    - `is_ensemble=False` with members: `ensemble_reduction="per_basin_oracle"` is passed
      through (`ensemble_source="per_basin_oracle"`); otherwise `qsim` must equal exactly one
      member (`rtol=atol=1e-5`) and that `single_seed_member_index` is reported, else HTTP 400.

    Returns `is_ensemble`, `num_members`, `ensemble_source`, `single_seed_member_index`,
    `ensemble_reduction`.
    """
    filename = str(parsed.meta.get("filename", "upload"))
    members = parsed.members
    reduction = parsed.ensemble_reduction

    if members is None:
        is_ensemble = bool(parsed.declared_is_ensemble) if parsed.declared_is_ensemble is not None else False
        return {
            "is_ensemble": is_ensemble,
            "num_members": 1,
            "ensemble_source": "stored_qsim_without_members",
            "single_seed_member_index": None,
            "ensemble_reduction": reduction,
        }

    n_members = int(members.shape[0])
    if not parsed.has_stored_qsim:
        return {
            "is_ensemble": n_members > 1,
            "num_members": n_members,
            "ensemble_source": ENSEMBLE_SOURCE_MEMBERS_ONLY,
            "single_seed_member_index": None,
            "ensemble_reduction": "mean",
        }

    sim = parsed.ensemble
    is_ensemble = parsed.declared_is_ensemble if parsed.declared_is_ensemble is not None else n_members > 1

    if is_ensemble:
        if reduction == PER_BASIN_ORACLE_REDUCTION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{filename}' declares is_ensemble=True together with "
                    f"ensemble_reduction='{PER_BASIN_ORACLE_REDUCTION}'; an oracle selection is "
                    f"not an ensemble mean and must be stored with is_ensemble=False."
                ),
            )
        reduced = np.mean(members, axis=0) if reduction == "mean" else np.median(members, axis=0)
        comparable = np.isfinite(sim) & np.isfinite(reduced)
        if not np.allclose(sim[comparable], reduced[comparable], rtol=MEMBER_MEAN_RTOL, atol=MEMBER_MEAN_ATOL):
            max_abs = float(np.max(np.abs(sim[comparable] - reduced[comparable])))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Ensemble consistency violation in '{filename}': stored 'qsim' is not the "
                    f"{reduction} of 'qsim_members' (max |qsim - {reduction}(members)| = {max_abs:.6g}, "
                    f"n_members = {n_members}; tolerance rtol={MEMBER_MEAN_RTOL}, atol={MEMBER_MEAN_ATOL}). "
                    f"Store the unmodified member {reduction} as 'qsim' (no clipping, weighting, or member selection)."
                ),
            )
        return {
            "is_ensemble": True,
            "num_members": n_members,
            "ensemble_source": f"{reduction}_of_qsim_members",
            "single_seed_member_index": None,
            "ensemble_reduction": reduction,
        }

    if reduction == PER_BASIN_ORACLE_REDUCTION:
        return {
            "is_ensemble": False,
            "num_members": n_members,
            "ensemble_source": PER_BASIN_ORACLE_REDUCTION,
            "single_seed_member_index": None,
            "ensemble_reduction": reduction,
        }

    matches: list[int] = []
    for i in range(n_members):
        comparable = np.isfinite(sim) & np.isfinite(members[i])
        if np.allclose(sim[comparable], members[i][comparable], rtol=SINGLE_MEMBER_RTOL, atol=SINGLE_MEMBER_ATOL):
            matches.append(int(i))
    if len(matches) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Single-model consistency violation in '{filename}': attribute 'is_ensemble' is False "
                f"but stored 'qsim' matches {len(matches)} of the {n_members} 'qsim_members' "
                f"(expected exactly one; tolerance rtol={SINGLE_MEMBER_RTOL}, atol={SINGLE_MEMBER_ATOL})."
            ),
        )
    return {
        "is_ensemble": False,
        "num_members": n_members,
        "ensemble_source": "single_member",
        "single_seed_member_index": matches[0],
        "ensemble_reduction": reduction,
    }


def parse_uploaded_hydrograph(
    filename: str,
    raw_bytes: bytes,
    eval_targets: dict[str, Any],
) -> ParsedHydrograph:
    """Validate and parse an uploaded coordinate-labeled `.zarr` (zipped Zarr store) or keyed `.npz` file.

    Enforces exact `(531, 3652)` basin-by-day dimensions (or `(N_members, 531, 3652)`) and
    canonical `basin` and `date` coordinate alignment without `try`/`except` error swallowing.
    Unlabeled `.npy` uploads are rejected because they carry no coordinate metadata and cannot
    detect silent basin-order or date misalignment.

    - Zarr with `qsim` only: the stored `qsim` is scored (`ensemble_source="stored_qsim_without_members"`).
    - Zarr with `qsim` and `qsim_members` (`("member", "basin", "date")`): the stored `qsim` is
      scored exactly as submitted; `resolve_member_info` then enforces the same consistency
      rules as `metrics.evaluator._check_members_against_qsim` (`is_ensemble` and
      `ensemble_reduction` attributes honoured; mismatches are HTTP 400).
    - Keyed `.npz`: must include `qsim`, `basin` (or `basins`), and `date` (or `dates`) matching
      the canonical 531 USGS IDs and 1989-10-01..1999-09-30 dates. A 2-D `qsim` is scored as
      submitted (optionally paired with 3-D `qsim_members`); a 3-D `qsim` (members only) scores
      `mean(members, axis=0)` (`ensemble_source="mean_of_members_no_stored_qsim"`).
    """
    if not filename or len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded hydrograph file is empty.")

    fname = filename.lower()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    meta: dict[str, Any] = {
        "filename": filename,
        "size_bytes": len(raw_bytes),
        "sha256": sha256,
    }
    canonical_basins: list[str] = eval_targets["basins"]
    canonical_dates: list[str] = eval_targets["dates"]

    if fname.endswith(".zarr") or fname.endswith(".zarr.zip") or fname.endswith(".zip"):
        if not zipfile.is_zipfile(io.BytesIO(raw_bytes)):
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded Zarr file '{filename}' is not a valid zip archive.",
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
                for member_name in zf.namelist():
                    if member_name.startswith("/") or ".." in Path(member_name).parts:
                        raise HTTPException(status_code=400, detail="Unsafe path in uploaded zip archive.")
                zf.extractall(tmp_path)

            zarr_root = _find_zarr_root_in_directory(tmp_path)
            if zarr_root is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded archive '{filename}' does not contain a valid Zarr store.",
                )

            consolidated = (zarr_root / ".zmetadata").exists()
            ds = xr.open_zarr(zarr_root, consolidated=consolidated)

            if "qsim" not in ds.data_vars:
                raise HTTPException(
                    status_code=400,
                    detail=f"Zarr store in '{filename}' is missing required 'qsim' data variable. Found: {list(ds.data_vars)}",
                )
            qsim_da = ds["qsim"]
            if tuple(qsim_da.dims) != ("basin", "date") or qsim_da.shape != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid Zarr 'qsim' dims {qsim_da.dims} or shape {qsim_da.shape}; "
                        f"expected dims ('basin', 'date') and shape ({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS})."
                    ),
                )
            if "basin" not in ds.coords or "date" not in ds.coords:
                raise HTTPException(
                    status_code=400,
                    detail="Zarr store must contain 'basin' and 'date' coordinates.",
                )
            store_basins = [str(b).zfill(8) for b in ds.coords["basin"].values]
            if store_basins != canonical_basins:
                raise HTTPException(
                    status_code=400,
                    detail="Zarr 'basin' coordinate does not match the canonical sorted 531 USGS gauge IDs.",
                )
            store_dates = pd.to_datetime(ds.coords["date"].values).strftime("%Y-%m-%d").tolist()
            if store_dates != canonical_dates:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Zarr 'date' coordinate does not match the canonical Newman evaluation window "
                        f"({NEWMAN_START_DATE} to {NEWMAN_END_DATE})."
                    ),
                )

            qsim_2d = np.asarray(qsim_da.values, dtype=np.float64)
            declared_is_ensemble = _parse_is_ensemble_attr(ds.attrs.get("is_ensemble"), filename)
            ensemble_reduction = _parse_ensemble_reduction_attr(ds.attrs.get("ensemble_reduction"), filename)
            members_3d: np.ndarray | None = None
            if "qsim_members" in ds.data_vars:
                members_da = ds["qsim_members"]
                if tuple(members_da.dims) != ("member", "basin", "date"):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid 'qsim_members' dims {members_da.dims}; "
                            f"expected ('member', 'basin', 'date')."
                        ),
                    )
                if members_da.shape[0] < 1 or members_da.shape[1:] != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'qsim_members' shape {members_da.shape}; expected (N_members, 531, 3652).",
                    )
                members_3d = np.asarray(members_da.values, dtype=np.float64)

            meta["format"] = "zarr"
            meta["primary_key"] = "qsim"
            meta["has_qsim_members"] = members_3d is not None
            meta["zarr_attrs"] = {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float, bool))}
            return ParsedHydrograph(
                ensemble=qsim_2d,
                members=members_3d,
                has_stored_qsim=True,
                declared_is_ensemble=declared_is_ensemble,
                ensemble_reduction=ensemble_reduction,
                meta=meta,
            )

    if fname.endswith(".npz"):
        if not zipfile.is_zipfile(io.BytesIO(raw_bytes)):
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file '{filename}' is not a valid .npz zip archive.",
            )
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as npz:
            keys = list(npz.files)
            meta["npz_keys"] = keys
            if "qsim" not in keys:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required 'qsim' array key in '{filename}'. Found keys: {keys}",
                )
            if "basin" not in keys and "basins" not in keys:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Missing required 'basin' (or 'basins') coordinate array in '{filename}'. "
                        "Coordinate-labeled uploads are required to verify canonical 531-basin alignment."
                    ),
                )
            bk = "basin" if "basin" in keys else "basins"
            file_basins = [str(b).zfill(8) for b in npz[bk]]
            if file_basins != canonical_basins:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded 'basin' coordinate does not match canonical sorted 531 USGS gauge IDs.",
                )
            if "date" not in keys and "dates" not in keys:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Missing required 'date' (or 'dates') coordinate array in '{filename}'. "
                        "Coordinate-labeled uploads are required to verify canonical Newman evaluation dates."
                    ),
                )
            dk = "date" if "date" in keys else "dates"
            file_dates = pd.to_datetime(npz[dk]).strftime("%Y-%m-%d").tolist()
            if file_dates != canonical_dates:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Uploaded 'date' coordinate does not match canonical Newman evaluation dates "
                        f"({NEWMAN_START_DATE} to {NEWMAN_END_DATE})."
                    ),
                )
            arr = np.asarray(npz["qsim"], dtype=np.float64)
            meta["format"] = "npz"
            meta["primary_key"] = "qsim"
            if "qsim_members" in keys:
                if arr.ndim != 2 or arr.shape != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid 'qsim' shape {arr.shape} alongside 'qsim_members'; "
                            f"expected 2D ({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS})."
                        ),
                    )
                members_arr = np.asarray(npz["qsim_members"], dtype=np.float64)
                if members_arr.ndim != 3 or members_arr.shape[0] < 1 or members_arr.shape[1:] != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'qsim_members' shape {members_arr.shape}; expected (N_members, {NEWMAN_N_BASINS}, {NEWMAN_N_DAYS}).",
                    )
                meta["has_qsim_members"] = True
                return ParsedHydrograph(
                    ensemble=arr,
                    members=members_arr,
                    has_stored_qsim=True,
                    declared_is_ensemble=None,
                    ensemble_reduction="mean",
                    meta=meta,
                )
            return _parsed_from_numpy_array(arr, meta)

    if fname.endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Bare .npy uploads ('{filename}') are not accepted because they carry no 'basin' "
                "or 'date' coordinate arrays to guard against silent basin-order or date misalignment. "
                "Upload a coordinate-labeled zipped Zarr store (.zarr.zip) or keyed .npz archive "
                "containing 'qsim', 'basin', and 'date'."
            ),
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file format for '{filename}'. Upload a coordinate-labeled zipped Zarr "
            f"store (.zarr, .zarr.zip, .zip) or keyed .npz archive with 'qsim', 'basin', and 'date' "
            f"aligned to ({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS})."
        ),
    )


def _validate_and_normalize_sim_array(arr: np.ndarray) -> np.ndarray:
    """Validate that a NumPy hydrograph array has shape `(531, 3652)` or `(N_members, 531, 3652)`."""
    if arr.ndim == 2:
        if arr.shape != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hydrograph shape {arr.shape}; expected ({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS}).",
            )
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        if arr.shape[0] < 1 or arr.shape[1:] != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid 3D hydrograph shape {arr.shape}; expected "
                    f"(N_members, {NEWMAN_N_BASINS}, {NEWMAN_N_DAYS})."
                ),
            )
        return arr
    raise HTTPException(
        status_code=400,
        detail=(
            f"Invalid hydrograph array rank {arr.ndim}D with shape {arr.shape}; "
            f"expected 2D ({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS}) or 3D (N_members, {NEWMAN_N_BASINS}, {NEWMAN_N_DAYS})."
        ),
    )


def _parsed_from_numpy_array(arr: np.ndarray, meta: dict[str, Any]) -> ParsedHydrograph:
    """Build a `ParsedHydrograph` from a 2-D (single hydrograph) or 3-D (members only) NumPy array.

    A 2-D array is scored as submitted (`ensemble_source="stored_qsim_without_members"`). A 3-D
    array carries members only, so the scored ensemble hydrograph is `mean(members, axis=0)`
    computed here (`ensemble_source="mean_of_members_no_stored_qsim"`).
    """
    sim_norm = _validate_and_normalize_sim_array(arr)
    if arr.ndim == 2:
        meta["has_qsim_members"] = False
        return ParsedHydrograph(
            ensemble=sim_norm[0],
            members=None,
            has_stored_qsim=True,
            declared_is_ensemble=None,
            ensemble_reduction="mean",
            meta=meta,
        )
    meta["has_qsim_members"] = True
    return ParsedHydrograph(
        ensemble=np.mean(sim_norm, axis=0),
        members=sim_norm,
        has_stored_qsim=False,
        declared_is_ensemble=None,
        ensemble_reduction="mean",
        meta=meta,
    )


def _evaluable_basin_mask(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(obs_valid, good_b)` exactly like `metrics.evaluator.calculate_model_leaderboard_metrics`.

    `obs_valid = (~isnan(qobs)) & (qobs >= 0)`; a basin is evaluable when it has at least
    `MIN_VALID_OBS_DAYS` valid observations and observed variance `SST > MIN_SST_THRESHOLD`.
    """
    obs_valid = (~np.isnan(obs)) & (obs >= 0.0)
    cnt = np.sum(obs_valid, axis=1)
    o_clean = np.where(obs_valid, obs, 0.0)
    cnt_safe = np.maximum(cnt, 1)
    o_mean = np.sum(o_clean, axis=1) / cnt_safe
    o_diff = np.where(obs_valid, obs - o_mean[:, None], 0.0)
    sst = np.sum(o_diff**2, axis=1)
    good_b = (cnt >= MIN_VALID_OBS_DAYS) & (sst > MIN_SST_THRESHOLD)
    return obs_valid, good_b


def _dates_ns(dates_np: np.ndarray | None) -> np.ndarray:
    if dates_np is None:
        return pd.date_range(NEWMAN_START_DATE, NEWMAN_END_DATE, freq="D").values.astype("datetime64[ns]")
    return np.asarray(dates_np).astype("datetime64[ns]")


def _compute_matrix_metrics(
    qobs: np.ndarray,
    qsim: np.ndarray,
    dates_np: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute all 14 Gauch et al. (2023) hydrological metrics per basin via `metrics/metrics.py`.

    Mirrors `metrics.evaluator.calculate_model_leaderboard_metrics` exactly: float64 inputs,
    predictions scored as submitted (negative values are NOT clipped), observations masked to
    `obs_valid`, and metrics computed only for evaluable basins (`cnt >= 30 & SST > 1e-12`);
    all other basins are `NaN`. Caller must verify `pred_nan_when_obs_valid == 0` first.
    """
    obs = np.asarray(qobs, dtype=np.float64)
    sim = np.asarray(qsim, dtype=np.float64)
    dates_ns = _dates_ns(dates_np)

    obs_valid, good_b = _evaluable_basin_mask(obs)
    cnt = np.sum(obs_valid, axis=1)
    n_basins = obs.shape[0]
    obs_for_da = np.where(obs_valid, obs, np.nan)

    out: dict[str, np.ndarray] = {
        m_name: np.full(n_basins, np.nan, dtype=np.float64)
        for m_name in GAUCH_2023_METRICS
    }
    out["Abs_FHV"] = np.full(n_basins, np.nan, dtype=np.float64)
    out["Abs_FMS"] = np.full(n_basins, np.nan, dtype=np.float64)
    out["Abs_FLV"] = np.full(n_basins, np.nan, dtype=np.float64)
    out["n_valid_days"] = cnt.astype(np.float64)
    out["is_evaluable"] = good_b.astype(np.float64)

    for i in np.where(good_b)[0]:
        obs_da = xr.DataArray(obs_for_da[i], coords={"date": dates_ns}, dims=("date",))
        sim_da = xr.DataArray(sim[i], coords={"date": dates_ns}, dims=("date",))
        m_vals = calculate_all_metrics(obs_da, sim_da, resolution="1D", datetime_coord="date")
        for m_name in GAUCH_2023_METRICS:
            out[m_name][i] = float(m_vals[m_name])
        out["Abs_FHV"][i] = abs(float(out["FHV"][i]))
        out["Abs_FMS"][i] = abs(float(out["FMS"][i]))
        out["Abs_FLV"][i] = abs(float(out["FLV"][i]))

    return out


def _compute_matrix_nse_only(
    qobs: np.ndarray,
    qsim: np.ndarray,
    dates_np: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-basin Gauch et al. (2023) `nse` via `metrics/metrics.py` for one ensemble member.

    Same evaluable-basin rule and unclipped float64 inputs as `_compute_matrix_metrics`
    (and as `metrics.evaluator._per_member_median_nse`).
    """
    obs = np.asarray(qobs, dtype=np.float64)
    sim = np.asarray(qsim, dtype=np.float64)
    dates_ns = _dates_ns(dates_np)

    obs_valid, good_b = _evaluable_basin_mask(obs)
    obs_for_da = np.where(obs_valid, obs, np.nan)

    nse_all = np.full(obs.shape[0], np.nan, dtype=np.float64)
    for i in np.where(good_b)[0]:
        obs_da = xr.DataArray(obs_for_da[i], coords={"date": dates_ns}, dims=("date",))
        sim_da = xr.DataArray(sim[i], coords={"date": dates_ns}, dims=("date",))
        nse_all[i] = float(gauch_nse(obs_da, sim_da))
    return nse_all


def _summarize_per_basin(per_basin: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, Any]:
    """Aggregate per-basin Gauch et al. (2023) metric arrays into standardized CAMELS-US MIP summary metrics.

    Aggregates (median, mean, percentiles) run over finite per-basin values only (NaN and
    +/-Inf excluded), exactly like `metrics.evaluator._summarize_metric_vector`.
    """
    def sel(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if mask is not None:
            a = a[mask]
        return a[np.isfinite(a)]

    nse = sel(per_basin["NSE"])
    kge = sel(per_basin["KGE"])
    fhv = sel(per_basin["FHV"])
    abs_fhv = sel(per_basin["Abs_FHV"])

    summary_out: dict[str, Any] = {
        "n_basins": int(len(nse)),
        "median_NSE": _clean_float(np.median(nse)) if len(nse) else None,
        "mean_NSE": _clean_float(np.mean(nse)) if len(nse) else None,
        "std_NSE": _clean_float(np.std(nse)) if len(nse) else None,
        "min_NSE": _clean_float(np.min(nse)) if len(nse) else None,
        "p01_NSE": _clean_float(np.percentile(nse, 1)) if len(nse) else None,
        "p05_NSE": _clean_float(np.percentile(nse, 5)) if len(nse) else None,
        "p10_NSE": _clean_float(np.percentile(nse, 10)) if len(nse) else None,
        "p25_NSE": _clean_float(np.percentile(nse, 25)) if len(nse) else None,
        "p75_NSE": _clean_float(np.percentile(nse, 75)) if len(nse) else None,
        "p90_NSE": _clean_float(np.percentile(nse, 90)) if len(nse) else None,
        "n_NSE_le_0": int(np.sum(nse <= 0.0)) if len(nse) else 0,
        "pct_NSE_le_0": _clean_float(100.0 * np.mean(nse <= 0.0), 2) if len(nse) else None,
        "median_KGE": _clean_float(np.median(kge)) if len(kge) else None,
        "mean_KGE": _clean_float(np.mean(kge)) if len(kge) else None,
        "median_FHV": _clean_float(np.median(fhv), 2) if len(fhv) else None,
        "median_abs_FHV": _clean_float(np.median(abs_fhv), 2) if len(abs_fhv) else None,
        "pred_nan_when_obs_valid": 0,
    }
    for m_name in GAUCH_2023_METRICS:
        vals = sel(per_basin[m_name])
        digits = 2 if m_name in ("FHV", "FMS", "FLV") else 4
        summary_out[f"median_{m_name}"] = _clean_float(np.median(vals), digits) if len(vals) else None
        summary_out[f"mean_{m_name}"] = _clean_float(np.mean(vals), digits) if len(vals) else None

    return summary_out


def _reject_non_finite_predictions(sim: np.ndarray, obs_valid: np.ndarray, label: str) -> None:
    """Raise HTTP 400 for any `Inf` anywhere or any `NaN` on a day with a valid observation.

    `sim` is `(531, 3652)` or `(N_members, 531, 3652)`; `obs_valid` is `(531, 3652)`.
    """
    if np.isinf(sim).any():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hydrograph submission: {label} contain infinite values.",
        )
    valid_b = obs_valid if sim.ndim == 2 else obs_valid[np.newaxis, :, :]
    pred_nan_mask = np.isnan(sim) & valid_b
    pred_nan_when_obs_valid = int(np.sum(pred_nan_mask))
    if pred_nan_when_obs_valid > 0:
        basin_axis = (1,) if sim.ndim == 2 else (0, 2)
        affected_basins = int(np.sum(np.any(pred_nan_mask, axis=basin_axis)))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid hydrograph submission: pred_nan_when_obs_valid={pred_nan_when_obs_valid} "
                f"across {affected_basins} basin(s) ({label} contain {pred_nan_when_obs_valid} NaN values "
                f"on days where observed discharge is valid; no missing basins or NaN predictions are allowed)."
            ),
        )


def load_basins_447_mask(basins: list[str]) -> np.ndarray:
    """Return the boolean mask of the Kratzert et al. (2019b) 447-basin subset over the 531 canonical basins.

    `splits/basins_447_kratzert2019b.json` is a required protocol artefact: a missing file
    raises `FileNotFoundError` rather than silently reporting the 531-basin summary as 447.
    """
    b447_path = get_splits_dir() / "basins_447_kratzert2019b.json"
    if not b447_path.is_file():
        raise FileNotFoundError(
            f"Required 447-basin subset file not found: {b447_path} "
            "(splits/basins_447_kratzert2019b.json must be present to report summary_447)."
        )
    with open(b447_path, "r", encoding="utf-8") as f:
        b447_set = {str(x).zfill(8) for x in json.load(f)}
    mask_447 = np.array([b in b447_set for b in basins], dtype=bool)
    if int(mask_447.sum()) != len(b447_set):
        raise ValueError(
            f"{b447_path} lists {len(b447_set)} basins but only {int(mask_447.sum())} are among the 531 canonical basins."
        )
    return mask_447


def score_hydrograph_predictions(parsed: ParsedHydrograph, eval_targets: dict[str, Any]) -> dict[str, Any]:
    """Score a parsed hydrograph upload against the canonical Newman evaluation targets.

    Parity with `metrics.evaluator.calculate_model_leaderboard_metrics`: the ensemble
    hydrograph (`parsed.ensemble`, the stored `qsim` for Zarr uploads) is scored unclipped in
    float64 on evaluable basins; headline metrics are median/mean over finite per-basin values.

    Strictly enforces that NO missing basins or partial NaNs are allowed (`pred_nan_when_obs_valid == 0`):
    any `NaN` in the ensemble hydrograph or in any member on a day where `qobs` is valid
    (`~np.isnan(qobs) & (qobs >= 0.0)`), including any omitted/all-`NaN` basin that has valid
    observations, immediately raises HTTP 400, as does any `Inf`.

    Single-seed skill (`single_median_NSE`) is the mean over members of each member's 531-basin
    median NSE, with `single_median_NSE_std` the population standard deviation (ddof=0) across
    members (0.0 for a single member) and `single_median_NSE_per_member` the per-member values.
    Exactly like `metrics.evaluator.calculate_model_leaderboard_metrics`: when members are
    shipped they are always used (ensemble, `single_member`, or `per_basin_oracle` stores); an
    upload without members is its own single member unless the Zarr attribute
    `is_ensemble=True` declares it an (unverifiable) ensemble, in which case the single-seed
    fields are `None`. Ensemble/single-model consistency between `qsim` and `qsim_members`
    (`is_ensemble`, `ensemble_reduction`) is enforced by `resolve_member_info` (HTTP 400).
    """
    qobs: np.ndarray = eval_targets["qobs"]
    basins: list[str] = eval_targets["basins"]
    huc_02: list[str] = eval_targets["huc_02"]
    dates_np: np.ndarray = eval_targets["dates_np"]

    obs_valid, good_b = _evaluable_basin_mask(np.asarray(qobs, dtype=np.float64))
    if int(np.sum(good_b)) == 0:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Zero basins in the evaluation targets satisfied >= {MIN_VALID_OBS_DAYS} valid "
                f"observations and SST > {MIN_SST_THRESHOLD}."
            ),
        )

    ens_hydrograph = np.asarray(parsed.ensemble, dtype=np.float64)
    members = None if parsed.members is None else np.asarray(parsed.members, dtype=np.float64)
    if ens_hydrograph.shape != (NEWMAN_N_BASINS, NEWMAN_N_DAYS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hydrograph shape {ens_hydrograph.shape}; expected ({NEWMAN_N_BASINS}, {NEWMAN_N_DAYS}).",
        )
    _reject_non_finite_predictions(ens_hydrograph, obs_valid, "predictions")
    if members is not None:
        _reject_non_finite_predictions(members, obs_valid, "member predictions")

    member_info = resolve_member_info(parsed)
    n_members = int(member_info["num_members"])
    is_ensemble = bool(member_info["is_ensemble"])

    ens_per_basin = _compute_matrix_metrics(qobs, ens_hydrograph, dates_np=dates_np)
    ens_summary_531 = _summarize_per_basin(ens_per_basin)
    if ens_summary_531["median_NSE"] is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid hydrograph submission: no finite NSE could be computed on any evaluable basin.",
        )

    mask_447 = load_basins_447_mask(basins)
    ens_summary_447 = _summarize_per_basin(ens_per_basin, mask=mask_447)

    nse_vec = ens_per_basin["NSE"]
    nse_finite = nse_vec[np.isfinite(nse_vec)]

    single_seed_medians: list[float] | None
    if members is not None:
        single_seed_medians = []
        for s_idx in range(n_members):
            s_nse = _compute_matrix_nse_only(qobs, members[s_idx], dates_np=dates_np)
            finite_s = s_nse[np.isfinite(s_nse)]
            if finite_s.size == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid hydrograph submission: member {s_idx} produced no finite NSE on any evaluable basin.",
                )
            single_seed_medians.append(float(np.median(finite_s)))
    elif not is_ensemble:
        single_seed_medians = [float(np.median(nse_finite))]
    else:
        single_seed_medians = None

    if single_seed_medians is not None:
        single_med_nse_mean = _clean_float(np.mean(single_seed_medians))
        single_med_nse_std = _clean_float(np.std(single_seed_medians))
        single_med_nse_per_member: list[float] | None = [round(v, 4) for v in single_seed_medians]
    else:
        single_med_nse_mean = None
        single_med_nse_std = None
        single_med_nse_per_member = None

    huc_summary: list[dict[str, Any]] = []
    regional_huc02: dict[str, float] = {}
    huc_arr = np.array(huc_02)
    for huc in sorted(set(huc_02)):
        h_mask = huc_arr == huc
        h_nse = nse_vec[h_mask]
        h_nse = h_nse[np.isfinite(h_nse)]
        h_kge = ens_per_basin["KGE"][h_mask]
        h_kge = h_kge[np.isfinite(h_kge)]
        med_h_nse = _clean_float(np.median(h_nse)) if len(h_nse) else None
        if med_h_nse is not None:
            regional_huc02[huc] = med_h_nse
        huc_summary.append(
            {
                "huc_02": huc,
                "n_basins": int(len(h_nse)),
                "median_NSE": med_h_nse,
                "mean_NSE": _clean_float(np.mean(h_nse)) if len(h_nse) else None,
                "median_KGE": _clean_float(np.median(h_kge)) if len(h_kge) else None,
                "n_NSE_le_0": int(np.sum(h_nse <= 0.0)) if len(h_nse) else 0,
            }
        )

    cdf_values = _empirical_cdf_101(nse_finite) if nse_finite.size > 0 else None

    per_basin_rows = []
    for idx, b in enumerate(basins):
        row_dict: dict[str, Any] = {
            "basin": b,
            "huc_02": huc_02[idx],
            "is_evaluable": bool(good_b[idx]),
            "NSE": _clean_float(ens_per_basin["NSE"][idx]),
            "MSE": _clean_float(ens_per_basin["MSE"][idx]),
            "RMSE": _clean_float(ens_per_basin["RMSE"][idx]),
            "KGE": _clean_float(ens_per_basin["KGE"][idx]),
            "logNSE": _clean_float(ens_per_basin["logNSE"][idx]),
            "logKGE": _clean_float(ens_per_basin["logKGE"][idx]),
            "Alpha-NSE": _clean_float(ens_per_basin["Alpha-NSE"][idx]),
            "Pearson-r": _clean_float(ens_per_basin["Pearson-r"][idx]),
            "Beta-KGE": _clean_float(ens_per_basin["Beta-KGE"][idx]),
            "Beta-NSE": _clean_float(ens_per_basin["Beta-NSE"][idx]),
            "FHV": _clean_float(ens_per_basin["FHV"][idx], 2),
            "Abs_FHV": _clean_float(ens_per_basin["Abs_FHV"][idx], 2),
            "FMS": _clean_float(ens_per_basin["FMS"][idx], 2),
            "FLV": _clean_float(ens_per_basin["FLV"][idx], 2),
            "Peak-Timing": _clean_float(ens_per_basin["Peak-Timing"][idx], 4),
        }
        per_basin_rows.append(row_dict)

    # 531-vector of per-basin NSE (None for non-evaluable basins), stored on community
    # leaderboard rows so `/api/entry/{id}/basins` can serve a real empirical CDF.
    per_basin_nse_531 = [_clean_float(v, 6) for v in nse_vec]

    gauch_medians = {
        m_name: ens_summary_531[f"median_{m_name}"]
        for m_name in GAUCH_2023_METRICS
    }

    standardized_summary = {
        "num_evaluated_basins": ens_summary_531["n_basins"],
        "pred_nan_when_obs_valid": 0,
        "is_ensemble": is_ensemble,
        "num_members": n_members,
        "has_qsim_members": members is not None,
        "ensemble_source": member_info["ensemble_source"],
        "ensemble_reduction": member_info["ensemble_reduction"],
        "single_seed_member_index": member_info["single_seed_member_index"],
        "median_nse": ens_summary_531["median_NSE"],
        "mean_nse": ens_summary_531["mean_NSE"],
        "median_kge": ens_summary_531["median_KGE"],
        "median_abs_fhv": ens_summary_531["median_abs_FHV"],
        "nse_le_zero_count": ens_summary_531["n_NSE_le_0"],
        "min_nse": ens_summary_531["min_NSE"],
        "p01_nse": ens_summary_531["p01_NSE"],
        "p05_nse": ens_summary_531["p05_NSE"],
        "p10_nse": ens_summary_531["p10_NSE"],
        "q25_nse": ens_summary_531["p25_NSE"],
        "q75_nse": ens_summary_531["p75_NSE"],
        "single_seed_median_nse_mean": single_med_nse_mean,
        "single_seed_median_nse_std": single_med_nse_std,
        "single_seed_median_nse_per_member": single_med_nse_per_member,
        "gauch_2023_medians": gauch_medians,
    }

    return {
        "n_basins": NEWMAN_N_BASINS,
        "n_days": NEWMAN_N_DAYS,
        "n_seeds": n_members,
        "n_members": n_members,
        "is_ensemble": is_ensemble,
        "declared_is_ensemble": parsed.declared_is_ensemble,
        "has_qsim_members": members is not None,
        "ensemble_source": member_info["ensemble_source"],
        "ensemble_reduction": member_info["ensemble_reduction"],
        "single_seed_member_index": member_info["single_seed_member_index"],
        "pred_nan_when_obs_valid": 0,
        "ens_median_NSE": ens_summary_531["median_NSE"] if is_ensemble else None,
        "single_median_NSE": single_med_nse_mean,
        "single_median_NSE_std": single_med_nse_std,
        "single_median_NSE_per_member": single_med_nse_per_member,
        "primary_ranking_NSE": ens_summary_531["median_NSE"],
        "median_nse": ens_summary_531["median_NSE"],
        "mean_nse": ens_summary_531["mean_NSE"],
        "median_kge": ens_summary_531["median_KGE"],
        "median_abs_fhv": ens_summary_531["median_abs_FHV"],
        "nse_le_zero_count": ens_summary_531["n_NSE_le_0"],
        "num_evaluated_basins": ens_summary_531["n_basins"],
        "gauch_2023_medians": gauch_medians,
        "summary": standardized_summary,
        "summary_531": ens_summary_531,
        "summary_447": ens_summary_447,
        "regional_huc02": regional_huc02,
        "huc_02_breakdown": huc_summary,
        "nse_cdf_101": cdf_values,
        "per_basin_nse": per_basin_nse_531,
        "per_basin": per_basin_rows,
    }


def persist_submission_zarr(
    parsed: ParsedHydrograph,
    eval_targets: dict[str, Any],
    leaderboard_id: str,
    entry_id: str,
    attrs: dict[str, Any],
) -> Path:
    """Persist the scored hydrograph(s) of a community submission as a consolidated Zarr v2 store.

    The store is written to `<submissions_dir>/<leaderboard_id>/<entry_id>.zarr` with `qsim`
    (`("basin", "date")`, the scored ensemble hydrograph exactly as scored) and, when present,
    `qsim_members` (`("member", "basin", "date")`), plus `basin`, `date`, and `huc_02`
    coordinates, so that `metrics.evaluator.calculate_model_leaderboard_metrics` can re-score
    it from `bucket_uri=<submissions_dir>` later. Returns the real on-disk path.
    """
    submissions_root = get_submissions_dir()
    store_path = submissions_root / leaderboard_id / f"{entry_id}.zarr"
    if store_path.exists():
        raise HTTPException(status_code=409, detail=f"Submission store already exists: {store_path}")
    store_path.parent.mkdir(parents=True, exist_ok=True)

    basins = np.array(eval_targets["basins"], dtype="<U8")
    dates = np.asarray(eval_targets["dates_np"]).astype("datetime64[ns]")
    huc_02 = np.array(eval_targets["huc_02"], dtype="<U2")

    data_vars: dict[str, Any] = {"qsim": (("basin", "date"), np.asarray(parsed.ensemble, dtype=np.float64))}
    if parsed.members is not None:
        data_vars["qsim_members"] = (
            ("member", "basin", "date"),
            np.asarray(parsed.members, dtype=np.float64),
        )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "basin": ("basin", basins),
            "date": ("date", dates),
            "huc_02": ("basin", huc_02),
        },
        attrs={k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool))},
    )
    ds.to_zarr(store_path, mode="w-", zarr_format=2, consolidated=True)

    if not (store_path / ".zmetadata").is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Submission store was not written with consolidated metadata: {store_path}",
        )
    return store_path


@app.get("/api/mips")
def api_list_mips() -> dict[str, Any]:
    """Auto-discover and return all registered MIP protocols under `mips/*/protocol.json`."""
    data_root = get_data_root()
    repo_for_mips = data_root if (data_root / "mips").is_dir() else REPO_ROOT
    mips = discover_mips(repo_root=repo_for_mips)
    return {
        "default_mip_id": DEFAULT_MIP_ID,
        "n_mips": len(mips),
        "mips": [proto.to_summary_dict() for proto in mips.values()],
    }


@app.get("/api/mips/{mip_id}/protocol")
def api_mip_protocol(mip_id: str) -> dict[str, Any]:
    """Return the declarative protocol specification for a registered MIP."""
    data_root = get_data_root()
    repo_for_mips = data_root if (data_root / "mips").is_dir() else REPO_ROOT
    mips = discover_mips(repo_root=repo_for_mips)
    if mip_id not in mips:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown mip_id '{mip_id}'. Registered MIPs: {sorted(mips.keys())}",
        )
    return get_mip(mip_id=mip_id, repo_root=repo_for_mips).to_summary_dict()


@app.get("/api/splits/{filename}")
def api_get_split_file(filename: str) -> Response:
    """Serve canonical basin split files (.json, .txt) for the default MIP."""
    safe_name = Path(filename).name
    split_path = get_splits_dir(DEFAULT_MIP_ID) / safe_name
    if not split_path.is_file():
        raise HTTPException(status_code=404, detail=f"Split file '{safe_name}' not found.")
    media_type = "application/json" if safe_name.endswith(".json") else "text/plain; charset=utf-8"
    return FileResponse(str(split_path), media_type=media_type)


@app.get("/api/mips/{mip_id}/splits/{filename}")
def api_get_mip_split_file(mip_id: str, filename: str) -> Response:
    """Serve canonical split files (.json, .txt) for a specific registered MIP."""
    data_root = get_data_root()
    repo_for_mips = data_root if (data_root / "mips").is_dir() else REPO_ROOT
    mips = discover_mips(repo_root=repo_for_mips)
    if mip_id not in mips:
        raise HTTPException(status_code=404, detail=f"Unknown mip_id '{mip_id}'.")
    safe_name = Path(filename).name
    split_path = get_splits_dir(mip_id) / safe_name
    if not split_path.is_file():
        raise HTTPException(status_code=404, detail=f"Split file '{safe_name}' not found for '{mip_id}'.")
    media_type = "application/json" if safe_name.endswith(".json") else "text/plain; charset=utf-8"
    return FileResponse(str(split_path), media_type=media_type)


@app.get("/api/figures/{filename}")
def api_get_figure(filename: str) -> Response:
    """Serve generated paper figures (.png, .pdf, .json) for the default MIP."""
    safe_name = Path(filename).name
    fig_path = get_figures_dir(DEFAULT_MIP_ID) / safe_name
    if not fig_path.is_file():
        raise HTTPException(status_code=404, detail=f"Figure '{safe_name}' not found.")
    media_type = "image/png"
    if safe_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif safe_name.endswith(".json"):
        media_type = "application/json"
    return FileResponse(str(fig_path), media_type=media_type, headers={"Cache-Control": "no-store"})


@app.get("/api/mips/{mip_id}/figures/{filename}")
def api_get_mip_figure(mip_id: str, filename: str) -> Response:
    """Serve generated paper figures (.png, .pdf, .json) for a specific registered MIP."""
    data_root = get_data_root()
    repo_for_mips = data_root if (data_root / "mips").is_dir() else REPO_ROOT
    mips = discover_mips(repo_root=repo_for_mips)
    if mip_id not in mips:
        raise HTTPException(status_code=404, detail=f"Unknown mip_id '{mip_id}'.")
    safe_name = Path(filename).name
    fig_path = get_figures_dir(mip_id) / safe_name
    if not fig_path.is_file():
        raise HTTPException(status_code=404, detail=f"Figure '{safe_name}' not found for '{mip_id}'.")
    media_type = "image/png"
    if safe_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif safe_name.endswith(".json"):
        media_type = "application/json"
    return FileResponse(str(fig_path), media_type=media_type, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Health check endpoint verifying leaderboard and registry readiness."""
    tables_data = load_tables_data()
    reg_data = load_registry_data()
    contrib_data = load_contributions_index()
    data_root = get_data_root()
    repo_for_mips = data_root if (data_root / "mips").is_dir() else REPO_ROOT
    mips = discover_mips(repo_root=repo_for_mips)
    return {
        "status": "ok",
        "service": "hydromip.org",
        "default_mip_id": DEFAULT_MIP_ID,
        "n_mips": len(mips),
        "n_leaderboards": len(tables_data["leaderboards"]),
        "n_families": len(reg_data["families"]),
        "total_entries": int(tables_data.get("total_entries", 0)),
        "total_contributions": int(contrib_data.get("total_contributions", 0)),
    }


@app.get("/api/protocol")
def api_protocol() -> dict[str, Any]:
    """Return the canonical CAMELS-US MIP spatial & temporal protocol specification."""
    splits_proto = load_splits_protocol()
    tables_data = load_tables_data()
    ui_proto = tables_data["protocol"]
    return {
        **splits_proto,
        "benchmark_name": ui_proto.get("benchmark_name", splits_proto["protocol_name"]),
        "n_basins": int(splits_proto["num_basins"]),
        "n_leaderboards": int(ui_proto.get("n_leaderboards", 12)),
        "forcing_regimes": ui_proto.get("forcing_regimes", []),
        "spatial_regimes_table": ui_proto.get("spatial_regimes", []),
    }


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    """Return high-level summary of all 12 leaderboards and the Table 2 model families (`total_families`)."""
    data = build_enriched_leaderboards()
    matrix_summary = [
        {
            "id": lb["id"],
            "table_number": lb["table_number"],
            "section": lb["section"],
            "spatial_regime": lb["spatial_regime"],
            "forcing_regime": lb["forcing_regime"],
            "title": lb["title"],
            "n_entries": lb["n_entries"],
            "top_model": lb["top_model"],
            "top_nse": lb["top_nse"],
        }
        for lb in data["leaderboards"]
    ]
    return {
        "updated_at": data["updated_at"],
        "protocol": data["protocol"],
        "total_entries": data["total_entries"],
        "total_families": data["total_families"],
        "total_contributions": data["total_contributions"],
        "matrix_summary": matrix_summary,
    }


@app.get("/api/leaderboards")
def api_leaderboards(
    spatial_regime: str | None = Query(default=None),
    forcing_regime: str | None = Query(default=None),
    paradigm: str | None = Query(default=None),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return all 12 leaderboards (Tables 3 to 14) with optional filtering."""
    return build_enriched_leaderboards(
        spatial_regime=spatial_regime,
        forcing_regime=forcing_regime,
        paradigm=paradigm,
        category=category,
        search=search,
    )


@app.get("/api/leaderboards/{leaderboard_id}")
def api_leaderboard_by_id(leaderboard_id: str) -> dict[str, Any]:
    """Return a single leaderboard table by its canonical `leaderboard_id`."""
    data = build_enriched_leaderboards()
    board = next((lb for lb in data["leaderboards"] if lb["id"] == leaderboard_id), None)
    if board is None:
        raise HTTPException(
            status_code=404,
            detail=f"Leaderboard '{leaderboard_id}' not found. Expected one of: {list(CANONICAL_LEADERBOARD_IDS)}",
        )
    return board


@app.get("/api/registry")
def api_registry() -> dict[str, Any]:
    """Return the Table 2 model-family registry (count read from `table_2_registry.json`) and contributions index."""
    data = build_enriched_leaderboards()
    contrib_data = load_contributions_index()
    return {
        "total_families": len(data["families"]),
        "families": data["families"],
        "total_contributions": int(contrib_data.get("total_contributions", len(contrib_data.get("entries", {})))),
        "contributions": contrib_data.get("entries", {}),
    }


@app.get("/api/families")
def api_families() -> dict[str, Any]:
    """Alias for `/api/registry` used by the browser UI."""
    return api_registry()


class FamilyCreateRequest(BaseModel):
    family_id: str | None = None
    name: str
    provenance_category: str = "Contribution"
    paradigm: str
    citation: str = ""
    paper_url: str = ""
    github_url: str = ""
    zenodo_doi: str = ""
    hydroshare_url: str = ""
    description: str = ""


@app.post("/api/families")
def api_create_or_update_family(req: FamilyCreateRequest) -> dict[str, Any]:
    """Create or update a model family in `table_2_registry.json`."""
    lb_dir = get_leaderboards_dir()
    reg_path = lb_dir / "table_2_registry.json"
    reg_data = load_registry_data()

    fid = req.family_id or re.sub(r"[^a-z0-9]+", "_", req.name.lower()).strip("_")
    github_clean = normalize_github_url(req.github_url)
    z_doi, z_url = normalize_zenodo(req.zenodo_doi)

    new_fam = {
        "family_id": fid,
        "name": req.name.strip(),
        "provenance_category": req.provenance_category.strip(),
        "paradigm": req.paradigm.strip(),
        "citation": req.citation.strip(),
        "paper_url": req.paper_url.strip(),
        "github_url": github_clean,
        "zenodo_doi": z_doi,
        "zenodo_url": z_url,
        "hydroshare_url": req.hydroshare_url.strip(),
        "description": req.description.strip(),
    }

    existing_idx = next((i for i, f in enumerate(reg_data["families"]) if f["family_id"] == fid), None)
    if existing_idx is not None:
        reg_data["families"][existing_idx] = {**reg_data["families"][existing_idx], **new_fam}
    else:
        reg_data["families"].insert(0, new_fam)
    reg_data["total_families"] = len(reg_data["families"])

    _write_json_file(reg_path, reg_data)
    return {"status": "ok", "family": new_fam}


class LinkValidationRequest(BaseModel):
    github_url: str | None = ""
    zenodo_doi: str | None = ""
    hydroshare_url: str | None = ""


@app.post("/api/validate-links")
def api_validate_links(req: LinkValidationRequest) -> dict[str, Any]:
    """Validate and normalize GitHub repository and Zenodo DOI links."""
    gh_url = normalize_github_url(req.github_url)
    z_doi, z_url = normalize_zenodo(req.zenodo_doi)
    hs_url = (req.hydroshare_url or "").strip()
    if hs_url and not hs_url.startswith("http"):
        hs_url = f"https://doi.org/{hs_url}"

    gh_status = {"valid_format": False, "normalized_url": gh_url, "repo_slug": "", "reachable": None}
    if gh_url:
        m = re.match(r"^https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", gh_url)
        if m:
            gh_status["valid_format"] = True
            gh_status["repo_slug"] = m.group(1)

    z_status = {
        "valid_format": bool(z_doi),
        "zenodo_doi": z_doi,
        "zenodo_url": z_url,
        "badge_markdown": f"[![DOI](https://zenodo.org/badge/DOI/{z_doi}.svg)]({z_url})" if z_doi else "",
    }
    return {
        "github": gh_status,
        "zenodo": z_status,
        "hydroshare": {"normalized_url": hs_url, "valid_format": bool(hs_url)},
    }


def _find_raw_leaderboard_row(entry_id: str, tables_data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the unmodified leaderboard JSON row for `entry_id` (including `per_basin_nse`), or None."""
    for lb in tables_data["leaderboards"]:
        for r in lb.get("rows", []):
            if str(r.get("entry_id")) == entry_id:
                return r
    return None


def _per_basin_nse_for_entry(
    entry_id: str,
    tables_data: dict[str, Any],
    contrib_data: dict[str, Any],
) -> tuple[np.ndarray | None, str | None]:
    """Locate real per-basin NSE values for `entry_id`.

    Looks, in order, at the leaderboard row's `per_basin_nse` (531-vector written by
    `leaderboards/update_leaderboard.py` or by `/api/submit`), then at the
    `contributions_index.json` entry's `per_basin_nse` or `per_basin.NSE`. Returns
    `(vector, origin)` or `(None, None)` when no real per-basin data exists. Nothing is ever
    synthesized from summary statistics.
    """
    raw_row = _find_raw_leaderboard_row(entry_id, tables_data)
    if raw_row is not None:
        arr = _per_basin_nse_array(raw_row.get("per_basin_nse"))
        if arr is not None:
            return arr, "leaderboard_row.per_basin_nse"

    contrib_entry = contrib_data.get("entries", {}).get(entry_id, {})
    arr = _per_basin_nse_array(contrib_entry.get("per_basin_nse"))
    if arr is not None:
        return arr, "contributions_index.per_basin_nse"
    per_basin_block = contrib_entry.get("per_basin")
    if isinstance(per_basin_block, dict):
        arr = _per_basin_nse_array(per_basin_block.get("NSE", per_basin_block.get("nse")))
        if arr is not None:
            return arr, "contributions_index.per_basin.NSE"
    return None, None


@app.get("/api/entry/{entry_id}/basins")
def api_entry_basins(entry_id: str) -> dict[str, Any]:
    """Return the empirical 531-basin NSE CDF and regional diagnostics for a leaderboard entry.

    The CDF is computed only from real per-basin NSE values stored with the entry
    (`source="per_basin_metrics"`: 101-point `np.quantile` over finite values at
    `q = linspace(0, 1, 101)`). Literature-only rows that carry summary statistics but no
    per-basin values get `source="unavailable"` with `nse_cdf_101=null` (HTTP 200). The
    reference curves (`ref_3f_lstm_cdf_101`, `ref_1f_daymet_cdf_101`) are likewise computed
    from the stored per-basin NSE of the real reference rows in `REFERENCE_CDF_ENTRY_IDS`
    and are `null` when those rows carry none. No defaults, no synthetic distributions.
    """
    data = build_enriched_leaderboards()
    entry = None
    for lb in data["leaderboards"]:
        for r in lb["rows"]:
            if r["entry_id"] == entry_id:
                entry = r
                break
        if entry is not None:
            break

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Entry '{entry_id}' not found.")

    tables_data = load_tables_data()
    contrib_data = load_contributions_index()
    nse_vec, origin = _per_basin_nse_for_entry(entry_id, tables_data, contrib_data)
    if nse_vec is not None:
        source = "per_basin_metrics"
        nse_cdf = _empirical_cdf_101(nse_vec)
        n_with_nse: int | None = int(np.isfinite(nse_vec).sum())
    else:
        source = "unavailable"
        nse_cdf = None
        n_with_nse = None

    ref_curves: dict[str, Any] = {}
    for ref_key, ref_entry_id in REFERENCE_CDF_ENTRY_IDS.items():
        ref_vec, _ = _per_basin_nse_for_entry(ref_entry_id, tables_data, contrib_data)
        ref_curves[f"{ref_key}_entry_id"] = ref_entry_id
        ref_curves[f"{ref_key}_cdf_101"] = _empirical_cdf_101(ref_vec) if ref_vec is not None else None

    return {
        "entry": entry,
        "source": source,
        "per_basin_nse_origin": origin,
        "n_basins_with_nse": n_with_nse,
        "nse_cdf_quantiles": [round(float(q), 2) for q in NSE_CDF_QUANTILES],
        "nse_cdf_101": nse_cdf,
        "ref_3f_lstm_entry_id": ref_curves["ref_3f_lstm_entry_id"],
        "ref_3f_lstm_cdf_101": ref_curves["ref_3f_lstm_cdf_101"],
        "ref_1f_daymet_entry_id": ref_curves["ref_1f_daymet_entry_id"],
        "ref_1f_daymet_cdf_101": ref_curves["ref_1f_daymet_cdf_101"],
        "huc_02_breakdown": entry.get("huc_02_breakdown", []),
    }


def _prospective_rank(lb_rows: list[dict[str, Any]], candidate_nse: float | None) -> int:
    """1-based rank a candidate median NSE would take on `lb_rows` (rows without a score sort last)."""
    if candidate_nse is None:
        return len(lb_rows) + 1
    better_count = 0
    for e in lb_rows:
        existing = _finite_or_none(e.get("primary_ranking_nse"))
        if existing is not None and existing > candidate_nse:
            better_count += 1
    return better_count + 1


@app.post("/api/score")
async def api_score_model_run(
    file: UploadFile | None = File(default=None),
    leaderboard_id: str = Form(default="gauged__single_camels"),
) -> dict[str, Any]:
    """Score an uploaded `.zarr` (zipped Zarr store), `.npz`, or `.npy` `(531, 3652)` hydrograph file.

    Scoring is identical to `metrics.evaluator.calculate_model_leaderboard_metrics` (see
    `score_hydrograph_predictions`). `ensemble_source` uses the evaluator's vocabulary
    (`mean_of_qsim_members`, `median_of_qsim_members`, `single_member`, `per_basin_oracle`,
    `stored_qsim_without_members`) plus `mean_of_members_no_stored_qsim` for a members-only
    3-D NumPy upload whose ensemble hydrograph had to be computed here.
    """
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid leaderboard_id '{leaderboard_id}'. Must be one of {list(CANONICAL_LEADERBOARD_IDS)}.",
        )
    if file is None or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload a 531x3652 daily hydrograph file (.zarr, .npz, or .npy).",
        )

    raw_bytes = await file.read()
    eval_targets = load_eval_targets()
    parsed = parse_uploaded_hydrograph(file.filename, raw_bytes, eval_targets)
    scored = score_hydrograph_predictions(parsed, eval_targets)

    tables_data = load_tables_data()
    lb_obj = next((lb for lb in tables_data["leaderboards"] if lb["id"] == leaderboard_id), None)
    lb_rows = lb_obj["rows"] if lb_obj is not None else []
    prospective_rank = _prospective_rank(lb_rows, _finite_or_none(scored["primary_ranking_NSE"]))

    return {
        "status": "ok",
        "file_metadata": parsed.meta,
        "leaderboard_id": leaderboard_id,
        "prospective_rank": prospective_rank,
        "total_on_leaderboard": len(lb_rows) + 1,
        "pred_nan_when_obs_valid": 0,
        "ensemble_source": scored["ensemble_source"],
        "ensemble_reduction": scored["ensemble_reduction"],
        "single_seed_member_index": scored["single_seed_member_index"],
        "is_ensemble": scored["is_ensemble"],
        "n_members": scored["n_members"],
        "has_qsim_members": scored["has_qsim_members"],
        "median_nse": scored["median_nse"],
        "mean_nse": scored["mean_nse"],
        "median_kge": scored["median_kge"],
        "median_abs_fhv": scored["median_abs_fhv"],
        "nse_le_zero_count": scored["nse_le_zero_count"],
        "num_evaluated_basins": scored["num_evaluated_basins"],
        "single_median_NSE": scored["single_median_NSE"],
        "single_median_NSE_std": scored["single_median_NSE_std"],
        "single_median_NSE_per_member": scored["single_median_NSE_per_member"],
        "summary": scored["summary"],
        "scoring": scored,
    }


def _format_nse_columns(scored: dict[str, Any]) -> dict[str, Any]:
    """Derive the leaderboard NSE columns from a scoring result with explicit single-seed/ensemble semantics.

    - Single-seed submission: `ens_med_nse` is `None` (display `"---"`); `single_med_nse` is
      the hydrograph's 531-basin median NSE (display `"0.812"`). A single-model store that ships
      its sibling seeds (`ensemble_source="single_member"`) reports the mean over all siblings
      with the population std (display `"0.812 ± 0.006"`), exactly like the evaluator.
    - Ensemble submission: `ens_med_nse` is the ensemble hydrograph's median NSE (display
      `"0.834"`); `single_med_nse` is the mean over members of the per-member median NSE
      (display `"0.812 ± 0.006"` with the population std), or `None`/`"---"` when the
      members were not supplied and the per-seed skill cannot be verified.
    - `primary_ranking_nse` is always the median NSE of the scored hydrograph.
    """
    primary = _finite_or_none(scored["primary_ranking_NSE"])
    is_ensemble = bool(scored["is_ensemble"])
    single_mean = _finite_or_none(scored["single_median_NSE"])
    single_std = _finite_or_none(scored["single_median_NSE_std"])
    per_member = scored.get("single_median_NSE_per_member") or []
    show_spread = len(per_member) > 1 and single_mean is not None and single_std is not None

    if show_spread:
        single_disp = f"{single_mean:.3f} ± {single_std:.3f}"
    elif single_mean is not None:
        single_disp = f"{single_mean:.3f}"
    else:
        single_disp = "---"

    if is_ensemble:
        ens_med = primary
        ens_disp = f"{primary:.3f}" if primary is not None else "---"
    else:
        ens_med = None
        ens_disp = "---"

    return {
        "ens_med_nse": ens_med,
        "ens_med_nse_display": ens_disp,
        "single_med_nse": single_mean,
        "single_med_nse_std": single_std,
        "single_med_nse_display": single_disp,
        "primary_ranking_nse": primary,
    }


@app.post("/api/submit")
async def api_submit_to_leaderboard(
    file: UploadFile | None = File(default=None),
    leaderboard_id: str = Form(...),
    model_name: str = Form(...),
    forcing: str = Form(...),
    category: str = Form(default="Contribution"),
    family_id: str = Form(default=""),
    new_family_name: str = Form(default=""),
    paradigm: str = Form(default="Community Contributed Model"),
    citation: str = Form(default=""),
    paper_url: str = Form(default=""),
    github_url: str = Form(default=""),
    zenodo_doi: str = Form(default=""),
    hydroshare_url: str = Form(default=""),
    family_description: str = Form(default=""),
    notes: str = Form(default=""),
) -> dict[str, Any]:
    """Score an uploaded `(531, 3652)` hydrograph file, persist it, and register the entry on the selected leaderboard.

    The scored hydrograph (and its members, if any) is written as a consolidated Zarr v2 store
    under `CAMELS_MIP_SUBMISSIONS_DIR` (default `<data_root>/submissions/<leaderboard_id>/<entry_id>.zarr`)
    BEFORE the leaderboard row is written; the recorded `hydrograph_path` is that real path.
    If the submission storage is not configured/writable the submission is refused with HTTP 503.
    """
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid leaderboard_id '{leaderboard_id}'. Must be one of {list(CANONICAL_LEADERBOARD_IDS)}.",
        )
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="model_name must not be empty.")
    if file is None or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please provide a 531x3652 hydrograph file (.zarr, .npz, or .npy) to score and submit.",
        )

    # Fail fast (HTTP 503) before any scoring work if submissions cannot be persisted.
    get_submissions_dir()

    raw_bytes = await file.read()
    eval_targets = load_eval_targets()
    parsed = parse_uploaded_hydrograph(file.filename, raw_bytes, eval_targets)
    scored = score_hydrograph_predictions(parsed, eval_targets)

    sum531 = scored["summary_531"]
    n_members = int(scored["n_members"])
    is_ensemble = bool(scored["is_ensemble"])

    lb_dir = get_leaderboards_dir()
    reg_data = load_registry_data()

    gh_clean = normalize_github_url(github_url)
    z_doi, z_url = normalize_zenodo(zenodo_doi)
    hs_clean = hydroshare_url.strip()

    resolved_family_id = family_id.strip()
    family_record: dict[str, Any] | None = None
    if new_family_name.strip() or not resolved_family_id:
        fname = new_family_name.strip() or model_name.strip()
        resolved_family_id = re.sub(r"[^a-z0-9]+", "_", fname.lower()).strip("_")
        existing_fam = next((f for f in reg_data["families"] if f["family_id"] == resolved_family_id), None)
        if existing_fam is None:
            family_record = {
                "family_id": resolved_family_id,
                "name": fname,
                "provenance_category": category.strip() or "Contribution",
                "paradigm": paradigm.strip() or "Hydrological Model",
                "citation": citation.strip() or "Community Submission (2026)",
                "paper_url": paper_url.strip(),
                "github_url": gh_clean,
                "zenodo_doi": z_doi,
                "zenodo_url": z_url,
                "hydroshare_url": hs_clean,
                "description": family_description.strip() or notes.strip(),
                "n_entries": 1,
                "leaderboards_covered": [leaderboard_id],
                "best_nse": scored["median_nse"],
            }

    entry_id = f"sub_{uuid.uuid4().hex[:8]}"
    submitted_at = dt.datetime.now(dt.timezone.utc).isoformat()
    nse_columns = _format_nse_columns(scored)

    store_path = persist_submission_zarr(
        parsed,
        eval_targets,
        leaderboard_id=leaderboard_id,
        entry_id=entry_id,
        attrs={
            "title": f"CAMELS-US MIP community submission {entry_id}",
            "entry_id": entry_id,
            "leaderboard_id": leaderboard_id,
            "model_name": model_name.strip(),
            "forcing": forcing.strip(),
            "is_ensemble": is_ensemble,
            "num_members": n_members,
            "ensemble_reduction": scored["ensemble_reduction"],
            "ensemble_source": scored["ensemble_source"],
            "upload_sha256": parsed.meta.get("sha256", ""),
            "upload_filename": parsed.meta.get("filename", ""),
            "submitted_at": submitted_at,
            "units": "mm/day",
        },
    )
    hydrograph_path = str(store_path)

    new_entry = {
        "entry_id": entry_id,
        "leaderboard_id": leaderboard_id,
        "model_name": model_name.strip(),
        "family_id": resolved_family_id,
        "category": category.strip(),
        "forcing": forcing.strip(),
        "seeds": str(n_members),
        "is_ensemble": is_ensemble,
        "num_members": n_members,
        "has_qsim_members": scored["has_qsim_members"],
        "ensemble_source": scored["ensemble_source"],
        "ensemble_reduction": scored["ensemble_reduction"],
        "single_seed_member_index": scored["single_seed_member_index"],
        **nse_columns,
        "single_med_nse_per_member": scored["single_median_NSE_per_member"],
        "mean_nse": sum531.get("mean_NSE"),
        "med_kge": sum531.get("median_KGE"),
        "med_fhv_abs": sum531.get("median_abs_FHV"),
        "nse_le_0": sum531.get("n_NSE_le_0"),
        "num_evaluated_basins": sum531.get("n_basins"),
        "asterisk": False,
        "archive_key": None,
        "hydrograph_path": hydrograph_path,
        "github_url": gh_clean,
        "code_url": gh_clean,
        "is_open_source": bool(gh_clean),
        "is_closed_source": not bool(gh_clean),
        "zenodo_doi": z_doi,
        "zenodo_url": z_url,
        "data_archive_url": z_url or hs_clean,
        "hydroshare_url": hs_clean,
        "paper_url": paper_url.strip(),
        "citation": citation.strip(),
        "paradigm": paradigm.strip(),
        "notes": notes.strip(),
        "sha256": parsed.meta.get("sha256", ""),
        "submitted_at": submitted_at,
        "is_user_submission": True,
        "huc_02_breakdown": scored["huc_02_breakdown"],
        "per_basin_nse": scored["per_basin_nse"],
    }

    contribution_record = {
        "entry_id": entry_id,
        "model_name": model_name.strip(),
        "category": "community",
        "spatial_regime": leaderboard_id.split("__", 1)[0],
        "forcing_regime": leaderboard_id.split("__", 1)[1],
        "forcings_used": forcing.strip(),
        "is_ensemble": is_ensemble,
        "num_members": n_members,
        "has_qsim_members": scored["has_qsim_members"],
        "ensemble_source": scored["ensemble_source"],
        "ensemble_reduction": scored["ensemble_reduction"],
        "single_seed_member_index": scored["single_seed_member_index"],
        "metrics": scored["summary"],
        "hydrograph_path": hydrograph_path,
        "code_url": gh_clean,
        "data_url": z_url or hs_clean,
        "target_leaderboard_tables": [leaderboard_id],
    }

    upsert_leaderboard_entry(
        leaderboard_id=leaderboard_id,
        entry=new_entry,
        family_record=family_record,
        contribution_record=contribution_record,
        leaderboards_dir=lb_dir,
    )

    return {
        "status": "ok",
        "entry": new_entry,
        "hydrograph_path": hydrograph_path,
        "scoring": scored,
    }


@app.get("/api/template/{kind}")
def api_download_template(kind: str) -> Response:
    """Download protocol or canonical fold split JSON templates."""
    if kind in ("protocol.json", "folds.json"):
        splits_proto = load_splits_protocol()
        filename = (
            "hydromip_camels531_protocol.json"
            if kind == "protocol.json"
            else "hydromip_camels531_canonical_folds.json"
        )
        return Response(
            content=json.dumps(splits_proto, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if kind == "sample.npz":
        splits_proto = load_splits_protocol()
        basins = sorted(splits_proto["basin_to_huc02"].keys())
        dates = pd.date_range(NEWMAN_START_DATE, NEWMAN_END_DATE, freq="D").strftime("%Y-%m-%d").tolist()
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            basin=np.array(basins, dtype="U8"),
            date=np.array(dates, dtype="U10"),
            qsim=np.zeros((NEWMAN_N_BASINS, NEWMAN_N_DAYS), dtype=np.float32),
        )
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="hydromip_camels531_sample_submission.npz"'},
        )

    raise HTTPException(
        status_code=404,
        detail=f"Unknown template '{kind}'. Use 'protocol.json', 'folds.json', or 'sample.npz'.",
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the main leaderboard application page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/styles.css")
def serve_styles() -> FileResponse:
    """Serve top-level `/styles.css` stylesheet."""
    css_path = STATIC_DIR / "styles.css"
    if not css_path.exists():
        raise HTTPException(status_code=404, detail="styles.css not found.")
    return FileResponse(str(css_path), media_type="text/css")


@app.get("/app.js")
def serve_app_js() -> FileResponse:
    """Serve top-level `/app.js` frontend script."""
    js_path = STATIC_DIR / "app.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="app.js not found.")
    return FileResponse(str(js_path), media_type="application/javascript")
