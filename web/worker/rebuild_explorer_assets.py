# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild canonical 531-basin Hydrograph Explorer assets from the 60 R2 Zarr stores.

Zero ``try``/``except`` blocks. Reads ``tables_3_to_14.json`` and ``manifest_index.json``,
loads ``newman_eval_targets.zarr`` and the 60 unique Zarr stores directly from R2 (via
``s3fs`` or HTTPS) or a local bucket directory, maps every series strictly by
``<leaderboard_id>__<entry_id>`` using only each row's declared ``zarr_path`` (never
colliding on filename stems such as Literature ``t3_21``), computes canonical per-basin
metrics via ``hydromip.metrics.deterministic_daily``, writes 531 ``hydro_assets/basin_*.json``
files plus ``hydro_assets/index.json``, packages ``hydro_assets_531.tar.gz``, and
optionally uploads it to Cloudflare R2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import tarfile
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import botocore.session
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from hydromip.metrics.deterministic_daily import (
    fdc_fhv,
    fdc_flv,
    kge,
    log_kge,
    log_nse,
    nse,
    pearsonr,
    rmse,
)
from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths

DEFAULT_MIP_DIR: Path = resolve_mip_paths(DEFAULT_MIP_ID).root
DEFAULT_R2_PUBLIC_BASE: str = "https://data.hydrologymip.org/camels_us_daily"
DEFAULT_R2_ENV_PATH: Path = Path.home() / ".config" / "hydrologymip" / "r2.env"
DEFAULT_BASIN_META_PATH: Path = Path(__file__).resolve().with_name("basin_map_metadata.json")

EXPECTED_NUM_BASINS: int = 531
EXPECTED_NUM_DAYS: int = 3652
MIN_VALID_OBS_DAYS: int = 30
EVAL_START_DATE: str = "1989-10-01"
EVAL_END_DATE: str = "1999-09-30"

REQUIRED_BASIN_METRIC_KEYS: tuple[str, ...] = (
    "nse",
    "kge",
    "logNSE",
    "logKGE",
    "pearson_r",
    "fhv",
    "flv",
    "rmse",
)

REQUIRED_NEWLY_INCLUDED_MODELS: tuple[str, ...] = (
    "gauged__single_camels__shahriar2026_mamba3_daymet",
    "gauged__lagged_streamflow__naive_lag1_persistence_gauged",
    "pub__lagged_streamflow__naive_lag1_persistence_pub",
    "pur__lagged_streamflow__naive_lag1_persistence_pur",
    "pub__lagged_streamflow__feng2020_di_lstm_pub_nldas",
    "pub__lagged_streamflow__feng2020_di_lstm_pub_nldas_single",
    "pur__lagged_streamflow__feng2020_di_lstm_pur_nldas",
    "pur__lagged_streamflow__feng2020_di_lstm_pur_nldas_single",
    "gauged__multi_camels__t4_24",
)

RERUN_PARITY_CHECK_MODELS: tuple[str, ...] = (
    "gauged__single_camels__tft_gauged_maurer",
    "gauged__lagged_streamflow__yin2022_rrformer_daymet",
    "gauged__single_camels__poudel_mlp_hbv_daymet",
    "pub__single_camels__poudel_mlp_hbv_pub_daymet",
    "pur__single_camels__poudel_mlp_hbv_pur_daymet",
)


def load_r2_credentials(r2_env_path: Path | None) -> dict[str, str]:
    """Load R2 S3 credentials from env vars or an optional ``r2.env`` file."""
    creds: dict[str, str] = {}
    if r2_env_path is not None and r2_env_path.is_file():
        for raw_line in r2_env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            creds[key] = v.strip().strip("'\"")
    for env_key in (
        "R2_ENDPOINT",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "R2_PUBLIC_BASE_URL",
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            creds[env_key] = val
    return creds


def build_r2_s3_filesystem(creds: dict[str, str]) -> s3fs.S3FileSystem:
    """Create a Cloudflare R2 ``s3fs.S3FileSystem`` from ``creds``."""
    for req_key in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        if not creds.get(req_key):
            raise ValueError(f"Missing required R2 credential key {req_key!r}.")
    return s3fs.S3FileSystem(
        key=creds["R2_ACCESS_KEY_ID"],
        secret=creds["R2_SECRET_ACCESS_KEY"],
        client_kwargs={"endpoint_url": creds["R2_ENDPOINT"], "region_name": "auto"},
    )


def zarr_uri_to_rel_path(zarr_uri: str) -> str:
    """Extract canonical subpath like ``hydrographs/<lb>/<store>.zarr`` from a gs:// or https:// URI."""
    clean = str(zarr_uri).strip().rstrip("/")
    idx = clean.find("hydrographs/")
    if idx < 0:
        raise ValueError(f"Cannot extract 'hydrographs/' relative path from {zarr_uri!r}.")
    return clean[idx:]


def open_canonical_zarr(
    rel_path: str,
    *,
    s3_fs: s3fs.S3FileSystem | None,
    r2_bucket: str,
    local_bucket_dir: Path | None,
    r2_public_base: str,
) -> xr.Dataset:
    """Open a consolidated Zarr v2 store from local dir, R2 S3 API, or public HTTPS."""
    clean_rel = rel_path.lstrip("/")
    if local_bucket_dir is not None:
        return xr.open_zarr(local_bucket_dir / clean_rel, consolidated=True)
    if s3_fs is not None:
        s3_map = s3fs.S3Map(f"{r2_bucket}/camels_us_daily/{clean_rel}", s3=s3_fs)
        return xr.open_zarr(s3_map, consolidated=True)
    https_url = f"{r2_public_base.rstrip('/')}/{clean_rel}"
    return xr.open_zarr(https_url, consolidated=True)


def fetch_manifest_index(
    *,
    s3_fs: s3fs.S3FileSystem | None,
    r2_bucket: str,
    local_bucket_dir: Path | None,
    r2_public_base: str,
) -> dict[str, Any]:
    """Load ``hydrographs/manifest_index.json`` from local bucket, R2 S3, or public HTTPS."""
    rel = "hydrographs/manifest_index.json"
    if local_bucket_dir is not None and (local_bucket_dir / rel).is_file():
        return json.loads((local_bucket_dir / rel).read_text(encoding="utf-8"))
    if s3_fs is not None:
        with s3_fs.open(f"{r2_bucket}/camels_us_daily/{rel}", "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    url = f"{r2_public_base.rstrip('/')}/{rel}"
    req = urllib.request.Request(url, headers={"User-Agent": "HydrologyMIP-ExplorerBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_list_round(vec: np.ndarray, dec: int = 2) -> list[float | None]:
    """Round 1-D values to ``dec`` decimals, converting non-finite values to ``None``."""
    rounded = np.round(vec.astype(np.float64), dec)
    finite = np.isfinite(rounded)
    if np.all(finite):
        return rounded.tolist()
    return [float(x) if f else None for x, f in zip(rounded.tolist(), finite.tolist())]


def compute_canonical_basin_metrics(
    qobs_2d: np.ndarray,
    qsim_2d: np.ndarray,
) -> list[dict[str, float | None]]:
    """Compute canonical per-basin metrics matching ``hydromip.metrics.deterministic_daily``."""
    n_basins = int(qobs_2d.shape[0])
    out: list[dict[str, float | None]] = []
    empty_metrics: dict[str, float | None] = {k: None for k in REQUIRED_BASIN_METRIC_KEYS}

    for b_idx in range(n_basins):
        qo = qobs_2d[b_idx]
        qs = qsim_2d[b_idx]
        mask = np.isfinite(qo) & np.isfinite(qs) & (qo >= 0.0)
        if int(np.count_nonzero(mask)) < MIN_VALID_OBS_DAYS:
            out.append(dict(empty_metrics))
            continue

        o_da = xr.DataArray(np.where(mask, qo, np.nan).astype(np.float64), dims=("date",))
        s_da = xr.DataArray(np.where(mask, qs, np.nan).astype(np.float64), dims=("date",))

        raw_vals = {
            "nse": (float(nse(o_da, s_da)), 4),
            "kge": (float(kge(o_da, s_da)), 4),
            "logNSE": (float(log_nse(o_da, s_da)), 4),
            "logKGE": (float(log_kge(o_da, s_da)), 4),
            "pearson_r": (float(pearsonr(o_da, s_da)), 4),
            "fhv": (float(fdc_fhv(o_da, s_da)), 2),
            "flv": (float(fdc_flv(o_da, s_da)), 2),
            "rmse": (float(rmse(o_da, s_da)), 4),
        }
        out.append(
            {
                key: (round(val, digits) if np.isfinite(val) else None)
                for key, (val, digits) in raw_vals.items()
            }
        )
    return out


def extract_forcings_from_bundle(
    bundle_path: Path,
    canonical_basins: Sequence[str],
    eval_start: str = EVAL_START_DATE,
    eval_end: str = EVAL_END_DATE,
) -> dict[str, dict[str, np.ndarray]]:
    """Load aligned `(531, 3652)` Daymet, Maurer, and NLDAS forcings from ``camels_671_1980_2014.npz``."""
    with np.load(bundle_path, allow_pickle=False) as d:
        bundle_basins = [str(b).strip().zfill(8) for b in d["basins"].tolist()]
        b_pos = {b: i for i, b in enumerate(bundle_basins)}
        missing_b = [b for b in canonical_basins if b not in b_pos]
        if missing_b:
            raise ValueError(f"Bundle {bundle_path} is missing basins: {missing_b[:5]}")
        b_indices = np.asarray([b_pos[b] for b in canonical_basins], dtype=np.int64)

        bundle_dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in d["dates"].tolist()]
        if eval_start not in bundle_dates or eval_end not in bundle_dates:
            raise ValueError(f"Bundle {bundle_path} does not cover {eval_start}..{eval_end}.")
        d0 = bundle_dates.index(eval_start)
        d1 = bundle_dates.index(eval_end) + 1
        if d1 - d0 != EXPECTED_NUM_DAYS:
            raise ValueError(f"Expected {EXPECTED_NUM_DAYS} days in slice, got {d1 - d0}.")

        var_names = [str(v) for v in d["forcing_variable_names"].tolist()]
        i_prcp = var_names.index("prcp(mm/day)")
        i_tmax = var_names.index("tmax(C)")
        i_tmin = var_names.index("tmin(C)")

        forcings_out: dict[str, dict[str, np.ndarray]] = {}
        for prod, key in (("daymet", "forcing_daymet"), ("maurer", "forcing_maurer"), ("nldas", "forcing_nldas")):
            arr = np.asarray(d[key][b_indices, d0:d1, :], dtype=np.float32)
            tmax = arr[:, :, i_tmax]
            tmin = arr[:, :, i_tmin]
            forcings_out[prod] = {
                "prcp": arr[:, :, i_prcp],
                "tmax": tmax,
                "tmin": tmin,
                "tmean": (0.5 * (tmax + tmin)).astype(np.float32),
            }
        return forcings_out


def verify_rebuilt_assets(
    hydro_dir: Path,
    tables_doc: dict[str, Any],
    basins: Sequence[str],
) -> dict[str, Any]:
    """Run strict parity and completeness checks on rebuilt ``hydro_assets/``."""
    idx_doc = json.loads((hydro_dir / "index.json").read_text(encoding="utf-8"))
    keys = set(idx_doc.get("available_qsim_keys", []))

    if "gauged__single_camels__t3_21" in keys:
        raise AssertionError("Literature entry gauged__single_camels__t3_21 (S4D-FT) leaked into index.json!")

    missing_required = [mk for mk in REQUIRED_NEWLY_INCLUDED_MODELS if mk not in keys]
    if missing_required:
        raise AssertionError(f"Missing required empirical models in rebuilt index: {missing_required}")

    row_by_key = {
        f"{lb['id']}__{r['entry_id']}": r
        for lb in tables_doc.get("leaderboards", [])
        for r in lb.get("rows", [])
    }

    expected_keys = {
        f"{lb['id']}__{r['entry_id']}"
        for lb in tables_doc.get("leaderboards", [])
        for r in lb.get("rows", [])
        if str(r.get("zarr_path") or "").strip()
    }
    if keys != expected_keys:
        raise AssertionError(
            f"available_qsim_keys mismatch: extra={sorted(keys - expected_keys)}, "
            f"missing={sorted(expected_keys - keys)}"
        )

    max_rerun_drift = 0.0
    worst_rerun: tuple[str, str, float, float] | None = None
    metric_key_set = set(REQUIRED_BASIN_METRIC_KEYS)

    for b_idx, bid in enumerate(basins):
        b_path = hydro_dir / f"basin_{bid}.json"
        if not b_path.is_file():
            raise AssertionError(f"Missing {b_path}")
        b_doc = json.loads(b_path.read_text(encoding="utf-8"))
        b_models = b_doc.get("models") or {}
        if set(b_models.keys()) != expected_keys:
            raise AssertionError(f"basin_{bid}.json model key set mismatch.")

        for mk in REQUIRED_NEWLY_INCLUDED_MODELS:
            series = b_models[mk]["qsim"]
            if len(series) != EXPECTED_NUM_DAYS or any(v is None for v in series):
                raise AssertionError(f"Model {mk} on basin {bid} has invalid/incomplete qsim series.")

        for mk, entry in b_models.items():
            bm = entry.get("basin_metrics") or {}
            if set(bm.keys()) != metric_key_set:
                raise AssertionError(f"{mk} on basin {bid} has unexpected metric keys {sorted(bm.keys())}")
            if mk in RERUN_PARITY_CHECK_MODELS:
                expected_pb = row_by_key[mk].get("per_basin_nse") or []
                exp_val = expected_pb[b_idx]
                got_val = bm.get("nse")
                if exp_val is not None and got_val is not None:
                    diff = abs(float(got_val) - float(exp_val))
                    if diff > max_rerun_drift:
                        max_rerun_drift = diff
                        worst_rerun = (mk, bid, float(got_val), float(exp_val))
                    if diff > 0.002:
                        raise AssertionError(
                            f"Stale drift for {mk} on basin {bid}: got {got_val} vs table {exp_val} (diff={diff:.4f})"
                        )

    return {
        "n_basins": len(basins),
        "n_models": len(expected_keys),
        "max_rerun_nse_drift": round(max_rerun_drift, 6),
        "worst_rerun_pair": worst_rerun,
        "s4dft_absent": "gauged__single_camels__t3_21" not in keys,
        "all_8_empirical_present": not missing_required,
    }


def rebuild_explorer_assets(
    *,
    mip_dir: Path = DEFAULT_MIP_DIR,
    out_dir: Path,
    tar_out: Path | None = None,
    camels_bundle_path: Path | None = None,
    forcings_from_dir: Path | None = None,
    r2_env_path: Path | None = DEFAULT_R2_ENV_PATH,
    r2_public_base: str = DEFAULT_R2_PUBLIC_BASE,
    local_bucket_dir: Path | None = None,
    basin_meta_path: Path = DEFAULT_BASIN_META_PATH,
    upload_r2: bool = False,
) -> dict[str, Any]:
    """Rebuild all 531 basin JSONs and ``index.json`` strictly from the 60 canonical Zarr stores."""
    tables_path = mip_dir / "leaderboards" / "tables_3_to_14.json"
    tables_doc = json.loads(tables_path.read_text(encoding="utf-8"))
    meta_doc = json.loads(basin_meta_path.read_text(encoding="utf-8"))
    basins_meta: list[dict[str, Any]] = list(meta_doc["basins"])
    basins = [str(b["basin_id"]).zfill(8) for b in basins_meta]
    if len(basins) != EXPECTED_NUM_BASINS or basins != sorted(basins):
        raise ValueError("Canonical basin_map_metadata.json does not contain 531 sorted basins.")

    creds = load_r2_credentials(r2_env_path) if local_bucket_dir is None else {}
    s3_fs: s3fs.S3FileSystem | None = None
    r2_bucket = creds.get("R2_BUCKET", "hydrologymip-data")
    if local_bucket_dir is None and creds.get("R2_ACCESS_KEY_ID"):
        s3_fs = build_r2_s3_filesystem(creds)

    manifest = fetch_manifest_index(
        s3_fs=s3_fs,
        r2_bucket=r2_bucket,
        local_bucket_dir=local_bucket_dir,
        r2_public_base=r2_public_base,
    )
    manifest_store_paths = {zarr_uri_to_rel_path(s["zarr_uri"]) for s in manifest.get("stores", [])}

    ds_targets = open_canonical_zarr(
        "observations/newman_eval_targets.zarr",
        s3_fs=s3_fs,
        r2_bucket=r2_bucket,
        local_bucket_dir=local_bucket_dir,
        r2_public_base=r2_public_base,
    )
    target_basins = [str(b).strip().zfill(8) for b in ds_targets["basin"].values.tolist()]
    if target_basins != basins:
        raise ValueError("Target Zarr basins do not match canonical sorted 531 basins.")
    target_dates = pd.to_datetime(ds_targets["date"].values).strftime("%Y-%m-%d").tolist()
    expected_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(EVAL_START_DATE, EVAL_END_DATE, freq="D")]
    if target_dates != expected_dates:
        raise ValueError("Target Zarr dates do not match canonical 1989-10-01..1999-09-30.")
    qobs_2d = np.asarray(ds_targets["qobs"].values, dtype=np.float64)
    if qobs_2d.shape != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
        raise ValueError(f"Unexpected qobs shape {qobs_2d.shape}.")

    # Group zarr-backed leaderboard rows by canonical store path (never by archive_key stem).
    store_rows: dict[str, list[dict[str, Any]]] = {}
    for lb in tables_doc.get("leaderboards", []):
        lbid = str(lb["id"])
        for row in lb.get("rows", []):
            zp = str(row.get("zarr_path") or "").strip()
            if not zp:
                continue
            rel_store = zarr_uri_to_rel_path(zp)
            eid = str(row["entry_id"]).strip()
            store_rows.setdefault(rel_store, []).append(
                {
                    "model_key": f"{lbid}__{eid}",
                    "leaderboard_id": lbid,
                    "entry_id": eid,
                    "is_ensemble": bool(row.get("is_ensemble")),
                    "num_members": int(row.get("num_members") or 1),
                    "category": str(row.get("category", "")),
                }
            )

    if not store_rows:
        raise ValueError("No Zarr stores found in tables_3_to_14.json.")
    if set(store_rows.keys()) != manifest_store_paths:
        raise ValueError(
            f"Leaderboard store paths differ from manifest_index.json: "
            f"extra={set(store_rows) - manifest_store_paths}, missing={manifest_store_paths - set(store_rows)}"
        )

    bundle_path = camels_bundle_path or (mip_dir / "data" / "camels_671_1980_2014.npz")
    forcings_arrays: dict[str, dict[str, np.ndarray]] | None = None
    if bundle_path.is_file():
        forcings_arrays = extract_forcings_from_bundle(bundle_path, basins)
    elif forcings_from_dir is None:
        raise FileNotFoundError(
            f"Missing CAMELS bundle {bundle_path}; pass --camels-bundle or --forcings-from."
        )

    model_series: dict[str, np.ndarray] = {}
    model_metrics: dict[str, list[dict[str, float | None]]] = {}
    sorted_stores = sorted(store_rows.keys())
    for s_idx, rel_store in enumerate(sorted_stores, start=1):
        rows_for_store = store_rows[rel_store]
        print(f"[{s_idx}/{len(sorted_stores)}] Reading {rel_store} ({len(rows_for_store)} row(s))...", flush=True)
        ds = open_canonical_zarr(
            rel_store,
            s3_fs=s3_fs,
            r2_bucket=r2_bucket,
            local_bucket_dir=local_bucket_dir,
            r2_public_base=r2_public_base,
        )
        ds_basins = [str(b).strip().zfill(8) for b in ds["basin"].values.tolist()]
        if ds_basins != basins:
            raise ValueError(f"Basin mismatch in {rel_store}.")
        ds_dates = pd.to_datetime(ds["date"].values).strftime("%Y-%m-%d").tolist()
        if ds_dates != expected_dates:
            raise ValueError(f"Date mismatch in {rel_store}.")

        qsim_ens: np.ndarray | None = None
        qsim_member0: np.ndarray | None = None
        for rspec in rows_for_store:
            mk = rspec["model_key"]
            if rspec["is_ensemble"]:
                if qsim_ens is None:
                    qsim_ens = np.asarray(ds["qsim"].values, dtype=np.float32)
                arr = qsim_ens
            elif "qsim_members" in ds.data_vars and int(ds.sizes.get("member", 0)) >= 1:
                if qsim_member0 is None:
                    qsim_member0 = np.asarray(ds["qsim_members"].isel(member=0).values, dtype=np.float32)
                arr = qsim_member0
            else:
                if qsim_ens is None:
                    qsim_ens = np.asarray(ds["qsim"].values, dtype=np.float32)
                arr = qsim_ens

            if arr.shape != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
                raise ValueError(f"Unexpected array shape {arr.shape} for {mk} in {rel_store}.")
            model_series[mk] = arr
            model_metrics[mk] = compute_canonical_basin_metrics(qobs_2d, arr)

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_model_keys = sorted(model_series.keys())
    for b_idx, b_meta in enumerate(basins_meta):
        bid = str(b_meta["basin_id"]).zfill(8)
        if forcings_arrays is not None:
            forcings_payload = {
                prod: {
                    "label": {"daymet": "Daymet", "maurer": "Maurer", "nldas": "NLDAS"}[prod],
                    "prcp": to_list_round(forcings_arrays[prod]["prcp"][b_idx], 2),
                    "tmax": to_list_round(forcings_arrays[prod]["tmax"][b_idx], 2),
                    "tmin": to_list_round(forcings_arrays[prod]["tmin"][b_idx], 2),
                    "tmean": to_list_round(forcings_arrays[prod]["tmean"][b_idx], 2),
                }
                for prod in ("daymet", "maurer", "nldas")
            }
        else:
            assert forcings_from_dir is not None
            prior_doc = json.loads((forcings_from_dir / f"basin_{bid}.json").read_text(encoding="utf-8"))
            forcings_payload = prior_doc["forcings"]

        models_payload = {
            mk: {
                "qsim": to_list_round(model_series[mk][b_idx], 2),
                "basin_metrics": model_metrics[mk][b_idx],
            }
            for mk in sorted_model_keys
        }
        basin_doc = {
            "basin": b_meta,
            "qobs": to_list_round(qobs_2d[b_idx], 2),
            "forcings": forcings_payload,
            "models": models_payload,
        }
        (out_dir / f"basin_{bid}.json").write_text(
            json.dumps(basin_doc, separators=(",", ":")),
            encoding="utf-8",
        )

    index_doc = {
        "n_basins": len(basins_meta),
        "n_models": len(sorted_model_keys),
        "n_stores": len(sorted_stores),
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basins": basins_meta,
        "available_qsim_keys": sorted_model_keys,
    }
    (out_dir / "index.json").write_text(
        json.dumps(index_doc, separators=(",", ":")),
        encoding="utf-8",
    )

    verification = verify_rebuilt_assets(out_dir, tables_doc, basins)

    tar_bytes = 0
    if tar_out is not None:
        tar_out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_out, "w:gz") as tf:
            for file_path in sorted(out_dir.iterdir()):
                if not file_path.is_file():
                    continue
                data = file_path.read_bytes()
                ti = tarfile.TarInfo(f"hydro_assets/{file_path.name}")
                ti.size = len(data)
                ti.mtime = 0
                ti.mode = 0o644
                tf.addfile(ti, io.BytesIO(data))
        tar_bytes = tar_out.stat().st_size

        if upload_r2:
            if not creds.get("R2_ACCESS_KEY_ID"):
                raise ValueError("--upload-r2 requires R2 credentials via --r2-env or environment variables.")
            r2_client = botocore.session.get_session().create_client(
                "s3",
                endpoint_url=creds["R2_ENDPOINT"],
                aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
            r2_key = "camels_us_daily/portal_cache/hydro_assets_531.tar.gz"
            with tar_out.open("rb") as fh:
                r2_client.put_object(
                    Bucket=r2_bucket,
                    Key=r2_key,
                    Body=fh,
                    ContentType="application/gzip",
                    CacheControl="public, max-age=300",
                )

    return {
        "out_dir": str(out_dir),
        "tar_out": str(tar_out) if tar_out is not None else None,
        "tar_bytes": tar_bytes,
        "uploaded_r2": bool(upload_r2 and tar_out is not None),
        **verification,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for rebuilding the 531-basin Hydrograph Explorer assets."""
    parser = argparse.ArgumentParser(description="Rebuild HydrologyMIP 531-basin Explorer assets from 60 R2 Zarr stores.")
    parser.add_argument("--mip-dir", type=Path, default=DEFAULT_MIP_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for hydro_assets JSONs")
    parser.add_argument("--tar-out", type=Path, default=None, help="Optional output path for hydro_assets_531.tar.gz")
    parser.add_argument("--camels-bundle", type=Path, default=None)
    parser.add_argument("--forcings-from", type=Path, default=None)
    parser.add_argument("--r2-env", type=Path, default=DEFAULT_R2_ENV_PATH)
    parser.add_argument("--r2-public-base", type=str, default=DEFAULT_R2_PUBLIC_BASE)
    parser.add_argument("--local-bucket-dir", type=Path, default=None)
    parser.add_argument("--upload-r2", action="store_true")
    args = parser.parse_args(argv)

    if args.out_dir is not None:
        summary = rebuild_explorer_assets(
            mip_dir=args.mip_dir,
            out_dir=args.out_dir,
            tar_out=args.tar_out,
            camels_bundle_path=args.camels_bundle,
            forcings_from_dir=args.forcings_from,
            r2_env_path=args.r2_env,
            r2_public_base=args.r2_public_base,
            local_bucket_dir=args.local_bucket_dir,
            upload_r2=bool(args.upload_r2),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="hmip_explorer_") as tmp_root:
            summary = rebuild_explorer_assets(
                mip_dir=args.mip_dir,
                out_dir=Path(tmp_root) / "hydro_assets",
                tar_out=args.tar_out,
                camels_bundle_path=args.camels_bundle,
                forcings_from_dir=args.forcings_from,
                r2_env_path=args.r2_env,
                r2_public_base=args.r2_public_base,
                local_bucket_dir=args.local_bucket_dir,
                upload_r2=bool(args.upload_r2),
            )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
