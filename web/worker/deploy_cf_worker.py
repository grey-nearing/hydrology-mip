# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standalone static bundle builder and Cloudflare Workers deployer for HydrologyMIP.

Reads all leaderboard tables, model families, splits, and figures directly from
``mips/camels_us_daily/`` and static portal assets from ``web/portal/``.

Zero ``try``/``except`` blocks. Reads Cloudflare credentials strictly from
``CLOUDFLARE_ACCOUNT_ID`` / ``CLOUDFLARE_API_TOKEN`` or ``~/.config/hydrologymip/cloudflare.env``.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from hydromip.io.doi_resolver import validate_and_fetch_archive_url
from hydromip.protocol import discover_mips, get_mip

REPO_ROOT = Path(__file__).resolve().parents[2]
MIP_DIR = REPO_ROOT / "mips" / "camels_us_daily"
PORTAL_DIR = REPO_ROOT / "web" / "portal"
WORKER_DIR = Path(__file__).resolve().parent
BASIN_MAP_META_PATH = WORKER_DIR / "basin_map_metadata.json"
DEFAULT_STAGE_DIR = WORKER_DIR / ".cf_staging"
CF_SCRIPT_NAME = "hydrologymip-router"
CANONICAL_HOST = "hydrologymip.org"
PUBLIC_R2_BASE = "https://data.hydrologymip.org/camels_us_daily"

PRIVATE_URL_MARKERS = (
    "gs://camels-us-mip",
    "console.cloud.google.com/storage/browser/camels-us-mip",
    "storage.googleapis.com/camels-us-mip",
)

PRIVATE_INLINE_PATTERNS = [
    re.compile(r"gs://camels-us-mip[^\s,;)\"']*"),
    re.compile(r"https?://console\.cloud\.google\.com/storage/browser/camels-us-mip[^\s,;)\"']*"),
    re.compile(r"https?://storage\.googleapis\.com/camels-us-mip[^\s,;)\"']*"),
]

URL_FIELDS = frozenset(
    {
        "code_url",
        "github_url",
        "paper_url",
        "data_archive_url",
        "zarr_path",
        "zenodo_url",
        "hydroshare_url",
        "repository_url",
        "manuscript_url",
    }
)

MIME_BY_EXT = {
    "html": "text/html",
    "css": "text/css",
    "js": "application/javascript",
    "json": "application/json",
    "txt": "text/plain",
    "xml": "application/xml",
    "svg": "image/svg+xml",
    "ico": "image/x-icon",
    "png": "image/png",
}

DEFAULT_METRICS_ORDER = [
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
]


def _is_truthy(val: str | None, *, default: bool = False) -> bool:
    if val is None or not val.strip():
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class _NonRaisingHTTPErrorProcessor(urllib.request.HTTPErrorProcessor):
    """Return HTTP error responses cleanly so uploads retry without try/except."""

    def http_response(self, request: Any, response: Any) -> Any:
        return response

    https_response = http_response


def load_cloudflare_credentials() -> tuple[str, str]:
    """Load Cloudflare account ID and API token from env or ~/.config/hydrologymip/cloudflare.env."""
    env_vals: dict[str, str] = {}
    env_file = Path.home() / ".config" / "hydrologymip" / "cloudflare.env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env_vals[k.strip()] = v.strip().strip('"').strip("'")

    account_id = (
        os.environ.get("CLOUDFLARE_ACCOUNT_ID") or env_vals.get("CLOUDFLARE_ACCOUNT_ID", "")
    ).strip()
    api_token = (
        os.environ.get("CLOUDFLARE_API_TOKEN") or env_vals.get("CLOUDFLARE_API_TOKEN", "")
    ).strip()
    if not account_id or not api_token:
        raise RuntimeError(
            "Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN. "
            f"Set them in the environment or in {env_file}."
        )
    return account_id, api_token


def is_private_url(val: str) -> bool:
    low = val.lower()
    return any(m.lower() in low for m in PRIVATE_URL_MARKERS)


def scrub_private_inline_text(text: str) -> str:
    out = text
    for pat in PRIVATE_INLINE_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r";\s*\)", ")", out)
    out = re.sub(r"\(\s*;\s*", "(", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def rewrite_gcs_to_r2(obj: Any) -> Any:
    """Recursively rewrite private ``gs://camels-us-mip`` URIs to the public R2 mirror."""
    if isinstance(obj, dict):
        return {k: rewrite_gcs_to_r2(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_gcs_to_r2(v) for v in obj]
    if isinstance(obj, str):
        out = obj.replace("gs://camels-us-mip/", f"{PUBLIC_R2_BASE}/")
        out = out.replace("gs://camels-us-mip", PUBLIC_R2_BASE)
        return out
    return obj


def sanitize_public_data(obj: Any) -> Any:
    """Recursively rewrite GCS URIs to R2 and strip private GCS console URLs."""
    rewritten = rewrite_gcs_to_r2(obj)
    if isinstance(rewritten, dict):
        cleaned = {}
        for k, v in rewritten.items():
            if isinstance(v, str):
                if k in URL_FIELDS and is_private_url(v):
                    cleaned[k] = ""
                elif is_private_url(v):
                    cleaned[k] = scrub_private_inline_text(v)
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = sanitize_public_data(v)
        return cleaned
    if isinstance(rewritten, list):
        return [sanitize_public_data(item) for item in rewritten]
    if isinstance(rewritten, str) and is_private_url(rewritten):
        return scrub_private_inline_text(rewritten)
    return rewritten


def _finite_or_none(val: Any) -> float | None:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        return None
    f = float(val)
    if not np.isfinite(f):
        return None
    return f


def _attach_flat_gauch_metrics(row: dict[str, Any]) -> None:
    """Attach flat numeric keys for all 14 Gauch et al. (2023) metrics (+ logNSE/logKGE)."""
    gauch = row.get("gauch_2023_metrics")
    g = gauch if isinstance(gauch, dict) else {}

    def _gmed(metric_key: str) -> float | None:
        m_obj = g.get(metric_key)
        return _finite_or_none(m_obj.get("median")) if isinstance(m_obj, dict) else None

    def _gmean(metric_key: str) -> float | None:
        m_obj = g.get(metric_key)
        return _finite_or_none(m_obj.get("mean")) if isinstance(m_obj, dict) else None

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
    if _finite_or_none(row.get("mean_kge")) is None and _gmean("KGE") is not None:
        row["mean_kge"] = _gmean("KGE")

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


def compute_fast_basin_metrics_matrix(
    qobs_2d: np.ndarray, qsim_2d: np.ndarray
) -> list[dict[str, float | None]]:
    """Compute per-basin NSE, KGE, Pearson_r, FHV, FLV, RMSE across 531 basins rapidly."""
    n_basins = qobs_2d.shape[0]
    out: list[dict[str, float | None]] = []
    for i in range(n_basins):
        qo = qobs_2d[i]
        qs = qsim_2d[i]
        mask = np.isfinite(qo) & np.isfinite(qs) & (qo >= 0.0)
        if np.count_nonzero(mask) < 30:
            out.append(
                {
                    "nse": None,
                    "kge": None,
                    "pearson_r": None,
                    "fhv": None,
                    "flv": None,
                    "rmse": None,
                }
            )
            continue
        o = qo[mask].astype(np.float64)
        s = qs[mask].astype(np.float64)
        diff = s - o
        mse = float(np.mean(diff * diff))
        rmse = float(np.sqrt(mse))
        o_mean = float(np.mean(o))
        s_mean = float(np.mean(s))
        o_dev = o - o_mean
        sst = float(np.sum(o_dev * o_dev))
        nse_val = 1.0 - float(np.sum(diff * diff)) / sst if sst > 1e-12 else float("nan")
        o_std = float(np.std(o))
        s_std = float(np.std(s))
        if o_std > 1e-12 and s_std > 1e-12 and o_mean > 1e-12:
            r_val = float(np.mean((o - o_mean) * (s - s_mean)) / (o_std * s_std))
            alpha = s_std / o_std
            beta = s_mean / o_mean
            kge_val = 1.0 - float(np.sqrt((r_val - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
        else:
            r_val = float("nan")
            kge_val = float("nan")

        o_sorted = np.sort(o)[::-1]
        s_sorted = np.sort(s)[::-1]
        n_obs = len(o_sorted)
        idx_hi = max(1, int(np.round(0.02 * n_obs)))
        o_hi_sum = float(np.sum(o_sorted[:idx_hi]))
        fhv_val = (
            100.0 * (float(np.sum(s_sorted[:idx_hi])) - o_hi_sum) / o_hi_sum
            if o_hi_sum > 1e-12
            else float("nan")
        )

        idx_lo_start = max(1, int(np.round(0.70 * n_obs)))
        o_low = o_sorted[idx_lo_start:]
        s_low = s_sorted[idx_lo_start:]
        pos_mask = (o_low > 1e-6) & (s_low > 1e-6)
        if np.count_nonzero(pos_mask) >= 10:
            lo_log = np.log(o_low[pos_mask])
            ls_log = np.log(s_low[pos_mask])
            o_min_l = float(lo_log[-1])
            s_min_l = float(ls_log[-1])
            denom_l = float(np.sum(lo_log - o_min_l))
            flv_val = (
                -100.0 * (float(np.sum(ls_log - s_min_l)) - denom_l) / denom_l
                if abs(denom_l) > 1e-9
                else float("nan")
            )
        else:
            flv_val = float("nan")

        out.append(
            {
                "nse": round(nse_val, 3) if np.isfinite(nse_val) else None,
                "kge": round(kge_val, 3) if np.isfinite(kge_val) else None,
                "pearson_r": round(r_val, 3) if np.isfinite(r_val) else None,
                "fhv": round(fhv_val, 1) if np.isfinite(fhv_val) else None,
                "flv": round(flv_val, 1) if np.isfinite(flv_val) else None,
                "rmse": round(rmse, 3) if np.isfinite(rmse) else None,
            }
        )
    return out


def build_portal_json_payloads(
    mip_dir: Path,
    explorer_store_path: Path | None = None,
    *,
    stage_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Construct standalone overview, leaderboards, families, and hydrographs index JSONs."""
    lb_dir = mip_dir / "leaderboards"
    splits_dir = mip_dir / "splits"
    t3_path = lb_dir / "tables_3_to_14.json"
    t2_path = lb_dir / "table_2_registry.json"
    if not t3_path.is_file() or not t2_path.is_file():
        raise FileNotFoundError(
            f"Missing canonical leaderboard files in {lb_dir}: tables_3_to_14.json / table_2_registry.json"
        )

    t3_doc = rewrite_gcs_to_r2(json.loads(t3_path.read_text(encoding="utf-8")))
    t2_doc = rewrite_gcs_to_r2(json.loads(t2_path.read_text(encoding="utf-8")))
    families_raw = list(t2_doc.get("families") or [])
    families_by_id = {str(f["family_id"]): f for f in families_raw}

    basin_hucs_531: list[str] = []
    if BASIN_MAP_META_PATH.is_file():
        bm_meta = json.loads(BASIN_MAP_META_PATH.read_text(encoding="utf-8"))
        basin_hucs_531 = [str(b.get("huc_02", "")).zfill(2) for b in (bm_meta.get("basins") or [])]

    # Deterministic timestamp derived from leaderboard files on disk
    lb_files = sorted(lb_dir.glob("*.json"))
    latest_mtime = max((p.stat().st_mtime for p in lb_files), default=time.time())
    updated_at = dt.datetime.fromtimestamp(latest_mtime, tz=dt.timezone.utc).isoformat()

    available_qsim_keys: set[str] = set()
    if explorer_store_path is not None and explorer_store_path.is_file():
        with np.load(explorer_store_path, allow_pickle=False) as d:
            available_qsim_keys = {k[len("qsim__") :] for k in d.files if k.startswith("qsim__")}
    elif stage_dir is not None:
        idx_file = stage_dir / "hydro_assets" / "index.json"
        if idx_file.is_file():
            idx_doc = json.loads(idx_file.read_text(encoding="utf-8"))
            available_qsim_keys = {str(k) for k in idx_doc.get("available_qsim_keys", [])}
        else:
            sample_basins = sorted((stage_dir / "hydro_assets").glob("basin_*.json"))
            if sample_basins:
                sample_doc = json.loads(sample_basins[0].read_text(encoding="utf-8"))
                available_qsim_keys = {str(k) for k in (sample_doc.get("models") or {}).keys()}

    boards_out: list[dict[str, Any]] = []
    matrix_summary: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    hydro_models: list[dict[str, Any]] = []
    model_basin_nse: dict[str, list[float | None]] = {}

    for lb in t3_doc.get("leaderboards", []):
        lb_id = str(lb["id"])
        rows_out: list[dict[str, Any]] = []
        for idx, raw_row in enumerate(lb.get("rows", []), start=1):
            row = dict(raw_row)
            eid = str(row.get("entry_id", ""))
            mk = f"{lb_id}__{eid}"
            fam = families_by_id.get(str(row.get("family_id", "")), {})
            row["rank"] = int(row.get("rank") or idx)
            row["leaderboard_id"] = lb_id
            row["model_key"] = mk
            row["family_name"] = row.get("family_name") or fam.get("name") or row.get("family_id", "")
            row["paradigm"] = row.get("paradigm") or fam.get("paradigm", "")
            row["citation"] = row.get("citation") or fam.get("citation", "")
            row["paper_url"] = row.get("paper_url") or fam.get("paper_url", "")
            row["github_url"] = row.get("github_url") or fam.get("github_url") or row.get("code_url") or ""
            row["code_url"] = row.get("code_url") or row["github_url"]
            row["data_archive_url"] = (
                row.get("data_archive_url")
                or row.get("zenodo_url")
                or row.get("hydroshare_url")
                or fam.get("zenodo_url")
                or ""
            )
            has_hydro = mk in available_qsim_keys
            row["has_hydrograph"] = has_hydro
            _attach_flat_gauch_metrics(row)

            pb_nse = row.get("per_basin_nse")
            if isinstance(pb_nse, list) and len(pb_nse) == 531 and len(basin_hucs_531) == 531:
                reg_raw = row.get("regional_huc02")
                needs_huc_enrich = not isinstance(reg_raw, dict) or any(
                    not isinstance(v, dict) for v in reg_raw.values()
                )
                if needs_huc_enrich:
                    by_huc: dict[str, list[float]] = {}
                    for h_code, val in zip(basin_hucs_531, pb_nse):
                        if h_code and val is not None and np.isfinite(float(val)):
                            by_huc.setdefault(h_code, []).append(float(val))
                    existing_reg = reg_raw if isinstance(reg_raw, dict) else {}
                    enriched_reg: dict[str, dict[str, Any]] = {}
                    for h_code in sorted(by_huc):
                        vals = by_huc[h_code]
                        prev = existing_reg.get(h_code)
                        prev_med = (
                            float(prev)
                            if isinstance(prev, (int, float)) and np.isfinite(float(prev))
                            else None
                        )
                        enriched_reg[h_code] = prev if isinstance(prev, dict) else {
                            "num_basins": len(vals),
                            "median_nse": round(
                                prev_med if prev_med is not None else float(np.median(vals)), 4
                            ),
                            "mean_nse": round(float(np.mean(vals)), 4),
                            "median_kge": None,
                            "nse_le_zero_count": int(sum(1 for v in vals if v <= 0.0)),
                        }
                    row["regional_huc02"] = enriched_reg

            rows_out.append(row)
            all_entries.append(row)

            pb_nse = row.get("per_basin_nse")
            if isinstance(pb_nse, list) and len(pb_nse) == 531:
                model_basin_nse[mk] = [
                    round(float(v), 3) if v is not None and np.isfinite(float(v)) else None
                    for v in pb_nse
                ]
            if has_hydro or (
                not available_qsim_keys
                and isinstance(pb_nse, list)
                and len(pb_nse) == 531
                and bool(str(row.get("zarr_path") or "").strip())
            ):
                hydro_models.append(
                    {
                        "model_key": mk,
                        "entry_id": eid,
                        "leaderboard_id": lb_id,
                        "model_name": row.get("model_name", eid),
                        "family_id": row.get("family_id", ""),
                        "forcing": row.get("forcing", ""),
                        "category": row.get("category", "Archive"),
                        "paradigm": row.get("paradigm", ""),
                        "is_ensemble": bool(row.get("is_ensemble")),
                        "primary_ranking_nse": row.get("primary_ranking_nse"),
                        "med_kge": row.get("med_kge"),
                    }
                )

        lb_copy = {**lb, "n_entries": len(rows_out), "rows": rows_out}
        boards_out.append(lb_copy)
        matrix_summary.append(
            {
                "id": lb_id,
                "table_number": lb.get("table_number"),
                "section": lb.get("section"),
                "spatial_regime": lb.get("spatial_regime"),
                "forcing_regime": lb.get("forcing_regime"),
                "title": lb.get("title"),
                "n_entries": len(rows_out),
                "top_model": rows_out[0]["model_name"] if rows_out else None,
                "top_nse": rows_out[0].get("primary_ranking_nse") if rows_out else None,
            }
        )

    families_out: list[dict[str, Any]] = []
    for fam in families_raw:
        fid = str(fam["family_id"])
        f_copy = dict(fam)
        members = [e for e in all_entries if str(e.get("family_id", "")) == fid]
        f_copy["n_entries"] = len(members)
        f_copy["leaderboards_covered"] = sorted({str(e["leaderboard_id"]) for e in members})
        nse_vals = [
            float(e["primary_ranking_nse"])
            for e in members
            if _finite_or_none(e.get("primary_ranking_nse")) is not None
        ]
        f_copy["best_nse"] = max(nse_vals, default=None)
        families_out.append(f_copy)

    protocol_obj = dict(t3_doc.get("protocol") or {})
    if "n_leaderboards" not in protocol_obj:
        protocol_obj["n_leaderboards"] = len(boards_out)

    overview_obj = {
        "updated_at": updated_at,
        "protocol": protocol_obj,
        "total_entries": len(all_entries),
        "total_families": len(families_out),
        "matrix_summary": matrix_summary,
    }
    leaderboards_obj = {
        "updated_at": updated_at,
        "protocol": protocol_obj,
        "total_entries": len(all_entries),
        "leaderboards": boards_out,
        "families": families_out,
    }
    families_obj = {"families": families_out}

    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("1989-10-01", "1999-09-30", freq="D")]
    basins_meta: list[dict[str, Any]] = []
    conus_map: dict[str, Any] = {}
    if BASIN_MAP_META_PATH.is_file():
        bm = json.loads(BASIN_MAP_META_PATH.read_text(encoding="utf-8"))
        basins_meta = list(bm.get("basins") or [])
        conus_map = dict(bm.get("conus_map") or {})
    else:
        basins_file = splits_dir / "basins_531.txt"
        basins_meta = [
            {"basin_id": b.strip().zfill(8), "gauge_name": b.strip().zfill(8)}
            for b in basins_file.read_text(encoding="utf-8").splitlines()
            if b.strip()
        ]

    hydro_index_obj = {
        "n_basins": len(basins_meta),
        "n_days": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
        "dates": dates,
        "basins": basins_meta,
        "models": hydro_models,
        "conus_map": conus_map,
        "model_basin_nse": model_basin_nse,
    }
    return overview_obj, leaderboards_obj, families_obj, hydro_index_obj


def _download_and_unpack_prebuilt_hydro_assets(url: str, stage_dir: Path) -> Path:
    """Safely download and extract prebuilt 531-basin JSON assets (rejecting traversal/symlinks)."""
    tar_path = stage_dir / "prebuilt_hydro_assets.tar.gz"
    stage_dir.mkdir(parents=True, exist_ok=True)
    validate_and_fetch_archive_url(
        url,
        dest_path=tar_path,
        max_bytes=500 * 1024 * 1024,
        timeout=120,
        user_agent="HydrologyMIP-Deploy/1.0",
    )
    out_dir = (stage_dir / "hydro_assets").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_resolved = stage_dir.resolve()

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            if m.issym() or m.islnk() or (not m.isdir() and not m.isreg()):
                raise ValueError(f"Unsafe tar member rejected in prebuilt hydro assets: {m.name!r}")
            posix = PurePosixPath(m.name)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"Tar path traversal rejected: {m.name!r}")
            rel_parts = posix.parts[1:] if posix.parts and posix.parts[0] == "hydro_assets" else posix.parts
            if not rel_parts or any(part in ("", ".", "..") for part in rel_parts):
                continue
            target = (out_dir / PurePosixPath(*rel_parts)).resolve()
            if target != out_dir and not target.is_relative_to(out_dir):
                raise ValueError(f"Tar member escapes hydro_assets root: {m.name!r}")
            if not target.is_relative_to(stage_resolved):
                raise ValueError(f"Tar member escapes staging root: {m.name!r}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                raise ValueError(f"Failed to read tar member {m.name!r}.")
            with src, open(target, "wb") as dst:
                dst.write(src.read())

    if tar_path.is_file():
        tar_path.unlink()
    basin_files = list(out_dir.glob("basin_*.json"))
    if len(basin_files) != 531:
        raise ValueError(
            f"Prebuilt hydro assets archive contained {len(basin_files)} basin files; expected 531."
        )
    print(f"Unpacked {len(basin_files)} prebuilt basin JSON files into {out_dir}")
    return out_dir


def build_per_basin_static_assets(
    hydro_index: dict[str, Any],
    store_path: Path | None,
    stage_dir: Path,
) -> dict[str, Path]:
    """Generate or reuse 531 compact per-basin JSON files in ``stage_dir/hydro_assets``."""
    out_dir = stage_dir / "hydro_assets"
    basins = hydro_index["basins"]
    models = hydro_index["models"]
    sample_file = out_dir / f"basin_{basins[0]['basin_id']}.json"

    if store_path is None or not store_path.is_file():
        if sample_file.exists() and len(list(out_dir.glob("basin_*.json"))) == len(basins):
            print(f"Using {len(basins)} prebuilt per-basin asset files in {out_dir}")
            return {
                f"/hydro_assets/basin_{b['basin_id']}.json": out_dir / f"basin_{b['basin_id']}.json"
                for b in basins
            }
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    latest_source_mtime = max(Path(__file__).stat().st_mtime, store_path.stat().st_mtime)
    if (
        sample_file.exists()
        and len(list(out_dir.glob("basin_*.json"))) == len(basins)
        and sample_file.stat().st_mtime >= latest_source_mtime
    ):
        print(f"Reusing {len(basins)} cached per-basin asset files in {out_dir}")
        return {
            f"/hydro_assets/basin_{b['basin_id']}.json": out_dir / f"basin_{b['basin_id']}.json"
            for b in basins
        }

    t0 = time.time()
    print(f"Building {len(basins)} per-basin static JSON assets from {store_path}...")
    with np.load(store_path, allow_pickle=False) as npz:
        qobs = np.asarray(npz["qobs"], dtype=np.float32)
        d_prcp = np.asarray(npz["daymet_prcp"], dtype=np.float32)
        d_tmax = np.asarray(npz["daymet_tmax"], dtype=np.float32)
        d_tmin = np.asarray(npz["daymet_tmin"], dtype=np.float32)
        m_prcp = np.asarray(npz["maurer_prcp"], dtype=np.float32)
        m_tmax = np.asarray(npz["maurer_tmax"], dtype=np.float32)
        m_tmin = np.asarray(npz["maurer_tmin"], dtype=np.float32)
        n_prcp = np.asarray(npz["nldas_prcp"], dtype=np.float32)
        n_tmax = np.asarray(npz["nldas_tmax"], dtype=np.float32)
        n_tmin = np.asarray(npz["nldas_tmin"], dtype=np.float32)

        model_arrays: dict[str, np.ndarray] = {}
        model_metrics: dict[str, list[dict[str, float | None]]] = {}
        for m in models:
            mk = m["model_key"]
            k = f"qsim__{mk}"
            if k in npz.files:
                arr = np.asarray(npz[k], dtype=np.float32)
                model_arrays[mk] = arr
                model_metrics[mk] = compute_fast_basin_metrics_matrix(qobs, arr)

    def to_list_round(vec: np.ndarray, dec: int = 2) -> list[float | None]:
        rounded = np.round(vec.astype(np.float64), dec)
        finite = np.isfinite(rounded)
        if np.all(finite):
            return rounded.tolist()
        return [float(x) if f else None for x, f in zip(rounded.tolist(), finite.tolist())]

    asset_map: dict[str, Path] = {}
    for b_idx, b_meta in enumerate(basins):
        bid = b_meta["basin_id"]
        dt_max, dt_min = d_tmax[b_idx], d_tmin[b_idx]
        mt_max, mt_min = m_tmax[b_idx], m_tmin[b_idx]
        nt_max, nt_min = n_tmax[b_idx], n_tmin[b_idx]

        models_payload = {
            mk: {
                "qsim": to_list_round(arr[b_idx], 2),
                "basin_metrics": model_metrics[mk][b_idx],
            }
            for mk, arr in model_arrays.items()
        }

        basin_doc = {
            "basin": b_meta,
            "qobs": to_list_round(qobs[b_idx], 2),
            "forcings": {
                "daymet": {
                    "label": "Daymet",
                    "prcp": to_list_round(d_prcp[b_idx], 2),
                    "tmax": to_list_round(dt_max, 2),
                    "tmin": to_list_round(dt_min, 2),
                    "tmean": to_list_round(0.5 * (dt_max + dt_min), 2),
                },
                "maurer": {
                    "label": "Maurer",
                    "prcp": to_list_round(m_prcp[b_idx], 2),
                    "tmax": to_list_round(mt_max, 2),
                    "tmin": to_list_round(mt_min, 2),
                    "tmean": to_list_round(0.5 * (mt_max + mt_min), 2),
                },
                "nldas": {
                    "label": "NLDAS",
                    "prcp": to_list_round(n_prcp[b_idx], 2),
                    "tmax": to_list_round(nt_max, 2),
                    "tmin": to_list_round(nt_min, 2),
                    "tmean": to_list_round(0.5 * (nt_max + nt_min), 2),
                },
            },
            "models": models_payload,
        }
        fpath = out_dir / f"basin_{bid}.json"
        fpath.write_text(json.dumps(basin_doc, separators=(",", ":")), encoding="utf-8")
        asset_map[f"/hydro_assets/basin_{bid}.json"] = fpath

    print(f"Built {len(asset_map)} basin JSON files in {time.time() - t0:.1f}s")
    return asset_map


def stage_data_and_portal_assets(
    overview_obj: Any,
    leaderboards_obj: Any,
    families_obj: Any,
    hydro_index_obj: dict[str, Any],
    *,
    mip_dir: Path = MIP_DIR,
    portal_dir: Path = PORTAL_DIR,
    stage_dir: Path = DEFAULT_STAGE_DIR,
) -> dict[str, Path]:
    """Stage portal files, sanitized JSON API payloads, splits, and figures as static assets."""
    stage_data = stage_dir / "data_assets"
    stage_data.mkdir(parents=True, exist_ok=True)

    protocol_spec = {
        "benchmark": "CAMELS-US Daily Deterministic Model Intercomparison Project",
        "version": "1.0",
        "canonical_url": f"https://{CANONICAL_HOST}/",
        "n_basins": 531,
        "training_window": {
            "start": "1999-10-01",
            "end": "2008-09-30",
            "water_years": "WY2000-WY2008",
        },
        "evaluation_window": {
            "start": "1989-10-01",
            "end": "1999-09-30",
            "water_years": "WY1990-WY1999",
            "n_days": 3652,
        },
        "target_variable": "USGS observed streamflow (mm/day)",
        "spatial_regimes": ["gauged", "pub_12fold_random", "pur_18fold_huc02"],
        "forcing_categories": ["single_camels", "multi_camels", "lagged_streamflow", "freeform"],
        "metrics": list(DEFAULT_METRICS_ORDER),
        "summary_metrics": ["Median-All-14", "Mean-NSE", "Mean-KGE", "NSE_le_0"],
        "metric_reference": "Gauch et al. (2023) + logNSE/logKGE",
        "splits_endpoints": {
            "basins_531": "/api/splits/basins_531.txt",
            "pub_12fold": "/api/splits/pub_12fold_random_splits.json",
            "pur_18fold": "/api/splits/pur_18fold_huc02_splits.json",
        },
        "public_data_mirror": "https://data.hydrologymip.org/camels_us_daily/",
        "public_targets_url": "https://data.hydrologymip.org/camels_us_daily/observations/newman_eval_targets.zarr",
        "zenodo_release_doi": "10.5281/zenodo.24383448",
        "submissions_email": "submissions@hydrologymip.org",
    }

    repo_for_mips = mip_dir.parent.parent
    mips_dict = discover_mips(repo_root=repo_for_mips)
    mips_payload = rewrite_gcs_to_r2(
        {
            "default_mip_id": mip_dir.name,
            "n_mips": len(mips_dict),
            "mips": [proto.to_summary_dict() for proto in mips_dict.values()],
        }
    )
    mip_camels_payload = rewrite_gcs_to_r2(
        get_mip(mip_id=mip_dir.name, repo_root=repo_for_mips).to_summary_dict()
    )

    json_assets = {
        "overview.json": overview_obj,
        "leaderboards.json": leaderboards_obj,
        "families.json": families_obj,
        "hydrographs_index.json": hydro_index_obj,
        "protocol.json": protocol_spec,
        "mips.json": mips_payload,
        "mip_camels_us_daily_protocol.json": mip_camels_payload,
    }
    asset_map: dict[str, Path] = {}
    for fname, payload in json_assets.items():
        p = stage_data / fname
        p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        asset_map[f"/data_assets/{fname}"] = p

    # Canonical splits directly from mips/camels_us_daily/splits/
    splits_dir = mip_dir / "splits"
    asset_map["/data_assets/pub_12fold_random_splits.json"] = splits_dir / "pub_12fold_random_splits.json"
    asset_map["/data_assets/pur_18fold_huc02_splits.json"] = splits_dir / "pur_18fold_huc02_splits.json"
    asset_map["/data_assets/basins_531.txt"] = splits_dir / "basins_531.txt"

    # Portal static files directly from web/portal/
    asset_map["/index.html"] = portal_dir / "index.html"
    asset_map["/static/styles.css"] = portal_dir / "styles.css"
    asset_map["/static/app.js"] = portal_dir / "app.js"
    asset_map["/favicon.svg"] = portal_dir / "favicon.svg"
    asset_map["/favicon.ico"] = portal_dir / "favicon.ico"
    asset_map["/og-card.png"] = portal_dir / "og-card.png"
    asset_map["/robots.txt"] = portal_dir / "robots.txt"
    asset_map["/sitemap.xml"] = portal_dir / "sitemap.xml"

    # Paper figures uploaded directly from mips/camels_us_daily/figures/
    figures_dir = mip_dir / "figures"
    if figures_dir.exists():
        for ext_pat in ("*.png", "*.pdf", "*.json"):
            for fp in sorted(figures_dir.glob(ext_pat)):
                asset_map[f"/api/figures/{fp.name}"] = fp

    return asset_map


def upload_cf_static_assets(
    asset_map: dict[str, Path], account_id: str, api_token: str
) -> str:
    """Upload static assets via Cloudflare Workers assets-upload-session and return completion JWT."""
    manifest = {}
    hash_to_path: dict[str, Path] = {}
    for url_path, file_path in asset_map.items():
        raw_bytes = file_path.read_bytes()
        ext = file_path.suffix.lstrip(".").lower()
        h = hashlib.sha256(base64.b64encode(raw_bytes) + ext.encode("utf-8")).hexdigest()[:32]
        manifest[url_path] = {"hash": h, "size": len(raw_bytes)}
        hash_to_path[h] = file_path

    opener = urllib.request.build_opener(_NonRaisingHTTPErrorProcessor())
    session_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/workers/scripts/{CF_SCRIPT_NAME}/assets-upload-session"
    )
    req = urllib.request.Request(
        session_url,
        data=json.dumps({"manifest": manifest}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "User-Agent": "HydrologyMIP/1.0",
        },
    )
    with opener.open(req, timeout=60) as resp:
        status_code = int(getattr(resp, "status", 200))
        raw_txt = resp.read().decode("utf-8", errors="replace")
        if status_code >= 400:
            raise RuntimeError(f"Cloudflare assets-upload-session failed ({status_code}): {raw_txt}")
        session_res = json.loads(raw_txt)["result"]

    jwt_token = session_res["jwt"]
    buckets = session_res.get("buckets") or []
    print(f"Cloudflare assets upload session: {len(asset_map)} total assets, {len(buckets)} buckets to upload")
    if not buckets:
        return jwt_token

    def upload_bucket(bucket_info: tuple[int, list[str]]) -> str | None:
        b_idx, b_hashes = bucket_info
        boundary = f"----CFAssetBatchBoundary{b_idx}"
        parts = []
        for h in b_hashes:
            fp = hash_to_path[h]
            ext = fp.suffix.lstrip(".").lower()
            mime = MIME_BY_EXT.get(ext, "application/octet-stream")
            b64_content = base64.b64encode(fp.read_bytes()).decode("ascii")
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{h}"; filename="{h}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
                f"{b64_content}\r\n"
            )
        parts.append(f"--{boundary}--\r\n")
        body = "".join(parts).encode("utf-8")

        up_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/workers/assets/upload?base64=true"
        )
        up_req = urllib.request.Request(
            up_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "HydrologyMIP/1.0",
            },
        )
        for attempt in range(3):
            with opener.open(up_req, timeout=120) as r:
                code = int(getattr(r, "status", 200))
                raw_resp = r.read().decode("utf-8", errors="ignore")
                if code < 400:
                    res_json = json.loads(raw_resp)
                    return res_json.get("result", {}).get("jwt")
                if attempt == 2:
                    raise RuntimeError(f"Bucket {b_idx} upload failed ({code}): {raw_resp}")
                time.sleep(1.5 * (attempt + 1))
        return None

    completion_jwt = jwt_token
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res_jwt in pool.map(upload_bucket, enumerate(buckets)):
            if res_jwt:
                completion_jwt = res_jwt

    print("All Cloudflare asset buckets uploaded successfully!")
    return completion_jwt


WORKER_JS = r"""
const CANONICAL_HOST = "hydrologymip.org";
const PREVIEW_GATE_ENABLED = __PREVIEW_GATE_ENABLED__;
const PREVIEW_ACCESS_KEY = __PREVIEW_ACCESS_KEY__;
const PREVIEW_COOKIE_NAME = "hmip_preview";
const PREVIEW_COOKIE_MAX_AGE = 604800;
const DEFAULT_SCORER_API_URL = "__SCORER_API_URL__";

const SECURITY_HEADERS = {
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "strict-origin-when-cross-origin",
};

const GATE_NO_CACHE_HEADERS = {
  "Cache-Control": "no-store",
  "X-Robots-Tag": "noindex, nofollow",
};

let CACHED_LEADERBOARDS = null;
let CACHED_HYDRO_INDEX = null;
let CACHED_HYDRO_MODELS_MAP = null;
let CACHED_HYDRO_ENTRY_MAP = null;

function hasValidPreviewCookie(request) {
  const cookieHeader = request.headers.get("Cookie") || "";
  if (!cookieHeader) return false;
  for (const part of cookieHeader.split(";")) {
    const [rawKey, ...valParts] = part.trim().split("=");
    if (rawKey === PREVIEW_COOKIE_NAME && valParts.join("=").trim() === PREVIEW_ACCESS_KEY) {
      return true;
    }
  }
  return false;
}

function previewCookieHeader(maxAge = PREVIEW_COOKIE_MAX_AGE) {
  const val = maxAge > 0 ? PREVIEW_ACCESS_KEY : "";
  return `${PREVIEW_COOKIE_NAME}=${val}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`;
}

function renderPreviewGateHtml(hasInvalidAttempt) {
  const errBanner = hasInvalidAttempt
    ? `<div class="err" role="alert">Invalid preview password. Please check your preview link or try again.</div>`
    : "";
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="robots" content="noindex, nofollow, noarchive" />
  <title>HydrologyMIP — Launching Soon</title>
  <style>
    :root {
      --bg: #faf8f4;
      --surface: #ffffff;
      --ink: #16191d;
      --muted: #5a626a;
      --rule: #e2ddd5;
      --accent: #1b5e55;
      --accent-hover: #13463f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .gate-card {
      width: 100%;
      max-width: 440px;
      background: var(--surface);
      border: 1px solid var(--rule);
      border-radius: 6px;
      padding: 2.25rem 2rem;
      box-shadow: 0 8px 24px rgba(22, 25, 29, 0.05);
    }
    .kicker {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--accent);
      margin: 0 0 0.5rem 0;
    }
    h1 {
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.55rem;
      font-weight: 600;
      margin: 0 0 0.65rem 0;
      letter-spacing: -0.01em;
    }
    p {
      font-size: 0.9rem;
      line-height: 1.55;
      color: var(--muted);
      margin: 0 0 1.4rem 0;
    }
    .err {
      margin-bottom: 1rem;
      padding: 0.65rem 0.85rem;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 4px;
      color: #991b1b;
      font-size: 0.84rem;
    }
    label {
      display: block;
      font-size: 0.78rem;
      font-weight: 600;
      margin-bottom: 0.4rem;
      color: var(--ink);
    }
    input[type="password"] {
      width: 100%;
      padding: 0.65rem 0.8rem;
      font-size: 0.92rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      border: 1px solid var(--rule);
      border-radius: 4px;
      background: #fdfcfa;
      color: var(--ink);
      margin-bottom: 1rem;
    }
    input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(27, 94, 85, 0.15);
    }
    button {
      width: 100%;
      padding: 0.7rem 1rem;
      font-size: 0.9rem;
      font-weight: 600;
      color: #ffffff;
      background: var(--accent);
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 150ms ease;
    }
    button:hover {
      background: var(--accent-hover);
    }
  </style>
</head>
<body>
  <main class="gate-card">
    <p class="kicker">Launching Soon — Preview Access</p>
    <h1>HydrologyMIP</h1>
    <p>HydrologyMIP is in pre-launch preview. Enter the preview password or follow your preview invitation link to continue.</p>
    ${errBanner}
    <form method="POST" action="">
      <label for="access-input">Preview Password</label>
      <input id="access-input" name="access" type="password" placeholder="Enter preview password..." required autofocus autocomplete="off" />
      <button type="submit">Enter Preview</button>
    </form>
  </main>
</body>
</html>`;
}

function withHeaders(resp, extraHeaders = {}) {
  const h = new Headers(resp.headers);
  for (const [k, v] of Object.entries(SECURITY_HEADERS)) {
    if (!h.has(k)) h.set(k, v);
  }
  for (const [k, v] of Object.entries(extraHeaders)) {
    h.set(k, v);
  }
  if (PREVIEW_GATE_ENABLED) {
    h.set("Cache-Control", "private, no-store");
    h.set("X-Robots-Tag", "noindex, nofollow");
  }
  return new Response(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers: h,
  });
}

function jsonResponse(obj, status = 200, cacheControl = "public, max-age=300, s-maxage=3600") {
  const effectiveCache = PREVIEW_GATE_ENABLED ? "private, no-store" : cacheControl;
  const headers = {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": effectiveCache,
    ...SECURITY_HEADERS,
  };
  if (PREVIEW_GATE_ENABLED) {
    headers["X-Robots-Tag"] = "noindex, nofollow";
  }
  return new Response(JSON.stringify(obj), {
    status,
    headers,
  });
}

async function fetchAsset(env, origin, assetPath) {
  const req = new Request(new URL(assetPath, origin));
  return await env.ASSETS.fetch(req);
}

async function getLeaderboardsObj(env, origin) {
  if (!CACHED_LEADERBOARDS) {
    const resp = await fetchAsset(env, origin, "/data_assets/leaderboards.json");
    if (resp.ok) CACHED_LEADERBOARDS = await resp.json();
  }
  return CACHED_LEADERBOARDS;
}

async function getHydroIndex(env, origin) {
  if (!CACHED_HYDRO_INDEX) {
    const resp = await fetchAsset(env, origin, "/data_assets/hydrographs_index.json");
    if (resp.ok) {
      CACHED_HYDRO_INDEX = await resp.json();
      CACHED_HYDRO_MODELS_MAP = new Map();
      CACHED_HYDRO_ENTRY_MAP = new Map();
      for (const m of (CACHED_HYDRO_INDEX.models || [])) {
        CACHED_HYDRO_MODELS_MAP.set(m.model_key, m);
        if (m.entry_id && !CACHED_HYDRO_ENTRY_MAP.has(m.entry_id)) {
          CACHED_HYDRO_ENTRY_MAP.set(m.entry_id, m.model_key);
        }
      }
    }
  }
  return CACHED_HYDRO_INDEX;
}

function computeQuantiles101(values) {
  const clean = (values || [])
    .filter((v) => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!clean.length) return [];
  const out = [];
  const n = clean.length;
  for (let i = 0; i <= 100; i++) {
    const pos = (i / 100) * (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(n - 1, Math.ceil(pos));
    const frac = pos - lo;
    const val = clean[lo] + frac * (clean[hi] - clean[lo]);
    out.push(Math.round(val * 10000) / 10000);
  }
  return out;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 1. Canonical 301 redirect from HTTP and alias domains to https://hydrologymip.org
    if (url.hostname !== CANONICAL_HOST || url.protocol !== "https:") {
      const target = `https://${CANONICAL_HOST}${url.pathname}${url.search}`;
      return Response.redirect(target, 301);
    }

    const p = url.pathname;

    // 2. Preview password gate
    if (PREVIEW_GATE_ENABLED) {
      if (p === "/robots.txt") {
        return new Response("User-agent: *\nDisallow: /\n", {
          status: 200,
          headers: {
            "Content-Type": "text/plain; charset=utf-8",
            ...GATE_NO_CACHE_HEADERS,
            ...SECURITY_HEADERS,
          },
        });
      }

      if (url.searchParams.get("logout") === "1") {
        const cleanUrl = new URL(request.url);
        cleanUrl.searchParams.delete("logout");
        cleanUrl.searchParams.delete("access");
        return new Response(null, {
          status: 302,
          headers: {
            "Location": cleanUrl.toString(),
            "Set-Cookie": previewCookieHeader(0),
            ...GATE_NO_CACHE_HEADERS,
            ...SECURITY_HEADERS,
          },
        });
      }

      const queryAccess = (url.searchParams.get("access") || "").trim();
      if ((request.method === "GET" || request.method === "HEAD") && queryAccess === PREVIEW_ACCESS_KEY) {
        const cleanUrl = new URL(request.url);
        cleanUrl.searchParams.delete("access");
        return new Response(null, {
          status: 302,
          headers: {
            "Location": cleanUrl.toString(),
            "Set-Cookie": previewCookieHeader(),
            ...GATE_NO_CACHE_HEADERS,
            ...SECURITY_HEADERS,
          },
        });
      }

      let postAttemptFailed = false;
      if (!p.startsWith("/api/") && request.method === "POST") {
        const ct = (request.headers.get("Content-Type") || "").toLowerCase();
        let submittedAccess = "";
        if (ct.includes("application/x-www-form-urlencoded")) {
          const rawBody = await request.text();
          submittedAccess = (new URLSearchParams(rawBody).get("access") || "").trim();
        } else if (ct.includes("multipart/form-data")) {
          const fd = await request.formData();
          submittedAccess = String(fd.get("access") || "").trim();
        }
        if (submittedAccess === PREVIEW_ACCESS_KEY) {
          const cleanUrl = new URL(request.url);
          cleanUrl.searchParams.delete("access");
          return new Response(null, {
            status: 302,
            headers: {
              "Location": cleanUrl.toString(),
              "Set-Cookie": previewCookieHeader(),
              ...GATE_NO_CACHE_HEADERS,
              ...SECURITY_HEADERS,
            },
          });
        }
        postAttemptFailed = true;
      }

      if (!hasValidPreviewCookie(request)) {
        if (p.startsWith("/api/")) {
          return new Response(
            JSON.stringify({ error: "Preview authentication required" }),
            {
              status: 401,
              headers: {
                "Content-Type": "application/json; charset=utf-8",
                ...GATE_NO_CACHE_HEADERS,
                ...SECURITY_HEADERS,
              },
            }
          );
        }
        return new Response(
          renderPreviewGateHtml(Boolean(queryAccess) || postAttemptFailed),
          {
            status: 401,
            headers: {
              "Content-Type": "text/html; charset=utf-8",
              ...GATE_NO_CACHE_HEADERS,
              ...SECURITY_HEADERS,
            },
          }
        );
      }
    }

    // 3. Reverse-proxy /api/preview/* to configurable SCORER_API_URL
    if (p.startsWith("/api/preview/")) {
      const scorerBase = String((env && env.SCORER_API_URL) || DEFAULT_SCORER_API_URL || "")
        .trim()
        .replace(/\/+$/, "");
      if (!scorerBase) {
        return jsonResponse(
          {
            error:
              "Instant preview scoring service is not yet configured for this deployment. You can still complete your metadata and submit an official GitHub PR or issue with your permanent DOI.",
            code: "SCORER_UNCONFIGURED",
          },
          503,
          "no-store"
        );
      }
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: { ...SECURITY_HEADERS } });
      }
      const targetUrl = `${scorerBase}${url.pathname}${url.search}`;
      const fwdHeaders = new Headers();
      const ct = request.headers.get("Content-Type");
      if (ct) fwdHeaders.set("Content-Type", ct);
      const accept = request.headers.get("Accept");
      if (accept) fwdHeaders.set("Accept", accept);
      const cfIp = request.headers.get("CF-Connecting-IP");
      if (cfIp) fwdHeaders.set("CF-Connecting-IP", cfIp);
      const rawCookie = request.headers.get("Cookie") || "";
      const gaesaCookie = rawCookie
        .split(";")
        .map((part) => part.trim())
        .find((part) => part.startsWith("GAESA="));
      if (gaesaCookie) fwdHeaders.set("Cookie", gaesaCookie);
      const init = { method: request.method, headers: fwdHeaders };
      if (request.method !== "GET" && request.method !== "HEAD") {
        init.body = request.body;
      }
      try {
        const upstream = await fetch(targetUrl, init);
        return withHeaders(upstream, { "Cache-Control": "no-store" });
      } catch (_err) {
        return jsonResponse(
          {
            error: "Preview scoring service is temporarily unreachable.",
            code: "SCORER_UNREACHABLE",
          },
          502,
          "no-store"
        );
      }
    }

    // 4. Serve canonical root index.html with security headers
    if (p === "/" || p === "/index.html") {
      const idxResp = await fetchAsset(env, url.origin, "/index.html");
      return withHeaders(idxResp, {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": PREVIEW_GATE_ENABLED ? "private, no-store" : "public, max-age=300, s-maxage=600",
      });
    }

    // 5. Reject unsupported write methods cleanly
    if (request.method !== "GET" && request.method !== "HEAD") {
      return jsonResponse(
        {
          detail:
            "Direct browser mutations are disabled on the public portal. " +
            "Please submit models via https://github.com/grey-nearing/hydrology-mip.",
        },
        405,
        "no-store"
      );
    }

    // 6. Static JSON/text API routes served directly from Cloudflare Static Assets
    const STATIC_API_MAP = {
      "/api/overview": ["/data_assets/overview.json", "application/json; charset=utf-8"],
      "/api/leaderboards": ["/data_assets/leaderboards.json", "application/json; charset=utf-8"],
      "/api/families": ["/data_assets/families.json", "application/json; charset=utf-8"],
      "/api/hydrographs/index": ["/data_assets/hydrographs_index.json", "application/json; charset=utf-8"],
      "/api/mips": ["/data_assets/mips.json", "application/json; charset=utf-8"],
      "/api/mips/camels_us_daily": ["/data_assets/mip_camels_us_daily_protocol.json", "application/json; charset=utf-8"],
      "/api/mips/camels_us_daily/protocol": ["/data_assets/mip_camels_us_daily_protocol.json", "application/json; charset=utf-8"],
      "/api/template/protocol.json": ["/data_assets/protocol.json", "application/json; charset=utf-8"],
      "/api/splits/pub_12fold_random_splits.json": ["/data_assets/pub_12fold_random_splits.json", "application/json; charset=utf-8"],
      "/api/splits/pur_18fold_huc02_splits.json": ["/data_assets/pur_18fold_huc02_splits.json", "application/json; charset=utf-8"],
      "/api/splits/basins_531.txt": ["/data_assets/basins_531.txt", "text/plain; charset=utf-8"],
    };

    if (STATIC_API_MAP[p]) {
      const [assetPath, contentType] = STATIC_API_MAP[p];
      const resp = await fetchAsset(env, url.origin, assetPath);
      return withHeaders(resp, {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=300, s-maxage=3600",
      });
    }

    // 7. Per-entry basin CDF & HUC-02 breakdown
    if (p.startsWith("/api/entry/") && p.endsWith("/basins")) {
      const parts = p.split("/");
      const entryId = decodeURIComponent(parts[3] || "");
      const lbs = await getLeaderboardsObj(env, url.origin);
      let foundRow = null;
      let ref3fRow = null;
      let ref1fRow = null;
      for (const lb of (lbs?.leaderboards || [])) {
        for (const r of (lb.rows || [])) {
          if (r.entry_id === entryId && !foundRow) foundRow = r;
          if (r.entry_id === "t4_05" && !ref3fRow) ref3fRow = r;
          if (r.entry_id === "t3_02" && !ref1fRow) ref1fRow = r;
        }
      }
      if (!foundRow) {
        return jsonResponse({ detail: `Entry '${entryId}' not found.` }, 404, "no-store");
      }
      const ref3fCdf = ref3fRow && Array.isArray(ref3fRow.per_basin_nse) ? computeQuantiles101(ref3fRow.per_basin_nse) : [];
      const ref1fCdf = ref1fRow && Array.isArray(ref1fRow.per_basin_nse) ? computeQuantiles101(ref1fRow.per_basin_nse) : [];
      let nseCdf = [];
      let source = "unavailable";
      if (Array.isArray(foundRow.per_basin_nse) && foundRow.per_basin_nse.length > 0) {
        nseCdf = computeQuantiles101(foundRow.per_basin_nse);
        source = "empirical_531_basins";
      }
      const hucRows = [];
      if (foundRow.regional_huc02 && typeof foundRow.regional_huc02 === "object") {
        for (const hucKey of Object.keys(foundRow.regional_huc02).sort()) {
          const rawHd = foundRow.regional_huc02[hucKey];
          if (typeof rawHd === "number" && Number.isFinite(rawHd)) {
            hucRows.push({
              huc_02: hucKey,
              n_basins: null,
              median_NSE: rawHd,
              mean_NSE: null,
              median_KGE: null,
              n_NSE_le_0: null,
            });
          } else {
            const hd = rawHd || {};
            hucRows.push({
              huc_02: hucKey,
              n_basins: hd.num_basins ?? null,
              median_NSE: hd.median_nse ?? null,
              mean_NSE: hd.mean_nse ?? null,
              median_KGE: hd.median_kge ?? null,
              n_NSE_le_0: hd.nse_le_zero_count ?? null,
            });
          }
        }
      }
      return jsonResponse({
        entry: foundRow,
        source,
        nse_cdf_101: nseCdf,
        ref_3f_lstm_cdf_101: ref3fCdf,
        ref_1f_daymet_cdf_101: ref1fCdf,
        huc_02_breakdown: hucRows,
      });
    }

    // 8. Per-basin hydrograph comparison endpoint
    if (p.startsWith("/api/hydrographs/basin/")) {
      const rawBid = decodeURIComponent(p.split("/").pop() || "").trim().padStart(8, "0");
      if (!/^\d{8}$/.test(rawBid)) {
        return jsonResponse({ detail: `Invalid basin ID '${rawBid}'.` }, 400, "no-store");
      }
      const [hydroIndex, assetResp] = await Promise.all([
        getHydroIndex(env, url.origin),
        fetchAsset(env, url.origin, `/hydro_assets/basin_${rawBid}.json`),
      ]);
      if (!assetResp || !assetResp.ok) {
        return jsonResponse({ detail: `Basin ${rawBid} not found.` }, 404, "no-store");
      }
      const basinDoc = await assetResp.json();
      const modelsDict = basinDoc.models || {};
      const rawModelsParam = (url.searchParams.get("models") || "").trim();
      const reqList = rawModelsParam ? rawModelsParam.split(",").map((s) => s.trim()).filter(Boolean) : [];
      const resolvedKeys = [];
      for (const rm of reqList) {
        if (CACHED_HYDRO_MODELS_MAP.has(rm) || modelsDict[rm]) resolvedKeys.push(rm);
        else if (CACHED_HYDRO_ENTRY_MAP.has(rm)) resolvedKeys.push(CACHED_HYDRO_ENTRY_MAP.get(rm));
      }
      if (resolvedKeys.length === 0) {
        for (const defKey of [
          "gauged__single_camels__t3_02",
          "gauged__single_camels__t3_18",
          "gauged__single_camels__sacsma_gauged_single_daymet",
        ]) {
          if (CACHED_HYDRO_MODELS_MAP.has(defKey)) resolvedKeys.push(defKey);
        }
      }
      const seriesOut = [];
      for (const mk of resolvedKeys.slice(0, 10)) {
        const mSeries = modelsDict[mk];
        const mMeta = CACHED_HYDRO_MODELS_MAP.get(mk) || {
          model_key: mk,
          entry_id: mk.split("__").pop() || mk,
          leaderboard_id: mk.includes("__") ? mk.split("__").slice(0, 2).join("__") : "",
          model_name: mk.split("__").pop() || mk,
          category: "Contribution",
        };
        if (mSeries) {
          seriesOut.push({
            ...mMeta,
            qsim: mSeries.qsim,
            basin_metrics: mSeries.basin_metrics,
          });
        }
      }
      return jsonResponse({
        basin: basinDoc.basin,
        dates: hydroIndex?.dates || [],
        qobs: basinDoc.qobs,
        forcings: basinDoc.forcings,
        series: seriesOut,
      });
    }

    // 9. Pre-generated PNG paper figures served directly from Cloudflare Assets
    if (p.startsWith("/api/figures/")) {
      const figName = decodeURIComponent(p.slice("/api/figures/".length));
      if (!/^[A-Za-z0-9_.\-]+\.(?:png|pdf|json)$/i.test(figName)) {
        return jsonResponse({ detail: "Invalid figure filename." }, 400, "no-store");
      }
      const figResp = await fetchAsset(env, url.origin, `/api/figures/${figName}`);
      if (!figResp || !figResp.ok) {
        return jsonResponse({ detail: `Figure '${figName}' not found.` }, 404, "no-store");
      }
      const lowerFig = figName.toLowerCase();
      const ct = lowerFig.endsWith(".pdf")
        ? "application/pdf"
        : lowerFig.endsWith(".json")
          ? "application/json; charset=utf-8"
          : "image/png";
      return withHeaders(figResp, {
        "Content-Type": ct,
        "Cache-Control": PREVIEW_GATE_ENABLED ? "private, no-store" : "public, max-age=3600, s-maxage=86400",
      });
    }

    // 10. Unknown /api/* route -> clean 404 JSON
    if (p.startsWith("/api/")) {
      return jsonResponse({ detail: `API endpoint '${p}' not found.` }, 404, "no-store");
    }

    // 11. Static asset passthrough or 404 for missing static/asset files; SPA fallback only for extensionless routes
    const staticResp = await env.ASSETS.fetch(request);
    if (staticResp.status !== 404) {
      return withHeaders(staticResp);
    }
    const isMissingAssetPath =
      p.startsWith("/static/") ||
      p.startsWith("/data_assets/") ||
      p.startsWith("/hydro_assets/") ||
      /\.[A-Za-z0-9]{1,8}$/.test(p);
    if (isMissingAssetPath) {
      return new Response("Not Found", {
        status: 404,
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-store",
          ...SECURITY_HEADERS,
        },
      });
    }
    const fallbackIdx = await fetchAsset(env, url.origin, "/index.html");
    return withHeaders(fallbackIdx, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=300",
    });
  },
};
""".lstrip()


def render_worker_script(
    *,
    preview_gate: bool,
    preview_access_key: str,
    scorer_api_url: str,
) -> str:
    """Render the Cloudflare Worker ES module with configured env flags."""
    return (
        WORKER_JS.replace("__PREVIEW_GATE_ENABLED__", "true" if preview_gate else "false")
        .replace("__PREVIEW_ACCESS_KEY__", json.dumps(preview_access_key))
        .replace('"__SCORER_API_URL__"', json.dumps(scorer_api_url.strip().rstrip("/")))
    )


def build_static_bundle(
    *,
    mip_dir: Path = MIP_DIR,
    portal_dir: Path = PORTAL_DIR,
    stage_dir: Path = DEFAULT_STAGE_DIR,
    explorer_store_path: Path | None = None,
    scrub_private_urls: bool = True,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Build all static Worker assets locally directly from ``mips/camels_us_daily/``."""
    overview_obj, leaderboards_obj, families_obj, hydro_index_obj = build_portal_json_payloads(
        mip_dir, explorer_store_path=explorer_store_path, stage_dir=stage_dir
    )
    if scrub_private_urls:
        overview_obj = sanitize_public_data(overview_obj)
        leaderboards_obj = sanitize_public_data(leaderboards_obj)
        families_obj = sanitize_public_data(families_obj)
        hydro_index_obj = sanitize_public_data(hydro_index_obj)

        for label, payload in (
            ("overview", overview_obj),
            ("leaderboards", leaderboards_obj),
            ("families", families_obj),
            ("hydrographs_index", hydro_index_obj),
        ):
            serialized = json.dumps(payload)
            for marker in PRIVATE_URL_MARKERS:
                if marker.lower() in serialized.lower():
                    raise RuntimeError(f"Sanitization check failed: found {marker!r} in {label}.")

    basin_assets = build_per_basin_static_assets(hydro_index_obj, explorer_store_path, stage_dir)
    site_assets = stage_data_and_portal_assets(
        overview_obj,
        leaderboards_obj,
        families_obj,
        hydro_index_obj,
        mip_dir=mip_dir,
        portal_dir=portal_dir,
        stage_dir=stage_dir,
    )
    all_assets = {**basin_assets, **site_assets}
    summary = {
        "total_assets": len(all_assets),
        "basin_assets": len(basin_assets),
        "site_assets": len(site_assets),
        "n_leaderboards": len(leaderboards_obj["leaderboards"]),
        "total_entries": leaderboards_obj["total_entries"],
        "total_families": len(families_obj["families"]),
        "stage_dir": str(stage_dir),
    }
    return all_assets, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and deploy the standalone HydrologyMIP Cloudflare Worker bundle."
    )
    parser.add_argument(
        "--explorer-store",
        type=Path,
        default=None,
        help="Optional path to hydrograph_explorer_store.npz for regenerating 531 basin JSONs.",
    )
    parser.add_argument(
        "--prebuilt-hydro-assets-url",
        type=str,
        default=None,
        help="HTTPS URL to prebuilt hydro_assets_531.tar.gz cache on R2 when explorer NPZ is absent.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=DEFAULT_STAGE_DIR,
        help="Staging directory for generated static assets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the static asset bundle + Worker JS locally without deploying.",
    )
    parser.add_argument(
        "--allow-partial-hydrographs",
        action="store_true",
        help="Permit deploy without hydrograph_explorer_store.npz or prebuilt cache.",
    )
    args = parser.parse_args(argv)

    env_explorer = os.environ.get("HYDROMIP_EXPLORER_STORE", "").strip()
    explorer_path = args.explorer_store or (Path(env_explorer) if env_explorer else None)
    prebuilt_url = (
        args.prebuilt_hydro_assets_url
        or os.environ.get("HYDROMIP_PREBUILT_HYDRO_ASSETS_URL", "")
    ).strip()

    has_explorer = explorer_path is not None and explorer_path.is_file()
    if not args.dry_run and not has_explorer:
        if prebuilt_url:
            _download_and_unpack_prebuilt_hydro_assets(prebuilt_url, args.stage_dir)
        elif not args.allow_partial_hydrographs:
            raise FileNotFoundError(
                "Live Cloudflare asset uploads replace the Workers asset manifest; pass "
                "--explorer-store <path/to/hydrograph_explorer_store.npz> or "
                "--prebuilt-hydro-assets-url <https://.../hydro_assets_531.tar.gz> so all 531 "
                "per-basin JSON assets are included."
            )

    preview_gate = _is_truthy(os.environ.get("HYDROLOGYMIP_PREVIEW_GATE"), default=True)
    preview_key = (
        (os.environ.get("HYDROLOGYMIP_PREVIEW_ACCESS_KEY", "camels-mip-2026").strip() or "camels-mip-2026")
        if preview_gate
        else ""
    )
    scrub_urls = _is_truthy(os.environ.get("HYDROLOGYMIP_SCRUB_PRIVATE_URLS"), default=True)
    scorer_api_url = (
        os.environ.get("HYDROLOGYMIP_SCORER_API_URL")
        or os.environ.get("SCORER_API_URL")
        or ""
    ).strip()

    all_assets, summary = build_static_bundle(
        mip_dir=MIP_DIR,
        portal_dir=PORTAL_DIR,
        stage_dir=args.stage_dir,
        explorer_store_path=explorer_path,
        scrub_private_urls=scrub_urls,
    )
    worker_js = render_worker_script(
        preview_gate=preview_gate,
        preview_access_key=preview_key,
        scorer_api_url=scorer_api_url,
    )

    if args.dry_run:
        worker_out = args.stage_dir / "worker.js"
        worker_out.parent.mkdir(parents=True, exist_ok=True)
        worker_out.write_text(worker_js, encoding="utf-8")
        print(json.dumps({**summary, "preview_gate": preview_gate, "worker_js": str(worker_out)}, indent=2))
        return 0

    account_id, api_token = load_cloudflare_credentials()
    assets_completion_jwt = upload_cf_static_assets(all_assets, account_id, api_token)

    boundary = "----CloudflareWorkerBoundary987654321"
    metadata = json.dumps(
        {
            "main_module": "worker.js",
            "compatibility_date": "2024-09-23",
            "assets": {
                "jwt": assets_completion_jwt,
                "config": {"run_worker_first": True},
            },
            "bindings": [{"type": "assets", "name": "ASSETS"}],
        }
    )
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{metadata}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="worker.js"; filename="worker.js"\r\n'
        f"Content-Type: application/javascript+module\r\n\r\n"
        f"{worker_js}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    print(f"Worker script bundle size: {len(worker_js.encode('utf-8')) / 1024:.2f} KB")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/workers/scripts/{CF_SCRIPT_NAME}"
    req = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "HydrologyMIP/1.0",
        },
    )
    opener = urllib.request.build_opener(_NonRaisingHTTPErrorProcessor())
    with opener.open(req, timeout=120) as resp:
        code = int(getattr(resp, "status", 200))
        res_text = resp.read().decode("utf-8", errors="replace")
        if code >= 400:
            raise RuntimeError(f"Cloudflare deploy failed ({code}): {res_text}")
        res = json.loads(res_text)
        print("Deploy success:", res.get("success"))
        if not res.get("success"):
            raise RuntimeError(f"Cloudflare deploy failed: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
