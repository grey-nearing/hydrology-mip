# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hardened stateless preview and canonical scoring API for HydrologyMIP.

Designed to live in ``web/scorer_api/app.py``:
- Imports and invokes the canonical 14-metric evaluator from ``hydrology-mip``
  (``metrics.evaluator.calculate_model_leaderboard_metrics`` and
  ``hydromip.metrics.deterministic_daily.calculate_all_metrics``) with zero
  duplicated metric formulas.
- Executes zero ``try``/``except`` blocks to comply with repository AST guardrails.
- Enforces HTTPS + archive host allowlist + DNS global-IP SSRF checks on all
  redirects, 250 MiB upload limits, per-IP sliding-window rate limits, bounded
  ephemeral preview storage with immediate post-score cleanup, and strict CORS.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.stats import wilcoxon
from starlette.datastructures import UploadFile

from fastapi.responses import FileResponse, JSONResponse, Response

from hydromip.io.doi_resolver import (
    resolve_and_download_archive as _shared_resolve_and_download_archive,
    validate_and_fetch_archive_url,
    validate_https_target_url as validate_https_target_url,
)
from hydromip.io.ingest import (
    EXPECTED_NUM_BASINS,
    EXPECTED_NUM_DAYS,
    IngestionRecipe,
    extract_file_metadata,
    ingest_hydrograph,
    resolve_deterministic_recipe,
    run_physical_sanity_checks,
    write_canonical_zarr_v2,
)
from hydromip.io.llm_schema_adapter import (
    build_stuck_report,
    get_llm_backend_mode,
    infer_recipe_from_metadata,
)
from hydromip.metrics.deterministic_daily import (
    calculate_all_metrics,
    get_available_metrics,
)
from hydromip.paths import DEFAULT_MIP_ID, REPO_ROOT, resolve_mip_subdir
from hydromip.protocol import discover_mips, get_mip
from metrics.evaluator import (
    CANONICAL_LEADERBOARD_IDS,
    GAUCH_2023_METRICS,
    calculate_model_leaderboard_metrics,
)

MAX_UPLOAD_BYTES: int = 250 * 1024 * 1024  # 250 MiB
MAX_ACTIVE_TOKENS: int = 20
TOKEN_TTL_SECONDS: int = 15 * 60
RATE_LIMIT_MAX_REQUESTS: int = 15
RATE_LIMIT_WINDOW_SECONDS: int = 60
DEFAULT_MAX_LLM_CALLS_PER_IP_HOUR: int = 3
DEFAULT_MAX_LLM_CALLS_PER_DAY: int = 100
MAX_REDIRECT_HOPS: int = 5
ARCHIVE_TIMEOUT_SECONDS: int = 60

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _preview_root_dir() -> Path:
    env_dir = os.environ.get("HYDROMIP_PREVIEW_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path(tempfile.gettempdir()) / "hydromip_preview"


def _targets_cache_dir() -> Path:
    env_dir = os.environ.get("HYDROMIP_TARGETS_CACHE_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path(tempfile.gettempdir()) / "hydromip_targets_cache"


def _resolve_leaderboards_dir() -> Path | None:
    env_dir = os.environ.get("HYDROMIP_LEADERBOARDS_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(REPO_ROOT / "mips" / "camels_us_daily" / "leaderboards")
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "mips" / "camels_us_daily" / "leaderboards")
        candidates.append(parent / "camels-us-mip" / "mips" / "camels_us_daily" / "leaderboards")
    for cand in candidates:
        if cand.is_dir() and (cand / "gauged__single_camels.json").is_file():
            return cand
    return None


_HERE = Path(__file__).resolve().parent


def _resolve_qobs_npz_path() -> Path | None:
    env_path = os.environ.get("HYDROMIP_QOBS_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_HERE / "data" / "qobs_531.npz")
    for parent in _HERE.parents:
        candidates.append(parent / "data" / "qobs_531.npz")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


@dataclass(frozen=True)
class CanonicalReference:
    """In-memory canonical CAMELS-US 531 observations & metadata."""

    basins: tuple[str, ...]
    dates: tuple[str, ...]
    huc_02: tuple[str, ...]
    gauge_names: tuple[str, ...]
    qobs: np.ndarray
    targets_bucket_dir: Path


_REF_LOCK = threading.Lock()
_CANONICAL_REF: CanonicalReference | None = None


def get_canonical_reference() -> CanonicalReference:
    """Load canonical 531-basin reference once and ensure ``newman_eval_targets.zarr`` exists."""
    global _CANONICAL_REF
    with _REF_LOCK:
        if _CANONICAL_REF is not None:
            return _CANONICAL_REF

        qobs_path = _resolve_qobs_npz_path()
        if qobs_path is None:
            raise FileNotFoundError(
                "Could not locate canonical qobs_531.npz (set HYDROMIP_QOBS_PATH)."
            )

        with np.load(qobs_path, allow_pickle=False) as d:
            basins = tuple(str(b).zfill(8) for b in d["basins"].tolist())
            dates = tuple(str(x) for x in d["dates"].tolist())
            qobs = np.asarray(d["qobs"], dtype=np.float32)
            if "huc_02" in d.files:
                huc_02 = tuple(str(h).zfill(2) for h in d["huc_02"].tolist())
            else:
                huc_02 = ("01",) * len(basins)
            if "gauge_names" in d.files:
                gauge_names = tuple(str(g) for g in d["gauge_names"].tolist())
            else:
                index_cand = qobs_path.parent / "hydrograph_explorer_index.json"
                if index_cand.is_file():
                    idx_doc = json.loads(index_cand.read_text(encoding="utf-8"))
                    b_list = idx_doc.get("basins", [])
                    huc_02 = tuple(str(item["huc_02"]).zfill(2) for item in b_list)
                    gauge_names = tuple(str(item["gauge_name"]) for item in b_list)
                else:
                    gauge_names = basins

        if len(basins) != EXPECTED_NUM_BASINS or len(dates) != EXPECTED_NUM_DAYS:
            raise ValueError(
                f"Invalid qobs bundle dimensions: {len(basins)} basins x {len(dates)} days."
            )

        cache_root = _targets_cache_dir()
        obs_zarr = cache_root / "observations" / "newman_eval_targets.zarr"
        if not (obs_zarr / ".zmetadata").is_file():
            obs_zarr.parent.mkdir(parents=True, exist_ok=True)
            sort_order = np.argsort(np.asarray(basins, dtype="<U8"))
            sorted_basins = np.asarray([basins[i] for i in sort_order], dtype="<U8")
            sorted_huc = np.asarray([huc_02[i] for i in sort_order], dtype="<U2")
            sorted_qobs = np.ascontiguousarray(qobs[sort_order, :], dtype=np.float32)
            date_ns = pd.to_datetime(list(dates)).values.astype("datetime64[ns]")
            ds = xr.Dataset(
                data_vars={"qobs": (("basin", "date"), sorted_qobs, {"units": "mm/day"})},
                coords={
                    "basin": ("basin", sorted_basins),
                    "date": ("date", date_ns),
                    "huc_02": ("basin", sorted_huc),
                },
            )
            ds.to_zarr(
                str(obs_zarr),
                mode="w",
                consolidated=True,
                zarr_format=2,
                encoding={"basin": {"dtype": "<U8"}, "huc_02": {"dtype": "<U2"}},
            )

        _CANONICAL_REF = CanonicalReference(
            basins=basins,
            dates=dates,
            huc_02=huc_02,
            gauge_names=gauge_names,
            qobs=qobs,
            targets_bucket_dir=cache_root,
        )
        return _CANONICAL_REF


def resolve_and_download_archive(raw_archive_url: str, target_dir: Path) -> Path:
    """Resolve a Zenodo/HydroShare/OSF/Figshare DOI or trusted HTTPS URL and download safely."""
    return _shared_resolve_and_download_archive(
        raw_archive_url,
        target_dir,
        max_bytes=MAX_UPLOAD_BYTES,
        allow_file_uri=False,
        fetch_fn=validate_and_fetch_archive_url,
    )


class SlidingWindowRateLimiter:
    """In-memory per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def check_and_record(self, client_ip: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._hits.setdefault(client_ip, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({self.max_requests} requests per {self.window_seconds}s).",
                )
            dq.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class EphemeralPreviewStore:
    """Manages bounded, TTL-expiring preview staging directories."""

    def __init__(
        self,
        root_dir: Path | None = None,
        max_tokens: int = MAX_ACTIVE_TOKENS,
        ttl_seconds: int = TOKEN_TTL_SECONDS,
    ) -> None:
        self._override_root = root_dir
        self.max_tokens = max_tokens
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    @property
    def root_dir(self) -> Path:
        return self._override_root if self._override_root is not None else _preview_root_dir()

    def prune_locked(self) -> None:
        root = self.root_dir
        if not root.exists():
            return
        now = time.time()
        dirs = [p for p in root.iterdir() if p.is_dir()]
        alive: list[tuple[float, Path]] = []
        for d in dirs:
            mtime = float(d.stat().st_mtime)
            if now - mtime > self.ttl_seconds:
                shutil.rmtree(d)
            else:
                alive.append((mtime, d))
        alive.sort(key=lambda item: item[0])
        while len(alive) >= self.max_tokens:
            _, oldest = alive.pop(0)
            if oldest.exists():
                shutil.rmtree(oldest)

    def allocate_token_dir(self) -> tuple[str, Path]:
        with self._lock:
            root = self.root_dir
            root.mkdir(parents=True, exist_ok=True)
            self.prune_locked()
            token = f"pv_{uuid.uuid4().hex[:16]}"
            token_dir = root / token
            token_dir.mkdir(parents=True, exist_ok=True)
            return token, token_dir

    def resolve_token_dir(self, token: str) -> Path:
        if not TOKEN_RE.fullmatch(token):
            raise HTTPException(status_code=400, detail="Invalid preview_token format.")
        root = self.root_dir.resolve()
        candidate = (root / token).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise HTTPException(status_code=400, detail="Invalid preview_token path.")
        if not candidate.is_dir():
            raise HTTPException(
                status_code=404,
                detail="Preview session expired or already completed; please re-run Step 1 inspect.",
            )
        age = time.time() - float(candidate.stat().st_mtime)
        if age > self.ttl_seconds:
            shutil.rmtree(candidate)
            raise HTTPException(status_code=404, detail="Preview token TTL (15 min) expired.")
        return candidate

    def delete_token_dir(self, token_dir: Path) -> None:
        if token_dir.exists():
            shutil.rmtree(token_dir)


def _env_nonneg_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


class LLMBudgetLimiter:
    """Thread-safe in-memory per-IP hourly and global UTC-daily cap for Tier 2 LLM calls."""

    def __init__(
        self,
        max_per_ip_hour: int | None = None,
        max_per_day: int | None = None,
    ) -> None:
        self._max_per_ip_hour = max_per_ip_hour
        self._max_per_day = max_per_day
        self._lock = threading.Lock()
        self._ip_hits: dict[str, deque[float]] = {}
        self._current_day: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_count: int = 0

    @property
    def per_ip_hour_limit(self) -> int:
        if self._max_per_ip_hour is not None:
            return max(0, int(self._max_per_ip_hour))
        return _env_nonneg_int(
            "HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR", DEFAULT_MAX_LLM_CALLS_PER_IP_HOUR
        )

    @property
    def daily_limit(self) -> int:
        if self._max_per_day is not None:
            return max(0, int(self._max_per_day))
        return _env_nonneg_int(
            "HYDROMIP_MAX_LLM_CALLS_PER_DAY", DEFAULT_MAX_LLM_CALLS_PER_DAY
        )

    def _rollover_utc_day_locked(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_day:
            self._current_day = today
            self._daily_count = 0

    def try_acquire(self, client_ip: str) -> tuple[bool, str | None]:
        now = time.monotonic()
        cutoff = now - 3600.0
        day_lim = self.daily_limit
        ip_lim = self.per_ip_hour_limit
        with self._lock:
            self._rollover_utc_day_locked()
            dq = self._ip_hits.setdefault(client_ip, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if self._daily_count >= day_lim:
                return False, (
                    f"Daily AI schema-helper cap reached ({day_lim} calls per UTC day)."
                )
            if len(dq) >= ip_lim:
                return False, (
                    f"Per-IP hourly AI schema-helper cap reached ({ip_lim} calls per hour)."
                )
            self._daily_count += 1
            dq.append(now)
            return True, None

    def daily_remaining(self) -> int:
        lim = self.daily_limit
        with self._lock:
            self._rollover_utc_day_locked()
            return max(0, lim - self._daily_count)

    def reset(self) -> None:
        with self._lock:
            self._ip_hits.clear()
            self._current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._daily_count = 0


RATE_LIMITER = SlidingWindowRateLimiter()
LLM_BUDGET_LIMITER = LLMBudgetLimiter()
PREVIEW_STORE = EphemeralPreviewStore()


def _client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    fwd = request.headers.get("x-forwarded-for", "").strip()
    if fwd:
        return fwd.split(",")[-1].strip()
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


class RecipeOverrideInput(BaseModel):
    var: str | None = None
    basin_dim: str | None = None
    time_dim: str | None = None
    member_dim: str | None = None
    basin_col: str | None = None
    date_col: str | None = None
    value_col: str | None = None
    member_col: str | None = None
    unit_scale: float = 1.0
    allow_positional_order: bool = False


class ScoreMetadataInput(BaseModel):
    leaderboard_id: str = "gauged__single_camels"
    spatial_regime: str = "gauged"
    forcing: str = "Daymet"
    model_name: str = "Community Model"
    family_id: str | None = None
    new_family_name: str | None = None
    paradigm: str = "Machine learning"
    category: str = "Contribution"
    ensemble_info: str = ""
    code_url: str = ""
    archive_url: str = ""
    citation: str = "Community Contribution (2026)"


class ScorePreviewRequest(BaseModel):
    preview_token: str
    recipe: RecipeOverrideInput = Field(default_factory=RecipeOverrideInput)
    metadata: ScoreMetadataInput = Field(default_factory=ScoreMetadataInput)


class ScoreMemberRequest(BaseModel):
    preview_token: str
    member_index: int


MAX_AUTO_SCORED_MEMBERS: int = 10
ENSEMBLE_SUFFIX_RE = re.compile(r"\s*\(\d+-model ensemble\)\s*$")


def _format_ensemble_model_name(base_name: str, is_ensemble: bool, num_members: int) -> str:
    cleaned = str(base_name or "").strip()
    if is_ensemble and int(num_members) >= 2 and "ensemble" not in cleaned.lower():
        return f"{cleaned} ({int(num_members)}-model ensemble)"
    return cleaned


def _format_single_model_name(base_name: str) -> str:
    stripped = ENSEMBLE_SUFFIX_RE.sub("", str(base_name or "").strip()).strip() or str(base_name or "").strip()
    return f"{stripped} (single model)"


def _default_member_labels(num_members: int) -> list[str]:
    return [f"Member {i + 1}" for i in range(max(0, int(num_members)))]


def _schema_summary_from_meta(meta: Mapping[str, Any], recipe: IngestionRecipe | None) -> dict[str, Any]:
    avail_vars = list(meta.get("vars") or meta.get("columns") or [])
    avail_dims = list(meta.get("dims") or meta.get("columns") or [])
    member_dim = (recipe.member_dim or recipe.member_col) if recipe else None
    if member_dim and member_dim not in avail_dims:
        avail_dims.append(member_dim)
    return {
        "var": (recipe.var or recipe.value_col) if recipe else (avail_vars[0] if avail_vars else None),
        "basin_dim": (recipe.basin_dim or recipe.basin_col) if recipe else None,
        "time_dim": (recipe.time_dim or recipe.date_col) if recipe else None,
        "member_dim": member_dim,
        "basin_col": recipe.basin_col if recipe else None,
        "date_col": recipe.date_col if recipe else None,
        "value_col": recipe.value_col if recipe else None,
        "member_col": recipe.member_col if recipe else None,
        "available_vars": avail_vars,
        "available_dims": avail_dims,
    }


def _load_live_leaderboard_rows(leaderboard_id: str) -> list[dict[str, Any]]:
    lb_dir = _resolve_leaderboards_dir()
    if lb_dir is None:
        return []
    lb_file = lb_dir / f"{leaderboard_id}.json"
    if not lb_file.is_file():
        return []
    doc = json.loads(lb_file.read_text(encoding="utf-8"))
    rows = doc.get("rows", [])
    return list(rows) if isinstance(rows, list) else []


def _paired_wilcoxon_p(vec_a: np.ndarray, vec_b: Sequence[Any] | None) -> float | None:
    if not vec_b or len(vec_b) != len(vec_a):
        return None
    b_arr = np.asarray([float(v) if v is not None else np.nan for v in vec_b], dtype=np.float64)
    mask = np.isfinite(vec_a) & np.isfinite(b_arr)
    if int(np.sum(mask)) < 10:
        return None
    diff = vec_a[mask] - b_arr[mask]
    if np.allclose(diff, 0.0, atol=1e-12):
        return 1.0
    res = wilcoxon(vec_a[mask], b_arr[mask], zero_method="wilcox", alternative="two-sided")
    pval = float(res.pvalue)
    return round(pval, 6) if np.isfinite(pval) else None


def _build_rich_huc02_summary(
    per_basin_nse: Sequence[float],
    per_basin_kge: Sequence[float],
    huc_02: Sequence[str],
) -> dict[str, dict[str, Any]]:
    nse_arr = np.asarray(per_basin_nse, dtype=np.float64)
    kge_arr = np.asarray(per_basin_kge, dtype=np.float64)
    huc_arr = np.asarray(huc_02, dtype=str)
    out: dict[str, dict[str, Any]] = {}
    for h in sorted(set(huc_arr)):
        mask = huc_arr == h
        hn = nse_arr[mask]
        hn = hn[np.isfinite(hn)]
        hk = kge_arr[mask]
        hk = hk[np.isfinite(hk)]
        out[str(h)] = {
            "num_basins": len(hn),
            "median_nse": round(float(np.median(hn)), 4) if len(hn) else None,
            "mean_nse": round(float(np.mean(hn)), 4) if len(hn) else None,
            "median_kge": round(float(np.median(hk)), 4) if len(hk) else None,
            "nse_le_zero_count": int(np.sum(hn <= 0.0)) if len(hn) else 0,
        }
    return out


def _evaluator_result_to_entry(
    eval_res: Mapping[str, Any],
    *,
    entry_id: str,
    meta_in: ScoreMetadataInput,
    ref: CanonicalReference,
    lb_rows: list[dict[str, Any]],
    hydrograph_samples: dict[str, Any] | None = None,
    model_name_override: str | None = None,
    selected_member_index: int | None = None,
    selected_member_label: str | None = None,
) -> dict[str, Any]:
    gauch = dict(eval_res["gauch_2023_metrics"])
    per_basin = eval_res["per_basin"]
    per_basin_nse = list(per_basin["NSE"])
    per_basin_kge = list(per_basin["KGE"])
    median_nse = float(eval_res["median_nse"])

    sota_row = lb_rows[0] if lb_rows else None
    p_nse = _paired_wilcoxon_p(
        np.asarray(per_basin_nse, dtype=np.float64),
        sota_row.get("per_basin_nse") if sota_row else None,
    )
    p_kge = _paired_wilcoxon_p(
        np.asarray(per_basin_kge, dtype=np.float64),
        sota_row.get("per_basin_kge") if sota_row else None,
    )
    rank = 1 + sum(
        1
        for r in lb_rows
        if r.get("primary_ranking_nse") is not None and float(r["primary_ranking_nse"]) > median_nse
    )

    # Sort huc_02 to match sorted canonical basins returned inside per_basin
    sort_order = np.argsort(np.asarray(ref.basins, dtype="<U8"))
    sorted_huc02 = [ref.huc_02[i] for i in sort_order]
    regional_huc02 = _build_rich_huc02_summary(per_basin_nse, per_basin_kge, sorted_huc02)

    is_ens = bool(eval_res["is_ensemble"])
    num_mem = int(eval_res["num_members"])
    if model_name_override is not None:
        model_label = model_name_override
    elif is_ens and num_mem >= 2:
        model_label = _format_ensemble_model_name(meta_in.model_name, True, num_mem)
    elif not is_ens and num_mem > 1:
        model_label = _format_single_model_name(meta_in.model_name)
    else:
        model_label = meta_in.model_name

    family_id = meta_in.family_id or f"preview_{entry_id}"
    family_name = meta_in.new_family_name or meta_in.model_name

    entry: dict[str, Any] = {
        "entry_id": entry_id,
        "leaderboard_id": meta_in.leaderboard_id,
        "model_name": model_label,
        "family_id": family_id,
        "family_name": family_name,
        "paradigm": meta_in.paradigm,
        "category": meta_in.category,
        "forcing": meta_in.forcing,
        "ensemble_info": meta_in.ensemble_info,
        "is_ensemble": is_ens,
        "num_members": num_mem,
        "has_qsim_members": bool(eval_res.get("has_qsim_members", False)),
        "ensemble_source": str(eval_res["ensemble_source"]),
        "ens_med_nse": round(median_nse, 4) if is_ens else None,
        "single_med_nse": eval_res.get("single_seed_median_nse_mean"),
        "single_med_nse_std": eval_res.get("single_seed_median_nse_std"),
        "single_med_nse_per_member": eval_res.get("single_seed_median_nse_per_member"),
        "primary_ranking_nse": round(median_nse, 4),
        "mean_nse": float(eval_res["mean_nse"]),
        "med_kge": float(eval_res["median_kge"]),
        "mean_kge": float(eval_res["mean_kge"]) if eval_res.get("mean_kge") is not None else float(gauch["KGE"]["mean"]),
        "med_fhv_abs": float(eval_res["median_abs_fhv"]),
        "gauch_fhv": float(gauch["FHV"]["median"]),
        "nse_le_0": int(eval_res["nse_le_zero_count"]),
        "num_evaluated_basins": int(eval_res.get("num_evaluated_basins", len(ref.basins))),
        "pred_nan_when_obs_valid": int(eval_res["pred_nan_when_obs_valid"]),
        "per_basin_nse": per_basin_nse,
        "per_basin_kge": per_basin_kge,
        "gauch_2023_metrics": gauch,
        "regional_huc02": regional_huc02,
        "p_value_vs_sota_nse": p_nse,
        "p_value_vs_sota_kge": p_kge,
        "rank": rank,
        "citation": meta_in.citation,
        "code_url": meta_in.code_url,
        "github_url": meta_in.code_url,
        "data_archive_url": meta_in.archive_url,
        "is_local_preview": True,
        "hydrograph_samples": hydrograph_samples,
    }
    if selected_member_index is not None:
        entry["selected_member_index"] = int(selected_member_index)
    if selected_member_label is not None:
        entry["selected_member_label"] = str(selected_member_label)
    return entry


def _build_full_hydrograph_samples(
    qsim: np.ndarray,
    ref: CanonicalReference,
    sample_basins: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    basin_to_idx = {b: i for i, b in enumerate(ref.basins)}
    dates_ns = pd.to_datetime(list(ref.dates)).values.astype("datetime64[ns]")
    samples: dict[str, dict[str, Any]] = {}
    for sb in sample_basins:
        bid = str(sb["basin_id"])
        idx = basin_to_idx.get(bid)
        if idx is None:
            continue
        obs_b = np.where((~np.isnan(ref.qobs[idx])) & (ref.qobs[idx] >= 0.0), ref.qobs[idx], np.nan)
        sim_b = np.asarray(qsim[idx], dtype=np.float64)
        obs_da = xr.DataArray(obs_b, coords={"date": dates_ns}, dims=("date",))
        sim_da = xr.DataArray(sim_b, coords={"date": dates_ns}, dims=("date",))
        b_metrics = calculate_all_metrics(obs_da, sim_da, resolution="1D", datetime_coord="date")
        samples[bid] = {
            "qsim": [None if not np.isfinite(v) else round(float(v), 3) for v in sim_b],
            "basin_metrics": {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in b_metrics.items()},
            "nse": sb.get("nse"),
        }
    return samples


def _run_canonical_scoring_sync(
    staged_path: Path,
    token_dir: Path,
    recipe_input: RecipeOverrideInput,
    meta_in: ScoreMetadataInput,
) -> dict[str, Any]:
    if meta_in.leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise ValueError(
            f"Invalid leaderboard_id {meta_in.leaderboard_id!r}; expected one of {CANONICAL_LEADERBOARD_IDS}."
        )
    ref = get_canonical_reference()
    base_recipe_raw = json.loads((token_dir / "resolved_recipe.json").read_text(encoding="utf-8"))

    merged_recipe = {**base_recipe_raw}
    for k in ("var", "basin_dim", "time_dim", "member_dim", "basin_col", "date_col", "value_col", "member_col"):
        override_v = getattr(recipe_input, k)
        if override_v is not None:
            merged_recipe[k] = override_v
    if recipe_input.member_dim is None and "member_dim" in recipe_input.model_fields_set:
        merged_recipe["member_dim"] = None
    merged_recipe["unit_scale"] = recipe_input.unit_scale
    merged_recipe["allow_positional_order"] = recipe_input.allow_positional_order
    recipe = IngestionRecipe.model_validate(merged_recipe)

    ingested = ingest_hydrograph(
        staged_path,
        recipe,
        ref.basins,
        ref.dates,
        strict=True,
        work_dir=token_dir / "extract_score",
    )
    qsim = ingested["qsim"]
    qsim_members = ingested["qsim_members"]

    sanity, sample_basins = run_physical_sanity_checks(
        qsim,
        ref.qobs,
        ref.basins,
        ref.dates,
        matched_basin_mask=ingested["matched_basin_mask"],
        matched_date_mask=ingested["matched_date_mask"],
        gauge_names=ref.gauge_names,
        initial_diagnostics=ingested["diagnostics"],
    )
    if sanity["nan_count"] > 0 or sanity["inf_count"] > 0:
        raise ValueError(
            f"Submission contains {sanity['nan_count']} NaNs or {sanity['inf_count']} Inf values on valid observation days."
        )

    safe_slug = SAFE_SLUG_RE.sub("_", meta_in.model_name.lower()).strip("_")[:24] or "model"
    entry_id = f"prev_{safe_slug}_{uuid.uuid4().hex[:6]}"
    score_bucket = token_dir / "score_bucket"
    score_bucket.mkdir(parents=True, exist_ok=True)
    obs_link = score_bucket / "observations"
    obs_link.symlink_to(ref.targets_bucket_dir / "observations")

    zarr_path = score_bucket / "hydrographs" / meta_in.leaderboard_id / f"{entry_id}.zarr"
    write_canonical_zarr_v2(
        qsim,
        ref.basins,
        zarr_path,
        metadata={
            "model_id": entry_id,
            "model_name": meta_in.model_name,
            "leaderboard_id": meta_in.leaderboard_id,
            "forcing": meta_in.forcing,
            "is_ensemble": bool(qsim_members is not None and qsim_members.shape[0] > 1),
            "num_members": int(qsim_members.shape[0]) if qsim_members is not None else 1,
            "ensemble_reduction": "mean",
        },
        qsim_members=qsim_members,
        huc_02=ref.huc_02,
        dates=ref.dates,
    )

    n_members_total = int(qsim_members.shape[0]) if qsim_members is not None else 1
    eval_res = calculate_model_leaderboard_metrics(
        model_id=entry_id,
        leaderboard_id=meta_in.leaderboard_id,
        bucket_uri=str(score_bucket),
        allow_partial_archive=False,
        evaluation_mode=None,
        max_scored_members=MAX_AUTO_SCORED_MEMBERS if n_members_total > MAX_AUTO_SCORED_MEMBERS else None,
    )
    lb_rows = _load_live_leaderboard_rows(meta_in.leaderboard_id)
    hydrograph_samples = _build_full_hydrograph_samples(qsim, ref, sample_basins)
    primary_entry = _evaluator_result_to_entry(
        eval_res,
        entry_id=entry_id,
        meta_in=meta_in,
        ref=ref,
        lb_rows=lb_rows,
        hydrograph_samples=hydrograph_samples,
    )

    ensemble_single_entry: dict[str, Any] | None = None
    ensemble_members_payload: dict[str, Any] | None = None
    if qsim_members is not None and n_members_total >= 2:
        member_labels = _default_member_labels(n_members_total)
        single_eval = calculate_model_leaderboard_metrics(
            model_id=entry_id,
            leaderboard_id=meta_in.leaderboard_id,
            bucket_uri=str(score_bucket),
            allow_partial_archive=False,
            evaluation_mode="single_member_mean",
            max_scored_members=MAX_AUTO_SCORED_MEMBERS,
            include_per_member_results=True,
        )
        single_name = _format_single_model_name(primary_entry["model_name"])
        ensemble_single_entry = _evaluator_result_to_entry(
            single_eval,
            entry_id=f"{entry_id}_single",
            meta_in=meta_in,
            ref=ref,
            lb_rows=lb_rows,
            hydrograph_samples=hydrograph_samples,
            model_name_override=single_name,
        )
        scored_members_list: list[dict[str, Any]] = []
        for m_res in single_eval.get("per_member_results") or []:
            m_idx = int(m_res["member_index"])
            m_lbl = member_labels[m_idx] if m_idx < len(member_labels) else f"Member {m_idx + 1}"
            m_entry = _evaluator_result_to_entry(
                m_res,
                entry_id=f"{entry_id}_single",
                meta_in=meta_in,
                ref=ref,
                lb_rows=lb_rows,
                hydrograph_samples=hydrograph_samples,
                model_name_override=single_name,
                selected_member_index=m_idx,
                selected_member_label=m_lbl,
            )
            scored_members_list.append(
                {
                    "member_index": m_idx,
                    "member_label": m_lbl,
                    "entry": m_entry,
                }
            )
        truncated = bool(single_eval.get("truncated_at_member_limit", n_members_total > MAX_AUTO_SCORED_MEMBERS))
        ensemble_members_payload = {
            "total_members": n_members_total,
            "auto_score_limit": MAX_AUTO_SCORED_MEMBERS,
            "scored_count": len(scored_members_list),
            "truncated": truncated,
            "member_labels": member_labels,
            "mean_entry": ensemble_single_entry,
            "members": scored_members_list,
        }
        if truncated:
            score_ctx = {
                "entry_id": entry_id,
                "leaderboard_id": meta_in.leaderboard_id,
                "primary_model_name": primary_entry["model_name"],
                "single_model_name": single_name,
                "total_members": n_members_total,
                "member_labels": member_labels,
                "metadata": meta_in.model_dump(),
                "sample_basins": sample_basins,
            }
            (token_dir / "score_context.json").write_text(
                json.dumps(score_ctx, indent=2), encoding="utf-8"
            )

    canonical_manifest = {
        "schema_version": "1.1",
        "benchmark": "camels_us_daily",
        "leaderboard_id": meta_in.leaderboard_id,
        "spatial_regime": meta_in.spatial_regime,
        "model_name": meta_in.model_name,
        "family_id": primary_entry["family_id"],
        "family_name": primary_entry["family_name"],
        "forcing": meta_in.forcing,
        "paradigm": meta_in.paradigm,
        "category": meta_in.category,
        "archive_url": meta_in.archive_url,
        "code_url": meta_in.code_url,
        "citation": meta_in.citation,
        "confirmed_recipe": recipe.model_dump(),
        "preview_summary": {
            "median_nse": primary_entry["primary_ranking_nse"],
            "mean_nse": primary_entry["mean_nse"],
            "median_kge": primary_entry["med_kge"],
            "nse_le_0": primary_entry["nse_le_0"],
        },
    }

    return {
        "status": "scored",
        "local_preview_only": True,
        "preview_token": token_dir.name if (ensemble_members_payload and ensemble_members_payload["truncated"]) else None,
        "entry": primary_entry,
        "ensemble_single_entry": ensemble_single_entry,
        "ensemble_members": ensemble_members_payload,
        "canonical_manifest": canonical_manifest,
        "sanity": sanity,
        "sample_basins": sample_basins,
    }


def _run_inspect_sync(
    staged_path: Path,
    token: str,
    token_dir: Path,
    leaderboard_id: str,
    recipe_override: Mapping[str, Any] | None,
    *,
    client_ip: str = "unknown",
) -> dict[str, Any]:
    ref = get_canonical_reference()
    meta = extract_file_metadata(staged_path, work_dir=token_dir / "extract_inspect")
    (token_dir / "file_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    recipe, unresolved = resolve_deterministic_recipe(meta, recipe_override=recipe_override)
    stuck_report: dict[str, Any] | None = None

    if recipe is None:
        if get_llm_backend_mode() == "disabled":
            llm_res = infer_recipe_from_metadata(
                meta,
                unresolved_reasons=unresolved,
                leaderboard_id=leaderboard_id,
            )
            recipe = llm_res["recipe"]
            stuck_report = llm_res["stuck_report"]
        else:
            allowed, cap_reason = LLM_BUDGET_LIMITER.try_acquire(client_ip)
            if not allowed:
                stuck_report = build_stuck_report(
                    reason=(
                        f"{cap_reason} Automated AI schema mapping is paused to stay within "
                        "public quota limits; select your dimensions manually below or package "
                        "locally with `hydromip package`."
                    ),
                    missing_or_ambiguous=unresolved
                    or ["Select axes manually below or package locally with `hydromip package`."],
                    metadata=meta,
                    leaderboard_id=leaderboard_id,
                )
            else:
                llm_res = infer_recipe_from_metadata(
                    meta,
                    unresolved_reasons=unresolved,
                    leaderboard_id=leaderboard_id,
                )
                recipe = llm_res["recipe"]
                stuck_report = llm_res["stuck_report"]

    if recipe is None:
        fallback_fmt = str(meta.get("format") or "netcdf")
        (token_dir / "resolved_recipe.json").write_text(
            json.dumps({"format": fallback_fmt, "recipe_source": "deterministic_tier1"}),
            encoding="utf-8",
        )
        report = stuck_report or build_stuck_report(
            reason="Unable to resolve hydrograph dimensions automatically.",
            missing_or_ambiguous=unresolved,
            metadata=meta,
            leaderboard_id=leaderboard_id,
        )
        return {
            "status": "ok",
            "preview_token": token,
            "format": fallback_fmt,
            "recipe_source": "deterministic_tier1",
            "schema": _schema_summary_from_meta(meta, None),
            "sanity": {
                "n_basins_matched": 0,
                "n_basins_expected": EXPECTED_NUM_BASINS,
                "nan_count": 0,
                "date_start": "",
                "date_end": "",
                "median_sim_obs_ratio": None,
                "unit_warning": None,
                "checks_passed": False,
                "diagnostics": [report["reason"], *report["missing_or_ambiguous"]],
            },
            "sample_basins": [],
            "stuck_report": report,
        }

    (token_dir / "resolved_recipe.json").write_text(recipe.model_dump_json(indent=2), encoding="utf-8")
    ingested = ingest_hydrograph(
        staged_path,
        recipe,
        ref.basins,
        ref.dates,
        strict=False,
        work_dir=token_dir / "extract_preview",
    )
    sanity, sample_basins = run_physical_sanity_checks(
        ingested["qsim"],
        ref.qobs,
        ref.basins,
        ref.dates,
        matched_basin_mask=ingested["matched_basin_mask"],
        matched_date_mask=ingested["matched_date_mask"],
        gauge_names=ref.gauge_names,
        initial_diagnostics=ingested["diagnostics"],
    )
    n_members_detected = (
        int(ingested["qsim_members"].shape[0])
        if ingested["qsim_members"] is not None
        else 1
    )
    schema_summary = _schema_summary_from_meta(meta, recipe)
    if n_members_detected >= 2 and not schema_summary.get("member_dim"):
        excluded = {schema_summary.get("basin_dim"), schema_summary.get("time_dim")}
        rem_dims = [d for d in schema_summary.get("available_dims", []) if d and d not in excluded]
        inferred_md = rem_dims[0] if len(rem_dims) == 1 else "member"
        schema_summary["member_dim"] = inferred_md
        if inferred_md not in schema_summary["available_dims"]:
            schema_summary["available_dims"].append(inferred_md)
    return {
        "status": "ok",
        "preview_token": token,
        "format": recipe.format,
        "recipe_source": recipe.recipe_source,
        "schema": schema_summary,
        "ensemble_detected": {
            "is_ensemble": n_members_detected >= 2,
            "num_members": n_members_detected,
            "auto_score_limit": MAX_AUTO_SCORED_MEMBERS,
            "truncated": n_members_detected > MAX_AUTO_SCORED_MEMBERS,
            "member_labels": _default_member_labels(n_members_detected) if n_members_detected >= 2 else [],
        },
        "sanity": sanity,
        "sample_basins": sample_basins,
        "stuck_report": None,
    }


def _cors_origins() -> list[str]:
    origins = ["https://hydrologymip.org", "https://www.hydrologymip.org"]
    if os.environ.get("HYDROMIP_DEV_CORS", "") == "1":
        origins.extend(
            [
                "http://localhost:8080",
                "http://127.0.0.1:8080",
                "http://localhost:8090",
                "http://127.0.0.1:8090",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        )
    return origins


app = FastAPI(title="HydrologyMIP Preview Scorer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(ValueError)
async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc), "detail": str(exc)})


@app.exception_handler(KeyError)
async def _key_error_handler(_request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc), "detail": str(exc)})


@app.get("/health")
@app.get("/api/preview/health")
async def health() -> dict[str, Any]:
    """Return service readiness, canonical metric count, and LLM/qobs status."""
    qobs_ready = _CANONICAL_REF is not None or _resolve_qobs_npz_path() is not None
    metric_names = list(GAUCH_2023_METRICS) or list(get_available_metrics())
    llm_backend = get_llm_backend_mode()
    return {
        "status": "ok",
        "metrics_count": len(metric_names),
        "metrics": metric_names,
        "llm_backend": llm_backend,
        "llm_adapter_enabled": llm_backend != "disabled",
        "llm_daily_limit": LLM_BUDGET_LIMITER.daily_limit,
        "llm_daily_remaining": LLM_BUDGET_LIMITER.daily_remaining(),
        "qobs_loaded": qobs_ready,
    }


def _append_upload_chunk(path: Path, chunk: bytes, *, create: bool) -> None:
    mode = "wb" if create else "ab"
    with open(path, mode) as dst:
        dst.write(chunk)


def _record_staged_path_and_inspect(
    staged_path: Path,
    token: str,
    token_dir: Path,
    leaderboard_id: str,
    recipe_override: Mapping[str, Any] | None,
    *,
    client_ip: str = "unknown",
) -> dict[str, Any]:
    (token_dir / "staged_path.txt").write_text(str(staged_path), encoding="utf-8")
    return _run_inspect_sync(
        staged_path,
        token,
        token_dir,
        leaderboard_id,
        recipe_override,
        client_ip=client_ip,
    )


def _resolve_download_and_inspect(
    archive_url: str,
    token: str,
    token_dir: Path,
    leaderboard_id: str,
    recipe_override: Mapping[str, Any] | None,
    *,
    client_ip: str = "unknown",
) -> dict[str, Any]:
    staged_path = resolve_and_download_archive(archive_url, token_dir)
    return _record_staged_path_and_inspect(
        staged_path,
        token,
        token_dir,
        leaderboard_id,
        recipe_override,
        client_ip=client_ip,
    )


def _score_and_cleanup_sync(payload: ScorePreviewRequest) -> dict[str, Any]:
    token_dir = PREVIEW_STORE.resolve_token_dir(payload.preview_token)
    staged_ref = token_dir / "staged_path.txt"
    if not staged_ref.is_file():
        PREVIEW_STORE.delete_token_dir(token_dir)
        raise HTTPException(status_code=404, detail="Staged file metadata missing for preview_token.")
    staged_path = Path(staged_ref.read_text(encoding="utf-8").strip())
    if not staged_path.is_file():
        PREVIEW_STORE.delete_token_dir(token_dir)
        raise HTTPException(status_code=404, detail="Staged archive file is no longer available.")

    result = _run_canonical_scoring_sync(
        staged_path,
        token_dir,
        payload.recipe,
        payload.metadata,
    )
    em = result.get("ensemble_members") or {}
    if bool(em.get("truncated")) and (token_dir / "score_context.json").is_file():
        # Purge raw upload/extract intermediates immediately; retain compact Zarr store until 15-min TTL
        # so the contributor can score additional individual members (> 10) on demand.
        if staged_path.is_file():
            staged_path.unlink()
        for sub in ("extract_inspect", "extract_preview", "extract_score"):
            sub_dir = token_dir / sub
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
    else:
        PREVIEW_STORE.delete_token_dir(token_dir)
    return result


def _score_single_member_sync(payload: ScoreMemberRequest) -> dict[str, Any]:
    token_dir = PREVIEW_STORE.resolve_token_dir(payload.preview_token)
    ctx_path = token_dir / "score_context.json"
    score_bucket = token_dir / "score_bucket"
    if not ctx_path.is_file() or not score_bucket.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Staged ensemble session expired; please re-run Step 1 & Step 2 to score additional members.",
        )
    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    total_members = int(ctx["total_members"])
    m_idx = int(payload.member_index)
    if m_idx < 0 or m_idx >= total_members:
        raise HTTPException(
            status_code=400,
            detail=f"member_index {m_idx} out of range (expected 0..{total_members - 1}).",
        )
    ref = get_canonical_reference()
    entry_id = str(ctx["entry_id"])
    leaderboard_id = str(ctx["leaderboard_id"])
    meta_in = ScoreMetadataInput.model_validate(ctx["metadata"])
    member_labels = list(ctx.get("member_labels") or _default_member_labels(total_members))
    m_lbl = member_labels[m_idx] if m_idx < len(member_labels) else f"Member {m_idx + 1}"
    single_name = str(ctx.get("single_model_name") or _format_single_model_name(meta_in.model_name))

    m_eval = calculate_model_leaderboard_metrics(
        model_id=entry_id,
        leaderboard_id=leaderboard_id,
        bucket_uri=str(score_bucket),
        allow_partial_archive=False,
        evaluation_mode="single_member",
        member_index=m_idx,
    )
    lb_rows = _load_live_leaderboard_rows(leaderboard_id)
    entry = _evaluator_result_to_entry(
        m_eval,
        entry_id=f"{entry_id}_single",
        meta_in=meta_in,
        ref=ref,
        lb_rows=lb_rows,
        model_name_override=single_name,
        selected_member_index=m_idx,
        selected_member_label=m_lbl,
    )
    return {
        "status": "scored",
        "member_index": m_idx,
        "member_label": m_lbl,
        "entry": entry,
    }


@app.post("/api/preview/inspect")
async def inspect_preview(request: Request) -> dict[str, Any]:
    """Inspect uploaded or SSRF-checked remote archive and run 531-basin physical sanity checks."""
    client_ip = _client_ip(request)
    RATE_LIMITER.check_and_record(client_ip)
    content_type = request.headers.get("content-type", "").lower()

    upload_obj: UploadFile | None = None
    archive_url = ""
    leaderboard_id = "gauged__single_camels"
    recipe_override: dict[str, Any] | None = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        maybe_file = form.get("file")
        if isinstance(maybe_file, UploadFile):
            upload_obj = maybe_file
        archive_url = str(form.get("archive_url") or "").strip()
        leaderboard_id = str(form.get("leaderboard_id") or "gauged__single_camels").strip()
        raw_override = str(form.get("recipe_override") or "").strip()
        if raw_override:
            parsed_ov = json.loads(raw_override)
            if isinstance(parsed_ov, dict):
                recipe_override = parsed_ov
    else:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        archive_url = str(body.get("archive_url") or "").strip()
        leaderboard_id = str(body.get("leaderboard_id") or "gauged__single_camels").strip()
        if isinstance(body.get("recipe_override"), dict):
            recipe_override = body["recipe_override"]

    if upload_obj is None and not archive_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a multipart hydrograph file upload or a trusted HTTPS archive_url / DOI.",
        )

    token, token_dir = await asyncio.to_thread(PREVIEW_STORE.allocate_token_dir)
    if upload_obj is not None:
        safe_name = Path(upload_obj.filename or "upload.nc").name
        staged_path = token_dir / safe_name
        total_bytes = 0
        chunk_bytes = 1024 * 1024
        first_chunk = True
        while True:
            buf = await upload_obj.read(chunk_bytes)
            if not buf:
                if first_chunk:
                    await asyncio.to_thread(_append_upload_chunk, staged_path, b"", create=True)
                break
            total_bytes += len(buf)
            if total_bytes > MAX_UPLOAD_BYTES:
                await asyncio.to_thread(PREVIEW_STORE.delete_token_dir, token_dir)
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MiB).",
                )
            await asyncio.to_thread(_append_upload_chunk, staged_path, buf, create=first_chunk)
            first_chunk = False
        return await asyncio.to_thread(
            _record_staged_path_and_inspect,
            staged_path,
            token,
            token_dir,
            leaderboard_id,
            recipe_override,
            client_ip=client_ip,
        )

    return await asyncio.to_thread(
        _resolve_download_and_inspect,
        archive_url,
        token,
        token_dir,
        leaderboard_id,
        recipe_override,
        client_ip=client_ip,
    )


@app.post("/api/preview/score")
async def score_preview(request: Request, payload: ScorePreviewRequest) -> dict[str, Any]:
    """Run canonical metric evaluation for a staged token and purge ephemeral temp files."""
    RATE_LIMITER.check_and_record(_client_ip(request))
    return await asyncio.to_thread(_score_and_cleanup_sync, payload)


@app.post("/api/preview/score-member")
async def score_preview_member(request: Request, payload: ScoreMemberRequest) -> dict[str, Any]:
    """Score one additional individual ensemble member on demand (for ensembles > 10 members)."""
    RATE_LIMITER.check_and_record(_client_ip(request))
    return await asyncio.to_thread(_score_single_member_sync, payload)


def _active_mips_root() -> Path:
    for var in ("CAMELS_MIP_DATA_DIR", "HYDROMIP_DATA_DIR", "HYDROMIP_REPO_ROOT"):
        raw = os.environ.get(var, "").strip()
        if raw and (Path(raw) / "mips").is_dir():
            return Path(raw)
    return REPO_ROOT


@app.get("/api/mips")
def api_list_mips() -> dict[str, Any]:
    """Auto-discover and return all registered MIP protocols under ``mips/*/protocol.json``."""
    repo_for_mips = _active_mips_root()
    mips = discover_mips(repo_root=repo_for_mips)
    return {
        "default_mip_id": DEFAULT_MIP_ID,
        "n_mips": len(mips),
        "mips": [proto.to_summary_dict() for proto in mips.values()],
    }


@app.get("/api/mips/{mip_id}/protocol")
def api_mip_protocol(mip_id: str) -> dict[str, Any]:
    """Return the declarative protocol specification for a registered MIP."""
    repo_for_mips = _active_mips_root()
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
    split_path = resolve_mip_subdir(_active_mips_root(), "splits", DEFAULT_MIP_ID) / safe_name
    if not split_path.is_file():
        raise HTTPException(status_code=404, detail=f"Split file '{safe_name}' not found.")
    media_type = "application/json" if safe_name.endswith(".json") else "text/plain; charset=utf-8"
    return FileResponse(str(split_path), media_type=media_type)


@app.get("/api/mips/{mip_id}/splits/{filename}")
def api_get_mip_split_file(mip_id: str, filename: str) -> Response:
    """Serve canonical split files (.json, .txt) for a specific registered MIP."""
    repo_for_mips = _active_mips_root()
    mips = discover_mips(repo_root=repo_for_mips)
    if mip_id not in mips:
        raise HTTPException(status_code=404, detail=f"Unknown mip_id '{mip_id}'.")
    safe_name = Path(filename).name
    split_path = resolve_mip_subdir(repo_for_mips, "splits", mip_id) / safe_name
    if not split_path.is_file():
        raise HTTPException(status_code=404, detail=f"Split file '{safe_name}' not found for '{mip_id}'.")
    media_type = "application/json" if safe_name.endswith(".json") else "text/plain; charset=utf-8"
    return FileResponse(str(split_path), media_type=media_type)


@app.get("/api/figures/{filename}")
def api_get_figure(filename: str) -> Response:
    """Serve generated paper figures (.png, .pdf, .json) for the default MIP."""
    safe_name = Path(filename).name
    fig_path = resolve_mip_subdir(_active_mips_root(), "figures", DEFAULT_MIP_ID) / safe_name
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
    repo_for_mips = _active_mips_root()
    mips = discover_mips(repo_root=repo_for_mips)
    if mip_id not in mips:
        raise HTTPException(status_code=404, detail=f"Unknown mip_id '{mip_id}'.")
    safe_name = Path(filename).name
    fig_path = resolve_mip_subdir(repo_for_mips, "figures", mip_id) / safe_name
    if not fig_path.is_file():
        raise HTTPException(status_code=404, detail=f"Figure '{safe_name}' not found for '{mip_id}'.")
    media_type = "image/png"
    if safe_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif safe_name.endswith(".json"):
        media_type = "application/json"
    return FileResponse(str(fig_path), media_type=media_type, headers={"Cache-Control": "no-store"})
