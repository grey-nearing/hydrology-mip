# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Automated community model submission processor for CI and maintainer workflows.

Designed to move cleanly into ``hydromip/tools/process_submission.py``:
- Zero ``try``/``except`` blocks and explicit ``allow_pickle=False`` for repository guardrails.
- Zero FastAPI/web framework dependency so CI runners execute cleanly with minimal deps.
- Evaluates submissions exclusively through ``metrics.evaluator.calculate_model_leaderboard_metrics``
  and updates leaderboards through ``leaderboards.update_leaderboard.upsert_leaderboard_entry``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wilcoxon

_HERE = Path(__file__).resolve().parent


from hydromip.evaluation.zarr_io import validate_store_id
from hydromip.io.doi_resolver import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    extract_zenodo_doi,
    resolve_and_download_archive,
    validate_and_fetch_archive_url,
)
from hydromip.io.ingest import (
    EXPECTED_NUM_DAYS,
    ensure_hydromip_on_path,
    extract_file_metadata,
    ingest_hydrograph,
    resolve_deterministic_recipe,
    run_physical_sanity_checks,
    write_canonical_zarr_v2,
)

_REPO_ROOT = ensure_hydromip_on_path()

from leaderboards.generate_figures import (
    classify_model_category,
    extract_publication_year,
)
from leaderboards.update_leaderboard import (
    default_code_provenance,
    format_row_metrics_from_evaluator,
    upsert_leaderboard_entry,
)
from metrics.evaluator import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_BUCKET_URI,
    GAUCH_2023_METRICS,
    calculate_model_leaderboard_metrics,
)

MAX_DOWNLOAD_BYTES: int = DEFAULT_MAX_DOWNLOAD_BYTES
DOWNLOAD_TIMEOUT_SECONDS: int = 120
SAFE_ID_RE = re.compile(r"[^a-z0-9_]+")
ENSEMBLE_SUFFIX_RE = re.compile(r"\s*\(\d+-model ensemble\)\s*$")


def _fetch_https(
    url: str,
    dest_path: Path | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> bytes:
    """Delegate HTTPS fetches through the shared SSRF-hardened downloader."""
    return validate_and_fetch_archive_url(
        url,
        dest_path=dest_path,
        max_bytes=max_bytes,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
        user_agent="HydrologyMIP-CI/1.0",
    )


def resolve_archive_to_local_file(
    manifest: Mapping[str, Any],
    work_dir: Path,
    *,
    explicit_archive_path: Path | None = None,
) -> Path:
    """Resolve submission archive from explicit path, ``file://`` URI, DOI, or HTTPS URL."""
    if explicit_archive_path is not None:
        if not explicit_archive_path.is_file():
            raise FileNotFoundError(f"Archive path does not exist: {explicit_archive_path}")
        return explicit_archive_path

    staged_raw = str(manifest.get("staged_archive_path") or "").strip()
    if staged_raw:
        staged = Path(staged_raw)
        if staged.is_file():
            return staged

    raw_url = str(manifest.get("archive_url") or manifest.get("data_archive_url") or "").strip()
    if not raw_url:
        raise ValueError("Submission manifest must define 'archive_url' (DOI or HTTPS file URL).")

    return resolve_and_download_archive(
        raw_url,
        work_dir,
        max_bytes=MAX_DOWNLOAD_BYTES,
        allow_file_uri=True,
        fetch_fn=_fetch_https,
    )


def _load_canonical_targets(
    mip_dir: Path,
    qobs_npz: Path | None = None,
) -> tuple[list[str], list[str], dict[str, str], np.ndarray]:
    splits_dir = mip_dir / "splits"
    basins_file = splits_dir / "basins_531.txt"
    if basins_file.is_file():
        basins_531 = [
            b.strip().zfill(8)
            for b in basins_file.read_text(encoding="utf-8").splitlines()
            if b.strip()
        ]
    else:
        basins_531 = []

    huc_map: dict[str, str] = {}
    pur_path = splits_dir / "pur_18fold_huc02_splits.json"
    if pur_path.is_file():
        pur_doc = json.loads(pur_path.read_text(encoding="utf-8"))
        folds = pur_doc.get("folds") or {}
        for huc_key, info in folds.items():
            huc_code = str(huc_key).replace("huc_", "").replace("HUC_", "").zfill(2)[:2]
            for b in info.get("test_basins", []):
                huc_map[str(b).zfill(8)] = huc_code

    if qobs_npz is not None and qobs_npz.is_file():
        with np.load(qobs_npz, allow_pickle=False) as d:
            b_arr = [str(b).zfill(8) for b in d["basins"].tolist()]
            d_arr = [str(x) for x in d["dates"].tolist()]
            qobs = np.asarray(d["qobs"], dtype=np.float32)
            if "huc_02" in d.files:
                for b_id, h_code in zip(b_arr, d["huc_02"].tolist()):
                    huc_map.setdefault(b_id, str(h_code).zfill(2))
            if basins_531 and b_arr != basins_531:
                idx_map = {b: i for i, b in enumerate(b_arr)}
                qobs = qobs[[idx_map[b] for b in basins_531], :]
                b_arr = basins_531
            return b_arr, d_arr, huc_map, qobs

    local_targets = mip_dir / "observations" / "newman_eval_targets.zarr"
    if local_targets.exists():
        with xr.open_zarr(
            local_targets, consolidated=(local_targets / ".zmetadata").is_file()
        ) as ds:
            b_arr = [str(b).zfill(8) for b in ds["basin"].values.tolist()]
            d_arr = pd.to_datetime(ds["date"].values).strftime("%Y-%m-%d").tolist()
            if "huc_02" in ds.coords:
                for b_id, h_code in zip(b_arr, ds["huc_02"].values.tolist()):
                    huc_map[b_id] = str(h_code).zfill(2)
            qobs = np.asarray(ds["qobs"].values, dtype=np.float32)
        if basins_531 and b_arr != basins_531:
            idx_map = {b: i for i, b in enumerate(b_arr)}
            qobs = qobs[[idx_map[b] for b in basins_531], :]
            b_arr = basins_531
        return b_arr, d_arr, huc_map, qobs

    raise FileNotFoundError(
        "Canonical observation targets missing: supply --qobs-npz or ensure "
        f"{local_targets} exists."
    )


def _materialize_targets_zarr(
    bucket_dir: Path,
    basins: Sequence[str],
    dates: Sequence[str],
    huc_map: Mapping[str, str],
    qobs: np.ndarray,
) -> Path:
    obs_zarr = bucket_dir / "observations" / "newman_eval_targets.zarr"
    obs_zarr.parent.mkdir(parents=True, exist_ok=True)
    sort_order = np.argsort(np.asarray(basins, dtype="<U8"))
    sorted_basins = np.asarray([basins[i] for i in sort_order], dtype="<U8")
    sorted_huc = np.asarray([huc_map.get(str(b), "01") for b in sorted_basins], dtype="<U2")
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
    return obs_zarr


def _paired_wilcoxon_p(vec_a: Sequence[Any], vec_b: Sequence[Any] | None) -> float | None:
    if not vec_b or len(vec_b) != len(vec_a):
        return None
    a_arr = np.asarray([float(v) if v is not None else np.nan for v in vec_a], dtype=np.float64)
    b_arr = np.asarray([float(v) if v is not None else np.nan for v in vec_b], dtype=np.float64)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if int(np.sum(mask)) < 10:
        return None
    diff = a_arr[mask] - b_arr[mask]
    if np.allclose(diff, 0.0, atol=1e-12):
        return 1.0
    pval = float(wilcoxon(a_arr[mask], b_arr[mask], zero_method="wilcox", alternative="two-sided").pvalue)
    return round(pval, 6) if np.isfinite(pval) else None


def _derive_entry_id(manifest: Mapping[str, Any], manifest_path: Path) -> str:
    explicit = str(manifest.get("entry_id") or "").strip()
    if explicit:
        validate_store_id(explicit, "entry_id")
        return explicit
    model_slug = SAFE_ID_RE.sub(
        "_", str(manifest.get("model_name") or manifest_path.stem).lower()
    ).strip("_")
    cleaned = f"contrib_{model_slug[:32]}".strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"contrib_{cleaned}"
    return cleaned[:48]


def _single_model_display_name(manifest: Mapping[str, Any], model_name: str) -> str:
    explicit_single = str(manifest.get("single_model_name") or "").strip()
    if explicit_single:
        return explicit_single
    base = ENSEMBLE_SUFFIX_RE.sub("", model_name).strip() or model_name
    return f"{base} (single model)"


def _render_pr_summary_markdown(
    *,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    ensemble_single_entry: Mapping[str, Any] | None,
    sanity: Mapping[str, Any],
    lb_rows: Sequence[Mapping[str, Any]],
    dry_run: bool,
    zarr_path: Path,
) -> str:
    lb_id = str(entry["leaderboard_id"])
    med_nse = float(entry["primary_ranking_nse"])
    rank = 1 + sum(
        1
        for r in lb_rows
        if str(r.get("entry_id")) != str(entry["entry_id"])
        and r.get("primary_ranking_nse") is not None
        and float(r["primary_ranking_nse"]) > med_nse
    )
    sota_row = lb_rows[0] if lb_rows else None
    sota_name = str(sota_row.get("model_name", "SOTA")) if sota_row else "N/A"
    sota_gauch = (sota_row.get("gauch_2023_metrics") or {}) if sota_row else {}
    sota_nse = sota_row.get("primary_ranking_nse") if sota_row else None
    delta_nse = f"{med_nse - float(sota_nse):+.4f}" if sota_nse is not None else "N/A"
    p_nse = entry.get("p_value_vs_sota_nse")
    p_str = f"{float(p_nse):.2e}" if p_nse is not None else "N/A"

    is_closed = bool(entry.get("closed_source") or entry.get("is_closed_source") or not str(entry.get("code_url") or "").strip())
    code_badge = "`Closed-Source` (no public code repository linked)" if is_closed else f"[Public Repository]({entry['code_url']})"

    gauch = entry["gauch_2023_metrics"]
    already_listed = any(str(r.get("entry_id")) == str(entry["entry_id"]) for r in lb_rows)
    total_count = len(lb_rows) + (1 if dry_run and not already_listed else 0)
    lines = [
        f"### HydrologyMIP Automated Submission Verification (`{lb_id}`)",
        "",
        f"- **Model:** `{entry['model_name']}` (`entry_id`: `{entry['entry_id']}`)",
        f"- **Mode:** {'PR Dry-Run Validation' if dry_run else 'Leaderboard Update Committed'}",
        f"- **Code Provenance:** {code_badge}",
        f"- **Permanent Archive:** {entry.get('data_archive_url') or manifest.get('archive_url')}",
        f"- **Estimated Rank:** **#{rank}** of {total_count} (`Δ Med NSE vs. {sota_name}`: `{delta_nse}`, Wilcoxon `p = {p_str}`)",
        "",
        "#### Physical Sanity & Protocol Checks",
        f"- **Basin Coverage:** `{sanity['n_basins_matched']}/{sanity['n_basins_expected']}` canonical CAMELS-US basins",
        f"- **Anti-Masking:** `{sanity['nan_count']}` NaNs on valid observation days, `{sanity['inf_count']}` Inf values",
        f"- **Evaluation Window:** `{sanity['date_start']}` to `{sanity['date_end']}` ({EXPECTED_NUM_DAYS} days)",
        f"- **Runoff Scale Ratio:** `{sanity['median_sim_obs_ratio']}x` (`mm/day` check passed: `{sanity['checks_passed']}`)",
        f"- **Canonical Zarr v2 Output:** `{zarr_path}` (`is_ensemble={entry['is_ensemble']}`, `num_members={entry['num_members']}`)",
        "",
    ]

    if ensemble_single_entry is not None:
        lines.append(
            f"- **Single-Seed Skill (`n=1` across {entry['num_members']} members):** "
            f"`{ensemble_single_entry.get('single_med_nse')} ± {ensemble_single_entry.get('single_med_nse_std')}` "
            f"(per-member: `{ensemble_single_entry.get('single_med_nse_per_member')}`)"
        )
        lines.append("")

    lines.extend(
        [
            "#### Canonical 14-Metric Evaluation Summary",
            "| Metric | Median | Mean (NSE/KGE only) | Q25 | Q75 | SOTA Median |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    def _fmt_num(val: Any) -> str:
        return f"{float(val):.4f}" if val is not None and np.isfinite(float(val)) else "—"

    for m_name in GAUCH_2023_METRICS:
        d = gauch.get(m_name) or {}
        med_v = d.get("median")
        mean_v = d.get("mean") if m_name in ("NSE", "KGE") else None
        q25_v = d.get("q25")
        q75_v = d.get("q75")
        s_med = (sota_gauch.get(m_name) or {}).get("median") if isinstance(sota_gauch, Mapping) else None
        lines.append(
            f"| `{m_name}` | {_fmt_num(med_v)} | {_fmt_num(mean_v)} | {_fmt_num(q25_v)} | {_fmt_num(q75_v)} | {_fmt_num(s_med)} |"
        )

    lines.extend(
        [
            "",
            "#### Reviewer Verification Checklist",
            "- [x] Schema, 531-basin alignment, zero NaN masking, and canonical Zarr v2 validated automatically",
            f"- [x] All {len(GAUCH_2023_METRICS)} canonical metrics scored cleanly via `metrics.evaluator`",
            "- [ ] Confirmed blind 1989–1999 temporal holdout (Rule 1)",
            "- [ ] Confirmed canonical spatial holdout fold isolation (Rule 2)",
            "- [ ] Confirmed forcing declarations & immutable DOI match manifest (Rule 3 & 4)",
        ]
    )
    return "\n".join(lines) + "\n"


def process_submission(
    *,
    manifest_path: Path,
    mip_dir: Path,
    out_zarr_dir: Path,
    summary_md_path: Path | None = None,
    qobs_npz_path: Path | None = None,
    explicit_archive_path: Path | None = None,
    dry_run: bool = True,
    update_figures: bool = False,
) -> dict[str, Any]:
    """Validate, ingest, score (14 metrics), and optionally persist a community submission."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leaderboard_id = str(manifest.get("leaderboard_id") or "").strip()
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise ValueError(
            f"Invalid leaderboard_id {leaderboard_id!r}; expected one of {CANONICAL_LEADERBOARD_IDS}."
        )
    model_name = str(manifest.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("Submission manifest must define a non-empty 'model_name'.")
    forcing = str(manifest.get("forcing") or "").strip()
    if not forcing:
        raise ValueError("Submission manifest must define 'forcing'.")

    att = manifest.get("compliance_attestation") or {}
    required_rules = (
        "blind_holdout_window",
        "strict_spatial_holdout_canonical_folds",
        "all_forcing_inputs_declared",
        "deterministic_daily_predictions_mm_per_day",
    )
    if not all(bool(att.get(r)) for r in required_rules):
        raise ValueError("All 4 benchmark compliance_attestation checkboxes must be true.")

    code_url = str(manifest.get("code_url") or manifest.get("github_url") or "").strip()
    closed_source = bool(manifest.get("closed_source") or manifest.get("is_closed_source") or not code_url)
    archive_url = str(manifest.get("archive_url") or manifest.get("data_archive_url") or "").strip()
    if not archive_url:
        raise ValueError("Submission manifest must define permanent 'archive_url' (DOI or HTTPS URL).")

    basins, dates, huc_map, qobs = _load_canonical_targets(mip_dir, qobs_npz=qobs_npz_path)
    entry_id = _derive_entry_id(manifest, manifest_path)

    existing_lb_file = mip_dir / "leaderboards" / f"{leaderboard_id}.json"
    if existing_lb_file.is_file():
        norm_new_name = " ".join(model_name.strip().lower().split())
        existing_rows = json.loads(existing_lb_file.read_text(encoding="utf-8")).get("rows", [])
        for existing_row in existing_rows:
            existing_eid = str(existing_row.get("entry_id") or "").strip()
            existing_name = str(existing_row.get("model_name") or "").strip()
            if existing_eid != entry_id and " ".join(existing_name.lower().split()) == norm_new_name:
                raise ValueError(
                    f"Duplicate model_name {model_name!r} already exists on {leaderboard_id} "
                    f"(entry_id={existing_eid!r}). Please choose a distinct model_name."
                )

    with tempfile.TemporaryDirectory() as tmp_str:
        work_dir = Path(tmp_str)
        archive_file = resolve_archive_to_local_file(
            manifest, work_dir, explicit_archive_path=explicit_archive_path
        )
        meta = extract_file_metadata(archive_file, work_dir=work_dir / "inspect")
        recipe_ov = manifest.get("confirmed_recipe")
        recipe, issues = resolve_deterministic_recipe(meta, recipe_override=recipe_ov)
        if recipe is None:
            raise ValueError(f"Could not deterministically map submission archive schema: {issues}")

        ingested = ingest_hydrograph(
            archive_file,
            recipe,
            basins,
            dates,
            strict=True,
            work_dir=work_dir / "ingest",
        )
        sanity, sample_basins = run_physical_sanity_checks(
            ingested["qsim"],
            qobs,
            basins,
            dates,
            matched_basin_mask=ingested["matched_basin_mask"],
            matched_date_mask=ingested["matched_date_mask"],
            initial_diagnostics=ingested["diagnostics"],
        )
        if not sanity["checks_passed"]:
            raise ValueError(f"Submission failed physical sanity checks: {sanity['diagnostics']}")

        out_zarr_path = out_zarr_dir / "hydrographs" / leaderboard_id / f"{entry_id}.zarr"
        write_canonical_zarr_v2(
            ingested["qsim"],
            basins,
            out_zarr_path,
            metadata={
                "model_id": entry_id,
                "model_name": model_name,
                "leaderboard_id": leaderboard_id,
                "forcing": forcing,
                "archive_url": archive_url,
                "code_url": "" if closed_source else code_url,
                "closed_source": closed_source,
            },
            qsim_members=ingested["qsim_members"],
            huc_02=[huc_map.get(b, "01") for b in basins],
            dates=dates,
        )

        # Build ephemeral target store symlinked alongside Zarr output so evaluator runs cleanly
        eval_bucket = work_dir / "eval_bucket"
        _materialize_targets_zarr(eval_bucket, basins, dates, huc_map, qobs)
        hydro_link = eval_bucket / "hydrographs" / leaderboard_id / f"{entry_id}.zarr"
        hydro_link.parent.mkdir(parents=True, exist_ok=True)
        hydro_link.symlink_to(out_zarr_path.resolve())

        eval_res = calculate_model_leaderboard_metrics(
            model_id=entry_id,
            leaderboard_id=leaderboard_id,
            bucket_uri=str(eval_bucket),
            allow_partial_archive=False,
        )
        ensemble_single_res: dict[str, Any] | None = None
        if ingested["qsim_members"] is not None and int(ingested["qsim_members"].shape[0]) >= 2:
            ensemble_single_res = calculate_model_leaderboard_metrics(
                model_id=entry_id,
                leaderboard_id=leaderboard_id,
                bucket_uri=str(eval_bucket),
                allow_partial_archive=False,
                evaluation_mode="single_member_mean",
            )

    lb_dir = mip_dir / "leaderboards"
    reg_path = lb_dir / "table_2_registry.json"
    existing_fams_by_id: dict[str, dict[str, Any]] = {}
    if reg_path.is_file():
        reg_doc = json.loads(reg_path.read_text(encoding="utf-8"))
        existing_fams_by_id = {
            str(f.get("family_id", "")): f for f in reg_doc.get("families", []) if f.get("family_id")
        }

    lb_file = lb_dir / f"{leaderboard_id}.json"
    lb_rows: list[dict[str, Any]] = []
    if lb_file.is_file():
        lb_rows = list(json.loads(lb_file.read_text(encoding="utf-8")).get("rows", []))
    sota_row = lb_rows[0] if lb_rows else None

    canonical_zarr_uri = f"{DEFAULT_BUCKET_URI.rstrip('/')}/hydrographs/{leaderboard_id}/{entry_id}.zarr"
    zenodo_doi = str(manifest.get("zenodo_doi") or extract_zenodo_doi(archive_url)).strip()
    zenodo_url = (
        str(manifest.get("zenodo_url") or (f"https://doi.org/{zenodo_doi}" if zenodo_doi else "")).strip()
    )
    paper_url = str(manifest.get("paper_url") or "").strip()
    winner_nb_url = str(manifest.get("winner_notebook_url") or "").strip()
    submitted_at = datetime.datetime.now(datetime.UTC).isoformat()

    def _assemble_entry(res_dict: dict[str, Any], target_id: str, display_name: str) -> dict[str, Any]:
        row_metrics = format_row_metrics_from_evaluator(res_dict)
        gauch = row_metrics["gauch_2023_metrics"]
        p_nse = _paired_wilcoxon_p(
            res_dict["per_basin"]["NSE"],
            sota_row.get("per_basin_nse") if sota_row else None,
        )
        p_kge = _paired_wilcoxon_p(
            res_dict["per_basin"]["KGE"],
            sota_row.get("per_basin_kge") if sota_row else None,
        )
        family_id = str(manifest.get("family_id") or f"contrib_{entry_id}").strip()
        existing_fam = existing_fams_by_id.get(family_id) or {}
        family_name = str(
            manifest.get("family_name")
            or manifest.get("new_family_name")
            or existing_fam.get("name")
            or model_name
        ).strip()
        category = str(manifest.get("category") or "Contribution").strip()
        num_mem = int(row_metrics.get("num_members") or 1)
        row_dict: dict[str, Any] = {
            **row_metrics,
            "entry_id": target_id,
            "leaderboard_id": leaderboard_id,
            "model_name": display_name,
            "family_id": family_id,
            "family_name": family_name,
            "paradigm": str(manifest.get("paradigm") or "Machine learning").strip(),
            "category": category,
            "forcing": forcing,
            "code_url": "" if closed_source else code_url,
            "github_url": "" if closed_source else code_url,
            "closed_source": closed_source,
            "is_closed_source": closed_source,
            "archive_url": archive_url,
            "data_archive_url": archive_url,
            "zenodo_doi": zenodo_doi,
            "zenodo_url": zenodo_url,
            "citation": str(manifest.get("citation") or "Community Contribution (2026)").strip(),
            "mean_kge": float(res_dict["mean_kge"]) if "mean_kge" in res_dict else gauch["KGE"]["mean"],
            "gauch_nse": gauch["NSE"]["median"],
            "gauch_kge": gauch["KGE"]["median"],
            "gauch_log_nse": gauch["logNSE"]["median"],
            "gauch_log_kge": gauch["logKGE"]["median"],
            "gauch_pearson_r": gauch["Pearson-r"]["median"],
            "gauch_alpha_nse": gauch["Alpha-NSE"]["median"],
            "gauch_beta_nse": gauch["Beta-NSE"]["median"],
            "gauch_beta_kge": gauch["Beta-KGE"]["median"],
            "gauch_mse": gauch["MSE"]["median"],
            "gauch_rmse": gauch["RMSE"]["median"],
            "gauch_fhv": gauch["FHV"]["median"],
            "gauch_fms": gauch["FMS"]["median"],
            "gauch_flv": gauch["FLV"]["median"],
            "gauch_peak_timing": gauch["Peak-Timing"]["median"],
            "p_value_vs_sota_nse": p_nse,
            "p_value_vs_sota_kge": p_kge,
            "asterisk": False,
            "archive_key": target_id,
            "zarr_path": canonical_zarr_uri,
            "eval_period": "1989-10-01 to 1999-09-30",
            "submitted_at": submitted_at,
        }
        if num_mem > 1 or bool(row_metrics.get("is_ensemble")):
            row_dict["seeds"] = str(num_mem)
        if paper_url:
            row_dict["paper_url"] = paper_url
        if winner_nb_url:
            row_dict["winner_notebook_url"] = winner_nb_url
        return row_dict

    primary_entry = _assemble_entry(eval_res, entry_id, model_name)
    single_entry = (
        _assemble_entry(
            ensemble_single_res,
            f"{entry_id}_single",
            _single_model_display_name(manifest, model_name),
        )
        if ensemble_single_res is not None
        else None
    )

    if not dry_run:
        family_record: dict[str, Any] | None = None
        if primary_entry["family_id"] not in existing_fams_by_id:
            row_for_classify = {
                **primary_entry,
                "code_provenance": default_code_provenance(primary_entry),
            }
            model_cat = classify_model_category(row_for_classify)
            pub_year = extract_publication_year(row_for_classify)
            family_record = {
                "family_id": primary_entry["family_id"],
                "name": primary_entry["family_name"],
                "category": model_cat,
                "model_category": model_cat,
                "provenance_category": primary_entry["category"],
                "paradigm": primary_entry["paradigm"],
                "citation": primary_entry["citation"],
                "paper_url": paper_url,
                "github_url": primary_entry["code_url"],
                "code_dir": "",
                "zenodo_doi": zenodo_doi,
                "zenodo_url": zenodo_url or primary_entry["data_archive_url"],
                "hydroshare_url": archive_url if "hydroshare.org" in archive_url.lower() or "10.4211/hs." in archive_url.lower() else "",
                "description": str(manifest.get("notes") or f"Community contribution: {model_name}.").strip(),
                "asterisk": False,
                "asterisk_reason": None,
                "n_entries": 0,
                "leaderboards_covered": [],
                "best_nse": primary_entry["primary_ranking_nse"],
                "year": pub_year if pub_year is not None else 2026,
            }

        spatial_reg, _, forcing_reg = leaderboard_id.partition("__")
        table_idx = CANONICAL_LEADERBOARD_IDS.index(leaderboard_id) + 3
        prim_gauch = primary_entry["gauch_2023_metrics"]
        contrib_record = {
            "entry_id": entry_id,
            "leaderboard_id": leaderboard_id,
            "model_name": primary_entry["model_name"],
            "family_id": primary_entry["family_id"],
            "family_name": primary_entry["family_name"],
            "forcing": forcing,
            "citation": primary_entry["citation"],
            "gcs_zarr_uri": canonical_zarr_uri,
            "archive_url": archive_url,
            "data_archive_url": archive_url,
            "zenodo_doi": zenodo_doi,
            "code_url": primary_entry["code_url"],
            "closed_source": closed_source,
            "manifest_path": str(manifest_path),
            "primary_ranking_nse": primary_entry["primary_ranking_nse"],
            "submitted_at": submitted_at,
            "category": primary_entry["category"],
            "code_provenance": default_code_provenance(primary_entry),
            "spatial_regime": spatial_reg,
            "forcing_regime": forcing_reg,
            "forcings_used": forcing,
            "is_ensemble": primary_entry["is_ensemble"],
            "num_members": primary_entry["num_members"],
            "has_qsim_members": primary_entry["has_qsim_members"],
            "hydrograph_path": canonical_zarr_uri,
            "target_leaderboard_tables": [f"table_{table_idx}", leaderboard_id],
            "metrics": {
                "median_nse": round(float(eval_res["median_nse"]), 4),
                "mean_nse": round(float(eval_res["mean_nse"]), 4),
                "median_kge": round(float(eval_res["median_kge"]), 4),
                "median_abs_fhv": round(float(eval_res["median_abs_fhv"]), 2),
                "nse_le_zero_count": int(eval_res["nse_le_zero_count"]),
                "single_seed_median_nse_mean": eval_res.get("single_seed_median_nse_mean"),
                "single_seed_median_nse_std": eval_res.get("single_seed_median_nse_std"),
                "single_seed_median_nse_per_member": eval_res.get("single_seed_median_nse_per_member"),
            },
            "gauch_2023_metrics": prim_gauch,
        }
        primary_entry = upsert_leaderboard_entry(
            leaderboard_id,
            primary_entry,
            leaderboards_dir=lb_dir,
            family_record=family_record,
            contribution_record=contrib_record,
            update_figures=(update_figures and single_entry is None),
        )
        if single_entry is not None:
            single_entry = upsert_leaderboard_entry(
                leaderboard_id,
                single_entry,
                leaderboards_dir=lb_dir,
                family_record=None,
                contribution_record=None,
                update_figures=update_figures,
            )

    summary_md = _render_pr_summary_markdown(
        manifest=manifest,
        entry=primary_entry,
        ensemble_single_entry=single_entry,
        sanity=sanity,
        lb_rows=lb_rows,
        dry_run=dry_run,
        zarr_path=out_zarr_path,
    )
    if summary_md_path is not None:
        summary_md_path.parent.mkdir(parents=True, exist_ok=True)
        summary_md_path.write_text(summary_md, encoding="utf-8")

    return {
        "status": "processed",
        "dry_run": dry_run,
        "entry": primary_entry,
        "ensemble_single_entry": single_entry,
        "sanity": sanity,
        "sample_basins": sample_basins,
        "zarr_path": str(out_zarr_path),
        "summary_markdown": summary_md,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, ingest, score, and register a HydrologyMIP community submission."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to submission manifest JSON.")
    parser.add_argument("--mip-dir", required=True, type=Path, help="Path to mips/camels_us_daily directory.")
    parser.add_argument("--qobs-npz", default=None, type=Path, help="Optional compact qobs_531.npz path.")
    parser.add_argument("--archive-path", default=None, type=Path, help="Optional explicit local archive path.")
    parser.add_argument("--out-zarr-dir", required=True, type=Path, help="Directory root to write canonical Zarr v2.")
    parser.add_argument("--summary-md", default=None, type=Path, help="Path to write Markdown PR summary.")
    parser.add_argument("--dry-run", action="store_true", help="Score and stage Zarr without mutating leaderboards.")
    parser.add_argument("--update-figures", action="store_true", help="Regenerate figures during non-dry-run update.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = process_submission(
        manifest_path=args.manifest,
        mip_dir=args.mip_dir,
        out_zarr_dir=args.out_zarr_dir,
        summary_md_path=args.summary_md,
        qobs_npz_path=args.qobs_npz,
        explicit_archive_path=args.archive_path,
        dry_run=bool(args.dry_run),
        update_figures=bool(args.update_figures),
    )
    print(json.dumps({"status": result["status"], "entry_id": result["entry"]["entry_id"], "rank": result["entry"].get("rank"), "zarr_path": result["zarr_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
