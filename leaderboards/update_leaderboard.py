"""Automated leaderboard JSON management and submission-triggered update pipeline.

Maintains the 12 individual leaderboard JSON files (`leaderboards/<leaderboard_id>.json`)
and keeps the combined `leaderboards/tables_3_to_14.json` synchronized whenever a
community submission or new model run is ingested.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import re
from pathlib import Path
from typing import Any

from hydromip.paths import DEFAULT_MIP_ID, REPO_ROOT, resolve_mip_paths
from metrics.evaluator import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_BUCKET_URI,
    calculate_model_leaderboard_metrics,
)

DEFAULT_LEADERBOARDS_DIR: Path = resolve_mip_paths(DEFAULT_MIP_ID).leaderboards_dir


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object from `path` with zero try/except blocks."""
    if not path.is_file():
        raise FileNotFoundError(f"Required leaderboard JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write `payload` to `path` as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def ensure_per_leaderboard_files(
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
) -> None:
    """Ensure all 12 `<leaderboard_id>.json` files exist in `leaderboards_dir`.

    If a directory only contains `tables_3_to_14.json` (for example in an isolated test
    fixture), splits `tables_3_to_14.json` into the 12 `<leaderboard_id>.json` files.
    """
    lb_dir = Path(leaderboards_dir)
    missing = [
        lbid
        for lbid in CANONICAL_LEADERBOARD_IDS
        if not (lb_dir / f"{lbid}.json").is_file()
    ]
    if not missing:
        return
    master_path = lb_dir / "tables_3_to_14.json"
    if not master_path.is_file():
        raise FileNotFoundError(
            f"Missing per-leaderboard JSON files {missing} and {master_path} in {lb_dir}"
        )
    master = _read_json_dict(master_path)
    protocol = master.get("protocol", {})
    for lb in master.get("leaderboards", []):
        lbid = str(lb["id"])
        per_lb_path = lb_dir / f"{lbid}.json"
        if not per_lb_path.is_file():
            payload = copy.deepcopy(lb)
            payload["protocol"] = copy.deepcopy(protocol)
            _write_json_dict(per_lb_path, payload)


def _is_finite_number(value: Any) -> bool:
    """Return True for a real, finite numeric value (bool excluded)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


# Legacy exports encoded "no value" as -999.0. A 531-basin median NSE below this bound is
# not a physically meaningful leaderboard score, so anything at or below it is a sentinel.
_SENTINEL_NSE_BOUND: float = -900.0


def _is_valid_nse(value: Any) -> bool:
    """Return True for a finite NSE-like value that is not a legacy `-999` sentinel."""
    return _is_finite_number(value) and float(value) > _SENTINEL_NSE_BOUND


def _primary_nse(row: dict[str, Any]) -> float | None:
    """Resolve the ranking NSE of a row: `primary_ranking_nse`, else ensemble, else single.

    Non-finite or non-numeric values and legacy `-999.0` sentinels resolve to `None`.
    """
    for key in ("primary_ranking_nse", "ens_med_nse", "single_med_nse"):
        value = row.get(key)
        if _is_valid_nse(value):
            return float(value)
    return None


def rank_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically sort and assign 1-based `rank` to leaderboard `rows`.

    Rows with a finite `primary_ranking_nse` (or `ens_med_nse` / `single_med_nse`)
    are sorted descending; ties break by `mean_nse` descending, then `med_kge`
    descending, then `model_name` ascending. Rows without any finite NSE sort last
    (in `model_name` order) and keep `primary_ranking_nse = None`.
    """
    ranked_copy = [copy.deepcopy(r) for r in rows]

    def _sort_key(r: dict[str, Any]) -> tuple[int, float, float, float, str]:
        p_nse = _primary_nse(r)
        if p_nse is None:
            return (1, 0.0, 0.0, 0.0, str(r.get("model_name", "")))
        mean_v = float(r["mean_nse"]) if _is_finite_number(r.get("mean_nse")) else -1e9
        kge_v = float(r["med_kge"]) if _is_finite_number(r.get("med_kge")) else -1e9
        return (0, -p_nse, -mean_v, -kge_v, str(r.get("model_name", "")))

    ranked_copy.sort(key=_sort_key)
    for idx, row in enumerate(ranked_copy, start=1):
        row["primary_ranking_nse"] = _primary_nse(row)
        row["rank"] = idx
    return ranked_copy


def load_leaderboard(
    leaderboard_id: str,
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
) -> dict[str, Any]:
    """Load a single leaderboard JSON file (`leaderboards/<leaderboard_id>.json`)."""
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise ValueError(
            f"Unknown leaderboard_id {leaderboard_id!r}; expected one of {CANONICAL_LEADERBOARD_IDS}."
        )
    ensure_per_leaderboard_files(leaderboards_dir)
    lb_path = Path(leaderboards_dir) / f"{leaderboard_id}.json"
    return _read_json_dict(lb_path)


def load_all_leaderboards(
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
) -> dict[str, Any]:
    """Load all 12 per-leaderboard JSON files directly from `leaderboards_dir`."""
    ensure_per_leaderboard_files(leaderboards_dir)
    lb_dir = Path(leaderboards_dir)
    leaderboards: list[dict[str, Any]] = []
    protocol: dict[str, Any] = {}
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_data = _read_json_dict(lb_dir / f"{lbid}.json")
        if not protocol and isinstance(lb_data.get("protocol"), dict):
            protocol = copy.deepcopy(lb_data["protocol"])
        lb_clean = copy.deepcopy(lb_data)
        lb_clean.pop("protocol", None)
        leaderboards.append(lb_clean)

    if not protocol and (lb_dir / "tables_3_to_14.json").is_file():
        master = _read_json_dict(lb_dir / "tables_3_to_14.json")
        protocol = copy.deepcopy(master.get("protocol", {}))

    total_entries = sum(len(lb.get("rows", [])) for lb in leaderboards)
    return {
        "protocol": protocol,
        "total_entries": total_entries,
        "leaderboards": leaderboards,
    }


def sync_master_tables_from_per_leaderboard_files(
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
) -> dict[str, Any]:
    """Synchronize `tables_3_to_14.json` from the 12 individual `<leaderboard_id>.json` files."""
    lb_dir = Path(leaderboards_dir)
    combined = load_all_leaderboards(lb_dir)
    _write_json_dict(lb_dir / "tables_3_to_14.json", combined)
    return combined


_AUTHOR_PREFIX_WITH_YEAR_RE = re.compile(
    r"^(?P<cit>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\s,&.\-]+?\(\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*\))\s+(?=\S)"
)
_AUTHOR_PREFIX_NO_YEAR_RE = re.compile(
    r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-]+(?:,\s*[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-]+)?"
    r"(?:\s+&\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-]+|\s+et\s+al\.)\s+(?=\S)"
)
_TRAILING_AUTHOR_PAREN_RE = re.compile(
    r"\s*\((?:Kratzert|Poudel|Wang|Gauch|Hoedt|Feng|Song)\)\s*$"
)
_LEGACY_ERA_NAME_MAP: dict[str, str] = {
    "ERA Gauged Minimal Guidance (Node 73 EntityAwareLSTM)": "ERA-Unguided-Gauged",
    "ERA PUB Minimal Guidance (Node 82 Enhanced EALSTM)": "ERA-Unguided-PUB",
    "ERA PUR Minimal Guidance (Node 135 DualFiLMLSTM)": "ERA-Unguided-PUR",
}


def clean_model_name(
    name: str,
    *,
    is_ensemble: bool | None = None,
    num_members: int | None = None,
    include_ensemble_suffix: bool = True,
) -> str:
    """Strip any author/citation prefix or suffix from `name`.

    Per the CAMELS-US MIP leaderboard naming convention, the author and publication year
    belong exclusively in `citation` (the row subtitle), never in `model_name`.
    """
    s = str(name or "").strip()
    if not s:
        return s
    if s == "Feng-dHBV":
        if include_ensemble_suffix and is_ensemble:
            n = int(num_members) if (num_members is not None and int(num_members) > 1) else 4
            return f"δHBV 1.0 ({n}-model ensemble)"
        return "δHBV 1.0"
    if s in _LEGACY_ERA_NAME_MAP:
        s = _LEGACY_ERA_NAME_MAP[s]

    s = _AUTHOR_PREFIX_WITH_YEAR_RE.sub("", s).strip()
    s = _AUTHOR_PREFIX_NO_YEAR_RE.sub("", s).strip()
    s = re.sub(r"\bHydroDiffusion\s*\(\d{4}\)\s*", "HydroDiffusion ", s).strip()
    s = _TRAILING_AUTHOR_PAREN_RE.sub("", s).strip()

    if s.startswith("ERA-"):
        base_era = re.sub(r"\s*\(\d+-model ensemble\)\s*$", "", s).strip()
        if not include_ensemble_suffix:
            return base_era
        if is_ensemble and num_members is not None and int(num_members) > 1:
            return f"{base_era} ({int(num_members)}-model ensemble)"
    return s


def _extract_leading_citation_if_missing(model_name: str, citation: str) -> str:
    """If `citation` is empty and `model_name` starts with `Author (YYYY)`, return that prefix."""
    cit = str(citation or "").strip()
    if cit:
        return cit
    m = _AUTHOR_PREFIX_WITH_YEAR_RE.match(str(model_name or "").strip())
    return m.group("cit").strip() if m else ""


def upsert_leaderboard_entry(
    leaderboard_id: str,
    entry: dict[str, Any],
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    family_record: dict[str, Any] | None = None,
    contribution_record: dict[str, Any] | None = None,
    update_figures: bool = True,
) -> dict[str, Any]:
    """Insert or update `entry` in `leaderboards/<leaderboard_id>.json` and re-rank.

    Also synchronizes `leaderboards/tables_3_to_14.json` (and optionally updates
    `table_2_registry.json`, `contributions_index.json`, and all paper figures in
    `figures/`) so all leaderboard JSON files and figures remain 100% consistent.
    """
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise ValueError(
            f"Unknown leaderboard_id {leaderboard_id!r}; expected one of {CANONICAL_LEADERBOARD_IDS}."
        )
    if "entry_id" not in entry or not str(entry["entry_id"]).strip():
        raise ValueError("Leaderboard entry must include a non-empty 'entry_id'.")
    if "model_name" not in entry or not str(entry["model_name"]).strip():
        raise ValueError("Leaderboard entry must include a non-empty 'model_name'.")

    from leaderboards.generate_figures import (
        classify_model_category,
        extract_publication_year,
        generate_all_paper_figures,
    )

    lb_dir = Path(leaderboards_dir)
    lb_data = load_leaderboard(leaderboard_id, lb_dir)
    rows: list[dict[str, Any]] = list(lb_data.get("rows", []))

    entry_copy = copy.deepcopy(entry)
    entry_copy["leaderboard_id"] = leaderboard_id
    inferred_cit = _extract_leading_citation_if_missing(
        str(entry_copy.get("model_name", "")),
        str(entry_copy.get("citation", "")),
    )
    if inferred_cit and not str(entry_copy.get("citation", "")).strip():
        entry_copy["citation"] = inferred_cit
    entry_copy["model_category"] = classify_model_category(entry_copy)
    entry_copy["publication_year"] = extract_publication_year(entry_copy)
    entry_copy["model_name"] = clean_model_name(
        str(entry_copy["model_name"]),
        is_ensemble=entry_copy.get("is_ensemble"),
        num_members=entry_copy.get("num_members"),
    )
    if entry_copy.get("family_name"):
        entry_copy["family_name"] = clean_model_name(
            str(entry_copy["family_name"]),
            include_ensemble_suffix=False,
        )
    target_eid = str(entry_copy["entry_id"])

    replaced = False
    for idx, existing in enumerate(rows):
        if str(existing.get("entry_id")) == target_eid:
            rows[idx] = entry_copy
            replaced = True
            break
    if not replaced:
        rows.append(entry_copy)

    ranked_rows = rank_leaderboard_rows(rows)
    lb_data["rows"] = ranked_rows
    lb_data["n_entries"] = len(ranked_rows)
    if ranked_rows and ranked_rows[0].get("primary_ranking_nse") is not None:
        lb_data["top_model"] = ranked_rows[0]["model_name"]
        lb_data["top_nse"] = ranked_rows[0]["primary_ranking_nse"]
    else:
        lb_data["top_model"] = None
        lb_data["top_nse"] = None

    _write_json_dict(lb_dir / f"{leaderboard_id}.json", lb_data)
    sync_master_tables_from_per_leaderboard_files(lb_dir)

    if family_record is not None and (lb_dir / "table_2_registry.json").is_file():
        reg_path = lb_dir / "table_2_registry.json"
        reg_data = _read_json_dict(reg_path)
        fid = str(family_record.get("family_id", ""))
        existing_idx = next(
            (i for i, f in enumerate(reg_data.get("families", [])) if f.get("family_id") == fid),
            None,
        )
        if existing_idx is not None:
            reg_data["families"][existing_idx] = {
                **reg_data["families"][existing_idx],
                **family_record,
            }
        else:
            reg_data.setdefault("families", []).insert(0, copy.deepcopy(family_record))
        reg_data["total_families"] = len(reg_data["families"])
        _write_json_dict(reg_path, reg_data)

    if contribution_record is not None and (lb_dir / "contributions_index.json").is_file():
        contrib_path = lb_dir / "contributions_index.json"
        contrib_data = _read_json_dict(contrib_path)
        contrib_data.setdefault("entries", {})[target_eid] = copy.deepcopy(contribution_record)
        contrib_data["total_contributions"] = len(contrib_data["entries"])
        _write_json_dict(contrib_path, contrib_data)

    if update_figures:
        generate_all_paper_figures(leaderboards_dir=lb_dir)

    matched = [r for r in ranked_rows if str(r.get("entry_id")) == target_eid]
    return matched[0]


def format_row_metrics_from_evaluator(metrics_summary: dict[str, Any]) -> dict[str, Any]:
    """Translate a `calculate_model_leaderboard_metrics` result into leaderboard row columns.

    The ensemble column (`ens_med_nse`) and the single-seed column (`single_med_nse`) are
    populated from different quantities and are never copies of each other:

    - Ensemble store (`is_ensemble=True`): `ens_med_nse` is the 531-basin median NSE of the
      stored ensemble-mean hydrograph; `single_med_nse` is the mean over members of the
      per-member median NSE (`single_med_nse_std` its population standard deviation),
      which exists only when the store ships `qsim_members`. Without members the
      single-seed column is `None` / `"---"`.
    - Single-model store: `single_med_nse` is the median NSE of the stored hydrograph and
      `ens_med_nse` is `None` / `"---"`.

    `primary_ranking_nse` is always the median NSE of the stored hydrograph.
    """
    med_nse = round(float(metrics_summary["median_nse"]), 3)
    mean_nse = round(float(metrics_summary["mean_nse"]), 3)
    med_kge = round(float(metrics_summary["median_kge"]), 3)
    med_fhv = round(float(metrics_summary["median_abs_fhv"]), 2)
    nse_le_0 = int(metrics_summary["nse_le_zero_count"])
    is_ensemble = bool(metrics_summary["is_ensemble"])
    num_members = int(metrics_summary["num_members"])
    ss_mean = metrics_summary.get("single_seed_median_nse_mean")
    ss_std = metrics_summary.get("single_seed_median_nse_std")

    if is_ensemble:
        ens_med: float | None = med_nse
        ens_disp = f"{med_nse:.3f}"
        if ss_mean is not None and ss_std is not None:
            single_med: float | None = round(float(ss_mean), 3)
            single_std: float | None = round(float(ss_std), 3)
            single_disp = f"{single_med:.3f} ± {single_std:.3f}"
        else:
            single_med = None
            single_std = None
            single_disp = "---"
    else:
        ens_med = None
        ens_disp = "---"
        if (
            metrics_summary.get("ensemble_source") == "mean_across_individual_members"
            and ss_mean is not None
            and ss_std is not None
            and num_members > 1
        ):
            single_med = round(float(ss_mean), 3)
            single_std = round(float(ss_std), 3)
            single_disp = f"{single_med:.3f} ± {single_std:.3f}"
            med_nse = single_med
        else:
            single_med = med_nse
            single_std = None
            single_disp = f"{med_nse:.3f}"

    per_basin_nse_raw = metrics_summary["per_basin"]["nse"]
    per_basin_nse = [
        round(float(v), 4) if _is_finite_number(v) else None for v in per_basin_nse_raw
    ]

    return {
        "is_ensemble": is_ensemble,
        "num_members": num_members,
        "has_qsim_members": bool(metrics_summary.get("has_qsim_members", False)),
        "ensemble_source": metrics_summary.get("ensemble_source"),
        "ens_med_nse": ens_med,
        "ens_med_nse_display": ens_disp,
        "single_med_nse": single_med,
        "single_med_nse_std": single_std,
        "single_med_nse_display": single_disp,
        "single_med_nse_per_member": metrics_summary.get("single_seed_median_nse_per_member"),
        "primary_ranking_nse": med_nse,
        "mean_nse": mean_nse,
        "med_kge": med_kge,
        "med_fhv_abs": med_fhv,
        "nse_le_0": nse_le_0,
        "num_evaluated_basins": int(metrics_summary["num_evaluated_basins"]),
        "pred_nan_when_obs_valid": int(metrics_summary["pred_nan_when_obs_valid"]),
        "per_basin_nse": per_basin_nse,
        "gauch_2023_metrics": metrics_summary["gauch_2023_metrics"],
        "regional_huc02": metrics_summary["regional_huc02"],
    }


def trigger_submission_from_gcs(
    model_id: str,
    leaderboard_id: str,
    model_name: str,
    family_id: str,
    forcing: str,
    is_ensemble: bool | None = None,
    category: str = "Contribution",
    asterisk: bool = False,
    family_name: str = "",
    paradigm: str = "Community Contribution",
    citation: str = "Community submission",
    paper_url: str = "",
    code_url: str = "",
    data_archive_url: str = "",
    bucket_uri: str = DEFAULT_BUCKET_URI,
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    update_figures: bool = True,
) -> dict[str, Any]:
    """Calculate metrics from `gs://camels-us-mip/` and update the leaderboard JSON files.

    Triggered upon each community submission (or new MIP model run):
    1. Scores `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr` against
       `<bucket_uri>/observations/newman_eval_targets.zarr` via
       `metrics.calculate_model_leaderboard_metrics` (using the Gauch et al. 2023
       `metrics/metrics.py` suite and enforcing zero missing basins).
    2. Formats the leaderboard entry and updates `leaderboards/<leaderboard_id>.json`,
       `leaderboards/tables_3_to_14.json`, and all paper figures in `figures/`.

    The ensemble / single-seed status is read from the Zarr store itself (`is_ensemble`
    attribute and `qsim_members`). The optional `is_ensemble` argument is only a
    cross-check: a claim that contradicts the store raises `ValueError`.
    """
    metrics_summary = calculate_model_leaderboard_metrics(
        model_id=model_id,
        leaderboard_id=leaderboard_id,
        bucket_uri=bucket_uri,
    )
    store_is_ensemble = bool(metrics_summary["is_ensemble"])
    if is_ensemble is not None and bool(is_ensemble) != store_is_ensemble:
        raise ValueError(
            f"Ensemble claim mismatch for '{model_id}' on '{leaderboard_id}': the submission "
            f"declares is_ensemble={bool(is_ensemble)} but the Zarr store resolves to "
            f"is_ensemble={store_is_ensemble} (attribute 'is_ensemble' / 'qsim_members'). "
            f"Fix the store attributes; claims are never taken from the command line."
        )

    zarr_uri = f"{bucket_uri.rstrip('/')}/hydrographs/{leaderboard_id}/{model_id}.zarr"
    resolved_data_url = data_archive_url.strip() if data_archive_url.strip() else zarr_uri

    entry: dict[str, Any] = {
        "entry_id": model_id,
        "leaderboard_id": leaderboard_id,
        "model_name": model_name,
        "family_id": family_id,
        "family_name": family_name.strip() if family_name.strip() else model_name,
        "paradigm": paradigm,
        "category": category,
        "forcing": forcing,
        **format_row_metrics_from_evaluator(metrics_summary),
        "asterisk": bool(asterisk),
        "archive_key": model_id,
        "citation": citation,
        "paper_url": paper_url,
        "github_url": code_url,
        "code_url": code_url,
        "zenodo_doi": "",
        "zenodo_url": resolved_data_url,
        "hydroshare_url": "",
        "data_archive_url": resolved_data_url,
        "family_description": f"Evaluated from {zarr_uri}",
    }
    updated_row = upsert_leaderboard_entry(
        leaderboard_id=leaderboard_id,
        entry=entry,
        leaderboards_dir=leaderboards_dir,
        update_figures=update_figures,
    )
    return {
        "updated_entry": updated_row,
        "metrics_summary": metrics_summary,
    }



def recalculate_all_self_run_models(
    bucket_uri: str = DEFAULT_BUCKET_URI,
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    update_figures: bool = True,
) -> dict[str, Any]:
    """Recalculate all statistics for all self-run models (MIP Re-run and Contribution) from GCS Zarr stores."""
    from leaderboards.generate_figures import generate_all_paper_figures

    lb_dir = Path(leaderboards_dir)
    ensure_per_leaderboard_files(lb_dir)

    self_run_categories = {
        "mip re-run",
        "mip-rerun",
        "mip rerun",
        "mip re-run (our code)",
        "mip re-run (author code)",
        "adapt (our code)",
        "contribution",
        "contributed",
    }
    archive_categories = {
        "archive",
        "archive*",
    }

    idx_path = lb_dir / "contributions_index.json"
    idx_data: dict[str, Any] = _read_json_dict(idx_path) if idx_path.is_file() else {"entries": {}}
    ci_entries: dict[str, dict[str, Any]] = idx_data.get("entries", {})

    eval_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    archive_pairs: set[tuple[str, str]] = set()
    pair_store_and_mode: dict[tuple[str, str], tuple[str, str | None]] = {}

    from metrics.evaluator import _get_gcs_storage_options, _zarr_store_exists

    root_uri = bucket_uri.rstrip("/")
    storage_opts = _get_gcs_storage_options() if root_uri.startswith("gs://") else None
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_path = lb_dir / f"{lbid}.json"
        if lb_path.is_file():
            data = _read_json_dict(lb_path)
            for row in data.get("rows", []):
                cat = str(row.get("category", "")).lower().strip()
                eid = str(row.get("entry_id", "")).strip()
                if not eid:
                    continue
                should_recalc = False
                if cat in self_run_categories:
                    should_recalc = True
                elif cat in archive_categories:
                    url = str(row.get("data_archive_url", "")).strip() or str(row.get("zenodo_url", "")).strip()
                    if url.startswith(f"{root_uri}/hydrographs/"):
                        should_recalc = True
                        archive_pairs.add((lbid, eid))
                if should_recalc:
                    store_id = eid
                    direct_uri = f"{root_uri}/hydrographs/{lbid}/{eid}.zarr"
                    if not _zarr_store_exists(direct_uri, storage_options=storage_opts):
                        akey = str(row.get("archive_key", "")).strip()
                        if akey and _zarr_store_exists(
                            f"{root_uri}/hydrographs/{lbid}/{akey}.zarr",
                            storage_options=storage_opts,
                        ):
                            store_id = akey
                        elif eid.endswith("_single") and _zarr_store_exists(
                            f"{root_uri}/hydrographs/{lbid}/{eid.removesuffix('_single')}.zarr",
                            storage_options=storage_opts,
                        ):
                            store_id = eid.removesuffix("_single")
                    eval_mode: str | None = None
                    if row.get("is_ensemble") is False and (
                        int(row.get("num_members") or 1) > 1 or eid.endswith("_single")
                    ):
                        eval_mode = "single_member_mean"
                    pair_store_and_mode[(lbid, eid)] = (store_id, eval_mode)
                    if (lbid, eid) not in seen_pairs:
                        seen_pairs.add((lbid, eid))
                        eval_pairs.append((lbid, eid))

    for cid, centry in ci_entries.items():
        if str(centry.get("category", "")).lower().strip() in self_run_categories:
            eid = str(centry.get("entry_id", cid)).strip()
            target_tables = centry.get("target_leaderboard_tables", [])
            lbid = (
                str(target_tables[1])
                if len(target_tables) >= 2
                else str(centry.get("leaderboard_id", ""))
            )
            if eid and lbid in CANONICAL_LEADERBOARD_IDS and (lbid, eid) not in seen_pairs:
                seen_pairs.add((lbid, eid))
                eval_pairs.append((lbid, eid))
                pair_store_and_mode.setdefault((lbid, eid), (eid, None))

    if not eval_pairs:
        return {"total_recalculated": 0, "reports": []}

    results_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for idx_pair, (lbid_i, eid_i) in enumerate(eval_pairs, start=1):
        store_id_i, eval_mode_i = pair_store_and_mode.get((lbid_i, eid_i), (eid_i, None))
        print(
            f"[{idx_pair}/{len(eval_pairs)}] Recalculating {eid_i} (store={store_id_i}, mode={eval_mode_i}) on {lbid_i}...",
            flush=True,
        )
        results_by_pair[(lbid_i, eid_i)] = calculate_model_leaderboard_metrics(
            model_id=store_id_i,
            leaderboard_id=lbid_i,
            bucket_uri=bucket_uri,
            allow_partial_archive=(lbid_i, eid_i) in archive_pairs,
            evaluation_mode=eval_mode_i,
        )

    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_path = lb_dir / f"{lbid}.json"
        if not lb_path.is_file():
            continue
        lb_data = _read_json_dict(lb_path)
        rows = list(lb_data.get("rows", []))
        modified = False
        for idx, row in enumerate(rows):
            eid = str(row.get("entry_id", "")).strip()
            if (lbid, eid) not in results_by_pair:
                continue
            metrics_summary = results_by_pair[(lbid, eid)]
            updated_row = copy.deepcopy(row)
            ci_meta = ci_entries.get(eid, {})
            declared = ci_meta.get("is_ensemble", row.get("is_ensemble"))
            if declared is not None and bool(declared) != bool(metrics_summary["is_ensemble"]):
                raise ValueError(
                    f"Ensemble claim mismatch for '{eid}' on '{lbid}': leaderboard / "
                    f"contributions index declare is_ensemble={bool(declared)} but the Zarr "
                    f"store resolves to is_ensemble={bool(metrics_summary['is_ensemble'])}."
                )
            # Every metric column is rewritten from the evaluator; the single-seed and
            # ensemble columns come from different quantities (see
            # `format_row_metrics_from_evaluator`) and are never copied into each other.
            updated_row.update(format_row_metrics_from_evaluator(metrics_summary))
            updated_row["model_name"] = clean_model_name(
                str(updated_row.get("model_name", "")),
                is_ensemble=updated_row.get("is_ensemble"),
                num_members=updated_row.get("num_members"),
            )
            if updated_row.get("family_name"):
                updated_row["family_name"] = clean_model_name(
                    str(updated_row["family_name"]),
                    include_ensemble_suffix=False,
                )
            rows[idx] = updated_row
            modified = True

        if modified:
            ranked_rows = rank_leaderboard_rows(rows)
            lb_data["rows"] = ranked_rows
            lb_data["n_entries"] = len(ranked_rows)
            if ranked_rows and ranked_rows[0].get("primary_ranking_nse") is not None:
                lb_data["top_model"] = ranked_rows[0]["model_name"]
                lb_data["top_nse"] = ranked_rows[0]["primary_ranking_nse"]
            _write_json_dict(lb_path, lb_data)

    sync_master_tables_from_per_leaderboard_files(lb_dir)
    if update_figures:
        generate_all_paper_figures(leaderboards_dir=lb_dir)

    if idx_path.is_file():
        for cid, centry in ci_entries.items():
            eid = str(centry.get("entry_id", cid)).strip()
            target_tables = centry.get("target_leaderboard_tables", [])
            lbid = (
                str(target_tables[1])
                if len(target_tables) >= 2
                else str(centry.get("leaderboard_id", ""))
            )
            if (lbid, eid) in results_by_pair:
                rep = results_by_pair[(lbid, eid)]
                centry["metrics"] = {
                    "median_nse": round(float(rep["median_nse"]), 4),
                    "mean_nse": round(float(rep["mean_nse"]), 4),
                    "median_kge": round(float(rep["median_kge"]), 4),
                    "median_abs_fhv": round(float(rep["median_abs_fhv"]), 2),
                    "nse_le_zero_count": int(rep["nse_le_zero_count"]),
                    "single_seed_median_nse_mean": rep["single_seed_median_nse_mean"],
                    "single_seed_median_nse_std": rep["single_seed_median_nse_std"],
                    "single_seed_median_nse_per_member": rep[
                        "single_seed_median_nse_per_member"
                    ],
                }
                centry["is_ensemble"] = bool(rep["is_ensemble"])
                centry["num_members"] = int(rep["num_members"])
                centry["has_qsim_members"] = bool(rep["has_qsim_members"])
                centry["gauch_2023_metrics"] = rep["gauch_2023_metrics"]
        _write_json_dict(idx_path, idx_data)

    reports: list[dict[str, Any]] = []
    for lbid, eid in eval_pairs:
        rep = results_by_pair[(lbid, eid)]
        reports.append(
            {
                "leaderboard_id": lbid,
                "model_id": eid,
                "hydrograph_uri": rep["hydrograph_uri"],
                "num_evaluated_basins": rep["num_evaluated_basins"],
                "pred_nan_when_obs_valid": rep["pred_nan_when_obs_valid"],
                "median_nse": rep["median_nse"],
                "mean_nse": rep["mean_nse"],
                "median_kge": rep["median_kge"],
                "median_abs_fhv": rep["median_abs_fhv"],
                "nse_le_zero_count": rep["nse_le_zero_count"],
                "gauch_2023_metrics": rep["gauch_2023_metrics"],
                "regional_huc02": rep["regional_huc02"],
            }
        )

    return {
        "bucket_uri": bucket_uri,
        "total_recalculated": len(reports),
        "reports": reports,
    }


_ENSEMBLE_NAME_PATTERN = re.compile(
    r"(?i)\bensembles?\b|\b\d+\s*-\s*(?:model|seed|member|calibration)\b"
)
_RESIDUAL_LOCAL_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:hydrographs/category_[a-z_]+/[^/]+\.np[yz]|submissions/category_[a-z_]+/.*)$"
)
_PLACEHOLDER_ZENODO_DOI = "10.5281/zenodo.15000531"
_RESIDUAL_ROW_KEYS = ("mip_hydrograph_path", "mip_manifest_path")


def infer_legacy_is_ensemble(
    row: dict[str, Any],
    contribution_entry: dict[str, Any] | None = None,
) -> bool:
    """Infer whether a legacy leaderboard row describes an ensemble-mean hydrograph.

    Precedence: an explicit boolean `is_ensemble` on the row, then on its
    `contributions_index.json` entry, then the model name (`"ensemble"` or an
    `"<n>-model/seed/member/calibration"` qualifier). Nothing else is consulted; a name
    without such a qualifier is a single model.
    """
    for source in (row, contribution_entry or {}):
        value = source.get("is_ensemble")
        if isinstance(value, bool):
            return value
    return bool(_ENSEMBLE_NAME_PATTERN.search(str(row.get("model_name", ""))))


def normalize_legacy_row(
    row: dict[str, Any],
    contribution_entry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return a normalized copy of a legacy leaderboard row and the list of changes applied.

    Repairs three audited defects without inventing any number:

    1. Sentinel values: a non-finite `primary_ranking_nse` / `ens_med_nse` /
       `single_med_nse` (e.g. the legacy `-999.0`) becomes `None` with display `"N/A"`.
    2. Collapsed columns: legacy rows carried the same value in `ens_med_nse` and
       `single_med_nse`. The column that the row cannot support is cleared: an ensemble
       row keeps `ens_med_nse` and gets `single_med_nse = None` (unknown without
       per-member hydrographs); a single-model row keeps `single_med_nse` and gets
       `ens_med_nse = None`. Rows that already carry a `single_med_nse_std` (i.e. were
       produced by the evaluator from `qsim_members`) are left untouched.
    3. Residual references: `mip_hydrograph_path` / `mip_manifest_path` keys and any URL
       pointing at non-existent in-repo `hydrographs/category_*/*.npz` or
       `submissions/category_*/...` paths are removed, and the placeholder Zenodo DOI is
       cleared.
    """
    out = copy.deepcopy(row)
    changes: list[str] = []

    for key, display_key in (
        ("ens_med_nse", "ens_med_nse_display"),
        ("single_med_nse", "single_med_nse_display"),
    ):
        value = out.get(key)
        if value is not None and not _is_valid_nse(value):
            out[key] = None
            out[display_key] = "N/A"
            changes.append(f"{key}: non-finite or sentinel value {value!r} -> None")
    p_value = out.get("primary_ranking_nse")
    if p_value is not None and not _is_valid_nse(p_value):
        out["primary_ranking_nse"] = None
        changes.append(f"primary_ranking_nse: non-finite or sentinel value {p_value!r} -> None")

    is_ensemble = infer_legacy_is_ensemble(out, contribution_entry)
    out["is_ensemble"] = is_ensemble
    ens_v = out.get("ens_med_nse")
    single_v = out.get("single_med_nse")
    already_split = out.get("single_med_nse_std") is not None
    if not already_split and ens_v is not None and single_v is not None and ens_v == single_v:
        if is_ensemble:
            out["single_med_nse"] = None
            out["single_med_nse_display"] = "---"
            changes.append("single_med_nse: duplicated ensemble value -> None (unknown)")
        else:
            out["ens_med_nse"] = None
            out["ens_med_nse_display"] = "---"
            changes.append("ens_med_nse: duplicated single value -> None (single model)")
    out.setdefault("single_med_nse_std", None)

    for key in _RESIDUAL_ROW_KEYS:
        if key in out:
            changes.append(f"{key}: removed residual local path {out[key]!r}")
            del out[key]
    for key in ("zenodo_url", "data_archive_url", "github_url", "code_url", "hydroshare_url"):
        value = out.get(key)
        if isinstance(value, str) and _RESIDUAL_LOCAL_PATH_PATTERN.search(value):
            changes.append(f"{key}: removed residual path {value!r}")
            out[key] = ""
    if out.get("zenodo_doi") == _PLACEHOLDER_ZENODO_DOI:
        out["zenodo_doi"] = ""
        changes.append("zenodo_doi: removed placeholder DOI")
    if isinstance(out.get("zenodo_url"), str) and _PLACEHOLDER_ZENODO_DOI in out["zenodo_url"]:
        out["zenodo_url"] = ""
        changes.append("zenodo_url: removed placeholder DOI URL")

    old_name = str(out.get("model_name", ""))
    inferred_cit = _extract_leading_citation_if_missing(old_name, str(out.get("citation", "")))
    if inferred_cit and not str(out.get("citation", "")).strip():
        out["citation"] = inferred_cit
        changes.append(f"citation: extracted {inferred_cit!r} from model_name")
    new_name = clean_model_name(
        old_name,
        is_ensemble=out.get("is_ensemble"),
        num_members=out.get("num_members"),
    )
    if new_name and new_name != old_name:
        out["model_name"] = new_name
        changes.append(f"model_name: {old_name!r} -> {new_name!r}")
    if out.get("family_name"):
        old_fam = str(out["family_name"])
        new_fam = clean_model_name(old_fam, include_ensemble_suffix=False)
        if new_fam and new_fam != old_fam:
            out["family_name"] = new_fam
            changes.append(f"family_name: {old_fam!r} -> {new_fam!r}")

    return out, changes


def normalize_legacy_leaderboards(
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    update_figures: bool = True,
) -> dict[str, Any]:
    """Apply `normalize_legacy_row` to every row of every leaderboard JSON file.

    Also scrubs residual local paths and the placeholder DOI from
    `contributions_index.json`, re-ranks every leaderboard, re-synchronizes
    `tables_3_to_14.json`, and (optionally) regenerates the paper figures. Returns a
    change log keyed by `<leaderboard_id>/<entry_id>`.
    """
    lb_dir = Path(leaderboards_dir)
    ensure_per_leaderboard_files(lb_dir)
    idx_path = lb_dir / "contributions_index.json"
    idx_data: dict[str, Any] = _read_json_dict(idx_path) if idx_path.is_file() else {"entries": {}}
    ci_entries: dict[str, dict[str, Any]] = idx_data.get("entries", {})

    change_log: dict[str, list[str]] = {}
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_path = lb_dir / f"{lbid}.json"
        lb_data = _read_json_dict(lb_path)
        new_rows: list[dict[str, Any]] = []
        for row in lb_data.get("rows", []):
            eid = str(row.get("entry_id", ""))
            normalized, changes = normalize_legacy_row(row, ci_entries.get(eid))
            if changes:
                change_log[f"{lbid}/{eid}"] = changes
            new_rows.append(normalized)
        ranked_rows = rank_leaderboard_rows(new_rows)
        lb_data["rows"] = ranked_rows
        lb_data["n_entries"] = len(ranked_rows)
        if ranked_rows and ranked_rows[0].get("primary_ranking_nse") is not None:
            lb_data["top_model"] = ranked_rows[0]["model_name"]
            lb_data["top_nse"] = ranked_rows[0]["primary_ranking_nse"]
        else:
            lb_data["top_model"] = None
            lb_data["top_nse"] = None
        _write_json_dict(lb_path, lb_data)

    if idx_path.is_file():
        for cid, centry in ci_entries.items():
            for key in ("hydrograph_path", "data_url", "code_url", "zenodo_url"):
                value = centry.get(key)
                if isinstance(value, str) and _RESIDUAL_LOCAL_PATH_PATTERN.search(value):
                    change_log.setdefault(f"contributions_index/{cid}", []).append(
                        f"{key}: removed residual path {value!r}"
                    )
                    centry[key] = ""
            if centry.get("zenodo_doi") == _PLACEHOLDER_ZENODO_DOI:
                centry["zenodo_doi"] = ""
                change_log.setdefault(f"contributions_index/{cid}", []).append(
                    "zenodo_doi: removed placeholder DOI"
                )
            if centry.get("model_name"):
                old_c_name = str(centry["model_name"])
                new_c_name = clean_model_name(
                    old_c_name,
                    is_ensemble=centry.get("is_ensemble"),
                    num_members=centry.get("num_members"),
                )
                if new_c_name and new_c_name != old_c_name:
                    centry["model_name"] = new_c_name
                    change_log.setdefault(f"contributions_index/{cid}", []).append(
                        f"model_name: {old_c_name!r} -> {new_c_name!r}"
                    )
        _write_json_dict(idx_path, idx_data)

    sync_master_tables_from_per_leaderboard_files(lb_dir)
    if update_figures:
        from leaderboards.generate_figures import generate_all_paper_figures

        generate_all_paper_figures(leaderboards_dir=lb_dir)
    return {"total_rows_changed": len(change_log), "changes": change_log}


def find_duplicate_metric_rows(
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
) -> list[dict[str, Any]]:
    """Find rows within one leaderboard that share identical headline metrics.

    Two different `entry_id`s with bit-identical (`primary_ranking_nse`, `mean_nse`,
    `med_kge`, `med_fhv_abs`, `nse_le_0`) cannot come from two different hydrographs.
    Returns one record per duplicate group (never modifies anything).
    """
    lb_dir = Path(leaderboards_dir)
    groups: list[dict[str, Any]] = []
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_path = lb_dir / f"{lbid}.json"
        if not lb_path.is_file():
            continue
        by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in _read_json_dict(lb_path).get("rows", []):
            signature = tuple(
                row.get(k) for k in ("primary_ranking_nse", "mean_nse", "med_kge", "med_fhv_abs", "nse_le_0")
            )
            if signature[0] is None or any(v is None for v in signature[1:]):
                continue
            by_signature.setdefault(signature, []).append(row)
        for signature, rows in by_signature.items():
            if len(rows) > 1:
                groups.append(
                    {
                        "leaderboard_id": lbid,
                        "signature": signature,
                        "entries": [
                            {
                                "entry_id": r.get("entry_id"),
                                "model_name": r.get("model_name"),
                                "is_ensemble": r.get("is_ensemble"),
                                "num_members": r.get("num_members"),
                            }
                            for r in rows
                        ],
                    }
                )
    return groups


def remove_leaderboard_entry(
    leaderboard_id: str,
    entry_id: str,
    reason: str,
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    update_figures: bool = True,
) -> dict[str, Any]:
    """Move one row out of `leaderboards/<leaderboard_id>.json` into its `removed_rows` ledger.

    The removal is transparent and reversible: the full original row is preserved under
    `removed_rows` together with `reason` and the UTC timestamp, the remaining rows are
    re-ranked, `tables_3_to_14.json` is re-synchronized, and the matching
    `contributions_index.json` entry (if any) is annotated with `removed_from_leaderboard`.

    Raises:
        ValueError: If `leaderboard_id` is unknown, `reason` is empty, or `entry_id` is absent.
    """
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise ValueError(
            f"Unknown leaderboard_id {leaderboard_id!r}; expected one of {CANONICAL_LEADERBOARD_IDS}."
        )
    if not reason.strip():
        raise ValueError("A non-empty reason is required to remove a leaderboard entry.")
    lb_dir = Path(leaderboards_dir)
    lb_data = load_leaderboard(leaderboard_id, lb_dir)
    rows = list(lb_data.get("rows", []))
    matches = [r for r in rows if str(r.get("entry_id")) == entry_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row with entry_id {entry_id!r} in {leaderboard_id}, "
            f"found {len(matches)}."
        )
    removed = copy.deepcopy(matches[0])
    remaining = [r for r in rows if str(r.get("entry_id")) != entry_id]
    ledger_record = {
        "removed_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reason": reason.strip(),
        "row": removed,
    }
    ranked_rows = rank_leaderboard_rows(remaining)
    lb_data["rows"] = ranked_rows
    lb_data["n_entries"] = len(ranked_rows)
    lb_data.setdefault("removed_rows", []).append(ledger_record)
    if ranked_rows and ranked_rows[0].get("primary_ranking_nse") is not None:
        lb_data["top_model"] = ranked_rows[0]["model_name"]
        lb_data["top_nse"] = ranked_rows[0]["primary_ranking_nse"]
    else:
        lb_data["top_model"] = None
        lb_data["top_nse"] = None
    _write_json_dict(lb_dir / f"{leaderboard_id}.json", lb_data)

    idx_path = lb_dir / "contributions_index.json"
    if idx_path.is_file():
        idx_data = _read_json_dict(idx_path)
        centry = idx_data.get("entries", {}).get(entry_id)
        if centry is not None:
            centry["removed_from_leaderboard"] = {
                "leaderboard_id": leaderboard_id,
                "reason": reason.strip(),
                "removed_at_utc": ledger_record["removed_at_utc"],
            }
            _write_json_dict(idx_path, idx_data)

    sync_master_tables_from_per_leaderboard_files(lb_dir)
    if update_figures:
        from leaderboards.generate_figures import generate_all_paper_figures

        generate_all_paper_figures(leaderboards_dir=lb_dir)
    return ledger_record


def reconcile_archived_rows(
    bucket_uri: str = DEFAULT_BUCKET_URI,
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    apply: bool = False,
    update_figures: bool = True,
) -> dict[str, Any]:
    """Re-score every leaderboard row whose hydrograph store exists in the bucket.

    Literature rows (`Literature` / `Archive*`) were seeded with author-reported numbers.
    Where the archived hydrograph `<bucket_uri>/hydrographs/<leaderboard_id>/<entry_id>.zarr`
    exists, this function scores it with `metrics.evaluator.calculate_model_leaderboard_metrics`
    and reports the discrepancy against the row. With `apply=False` (default) nothing is
    modified; with `apply=True` every such row is rewritten from the evaluator (the previous
    value is preserved under `author_reported_median_nse`) and re-ranked. A store that fails
    validation (e.g. an ensemble `qsim` that is not the member mean) raises immediately; fix
    or remove the store rather than skipping it.
    """
    from metrics.evaluator import _get_gcs_storage_options, _zarr_store_exists

    lb_dir = Path(leaderboards_dir)
    ensure_per_leaderboard_files(lb_dir)
    root_uri = bucket_uri.rstrip("/")
    storage_options = _get_gcs_storage_options() if root_uri.startswith("gs://") else None

    compared: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_path = lb_dir / f"{lbid}.json"
        lb_data = _read_json_dict(lb_path)
        rows = list(lb_data.get("rows", []))
        modified = False
        self_run_categories = {
            "mip re-run",
            "mip-rerun",
            "mip rerun",
            "contribution",
            "contributed",
        }
        archive_categories = {
            "archive",
        }
        for idx, row in enumerate(rows):
            eid = str(row.get("entry_id", "")).strip()
            cat = str(row.get("category", "")).lower().strip()
            store_uri = f"{root_uri}/hydrographs/{lbid}/{eid}.zarr"

            is_self_run = cat in self_run_categories
            is_backed_archive = False
            if cat in archive_categories:
                url = str(row.get("data_archive_url", "")).strip() or str(row.get("zenodo_url", "")).strip()
                if url.startswith(f"{root_uri}/hydrographs/"):
                    is_backed_archive = True

            if not _zarr_store_exists(store_uri, storage_options=storage_options):
                if is_self_run or is_backed_archive:
                    raise FileNotFoundError(
                        f"Missing required Zarr store {store_uri} for {lbid}/{eid} (category {cat})."
                    )
                missing.append({"leaderboard_id": lbid, "entry_id": eid, "category": row.get("category")})
                continue
            print(f"Reconciling {lbid}/{eid} ...", flush=True)
            ms = calculate_model_leaderboard_metrics(
                model_id=eid, leaderboard_id=lbid, bucket_uri=root_uri
            )
            row_nse = _primary_nse(row)
            record = {
                "leaderboard_id": lbid,
                "entry_id": eid,
                "category": row.get("category"),
                "model_name": row.get("model_name"),
                "leaderboard_median_nse": row_nse,
                "archived_hydrograph_median_nse": round(float(ms["median_nse"]), 4),
                "difference": (
                    round(float(row_nse) - float(ms["median_nse"]), 4) if row_nse is not None else None
                ),
                "is_ensemble": bool(ms["is_ensemble"]),
                "num_members": int(ms["num_members"]),
                "single_seed_median_nse_mean": ms["single_seed_median_nse_mean"],
                "single_seed_median_nse_std": ms["single_seed_median_nse_std"],
            }
            compared.append(record)
            if apply:
                updated_row = copy.deepcopy(row)
                updated_row["author_reported_median_nse"] = row_nse
                updated_row.update(format_row_metrics_from_evaluator(ms))
                updated_row["model_name"] = clean_model_name(
                    str(updated_row.get("model_name", "")),
                    is_ensemble=updated_row.get("is_ensemble"),
                    num_members=updated_row.get("num_members"),
                )
                if updated_row.get("family_name"):
                    updated_row["family_name"] = clean_model_name(
                        str(updated_row["family_name"]),
                        include_ensemble_suffix=False,
                    )
                rows[idx] = updated_row
                modified = True
        if apply and modified:
            ranked_rows = rank_leaderboard_rows(rows)
            lb_data["rows"] = ranked_rows
            lb_data["n_entries"] = len(ranked_rows)
            if ranked_rows and ranked_rows[0].get("primary_ranking_nse") is not None:
                lb_data["top_model"] = ranked_rows[0]["model_name"]
                lb_data["top_nse"] = ranked_rows[0]["primary_ranking_nse"]
            _write_json_dict(lb_path, lb_data)

    if apply:
        sync_master_tables_from_per_leaderboard_files(lb_dir)
        if update_figures:
            from leaderboards.generate_figures import generate_all_paper_figures

            generate_all_paper_figures(leaderboards_dir=lb_dir)

    return {
        "bucket_uri": root_uri,
        "applied": bool(apply),
        "num_compared": len(compared),
        "num_missing_store": len(missing),
        "compared": compared,
        "missing_store": missing,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mip-update-leaderboard",
        description=(
            "Score a model Zarr store in gs://camels-us-mip/ and update the "
            "corresponding leaderboard JSON file in leaderboards/."
        ),
    )
    parser.add_argument(
        "--recalculate-all-self-run",
        action="store_true",
        help="Recalculate metrics for all self-run models in the repository.",
    )
    parser.add_argument(
        "--normalize-legacy-rows",
        action="store_true",
        help=(
            "One-off repair of legacy leaderboard rows: clear non-finite sentinels, split "
            "the collapsed ensemble/single-seed columns, and drop residual local paths."
        ),
    )
    parser.add_argument(
        "--reconcile-archived",
        action="store_true",
        help=(
            "Re-score every leaderboard row whose hydrograph store exists in the bucket and "
            "report discrepancies against the stored (author-reported) numbers; combine with "
            "--apply to rewrite those rows from the evaluator."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --reconcile-archived: write the evaluator numbers into the leaderboards.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the JSON report of --reconcile-archived / --normalize-legacy-rows.",
    )
    parser.add_argument("--model-id", help="Model Zarr store ID.")
    parser.add_argument(
        "--leaderboard-id",
        choices=CANONICAL_LEADERBOARD_IDS,
        help="Target leaderboard ID.",
    )
    parser.add_argument("--model-name", help="Display name for leaderboard row.")
    parser.add_argument("--family-id", help="Model family identifier.")
    parser.add_argument("--forcing", help="Forcing product label (e.g., Daymet).")
    parser.add_argument(
        "--is-ensemble",
        action="store_true",
        default=None,
        help=(
            "Cross-check only: assert that the Zarr store is an ensemble-mean hydrograph "
            "(attribute 'is_ensemble' / 'qsim_members'). A claim that contradicts the "
            "store is rejected; the store is always authoritative."
        ),
    )
    parser.add_argument(
        "--category",
        default="Contribution",
        help="Provenance category label (default: 'Contribution').",
    )
    parser.add_argument("--citation", default="Community submission", help="Citation string.")
    parser.add_argument("--code-url", default="", help="Open-source code repository URL.")
    parser.add_argument("--data-archive-url", default="", help="Data archive URL or Zarr URI.")
    parser.add_argument(
        "--bucket-uri",
        default=DEFAULT_BUCKET_URI,
        help=f"Cloud bucket root URI (default: {DEFAULT_BUCKET_URI}).",
    )
    parser.add_argument(
        "--mip",
        default=DEFAULT_MIP_ID,
        help=f"Registered MIP identifier under mips/ (default: {DEFAULT_MIP_ID}).",
    )
    parser.add_argument(
        "--leaderboards-dir",
        default=None,
        help="Path to the MIP leaderboards/ directory (defaults to mips/<mip>/leaderboards).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    leaderboards_dir = (
        Path(args.leaderboards_dir)
        if args.leaderboards_dir is not None
        else resolve_mip_paths(args.mip).leaderboards_dir
    )

    def _emit(result: dict[str, Any]) -> None:
        text = json.dumps(result, indent=2) + "\n"
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        print(text)

    if args.normalize_legacy_rows:
        _emit(normalize_legacy_leaderboards(leaderboards_dir=leaderboards_dir))
        return 0
    if args.reconcile_archived:
        _emit(
            reconcile_archived_rows(
                bucket_uri=args.bucket_uri,
                leaderboards_dir=leaderboards_dir,
                apply=bool(args.apply),
            )
        )
        return 0
    if args.recalculate_all_self_run:
        result = recalculate_all_self_run_models(
            bucket_uri=args.bucket_uri,
            leaderboards_dir=leaderboards_dir,
        )
        report_path = Path(REPO_ROOT) / "benchmarks" / "recalculated_self_run_models_report.json"
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Recalculation complete. Report saved to {report_path}")
        return 0
    missing = [
        name
        for name, value in (
            ("--model-id", args.model_id),
            ("--leaderboard-id", args.leaderboard_id),
            ("--model-name", args.model_name),
            ("--family-id", args.family_id),
            ("--forcing", args.forcing),
        )
        if not value
    ]
    if missing:
        parser.error(f"Missing required arguments for a submission: {', '.join(missing)}")
    result = trigger_submission_from_gcs(
        model_id=args.model_id,
        leaderboard_id=args.leaderboard_id,
        model_name=args.model_name,
        family_id=args.family_id,
        forcing=args.forcing,
        is_ensemble=args.is_ensemble,
        category=args.category,
        citation=args.citation,
        code_url=args.code_url,
        data_archive_url=args.data_archive_url,
        bucket_uri=args.bucket_uri,
        leaderboards_dir=leaderboards_dir,
    )
    print(json.dumps(result["updated_entry"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
