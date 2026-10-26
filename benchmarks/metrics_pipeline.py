"""Canonical cloud-only Zarr metrics benchmark pipeline (`benchmarks/metrics_pipeline.py`).

Iterates over a manifest or all Zarr stores in
`<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr`, invokes
`calculate_model_leaderboard_metrics(model_id, leaderboard_id, bucket_uri)` for each
entry, enforces strict anti-masking (`pred_nan_when_obs_valid == 0`), 531-basin
coverage, and full lower-tail / upper-tail / 18-region HUC-02 reporting, and emits
a structured verification report.
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np

from metrics.evaluator import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_BUCKET_URI,
    NUM_BASINS,
    NUM_EVAL_DAYS,
    _get_gcs_storage_options,
    calculate_model_leaderboard_metrics,
)

REQUIRED_HEADLINE_KEYS: tuple[str, ...] = (
    "median_nse",
    "mean_nse",
    "median_kge",
    "median_abs_fhv",
    "nse_le_zero_count",
)

REQUIRED_LOWER_TAIL_NSE_KEYS: tuple[str, ...] = (
    "min_nse",
    "p01_nse",
    "p05_nse",
    "p10_nse",
    "q25_nse",
    "median_nse",
    "q75_nse",
)

REQUIRED_LOWER_TAIL_KGE_KEYS: tuple[str, ...] = (
    "min_kge",
    "p01_kge",
    "p05_kge",
    "p10_kge",
    "q25_kge",
    "median_kge",
    "q75_kge",
)

REQUIRED_UPPER_TAIL_FHV_KEYS: tuple[str, ...] = (
    "p50_abs_fhv",
    "p75_abs_fhv",
    "p90_abs_fhv",
    "p95_abs_fhv",
    "p99_abs_fhv",
    "max_abs_fhv",
)

EXPECTED_HUC02_REGIONS: tuple[str, ...] = tuple(f"{i:02d}" for i in range(1, 19))


def discover_hydrograph_zarr_entries(
    bucket_uri: str = DEFAULT_BUCKET_URI,
    leaderboard_ids: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Discover all `(leaderboard_id, model_id)` Zarr stores in `<bucket_uri>/hydrographs/`.

    Args:
        bucket_uri: Root URI of the cloud bucket or local Zarr directory.
        leaderboard_ids: Optional subset of canonical leaderboard IDs to scan.

    Returns:
        Sorted list of `(leaderboard_id, model_id)` tuples discovered.

    Raises:
        ValueError: If any requested `leaderboard_id` is not in `CANONICAL_LEADERBOARD_IDS`.
        FileNotFoundError: If no hydrograph Zarr stores are found across the scanned partitions.
    """
    target_leaderboards = (
        tuple(leaderboard_ids)
        if leaderboard_ids is not None
        else CANONICAL_LEADERBOARD_IDS
    )
    for lb_id in target_leaderboards:
        if lb_id not in CANONICAL_LEADERBOARD_IDS:
            raise ValueError(
                f"Invalid leaderboard_id '{lb_id}'. Must be one of {CANONICAL_LEADERBOARD_IDS}."
            )

    root_uri = bucket_uri.rstrip("/")
    discovered: list[tuple[str, str]] = []

    if root_uri.startswith("gs://"):
        storage_options = _get_gcs_storage_options()
        fs = gcsfs.GCSFileSystem(**storage_options)
        for lb_id in target_leaderboards:
            lb_prefix = f"{root_uri}/hydrographs/{lb_id}"
            if fs.exists(lb_prefix):
                items = fs.ls(lb_prefix, detail=False)
                for item in sorted(items):
                    name = item.rstrip("/").split("/")[-1]
                    if name.endswith(".zarr"):
                        model_id = name[: -len(".zarr")]
                        if model_id:
                            discovered.append((lb_id, model_id))
    else:
        local_str = (
            root_uri.removeprefix("file://")
        )
        hydro_root = Path(local_str) / "hydrographs"
        if not hydro_root.exists() or not hydro_root.is_dir():
            raise FileNotFoundError(
                f"Hydrographs directory not found at '{hydro_root}'."
            )
        for lb_id in target_leaderboards:
            lb_dir = hydro_root / lb_id
            if lb_dir.exists() and lb_dir.is_dir():
                for child in sorted(lb_dir.iterdir()):
                    if child.is_dir() and child.name.endswith(".zarr"):
                        model_id = child.name[: -len(".zarr")]
                        if model_id:
                            discovered.append((lb_id, model_id))

    if not discovered:
        raise FileNotFoundError(
            f"Zero hydrograph Zarr stores discovered under '{root_uri}/hydrographs/'."
        )
    return discovered


def load_manifest_entries(manifest_path: str | Path) -> list[tuple[str, str]]:
    """Load `(leaderboard_id, model_id)` pairs from a JSON manifest file.

    Supports either:
      1. A JSON list of `{"leaderboard_id": ..., "model_id": ...}` objects, or
      2. A JSON dict with `"entries"` mapping `<model_id>` to metadata containing
         `"leaderboard_id"` or `"target_leaderboard_tables"`.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []

    if isinstance(payload, list):
        for item in payload:
            if (
                not isinstance(item, dict)
                or "leaderboard_id" not in item
                or "model_id" not in item
            ):
                raise KeyError(
                    "Each manifest list item must contain 'leaderboard_id' and 'model_id'."
                )
            lb_id = str(item["leaderboard_id"])
            m_id = str(item["model_id"])
            if lb_id not in CANONICAL_LEADERBOARD_IDS:
                raise ValueError(f"Invalid leaderboard_id '{lb_id}' in manifest.")
            pairs.append((lb_id, m_id))
    elif isinstance(payload, dict) and "entries" in payload:
        entries_obj = payload["entries"]
        if isinstance(entries_obj, list):
            for item in entries_obj:
                if (
                    not isinstance(item, dict)
                    or "leaderboard_id" not in item
                    or "model_id" not in item
                ):
                    raise KeyError(
                        "Each manifest entry must contain 'leaderboard_id' and 'model_id'."
                    )
                lb_id = str(item["leaderboard_id"])
                m_id = str(item["model_id"])
                if lb_id not in CANONICAL_LEADERBOARD_IDS:
                    raise ValueError(
                        f"Invalid leaderboard_id '{lb_id}' in manifest."
                    )
                pairs.append((lb_id, m_id))
        elif isinstance(entries_obj, dict):
            for entry_key, meta in entries_obj.items():
                if not isinstance(meta, dict):
                    raise TypeError(
                        f"Manifest entry '{entry_key}' must be a dictionary."
                    )
                m_id = str(entry_key)
                if "entry_id" in meta:
                    m_id = str(meta["entry_id"])
                if "leaderboard_id" in meta:
                    lb_id = str(meta["leaderboard_id"])
                    if lb_id not in CANONICAL_LEADERBOARD_IDS:
                        raise ValueError(
                            f"Invalid leaderboard_id '{lb_id}' for '{m_id}'."
                        )
                    pairs.append((lb_id, m_id))
                elif "target_leaderboard_tables" in meta:
                    matched_lbs = [
                        str(t)
                        for t in meta["target_leaderboard_tables"]
                        if str(t) in CANONICAL_LEADERBOARD_IDS
                    ]
                    if not matched_lbs:
                        raise ValueError(
                            f"No canonical leaderboard ID found in target_leaderboard_tables for '{m_id}'."
                        )
                    for lb_id in matched_lbs:
                        pairs.append((lb_id, m_id))
                else:
                    raise KeyError(
                        f"Manifest entry '{m_id}' is missing 'leaderboard_id' or 'target_leaderboard_tables'."
                    )
        else:
            raise TypeError("Manifest 'entries' field must be a list or dict.")
    else:
        raise ValueError(
            f"Unsupported manifest structure in '{path}'; expected a list or dict with 'entries'."
        )

    if not pairs:
        raise ValueError(f"Manifest '{path}' contained zero model entries.")
    return pairs


def verify_single_entry_report(entry_metrics: dict[str, Any]) -> None:
    """Verify all 5 non-negotiable CAMELS-US MIP benchmarking invariants on one model result.

    Raises:
        KeyError: If any required metric, lower-tail percentile, upper-tail percentile,
            or HUC-02 regional key is missing.
        ValueError: If `pred_nan_when_obs_valid != 0`, basin coverage is incomplete,
            metrics are non-finite, or percentile ordering is violated.
    """
    for req_key in (
        "model_id",
        "leaderboard_id",
        "num_total_basins",
        "num_evaluated_basins",
        "num_eval_days",
        "pred_nan_when_obs_valid",
        "regional_huc02",
        "per_basin",
        *REQUIRED_HEADLINE_KEYS,
        *REQUIRED_LOWER_TAIL_NSE_KEYS,
        *REQUIRED_LOWER_TAIL_KGE_KEYS,
        *REQUIRED_UPPER_TAIL_FHV_KEYS,
        "mean_kge",
        "median_of_18_huc02_medians",
        "worst_huc02_region",
        "best_huc02_region",
    ):
        if req_key not in entry_metrics:
            raise KeyError(
                f"Entry report is missing mandatory benchmark key '{req_key}'."
            )

    if int(entry_metrics["pred_nan_when_obs_valid"]) != 0:
        raise ValueError(
            f"Anti-masking violation for '{entry_metrics['model_id']}': "
            f"pred_nan_when_obs_valid = {entry_metrics['pred_nan_when_obs_valid']} (expected 0)."
        )

    if int(entry_metrics["num_total_basins"]) != NUM_BASINS:
        raise ValueError(
            f"Expected num_total_basins == {NUM_BASINS}, got {entry_metrics['num_total_basins']}."
        )
    # The canonical `newman_eval_targets.zarr` has 531/531 basins with 100% valid daily
    # observations over 1989-10-01..1999-09-30 (verified against raw USGS files), so every
    # basin is evaluable and full coverage is required. Evaluability depends only on the
    # observations, never on the model, so this cannot be gamed by a submission.
    if int(entry_metrics["num_evaluated_basins"]) != NUM_BASINS:
        raise ValueError(
            f"Incomplete basin coverage for '{entry_metrics['model_id']}': "
            f"num_evaluated_basins = {entry_metrics['num_evaluated_basins']} (expected {NUM_BASINS})."
        )
    if int(entry_metrics["num_eval_days"]) != NUM_EVAL_DAYS:
        raise ValueError(
            f"Expected num_eval_days == {NUM_EVAL_DAYS}, got {entry_metrics['num_eval_days']}."
        )

    # Verify all required scalar metrics are finite
    for scalar_key in (
        *REQUIRED_HEADLINE_KEYS,
        *REQUIRED_LOWER_TAIL_NSE_KEYS,
        *REQUIRED_LOWER_TAIL_KGE_KEYS,
        *REQUIRED_UPPER_TAIL_FHV_KEYS,
        "mean_kge",
        "median_of_18_huc02_medians",
    ):
        val = float(entry_metrics[scalar_key])
        if not np.isfinite(val):
            raise ValueError(
                f"Metric '{scalar_key}' for '{entry_metrics['model_id']}' is non-finite ({val})."
            )

    # Verify monotonic ordering of lower-tail NSE percentiles
    nse_p = [float(entry_metrics[k]) for k in REQUIRED_LOWER_TAIL_NSE_KEYS]
    if any(nse_p[i] > nse_p[i + 1] + 1e-6 for i in range(len(nse_p) - 1)):
        raise ValueError(
            f"Lower-tail NSE percentiles are not monotonically ordered for "
            f"'{entry_metrics['model_id']}': {dict(zip(REQUIRED_LOWER_TAIL_NSE_KEYS, nse_p))}"
        )

    # Verify monotonic ordering of lower-tail KGE percentiles
    kge_p = [float(entry_metrics[k]) for k in REQUIRED_LOWER_TAIL_KGE_KEYS]
    if any(kge_p[i] > kge_p[i + 1] + 1e-6 for i in range(len(kge_p) - 1)):
        raise ValueError(
            f"Lower-tail KGE percentiles are not monotonically ordered for "
            f"'{entry_metrics['model_id']}': {dict(zip(REQUIRED_LOWER_TAIL_KGE_KEYS, kge_p))}"
        )

    # Verify monotonic ordering of upper-tail |FHV| percentiles
    fhv_p = [float(entry_metrics[k]) for k in REQUIRED_UPPER_TAIL_FHV_KEYS]
    if fhv_p[0] < 0.0 or any(
        fhv_p[i] > fhv_p[i + 1] + 1e-6 for i in range(len(fhv_p) - 1)
    ):
        raise ValueError(
            f"Upper-tail |FHV| percentiles are invalid for '{entry_metrics['model_id']}': "
            f"{dict(zip(REQUIRED_UPPER_TAIL_FHV_KEYS, fhv_p))}"
        )

    # Verify all 18 USGS HUC-02 regional medians exist and are finite
    regional = entry_metrics["regional_huc02"]
    for huc_code in EXPECTED_HUC02_REGIONS:
        if huc_code not in regional:
            raise KeyError(
                f"Missing USGS HUC-02 region '{huc_code}' in regional_huc02 for "
                f"'{entry_metrics['model_id']}'."
            )
        if not np.isfinite(float(regional[huc_code])):
            raise ValueError(
                f"Non-finite HUC-02 regional median for region '{huc_code}' in "
                f"'{entry_metrics['model_id']}'."
            )


def run_metrics_benchmark_pipeline(
    bucket_uri: str = DEFAULT_BUCKET_URI,
    entries: Sequence[tuple[str, str]] | None = None,
    manifest_path: str | Path | None = None,
    leaderboard_ids: Sequence[str] | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Score and verify all model hydrograph Zarr stores across leaderboards.

    Args:
        bucket_uri: Root URI of the cloud bucket or local Zarr store directory.
        entries: Optional explicit sequence of `(leaderboard_id, model_id)` pairs.
        manifest_path: Optional JSON manifest listing entries to score.
        leaderboard_ids: Optional subset of canonical leaderboard IDs to discover.
        output_json: Optional output file path to write the structured JSON report.

    Returns:
        Structured benchmark verification report containing per-leaderboard ranked
        summaries (headline metrics, lower-tail NSE/KGE percentiles, upper-tail
        |FHV| percentiles, 18-region HUC-02 medians) and full entry records.
    """
    if entries is not None:
        target_entries = list(entries)
        if not target_entries:
            raise ValueError("Provided 'entries' sequence must not be empty.")
    elif manifest_path is not None:
        target_entries = load_manifest_entries(manifest_path)
    else:
        target_entries = discover_hydrograph_zarr_entries(
            bucket_uri=bucket_uri,
            leaderboard_ids=leaderboard_ids,
        )

    evaluated_entries: dict[str, dict[str, Any]] = {}
    by_leaderboard: dict[str, list[dict[str, Any]]] = {}
    total_pred_nan = 0

    for leaderboard_id, model_id in target_entries:
        metrics_res = calculate_model_leaderboard_metrics(
            model_id=model_id,
            leaderboard_id=leaderboard_id,
            bucket_uri=bucket_uri,
        )
        verify_single_entry_report(metrics_res)

        total_pred_nan += int(metrics_res["pred_nan_when_obs_valid"])
        composite_key = f"{leaderboard_id}/{model_id}"
        evaluated_entries[composite_key] = metrics_res

        row_summary = {
            "model_id": model_id,
            "leaderboard_id": leaderboard_id,
            "hydrograph_uri": metrics_res["hydrograph_uri"],
            "num_evaluated_basins": metrics_res["num_evaluated_basins"],
            "pred_nan_when_obs_valid": metrics_res["pred_nan_when_obs_valid"],
            # Ensemble / single-seed provenance (from the store, never from names)
            "is_ensemble": metrics_res["is_ensemble"],
            "num_members": metrics_res["num_members"],
            "has_qsim_members": metrics_res["has_qsim_members"],
            "ensemble_source": metrics_res["ensemble_source"],
            "single_seed_median_nse_mean": metrics_res["single_seed_median_nse_mean"],
            "single_seed_median_nse_std": metrics_res["single_seed_median_nse_std"],
            # 5 headline metrics
            "median_nse": metrics_res["median_nse"],
            "mean_nse": metrics_res["mean_nse"],
            "median_kge": metrics_res["median_kge"],
            "median_abs_fhv": metrics_res["median_abs_fhv"],
            "nse_le_zero_count": metrics_res["nse_le_zero_count"],
            # Lower-tail NSE distribution
            "lower_tail_nse": {
                k: metrics_res[k] for k in REQUIRED_LOWER_TAIL_NSE_KEYS
            },
            # Lower-tail KGE distribution
            "lower_tail_kge": {
                k: metrics_res[k] for k in REQUIRED_LOWER_TAIL_KGE_KEYS
            },
            "mean_kge": metrics_res["mean_kge"],
            # Upper-tail |FHV| distribution
            "upper_tail_abs_fhv": {
                k: metrics_res[k] for k in REQUIRED_UPPER_TAIL_FHV_KEYS
            },
            # Regional HUC-02 breakdown
            "median_of_18_huc02_medians": metrics_res[
                "median_of_18_huc02_medians"
            ],
            "worst_huc02_region": metrics_res["worst_huc02_region"],
            "best_huc02_region": metrics_res["best_huc02_region"],
            "regional_huc02": metrics_res["regional_huc02"],
        }
        if leaderboard_id not in by_leaderboard:
            by_leaderboard[leaderboard_id] = []
        by_leaderboard[leaderboard_id].append(row_summary)

    # Sort each leaderboard descending by median_nse (then mean_nse) and assign rank
    leaderboards_report: dict[str, dict[str, Any]] = {}
    for lb_id in CANONICAL_LEADERBOARD_IDS:
        if lb_id in by_leaderboard:
            rows = sorted(
                by_leaderboard[lb_id],
                key=lambda r: (float(r["median_nse"]), float(r["mean_nse"])),
                reverse=True,
            )
            for idx, row in enumerate(rows, start=1):
                row["rank"] = idx
            leaderboards_report[lb_id] = {
                "leaderboard_id": lb_id,
                "num_entries": len(rows),
                "top_model_id": rows[0]["model_id"],
                "top_median_nse": rows[0]["median_nse"],
                "rows": rows,
            }

    report: dict[str, Any] = {
        "bucket_uri": bucket_uri.rstrip("/"),
        "verified_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_evaluated_entries": len(evaluated_entries),
        "total_pred_nan_when_obs_valid": total_pred_nan,
        "all_invariants_verified": True,
        "leaderboards": leaderboards_report,
        "entries": evaluated_entries,
    }

    if output_json is not None:
        out_path = Path(output_json)
        if out_path.parent != Path(""):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    return report


def format_verification_markdown_table(report: dict[str, Any]) -> str:
    """Format a Markdown summary table of all benchmarked models and lower-tail metrics."""
    lines = [
        "| Leaderboard | Rank | Model ID | Med NSE | Mean NSE | Min NSE | P5 NSE | P10 NSE | Q25 NSE | Q75 NSE | Med KGE | Med |FHV| (%) | P95 |FHV| (%) | NSE <= 0 | Pred NaN |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    leaderboards = report["leaderboards"]
    for lb_id, lb_block in leaderboards.items():
        for row in lb_block["rows"]:
            lt_nse = row["lower_tail_nse"]
            ut_fhv = row["upper_tail_abs_fhv"]
            lines.append(
                f"| `{lb_id}` | {row['rank']} | `{row['model_id']}` | "
                f"{row['median_nse']:.4f} | {row['mean_nse']:.4f} | "
                f"{lt_nse['min_nse']:.4f} | {lt_nse['p05_nse']:.4f} | "
                f"{lt_nse['p10_nse']:.4f} | {lt_nse['q25_nse']:.4f} | "
                f"{lt_nse['q75_nse']:.4f} | {row['median_kge']:.4f} | "
                f"{row['median_abs_fhv']:.2f} | {ut_fhv['p95_abs_fhv']:.2f} | "
                f"{row['nse_le_zero_count']} | {row['pred_nan_when_obs_valid']} |"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for `benchmark-mip-metrics`."""
    parser = argparse.ArgumentParser(
        prog="benchmark-mip-metrics",
        description=(
            "Run the end-to-end cloud Zarr metrics verification benchmark across "
            "all leaderboards in gs://camels-us-mip/."
        ),
    )
    parser.add_argument(
        "--bucket-uri",
        default=DEFAULT_BUCKET_URI,
        type=str,
        help=f"Root URI of the CAMELS-US MIP Zarr bucket (default: {DEFAULT_BUCKET_URI}).",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        type=str,
        help="Optional path to a JSON manifest specifying (leaderboard_id, model_id) entries.",
    )
    parser.add_argument(
        "--leaderboard-id",
        action="append",
        dest="leaderboard_ids",
        choices=list(CANONICAL_LEADERBOARD_IDS),
        default=None,
        help="Optional canonical leaderboard ID filter (can be passed multiple times).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        type=str,
        help="Optional path to write the structured verification JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the `benchmark-mip-metrics` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_metrics_benchmark_pipeline(
        bucket_uri=args.bucket_uri,
        manifest_path=args.manifest_path,
        leaderboard_ids=args.leaderboard_ids,
        output_json=args.output_json,
    )
    print(format_verification_markdown_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
