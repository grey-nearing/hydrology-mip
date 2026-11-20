# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hermetic Stage-1 (author-protocol) reproduction harness for the NeuralHydrology-family models.

Every `models/<model>/reproduce_author_metrics.py` in this family delegates to `run_reproduction`:
the model is **trained in-process** with the model's own `runner.run_fold` on the author's basin
list, forcing products, and train/evaluation periods; the resulting hydrograph is scored with
`metrics.metrics.calculate_all_metrics` against discharge loaded through
`models.camels_data.load_camels_slice`; and the summary is compared with the published median
NSE.  No precomputed hydrograph, checkpoint, or metric file is ever read.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hydromip.paths import DEFAULT_MIP_ID, REPO_ROOT, resolve_mip_paths
from metrics.metrics import calculate_all_metrics, nse
from models.camels_data import DEFAULT_DATA_DIR, load_camels_slice

SPLITS_DIR: Path = resolve_mip_paths(DEFAULT_MIP_ID).splits_dir
RESULTS_FILENAME: str = "stage1_author_reproduction_results.json"
# Default target of a full (non-smoke) run: a sibling of the tracked JSON that is never
# committed, so a re-run cannot silently overwrite the published Stage-1 record.  Pass
# `--write-tracked` to (deliberately) regenerate `RESULTS_FILENAME`.
LOCAL_RESULTS_FILENAME: str = "stage1_author_reproduction_results.local.json"
SMOKE_RESULTS_FILENAME: str = "stage1_smoke_results.json"
HYDROGRAPH_FILENAME: str = "stage1_author_reproduction_hydrograph.npz"


def load_basin_list(path: str | Path) -> list[str]:
    """Load a basin list from a JSON array or a one-ID-per-line text file (zero-padded to 8)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Basin list not found: {p}")
    if p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{p} must contain a JSON array of gauge IDs.")
        ids = [str(b).strip().zfill(8) for b in raw]
    else:
        ids = [line.strip().zfill(8) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"{p} contains no basin IDs.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{p} contains duplicate basin IDs.")
    return ids


def _per_basin_dataarrays(
    obs_row: np.ndarray, sim_row: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[xr.DataArray, xr.DataArray]:
    obs_da = xr.DataArray(obs_row.astype(np.float64), dims=["date"], coords={"date": dates})
    sim_da = xr.DataArray(sim_row.astype(np.float64), dims=["date"], coords={"date": dates})
    return obs_da, sim_da


MetricsFn = Callable[..., Mapping[str, float]]


def score_hydrograph(
    qsim: np.ndarray,
    qobs: np.ndarray,
    dates: Sequence[np.datetime64] | pd.DatetimeIndex,
    basins: Sequence[str],
    metrics_fn: MetricsFn = calculate_all_metrics,
) -> dict[str, object]:
    """Score `(n_basins, n_days)` simulations against observations with the Gauch et al. (2023) suite.

    Raises if any simulated value is non-finite on a day with a valid observation
    (`pred_nan_when_obs_valid` must be 0).  Basins without a single valid observation are
    reported separately and excluded from the aggregates.
    """
    qsim = np.asarray(qsim, dtype=np.float64)
    qobs = np.asarray(qobs, dtype=np.float64)
    if qsim.shape != qobs.shape or qsim.ndim != 2 or qsim.shape[0] != len(basins):
        raise ValueError(f"qsim {qsim.shape} / qobs {qobs.shape} / basins {len(basins)} are inconsistent.")
    date_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    if len(date_index) != qsim.shape[1]:
        raise ValueError("dates length does not match the hydrograph length.")

    obs_valid = np.isfinite(qobs) & (qobs >= 0.0)
    pred_nan_when_obs_valid = int(np.sum(~np.isfinite(qsim) & obs_valid))
    if pred_nan_when_obs_valid > 0:
        raise ValueError(
            f"{pred_nan_when_obs_valid} simulated values are non-finite on days with valid observations."
        )

    per_basin: dict[str, dict[str, float]] = {}
    unscored: list[str] = []
    for i, basin in enumerate(basins):
        if int(obs_valid[i].sum()) == 0:
            unscored.append(str(basin))
            continue
        obs_row = np.where(obs_valid[i], qobs[i], np.nan)
        obs_da, sim_da = _per_basin_dataarrays(obs_row, qsim[i], date_index)
        values = metrics_fn(obs_da, sim_da, resolution="1D", datetime_coord="date")
        per_basin[str(basin)] = {k: float(v) for k, v in values.items()}

    if not per_basin:
        raise ValueError("No basin has a valid observation in the evaluation window.")
    nse_values = np.asarray([v["NSE"] for v in per_basin.values()], dtype=np.float64)
    kge_values = np.asarray([v["KGE"] for v in per_basin.values()], dtype=np.float64)
    return {
        "n_scored_basins": int(len(per_basin)),
        "unscored_basins": unscored,
        "pred_nan_when_obs_valid": pred_nan_when_obs_valid,
        "median_nse": float(np.nanmedian(nse_values)),
        "mean_nse": float(np.nanmean(nse_values)),
        "median_kge": float(np.nanmedian(kge_values)),
        "nse_le_zero_count": int(np.sum(nse_values <= 0.0)),
        "per_basin": per_basin,
    }


def member_median_nse(
    members: np.ndarray, qobs: np.ndarray, dates: Sequence[np.datetime64] | pd.DatetimeIndex
) -> list[float]:
    """Median NSE across basins for every ensemble member `(n_members, n_basins, n_days)`."""
    members = np.asarray(members, dtype=np.float64)
    qobs = np.asarray(qobs, dtype=np.float64)
    date_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    obs_valid = np.isfinite(qobs) & (qobs >= 0.0)
    out: list[float] = []
    for m in range(members.shape[0]):
        vals: list[float] = []
        for i in range(qobs.shape[0]):
            if int(obs_valid[i].sum()) == 0:
                continue
            obs_da, sim_da = _per_basin_dataarrays(np.where(obs_valid[i], qobs[i], np.nan), members[m, i], date_index)
            vals.append(float(nse(obs_da, sim_da)))
        out.append(float(np.nanmedian(np.asarray(vals, dtype=np.float64))))
    return out


def run_reproduction(
    *,
    model_key: str,
    model_name: str,
    run_fold: Callable[..., object],
    basins: Sequence[str],
    basins_source: str,
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    seeds: Sequence[int],
    runner_kwargs: Mapping[str, object],
    published_single_median_nse: float | None,
    published_ensemble_median_nse: float | None,
    published_source: str,
    tolerance: float,
    output_dir: str | Path,
    results_filename: str = RESULTS_FILENAME,
    metrics_fn: MetricsFn = calculate_all_metrics,
) -> dict[str, object]:
    """Train with `run_fold`, score, compare with the published value, and write the results JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    t0 = time.time()
    result = run_fold(
        train_basins=list(basins),
        test_basins=list(basins),
        train_period=train_period,
        eval_period=eval_period,
        forcings=list(forcings),
        data_dir=data_dir,
        seeds=seeds,
        return_members=True,
        **dict(runner_kwargs),
    )
    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError("run_fold(return_members=True) must return (mean, members).")
    qsim_mean, qsim_members = (np.asarray(result[0], dtype=np.float32), np.asarray(result[1], dtype=np.float32))
    train_seconds = time.time() - t0

    obs_slice = load_camels_slice(
        basins=list(basins), period=eval_period, forcings=list(forcings), data_dir=data_dir, spinup_days=0
    )
    qobs = np.asarray(obs_slice["target_qobs"], dtype=np.float32)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(obs_slice["target_dates"])))
    if qsim_mean.shape != qobs.shape:
        raise ValueError(f"Simulated hydrograph {qsim_mean.shape} does not match observations {qobs.shape}.")

    summary = score_hydrograph(qsim_mean, qobs, dates, basins, metrics_fn=metrics_fn)
    member_medians = member_median_nse(qsim_members, qobs, dates)

    reference_used: str | None
    published: float | None
    if len(seeds) > 1 and published_ensemble_median_nse is not None:
        reference_used, published = "ensemble", float(published_ensemble_median_nse)
    elif published_single_median_nse is not None:
        reference_used, published = "single", float(published_single_median_nse)
    else:
        reference_used, published = None, None
    if published is None:
        status = "COMPLETED_NO_PUBLISHED_REFERENCE"
        abs_diff = None
    else:
        abs_diff = abs(float(summary["median_nse"]) - published)
        status = "PASS" if abs_diff <= float(tolerance) else "FAIL"

    hydrograph_path = out_dir / HYDROGRAPH_FILENAME
    np.savez_compressed(
        hydrograph_path,
        qsim=qsim_mean,
        qsim_members=qsim_members,
        basins=np.asarray(list(basins), dtype="<U8"),
        dates=np.asarray(dates.strftime("%Y-%m-%d").tolist(), dtype="<U10"),
        seeds=np.asarray(seeds, dtype=np.int64),
    )

    def _rel_path_str(p: str | Path) -> str:
        # Never record absolute (machine-specific) paths: repo-relative inside the repo,
        # otherwise only the file name.
        resolved = Path(p).resolve()
        repo_resolved = REPO_ROOT.resolve()
        if resolved.is_relative_to(repo_resolved):
            return resolved.relative_to(repo_resolved).as_posix()
        return resolved.name

    results: dict[str, object] = {
        "status": status,
        "model_key": model_key,
        "model_name": model_name,
        "provenance": (
            f"Trained in-process via models.{model_key}.runner.run_fold on the author protocol; "
            "scored with metrics.metrics.calculate_all_metrics; no precomputed hydrographs, checkpoints, "
            "or metric files were read."
        ),
        "protocol": {
            "basins_source": _rel_path_str(basins_source),
            "n_basins": int(len(basins)),
            "train_period": list(train_period),
            "eval_period": list(eval_period),
            "forcings": list(forcings),
            "seeds": list(seeds),
            "runner_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in dict(runner_kwargs).items()},
            "data_dir": _rel_path_str(data_dir),
        },
        "published": {
            "single_median_nse": published_single_median_nse,
            "ensemble_median_nse": published_ensemble_median_nse,
            "source": published_source,
        },
        "reference_used": reference_used,
        "published_median_nse": published,
        "tolerance": float(tolerance),
        "abs_diff_median_nse": abs_diff,
        "ours_median_nse": float(summary["median_nse"]),
        "ours_mean_nse": float(summary["mean_nse"]),
        "ours_median_kge": float(summary["median_kge"]),
        "ours_nse_le_zero_count": int(summary["nse_le_zero_count"]),
        "ours_n_scored_basins": int(summary["n_scored_basins"]),
        "unscored_basins": list(summary["unscored_basins"]),
        "pred_nan_when_obs_valid": int(summary["pred_nan_when_obs_valid"]),
        "single_seed_median_nse_per_member": member_medians,
        "single_seed_median_nse_mean": float(np.mean(member_medians)),
        "single_seed_median_nse_std": float(np.std(member_medians)),
        "hydrograph_path": _rel_path_str(hydrograph_path),
        "train_and_predict_seconds": float(train_seconds),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = out_dir / results_filename
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[{model_key}] status={status} median NSE={results['ours_median_nse']:.4f} "
          f"(published {published}) -> {out_path}")
    return results


def build_arg_parser(
    description: str,
    default_basins_file: Path,
    default_forcings: Sequence[str],
    default_seeds: Sequence[int],
) -> argparse.ArgumentParser:
    """Common CLI for the family's reproduction scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default="verify",
        choices=["verify", "train_and_eval"],
        help="'verify' trains, scores, and exits non-zero unless the published median NSE is reproduced "
        "within tolerance; 'train_and_eval' trains and scores without enforcing the tolerance.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="CAMELS-US bundle or directory.")
    parser.add_argument("--basins-file", type=Path, default=Path(default_basins_file))
    parser.add_argument("--forcings", nargs="+", default=list(default_forcings))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(default_seeds))
    parser.add_argument("--epochs", type=int, default=None, help="Override the author epoch count.")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cuda if available).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write results (default: model dir).")
    parser.add_argument(
        "--write-tracked",
        action="store_true",
        help=f"Write the tracked {RESULTS_FILENAME} instead of the untracked default "
        f"{LOCAL_RESULTS_FILENAME}. Use only when deliberately regenerating the published Stage-1 record.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pipeline check only: first --smoke-basins basins, 1 epoch, 2 batches, tiny hidden size; "
        "results go to a temporary directory and never overwrite the Stage-1 JSON.",
    )
    parser.add_argument("--smoke-basins", type=int, default=4)
    parser.add_argument(
        "--runner-kwarg",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Extra keyword argument forwarded to run_fold (value parsed as JSON; repeatable), e.g. "
        "`--runner-kwarg n_batches_per_epoch=250` to cap the per-epoch budget on CPU. Recorded in the results JSON.",
    )
    return parser


def parse_runner_kwarg_items(items: Sequence[str]) -> dict[str, object]:
    """Parse repeated `KEY=JSON` command-line items into a keyword dictionary."""
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--runner-kwarg expects KEY=JSON, got {item!r}.")
        key, raw = item.split("=", 1)
        out[key.strip()] = json.loads(raw)
    return out


def run_cli(
    *,
    args: argparse.Namespace,
    model_key: str,
    model_name: str,
    model_dir: Path,
    run_fold: Callable[..., object],
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    published_single_median_nse: float | None,
    published_ensemble_median_nse: float | None,
    published_source: str,
    tolerance: float,
    smoke_kwargs: Mapping[str, object],
    metrics_fn: MetricsFn = calculate_all_metrics,
) -> int:
    """Shared `main` body: resolve the protocol, run the reproduction, and return an exit code."""
    basins = load_basin_list(args.basins_file)
    runner_kwargs: dict[str, object] = parse_runner_kwarg_items(args.runner_kwarg)
    if args.epochs is not None:
        runner_kwargs["epochs"] = int(args.epochs)
    if args.device is not None:
        runner_kwargs["device"] = str(args.device)

    if args.smoke:
        basins = basins[: max(1, int(args.smoke_basins))]
        runner_kwargs.update(dict(smoke_kwargs))
        runner_kwargs.setdefault("epochs", 1)
        runner_kwargs.setdefault("n_batches_per_epoch", 2)
        output_dir = args.output_dir if args.output_dir is not None else Path(tempfile.mkdtemp(prefix=f"{model_key}_smoke_"))
        results_filename = SMOKE_RESULTS_FILENAME
    else:
        output_dir = args.output_dir if args.output_dir is not None else model_dir
        results_filename = RESULTS_FILENAME if args.write_tracked else LOCAL_RESULTS_FILENAME

    results = run_reproduction(
        model_key=model_key,
        model_name=model_name,
        run_fold=run_fold,
        basins=basins,
        basins_source=str(args.basins_file),
        train_period=train_period,
        eval_period=eval_period,
        forcings=list(args.forcings),
        data_dir=args.data_dir,
        seeds=list(args.seeds),
        runner_kwargs=runner_kwargs,
        published_single_median_nse=published_single_median_nse,
        published_ensemble_median_nse=published_ensemble_median_nse,
        published_source=published_source,
        tolerance=tolerance,
        output_dir=output_dir,
        results_filename=results_filename,
        metrics_fn=metrics_fn,
    )
    if args.mode == "verify" and not args.smoke and results["status"] == "FAIL":
        print(f"[{model_key}] FAIL: |{results['ours_median_nse']:.4f} - {results['published_median_nse']}| > {tolerance}")
        return 1
    return 0
