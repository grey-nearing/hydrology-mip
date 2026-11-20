# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 Author Metric Reproduction Script: Song et al. (2024) dHBV 1.1p (dMG).

Paper:
  Song, Y., Knoben, W. J. M., Clark, M. P., Bindas, T., & Shen, C. (2024).
  When ancient numerical demons meet physics-informed machine learning:
  adjoint-based gradients for implicit differentiable modeling.
  Hydrology and Earth System Sciences, 28(13), 3051-3077.
  https://doi.org/10.5194/hess-28-3051-2024

Published Benchmark Metrics:
  1. Author Split-3 Protocol (WY1980-1995 train: 1980-10-01 to 1995-09-30;
     WY1995-2010 test: 1995-10-01 to 2010-09-30):
     - Published dHBV 1.1p (dynamic beta, K0, gamma) single-seed Median NSE = 0.750
       (published dHBV 1.0 static Median NSE = 0.730).
     - Clean-room reproduction (1-seed mean on 531 basins Maurer / 671 basins Daymet):
       Median NSE = 0.733 (0.7333 on 531 Maurer; 0.7330 on 671 Daymet; 0.7407 on 531 Daymet;
       2-seed ensemble = 0.7460 Maurer / 0.7556 Daymet), delta = -0.017 (PASS).
  2. Split-1 Verification vs. Released MHPI Checkpoint (WY1999-2008 train / WY1989-1999 test,
     531 CAMELS-US basins, Daymet forcing):
     - Released MHPI generic_deltamodel checkpoint Median NSE = 0.7574 (10-seed ensemble)
       / 0.7548 (single seed).
     - Clean-room 5-seed reproduction (t3_18.zarr): Median NSE = 0.7589 (ensemble)
       / 0.7537 +/- 0.0011 (single-member mean across 5 seeds), PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

from metrics.evaluator import calculate_model_leaderboard_metrics
from metrics.metrics import calculate_all_metrics
from models.camels_data import DEFAULT_DATA_DIR, load_camels_slice
from models.run_model import load_canonical_basins
from models.song_dhbv11p.runner import run_fold
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
)

MODEL_DIR = Path(__file__).resolve().parent
# Tracked, committed Stage-1 record. Only rewritten when --out explicitly points here.
TRACKED_RESULTS_PATH = MODEL_DIR / "stage1_author_reproduction_results.json"
# Default output for new runs: untracked, so reproducing never mutates the committed record.
DEFAULT_OUT_PATH = MODEL_DIR / "stage1_author_reproduction_results.local.json"

# Song et al. (2024) Split-3 author protocol windows (eval end depends on forcing coverage).
SPLIT3_TRAIN = ("1980-10-01", "1995-09-30")
SPLIT3_EVAL_START = "1995-10-01"
SPLIT3_EVAL_END_MAURER = "2008-09-30"
SPLIT3_EVAL_END_DAYMET = "2010-09-30"
# First day of the canonical CAMELS bundle (camels_671_1980_2014.npz).
BUNDLE_FIRST_DAY = "1980-01-01"
# Spin-up that fits inside the bundle: 1980-01-01 .. 1980-09-30 (274 days) instead of the runner's
# default 365, which would reach back to 1979-10-02 and fail the bundle coverage check.
SPLIT3_SPINUP_DAYS = int((pd.Timestamp(SPLIT3_TRAIN[0]) - pd.Timestamp(BUNDLE_FIRST_DAY)).days)
SPLIT1_SPINUP_DAYS = 365


def _evaluate_predictions(
    qsim: np.ndarray,
    qobs: np.ndarray,
    dates: pd.DatetimeIndex,
    basins: list[str],
) -> Dict[str, object]:
    """Evaluate (N, T) predictions against (N, T) observations using canonical metrics."""
    obs_valid = ~np.isnan(qobs) & (qobs >= 0.0)
    pred_nan_when_obs_valid = int(np.sum(np.isnan(qsim) & obs_valid))
    if pred_nan_when_obs_valid > 0:
        raise ValueError(
            f"Predictions contain {pred_nan_when_obs_valid} NaNs on valid observation days."
        )

    nses: list[float] = []
    kges: list[float] = []
    fhvs: list[float] = []
    for i, b in enumerate(basins):
        valid_i = obs_valid[i]
        if int(np.sum(valid_i)) < 30:
            continue
        da_obs = xr.DataArray(
            np.where(valid_i, qobs[i], np.nan).astype(np.float64),
            coords={"date": dates},
            dims=["date"],
            attrs={"basin_id": b},
        )
        da_sim = xr.DataArray(
            qsim[i].astype(np.float64),
            coords={"date": dates},
            dims=["date"],
            attrs={"basin_id": b},
        )
        res = calculate_all_metrics(da_obs, da_sim, resolution="1D", datetime_coord="date")
        nse_val = float(res["NSE"])
        kge_val = float(res["KGE"])
        fhv_val = float(res["FHV"])
        if np.isfinite(nse_val):
            nses.append(nse_val)
        if np.isfinite(kge_val):
            kges.append(kge_val)
        if np.isfinite(fhv_val):
            fhvs.append(abs(fhv_val))

    return {
        "median_nse": float(np.median(nses)),
        "mean_nse": float(np.mean(nses)),
        "median_kge": float(np.median(kges)),
        "median_abs_fhv": float(np.median(fhvs)),
        "num_evaluated_basins": len(nses),
        "pred_nan_when_obs_valid": pred_nan_when_obs_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce or verify Song et al. (2024) dHBV 1.1p (dMG) author metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "train_and_eval"],
        default="verify",
        help="Verify stored reproduction metrics or train and evaluate from scratch.",
    )
    parser.add_argument(
        "--split",
        choices=["split3", "split1"],
        default="split3",
        help="Evaluation split: 'split3' (Song et al. 2024 author split WY1980-1995 / WY1995-2010) "
        "or 'split1' (CAMELS-US MIP benchmark split WY1999-2008 / WY1989-1999).",
    )
    parser.add_argument(
        "--forcing",
        choices=["maurer", "daymet"],
        default="maurer",
        help="Meteorological forcing product for from-scratch training.",
    )
    parser.add_argument(
        "--verify-gcs",
        action="store_true",
        help="Re-score the canonical Split-1 Zarr store t3_18 directly from GCS.",
    )
    parser.add_argument("--bucket-uri", type=str, default="gs://camels-us-mip")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-batches-per-epoch", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[111, 222],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Where to write a newly computed report (default: untracked *.local.json). The tracked "
        "stage1_author_reproduction_results.json is only overwritten if --out points at it.",
    )
    args = parser.parse_args()

    tolerance = 0.03

    if args.mode == "verify":
        if args.verify_gcs:
            published_median_nse = 0.757
            metrics_out = calculate_model_leaderboard_metrics(
                model_id="t3_18",
                leaderboard_id="gauged__single_camels",
                bucket_uri=args.bucket_uri,
            )
            reproduced = float(metrics_out["median_nse"])
            delta = reproduced - published_median_nse
            status = "PASS" if abs(delta) <= tolerance else "FAIL"
            report = {
                "model_dir": "song_dhbv11p",
                "paper": "Song et al. (2024) HESS 28, 3051-3077 - dHBV 1.1p / dMG (Dynamic Parameters)",
                "paper_url": "https://doi.org/10.5194/hess-28-3051-2024",
                "verification_source": f"{args.bucket_uri}/hydrographs/gauged__single_camels/t3_18.zarr",
                "published_median_nse": published_median_nse,
                "reproduced_median_nse": round(reproduced, 4),
                "reproduced_mean_nse": round(float(metrics_out["mean_nse"]), 4),
                "reproduced_median_kge": round(float(metrics_out["median_kge"]), 4),
                "reproduced_median_abs_fhv": round(float(metrics_out["median_abs_fhv"]), 4),
                "delta_median_nse": round(delta, 4),
                "tolerance": tolerance,
                "num_evaluated_basins": int(metrics_out["num_evaluated_basins"]),
                "pred_nan_when_obs_valid": int(metrics_out["pred_nan_when_obs_valid"]),
                "status": status,
            }
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            if not TRACKED_RESULTS_PATH.is_file():
                raise FileNotFoundError(f"Missing reproduction report: {TRACKED_RESULTS_PATH}")
            report = json.loads(TRACKED_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        basins = load_canonical_basins()
        if args.split == "split3":
            train_period = SPLIT3_TRAIN
            # Daymet/Maurer in standard CAMELS-US v1.2 extends through 2014-12-31 / 2008-12-31
            eval_end = SPLIT3_EVAL_END_MAURER if args.forcing == "maurer" else SPLIT3_EVAL_END_DAYMET
            eval_period = (SPLIT3_EVAL_START, eval_end)
            spinup_days = SPLIT3_SPINUP_DAYS
            published_median_nse = 0.750
        else:
            train_period = (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END)
            eval_period = (CANONICAL_EVAL_START, CANONICAL_EVAL_END)
            spinup_days = SPLIT1_SPINUP_DAYS
            published_median_nse = 0.757

        extra_kw = {}
        if args.n_batches_per_epoch is not None:
            extra_kw["n_batches_per_epoch"] = args.n_batches_per_epoch
        qsim = run_fold(
            train_basins=basins,
            test_basins=basins,
            train_period=train_period,
            eval_period=eval_period,
            forcings=[args.forcing],
            data_dir=args.data_dir,
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            spinup_days=spinup_days,
            **extra_kw,
        )
        eval_slice = load_camels_slice(
            basins=basins,
            period=eval_period,
            forcings=[args.forcing],
            data_dir=args.data_dir,
            spinup_days=0,
            require_valid_targets=True,
        )
        eval_dates = pd.DatetimeIndex(eval_slice["target_dates"])
        metrics_out = _evaluate_predictions(
            qsim=qsim,
            qobs=eval_slice["target_qobs"],
            dates=eval_dates,
            basins=basins,
        )
        reproduced = float(metrics_out["median_nse"])
        delta = reproduced - published_median_nse
        status = "PASS" if reproduced >= published_median_nse - tolerance else "FAIL"
        report = {
            "model_dir": "song_dhbv11p",
            "paper": "Song et al. (2024) HESS 28, 3051-3077 - dHBV 1.1p / dMG (Dynamic Parameters)",
            "paper_url": "https://doi.org/10.5194/hess-28-3051-2024",
            "provenance": "from_scratch_train_and_eval",
            "protocol": {
                "spatial_regime": "gauged",
                "num_basins": len(basins),
                "train_period": f"{train_period[0]}/{train_period[1]}",
                "eval_period": f"{eval_period[0]}/{eval_period[1]}",
                "forcings": [args.forcing],
                "epochs": int(args.epochs),
                "n_batches_per_epoch": args.n_batches_per_epoch,
                "seeds": list(args.seeds),
                "spinup_days": spinup_days,
            },
            "verification_source": "from_scratch_train_and_eval",
            "published_median_nse": published_median_nse,
            "reproduced_median_nse": round(reproduced, 4),
            "reproduced_mean_nse": round(float(metrics_out["mean_nse"]), 4),
            "reproduced_median_kge": round(float(metrics_out["median_kge"]), 4),
            "reproduced_median_abs_fhv": round(float(metrics_out["median_abs_fhv"]), 4),
            "delta_median_nse": round(delta, 4),
            "tolerance": tolerance,
            "num_evaluated_basins": int(metrics_out["num_evaluated_basins"]),
            "pred_nan_when_obs_valid": int(metrics_out["pred_nan_when_obs_valid"]),
            "status": status,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
