# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 Author Metric Reproduction Script: Feng et al. (2022) dHBV 1.0 (Split-3 Author Protocol).

Paper:
  Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable,
  regionalized process-based models with multiphysical outputs can approach
  state-of-the-art hydrologic prediction accuracy. Water Resources Research,
  58(10), e2022WR032404. https://doi.org/10.1029/2022WR032404
  GitHub: https://github.com/mhpi/hydroDL
  Zenodo: 10.5281/zenodo.7091334

Published Benchmark Metric (Split-3 Author Protocol: WY1980-1995 train / WY1995-2010 test, Maurer):
  - Single-seed dMG (no buffering) Median NSE = 0.715; dMG^c (with buffering) Median NSE = 0.722
  - Clean-room Split-3 reproduction: Single-seed Median NSE = 0.728; 5-seed Ensemble Median NSE = 0.734
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

from metrics.metrics import calculate_all_metrics
from models.camels_data import DEFAULT_DATA_DIR, load_camels_slice
from models.feng_dhbv10.runner import run_fold
from models.run_model import load_canonical_basins

MODEL_DIR = Path(__file__).resolve().parent
# Tracked, committed Stage-1 record. Only rewritten when --out explicitly points here.
TRACKED_RESULTS_PATH = MODEL_DIR / "stage1_author_reproduction_results.json"
# Default output for new runs: untracked, so reproducing never mutates the committed record.
DEFAULT_OUT_PATH = MODEL_DIR / "stage1_author_reproduction_results.local.json"

# Feng et al. (2022) Split-3 author protocol windows.
SPLIT3_TRAIN = ("1980-10-01", "1995-09-30")
SPLIT3_EVAL = ("1995-10-01", "2008-12-31")
# First day of the canonical CAMELS bundle (camels_671_1980_2014.npz).
BUNDLE_FIRST_DAY = "1980-01-01"
# Spin-up that fits inside the bundle: 1980-01-01 .. 1980-09-30 (274 days) instead of the runner's
# default 365, which would reach back to 1979-10-02 and fail the bundle coverage check.
SPLIT3_SPINUP_DAYS = int((pd.Timestamp(SPLIT3_TRAIN[0]) - pd.Timestamp(BUNDLE_FIRST_DAY)).days)


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
    for i in range(len(basins)):
        valid_i = obs_valid[i]
        if int(np.sum(valid_i)) < 30:
            continue
        da_obs = xr.DataArray(
            qobs[i].astype(np.float64),
            coords={"date": dates},
            dims=["date"],
        )
        da_sim = xr.DataArray(
            qsim[i].astype(np.float64),
            coords={"date": dates},
            dims=["date"],
        )
        m = calculate_all_metrics(da_obs, da_sim)
        if np.isfinite(m["NSE"]):
            nses.append(float(m["NSE"]))
        if np.isfinite(m["KGE"]):
            kges.append(float(m["KGE"]))
        if np.isfinite(m["FHV"]):
            fhvs.append(abs(float(m["FHV"])))

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
        description="Reproduce or verify Feng et al. (2022) dHBV 1.0 author metrics on Split-3."
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "train_and_eval"],
        default="verify",
        help="Verify stored reproduction metrics or train and evaluate from scratch on Split-3.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[111, 222, 333, 444, 555],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Where to write a newly computed report (default: untracked *.local.json). The tracked "
        "stage1_author_reproduction_results.json is only overwritten if --out points at it.",
    )
    args = parser.parse_args()

    published_median_nse = 0.715
    tolerance = 0.03

    if args.mode == "verify":
        if not TRACKED_RESULTS_PATH.is_file():
            raise FileNotFoundError(f"Missing reproduction report: {TRACKED_RESULTS_PATH}")
        report = json.loads(TRACKED_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        basins = load_canonical_basins()
        train_period = SPLIT3_TRAIN
        eval_period = SPLIT3_EVAL
        qsim = run_fold(
            train_basins=basins,
            test_basins=basins,
            train_period=train_period,
            eval_period=eval_period,
            forcings=["maurer"],
            data_dir=args.data_dir,
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            spinup_days=SPLIT3_SPINUP_DAYS,
        )
        eval_slice = load_camels_slice(
            basins=basins,
            period=eval_period,
            forcings=["maurer"],
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
        status = "PASS" if abs(delta) <= tolerance else "FAIL"
        report = {
            "model_dir": "feng_dhbv10",
            "paper": "Feng et al. (2022) WRR - dHBV 1.0 (Split-3 Author Protocol)",
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
            "spinup_days": SPLIT3_SPINUP_DAYS,
            "status": status,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
