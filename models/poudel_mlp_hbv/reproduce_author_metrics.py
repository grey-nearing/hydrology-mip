# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 Author Metric Reproduction Script: Poudel & Steinschneider (2026) MLP+16HBV.

Paper:
  Poudel, S., & Steinschneider, S. (2026). An argument for parsimony in
  differentiable hydrologic models. Hydrology and Earth System Sciences (HESS),
  30(15), 5173-5193. https://doi.org/10.5194/hess-30-5173-2026

Official Code Repository:
  https://github.com/snpoudel/diff-hydro

Author Protocol (Split-2C):
  - Basins: 531 CAMELS-US basins partitioned via an 80%/20% random split
    (random_state=42) into 425 in-sample training gauges and 106 held-out PUB
    test gauges (camels_test_basins.csv).
  - Forcing: Daymet daily meteorological forcing + 30 static catchment attributes.
  - Temporal Split:
    - Training: 1990-01-01 to 2005-12-31 (1095-day windows, 730-day spinup, 60-day stride)
    - Validation: 2006-01-01 to 2014-12-31 (early stopping on validation MSE)
    - Testing / Evaluation: 1980-01-01 to 1989-12-31
  - Published Benchmark Metrics (Poudel & Steinschneider, 2026, Figure 3 & Table 2):
    - OOS in Time (Gauged, 425 basins): Published Median NSE = 0.71 (0.710)
    - OOS in Space (PUB hold-out, 106 basins): Published Median NSE = 0.63 (0.630; 0.67 reference)
  - Stage-1 Confirmation Run Results (verified on Split-2C):
    - OOS in Time (Gauged, 425 basins): Reproduced Median NSE = 0.7177 (0.718, delta = +0.0077, PASS; WY1981-1989 = 0.7334)
    - OOS in Space (PUB, 106 basins): Reproduced Median NSE = 0.6539 (0.654, delta = +0.0239 vs 0.630 / -0.0161 vs 0.670, PASS; WY1981-1989 = 0.6609)
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
from models.poudel_mlp_hbv.runner import run_fold
from models.run_model import load_canonical_basins

PUBLISHED_GAUGED_MEDIAN_NSE = 0.710
PUBLISHED_PUB_MEDIAN_NSE = 0.630
PUBLISHED_PUB_REFERENCE_NSE = 0.670
TOLERANCE_NSE = 0.030

MODEL_DIR = Path(__file__).resolve().parent
# Tracked, committed Stage-1 record. Only rewritten when --out explicitly points here.
TRACKED_RESULTS_PATH = MODEL_DIR / "stage1_author_reproduction_results.json"
# Default output for new runs: untracked, so reproducing never mutates the committed record.
DEFAULT_OUT_PATH = MODEL_DIR / "stage1_author_reproduction_results.local.json"

# Poudel & Steinschneider (2026) Split-2C author protocol windows.
SPLIT2C_TRAIN = ("1990-01-01", "2005-12-31")
SPLIT2C_EVAL = ("1980-01-01", "1989-12-31")
# Training keeps the runner's 730-day spin-up (1988-01-02 .. 1989-12-31). The evaluation window starts
# on the first day of the CAMELS bundle (1980-01-01), so no forcing exists before it: evaluate with
# no prepended spin-up instead of reaching back to 1978-01-02 and failing the bundle coverage check.
SPLIT2C_EVAL_SPINUP_DAYS = 0


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


def _split2c_basins(basins_531: list[str]) -> tuple[list[str], list[str]]:
    """Return the 425 training and 106 PUB test basins from Poudel & Steinschneider (2026)."""
    rng = np.random.RandomState(42)
    indices = np.arange(len(basins_531))
    rng.shuffle(indices)
    n_test = 106
    test_set = {basins_531[i] for i in indices[:n_test]}
    train_basins = [b for b in basins_531 if b not in test_set]
    test_basins = [b for b in basins_531 if b in test_set]
    return train_basins, test_basins


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce or verify Poudel & Steinschneider (2026) Split-2C MLP+16HBV author metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "train_and_eval"],
        default="verify",
        help="Verify stored Split-2C reproduction metrics or train and evaluate from scratch.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-batches-per-epoch", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Where to write a newly computed report (default: untracked *.local.json). The tracked "
        "stage1_author_reproduction_results.json is only overwritten if --out points at it.",
    )
    args = parser.parse_args()


    if args.mode == "verify":
        if not TRACKED_RESULTS_PATH.is_file():
            raise FileNotFoundError(f"Missing reproduction report: {TRACKED_RESULTS_PATH}")
        report = json.loads(TRACKED_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        basins_531 = load_canonical_basins()
        train_basins, pub_basins = _split2c_basins(basins_531)
        train_period = SPLIT2C_TRAIN
        eval_period = SPLIT2C_EVAL

        extra_kw = {}
        if args.n_batches_per_epoch is not None:
            extra_kw["n_batches_per_epoch"] = args.n_batches_per_epoch

        qsim_all = run_fold(
            train_basins=train_basins,
            test_basins=basins_531,
            train_period=train_period,
            eval_period=eval_period,
            forcings=["daymet"],
            data_dir=args.data_dir,
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            eval_spinup_days=SPLIT2C_EVAL_SPINUP_DAYS,
            **extra_kw,
        )
        eval_slice = load_camels_slice(
            basins=basins_531,
            period=eval_period,
            forcings=["daymet"],
            data_dir=args.data_dir,
            spinup_days=0,
            require_valid_targets=True,
        )
        eval_dates = pd.DatetimeIndex(eval_slice["target_dates"])
        qobs_all = eval_slice["target_qobs"]

        train_idx = [basins_531.index(b) for b in train_basins]
        pub_idx = [basins_531.index(b) for b in pub_basins]

        gauged_metrics = _evaluate_predictions(
            qsim=qsim_all[train_idx],
            qobs=qobs_all[train_idx],
            dates=eval_dates,
            basins=train_basins,
        )
        pub_metrics = _evaluate_predictions(
            qsim=qsim_all[pub_idx],
            qobs=qobs_all[pub_idx],
            dates=eval_dates,
            basins=pub_basins,
        )

        reproduced_gauged = float(gauged_metrics["median_nse"])
        reproduced_pub = float(pub_metrics["median_nse"])
        delta_gauged = reproduced_gauged - PUBLISHED_GAUGED_MEDIAN_NSE
        delta_pub_063 = reproduced_pub - PUBLISHED_PUB_MEDIAN_NSE
        delta_pub_067 = reproduced_pub - PUBLISHED_PUB_REFERENCE_NSE

        pass_gauged = reproduced_gauged >= PUBLISHED_GAUGED_MEDIAN_NSE - TOLERANCE_NSE
        pass_pub = reproduced_pub >= PUBLISHED_PUB_MEDIAN_NSE - TOLERANCE_NSE
        status = "PASS" if (pass_gauged and pass_pub) else "FAIL"

        report = {
            "model_dir": "poudel_mlp_hbv",
            "paper": "Poudel, S., & Steinschneider, S. (2026). An argument for parsimony in differentiable hydrologic models. Hydrology and Earth System Sciences, 30(15), 5173-5193.",
            "citation": "Poudel & Steinschneider (2026)",
            "paper_url": "https://doi.org/10.5194/hess-30-5173-2026",
            "github_url": "https://github.com/snpoudel/diff-hydro",
            "provenance": "from_scratch_split2c_train_and_eval",
            "protocol": {
                "split_name": "Split-2C (Poudel & Steinschneider, 2026)",
                "spatial_regime": "split2c_gauged_425_and_pub_106",
                "num_train_basins": len(train_basins),
                "num_pub_test_basins": len(pub_basins),
                "num_total_basins": len(basins_531),
                "train_period": f"{train_period[0]}/{train_period[1]}",
                "eval_period": f"{eval_period[0]}/{eval_period[1]}",
                "forcings": ["daymet"],
                "epochs": int(args.epochs),
                "n_batches_per_epoch": args.n_batches_per_epoch,
                "seeds": list(args.seeds),
                "eval_spinup_days": SPLIT2C_EVAL_SPINUP_DAYS,
            },
            "verification_source": "from_scratch_split2c_train_and_eval",
            "published_median_nse": PUBLISHED_GAUGED_MEDIAN_NSE,
            "reproduced_median_nse": round(reproduced_gauged, 4),
            "reproduced_mean_nse": round(float(gauged_metrics["mean_nse"]), 4),
            "reproduced_median_kge": round(float(gauged_metrics["median_kge"]), 4),
            "reproduced_median_abs_fhv": round(float(gauged_metrics["median_abs_fhv"]), 4),
            "delta_median_nse": round(delta_gauged, 4),
            "published_pub_median_nse": PUBLISHED_PUB_MEDIAN_NSE,
            "published_pub_10fold_reference_nse": PUBLISHED_PUB_REFERENCE_NSE,
            "reproduced_pub_median_nse": round(reproduced_pub, 4),
            "reproduced_pub_mean_nse": round(float(pub_metrics["mean_nse"]), 4),
            "reproduced_pub_median_kge": round(float(pub_metrics["median_kge"]), 4),
            "reproduced_pub_median_abs_fhv": round(float(pub_metrics["median_abs_fhv"]), 4),
            "delta_pub_median_nse_vs_063": round(delta_pub_063, 4),
            "delta_pub_median_nse_vs_067": round(delta_pub_067, 4),
            "tolerance": TOLERANCE_NSE,
            "num_evaluated_basins": int(gauged_metrics["num_evaluated_basins"]),
            "num_evaluated_pub_basins": int(pub_metrics["num_evaluated_basins"]),
            "pred_nan_when_obs_valid": int(
                gauged_metrics["pred_nan_when_obs_valid"]
                + pub_metrics["pred_nan_when_obs_valid"]
            ),
            "status": status,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
