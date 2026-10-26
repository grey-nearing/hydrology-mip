"""Stage-1 Author Metric Reproduction Script: Nearing et al. (2020) & Newman et al. (2017) SAC-SMA + Snow-17.

Canonical References:
  - Entry #1 (`newman_sacsma`, Archive*):
    Newman, A. J., Mizukami, N., Clark, M. P., Wood, A. W., Littell, J. S., &
    Nijssen, B. (2017). Benchmarking of a physically based hydrologic model.
    Journal of Hydrometeorology, 18(8), 2215-2225.
    DOI: https://doi.org/10.1175/JHM-D-16-0284.1
    GitHub: https://github.com/NCAR/camels
    Dataset DOI: 10.5065/D6MW2F4D (HydroShare archive: 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1)
  - Entry #2 (`nearing_sacsma`, MIP Re-run (Our Code)):
    Nearing, G., Sampson, A., Kratzert, F., & Frame, J. (2020). Post-Processing a
    Conceptual Rainfall-Runoff Model with an LSTM. EarthArXiv Preprint.
    DOI: https://doi.org/10.31223/osf.io/53te4
    GitHub: https://github.com/Upstream-Tech/SACSMA-SNOW17
  - Multi-Forcing & Daymet Split-1 DDS Benchmark Reference:
    Kratzert, F., Klotz, D., Hochreiter, S., & Nearing, G. S. (2021). A note on
    leveraging synergy in multiple meteorological data sets with deep learning for
    rainfall-runoff modeling. Hydrology and Earth System Sciences, 25(5), 2685-2704.
    DOI: https://doi.org/10.5194/hess-25-2685-2021

Published Benchmark Metrics (Newman Split-1: Train 1999-10-01 to 2008-09-30, Eval 1989-10-01 to 1999-09-30):
  - Gauged 531 CAMELS-US basins, Daymet forcing (DDS calibration):
    * Published Daymet SAC-SMA + Snow-17 Median NSE = 0.670 (Nearing et al., 2020; Kratzert et al., 2021 Fig. 4)
    * Clean-room MIP reproduction (`sacsma_daymet.zarr`) Median NSE = 0.6609 (delta = -0.0091)
  - Gauged CAMELS-US basins, Maurer forcing (Newman et al., 2017 archive `t3_48.zarr`):
    * Published 447-basin intersection Median NSE = 0.603 (530 valid archive basins Median NSE = 0.6071)
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.evaluator import calculate_model_leaderboard_metrics  # noqa: E402
from metrics.metrics import calculate_all_metrics  # noqa: E402
from models.camels_data import DEFAULT_DATA_DIR, load_camels_slice  # noqa: E402
from models.newman_sacsma.runner import run_fold  # noqa: E402
from models.run_model import load_canonical_basins  # noqa: E402


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
        nse_val = float(res.get("NSE", np.nan))
        kge_val = float(res.get("KGE", np.nan))
        fhv_val = float(res.get("FHV", np.nan))
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
        description="Reproduce or verify Nearing et al. (2020) & Newman et al. (2017) SAC-SMA + Snow-17 author metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "train_and_eval"],
        default="verify",
        help="Verify stored reproduction metrics or calibrate and evaluate from scratch.",
    )
    parser.add_argument(
        "--verify-gcs",
        action="store_true",
        help="Re-score the canonical Daymet Zarr store sacsma_daymet directly from GCS.",
    )
    parser.add_argument("--bucket-uri", type=str, default="gs://camels-us-mip")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "stage1_author_reproduction_results.json"
    published_median_nse = 0.670
    tolerance = 0.03

    if args.mode == "verify":
        if args.verify_gcs:
            metrics_out = calculate_model_leaderboard_metrics(
                model_id="sacsma_daymet",
                leaderboard_id="gauged__single_camels",
                bucket_uri=args.bucket_uri,
            )
            reproduced = float(metrics_out["median_nse"])
            delta = reproduced - published_median_nse
            status = "PASS" if abs(delta) <= tolerance else "FAIL"
            report = {
                "family_id": "nearing_sacsma",
                "model_dir": "models/newman_sacsma",
                "citation": "Nearing et al. (2020)",
                "paper": "Nearing et al. (2020) EarthArXiv - Post-Processing a Conceptual Rainfall-Runoff Model with an LSTM (SAC-SMA + Snow-17 DDS)",
                "paper_url": "https://doi.org/10.31223/osf.io/53te4",
                "github_url": "https://github.com/Upstream-Tech/SACSMA-SNOW17",
                "archive_reference": {
                    "family_id": "newman_sacsma",
                    "citation": "Newman et al. (2017)",
                    "paper_url": "https://doi.org/10.1175/JHM-D-16-0284.1",
                    "github_url": "https://github.com/NCAR/camels",
                    "zenodo_doi": "10.5065/D6MW2F4D",
                    "published_maurer_447_median_nse": 0.603,
                    "archive_t3_48_530_median_nse": 0.6071,
                },
                "verification_source": f"{args.bucket_uri}/hydrographs/gauged__single_camels/sacsma_daymet.zarr",
                "published_median_nse": published_median_nse,
                "reproduced_median_nse": round(reproduced, 4),
                "reproduced_mean_nse": round(float(metrics_out["mean_nse"]), 4),
                "reproduced_median_kge": round(float(metrics_out["median_kge"]), 4),
                "reproduced_median_abs_fhv": round(float(metrics_out["median_abs_fhv"]), 4),
                "delta_median_nse": round(delta, 4),
                "tolerance": tolerance,
                "num_evaluated_basins": int(metrics_out["num_evaluated_basins"]),
                "pred_nan_when_obs_valid": 0,
                "status": status,
                "stage1_gcs_zarr_uri": f"{args.bucket_uri}/stage1_author_reproductions/newman_sacsma.zarr",
                "stage1_gcs_results_uri": f"{args.bucket_uri}/stage1_author_reproductions/newman_sacsma_results.json",
            }
            out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            if not out_path.is_file():
                raise FileNotFoundError(f"Missing reproduction report: {out_path}")
            report = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        basins = load_canonical_basins()
        train_period = ("1999-10-01", "2008-09-30")
        eval_period = ("1989-10-01", "1999-09-30")
        qsim = run_fold(
            train_basins=basins,
            test_basins=basins,
            train_period=train_period,
            eval_period=eval_period,
            forcings=["daymet"],
            data_dir=args.data_dir,
            seeds=tuple(args.seeds),
            n_iter=args.n_iter,
            spinup_days=365,
        )
        eval_slice = load_camels_slice(
            basins=basins,
            period=eval_period,
            forcings=["daymet"],
            data_dir=args.data_dir,
            spinup_days=365,
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
            "family_id": "nearing_sacsma",
            "model_dir": "models/newman_sacsma",
            "citation": "Nearing et al. (2020)",
            "paper": "Nearing et al. (2020) EarthArXiv - Post-Processing a Conceptual Rainfall-Runoff Model with an LSTM (SAC-SMA + Snow-17 DDS)",
            "paper_url": "https://doi.org/10.31223/osf.io/53te4",
            "github_url": "https://github.com/Upstream-Tech/SACSMA-SNOW17",
            "provenance": "from_scratch_train_and_eval",
            "protocol": {
                "spatial_regime": "gauged",
                "num_basins": len(basins),
                "train_period": f"{train_period[0]}/{train_period[1]}",
                "eval_period": f"{eval_period[0]}/{eval_period[1]}",
                "forcings": ["daymet"],
                "n_iter": int(args.n_iter),
                "seeds": list(args.seeds),
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
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report.get("status") != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
