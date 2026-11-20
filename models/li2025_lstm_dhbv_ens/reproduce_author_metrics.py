# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 Author Metric Reproduction Script: Li et al. (2025) (LSTM + dHBV 1.1p) Ensemble.

Paper:
  Li, B., Song, Y., Pan, M., Lawson, K., & Shen, C. (2025).
  Advancing spatial and temporal generalization of hydrologic models using
  differentiable and machine learning architectures.
  Hydrology and Earth System Sciences, 29, 6829-6861.
  https://doi.org/10.5194/hess-29-6829-2025

Zenodo Archive:
  https://doi.org/10.5281/zenodo.16895228

Published Benchmark Metrics (Split-3: WY1980-1995 train / WY1995-2008 test, 531 basins):
  1. (LSTM + dHBV)_seed^123 (60-member multi-forcing hybrid ensemble: 3 forcings x
     2 structures x 10 seeds):
     - Published Median NSE = 0.823 (Figure 1a / Section 3.1).
     - Recomputed from Zenodo 10.5281/zenodo.16895228 (all 4,749 days):
       Median NSE = 0.8225 (post-365d warmup = 0.8286), delta = -0.0005 (PASS).
     - Independent Split-3 StandaloneDHBV re-run hybrid: Median NSE = 0.8195,
       delta = -0.0035 (PASS).
  2. (LSTM + dHBV)^123 (6-member single-seed 3-forcing hybrid):
     - Published Median NSE = 0.818.
     - Recomputed from Zenodo archive: member[0] = 0.8184, 10-seed mean = 0.8176 +/- 0.0040;
       independent Split-3 re-run = 0.8165 (PASS).
  3. (LSTM + dHBV)^1_Daymet (2-member single-seed Daymet hybrid):
     - Published Median NSE = 0.790.
     - Recomputed from Zenodo archive: member[0] = 0.7895 (post-warmup = 0.7915),
       10-seed mean = 0.7871 +/- 0.0013; independent Split-3 re-run = 0.7872 (PASS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from metrics.evaluator import calculate_model_leaderboard_metrics
from models.li2025_lstm_dhbv_ens.runner import run_fold  # noqa: F401
from models.zarr_writer import build_hydrograph_dataset, write_hydrograph_zarr


MODEL_DIR = Path(__file__).resolve().parent
# Tracked, committed Stage-1 record. Only rewritten when --out explicitly points here.
TRACKED_RESULTS_PATH = MODEL_DIR / "stage1_author_reproduction_results.json"
# Default output for newly computed reports: untracked, so verification never mutates the record.
DEFAULT_OUT_PATH = MODEL_DIR / "stage1_author_reproduction_results.local.json"


def build_stage2_multi_model_ensemble(
    bucket_uri: str = "gs://camels-us-mip",
    upload: bool = False,
) -> dict[str, object]:
    """Score the Stage-2 (LSTM + dHBV 1.1p) multi-model ensemble t4_06.

    With ``upload=True`` (maintainers only, opt-in) t4_06.zarr is first rebuilt from t4_05.zarr and
    t4_24.zarr and written to the bucket; otherwise nothing is written.
    """
    uri_lstm = f"{bucket_uri.rstrip('/')}/hydrographs/gauged__multi_camels/t4_05.zarr"
    uri_dhbv = f"{bucket_uri.rstrip('/')}/hydrographs/gauged__multi_camels/t4_24.zarr"
    uri_out = f"{bucket_uri.rstrip('/')}/hydrographs/gauged__multi_camels/t4_06.zarr"

    if upload:
        g_lstm = zarr.open_group(uri_lstm, mode="r")
        g_dhbv = zarr.open_group(uri_dhbv, mode="r")
        basins = [str(b) for b in g_dhbv["basin"][:]]
        hucs = [str(h) for h in g_dhbv["huc_02"][:]]
        basin_to_huc02 = dict(zip(basins, hucs, strict=True))
        q_lstm = np.asarray(g_lstm["qsim"][:], dtype=np.float32)
        q_dhbv = np.asarray(g_dhbv["qsim"][:], dtype=np.float32)
        qsim_members = np.stack([q_lstm, q_dhbv], axis=0).astype(np.float32)
        qsim = (0.5 * (q_lstm + q_dhbv)).astype(np.float32)

        ds = build_hydrograph_dataset(
            qsim=qsim,
            basins=basins,
            basin_to_huc02=basin_to_huc02,
            model_id="t4_06",
            model_name="LSTM + dHBV 1.1p (Multi-Model Ensemble)",
            model_family="li2025_lstm_dhbv_ens",
            family_id="li2025_lstm_dhbv_ens",
            provenance_category="MIP Re-run",
            spatial_regime="gauged",
            forcing_regime="multi_camels",
            forcings_used=["daymet", "maurer", "nldas"],
            code_url="https://github.com/mhpi/generic_deltamodel",
            qsim_members=qsim_members,
            is_ensemble=True,
            num_members=2,
            leaderboard_id="gauged__multi_camels",
            data_url=uri_out,
        )
        write_hydrograph_zarr(ds, target_uri=uri_out, expected_basins=basins)

    return calculate_model_leaderboard_metrics(
        model_id="t4_06",
        leaderboard_id="gauged__multi_camels",
        bucket_uri=bucket_uri,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify or evaluate Li et al. (2025) (LSTM + dHBV 1.1p) Multi-Model Ensemble."
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "evaluate_stage2"],
        default="verify",
        help="Verify stored Stage-1 reproduction metrics or evaluate the Stage-2 ensemble.",
    )
    parser.add_argument(
        "--verify-gcs",
        action="store_true",
        help="Re-score the canonical Stage-2 Zarr store t4_06 directly from GCS.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Maintainers only: rebuild t4_06.zarr from t4_05.zarr + t4_24.zarr and WRITE it to the "
        "bucket before scoring. Off by default; without it this script never writes to GCS.",
    )
    parser.add_argument("--bucket-uri", type=str, default="gs://camels-us-mip")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Where to write a newly computed report (default: untracked *.local.json). The tracked "
        "stage1_author_reproduction_results.json is only overwritten if --out points at it.",
    )
    args = parser.parse_args()

    tolerance = 0.03

    if args.mode == "verify" and not args.verify_gcs:
        if not TRACKED_RESULTS_PATH.is_file():
            raise FileNotFoundError(f"Missing reproduction report: {TRACKED_RESULTS_PATH}")
        report = json.loads(TRACKED_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        published_median_nse = 0.823
        metrics_out = build_stage2_multi_model_ensemble(
            bucket_uri=args.bucket_uri,
            upload=args.upload,
        )
        reproduced = float(metrics_out["median_nse"])
        delta = reproduced - published_median_nse
        status = "PASS" if abs(delta) <= tolerance else "FAIL"
        report = {
            "model_dir": "li2025_lstm_dhbv_ens",
            "paper": "Li et al. (2025) HESS 29, 6829-6861 - (LSTM + dHBV 1.1p) Multi-Model Ensemble",
            "paper_url": "https://doi.org/10.5194/hess-29-6829-2025",
            "zenodo_doi": "10.5281/zenodo.16895228",
            "verification_source": f"{args.bucket_uri}/hydrographs/gauged__multi_camels/t4_06.zarr",
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
