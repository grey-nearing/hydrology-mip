# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 author-protocol reproduction for Yin et al. (2022, J. Hydrol.) RR-Former.

Protocol matching Yin et al. (2022):
  - All 671 CAMELS-US basins (plus `06775500` & `06846500` if available), Daymet 5
    dynamic channels (`prcp, srad, tmax, tmin, vp`), `use_static=False`.
  - Train `1980-10-01..1995-09-30`, Val `1995-10-01..2000-09-30`, Test `2000-10-01..2014-09-30`.
  - Window: `past_len=15`, `pred_len=7`, `use_future_fea=True` (`src_len=22`, `tgt_len=22`).
  - Logs BOTH Day-1 `nse1` and 7-day mean `test_nse_mean`.
  - Hard Pass Gate: `|median(test_nse_mean) - 0.8265| <= 0.01`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from metrics.metrics import calculate_all_metrics
from models.camels_data import load_camels_slice
from models.kratzert_lstm.reproduction import (
    HYDROGRAPH_FILENAME,
    LOCAL_RESULTS_FILENAME,
    RESULTS_FILENAME,
    SMOKE_RESULTS_FILENAME,
    SPLITS_DIR,
    build_arg_parser,
    load_basin_list,
    parse_runner_kwarg_items,
    score_hydrograph,
)
from models.yin2022_rrformer.runner import DEFAULT_SEEDS, run_fold

MODEL_KEY = "yin2022_rrformer"
MODEL_NAME = "Yin et al. (2022) RR-Former"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_671.json",
    "train_period": ("1980-10-01", "1995-09-30"),
    "val_period": ("1995-10-01", "2000-09-30"),
    "eval_period": ("2000-10-01", "2014-09-30"),
    "forcings": ["daymet"],
    "seeds": DEFAULT_SEEDS,
}
PUBLISHED_SINGLE_MEDIAN_NSE = 0.8265
PUBLISHED_ENSEMBLE_MEDIAN_NSE = 0.8265
PUBLISHED_SOURCE = (
    "Yin et al. (2022), J. Hydrol. 609, 127781, Section 4.1 (p. 6), Fig. 4a & Tables 5-6 "
    "(RR-Former 7-day mean NSE = 0.8265 median / 0.7904 mean, 673 basins, Daymet, 2000-2014 eval)"
)
TOLERANCE = 0.01
SMOKE_KWARGS = {
    "d_model": 16,
    "nhead": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dim_feedforward": 32,
    "past_len": 15,
    "pred_len": 7,
    "batch_size": 32,
    "finetune_epochs": 1,
}


def _lead_nse_per_basin(pred_3d: np.ndarray, qobs_2d: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, list[float]]:
    """Compute `(n_basins, pred_len)` NSE matrix and per-basin 7-day mean NSE."""
    n_basins, n_win, pred_len = pred_3d.shape
    nse_mat = np.full((n_basins, pred_len), np.nan, dtype=np.float64)
    for h in range(pred_len):
        obs_h = qobs_2d[:, h : h + n_win]
        sim_h = pred_3d[:, :, h]
        d_h = dates[h : h + n_win]
        summary_h = score_hydrograph(sim_h, obs_h, d_h, [str(i) for i in range(n_basins)], metrics_fn=calculate_all_metrics)
        for i in range(n_basins):
            b_res = summary_h["per_basin"].get(str(i))  # type: ignore[union-attr]
            if b_res is not None:
                nse_mat[i, h] = float(b_res["NSE"])
    per_basin_mean = [float(np.nanmean(nse_mat[i])) for i in range(n_basins)]
    return nse_mat, per_basin_mean


def main() -> int:
    parser = build_arg_parser(
        description=__doc__,
        default_basins_file=AUTHOR_PROTOCOL["basins_file"],
        default_forcings=AUTHOR_PROTOCOL["forcings"],
        default_seeds=AUTHOR_PROTOCOL["seeds"],
    )
    args = parser.parse_args()
    basins = load_basin_list(args.basins_file)
    runner_kwargs = parse_runner_kwarg_items(args.runner_kwarg)
    runner_kwargs.setdefault("past_len", 15)
    runner_kwargs.setdefault("pred_len", 7)
    runner_kwargs.setdefault("use_static", False)
    if args.epochs is not None:
        runner_kwargs["epochs"] = int(args.epochs)
    if args.device is not None:
        runner_kwargs["device"] = str(args.device)

    if args.smoke:
        basins = basins[: max(1, int(args.smoke_basins))]
        runner_kwargs.update(dict(SMOKE_KWARGS))
        runner_kwargs.setdefault("epochs", 1)
        runner_kwargs.setdefault("n_batches_per_epoch", 2)
        out_dir = args.output_dir if args.output_dir is not None else Path(tempfile.mkdtemp(prefix=f"{MODEL_KEY}_smoke_"))
        results_filename = SMOKE_RESULTS_FILENAME
    else:
        out_dir = args.output_dir if args.output_dir is not None else MODEL_DIR
        results_filename = RESULTS_FILENAME if args.write_tracked else LOCAL_RESULTS_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    qsim_mean_3d, qsim_members_4d = run_fold(
        train_basins=basins,
        test_basins=basins,
        train_period=AUTHOR_PROTOCOL["train_period"],
        eval_period=AUTHOR_PROTOCOL["eval_period"],
        val_period=AUTHOR_PROTOCOL["val_period"],
        forcings=list(args.forcings),
        data_dir=args.data_dir,
        seeds=list(args.seeds),
        return_members=True,
        return_lead_predictions=True,
        **runner_kwargs,
    )
    train_seconds = time.time() - t0

    obs_slice = load_camels_slice(
        basins=basins, period=AUTHOR_PROTOCOL["eval_period"], forcings=list(args.forcings), data_dir=args.data_dir, spinup_days=0
    )
    qobs = np.asarray(obs_slice["target_qobs"], dtype=np.float32)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(obs_slice["target_dates"])))

    n_win = qsim_mean_3d.shape[1]
    day1_summary = score_hydrograph(
        qsim_mean_3d[:, :, 0],
        qobs[:, :n_win],
        dates[:n_win],
        basins,
        metrics_fn=calculate_all_metrics,
    )
    nse_mat, basin_lead_mean_nse = _lead_nse_per_basin(qsim_mean_3d, qobs, dates)
    test_nse_mean_median = float(np.nanmedian(basin_lead_mean_nse))
    test_nse_mean_mean = float(np.nanmean(basin_lead_mean_nse))

    member_nse1: list[float] = []
    member_nse_mean: list[float] = []
    for m in range(qsim_members_4d.shape[0]):
        s1 = score_hydrograph(
            qsim_members_4d[m, :, :, 0],
            qobs[:, :n_win],
            dates[:n_win],
            basins,
            metrics_fn=calculate_all_metrics,
        )
        member_nse1.append(float(s1["median_nse"]))
        _, m_means = _lead_nse_per_basin(qsim_members_4d[m], qobs, dates)
        member_nse_mean.append(float(np.nanmedian(m_means)))

    abs_diff = abs(test_nse_mean_median - PUBLISHED_ENSEMBLE_MEDIAN_NSE)
    status = "PASS" if abs_diff <= TOLERANCE else "FAIL"

    hydrograph_path = out_dir / HYDROGRAPH_FILENAME
    np.savez_compressed(
        hydrograph_path,
        qsim=qsim_mean_3d[:, :, 0],
        qsim_leads=qsim_mean_3d,
        qsim_members=qsim_members_4d[:, :, :, 0],
        basins=np.asarray(basins, dtype="<U8"),
        dates=np.asarray(dates[:n_win].strftime("%Y-%m-%d").tolist(), dtype="<U10"),
        seeds=np.asarray(list(args.seeds), dtype=np.int64),
    )

    results = {
        "status": status,
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "provenance": (
            "Trained in-process via models.yin2022_rrformer.runner.run_fold on the author protocol "
            "(global NSELoss pretraining + full-network per-basin Adam MSE fine-tuning)."
        ),
        "protocol": {
            "basins_source": str(args.basins_file),
            "n_basins": len(basins),
            "train_period": list(AUTHOR_PROTOCOL["train_period"]),
            "val_period": list(AUTHOR_PROTOCOL["val_period"]),
            "eval_period": list(AUTHOR_PROTOCOL["eval_period"]),
            "forcings": list(args.forcings),
            "seeds": list(args.seeds),
            "runner_kwargs": {k: v for k, v in runner_kwargs.items()},
        },
        "published": {
            "single_median_nse": PUBLISHED_SINGLE_MEDIAN_NSE,
            "ensemble_median_nse": PUBLISHED_ENSEMBLE_MEDIAN_NSE,
            "source": PUBLISHED_SOURCE,
        },
        "reference_used": "ensemble",
        "published_median_nse": PUBLISHED_ENSEMBLE_MEDIAN_NSE,
        "tolerance": TOLERANCE,
        "abs_diff_median_nse": abs_diff,
        "ours_median_nse": test_nse_mean_median,
        "ours_mean_nse": test_nse_mean_mean,
        "test_nse_mean_median": test_nse_mean_median,
        "test_nse_mean_mean": test_nse_mean_mean,
        "nse1_median": float(day1_summary["median_nse"]),
        "nse1_mean": float(day1_summary["mean_nse"]),
        "nse_per_lead_median": [float(np.nanmedian(nse_mat[:, h])) for h in range(nse_mat.shape[1])],
        "ours_median_kge": float(day1_summary["median_kge"]),
        "ours_nse_le_zero_count": int(sum(1 for v in basin_lead_mean_nse if v <= 0.0)),
        "ours_n_scored_basins": int(day1_summary["n_scored_basins"]),
        "unscored_basins": list(day1_summary["unscored_basins"]),
        "pred_nan_when_obs_valid": int(day1_summary["pred_nan_when_obs_valid"]),
        "single_seed_median_nse_per_member": member_nse_mean,
        "single_seed_median_nse_mean": float(np.mean(member_nse_mean)),
        "single_seed_median_nse_std": float(np.std(member_nse_mean)),
        "single_seed_nse1_per_member": member_nse1,
        "train_and_predict_seconds": float(train_seconds),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = out_dir / results_filename
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{MODEL_KEY}] status={status} test_nse_mean={test_nse_mean_median:.4f} "
        f"nse1={results['nse1_median']:.4f} (published 0.8265) -> {out_path}"
    )
    if args.mode == "verify" and not args.smoke and status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
