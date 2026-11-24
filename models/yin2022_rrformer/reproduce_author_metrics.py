# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 author-protocol reproduction for Yin et al. (2022, J. Hydrol.) RR-Former.

Protocol matching Yin et al. (2022, J. Hydrol. 609, 127781, Tables 5-6) and
`github.com/iThronne/RR-Former`:
  - All 671 CAMELS-US basins present in the canonical bundle (`06775500` & `06846500`
    absent from `camels_671_1980_2014.npz`), Daymet 5 dynamic channels
    (`prcp, srad, tmax, tmin, vp`), `use_static=False`.
  - Train `1980-10-01..1995-09-30`, Val `1995-10-01..2000-09-30`, Test `2000-10-01..2014-09-30`.
  - Window: `past_len=15`, `pred_len=7`, `use_future_fea=True` (`src_len=22`, `tgt_len=22`).
  - Zero-shortcut training: `epochs=200`, `finetune_epochs=200`, `batch_size=512`,
    `n_batches_per_epoch=None`, global z-score pretraining with `NSELoss`, per-basin
    z-score full-network fine-tuning with `MSELoss`.
  - Evaluates Day-1 ahead NSE (`nse1`) and all 7 lead-time curves (Days 1..7 medians
    and means plus 7-day mean `test_nse_mean`) on both upstream-aligned windows
    (`2000-10-16..2014-09-24`) and full eval windows (`2000-10-01..2014-09-24`).
  - Hard Pass Gate (Tables 5-6 Day-1 headline metric): `|median(nse1) - 0.8265| <= 0.01`.
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
PUBLISHED_DAY1_MEAN_NSE = 0.7904
PUBLISHED_PER_LEAD_MEDIAN_NSE: tuple[float, ...] = (
    0.8265,
    0.7832,
    0.7674,
    0.7582,
    0.7484,
    0.7423,
    0.7282,
)
PUBLISHED_PER_LEAD_MEAN_NSE: tuple[float, ...] = (
    0.7904,
    0.7421,
    0.7241,
    0.7107,
    0.7003,
    0.6901,
    0.6734,
)
PUBLISHED_SOURCE = (
    "Yin et al. (2022), J. Hydrol. 609, 127781, Section 4.1 & Tables 5-6 "
    "(Task 1 RR-Former 673 basins, Daymet, 2000-2014 eval; Day-1 nse1 = 0.8265 median / "
    "0.7904 mean; Days 1-7 med NSE = [0.8265, 0.7832, 0.7674, 0.7582, 0.7484, 0.7423, 0.7282]; "
    "Days 1-7 mean NSE = [0.7904, 0.7421, 0.7241, 0.7107, 0.7003, 0.6901, 0.6734]; "
    "Table 6 Day-1 ablations: RR-Former1 no-pretrain = 0.7756, RR-Former2 no-finetune = 0.7598)"
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


def _lead_nse_per_basin(
    pred_3d: np.ndarray,
    qobs_2d: np.ndarray,
    dates: pd.DatetimeIndex,
    start_offset: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Compute `(n_basins, pred_len)` NSE matrix and per-basin 7-day mean NSE."""
    pred_sub = pred_3d[:, start_offset:, :]
    n_basins, n_win, pred_len = pred_sub.shape
    nse_mat = np.full((n_basins, pred_len), np.nan, dtype=np.float64)
    for h in range(pred_len):
        obs_h = qobs_2d[:, start_offset + h : start_offset + h + n_win]
        sim_h = pred_sub[:, :, h]
        d_h = dates[start_offset + h : start_offset + h + n_win]
        summary_h = score_hydrograph(
            sim_h,
            obs_h,
            d_h,
            [str(i) for i in range(n_basins)],
            metrics_fn=calculate_all_metrics,
        )
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
        out_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path(tempfile.mkdtemp(prefix=f"{MODEL_KEY}_smoke_"))
        )
        results_filename = SMOKE_RESULTS_FILENAME
    else:
        out_dir = args.output_dir if args.output_dir is not None else MODEL_DIR
        results_filename = RESULTS_FILENAME if args.write_tracked else LOCAL_RESULTS_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    past_len = int(runner_kwargs["past_len"])
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
        basins=basins,
        period=AUTHOR_PROTOCOL["eval_period"],
        forcings=list(args.forcings),
        data_dir=args.data_dir,
        spinup_days=0,
    )
    qobs = np.asarray(obs_slice["target_qobs"], dtype=np.float32)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(obs_slice["target_dates"])))

    n_win = qsim_mean_3d.shape[1]
    upstream_off = past_len if n_win > past_len + 10 else 0

    day1_summary = score_hydrograph(
        qsim_mean_3d[:, upstream_off:, 0],
        qobs[:, upstream_off:n_win],
        dates[upstream_off:n_win],
        basins,
        metrics_fn=calculate_all_metrics,
    )
    nse_mat, basin_lead_mean_nse = _lead_nse_per_basin(
        qsim_mean_3d, qobs, dates, start_offset=upstream_off
    )
    per_lead_medians = [float(np.nanmedian(nse_mat[:, h])) for h in range(nse_mat.shape[1])]
    per_lead_means = [float(np.nanmean(nse_mat[:, h])) for h in range(nse_mat.shape[1])]
    test_nse_mean_median = float(np.nanmedian(basin_lead_mean_nse))
    test_nse_mean_mean = float(np.nanmean(basin_lead_mean_nse))

    nse_mat_full, basin_lead_mean_nse_full = _lead_nse_per_basin(
        qsim_mean_3d, qobs, dates, start_offset=0
    )
    per_lead_full_medians = [
        float(np.nanmedian(nse_mat_full[:, h])) for h in range(nse_mat_full.shape[1])
    ]
    test_nse_mean_full_median = float(np.nanmedian(basin_lead_mean_nse_full))
    test_nse_mean_full_mean = float(np.nanmean(basin_lead_mean_nse_full))

    member_nse1: list[float] = []
    member_nse_mean: list[float] = []
    for m in range(qsim_members_4d.shape[0]):
        s1 = score_hydrograph(
            qsim_members_4d[m, :, upstream_off:, 0],
            qobs[:, upstream_off:n_win],
            dates[upstream_off:n_win],
            basins,
            metrics_fn=calculate_all_metrics,
        )
        member_nse1.append(float(s1["median_nse"]))
        _, m_means = _lead_nse_per_basin(qsim_members_4d[m], qobs, dates, start_offset=upstream_off)
        member_nse_mean.append(float(np.nanmedian(m_means)))

    nse1_median = float(day1_summary["median_nse"])
    nse1_mean = float(day1_summary["mean_nse"])
    abs_diff = abs(nse1_median - PUBLISHED_ENSEMBLE_MEDIAN_NSE)
    status = "PASS" if abs_diff <= TOLERANCE else "FAIL"

    lead_comparison: list[dict[str, float | int | bool]] = []
    for h in range(min(len(per_lead_medians), len(PUBLISHED_PER_LEAD_MEDIAN_NSE))):
        pub_med = PUBLISHED_PER_LEAD_MEDIAN_NSE[h]
        pub_mean = PUBLISHED_PER_LEAD_MEAN_NSE[h]
        delta_med = per_lead_medians[h] - pub_med
        delta_mean = per_lead_means[h] - pub_mean
        lead_comparison.append(
            {
                "lead_day": h + 1,
                "published_median_nse": pub_med,
                "ours_median_nse": per_lead_medians[h],
                "delta_median_nse": delta_med,
                "passes_tolerance": abs(delta_med) <= TOLERANCE,
                "published_mean_nse": pub_mean,
                "ours_mean_nse": per_lead_means[h],
                "delta_mean_nse": delta_mean,
            }
        )

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
            "(global z-score NSELoss pretraining + per-basin z-score full-network Adam MSE fine-tuning)."
        ),
        "protocol": {
            "basins_source": str(args.basins_file),
            "n_basins": len(basins),
            "train_period": list(AUTHOR_PROTOCOL["train_period"]),
            "val_period": list(AUTHOR_PROTOCOL["val_period"]),
            "eval_period": list(AUTHOR_PROTOCOL["eval_period"]),
            "upstream_eval_start_offset_days": upstream_off,
            "forcings": list(args.forcings),
            "seeds": list(args.seeds),
            "runner_kwargs": {k: v for k, v in runner_kwargs.items()},
        },
        "published": {
            "single_median_nse": PUBLISHED_SINGLE_MEDIAN_NSE,
            "ensemble_median_nse": PUBLISHED_ENSEMBLE_MEDIAN_NSE,
            "day1_mean_nse": PUBLISHED_DAY1_MEAN_NSE,
            "per_lead_median_nse": list(PUBLISHED_PER_LEAD_MEDIAN_NSE),
            "per_lead_mean_nse": list(PUBLISHED_PER_LEAD_MEAN_NSE),
            "source": PUBLISHED_SOURCE,
        },
        "reference_used": "ensemble" if len(args.seeds) > 1 else "single",
        "published_median_nse": PUBLISHED_ENSEMBLE_MEDIAN_NSE,
        "tolerance": TOLERANCE,
        "abs_diff_median_nse": abs_diff,
        "delta_med_nse": nse1_median - PUBLISHED_ENSEMBLE_MEDIAN_NSE,
        "ours_median_nse": nse1_median,
        "ours_mean_nse": nse1_mean,
        "nse1_median": nse1_median,
        "nse1_mean": nse1_mean,
        "test_nse_mean_median": test_nse_mean_median,
        "test_nse_mean_mean": test_nse_mean_mean,
        "test_nse_mean_full_period_median": test_nse_mean_full_median,
        "test_nse_mean_full_period_mean": test_nse_mean_full_mean,
        "nse_per_lead_median": per_lead_medians,
        "nse_per_lead_mean": per_lead_means,
        "nse_per_lead_full_period_median": per_lead_full_medians,
        "lead_day_comparison_vs_tables_5_6": lead_comparison,
        "ours_median_kge": float(day1_summary["median_kge"]),
        "ours_nse_le_zero_count": int(day1_summary["nse_le_zero_count"]),
        "ours_n_scored_basins": int(day1_summary["n_scored_basins"]),
        "unscored_basins": list(day1_summary["unscored_basins"]),
        "pred_nan_when_obs_valid": int(day1_summary["pred_nan_when_obs_valid"]),
        "single_seed_median_nse_per_member": member_nse1,
        "single_seed_median_nse_mean": float(np.mean(member_nse1)),
        "single_seed_median_nse_std": float(np.std(member_nse1)),
        "single_seed_7d_mean_nse_per_member": member_nse_mean,
        "train_and_predict_seconds": float(train_seconds),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = out_dir / results_filename
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{MODEL_KEY}] status={status} nse1_med={nse1_median:.4f} (published 0.8265, "
        f"delta={nse1_median - PUBLISHED_ENSEMBLE_MEDIAN_NSE:+.4f}) "
        f"test_nse_mean={test_nse_mean_median:.4f} -> {out_path}"
    )
    if args.mode == "verify" and not args.smoke and status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
