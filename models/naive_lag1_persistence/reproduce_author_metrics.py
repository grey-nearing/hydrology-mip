# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 reproduction script for Naive Lag-1 Persistence lower benchmark."""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli
from models.naive_lag1_persistence.runner import run_fold

MODEL_KEY = "naive_lag1_persistence"
MODEL_NAME = "Naive Lag-1 Persistence (Q(t-1), Lower Benchmark)"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": ("1999-10-01", "2008-09-30"),
    "eval_period": ("1989-10-01", "1999-09-30"),
    "forcings": ["daymet"],
    "seeds": (0,),
}
PUBLISHED_SINGLE_MEDIAN_NSE = None
PUBLISHED_ENSEMBLE_MEDIAN_NSE = None
PUBLISHED_SOURCE = "none (zero-parameter lag-1 persistence lower benchmark contributed by HydroMIP)"
TOLERANCE = 0.01
SMOKE_KWARGS = {
    "spinup_days": 7,
}


def main() -> int:
    parser = build_arg_parser(
        description=__doc__,
        default_basins_file=AUTHOR_PROTOCOL["basins_file"],
        default_forcings=AUTHOR_PROTOCOL["forcings"],
        default_seeds=AUTHOR_PROTOCOL["seeds"],
    )
    args = parser.parse_args()
    return run_cli(
        args=args,
        model_key=MODEL_KEY,
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        run_fold=run_fold,
        train_period=AUTHOR_PROTOCOL["train_period"],
        eval_period=AUTHOR_PROTOCOL["eval_period"],
        published_single_median_nse=PUBLISHED_SINGLE_MEDIAN_NSE,
        published_ensemble_median_nse=PUBLISHED_ENSEMBLE_MEDIAN_NSE,
        published_source=PUBLISHED_SOURCE,
        tolerance=TOLERANCE,
        smoke_kwargs=SMOKE_KWARGS,
        metrics_fn=calculate_all_metrics,
    )


if __name__ == "__main__":
    raise SystemExit(main())
