# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 reproduction script for ERA-Physics-Guided-Gauged (Node 187; category iv).

Contributed by this study via the Evolving Research Agent (ERA) tree search on the Newman
training window; there is no external published reference to compare against.
"""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.era_gauged_physics_guided.runner import run_fold
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
)

MODEL_KEY = "era_gauged_physics_guided"
MODEL_NAME = "ERA-Physics-Guided-Gauged"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END),
    "eval_period": (CANONICAL_EVAL_START, CANONICAL_EVAL_END),
    "forcings": ["daymet", "maurer", "nldas"],
    "seeds": (100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
}
PUBLISHED_SINGLE_MEDIAN_NSE = None
PUBLISHED_ENSEMBLE_MEDIAN_NSE = None
PUBLISHED_SOURCE = "none (category iv ERA model contributed by this study)"
TOLERANCE = 0.05
SMOKE_KWARGS = {
    "hidden_size": 16,
    "embed_dim": 8,
    "seq_len": 30,
    "spinup_days": 30,
    "warmup_len": 10,
    "samples_per_basin": 4,
    "eval_batch_size": 32,
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
