# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 record for the Naive Lag-1 Persistence lower benchmark (no published reference).

Runs `models.naive_lag1_persistence.runner.run_fold` on the 531 canonical basins over the canonical
evaluation window (gauged protocol, lagged-streamflow regime) and scores the hydrograph with
`metrics.metrics.calculate_all_metrics` through the shared hermetic harness
(`models.kratzert_lstm.reproduction`). The model has no parameters, so the training period is
unused and there is nothing to compare against: the status is `COMPLETED_NO_PUBLISHED_REFERENCE`.

Usage (from the repository root)::

    python -m models.naive_lag1_persistence.reproduce_author_metrics --mode train_and_eval
    # add --write-tracked to regenerate the committed stage1_author_reproduction_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics.metrics import calculate_all_metrics
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli
from models.naive_lag1_persistence.runner import run_fold
from models.run_model import MODEL_REGISTRY
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
)

MODEL_KEY = "naive_lag1_persistence"
MODEL_NAME = "Naive Lag-1 Persistence (Q(t-1), Lower Benchmark)"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END),
    "eval_period": (CANONICAL_EVAL_START, CANONICAL_EVAL_END),
    "forcings": ["daymet"],
    "seeds": (0,),
}
PUBLISHED_SINGLE_MEDIAN_NSE = None
PUBLISHED_ENSEMBLE_MEDIAN_NSE = None
PUBLISHED_SOURCE = "none (zero-parameter lag-1 persistence lower benchmark contributed by HydrologyMIP)"
TOLERANCE = 0.01
SMOKE_KWARGS: dict[str, object] = {}


def run_fold_with_members(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
    """Adapter for the shared harness, which requests ``return_members=True``.

    The deterministic model has exactly one member.
    """
    if not kwargs.pop("return_members"):
        raise ValueError("The Stage-1 harness must request return_members=True.")
    qsim = np.asarray(run_fold(**kwargs), dtype=np.float32)  # type: ignore[arg-type]
    return qsim, qsim[np.newaxis, :, :]


def main() -> int:
    parser = build_arg_parser(
        description=__doc__,
        default_basins_file=AUTHOR_PROTOCOL["basins_file"],
        default_forcings=AUTHOR_PROTOCOL["forcings"],
        default_seeds=AUTHOR_PROTOCOL["seeds"],
    )
    args = parser.parse_args()
    # Forward the registry's explicit cold-start constant unless overridden, so the harness records
    # it in the results JSON exactly as run_model records it in the Zarr provenance.
    given = {item.split("=", 1)[0].strip() for item in args.runner_kwarg}
    for key, value in MODEL_REGISTRY[MODEL_KEY].default_kwargs.items():
        if key not in given:
            args.runner_kwarg.append(f"{key}={json.dumps(value)}")
    return run_cli(
        args=args,
        model_key=MODEL_KEY,
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        run_fold=run_fold_with_members,
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
