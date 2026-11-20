# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 reproduction for the Rasiya Koya et al. (2023) Temporal Fusion Transformer.

Trains the TFT in-process with `models.rasiya_koya_tft.runner.run_fold` on the Newman 531 basins
(train 1999-10-01..2008-09-30, evaluate 1989-10-01..1999-09-30, Maurer forcings as used for the
MIP TFT re-run) and scores the hydrograph with the Gauch et al. (2023) metric suite
(`metrics.metrics.calculate_all_metrics`).  Nothing precomputed is read.

    python -m models.rasiya_koya_tft.reproduce_author_metrics --data-dir /path/to/camels --mode verify
    python -m models.rasiya_koya_tft.reproduce_author_metrics --smoke      # pipeline check only
    python -m models.rasiya_koya_tft.reproduce_author_metrics --write-tracked  # regenerate the tracked JSON

Full runs write `stage1_author_reproduction_results.local.json` (untracked) unless `--write-tracked`.
"""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli
from models.rasiya_koya_tft.runner import run_fold
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
)

MODEL_KEY = "rasiya_koya_tft"
MODEL_NAME = "Rasiya Koya et al. (2023) Temporal Fusion Transformer"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END),
    "eval_period": (CANONICAL_EVAL_START, CANONICAL_EVAL_END),
    "forcings": ["maurer"],
    "seeds": (123,),
}
# The only Newman-protocol TFT reference available in this repository is the MIP's own earlier
# NeuralHydrology TFT re-run on Maurer forcings (`leaderboards/gauged__single_camels.json`
# entry t3_21, single model, median NSE 0.725).  Rasiya Koya & Roy (2024) use a different
# basin set / split, so their headline number is not asserted here.
PUBLISHED_SINGLE_MEDIAN_NSE = 0.725
PUBLISHED_ENSEMBLE_MEDIAN_NSE = None
PUBLISHED_SOURCE = (
    "mips/camels_us_daily/leaderboards/gauged__single_camels.json entry t3_21 (NeuralHydrology TFT, Maurer, single model); "
    "architecture after Rasiya Koya & Roy (2024), Journal of Hydrology"
)
TOLERANCE = 0.03
SMOKE_KWARGS = {"hidden_size": 8, "seq_len": 20, "eval_batch_size": 2048}


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
