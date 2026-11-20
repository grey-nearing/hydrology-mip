# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 author-protocol reproduction for Kratzert et al. (2021) CudaLSTM.

Trains the CudaLSTM in-process with `models.kratzert_lstm.runner.run_fold` on the Newman 531
basins (Kratzert et al., 2021 single-forcing protocol: train 1999-10-01..2008-09-30, evaluate
1989-10-01..1999-09-30) and scores the hydrograph with the Gauch et al. (2023) metric suite
(`metrics.metrics.calculate_all_metrics`).  Nothing precomputed is read.

    python -m models.kratzert_lstm.reproduce_author_metrics --data-dir /path/to/camels --mode verify
    python -m models.kratzert_lstm.reproduce_author_metrics --smoke      # pipeline check only
    python -m models.kratzert_lstm.reproduce_author_metrics --write-tracked  # regenerate the tracked JSON

Full runs write `stage1_author_reproduction_results.local.json` (untracked) unless `--write-tracked`.
"""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli
from models.kratzert_lstm.runner import run_fold
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
)

MODEL_KEY = "kratzert_lstm"
MODEL_NAME = "Kratzert et al. (2019, 2021) CudaLSTM"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END),
    "eval_period": (CANONICAL_EVAL_START, CANONICAL_EVAL_END),
    "forcings": ["daymet"],
    "seeds": (101, 102, 103),
}
# Median NSE over the 531 Newman basins for the Daymet single-forcing LSTM of Kratzert et al.
# (2021, HESS 25, 2685-2706, Table 1 & Table A2; Zenodo 10.5281/zenodo.4670268 / t3_02):
# 0.744 ± 0.003 single-seed, 0.772 10-seed ensemble.
PUBLISHED_SINGLE_MEDIAN_NSE = 0.744
PUBLISHED_ENSEMBLE_MEDIAN_NSE = 0.772
PUBLISHED_SOURCE = (
    "Kratzert et al. (2021), HESS 25, 2685-2706, Table 1 & Table A2; Zenodo 10.5281/zenodo.4670268 "
    "and mips/camels_us_daily/leaderboards/gauged__single_camels.json entry t3_02 (10-model ensemble 0.772, single model 0.744)"
)
TOLERANCE = 0.04
SMOKE_KWARGS = {"hidden_size": 8, "seq_len": 30, "eval_batch_size": 4096}


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
