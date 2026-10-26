"""Stage-1 author-protocol reproduction for Feng, Fang & Shen (2020, WRR) `DI(1)` LSTM.

Trains the 6-seed Feng et al. (2020) `DI(1)` Data-Integration Lagged-Discharge LSTM
in-process with `models.feng2020_di_lstm.runner.run_fold` on the authors' Split-3
protocol (671 CAMELS-US basins, NLDAS meteorological forcings + 1-day-lagged observed
specific discharge `Q_{t-1}`, train 1985-10-01..1995-09-30, evaluate 1995-10-01..2005-09-30)
and scores the resulting hydrographs with `metrics.metrics.calculate_all_metrics`.

    python -m models.feng2020_di_lstm.reproduce_author_metrics --data-dir /path/to/camels --mode verify
    python -m models.feng2020_di_lstm.reproduce_author_metrics --smoke
"""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.feng2020_di_lstm.runner import DEFAULT_SEEDS, run_fold
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli

MODEL_KEY = "feng2020_di_lstm"
MODEL_NAME = "Feng et al. (2020) DI(1) Lagged-Discharge LSTM"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_671.json",
    "train_period": ("1985-10-01", "1995-09-30"),
    "eval_period": ("1995-10-01", "2005-09-30"),
    "forcings": ["nldas"],
    "seeds": DEFAULT_SEEDS,
}
PUBLISHED_SINGLE_MEDIAN_NSE = 0.851
PUBLISHED_ENSEMBLE_MEDIAN_NSE = 0.859
PUBLISHED_SOURCE = (
    "Feng, Fang & Shen (2020), WRR, Table 3 (DI(1), 671 basins, NLDAS, 1995-2005 eval)"
)
TOLERANCE = 0.04
SMOKE_KWARGS = {"hidden_size": 16, "seq_len": 30, "batch_size": 32}


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
