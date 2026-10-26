"""Stage-1 author-protocol reproduction for Hoedt et al. (2021) MC-LSTM.

Trains the MC-LSTM in-process with `models.hoedt_mclstm.runner.run_fold` on the Newman 531
basins (Hoedt et al., 2021, ICML, hydrology benchmark following Kratzert et al. 2019: Maurer
forcings, train 1999-10-01..2008-09-30, evaluate 1989-10-01..1999-09-30, NSE* loss, 64 cells,
unnormalized precipitation mass input) and scores the hydrograph with the Gauch et al. (2023)
metric suite (`metrics.metrics.calculate_all_metrics`).  Nothing precomputed is read.

    python -m models.hoedt_mclstm.reproduce_author_metrics --data-dir /path/to/camels --mode verify
    python -m models.hoedt_mclstm.reproduce_author_metrics --smoke      # pipeline check only
"""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.hoedt_mclstm.runner import run_fold
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli

MODEL_KEY = "hoedt_mclstm"
MODEL_NAME = "Hoedt et al. (2021) MC-LSTM"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": ("1999-10-01", "2008-09-30"),
    "eval_period": ("1989-10-01", "1999-09-30"),
    "forcings": ["maurer"],
    "seeds": (301, 302, 303),
}
# Median NSE over the Newman CAMELS-US benchmark for the MC-LSTM of Hoedt et al. (2021),
# ICML (PMLR 139:4275-4286), Table 2/3 & Appendix B (0.744 5-seed/10-seed ensemble on 447/498
# basins, 0.742 on 531 basins; 0.726 +/- 0.004 single model on Maurer forcings).
PUBLISHED_SINGLE_MEDIAN_NSE = 0.726
PUBLISHED_ENSEMBLE_MEDIAN_NSE = 0.744
PUBLISHED_SOURCE = (
    "Hoedt et al. (2021), ICML (PMLR 139:4275-4286), Table 2/3 & Appendix B "
    "(0.744 ensemble on 447/498 basins / 0.742 on 531 basins; 0.726 +/- 0.004 single model on Maurer)"
)
TOLERANCE = 0.02
SMOKE_KWARGS = {"hidden_size": 4, "seq_len": 30, "eval_batch_size": 4096}


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
