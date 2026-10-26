"""Stage-1 author-protocol reproduction for Yin et al. (2022, J. Hydrol.) RR-Former.

Trains the 3-seed RR-Former (global pretraining + per-basin fine-tuning with
1-day-lagged observed discharge) in-process with `models.yin2022_rrformer.runner.run_fold`
on the 531 CAMELS-US basins (Daymet forcings, train 1980-10-01..1995-09-30,
evaluate 1995-10-01..2010-09-30) and scores the hydrograph with
`metrics.metrics.calculate_all_metrics`.

    python -m models.yin2022_rrformer.reproduce_author_metrics --data-dir /path/to/camels --mode verify
    python -m models.yin2022_rrformer.reproduce_author_metrics --smoke
"""

from __future__ import annotations

from pathlib import Path

from metrics.metrics import calculate_all_metrics
from models.kratzert_lstm.reproduction import SPLITS_DIR, build_arg_parser, run_cli
from models.yin2022_rrformer.runner import DEFAULT_SEEDS, run_fold

MODEL_KEY = "yin2022_rrformer"
MODEL_NAME = "Yin et al. (2022) RR-Former"
MODEL_DIR = Path(__file__).resolve().parent

AUTHOR_PROTOCOL = {
    "basins_file": SPLITS_DIR / "basins_531.json",
    "train_period": ("1980-10-01", "1995-09-30"),
    "eval_period": ("1995-10-01", "2010-09-30"),
    "forcings": ["daymet"],
    "seeds": DEFAULT_SEEDS,
}
# Yin et al. (2022, J. Hydrol. 609, 127781) report Day-1 Median NSE = 0.8265
# (Mean NSE = 0.7904; global pretraining w/o fine-tuning RR-Former2 = 0.7598) in
# Section 4.1 (p. 6), Abstract, Fig. 4a, and Tables 5-6 (whereas 0.862 in Table 4
# is Mean NSE on a 40-basin subset / Feng et al. 2020 DI(1) phantom target).
PUBLISHED_SINGLE_MEDIAN_NSE = 0.8265
PUBLISHED_ENSEMBLE_MEDIAN_NSE = 0.8265
PUBLISHED_SOURCE = (
    "Yin et al. (2022), J. Hydrol. 609, 127781, Section 4.1 (p. 6), Fig. 4a & Tables 5-6 "
    "(RR-Former Day-1 Median NSE = 0.8265, Mean NSE = 0.7904; note: 0.862 in Table 4 is "
    "40-basin subset Mean NSE)"
)
TOLERANCE = 0.05
SMOKE_KWARGS = {
    "d_model": 16,
    "nhead": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dim_feedforward": 32,
    "seq_len": 15,
    "batch_size": 32,
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
