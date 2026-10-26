"""Standalone cloud-only hydrograph metrics evaluation package for CAMELS-US MIP.

Exposes the single primary public API function `calculate_model_leaderboard_metrics`
for scoring standardized model hydrograph Zarr stores directly from
`gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr` against
`gs://camels-us-mip/observations/newman_eval_targets.zarr`, alongside the 14
Gauch et al. (2023) hydrological metrics in `metrics.metrics`.
"""

from metrics.evaluator import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_BUCKET_URI,
    EVAL_END_DATE,
    EVAL_START_DATE,
    GAUCH_2023_METRICS,
    NUM_BASINS,
    NUM_EVAL_DAYS,
    calculate_model_leaderboard_metrics,
)
from metrics.metrics import (
    AllNaNError,
    alpha_nse,
    beta_kge,
    beta_nse,
    calculate_all_metrics,
    calculate_metrics,
    fdc_fhv,
    fdc_flv,
    fdc_fms,
    get_available_metrics,
    get_frequency_factor,
    infer_datetime_coord,
    kge,
    mean_absolute_percentage_peak_error,
    mean_peak_timing,
    missed_peaks,
    mse,
    nse,
    pearsonr,
    rmse,
)

__all__ = [
    "CANONICAL_LEADERBOARD_IDS",
    "DEFAULT_BUCKET_URI",
    "EVAL_END_DATE",
    "EVAL_START_DATE",
    "GAUCH_2023_METRICS",
    "NUM_BASINS",
    "NUM_EVAL_DAYS",
    "AllNaNError",
    "alpha_nse",
    "beta_kge",
    "beta_nse",
    "calculate_all_metrics",
    "calculate_metrics",
    "calculate_model_leaderboard_metrics",
    "fdc_fhv",
    "fdc_flv",
    "fdc_fms",
    "get_available_metrics",
    "get_frequency_factor",
    "infer_datetime_coord",
    "kge",
    "mean_absolute_percentage_peak_error",
    "mean_peak_timing",
    "missed_peaks",
    "mse",
    "nse",
    "pearsonr",
    "rmse",
]
