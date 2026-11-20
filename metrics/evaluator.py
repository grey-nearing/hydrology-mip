# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cloud-only Zarr hydrograph evaluator for the CAMELS-US Model Intercomparison Project.

Scores standardized daily simulated streamflow hydrographs (`qsim`, shape `(531, 3652)`
in `mm/day`) stored in `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr`
against the ground-truth USGS observations (`qobs`, shape `(531, 3652)` in `mm/day`)
stored in `<bucket_uri>/observations/newman_eval_targets.zarr`.

Computes all 14 Gauch et al. (2023) ("In Defense of Metrics") hydrological metrics
via `metrics/metrics.py` and strictly enforces that no missing basins or missing
predictions on valid observation days are allowed (`pred_nan_when_obs_valid == 0`).
The only exception is an explicit `allow_partial_archive=True` for declared partial
author archives, which permits whole-basin exclusion but never within-basin masking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hydromip.evaluation.zarr_io import (
    deterministic_io,
    get_gcs_storage_options,
    open_zarr_store,
    validate_store_id,
    zarr_store_exists,
)
from metrics.metrics import calculate_all_metrics, get_available_metrics, nse

LOGGER = logging.getLogger(__name__)

DEFAULT_BUCKET_URI: str = "gs://camels-us-mip"
NUM_BASINS: int = 531
NUM_EVAL_DAYS: int = 3652
EVAL_START_DATE: str = "1989-10-01"
EVAL_END_DATE: str = "1999-09-30"
MIN_VALID_OBS_DAYS: int = 30
MIN_SST_THRESHOLD: float = 1e-12

GAUCH_2023_METRICS: tuple[str, ...] = (
    "NSE",
    "MSE",
    "RMSE",
    "KGE",
    "logNSE",
    "logKGE",
    "Alpha-NSE",
    "Pearson-r",
    "Beta-KGE",
    "Beta-NSE",
    "FHV",
    "FMS",
    "FLV",
    "Peak-Timing",
)

GAUCH_METRIC_TO_SNAKE: dict[str, str] = {
    "NSE": "nse",
    "MSE": "mse",
    "RMSE": "rmse",
    "KGE": "kge",
    "logNSE": "log_nse",
    "logKGE": "log_kge",
    "Alpha-NSE": "alpha_nse",
    "Pearson-r": "pearson_r",
    "Beta-KGE": "beta_kge",
    "Beta-NSE": "beta_nse",
    "FHV": "fhv",
    "FMS": "fms",
    "FLV": "flv",
    "Peak-Timing": "peak_timing",
}

CANONICAL_LEADERBOARD_IDS: tuple[str, ...] = (
    "gauged__single_camels",
    "gauged__multi_camels",
    "gauged__lagged_streamflow",
    "gauged__freeform",
    "pub__single_camels",
    "pub__multi_camels",
    "pub__lagged_streamflow",
    "pub__freeform",
    "pur__single_camels",
    "pur__multi_camels",
    "pur__lagged_streamflow",
    "pur__freeform",
)

# Backwards-compatible private aliases; the single implementation lives in
# `hydromip.evaluation.zarr_io` (credential resolution, existence checks, store opening).
_get_gcs_storage_options = get_gcs_storage_options
_zarr_store_exists = zarr_store_exists
_open_zarr_store = open_zarr_store


_GCS_TARGETS_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _validate_targets_dataset(
    targets_ds: xr.Dataset,
    targets_uri: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate coordinate and variable structure of `newman_eval_targets.zarr`."""
    if "qobs" not in targets_ds.data_vars:
        raise KeyError(
            f"Target Zarr store '{targets_uri}' is missing required 'qobs' variable."
        )
    for coord_name in ("basin", "date", "huc_02"):
        if coord_name not in targets_ds.coords and coord_name not in targets_ds:
            raise KeyError(
                f"Target Zarr store '{targets_uri}' is missing required '{coord_name}' coordinate."
            )

    qobs_da = targets_ds["qobs"]
    if tuple(qobs_da.dims) != ("basin", "date"):
        raise ValueError(
            f"Target 'qobs' in '{targets_uri}' must have dimensions ('basin', 'date'), "
            f"got {tuple(qobs_da.dims)}."
        )
    if qobs_da.shape != (NUM_BASINS, NUM_EVAL_DAYS):
        raise ValueError(
            f"Target 'qobs' in '{targets_uri}' must have shape ({NUM_BASINS}, {NUM_EVAL_DAYS}), "
            f"got {qobs_da.shape}."
        )

    target_basins = np.asarray(targets_ds["basin"].values, dtype=str)
    if target_basins.shape != (NUM_BASINS,):
        raise ValueError(
            f"Target 'basin' coordinate must have shape ({NUM_BASINS},), got {target_basins.shape}."
        )
    if not all(len(b) == 8 and b.isdigit() for b in target_basins):
        raise ValueError(
            f"Target 'basin' coordinate in '{targets_uri}' must contain 8-digit USGS gauge IDs."
        )
    if list(target_basins) != sorted(target_basins):
        raise ValueError(
            f"Target 'basin' coordinate in '{targets_uri}' must be monotonically sorted."
        )

    expected_dates = pd.date_range(
        EVAL_START_DATE, EVAL_END_DATE, freq="D"
    ).values.astype("datetime64[D]")
    target_dates = pd.to_datetime(targets_ds["date"].values).values.astype(
        "datetime64[D]"
    )
    if target_dates.shape != (NUM_EVAL_DAYS,) or not np.array_equal(
        target_dates, expected_dates
    ):
        raise ValueError(
            f"Target 'date' coordinate in '{targets_uri}' must match daily dates "
            f"{EVAL_START_DATE} to {EVAL_END_DATE} ({NUM_EVAL_DAYS} days)."
        )

    target_huc02 = np.asarray(targets_ds["huc_02"].values, dtype=str)
    if target_huc02.shape != (NUM_BASINS,):
        raise ValueError(
            f"Target 'huc_02' coordinate must have shape ({NUM_BASINS},), got {target_huc02.shape}."
        )

    qobs = np.asarray(qobs_da.values, dtype=np.float64)
    return qobs, target_basins, target_dates, target_huc02


def _validate_hydrograph_dataset(
    hydrograph_ds: xr.Dataset,
    hydrograph_uri: str,
    target_basins: np.ndarray,
    target_dates: np.ndarray,
    target_huc02: np.ndarray,
) -> np.ndarray:
    """Validate coordinate and variable structure of a model hydrograph Zarr store."""
    if "qsim" not in hydrograph_ds.data_vars:
        raise KeyError(
            f"Hydrograph Zarr store '{hydrograph_uri}' is missing required 'qsim' variable."
        )
    for coord_name in ("basin", "date"):
        if coord_name not in hydrograph_ds.coords:
            raise KeyError(
                f"Hydrograph Zarr store '{hydrograph_uri}' is missing required '{coord_name}' coordinate."
            )

    qsim_da = hydrograph_ds["qsim"]
    if tuple(qsim_da.dims) != ("basin", "date"):
        raise ValueError(
            f"Hydrograph 'qsim' in '{hydrograph_uri}' must have dimensions ('basin', 'date'), "
            f"got {tuple(qsim_da.dims)}."
        )
    if qsim_da.shape != (NUM_BASINS, NUM_EVAL_DAYS):
        raise ValueError(
            f"Hydrograph 'qsim' in '{hydrograph_uri}' must have shape "
            f"({NUM_BASINS}, {NUM_EVAL_DAYS}), got {qsim_da.shape}."
        )

    sim_basins = np.asarray(hydrograph_ds["basin"].values, dtype=str)
    if sim_basins.shape != (NUM_BASINS,) or not np.array_equal(
        sim_basins, target_basins
    ):
        raise ValueError(
            f"Hydrograph 'basin' coordinate in '{hydrograph_uri}' does not match "
            f"canonical sorted 531 USGS basin IDs in exact order."
        )

    sim_dates = pd.to_datetime(hydrograph_ds["date"].values).values.astype(
        "datetime64[D]"
    )
    if sim_dates.shape != (NUM_EVAL_DAYS,) or not np.array_equal(
        sim_dates, target_dates
    ):
        raise ValueError(
            f"Hydrograph 'date' coordinate in '{hydrograph_uri}' does not match "
            f"canonical evaluation window {EVAL_START_DATE} to {EVAL_END_DATE} ({NUM_EVAL_DAYS} days)."
        )

    if "huc_02" in hydrograph_ds.coords:
        sim_huc02 = np.asarray(hydrograph_ds["huc_02"].values, dtype=str)
        if sim_huc02.shape != (NUM_BASINS,) or not np.array_equal(
            sim_huc02, target_huc02
        ):
            raise ValueError(
                f"Hydrograph 'huc_02' coordinate in '{hydrograph_uri}' does not match "
                f"canonical USGS HUC-02 basin assignments."
            )

    return np.asarray(qsim_da.values, dtype=np.float64)


def _validate_hydrograph_members(
    hydrograph_ds: xr.Dataset,
    hydrograph_uri: str,
) -> np.ndarray | None:
    """Validate the optional per-member hydrograph variable `qsim_members`.

    Returns the members as a float64 array of shape `(n_members, 531, 3652)`, or `None`
    when the store carries no `qsim_members` variable.

    Raises:
        ValueError: If `qsim_members` does not have dimensions `("member", "basin", "date")`
            and shape `(n_members >= 1, 531, 3652)`.
    """
    if "qsim_members" not in hydrograph_ds.data_vars:
        return None
    members_da = hydrograph_ds["qsim_members"]
    if tuple(members_da.dims) != ("member", "basin", "date"):
        raise ValueError(
            f"Hydrograph 'qsim_members' in '{hydrograph_uri}' must have dimensions "
            f"('member', 'basin', 'date'), got {tuple(members_da.dims)}."
        )
    if (
        members_da.ndim != 3
        or members_da.shape[0] < 1
        or members_da.shape[1:] != (NUM_BASINS, NUM_EVAL_DAYS)
    ):
        raise ValueError(
            f"Hydrograph 'qsim_members' in '{hydrograph_uri}' must have shape "
            f"(n_members >= 1, {NUM_BASINS}, {NUM_EVAL_DAYS}), got {tuple(members_da.shape)}."
        )
    return np.asarray(members_da.values, dtype=np.float64)


def _resolve_is_ensemble(
    hydrograph_ds: xr.Dataset,
    hydrograph_uri: str,
    members: np.ndarray | None,
) -> bool:
    """Resolve whether the stored `qsim` is an ensemble mean from the store attributes.

    The `is_ensemble` global attribute is authoritative when present. Without it, a store
    with more than one member is treated as an ensemble and a store without members as a
    single model.

    Raises:
        ValueError: If `is_ensemble` is present but not a boolean-like value.
    """
    raw = hydrograph_ds.attrs.get("is_ensemble")
    if raw is None:
        return members is not None and members.shape[0] > 1
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, (int, np.integer)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        return raw.strip().lower() == "true"
    raise ValueError(
        f"Hydrograph attribute 'is_ensemble' in '{hydrograph_uri}' must be boolean, got {raw!r}."
    )


SUPPORTED_ENSEMBLE_REDUCTIONS: tuple[str, ...] = ("mean", "median", "nanmean", "nanmedian")
PER_BASIN_ORACLE_REDUCTION: str = "per_basin_oracle"
# Reductions under which individual members may be missing (NaN) on valid observation days:
# archived multi-structure suites (e.g. MARRMoT) where some structures were never run for some
# basins. `qsim` itself must still be complete; for `mean`/`median`/single-member stores every
# member must be complete.
_REDUCTIONS_ALLOWING_PARTIAL_MEMBERS: tuple[str, ...] = (
    "nanmean",
    "nanmedian",
    PER_BASIN_ORACLE_REDUCTION,
)


def _resolve_ensemble_reduction(hydrograph_ds: xr.Dataset, hydrograph_uri: str) -> str:
    """Read the declared member reduction (`ensemble_reduction` attribute; default `"mean"`)."""
    raw = hydrograph_ds.attrs.get("ensemble_reduction", "mean")
    if not isinstance(raw, str):
        raise ValueError(
            f"Hydrograph attribute 'ensemble_reduction' in '{hydrograph_uri}' must be a string."
        )
    reduction = raw.strip().lower()
    if reduction not in SUPPORTED_ENSEMBLE_REDUCTIONS + (PER_BASIN_ORACLE_REDUCTION,):
        raise ValueError(
            f"Unsupported 'ensemble_reduction' {raw!r} in '{hydrograph_uri}'; expected one of "
            f"{SUPPORTED_ENSEMBLE_REDUCTIONS + (PER_BASIN_ORACLE_REDUCTION,)}."
        )
    return reduction


def _mean_over_finite_members(stack: np.ndarray) -> np.ndarray:
    """Mean over axis 0 of the finite entries; NaN where no member is finite (no warnings)."""
    finite = np.isfinite(stack)
    n_finite = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, stack, 0.0), axis=0)
    return np.where(n_finite > 0, total / np.maximum(n_finite, 1), np.nan)


def _median_over_finite_members(stack: np.ndarray) -> np.ndarray:
    """Median over axis 0 of the finite entries; NaN where no member is finite (no warnings)."""
    n_finite = np.sum(np.isfinite(stack), axis=0)
    ordered = np.sort(stack, axis=0)  # NaNs sort to the end of axis 0
    last = int(stack.shape[0]) - 1
    lower = np.clip((n_finite - 1) // 2, 0, last)[np.newaxis, ...]
    upper = np.clip(n_finite // 2, 0, last)[np.newaxis, ...]
    lower_vals = np.take_along_axis(ordered, lower, axis=0)[0]
    upper_vals = np.take_along_axis(ordered, upper, axis=0)[0]
    return np.where(n_finite > 0, 0.5 * (lower_vals + upper_vals), np.nan)


def _check_members_against_qsim(
    sim: np.ndarray,
    members: np.ndarray,
    obs_valid: np.ndarray,
    is_ensemble: bool,
    reduction: str,
    hydrograph_uri: str,
) -> dict[str, Any]:
    """Enforce consistency between `qsim` and `qsim_members` and derive per-member bookkeeping.

    - Every member must be finite on every valid observation day (same anti-masking rule
      as `qsim`), except under the explicitly declared partial-member reductions
      (`nanmean`, `nanmedian`, `per_basin_oracle`), where a member may be absent for a
      whole basin. Per-member scoring still rejects a member that is only partially
      missing within a basin (no within-basin masking).
    - For an ensemble store, `qsim` must equal the declared member reduction
      (`ensemble_reduction` attribute: `"mean"` (default) or `"median"`, compared with
      `np.allclose(rtol=1e-4, atol=1e-4)`); a store whose `qsim` was clipped, re-weighted,
      or cherry-picked is rejected rather than silently scored.
    - For a single-model store that ships its sibling seeds, `qsim` must equal exactly one
      of the members; the matching member index is reported. The only exception is an
      explicitly declared `ensemble_reduction="per_basin_oracle"` (a per-basin best-member
      selection, which is a diagnostic bound rather than a model) which is passed through
      with `ensemble_source="per_basin_oracle"`.

    Returns:
        Dictionary with `num_members`, `ensemble_source`, and `single_seed_member_index`.

    Raises:
        ValueError: On any `NaN`/`Inf` on valid observation days or on `qsim`/member mismatch.
    """
    n_members = int(members.shape[0])
    member_inf = np.isinf(members)
    if np.any(member_inf):
        raise ValueError(
            f"Non-finite prediction violation in '{hydrograph_uri}': found "
            f"{int(np.sum(member_inf))} infinite values in 'qsim_members'."
        )
    member_nan = np.isnan(members) & obs_valid[np.newaxis, :, :]
    member_nan_count = int(np.sum(member_nan))
    if member_nan_count > 0 and reduction not in _REDUCTIONS_ALLOWING_PARTIAL_MEMBERS:
        affected_members = int(np.sum(np.any(member_nan, axis=(1, 2))))
        raise ValueError(
            f"Anti-masking violation in '{hydrograph_uri}': 'qsim_members' contains "
            f"{member_nan_count} NaN values on valid observation days across "
            f"{affected_members} member(s). No silent NaN masking is allowed."
        )

    if is_ensemble:
        if reduction == PER_BASIN_ORACLE_REDUCTION:
            raise ValueError(
                f"'{hydrograph_uri}' declares is_ensemble=True together with "
                f"ensemble_reduction='{PER_BASIN_ORACLE_REDUCTION}'; an oracle selection is "
                f"not an ensemble mean and must be stored with is_ensemble=False."
            )
        if reduction == "mean":
            reduced = np.mean(members, axis=0)
        elif reduction == "median":
            reduced = np.median(members, axis=0)
        elif reduction == "nanmean":
            reduced = _mean_over_finite_members(members)
        else:
            reduced = _median_over_finite_members(members)
        comparable = np.isfinite(sim) & np.isfinite(reduced)
        if not np.array_equal(np.isfinite(sim) & obs_valid, np.isfinite(reduced) & obs_valid):
            raise ValueError(
                f"Ensemble consistency violation in '{hydrograph_uri}': the {reduction} of "
                f"'qsim_members' is undefined on valid observation days where 'qsim' is finite."
            )
        if not np.allclose(sim[comparable], reduced[comparable], rtol=1e-4, atol=1e-4):
            max_abs = float(np.max(np.abs(sim[comparable] - reduced[comparable])))
            raise ValueError(
                f"Ensemble consistency violation in '{hydrograph_uri}': stored 'qsim' is "
                f"not the {reduction} of 'qsim_members' (max |qsim - {reduction}(members)| = "
                f"{max_abs:.6g}, n_members = {n_members}). Store the unmodified member "
                f"{reduction} as 'qsim' (no clipping, weighting, or member selection)."
            )
        return {
            "num_members": n_members,
            "ensemble_source": f"{reduction}_of_qsim_members",
            "single_seed_member_index": None,
        }

    if reduction == PER_BASIN_ORACLE_REDUCTION:
        return {
            "num_members": n_members,
            "ensemble_source": PER_BASIN_ORACLE_REDUCTION,
            "single_seed_member_index": None,
        }

    matches = []
    for i in range(n_members):
        comparable = np.isfinite(sim) & np.isfinite(members[i])
        if np.allclose(sim[comparable], members[i][comparable], rtol=1e-5, atol=1e-5):
            matches.append(int(i))
    if len(matches) != 1:
        raise ValueError(
            f"Single-model consistency violation in '{hydrograph_uri}': attribute "
            f"'is_ensemble' is False but stored 'qsim' matches {len(matches)} of the "
            f"{n_members} 'qsim_members' (expected exactly one)."
        )
    return {
        "num_members": n_members,
        "ensemble_source": "single_member",
        "single_seed_member_index": matches[0],
    }


def _per_member_median_nse(
    members: np.ndarray,
    obs_for_da: np.ndarray,
    dates_ns: np.ndarray,
    good_b: np.ndarray,
) -> list[float]:
    """Compute the 531-basin median NSE of every individual member via `metrics.metrics.nse`.

    A member may be entirely absent for a basin (partial multi-structure suites); that basin
    is then excluded from the member's median. A member that is only *partially* missing on
    a basin's valid observation days is rejected: no within-basin masking is ever applied.
    """
    obs_valid = np.isfinite(obs_for_da)
    per_member: list[float] = []
    for m_idx in range(int(members.shape[0])):
        nse_vec = np.full(NUM_BASINS, np.nan, dtype=np.float64)
        for b_idx in np.where(good_b)[0]:
            member_row = members[m_idx, b_idx]
            missing = np.isnan(member_row) & obs_valid[b_idx]
            if np.all(missing[obs_valid[b_idx]]):
                continue
            if np.any(missing):
                raise ValueError(
                    f"Member {m_idx} is partially missing on {int(np.sum(missing))} valid "
                    f"observation days of basin index {b_idx}; within-basin masking is not allowed."
                )
            obs_da = xr.DataArray(
                obs_for_da[b_idx], coords={"date": dates_ns}, dims=("date",)
            )
            sim_da = xr.DataArray(member_row, coords={"date": dates_ns}, dims=("date",))
            nse_vec[b_idx] = float(nse(obs_da, sim_da))
        finite = nse_vec[np.isfinite(nse_vec)]
        if finite.size == 0:
            raise ValueError(
                f"Member {m_idx} produced no finite NSE on any evaluable basin."
            )
        per_member.append(float(np.median(finite)))
    return per_member


def _summarize_metric_vector(
    values: np.ndarray,
    ndigits: int = 4,
) -> dict[str, float]:
    """Compute quantiles and mean across finite values of a per-basin metric vector.

    Both `NaN` (metric undefined for a basin) and `+/-Inf` (degenerate ratio) entries are
    excluded, so no aggregate can silently become infinite.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {
            k: float("nan")
            for k in (
                "median",
                "mean",
                "min",
                "p01",
                "p05",
                "p10",
                "q25",
                "q75",
                "p90",
                "p95",
                "p99",
                "max",
            )
        }
    return {
        "median": round(float(np.median(valid)), ndigits),
        "mean": round(float(np.mean(valid)), ndigits),
        "min": round(float(np.min(valid)), ndigits),
        "p01": round(float(np.percentile(valid, 1)), ndigits),
        "p05": round(float(np.percentile(valid, 5)), ndigits),
        "p10": round(float(np.percentile(valid, 10)), ndigits),
        "q25": round(float(np.percentile(valid, 25)), ndigits),
        "q75": round(float(np.percentile(valid, 75)), ndigits),
        "p90": round(float(np.percentile(valid, 90)), ndigits),
        "p95": round(float(np.percentile(valid, 95)), ndigits),
        "p99": round(float(np.percentile(valid, 99)), ndigits),
        "max": round(float(np.max(valid)), ndigits),
    }


def calculate_model_leaderboard_metrics(
    model_id: str,
    leaderboard_id: str,
    bucket_uri: str = DEFAULT_BUCKET_URI,
    allow_partial_archive: bool = False,
    evaluation_mode: str | None = None,
) -> dict[str, Any]:
    """Calculate all 14 Gauch et al. (2023) & CAMELS-US MIP metrics for one model and leaderboard.

    Reads the standardized model hydrograph Zarr store at
    `<bucket_uri>/hydrographs/<leaderboard_id>/<model_id>.zarr` and scores it against
    the canonical observed target Zarr store at
    `<bucket_uri>/observations/newman_eval_targets.zarr`.

    Enforces strict completeness and anti-masking:
      - Every Zarr store must have shape `(531, 3652)`.
      - No missing basins and no partial-time-series `NaN`s are permitted: on every
        basin and day where `obs_valid` (`(~np.isnan(qobs)) & (qobs >= 0.0)`) is True,
        `qsim` must be finite (`pred_nan_when_obs_valid == 0`).
      - If the store carries per-member hydrographs `qsim_members`
        (`("member", "basin", "date")`, shape `(n_members, 531, 3652)`), every member is
        held to the same rule, and `qsim` must be exactly the member mean for an
        ensemble store (`is_ensemble=True`) or exactly one member for a single-model
        store (`is_ensemble=False`). Single-seed skill is then reported as the mean and
        population standard deviation of the per-member 531-basin median NSE
        (`single_seed_median_nse_mean`, `single_seed_median_nse_std`,
        `single_seed_median_nse_per_member`). A store that declares `is_ensemble=True`
        without members yields `None` for these fields: the claim cannot be verified and
        is never substituted with the ensemble score.

    Args:
        model_id: Model entry identifier corresponding to `<model_id>.zarr`.
        leaderboard_id: One of the 12 canonical leaderboard IDs (`CANONICAL_LEADERBOARD_IDS`).
        bucket_uri: Root URI of the CAMELS-US MIP bucket (default `"gs://camels-us-mip"`).

    Returns:
        Dictionary containing the 5 headline leaderboard metrics, all 14 Gauch et al.
        (2023) metric distributions (`gauch_2023_metrics`), lower-tail skill percentiles,
        upper-tail peak-flow bias percentiles, `pred_nan_when_obs_valid` (`0`),
        18-region USGS HUC-02 median NSE breakdown (`regional_huc02`), and per-basin
        metric vectors (`per_basin`).

    Raises:
        ValueError: If `model_id`, `leaderboard_id`, or `bucket_uri` is invalid, if
            coordinates or shapes do not match `(531, 3652)`, or if any evaluable basin
            is missing or contains any `NaN` or `Inf` on valid observation days
            (`pred_nan_when_obs_valid > 0`). With `allow_partial_archive=True`, whole
            basins may be absent, but a basin that is only partially predicted still
            raises. `evaluation_mode="single_member_mean"` raises unless the store
            carries `qsim_members` with at least two members.
        FileNotFoundError: If either the target Zarr store or the model hydrograph
            Zarr store does not exist.
        KeyError: If required variables (`qobs`, `qsim`) or coordinates (`basin`,
            `date`, `huc_02`) are missing from the Zarr stores.
    """
    validate_store_id(model_id, "model_id")
    with deterministic_io():
        return _score_model_leaderboard(
            model_id=model_id,
            leaderboard_id=leaderboard_id,
            bucket_uri=bucket_uri,
            allow_partial_archive=allow_partial_archive,
            evaluation_mode=evaluation_mode,
        )


def _score_model_leaderboard(
    model_id: str,
    leaderboard_id: str,
    bucket_uri: str,
    allow_partial_archive: bool,
    evaluation_mode: str | None,
) -> dict[str, Any]:
    """Implementation of `calculate_model_leaderboard_metrics` (called inside scoped I/O)."""
    if leaderboard_id not in CANONICAL_LEADERBOARD_IDS:
        raise ValueError(
            f"Invalid leaderboard_id '{leaderboard_id}'. Must be one of the 12 canonical "
            f"leaderboards: {CANONICAL_LEADERBOARD_IDS}"
        )
    if not isinstance(bucket_uri, str) or not bucket_uri.strip():
        raise ValueError("bucket_uri must be a non-empty string.")
    if evaluation_mode is not None and evaluation_mode not in (
        "ensemble",
        "single_member_mean",
    ):
        raise ValueError(
            f"Unsupported evaluation_mode {evaluation_mode!r}; expected None, "
            f"'ensemble', or 'single_member_mean'."
        )

    root_uri = bucket_uri.rstrip("/")
    targets_uri = f"{root_uri}/observations/newman_eval_targets.zarr"
    hydrograph_uri = f"{root_uri}/hydrographs/{leaderboard_id}/{model_id}.zarr"

    if targets_uri.startswith("gs://") and targets_uri in _GCS_TARGETS_CACHE:
        obs, target_basins, target_dates, target_huc02 = _GCS_TARGETS_CACHE[targets_uri]
    else:
        targets_ds = _open_zarr_store(targets_uri)
        obs, target_basins, target_dates, target_huc02 = _validate_targets_dataset(
            targets_ds, targets_uri
        )
        if targets_uri.startswith("gs://"):
            _GCS_TARGETS_CACHE[targets_uri] = (
                obs,
                target_basins,
                target_dates,
                target_huc02,
            )
    hydrograph_ds = _open_zarr_store(hydrograph_uri)
    sim = _validate_hydrograph_dataset(
        hydrograph_ds,
        hydrograph_uri,
        target_basins=target_basins,
        target_dates=target_dates,
        target_huc02=target_huc02,
    )
    members = _validate_hydrograph_members(hydrograph_ds, hydrograph_uri)
    if evaluation_mode == "single_member_mean" and (
        members is None or int(members.shape[0]) < 2
    ):
        n_found = 0 if members is None else int(members.shape[0])
        raise ValueError(
            f"evaluation_mode='single_member_mean' requires 'qsim_members' with at least 2 "
            f"members in '{hydrograph_uri}', found {n_found}. The stored 'qsim' (possibly an "
            f"ensemble mean) is never substituted for a single-seed score."
        )
    is_ensemble = _resolve_is_ensemble(hydrograph_ds, hydrograph_uri, members)
    if is_ensemble and members is None:
        declared = hydrograph_ds.attrs.get("num_members")
        LOGGER.warning(
            "Store '%s' declares is_ensemble=True (num_members=%r) but carries no "
            "'qsim_members'; per-seed statistics cannot be computed and the ensemble "
            "claim cannot be verified.",
            hydrograph_uri,
            declared,
        )

    pred_inf_mask = np.isinf(sim)
    if np.any(pred_inf_mask):
        inf_count = int(np.sum(pred_inf_mask))
        raise ValueError(
            f"Non-finite prediction violation in '{hydrograph_uri}': found {inf_count} "
            f"infinite (+/-Inf) values in 'qsim'."
        )

    obs_valid = (~np.isnan(obs)) & (obs >= 0.0)
    pred_nan_mask = obs_valid & np.isnan(sim)
    pred_nan_when_obs_valid = int(np.sum(pred_nan_mask))
    n_valid_per_basin = np.sum(obs_valid, axis=1)
    n_pred_nan_per_basin = np.sum(pred_nan_mask, axis=1)
    excluded_basin_mask = np.zeros(NUM_BASINS, dtype=bool)
    if pred_nan_when_obs_valid > 0:
        affected_basins = int(np.sum(n_pred_nan_per_basin > 0))
        missing_basin_mask = (n_pred_nan_per_basin > 0) & (
            n_pred_nan_per_basin == n_valid_per_basin
        )
        partial_basin_mask = (n_pred_nan_per_basin > 0) & (
            n_pred_nan_per_basin < n_valid_per_basin
        )
        if not allow_partial_archive:
            raise ValueError(
                f"Anti-masking / missing-basin violation in '{hydrograph_uri}': "
                f"pred_nan_when_obs_valid = {pred_nan_when_obs_valid} predicted NaN values "
                f"on valid observation days across {affected_basins} basin(s) "
                f"({int(np.sum(missing_basin_mask))} completely missing basin(s)). No missing "
                f"basins or silent NaN masking are allowed."
            )
        if np.any(partial_basin_mask):
            sample = [str(b) for b in target_basins[partial_basin_mask][:5]]
            raise ValueError(
                f"Partial-archive violation in '{hydrograph_uri}': "
                f"{int(np.sum(partial_basin_mask))} basin(s) are only partially predicted on "
                f"valid observation days (e.g. {sample}). allow_partial_archive permits "
                f"whole-basin exclusion only, never within-basin masking."
            )
        excluded_basin_mask = missing_basin_mask

    eval_valid = obs_valid & ~excluded_basin_mask[:, np.newaxis]
    if not np.all(np.isfinite(sim[eval_valid])):
        raise ValueError(
            f"Non-finite prediction violation in '{hydrograph_uri}' on valid observation days."
        )

    if members is not None:
        member_info = _check_members_against_qsim(
            sim=sim,
            members=members,
            obs_valid=eval_valid,
            is_ensemble=is_ensemble,
            reduction=_resolve_ensemble_reduction(hydrograph_ds, hydrograph_uri),
            hydrograph_uri=hydrograph_uri,
        )
    else:
        member_info = {
            "num_members": 1,
            "ensemble_source": "stored_qsim_without_members",
            "single_seed_member_index": None,
        }

    cnt = np.sum(eval_valid, axis=1)
    o_clean = np.where(eval_valid, obs, 0.0)
    cnt_safe = np.maximum(cnt, 1)
    o_mean = np.sum(o_clean, axis=1) / cnt_safe
    o_diff = np.where(eval_valid, obs - o_mean[:, None], 0.0)
    sst = np.sum(o_diff**2, axis=1)

    good_b = (cnt >= MIN_VALID_OBS_DAYS) & (sst > MIN_SST_THRESHOLD)
    num_evaluated_basins = int(np.sum(good_b))
    if num_evaluated_basins == 0:
        raise ValueError(
            f"Zero basins in '{targets_uri}' satisfied >= {MIN_VALID_OBS_DAYS} valid "
            f"observations and SST > {MIN_SST_THRESHOLD}."
        )

    # Compute all 14 Gauch et al. (2023) metrics per evaluable basin via metrics/metrics.py
    per_basin_vectors: dict[str, np.ndarray] = {
        m: np.full(NUM_BASINS, np.nan, dtype=np.float64)
        for m in GAUCH_2023_METRICS
    }
    dates_ns = target_dates.astype("datetime64[ns]")
    obs_for_da = np.where(obs_valid, obs, np.nan)

    if evaluation_mode == "single_member_mean" and members is not None:
        is_ensemble = False
        n_m = int(members.shape[0])
        member_info["num_members"] = n_m
        member_info["ensemble_source"] = "mean_across_individual_members"
        member_info["single_seed_member_index"] = None

        member_per_basin: list[dict[str, np.ndarray]] = []
        member_distributions: list[dict[str, dict[str, float]]] = []
        member_nse_le_zero_counts: list[int] = []
        member_regional_huc02: list[dict[str, float]] = []

        for m_idx in range(n_m):
            m_vecs: dict[str, np.ndarray] = {
                m: np.full(NUM_BASINS, np.nan, dtype=np.float64)
                for m in GAUCH_2023_METRICS
            }
            for b_idx in np.where(good_b)[0]:
                m_row = members[m_idx, b_idx]
                basin_valid = eval_valid[b_idx]
                n_missing = int(np.sum(np.isnan(m_row) & basin_valid))
                if n_missing == int(np.sum(basin_valid)):
                    # Member entirely absent for this basin (declared partial-member suite).
                    continue
                if n_missing > 0:
                    raise ValueError(
                        f"Member {m_idx} in '{hydrograph_uri}' is partially missing on "
                        f"{n_missing} valid observation days of basin {target_basins[b_idx]}; "
                        f"within-basin masking is not allowed."
                    )
                obs_da = xr.DataArray(
                    obs_for_da[b_idx], coords={"date": dates_ns}, dims=("date",)
                )
                sim_da = xr.DataArray(
                    m_row, coords={"date": dates_ns}, dims=("date",)
                )
                basin_metrics = calculate_all_metrics(
                    obs_da, sim_da, resolution="1D", datetime_coord="date"
                )
                for m_name in GAUCH_2023_METRICS:
                    m_vecs[m_name][b_idx] = float(basin_metrics[m_name])
            m_vecs["abs_fhv"] = np.abs(m_vecs["FHV"])
            m_vecs["abs_fms"] = np.abs(m_vecs["FMS"])
            m_vecs["abs_flv"] = np.abs(m_vecs["FLV"])
            member_per_basin.append(m_vecs)

            m_dist: dict[str, dict[str, float]] = {}
            for m_name in GAUCH_2023_METRICS + ("abs_fhv", "abs_fms", "abs_flv"):
                ndigits = (
                    2
                    if m_name
                    in ("FHV", "FMS", "FLV", "abs_fhv", "abs_fms", "abs_flv")
                    else 4
                )
                m_dist[m_name] = _summarize_metric_vector(
                    m_vecs[m_name][good_b], ndigits=ndigits
                )
            member_distributions.append(m_dist)

            m_valid_nses = m_vecs["NSE"][good_b]
            m_valid_nses = m_valid_nses[~np.isnan(m_valid_nses)]
            member_nse_le_zero_counts.append(int(np.sum(m_valid_nses <= 0.0)))

            m_huc: dict[str, float] = {}
            for h in sorted(set(target_huc02)):
                h_mask = (target_huc02 == h) & good_b
                if np.any(h_mask):
                    h_nses = m_vecs["NSE"][h_mask]
                    h_nses = h_nses[~np.isnan(h_nses)]
                    if len(h_nses) > 0:
                        m_huc[str(h)] = float(np.median(h_nses))
            member_regional_huc02.append(m_huc)

        for m_name in GAUCH_2023_METRICS + ("abs_fhv", "abs_fms", "abs_flv"):
            stacked = np.stack([mpb[m_name] for mpb in member_per_basin], axis=0)
            # Mean over the members that score each basin (a member absent for a whole
            # basin contributes nothing; partial within-basin members were rejected above).
            per_basin_vectors[m_name] = _mean_over_finite_members(stacked)

        single_seed_per_member = [
            round(float(md["NSE"]["median"]), 4) for md in member_distributions
        ]
        single_seed_mean = round(float(np.mean(single_seed_per_member)), 4)
        single_seed_std = round(float(np.std(single_seed_per_member)), 4)

        stat_keys = (
            "median",
            "mean",
            "min",
            "p01",
            "p05",
            "p10",
            "q25",
            "q75",
            "p90",
            "p95",
            "p99",
            "max",
        )
        gauch_distributions = {}
        for m_name in GAUCH_2023_METRICS:
            ndigits = 2 if m_name in ("FHV", "FMS", "FLV") else 4
            agg_d: dict[str, float] = {}
            for sk in stat_keys:
                vals = [
                    md[m_name][sk]
                    for md in member_distributions
                    if np.isfinite(md[m_name][sk])
                ]
                agg_d[sk] = (
                    round(float(np.mean(vals)), ndigits) if vals else float("nan")
                )
            med_vals = [
                md[m_name]["median"]
                for md in member_distributions
                if np.isfinite(md[m_name]["median"])
            ]
            agg_d["median_std"] = (
                round(float(np.std(med_vals)), ndigits) if med_vals else float("nan")
            )
            gauch_distributions[m_name] = agg_d

        def _agg_aux_dist(aux_key: str) -> dict[str, float]:
            out_d: dict[str, float] = {}
            for sk in stat_keys:
                vals = [
                    md[aux_key][sk]
                    for md in member_distributions
                    if np.isfinite(md[aux_key][sk])
                ]
                out_d[sk] = round(float(np.mean(vals)), 2) if vals else float("nan")
            return out_d

        abs_fhv_d = _agg_aux_dist("abs_fhv")
        abs_fms_d = _agg_aux_dist("abs_fms")
        abs_flv_d = _agg_aux_dist("abs_flv")

        nse_le_zero_agg = int(round(float(np.mean(member_nse_le_zero_counts))))

        regional_huc02 = {}
        for h in sorted(set(target_huc02)):
            h_vals_m = [
                mh[str(h)] for mh in member_regional_huc02 if str(h) in mh
            ]
            if h_vals_m:
                regional_huc02[str(h)] = round(float(np.mean(h_vals_m)), 4)
    else:
        for b_idx in np.where(good_b)[0]:
            obs_da = xr.DataArray(
                obs_for_da[b_idx],
                coords={"date": dates_ns},
                dims=("date",),
            )
            sim_da = xr.DataArray(
                sim[b_idx],
                coords={"date": dates_ns},
                dims=("date",),
            )
            basin_metrics = calculate_all_metrics(
                obs_da,
                sim_da,
                resolution="1D",
                datetime_coord="date",
            )
            for m_name in GAUCH_2023_METRICS:
                per_basin_vectors[m_name][b_idx] = float(basin_metrics[m_name])

        per_basin_vectors["abs_fhv"] = np.abs(per_basin_vectors["FHV"])
        per_basin_vectors["abs_fms"] = np.abs(per_basin_vectors["FMS"])
        per_basin_vectors["abs_flv"] = np.abs(per_basin_vectors["FLV"])

        # Single-seed skill: median NSE of each individual member, summarized as mean +/- std
        # (population std, ddof=0) across members. For a store without members this is the
        # score of the stored hydrograph itself when it is a single model, and undefined (None)
        # when the store claims to be an ensemble mean without shipping its members.
        if members is not None:
            single_seed_per_member = _per_member_median_nse(
                members, obs_for_da, dates_ns, good_b
            )
        elif not is_ensemble:
            stored_nse = per_basin_vectors["NSE"][good_b]
            stored_nse = stored_nse[np.isfinite(stored_nse)]
            single_seed_per_member = [float(np.median(stored_nse))]
        else:
            single_seed_per_member = None

        if single_seed_per_member is not None:
            single_seed_mean = round(float(np.mean(single_seed_per_member)), 4)
            single_seed_std = round(float(np.std(single_seed_per_member)), 4)
            single_seed_per_member = [round(v, 4) for v in single_seed_per_member]
        else:
            single_seed_mean = None
            single_seed_std = None

        gauch_distributions = {}
        for m_name in GAUCH_2023_METRICS:
            ndigits = 2 if m_name in ("FHV", "FMS", "FLV") else 4
            gauch_distributions[m_name] = _summarize_metric_vector(
                per_basin_vectors[m_name][good_b],
                ndigits=ndigits,
            )

        abs_fhv_d = _summarize_metric_vector(
            per_basin_vectors["abs_fhv"][good_b], ndigits=2
        )
        abs_fms_d = _summarize_metric_vector(
            per_basin_vectors["abs_fms"][good_b], ndigits=2
        )
        abs_flv_d = _summarize_metric_vector(
            per_basin_vectors["abs_flv"][good_b], ndigits=2
        )

        valid_nses = per_basin_vectors["NSE"][good_b]
        valid_nses = valid_nses[~np.isnan(valid_nses)]
        nse_le_zero_agg = int(np.sum(valid_nses <= 0.0))

        regional_huc02 = {}
        for h in sorted(set(target_huc02)):
            h_mask = (target_huc02 == h) & good_b
            if np.any(h_mask):
                h_nses = per_basin_vectors["NSE"][h_mask]
                h_nses = h_nses[~np.isnan(h_nses)]
                if len(h_nses) > 0:
                    regional_huc02[str(h)] = round(float(np.median(h_nses)), 4)

    if not regional_huc02:
        raise ValueError("No valid HUC-02 regions found across evaluated basins.")

    huc_vals = list(regional_huc02.values())
    worst_huc = min(regional_huc02.items(), key=lambda kv: kv[1])
    best_huc = max(regional_huc02.items(), key=lambda kv: kv[1])

    nse_d = gauch_distributions["NSE"]
    kge_d = gauch_distributions["KGE"]
    log_nse_d = gauch_distributions["logNSE"]
    log_kge_d = gauch_distributions["logKGE"]
    pr_d = gauch_distributions["Pearson-r"]
    mse_d = gauch_distributions["MSE"]
    rmse_d = gauch_distributions["RMSE"]
    alpha_d = gauch_distributions["Alpha-NSE"]
    beta_kge_d = gauch_distributions["Beta-KGE"]
    beta_nse_d = gauch_distributions["Beta-NSE"]
    fhv_d = gauch_distributions["FHV"]
    fms_d = gauch_distributions["FMS"]
    flv_d = gauch_distributions["FLV"]
    pt_d = gauch_distributions["Peak-Timing"]

    summary: dict[str, Any] = {
        "num_total_basins": NUM_BASINS,
        "num_evaluated_basins": num_evaluated_basins,
        "num_eval_days": NUM_EVAL_DAYS,
        "pred_nan_when_obs_valid": pred_nan_when_obs_valid,
        # Whole basins excluded under an explicit allow_partial_archive=True (never masked days)
        "allow_partial_archive": bool(allow_partial_archive),
        "num_excluded_basins": int(np.sum(excluded_basin_mask)),
        "excluded_basins": [str(b) for b in target_basins[excluded_basin_mask]],
        # Ensemble / single-seed bookkeeping (never inferred from names or claims)
        "is_ensemble": bool(is_ensemble),
        "num_members": int(member_info["num_members"]),
        "has_qsim_members": members is not None,
        "ensemble_source": member_info["ensemble_source"],
        "single_seed_member_index": member_info["single_seed_member_index"],
        "single_seed_median_nse_mean": single_seed_mean,
        "single_seed_median_nse_std": single_seed_std,
        "single_seed_median_nse_per_member": single_seed_per_member,
        # 5 required headline leaderboard metrics
        "median_nse": nse_d["median"],
        "mean_nse": nse_d["mean"],
        "median_kge": kge_d["median"],
        "median_abs_fhv": abs_fhv_d["median"],
        "nse_le_zero_count": nse_le_zero_agg,
        # Lower-tail & quartile NSE distribution
        "min_nse": nse_d["min"],
        "p01_nse": nse_d["p01"],
        "p05_nse": nse_d["p05"],
        "p10_nse": nse_d["p10"],
        "q25_nse": nse_d["q25"],
        "p25_nse": nse_d["q25"],
        "p50_nse": nse_d["median"],
        "q75_nse": nse_d["q75"],
        "p75_nse": nse_d["q75"],
        "max_nse": nse_d["max"],
        # Lower-tail & quartile KGE distribution
        "min_kge": kge_d["min"],
        "p01_kge": kge_d["p01"],
        "p05_kge": kge_d["p05"],
        "p10_kge": kge_d["p10"],
        "q25_kge": kge_d["q25"],
        "p25_kge": kge_d["q25"],
        "p50_kge": kge_d["median"],
        "q75_kge": kge_d["q75"],
        "p75_kge": kge_d["q75"],
        "max_kge": kge_d["max"],
        "mean_kge": kge_d["mean"],
        # Log-space efficiencies (low-flow emphasis): 531-basin median and quartiles
        "median_log_nse": log_nse_d["median"],
        "q25_log_nse": log_nse_d["q25"],
        "q75_log_nse": log_nse_d["q75"],
        "median_log_kge": log_kge_d["median"],
        "q25_log_kge": log_kge_d["q25"],
        "q75_log_kge": log_kge_d["q75"],
        # Upper-tail peak flow bias (|FHV| %) distribution
        "min_abs_fhv": abs_fhv_d["min"],
        "q25_abs_fhv": abs_fhv_d["q25"],
        "p50_abs_fhv": abs_fhv_d["median"],
        "p75_abs_fhv": abs_fhv_d["q75"],
        "q75_abs_fhv": abs_fhv_d["q75"],
        "p90_abs_fhv": abs_fhv_d["p90"],
        "p95_abs_fhv": abs_fhv_d["p95"],
        "p99_abs_fhv": abs_fhv_d["p99"],
        "max_abs_fhv": abs_fhv_d["max"],
        "mean_abs_fhv": abs_fhv_d["mean"],
        # Additional Gauch et al. (2023) summary scalars
        "median_mse": mse_d["median"],
        "q25_mse": mse_d["q25"],
        "q75_mse": mse_d["q75"],
        "median_rmse": rmse_d["median"],
        "q25_rmse": rmse_d["q25"],
        "q75_rmse": rmse_d["q75"],
        "median_alpha_nse": alpha_d["median"],
        "q25_alpha_nse": alpha_d["q25"],
        "q75_alpha_nse": alpha_d["q75"],
        "median_pearson_r": pr_d["median"],
        "q25_pearson_r": pr_d["q25"],
        "q75_pearson_r": pr_d["q75"],
        "median_beta_kge": beta_kge_d["median"],
        "q25_beta_kge": beta_kge_d["q25"],
        "q75_beta_kge": beta_kge_d["q75"],
        "median_beta_nse": beta_nse_d["median"],
        "q25_beta_nse": beta_nse_d["q25"],
        "q75_beta_nse": beta_nse_d["q75"],
        "median_fhv": fhv_d["median"],
        "q25_fhv": fhv_d["q25"],
        "q75_fhv": fhv_d["q75"],
        "median_fms": fms_d["median"],
        "q25_fms": fms_d["q25"],
        "q75_fms": fms_d["q75"],
        "median_abs_fms": abs_fms_d["median"],
        "q25_abs_fms": abs_fms_d["q25"],
        "q75_abs_fms": abs_fms_d["q75"],
        "median_flv": flv_d["median"],
        "q25_flv": flv_d["q25"],
        "q75_flv": flv_d["q75"],
        "median_abs_flv": abs_flv_d["median"],
        "q25_abs_flv": abs_flv_d["q25"],
        "q75_abs_flv": abs_flv_d["q75"],
        "median_peak_timing": pt_d["median"],
        "q25_peak_timing": pt_d["q25"],
        "q75_peak_timing": pt_d["q75"],
        # Regional HUC-02 summary statistics
        "median_of_18_huc02_medians": round(float(np.median(huc_vals)), 4),
        "worst_huc02_region": f"{worst_huc[1]:.3f} ({worst_huc[0]})",
        "best_huc02_region": f"{best_huc[1]:.3f} ({best_huc[0]})",
        "gauch_2023_metrics": gauch_distributions,
    }

    per_basin: dict[str, Any] = {
        "basins": [str(b) for b in target_basins],
        "huc_02": [str(h) for h in target_huc02],
        "is_evaluable": [bool(v) for v in good_b],
    }
    for m_name in GAUCH_2023_METRICS:
        ndigits = 4 if m_name in ("FHV", "FMS", "FLV") else 6
        vec_list = [
            float("nan") if np.isnan(v) else round(float(v), ndigits)
            for v in per_basin_vectors[m_name]
        ]
        per_basin[m_name] = vec_list
        per_basin[GAUCH_METRIC_TO_SNAKE[m_name]] = vec_list

    for abs_key in ("abs_fhv", "abs_fms", "abs_flv"):
        per_basin[abs_key] = [
            float("nan") if np.isnan(v) else round(float(v), 4)
            for v in per_basin_vectors[abs_key]
        ]

    return {
        "model_id": model_id,
        "leaderboard_id": leaderboard_id,
        "bucket_uri": root_uri,
        "hydrograph_uri": hydrograph_uri,
        "targets_uri": targets_uri,
        "available_gauch_2023_metrics": get_available_metrics(),
        **summary,
        "summary": summary,
        "regional_huc02": regional_huc02,
        "per_basin": per_basin,
        "attributes": dict(hydrograph_ds.attrs),
    }
