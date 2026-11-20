"""Regression tests for the evaluator's anti-masking and identifier guardrails.

Each test pins one fixed defect from the pre-launch architecture audit:

* C3a: `allow_partial_archive=True` permits whole-basin exclusion only, never within-basin
  masking, and reports the excluded basins.
* C3b: `evaluation_mode="single_member_mean"` never relabels a stored (ensemble-mean)
  hydrograph as a single-seed score when the store has fewer than two members.
* C3c: a member that is only partially missing inside a basin is rejected in
  `single_member_mean` mode instead of being silently masked.
* H4: store identifiers cannot traverse out of `hydrographs/<leaderboard_id>/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metrics.evaluator import calculate_model_leaderboard_metrics
from metrics.tests.test_evaluator import (
    _build_synthetic_qobs,
    _canonical_dates,
    _load_canonical_basins_and_huc02,
    _write_hydrograph_zarr,
    _write_targets_zarr,
)

LB_ID = "gauged__single_camels"


def _bucket_with_targets(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins).astype(np.float64)
    _write_targets_zarr(root, qobs, basins, dates, huc_02)
    return qobs, basins, dates, huc_02


def _members(qobs: np.ndarray, n_members: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [qobs * (1.0 + 0.1 * rng.standard_normal(qobs.shape)) for _ in range(n_members)]
    ).astype(np.float32)


@pytest.mark.unit
def test_partial_archive_allows_whole_basin_exclusion_only(tmp_path: Path) -> None:
    """C3a: whole missing basins are excluded and reported; partial basins always raise."""
    qobs, basins, dates, huc_02 = _bucket_with_targets(tmp_path)

    whole = qobs.copy()
    whole[[3, 17], :] = np.nan
    _write_hydrograph_zarr(tmp_path, LB_ID, "whole_basin_gaps", whole, basins, dates, huc_02)
    with pytest.raises(ValueError, match="Anti-masking"):
        calculate_model_leaderboard_metrics("whole_basin_gaps", LB_ID, bucket_uri=str(tmp_path))
    res = calculate_model_leaderboard_metrics(
        "whole_basin_gaps", LB_ID, bucket_uri=str(tmp_path), allow_partial_archive=True
    )
    assert res["num_evaluated_basins"] == 529
    assert res["num_excluded_basins"] == 2
    assert res["excluded_basins"] == [str(basins[3]), str(basins[17])]
    assert res["pred_nan_when_obs_valid"] == 2 * 3652
    assert res["allow_partial_archive"] is True
    assert np.isnan(res["per_basin"]["NSE"][3]) and np.isnan(res["per_basin"]["NSE"][17])

    partial = qobs.copy()
    partial[:, :1000] = np.nan
    _write_hydrograph_zarr(tmp_path, LB_ID, "within_basin_gaps", partial, basins, dates, huc_02)
    with pytest.raises(ValueError, match="whole-basin exclusion only"):
        calculate_model_leaderboard_metrics(
            "within_basin_gaps", LB_ID, bucket_uri=str(tmp_path), allow_partial_archive=True
        )


@pytest.mark.unit
def test_single_member_mean_never_relabels_an_ensemble_store(tmp_path: Path) -> None:
    """C3b: single_member_mean raises without >= 2 members instead of reusing `qsim`."""
    qobs, basins, dates, huc_02 = _bucket_with_targets(tmp_path)
    members = _members(qobs, 5)
    ens_mean = members.astype(np.float64).mean(axis=0)
    _write_hydrograph_zarr(
        tmp_path,
        LB_ID,
        "ensemble_without_members",
        ens_mean,
        basins,
        dates,
        huc_02,
        extra_attrs={"is_ensemble": True, "num_members": 5},
    )
    default = calculate_model_leaderboard_metrics(
        "ensemble_without_members", LB_ID, bucket_uri=str(tmp_path)
    )
    assert default["is_ensemble"] is True
    assert default["single_seed_median_nse_mean"] is None
    with pytest.raises(ValueError, match="at least 2 members"):
        calculate_model_leaderboard_metrics(
            "ensemble_without_members",
            LB_ID,
            bucket_uri=str(tmp_path),
            evaluation_mode="single_member_mean",
        )

    _write_hydrograph_zarr(
        tmp_path,
        LB_ID,
        "one_member",
        members[0],
        basins,
        dates,
        huc_02,
        qsim_members=members[:1],
        extra_attrs={"is_ensemble": False},
    )
    with pytest.raises(ValueError, match="at least 2 members"):
        calculate_model_leaderboard_metrics(
            "one_member", LB_ID, bucket_uri=str(tmp_path), evaluation_mode="single_member_mean"
        )


@pytest.mark.unit
def test_single_member_mean_rejects_partially_missing_members(tmp_path: Path) -> None:
    """C3c: a member partially missing inside a basin raises; a whole absent basin is skipped."""
    qobs, basins, dates, huc_02 = _bucket_with_targets(tmp_path)
    members = _members(qobs, 4).astype(np.float64)

    absent = members.copy()
    absent[0, 7, :] = np.nan  # member 0 never ran basin 7 (declared partial-member suite)
    _write_hydrograph_zarr(
        tmp_path,
        LB_ID,
        "member_absent_for_basin",
        _nanmean(absent),
        basins,
        dates,
        huc_02,
        qsim_members=absent,
        extra_attrs={"is_ensemble": True, "ensemble_reduction": "nanmean"},
    )
    ok = calculate_model_leaderboard_metrics(
        "member_absent_for_basin",
        LB_ID,
        bucket_uri=str(tmp_path),
        evaluation_mode="single_member_mean",
    )
    assert ok["pred_nan_when_obs_valid"] == 0
    assert np.isfinite(ok["per_basin"]["NSE"][7])

    partial = members.copy()
    partial[0, 7, 100:1900] = np.nan  # member 0 missing 1,800 valid days of basin 7
    _write_hydrograph_zarr(
        tmp_path,
        LB_ID,
        "member_partial_in_basin",
        _nanmean(partial),
        basins,
        dates,
        huc_02,
        qsim_members=partial,
        extra_attrs={"is_ensemble": True, "ensemble_reduction": "nanmean"},
    )
    with pytest.raises(ValueError, match="within-basin masking is not allowed"):
        calculate_model_leaderboard_metrics(
            "member_partial_in_basin",
            LB_ID,
            bucket_uri=str(tmp_path),
            evaluation_mode="single_member_mean",
        )


def _nanmean(stack: np.ndarray) -> np.ndarray:
    finite = np.isfinite(stack)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    count = finite.sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


@pytest.mark.unit
def test_store_identifiers_cannot_traverse_the_bucket(tmp_path: Path) -> None:
    """H4: `model_id` values with path components are rejected before any I/O."""
    qobs, basins, dates, huc_02 = _bucket_with_targets(tmp_path)
    _write_hydrograph_zarr(tmp_path, "gauged__multi_camels", "victim", qobs, basins, dates, huc_02)
    for bad in ("../gauged__multi_camels/victim", "../../observations/newman_eval_targets", "a b"):
        with pytest.raises(ValueError, match="not a valid identifier"):
            calculate_model_leaderboard_metrics(bad, LB_ID, bucket_uri=str(tmp_path))
