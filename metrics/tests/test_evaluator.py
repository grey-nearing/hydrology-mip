"""Native-resolution (531 x 3652) unit and data-integrity tests for `metrics/`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import FsspecStore

from metrics import (
    CANONICAL_LEADERBOARD_IDS,
    EVAL_END_DATE,
    EVAL_START_DATE,
    GAUCH_2023_METRICS,
    NUM_BASINS,
    NUM_EVAL_DAYS,
    AllNaNError,
    calculate_all_metrics,
    calculate_metrics,
    calculate_model_leaderboard_metrics,
    get_available_metrics,
    get_frequency_factor,
    infer_datetime_coord,
)
from metrics.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_canonical_basins_and_huc02() -> tuple[np.ndarray, np.ndarray]:
    """Load the real 531 sorted USGS basin IDs and 2-digit HUC-02 region codes."""
    spec_path = REPO_ROOT / "splits" / "splits_and_regimes.json"
    assert spec_path.exists(), f"Missing canonical split specification at {spec_path}"

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    basin_to_huc02: dict[str, str] = spec["basin_to_huc02"]
    basins = np.array(sorted(basin_to_huc02.keys()), dtype="<U8")
    huc_02 = np.array([basin_to_huc02[str(b)] for b in basins], dtype="<U2")
    assert basins.shape == (NUM_BASINS,)
    assert huc_02.shape == (NUM_BASINS,)
    return basins, huc_02


def _canonical_dates() -> np.ndarray:
    """Return the 3,652 daily timestamps over 1989-10-01 to 1999-09-30."""
    dates = pd.date_range(EVAL_START_DATE, EVAL_END_DATE, freq="D").values.astype(
        "datetime64[ns]"
    )
    assert dates.shape == (NUM_EVAL_DAYS,)
    return dates


def _write_targets_zarr(
    bucket_root: Path,
    qobs: np.ndarray,
    basins: np.ndarray,
    dates: np.ndarray,
    huc_02: np.ndarray,
) -> Path:
    """Write a consolidated target Zarr store at `<bucket_root>/observations/newman_eval_targets.zarr`."""
    store_path = bucket_root / "observations" / "newman_eval_targets.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "qobs": (
                ("basin", "date"),
                qobs.astype(np.float32),
                {"units": "mm/day"},
            ),
        },
        coords={
            "basin": ("basin", basins.astype("<U8")),
            "date": ("date", dates.astype("datetime64[ns]")),
            "huc_02": ("basin", huc_02.astype("<U2")),
        },
        attrs={
            "evaluation_period": f"{EVAL_START_DATE} to {EVAL_END_DATE}",
            "units": "mm/day",
        },
    )
    ds.to_zarr(store_path, mode="w", consolidated=True, zarr_format=2)
    return store_path


def _write_hydrograph_zarr(
    bucket_root: Path,
    leaderboard_id: str,
    model_id: str,
    qsim: np.ndarray,
    basins: np.ndarray,
    dates: np.ndarray,
    huc_02: np.ndarray | None = None,
    qsim_members: np.ndarray | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write a consolidated hydrograph Zarr store at `<bucket_root>/hydrographs/<leaderboard_id>/<model_id>.zarr`."""
    store_path = bucket_root / "hydrographs" / leaderboard_id / f"{model_id}.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    coords: dict[str, Any] = {
        "basin": ("basin", basins.astype("<U8")),
        "date": ("date", dates.astype("datetime64[ns]")),
    }
    if huc_02 is not None:
        coords["huc_02"] = ("basin", huc_02.astype("<U2"))

    data_vars: dict[str, Any] = {
        "qsim": (
            ("basin", "date"),
            qsim.astype(np.float32),
            {"units": "mm/day"},
        ),
    }
    if qsim_members is not None:
        data_vars["qsim_members"] = (
            ("member", "basin", "date"),
            qsim_members.astype(np.float32),
            {"units": "mm/day"},
        )
        coords["member"] = ("member", np.arange(qsim_members.shape[0], dtype=np.int64))

    attrs: dict[str, Any] = {
        "model_id": model_id,
        "leaderboard_id": leaderboard_id,
        "evaluation_period": f"{EVAL_START_DATE} to {EVAL_END_DATE}",
        "units": "mm/day",
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds.to_zarr(store_path, mode="w", consolidated=True, zarr_format=2)
    return store_path


def _build_synthetic_qobs(basins: np.ndarray) -> np.ndarray:
    """Generate deterministic positive daily streamflow observations of shape (531, 3652)."""
    t = np.linspace(0.0, 20.0 * np.pi, NUM_EVAL_DAYS, dtype=np.float64)
    basin_idx = np.arange(len(basins), dtype=np.float64)[:, None]
    seasonal = 2.0 + 1.5 * np.sin(t[None, :] + 0.05 * basin_idx)
    peaks = np.where((np.arange(NUM_EVAL_DAYS)[None, :] % 45) == 0, 14.0, 0.0)
    scale = 1.0 + 0.002 * basin_idx
    qobs = (scale * (seasonal + peaks)).astype(np.float32)
    return qobs


@pytest.mark.unit
def test_gauch_2023_all_12_metrics_and_utilities() -> None:
    """Verify all 12 Gauch et al. (2023) metrics and helper functions in `metrics/metrics.py`."""
    available = get_available_metrics()
    assert tuple(available) == GAUCH_2023_METRICS
    assert len(available) == 12

    dates = _canonical_dates()
    t = np.linspace(0.0, 20.0 * np.pi, NUM_EVAL_DAYS, dtype=np.float64)
    obs_vals = 2.5 + 1.2 * np.sin(t) + np.where(
        (np.arange(NUM_EVAL_DAYS) % 120) == 50, 15.0, 0.0
    )
    sim_vals = 1.10 * obs_vals

    obs_da = xr.DataArray(obs_vals, coords={"date": dates}, dims=("date",))
    sim_da = xr.DataArray(sim_vals, coords={"date": dates}, dims=("date",))

    assert infer_datetime_coord(obs_da) == "date"
    assert get_frequency_factor("1D", "1D") == 1
    assert get_frequency_factor("1D", "12h") == pytest.approx(2.0)

    all_m = calculate_all_metrics(obs_da, sim_da, resolution="1D")
    assert set(all_m.keys()) == set(GAUCH_2023_METRICS)
    assert all_m["Alpha-NSE"] == pytest.approx(1.10, abs=1e-5)
    assert all_m["Beta-KGE"] == pytest.approx(1.10, abs=1e-5)
    assert all_m["Pearson-r"] == pytest.approx(1.0, abs=1e-5)
    assert all_m["FHV"] == pytest.approx(10.0, abs=1e-4)
    assert all_m["Peak-Timing"] == pytest.approx(0.0, abs=1e-5)

    subset_m = calculate_metrics(obs_da, sim_da, ["nse", "kge"])
    assert set(subset_m.keys()) == {"NSE", "KGE"}

    nan_da = xr.DataArray(
        np.full(NUM_EVAL_DAYS, np.nan), coords={"date": dates}, dims=("date",)
    )
    with pytest.raises(AllNaNError):
        calculate_all_metrics(obs_da, nan_da)


@pytest.mark.unit
def test_perfect_predictions_with_530_valid_basins(tmp_path: Path) -> None:
    """Verify exact metrics when qsim == qobs on 530 valid basins and 1 basin has 0 valid obs.

    This is a purely synthetic edge case exercising the evaluability rule
    (`cnt >= 30 & SST > 1e-12`). The real canonical `newman_eval_targets.zarr` has
    531/531 basins with 100% valid daily observations over 1989-10-01..1999-09-30
    (verified against the raw USGS `usgs_streamflow` files), so on real data all 531
    basins are always evaluated.
    """
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)

    # Synthetic: blank out one basin's observations entirely (not a property of real data)
    unobserved_idx = 250
    qobs[unobserved_idx, :] = np.nan

    # Model predicts exact qobs on valid basins and finite values on the unobserved basin
    qsim = np.where(np.isnan(qobs), 1.0, qobs).astype(np.float32)

    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    _write_hydrograph_zarr(
        tmp_path, "gauged__single_camels", "perfect_model", qsim, basins, dates, huc_02
    )

    result = calculate_model_leaderboard_metrics(
        model_id="perfect_model",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
    )

    assert result["num_total_basins"] == 531
    assert result["num_evaluated_basins"] == 530
    assert result["num_eval_days"] == 3652
    assert result["pred_nan_when_obs_valid"] == 0

    assert result["median_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["mean_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["min_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["p01_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["p05_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["p10_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["q25_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["q75_nse"] == pytest.approx(1.0, abs=1e-4)

    assert result["median_kge"] == pytest.approx(1.0, abs=1e-4)
    assert result["mean_kge"] == pytest.approx(1.0, abs=1e-4)
    assert result["min_kge"] == pytest.approx(1.0, abs=1e-4)

    assert result["median_abs_fhv"] == pytest.approx(0.0, abs=1e-4)
    assert result["p50_abs_fhv"] == pytest.approx(0.0, abs=1e-4)
    assert result["p95_abs_fhv"] == pytest.approx(0.0, abs=1e-4)
    assert result["max_abs_fhv"] == pytest.approx(0.0, abs=1e-4)
    assert result["nse_le_zero_count"] == 0

    assert set(result["gauch_2023_metrics"].keys()) == set(GAUCH_2023_METRICS)
    assert result["median_peak_timing"] == pytest.approx(0.0, abs=1e-4)

    assert len(result["regional_huc02"]) == 18
    for huc_code, med_val in result["regional_huc02"].items():
        assert len(huc_code) == 2
        assert med_val == pytest.approx(1.0, abs=1e-4)

    assert np.isnan(result["per_basin"]["nse"][unobserved_idx])
    assert np.isnan(result["per_basin"]["kge"][unobserved_idx])
    assert np.isnan(result["per_basin"]["abs_fhv"][unobserved_idx])


@pytest.mark.unit
def test_exact_numerical_verification_of_nse_kge_fhv_and_failure_count(
    tmp_path: Path,
) -> None:
    """Verify exact closed-form NSE, KGE, |FHV|, and NSE <= 0 counts across all 531 basins."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)

    # Construct qsim:
    # - First 15 basins: constant prediction equal to each basin's mean qobs -> NSE == 0.0 (catastrophic failure)
    # - Remaining 516 basins: qsim = 1.25 * qobs ->
    #   alpha = 1.25, beta = 1.25, r = 1.0 -> KGE = 1 - sqrt(0.25^2 + 0.25^2)
    #   |FHV| = |1.25 - 1.0| * 100 = 25.0%
    qsim = (1.25 * qobs).astype(np.float32)
    obs_means = np.mean(qobs[:15].astype(np.float64), axis=1, keepdims=True)
    qsim[:15, :] = obs_means.astype(np.float32)

    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    _write_hydrograph_zarr(
        tmp_path, "pub__multi_camels", "analytical_model", qsim, basins, dates, huc_02
    )

    result = calculate_model_leaderboard_metrics(
        model_id="analytical_model",
        leaderboard_id="pub__multi_camels",
        bucket_uri=str(tmp_path),
    )

    assert result["num_evaluated_basins"] == 531
    assert result["nse_le_zero_count"] == 15

    for idx in range(15):
        assert result["per_basin"]["nse"][idx] == pytest.approx(0.0, abs=1e-5)

    b15_obs = qobs[15].astype(np.float64)
    b15_sim = qsim[15].astype(np.float64)
    expected_nse_15 = 1.0 - np.sum((b15_obs - b15_sim) ** 2) / np.sum(
        (b15_obs - np.mean(b15_obs)) ** 2
    )
    expected_kge_15 = 1.0 - np.sqrt(0.25**2 + 0.25**2)
    expected_fhv_15 = 25.0

    assert result["per_basin"]["nse"][15] == pytest.approx(expected_nse_15, abs=1e-4)
    assert result["per_basin"]["kge"][15] == pytest.approx(expected_kge_15, abs=1e-4)
    assert result["per_basin"]["abs_fhv"][15] == pytest.approx(
        expected_fhv_15, abs=1e-2
    )
    assert result["median_kge"] == pytest.approx(expected_kge_15, abs=1e-3)
    assert result["median_abs_fhv"] == pytest.approx(25.0, abs=1e-2)

    assert (
        result["min_nse"]
        <= result["p01_nse"]
        <= result["p05_nse"]
        <= result["p10_nse"]
        <= result["q25_nse"]
        <= result["median_nse"]
        <= result["q75_nse"]
        <= result["max_nse"]
    )
    assert (
        result["p50_abs_fhv"]
        <= result["p75_abs_fhv"]
        <= result["p90_abs_fhv"]
        <= result["p95_abs_fhv"]
        <= result["p99_abs_fhv"]
        <= result["max_abs_fhv"]
    )


@pytest.mark.unit
def test_qsim_nan_allowed_only_when_qobs_is_nan(tmp_path: Path) -> None:
    """Verify qsim NaNs on days where qobs is also NaN do not trigger anti-masking errors."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    qsim = qobs.copy()

    # Mark 10 days in basin 0 as missing in BOTH qobs and qsim
    qobs[0, 100:110] = np.nan
    qsim[0, 100:110] = np.nan

    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    _write_hydrograph_zarr(
        tmp_path, "pur__single_camels", "aligned_missing_model", qsim, basins, dates
    )

    result = calculate_model_leaderboard_metrics(
        model_id="aligned_missing_model",
        leaderboard_id="pur__single_camels",
        bucket_uri=str(tmp_path),
    )
    assert result["pred_nan_when_obs_valid"] == 0
    assert result["median_nse"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
def test_missing_basins_or_pred_nan_when_obs_valid_raises_immediately(
    tmp_path: Path,
) -> None:
    """Verify missing basins (all-NaN) or partial NaNs in qsim on valid obs days raise ValueError."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)

    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)

    # 1. Partial NaNs on 3 valid observation days
    qsim_partial = qobs.copy()
    qsim_partial[0, 10] = np.nan
    qsim_partial[42, 500] = np.nan
    qsim_partial[530, 3651] = np.nan
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__multi_camels",
        "masked_nan_model",
        qsim_partial,
        basins,
        dates,
    )
    with pytest.raises(ValueError, match="pred_nan_when_obs_valid = 3"):
        calculate_model_leaderboard_metrics(
            model_id="masked_nan_model",
            leaderboard_id="gauged__multi_camels",
            bucket_uri=str(tmp_path),
        )

    # 2. Completely missing basin (all 3,652 days NaN for 1 evaluable basin)
    qsim_missing_basin = qobs.copy()
    qsim_missing_basin[10, :] = np.nan
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__multi_camels",
        "missing_basin_model",
        qsim_missing_basin,
        basins,
        dates,
    )
    with pytest.raises(
        ValueError, match="1 completely missing basin\\(s\\)"
    ):
        calculate_model_leaderboard_metrics(
            model_id="missing_basin_model",
            leaderboard_id="gauged__multi_camels",
            bucket_uri=str(tmp_path),
        )


@pytest.mark.unit
def test_ensemble_and_single_seed_semantics_from_qsim_members(tmp_path: Path) -> None:
    """Verify `qsim_members` validation and the ensemble vs single-seed bookkeeping.

    - An ensemble store must carry `qsim == mean(qsim_members)`; its single-seed skill is
      the mean +/- std of the per-member 531-basin median NSE and is strictly below the
      ensemble-mean skill for members with independent errors.
    - A single-model store that ships sibling seeds must have `qsim` equal to exactly one
      member, and the matching member index is reported.
    - A clipped / cherry-picked `qsim`, a member with NaN on a valid observation day, a
      wrong member layout, or an `is_ensemble=True` store whose `qsim` matches no mean all
      raise `ValueError`.
    - A store that claims `is_ensemble=True` without shipping members yields `None`
      single-seed statistics (never a copy of the ensemble score).
    """
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)

    rng = np.random.default_rng(20261020)
    n_members = 4
    noise = rng.normal(0.0, 0.6, size=(n_members, NUM_BASINS, NUM_EVAL_DAYS))
    members = (qobs[np.newaxis, :, :].astype(np.float64) + noise).astype(np.float32)
    # Negative predictions are legal input and must be scored as-is (no clipping).
    assert np.any(members < 0.0)
    ens_mean = members.astype(np.float64).mean(axis=0).astype(np.float32)

    # 1. Valid ensemble store
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "ens_model",
        ens_mean,
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": True, "num_members": n_members},
    )
    res = calculate_model_leaderboard_metrics(
        model_id="ens_model", leaderboard_id="gauged__single_camels", bucket_uri=str(tmp_path)
    )
    assert res["is_ensemble"] is True
    assert res["has_qsim_members"] is True
    assert res["num_members"] == n_members
    assert res["ensemble_source"] == "mean_of_qsim_members"
    assert res["single_seed_member_index"] is None
    per_member = res["single_seed_median_nse_per_member"]
    assert len(per_member) == n_members
    assert res["single_seed_median_nse_mean"] == pytest.approx(np.mean(per_member), abs=1e-4)
    assert res["single_seed_median_nse_std"] == pytest.approx(np.std(per_member), abs=1e-4)
    # Averaging independent noise raises skill: the ensemble must beat every member.
    assert res["median_nse"] > max(per_member)
    assert res["single_seed_median_nse_mean"] < res["median_nse"]
    # Per-member medians are the real per-seed scores, not copies of the ensemble score.
    assert all(abs(v - res["median_nse"]) > 1e-3 for v in per_member)

    # 2. Valid single-model store shipping its sibling seeds: qsim == members[2]
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "single_model",
        members[2],
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": False, "num_members": n_members},
    )
    res_single = calculate_model_leaderboard_metrics(
        model_id="single_model",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
    )
    assert res_single["is_ensemble"] is False
    assert res_single["ensemble_source"] == "single_member"
    assert res_single["single_seed_member_index"] == 2
    assert res_single["median_nse"] == pytest.approx(per_member[2], abs=1e-4)
    assert res_single["single_seed_median_nse_per_member"] == pytest.approx(per_member, abs=1e-4)

    # 3. Store without members and without an ensemble claim: single seed == stored score
    _write_hydrograph_zarr(
        tmp_path, "gauged__single_camels", "plain_model", members[0], basins, dates, huc_02
    )
    res_plain = calculate_model_leaderboard_metrics(
        model_id="plain_model", leaderboard_id="gauged__single_camels", bucket_uri=str(tmp_path)
    )
    assert res_plain["is_ensemble"] is False
    assert res_plain["has_qsim_members"] is False
    assert res_plain["num_members"] == 1
    assert res_plain["ensemble_source"] == "stored_qsim_without_members"
    assert res_plain["single_seed_median_nse_mean"] == pytest.approx(
        res_plain["median_nse"], abs=1e-4
    )
    assert res_plain["single_seed_median_nse_std"] == pytest.approx(0.0, abs=1e-9)

    # 4. Ensemble claim without members: single-seed statistics are unknown, not copied
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "claimed_ens_model",
        ens_mean,
        basins,
        dates,
        huc_02,
        extra_attrs={"is_ensemble": True, "num_members": 8},
    )
    res_claim = calculate_model_leaderboard_metrics(
        model_id="claimed_ens_model",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
    )
    assert res_claim["is_ensemble"] is True
    assert res_claim["has_qsim_members"] is False
    assert res_claim["single_seed_median_nse_mean"] is None
    assert res_claim["single_seed_median_nse_std"] is None
    assert res_claim["single_seed_median_nse_per_member"] is None

    # 5. Clipped ensemble mean (qsim = max(mean, 0)) is rejected
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "clipped_ens_model",
        np.maximum(ens_mean, 0.0),
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": True, "num_members": n_members},
    )
    with pytest.raises(ValueError, match="Ensemble consistency violation"):
        calculate_model_leaderboard_metrics(
            model_id="clipped_ens_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 6. Cherry-picked "single" qsim matching no member is rejected
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "cherry_model",
        ens_mean,
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": False, "num_members": n_members},
    )
    with pytest.raises(ValueError, match="Single-model consistency violation"):
        calculate_model_leaderboard_metrics(
            model_id="cherry_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 7. NaN in one member on a valid observation day is rejected even if qsim is complete
    members_nan = members.copy()
    members_nan[1, 7, 123] = np.nan
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "member_nan_model",
        ens_mean,
        basins,
        dates,
        huc_02,
        qsim_members=members_nan,
        extra_attrs={"is_ensemble": True, "num_members": n_members},
    )
    with pytest.raises(ValueError, match="'qsim_members' contains 1 NaN"):
        calculate_model_leaderboard_metrics(
            model_id="member_nan_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 8. Wrong member layout is rejected
    store_path = tmp_path / "hydrographs" / "gauged__single_camels" / "bad_layout_model.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    ds_bad = xr.Dataset(
        data_vars={
            "qsim": (("basin", "date"), ens_mean),
            "qsim_members": (
                ("basin", "member", "date"),
                np.transpose(members, (1, 0, 2)),
            ),
        },
        coords={
            "basin": ("basin", basins.astype("<U8")),
            "date": ("date", dates.astype("datetime64[ns]")),
            "member": ("member", np.arange(n_members, dtype=np.int64)),
        },
        attrs={"is_ensemble": True},
    )
    ds_bad.to_zarr(store_path, mode="w", consolidated=True, zarr_format=2)
    with pytest.raises(ValueError, match="must have dimensions \\('member', 'basin', 'date'\\)"):
        calculate_model_leaderboard_metrics(
            model_id="bad_layout_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 9. A declared median ensemble is checked against the member median (not the mean)
    ens_median = np.median(members.astype(np.float64), axis=0).astype(np.float32)
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "median_ens_model",
        ens_median,
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": True, "ensemble_reduction": "median"},
    )
    res_med = calculate_model_leaderboard_metrics(
        model_id="median_ens_model",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
    )
    assert res_med["ensemble_source"] == "median_of_qsim_members"
    assert res_med["single_seed_median_nse_per_member"] == pytest.approx(per_member, abs=1e-4)

    # 10. An undeclared / unsupported reduction is rejected loudly
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "weird_reduction_model",
        ens_mean,
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": True, "ensemble_reduction": "weighted_blend"},
    )
    with pytest.raises(ValueError, match="Unsupported 'ensemble_reduction'"):
        calculate_model_leaderboard_metrics(
            model_id="weird_reduction_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 11. A partial multi-structure suite (`nanmedian`): whole-basin gaps in some members are
    #     allowed, qsim must equal the nanmedian, and per-member medians skip absent basins.
    members_partial = members.copy()
    members_partial[0, :40, :] = np.nan  # structure 0 never run on the first 40 basins
    members_partial[3, 100:110, :] = np.nan
    nanmed = np.nanmedian(members_partial.astype(np.float64), axis=0).astype(np.float32)
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "nanmedian_suite",
        nanmed,
        basins,
        dates,
        huc_02,
        qsim_members=members_partial,
        extra_attrs={"is_ensemble": True, "ensemble_reduction": "nanmedian"},
    )
    res_nm = calculate_model_leaderboard_metrics(
        model_id="nanmedian_suite",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
    )
    assert res_nm["ensemble_source"] == "nanmedian_of_qsim_members"
    assert len(res_nm["single_seed_median_nse_per_member"]) == n_members
    # The same partial members under a plain `mean` declaration are rejected (anti-masking).
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "partial_mean_model",
        np.nanmean(members_partial.astype(np.float64), axis=0).astype(np.float32),
        basins,
        dates,
        huc_02,
        qsim_members=members_partial,
        extra_attrs={"is_ensemble": True},
    )
    with pytest.raises(ValueError, match="'qsim_members' contains"):
        calculate_model_leaderboard_metrics(
            model_id="partial_mean_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )
    # Within-basin partial gaps are rejected even for nanmedian suites.
    members_within = members.copy()
    members_within[1, 5, 200:260] = np.nan
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "within_basin_gap_suite",
        np.nanmedian(members_within.astype(np.float64), axis=0).astype(np.float32),
        basins,
        dates,
        huc_02,
        qsim_members=members_within,
        extra_attrs={"is_ensemble": True, "ensemble_reduction": "nanmedian"},
    )
    with pytest.raises(ValueError, match="partially missing"):
        calculate_model_leaderboard_metrics(
            model_id="within_basin_gap_suite",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )


@pytest.mark.unit
def test_summary_statistics_exclude_non_finite_metric_values() -> None:
    """`_summarize_metric_vector` must drop NaN and +/-Inf so no aggregate becomes infinite."""
    from metrics.evaluator import _summarize_metric_vector

    values = np.array([1.0, np.nan, np.inf, -np.inf, 3.0, 2.0], dtype=np.float64)
    summary = _summarize_metric_vector(values)
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(3.0)
    assert all(np.isfinite(v) for v in summary.values())

    empty = _summarize_metric_vector(np.array([np.nan, np.inf]))
    assert all(np.isnan(v) for v in empty.values())


@pytest.mark.unit
def test_inf_in_qsim_raises_immediately(tmp_path: Path) -> None:
    """Verify injecting +Inf or -Inf into qsim raises ValueError."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    qsim = qobs.copy()
    qsim[12, 200] = np.inf
    qsim[19, 400] = -np.inf

    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    _write_hydrograph_zarr(
        tmp_path, "gauged__freeform", "inf_model", qsim, basins, dates
    )

    with pytest.raises(ValueError, match="Non-finite prediction violation"):
        calculate_model_leaderboard_metrics(
            model_id="inf_model",
            leaderboard_id="gauged__freeform",
            bucket_uri=str(tmp_path),
        )


@pytest.mark.unit
def test_invalid_arguments_and_missing_stores_fail_loudly(tmp_path: Path) -> None:
    """Verify invalid leaderboard_id, empty model_id, or missing Zarr stores raise immediately."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)

    with pytest.raises(ValueError, match="Invalid leaderboard_id"):
        calculate_model_leaderboard_metrics(
            model_id="m1",
            leaderboard_id="gauged__invalid_leaderboard",
            bucket_uri=str(tmp_path),
        )

    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        calculate_model_leaderboard_metrics(
            model_id="   ",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # Missing targets store
    with pytest.raises(FileNotFoundError, match="newman_eval_targets.zarr"):
        calculate_model_leaderboard_metrics(
            model_id="m1",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # Write targets store, but leave hydrograph store missing
    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    with pytest.raises(FileNotFoundError, match="missing_model.zarr"):
        calculate_model_leaderboard_metrics(
            model_id="missing_model",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )


@pytest.mark.unit
def test_corrupt_shapes_and_scrambled_coordinates_fail_loudly(
    tmp_path: Path,
) -> None:
    """Verify wrong shapes (530, 3652), scrambled basin order, and wrong dates raise ValueError."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)

    # 1. Wrong basin count (530, 3652)
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "wrong_shape_530",
        qobs[:530, :],
        basins[:530],
        dates,
    )
    with pytest.raises(ValueError, match="must have shape \\(531, 3652\\)"):
        calculate_model_leaderboard_metrics(
            model_id="wrong_shape_530",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 2. Scrambled basin order (swap first two basins)
    scrambled_basins = basins.copy()
    scrambled_basins[0], scrambled_basins[1] = (
        scrambled_basins[1],
        scrambled_basins[0],
    )
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "scrambled_basins",
        qobs,
        scrambled_basins,
        dates,
    )
    with pytest.raises(
        ValueError,
        match="does not match canonical sorted 531 USGS basin IDs in exact order",
    ):
        calculate_model_leaderboard_metrics(
            model_id="scrambled_basins",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 3. Wrong date range (shifted by 1 day)
    wrong_dates = pd.date_range("1989-10-02", periods=NUM_EVAL_DAYS, freq="D").values
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "wrong_dates",
        qobs,
        basins,
        wrong_dates,
    )
    with pytest.raises(ValueError, match="does not match canonical evaluation window"):
        calculate_model_leaderboard_metrics(
            model_id="wrong_dates",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )

    # 4. Missing 'qsim' variable in Zarr dataset
    bad_var_path = (
        tmp_path / "hydrographs" / "gauged__single_camels" / "missing_qsim.zarr"
    )
    bad_ds = xr.Dataset(
        data_vars={"wrong_var": (("basin", "date"), qobs)},
        coords={"basin": basins, "date": dates},
    )
    bad_ds.to_zarr(bad_var_path, mode="w", consolidated=True, zarr_format=2)
    with pytest.raises(KeyError, match="missing required 'qsim' variable"):
        calculate_model_leaderboard_metrics(
            model_id="missing_qsim",
            leaderboard_id="gauged__single_camels",
            bucket_uri=str(tmp_path),
        )


@pytest.mark.unit
def test_gcs_uri_resolution_at_storage_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify gs://camels-us-mip URI resolution routed to real Zarr stores at the fsspec/gcsfs boundary."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    _write_hydrograph_zarr(
        tmp_path, "gauged__lagged_streamflow", "gcs_test_model", qobs, basins, dates, huc_02
    )

    class _LocalMappedGCSFileSystem:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def exists(self, uri: str) -> bool:
            prefix = "gs://camels-us-mip/"
            assert uri.startswith(prefix)
            rel = uri[len(prefix) :]
            return (tmp_path / rel).exists()

    orig_from_url = FsspecStore.from_url

    def _mapped_from_url(
        url: str, storage_options: dict[str, Any] | None = None, **kwargs: Any
    ) -> FsspecStore:
        prefix = "gs://camels-us-mip/"
        if url.startswith(prefix):
            local_url = f"file://{tmp_path}/{url[len(prefix):]}"
            return orig_from_url(local_url, storage_options=None, **kwargs)
        return orig_from_url(url, storage_options=storage_options, **kwargs)

    monkeypatch.setattr(gcsfs, "GCSFileSystem", _LocalMappedGCSFileSystem)
    monkeypatch.setattr(FsspecStore, "from_url", _mapped_from_url)

    result = calculate_model_leaderboard_metrics(
        model_id="gcs_test_model",
        leaderboard_id="gauged__lagged_streamflow",
        bucket_uri="gs://camels-us-mip",
    )
    assert result["hydrograph_uri"] == (
        "gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/gcs_test_model.zarr"
    )
    assert result["median_nse"] == pytest.approx(1.0, abs=1e-4)
    assert result["pred_nan_when_obs_valid"] == 0


@pytest.mark.unit
def test_cli_entrypoint_writes_output_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `metrics.cli.main` scores a model and writes the JSON output file."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)
    _write_hydrograph_zarr(
        tmp_path, "pur__freeform", "cli_model", qobs, basins, dates, huc_02
    )

    out_json = tmp_path / "outputs" / "cli_model_metrics.json"
    exit_code = cli_main(
        [
            "--model-id",
            "cli_model",
            "--leaderboard-id",
            "pur__freeform",
            "--bucket-uri",
            str(tmp_path),
            "--output-json",
            str(out_json),
        ]
    )
    assert exit_code == 0
    assert out_json.exists()

    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["model_id"] == "cli_model"
    assert saved["leaderboard_id"] == "pur__freeform"
    assert saved["median_nse"] == pytest.approx(1.0, abs=1e-4)
    assert len(saved["per_basin"]["basins"]) == len(CANONICAL_LEADERBOARD_IDS) * 0 + 531

    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["summary"]["median_nse"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
def test_single_member_mean_evaluation_mode(tmp_path: Path) -> None:
    """Verify evaluation_mode='single_member_mean' averages all 12 Gauch metrics across all N members."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)
    _write_targets_zarr(tmp_path, qobs, basins, dates, huc_02)

    rng = np.random.default_rng(42)
    n_members = 3
    members = np.stack(
        [
            (qobs + rng.normal(0.0, 0.3 * (m + 1), size=qobs.shape)).astype(np.float32)
            for m in range(n_members)
        ],
        axis=0,
    )
    ens_mean = np.mean(members.astype(np.float64), axis=0).astype(np.float32)
    _write_hydrograph_zarr(
        tmp_path,
        "gauged__single_camels",
        "multi_seed_model",
        ens_mean,
        basins,
        dates,
        huc_02,
        qsim_members=members,
        extra_attrs={"is_ensemble": True, "num_members": n_members},
    )

    res_ens = calculate_model_leaderboard_metrics(
        model_id="multi_seed_model",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
    )
    res_single_mean = calculate_model_leaderboard_metrics(
        model_id="multi_seed_model",
        leaderboard_id="gauged__single_camels",
        bucket_uri=str(tmp_path),
        evaluation_mode="single_member_mean",
    )

    assert res_single_mean["is_ensemble"] is False
    assert res_single_mean["num_members"] == n_members
    assert res_single_mean["ensemble_source"] == "mean_across_individual_members"
    assert res_single_mean["median_nse"] == pytest.approx(
        res_ens["single_seed_median_nse_mean"], abs=1e-4
    )
    assert res_single_mean["single_seed_median_nse_std"] == pytest.approx(
        res_ens["single_seed_median_nse_std"], abs=1e-4
    )
    assert "median_std" in res_single_mean["gauch_2023_metrics"]["NSE"]
    assert res_single_mean["median_nse"] < res_ens["median_nse"]

