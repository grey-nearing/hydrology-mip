"""Hermetic unit tests and an opt-in real-data integration test for `metrics.build_eval_targets`.

The unit tests synthesize a tiny NCAR CAMELS-US v1.2 directory tree under `tmp_path` for a
handful of *real* basin IDs / HUC-02 codes (taken from `splits/splits_and_regimes.json`) and
exercise the full build -> write -> re-open -> verify path without network access or real raw data.

Set `CAMELS_RAW_DIR=/path/to/basin_dataset_public_v1p2` to additionally run the integration test
that rebuilds all 531 basins and verifies them against the canonical GCS store.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths
from metrics import build_eval_targets as bet
from metrics.evaluator import _open_zarr_store

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = resolve_mip_paths(DEFAULT_MIP_ID).splits_dir

FULL_INDEX = pd.date_range("1980-01-01", "2008-12-31", freq="D")
KNOWN_AREA_M2 = 587_675_987.0  # real Daymet header area of a CAMELS basin (m^2)
KNOWN_Q_CFS = 395.0
KNOWN_VALUE_DATE = "1995-03-15"  # inside the evaluation decade
SENTINEL_DATE = "1990-01-10"  # written as -999.00 with flag M (inside the evaluation decade)
MISSING_DATE = "1998-07-04"  # row deleted from the file entirely (inside the evaluation decade)
ZERO_FLOW_DATE = "2003-08-20"  # genuine 0.00 cfs row (inside the training period)


@dataclass(frozen=True)
class FakeTree:
    """Description of a synthetic CAMELS-US tree written by `_write_fake_camels_tree`."""

    root: Path
    basins: list[str]
    basin_to_huc02: dict[str, str]
    series_cfs: dict[str, pd.Series]
    areas_m2: dict[str, float]


def _real_basin_to_huc02() -> dict[str, str]:
    """Load the canonical basin -> HUC-02 map from the repository split specification."""
    spec = json.loads((SPLITS_DIR / "splits_and_regimes.json").read_text(encoding="utf-8"))
    return {str(b): str(h) for b, h in spec["basin_to_huc02"].items()}


def _pick_basins(basin_to_huc02: dict[str, str]) -> list[str]:
    """Pick three real basin IDs from three different HUC-02 regions (first, middle, last)."""
    ordered = sorted(basin_to_huc02)
    picked = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    assert len({basin_to_huc02[b] for b in picked}) == 3
    return picked


def _fake_series_cfs(seed: int) -> pd.Series:
    """Deterministic fake daily discharge (cfs) as recorded in a raw file, with special rows.

    The record deliberately extends beyond 1980-2008 on both sides (those rows must be dropped),
    contains a `-999` sentinel row, a genuine zero-flow row, a known-value row and one date that
    is missing from the file entirely.
    """
    dates = pd.date_range("1979-12-30", "2009-01-02", freq="D")
    rng = np.random.default_rng(seed)
    series = pd.Series(np.round(rng.uniform(0.5, 500.0, size=len(dates)), 2), index=dates)
    series[pd.Timestamp(KNOWN_VALUE_DATE)] = KNOWN_Q_CFS
    series[pd.Timestamp(ZERO_FLOW_DATE)] = 0.0
    series[pd.Timestamp(SENTINEL_DATE)] = -999.0
    return series.drop(pd.Timestamp(MISSING_DATE))


def _write_streamflow_file(path: Path, basin: str, series_cfs: pd.Series) -> None:
    """Write a raw `<basin>_streamflow_qc.txt` file (`gauge_id year month day q_cfs flag`)."""
    lines = []
    for i, (timestamp, q_cfs) in enumerate(series_cfs.items()):
        flag = "M" if q_cfs < 0 else ("A:e" if i % 7 == 0 else "A")
        lines.append(
            f"{basin} {timestamp.year:04d} {timestamp.month:02d} {timestamp.day:02d} "
            f"{q_cfs:9.2f} {flag}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_forcing_file(path: Path, area_m2: float) -> None:
    """Write a Daymet forcing file whose third header line is the catchment area in m^2."""
    path.write_text(
        f"  44.82\n 133.00\n {area_m2:.0f}\n"
        "Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)\n"
        "1980 01 01 12\t31185.97\t0.00\t178.46\t0.00\t-0.72\t-10.59\t271.45\n",
        encoding="utf-8",
    )


def _write_fake_camels_tree(
    root: Path,
    basins: list[str],
    basin_to_huc02: dict[str, str],
    huc_override: dict[str, str] | None = None,
) -> FakeTree:
    """Synthesize `usgs_streamflow/` and `basin_mean_forcing/daymet/` for the given basins."""
    huc_override = huc_override or {}
    series_cfs: dict[str, pd.Series] = {}
    areas_m2: dict[str, float] = {}
    for i, basin in enumerate(sorted(basins)):
        huc = huc_override.get(basin, basin_to_huc02[basin])
        streamflow_dir = root / bet.STREAMFLOW_SUBDIR / huc
        forcing_dir = root / bet.FORCING_SUBDIR / huc
        streamflow_dir.mkdir(parents=True, exist_ok=True)
        forcing_dir.mkdir(parents=True, exist_ok=True)
        series_cfs[basin] = _fake_series_cfs(seed=i)
        areas_m2[basin] = KNOWN_AREA_M2 if i == 0 else 1.0e8 + 12_345.0 * i
        _write_streamflow_file(
            streamflow_dir / f"{basin}{bet.STREAMFLOW_SUFFIX}", basin, series_cfs[basin]
        )
        _write_forcing_file(forcing_dir / f"{basin}{bet.FORCING_SUFFIX}", areas_m2[basin])
    return FakeTree(
        root=root,
        basins=sorted(basins),
        basin_to_huc02={b: basin_to_huc02[b] for b in basins},
        series_cfs=series_cfs,
        areas_m2=areas_m2,
    )


def _expected_mm_per_day(series_cfs: pd.Series, area_m2: float) -> np.ndarray:
    """Independent re-implementation of the raw -> mm/day transformation on the full index."""
    q_cfs = series_cfs.reindex(FULL_INDEX).to_numpy(dtype=np.float64)
    q_cfs = np.where(q_cfs < 0.0, np.nan, q_cfs)
    return (q_cfs * 0.028316846592 * 86400 * 1000 / area_m2).astype(np.float32)


def _write_reduced_splits_dir(path: Path, tree: FakeTree) -> Path:
    """Write `basins_531.json` / `splits_and_regimes.json` restricted to the fake tree's basins."""
    path.mkdir(parents=True, exist_ok=True)
    (path / bet.BASINS_FILENAME).write_text(json.dumps(tree.basins), encoding="utf-8")
    (path / bet.SPLITS_AND_REGIMES_FILENAME).write_text(
        json.dumps({"num_basins": len(tree.basins), "basin_to_huc02": tree.basin_to_huc02}),
        encoding="utf-8",
    )
    return path


def _streamflow_path(tree: FakeTree, basin: str) -> Path:
    """Path of a basin's raw streamflow file inside the fake tree."""
    huc = tree.basin_to_huc02[basin]
    return tree.root / bet.STREAMFLOW_SUBDIR / huc / f"{basin}{bet.STREAMFLOW_SUFFIX}"


def _forcing_path(tree: FakeTree, basin: str) -> Path:
    """Path of a basin's Daymet forcing file inside the fake tree."""
    huc = tree.basin_to_huc02[basin]
    return tree.root / bet.FORCING_SUBDIR / huc / f"{basin}{bet.FORCING_SUFFIX}"


@pytest.fixture(scope="module")
def fake_tree(tmp_path_factory: pytest.TempPathFactory) -> FakeTree:
    """A module-scoped synthetic CAMELS-US tree for three real basin IDs."""
    basin_to_huc02 = _real_basin_to_huc02()
    root = tmp_path_factory.mktemp("camels_v1p2")
    return _write_fake_camels_tree(root, _pick_basins(basin_to_huc02), basin_to_huc02)


@pytest.fixture(scope="module")
def built(fake_tree: FakeTree) -> xr.Dataset:
    """The dataset built from `fake_tree`, passing the basins deliberately out of order."""
    return bet.build_eval_targets(fake_tree.root, SPLITS_DIR, basins=fake_tree.basins[::-1])


# --------------------------------------------------------------------------------------------
# Schema, coordinates, masks and attributes
# --------------------------------------------------------------------------------------------


def test_build_reproduces_canonical_schema(fake_tree: FakeTree, built: xr.Dataset) -> None:
    assert dict(built.sizes) == {"basin": 3, "date": 3652, "date_all": 10593}
    assert set(built.data_vars) == {"qobs", "qobs_all", "newman_test_mask", "newman_train_mask"}
    assert set(built.coords) == {"basin", "date", "date_all", "huc_02"}

    assert built["qobs"].dims == ("basin", "date")
    assert built["qobs_all"].dims == ("basin", "date_all")
    assert built["newman_test_mask"].dims == ("date_all",)
    assert built["newman_train_mask"].dims == ("date_all",)
    assert built["huc_02"].dims == ("basin",)

    assert built["qobs"].dtype == np.float32
    assert built["qobs_all"].dtype == np.float32
    assert built["newman_test_mask"].dtype == bool
    assert built["newman_train_mask"].dtype == bool
    assert built["basin"].dtype == np.dtype("<U8")
    assert built["huc_02"].dtype == np.dtype("<U2")
    assert built["date"].dtype == np.dtype("datetime64[ns]")
    assert built["date_all"].dtype == np.dtype("datetime64[ns]")

    assert built.attrs == {
        "title": bet.DATASET_TITLE,
        "num_basins": 3,
        "train_period": "1999-10-01/2008-09-30",
        "test_period": "1989-10-01/1999-09-30",
        "full_period": "1980-01-01/2008-12-31",
        "units": "mm/day",
    }
    assert built["qobs"].attrs["units"] == "mm/day"
    assert built["qobs_all"].attrs["units"] == "mm/day"


def test_coordinates_are_sorted_and_huc02_matches_splits(
    fake_tree: FakeTree, built: xr.Dataset
) -> None:
    basins = list(built["basin"].values)
    assert basins == sorted(fake_tree.basins)
    assert basins == sorted(basins)
    assert all(len(b) == 8 and b.isdigit() for b in basins)
    assert list(built["huc_02"].values) == [fake_tree.basin_to_huc02[b] for b in basins]

    date_all = pd.DatetimeIndex(built["date_all"].values)
    assert date_all.equals(FULL_INDEX)
    date = pd.DatetimeIndex(built["date"].values)
    assert date.equals(pd.date_range("1989-10-01", "1999-09-30", freq="D"))


def test_newman_masks_cover_exact_periods(built: xr.Dataset) -> None:
    date_all = pd.DatetimeIndex(built["date_all"].values)
    test_mask = built["newman_test_mask"].values
    train_mask = built["newman_train_mask"].values

    assert int(test_mask.sum()) == 3652
    assert date_all[test_mask][0] == pd.Timestamp("1989-10-01")
    assert date_all[test_mask][-1] == pd.Timestamp("1999-09-30")
    assert np.array_equal(date_all[test_mask].values, built["date"].values)

    assert int(train_mask.sum()) == 3288
    assert date_all[train_mask][0] == pd.Timestamp("1999-10-01")
    assert date_all[train_mask][-1] == pd.Timestamp("2008-09-30")

    assert not np.any(test_mask & train_mask)
    first_test = int(np.argmax(test_mask))
    assert test_mask[first_test : first_test + 3652].all()
    first_train = int(np.argmax(train_mask))
    assert train_mask[first_train : first_train + 3288].all()
    assert first_train == first_test + 3652  # training starts the day after evaluation ends


def test_qobs_is_the_evaluation_slice_of_qobs_all(built: xr.Dataset) -> None:
    sliced = built["qobs_all"].sel(date_all=slice("1989-10-01", "1999-09-30")).values
    assert np.array_equal(built["qobs"].values, sliced, equal_nan=True)


# --------------------------------------------------------------------------------------------
# Values: missing data handling and unit conversion
# --------------------------------------------------------------------------------------------


def test_missing_values_become_nan_not_zero(fake_tree: FakeTree, built: xr.Dataset) -> None:
    for basin in fake_tree.basins:
        sentinel_all = built["qobs_all"].sel(basin=basin, date_all=SENTINEL_DATE).item()
        sentinel_eval = built["qobs"].sel(basin=basin, date=SENTINEL_DATE).item()
        assert np.isnan(sentinel_all) and np.isnan(sentinel_eval)

        missing_all = built["qobs_all"].sel(basin=basin, date_all=MISSING_DATE).item()
        missing_eval = built["qobs"].sel(basin=basin, date=MISSING_DATE).item()
        assert np.isnan(missing_all) and np.isnan(missing_eval)

        # Exactly the two special days are missing; rows outside 1980-2008 were dropped, not
        # turned into NaN, and no other day was lost.
        assert int(np.isnan(built["qobs_all"].sel(basin=basin).values).sum()) == 2
        assert int(np.isnan(built["qobs"].sel(basin=basin).values).sum()) == 2

        # Genuine zero flow stays exactly zero (never confused with missing data).
        assert built["qobs_all"].sel(basin=basin, date_all=ZERO_FLOW_DATE).item() == 0.0
        for neighbour in ("1990-01-09", "1990-01-11", "1998-07-03", "1998-07-05"):
            value = built["qobs_all"].sel(basin=basin, date_all=neighbour).item()
            assert np.isfinite(value) and value > 0.0

    assert not np.any(built["qobs_all"].values < 0.0)


def test_unit_conversion_exact_for_known_value(fake_tree: FakeTree, built: xr.Dataset) -> None:
    basin = fake_tree.basins[0]
    assert fake_tree.areas_m2[basin] == KNOWN_AREA_M2
    # 395 cfs = 395 * 0.3048^3 m^3/s = 11.18515 m^3/s; over 587.676 km^2 for one day that is
    # 11.18515 * 86400 / 587675987 m = 1.644439 mm.
    assert bet.CUBIC_FEET_TO_CUBIC_METRES == 0.028316846592
    assert bet.CUBIC_FEET_TO_CUBIC_METRES == pytest.approx(0.3048**3, abs=1e-15)
    expected = np.float32(KNOWN_Q_CFS * 0.028316846592 * 86400 * 1000 / KNOWN_AREA_M2)
    assert expected == pytest.approx(1.644439, abs=1e-6)
    assert built["qobs_all"].sel(basin=basin, date_all=KNOWN_VALUE_DATE).item() == expected
    assert built["qobs"].sel(basin=basin, date=KNOWN_VALUE_DATE).item() == expected

    scalar = bet.cfs_to_mm_per_day(KNOWN_Q_CFS, KNOWN_AREA_M2)
    assert np.float32(scalar) == expected
    with pytest.raises(ValueError, match="area"):
        bet.cfs_to_mm_per_day(1.0, 0.0)


def test_every_basin_matches_independent_transformation(
    fake_tree: FakeTree, built: xr.Dataset
) -> None:
    for basin in fake_tree.basins:
        expected = _expected_mm_per_day(fake_tree.series_cfs[basin], fake_tree.areas_m2[basin])
        actual = built["qobs_all"].sel(basin=basin).values
        assert actual.dtype == np.float32
        assert np.array_equal(actual, expected, equal_nan=True)


def test_build_is_deterministic(fake_tree: FakeTree, built: xr.Dataset) -> None:
    again = bet.build_eval_targets(fake_tree.root, SPLITS_DIR, basins=fake_tree.basins)
    xr.testing.assert_identical(built, again)


# --------------------------------------------------------------------------------------------
# Failure modes: nothing is skipped or silently repaired
# --------------------------------------------------------------------------------------------


def test_huc02_directory_mismatch_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    wrong_huc = basin_to_huc02[basins[-1]]
    assert wrong_huc != basin_to_huc02[basins[0]]
    tree = _write_fake_camels_tree(
        tmp_path, basins, basin_to_huc02, huc_override={basins[0]: wrong_huc}
    )
    with pytest.raises(ValueError, match=f"Basin {basins[0]}: HUC-02 directory '{wrong_huc}'"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_forcing_file_in_other_huc02_directory_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    source = _forcing_path(tree, basins[0])
    target = _forcing_path(tree, basins[-1]).with_name(source.name)
    shutil.move(str(source), str(target))
    with pytest.raises(ValueError, match="forcing file sits in"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_missing_streamflow_file_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    victim = basins[1]
    _streamflow_path(tree, victim).unlink()
    with pytest.raises(FileNotFoundError, match=f"Missing streamflow file for basin {victim}"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_missing_forcing_file_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    victim = basins[2]
    _forcing_path(tree, victim).unlink()
    with pytest.raises(FileNotFoundError, match=f"Missing Daymet forcing file for basin {victim}"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_missing_camels_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CAMELS directory not found"):
        bet.build_eval_targets(tmp_path / "nope", SPLITS_DIR, basins=["01022500"])


def test_basin_outside_split_specification_raises(fake_tree: FakeTree) -> None:
    with pytest.raises(ValueError, match="not present in splits_and_regimes.json"):
        bet.build_eval_targets(fake_tree.root, SPLITS_DIR, basins=["99999999"])
    with pytest.raises(ValueError, match="malformed"):
        bet.build_eval_targets(fake_tree.root, SPLITS_DIR, basins=["1022500"])
    with pytest.raises(ValueError, match="duplicate"):
        bet.build_eval_targets(
            fake_tree.root, SPLITS_DIR, basins=[fake_tree.basins[0], fake_tree.basins[0]]
        )


def test_basin_under_two_huc02_directories_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    source = _streamflow_path(tree, basins[0])
    shutil.copy(source, _streamflow_path(tree, basins[-1]).with_name(source.name))
    with pytest.raises(ValueError, match="more than one HUC-02 directory"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_gauge_id_mismatch_inside_file_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    basin = basins[1]
    path = _streamflow_path(tree, basin)
    path.write_text(path.read_text(encoding="utf-8").replace(basin, "00000000"), encoding="utf-8")
    with pytest.raises(ValueError, match="gauge IDs"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_duplicate_date_inside_file_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    path = _streamflow_path(tree, basins[0])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[100]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate dates"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


def test_non_positive_area_raises(tmp_path: Path) -> None:
    basin_to_huc02 = _real_basin_to_huc02()
    basins = _pick_basins(basin_to_huc02)
    tree = _write_fake_camels_tree(tmp_path, basins, basin_to_huc02)
    _write_forcing_file(_forcing_path(tree, basins[0]), 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        bet.build_eval_targets(tree.root, SPLITS_DIR, basins=basins)


# --------------------------------------------------------------------------------------------
# Zarr round trip and verification against an existing store
# --------------------------------------------------------------------------------------------


def test_write_reopen_roundtrip_and_verify(built: xr.Dataset, tmp_path: Path) -> None:
    store = tmp_path / "newman_eval_targets.zarr"
    written = bet.write_eval_targets_zarr(built, str(store))
    assert written == str(store)
    assert (store / ".zmetadata").is_file()  # consolidated Zarr v2
    assert (store / ".zgroup").is_file()

    reopened = _open_zarr_store(str(store))
    assert dict(reopened.sizes) == dict(built.sizes)
    assert reopened["qobs"].dtype == np.float32
    assert reopened["qobs_all"].dtype == np.float32
    assert reopened["newman_test_mask"].dtype == bool
    assert reopened["basin"].dtype == np.dtype("<U8")
    assert reopened["huc_02"].dtype == np.dtype("<U2")
    assert reopened["date"].dtype == np.dtype("datetime64[ns]")
    assert reopened["date_all"].dtype == np.dtype("datetime64[ns]")
    assert reopened["qobs"].encoding["chunks"] == (3, 3652)
    assert reopened["qobs_all"].encoding["chunks"] == (3, 10593)
    assert dict(reopened.attrs) == built.attrs
    assert np.array_equal(reopened["qobs"].values, built["qobs"].values, equal_nan=True)
    assert np.array_equal(reopened["qobs_all"].values, built["qobs_all"].values, equal_nan=True)

    diffs = bet.verify_against_existing(built, reopened)
    assert diffs == {"qobs": 0.0, "qobs_all": 0.0}


def test_verify_against_existing_detects_mismatches(built: xr.Dataset) -> None:
    def _with_qobs(values: np.ndarray) -> xr.Dataset:
        other = built.copy(deep=True)
        other["qobs"] = (("basin", "date"), values, built["qobs"].attrs)
        return other

    perturbed = built["qobs"].values.copy()
    perturbed[0, 0] += 0.01
    with pytest.raises(ValueError, match="'qobs' values differ"):
        bet.verify_against_existing(built, _with_qobs(perturbed), atol=1e-3)
    assert bet.verify_against_existing(built, _with_qobs(perturbed), atol=0.1)["qobs"] == (
        pytest.approx(0.01, abs=1e-6)
    )

    extra_nan = built["qobs"].values.copy()
    extra_nan[1, 5] = np.nan
    with pytest.raises(ValueError, match="missing-data pattern differs"):
        bet.verify_against_existing(built, _with_qobs(extra_nan))

    renamed = built.assign_coords(basin=[b[:-1] + "9" for b in built["basin"].values])
    with pytest.raises(ValueError, match="Coordinate 'basin' mismatch"):
        bet.verify_against_existing(built, renamed)

    shifted = built.assign_coords(date=built["date"].values + np.timedelta64(1, "D"))
    with pytest.raises(ValueError, match="Coordinate 'date' mismatch"):
        bet.verify_against_existing(built, shifted)

    wrong_huc = built.assign_coords(huc_02=("basin", np.array(["99"] * 3, dtype="<U2")))
    with pytest.raises(ValueError, match="Coordinate 'huc_02' mismatch"):
        bet.verify_against_existing(built, wrong_huc)

    flipped = built.copy(deep=True)
    flipped["newman_train_mask"] = (("date_all",), ~built["newman_train_mask"].values)
    with pytest.raises(ValueError, match="newman_train_mask"):
        bet.verify_against_existing(built, flipped)

    with pytest.raises(ValueError, match="no 'qobs' variable"):
        bet.verify_against_existing(built, built.drop_vars("qobs"))


# --------------------------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------------------------


def test_cli_builds_writes_and_verifies(fake_tree: FakeTree, tmp_path: Path) -> None:
    splits_dir = _write_reduced_splits_dir(tmp_path / "splits", fake_tree)
    first = tmp_path / "first.zarr"
    base = ["--camels-dir", str(fake_tree.root), "--splits-dir", str(splits_dir)]
    assert bet.main(base + ["--output", str(first)]) == 0
    reopened = _open_zarr_store(str(first))
    assert list(reopened["basin"].values) == fake_tree.basins
    assert reopened.attrs["num_basins"] == 3

    second = tmp_path / "second.zarr"
    assert bet.main(base + ["--output", str(second), "--verify-against", str(first)]) == 0
    assert (second / ".zmetadata").is_file()

    # A store that does not match must fail verification *before* anything is written.
    bad = reopened.load().copy(deep=True)
    bad_qobs = bad["qobs"].values.copy()
    bad_qobs[2, 100] += 0.5
    bad["qobs"] = (("basin", "date"), bad_qobs, bad["qobs"].attrs)
    bad_store = tmp_path / "bad.zarr"
    bet.write_eval_targets_zarr(bad, str(bad_store))
    never_written = tmp_path / "never_written.zarr"
    with pytest.raises(ValueError, match="'qobs' values differ"):
        bet.main(base + ["--output", str(never_written), "--verify-against", str(bad_store)])
    assert not never_written.exists()


def test_cli_entry_point_exit_status(fake_tree: FakeTree, tmp_path: Path) -> None:
    splits_dir = _write_reduced_splits_dir(tmp_path / "splits", fake_tree)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    base = [
        sys.executable, "-m", "metrics.build_eval_targets",
        "--camels-dir", str(fake_tree.root),
        "--splits-dir", str(splits_dir),
    ]
    good = tmp_path / "good.zarr"
    ok = subprocess.run(
        base + ["--output", str(good)], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stderr
    assert "Wrote consolidated Zarr v2 store" in ok.stdout

    bad_splits = _write_reduced_splits_dir(tmp_path / "bad_splits", fake_tree)
    bad_map = dict(fake_tree.basin_to_huc02)
    bad_map[fake_tree.basins[0]] = "99"
    (bad_splits / bet.SPLITS_AND_REGIMES_FILENAME).write_text(
        json.dumps({"num_basins": 3, "basin_to_huc02": bad_map}), encoding="utf-8"
    )
    failed = subprocess.run(
        base[:-2] + ["--splits-dir", str(bad_splits), "--output", str(tmp_path / "never.zarr")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "HUC-02 directory" in failed.stderr
    assert not (tmp_path / "never.zarr").exists()


def test_cli_requires_camels_dir_splits_dir_and_output() -> None:
    with pytest.raises(SystemExit) as exc_info:
        bet.build_arg_parser().parse_args([])
    assert exc_info.value.code != 0


# --------------------------------------------------------------------------------------------
# Opt-in real-data integration test
# --------------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("CAMELS_RAW_DIR"),
    reason="Set CAMELS_RAW_DIR=/path/to/basin_dataset_public_v1p2 to run the full rebuild.",
)
def test_full_531_basin_rebuild_matches_canonical_gcs_store() -> None:
    camels_dir = Path(os.environ["CAMELS_RAW_DIR"])
    rebuilt = bet.build_eval_targets(camels_dir, SPLITS_DIR)
    assert dict(rebuilt.sizes) == {"basin": 531, "date": 3652, "date_all": 10593}
    assert rebuilt.attrs["num_basins"] == 531

    canonical = _open_zarr_store(bet.CANONICAL_TARGETS_URI)
    assert dict(canonical.attrs) == rebuilt.attrs
    diffs = bet.verify_against_existing(rebuilt, canonical, atol=1e-3)
    assert set(diffs) == {"qobs", "qobs_all"}
    # The canonical store was produced with the exact 0.3048^3 cubic-foot constant, so the rebuild
    # is expected to be bit-for-bit identical in float32, not merely within tolerance.
    assert diffs == {"qobs": 0.0, "qobs_all": 0.0}
    canonical_qobs = np.asarray(canonical["qobs"].values, dtype=np.float32)
    assert np.array_equal(rebuilt["qobs"].values, canonical_qobs, equal_nan=True)
