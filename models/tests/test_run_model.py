"""Tests for `models/camels_data.py`, `models/zarr_writer.py`, and `models/run_model.py`.

All hydrograph dataset, fold-stitching, and Zarr store tests operate at the full native `(531, 3652)`
CAMELS-US MIP resolution. Loader tests use a hermetic synthetic raw NCAR CAMELS-US v1.2 tree built in
`tmp_path`; nothing here reads data outside the repository.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from models import camels_data as cd
from models.run_model import (
    EXPECTED_NUM_BASINS,
    EXPECTED_NUM_DAYS,
    MODEL_REGISTRY,
    FoldSpec,
    load_basin_to_huc02,
    load_canonical_basins,
    load_model_fold_runner,
    load_spatial_folds,
    run_model_experiment,
    stitch_fold_predictions,
    validate_periods,
)
from models.run_model import (
    main as run_model_cli_main,
)
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_LEADERBOARD_IDS,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
    REQUIRED_GLOBAL_ATTRS,
    build_hydrograph_dataset,
    canonical_evaluation_dates,
    resolve_leaderboard_and_table_id,
    validate_hydrograph_dataset,
    write_hydrograph_zarr,
)

MODELS_DIR: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = MODELS_DIR.parent

EXPECTED_16_MODEL_DIRS: tuple[str, ...] = (
    "kratzert_lstm",
    "kratzert_ealstm",
    "hoedt_mclstm",
    "rasiya_koya_tft",
    "poudel_mlp_hbv",
    "feng_dhbv10",
    "song_dhbv11p",
    "li2025_lstm_dhbv_ens",
    "newman_sacsma",
    "era_gauged_minimal_guidance",
    "era_pur_minimal_guidance",
    "era_pub_minimal_guidance",
    "era_gauged_physics_guided",
    "era_pub_physics_guided",
    "era_pur_physics_guided",
    "era_gauged_multi_paradigm",
    "feng2020_di_lstm",
    "shahriar2026_mamba3",
    "yin2022_rrformer",
)

# Four real canonical basins / HUCs (values come from splits/splits_and_regimes.json at test time).
SYNTHETIC_BASINS: tuple[str, ...] = ("01022500", "02046000", "06221400", "14400000")
SYNTHETIC_START: str = "1997-01-01"
SYNTHETIC_END: str = "2001-12-31"


# =============================================================================
# Synthetic raw NCAR CAMELS-US v1.2 tree
# =============================================================================


def _synthetic_series(basin: str, dates: pd.DatetimeIndex, product: str) -> pd.DataFrame:
    """Deterministic, basin- and product-unique daily forcing table."""
    rng = np.random.RandomState(int(basin[-4:]) + {"daymet": 1, "maurer": 2, "nldas": 3}[product])
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    df = pd.DataFrame(
        {
            "Year": dates.year,
            "Mnth": dates.month,
            "Day": dates.day,
            "Hr": 12,
            "dayl(s)": 40000.0 + 10000.0 * np.sin(2 * np.pi * doy / 365.25),
            "prcp(mm/day)": np.round(rng.gamma(0.6, 4.0, size=n), 2),
            "srad(W/m2)": np.round(200.0 + 100.0 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 5, n), 2),
            "swe(mm)": np.round(np.maximum(0.0, rng.normal(5.0, 3.0, size=n)), 2),
            "tmax(C)": np.round(15.0 + 12.0 * np.cos(2 * np.pi * (doy - 200) / 365.25) + rng.normal(0, 2, n), 2),
            "tmin(C)": np.round(3.0 + 12.0 * np.cos(2 * np.pi * (doy - 200) / 365.25) + rng.normal(0, 2, n), 2),
            "vp(Pa)": np.round(600.0 + 300.0 * np.cos(2 * np.pi * (doy - 200) / 365.25), 2),
        }
    )
    return df


def _write_forcing_file(path: Path, df: pd.DataFrame, lat: float, elev: float, area_m2: float, product: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    header_names = cols
    if product == "nldas":  # NLDAS files use different capitalization in the official release
        header_names = ["Year", "Mnth", "Day", "Hr", "Dayl(s)", "PRCP(mm/day)", "SRAD(W/m2)", "SWE(mm)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"  {lat:.2f}\n {elev:.2f}\n {int(area_m2)}\n")
        fh.write(("\t" if product != "daymet" else " ").join(header_names) + "\n")
        for row in df[cols].itertuples(index=False):
            fh.write("\t".join(str(v) for v in row) + "\n")


def build_synthetic_raw_camels(
    root: Path,
    basins: Sequence[str] = SYNTHETIC_BASINS,
    start: str = SYNTHETIC_START,
    end: str = SYNTHETIC_END,
    maurer_end: str | None = None,
    qobs_cfs_override: dict[str, np.ndarray] | None = None,
    negative_sentinel_day: str | None = "1999-10-15",
    drop_attribute_column: str | None = None,
) -> tuple[dict[str, float], pd.DatetimeIndex]:
    """Create a hermetic raw CAMELS-US v1.2 tree; returns ({basin: area_m2}, dates)."""
    basin_to_huc = load_basin_to_huc02()
    dates = pd.date_range(start, end, freq="D")
    maurer_dates = pd.date_range(start, maurer_end or end, freq="D")
    areas: dict[str, float] = {}
    attr_rows: dict[str, dict[str, object]] = {}

    for b in basins:
        huc = basin_to_huc[b]
        area_m2 = 1.0e6 * (100.0 + int(b[-3:]))  # km2 -> m2, basin-unique
        areas[b] = area_m2
        lat = 30.0 + int(b[:2])
        for product, pdates in (("daymet", dates), ("maurer", maurer_dates), ("nldas", dates)):
            df = _synthetic_series(b, pdates, product)
            _write_forcing_file(cd.raw_forcing_path(root, product, huc, b), df, lat, 250.0, area_m2, product)
        if qobs_cfs_override is not None and b in qobs_cfs_override:
            q_cfs = np.asarray(qobs_cfs_override[b], dtype=np.float64)
        else:
            rng = np.random.RandomState(int(b[-4:]) + 7)
            q_cfs = np.round(rng.gamma(2.0, 50.0, size=len(dates)), 2)
        flags = np.array(["A"] * len(dates), dtype=object)
        if negative_sentinel_day is not None:
            k = int(dates.get_loc(pd.Timestamp(negative_sentinel_day)))
            q_cfs = q_cfs.copy()
            q_cfs[k] = -999.0
            flags[k] = "M"
        sf_path = cd.raw_streamflow_path(root, huc, b)
        sf_path.parent.mkdir(parents=True, exist_ok=True)
        with sf_path.open("w", encoding="utf-8") as fh:
            for d, q, f in zip(dates, q_cfs, flags):
                fh.write(f"{b} {d.year:04d} {d.month:02d} {d.day:02d} {q:9.2f} {f}\n")
        attr_rows[b] = {
            "gauge_lat": lat,
            "gauge_lon": -100.0 - int(b[2:4]),
            "huc_02": huc,
            "gauge_name": f"Synthetic {b}",
            "area_gages2": area_m2 / 1.0e6,
            "area_geospa_fabric": area_m2 / 1.0e6 * 1.05,
            "elev_mean": 250.0 + int(b[-2:]),
            "slope_mean": 20.0,
        }
        for i, name in enumerate(cd.STATIC_ATTRIBUTE_NAMES_27):
            attr_rows[b].setdefault(name, float(i + 1) + int(b[-2:]) / 100.0)

    attr_dir = root / "camels_attributes_v2.0"
    attr_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "camels_name.txt": ["huc_02", "gauge_name"],
        "camels_topo.txt": ["gauge_lat", "gauge_lon", "elev_mean", "slope_mean", "area_gages2", "area_geospa_fabric"],
        "camels_clim.txt": ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"],
        "camels_vege.txt": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff"],
        "camels_soil.txt": ["soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac"],
        "camels_geol.txt": ["carbonate_rocks_frac", "geol_permeability"],
        "camels_hydro.txt": ["q_mean", "runoff_ratio"],
    }
    for fname, cols in groups.items():
        use_cols = [c for c in cols if c != drop_attribute_column]
        with (attr_dir / fname).open("w", encoding="utf-8") as fh:
            fh.write(";".join(["gauge_id"] + use_cols) + "\n")
            for b in basins:
                vals = [str(attr_rows[b].get(c, 0.5)) for c in use_cols]
                fh.write(";".join([b] + vals) + "\n")
    return areas, dates


def _expected_qobs_mm(root: Path, basin: str, dates: pd.DatetimeIndex, area_m2: float) -> np.ndarray:
    huc = load_basin_to_huc02()[basin]
    q = cd.read_raw_streamflow_file(cd.raw_streamflow_path(root, huc, basin), basin).reindex(dates)
    return (q.to_numpy(dtype=np.float64) * cd.CFS_TO_M3_PER_DAY * 1000.0 / area_m2).astype(np.float32)


# =============================================================================
# Shared loader
# =============================================================================


def test_loader_masks_spinup_discharge_and_sanitizes_negatives(tmp_path: Path) -> None:
    """Spin-up window discharge must never leave the loader; -999 sentinels become NaN."""
    root = tmp_path / "raw"
    areas, _ = build_synthetic_raw_camels(root)
    period = ("1999-10-01", "2001-09-30")
    spin = 364
    sl = cd.load_camels_slice(list(SYNTHETIC_BASINS), period, ["daymet"], root, spinup_days=spin)

    full_dates, n_target = cd.period_date_range(period, spin)
    assert int(sl["n_target_days"]) == n_target == 731
    assert int(sl["spinup_days"]) == spin
    assert sl["x_dyn"].shape == (4, len(full_dates), 5)
    assert sl["qobs"].shape == (4, len(full_dates))
    assert np.isnan(sl["qobs"][:, :spin]).all(), "spin-up discharge leaked out of the loader"
    assert sl["target_qobs"].shape == (4, n_target)
    np.testing.assert_array_equal(sl["target_qobs"], sl["qobs"][:, spin:])
    assert pd.Timestamp(sl["dates"][0]) == pd.Timestamp("1998-10-02")
    assert pd.Timestamp(sl["target_dates"][0]) == pd.Timestamp("1999-10-01")

    # Target discharge equals the exact cfs -> mm/day conversion using the Daymet header area.
    for i, b in enumerate(SYNTHETIC_BASINS):
        exp = _expected_qobs_mm(root, b, pd.DatetimeIndex(sl["target_dates"]), areas[b])
        np.testing.assert_allclose(sl["target_qobs"][i], exp, rtol=1e-6, atol=1e-6, equal_nan=True)
        assert abs(float(sl["area_km2"][i]) - areas[b] / 1e6) < 1e-9

    # The -999 day (1999-10-15, inside the target window) is NaN, never 0 or negative.
    k = int(pd.DatetimeIndex(sl["target_dates"]).get_loc(pd.Timestamp("1999-10-15")))
    assert np.isnan(sl["target_qobs"][:, k]).all()
    assert not np.any(sl["qobs"][np.isfinite(sl["qobs"])] < 0.0)
    assert int(sl["n_valid_targets"]) == 4 * (n_target - 1)

    # Channel order and metadata
    assert sl["dynamic_feature_names"].tolist() == cd.PRODUCT_FORCING_COLUMNS["daymet"]
    assert sl["x_stat"].shape == (4, 27)
    assert sl["huc_02"].tolist() == [load_basin_to_huc02()[b] for b in SYNTHETIC_BASINS]
    assert np.all(np.isfinite(sl["lat"])) and np.all(np.isfinite(sl["lon"])) and np.all(np.isfinite(sl["elev"]))


def test_loader_statistics_are_invariant_to_spinup_discharge(tmp_path: Path) -> None:
    """Poisoning the evaluation-decade spin-up discharge must not change any training statistic."""
    period = ("1999-10-01", "2001-09-30")
    spin = 365
    dates = pd.date_range(SYNTHETIC_START, SYNTHETIC_END, freq="D")
    rng = np.random.RandomState(0)
    base = {b: np.round(rng.gamma(2.0, 50.0, size=len(dates)), 2) for b in SYNTHETIC_BASINS}
    poisoned = {b: v.copy() for b, v in base.items()}
    spin_mask = (dates >= pd.Timestamp("1998-10-01")) & (dates <= pd.Timestamp("1999-09-30"))
    for b in poisoned:
        poisoned[b][spin_mask] = 1.0e6  # absurd eval-decade discharge

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    build_synthetic_raw_camels(root_a, qobs_cfs_override=base, negative_sentinel_day=None)
    build_synthetic_raw_camels(root_b, qobs_cfs_override=poisoned, negative_sentinel_day=None)
    sl_a = cd.load_camels_slice(list(SYNTHETIC_BASINS), period, ["daymet", "nldas"], root_a, spinup_days=spin)
    sl_b = cd.load_camels_slice(list(SYNTHETIC_BASINS), period, ["daymet", "nldas"], root_b, spinup_days=spin)

    np.testing.assert_array_equal(sl_a["qobs"], sl_b["qobs"])
    np.testing.assert_array_equal(sl_a["target_qobs"], sl_b["target_qobs"])
    stats_a = cd.compute_train_stats(sl_a)
    stats_b = cd.compute_train_stats(sl_b)
    for k in stats_a:
        np.testing.assert_array_equal(stats_a[k], stats_b[k])
    assert float(stats_a["y_mean"]) < 1000.0
    assert stats_a["basin_stds"].shape == (4,)
    assert stats_a["xd_mean"].shape == (1, 1, 10)


def test_loader_raises_on_missing_forcings_attributes_and_targets(tmp_path: Path) -> None:
    """Missing data in means an exception out: no imputation, no defaults, no silent NaN targets."""
    root = tmp_path / "raw"
    build_synthetic_raw_camels(root, maurer_end="2000-12-31")

    # Maurer coverage ends 2000-12-31 -> requesting 2001 raises instead of filling zeros.
    with pytest.raises(ValueError, match="non-finite forcing value for product 'maurer'"):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), ("2001-01-01", "2001-06-30"), ["maurer"], root)
    # Same request on Daymet is fine.
    cd.load_camels_slice(list(SYNTHETIC_BASINS), ("2001-01-01", "2001-06-30"), ["daymet"], root)

    # Requested dates outside any file coverage raise.
    with pytest.raises(ValueError, match="non-finite forcing value"):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), ("1990-01-01", "1990-12-31"), ["daymet"], root)

    # Target window with zero valid observations is rejected when required.
    zero = {b: np.full(len(pd.date_range(SYNTHETIC_START, SYNTHETIC_END, freq="D")), -999.0) for b in SYNTHETIC_BASINS}
    root_z = tmp_path / "zero"
    build_synthetic_raw_camels(root_z, qobs_cfs_override=zero, negative_sentinel_day=None)
    sl = cd.load_camels_slice(list(SYNTHETIC_BASINS), ("1999-10-01", "2000-09-30"), ["daymet"], root_z)
    assert int(sl["n_valid_targets"]) == 0 and np.isnan(sl["target_qobs"]).all()
    with pytest.raises(ValueError, match="zero valid discharge observations"):
        cd.load_camels_slice(
            list(SYNTHETIC_BASINS), ("1999-10-01", "2000-09-30"), ["daymet"], root_z, require_valid_targets=True
        )

    # Missing required attribute column (gauge_lat) raises; nothing defaults to 40.0.
    root_m = tmp_path / "noattr"
    build_synthetic_raw_camels(root_m, drop_attribute_column="gauge_lat")
    with pytest.raises(KeyError, match="gauge_lat"):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), ("1999-10-01", "2000-09-30"), ["daymet"], root_m)

    # Unknown basin, duplicate forcings, unknown product, empty data dir.
    with pytest.raises(KeyError, match="01013500"):
        cd.load_camels_slice(["01013500"], ("1999-10-01", "2000-09-30"), ["daymet"], root)
    with pytest.raises(ValueError, match="Duplicate forcing products"):
        cd.normalize_forcing_names(["daymet", "daymet"])
    with pytest.raises(ValueError, match="Unsupported forcing products"):
        cd.normalize_forcing_names(["era5"])
    with pytest.raises(FileNotFoundError):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), ("1999-10-01", "2000-09-30"), ["daymet"], tmp_path / "empty")


def test_bundle_roundtrip_matches_raw_and_validates(tmp_path: Path) -> None:
    """compile -> load gives bit-identical slices to the raw reader; schema corruption is rejected."""
    root = tmp_path / "raw"
    build_synthetic_raw_camels(root)
    out = tmp_path / "data" / cd.CANONICAL_BUNDLE_FILENAME
    cd.compile_camels_bundle(root, out, start=SYNTHETIC_START, end=SYNTHETIC_END)
    assert out.is_file()

    header = cd.open_camels_bundle_header(out)
    assert [str(b) for b in header["basins"].tolist()] == sorted(SYNTHETIC_BASINS)
    summary = cd.validate_bundle_against_raw(out, root)
    assert summary["basins_checked"] == 4 and summary["basins_skipped_no_raw_file"] == 0

    period = ("1999-10-01", "2001-09-30")
    for forcings in (["daymet"], ["nldas", "daymet"], ["daymet", "maurer", "nldas"]):
        sl_raw = cd.load_camels_slice(list(SYNTHETIC_BASINS), period, forcings, root, spinup_days=100)
        sl_dir = cd.load_camels_slice(list(SYNTHETIC_BASINS), period, forcings, out.parent, spinup_days=100)
        sl_file = cd.load_camels_slice(list(SYNTHETIC_BASINS), period, forcings, out, spinup_days=100)
        for sl in (sl_dir, sl_file):
            for key in ("x_dyn", "x_stat", "qobs", "target_qobs", "lat", "lon", "elev", "area_km2"):
                np.testing.assert_array_equal(sl[key], sl_raw[key])
            assert sl["dynamic_feature_names"].tolist() == sl_raw["dynamic_feature_names"].tolist()

    # Requested dates outside bundle coverage raise.
    with pytest.raises(ValueError, match="outside bundle coverage"):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), ("1990-01-01", "1990-12-31"), ["daymet"], out)

    # A bundle with a different forcing channel order is rejected instead of being silently mislabelled.
    with np.load(out, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["forcing_variable_names"] = np.asarray(["srad(W/m2)", "prcp(mm/day)", "tmax(C)", "tmin(C)", "vp(Pa)"])
    bad = tmp_path / "bad.npz"
    np.savez(bad, **payload)
    with pytest.raises(ValueError, match="forcing_variable_names"):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), period, ["daymet"], bad)
    payload["forcing_variable_names"] = np.asarray(cd.FORCING_VARIABLES)
    del payload["gauge_lat"]
    np.savez(bad, **payload)
    with pytest.raises(KeyError, match="gauge_lat"):
        cd.load_camels_slice(list(SYNTHETIC_BASINS), period, ["daymet"], bad)


def test_forcing_regime_resolution() -> None:
    assert cd.resolve_forcing_regime(["daymet"]) == "single_camels"
    assert cd.resolve_forcing_regime(["daymet", "maurer"]) == "multi_camels"
    assert cd.resolve_forcing_regime(["daymet", "maurer", "nldas"]) == "multi_camels"
    assert cd.forcing_display_label(["nldas", "daymet"]) == "NLDAS, Daymet"
    assert resolve_leaderboard_and_table_id("pub_12fold", "multi_camels") == ("pub__multi_camels", "table_8")
    assert resolve_leaderboard_and_table_id("pur", "freeform") == ("pur__freeform", "table_14")


# =============================================================================
# Zarr writer
# =============================================================================


def _deterministic_basin_hydrograph(basins: Sequence[str]) -> np.ndarray:
    """Generate a deterministic, basin-unique `(len(basins), 3652)` float32 hydrograph array."""
    n_b = len(basins)
    t_axis = np.linspace(0.0, 2.0 * np.pi, EXPECTED_NUM_DAYS, dtype=np.float32)
    wave = 1.5 + np.sin(t_axis)[None, :]
    basin_offsets = np.asarray([int(str(b).strip()) * 1e-7 for b in basins], dtype=np.float32)[:, None]
    out = (wave + basin_offsets).astype(np.float32)
    assert out.shape == (n_b, EXPECTED_NUM_DAYS)
    return out


def test_zarr_writer_full_resolution_and_reordering(tmp_path: Path) -> None:
    """Verify Zarr creation, sorting of unsorted basins, consolidated metadata, and schema at (531, 3652)."""
    basins = load_canonical_basins()
    basin_to_huc02 = load_basin_to_huc02()
    assert len(basins) == 531

    reversed_basins = list(reversed(basins))
    members_reversed = np.stack(
        [_deterministic_basin_hydrograph(reversed_basins), _deterministic_basin_hydrograph(reversed_basins) * 1.1],
        axis=0,
    ).astype(np.float32)
    qsim_reversed = np.mean(members_reversed, axis=0).astype(np.float32)

    ds = build_hydrograph_dataset(
        qsim=qsim_reversed,
        basins=reversed_basins,
        basin_to_huc02=basin_to_huc02,
        model_id="kratzert_lstm_gauged_single_camels",
        model_name="Kratzert et al. (2019, 2021) CudaLSTM",
        model_family="LSTM",
        family_id="kratzert_lstm",
        provenance_category="MIP Re-run",
        spatial_regime="gauged",
        forcing_regime="single_camels",
        forcings_used=["daymet"],
        code_url="https://github.com/neuralhydrology/neuralhydrology",
        qsim_members=members_reversed,
    )

    assert ds["qsim"].shape == (531, 3652)
    assert ds["qsim"].dtype == np.float32
    assert ds["qsim_members"].shape == (2, 531, 3652)
    assert list(ds.coords["basin"].values) == basins
    assert ds.coords["basin"].dtype == np.dtype("<U8")
    assert ds.coords["huc_02"].dtype == np.dtype("<U2")
    assert ds.coords["date"].dtype == np.dtype("datetime64[ns]")
    np.testing.assert_allclose(ds["qsim"].values[0], qsim_reversed[-1], rtol=1e-6)
    assert ds.attrs["provenance_category"] == "MIP Re-run"
    assert ds.attrs["leaderboard_id"] == "gauged__single_camels"
    assert ds.attrs["spatial_regime"] == "gauged"
    assert ds.attrs["forcings_used"] == "Daymet"
    assert ds.attrs["forcing_products"] == ["daymet"]
    assert ds.attrs["train_period"] == "1999-10-01/2008-09-30"
    assert ds.attrs["test_period"] == "1989-10-01/1999-09-30"
    assert ds.attrs["entry_id"] == ds.attrs["model_id"]
    assert ds.attrs["archive_url"] == "gs://camels-us-mip/hydrographs/gauged__single_camels/kratzert_lstm_gauged_single_camels.zarr"
    assert ds.attrs["is_ensemble"] is True and ds.attrs["num_members"] == 2

    target_zarr = tmp_path / "kratzert_lstm_gauged_single_camels.zarr"
    written_path = write_hydrograph_zarr(ds, target_uri=target_zarr, expected_basins=basins)
    assert Path(written_path).is_dir()
    assert (target_zarr / ".zmetadata").is_file()
    assert (target_zarr / ".zattrs").is_file()

    reopened = xr.open_zarr(target_zarr, consolidated=True)
    validate_hydrograph_dataset(reopened, expected_basins=basins)
    for attr_key in REQUIRED_GLOBAL_ATTRS:
        assert attr_key in reopened.attrs


def test_zarr_writer_rejects_invalid_inputs() -> None:
    """NaN/Inf, wrong shapes, inconsistent ensembles, overlapping periods, and date mismatches are rejected."""
    basins = load_canonical_basins()
    basin_to_huc02 = load_basin_to_huc02()
    valid_qsim = _deterministic_basin_hydrograph(basins)
    common = dict(
        basins=basins,
        basin_to_huc02=basin_to_huc02,
        model_name="Bad",
        model_family="LSTM",
        provenance_category="MIP Re-run",
        spatial_regime="gauged",
        forcing_regime="single_camels",
        forcings_used=["daymet"],
        code_url="https://example.com",
    )

    qsim_with_nan = valid_qsim.copy()
    qsim_with_nan[250, 1800] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_hydrograph_dataset(qsim=qsim_with_nan, model_id="bad_nan", **common)
    qsim_with_inf = valid_qsim.copy()
    qsim_with_inf[10, 20] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        build_hydrograph_dataset(qsim=qsim_with_inf, model_id="bad_inf", **common)
    with pytest.raises(ValueError, match="shape"):
        build_hydrograph_dataset(qsim=valid_qsim[:, :-1], model_id="bad_shape", **common)

    # qsim must be the mean of qsim_members.
    members = np.stack([valid_qsim, valid_qsim * 2.0], axis=0).astype(np.float32)
    with pytest.raises(ValueError, match="arithmetic mean"):
        build_hydrograph_dataset(qsim=valid_qsim, model_id="bad_mean", qsim_members=members, **common)
    # is_ensemble / num_members consistency.
    with pytest.raises(ValueError, match="inconsistent"):
        build_hydrograph_dataset(qsim=valid_qsim, model_id="bad_ens", is_ensemble=True, num_members=1, **common)
    # Forcing regime vs forcing list.
    with pytest.raises(ValueError, match="single_camels"):
        build_hydrograph_dataset(
            qsim=valid_qsim, model_id="bad_forcing", **{**common, "forcings_used": ["daymet", "nldas"]}
        )
    # Overlapping periods.
    with pytest.raises(ValueError, match="overlap"):
        build_hydrograph_dataset(
            qsim=valid_qsim, model_id="bad_overlap", training_period=("1995-10-01", "2004-09-30"), **common
        )
    # Evaluation period that is not 3652 days.
    with pytest.raises(ValueError, match="3652"):
        build_hydrograph_dataset(
            qsim=valid_qsim, model_id="bad_eval", evaluation_period=("1989-10-01", "1998-09-30"), **common
        )
    # Dates disagreeing with the evaluation period.
    shifted = canonical_evaluation_dates() + np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="dates do not match"):
        build_hydrograph_dataset(qsim=valid_qsim, model_id="bad_dates", dates=shifted, **common)
    # Non-canonical evaluation window cannot be filed under a canonical leaderboard id.
    with pytest.raises(ValueError, match="canonical evaluation window"):
        build_hydrograph_dataset(
            qsim=valid_qsim,
            model_id="bad_window",
            training_period=("2000-10-01", "2008-09-30"),
            evaluation_period=("1990-10-01", "2000-09-29"),
            **common,
        )
    # Unknown provenance / regime vocabulary.
    with pytest.raises(ValueError, match="provenance_category"):
        build_hydrograph_dataset(qsim=valid_qsim, model_id="bad_prov", **{**common, "provenance_category": "V"})
    with pytest.raises(ValueError, match="forcing_regime"):
        build_hydrograph_dataset(qsim=valid_qsim, model_id="bad_fr", **{**common, "forcing_regime": "3f"})
    with pytest.raises(ValueError, match="leaderboard_id"):
        build_hydrograph_dataset(qsim=valid_qsim, model_id="bad_lb", leaderboard_id="table_3_gauged_1f", **common)


def test_validator_accepts_bucket_vocabulary_dataset() -> None:
    """A store written with the existing gs://camels-us-mip attribute vocabulary validates as-is."""
    basins = np.asarray(load_canonical_basins(), dtype="<U8")
    huc = load_basin_to_huc02()
    ds = xr.Dataset(
        data_vars={"qsim": (("basin", "date"), _deterministic_basin_hydrograph(basins.tolist()))},
        coords={
            "basin": ("basin", basins),
            "date": ("date", canonical_evaluation_dates()),
            "huc_02": ("basin", np.asarray([huc[b] for b in basins], dtype="<U2")),
        },
        attrs={
            "entry_id": "t4_01",
            "model_name": "Li et al. (2025) LSTM + dHBV 1.0 Ensemble",
            "family_id": "li2025_lstm_dhbv_ens",
            "provenance_category": "Archive",
            "leaderboard_id": "gauged__multi_camels",
            "spatial_regime": "gauged",
            "forcing_regime": "multi_camels",
            "forcings_used": "Daymet, Maurer, NLDAS",
            "is_ensemble": False,
            "num_members": 1,
            "train_period": "1999-10-01/2008-09-30",
            "test_period": "1989-10-01/1999-09-30",
            "units": "mm/day",
            "code_url": "https://github.com/mhpi/hydroDL",
            "archive_url": "gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_01.zarr",
        },
    )
    validate_hydrograph_dataset(ds, expected_basins=basins.tolist())
    ds.attrs["spatial_regime"] = "pub"
    with pytest.raises(ValueError, match="does not match leaderboard_id"):
        validate_hydrograph_dataset(ds)


def test_leaderboard_ids_match_metrics_evaluator() -> None:
    """The writer's 12 leaderboard ids must be exactly the scorer's."""
    from metrics.evaluator import CANONICAL_LEADERBOARD_IDS as SCORER_IDS

    assert set(CANONICAL_LEADERBOARD_IDS) == set(SCORER_IDS)
    assert len(CANONICAL_LEADERBOARD_IDS) == 12


# =============================================================================
# Launcher
# =============================================================================


@pytest.mark.parametrize(
    "spatial_regime,expected_folds",
    [("gauged", 1), ("pub_12fold", 12), ("pur_18huc02", 18)],
)
def test_run_model_experiment_all_spatial_regimes_full_531x3652(
    tmp_path: Path,
    spatial_regime: str,
    expected_folds: int,
) -> None:
    """`run_model_experiment` runs every fold once per seed and stores per-seed members plus their mean."""
    basins = load_canonical_basins()
    folds = load_spatial_folds(spatial_regime)
    assert len(folds) == expected_folds

    observed_calls: list[tuple[str, int, int, int]] = []

    def _recording_fold_runner(
        train_basins: Sequence[str],
        test_basins: Sequence[str],
        train_period: tuple[str, str],
        eval_period: tuple[str, str],
        forcings: Sequence[str],
        data_dir: str | Path,
        seeds: Sequence[int] = (123,),
        **kwargs: object,
    ) -> np.ndarray:
        del data_dir
        assert list(forcings) == ["daymet", "maurer", "nldas"]
        assert train_period == (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END)
        assert eval_period == (CANONICAL_EVAL_START, CANONICAL_EVAL_END)
        assert len(seeds) == 1, "launcher must invoke run_fold once per seed"
        if spatial_regime != "gauged":
            assert set(train_basins).isdisjoint(set(test_basins))
            assert len(train_basins) + len(test_basins) == EXPECTED_NUM_BASINS
        observed_calls.append((str(kwargs["fold_id"]), int(kwargs["fold_index"]), len(test_basins), int(seeds[0])))
        return _deterministic_basin_hydrograph(test_basins) * (1.0 + 0.1 * (int(seeds[0]) - 101))

    out_zarr = tmp_path / f"kratzert_lstm_{spatial_regime}.zarr"
    ds, written_uri = run_model_experiment(
        model_name="kratzert_lstm",
        spatial_regime=spatial_regime,
        forcings=["daymet", "maurer", "nldas"],
        data_dir=tmp_path,
        output_zarr=out_zarr,
        seeds=[101, 102],
        max_workers=2 if expected_folds > 1 else 1,
        fold_runner=_recording_fold_runner,
    )

    assert len(observed_calls) == expected_folds * 2
    assert {c[3] for c in observed_calls} == {101, 102}
    assert sum(c[2] for c in observed_calls) == 2 * EXPECTED_NUM_BASINS
    assert Path(written_uri).is_dir()
    assert ds["qsim"].shape == (531, 3652)
    assert ds["qsim_members"].shape == (2, 531, 3652)
    assert ds.attrs["spatial_regime"] == {"gauged": "gauged", "pub_12fold": "pub", "pur_18huc02": "pur"}[spatial_regime]
    assert ds.attrs["spatial_protocol"] == spatial_regime
    assert ds.attrs["forcing_regime"] == "multi_camels"
    assert ds.attrs["leaderboard_id"].endswith("__multi_camels")
    assert ds.attrs["leaderboard_id"] in CANONICAL_LEADERBOARD_IDS
    assert ds.attrs["num_members"] == 2 and ds.attrs["is_ensemble"] is True
    assert ds.attrs["seeds"] == [101, 102]
    assert ds.attrs["n_folds"] == expected_folds
    assert ds.attrs["provenance_category"] == "MIP Re-run"
    assert ds.attrs["family_id"] == "kratzert_lstm"

    expected_full = _deterministic_basin_hydrograph(basins)
    np.testing.assert_allclose(ds["qsim_members"].values[0], expected_full, rtol=1e-6)
    np.testing.assert_allclose(ds["qsim_members"].values[1], expected_full * 1.1, rtol=1e-6)
    np.testing.assert_allclose(ds["qsim"].values, expected_full * 1.05, rtol=1e-5)

    reopened = xr.open_zarr(out_zarr, consolidated=True)
    validate_hydrograph_dataset(reopened, expected_basins=basins)


def test_run_model_experiment_guards(tmp_path: Path) -> None:
    """Period overlap, duplicate forcings/seeds, bad forcing regime overrides, and bad outputs raise."""

    def _ok_runner(train_basins, test_basins, train_period, eval_period, forcings, data_dir, seeds=(1,), **kw):
        return _deterministic_basin_hydrograph(test_basins)

    def _nan_runner(train_basins, test_basins, train_period, eval_period, forcings, data_dir, seeds=(1,), **kw):
        out = _deterministic_basin_hydrograph(test_basins)
        out[0, 0] = np.nan
        return out

    def _short_runner(train_basins, test_basins, train_period, eval_period, forcings, data_dir, seeds=(1,), **kw):
        return _deterministic_basin_hydrograph(test_basins)[:, :-1]

    base = dict(model_name="kratzert_lstm", spatial_regime="gauged", forcings=["daymet"], data_dir=tmp_path, seeds=[1])
    with pytest.raises(ValueError, match="overlap"):
        run_model_experiment(
            **base, output_zarr=tmp_path / "a.zarr", train_period=("1995-10-01", "2004-09-30"), fold_runner=_ok_runner
        )
    with pytest.raises(ValueError, match="3652"):
        run_model_experiment(
            **base, output_zarr=tmp_path / "b.zarr", eval_period=("1989-10-01", "1998-09-30"), fold_runner=_ok_runner
        )
    with pytest.raises(ValueError, match="Duplicate forcing"):
        run_model_experiment(**{**base, "forcings": ["daymet", "daymet"]}, output_zarr=tmp_path / "c.zarr", fold_runner=_ok_runner)
    with pytest.raises(ValueError, match="Duplicate seeds"):
        run_model_experiment(**{**base, "seeds": [1, 1]}, output_zarr=tmp_path / "d.zarr", fold_runner=_ok_runner)
    with pytest.raises(ValueError, match="contradicts"):
        run_model_experiment(**base, output_zarr=tmp_path / "e.zarr", forcing_regime="multi_camels", fold_runner=_ok_runner)
    with pytest.raises(ValueError, match="non-finite"):
        run_model_experiment(**base, output_zarr=tmp_path / "f.zarr", fold_runner=_nan_runner)
    with pytest.raises(ValueError, match="returned shape"):
        run_model_experiment(**base, output_zarr=tmp_path / "g.zarr", fold_runner=_short_runner)

    # Two products -> multi_camels; explicit lagged_streamflow override is honoured and routed to table 5.
    ds, _ = run_model_experiment(
        **{**base, "forcings": ["daymet", "nldas"]}, output_zarr=tmp_path / "h.zarr", fold_runner=_ok_runner
    )
    assert ds.attrs["leaderboard_id"] == "gauged__multi_camels" and ds.attrs["table_id"] == "table_4"
    assert ds.attrs["num_members"] == 1 and ds.attrs["is_ensemble"] is False
    assert "qsim_members" in ds.data_vars and ds["qsim_members"].shape[0] == 1
    ds2, _ = run_model_experiment(
        **base, output_zarr=tmp_path / "i.zarr", forcing_regime="lagged_streamflow", fold_runner=_ok_runner
    )
    assert ds2.attrs["leaderboard_id"] == "gauged__lagged_streamflow" and ds2.attrs["table_id"] == "table_5"

    validate_periods((CANONICAL_TRAIN_START, CANONICAL_TRAIN_END), (CANONICAL_EVAL_START, CANONICAL_EVAL_END))

    # Sub-model selectors are part of the default store id so kinds/variants never overwrite each other.
    ds3, uri3 = run_model_experiment(
        model_name="kratzert_lstm",
        spatial_regime="gauged",
        forcings=["daymet"],
        data_dir=tmp_path,
        seeds=[1],
        output_zarr=tmp_path / "j.zarr",
        runner_kwargs={"model_kind": "cudalstm"},
        fold_runner=_ok_runner,
    )
    assert ds3.attrs["entry_id"] == "kratzert_lstm_cudalstm_gauged_single_camels"
    assert ds3.attrs["family_id"] == "kratzert_lstm"
    assert json.loads(ds3.attrs["runner_kwargs_json"]) == {"model_kind": "cudalstm"}
    # The process executor needs a registered runner module; injected callables are thread-only.
    with pytest.raises(ValueError, match="executor='process'"):
        run_model_experiment(**base, output_zarr=tmp_path / "k.zarr", max_workers=2, executor="process", fold_runner=_ok_runner)
    with pytest.raises(ValueError, match="executor must be"):
        run_model_experiment(**base, output_zarr=tmp_path / "l.zarr", executor="ray", fold_runner=_ok_runner)


def test_stitch_fold_predictions_detects_missing_and_overlapping_basins() -> None:
    """Verify `stitch_fold_predictions` raises on missing or overlapping out-of-fold basins."""
    basins = load_canonical_basins()
    folds = load_spatial_folds("pub_12fold")

    incomplete_results = [(f, _deterministic_basin_hydrograph(f.test_basins)) for f in folds[:-1]]
    with pytest.raises(ValueError, match="Incomplete basin coverage"):
        stitch_fold_predictions(incomplete_results, canonical_basins=basins)

    dup_fold = FoldSpec(
        fold_id="fold_dup", fold_index=99, train_basins=folds[-1].train_basins, test_basins=folds[0].test_basins
    )
    overlapping_results = incomplete_results + [(dup_fold, _deterministic_basin_hydrograph(dup_fold.test_basins))]
    with pytest.raises(ValueError, match="predicted more than once"):
        stitch_fold_predictions(overlapping_results, canonical_basins=basins)


def test_split_loading_has_no_fallback_and_validates_pur_purity(tmp_path: Path) -> None:
    """Only the canonical full split files are accepted; a missing file is an error, never a fallback."""
    src = REPO_ROOT / "splits"
    partial = tmp_path / "splits"
    partial.mkdir()
    for name in ("basins_531.txt", "splits_and_regimes.json", "pur_18fold_huc02_splits.json"):
        (partial / name).write_bytes((src / name).read_bytes())
    with pytest.raises(FileNotFoundError, match="pub_12fold_random_splits.json"):
        load_spatial_folds("pub_12fold", splits_dir=partial)
    pur = load_spatial_folds("pur_18huc02", splits_dir=partial)
    assert len(pur) == 18

    # Corrupt a PUR fold by moving one test basin into the training list -> overlap is detected.
    payload = json.loads((partial / "pur_18fold_huc02_splits.json").read_text())
    f01 = payload["folds"]["huc_01"]
    f01["train_basins"].append(f01["test_basins"][0])
    (partial / "pur_18fold_huc02_splits.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlapping train and test basins"):
        load_spatial_folds("pur_18huc02", splits_dir=partial)

    # The canonical files satisfy every invariant.
    pub = load_spatial_folds("pub_12fold")
    assert sorted(len(f.test_basins) for f in pub) == [44] * 9 + [45] * 3
    assert set().union(*(set(f.test_basins) for f in pub)) == set(load_canonical_basins())
    huc = load_basin_to_huc02()
    for f in load_spatial_folds("pur_18huc02"):
        assert len({huc[b] for b in f.test_basins}) == 1


def test_all_16_model_directories_registered_and_expose_run_fold() -> None:
    """Verify all 16 model subdirectories exist, are registered, and expose a compliant `run_fold`."""
    assert set(MODEL_REGISTRY.keys()) == set(EXPECTED_16_MODEL_DIRS)
    for model_key in EXPECTED_16_MODEL_DIRS:
        model_dir = MODELS_DIR / model_key
        assert model_dir.is_dir(), f"Missing model directory: {model_dir}"
        assert (model_dir / "__init__.py").is_file()
        assert (model_dir / "model.py").is_file()
        assert (model_dir / "runner.py").is_file()
        entry = MODEL_REGISTRY[model_key]
        assert entry.provenance_category in ("MIP Re-run", "Contribution")
        runner_fn = load_model_fold_runner(entry)
        param_names = list(inspect.signature(runner_fn).parameters.keys())
        for required_param in ("train_basins", "test_basins", "train_period", "eval_period", "forcings", "data_dir", "seeds"):
            assert required_param in param_names, f"{model_key}/runner.py:run_fold is missing parameter '{required_param}'"


def test_ast_audit_zero_try_except_zero_split_loading_zero_hardcoded_dates_in_models() -> None:
    """Enforce via AST that `models/` has zero `try`/`except` and `models/<model>/` has zero split or date literals."""
    forbidden_split_tokens = (
        "pub_12fold_random_splits.json",
        "pur_18fold_huc02_splits.json",
        "basins_531.txt",
        "splits_and_regimes.json",
    )
    forbidden_date_tokens = (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END, CANONICAL_EVAL_START, CANONICAL_EVAL_END)

    py_files = [p for p in sorted(MODELS_DIR.rglob("*.py")) if "tests" not in p.parts]
    assert len(py_files) >= 15 * 3 + 3

    for py_path in py_files:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_path))
        rel_path = py_path.relative_to(MODELS_DIR)

        try_nodes = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Try)]
        assert not try_nodes, f"{rel_path} contains forbidden try/except block at lines {try_nodes}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "suppress":
                pytest.fail(f"{rel_path} uses forbidden contextlib.suppress at line {node.lineno}")

        if (
            len(rel_path.parts) > 1
            and rel_path.parts[0] in EXPECTED_16_MODEL_DIRS
            and py_path.name != "reproduce_author_metrics.py"
        ):
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value
                    for split_tok in forbidden_split_tokens:
                        assert split_tok not in val, (
                            f"{rel_path}:{node.lineno} references split file {split_tok!r} inside a model package"
                        )
                    for date_tok in forbidden_date_tokens:
                        assert date_tok not in val, (
                            f"{rel_path}:{node.lineno} hardcodes canonical split date {date_tok!r} inside a model package"
                        )


def test_all_16_reproduce_author_metrics_scripts_and_json_results() -> None:
    """Verify all 16 model subdirectories contain a hermetic `reproduce_author_metrics.py` and an honest Stage-1 status JSON."""
    for model_key in EXPECTED_16_MODEL_DIRS:
        model_dir = MODELS_DIR / model_key
        script_path = model_dir / "reproduce_author_metrics.py"
        json_path = model_dir / "stage1_author_reproduction_results.json"
        assert script_path.is_file(), f"Missing {script_path}"
        assert json_path.is_file(), f"Missing {json_path}"
        script_src = script_path.read_text(encoding="utf-8")
        assert "MOCK_GCS" not in script_src, f"{script_path} must not contain MOCK_GCS"
        assert "camels_leaderboard" not in script_src, f"{script_path} references the external staging directory"
        assert "from metrics." in script_src, f"{script_path} must import from metrics.metrics or metrics.evaluator"
        assert "run_fold" in script_src, f"{script_path} must reproduce via the model's runner.run_fold"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("status") in ("PASS", "FAIL", "NOT_RUN", "COMPLETED_NO_PUBLISHED_REFERENCE"), (
            f"{json_path} has an unknown status: {payload.get('status')!r}"
        )


def test_run_model_cli_list_models(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `--list-models` prints all registered model families and exits 0."""
    rc = run_model_cli_main(["--list-models"])
    assert rc == 0
    captured = capsys.readouterr().out
    for model_key in EXPECTED_16_MODEL_DIRS:
        assert model_key in captured
    assert len(canonical_evaluation_dates()) == EXPECTED_NUM_DAYS


def test_feng2020_yin2022_models_finite_predictions_with_missing_qobs(
    tmp_path: Path,
) -> None:
    """Verify DI(1) and RR-Former produce 100% finite outputs."""
    from models.feng2020_di_lstm.model import build_lagged_discharge_features
    from models.feng2020_di_lstm.runner import run_fold as run_di
    from models.yin2022_rrformer.runner import run_fold as run_rr

    synthetic_camels_dir = tmp_path / "raw"
    build_synthetic_raw_camels(synthetic_camels_dir, negative_sentinel_day="2000-06-15")

    qobs = np.array([[1.0, np.nan, -999.0, 4.0, 2.0]], dtype=np.float32)
    lag_feats = build_lagged_discharge_features(qobs, q_mean=2.0, q_std=1.0, lag_days=1)
    assert lag_feats.shape == (1, 5, 2)
    assert np.all(np.isfinite(lag_feats))
    # t=0 (shifted), t=2 (from NaN at t=1), t=3 (from -999 at t=2) must have validity=0 and value=0
    assert list(lag_feats[0, :, 1]) == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert lag_feats[0, 0, 0] == 0.0 and lag_feats[0, 2, 0] == 0.0 and lag_feats[0, 3, 0] == 0.0

    train_b = list(SYNTHETIC_BASINS[:2])
    test_b = list(SYNTHETIC_BASINS[2:])
    train_p = ("1998-01-01", "1999-12-31")
    eval_p = ("2000-01-01", "2001-12-31")

    q_di = run_di(
        train_basins=train_b,
        test_basins=train_b,
        train_period=train_p,
        eval_period=eval_p,
        forcings=["nldas"],
        data_dir=synthetic_camels_dir,
        seeds=(1, 2),
        hidden_size=8,
        seq_len=15,
        batch_size=16,
        epochs=1,
        n_batches_per_epoch=2,
        device="cpu",
    )
    assert q_di.shape == (2, 731) and np.all(np.isfinite(q_di)) and np.all(q_di >= 0.0)

    q_di_pub = run_di(
        train_basins=train_b,
        test_basins=test_b,
        train_period=train_p,
        eval_period=eval_p,
        forcings=["nldas"],
        data_dir=synthetic_camels_dir,
        seeds=(1,),
        hidden_size=8,
        seq_len=15,
        batch_size=16,
        epochs=1,
        n_batches_per_epoch=2,
        device="cpu",
    )
    assert q_di_pub.shape == (2, 731) and np.all(np.isfinite(q_di_pub)) and np.all(q_di_pub >= 0.0)

    q_rr = run_rr(
        train_basins=train_b,
        test_basins=train_b,
        train_period=train_p,
        eval_period=eval_p,
        forcings=["daymet"],
        data_dir=synthetic_camels_dir,
        seeds=(1,),
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        seq_len=15,
        batch_size=16,
        epochs=1,
        n_batches_per_epoch=2,
        device="cpu",
    )
    assert q_rr.shape == (2, 731) and np.all(np.isfinite(q_rr)) and np.all(q_rr >= 0.0)
