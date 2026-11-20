"""Shared CAMELS-US forcing, static-attribute, and discharge loader for all MIP model runners.

Design rules (see `skills/algorithm-rules-and-norms.md`):

* All functions take explicit basin lists and date periods handed down from `models/run_model.py`;
  no split files or canonical dates are loaded or hard-coded here.
* Discharge inside the spin-up window is **masked to NaN before it leaves this module** so that no
  trainer can ever use spin-up discharge for loss targets, target scalers, or per-basin NSE weights.
  Under the Newman protocol the training spin-up window (`1998-10-0x`..`1999-09-30`) lies inside the
  `1989-10-01`..`1999-09-30` evaluation decade, so this masking is a hard anti-leakage guarantee.
* Negative discharge sentinels (USGS `-999`) are converted to NaN in every branch.
* Missing data in -> error out: non-finite forcings or attributes in the requested window raise
  `ValueError`; missing basins, files, or coordinates raise `FileNotFoundError` / `KeyError` /
  `ValueError`. Nothing is imputed and no silent defaults (latitude, area, ...) exist.
* Zero `try`/`except` blocks.

Three on-disk layouts are supported, resolved in this order by `resolve_data_source`:

1. **Canonical compiled bundle** (`camels_671_1980_2014.npz`, schema `BUNDLE_SCHEMA_VERSION`) built by
   `compile_camels_bundle` from the raw NCAR CAMELS-US v1.2 download, or converted from a legacy
   per-product bundle by `convert_legacy_bundle`. Feature names, attribute names, basins, and dates
   are stored inside the file and validated on every read; nothing about the layout is assumed.
2. **Raw NCAR CAMELS-US v1.2 directory** containing `basin_mean_forcing/<product>/<HUC02>/<basin>_lump_*`,
   `usgs_streamflow/<HUC02>/<basin>_streamflow_qc.txt`, and the `camels_attributes_v2.0` text files.
3. **Per-basin NetCDF layout** (`time_series/<basin>.nc` + `attributes/camels_*.txt`), e.g. as produced
   by neuralhydrology's CAMELS-US preprocessing.

Command line::

    python -m models.camels_data build --raw-dir /path/to/camels_v1p2 --out data/camels_671_1980_2014.npz
    python -m models.camels_data convert-legacy --legacy-npz old.npz --attributes-dir ... --out ...
    python -m models.camels_data validate --bundle data/camels_671_1980_2014.npz [--raw-dir ...]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hydromip.paths import DEFAULT_MIP_ID, REPO_ROOT, resolve_mip_paths  # noqa: F401

DEFAULT_DATA_DIR: Path = resolve_mip_paths(DEFAULT_MIP_ID).data_dir

BUNDLE_SCHEMA_VERSION: str = "camels-us-mip-bundle-v1"
CANONICAL_BUNDLE_FILENAME: str = "camels_671_1980_2014.npz"

FORCING_PRODUCTS: tuple[str, ...] = ("daymet", "maurer", "nldas")
FORCING_PRODUCT_LABELS: dict[str, str] = {"daymet": "Daymet", "maurer": "Maurer", "nldas": "NLDAS"}
FORCING_VARIABLES: tuple[str, ...] = ("prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)")
AUX_VARIABLES: tuple[str, ...] = ("dayl(s)", "swe(mm)")

PRODUCT_FORCING_COLUMNS: dict[str, list[str]] = {
    p: [f"{v}_{p}" for v in FORCING_VARIABLES] for p in FORCING_PRODUCTS
}

# Raw NCAR CAMELS-US v1.2 layout (`basin_timeseries_v1p2_metForcing_obsFlow`).
RAW_PRODUCT_DIRS: dict[str, str] = {
    "daymet": "daymet",
    "maurer": "maurer_extended",
    "nldas": "nldas_extended",
}
RAW_FORCING_SUFFIX: dict[str, str] = {
    "daymet": "_lump_cida_forcing_leap.txt",
    "maurer": "_lump_maurer_forcing_leap.txt",
    "nldas": "_lump_nldas_forcing_leap.txt",
}
RAW_STREAMFLOW_SUFFIX: str = "_streamflow_qc.txt"
RAW_FORCING_COLUMN_ALIASES: dict[str, str] = {
    "dayl(s)": "dayl(s)",
    "prcp(mm/day)": "prcp(mm/day)",
    "srad(w/m2)": "srad(W/m2)",
    "swe(mm)": "swe(mm)",
    "tmax(c)": "tmax(C)",
    "tmin(c)": "tmin(C)",
    "vp(pa)": "vp(Pa)",
}
CAMELS_ATTRIBUTE_FILES: tuple[str, ...] = (
    "camels_clim.txt",
    "camels_geol.txt",
    "camels_hydro.txt",
    "camels_name.txt",
    "camels_soil.txt",
    "camels_topo.txt",
    "camels_vege.txt",
)

# Cubic feet per second -> mm/day over the catchment area in m^2. The exact cubic-foot constant
# (0.3048^3 m^3) is the one neuralhydrology and `metrics/build_eval_targets.py` use, so training
# targets built here are bit-consistent with the canonical `observations/newman_eval_targets.zarr`.
CUBIC_FEET_TO_CUBIC_METRES: float = 0.028316846592
CFS_TO_M3_PER_DAY: float = CUBIC_FEET_TO_CUBIC_METRES * 86400.0
# Constant a legacy per-product bundle was built with; `convert_legacy_bundle` rescales exactly when it
# differs from `CUBIC_FEET_TO_CUBIC_METRES` (the step4_tier2 bundle already used the exact constant).
LEGACY_CUBIC_FEET_TO_CUBIC_METRES: float = CUBIC_FEET_TO_CUBIC_METRES

STATIC_ATTRIBUTE_NAMES_27: tuple[str, ...] = (
    "p_mean",
    "pet_mean",
    "aridity",
    "p_seasonality",
    "frac_snow",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "carbonate_rocks_frac",
    "geol_permeability",
)
REQUIRED_ATTRIBUTE_COLUMNS: tuple[str, ...] = STATIC_ATTRIBUTE_NAMES_27 + (
    "gauge_lat",
    "gauge_lon",
    "huc_02",
)

BUNDLE_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "basins",
    "dates",
    "forcing_variable_names",
    "forcing_daymet",
    "forcing_maurer",
    "forcing_nldas",
    "qobs",
    "static_attributes",
    "static_attribute_names",
    "gauge_lat",
    "gauge_lon",
    "elev_mean",
    "area_km2",
    "huc_02",
    "provenance",
)

_BUNDLE_CACHE: dict[tuple[str, int, int, str], np.ndarray] = {}


# =============================================================================
# Small helpers
# =============================================================================


def _zfill_basins(basins: Sequence[str]) -> list[str]:
    """Normalize basin identifiers to 8-digit zero-padded USGS gauge IDs."""
    out = [str(b).strip().zfill(8) for b in basins]
    if not out:
        raise ValueError("basins sequence must be non-empty.")
    bad = [b for b in out if len(b) != 8 or not b.isdigit()]
    if bad:
        raise ValueError(f"Invalid USGS gauge IDs (must be <= 8 digits): {bad[:5]}")
    dupes = sorted({b for b in out if out.count(b) > 1})
    if dupes:
        raise ValueError(f"Duplicate basin IDs requested: {dupes[:5]}")
    return out


def normalize_forcing_names(forcings: Sequence[str]) -> list[str]:
    """Validate forcing product names ('daymet', 'maurer', 'nldas'); duplicates are an error."""
    normalized = [str(f).strip().lower() for f in forcings]
    if not normalized:
        raise ValueError("At least one forcing product must be specified.")
    invalid = [f for f in normalized if f not in PRODUCT_FORCING_COLUMNS]
    if invalid:
        raise ValueError(
            f"Unsupported forcing products {invalid}; valid products are {FORCING_PRODUCTS}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate forcing products requested: {normalized}.")
    return normalized


def resolve_forcing_regime(forcings: Sequence[str]) -> str:
    """Map a forcing list to the leaderboard forcing regime ('single_camels' or 'multi_camels')."""
    n = len(normalize_forcing_names(forcings))
    return "single_camels" if n == 1 else "multi_camels"


def forcing_display_label(forcings: Sequence[str]) -> str:
    """Return the human-readable forcing label used in Zarr attrs, e.g. 'Daymet, Maurer, NLDAS'."""
    return ", ".join(FORCING_PRODUCT_LABELS[p] for p in normalize_forcing_names(forcings))


def period_date_range(
    period: tuple[str, str], spinup_days: int = 0
) -> tuple[pd.DatetimeIndex, int]:
    """Build a daily DatetimeIndex covering [start - spinup_days, end]; return (dates, n_target_days)."""
    if int(spinup_days) < 0:
        raise ValueError(f"spinup_days must be >= 0, got {spinup_days}.")
    start_ts = pd.Timestamp(period[0])
    end_ts = pd.Timestamp(period[1])
    if end_ts < start_ts:
        raise ValueError(f"Invalid period {period!r}: end date precedes start date.")
    target_dates = pd.date_range(start_ts, end_ts, freq="D")
    n_target_days = len(target_dates)
    if int(spinup_days) > 0:
        spinup_start = start_ts - pd.Timedelta(days=int(spinup_days))
        full_dates = pd.date_range(spinup_start, end_ts, freq="D")
    else:
        full_dates = target_dates
    return full_dates, n_target_days


def _assert_contiguous_daily(dates: pd.DatetimeIndex, what: str) -> None:
    if len(dates) == 0:
        raise ValueError(f"{what}: empty date axis.")
    expected = pd.date_range(dates[0], periods=len(dates), freq="D")
    if not dates.equals(expected):
        raise ValueError(f"{what}: date axis is not contiguous daily.")


# =============================================================================
# Raw NCAR CAMELS-US v1.2 readers
# =============================================================================


def read_raw_forcing_file(path: str | Path) -> tuple[pd.DataFrame, float, float, float]:
    """Read one `basin_mean_forcing` text file.

    Returns ``(df, gauge_lat, elevation_m, area_m2)`` where `df` is indexed by date and has the
    columns `FORCING_VARIABLES + AUX_VARIABLES` in canonical capitalization.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing CAMELS forcing file: {p}")
    with p.open("r", encoding="utf-8") as fh:
        header = [fh.readline().strip() for _ in range(3)]
    lat = float(header[0])
    elev = float(header[1])
    area_m2 = float(header[2])
    if not (area_m2 > 0.0):
        raise ValueError(f"Non-positive catchment area {area_m2} in header of {p}.")
    df = pd.read_csv(p, sep=r"\s+", skiprows=3)
    rename = {c: RAW_FORCING_COLUMN_ALIASES[c.strip().lower()] for c in df.columns
              if c.strip().lower() in RAW_FORCING_COLUMN_ALIASES}
    df = df.rename(columns=rename)
    for c in ("Year", "Mnth", "Day"):
        if c not in df.columns:
            raise KeyError(f"Column {c!r} missing in {p}.")
    missing = [v for v in FORCING_VARIABLES + AUX_VARIABLES if v not in df.columns]
    if missing:
        raise KeyError(f"Forcing variables {missing} missing in {p}; columns = {list(df.columns)}.")
    dates = pd.to_datetime(
        {"year": df["Year"].astype(int), "month": df["Mnth"].astype(int), "day": df["Day"].astype(int)}
    )
    out = df[list(FORCING_VARIABLES + AUX_VARIABLES)].astype(np.float64)
    out.index = pd.DatetimeIndex(dates)
    if out.index.has_duplicates:
        raise ValueError(f"Duplicate dates in {p}.")
    _assert_contiguous_daily(out.index, str(p))
    return out, lat, elev, area_m2


def read_raw_streamflow_file(path: str | Path, basin: str) -> pd.Series:
    """Read one `usgs_streamflow` file; returns daily discharge in cfs with `-999`/negatives as NaN."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing CAMELS streamflow file: {p}")
    df = pd.read_csv(
        p,
        sep=r"\s+",
        header=None,
        names=["gauge_id", "Year", "Mnth", "Day", "q_cfs", "flag"],
        dtype={"gauge_id": str, "flag": str},
    )
    ids = df["gauge_id"].astype(str).str.zfill(8).unique().tolist()
    if ids != [basin]:
        raise ValueError(f"Streamflow file {p} contains gauge ids {ids}, expected [{basin!r}].")
    dates = pd.to_datetime(
        {"year": df["Year"].astype(int), "month": df["Mnth"].astype(int), "day": df["Day"].astype(int)}
    )
    q = df["q_cfs"].astype(np.float64).to_numpy()
    q = np.where(q < 0.0, np.nan, q)
    s = pd.Series(q, index=pd.DatetimeIndex(dates), name=basin)
    if s.index.has_duplicates:
        raise ValueError(f"Duplicate dates in {p}.")
    return s


def load_camels_attributes(attributes_dir: str | Path) -> pd.DataFrame:
    """Concatenate the official `camels_*.txt` attribute tables (semicolon separated) on `gauge_id`."""
    d = Path(attributes_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"CAMELS attributes directory not found: {d}")
    frames: list[pd.DataFrame] = []
    for name in CAMELS_ATTRIBUTE_FILES:
        f = d / name
        if not f.is_file():
            raise FileNotFoundError(f"Missing CAMELS attribute file: {f}")
        frames.append(pd.read_csv(f, sep=";", dtype={"gauge_id": str, "huc_02": str}).set_index("gauge_id"))
    df = pd.concat(frames, axis=1)
    if df.columns.has_duplicates:
        dup = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate attribute columns across CAMELS tables: {dup}")
    df.index = df.index.map(lambda s: str(s).strip().zfill(8))
    if df.index.has_duplicates:
        raise ValueError("Duplicate gauge_id rows in CAMELS attribute tables.")
    missing = [c for c in REQUIRED_ATTRIBUTE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"CAMELS attribute tables are missing required columns: {missing}")
    df["huc_02"] = df["huc_02"].astype(str).str.strip().str.zfill(2)
    return df


def find_raw_attributes_dir(raw_dir: str | Path) -> Path:
    """Locate the `camels_attributes_v2.0` tables under (or next to) a raw CAMELS directory."""
    root = Path(raw_dir)
    candidates = [
        root,
        root / "camels_attributes_v2.0",
        root.parent / "camels_attributes_v2.0",
        root / "camels_attributes_v2.0" / "camels_attributes_v2.0",
    ]
    for c in candidates:
        if c.is_dir() and all((c / n).is_file() for n in CAMELS_ATTRIBUTE_FILES):
            return c
    raise FileNotFoundError(
        f"Could not find the CAMELS attribute tables {CAMELS_ATTRIBUTE_FILES} in any of {candidates}."
    )


def raw_forcing_path(raw_dir: str | Path, product: str, huc_02: str, basin: str) -> Path:
    """Path of a raw basin-mean forcing file for `product` inside the NCAR v1.2 layout."""
    return (
        Path(raw_dir)
        / "basin_mean_forcing"
        / RAW_PRODUCT_DIRS[product]
        / str(huc_02).zfill(2)
        / f"{basin}{RAW_FORCING_SUFFIX[product]}"
    )


def raw_streamflow_path(raw_dir: str | Path, huc_02: str, basin: str) -> Path:
    """Path of a raw USGS streamflow file inside the NCAR v1.2 layout."""
    return Path(raw_dir) / "usgs_streamflow" / str(huc_02).zfill(2) / f"{basin}{RAW_STREAMFLOW_SUFFIX}"


def is_raw_camels_dir(path: str | Path) -> bool:
    """True when `path` looks like a raw NCAR CAMELS-US v1.2 directory."""
    p = Path(path)
    return (p / "basin_mean_forcing").is_dir() and (p / "usgs_streamflow").is_dir()


def is_netcdf_layout_dir(path: str | Path) -> bool:
    """True when `path` looks like a per-basin NetCDF layout (`time_series/` + `attributes/`)."""
    p = Path(path)
    return (p / "time_series").is_dir() and (p / "attributes").is_dir()


def load_raw_camels(
    raw_dir: str | Path,
    basins: Sequence[str],
    dates: pd.DatetimeIndex,
    products: Sequence[str],
    attributes_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load raw NCAR CAMELS-US forcings, discharge, and attributes for `basins` over `dates`.

    Returns the canonical in-memory bundle dictionary (same keys as a compiled bundle) restricted
    to the requested basins, dates, and products. Discharge is converted from cfs to mm/day using
    the catchment area in the Daymet forcing-file header.
    """
    raw_root = Path(raw_dir)
    if not is_raw_camels_dir(raw_root):
        raise FileNotFoundError(
            f"{raw_root} is not a raw CAMELS-US v1.2 directory (needs basin_mean_forcing/ and usgs_streamflow/)."
        )
    basin_ids = _zfill_basins(basins)
    prods = normalize_forcing_names(products)
    _assert_contiguous_daily(dates, "requested dates")

    attr_dir = Path(attributes_dir) if attributes_dir is not None else find_raw_attributes_dir(raw_root)
    attr_df = load_camels_attributes(attr_dir)
    missing_attr = [b for b in basin_ids if b not in attr_df.index]
    if missing_attr:
        raise KeyError(f"Basins missing from CAMELS attribute tables: {missing_attr[:5]}")

    n_b, n_t = len(basin_ids), len(dates)
    forcing_arrays = {p: np.full((n_b, n_t, len(FORCING_VARIABLES)), np.nan, dtype=np.float32) for p in prods}
    aux_arrays = {p: np.full((n_b, n_t, len(AUX_VARIABLES)), np.nan, dtype=np.float32) for p in prods}
    qobs = np.full((n_b, n_t), np.nan, dtype=np.float32)
    area_km2 = np.full(n_b, np.nan, dtype=np.float64)
    header_lat = np.full(n_b, np.nan, dtype=np.float64)
    huc_02 = np.asarray([attr_df.loc[b, "huc_02"] for b in basin_ids], dtype="<U2")

    for i, b in enumerate(basin_ids):
        huc = str(huc_02[i])
        area_for_basin: float | None = None
        for p in prods:
            df, lat, _elev, area_m2 = read_raw_forcing_file(raw_forcing_path(raw_root, p, huc, b))
            sub = df.reindex(dates)
            forcing_arrays[p][i] = sub[list(FORCING_VARIABLES)].to_numpy(dtype=np.float32)
            aux_arrays[p][i] = sub[list(AUX_VARIABLES)].to_numpy(dtype=np.float32)
            if p == "daymet":
                area_for_basin = area_m2
                header_lat[i] = lat
        if area_for_basin is None:
            # Area must come from the Daymet header to match the canonical discharge convention.
            _df, lat, _elev, area_for_basin = read_raw_forcing_file(
                raw_forcing_path(raw_root, "daymet", huc, b)
            )
            header_lat[i] = lat
        area_km2[i] = area_for_basin / 1.0e6
        q_cfs = read_raw_streamflow_file(raw_streamflow_path(raw_root, huc, b), b).reindex(dates)
        qobs[i] = (q_cfs.to_numpy(dtype=np.float64) * CFS_TO_M3_PER_DAY * 1000.0 / area_for_basin).astype(
            np.float32
        )

    static = np.asarray(attr_df.loc[basin_ids, list(STATIC_ATTRIBUTE_NAMES_27)].to_numpy(), dtype=np.float32)
    bundle: dict[str, np.ndarray] = {
        "schema_version": np.asarray(BUNDLE_SCHEMA_VERSION),
        "basins": np.asarray(basin_ids, dtype="<U8"),
        "dates": np.asarray(dates.strftime("%Y-%m-%d").tolist(), dtype="<U10"),
        "forcing_variable_names": np.asarray(FORCING_VARIABLES, dtype=str),
        "aux_variable_names": np.asarray(AUX_VARIABLES, dtype=str),
        "qobs": qobs,
        "static_attributes": static,
        "static_attribute_names": np.asarray(STATIC_ATTRIBUTE_NAMES_27, dtype=str),
        "gauge_lat": np.asarray(attr_df.loc[basin_ids, "gauge_lat"].to_numpy(), dtype=np.float32),
        "gauge_lon": np.asarray(attr_df.loc[basin_ids, "gauge_lon"].to_numpy(), dtype=np.float32),
        "elev_mean": np.asarray(attr_df.loc[basin_ids, "elev_mean"].to_numpy(), dtype=np.float32),
        "area_km2": area_km2,
        "area_source": np.asarray("daymet_forcing_header"),
        "huc_02": huc_02,
    }
    for p in prods:
        bundle[f"forcing_{p}"] = forcing_arrays[p]
        bundle[f"aux_{p}"] = aux_arrays[p]
    bundle["provenance"] = np.asarray(
        json.dumps(
            {
                "source": "raw NCAR CAMELS-US v1.2",
                "raw_dir": str(raw_root),
                "attributes_dir": str(attr_dir),
                "products": prods,
                "n_basins": n_b,
                "date_range": [str(dates[0].date()), str(dates[-1].date())],
                "discharge_units": "mm/day = cfs * 0.028316846592 * 86400 * 1000 / area_m2(daymet header)",
            }
        )
    )
    return bundle


# =============================================================================
# Canonical compiled bundle (.npz)
# =============================================================================


def validate_bundle_arrays(bundle: Mapping[str, np.ndarray], source: str = "<bundle>") -> None:
    """Strictly validate the schema of an in-memory or on-disk canonical bundle."""
    missing = [k for k in BUNDLE_REQUIRED_KEYS if k not in bundle]
    if missing:
        raise KeyError(f"{source}: bundle is missing required keys {missing}.")
    schema = str(np.asarray(bundle["schema_version"]))
    if schema != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"{source}: schema_version {schema!r} != {BUNDLE_SCHEMA_VERSION!r}.")
    basins = [str(b) for b in np.asarray(bundle["basins"]).tolist()]
    if len(basins) != len(set(basins)) or any(len(b) != 8 for b in basins):
        raise ValueError(f"{source}: basins must be unique 8-digit IDs.")
    if basins != sorted(basins):
        raise ValueError(f"{source}: basins must be sorted ascending.")
    dates = pd.to_datetime(np.asarray(bundle["dates"]).astype(str))
    _assert_contiguous_daily(pd.DatetimeIndex(dates), f"{source}: dates")
    n_b, n_t = len(basins), len(dates)
    fnames = [str(x) for x in np.asarray(bundle["forcing_variable_names"]).tolist()]
    if fnames != list(FORCING_VARIABLES):
        raise ValueError(f"{source}: forcing_variable_names {fnames} != {list(FORCING_VARIABLES)}.")
    for p in FORCING_PRODUCTS:
        arr = np.asarray(bundle[f"forcing_{p}"])
        if arr.shape != (n_b, n_t, len(FORCING_VARIABLES)):
            raise ValueError(
                f"{source}: forcing_{p} has shape {arr.shape}, expected {(n_b, n_t, len(FORCING_VARIABLES))}."
            )
    snames = [str(x) for x in np.asarray(bundle["static_attribute_names"]).tolist()]
    if snames != list(STATIC_ATTRIBUTE_NAMES_27):
        raise ValueError(f"{source}: static_attribute_names do not match STATIC_ATTRIBUTE_NAMES_27.")
    if np.asarray(bundle["static_attributes"]).shape != (n_b, len(STATIC_ATTRIBUTE_NAMES_27)):
        raise ValueError(f"{source}: static_attributes has wrong shape.")
    if np.asarray(bundle["qobs"]).shape != (n_b, n_t):
        raise ValueError(f"{source}: qobs has shape {np.asarray(bundle['qobs']).shape}, expected {(n_b, n_t)}.")
    for k in ("gauge_lat", "gauge_lon", "elev_mean", "area_km2", "huc_02"):
        if np.asarray(bundle[k]).shape != (n_b,):
            raise ValueError(f"{source}: {k} must have shape {(n_b,)}.")
    if not np.all(np.isfinite(np.asarray(bundle["area_km2"], dtype=np.float64))):
        raise ValueError(f"{source}: area_km2 contains non-finite values.")
    if np.any(np.asarray(bundle["area_km2"], dtype=np.float64) <= 0.0):
        raise ValueError(f"{source}: area_km2 contains non-positive values.")
    hucs = [str(h).zfill(2) for h in np.asarray(bundle["huc_02"]).tolist()]
    bad_h = sorted({h for h in hucs if h not in {f"{i:02d}" for i in range(1, 19)}})
    if bad_h:
        raise ValueError(f"{source}: invalid huc_02 codes {bad_h}.")


def write_camels_bundle(bundle: Mapping[str, np.ndarray], out_path: str | Path, compress: bool = True) -> Path:
    """Validate and write a canonical bundle to `out_path` (.npz)."""
    validate_bundle_arrays(bundle, source=str(out_path))
    out = Path(out_path)
    if out.suffix != ".npz":
        raise ValueError(f"Bundle path must end with .npz, got {out}.")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in bundle.items()}
    if compress:
        np.savez_compressed(out, **payload)
    else:
        np.savez(out, **payload)
    return out


def _bundle_stat_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def clear_bundle_cache() -> None:
    """Drop every cached bundle array (read-only input cache keyed by file identity)."""
    _BUNDLE_CACHE.clear()


def _bundle_array(path: Path, key: str) -> np.ndarray:
    """Return one array from a bundle, cached by (path, mtime, size, key)."""
    ident = _bundle_stat_key(path) + (key,)
    if ident in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[ident]
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"{path}: bundle has no key {key!r} (keys: {z.files}).")
        arr = np.asarray(z[key])
    _BUNDLE_CACHE[ident] = arr
    return arr


def open_camels_bundle_header(path: str | Path) -> dict[str, np.ndarray]:
    """Open a bundle and validate its schema using only the small metadata arrays."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Bundle not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        keys = list(z.files)
    missing = [k for k in BUNDLE_REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(f"{p}: bundle is missing required keys {missing}.")
    header = {
        k: _bundle_array(p, k)
        for k in (
            "schema_version",
            "basins",
            "dates",
            "forcing_variable_names",
            "static_attribute_names",
            "gauge_lat",
            "gauge_lon",
            "elev_mean",
            "area_km2",
            "huc_02",
            "provenance",
        )
    }
    schema = str(header["schema_version"])
    if schema != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"{p}: schema_version {schema!r} != {BUNDLE_SCHEMA_VERSION!r}.")
    fnames = [str(x) for x in header["forcing_variable_names"].tolist()]
    if fnames != list(FORCING_VARIABLES):
        raise ValueError(f"{p}: forcing_variable_names {fnames} != {list(FORCING_VARIABLES)}.")
    snames = [str(x) for x in header["static_attribute_names"].tolist()]
    if snames != list(STATIC_ATTRIBUTE_NAMES_27):
        raise ValueError(f"{p}: static_attribute_names do not match STATIC_ATTRIBUTE_NAMES_27.")
    basins = [str(b) for b in header["basins"].tolist()]
    if len(basins) != len(set(basins)) or basins != sorted(basins) or any(len(b) != 8 for b in basins):
        raise ValueError(f"{p}: basins must be unique, sorted, 8-digit IDs.")
    _assert_contiguous_daily(pd.DatetimeIndex(pd.to_datetime(header["dates"].astype(str))), f"{p}: dates")
    return header


def load_bundle_slice(
    bundle_path: str | Path,
    basins: Sequence[str],
    dates: pd.DatetimeIndex,
    products: Sequence[str],
) -> dict[str, np.ndarray]:
    """Slice a canonical bundle to `basins` x `dates` for the requested products."""
    p = Path(bundle_path)
    header = open_camels_bundle_header(p)
    basin_ids = _zfill_basins(basins)
    prods = normalize_forcing_names(products)
    _assert_contiguous_daily(dates, "requested dates")

    all_basins = [str(b) for b in header["basins"].tolist()]
    b_index = {b: i for i, b in enumerate(all_basins)}
    missing_b = [b for b in basin_ids if b not in b_index]
    if missing_b:
        raise KeyError(f"Basins missing from bundle {p}: {missing_b[:5]} (bundle has {len(all_basins)}).")
    b_idx = np.asarray([b_index[b] for b in basin_ids], dtype=np.int64)

    bundle_dates = pd.DatetimeIndex(pd.to_datetime(header["dates"].astype(str)))
    if dates[0] < bundle_dates[0] or dates[-1] > bundle_dates[-1]:
        raise ValueError(
            f"Requested dates {dates[0].date()}..{dates[-1].date()} fall outside bundle coverage "
            f"{bundle_dates[0].date()}..{bundle_dates[-1].date()} ({p})."
        )
    t0 = int((dates[0] - bundle_dates[0]).days)
    t_idx = np.arange(t0, t0 + len(dates), dtype=np.int64)

    out: dict[str, np.ndarray] = {
        "schema_version": np.asarray(BUNDLE_SCHEMA_VERSION),
        "basins": np.asarray(basin_ids, dtype="<U8"),
        "dates": np.asarray(dates.strftime("%Y-%m-%d").tolist(), dtype="<U10"),
        "forcing_variable_names": np.asarray(FORCING_VARIABLES, dtype=str),
        "static_attribute_names": np.asarray(STATIC_ATTRIBUTE_NAMES_27, dtype=str),
        "qobs": np.asarray(_bundle_array(p, "qobs")[np.ix_(b_idx, t_idx)], dtype=np.float32),
        "static_attributes": np.asarray(_bundle_array(p, "static_attributes")[b_idx], dtype=np.float32),
        "gauge_lat": np.asarray(header["gauge_lat"][b_idx], dtype=np.float32),
        "gauge_lon": np.asarray(header["gauge_lon"][b_idx], dtype=np.float32),
        "elev_mean": np.asarray(header["elev_mean"][b_idx], dtype=np.float32),
        "area_km2": np.asarray(header["area_km2"][b_idx], dtype=np.float64),
        "huc_02": np.asarray([str(h).zfill(2) for h in header["huc_02"][b_idx].tolist()], dtype="<U2"),
        "provenance": header["provenance"],
    }
    with np.load(p, allow_pickle=False) as z:
        bundle_keys = set(z.files)
    for prod in prods:
        out[f"forcing_{prod}"] = np.asarray(
            _bundle_array(p, f"forcing_{prod}")[np.ix_(b_idx, t_idx)], dtype=np.float32
        )
        aux_key = (
            f"aux_{prod}"
            if f"aux_{prod}" in bundle_keys
            else ("aux_daymet" if "aux_daymet" in bundle_keys else None)
        )
        if aux_key is not None:
            out[f"aux_{prod}"] = np.asarray(
                _bundle_array(p, aux_key)[np.ix_(b_idx, t_idx)], dtype=np.float32
            )
    return out


# =============================================================================
# Per-basin NetCDF layout (neuralhydrology-style)
# =============================================================================


def load_netcdf_layout(
    data_dir: str | Path,
    basins: Sequence[str],
    dates: pd.DatetimeIndex,
    products: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load `time_series/<basin>.nc` + `attributes/camels_*.txt` into the canonical bundle dict."""
    import xarray as xr

    root = Path(data_dir)
    basin_ids = _zfill_basins(basins)
    prods = normalize_forcing_names(products)
    attr_df = load_camels_attributes(root / "attributes")
    missing_attr = [b for b in basin_ids if b not in attr_df.index]
    if missing_attr:
        raise KeyError(f"Basins missing from attribute tables in {root / 'attributes'}: {missing_attr[:5]}")

    n_b, n_t = len(basin_ids), len(dates)
    forcing_arrays = {p: np.full((n_b, n_t, len(FORCING_VARIABLES)), np.nan, dtype=np.float32) for p in prods}
    qobs = np.full((n_b, n_t), np.nan, dtype=np.float32)
    ts_dir = root / "time_series"
    for i, b in enumerate(basin_ids):
        nc_path = ts_dir / f"{b}.nc"
        if not nc_path.is_file():
            raise FileNotFoundError(f"Missing basin NetCDF file: {nc_path}")
        ds_b = xr.open_dataset(nc_path)
        if "date" not in ds_b.coords:
            raise KeyError(f"{nc_path} has no 'date' coordinate.")
        ds_sel = ds_b.sel(date=dates)
        for p in prods:
            for k, col in enumerate(PRODUCT_FORCING_COLUMNS[p]):
                if col not in ds_sel.data_vars:
                    raise KeyError(f"{nc_path} is missing forcing variable {col!r}.")
                forcing_arrays[p][i, :, k] = np.asarray(ds_sel[col].values, dtype=np.float32)
        if "QObs(mm/d)" not in ds_sel.data_vars:
            raise KeyError(f"{nc_path} is missing discharge variable 'QObs(mm/d)'.")
        qobs[i] = np.asarray(ds_sel["QObs(mm/d)"].values, dtype=np.float32)
        ds_b.close()

    out: dict[str, np.ndarray] = {
        "schema_version": np.asarray(BUNDLE_SCHEMA_VERSION),
        "basins": np.asarray(basin_ids, dtype="<U8"),
        "dates": np.asarray(dates.strftime("%Y-%m-%d").tolist(), dtype="<U10"),
        "forcing_variable_names": np.asarray(FORCING_VARIABLES, dtype=str),
        "static_attribute_names": np.asarray(STATIC_ATTRIBUTE_NAMES_27, dtype=str),
        "qobs": qobs,
        "static_attributes": np.asarray(
            attr_df.loc[basin_ids, list(STATIC_ATTRIBUTE_NAMES_27)].to_numpy(), dtype=np.float32
        ),
        "gauge_lat": np.asarray(attr_df.loc[basin_ids, "gauge_lat"].to_numpy(), dtype=np.float32),
        "gauge_lon": np.asarray(attr_df.loc[basin_ids, "gauge_lon"].to_numpy(), dtype=np.float32),
        "elev_mean": np.asarray(attr_df.loc[basin_ids, "elev_mean"].to_numpy(), dtype=np.float32),
        "area_km2": np.asarray(attr_df.loc[basin_ids, "area_gages2"].to_numpy(), dtype=np.float64),
        "area_source": np.asarray("attributes:area_gages2"),
        "huc_02": np.asarray(attr_df.loc[basin_ids, "huc_02"].tolist(), dtype="<U2"),
        "provenance": np.asarray(json.dumps({"source": "netcdf layout", "data_dir": str(root)})),
    }
    for p in prods:
        out[f"forcing_{p}"] = forcing_arrays[p]
    return out


# =============================================================================
# Source resolution and the public slice API
# =============================================================================


def resolve_data_source(data_dir: str | Path) -> tuple[str, Path]:
    """Return ``(kind, path)`` with kind in {'bundle', 'raw', 'netcdf'}; raise if nothing matches."""
    p = Path(data_dir)
    if p.is_file():
        if p.suffix != ".npz":
            raise ValueError(f"data_dir file {p} is not a .npz bundle.")
        return "bundle", p
    if not p.is_dir():
        raise FileNotFoundError(f"CAMELS data directory not found: {p}")
    canonical = p / CANONICAL_BUNDLE_FILENAME
    if canonical.is_file():
        return "bundle", canonical
    npz_files = sorted(p.glob("*.npz"))
    if len(npz_files) == 1:
        return "bundle", npz_files[0]
    if len(npz_files) > 1:
        raise ValueError(
            f"{p} contains several .npz files {[f.name for f in npz_files]}; pass the bundle file explicitly."
        )
    if is_raw_camels_dir(p):
        return "raw", p
    if is_netcdf_layout_dir(p):
        return "netcdf", p
    raise FileNotFoundError(
        f"{p} contains neither a canonical bundle ({CANONICAL_BUNDLE_FILENAME}), a raw CAMELS-US v1.2 "
        "layout (basin_mean_forcing/ + usgs_streamflow/), nor a NetCDF layout (time_series/ + attributes/)."
    )


def _finalize_slice(
    raw: Mapping[str, np.ndarray],
    products: Sequence[str],
    full_dates: pd.DatetimeIndex,
    n_target_days: int,
    spinup_days: int,
    require_valid_targets: bool,
    source_desc: str,
) -> dict[str, np.ndarray]:
    """Assemble, validate, and anti-leakage-mask a loaded slice."""
    prods = normalize_forcing_names(products)
    basin_ids = [str(b) for b in np.asarray(raw["basins"]).tolist()]
    n_b, n_t = len(basin_ids), len(full_dates)

    parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for p in prods:
        arr = np.asarray(raw[f"forcing_{p}"], dtype=np.float32)
        if arr.shape != (n_b, n_t, len(FORCING_VARIABLES)):
            raise ValueError(f"{source_desc}: forcing_{p} has shape {arr.shape}, expected {(n_b, n_t, 5)}.")
        bad = ~np.isfinite(arr)
        if bad.any():
            bi, ti, vi = np.argwhere(bad)[0]
            raise ValueError(
                f"{source_desc}: non-finite forcing value for product {p!r}, variable "
                f"{FORCING_VARIABLES[int(vi)]!r}, basin {basin_ids[int(bi)]}, date "
                f"{full_dates[int(ti)].date()} ({int(bad.sum())} non-finite values in the requested window). "
                "Missing meteorological inputs are never imputed; choose a product/period with complete coverage."
            )
        parts.append(arr)
        feature_names.extend(PRODUCT_FORCING_COLUMNS[p])
    x_dyn = np.ascontiguousarray(np.concatenate(parts, axis=-1), dtype=np.float32)

    aux_parts: list[np.ndarray] = []
    aux_feature_names: list[str] = []
    doy = full_dates.dayofyear.to_numpy(dtype=np.float32)[None, :]
    lat_rad = np.deg2rad(np.asarray(raw["gauge_lat"], dtype=np.float32))[:, None]
    for p in prods:
        if f"aux_{p}" in raw:
            aux_arr = np.asarray(raw[f"aux_{p}"], dtype=np.float32)
        elif "aux_daymet" in raw:
            aux_arr = np.asarray(raw["aux_daymet"], dtype=np.float32)
        else:
            decl = 0.409 * np.sin(2.0 * np.pi * (doy - 81.0) / 365.25)
            cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
            dayl_s = (2.0 * np.arccos(cos_omega) / np.pi * 43200.0).astype(np.float32)
            swe_mm = np.zeros_like(dayl_s, dtype=np.float32)
            aux_arr = np.stack([dayl_s, swe_mm], axis=-1)
        aux_parts.append(aux_arr)
        aux_feature_names.extend([f"{v}_{p}" for v in AUX_VARIABLES])
    x_aux = np.ascontiguousarray(np.concatenate(aux_parts, axis=-1), dtype=np.float32)

    x_stat = np.asarray(raw["static_attributes"], dtype=np.float32)
    if x_stat.shape != (n_b, len(STATIC_ATTRIBUTE_NAMES_27)):
        raise ValueError(f"{source_desc}: static_attributes has shape {x_stat.shape}.")
    bad_s = ~np.isfinite(x_stat)
    if bad_s.any():
        bi, ai = np.argwhere(bad_s)[0]
        raise ValueError(
            f"{source_desc}: non-finite static attribute {STATIC_ATTRIBUTE_NAMES_27[int(ai)]!r} for basin "
            f"{basin_ids[int(bi)]}. Attributes are never imputed."
        )

    for key in ("gauge_lat", "gauge_lon", "elev_mean", "area_km2"):
        vals = np.asarray(raw[key], dtype=np.float64)
        if vals.shape != (n_b,) or not np.all(np.isfinite(vals)):
            raise ValueError(f"{source_desc}: {key} is missing or non-finite for at least one basin.")
    area = np.asarray(raw["area_km2"], dtype=np.float64)
    if np.any(area <= 0.0):
        raise ValueError(f"{source_desc}: non-positive catchment area.")

    qobs = np.array(raw["qobs"], dtype=np.float32, copy=True)
    if qobs.shape != (n_b, n_t):
        raise ValueError(f"{source_desc}: qobs has shape {qobs.shape}, expected {(n_b, n_t)}.")
    qobs = np.where(qobs < 0.0, np.nan, qobs).astype(np.float32)
    qobs[~np.isfinite(qobs)] = np.nan
    if spinup_days > 0:
        # Hard anti-leakage guarantee: spin-up discharge never leaves this module.
        qobs[:, :spinup_days] = np.nan
    target_qobs = np.ascontiguousarray(qobs[:, spinup_days:], dtype=np.float32)
    if target_qobs.shape[1] != n_target_days:
        raise ValueError(
            f"{source_desc}: target window has {target_qobs.shape[1]} days, expected {n_target_days}."
        )
    n_valid_targets = int(np.sum(~np.isnan(target_qobs)))
    if require_valid_targets and n_valid_targets == 0:
        raise ValueError(
            f"{source_desc}: zero valid discharge observations in the target window "
            f"{full_dates[spinup_days].date()}..{full_dates[-1].date()} for the requested basins."
        )

    return {
        "x_dyn": x_dyn,
        "x_aux": x_aux,
        "x_stat": x_stat,
        "qobs": qobs,
        "target_qobs": target_qobs,
        "dates": full_dates.values.astype("datetime64[ns]"),
        "target_dates": full_dates[spinup_days:].values.astype("datetime64[ns]"),
        "n_target_days": np.asarray(n_target_days, dtype=np.int64),
        "spinup_days": np.asarray(spinup_days, dtype=np.int64),
        "n_valid_targets": np.asarray(n_valid_targets, dtype=np.int64),
        "lat": np.asarray(raw["gauge_lat"], dtype=np.float32),
        "lon": np.asarray(raw["gauge_lon"], dtype=np.float32),
        "elev": np.asarray(raw["elev_mean"], dtype=np.float32),
        "area_km2": area,
        "huc_02": np.asarray([str(h).zfill(2) for h in np.asarray(raw["huc_02"]).tolist()], dtype="<U2"),
        "basins": np.asarray(basin_ids, dtype="<U8"),
        "dynamic_feature_names": np.asarray(feature_names, dtype=str),
        "aux_feature_names": np.asarray(aux_feature_names, dtype=str),
        "static_attribute_names": np.asarray(STATIC_ATTRIBUTE_NAMES_27, dtype=str),
    }


def load_camels_slice(
    basins: Sequence[str],
    period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str | Path,
    spinup_days: int = 0,
    require_valid_targets: bool = False,
) -> dict[str, np.ndarray]:
    """Load CAMELS-US dynamic forcings, 27 static attributes, and discharge for `basins` over `period`.

    Args:
        basins: USGS gauge IDs (any zero-padding); order is preserved in every returned array.
        period: ``(start, end)`` of the *target* window (training or evaluation period).
        forcings: subset of ``('daymet', 'maurer', 'nldas')`` without duplicates; the dynamic input
            channels are product-major in this order, five variables per product
            (``prcp(mm/day), srad(W/m2), tmax(C), tmin(C), vp(Pa)``).
        data_dir: canonical bundle file / directory, raw NCAR CAMELS-US v1.2 directory, or NetCDF layout.
        spinup_days: number of days prepended before ``period[0]`` for model warm-up. **Discharge in
            the spin-up window is returned as NaN** (`qobs[:, :spinup_days]`), so neither loss targets,
            target scalers, nor per-basin NSE weights can ever see it. Forcings are returned for the
            whole window.
        require_valid_targets: raise ``ValueError`` when the target window holds zero valid
            observations (set this for training slices so an untrained model can never be emitted).

    Returns:
        dict with ``x_dyn (n_basins, n_days, 5*n_products)``, ``x_stat (n_basins, 27)``,
        ``qobs (n_basins, n_days)`` (spin-up masked), ``target_qobs (n_basins, n_target_days)``,
        ``dates``, ``target_dates``, ``n_target_days``, ``spinup_days``, ``n_valid_targets``,
        ``lat``, ``lon``, ``elev``, ``area_km2``, ``huc_02``, ``basins``, ``dynamic_feature_names``,
        ``static_attribute_names``. Discharge is in mm/day.
    """
    basin_ids = _zfill_basins(basins)
    prods = normalize_forcing_names(forcings)
    spin = int(spinup_days)
    full_dates, n_target_days = period_date_range(period, spinup_days=spin)
    kind, path = resolve_data_source(data_dir)
    if kind == "bundle":
        raw = load_bundle_slice(path, basin_ids, full_dates, prods)
    elif kind == "raw":
        raw = load_raw_camels(path, basin_ids, full_dates, prods)
    else:
        raw = load_netcdf_layout(path, basin_ids, full_dates, prods)
    return _finalize_slice(
        raw,
        products=prods,
        full_dates=full_dates,
        n_target_days=n_target_days,
        spinup_days=spin,
        require_valid_targets=bool(require_valid_targets),
        source_desc=f"{kind}:{path}",
    )


def compute_train_stats(train_slice: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute standardization statistics strictly on the training target window of `train_slice`.

    Forcing statistics use ``x_dyn[:, spinup_days:]`` (target days only), static statistics use all
    training basins, target statistics use ``target_qobs`` with ``qobs >= 0``. Per-basin standard
    deviations are NaN for basins with fewer than two valid observations; callers must decide
    explicitly how to treat them (nothing is defaulted here).
    """
    spin = int(train_slice["spinup_days"])
    x_dyn = np.asarray(train_slice["x_dyn"], dtype=np.float32)[:, spin:, :]
    x_stat = np.asarray(train_slice["x_stat"], dtype=np.float32)
    target = np.asarray(train_slice["target_qobs"], dtype=np.float32)
    if x_dyn.shape[1] != target.shape[1]:
        raise ValueError("train_slice x_dyn target window and target_qobs disagree in length.")

    xd_mean = np.mean(x_dyn, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    xd_std = (np.std(x_dyn, axis=(0, 1), keepdims=True, dtype=np.float64) + 1e-6).astype(np.float32)
    xs_mean = np.mean(x_stat, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    xs_std = (np.std(x_stat, axis=0, keepdims=True, dtype=np.float64) + 1e-6).astype(np.float32)

    valid = ~np.isnan(target) & (target >= 0.0)
    if int(valid.sum()) == 0:
        raise ValueError("Training target_qobs contains zero valid observations.")
    valid_q = target[valid].astype(np.float64)
    y_mean = np.float32(np.mean(valid_q))
    y_std = np.float32(np.std(valid_q) + 1e-6)

    basin_stds = np.full(target.shape[0], np.nan, dtype=np.float32)
    for i in range(target.shape[0]):
        v = target[i][valid[i]].astype(np.float64)
        if v.size >= 2:
            basin_stds[i] = np.float32(np.std(v))

    return {
        "xd_mean": xd_mean,
        "xd_std": xd_std,
        "xs_mean": xs_mean,
        "xs_std": xs_std,
        "y_mean": np.asarray(y_mean, dtype=np.float32),
        "y_std": np.asarray(y_std, dtype=np.float32),
        "basin_stds": basin_stds,
    }


# =============================================================================
# Bundle construction and verification
# =============================================================================


def compile_camels_bundle(
    raw_dir: str | Path,
    out_path: str | Path,
    basins: Sequence[str] | None = None,
    start: str = "1980-01-01",
    end: str = "2014-12-31",
    attributes_dir: str | Path | None = None,
    compress: bool = True,
) -> Path:
    """Compile a canonical bundle from a raw NCAR CAMELS-US v1.2 directory for all three products.

    When `basins` is None every basin listed in the attribute tables (671 in CAMELS-US) is compiled.
    """
    raw_root = Path(raw_dir)
    attr_dir = Path(attributes_dir) if attributes_dir is not None else find_raw_attributes_dir(raw_root)
    attr_df = load_camels_attributes(attr_dir)
    basin_ids = sorted(_zfill_basins(basins)) if basins is not None else sorted(attr_df.index.tolist())
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    bundle = load_raw_camels(raw_root, basin_ids, dates, FORCING_PRODUCTS, attributes_dir=attr_dir)
    prov = json.loads(str(bundle["provenance"]))
    prov["built_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    prov["builder"] = "models.camels_data.compile_camels_bundle"
    bundle["provenance"] = np.asarray(json.dumps(prov))
    return write_camels_bundle(bundle, out_path, compress=compress)


LEGACY_PRODUCT_KEYS: dict[str, str] = {"daymet": "dyn_daymet", "maurer": "dyn_maurer", "nldas": "dyn_nldas"}


def convert_legacy_bundle(
    legacy_npz: str | Path,
    attributes_dir: str | Path,
    out_path: str | Path,
    raw_dir_for_validation: str | Path | None = None,
    compress: bool = True,
    legacy_cfs_constant: float = LEGACY_CUBIC_FEET_TO_CUBIC_METRES,
) -> Path:
    """Convert a legacy per-product bundle (`dyn_daymet/dyn_maurer/dyn_nldas/qobs/area_km2/...`) to the
    canonical schema, taking static attributes, coordinates, and HUC codes from the official CAMELS
    attribute tables, and (optionally) cross-validating forcings and discharge against raw files.

    Legacy discharge was converted with ``legacy_cfs_constant`` (cubic feet -> cubic metres); it is
    rescaled exactly to the canonical ``CUBIC_FEET_TO_CUBIC_METRES`` so training targets match the
    canonical observation store.
    """
    src = Path(legacy_npz)
    if not src.is_file():
        raise FileNotFoundError(f"Legacy bundle not found: {src}")
    if not (legacy_cfs_constant > 0.0):
        raise ValueError("legacy_cfs_constant must be positive.")
    attr_df = load_camels_attributes(attributes_dir)
    with np.load(src, allow_pickle=True) as z:
        keys = list(z.files)
        required = ["basins", "dates", "qobs", "area_km2"] + list(LEGACY_PRODUCT_KEYS.values())
        missing = [k for k in required if k not in keys]
        if missing:
            raise KeyError(f"{src}: legacy bundle is missing keys {missing}.")
        basins = [str(b).zfill(8) for b in z["basins"].tolist()]
        dates = pd.DatetimeIndex(pd.to_datetime(z["dates"].astype(str)))
        _assert_contiguous_daily(dates, f"{src}: dates")
        order = np.argsort(np.asarray(basins))
        sorted_basins = [basins[i] for i in order]
        forcing = {p: np.asarray(z[k], dtype=np.float32)[order] for p, k in LEGACY_PRODUCT_KEYS.items()}
        qobs = np.asarray(z["qobs"], dtype=np.float64)[order]
        area_km2 = np.asarray(z["area_km2"], dtype=np.float64)[order]
        aux = np.asarray(z["aux_daymet"], dtype=np.float32)[order] if "aux_daymet" in keys else None

    missing_attr = [b for b in sorted_basins if b not in attr_df.index]
    if missing_attr:
        raise KeyError(f"Legacy bundle basins missing from attribute tables: {missing_attr[:5]}")
    qobs = np.where(qobs < 0.0, np.nan, qobs)
    if legacy_cfs_constant != CUBIC_FEET_TO_CUBIC_METRES:
        qobs = qobs * (CUBIC_FEET_TO_CUBIC_METRES / legacy_cfs_constant)
    qobs = qobs.astype(np.float32)

    bundle: dict[str, np.ndarray] = {
        "schema_version": np.asarray(BUNDLE_SCHEMA_VERSION),
        "basins": np.asarray(sorted_basins, dtype="<U8"),
        "dates": np.asarray(dates.strftime("%Y-%m-%d").tolist(), dtype="<U10"),
        "forcing_variable_names": np.asarray(FORCING_VARIABLES, dtype=str),
        "aux_variable_names": np.asarray(AUX_VARIABLES, dtype=str),
        "qobs": qobs,
        "static_attributes": np.asarray(
            attr_df.loc[sorted_basins, list(STATIC_ATTRIBUTE_NAMES_27)].to_numpy(), dtype=np.float32
        ),
        "static_attribute_names": np.asarray(STATIC_ATTRIBUTE_NAMES_27, dtype=str),
        "gauge_lat": np.asarray(attr_df.loc[sorted_basins, "gauge_lat"].to_numpy(), dtype=np.float32),
        "gauge_lon": np.asarray(attr_df.loc[sorted_basins, "gauge_lon"].to_numpy(), dtype=np.float32),
        "elev_mean": np.asarray(attr_df.loc[sorted_basins, "elev_mean"].to_numpy(), dtype=np.float32),
        "area_km2": area_km2,
        "area_source": np.asarray("daymet_forcing_header"),
        "huc_02": np.asarray(attr_df.loc[sorted_basins, "huc_02"].tolist(), dtype="<U2"),
    }
    for p in FORCING_PRODUCTS:
        bundle[f"forcing_{p}"] = forcing[p]
    if aux is not None:
        bundle["aux_daymet"] = aux

    prov: dict[str, object] = {
        "source": "legacy per-product bundle converted to canonical schema",
        "legacy_bundle": str(src),
        "attributes_dir": str(Path(attributes_dir)),
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "builder": "models.camels_data.convert_legacy_bundle",
        "discharge_units": "mm/day = cfs * 0.028316846592 * 86400 * 1000 / area_m2(daymet header)",
    }
    bundle["provenance"] = np.asarray(json.dumps(prov))
    if raw_dir_for_validation is not None:
        prov["raw_validation"] = validate_bundle_against_raw(bundle, raw_dir_for_validation)
        bundle["provenance"] = np.asarray(json.dumps(prov))
    return write_camels_bundle(bundle, out_path, compress=compress)


def validate_bundle_against_raw(
    bundle: Mapping[str, np.ndarray] | str | Path,
    raw_dir: str | Path,
    basins: Sequence[str] | None = None,
    qobs_atol: float = 1e-3,
) -> dict[str, object]:
    """Cross-check a bundle's forcings (exact) and discharge (``qobs_atol``) against raw CAMELS files.

    Only basins for which raw Daymet files exist are checked (the raw download may be partial); the
    returned summary states how many basins were checked. Any mismatch raises ``ValueError``.
    """
    if isinstance(bundle, (str, Path)):
        header = open_camels_bundle_header(bundle)
        bundle_path: Path | None = Path(bundle)
        all_basins = [str(b) for b in header["basins"].tolist()]
        dates = pd.DatetimeIndex(pd.to_datetime(header["dates"].astype(str)))
        hucs = [str(h).zfill(2) for h in header["huc_02"].tolist()]
        area_km2 = np.asarray(header["area_km2"], dtype=np.float64)
    else:
        validate_bundle_arrays(bundle, source="<in-memory bundle>")
        bundle_path = None
        all_basins = [str(b) for b in np.asarray(bundle["basins"]).tolist()]
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(bundle["dates"]).astype(str)))
        hucs = [str(h).zfill(2) for h in np.asarray(bundle["huc_02"]).tolist()]
        area_km2 = np.asarray(bundle["area_km2"], dtype=np.float64)

    raw_root = Path(raw_dir)
    if not is_raw_camels_dir(raw_root):
        raise FileNotFoundError(f"{raw_root} is not a raw CAMELS-US v1.2 directory.")
    check = _zfill_basins(basins) if basins is not None else all_basins
    b_index = {b: i for i, b in enumerate(all_basins)}
    unknown = [b for b in check if b not in b_index]
    if unknown:
        raise KeyError(f"Basins {unknown[:5]} are not in the bundle.")

    def _arr(key: str) -> np.ndarray:
        if bundle_path is not None:
            return _bundle_array(bundle_path, key)
        return np.asarray(bundle[key])  # type: ignore[index]

    checked = 0
    skipped: list[str] = []
    max_q_diff = 0.0
    forcing_cache = {p: _arr(f"forcing_{p}") for p in FORCING_PRODUCTS}
    qobs_all = _arr("qobs")
    for b in check:
        i = b_index[b]
        huc = hucs[i]
        if not raw_forcing_path(raw_root, "daymet", huc, b).is_file():
            skipped.append(b)
            continue
        for p in FORCING_PRODUCTS:
            fpath = raw_forcing_path(raw_root, p, huc, b)
            if not fpath.is_file():
                raise FileNotFoundError(f"Raw {p} forcing file missing for basin {b}: {fpath}")
            df, _lat, _elev, area_m2 = read_raw_forcing_file(fpath)
            raw_vals = df.reindex(dates)[list(FORCING_VARIABLES)].to_numpy(dtype=np.float32)
            bundle_vals = np.asarray(forcing_cache[p][i], dtype=np.float32)
            both_nan = np.isnan(raw_vals) & np.isnan(bundle_vals)
            if not np.array_equal(np.where(both_nan, 0.0, raw_vals), np.where(both_nan, 0.0, bundle_vals)):
                raise ValueError(f"Bundle {p} forcings differ from raw file for basin {b}.")
            if p == "daymet":
                if abs(area_m2 / 1.0e6 - float(area_km2[i])) > 1e-6 * max(1.0, float(area_km2[i])):
                    raise ValueError(
                        f"Bundle area_km2 {area_km2[i]} differs from Daymet header area {area_m2 / 1e6} for {b}."
                    )
                q_cfs = read_raw_streamflow_file(raw_streamflow_path(raw_root, huc, b), b).reindex(dates)
                q_mm = (q_cfs.to_numpy(dtype=np.float64) * CFS_TO_M3_PER_DAY * 1000.0 / area_m2).astype(np.float32)
                q_b = np.asarray(qobs_all[i], dtype=np.float32)
                nan_mismatch = np.isnan(q_mm) != np.isnan(q_b)
                if nan_mismatch.any():
                    raise ValueError(f"Bundle qobs NaN pattern differs from raw streamflow for basin {b}.")
                ok = ~np.isnan(q_mm)
                diff = float(np.max(np.abs(q_mm[ok] - q_b[ok]))) if ok.any() else 0.0
                max_q_diff = max(max_q_diff, diff)
                if diff > qobs_atol:
                    raise ValueError(f"Bundle qobs differs from raw streamflow for basin {b} (max abs diff {diff}).")
        checked += 1
    return {
        "raw_dir": str(raw_root),
        "basins_checked": checked,
        "basins_skipped_no_raw_file": len(skipped),
        "max_abs_qobs_diff_mm_per_day": max_q_diff,
        "forcings": "bit-identical for all checked basins and products",
    }


# =============================================================================
# Command line
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m models.camels_data",
        description="Build, convert, and validate the canonical CAMELS-US input bundle.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Compile a canonical bundle from a raw NCAR CAMELS-US v1.2 directory.")
    b.add_argument("--raw-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / CANONICAL_BUNDLE_FILENAME)
    b.add_argument("--attributes-dir", type=Path, default=None)
    b.add_argument("--start", type=str, default="1980-01-01")
    b.add_argument("--end", type=str, default="2014-12-31")
    b.add_argument("--basins-file", type=Path, default=None, help="Optional text file with one gauge ID per line.")
    b.add_argument("--no-compress", action="store_true")

    c = sub.add_parser("convert-legacy", help="Convert a legacy per-product .npz into the canonical schema.")
    c.add_argument("--legacy-npz", type=Path, required=True)
    c.add_argument("--attributes-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / CANONICAL_BUNDLE_FILENAME)
    c.add_argument("--raw-dir", type=Path, default=None, help="Optional raw directory for cross-validation.")
    c.add_argument("--no-compress", action="store_true")

    v = sub.add_parser("validate", help="Validate a bundle's schema and optionally cross-check against raw files.")
    v.add_argument("--bundle", type=Path, required=True)
    v.add_argument("--raw-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "build":
        basins = None
        if args.basins_file is not None:
            basins = [ln.strip() for ln in Path(args.basins_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = compile_camels_bundle(
            raw_dir=args.raw_dir,
            out_path=args.out,
            basins=basins,
            start=args.start,
            end=args.end,
            attributes_dir=args.attributes_dir,
            compress=not args.no_compress,
        )
        print(f"Wrote canonical bundle to {out}")
        return 0
    if args.command == "convert-legacy":
        out = convert_legacy_bundle(
            legacy_npz=args.legacy_npz,
            attributes_dir=args.attributes_dir,
            out_path=args.out,
            raw_dir_for_validation=args.raw_dir,
            compress=not args.no_compress,
        )
        print(f"Wrote canonical bundle to {out}")
        return 0
    header = open_camels_bundle_header(args.bundle)
    print(f"{args.bundle}: schema OK; {len(header['basins'])} basins, {len(header['dates'])} days, "
          f"{str(header['dates'][0])}..{str(header['dates'][-1])}")
    print("provenance:", str(header["provenance"]))
    if args.raw_dir is not None:
        summary = validate_bundle_against_raw(args.bundle, args.raw_dir)
        print("raw cross-validation:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
