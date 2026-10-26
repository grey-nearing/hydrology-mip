"""Centralized consolidated Zarr v2 hydrograph writer and validator for CAMELS-US MIP.

The attribute vocabulary written here is the one used by the stores already in
``gs://camels-us-mip/hydrographs/``, by ``gs://camels-us-mip/hydrographs/manifest_index.json``, and by
``metrics/evaluator.py`` (``CANONICAL_LEADERBOARD_IDS``):

* ``leaderboard_id``  in ``<spatial>__<forcing>`` form, e.g. ``gauged__single_camels`` (12 IDs);
* ``spatial_regime``  in ``{"gauged", "pub", "pur"}`` (``spatial_protocol`` carries the detailed
  ``gauged`` / ``pub_12fold`` / ``pur_18huc02`` protocol name);
* ``forcing_regime``  in ``{"single_camels", "multi_camels", "lagged_streamflow", "freeform"}``;
* ``provenance_category`` in ``{"Literature", "MIP Re-run", "Archive",
  "Archive*", "Contribution"}``;
* ``train_period`` / ``test_period`` as ``"YYYY-MM-DD/YYYY-MM-DD"``.

Repository-side aliases (``model_id``, ``model_family``, ``training_period``, ``evaluation_period``,
``data_url``, ``forcing_products``) are written in addition so both vocabularies resolve.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

EXPECTED_NUM_BASINS: int = 531
EXPECTED_NUM_DAYS: int = 3652

CANONICAL_TRAIN_START: str = "1999-10-01"
CANONICAL_TRAIN_END: str = "2008-09-30"
CANONICAL_EVAL_START: str = "1989-10-01"
CANONICAL_EVAL_END: str = "1999-09-30"

GCS_BUCKET_URI: str = "gs://camels-us-mip"
GCS_HYDROGRAPHS_PREFIX: str = f"{GCS_BUCKET_URI}/hydrographs"

VALID_PROVENANCE_CATEGORIES: tuple[str, ...] = (
    "Literature",
    "MIP Re-run",
    "Archive",
    "Archive*",
    "Contribution",
)
PROVENANCE_CATEGORY_ALIASES: dict[str, str] = {
    "literature": "Literature",
    "lit": "Literature",
    "mip re-run": "MIP Re-run",
    "mip-rerun": "MIP Re-run",
    "mip rerun": "MIP Re-run",
    "archive": "Archive",
    "archive*": "Archive*",
    "contribution": "Contribution",
    "contributed": "Contribution",
    "community": "Contribution",
}

# Experiment protocols accepted by `models/run_model.py` and their leaderboard prefixes.
VALID_SPATIAL_REGIMES: tuple[str, ...] = ("gauged", "pub_12fold", "pur_18huc02")
SPATIAL_PROTOCOL_TO_PREFIX: dict[str, str] = {
    "gauged": "gauged",
    "pub_12fold": "pub",
    "pur_18huc02": "pur",
}
VALID_SPATIAL_PREFIXES: tuple[str, ...] = ("gauged", "pub", "pur")
VALID_FORCING_REGIMES: tuple[str, ...] = (
    "single_camels",
    "multi_camels",
    "lagged_streamflow",
    "freeform",
)
VALID_HUC02_CODES: frozenset[str] = frozenset(f"{i:02d}" for i in range(1, 19))

_TABLE_NUMBERS: dict[tuple[str, str], int] = {
    ("gauged", "single_camels"): 3,
    ("gauged", "multi_camels"): 4,
    ("gauged", "lagged_streamflow"): 5,
    ("gauged", "freeform"): 6,
    ("pub", "single_camels"): 7,
    ("pub", "multi_camels"): 8,
    ("pub", "lagged_streamflow"): 9,
    ("pub", "freeform"): 10,
    ("pur", "single_camels"): 11,
    ("pur", "multi_camels"): 12,
    ("pur", "lagged_streamflow"): 13,
    ("pur", "freeform"): 14,
}
REGIME_TO_LEADERBOARD: dict[tuple[str, str], tuple[str, str]] = {
    (prefix, forcing): (f"{prefix}__{forcing}", f"table_{num}")
    for (prefix, forcing), num in _TABLE_NUMBERS.items()
}
CANONICAL_LEADERBOARD_IDS: tuple[str, ...] = tuple(
    REGIME_TO_LEADERBOARD[(prefix, forcing)][0]
    for prefix in VALID_SPATIAL_PREFIXES
    for forcing in VALID_FORCING_REGIMES
)
VALID_LEADERBOARD_IDS: frozenset[str] = frozenset(CANONICAL_LEADERBOARD_IDS)

# Attributes every hydrograph store must carry (bucket / manifest vocabulary).
REQUIRED_GLOBAL_ATTRS: tuple[str, ...] = (
    "entry_id",
    "model_name",
    "family_id",
    "provenance_category",
    "leaderboard_id",
    "spatial_regime",
    "forcing_regime",
    "forcings_used",
    "is_ensemble",
    "num_members",
    "train_period",
    "test_period",
    "units",
    "code_url",
    "archive_url",
)
# Aliases additionally written by this repository's writer.
REPO_ALIAS_ATTRS: tuple[str, ...] = (
    "model_id",
    "model_family",
    "spatial_protocol",
    "forcing_products",
    "training_period",
    "evaluation_period",
    "data_url",
)


def _period_to_slash(period: str | tuple[str, str]) -> str:
    """Normalize 'A to B', 'A/B', or (A, B) into the canonical 'A/B' attribute string."""
    if isinstance(period, tuple):
        a, b = period
    else:
        text = str(period).strip()
        if " to " in text:
            a, b = (s.strip() for s in text.split(" to ", 1))
        elif "/" in text:
            a, b = (s.strip() for s in text.split("/", 1))
        else:
            raise ValueError(f"Cannot parse period string {period!r}; expected 'A/B' or 'A to B'.")
    a_ts, b_ts = pd.Timestamp(a), pd.Timestamp(b)
    if b_ts < a_ts:
        raise ValueError(f"Period {period!r} ends before it starts.")
    return f"{a_ts.strftime('%Y-%m-%d')}/{b_ts.strftime('%Y-%m-%d')}"


def _period_bounds(period_slash: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = period_slash.split("/", 1)
    return pd.Timestamp(a), pd.Timestamp(b)


def normalize_provenance_category(value: str) -> str:
    """Map provenance category strings to the canonical word-based category name."""
    key = str(value).strip().lower()
    if key not in PROVENANCE_CATEGORY_ALIASES:
        raise ValueError(
            f"Invalid provenance_category {value!r}; must be one of {VALID_PROVENANCE_CATEGORIES}."
        )
    return PROVENANCE_CATEGORY_ALIASES[key]


def spatial_prefix_for(spatial_regime: str) -> str:
    """Return the leaderboard prefix ('gauged', 'pub', 'pur') for a protocol or prefix name."""
    s = str(spatial_regime).strip().lower()
    if s in SPATIAL_PROTOCOL_TO_PREFIX:
        return SPATIAL_PROTOCOL_TO_PREFIX[s]
    if s in VALID_SPATIAL_PREFIXES:
        return s
    raise ValueError(
        f"Invalid spatial_regime {spatial_regime!r}; must be one of {VALID_SPATIAL_REGIMES} "
        f"or {VALID_SPATIAL_PREFIXES}."
    )


def canonical_evaluation_dates(
    eval_start: str = CANONICAL_EVAL_START,
    eval_end: str = CANONICAL_EVAL_END,
) -> np.ndarray:
    """Return the daily evaluation date coordinate for [eval_start, eval_end] as datetime64[ns]."""
    dates = pd.date_range(eval_start, eval_end, freq="D").values.astype("datetime64[ns]")
    if dates.shape != (EXPECTED_NUM_DAYS,):
        raise ValueError(
            f"Evaluation date range {eval_start}..{eval_end} produced {dates.shape[0]} days; "
            f"expected {EXPECTED_NUM_DAYS}."
        )
    return dates


def resolve_leaderboard_and_table_id(
    spatial_regime: str,
    forcing_regime: str,
) -> tuple[str, str]:
    """Map (spatial protocol/prefix, forcing_regime) to the canonical (leaderboard_id, table_id)."""
    prefix = spatial_prefix_for(spatial_regime)
    if forcing_regime not in VALID_FORCING_REGIMES:
        raise ValueError(
            f"Unsupported forcing_regime {forcing_regime!r}; valid forcing regimes: {VALID_FORCING_REGIMES}."
        )
    return REGIME_TO_LEADERBOARD[(prefix, forcing_regime)]


def leaderboard_id_components(leaderboard_id: str) -> tuple[str, str]:
    """Split a canonical leaderboard_id into (spatial prefix, forcing regime)."""
    if leaderboard_id not in VALID_LEADERBOARD_IDS:
        raise ValueError(
            f"Invalid leaderboard_id {leaderboard_id!r}; must be one of {sorted(VALID_LEADERBOARD_IDS)}."
        )
    prefix, forcing = leaderboard_id.split("__", 1)
    return prefix, forcing


def default_store_uri(leaderboard_id: str, entry_id: str, bucket_uri: str = GCS_BUCKET_URI) -> str:
    """Canonical store URI `<bucket>/hydrographs/<leaderboard_id>/<entry_id>.zarr`."""
    if leaderboard_id not in VALID_LEADERBOARD_IDS:
        raise ValueError(f"Invalid leaderboard_id {leaderboard_id!r}.")
    if not str(entry_id).strip():
        raise ValueError("entry_id must be non-empty.")
    return f"{bucket_uri.rstrip('/')}/hydrographs/{leaderboard_id}/{entry_id}.zarr"


def build_hydrograph_dataset(
    qsim: np.ndarray,
    basins: Sequence[str],
    basin_to_huc02: Mapping[str, str],
    *,
    model_id: str,
    model_name: str,
    model_family: str,
    provenance_category: str,
    spatial_regime: str,
    forcing_regime: str,
    forcings_used: Sequence[str] | str,
    code_url: str,
    family_id: str | None = None,
    qsim_members: np.ndarray | None = None,
    is_ensemble: bool | None = None,
    num_members: int | None = None,
    leaderboard_id: str | None = None,
    table_id: str | None = None,
    training_period: str | tuple[str, str] = (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END),
    evaluation_period: str | tuple[str, str] = (CANONICAL_EVAL_START, CANONICAL_EVAL_END),
    units: str = "mm/day",
    data_url: str | None = None,
    dates: np.ndarray | None = None,
    extra_attrs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Build and strictly validate a canonical CAMELS-US MIP hydrograph ``xarray.Dataset``.

    The ``date`` coordinate is derived from ``evaluation_period`` (or taken from ``dates`` and checked
    against it) so a store can never carry dates that disagree with the period it was simulated for.
    """
    basin_list = [str(b).strip().zfill(8) for b in basins]
    if len(basin_list) != EXPECTED_NUM_BASINS:
        raise ValueError(f"Expected {EXPECTED_NUM_BASINS} basins, received {len(basin_list)}.")
    if len(set(basin_list)) != EXPECTED_NUM_BASINS:
        raise ValueError("Duplicate basin IDs detected in input basin sequence.")

    qsim_arr = np.asarray(qsim, dtype=np.float32)
    if qsim_arr.shape != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
        raise ValueError(
            f"qsim must have shape ({EXPECTED_NUM_BASINS}, {EXPECTED_NUM_DAYS}), got {qsim_arr.shape}."
        )

    sort_order = np.argsort(np.asarray(basin_list, dtype="<U8"))
    sorted_basins = np.asarray([basin_list[i] for i in sort_order], dtype="<U8")
    qsim_sorted = np.ascontiguousarray(qsim_arr[sort_order, :], dtype=np.float32)

    missing_hucs = [b for b in sorted_basins if str(b) not in basin_to_huc02]
    if missing_hucs:
        raise ValueError(
            f"Missing HUC-02 mapping for {len(missing_hucs)} basins (first: {missing_hucs[0]})."
        )
    huc_02_arr = np.asarray(
        [str(basin_to_huc02[str(b)]).zfill(2) for b in sorted_basins], dtype="<U2"
    )

    train_slash = _period_to_slash(training_period)
    test_slash = _period_to_slash(evaluation_period)
    tr_a, tr_b = _period_bounds(train_slash)
    te_a, te_b = _period_bounds(test_slash)
    if not (te_b < tr_a or tr_b < te_a):
        raise ValueError(
            f"training_period {train_slash} and evaluation_period {test_slash} overlap; "
            "the evaluation window must be disjoint from the training window."
        )
    expected_dates = pd.date_range(te_a, te_b, freq="D").values.astype("datetime64[ns]")
    if expected_dates.shape != (EXPECTED_NUM_DAYS,):
        raise ValueError(
            f"evaluation_period {test_slash} spans {expected_dates.shape[0]} days; "
            f"hydrograph stores must cover exactly {EXPECTED_NUM_DAYS} daily steps."
        )
    if dates is None:
        date_arr = expected_dates
    else:
        date_arr = np.asarray(dates, dtype="datetime64[ns]")
        if date_arr.shape != (EXPECTED_NUM_DAYS,) or not np.array_equal(date_arr, expected_dates):
            raise ValueError(
                f"dates do not match evaluation_period {test_slash} "
                f"({EXPECTED_NUM_DAYS} contiguous daily steps)."
            )

    prefix = spatial_prefix_for(spatial_regime)
    spatial_protocol = str(spatial_regime).strip().lower()
    if forcing_regime not in VALID_FORCING_REGIMES:
        raise ValueError(
            f"Invalid forcing_regime {forcing_regime!r}; must be one of {VALID_FORCING_REGIMES}."
        )
    default_lb, default_tbl = resolve_leaderboard_and_table_id(prefix, forcing_regime)
    leaderboard_id = leaderboard_id or default_lb
    table_id = table_id or default_tbl
    lb_prefix, lb_forcing = leaderboard_id_components(leaderboard_id)
    if lb_prefix != prefix or lb_forcing != forcing_regime:
        raise ValueError(
            f"leaderboard_id {leaderboard_id!r} is inconsistent with spatial_regime {spatial_regime!r} "
            f"and forcing_regime {forcing_regime!r} (expected {default_lb!r})."
        )
    if table_id != default_tbl:
        raise ValueError(f"table_id {table_id!r} does not match leaderboard {leaderboard_id!r} ({default_tbl}).")

    provenance = normalize_provenance_category(provenance_category)
    model_id_str = str(model_id).strip()
    if not model_id_str:
        raise ValueError("model_id must be a non-empty string.")
    family_id_str = str(family_id).strip() if family_id is not None else model_id_str

    if isinstance(forcings_used, str):
        forcings_label = forcings_used.strip()
        forcing_products = [s.strip().lower() for s in forcings_label.split(",") if s.strip()]
    else:
        forcing_products = [str(f).strip().lower() for f in forcings_used]
        forcings_label = ", ".join(
            {"daymet": "Daymet", "maurer": "Maurer", "nldas": "NLDAS"}.get(f, f) for f in forcing_products
        )
    if not forcing_products:
        raise ValueError("forcings_used must name at least one forcing product.")
    if forcing_regime == "single_camels" and len(forcing_products) != 1:
        raise ValueError(
            f"forcing_regime 'single_camels' requires exactly one forcing product, got {forcing_products}."
        )
    if forcing_regime == "multi_camels" and len(forcing_products) < 2:
        raise ValueError(
            f"forcing_regime 'multi_camels' requires at least two forcing products, got {forcing_products}."
        )

    if data_url is None:
        data_url = default_store_uri(leaderboard_id, model_id_str)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]] = {
        "qsim": (
            ("basin", "date"),
            qsim_sorted,
            {"units": units, "long_name": "Simulated daily streamflow"},
        ),
    }
    coords: dict[str, tuple[str, np.ndarray]] = {
        "basin": ("basin", sorted_basins),
        "date": ("date", date_arr),
        "huc_02": ("basin", huc_02_arr),
    }

    if qsim_members is not None:
        members_arr = np.asarray(qsim_members, dtype=np.float32)
        if members_arr.ndim != 3 or members_arr.shape[1:] != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
            raise ValueError(
                f"qsim_members must have shape (n_members, {EXPECTED_NUM_BASINS}, {EXPECTED_NUM_DAYS}), "
                f"got {members_arr.shape}."
            )
        n_mem = int(members_arr.shape[0])
        if n_mem < 1:
            raise ValueError("qsim_members must have at least 1 ensemble member.")
        members_sorted = np.ascontiguousarray(members_arr[:, sort_order, :], dtype=np.float32)
        data_vars["qsim_members"] = (
            ("member", "basin", "date"),
            members_sorted,
            {"units": units, "long_name": "Simulated daily streamflow per ensemble member"},
        )
        coords["member"] = ("member", np.arange(n_mem, dtype=np.int64))
        if num_members is None:
            num_members = n_mem
        if int(num_members) != n_mem:
            raise ValueError(f"num_members={num_members} but qsim_members has {n_mem} members.")
        if is_ensemble is None:
            is_ensemble = n_mem > 1
    else:
        if num_members is None:
            num_members = 1
        if is_ensemble is None:
            is_ensemble = bool(int(num_members) > 1)
    if bool(is_ensemble) != bool(int(num_members) > 1):
        raise ValueError(
            f"is_ensemble={is_ensemble} is inconsistent with num_members={num_members} "
            "(an ensemble has more than one member)."
        )

    attrs: dict[str, object] = {
        "entry_id": model_id_str,
        "model_id": model_id_str,
        "model_name": str(model_name),
        "family_id": family_id_str,
        "model_family": str(model_family),
        "provenance_category": provenance,
        "leaderboard_id": str(leaderboard_id),
        "table_id": str(table_id),
        "spatial_regime": prefix,
        "spatial_protocol": spatial_protocol,
        "forcing_regime": str(forcing_regime),
        "forcings_used": forcings_label,
        "forcing_products": forcing_products,
        "is_ensemble": bool(is_ensemble),
        "num_members": int(num_members),
        "train_period": train_slash,
        "test_period": test_slash,
        "training_period": train_slash.replace("/", " to "),
        "evaluation_period": test_slash.replace("/", " to "),
        "units": str(units),
        "code_url": str(code_url),
        "archive_url": str(data_url),
        "data_url": str(data_url),
    }
    if extra_attrs:
        for k, v in extra_attrs.items():
            if k in attrs:
                raise ValueError(f"extra_attrs may not override the canonical attribute {k!r}.")
            attrs[str(k)] = v

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    validate_hydrograph_dataset(ds, expected_basins=sorted_basins)
    return ds


def validate_hydrograph_dataset(
    ds: xr.Dataset,
    expected_basins: Sequence[str] | None = None,
) -> None:
    """Strictly validate a hydrograph ``xarray.Dataset`` against the CAMELS-US MIP schema."""
    if "qsim" not in ds.data_vars:
        raise ValueError("Dataset is missing required data variable 'qsim'.")

    qsim_da = ds["qsim"]
    if qsim_da.dims != ("basin", "date"):
        raise ValueError(f"qsim must have dimensions ('basin', 'date'), got {qsim_da.dims}.")
    if qsim_da.shape != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS):
        raise ValueError(
            f"qsim must have shape ({EXPECTED_NUM_BASINS}, {EXPECTED_NUM_DAYS}), got {qsim_da.shape}."
        )
    if qsim_da.dtype != np.float32:
        raise ValueError(f"qsim must have dtype float32, got {qsim_da.dtype}.")

    qsim_values = np.asarray(qsim_da.values)
    nan_count = int(np.isnan(qsim_values).sum())
    inf_count = int(np.isinf(qsim_values).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"qsim contains non-finite values: {nan_count} NaNs and {inf_count} Infs. "
            "All 531 basins x 3652 days must be finite."
        )

    for required_coord in ("basin", "date", "huc_02"):
        if required_coord not in ds.coords:
            raise ValueError(f"Dataset is missing required coordinate '{required_coord}'.")

    basin_vals = np.asarray(ds.coords["basin"].values)
    if basin_vals.shape != (EXPECTED_NUM_BASINS,):
        raise ValueError(
            f"Coordinate 'basin' must have shape ({EXPECTED_NUM_BASINS},), got {basin_vals.shape}."
        )
    if basin_vals.dtype != np.dtype("<U8"):
        raise ValueError(f"Coordinate 'basin' must have dtype '<U8', got {basin_vals.dtype}.")
    if not np.all(basin_vals[:-1] < basin_vals[1:]):
        raise ValueError("Coordinate 'basin' must be strictly sorted in ascending order.")

    if expected_basins is not None:
        exp_arr = np.asarray([str(b).strip().zfill(8) for b in expected_basins], dtype="<U8")
        if exp_arr.shape != (EXPECTED_NUM_BASINS,) or not np.array_equal(basin_vals, np.sort(exp_arr)):
            raise ValueError("Dataset 'basin' coordinate does not match the canonical 531 USGS basin list.")

    missing_attrs = [k for k in REQUIRED_GLOBAL_ATTRS if k not in ds.attrs]
    if missing_attrs:
        raise ValueError(f"Dataset is missing required global attributes: {missing_attrs}.")

    provenance = str(ds.attrs["provenance_category"])
    if provenance not in VALID_PROVENANCE_CATEGORIES:
        raise ValueError(
            f"Invalid provenance_category {provenance!r}; must be one of {VALID_PROVENANCE_CATEGORIES}."
        )
    leaderboard_id = str(ds.attrs["leaderboard_id"])
    lb_prefix, lb_forcing = leaderboard_id_components(leaderboard_id)
    spatial = str(ds.attrs["spatial_regime"])
    if spatial not in VALID_SPATIAL_PREFIXES:
        raise ValueError(
            f"Invalid spatial_regime {spatial!r}; must be one of {VALID_SPATIAL_PREFIXES}."
        )
    if spatial != lb_prefix:
        raise ValueError(
            f"spatial_regime {spatial!r} does not match leaderboard_id {leaderboard_id!r} prefix {lb_prefix!r}."
        )
    forcing = str(ds.attrs["forcing_regime"])
    if forcing not in VALID_FORCING_REGIMES:
        raise ValueError(f"Invalid forcing_regime {forcing!r}; must be one of {VALID_FORCING_REGIMES}.")
    if forcing != lb_forcing:
        raise ValueError(
            f"forcing_regime {forcing!r} does not match leaderboard_id {leaderboard_id!r} suffix {lb_forcing!r}."
        )
    if ds.attrs["units"] != "mm/day":
        raise ValueError(f"Invalid units {ds.attrs['units']!r}; must be 'mm/day'.")
    forcings_used = ds.attrs["forcings_used"]
    if not isinstance(forcings_used, str) or not forcings_used.strip():
        raise ValueError("Global attribute 'forcings_used' must be a non-empty string.")
    if "forcing_products" in ds.attrs:
        fp = ds.attrs["forcing_products"]
        if not isinstance(fp, (list, tuple, np.ndarray)) or len(fp) < 1:
            raise ValueError("Global attribute 'forcing_products' must be a non-empty list of strings.")
    for key in ("entry_id", "model_name", "family_id", "code_url", "archive_url"):
        if not isinstance(ds.attrs[key], str) or not str(ds.attrs[key]).strip():
            raise ValueError(f"Global attribute {key!r} must be a non-empty string.")

    num_members = int(ds.attrs["num_members"])
    if num_members < 1:
        raise ValueError("Global attribute 'num_members' must be >= 1.")
    is_ensemble = bool(ds.attrs["is_ensemble"])
    if is_ensemble != (num_members > 1):
        raise ValueError(
            f"is_ensemble={is_ensemble} is inconsistent with num_members={num_members}."
        )

    train_slash = _period_to_slash(str(ds.attrs["train_period"]))
    test_slash = _period_to_slash(str(ds.attrs["test_period"]))
    tr_a, tr_b = _period_bounds(train_slash)
    te_a, te_b = _period_bounds(test_slash)
    if not (te_b < tr_a or tr_b < te_a):
        raise ValueError(f"train_period {train_slash} overlaps test_period {test_slash}.")

    date_vals = np.asarray(ds.coords["date"].values)
    if date_vals.shape != (EXPECTED_NUM_DAYS,):
        raise ValueError(
            f"Coordinate 'date' must have shape ({EXPECTED_NUM_DAYS},), got {date_vals.shape}."
        )
    if date_vals.dtype != np.dtype("datetime64[ns]"):
        raise ValueError(
            f"Coordinate 'date' must have dtype 'datetime64[ns]', got {date_vals.dtype}."
        )
    expected_dates = pd.date_range(te_a, te_b, freq="D").values.astype("datetime64[ns]")
    if expected_dates.shape != (EXPECTED_NUM_DAYS,) or not np.array_equal(date_vals, expected_dates):
        raise ValueError(
            f"Coordinate 'date' does not match test_period {test_slash} "
            f"({EXPECTED_NUM_DAYS} contiguous daily steps)."
        )
    canonical_dates = canonical_evaluation_dates()
    if leaderboard_id in VALID_LEADERBOARD_IDS and not np.array_equal(date_vals, canonical_dates):
        raise ValueError(
            f"Leaderboard stores must cover the canonical evaluation window "
            f"{CANONICAL_EVAL_START} to {CANONICAL_EVAL_END}; got {test_slash}."
        )

    huc_da = ds.coords["huc_02"]
    if huc_da.dims != ("basin",):
        raise ValueError(f"Coordinate 'huc_02' must have dimension ('basin',), got {huc_da.dims}.")
    huc_vals = np.asarray(huc_da.values)
    if huc_vals.dtype != np.dtype("<U2"):
        raise ValueError(f"Coordinate 'huc_02' must have dtype '<U2', got {huc_vals.dtype}.")
    invalid_hucs = [str(h) for h in huc_vals if str(h) not in VALID_HUC02_CODES]
    if invalid_hucs:
        raise ValueError(
            f"Coordinate 'huc_02' contains invalid HUC-02 codes (first: {invalid_hucs[0]!r})."
        )

    if "qsim_members" in ds.data_vars:
        mem_da = ds["qsim_members"]
        if mem_da.dims != ("member", "basin", "date"):
            raise ValueError(
                f"qsim_members must have dimensions ('member', 'basin', 'date'), got {mem_da.dims}."
            )
        if mem_da.dtype != np.float32:
            raise ValueError(f"qsim_members must have dtype float32, got {mem_da.dtype}.")
        if mem_da.shape[1:] != (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS) or mem_da.shape[0] < 1:
            raise ValueError(f"qsim_members has invalid shape {mem_da.shape}.")
        mem_vals = np.asarray(mem_da.values)
        if not np.all(np.isfinite(mem_vals)):
            raise ValueError("qsim_members contains NaN or Inf values.")
        if int(mem_da.shape[0]) != num_members:
            raise ValueError(
                f"num_members={num_members} but qsim_members has {int(mem_da.shape[0])} members."
            )
        member_mean = np.mean(mem_vals.astype(np.float64), axis=0)
        if not np.allclose(qsim_values.astype(np.float64), member_mean, rtol=1e-4, atol=1e-4):
            raise ValueError("qsim is not the arithmetic mean of qsim_members (rtol/atol 1e-4).")


def write_hydrograph_zarr(
    ds: xr.Dataset,
    target_uri: str | Path | None = None,
    expected_basins: Sequence[str] | None = None,
) -> str:
    """Validate and write a consolidated Zarr v2 hydrograph store to local disk or GCS."""
    validate_hydrograph_dataset(ds, expected_basins=expected_basins)

    resolved_uri = str(target_uri) if target_uri is not None else str(ds.attrs["archive_url"])
    if not resolved_uri:
        raise ValueError("target_uri or ds.attrs['archive_url'] must be non-empty.")

    ds_to_write = ds.chunk({"basin": EXPECTED_NUM_BASINS, "date": EXPECTED_NUM_DAYS})
    encoding: dict[str, dict[str, object]] = {
        "qsim": {
            "chunks": (EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS),
            "dtype": "float32",
            "_FillValue": np.nan,
        },
        "basin": {"dtype": "<U8"},
        "huc_02": {"dtype": "<U2"},
    }
    if "qsim_members" in ds_to_write.data_vars:
        ds_to_write = ds_to_write.chunk(
            {"member": 1, "basin": EXPECTED_NUM_BASINS, "date": EXPECTED_NUM_DAYS}
        )
        encoding["qsim_members"] = {
            "chunks": (1, EXPECTED_NUM_BASINS, EXPECTED_NUM_DAYS),
            "dtype": "float32",
            "_FillValue": np.nan,
        }

    ds_to_write.to_zarr(
        resolved_uri,
        mode="w",
        consolidated=True,
        zarr_format=2,
        encoding=encoding,
    )
    return resolved_uri


def run_provenance_attrs(
    *,
    seeds: Sequence[int],
    fold_ids: Sequence[str],
    runner_kwargs: Mapping[str, object] | None,
    git_commit: str | None,
    data_source: str,
) -> dict[str, object]:
    """Provenance attributes recorded by `models/run_model.py` alongside the canonical attrs."""
    return {
        "generator": "camels-us-mip/models/run_model.py",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "seeds": [int(s) for s in seeds],
        "n_folds": int(len(fold_ids)),
        "fold_ids": [str(f) for f in fold_ids],
        "runner_kwargs_json": json.dumps(dict(runner_kwargs or {}), sort_keys=True, default=str),
        "git_commit": str(git_commit) if git_commit else "unknown",
        "data_source": str(data_source),
    }
