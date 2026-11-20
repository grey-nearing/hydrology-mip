# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Declarative MIP protocol schema and auto-discovery for HydrologyMIP.

Every benchmark in ``mips/<mip_id>/`` ships a ``protocol.json`` file describing its
scientific task, target variable & units, prediction tensor dimensions, temporal windows,
spatial/input regimes, metric suite, leaderboard taxonomy, and cloud storage locations.

Adding a future MIP (e.g., Caravan global daily streamflow, probabilistic ensemble
streamflow, soil moisture, 2D flood inundation, real-time operational, or weather-driven
forecasting) requires adding a new directory ``mips/<new_mip_id>/protocol.json`` (and, if
needed, registering a new metric suite in ``hydromip.metrics.registry``) without editing
existing MIPs or core orchestration code.

Zero ``try``/``except`` blocks. All schema preconditions are validated explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydromip.metrics.registry import METRIC_SUITES
from hydromip.paths import (
    DEFAULT_MIP_ID,
    MIP_SUBDIRS,
    MipPaths,
    get_repo_root,
    resolve_mip_paths,
    validate_mip_id,
)

SCHEMA_VERSION: str = "hydromip-protocol-v1"

SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(
    {
        "streamflow_daily_deterministic",
        "streamflow_daily_probabilistic",
        "streamflow_forecast_deterministic",
        "streamflow_forecast_probabilistic",
        "streamflow_realtime",
        "soil_moisture_daily",
        "flood_inundation_2d",
    }
)

SUPPORTED_STATUSES: frozenset[str] = frozenset(
    {"active", "beta", "planned", "deprecated", "archived"}
)


@dataclass(frozen=True)
class SpatialRegimeSpec:
    """Specification for one spatial evaluation regime (e.g. gauged, PUB, PUR, leave-one-CAMELS-out)."""

    id: str
    leaderboard_prefix: str
    title: str
    num_folds: int
    description: str
    split_file: str | None = None
    section: str | None = None


@dataclass(frozen=True)
class ForcingRegimeSpec:
    """Specification for one input/forcing regime column in a MIP leaderboard matrix."""

    id: str
    title: str
    description: str


@dataclass(frozen=True)
class LeaderboardSpec:
    """Specification for one canonical leaderboard table inside a MIP.

    ``spatial_regime`` and ``forcing_regime`` reference ``SpatialRegimeSpec.id`` and
    ``ForcingRegimeSpec.id``; ``id`` is ``f"{leaderboard_prefix}__{forcing_regime}"``.
    """

    id: str
    spatial_regime: str
    forcing_regime: str
    title: str
    table_number: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class MipProtocol:
    """Validated declarative protocol specification loaded from ``mips/<mip_id>/protocol.json``."""

    schema_version: str
    mip_id: str
    name: str
    short_name: str
    status: str
    version: str
    task_type: str
    metric_suite: str
    primary_metric: str
    primary_metric_higher_is_better: bool
    target_variable: str
    target_units: str
    prediction_dims: tuple[str, ...]
    entity_unit: str
    num_entities: int
    num_eval_steps: int
    temporal_windows: dict[str, Any]
    spatial_regimes: tuple[SpatialRegimeSpec, ...]
    forcing_regimes: tuple[ForcingRegimeSpec, ...]
    leaderboards: tuple[LeaderboardSpec, ...]
    storage: dict[str, Any]
    paths: MipPaths
    raw: dict[str, Any]

    @property
    def leaderboard_ids(self) -> tuple[str, ...]:
        """Ordered tuple of all canonical leaderboard IDs declared by this MIP."""
        return tuple(lb.id for lb in self.leaderboards)

    @property
    def spatial_regime_ids(self) -> tuple[str, ...]:
        """Ordered tuple of all spatial regime IDs declared by this MIP."""
        return tuple(sr.id for sr in self.spatial_regimes)

    @property
    def forcing_regime_ids(self) -> tuple[str, ...]:
        """Ordered tuple of all forcing/input regime IDs declared by this MIP."""
        return tuple(fr.id for fr in self.forcing_regimes)

    def get_spatial_regime(self, regime_id: str) -> SpatialRegimeSpec:
        """Return the spatial regime with ``id == regime_id`` or raise ``KeyError``."""
        for sr in self.spatial_regimes:
            if sr.id == regime_id:
                return sr
        raise KeyError(
            f"Unknown spatial regime {regime_id!r} for MIP {self.mip_id!r}; "
            f"declared: {list(self.spatial_regime_ids)}"
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dictionary suitable for APIs and portals."""
        return {
            "schema_version": self.schema_version,
            "mip_id": self.mip_id,
            "name": self.name,
            "short_name": self.short_name,
            "status": self.status,
            "version": self.version,
            "task_type": self.task_type,
            "metric_suite": self.metric_suite,
            "primary_metric": self.primary_metric,
            "primary_metric_higher_is_better": self.primary_metric_higher_is_better,
            "target_variable": self.target_variable,
            "target_units": self.target_units,
            "prediction_dims": list(self.prediction_dims),
            "entity_unit": self.entity_unit,
            "num_entities": self.num_entities,
            "num_eval_steps": self.num_eval_steps,
            "temporal_windows": self.temporal_windows,
            "n_spatial_regimes": len(self.spatial_regimes),
            "n_forcing_regimes": len(self.forcing_regimes),
            "n_leaderboards": len(self.leaderboards),
            "leaderboard_ids": list(self.leaderboard_ids),
            "spatial_regimes": [
                {
                    "id": sr.id,
                    "leaderboard_prefix": sr.leaderboard_prefix,
                    "title": sr.title,
                    "num_folds": sr.num_folds,
                    "split_file": sr.split_file,
                    "section": sr.section,
                    "description": sr.description,
                }
                for sr in self.spatial_regimes
            ],
            "forcing_regimes": [
                {
                    "id": fr.id,
                    "title": fr.title,
                    "description": fr.description,
                }
                for fr in self.forcing_regimes
            ],
            "storage": self.storage,
        }


_REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "mip_id",
    "name",
    "short_name",
    "status",
    "version",
    "task_type",
    "metric_suite",
    "primary_metric",
    "primary_metric_higher_is_better",
    "target_variable",
    "target_units",
    "prediction_dims",
    "entity_unit",
    "num_entities",
    "num_eval_steps",
    "temporal_windows",
    "spatial_regimes",
    "forcing_regimes",
    "leaderboards",
    "storage",
)
_REQUIRED_STRING_KEYS: tuple[str, ...] = (
    "name",
    "short_name",
    "status",
    "version",
    "task_type",
    "metric_suite",
    "primary_metric",
    "target_variable",
    "target_units",
    "entity_unit",
)
_REQUIRED_SPATIAL_KEYS: tuple[str, ...] = (
    "id",
    "leaderboard_prefix",
    "title",
    "num_folds",
    "description",
)
_REQUIRED_FORCING_KEYS: tuple[str, ...] = ("id", "title", "description")
_REQUIRED_LEADERBOARD_KEYS: tuple[str, ...] = ("id", "spatial_regime", "forcing_regime", "title")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require_entries(
    payload: dict[str, Any], key: str, required: tuple[str, ...], mip_id: str
) -> list[dict[str, Any]]:
    entries = payload[key]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"protocol.json '{key}' must be a non-empty list ({mip_id!r}).")
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"protocol.json '{key}[{idx}]' must be an object ({mip_id!r}).")
        missing = [k for k in required if k not in entry]
        if missing:
            raise KeyError(f"protocol.json '{key}[{idx}]' is missing keys {missing} ({mip_id!r}).")
        for k in required:
            if k != "num_folds" and (not isinstance(entry[k], str) or not entry[k].strip()):
                raise ValueError(
                    f"protocol.json '{key}[{idx}].{k}' must be a non-empty string ({mip_id!r})."
                )
    return entries


def _require_unique(values: list[str], what: str, mip_id: str) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {what} in protocol.json for {mip_id!r}: {duplicates}")


def validate_protocol_dict(
    payload: dict[str, Any],
    expected_mip_id: str | None = None,
) -> None:
    """Validate raw ``protocol.json`` dictionary preconditions strictly.

    Checks required keys and types, the registered metric suite and primary metric,
    unique regime and leaderboard identifiers, and the referential integrity of every
    leaderboard (``spatial_regime`` / ``forcing_regime`` must reference declared regime
    ``id`` values and ``id`` must equal ``f"{leaderboard_prefix}__{forcing_regime}"``).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"protocol.json must contain a JSON object, got {type(payload).__name__}.")
    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in payload]
    if missing:
        raise KeyError(f"protocol.json missing required keys: {missing}")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported protocol schema_version {payload['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}."
        )
    if not isinstance(payload["mip_id"], str):
        raise ValueError("protocol.json 'mip_id' must be a string.")
    mip_id = validate_mip_id(payload["mip_id"])
    if expected_mip_id is not None and mip_id != expected_mip_id:
        raise ValueError(
            f"protocol.json mip_id={mip_id!r} does not match directory name {expected_mip_id!r}."
        )
    for key in _REQUIRED_STRING_KEYS:
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"protocol.json '{key}' must be a non-empty string ({mip_id!r}).")
    if payload["status"] not in SUPPORTED_STATUSES:
        raise ValueError(
            f"Unsupported status {payload['status']!r}; expected one of {sorted(SUPPORTED_STATUSES)}."
        )
    if payload["task_type"] not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type {payload['task_type']!r}; "
            f"expected one of {sorted(SUPPORTED_TASK_TYPES)}."
        )
    if not isinstance(payload["primary_metric_higher_is_better"], bool):
        raise ValueError("protocol.json 'primary_metric_higher_is_better' must be a boolean.")

    suite_id = payload["metric_suite"]
    if suite_id not in METRIC_SUITES:
        raise ValueError(
            f"protocol.json metric_suite {suite_id!r} is not registered; register it via "
            f"hydromip.metrics.register_metric_suite (registered: {sorted(METRIC_SUITES)})."
        )
    suite = METRIC_SUITES[suite_id]
    if payload["primary_metric"] not in suite.metric_names:
        raise ValueError(
            f"primary_metric {payload['primary_metric']!r} is not part of metric suite "
            f"{suite_id!r} {list(suite.metric_names)}."
        )
    if (
        payload["primary_metric"] == suite.primary_metric
        and payload["primary_metric_higher_is_better"] != suite.higher_is_better
    ):
        raise ValueError(
            f"primary_metric_higher_is_better={payload['primary_metric_higher_is_better']} "
            f"contradicts metric suite {suite_id!r} (higher_is_better={suite.higher_is_better})."
        )

    dims = payload["prediction_dims"]
    if (
        not isinstance(dims, list)
        or not dims
        or not all(isinstance(d, str) and d.strip() for d in dims)
    ):
        raise ValueError("protocol.json 'prediction_dims' must be a non-empty list of dimension names.")
    _require_unique(list(dims), "prediction_dims", mip_id)
    if payload["entity_unit"] not in dims:
        raise ValueError(
            f"protocol.json entity_unit {payload['entity_unit']!r} must be one of prediction_dims {dims}."
        )
    if not _is_positive_int(payload["num_entities"]):
        raise ValueError("protocol.json 'num_entities' must be a positive integer.")
    if not _is_positive_int(payload["num_eval_steps"]):
        raise ValueError("protocol.json 'num_eval_steps' must be a positive integer.")
    if not isinstance(payload["temporal_windows"], dict) or not payload["temporal_windows"]:
        raise ValueError("protocol.json 'temporal_windows' must be a non-empty object.")
    if not isinstance(payload["storage"], dict):
        raise ValueError("protocol.json 'storage' must be an object.")
    if "paths" in payload:
        paths_block = payload["paths"]
        if not isinstance(paths_block, dict) or any(
            k not in MIP_SUBDIRS or v != k for k, v in paths_block.items()
        ):
            raise ValueError(
                "protocol.json 'paths' may only restate the canonical subdirectory names "
                f"{MIP_SUBDIRS}; custom per-MIP paths are not supported."
            )

    spatial = _require_entries(payload, "spatial_regimes", _REQUIRED_SPATIAL_KEYS, mip_id)
    for sr in spatial:
        if not _is_positive_int(sr["num_folds"]):
            raise ValueError(f"Spatial regime {sr['id']!r} 'num_folds' must be a positive integer.")
        split_file = sr.get("split_file")
        if split_file is not None and (not isinstance(split_file, str) or not split_file.strip()):
            raise ValueError(f"Spatial regime {sr['id']!r} 'split_file' must be null or a file name.")
        if sr["num_folds"] > 1 and split_file is None:
            raise ValueError(
                f"Spatial regime {sr['id']!r} declares {sr['num_folds']} folds but no 'split_file'."
            )
    _require_unique([str(sr["id"]) for sr in spatial], "spatial regime ids", mip_id)
    _require_unique(
        [str(sr["leaderboard_prefix"]) for sr in spatial], "spatial leaderboard prefixes", mip_id
    )

    forcing = _require_entries(payload, "forcing_regimes", _REQUIRED_FORCING_KEYS, mip_id)
    _require_unique([str(fr["id"]) for fr in forcing], "forcing regime ids", mip_id)

    leaderboards = _require_entries(payload, "leaderboards", _REQUIRED_LEADERBOARD_KEYS, mip_id)
    lb_ids = [str(lb["id"]) for lb in leaderboards]
    if len(set(lb_ids)) != len(lb_ids):
        raise ValueError(f"Duplicate leaderboard IDs in protocol.json for {mip_id!r}: {lb_ids}")
    prefix_by_regime = {str(sr["id"]): str(sr["leaderboard_prefix"]) for sr in spatial}
    forcing_ids = {str(fr["id"]) for fr in forcing}
    seen_cells: set[tuple[str, str]] = set()
    for lb in leaderboards:
        regime = str(lb["spatial_regime"])
        forcing_id = str(lb["forcing_regime"])
        if regime not in prefix_by_regime:
            raise ValueError(
                f"Leaderboard {lb['id']!r} references undeclared spatial_regime {regime!r}; "
                f"declared spatial regime ids: {sorted(prefix_by_regime)}."
            )
        if forcing_id not in forcing_ids:
            raise ValueError(
                f"Leaderboard {lb['id']!r} references undeclared forcing_regime {forcing_id!r}; "
                f"declared forcing regime ids: {sorted(forcing_ids)}."
            )
        expected_id = f"{prefix_by_regime[regime]}__{forcing_id}"
        if lb["id"] != expected_id:
            raise ValueError(
                f"Leaderboard id {lb['id']!r} must equal '<leaderboard_prefix>__<forcing_regime>' "
                f"= {expected_id!r}."
            )
        if (regime, forcing_id) in seen_cells:
            raise ValueError(f"Duplicate leaderboard cell ({regime!r}, {forcing_id!r}) in {mip_id!r}.")
        seen_cells.add((regime, forcing_id))
        table_number = lb.get("table_number")
        if table_number is not None and not _is_positive_int(table_number):
            raise ValueError(f"Leaderboard {lb['id']!r} 'table_number' must be null or a positive integer.")


def load_mip_protocol(
    protocol_path: str | Path,
    repo_root: str | Path | None = None,
) -> MipProtocol:
    """Load and validate a single ``protocol.json`` file into a ``MipProtocol``."""
    p = Path(protocol_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"MIP protocol file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    expected_id = p.parent.name
    validate_protocol_dict(payload, expected_mip_id=expected_id)

    resolved_root = get_repo_root(repo_root) if repo_root is not None else p.parents[2]
    mip_paths = resolve_mip_paths(mip_id=str(payload["mip_id"]), repo_root=resolved_root)

    spatial_regimes = tuple(
        SpatialRegimeSpec(
            id=str(sr["id"]),
            leaderboard_prefix=str(sr["leaderboard_prefix"]),
            title=str(sr["title"]),
            num_folds=int(sr["num_folds"]),
            description=str(sr["description"]),
            split_file=str(sr["split_file"]) if sr.get("split_file") else None,
            section=str(sr["section"]) if sr.get("section") else None,
        )
        for sr in payload["spatial_regimes"]
    )
    forcing_regimes = tuple(
        ForcingRegimeSpec(
            id=str(fr["id"]),
            title=str(fr["title"]),
            description=str(fr["description"]),
        )
        for fr in payload["forcing_regimes"]
    )
    leaderboards = tuple(
        LeaderboardSpec(
            id=str(lb["id"]),
            spatial_regime=str(lb["spatial_regime"]),
            forcing_regime=str(lb["forcing_regime"]),
            title=str(lb["title"]),
            table_number=int(lb["table_number"]) if lb.get("table_number") is not None else None,
            section=str(lb["section"]) if lb.get("section") else None,
        )
        for lb in payload["leaderboards"]
    )

    return MipProtocol(
        schema_version=str(payload["schema_version"]),
        mip_id=str(payload["mip_id"]),
        name=str(payload["name"]),
        short_name=str(payload["short_name"]),
        status=str(payload["status"]),
        version=str(payload["version"]),
        task_type=str(payload["task_type"]),
        metric_suite=str(payload["metric_suite"]),
        primary_metric=str(payload["primary_metric"]),
        primary_metric_higher_is_better=bool(payload["primary_metric_higher_is_better"]),
        target_variable=str(payload["target_variable"]),
        target_units=str(payload["target_units"]),
        prediction_dims=tuple(str(d) for d in payload["prediction_dims"]),
        entity_unit=str(payload["entity_unit"]),
        num_entities=int(payload["num_entities"]),
        num_eval_steps=int(payload["num_eval_steps"]),
        temporal_windows=dict(payload["temporal_windows"]),
        spatial_regimes=spatial_regimes,
        forcing_regimes=forcing_regimes,
        leaderboards=leaderboards,
        storage=dict(payload["storage"]),
        paths=mip_paths,
        raw=payload,
    )


def validate_mip_assets(proto: MipProtocol) -> None:
    """Verify that the on-disk assets referenced by ``proto`` exist and are consistent.

    Checks ``splits/`` (each declared ``split_file`` exists and, for JSON split files,
    defines exactly ``num_folds`` folds), ``leaderboards/`` (one ``<leaderboard_id>.json``
    per declared leaderboard whose ``id`` matches), and ``figures/``. Used by tests and CI;
    protocol loading itself stays lightweight.
    """
    paths = proto.paths
    for directory in (paths.splits_dir, paths.leaderboards_dir, paths.figures_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"MIP {proto.mip_id!r} is missing directory {directory}")
    for sr in proto.spatial_regimes:
        if sr.split_file is None:
            continue
        split_path = paths.splits_dir / sr.split_file
        if not split_path.is_file():
            raise FileNotFoundError(f"Spatial regime {sr.id!r} split file not found: {split_path}")
        if split_path.suffix == ".json":
            split_payload = json.loads(split_path.read_text(encoding="utf-8"))
            folds = split_payload["folds"] if isinstance(split_payload, dict) else None
            if not isinstance(folds, dict) or len(folds) != sr.num_folds:
                raise ValueError(
                    f"Split file {split_path} must define {sr.num_folds} folds under 'folds'."
                )
    for lb in proto.leaderboards:
        lb_path = paths.leaderboards_dir / f"{lb.id}.json"
        if not lb_path.is_file():
            raise FileNotFoundError(f"Leaderboard file not found for {lb.id!r}: {lb_path}")
        lb_payload = json.loads(lb_path.read_text(encoding="utf-8"))
        if not isinstance(lb_payload, dict) or lb_payload.get("id") != lb.id:
            raise ValueError(f"Leaderboard file {lb_path} must be an object with id {lb.id!r}.")


def discover_mips(repo_root: str | Path | None = None) -> dict[str, MipProtocol]:
    """Auto-discover all registered MIPs under ``<repo_root>/mips/*/protocol.json``."""
    root = get_repo_root(repo_root)
    mips_root = root / "mips"
    if not mips_root.is_dir():
        raise FileNotFoundError(f"MIP registry directory not found: {mips_root}")
    discovered: dict[str, MipProtocol] = {}
    for entry in sorted(mips_root.iterdir()):
        if not entry.is_dir():
            continue
        proto_path = entry / "protocol.json"
        if proto_path.is_file():
            proto = load_mip_protocol(proto_path, repo_root=root)
            discovered[proto.mip_id] = proto
    if not discovered:
        raise FileNotFoundError(f"No MIP protocol.json files found under {mips_root}")
    return discovered


def get_mip(
    mip_id: str = DEFAULT_MIP_ID,
    repo_root: str | Path | None = None,
) -> MipProtocol:
    """Load one registered MIP protocol by ``mip_id``, raising ``KeyError`` if unknown."""
    mips = discover_mips(repo_root=repo_root)
    if mip_id not in mips:
        raise KeyError(
            f"Unknown mip_id {mip_id!r}; registered MIPs: {sorted(mips.keys())}"
        )
    return mips[mip_id]
