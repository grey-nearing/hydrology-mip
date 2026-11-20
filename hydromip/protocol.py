# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Declarative MIP protocol schema and auto-discovery for HydroMIP.

Every benchmark in ``mips/<mip_id>/`` ships a ``protocol.json`` file describing its
scientific task, target variable & units, prediction tensor dimensions, temporal windows,
spatial/input regimes, metric suite, leaderboard taxonomy, and cloud storage locations.

Adding a future MIP (e.g., Caravan global daily streamflow, probabilistic ensemble
streamflow, soil moisture, 2D flood inundation, real-time operational, or weather-driven
forecasting) requires adding a new directory ``mips/<new_mip_id>/protocol.json`` without
editing existing MIPs or core orchestration code.

Zero ``try``/``except`` blocks. All schema preconditions are validated explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydromip.paths import DEFAULT_MIP_ID, MipPaths, get_repo_root, resolve_mip_paths

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
    """Specification for one canonical leaderboard table inside a MIP."""

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


def validate_protocol_dict(
    payload: dict[str, Any],
    expected_mip_id: str | None = None,
) -> None:
    """Validate raw ``protocol.json`` dictionary preconditions strictly."""
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
    mip_id = str(payload["mip_id"]).strip()
    if not mip_id:
        raise ValueError("protocol.json 'mip_id' must be a non-empty string.")
    if expected_mip_id is not None and mip_id != expected_mip_id:
        raise ValueError(
            f"protocol.json mip_id={mip_id!r} does not match directory name {expected_mip_id!r}."
        )
    task_type = str(payload["task_type"]).strip()
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type {task_type!r}; expected one of {sorted(SUPPORTED_TASK_TYPES)}."
        )
    if not isinstance(payload["prediction_dims"], list) or not payload["prediction_dims"]:
        raise ValueError("protocol.json 'prediction_dims' must be a non-empty list of dimension names.")
    if not isinstance(payload["num_entities"], int) or payload["num_entities"] <= 0:
        raise ValueError("protocol.json 'num_entities' must be a positive integer.")
    if not isinstance(payload["num_eval_steps"], int) or payload["num_eval_steps"] <= 0:
        raise ValueError("protocol.json 'num_eval_steps' must be a positive integer.")
    if not isinstance(payload["spatial_regimes"], list) or not payload["spatial_regimes"]:
        raise ValueError("protocol.json 'spatial_regimes' must be a non-empty list.")
    if not isinstance(payload["forcing_regimes"], list) or not payload["forcing_regimes"]:
        raise ValueError("protocol.json 'forcing_regimes' must be a non-empty list.")
    if not isinstance(payload["leaderboards"], list) or not payload["leaderboards"]:
        raise ValueError("protocol.json 'leaderboards' must be a non-empty list.")

    lb_ids = [str(lb["id"]) for lb in payload["leaderboards"]]
    if len(set(lb_ids)) != len(lb_ids):
        raise ValueError(f"Duplicate leaderboard IDs in protocol.json for {mip_id!r}: {lb_ids}")


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
