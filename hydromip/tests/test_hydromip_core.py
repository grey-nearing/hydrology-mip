# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and extensibility tests for the multi-MIP ``hydromip`` core framework."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from frontend.backend.app import app as frontend_app
from hydromip import (
    DEFAULT_MIP_ID,
    REPO_ROOT,
    CvFold,
    MetricSuiteSpec,
    discover_mips,
    get_metric_suite,
    get_mip,
    list_metric_suites,
    register_metric_suite,
    resolve_mip_paths,
    validate_cv_folds,
)
from hydromip.metrics.registry import METRIC_SUITES
from mips.camels_us_daily import load_canonical_basins, load_splits_for_regime
from models.run_model import MODEL_REGISTRY

HYDROMIP_PKG_DIR = REPO_ROOT / "hydromip"
MIPS_PKG_DIR = REPO_ROOT / "mips"


@pytest.mark.unit
def test_zero_try_except_blocks_in_hydromip_and_mips_adapters() -> None:
    """Enforce zero try/except blocks across ``hydromip/`` and ``mips/`` Python code."""
    py_files = sorted(HYDROMIP_PKG_DIR.rglob("*.py")) + sorted(MIPS_PKG_DIR.rglob("*.py"))
    assert py_files, "Expected Python modules under hydromip/ and mips/."
    for py_path in py_files:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        assert not try_nodes, f"Forbidden try/except block found in {py_path}"


@pytest.mark.unit
def test_discover_mips_finds_only_camels_us_daily_and_validates_assets() -> None:
    """Verify ``discover_mips`` registers ``camels_us_daily`` with complete assets."""
    mips = discover_mips()
    assert list(mips.keys()) == [DEFAULT_MIP_ID]

    proto = get_mip(DEFAULT_MIP_ID)
    assert proto.mip_id == "camels_us_daily"
    assert proto.task_type == "streamflow_daily_deterministic"
    assert proto.metric_suite == "gauch_2023_deterministic"
    assert proto.num_entities == 531
    assert proto.num_eval_steps == 3652
    assert len(proto.leaderboard_ids) == 12

    paths = resolve_mip_paths(DEFAULT_MIP_ID)
    assert paths.protocol_file.is_file()
    assert (paths.splits_dir / "basins_531.txt").is_file()
    assert (paths.splits_dir / "pub_12fold_random_splits.json").is_file()
    assert (paths.splits_dir / "pur_18fold_huc02_splits.json").is_file()
    assert (paths.leaderboards_dir / "table_2_registry.json").is_file()
    assert (paths.leaderboards_dir / "tables_3_to_14.json").is_file()
    assert (paths.figures_dir / "figures_manifest.json").is_file()
    assert paths.era_experiments_dir.is_dir()


@pytest.mark.unit
def test_camels_us_adapter_enforces_strict_cv_and_huc02_separation() -> None:
    """Verify ``mips.camels_us_daily`` adapter validates gauged, 12-fold PUB, and 18-fold PUR."""
    basins = load_canonical_basins()
    assert len(basins) == 531

    gauged = load_splits_for_regime("gauged")
    assert len(gauged) == 1
    assert len(gauged[0].test_entities) == 531

    pub = load_splits_for_regime("pub_12fold")
    assert len(pub) == 12
    assert sum(len(f.test_entities) for f in pub) == 531

    pur = load_splits_for_regime("pur_18huc02")
    assert len(pur) == 18
    assert sum(len(f.test_entities) for f in pur) == 531


@pytest.mark.unit
def test_validate_cv_folds_rejects_leakage_and_group_overlap() -> None:
    """Verify generic CV hygiene rejects train/test overlap and shared regional groups."""
    entities = ["b1", "b2", "b3", "b4"]
    entity_to_group = {"b1": "g1", "b2": "g1", "b3": "g2", "b4": "g2"}

    # Valid group-separated split passes
    good_folds = [
        CvFold("f0", 0, ("b3", "b4"), ("b1", "b2")),
        CvFold("f1", 1, ("b1", "b2"), ("b3", "b4")),
    ]
    validate_cv_folds(good_folds, entities, entity_to_group=entity_to_group)

    # Train/test entity leakage fails
    leaky_folds = [
        CvFold("f0", 0, ("b2", "b3", "b4"), ("b1", "b2")),
        CvFold("f1", 1, ("b1", "b2"), ("b3", "b4")),
    ]
    with pytest.raises(ValueError, match="CV leakage"):
        validate_cv_folds(leaky_folds, entities)

    # Group separation violation fails (b1 and b2 share group g1)
    bad_group_folds = [
        CvFold("f0", 0, ("b2", "b4"), ("b1", "b3")),
        CvFold("f1", 1, ("b1", "b3"), ("b2", "b4")),
    ]
    with pytest.raises(ValueError, match="Group separation violation"):
        validate_cv_folds(bad_group_folds, entities, entity_to_group=entity_to_group)


@pytest.mark.unit
def test_metric_suite_registry_is_pluggable_for_future_mips() -> None:
    """Verify new metric suites (e.g. probabilistic or flood) register cleanly without touching Gauch 2023."""
    gauch = get_metric_suite("gauch_2023_deterministic")
    assert gauch.primary_metric == "NSE"
    assert len(gauch.metric_names) == 12
    assert any(s["suite_id"] == "gauch_2023_deterministic" for s in list_metric_suites())

    custom_id = "_test_probabilistic_crps"
    custom = MetricSuiteSpec(
        suite_id=custom_id,
        description="Synthetic probabilistic metric suite test",
        metric_names=("CRPS", "PIT_Uniformity"),
        primary_metric="CRPS",
        higher_is_better=False,
        compute_fn=lambda obs, sim: {"CRPS": 0.0, "PIT_Uniformity": 1.0},
        citation="Test Suite",
    )
    register_metric_suite(custom)
    loaded = get_metric_suite(custom_id)
    assert loaded.primary_metric == "CRPS"
    assert loaded.higher_is_better is False
    del METRIC_SUITES[custom_id]


@pytest.mark.unit
def test_model_catalog_matches_registered_models() -> None:
    """Verify ``models/catalog.json`` covers every model in ``MODEL_REGISTRY`` with real directories."""
    catalog_path = REPO_ROOT / "models" / "catalog.json"
    assert catalog_path.is_file()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    keys = {entry["key"] for entry in catalog["models"]}
    assert keys == set(MODEL_REGISTRY.keys())
    for key in keys:
        assert (REPO_ROOT / "models" / key / "runner.py").is_file()


@pytest.mark.integration
def test_adding_a_future_mip_is_purely_additive_and_auto_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify dropping a second MIP under ``mips/<new_id>/protocol.json`` is auto-discovered by core and API."""
    # Mirror camles_us_daily protocol plus add a synthetic soil_moisture MIP
    camels_dir = tmp_path / "mips" / "camels_us_daily"
    camels_dir.mkdir(parents=True, exist_ok=True)
    camels_proto = (REPO_ROOT / "mips" / "camels_us_daily" / "protocol.json").read_text(encoding="utf-8")
    (camels_dir / "protocol.json").write_text(camels_proto, encoding="utf-8")

    sm_dir = tmp_path / "mips" / "ismn_soil_moisture"
    sm_dir.mkdir(parents=True, exist_ok=True)
    sm_proto = {
        "schema_version": "hydromip-protocol-v1",
        "mip_id": "ismn_soil_moisture",
        "name": "ISMN Daily Soil Moisture MIP",
        "short_name": "Soil Moisture",
        "status": "planned",
        "version": "0.1.0",
        "task_type": "soil_moisture_daily",
        "metric_suite": "soil_moisture_daily",
        "primary_metric": "ubRMSE",
        "primary_metric_higher_is_better": False,
        "target_variable": "soil_moisture",
        "target_units": "m3/m3",
        "prediction_dims": ["station", "date"],
        "entity_unit": "station",
        "num_entities": 800,
        "num_eval_steps": 1826,
        "temporal_windows": {"train_start": "2010-01-01", "train_end": "2018-12-31"},
        "spatial_regimes": [
            {
                "id": "gauged",
                "leaderboard_prefix": "gauged",
                "title": "In-Situ Stations",
                "num_folds": 1,
                "description": "In-situ temporal split.",
            }
        ],
        "forcing_regimes": [
            {
                "id": "era5land",
                "title": "ERA5-Land",
                "description": "ERA5-Land meteorological forcing.",
            }
        ],
        "leaderboards": [
            {
                "id": "gauged__era5land",
                "spatial_regime": "gauged",
                "forcing_regime": "era5land",
                "title": "In-Situ Stations — ERA5-Land",
            }
        ],
        "storage": {"bucket_uri": "gs://hydromip/ismn_soil_moisture"},
    }
    (sm_dir / "protocol.json").write_text(json.dumps(sm_proto, indent=2), encoding="utf-8")

    discovered = discover_mips(repo_root=tmp_path)
    assert set(discovered.keys()) == {"camels_us_daily", "ismn_soil_moisture"}
    assert discovered["ismn_soil_moisture"].primary_metric == "ubRMSE"
    assert discovered["ismn_soil_moisture"].primary_metric_higher_is_better is False

    # Verify the web portal API auto-discovers both MIPs without code changes
    monkeypatch.setenv("CAMELS_MIP_DATA_DIR", str(tmp_path))
    client = TestClient(frontend_app)
    resp = client.get("/api/mips")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["n_mips"] == 2
    ids = {m["mip_id"] for m in payload["mips"]}
    assert ids == {"camels_us_daily", "ismn_soil_moisture"}

    resp_sm = client.get("/api/mips/ismn_soil_moisture/protocol")
    assert resp_sm.status_code == 200
    assert resp_sm.json()["task_type"] == "soil_moisture_daily"


@pytest.mark.integration
def test_frontend_splits_and_figures_endpoints_work_with_multi_mip_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``/api/mips``, ``/api/splits/*``, and ``/api/figures/*`` serve CAMELS-US files."""
    monkeypatch.delenv("CAMELS_MIP_DATA_DIR", raising=False)
    monkeypatch.delenv("HYDROMIP_DATA_DIR", raising=False)
    client = TestClient(frontend_app)

    assert client.get("/api/mips").json()["n_mips"] == 1
    assert client.get("/api/splits/basins_531.txt").status_code == 200
    assert client.get("/api/mips/camels_us_daily/splits/pub_12fold_random_splits.json").status_code == 200
    assert client.get("/api/figures/figures_manifest.json").status_code == 200
    assert client.get("/api/mips/camels_us_daily/figures/figure_1_water_year_coverage.png").status_code == 200
