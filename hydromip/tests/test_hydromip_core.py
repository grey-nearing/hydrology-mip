# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and extensibility tests for the multi-MIP ``hydromip`` core framework."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

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
    resolve_mip_subdir,
    validate_cv_folds,
    validate_protocol_dict,
)
from hydromip.evaluation import zarr_io
from hydromip.evaluation.zarr_io import get_gcs_storage_options, validate_store_id
from hydromip.metrics.registry import METRIC_SUITES
from hydromip.protocol import validate_mip_assets
from hydromip.tools.ci_guardrails import (
    TRY_EXCEPT_FUNCTION_ALLOWLIST,
    _ast_try_types,
    _iter_try_nodes_with_scope,
)
from mips.camels_us_daily import load_canonical_basins, load_splits_for_regime

HYDROMIP_PKG_DIR = REPO_ROOT / "hydromip"
MIPS_PKG_DIR = REPO_ROOT / "mips"
#: Packages whose non-test Python code must only contain allowlisted boundary try/excepts (or none).
TRY_FREE_PACKAGES: tuple[str, ...] = ("hydromip", "mips", "metrics", "leaderboards", "benchmarks")


def _camels_protocol_payload() -> dict:
    path = resolve_mip_paths(DEFAULT_MIP_ID).protocol_file
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_zero_try_except_blocks_in_core_packages() -> None:
    """Enforce zero unallowlisted try/except (incl. ``except*``) blocks across core packages."""
    py_files = [
        p
        for pkg in TRY_FREE_PACKAGES
        for p in sorted((REPO_ROOT / pkg).rglob("*.py"))
        if "tests" not in p.relative_to(REPO_ROOT).parts
    ]
    assert py_files, "Expected Python modules under the core packages."
    try_types = _ast_try_types()
    for py_path in py_files:
        rel_str = py_path.relative_to(REPO_ROOT).as_posix()
        allowed = TRY_EXCEPT_FUNCTION_ALLOWLIST.get(rel_str, frozenset())
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel_str)
        violations = [
            node.lineno
            for node, scope in _iter_try_nodes_with_scope(tree, try_types)
            if scope not in allowed
        ]
        assert not violations, f"Forbidden try/except block found in {rel_str}:{violations}"


@pytest.mark.unit
def test_discover_mips_registers_camels_us_daily_and_validates_assets() -> None:
    """Verify ``discover_mips`` registers ``camels_us_daily`` with complete assets.

    Other registered MIPs may exist; adding one under ``mips/<mip_id>/`` is purely additive.
    """
    mips = discover_mips()
    assert len(mips) >= 1
    assert DEFAULT_MIP_ID in mips

    proto = get_mip(DEFAULT_MIP_ID)
    assert proto.mip_id == "camels_us_daily"
    assert proto.task_type == "streamflow_daily_deterministic"
    assert proto.metric_suite == "gauch_2023_deterministic"
    assert proto.num_entities == 531
    assert proto.num_eval_steps == 3652
    assert len(proto.leaderboard_ids) == 12
    validate_mip_assets(proto)

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
def test_protocol_leaderboards_reference_declared_regimes() -> None:
    """Every leaderboard references declared regime ids and is named `<prefix>__<forcing>`."""
    proto = get_mip(DEFAULT_MIP_ID)
    prefix = {sr.id: sr.leaderboard_prefix for sr in proto.spatial_regimes}
    for lb in proto.leaderboards:
        assert lb.spatial_regime in proto.spatial_regime_ids
        assert lb.forcing_regime in proto.forcing_regime_ids
        assert lb.id == f"{prefix[lb.spatial_regime]}__{lb.forcing_regime}"
    assert "paths" not in proto.raw, "custom per-MIP paths are not supported; drop the block"
    authors = [a["name"] for a in proto.raw["authors"]]
    assert "Gauch, Martin" in authors


@pytest.mark.unit
def test_protocol_matches_leaderboard_json_metadata() -> None:
    """`protocol.json` and the 12 leaderboard JSON files agree on table, section, and regimes."""
    proto = get_mip(DEFAULT_MIP_ID)
    prefix = {sr.id: sr.leaderboard_prefix for sr in proto.spatial_regimes}
    for lb in proto.leaderboards:
        payload = json.loads(
            (proto.paths.leaderboards_dir / f"{lb.id}.json").read_text(encoding="utf-8")
        )
        assert payload["table_number"] == f"Table {lb.table_number}"
        assert payload["section"] == lb.section
        assert payload["spatial_regime"] == prefix[lb.spatial_regime]
        assert payload["forcing_regime"] == lb.forcing_regime


@pytest.mark.unit
def test_validate_protocol_dict_rejects_invalid_protocols() -> None:
    """Dangling regime references, unregistered suites, bad ids, and custom paths raise."""
    good = _camels_protocol_payload()
    validate_protocol_dict(good, expected_mip_id=DEFAULT_MIP_ID)

    dangling = copy.deepcopy(good)
    dangling["leaderboards"][4]["spatial_regime"] = "pub"
    with pytest.raises(ValueError, match="undeclared spatial_regime"):
        validate_protocol_dict(dangling)

    unknown_suite = copy.deepcopy(good)
    unknown_suite["metric_suite"] = "not_a_registered_suite"
    with pytest.raises(ValueError, match="is not registered"):
        validate_protocol_dict(unknown_suite)

    wrong_metric = copy.deepcopy(good)
    wrong_metric["primary_metric"] = "CRPS"
    with pytest.raises(ValueError, match="is not part of metric suite"):
        validate_protocol_dict(wrong_metric)

    bad_id = copy.deepcopy(good)
    bad_id["leaderboards"][0]["id"] = "gauged__daymet_only"
    with pytest.raises(ValueError, match="must equal"):
        validate_protocol_dict(bad_id)

    custom_paths = copy.deepcopy(good)
    custom_paths["paths"] = {"splits": "folds"}
    with pytest.raises(ValueError, match="custom per-MIP paths"):
        validate_protocol_dict(custom_paths)

    duplicate_regime = copy.deepcopy(good)
    duplicate_regime["spatial_regimes"][1]["id"] = "gauged"
    with pytest.raises(ValueError, match="Duplicate spatial regime ids"):
        validate_protocol_dict(duplicate_regime)

    traversal = copy.deepcopy(good)
    traversal["mip_id"] = "../camels_us_daily"
    with pytest.raises(ValueError, match="Invalid mip_id"):
        validate_protocol_dict(traversal)


@pytest.mark.unit
def test_resolve_mip_subdir_never_aliases_code_packages(tmp_path: Path) -> None:
    """A missing per-MIP directory raises instead of resolving to the `leaderboards/` package."""
    lb_dir = resolve_mip_subdir(REPO_ROOT, "leaderboards", marker_filename="tables_3_to_14.json")
    assert lb_dir == resolve_mip_paths(DEFAULT_MIP_ID).leaderboards_dir
    with pytest.raises(FileNotFoundError):
        resolve_mip_subdir(REPO_ROOT, "leaderboards", mip_id="brand_new_mip")
    with pytest.raises(ValueError, match="Invalid mip_id"):
        resolve_mip_subdir(REPO_ROOT, "leaderboards", mip_id="..")
    with pytest.raises(ValueError, match="Unknown MIP subdirectory"):
        resolve_mip_subdir(REPO_ROOT, "update_leaderboard")
    with pytest.raises(FileNotFoundError, match="Required file"):
        resolve_mip_subdir(REPO_ROOT, "leaderboards", marker_filename="missing_marker.json")

    # An override data root without a `mips/` directory uses the flat layout.
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "splits_and_regimes.json").write_text("{}", encoding="utf-8")
    flat = resolve_mip_subdir(tmp_path, "splits", marker_filename="splits_and_regimes.json")
    assert flat == (tmp_path / "splits").resolve()


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
    assert [f.fold_index for f in pub] == list(range(12))

    pur = load_splits_for_regime("pur_18huc02")
    assert len(pur) == 18
    assert sum(len(f.test_entities) for f in pur) == 531

    with pytest.raises(KeyError, match="Unknown spatial regime"):
        load_splits_for_regime("pub")


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
    validate_cv_folds(
        good_folds,
        entities,
        entity_to_group=entity_to_group,
        require_train_test_union_equals_cohort=True,
        require_single_test_group=True,
    )

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

    # A fold that silently drops an entity from both train and test fails the union check
    dropped = [
        CvFold("f0", 0, ("b3",), ("b1", "b2")),
        CvFold("f1", 1, ("b1", "b2"), ("b3", "b4")),
    ]
    with pytest.raises(ValueError, match="do not cover the canonical cohort"):
        validate_cv_folds(dropped, entities, require_train_test_union_equals_cohort=True)

    # Leave-one-group-out requires exactly one held-out group per fold
    two_groups = [CvFold("f0", 0, ("b2",), ("b1", "b3", "b4"))]
    with pytest.raises(ValueError):
        validate_cv_folds(
            two_groups,
            entities,
            entity_to_group={"b1": "g1", "b2": "g2", "b3": "g3", "b4": "g3"},
            require_exact_test_partition=False,
            require_single_test_group=True,
        )

    # Duplicate fold indices fail
    with pytest.raises(ValueError, match="Duplicate fold_index"):
        validate_cv_folds(
            [CvFold("f0", 0, ("b3", "b4"), ("b1", "b2")), CvFold("f1", 0, ("b1", "b2"), ("b3", "b4"))],
            entities,
        )


@pytest.mark.unit
def test_metric_suite_registry_is_pluggable_for_future_mips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify new metric suites (e.g. probabilistic or flood) register cleanly without touching Gauch 2023."""
    gauch = get_metric_suite("gauch_2023_deterministic")
    assert gauch.primary_metric == "NSE"
    assert len(gauch.metric_names) == 14
    assert gauch.metric_names[3:6] == ("KGE", "logNSE", "logKGE")
    assert any(s["suite_id"] == "gauch_2023_deterministic" for s in list_metric_suites())

    # Isolate the global registry so a failing assertion can never leak the test suite.
    monkeypatch.setattr("hydromip.metrics.registry.METRIC_SUITES", dict(METRIC_SUITES))
    custom_id = "_test_probabilistic_crps"
    custom = MetricSuiteSpec(
        suite_id=custom_id,
        description="Synthetic probabilistic metric suite test",
        metric_names=("CRPS", "PIT_Uniformity"),
        primary_metric="CRPS",
        higher_is_better=False,
        compute_fn=lambda obs, sim: {"CRPS": float(obs.size - sim.size), "PIT_Uniformity": 1.0},
        citation="Test Suite",
    )
    register_metric_suite(custom)
    loaded = get_metric_suite(custom_id)
    assert loaded.primary_metric == "CRPS"
    assert loaded.higher_is_better is False
    with pytest.raises(ValueError, match="already registered"):
        register_metric_suite(custom)


@pytest.mark.unit
def test_metric_names_are_identical_across_the_registry_and_evaluator() -> None:
    """The 14 canonical Gauch et al. (2023) metric names agree in every implementation list."""
    from hydromip.metrics.deterministic_daily import get_available_metrics
    from metrics.evaluator import GAUCH_2023_METRICS

    suite = get_metric_suite("gauch_2023_deterministic")
    assert suite.metric_names == GAUCH_2023_METRICS == tuple(get_available_metrics())


@pytest.mark.unit
def test_model_catalog_matches_registered_models() -> None:
    """Verify ``models/catalog.json`` covers every model in ``MODEL_REGISTRY`` with real directories."""
    from models.run_model import MODEL_REGISTRY

    catalog_path = REPO_ROOT / "models" / "catalog.json"
    assert catalog_path.is_file()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    keys = {entry["key"] for entry in catalog["models"]}
    assert keys == set(MODEL_REGISTRY.keys())
    for key in keys:
        assert (REPO_ROOT / "models" / key / "runner.py").is_file()


@pytest.mark.unit
def test_validate_store_id_rejects_traversal_and_bad_characters() -> None:
    """Store / entry identifiers interpolated into Zarr URIs can never escape their prefix."""
    assert validate_store_id("naive_lag1_persistence_gauged") == "naive_lag1_persistence_gauged"
    assert validate_store_id("t3_02") == "t3_02"
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_store_id(blank)
    for bad in ("../observations/newman_eval_targets", "a/b", "a..b", ".hidden", "x" * 129, "a b"):
        with pytest.raises(ValueError, match="not a valid identifier"):
            validate_store_id(bad)


@pytest.mark.unit
def test_gcs_storage_options_resolution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials resolve explicitly: anon flag, env files (fail loudly), ADC, GCE, then anon."""
    for var in (
        "HYDROMIP_GCS_ANON",
        "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(zarr_io, "running_on_gce", lambda: False)
    assert get_gcs_storage_options() == {"token": "anon"}

    monkeypatch.setattr(zarr_io, "running_on_gce", lambda: True)
    assert get_gcs_storage_options() == {"token": "google_default"}

    adc = tmp_path / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}", encoding="utf-8")
    assert get_gcs_storage_options() == {"token": str(adc)}

    explicit = tmp_path / "service_account.json"
    explicit.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(explicit))
    assert get_gcs_storage_options() == {"token": str(explicit)}

    monkeypatch.setenv("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError, match="CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"):
        get_gcs_storage_options()

    monkeypatch.setenv("HYDROMIP_GCS_ANON", "1")
    assert get_gcs_storage_options() == {"token": "anon"}


@pytest.mark.integration
def test_adding_a_future_mip_is_purely_additive_and_auto_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify dropping a second MIP under ``mips/<new_id>/protocol.json`` is auto-discovered by core and API."""
    from fastapi.testclient import TestClient

    from web.scorer_api.app import app as web_app

    # A future MIP registers its metric suite first (here: a synthetic soil-moisture suite).
    monkeypatch.setitem(
        METRIC_SUITES,
        "soil_moisture_daily",
        MetricSuiteSpec(
            suite_id="soil_moisture_daily",
            description="Synthetic soil-moisture suite for the extensibility test",
            metric_names=("ubRMSE", "anomaly_r"),
            primary_metric="ubRMSE",
            higher_is_better=False,
            compute_fn=lambda obs, sim: {"ubRMSE": 0.0, "anomaly_r": 1.0},
            citation="Test Suite",
        ),
    )

    # Mirror camels_us_daily protocol plus add a synthetic soil_moisture MIP
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
    client = TestClient(web_app)
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
def test_portal_splits_and_figures_endpoints_work_with_multi_mip_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``/api/mips``, ``/api/splits/*``, and ``/api/figures/*`` serve CAMELS-US files."""
    from fastapi.testclient import TestClient

    from web.scorer_api.app import app as web_app

    monkeypatch.delenv("CAMELS_MIP_DATA_DIR", raising=False)
    monkeypatch.delenv("HYDROMIP_DATA_DIR", raising=False)
    client = TestClient(web_app)

    assert client.get("/api/mips").json()["n_mips"] == 1
    assert client.get("/api/splits/basins_531.txt").status_code == 200
    assert client.get("/api/mips/camels_us_daily/splits/pub_12fold_random_splits.json").status_code == 200
    assert client.get("/api/figures/figures_manifest.json").status_code == 200
    assert client.get("/api/mips/camels_us_daily/figures/figure_1_water_year_coverage.png").status_code == 200
