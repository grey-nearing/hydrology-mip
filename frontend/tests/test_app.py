# Copyright 2026 HydrologyMIP Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit, integration, and data-integrity tests for `frontend/backend/app.py`.

Tests all endpoints at full native CAMELS-US MIP resolution (531 basins x 3,652 days)
using real Zarr stores on `tmp_path` without mocking internal scoring or Zarr logic.
"""

from __future__ import annotations

import ast
import datetime as dt
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

import frontend.backend.app as backend_app
from frontend.backend.app import (
    CANONICAL_LEADERBOARD_IDS,
    NEWMAN_END_DATE,
    NEWMAN_N_BASINS,
    NEWMAN_N_DAYS,
    NEWMAN_START_DATE,
    REPO_ROOT,
    app,
)
from metrics.evaluator import calculate_model_leaderboard_metrics


def _load_canonical_splits_dict() -> dict:
    """Load the repository's canonical `splits_and_regimes.json` for test fixture setup."""
    splits_candidate = REPO_ROOT / "splits" / "splits_and_regimes.json"
    assert splits_candidate.exists(), f"Missing canonical splits at {splits_candidate}"
    with open(splits_candidate, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_native_eval_targets_zarr(store_path: Path, splits_dict: dict) -> dict[str, np.ndarray]:
    """Write a real (531, 3652) `newman_eval_targets.zarr` store (consolidated Zarr v2, like the bucket)."""
    basin_to_huc = splits_dict["basin_to_huc02"]
    basins = sorted(basin_to_huc.keys())
    assert len(basins) == NEWMAN_N_BASINS
    huc_02 = [str(basin_to_huc[b]).zfill(2) for b in basins]
    dates = pd.date_range(NEWMAN_START_DATE, NEWMAN_END_DATE, freq="D")
    assert len(dates) == NEWMAN_N_DAYS

    rng = np.random.default_rng(20261020)
    seasonal = 2.0 + 1.5 * np.sin(np.linspace(0.0, 20.0 * np.pi, NEWMAN_N_DAYS))[np.newaxis, :]
    basin_scale = np.linspace(0.5, 2.5, NEWMAN_N_BASINS)[:, np.newaxis]
    noise = rng.gamma(shape=2.0, scale=0.6, size=(NEWMAN_N_BASINS, NEWMAN_N_DAYS))
    qobs = (basin_scale * seasonal + noise).astype(np.float32)

    # Include realistic missing USGS observation days (NaN in qobs)
    qobs[0, 10:15] = np.nan
    qobs[100, 500:520] = np.nan

    ds = xr.Dataset(
        data_vars={
            "qobs": (("basin", "date"), qobs),
        },
        coords={
            "basin": ("basin", np.array(basins, dtype="<U8")),
            "date": ("date", dates.values.astype("datetime64[ns]")),
            "huc_02": ("basin", np.array(huc_02, dtype="<U2")),
        },
        attrs={
            "title": "CAMELS-US MIP Canonical Newman Evaluation Targets",
            "units": "mm/day",
        },
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(store_path, mode="w", consolidated=True, zarr_format=2)
    return {
        "basins": np.array(basins, dtype="<U8"),
        "dates": dates.values.astype("datetime64[ns]"),
        "huc_02": np.array(huc_02, dtype="<U2"),
        "qobs": qobs,
    }


def _zip_directory_bytes(store_dir: Path) -> bytes:
    """Zip an on-disk Zarr store directory (archive paths relative to the store's parent)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(store_dir.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(store_dir.parent)
                zf.write(file_path, arcname=str(rel))
    return buf.getvalue()


def _write_hydrograph_zarr(
    store_dir: Path,
    qsim: np.ndarray,
    basins: np.ndarray,
    dates: np.ndarray,
    qsim_members: np.ndarray | None = None,
    dtype: type = np.float32,
    attrs: dict | None = None,
) -> Path:
    """Write a consolidated Zarr v2 hydrograph store (`qsim` and optional `qsim_members`)."""
    if store_dir.exists():
        shutil.rmtree(store_dir)
    data_vars = {"qsim": (("basin", "date"), qsim.astype(dtype))}
    coords: dict = {"basin": ("basin", basins), "date": ("date", dates)}
    if qsim_members is not None:
        data_vars["qsim_members"] = (("member", "basin", "date"), qsim_members.astype(dtype))
        coords["member"] = ("member", np.arange(qsim_members.shape[0], dtype=np.int64))
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs or {})
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(store_dir, mode="w", consolidated=True, zarr_format=2)
    return store_dir


def _create_zipped_zarr_bytes(
    qsim: np.ndarray,
    basins: np.ndarray,
    dates: np.ndarray,
    tmp_dir: Path,
    var_name: str = "qsim",
    qsim_members: np.ndarray | None = None,
    dtype: type = np.float32,
    attrs: dict | None = None,
) -> bytes:
    """Create an in-memory zipped `.zarr` archive from a real Zarr dataset."""
    zarr_dir = tmp_dir / "upload_hydrograph.zarr"
    if zarr_dir.exists():
        shutil.rmtree(zarr_dir)

    data_vars = {var_name: (("basin", "date"), qsim.astype(dtype))}
    coords: dict = {"basin": ("basin", basins), "date": ("date", dates)}
    if qsim_members is not None:
        data_vars["qsim_members"] = (("member", "basin", "date"), qsim_members.astype(dtype))
        coords["member"] = ("member", np.arange(qsim_members.shape[0], dtype=np.int64))

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs or {})
    ds.to_zarr(zarr_dir, mode="w", consolidated=True, zarr_format=2)
    return _zip_directory_bytes(zarr_dir)


@pytest.fixture()
def mip_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Set up an isolated native-resolution (531 x 3652) data & Zarr environment."""
    backend_app._EVAL_TARGETS_CACHE.clear()

    data_dir = tmp_path / "camels_mip_data"
    lb_dir = data_dir / "leaderboards"
    splits_dir = data_dir / "splits"
    lb_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("tables_3_to_14.json", "table_2_registry.json", "contributions_index.json"):
        shutil.copy2(REPO_ROOT / "leaderboards" / fname, lb_dir / fname)
    for lb_id in CANONICAL_LEADERBOARD_IDS:
        shutil.copy2(REPO_ROOT / "leaderboards" / f"{lb_id}.json", lb_dir / f"{lb_id}.json")

    splits_dict = _load_canonical_splits_dict()
    with open(splits_dir / "splits_and_regimes.json", "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2)
    shutil.copy2(
        REPO_ROOT / "splits" / "basins_447_kratzert2019b.json",
        splits_dir / "basins_447_kratzert2019b.json",
    )

    targets_zarr_path = data_dir / "observations" / "newman_eval_targets.zarr"
    target_arrays = _build_native_eval_targets_zarr(targets_zarr_path, splits_dict)

    monkeypatch.setenv("CAMELS_MIP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CAMELS_MIP_GCS_ROOT", str(data_dir))
    monkeypatch.delenv("CAMELS_MIP_SUBMISSIONS_DIR", raising=False)

    client = TestClient(app)
    return {
        "client": client,
        "data_dir": data_dir,
        "targets": target_arrays,
        "tmp_path": tmp_path,
    }


@pytest.fixture()
def no_figure_regeneration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the ~50 s paper-figure regeneration that `upsert_leaderboard_entry` triggers on every submit.

    Figure generation is exercised by `leaderboards/tests` and by
    `test_submit_community_hydrograph_end_to_end`; the submission-persistence tests below only
    need the leaderboard/contributions JSON side effects.
    """
    import leaderboards.generate_figures as generate_figures

    monkeypatch.setattr(generate_figures, "generate_all_paper_figures", lambda **kwargs: {"skipped": True})


def test_zero_try_except_blocks_in_backend() -> None:
    """Verify `frontend/backend/app.py` contains zero `try`/`except` blocks."""
    app_py_path = Path(backend_app.__file__).resolve()
    source = app_py_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_py_path))
    try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert len(try_nodes) == 0, f"Found forbidden try/except blocks at lines: {[n.lineno for n in try_nodes]}"


def _expected_counts(lb_dir: Path) -> dict[str, int]:
    """Read the expected family / entry / contribution counts straight from the JSON files in `lb_dir`."""
    with open(lb_dir / "table_2_registry.json", "r", encoding="utf-8") as f:
        n_families = len(json.load(f)["families"])
    with open(lb_dir / "contributions_index.json", "r", encoding="utf-8") as f:
        n_contributions = len(json.load(f)["entries"])
    n_entries = 0
    for lb_id in CANONICAL_LEADERBOARD_IDS:
        with open(lb_dir / f"{lb_id}.json", "r", encoding="utf-8") as f:
            n_entries += len(json.load(f)["rows"])
    return {"n_families": n_families, "n_entries": n_entries, "n_contributions": n_contributions}


def test_health_endpoint(mip_env: dict) -> None:
    """Verify `/api/health` returns 200 OK with 12 leaderboards and the registry's model-family count."""
    client: TestClient = mip_env["client"]
    expected = _expected_counts(mip_env["data_dir"] / "leaderboards")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_leaderboards"] == 12
    assert body["n_families"] == expected["n_families"] >= 25
    assert body["total_entries"] == expected["n_entries"] >= 100
    assert body["total_contributions"] == expected["n_contributions"] >= 30


def test_protocol_endpoint(mip_env: dict) -> None:
    """Verify `/api/protocol` returns the 531-basin Newman protocol and PUB/PUR splits."""
    client: TestClient = mip_env["client"]
    resp = client.get("/api/protocol")
    assert resp.status_code == 200
    proto = resp.json()
    assert proto["num_basins"] == 531
    assert proto["n_basins"] == 531
    assert proto["n_leaderboards"] == 12
    assert proto["training_period"]["start_date"] == "1999-10-01"
    assert proto["training_period"]["end_date"] == "2008-09-30"
    assert proto["evaluation_period"]["start_date"] == "1989-10-01"
    assert proto["evaluation_period"]["end_date"] == "1999-09-30"
    assert proto["evaluation_period"]["num_days"] == 3652
    assert len(proto["basin_to_huc02"]) == 531
    assert len(proto["pub_12fold_test_basins"]) == 12
    assert len(proto["pur_18huc02_test_basins"]) == 18


def test_all_12_leaderboards_endpoints(mip_env: dict) -> None:
    """Verify `/api/leaderboards` and `/api/leaderboards/{leaderboard_id}` across all 12 leaderboards."""
    client: TestClient = mip_env["client"]
    resp = client.get("/api/leaderboards")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["leaderboards"]) == 12
    assert len(data["families"]) == _expected_counts(mip_env["data_dir"] / "leaderboards")["n_families"]

    returned_ids = [lb["id"] for lb in data["leaderboards"]]
    assert tuple(returned_ids) == CANONICAL_LEADERBOARD_IDS

    for lb_id in CANONICAL_LEADERBOARD_IDS:
        single_resp = client.get(f"/api/leaderboards/{lb_id}")
        assert single_resp.status_code == 200
        board = single_resp.json()
        assert board["id"] == lb_id
        assert board["spatial_regime"] in ("gauged", "pub", "pur")
        assert board["forcing_regime"] in (
            "single_camels",
            "multi_camels",
            "lagged_streamflow",
            "freeform",
        )
        assert board["n_entries"] == len(board["rows"])
        if board["rows"]:
            ranks = [r["rank"] for r in board["rows"]]
            assert ranks == list(range(1, len(board["rows"]) + 1))

    # Non-existent leaderboard returns 404
    not_found = client.get("/api/leaderboards/nonexistent__board")
    assert not_found.status_code == 404


def test_registry_endpoint(mip_env: dict) -> None:
    """Verify `/api/registry` returns every registry model family and the contribution records."""
    client: TestClient = mip_env["client"]
    expected = _expected_counts(mip_env["data_dir"] / "leaderboards")
    resp = client.get("/api/registry")
    assert resp.status_code == 200
    reg = resp.json()
    assert reg["total_families"] == expected["n_families"]
    assert len(reg["families"]) == expected["n_families"]
    family_ids = {f["family_id"] for f in reg["families"]}
    assert "kratzert_lstm" in family_ids
    assert "hoedt_mclstm" in family_ids
    assert reg["total_contributions"] == expected["n_contributions"]
    assert len(reg["contributions"]) == expected["n_contributions"]


def test_static_ui_serving(mip_env: dict) -> None:
    """Verify static UI assets are served at `/`, `/styles.css`, `/app.js`, and `/static/*`."""
    client: TestClient = mip_env["client"]

    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "HydrologyMIP" in r_index.text
    assert "hydrologymip.org" in r_index.text

    r_css = client.get("/styles.css")
    assert r_css.status_code == 200
    assert "--accent-primary" in r_css.text

    r_js = client.get("/app.js")
    assert r_js.status_code == 200
    assert "loadAllData" in r_js.text

    r_static_css = client.get("/static/styles.css")
    assert r_static_css.status_code == 200

    r_static_js = client.get("/static/app.js")
    assert r_static_js.status_code == 200


def test_score_valid_zarr_and_npz_hydrographs(mip_env: dict) -> None:
    """Verify scoring a native (531, 3652) `.zarr` upload and `.npz` upload succeeds with exact metrics."""
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]

    # Perfect prediction on all valid observation days (and 0.0 where qobs is NaN)
    qsim_perfect = np.where(np.isnan(qobs), 0.0, qobs).astype(np.float32)

    zarr_bytes = _create_zipped_zarr_bytes(
        qsim=qsim_perfect,
        basins=basins,
        dates=dates,
        tmp_dir=mip_env["tmp_path"],
    )
    resp_zarr = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("perfect_model.zarr.zip", zarr_bytes, "application/zip")},
    )
    assert resp_zarr.status_code == 200, resp_zarr.text
    res_zarr = resp_zarr.json()
    assert res_zarr["status"] == "ok"
    assert res_zarr["pred_nan_when_obs_valid"] == 0
    assert res_zarr["median_nse"] == pytest.approx(1.0, abs=1e-4)
    assert res_zarr["mean_nse"] == pytest.approx(1.0, abs=1e-4)
    assert res_zarr["median_kge"] == pytest.approx(1.0, abs=1e-4)
    assert res_zarr["median_abs_fhv"] == pytest.approx(0.0, abs=1e-4)
    assert res_zarr["nse_le_zero_count"] == 0
    assert res_zarr["prospective_rank"] == 1
    assert len(res_zarr["scoring"]["huc_02_breakdown"]) == 18
    assert len(res_zarr["scoring"]["per_basin"]) == 531

    # Verify all 12 Gauch et al. (2023) metrics are present in gauch_2023_medians and per_basin
    expected_gauch_metrics = {
        "NSE",
        "MSE",
        "RMSE",
        "KGE",
        "Alpha-NSE",
        "Pearson-r",
        "Beta-KGE",
        "Beta-NSE",
        "FHV",
        "FMS",
        "FLV",
        "Peak-Timing",
    }
    gauch_medians = res_zarr["scoring"]["gauch_2023_medians"]
    assert set(gauch_medians.keys()) == expected_gauch_metrics
    assert gauch_medians["NSE"] == pytest.approx(1.0, abs=1e-4)
    assert gauch_medians["MSE"] == pytest.approx(0.0, abs=1e-4)
    assert gauch_medians["RMSE"] == pytest.approx(0.0, abs=1e-4)
    assert gauch_medians["KGE"] == pytest.approx(1.0, abs=1e-4)
    assert gauch_medians["Alpha-NSE"] == pytest.approx(1.0, abs=1e-4)
    assert gauch_medians["Pearson-r"] == pytest.approx(1.0, abs=1e-4)
    assert gauch_medians["Beta-KGE"] == pytest.approx(1.0, abs=1e-4)
    assert gauch_medians["Beta-NSE"] == pytest.approx(0.0, abs=1e-4)
    assert gauch_medians["FHV"] == pytest.approx(0.0, abs=1e-4)
    assert gauch_medians["FMS"] == pytest.approx(0.0, abs=1e-4)
    assert gauch_medians["FLV"] == pytest.approx(0.0, abs=1e-4)
    assert gauch_medians["Peak-Timing"] == pytest.approx(0.0, abs=1e-4)
    assert expected_gauch_metrics.issubset(set(res_zarr["scoring"]["per_basin"][0].keys()))

    # Also verify that NaN in qsim ONLY where qobs is already NaN is permitted (pred_nan_when_obs_valid == 0)
    qsim_matching_obs_nan = qobs.copy()
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, qsim=qsim_matching_obs_nan, basin=basins)
    resp_npz = client.post(
        "/api/score",
        data={"leaderboard_id": "pub__single_camels"},
        files={"file": ("model_531x3652.npz", npz_buf.getvalue(), "application/octet-stream")},
    )
    assert resp_npz.status_code == 200, resp_npz.text
    res_npz = resp_npz.json()
    assert res_npz["pred_nan_when_obs_valid"] == 0
    assert res_npz["median_nse"] == pytest.approx(1.0, abs=1e-4)


def test_reject_pred_nan_when_obs_valid_and_invalid_shapes(mip_env: dict) -> None:
    """Verify explicit HTTP 400 rejection when `pred_nan_when_obs_valid > 0` or shape != `(531, 3652)`."""
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]

    # 1. Inject a single NaN on a day where observed discharge is valid (basin 0, day 0)
    assert not np.isnan(qobs[0, 0]) and qobs[0, 0] >= 0.0
    qsim_with_nan = np.where(np.isnan(qobs), 0.0, qobs).astype(np.float32)
    qsim_with_nan[0, 0] = np.nan

    zarr_nan_bytes = _create_zipped_zarr_bytes(
        qsim=qsim_with_nan,
        basins=basins,
        dates=dates,
        tmp_dir=mip_env["tmp_path"],
    )
    resp_nan_zarr = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("nan_on_valid_obs.zarr", zarr_nan_bytes, "application/zip")},
    )
    assert resp_nan_zarr.status_code == 400
    assert "pred_nan_when_obs_valid=1" in resp_nan_zarr.json()["detail"]

    # Also verify rejection via .npz when pred_nan_when_obs_valid > 0
    npz_nan_buf = io.BytesIO()
    np.savez_compressed(npz_nan_buf, qsim=qsim_with_nan)
    resp_nan_npz = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("nan_on_valid_obs.npz", npz_nan_buf.getvalue(), "application/octet-stream")},
    )
    assert resp_nan_npz.status_code == 400
    assert "pred_nan_when_obs_valid=1" in resp_nan_npz.json()["detail"]

    # 2. Wrong shape (530, 3652) must be rejected with HTTP 400
    wrong_shape_buf = io.BytesIO()
    np.savez_compressed(
        wrong_shape_buf,
        qsim=np.ones((530, NEWMAN_N_DAYS), dtype=np.float32),
    )
    resp_wrong_shape = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("wrong_basins.npz", wrong_shape_buf.getvalue(), "application/octet-stream")},
    )
    assert resp_wrong_shape.status_code == 400
    assert "531, 3652" in resp_wrong_shape.json()["detail"]

    # 3. Missing 'qsim' key must be rejected with HTTP 400
    missing_key_buf = io.BytesIO()
    np.savez_compressed(
        missing_key_buf,
        wrong_var=np.ones((NEWMAN_N_BASINS, NEWMAN_N_DAYS), dtype=np.float32),
    )
    resp_missing_key = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("missing_key.npz", missing_key_buf.getvalue(), "application/octet-stream")},
    )
    assert resp_missing_key.status_code == 400
    assert "qsim" in resp_missing_key.json()["detail"]

    # 4. Omitting even a single basin (setting a basin with valid observations to all-NaN) must be rejected with HTTP 400
    qsim_missing_basin = np.where(np.isnan(qobs), 0.0, qobs).astype(np.float32)
    qsim_missing_basin[5, :] = np.nan
    expected_nan_days = int(np.sum((~np.isnan(qobs[5, :])) & (qobs[5, :] >= 0.0)))
    assert expected_nan_days == NEWMAN_N_DAYS

    missing_basin_zarr = _create_zipped_zarr_bytes(
        qsim=qsim_missing_basin,
        basins=basins,
        dates=dates,
        tmp_dir=mip_env["tmp_path"],
    )
    resp_missing_basin = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("missing_basin.zarr", missing_basin_zarr, "application/zip")},
    )
    assert resp_missing_basin.status_code == 400
    assert f"pred_nan_when_obs_valid={expected_nan_days}" in resp_missing_basin.json()["detail"]
    assert "no missing basins" in resp_missing_basin.json()["detail"]


def test_default_repo_root_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default loading from repo `leaderboards/` and `splits/` when `CAMELS_MIP_DATA_DIR` is unset."""
    monkeypatch.delenv("CAMELS_MIP_DATA_DIR", raising=False)
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["n_leaderboards"] == 12
    assert r_health.json()["n_families"] == _expected_counts(REPO_ROOT / "leaderboards")["n_families"]

    r_proto = client.get("/api/protocol")
    assert r_proto.status_code == 200
    assert r_proto.json()["num_basins"] == 531
    assert len(r_proto.json()["pub_12fold_test_basins"]) == 12
    assert len(r_proto.json()["pur_18huc02_test_basins"]) == 18


def test_submit_community_hydrograph_end_to_end(mip_env: dict) -> None:
    """Verify `/api/submit` scores a (531, 3652) Zarr upload and adds it to the target leaderboard."""
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]

    qsim = np.where(np.isnan(qobs), 0.0, qobs * 0.95).astype(np.float32)
    zarr_bytes = _create_zipped_zarr_bytes(
        qsim=qsim,
        basins=basins,
        dates=dates,
        tmp_dir=mip_env["tmp_path"],
    )

    resp = client.post(
        "/api/submit",
        data={
            "leaderboard_id": "pur__freeform",
            "model_name": "Community Test Zarr Model",
            "forcing": "ERA5-Land",
            "github_url": "https://github.com/example/hydro-model",
            "zenodo_doi": "10.5281/zenodo.9999999",
            "citation": "Test et al. (2026)",
        },
        files={"file": ("community_model.zarr", zarr_bytes, "application/zip")},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "ok"
    entry_id = payload["entry"]["entry_id"]

    # Confirm the new entry appears on Table 14 (pur__freeform) both via API and in leaderboards/pur__freeform.json
    board_resp = client.get("/api/leaderboards/pur__freeform")
    assert board_resp.status_code == 200
    board = board_resp.json()
    assert any(r["entry_id"] == entry_id for r in board["rows"])

    per_lb_path = mip_env["data_dir"] / "leaderboards" / "pur__freeform.json"
    with open(per_lb_path, "r", encoding="utf-8") as f:
        per_lb_json = json.load(f)
    assert any(r["entry_id"] == entry_id for r in per_lb_json["rows"])


def _build_parity_hydrographs(qobs: np.ndarray, n_members: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build float64 `(qsim, qsim_members)` with negative low-flow predictions and `qsim == mean(members)`.

    Predictions are NaN only where `qobs` is NaN (allowed: `pred_nan_when_obs_valid == 0`).
    """
    rng = np.random.default_rng(7)
    base = np.where(np.isnan(qobs), np.nan, 0.9 * qobs.astype(np.float64) - 0.6)
    perturb = rng.normal(0.0, 0.15, size=(n_members,) + base.shape)
    perturb -= perturb.mean(axis=0, keepdims=True)
    members = base[np.newaxis, :, :] + perturb
    qsim = members.mean(axis=0)
    assert np.nanmin(qsim) < 0.0, "parity hydrograph must contain negative predictions"
    assert np.array_equal(np.isnan(qsim), np.isnan(qobs))
    return qsim, members


def test_score_parity_with_metrics_evaluator(mip_env: dict) -> None:
    """`/api/score` must reproduce `metrics.evaluator.calculate_model_leaderboard_metrics` to 1e-9.

    Same consolidated Zarr v2 store (negative predictions, `qsim == mean(qsim_members)`), scored
    once by the cloud evaluator from `bucket_uri=<tmp>` and once via `/api/score` (zipped store):
    float64, unclipped, identical evaluable-basin rule and aggregation.
    """
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    data_dir: Path = mip_env["data_dir"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]

    qsim, members = _build_parity_hydrographs(qobs, n_members=3)
    leaderboard_id = "gauged__multi_camels"
    model_id = "parity_model"
    store_dir = data_dir / "hydrographs" / leaderboard_id / f"{model_id}.zarr"
    _write_hydrograph_zarr(
        store_dir,
        qsim=qsim,
        basins=basins,
        dates=dates,
        qsim_members=members,
        dtype=np.float64,
        attrs={"is_ensemble": True, "num_members": 3},
    )

    evaluator_res = calculate_model_leaderboard_metrics(model_id, leaderboard_id, bucket_uri=str(data_dir))

    resp = client.post(
        "/api/score",
        data={"leaderboard_id": leaderboard_id},
        files={"file": (f"{model_id}.zarr.zip", _zip_directory_bytes(store_dir), "application/zip")},
    )
    assert resp.status_code == 200, resp.text
    api_res = resp.json()

    assert api_res["pred_nan_when_obs_valid"] == 0
    assert api_res["ensemble_source"] == "mean_of_qsim_members"
    assert api_res["ensemble_source"] == evaluator_res["ensemble_source"]
    assert api_res["ensemble_reduction"] == "mean"
    assert api_res["single_seed_member_index"] is None
    assert api_res["single_seed_member_index"] == evaluator_res["single_seed_member_index"]
    assert api_res["is_ensemble"] is True
    assert api_res["is_ensemble"] == evaluator_res["is_ensemble"]
    assert api_res["n_members"] == 3
    assert api_res["has_qsim_members"] is True
    assert api_res["num_evaluated_basins"] == evaluator_res["num_evaluated_basins"] == NEWMAN_N_BASINS
    assert api_res["nse_le_zero_count"] == evaluator_res["nse_le_zero_count"]
    for key in ("median_nse", "mean_nse", "median_kge", "median_abs_fhv"):
        assert api_res[key] is not None, key
        assert abs(float(api_res[key]) - float(evaluator_res[key])) <= 1e-9, (
            f"{key}: api={api_res[key]!r} evaluator={evaluator_res[key]!r}"
        )

    # Unclipped scoring: negative predictions must change the score relative to a clipped copy.
    clipped_resp = client.post(
        "/api/score",
        data={"leaderboard_id": leaderboard_id},
        files={
            "file": (
                "clipped.zarr.zip",
                _create_zipped_zarr_bytes(
                    qsim=np.where(np.isnan(qsim), np.nan, np.maximum(qsim, 0.0)),
                    basins=basins,
                    dates=dates,
                    tmp_dir=mip_env["tmp_path"],
                    dtype=np.float64,
                ),
                "application/zip",
            )
        },
    )
    assert clipped_resp.status_code == 200, clipped_resp.text
    unclipped_pb = np.array([np.nan if v is None else v for v in api_res["scoring"]["per_basin_nse"]], dtype=float)
    clipped_pb = np.array(
        [np.nan if v is None else v for v in clipped_resp.json()["scoring"]["per_basin_nse"]], dtype=float
    )
    both = np.isfinite(unclipped_pb) & np.isfinite(clipped_pb)
    # Clipping negative predictions toward the non-negative observations can only raise NSE, so an
    # unclipped scorer must report strictly lower NSE on the basins with negative predictions.
    assert np.all(clipped_pb[both] >= unclipped_pb[both] - 1e-9)
    assert np.any(clipped_pb[both] > unclipped_pb[both] + 1e-6)

    # Per-basin NSE vectors agree (both rounded to 6 digits; NaN <-> None for non-evaluable basins).
    api_per_basin = api_res["scoring"]["per_basin_nse"]
    ev_per_basin = evaluator_res["per_basin"]["NSE"]
    assert len(api_per_basin) == len(ev_per_basin) == NEWMAN_N_BASINS
    for a_val, e_val in zip(api_per_basin, ev_per_basin):
        if a_val is None:
            assert np.isnan(e_val)
        else:
            assert abs(float(a_val) - float(e_val)) <= 1e-9

    # Per-member single-seed skill (keys added to the evaluator; compared when present).
    assert api_res["single_median_NSE_per_member"] is not None
    assert len(api_res["single_median_NSE_per_member"]) == 3
    assert api_res["single_median_NSE"] == pytest.approx(
        float(np.mean(api_res["single_median_NSE_per_member"])), abs=1e-4
    )
    assert api_res["single_median_NSE_std"] == pytest.approx(
        float(np.std(api_res["single_median_NSE_per_member"])), abs=1e-4
    )
    if evaluator_res.get("single_seed_median_nse_per_member") is not None:
        for a_val, e_val in zip(
            api_res["single_median_NSE_per_member"], evaluator_res["single_seed_median_nse_per_member"]
        ):
            assert abs(float(a_val) - float(e_val)) <= 1e-9
        assert abs(api_res["single_median_NSE"] - evaluator_res["single_seed_median_nse_mean"]) <= 1e-9
        assert abs(api_res["single_median_NSE_std"] - evaluator_res["single_seed_median_nse_std"]) <= 1e-9
    if "num_members" in evaluator_res:
        assert api_res["n_members"] == evaluator_res["num_members"]

    # The stored qsim is scored directly, never recomputed: scoring the ensemble hydrograph
    # alone (no members) yields exactly the same headline metrics.
    qsim_only_resp = client.post(
        "/api/score",
        data={"leaderboard_id": leaderboard_id},
        files={
            "file": (
                "qsim_only.zarr.zip",
                _create_zipped_zarr_bytes(
                    qsim=qsim, basins=basins, dates=dates, tmp_dir=mip_env["tmp_path"], dtype=np.float64
                ),
                "application/zip",
            )
        },
    )
    assert qsim_only_resp.status_code == 200, qsim_only_resp.text
    qsim_only = qsim_only_resp.json()
    assert qsim_only["is_ensemble"] is False
    assert qsim_only["n_members"] == 1
    assert qsim_only["ensemble_source"] == "stored_qsim_without_members"
    assert qsim_only["single_median_NSE_std"] == 0.0  # population std of a single member, like the evaluator
    assert qsim_only["single_median_NSE"] == qsim_only["median_nse"]
    for key in ("median_nse", "mean_nse", "median_kge", "median_abs_fhv", "nse_le_zero_count"):
        assert qsim_only[key] == api_res[key], key


def test_reject_zarr_when_qsim_is_not_mean_of_members(mip_env: dict) -> None:
    """A Zarr upload whose stored `qsim` is not `mean(qsim_members)` (e.g. clipped) is rejected with 400."""
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]

    qsim, members = _build_parity_hydrographs(qobs, n_members=2)
    qsim_clipped = np.where(np.isnan(qsim), np.nan, np.maximum(qsim, 0.0))
    assert not np.allclose(qsim_clipped[~np.isnan(qsim)], qsim[~np.isnan(qsim)], rtol=1e-4, atol=1e-4)

    resp = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={
            "file": (
                "clipped_ensemble.zarr",
                _create_zipped_zarr_bytes(
                    qsim=qsim_clipped,
                    basins=basins,
                    dates=dates,
                    tmp_dir=mip_env["tmp_path"],
                    qsim_members=members,
                    dtype=np.float64,
                ),
                "application/zip",
            )
        },
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "not the mean of 'qsim_members'" in detail
    assert "rtol=0.0001" in detail

    # Wrong member dimension layout is also rejected (must be ("member", "basin", "date")).
    zarr_dir = mip_env["tmp_path"] / "bad_dims.zarr"
    ds = xr.Dataset(
        data_vars={
            "qsim": (("basin", "date"), qsim),
            "qsim_members": (("basin", "date", "member"), np.moveaxis(members, 0, -1)),
        },
        coords={"basin": ("basin", basins), "date": ("date", dates)},
    )
    ds.to_zarr(zarr_dir, mode="w", consolidated=True, zarr_format=2)
    resp_dims = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("bad_dims.zarr", _zip_directory_bytes(zarr_dir), "application/zip")},
    )
    assert resp_dims.status_code == 400
    assert "('member', 'basin', 'date')" in resp_dims.json()["detail"]

    # A members-only 3-D .npz upload is scored as mean(members) and says so.
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, qsim=members.astype(np.float32), basin=basins)
    resp_npz = client.post(
        "/api/score",
        data={"leaderboard_id": "gauged__single_camels"},
        files={"file": ("members_only.npz", npz_buf.getvalue(), "application/octet-stream")},
    )
    assert resp_npz.status_code == 200, resp_npz.text
    body = resp_npz.json()
    assert body["ensemble_source"] == "mean_of_members_no_stored_qsim"
    assert body["has_qsim_members"] is True
    assert body["is_ensemble"] is True
    assert body["n_members"] == 2
    assert len(body["single_median_NSE_per_member"]) == 2


def test_entry_basins_cdf_is_empirical_or_unavailable(mip_env: dict) -> None:
    """`/api/entry/{id}/basins` never fabricates a CDF: `unavailable` without per-basin data, empirical with it."""
    client: TestClient = mip_env["client"]
    lb_dir: Path = mip_env["data_dir"] / "leaderboards"

    # Clear per_basin_nse on reference rows in the isolated test fixture to verify the None fallback first.
    for fname, ref_eid in (("gauged__single_camels.json", "t3_02"), ("gauged__multi_camels.json", "t4_05")):
        fpath = lb_dir / fname
        fdata = json.loads(fpath.read_text(encoding="utf-8"))
        for r in fdata["rows"]:
            if r["entry_id"] == ref_eid:
                r.pop("per_basin_nse", None)
        fpath.write_text(json.dumps(fdata), encoding="utf-8")

    # 1. Literature-only row (summary metrics only): source "unavailable", null CDF and null references.
    resp_lit = client.get("/api/entry/t3_55/basins")
    assert resp_lit.status_code == 200, resp_lit.text
    lit = resp_lit.json()
    assert lit["entry"]["entry_id"] == "t3_55"
    assert lit["entry"]["has_per_basin_nse"] is False
    assert lit["source"] == "unavailable"
    assert lit["nse_cdf_101"] is None
    assert lit["n_basins_with_nse"] is None
    assert lit["ref_3f_lstm_cdf_101"] is None
    assert lit["ref_1f_daymet_cdf_101"] is None
    assert lit["ref_3f_lstm_entry_id"] == "t4_05"
    assert lit["ref_1f_daymet_entry_id"] == "t3_02"
    assert isinstance(lit["huc_02_breakdown"], list)

    # 2. Unknown entry -> 404.
    assert client.get("/api/entry/does_not_exist/basins").status_code == 404

    # 3. Attach real per-basin NSE values to a row (as the fixed update_leaderboard.py does)
    #    and to the 3-forcing reference row; the CDF must be the empirical np.quantile curve.
    rng = np.random.default_rng(11)
    per_basin = rng.beta(8.0, 2.0, size=NEWMAN_N_BASINS).astype(np.float64).tolist()
    per_basin[3] = None
    per_basin[77] = None
    ref_per_basin = rng.beta(9.0, 2.0, size=NEWMAN_N_BASINS).astype(np.float64).tolist()

    lb_path = lb_dir / "gauged__single_camels.json"
    with open(lb_path, "r", encoding="utf-8") as f:
        lb_json = json.load(f)
    target_row = next(r for r in lb_json["rows"] if r["entry_id"] == "t3_55")
    target_row["per_basin_nse"] = per_basin
    with open(lb_path, "w", encoding="utf-8") as f:
        json.dump(lb_json, f)

    ref_path = lb_dir / "gauged__multi_camels.json"
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_json = json.load(f)
    ref_row = next(r for r in ref_json["rows"] if r["entry_id"] == "t4_05")
    ref_row["per_basin_nse"] = ref_per_basin
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref_json, f)

    resp_emp = client.get("/api/entry/t3_55/basins")
    assert resp_emp.status_code == 200, resp_emp.text
    emp = resp_emp.json()
    assert emp["source"] == "per_basin_metrics"
    assert emp["per_basin_nse_origin"] == "leaderboard_row.per_basin_nse"
    assert emp["n_basins_with_nse"] == NEWMAN_N_BASINS - 2
    assert emp["entry"]["has_per_basin_nse"] is True
    assert "per_basin_nse" not in emp["entry"]
    finite_vals = np.array([v for v in per_basin if v is not None], dtype=np.float64)
    expected_cdf = np.quantile(finite_vals, np.linspace(0.0, 1.0, 101))
    assert len(emp["nse_cdf_101"]) == 101
    assert np.allclose(np.array(emp["nse_cdf_101"]), np.round(expected_cdf, 4), atol=1e-9)
    assert emp["nse_cdf_101"] == sorted(emp["nse_cdf_101"])
    assert emp["nse_cdf_101"][0] == pytest.approx(float(finite_vals.min()), abs=1e-4)
    assert emp["nse_cdf_101"][-1] == pytest.approx(float(finite_vals.max()), abs=1e-4)

    expected_ref = np.quantile(np.array(ref_per_basin), np.linspace(0.0, 1.0, 101))
    assert np.allclose(np.array(emp["ref_3f_lstm_cdf_101"]), np.round(expected_ref, 4), atol=1e-9)
    assert emp["ref_1f_daymet_cdf_101"] is None

    # The heavy per-basin vector is not shipped with the leaderboard payloads.
    board = client.get("/api/leaderboards/gauged__single_camels").json()
    t3_55 = next(r for r in board["rows"] if r["entry_id"] == "t3_55")
    assert t3_55["has_per_basin_nse"] is True
    assert "per_basin_nse" not in t3_55


def test_submit_persists_zarr_and_single_seed_vs_ensemble_columns(mip_env: dict, no_figure_regeneration: None) -> None:
    """`/api/submit` writes a readable consolidated Zarr v2 store at the recorded path with correct NSE columns."""
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    data_dir: Path = mip_env["data_dir"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]

    # --- Single-seed submission (2-D qsim only) ---
    qsim_single = np.where(np.isnan(qobs), np.nan, 0.97 * qobs.astype(np.float64) - 0.3)
    resp_single = client.post(
        "/api/submit",
        data={
            "leaderboard_id": "pub__lagged_streamflow",
            "model_name": "Single Seed Community Model",
            "forcing": "Daymet",
            "citation": "Solo et al. (2026)",
        },
        files={
            "file": (
                "single_seed.zarr",
                _create_zipped_zarr_bytes(
                    qsim=qsim_single, basins=basins, dates=dates, tmp_dir=mip_env["tmp_path"], dtype=np.float64
                ),
                "application/zip",
            )
        },
    )
    assert resp_single.status_code == 200, resp_single.text
    single_payload = resp_single.json()
    single_entry = single_payload["entry"]
    single_scoring = single_payload["scoring"]

    assert single_entry["is_ensemble"] is False
    assert single_entry["num_members"] == 1
    assert single_entry["seeds"] == "1"
    assert single_entry["ens_med_nse"] is None
    assert single_entry["ens_med_nse_display"] == "---"
    assert single_entry["single_med_nse"] == single_scoring["median_nse"]
    assert single_entry["single_med_nse_display"] == f"{single_scoring['median_nse']:.3f}"
    assert single_entry["single_med_nse_std"] == 0.0
    assert single_entry["primary_ranking_nse"] == single_scoring["median_nse"]
    assert single_entry["ensemble_source"] == "stored_qsim_without_members"
    assert single_entry["has_qsim_members"] is False
    assert single_entry["single_seed_member_index"] is None
    assert len(single_entry["per_basin_nse"]) == NEWMAN_N_BASINS

    single_store = Path(single_entry["hydrograph_path"])
    assert single_store == data_dir / "submissions" / "pub__lagged_streamflow" / f"{single_entry['entry_id']}.zarr"
    assert single_store.is_dir()
    assert (single_store / ".zmetadata").is_file()  # consolidated Zarr v2
    assert (single_store / ".zgroup").is_file()
    ds_single = xr.open_zarr(single_store, consolidated=True)
    assert tuple(ds_single["qsim"].dims) == ("basin", "date")
    assert ds_single["qsim"].shape == (NEWMAN_N_BASINS, NEWMAN_N_DAYS)
    assert "qsim_members" not in ds_single.data_vars
    assert [str(b) for b in ds_single["basin"].values] == [str(b) for b in basins]
    assert [str(h) for h in ds_single["huc_02"].values] == [str(h) for h in targets["huc_02"]]
    stored_single = np.asarray(ds_single["qsim"].values, dtype=np.float64)
    assert np.array_equal(np.isnan(stored_single), np.isnan(qsim_single))
    assert np.allclose(stored_single[~np.isnan(qsim_single)], qsim_single[~np.isnan(qsim_single)], rtol=0, atol=0)
    assert ds_single.attrs["entry_id"] == single_entry["entry_id"]
    assert bool(ds_single.attrs["is_ensemble"]) is False

    # The persisted store is exactly what was scored: re-scoring it with the cloud evaluator
    # (bucket layout: <bucket>/hydrographs/<lb>/<id>.zarr + <bucket>/observations/...) reproduces
    # the recorded headline metrics.
    bucket2 = mip_env["tmp_path"] / "rescoring_bucket"
    shutil.copytree(data_dir / "observations", bucket2 / "observations")
    shutil.copytree(
        single_store, bucket2 / "hydrographs" / "pub__lagged_streamflow" / f"{single_entry['entry_id']}.zarr"
    )
    rescored = calculate_model_leaderboard_metrics(
        single_entry["entry_id"], "pub__lagged_streamflow", bucket_uri=str(bucket2)
    )
    assert abs(rescored["median_nse"] - single_entry["primary_ranking_nse"]) <= 1e-9
    assert abs(rescored["mean_nse"] - single_entry["mean_nse"]) <= 1e-9
    assert abs(rescored["median_kge"] - single_entry["med_kge"]) <= 1e-9
    assert abs(rescored["median_abs_fhv"] - single_entry["med_fhv_abs"]) <= 1e-9
    assert rescored["nse_le_zero_count"] == single_entry["nse_le_0"]

    # --- Ensemble submission (Zarr with qsim == mean(qsim_members)) ---
    qsim_ens, members = _build_parity_hydrographs(qobs, n_members=3)
    resp_ens = client.post(
        "/api/submit",
        data={
            "leaderboard_id": "pub__lagged_streamflow",
            "model_name": "Ensemble Community Model",
            "forcing": "Daymet",
            "citation": "Crowd et al. (2026)",
        },
        files={
            "file": (
                "ensemble.zarr",
                _create_zipped_zarr_bytes(
                    qsim=qsim_ens,
                    basins=basins,
                    dates=dates,
                    tmp_dir=mip_env["tmp_path"],
                    qsim_members=members,
                    dtype=np.float64,
                    attrs={"is_ensemble": True, "num_members": 3},
                ),
                "application/zip",
            )
        },
    )
    assert resp_ens.status_code == 200, resp_ens.text
    ens_payload = resp_ens.json()
    ens_entry = ens_payload["entry"]
    ens_scoring = ens_payload["scoring"]

    per_member = ens_scoring["single_median_NSE_per_member"]
    assert len(per_member) == 3
    assert ens_entry["is_ensemble"] is True
    assert ens_entry["num_members"] == 3
    assert ens_entry["seeds"] == "3"
    assert ens_entry["ens_med_nse"] == ens_scoring["median_nse"]
    assert ens_entry["ens_med_nse_display"] == f"{ens_scoring['median_nse']:.3f}"
    assert ens_entry["primary_ranking_nse"] == ens_scoring["median_nse"]
    assert ens_entry["single_med_nse"] == pytest.approx(float(np.mean(per_member)), abs=1e-4)
    assert ens_entry["single_med_nse_std"] == pytest.approx(float(np.std(per_member)), abs=1e-4)
    assert ens_entry["single_med_nse_display"] == (
        f"{ens_entry['single_med_nse']:.3f} ± {ens_entry['single_med_nse_std']:.3f}"
    )
    assert ens_entry["single_med_nse_per_member"] == per_member

    ens_store = Path(ens_entry["hydrograph_path"])
    assert ens_store.is_dir() and (ens_store / ".zmetadata").is_file()
    ds_ens = xr.open_zarr(ens_store, consolidated=True)
    assert tuple(ds_ens["qsim_members"].dims) == ("member", "basin", "date")
    assert ds_ens["qsim_members"].shape == (3, NEWMAN_N_BASINS, NEWMAN_N_DAYS)
    stored_members = np.asarray(ds_ens["qsim_members"].values, dtype=np.float64)
    stored_qsim = np.asarray(ds_ens["qsim"].values, dtype=np.float64)
    valid = ~np.isnan(qsim_ens)
    assert np.allclose(stored_members[:, valid], members[:, valid], rtol=0, atol=0)
    assert np.allclose(stored_qsim[valid], qsim_ens[valid], rtol=0, atol=0)
    assert bool(ds_ens.attrs["is_ensemble"]) is True
    assert int(ds_ens.attrs["num_members"]) == 3

    # Both rows are on the board, ensemble ranked by its median NSE, and the contributions index
    # records the same real paths (never a gs:// path that was not written).
    board = client.get("/api/leaderboards/pub__lagged_streamflow").json()
    board_ids = [r["entry_id"] for r in board["rows"]]
    assert single_entry["entry_id"] in board_ids and ens_entry["entry_id"] in board_ids
    for row in board["rows"]:
        assert row["primary_ranking_nse"] is None or row["primary_ranking_nse"] > -100
    with open(data_dir / "leaderboards" / "contributions_index.json", "r", encoding="utf-8") as f:
        contrib = json.load(f)["entries"]
    for e in (single_entry, ens_entry):
        rec = contrib[e["entry_id"]]
        assert rec["hydrograph_path"] == e["hydrograph_path"]
        assert Path(rec["hydrograph_path"]).is_dir()
        assert not rec["hydrograph_path"].startswith("gs://")
        assert rec["ensemble_source"] == e["ensemble_source"]
    assert contrib[ens_entry["entry_id"]]["ensemble_source"] == "mean_of_qsim_members"
    assert contrib[single_entry["entry_id"]]["ensemble_source"] == "stored_qsim_without_members"
    assert contrib[ens_entry["entry_id"]]["is_ensemble"] is True
    assert contrib[ens_entry["entry_id"]]["num_members"] == 3
    assert contrib[single_entry["entry_id"]]["is_ensemble"] is False

    # Re-scoring the persisted ensemble store with the evaluator reproduces the row's bookkeeping.
    shutil.copytree(ens_store, bucket2 / "hydrographs" / "pub__lagged_streamflow" / f"{ens_entry['entry_id']}.zarr")
    rescored_ens = calculate_model_leaderboard_metrics(
        ens_entry["entry_id"], "pub__lagged_streamflow", bucket_uri=str(bucket2)
    )
    assert rescored_ens["ensemble_source"] == "mean_of_qsim_members"
    assert rescored_ens["is_ensemble"] is True
    assert rescored_ens["num_members"] == 3
    assert abs(rescored_ens["median_nse"] - ens_entry["primary_ranking_nse"]) <= 1e-9
    assert abs(rescored_ens["single_seed_median_nse_mean"] - ens_entry["single_med_nse"]) <= 1e-9
    assert abs(rescored_ens["single_seed_median_nse_std"] - ens_entry["single_med_nse_std"]) <= 1e-9

    # The new rows serve a real empirical CDF through the basins endpoint.
    basins_resp = client.get(f"/api/entry/{ens_entry['entry_id']}/basins")
    assert basins_resp.status_code == 200
    assert basins_resp.json()["source"] == "per_basin_metrics"
    assert len(basins_resp.json()["nse_cdf_101"]) == 101
    assert len(basins_resp.json()["huc_02_breakdown"]) == 18


def test_submit_honours_submissions_dir_env_and_refuses_when_unwritable(
    mip_env: dict, monkeypatch: pytest.MonkeyPatch, no_figure_regeneration: None
) -> None:
    """`CAMELS_MIP_SUBMISSIONS_DIR` overrides the store root; an unwritable root yields HTTP 503 and no row."""
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]
    qsim = np.where(np.isnan(qobs), np.nan, qobs.astype(np.float64))
    zarr_bytes = _create_zipped_zarr_bytes(
        qsim=qsim, basins=basins, dates=dates, tmp_dir=mip_env["tmp_path"], dtype=np.float64
    )

    custom_root = mip_env["tmp_path"] / "custom_submissions_root"
    monkeypatch.setenv("CAMELS_MIP_SUBMISSIONS_DIR", str(custom_root))
    resp_ok = client.post(
        "/api/submit",
        data={"leaderboard_id": "pur__multi_camels", "model_name": "Env Root Model", "forcing": "NLDAS"},
        files={"file": ("env_root.zarr", zarr_bytes, "application/zip")},
    )
    assert resp_ok.status_code == 200, resp_ok.text
    recorded = Path(resp_ok.json()["entry"]["hydrograph_path"])
    assert recorded.parent.parent == custom_root.resolve()
    assert (recorded / ".zmetadata").is_file()

    if os.geteuid() == 0:
        pytest.skip("permission-based refusal cannot be exercised as root")
    read_only_parent = mip_env["tmp_path"] / "read_only_parent"
    read_only_parent.mkdir()
    read_only_parent.chmod(0o500)
    monkeypatch.setenv("CAMELS_MIP_SUBMISSIONS_DIR", str(read_only_parent / "submissions"))
    n_rows_before = len(client.get("/api/leaderboards/pur__multi_camels").json()["rows"])
    resp_refused = client.post(
        "/api/submit",
        data={"leaderboard_id": "pur__multi_camels", "model_name": "Refused Model", "forcing": "NLDAS"},
        files={"file": ("refused.zarr", zarr_bytes, "application/zip")},
    )
    read_only_parent.chmod(0o700)
    assert resp_refused.status_code == 503, resp_refused.text
    assert "CAMELS_MIP_SUBMISSIONS_DIR" in resp_refused.json()["detail"]
    assert len(client.get("/api/leaderboards/pur__multi_camels").json()["rows"]) == n_rows_before


def test_updated_at_is_max_mtime_of_leaderboard_files(mip_env: dict) -> None:
    """`updated_at` is the UTC mtime of the newest of the 12 leaderboard JSON files, not a hard-coded date."""
    client: TestClient = mip_env["client"]
    lb_dir: Path = mip_env["data_dir"] / "leaderboards"

    newest = lb_dir / "pur__freeform.json"
    future = dt.datetime(2031, 3, 4, 5, 6, 7, tzinfo=dt.timezone.utc).timestamp()
    os.utime(newest, (future, future))

    overview = client.get("/api/overview").json()
    assert overview["updated_at"] == "2031-03-04T05:06:07Z"
    assert client.get("/api/leaderboards").json()["updated_at"] == "2031-03-04T05:06:07Z"

    # Sanity: rows with no numeric score sort last and are reported as None, never a sentinel.
    boards = client.get("/api/leaderboards").json()["leaderboards"]
    for lb in boards:
        scores = [r["primary_ranking_nse"] for r in lb["rows"]]
        numeric = [s for s in scores if s is not None]
        assert scores[: len(numeric)] == numeric
        assert numeric == sorted(numeric, reverse=True)


def test_member_semantics_match_evaluator(mip_env: dict) -> None:
    """`is_ensemble` / `ensemble_reduction` attributes are honoured exactly like `metrics.evaluator`.

    - `is_ensemble=False` + N members: `qsim` must equal exactly one member (rtol=atol=1e-5) →
      `ensemble_source="single_member"` with `single_seed_member_index`; otherwise HTTP 400.
    - `ensemble_reduction="median"`: `qsim` is checked against `np.median(members, 0)`.
    - `ensemble_reduction="per_basin_oracle"` with `is_ensemble=False` is passed through;
      with `is_ensemble=True` it is HTTP 400; unsupported reductions are HTTP 400.
    """
    client: TestClient = mip_env["client"]
    targets = mip_env["targets"]
    data_dir: Path = mip_env["data_dir"]
    qobs: np.ndarray = targets["qobs"]
    basins: np.ndarray = targets["basins"]
    dates: np.ndarray = targets["dates"]
    tmp: Path = mip_env["tmp_path"]
    _, members = _build_parity_hydrographs(qobs, n_members=3)
    valid = ~np.isnan(qobs)

    def post(name: str, store_dir: Path):
        return client.post(
            "/api/score",
            data={"leaderboard_id": "gauged__single_camels"},
            files={"file": (f"{name}.zarr.zip", _zip_directory_bytes(store_dir), "application/zip")},
        )

    # 1. Single model shipping its sibling seeds: qsim == member 1 exactly.
    single_dir = data_dir / "hydrographs" / "gauged__single_camels" / "single_member_model.zarr"
    _write_hydrograph_zarr(
        single_dir, qsim=members[1], basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": False, "num_members": 3},
    )
    resp_single = post("single_member_model", single_dir)
    assert resp_single.status_code == 200, resp_single.text
    body = resp_single.json()
    assert body["ensemble_source"] == "single_member"
    assert body["single_seed_member_index"] == 1
    assert body["is_ensemble"] is False
    assert body["n_members"] == 3
    assert len(body["single_median_NSE_per_member"]) == 3
    ev_single = calculate_model_leaderboard_metrics(
        "single_member_model", "gauged__single_camels", bucket_uri=str(data_dir)
    )
    assert ev_single["ensemble_source"] == "single_member"
    assert ev_single["single_seed_member_index"] == 1
    assert ev_single["is_ensemble"] is False
    assert abs(ev_single["median_nse"] - body["median_nse"]) <= 1e-9
    assert abs(ev_single["single_seed_median_nse_mean"] - body["single_median_NSE"]) <= 1e-9
    assert abs(ev_single["single_seed_median_nse_std"] - body["single_median_NSE_std"]) <= 1e-9
    # Leaderboard column semantics for a single model with siblings: no ensemble score, "mean ± std".
    cols = backend_app._format_nse_columns(body["scoring"])
    assert cols["ens_med_nse"] is None and cols["ens_med_nse_display"] == "---"
    assert cols["single_med_nse_display"] == f"{body['single_median_NSE']:.3f} ± {body['single_median_NSE_std']:.3f}"
    assert cols["primary_ranking_nse"] == body["median_nse"]

    # 2. is_ensemble=False but qsim matches no member → 400.
    qsim_none = np.where(valid, 0.5 * (members[0] + members[1]), np.nan)
    none_dir = tmp / "matches_none.zarr"
    _write_hydrograph_zarr(
        none_dir, qsim=qsim_none, basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": False},
    )
    resp_none = post("matches_none", none_dir)
    assert resp_none.status_code == 400
    assert "matches 0 of the 3" in resp_none.json()["detail"]

    # 3. Median reduction: accepted when qsim == median(members), rejected when qsim == mean(members).
    qsim_median = np.where(valid, np.median(members, axis=0), np.nan)
    median_dir = data_dir / "hydrographs" / "gauged__single_camels" / "median_model.zarr"
    _write_hydrograph_zarr(
        median_dir, qsim=qsim_median, basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": True, "ensemble_reduction": "median"},
    )
    resp_median = post("median_model", median_dir)
    assert resp_median.status_code == 200, resp_median.text
    assert resp_median.json()["ensemble_source"] == "median_of_qsim_members"
    assert resp_median.json()["ensemble_reduction"] == "median"
    ev_median = calculate_model_leaderboard_metrics("median_model", "gauged__single_camels", bucket_uri=str(data_dir))
    assert ev_median["ensemble_source"] == "median_of_qsim_members"
    assert abs(ev_median["median_nse"] - resp_median.json()["median_nse"]) <= 1e-9

    qsim_mean = np.where(valid, np.mean(members, axis=0), np.nan)
    wrong_red_dir = tmp / "mean_declared_median.zarr"
    _write_hydrograph_zarr(
        wrong_red_dir, qsim=qsim_mean, basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": True, "ensemble_reduction": "median"},
    )
    resp_wrong_red = post("mean_declared_median", wrong_red_dir)
    assert resp_wrong_red.status_code == 400
    assert "not the median of 'qsim_members'" in resp_wrong_red.json()["detail"]

    # 4. per_basin_oracle: pass-through with is_ensemble=False, rejected with is_ensemble=True.
    oracle_qsim = np.where(valid, np.max(members, axis=0), np.nan)
    oracle_dir = tmp / "oracle.zarr"
    _write_hydrograph_zarr(
        oracle_dir, qsim=oracle_qsim, basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": False, "ensemble_reduction": "per_basin_oracle"},
    )
    resp_oracle = post("oracle", oracle_dir)
    assert resp_oracle.status_code == 200, resp_oracle.text
    assert resp_oracle.json()["ensemble_source"] == "per_basin_oracle"
    assert resp_oracle.json()["is_ensemble"] is False
    assert resp_oracle.json()["single_seed_member_index"] is None
    assert len(resp_oracle.json()["single_median_NSE_per_member"]) == 3

    oracle_ens_dir = tmp / "oracle_as_ensemble.zarr"
    _write_hydrograph_zarr(
        oracle_ens_dir, qsim=oracle_qsim, basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": True, "ensemble_reduction": "per_basin_oracle"},
    )
    resp_oracle_ens = post("oracle_as_ensemble", oracle_ens_dir)
    assert resp_oracle_ens.status_code == 400
    assert "per_basin_oracle" in resp_oracle_ens.json()["detail"]

    # 5. Unsupported reduction / non-boolean is_ensemble → 400 before any scoring.
    bad_red_dir = tmp / "bad_reduction.zarr"
    _write_hydrograph_zarr(
        bad_red_dir, qsim=qsim_mean, basins=basins, dates=dates, qsim_members=members, dtype=np.float64,
        attrs={"is_ensemble": True, "ensemble_reduction": "weighted"},
    )
    resp_bad_red = post("bad_reduction", bad_red_dir)
    assert resp_bad_red.status_code == 400
    assert "Unsupported 'ensemble_reduction'" in resp_bad_red.json()["detail"]

    bad_flag_dir = tmp / "bad_flag.zarr"
    _write_hydrograph_zarr(
        bad_flag_dir, qsim=qsim_mean, basins=basins, dates=dates, dtype=np.float64, attrs={"is_ensemble": "maybe"}
    )
    resp_bad_flag = post("bad_flag", bad_flag_dir)
    assert resp_bad_flag.status_code == 400
    assert "'is_ensemble'" in resp_bad_flag.json()["detail"]

    # 6. is_ensemble=True declared without members: ensemble score kept, single-seed skill None.
    declared_dir = tmp / "declared_ensemble_no_members.zarr"
    _write_hydrograph_zarr(
        declared_dir, qsim=qsim_mean, basins=basins, dates=dates, dtype=np.float64, attrs={"is_ensemble": True}
    )
    resp_declared = post("declared_ensemble_no_members", declared_dir)
    assert resp_declared.status_code == 200, resp_declared.text
    declared = resp_declared.json()
    assert declared["is_ensemble"] is True
    assert declared["ensemble_source"] == "stored_qsim_without_members"
    assert declared["single_median_NSE"] is None
    assert declared["single_median_NSE_std"] is None
    assert declared["single_median_NSE_per_member"] is None
    cols_declared = backend_app._format_nse_columns(declared["scoring"])
    assert cols_declared["ens_med_nse"] == declared["median_nse"]
    assert cols_declared["single_med_nse"] is None and cols_declared["single_med_nse_display"] == "---"
