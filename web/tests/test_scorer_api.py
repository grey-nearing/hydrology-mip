# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for staged HydrologyMIP ingestion, LLM schema fallback, and Scorer API."""

from __future__ import annotations

import json
import socket
import stat
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, ClassVar, Self

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from fastapi.testclient import TestClient

from hydromip.io import ingest, llm_schema_adapter
from hydromip.metrics.deterministic_daily import calculate_all_metrics
from hydromip.paths import REPO_ROOT
from metrics.evaluator import GAUCH_2023_METRICS
from web.scorer_api import app as scorer_app


@pytest.fixture(scope="session")
def hermetic_qobs_npz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    splits_dir = REPO_ROOT / "mips" / "camels_us_daily" / "splits"
    sr_doc = json.loads((splits_dir / "splits_and_regimes.json").read_text(encoding="utf-8"))
    b_map = sr_doc["basin_to_huc02"]
    basins = sorted(str(b).zfill(8) for b in b_map.keys())
    huc_02 = [str(b_map[b]).zfill(2) for b in basins]
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("1989-10-01", "1999-09-30", freq="D")]

    rng = np.random.default_rng(20261120)
    t_grid = np.linspace(0.0, 20.0 * np.pi, len(dates))[np.newaxis, :]
    seasonal = 2.0 + 1.4 * np.sin(t_grid)
    scale = np.linspace(0.6, 2.2, len(basins))[:, np.newaxis]
    qobs = (scale * seasonal + rng.gamma(shape=2.0, scale=0.5, size=(len(basins), len(dates)))).astype(np.float32)
    qobs[0, 10:15] = np.nan

    out = tmp_path_factory.mktemp("qobs_ref") / "qobs_531.npz"
    np.savez(
        out,
        basins=np.asarray(basins, dtype="<U8"),
        dates=np.asarray(dates, dtype="<U10"),
        huc_02=np.asarray(huc_02, dtype="<U2"),
        qobs=qobs,
    )
    return out


@pytest.fixture(scope="session")
def canonical_qobs_bundle(
    hermetic_qobs_npz: Path,
) -> tuple[list[str], list[str], list[str], np.ndarray]:
    with np.load(hermetic_qobs_npz, allow_pickle=False) as d:
        basins = [str(b).zfill(8) for b in d["basins"].tolist()]
        dates = [str(x) for x in d["dates"].tolist()]
        huc_02 = [str(h).zfill(2) for h in d["huc_02"].tolist()]
        qobs = np.asarray(d["qobs"], dtype=np.float32)
    return basins, dates, huc_02, qobs


@pytest.fixture()
def isolated_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hermetic_qobs_npz: Path,
) -> TestClient:
    preview_dir = tmp_path / "preview_store"
    targets_cache = tmp_path / "targets_cache"
    monkeypatch.setenv("HYDROMIP_QOBS_PATH", str(hermetic_qobs_npz))
    monkeypatch.setenv("HYDROMIP_PREVIEW_DIR", str(preview_dir))
    monkeypatch.setenv("HYDROMIP_TARGETS_CACHE_DIR", str(targets_cache))
    monkeypatch.delenv("VERTEX_AI_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_AI_LOCATION", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HYDROMIP_MAX_LLM_CALLS_PER_DAY", raising=False)
    monkeypatch.delenv("HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR", raising=False)
    scorer_app._CANONICAL_REF = None
    scorer_app.RATE_LIMITER.reset()
    scorer_app.LLM_BUDGET_LIMITER.reset()
    return TestClient(scorer_app.app)


def _synthetic_qsim(qobs: np.ndarray, scale: float = 0.92, offset: float = 0.05) -> np.ndarray:
    sim = qobs.astype(np.float32) * np.float32(scale) + np.float32(offset)
    return np.nan_to_num(sim, nan=0.1).astype(np.float32)


def test_transposed_20yr_netcdf_7digit_ids_and_metric_parity(
    tmp_path: Path,
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, _, qobs = canonical_qobs_bundle
    # Build 20-year daily superset (1985-01-01 to 2004-12-31) transposed as (time, basin)
    full_dates = pd.date_range("1985-01-01", "2004-12-31", freq="D")
    n_days_20y = len(full_dates)
    rng = np.random.default_rng(7)
    arr_20y = rng.uniform(0.1, 5.0, size=(n_days_20y, len(basins))).astype(np.float32)

    qsim_expected = _synthetic_qsim(qobs)
    newman_start_idx = int(np.where(full_dates == pd.Timestamp(dates[0]))[0][0])
    newman_end_idx = newman_start_idx + len(dates)
    # Place transposed (date, basin) predictions inside the 20-year slice with shuffled basin order
    perm = rng.permutation(len(basins))
    rev_basins_7digit = [str(int(basins[i])) for i in perm]  # strips leading zeros -> 7-digit IDs
    arr_20y[newman_start_idx:newman_end_idx, :] = qsim_expected[perm, :].T

    nc_path = tmp_path / "transposed_20y.nc"
    ds = xr.Dataset(
        data_vars={"streamflow": (("time", "gauge_id"), arr_20y)},
        coords={"time": full_dates, "gauge_id": np.asarray(rev_basins_7digit)},
    )
    ds.to_netcdf(nc_path)

    meta = ingest.extract_file_metadata(nc_path)
    recipe, issues = ingest.resolve_deterministic_recipe(meta)
    assert recipe is not None
    assert not issues
    assert recipe.format == "netcdf"
    assert recipe.var == "streamflow"
    assert recipe.basin_dim == "gauge_id"
    assert recipe.time_dim == "time"

    out = ingest.ingest_hydrograph(nc_path, recipe, basins, dates, strict=True)
    assert out["qsim"].shape == (531, 3652)
    assert out["qsim"].dtype == np.float32
    np.testing.assert_allclose(out["qsim"], qsim_expected, rtol=1e-5, atol=1e-5)

    sanity, sample_basins = ingest.run_physical_sanity_checks(
        out["qsim"], qobs, basins, dates
    )
    assert sanity["checks_passed"] is True
    assert sanity["n_basins_matched"] == 531
    assert sanity["nan_count"] == 0
    assert sanity["date_start"] == "1989-10-01"
    assert sanity["date_end"] == "1999-09-30"
    assert len(sample_basins) == 3
    assert len(sample_basins[0]["dates"]) == 365

    # Exact parity check on single-basin metrics against hydromip.metrics.deterministic_daily
    dates_ns = pd.to_datetime(dates).values.astype("datetime64[ns]")
    b_idx = 0
    obs_da = xr.DataArray(qobs[b_idx].astype(np.float64), coords={"date": dates_ns}, dims=("date",))
    sim_da = xr.DataArray(out["qsim"][b_idx].astype(np.float64), coords={"date": dates_ns}, dims=("date",))
    direct_metrics = calculate_all_metrics(obs_da, sim_da, resolution="1D", datetime_coord="date")
    assert set(direct_metrics.keys()) == set(GAUCH_2023_METRICS)
    assert sample_basins[0]["nse"] is None or isinstance(sample_basins[0]["nse"], float)


def test_multiseed_zarr_zip_both_ensemble_and_single_seed_via_api(
    tmp_path: Path,
    isolated_api: TestClient,
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, huc_02, qobs = canonical_qobs_bundle
    base_sim = _synthetic_qsim(qobs)
    members = np.stack(
        [base_sim * 0.96, base_sim * 1.00, base_sim * 1.04],
        axis=0,
    ).astype(np.float32)
    ens_mean = np.mean(members.astype(np.float64), axis=0).astype(np.float32)

    zarr_dir = tmp_path / "multi_seed.zarr"
    ingest.write_canonical_zarr_v2(
        ens_mean,
        basins,
        zarr_dir,
        metadata={"model_id": "test_ens", "model_name": "Test Ens LSTM", "leaderboard_id": "gauged__single_camels"},
        qsim_members=members,
        huc_02=huc_02,
        dates=dates,
    )
    zip_path = tmp_path / "multi_seed.zarr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(zarr_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(zarr_dir).as_posix())

    with open(zip_path, "rb") as f:
        inspect_resp = isolated_api.post(
            "/api/preview/inspect",
            files={"file": ("multi_seed.zarr.zip", f, "application/zip")},
            data={
                "leaderboard_id": "gauged__single_camels",
                "spatial_regime": "gauged",
                "forcing": "Daymet",
                "model_name": "Test Ens LSTM",
            },
        )
    assert inspect_resp.status_code == 200, inspect_resp.text
    inspect_data = inspect_resp.json()
    assert inspect_data["format"] == "zarr.zip"
    assert inspect_data["sanity"]["checks_passed"] is True
    token = inspect_data["preview_token"]

    score_resp = isolated_api.post(
        "/api/preview/score",
        json={
            "preview_token": token,
            "recipe": {
                "var": inspect_data["schema"]["var"],
                "basin_dim": inspect_data["schema"]["basin_dim"],
                "time_dim": inspect_data["schema"]["time_dim"],
                "member_dim": inspect_data["schema"]["member_dim"],
            },
            "metadata": {
                "leaderboard_id": "gauged__single_camels",
                "spatial_regime": "gauged",
                "forcing": "Daymet",
                "model_name": "Test Ens LSTM",
                "code_url": "https://github.com/example/model",
                "archive_url": "https://doi.org/10.5281/zenodo.1234567",
            },
        },
    )
    assert score_resp.status_code == 200, score_resp.text
    scored = score_resp.json()
    assert scored["status"] == "scored"
    assert scored["local_preview_only"] is True

    entry = scored["entry"]
    assert entry["is_ensemble"] is True
    assert entry["num_members"] == 3
    assert len(entry["per_basin_nse"]) == 531
    assert set(entry["gauch_2023_metrics"].keys()) == set(GAUCH_2023_METRICS)
    assert len(entry["regional_huc02"]) == 18
    assert entry["p_value_vs_sota_nse"] is None or 0.0 <= entry["p_value_vs_sota_nse"] <= 1.0
    assert entry["hydrograph_samples"] is not None
    first_bid = next(iter(entry["hydrograph_samples"]))
    assert len(entry["hydrograph_samples"][first_bid]["qsim"]) == 3652

    single_entry = scored["ensemble_single_entry"]
    assert single_entry is not None
    assert single_entry["is_ensemble"] is False
    assert single_entry["num_members"] == 3
    assert len(single_entry["single_med_nse_per_member"]) == 3

    # Verify ephemeral preview directory was immediately purged after /api/preview/score
    preview_root = scorer_app.PREVIEW_STORE.root_dir
    assert list(preview_root.iterdir()) == []


def test_wide_csv_long_csv_and_parquet_ingestion(
    tmp_path: Path,
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, _, qobs = canonical_qobs_bundle
    qsim = _synthetic_qsim(qobs)

    # 1. Wide CSV (rows=dates, cols=basin IDs with mixed usgs_ prefix and 7-digit unpadded IDs)
    wide_cols = {
        (f"usgs_{int(b)}" if idx % 2 == 0 else b): qsim[idx, :]
        for idx, b in enumerate(basins)
    }
    df_wide = pd.DataFrame({"date": dates, **wide_cols})
    wide_csv = tmp_path / "wide_preds.csv"
    df_wide.to_csv(wide_csv, index=False)

    meta_w = ingest.extract_file_metadata(wide_csv)
    recipe_w, issues_w = ingest.resolve_deterministic_recipe(meta_w)
    assert recipe_w is not None and not issues_w
    res_w = ingest.ingest_hydrograph(wide_csv, recipe_w, basins, dates, strict=True)
    np.testing.assert_allclose(res_w["qsim"], qsim, rtol=1e-4, atol=1e-4)

    # 2. Long/tidy CSV & Parquet at full 531 x 3652 native resolution
    b_repeat = np.repeat(np.asarray(basins), len(dates))
    d_tile = np.tile(np.asarray(dates), len(basins))
    df_long = pd.DataFrame(
        {"basin_id": b_repeat, "date": d_tile, "qsim": qsim.ravel()}
    )

    pq_path = tmp_path / "long_preds.parquet"
    df_long.to_parquet(pq_path, index=False)
    meta_pq = ingest.extract_file_metadata(pq_path)
    recipe_pq, issues_pq = ingest.resolve_deterministic_recipe(meta_pq)
    assert recipe_pq is not None and not issues_pq
    res_pq = ingest.ingest_hydrograph(pq_path, recipe_pq, basins, dates, strict=True)
    np.testing.assert_allclose(res_pq["qsim"], qsim, rtol=1e-5, atol=1e-5)

    long_csv = tmp_path / "long_preds.csv"
    df_long.to_csv(long_csv, index=False)
    meta_l = ingest.extract_file_metadata(long_csv)
    recipe_l, issues_l = ingest.resolve_deterministic_recipe(meta_l)
    assert recipe_l is not None and not issues_l
    res_l = ingest.ingest_hydrograph(long_csv, recipe_l, basins, dates, strict=True)
    np.testing.assert_allclose(res_l["qsim"], qsim, rtol=1e-4, atol=1e-4)

    # 3. NPZ with explicit basin & date keys (and rejected when basin keys missing unless positional)
    npz_path = tmp_path / "explicit_preds.npz"
    np.savez_compressed(
        npz_path,
        qsim=qsim.T,  # (date, basin) transposed
        basins=np.asarray([str(int(b)) for b in basins]),
        dates=np.asarray(dates),
    )
    meta_npz = ingest.extract_file_metadata(npz_path)
    recipe_npz, issues_npz = ingest.resolve_deterministic_recipe(meta_npz)
    assert recipe_npz is not None and not issues_npz
    res_npz = ingest.ingest_hydrograph(npz_path, recipe_npz, basins, dates, strict=True)
    np.testing.assert_allclose(res_npz["qsim"], qsim, rtol=1e-5, atol=1e-5)

    unlabeled_npz = tmp_path / "unlabeled.npz"
    np.savez_compressed(unlabeled_npz, qsim=qsim)
    meta_un = ingest.extract_file_metadata(unlabeled_npz)
    rec_un, issues_un = ingest.resolve_deterministic_recipe(meta_un)
    assert rec_un is None and issues_un


def test_rate_limit_cors_and_yaml_templates(isolated_api: TestClient) -> None:
    # Allowed CORS origin vs blocked untrusted origin
    cors_ok = isolated_api.get("/health", headers={"Origin": "https://hydrologymip.org"})
    assert cors_ok.headers.get("access-control-allow-origin") == "https://hydrologymip.org"
    cors_bad = isolated_api.get("/health", headers={"Origin": "https://untrusted.example.com"})
    assert "access-control-allow-origin" not in cors_bad.headers

    # 15 requests allowed, 16th returns 429
    for idx in range(15):
        r = isolated_api.post("/api/preview/score", json={"preview_token": f"pv_{idx:012d}"})
        assert r.status_code in (400, 404)
    r_16 = isolated_api.post("/api/preview/score", json={"preview_token": "pv_000000000015"})
    assert r_16.status_code == 429

    # Validate GitHub templates parse cleanly with PyYAML
    gh_dir = REPO_ROOT / ".github"
    yml_files = list(gh_dir.rglob("*.yml"))
    names = {p.name for p in yml_files}
    for req in ("model_submission.yml", "score_submission.yml", "deploy_website.yml", "issue_to_pr.yml"):
        assert req in names
    for yp in yml_files:
        doc = yaml.safe_load(yp.read_text(encoding="utf-8"))
        assert isinstance(doc, dict)


def test_unit_mismatch_cfs_warning(
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, _, qobs = canonical_qobs_bundle
    cfs_sim = _synthetic_qsim(qobs, scale=35.3147, offset=0.2)
    sanity, _ = ingest.run_physical_sanity_checks(cfs_sim, qobs, basins, dates)
    assert sanity["checks_passed"] is False
    assert sanity["unit_warning"] is not None
    assert sanity["median_sim_obs_ratio"] > 5.0


def test_pickle_rejection_and_zip_slip_blocking(tmp_path: Path, isolated_api: TestClient) -> None:
    pkl_path = tmp_path / "model.pkl"
    pkl_path.write_bytes(b"not-a-pickle")
    with pytest.raises(ValueError, match="Pickle files are rejected"):
        ingest.detect_file_format(pkl_path)

    resp = isolated_api.post(
        "/api/preview/inspect",
        files={"file": ("model.pkl", b"dummy", "application/octet-stream")},
        data={"leaderboard_id": "gauged__single_camels", "model_name": "Bad"},
    )
    assert resp.status_code == 400
    assert "Pickle files are rejected" in resp.json()["error"]

    # ZIP path traversal
    slip_zip = tmp_path / "slip.zarr.zip"
    with zipfile.ZipFile(slip_zip, "w") as zf:
        zf.writestr("../../etc/passwd", "x")
    with pytest.raises(ValueError, match="traversal"):
        ingest.safe_extract_zip(slip_zip, tmp_path / "extract1")

    # ZIP symlink rejection
    sym_zip = tmp_path / "symlink.zarr.zip"
    with zipfile.ZipFile(sym_zip, "w") as zf:
        zi = zipfile.ZipInfo("evil_link")
        zi.create_system = 3  # Unix
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zi, "/etc/passwd")
    with pytest.raises(ValueError, match="Symlink"):
        ingest.safe_extract_zip(sym_zip, tmp_path / "extract2")


def test_ssrf_blocking_hosts_private_ips_and_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    for blocked_url in (
        "http://zenodo.org/records/123",
        "https://169.254.169.254/latest/meta-data/",
        "https://127.0.0.1/preds.nc",
        "https://evil.untrusted.example/preds.nc",
        "file:///etc/passwd",
        "https://user:pass@zenodo.org/records/123",
    ):
        with pytest.raises(ValueError):
            scorer_app.validate_https_target_url(blocked_url)

    # Mock DNS resolution of trusted host returning private RFC1918 IP
    def _fake_private_dns(*_args: Any, **_kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_private_dns)
    with pytest.raises(ValueError, match="non-global"):
        scorer_app.validate_https_target_url("https://zenodo.org/records/12345")

    # Mock DNS returning public IP, then redirect targeting metadata IP -> must fail
    def _fake_public_dns(*_args: Any, **_kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("188.184.95.95", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_public_dns)

    class _FakeRedirectResponse:
        status = 302
        headers: ClassVar[dict[str, str]] = {"Location": "http://169.254.169.254/latest/meta-data/"}

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    class _FakeOpener:
        def open(self, _req: Any, timeout: int = 60) -> _FakeRedirectResponse:
            return _FakeRedirectResponse()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _FakeOpener())
    with pytest.raises(ValueError):
        scorer_app.validate_and_fetch_archive_url("https://zenodo.org/records/12345/files/q.nc")


def test_gemini_tier2_mock_and_stuck_report_and_full_http_flow(
    tmp_path: Path,
    isolated_api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, _, qobs = canonical_qobs_bundle
    qsim = _synthetic_qsim(qobs)

    # Ambiguous multi-variable NetCDF so Tier 1 declines and falls back to Tier 2 Gemini Flash
    weird_nc = tmp_path / "weird_schema.nc"
    ds = xr.Dataset(
        data_vars={
            "custom_out_v9": (("gauge_axis", "day_axis"), qsim),
            "aux_state_v9": (("gauge_axis", "day_axis"), qsim * np.float32(0.5)),
        },
        coords={"gauge_axis": np.asarray(basins), "day_axis": pd.to_datetime(dates)},
    )
    ds.to_netcdf(weird_nc)

    meta = ingest.extract_file_metadata(weird_nc)
    recipe_t1, reasons = ingest.resolve_deterministic_recipe(meta)
    assert recipe_t1 is None and reasons

    # 1. Missing GEMINI_API_KEY -> stuck_report with hydromip package CLI snippet
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res_nokey = llm_schema_adapter.infer_recipe_from_metadata(meta, reasons)
    assert res_nokey["status"] == "stuck"
    assert res_nokey["recipe"] is None
    assert "hydromip package" in res_nokey["stuck_report"]["cli_snippet"]

    # 2. Failed request -> stuck_report
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")

    def _fail_urlopen(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.URLError("simulated outage")

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)
    res_fail = llm_schema_adapter.infer_recipe_from_metadata(meta, reasons)
    assert res_fail["status"] == "stuck"
    assert "simulated outage" in res_fail["stuck_report"]["reason"]

    # 3. LLM sets cannot_map=True -> stuck_report
    class _MockResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status = 200
            self._body = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    def _cannot_map_urlopen(*_a: Any, **_k: Any) -> _MockResp:
        inner = {"cannot_map": True, "missing_or_ambiguous": ["Unknown custom axis"], "notes": "Unclear."}
        return _MockResp({"candidates": [{"content": {"parts": [{"text": json.dumps(inner)}]}}]})

    monkeypatch.setattr(urllib.request, "urlopen", _cannot_map_urlopen)
    res_stuck = llm_schema_adapter.infer_recipe_from_metadata(meta, reasons)
    assert res_stuck["status"] == "stuck"
    assert "Unknown custom axis" in res_stuck["stuck_report"]["missing_or_ambiguous"]

    # 4. Valid Tier 2 schema JSON -> full FastAPI /health -> /inspect -> /score flow
    captured_prompts: list[str] = []

    def _valid_gemini_urlopen(req: urllib.request.Request, **_k: Any) -> _MockResp:
        body = json.loads(req.data.decode("utf-8"))
        captured_prompts.append(body["contents"][0]["parts"][0]["text"])
        recipe_json = {
            "cannot_map": False,
            "missing_or_ambiguous": [],
            "format": "netcdf",
            "var": "custom_out_v9",
            "basin_dim": "gauge_axis",
            "time_dim": "day_axis",
            "member_dim": None,
            "confidence": 0.96,
            "notes": "Mapped custom axes cleanly.",
        }
        return _MockResp(
            {"candidates": [{"content": {"parts": [{"text": json.dumps(recipe_json)}]}}]}
        )

    monkeypatch.setattr(urllib.request, "urlopen", _valid_gemini_urlopen)

    h_resp = isolated_api.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "ok"
    assert h_data["metrics_count"] == len(GAUCH_2023_METRICS)
    assert h_data["llm_adapter_enabled"] is True
    assert h_data["qobs_loaded"] is True

    with open(weird_nc, "rb") as f:
        ins_resp = isolated_api.post(
            "/api/preview/inspect",
            files={"file": ("weird_schema.nc", f, "application/x-netcdf")},
            data={
                "leaderboard_id": "gauged__single_camels",
                "spatial_regime": "gauged",
                "forcing": "Daymet",
                "model_name": "Weird Schema Model",
            },
        )
    assert ins_resp.status_code == 200, ins_resp.text
    ins = ins_resp.json()
    assert ins["recipe_source"] == "llm_tier2"
    assert ins["schema"]["var"] == "custom_out_v9"
    assert ins["sanity"]["checks_passed"] is True
    # Ensure structural metadata only was sent to Gemini (no 531x3652 raw array numbers)
    assert len(captured_prompts) == 1
    assert "gauge_axis" in captured_prompts[0]

    sc_resp = isolated_api.post(
        "/api/preview/score",
        json={
            "preview_token": ins["preview_token"],
            "recipe": {
                "var": ins["schema"]["var"],
                "basin_dim": ins["schema"]["basin_dim"],
                "time_dim": ins["schema"]["time_dim"],
                "member_dim": None,
            },
            "metadata": {
                "leaderboard_id": "gauged__single_camels",
                "spatial_regime": "gauged",
                "forcing": "Daymet",
                "model_name": "Weird Schema Model",
            },
        },
    )
    assert sc_resp.status_code == 200, sc_resp.text
    sc_data = sc_resp.json()
    assert sc_data["status"] == "scored"
    assert sc_data["ensemble_single_entry"] is None
    assert sc_data["entry"]["num_evaluated_basins"] == 531
    assert list(scorer_app.PREVIEW_STORE.root_dir.iterdir()) == []


def test_preview_health_and_vertex_ai_iam_path(
    tmp_path: Path,
    isolated_api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, _, qobs = canonical_qobs_bundle
    qsim = _synthetic_qsim(qobs)

    weird_nc = tmp_path / "vertex_weird_schema.nc"
    xr.Dataset(
        data_vars={
            "custom_q_a": (("station_axis", "calendar_axis"), qsim),
            "custom_q_b": (("station_axis", "calendar_axis"), qsim * np.float32(0.5)),
        },
        coords={"station_axis": np.asarray(basins), "calendar_axis": pd.to_datetime(dates)},
    ).to_netcdf(weird_nc)

    h0 = isolated_api.get("/api/preview/health")
    assert h0.status_code == 200
    h0_data = h0.json()
    assert h0_data["llm_backend"] == "disabled"
    assert h0_data["llm_adapter_enabled"] is False
    assert h0_data["metrics_count"] == len(GAUCH_2023_METRICS)
    assert h0_data["llm_daily_limit"] == 100
    assert h0_data["llm_daily_remaining"] == 100

    monkeypatch.setenv("VERTEX_AI_PROJECT", "hydrologymip")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")
    monkeypatch.setenv("HYDROMIP_LLM_MODEL", "gemini-2.5-flash")

    h1 = isolated_api.get("/api/preview/health")
    assert h1.status_code == 200
    h1_data = h1.json()
    assert h1_data["llm_backend"] == "vertex_ai"
    assert h1_data["llm_adapter_enabled"] is True
    assert h1_data["llm_daily_remaining"] == 100

    # ADC credential failure cleanly degrades into stuck_report without crashing
    meta = ingest.extract_file_metadata(weird_nc)
    monkeypatch.setattr(
        google.auth,
        "default",
        lambda **_k: (_ for _ in ()).throw(google.auth.exceptions.DefaultCredentialsError("missing adc")),
    )
    adc_fail = llm_schema_adapter.infer_recipe_from_metadata(meta, ["Ambiguous variables"])
    assert adc_fail["status"] == "stuck"
    assert "missing adc" in adc_fail["stuck_report"]["reason"]

    class _FakeCreds:
        def __init__(self) -> None:
            self.token: str | None = None

        def refresh(self, _req: object) -> None:
            self.token = "test-vertex-iam-token"

    monkeypatch.setattr(google.auth, "default", lambda **_k: (_FakeCreds(), "hydrologymip"))
    monkeypatch.setattr(google.auth.transport.requests, "Request", lambda: object())

    seen_requests: list[tuple[str, str | None, dict[str, Any]]] = []

    class _MockResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status = 200
            self._body = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    def _vertex_urlopen(req: urllib.request.Request, **_k: Any) -> _MockResp:
        auth_header = req.get_header("Authorization")
        body = json.loads(req.data.decode("utf-8"))
        seen_requests.append((req.full_url, auth_header, body))
        recipe_json = {
            "cannot_map": False,
            "missing_or_ambiguous": [],
            "format": "netcdf",
            "var": "custom_q_a",
            "basin_dim": "station_axis",
            "time_dim": "calendar_axis",
            "member_dim": None,
            "confidence": 0.98,
            "notes": "Mapped Vertex schema.",
        }
        return _MockResp(
            {"candidates": [{"content": {"parts": [{"text": json.dumps(recipe_json)}]}}]}
        )

    monkeypatch.setattr(urllib.request, "urlopen", _vertex_urlopen)

    with open(weird_nc, "rb") as f:
        ins_resp = isolated_api.post(
            "/api/preview/inspect",
            files={"file": ("vertex_weird_schema.nc", f, "application/x-netcdf")},
            data={"leaderboard_id": "gauged__single_camels"},
        )
    assert ins_resp.status_code == 200, ins_resp.text
    ins = ins_resp.json()
    assert ins["recipe_source"] == "llm_tier2"
    assert ins["schema"]["var"] == "custom_q_a"
    assert ins["sanity"]["checks_passed"] is True
    assert len(seen_requests) == 1
    req_url, req_auth, req_body = seen_requests[0]
    assert req_url == (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/hydrologymip/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
    )
    assert req_auth == "Bearer test-vertex-iam-token"
    assert req_body["contents"][0]["role"] == "user"
    assert req_body["generationConfig"]["responseMimeType"] == "application/json"

    h2 = isolated_api.get("/api/preview/health")
    assert h2.status_code == 200
    assert h2.json()["llm_daily_remaining"] == 99


def test_llm_budget_limiter_per_ip_hour_and_daily_caps(
    tmp_path: Path,
    isolated_api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    canonical_qobs_bundle: tuple[list[str], list[str], list[str], np.ndarray],
) -> None:
    basins, dates, _, qobs = canonical_qobs_bundle
    qsim = _synthetic_qsim(qobs)

    weird_nc = tmp_path / "budget_weird.nc"
    xr.Dataset(
        data_vars={
            "custom_q_x": (("st_ax", "dt_ax"), qsim),
            "custom_q_y": (("st_ax", "dt_ax"), qsim * np.float32(0.5)),
        },
        coords={"st_ax": np.asarray(basins), "dt_ax": pd.to_datetime(dates)},
    ).to_netcdf(weird_nc)

    monkeypatch.setenv("VERTEX_AI_PROJECT", "hydrologymip")
    monkeypatch.setenv("HYDROMIP_MAX_LLM_CALLS_PER_IP_HOUR", "2")
    monkeypatch.setenv("HYDROMIP_MAX_LLM_CALLS_PER_DAY", "3")
    scorer_app.LLM_BUDGET_LIMITER.reset()

    class _FakeCreds:
        token = "test-budget-token"

        def refresh(self, _req: object) -> None:
            return None

    monkeypatch.setattr(google.auth, "default", lambda **_k: (_FakeCreds(), "hydrologymip"))
    monkeypatch.setattr(google.auth.transport.requests, "Request", lambda: object())

    llm_calls = 0

    class _MockResp:
        status = 200

        def read(self) -> bytes:
            recipe_json = {
                "cannot_map": False,
                "missing_or_ambiguous": [],
                "format": "netcdf",
                "var": "custom_q_x",
                "basin_dim": "st_ax",
                "time_dim": "dt_ax",
                "member_dim": None,
                "confidence": 0.95,
                "notes": "Mapped.",
            }
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"text": json.dumps(recipe_json)}]}}]}
            ).encode("utf-8")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    def _counting_urlopen(*_a: Any, **_k: Any) -> _MockResp:
        nonlocal llm_calls
        llm_calls += 1
        return _MockResp()

    monkeypatch.setattr(urllib.request, "urlopen", _counting_urlopen)

    def _post_inspect(client_ip: str) -> dict[str, Any]:
        with open(weird_nc, "rb") as f:
            resp = isolated_api.post(
                "/api/preview/inspect",
                headers={"X-Forwarded-For": client_ip},
                files={"file": ("budget_weird.nc", f, "application/x-netcdf")},
                data={"leaderboard_id": "gauged__single_camels"},
            )
        assert resp.status_code == 200, resp.text
        return resp.json()

    # IP1 calls 1 & 2 succeed via LLM; call 3 hits per-IP hourly cap (2/hr) without calling Vertex
    assert _post_inspect("203.0.113.10")["recipe_source"] == "llm_tier2"
    assert _post_inspect("203.0.113.10")["recipe_source"] == "llm_tier2"
    assert llm_calls == 2

    capped_ip = _post_inspect("203.0.113.10")
    assert llm_calls == 2
    assert capped_ip["stuck_report"] is not None
    assert "Per-IP hourly" in capped_ip["stuck_report"]["reason"]
    assert "hydromip package" in capped_ip["stuck_report"]["cli_snippet"]

    # IP2 uses the 3rd global daily call, then IP3 hits global daily cap (3/day)
    assert _post_inspect("203.0.113.11")["recipe_source"] == "llm_tier2"
    assert llm_calls == 3

    capped_day = _post_inspect("203.0.113.12")
    assert llm_calls == 3
    assert capped_day["stuck_report"] is not None
    assert "Daily AI schema-helper cap" in capped_day["stuck_report"]["reason"]
    assert isolated_api.get("/api/preview/health").json()["llm_daily_remaining"] == 0
