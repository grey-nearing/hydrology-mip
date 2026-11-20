# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for standalone portal bundle build, CLI packaging (`hydromip package`), and HTTP Zarr I/O."""

from __future__ import annotations

import ast
import http.server
import json
import shutil
import socketserver
import subprocess
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromip.evaluation.zarr_io import open_zarr_store, zarr_store_exists
from hydromip.paths import REPO_ROOT
from hydromip.tools import cli as hydromip_cli
from hydromip.tools import package_submission
from metrics.evaluator import GAUCH_2023_METRICS
from web.worker import deploy_cf_worker


def test_unified_portal_bundle_standalone_14_metrics_and_no_legacy(tmp_path: Path) -> None:
    stage_dir = tmp_path / "cf_stage"
    assets, summary = deploy_cf_worker.build_static_bundle(
        mip_dir=REPO_ROOT / "mips" / "camels_us_daily",
        portal_dir=REPO_ROOT / "web" / "portal",
        stage_dir=stage_dir,
        explorer_store_path=None,
        scrub_private_urls=True,
    )
    assert summary["n_leaderboards"] == 12
    assert summary["total_entries"] > 0
    assert summary["total_families"] > 0

    # All required portal files & figures exist in bundle
    for expected_key in (
        "/index.html",
        "/static/styles.css",
        "/static/app.js",
        "/data_assets/overview.json",
        "/data_assets/leaderboards.json",
        "/data_assets/families.json",
        "/data_assets/hydrographs_index.json",
        "/data_assets/protocol.json",
        "/data_assets/basins_531.txt",
        "/api/figures/figures_manifest.json",
        "/api/figures/figure_1_water_year_coverage.png",
    ):
        assert expected_key in assets, f"Missing static asset {expected_key}"
        assert assets[expected_key].is_file()

    # Check 14 canonical metrics propagated in protocol and rows
    proto_doc = json.loads(assets["/data_assets/protocol.json"].read_text(encoding="utf-8"))
    assert proto_doc["metrics"] == list(GAUCH_2023_METRICS)
    lbs_doc = json.loads(assets["/data_assets/leaderboards.json"].read_text(encoding="utf-8"))
    first_row = lbs_doc["leaderboards"][0]["rows"][0]
    for metric_key in ("gauch_nse", "gauch_kge", "gauch_log_nse", "gauch_log_kge"):
        assert metric_key in first_row
        assert first_row[metric_key] is not None

    # Enforce zero Caravan & zero public comments in web/portal/
    for fname in ("index.html", "app.js", "styles.css"):
        text = (REPO_ROOT / "web" / "portal" / fname).read_text(encoding="utf-8").lower()
        assert "caravan" not in text, f"Unexpected 'caravan' text in web/portal/{fname}"
        assert "api/comments" not in text and "community discussion" not in text

    # Check 0 AST Try nodes in deploy_cf_worker.py
    tree = ast.parse((REPO_ROOT / "web" / "worker" / "deploy_cf_worker.py").read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)]

    # Syntax-check portal app.js and rendered worker.js if gjs is installed
    gjs_bin = shutil.which("gjs")
    if gjs_bin:
        worker_js = deploy_cf_worker.render_worker_script(
            preview_gate=True,
            preview_access_key="camels-mip-2026",
            scorer_api_url="https://example-scorer.run.app",
        )
        worker_path = tmp_path / "worker.mjs"
        worker_path.write_text(worker_js, encoding="utf-8")
        app_js_path = (REPO_ROOT / "web" / "portal" / "app.js").as_posix()
        subprocess.run(
            [
                gjs_bin,
                "-c",
                f"const s = new TextDecoder().decode(imports.gi.GLib.file_get_contents({json.dumps(app_js_path)})[1]); new Function(s);",
            ],
            check=True,
        )
        subprocess.run(
            [
                gjs_bin,
                "-c",
                f"const s = new TextDecoder().decode(imports.gi.GLib.file_get_contents({json.dumps(worker_path.as_posix())})[1]); new Function(s.replace(/\\bexport\\s+default\\s+/, 'return '));",
            ],
            check=True,
        )


def test_hydromip_package_cli_transposed_netcdf_to_zarr_zip(tmp_path: Path) -> None:
    basins, dates, _ = package_submission.load_canonical_camels_coordinates()
    rng = np.random.default_rng(77)
    arr_t = rng.uniform(0.2, 4.5, size=(len(dates), len(basins))).astype(np.float32)
    short_ids = [b.lstrip("0") or "0" for b in basins]

    in_nc = tmp_path / "raw_transposed.nc"
    xr.Dataset(
        data_vars={"streamflow_pred": (("time", "gauge"), arr_t)},
        coords={"time": pd.to_datetime(dates), "gauge": short_ids},
    ).to_netcdf(in_nc)

    out_zip = tmp_path / "packaged.zarr.zip"
    rc = hydromip_cli.main(
        [
            "package",
            "--input",
            str(in_nc),
            "--output",
            str(out_zip),
            "--forcing",
            "Daymet",
            "--regime",
            "gauged",
            "--model-id",
            "cli_pkg_test",
            "--model-name",
            "CLI Packaged Test",
        ]
    )
    assert rc == 0
    assert out_zip.is_file()

    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(out_zip, "r") as zf:
        zf.extractall(extract_dir)
    zarr_dirs = list(extract_dir.glob("*.zarr"))
    assert len(zarr_dirs) == 1
    ds = xr.open_zarr(zarr_dirs[0], consolidated=True)
    assert ds["qsim"].shape == (531, 3652)
    assert ds.attrs["leaderboard_id"] == "gauged__single_camels"
    assert ds.attrs["entry_id"] == "cli_pkg_test"


def test_http_zarr_io_open_and_missing_raises(tmp_path: Path) -> None:
    zarr_dir = tmp_path / "http_demo.zarr"
    xr.Dataset(
        data_vars={"qobs": (("basin",), np.array([1.25, 2.5], dtype=np.float32))},
        coords={"basin": np.asarray(["01022500", "01031500"], dtype="<U8")},
    ).to_zarr(zarr_dir, mode="w", consolidated=True, zarr_format=2)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(tmp_path), **kwargs)  # type: ignore[arg-type]

        def log_message(self, format: str, *args: object) -> None:
            return None

    srv = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/http_demo.zarr"
    missing_url = f"http://127.0.0.1:{port}/does_not_exist.zarr"
    assert zarr_store_exists(url) is True
    assert zarr_store_exists(missing_url) is False
    opened = open_zarr_store(url)
    np.testing.assert_allclose(opened["qobs"].values, [1.25, 2.5])
    with pytest.raises(FileNotFoundError):
        open_zarr_store(missing_url)
    srv.shutdown()
    srv.server_close()
