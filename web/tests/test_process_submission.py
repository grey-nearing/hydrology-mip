# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for automated CI submission processing (`hydromip.tools.process_submission`)."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from hydromip.io import doi_resolver
from hydromip.paths import REPO_ROOT
from hydromip.tools import process_submission
from metrics.evaluator import GAUCH_2023_METRICS


def _make_temp_mip_dir(tmp_path: Path) -> Path:
    camels_mip = REPO_ROOT / "mips" / "camels_us_daily"
    mip_copy = tmp_path / "mips" / "camels_us_daily"
    (mip_copy / "splits").mkdir(parents=True, exist_ok=True)
    shutil.copytree(camels_mip / "splits", mip_copy / "splits", dirs_exist_ok=True)
    shutil.copytree(camels_mip / "leaderboards", mip_copy / "leaderboards", dirs_exist_ok=True)
    if (camels_mip / "protocol.json").is_file():
        shutil.copy2(camels_mip / "protocol.json", mip_copy / "protocol.json")
    return mip_copy


def _write_hermetic_qobs_npz(out_npz: Path) -> tuple[list[str], list[str], np.ndarray]:
    splits_dir = REPO_ROOT / "mips" / "camels_us_daily" / "splits"
    sr_doc = json.loads((splits_dir / "splits_and_regimes.json").read_text(encoding="utf-8"))
    b_map = sr_doc["basin_to_huc02"]
    basins = sorted(str(b).zfill(8) for b in b_map.keys())
    huc_02 = [str(b_map[b]).zfill(2) for b in basins]
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("1989-10-01", "1999-09-30", freq="D")]
    rng = np.random.default_rng(20261121)
    qobs = rng.gamma(shape=2.0, scale=1.2, size=(len(basins), len(dates))).astype(np.float32)
    qobs[0, 5:10] = np.nan
    np.savez(
        out_npz,
        basins=np.asarray(basins, dtype="<U8"),
        dates=np.asarray(dates, dtype="<U10"),
        huc_02=np.asarray(huc_02, dtype="<U2"),
        qobs=qobs,
    )
    return basins, dates, qobs


def _write_synthetic_netcdf(
    out_nc: Path,
    basins: list[str],
    dates: list[str],
    qobs: np.ndarray,
    scale: float = 0.93,
) -> Path:
    qsim = np.nan_to_num(qobs * np.float32(scale) + np.float32(0.05), nan=0.1).astype(np.float32)
    xr.Dataset(
        data_vars={"qsim": (("basin", "date"), qsim)},
        coords={"basin": np.asarray(basins), "date": pd.to_datetime(dates)},
    ).to_netcdf(out_nc)
    return out_nc


def _manifest_payload(
    *,
    model_name: str,
    archive_url: str,
    code_url: str = "",
    closed_source: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "benchmark": "camels_us_daily",
        "leaderboard_id": "gauged__single_camels",
        "spatial_regime": "gauged",
        "model_name": model_name,
        "forcing": "Daymet",
        "paradigm": "Machine learning",
        "category": "Contribution",
        "archive_url": archive_url,
        "code_url": code_url,
        "closed_source": closed_source,
        "citation": "Community Test (2026)",
        "compliance_attestation": {
            "blind_holdout_window": True,
            "strict_spatial_holdout_canonical_folds": True,
            "all_forcing_inputs_declared": True,
            "deterministic_daily_predictions_mm_per_day": True,
        },
    }


def test_process_submission_zero_try_except_and_workflow_yaml() -> None:
    proc_path = REPO_ROOT / "hydromip" / "tools" / "process_submission.py"
    tree = ast.parse(proc_path.read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    doi_path = REPO_ROOT / "hydromip" / "io" / "doi_resolver.py"
    doi_tree = ast.parse(doi_path.read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(doi_tree) if isinstance(n, ast.Try)]
    for wf in (
        REPO_ROOT / ".github" / "workflows" / "score_submission.yml",
        REPO_ROOT / ".github" / "workflows" / "issue_to_pr.yml",
        REPO_ROOT / ".github" / "workflows" / "deploy_website.yml",
        REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "model_submission.yml",
    ):
        assert wf.is_file(), f"Missing GitHub file {wf}"
        parsed = yaml.safe_load(wf.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)


def test_shared_doi_resolver_zenodo_osf_figshare_and_ssrf_guard() -> None:
    def _mock_fetch(url: str, **_kwargs: Any) -> bytes:
        if "zenodo.org/api/records/23641695" in url:
            return json.dumps(
                {
                    "files": [
                        {"key": "README.md", "size": 5740, "links": {"content": "https://zenodo.org/api/records/23641695/files/README.md/content"}},
                        {"key": "era_gauged_multi_paradigm_gauged_multi_camels_hydrographs_531x3652.npz", "size": 145803779, "links": {"content": "https://zenodo.org/api/records/23641695/files/era.npz/content"}},
                        {"key": "era_gauged_multi_paradigm_gauged_multi_camels.zarr.zip", "size": 69714486, "links": {"content": "https://zenodo.org/api/records/23641695/files/era.zarr.zip/content"}},
                    ]
                }
            ).encode("utf-8")
        if "api.osf.io/v2/nodes/abc12/files/osfstorage/" in url:
            return json.dumps(
                {
                    "data": [
                        {
                            "attributes": {"kind": "file", "name": "preds.nc", "size": 1024},
                            "links": {"download": "https://osf.io/download/xyz/"},
                        }
                    ]
                }
            ).encode("utf-8")
        if "api.figshare.com/v2/articles/987654/files" in url:
            return json.dumps(
                [
                    {"name": "README.md", "size": 200, "download_url": "https://figshare.com/ndownloader/files/1"},
                    {"name": "runoff.parquet", "size": 2048, "download_url": "https://figshare.com/ndownloader/files/2"},
                ]
            ).encode("utf-8")
        raise AssertionError(f"Unexpected mocked URL: {url}")

    z_name, z_dl = doi_resolver.resolve_archive_download_url(
        "https://doi.org/10.5281/zenodo.23641695", fetch_fn=_mock_fetch
    )
    assert z_name == "era_gauged_multi_paradigm_gauged_multi_camels.zarr.zip"
    assert z_dl.endswith("/era.zarr.zip/content")

    o_name, o_dl = doi_resolver.resolve_archive_download_url(
        "https://doi.org/10.17605/OSF.IO/abc12", fetch_fn=_mock_fetch
    )
    assert o_name == "preds.nc"
    assert o_dl == "https://osf.io/download/xyz/"

    f_name, f_dl = doi_resolver.resolve_archive_download_url(
        "https://doi.org/10.6084/m9.figshare.987654", fetch_fn=_mock_fetch
    )
    assert f_name == "runoff.parquet"
    assert f_dl == "https://figshare.com/ndownloader/files/2"

    with pytest.raises(ValueError, match="SSRF guard"):
        doi_resolver.validate_https_target_url("https://169.254.169.254/latest/meta-data/")


def test_process_submission_open_source_dry_run_and_closed_source_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mip_copy = _make_temp_mip_dir(tmp_path)
    qobs_npz = tmp_path / "qobs_531.npz"
    basins, dates, qobs = _write_hermetic_qobs_npz(qobs_npz)

    # 1. Dry-run open-source submission via file:// URI (must NOT mutate leaderboard JSONs)
    nc_open = _write_synthetic_netcdf(tmp_path / "open_model.nc", basins, dates, qobs, scale=0.94)
    open_manifest_path = tmp_path / "open_manifest.json"
    open_manifest_path.write_text(
        json.dumps(
            _manifest_payload(
                model_name="CI Open NetCDF Model",
                archive_url=nc_open.as_uri(),
                code_url="https://github.com/example/open-hydro-model",
                closed_source=False,
            )
        ),
        encoding="utf-8",
    )
    zarr_stage = tmp_path / "staged_zarr"
    md_open = tmp_path / "open_summary.md"

    lb_before = (mip_copy / "leaderboards" / "gauged__single_camels.json").read_text(encoding="utf-8")
    res_open = process_submission.process_submission(
        manifest_path=open_manifest_path,
        mip_dir=mip_copy,
        out_zarr_dir=zarr_stage,
        summary_md_path=md_open,
        qobs_npz_path=qobs_npz,
        dry_run=True,
    )
    assert res_open["status"] == "processed"
    assert res_open["dry_run"] is True
    assert set(res_open["entry"]["gauch_2023_metrics"].keys()) == set(GAUCH_2023_METRICS)
    assert res_open["entry"]["closed_source"] is False
    assert (Path(res_open["zarr_path"]) / ".zmetadata").is_file()
    assert (mip_copy / "leaderboards" / "gauged__single_camels.json").read_text(encoding="utf-8") == lb_before
    md_text = md_open.read_text(encoding="utf-8")
    assert "Canonical 14-Metric Evaluation Summary" in md_text
    assert "`logNSE`" in md_text and "`logKGE`" in md_text
    assert "Public Repository" in md_text

    # 2. Non-dry-run closed-source submission via mocked HTTPS Zenodo DOI
    nc_closed = _write_synthetic_netcdf(tmp_path / "closed_model.nc", basins, dates, qobs, scale=0.89)
    closed_bytes = nc_closed.read_bytes()

    def _fake_fetch_https(url: str, dest_path: Path | None = None, **_kwargs: Any) -> bytes:
        if "zenodo.org/api/records/9990001" in url:
            payload = {
                "files": [
                    {
                        "key": "closed_model.nc",
                        "size": len(closed_bytes),
                        "links": {"content": "https://zenodo.org/records/9990001/files/closed_model.nc?download=1"},
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")
        assert dest_path is not None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(closed_bytes)
        return b""

    monkeypatch.setattr(process_submission, "_fetch_https", _fake_fetch_https)

    closed_manifest_path = tmp_path / "closed_manifest.json"
    closed_manifest_path.write_text(
        json.dumps(
            _manifest_payload(
                model_name="Proprietary Hydro Suite",
                archive_url="https://doi.org/10.5281/zenodo.9990001",
                code_url="",
                closed_source=True,
            )
        ),
        encoding="utf-8",
    )
    md_closed = tmp_path / "closed_summary.md"
    res_closed = process_submission.process_submission(
        manifest_path=closed_manifest_path,
        mip_dir=mip_copy,
        out_zarr_dir=zarr_stage,
        summary_md_path=md_closed,
        qobs_npz_path=qobs_npz,
        dry_run=False,
        update_figures=False,
    )
    assert res_closed["dry_run"] is False
    assert res_closed["entry"]["closed_source"] is True
    assert "`Closed-Source`" in md_closed.read_text(encoding="utf-8")

    updated_lb = json.loads((mip_copy / "leaderboards" / "gauged__single_camels.json").read_text(encoding="utf-8"))
    matched_rows = [r for r in updated_lb["rows"] if r["entry_id"] == res_closed["entry"]["entry_id"]]
    assert len(matched_rows) == 1
    assert matched_rows[0]["closed_source"] is True
    assert set(matched_rows[0]["gauch_2023_metrics"].keys()) == set(GAUCH_2023_METRICS)

    master_tables = json.loads((mip_copy / "leaderboards" / "tables_3_to_14.json").read_text(encoding="utf-8"))
    master_lb = next(b for b in master_tables["leaderboards"] if b["id"] == "gauged__single_camels")
    assert any(r["entry_id"] == res_closed["entry"]["entry_id"] for r in master_lb["rows"])

    reg = json.loads((mip_copy / "leaderboards" / "table_2_registry.json").read_text(encoding="utf-8"))
    assert any(f["family_id"] == res_closed["entry"]["family_id"] for f in reg["families"])
    # Verify exact 20-key registry schema uniformity even when adding a brand new family
    assert len({frozenset(f.keys()) for f in reg["families"]}) == 1


def test_process_submission_zarr_targets_explicit_entry_id_and_ensemble_single_parity(
    tmp_path: Path,
) -> None:
    mip_copy = _make_temp_mip_dir(tmp_path)
    qobs_npz = tmp_path / "qobs_531.npz"
    basins, dates, qobs = _write_hermetic_qobs_npz(qobs_npz)

    # Materialize datetime64[ns] Zarr targets (matching CI R2/GCS format) so we cover zarr target loading
    targets_zarr = mip_copy / "observations" / "newman_eval_targets.zarr"
    targets_zarr.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={"qobs": (("basin", "date"), qobs)},
        coords={
            "basin": np.asarray(basins, dtype="<U8"),
            "date": pd.to_datetime(dates).values.astype("datetime64[ns]"),
        },
    ).to_zarr(str(targets_zarr), mode="w", consolidated=True, zarr_format=2)

    # Build a 3-seed .zarr.zip ensemble archive
    rng = np.random.default_rng(11)
    members = np.stack(
        [
            np.nan_to_num(qobs * np.float32(0.95) + np.float32(0.03 * i + rng.normal(0, 0.02, qobs.shape)), nan=0.1).astype(np.float32)
            for i in range(3)
        ],
        axis=0,
    )
    members = np.maximum(members, np.float32(0.0))
    qsim_mean = np.mean(members, axis=0).astype(np.float32)
    ens_zarr_dir = tmp_path / "ens_src.zarr"
    xr.Dataset(
        data_vars={
            "qsim": (("basin", "date"), qsim_mean),
            "qsim_members": (("member", "basin", "date"), members),
        },
        coords={
            "basin": np.asarray(basins, dtype="<U8"),
            "date": pd.to_datetime(dates),
            "member": np.arange(3, dtype=np.int64),
        },
        attrs={"is_ensemble": True, "num_members": 3},
    ).to_zarr(str(ens_zarr_dir), mode="w", consolidated=True, zarr_format=2)

    import zipfile as _zf

    ens_zip = tmp_path / "ens_model.zarr.zip"
    with _zf.ZipFile(ens_zip, "w") as zf:
        for fp in sorted(p for p in ens_zarr_dir.rglob("*") if p.is_file()):
            zf.write(fp, arcname=fp.relative_to(ens_zarr_dir).as_posix())

    manifest = {
        **_manifest_payload(
            model_name="Multi-Seed Community (3-model ensemble)",
            archive_url=ens_zip.as_uri(),
            code_url="https://github.com/example/multiseed",
        ),
        "leaderboard_id": "gauged__multi_camels",
        "entry_id": "custom_multiseed_model",
        "single_model_name": "Multi-Seed Community (single model)",
        "family_id": "custom_multiseed_fam",
        "new_family_name": "Multi-Seed Community Family",
    }
    manifest_path = tmp_path / "ens_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    res = process_submission.process_submission(
        manifest_path=manifest_path,
        mip_dir=mip_copy,
        out_zarr_dir=tmp_path / "staged_zarr",
        summary_md_path=tmp_path / "ens_summary.md",
        qobs_npz_path=None,  # force reading newman_eval_targets.zarr
        dry_run=False,
        update_figures=False,
    )
    assert res["entry"]["entry_id"] == "custom_multiseed_model"
    assert res["entry"]["zarr_path"] == "gs://camels-us-mip/hydrographs/gauged__multi_camels/custom_multiseed_model.zarr"
    assert res["ensemble_single_entry"] is not None
    assert res["ensemble_single_entry"]["entry_id"] == "custom_multiseed_model_single"
    assert res["ensemble_single_entry"]["model_name"] == "Multi-Seed Community (single model)"
    assert res["ensemble_single_entry"]["single_med_nse_std"] is not None

    from leaderboards.update_leaderboard import build_hydrograph_manifest, refresh_registry_counts

    assert refresh_registry_counts(mip_copy / "leaderboards") is False
    hm = build_hydrograph_manifest(leaderboards_dir=mip_copy / "leaderboards", updated_utc="2026-01-01T00:00:00Z")
    assert any(s["store_name"] == "custom_multiseed_model.zarr" for s in hm["stores"])

    # Submit a second model into the same existing family (like portal dropdown reuse without new_family_name)
    manifest_reuse = {
        **_manifest_payload(
            model_name="Multi-Seed Community PUB Follow-Up",
            archive_url=ens_zip.as_uri(),
            code_url="https://github.com/example/multiseed",
        ),
        "leaderboard_id": "pub__multi_camels",
        "entry_id": "custom_multiseed_pub",
        "family_id": "custom_multiseed_fam",
        "new_family_name": None,
    }
    manifest_reuse_path = tmp_path / "ens_reuse_manifest.json"
    manifest_reuse_path.write_text(json.dumps(manifest_reuse), encoding="utf-8")
    res_reuse = process_submission.process_submission(
        manifest_path=manifest_reuse_path,
        mip_dir=mip_copy,
        out_zarr_dir=tmp_path / "staged_zarr",
        summary_md_path=tmp_path / "ens_reuse_summary.md",
        qobs_npz_path=None,
        dry_run=False,
        update_figures=False,
    )
    assert res_reuse["entry"]["family_id"] == "custom_multiseed_fam"
    assert res_reuse["entry"]["family_name"] == "Multi-Seed Community Family"
    reg_after = json.loads((mip_copy / "leaderboards" / "table_2_registry.json").read_text(encoding="utf-8"))
    fam_entries = [f for f in reg_after["families"] if f["family_id"] == "custom_multiseed_fam"]
    assert len(fam_entries) == 1
    assert set(fam_entries[0]["leaderboards_covered"]) == {"gauged__multi_camels", "pub__multi_camels"}
