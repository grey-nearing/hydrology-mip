"""Native-resolution (531 x 3652) tests for `leaderboards/update_leaderboard.py`."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from leaderboards import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_LEADERBOARDS_DIR,
    clean_model_name,
    load_all_leaderboards,
    load_leaderboard,
    rank_leaderboard_rows,
    recalculate_all_self_run_models,
    trigger_submission_from_gcs,
    upsert_leaderboard_entry,
)
from leaderboards.update_leaderboard import main as cli_main
from leaderboards.update_leaderboard import normalize_legacy_row

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_zero_try_except_blocks_in_leaderboards() -> None:
    """Enforce zero try/except blocks in `leaderboards/`."""
    for py_path in sorted(DEFAULT_LEADERBOARDS_DIR.rglob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        assert not try_nodes, f"Forbidden try/except block found in {py_path}"


@pytest.mark.unit
def test_all_12_per_leaderboard_json_files_exist_and_load() -> None:
    """Verify all 12 individual `<leaderboard_id>.json` files exist in `leaderboards/`."""
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb_path = DEFAULT_LEADERBOARDS_DIR / f"{lbid}.json"
        assert lb_path.is_file(), f"Missing standalone leaderboard file: {lb_path}"
        lb_data = load_leaderboard(lbid)
        assert lb_data["id"] == lbid
        assert "rows" in lb_data
        assert lb_data["n_entries"] == len(lb_data["rows"])

    all_lbs = load_all_leaderboards()
    assert len(all_lbs["leaderboards"]) == 12
    assert "protocol" in all_lbs


@pytest.mark.integration
def test_upsert_and_trigger_submission_from_gcs(tmp_path: Path) -> None:
    """Verify submission-triggered scoring from Zarr and deterministic JSON re-ranking."""
    lb_dir = tmp_path / "leaderboards"
    lb_dir.mkdir(parents=True, exist_ok=True)
    for lbid in CANONICAL_LEADERBOARD_IDS:
        shutil.copy2(DEFAULT_LEADERBOARDS_DIR / f"{lbid}.json", lb_dir / f"{lbid}.json")
    shutil.copy2(DEFAULT_LEADERBOARDS_DIR / "tables_3_to_14.json", lb_dir / "tables_3_to_14.json")

    splits = json.loads(
        (REPO_ROOT / "splits" / "splits_and_regimes.json").read_text(encoding="utf-8")
    )
    basin_to_huc = splits["basin_to_huc02"]
    basins = np.array(sorted(basin_to_huc.keys()), dtype="<U8")
    huc_02 = np.array([str(basin_to_huc[str(b)]).zfill(2) for b in basins], dtype="<U2")
    dates = pd.date_range("1989-10-01", "1999-09-30", freq="D").values.astype("datetime64[ns]")

    rng = np.random.default_rng(7)
    t_idx = np.arange(3652, dtype=np.float32)
    seasonal = 2.0 + 1.5 * np.sin(2.0 * np.pi * t_idx / 365.25)
    qobs = np.broadcast_to(seasonal[None, :], (531, 3652)).copy() + rng.uniform(
        0.1, 0.5, size=(531, 3652)
    ).astype(np.float32)
    qsim = (qobs * 0.99).astype(np.float32)

    bucket_root = tmp_path / "bucket"
    obs_store = bucket_root / "observations" / "newman_eval_targets.zarr"
    obs_store.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={"qobs": (("basin", "date"), qobs, {"units": "mm/day"})},
        coords={"basin": ("basin", basins), "date": ("date", dates), "huc_02": ("basin", huc_02)},
    ).to_zarr(obs_store, mode="w", consolidated=True, zarr_format=2)

    sim_store = (
        bucket_root / "hydrographs" / "gauged__single_camels" / "community_test_model.zarr"
    )
    sim_store.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={"qsim": (("basin", "date"), qsim, {"units": "mm/day"})},
        coords={"basin": ("basin", basins), "date": ("date", dates), "huc_02": ("basin", huc_02)},
    ).to_zarr(sim_store, mode="w", consolidated=True, zarr_format=2)

    rc = cli_main(
        [
            "--model-id",
            "community_test_model",
            "--leaderboard-id",
            "gauged__single_camels",
            "--model-name",
            "Community Test Model",
            "--family-id",
            "kratzert_lstm",
            "--forcing",
            "Daymet",
            "--code-url",
            "https://github.com/example/community-test",
            "--bucket-uri",
            str(bucket_root),
            "--leaderboards-dir",
            str(lb_dir),
        ]
    )
    assert rc == 0

    updated_lb = load_leaderboard("gauged__single_camels", lb_dir)
    assert updated_lb["top_model"] == "Community Test Model"
    assert updated_lb["rows"][0]["entry_id"] == "community_test_model"
    assert updated_lb["rows"][0]["rank"] == 1
    assert "gauch_2023_metrics" in updated_lb["rows"][0]

    # Verify master tables_3_to_14.json was also synchronized
    master_synced = json.loads((lb_dir / "tables_3_to_14.json").read_text(encoding="utf-8"))
    t3 = [lb for lb in master_synced["leaderboards"] if lb["id"] == "gauged__single_camels"][0]
    assert t3["rows"][0]["entry_id"] == "community_test_model"
    assert t3["top_model"] == "Community Test Model"

    # Verify upserting an existing entry_id updates in-place without duplicating
    count_before = len(updated_lb["rows"])
    res2 = trigger_submission_from_gcs(
        model_id="community_test_model",
        leaderboard_id="gauged__single_camels",
        model_name="Community Test Model v2",
        family_id="kratzert_lstm",
        forcing="Daymet",
        bucket_uri=str(bucket_root),
        leaderboards_dir=lb_dir,
        update_figures=False,
    )
    assert res2["updated_entry"]["model_name"] == "Community Test Model v2"
    updated_lb2 = load_leaderboard("gauged__single_camels", lb_dir)
    assert len(updated_lb2["rows"]) == count_before

    # A memberless store without an `is_ensemble` attribute is a single model: the
    # single-seed column carries the score and the ensemble column is empty.
    row = res2["updated_entry"]
    assert row["is_ensemble"] is False
    assert row["num_members"] == 1
    assert row["ens_med_nse"] is None and row["ens_med_nse_display"] == "---"
    assert row["single_med_nse"] == row["primary_ranking_nse"]
    assert row["single_med_nse_std"] is None
    assert len(row["per_basin_nse"]) == 531

    # An ensemble claim from the command line that the store cannot back is rejected.
    with pytest.raises(ValueError, match="Ensemble claim mismatch"):
        trigger_submission_from_gcs(
            model_id="community_test_model",
            leaderboard_id="gauged__single_camels",
            model_name="Community Test Model v3",
            family_id="kratzert_lstm",
            forcing="Daymet",
            is_ensemble=True,
            bucket_uri=str(bucket_root),
            leaderboards_dir=lb_dir,
            update_figures=False,
        )

    # Verify upsert_leaderboard_entry validation errors
    with pytest.raises(ValueError, match="Unknown leaderboard_id"):
        upsert_leaderboard_entry("invalid_lb", {"entry_id": "x", "model_name": "y"}, lb_dir)


def _write_member_bucket(
    bucket_root: Path,
    leaderboard_id: str,
    model_id: str,
    n_members: int,
    is_ensemble: bool,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Write synthetic targets plus a hydrograph store carrying `qsim_members`."""
    splits = json.loads(
        (REPO_ROOT / "splits" / "splits_and_regimes.json").read_text(encoding="utf-8")
    )
    basin_to_huc = splits["basin_to_huc02"]
    basins = np.array(sorted(basin_to_huc.keys()), dtype="<U8")
    huc_02 = np.array([str(basin_to_huc[str(b)]).zfill(2) for b in basins], dtype="<U2")
    dates = pd.date_range("1989-10-01", "1999-09-30", freq="D").values.astype("datetime64[ns]")

    rng = np.random.default_rng(seed)
    t_idx = np.arange(3652, dtype=np.float64)
    seasonal = 2.0 + 1.5 * np.sin(2.0 * np.pi * t_idx / 365.25)
    qobs = (seasonal[None, :] + rng.uniform(0.1, 0.5, size=(531, 3652))).astype(np.float32)
    members = (
        qobs[np.newaxis].astype(np.float64)
        + rng.normal(0.0, 0.5, size=(n_members, 531, 3652))
    ).astype(np.float32)
    qsim = members.astype(np.float64).mean(axis=0).astype(np.float32) if is_ensemble else members[0]

    obs_store = bucket_root / "observations" / "newman_eval_targets.zarr"
    obs_store.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={"qobs": (("basin", "date"), qobs, {"units": "mm/day"})},
        coords={"basin": (("basin"), basins), "date": ("date", dates), "huc_02": ("basin", huc_02)},
    ).to_zarr(obs_store, mode="w", consolidated=True, zarr_format=2)

    sim_store = bucket_root / "hydrographs" / leaderboard_id / f"{model_id}.zarr"
    sim_store.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={
            "qsim": (("basin", "date"), qsim, {"units": "mm/day"}),
            "qsim_members": (("member", "basin", "date"), members, {"units": "mm/day"}),
        },
        coords={
            "basin": ("basin", basins),
            "date": ("date", dates),
            "huc_02": ("basin", huc_02),
            "member": ("member", np.arange(n_members, dtype=np.int64)),
        },
        attrs={"is_ensemble": is_ensemble, "num_members": n_members},
    ).to_zarr(sim_store, mode="w", consolidated=True, zarr_format=2)
    return qobs, members


@pytest.mark.integration
def test_recalculation_splits_ensemble_and_single_seed_columns(tmp_path: Path) -> None:
    """`recalculate_all_self_run_models` must never copy the ensemble score into the single column."""
    lb_dir = tmp_path / "leaderboards"
    lb_dir.mkdir(parents=True, exist_ok=True)
    for lbid in CANONICAL_LEADERBOARD_IDS:
        # Keep the real file structure but no rows, so only the synthetic entry is scored.
        src = json.loads((DEFAULT_LEADERBOARDS_DIR / f"{lbid}.json").read_text(encoding="utf-8"))
        src["rows"] = []
        src["n_entries"] = 0
        (lb_dir / f"{lbid}.json").write_text(json.dumps(src, indent=2), encoding="utf-8")

    bucket_root = tmp_path / "bucket"
    _write_member_bucket(bucket_root, "pur__freeform", "ens_ii_model", 3, True)

    # Seed a legacy-style self-run row with the collapsed columns (the audited defect)
    lb_path = lb_dir / "pur__freeform.json"
    lb_data = json.loads(lb_path.read_text(encoding="utf-8"))
    lb_data["rows"] = [
        {
            "entry_id": "ens_ii_model",
            "leaderboard_id": "pur__freeform",
            "model_name": "Synthetic ERA model (3-model ensemble)",
            "family_id": "synthetic_family",
            "category": "MIP Re-run",
            "forcing": "Daymet",
            "ens_med_nse": 0.5,
            "single_med_nse": 0.5,
            "primary_ranking_nse": 0.5,
            "mean_nse": 0.4,
            "med_kge": 0.4,
            "med_fhv_abs": 10.0,
            "nse_le_0": 0,
        }
    ]
    lb_path.write_text(json.dumps(lb_data, indent=2), encoding="utf-8")
    (lb_dir / "contributions_index.json").write_text(
        json.dumps(
            {
                "total_contributions": 1,
                "entries": {
                    "ens_ii_model": {
                        "entry_id": "ens_ii_model",
                        "category": "MIP Re-run",
                        "is_ensemble": True,
                        "num_members": 3,
                        "target_leaderboard_tables": ["table_14", "pur__freeform"],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = recalculate_all_self_run_models(
        bucket_uri=str(bucket_root), leaderboards_dir=lb_dir, update_figures=False
    )
    assert report["total_recalculated"] == 1

    row = load_leaderboard("pur__freeform", lb_dir)["rows"][0]
    assert row["is_ensemble"] is True
    assert row["num_members"] == 3
    assert row["has_qsim_members"] is True
    assert row["ens_med_nse"] == row["primary_ranking_nse"]
    assert row["single_med_nse"] is not None
    assert row["single_med_nse"] < row["ens_med_nse"]
    assert row["single_med_nse_std"] is not None
    assert "±" in row["single_med_nse_display"]
    assert len(row["single_med_nse_per_member"]) == 3
    assert len(row["per_basin_nse"]) == 531

    ci = json.loads((lb_dir / "contributions_index.json").read_text(encoding="utf-8"))
    ci_metrics = ci["entries"]["ens_ii_model"]["metrics"]
    assert ci_metrics["single_seed_median_nse_mean"] == pytest.approx(
        np.mean(row["single_med_nse_per_member"]), abs=1e-3
    )


@pytest.mark.unit
def test_rank_rows_sorts_none_and_sentinels_last() -> None:
    """Non-finite NSE values (legacy `-999.0`) and None never rank above real scores."""
    rows = [
        {"entry_id": "a", "model_name": "A", "primary_ranking_nse": -999.0, "mean_nse": 0.6},
        {"entry_id": "b", "model_name": "B", "ens_med_nse": 0.70, "single_med_nse": None},
        {"entry_id": "c", "model_name": "C", "primary_ranking_nse": None},
        {"entry_id": "d", "model_name": "D", "single_med_nse": 0.72},
    ]
    ranked = rank_leaderboard_rows(rows)
    assert [r["entry_id"] for r in ranked] == ["d", "b", "a", "c"]
    assert ranked[0]["primary_ranking_nse"] == 0.72
    assert ranked[2]["primary_ranking_nse"] is None
    assert ranked[3]["primary_ranking_nse"] is None


@pytest.mark.unit
def test_normalize_legacy_row_splits_columns_and_strips_residue() -> None:
    """Legacy rows: sentinel -> None, collapsed columns split by ensemble status, residue removed."""
    ens_row = {
        "entry_id": "t_ens",
        "model_name": "Some LSTM (8-model ensemble)",
        "ens_med_nse": 0.75,
        "ens_med_nse_display": "0.750",
        "single_med_nse": 0.75,
        "single_med_nse_display": "0.750",
        "primary_ranking_nse": 0.75,
        "zenodo_url": "https://github.com/x/y/blob/main/hydrographs/category_contribution/t_ens.npz",
        "mip_hydrograph_path": "hydrographs/category_contribution/t_ens.npz",
        "mip_manifest_path": "submissions/category_contribution/t_ens/manifest.json",
        "zenodo_doi": "10.5281/zenodo.15000531",
    }
    out, changes = normalize_legacy_row(ens_row)
    assert out["is_ensemble"] is True
    assert out["ens_med_nse"] == 0.75
    assert out["single_med_nse"] is None and out["single_med_nse_display"] == "---"
    assert out["single_med_nse_std"] is None
    assert "mip_hydrograph_path" not in out and "mip_manifest_path" not in out
    assert out["zenodo_url"] == "" and out["zenodo_doi"] == ""
    assert out["primary_ranking_nse"] == 0.75
    assert len(changes) >= 5

    single_row = {
        "entry_id": "t_single",
        "model_name": "Some LSTM",
        "ens_med_nse": 0.70,
        "single_med_nse": 0.70,
        "primary_ranking_nse": 0.70,
    }
    out_s, _ = normalize_legacy_row(single_row, {"is_ensemble": False})
    assert out_s["is_ensemble"] is False
    assert out_s["ens_med_nse"] is None and out_s["ens_med_nse_display"] == "---"
    assert out_s["single_med_nse"] == 0.70

    sentinel_row = {
        "entry_id": "t_sent",
        "model_name": "Geo-ViT-LSTM",
        "ens_med_nse": None,
        "single_med_nse": None,
        "primary_ranking_nse": -999.0,
        "mean_nse": 0.62,
    }
    out_n, changes_n = normalize_legacy_row(sentinel_row)
    assert out_n["primary_ranking_nse"] is None
    assert any("sentinel" in c for c in changes_n)

    # Rows already split by the evaluator (carrying a real std) are left untouched
    split_row = {
        "entry_id": "t_split",
        "model_name": "X (4-model ensemble)",
        "ens_med_nse": 0.80,
        "single_med_nse": 0.80,
        "single_med_nse_std": 0.0,
        "primary_ranking_nse": 0.80,
    }
    out_split, _ = normalize_legacy_row(split_row)
    assert out_split["single_med_nse"] == 0.80


@pytest.mark.unit
def test_repository_leaderboards_carry_no_sentinels_or_collapsed_columns() -> None:
    """The committed JSON files must be free of the audited defects after normalization."""
    for lbid in CANONICAL_LEADERBOARD_IDS:
        for row in load_leaderboard(lbid)["rows"]:
            for key in ("primary_ranking_nse", "ens_med_nse", "single_med_nse"):
                value = row.get(key)
                assert value is None or (
                    isinstance(value, (int, float)) and np.isfinite(value)
                ), f"{lbid}/{row['entry_id']}: non-finite {key}={value!r}"
            assert "mip_hydrograph_path" not in row and "mip_manifest_path" not in row
            assert row.get("zenodo_doi") != "10.5281/zenodo.15000531"
            assert isinstance(row.get("is_ensemble"), bool), f"{lbid}/{row['entry_id']}"
            ens_v, single_v = row.get("ens_med_nse"), row.get("single_med_nse")
            if ens_v is not None and single_v is not None:
                # Both columns may coexist only when the single-seed value was derived
                # from per-member hydrographs (then a std accompanies it).
                assert row.get("single_med_nse_std") is not None, (
                    f"{lbid}/{row['entry_id']}: ensemble and single columns both set "
                    f"without per-member provenance"
                )


@pytest.mark.unit
def test_no_author_or_citation_in_leaderboard_model_names() -> None:
    """Every leaderboard `model_name` must be free of author/citation prefixes; citation goes in `citation`."""
    assert (
        clean_model_name("Kratzert et al. (2021) LSTM (10-model ensemble)")
        == "LSTM (10-model ensemble)"
    )
    assert (
        clean_model_name("Rasiya Koya & Roy (2025) TFT (10-model ensemble)")
        == "TFT (10-model ensemble)"
    )
    assert (
        clean_model_name(
            "ERA Gauged Minimal Guidance (Node 73 EntityAwareLSTM)",
            is_ensemble=True,
            num_members=10,
        )
        == "ERA-Unguided-Gauged (10-model ensemble)"
    )

    for lbid in CANONICAL_LEADERBOARD_IDS:
        for row in load_leaderboard(lbid)["rows"]:
            mname = str(row.get("model_name", ""))
            assert mname == clean_model_name(
                mname,
                is_ensemble=row.get("is_ensemble"),
                num_members=row.get("num_members"),
            ), f"{lbid}/{row['entry_id']}: uncleaned model_name {mname!r}"
            assert "et al." not in mname, f"{lbid}/{row['entry_id']}: author in model_name {mname!r}"
            assert str(row.get("citation", "")).strip(), (
                f"{lbid}/{row['entry_id']}: missing subtitle citation"
            )

