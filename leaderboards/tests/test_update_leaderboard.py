# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

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
    build_hydrograph_manifest,
    clean_model_name,
    load_all_leaderboards,
    load_leaderboard,
    rank_leaderboard_rows,
    recalculate_all_self_run_models,
    render_stage1_appendix_csv,
    render_stage1_appendix_markdown,
    resolve_row_store_id,
    trigger_submission_from_gcs,
    upsert_leaderboard_entry,
    validate_stage1_appendix,
    write_hydrograph_manifest,
)
from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths
from leaderboards.update_leaderboard import main as cli_main
from leaderboards.update_leaderboard import normalize_legacy_row

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADERBOARDS_PKG_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = resolve_mip_paths(DEFAULT_MIP_ID).splits_dir


@pytest.mark.unit
def test_zero_try_except_blocks_in_leaderboards() -> None:
    """Enforce zero try/except blocks in `leaderboards/`."""
    py_files = sorted(LEADERBOARDS_PKG_DIR.rglob("*.py"))
    assert py_files, f"Expected Python modules under {LEADERBOARDS_PKG_DIR}"
    for py_path in py_files:
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
        (SPLITS_DIR / "splits_and_regimes.json").read_text(encoding="utf-8")
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
        (SPLITS_DIR / "splits_and_regimes.json").read_text(encoding="utf-8")
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



# --- Repository data invariants (pre-launch audit regression tests) ---------------------

_PARTIAL_CATEGORIES = {"Archive*", "Literature*"}


def _repo_rows() -> list[tuple[str, dict]]:
    return [(lbid, row) for lbid in CANONICAL_LEADERBOARD_IDS for row in load_leaderboard(lbid)["rows"]]


@pytest.mark.unit
def test_repository_leaderboard_metadata_is_consistent() -> None:
    """Counts, ranks, and top model agree with the rows; no stale `num_entries` field remains."""
    for lbid in CANONICAL_LEADERBOARD_IDS:
        lb = load_leaderboard(lbid)
        rows = lb["rows"]
        assert "num_entries" not in lb, f"{lbid}: stale num_entries field"
        assert lb["n_entries"] == len(rows)
        assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
        assert [r["entry_id"] for r in rank_leaderboard_rows(rows)] == [r["entry_id"] for r in rows]
        assert len({r["entry_id"] for r in rows}) == len(rows), f"{lbid}: duplicate entry_id"
        if rows:
            assert lb["top_model"] == rows[0]["model_name"]
            assert lb["top_nse"] == rows[0]["primary_ranking_nse"]
        for row in rows:
            assert row["leaderboard_id"] == lbid
            resolve_row_store_id(row, lbid)  # valid ids; zarr_path points into this leaderboard


@pytest.mark.unit
def test_partial_coverage_rows_are_starred_and_unverifiable_claims_are_null() -> None:
    """Partial basin coverage implies a starred category; rows without a store claim no NaN count."""
    for lbid, row in _repo_rows():
        tag = f"{lbid}/{row['entry_id']}"
        n_basins = row.get("num_evaluated_basins")
        pred_nan = row.get("pred_nan_when_obs_valid")
        partial = (isinstance(n_basins, int) and n_basins < 531) or (
            isinstance(pred_nan, int) and pred_nan > 0
        )
        if partial:
            assert row["category"] in _PARTIAL_CATEGORIES, f"{tag}: partial coverage but {row['category']!r}"
            assert row["asterisk"] is True and str(row.get("asterisk_reason") or "").strip(), tag
        if row["category"] in _PARTIAL_CATEGORIES:
            assert row["asterisk"] is True, tag
        if str(row["category"]).startswith("Literature") and not row.get("zarr_path"):
            assert pred_nan is None, f"{tag}: literature row without a hydrograph claims pred_nan={pred_nan}"
        if row["category"] in ("MIP Re-run", "Contribution"):
            assert row.get("zarr_path"), f"{tag}: self-run row without a zarr_path"
            assert pred_nan == 0, f"{tag}: self-run row with pred_nan_when_obs_valid={pred_nan}"


@pytest.mark.unit
def test_registry_counts_and_family_records_match_the_leaderboards() -> None:
    """`table_2_registry.json` counts equal len(families); every family follows one schema."""
    registry = json.loads((DEFAULT_LEADERBOARDS_DIR / "table_2_registry.json").read_text(encoding="utf-8"))
    families = registry["families"]
    for key in ("total_families", "num_families", "n_families"):
        assert registry[key] == len(families), key
    key_sets = {frozenset(f) for f in families}
    assert len(key_sets) == 1, f"families use {len(key_sets)} different key sets"
    rows_by_family: dict[str, list[tuple[str, dict]]] = {}
    for lbid, row in _repo_rows():
        rows_by_family.setdefault(row["family_id"], []).append((lbid, row))
    assert set(rows_by_family) == {f["family_id"] for f in families}
    for fam in families:
        fam_rows = rows_by_family[fam["family_id"]]
        assert fam["n_entries"] == len(fam_rows), fam["family_id"]
        covered = [lb for lb in CANONICAL_LEADERBOARD_IDS if any(lb == x for x, _ in fam_rows)]
        assert fam["leaderboards_covered"] == covered, fam["family_id"]
        assert fam["best_nse"] == max(r["primary_ranking_nse"] for _, r in fam_rows), fam["family_id"]


@pytest.mark.unit
def test_stage1_appendix_renderings_match_the_json_source() -> None:
    """The appendix `.md` and `.csv` are exact renderings of the validated `.json` source."""
    base = DEFAULT_LEADERBOARDS_DIR / "appendix_stage1_confirmation_runs"
    payload = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    validate_stage1_appendix(payload)
    assert base.with_suffix(".md").read_text(encoding="utf-8") == render_stage1_appendix_markdown(payload)
    assert base.with_suffix(".csv").read_text(encoding="utf-8") == render_stage1_appendix_csv(payload)
    families = {row["family_id"] for row in payload["rows"]}
    assert "shahriar2026_mamba3" in families

    bad = json.loads(json.dumps(payload))
    kratzert = next(r for r in bad["rows"] if r["model_dir"] == "models/kratzert_lstm")
    kratzert["status"] = "PASS"
    with pytest.raises(ValueError, match="implied by its tolerance"):
        validate_stage1_appendix(bad)


@pytest.mark.unit
def test_resolve_row_store_id_uses_the_explicit_store_without_fallbacks() -> None:
    """A row is scored from its `zarr_path` store (or its `entry_id`), never a guessed sibling."""
    lbid = "gauged__single_camels"
    assert resolve_row_store_id({"entry_id": "t3_02"}, lbid) == "t3_02"
    assert (
        resolve_row_store_id(
            {"entry_id": "t3_19", "zarr_path": f"gs://camels-us-mip/hydrographs/{lbid}/t3_18.zarr"}, lbid
        )
        == "t3_18"
    )
    with pytest.raises(ValueError, match="expected"):
        resolve_row_store_id(
            {"entry_id": "x", "zarr_path": "gs://camels-us-mip/hydrographs/pub__single_camels/x.zarr"}, lbid
        )
    with pytest.raises(ValueError, match="not a valid identifier"):
        resolve_row_store_id({"entry_id": "../observations/newman_eval_targets"}, lbid)


@pytest.mark.unit
def test_recalculation_fails_fast_when_a_self_run_store_is_missing(tmp_path: Path) -> None:
    """Missing stores are reported together before any scoring (no silent store substitution)."""
    lb_dir = tmp_path / "leaderboards"
    lb_dir.mkdir()
    for lbid in CANONICAL_LEADERBOARD_IDS:
        src = json.loads((DEFAULT_LEADERBOARDS_DIR / f"{lbid}.json").read_text(encoding="utf-8"))
        src["rows"] = []
        src["n_entries"] = 0
        (lb_dir / f"{lbid}.json").write_text(json.dumps(src), encoding="utf-8")
    lb_path = lb_dir / "pur__freeform.json"
    data = json.loads(lb_path.read_text(encoding="utf-8"))
    data["rows"] = [
        {"entry_id": "missing_model_a", "model_name": "A", "category": "MIP Re-run"},
        {"entry_id": "missing_model_b_single", "model_name": "B", "category": "Contribution",
         "is_ensemble": False, "num_members": 3},
    ]
    lb_path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "bucket" / "hydrographs" / "pur__freeform" / "missing_model_b.zarr").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        recalculate_all_self_run_models(
            bucket_uri=str(tmp_path / "bucket"), leaderboards_dir=lb_dir, update_figures=False
        )
    message = str(excinfo.value)
    assert "2 self-run leaderboard row(s) have no Zarr store" in message
    assert "missing_model_a.zarr" in message and "missing_model_b_single.zarr" in message


@pytest.mark.unit
def test_clean_model_name_never_invents_member_counts() -> None:
    """The δHBV display name only states a member count that the row actually carries."""
    assert clean_model_name("Feng-dHBV", is_ensemble=True, num_members=5) == "δHBV 1.0 (5-model ensemble)"
    assert clean_model_name("Feng-dHBV", is_ensemble=True, num_members=None) == "δHBV 1.0 (ensemble)"
    assert clean_model_name("Feng-dHBV", is_ensemble=False) == "δHBV 1.0"


@pytest.mark.unit
def test_hydrograph_manifest_lists_exactly_the_stores_backing_leaderboard_rows() -> None:
    """The bucket manifest is a deterministic function of the rows' `zarr_path` values."""
    stamp = "2026-01-01T00:00:00Z"
    manifest = build_hydrograph_manifest(updated_utc=stamp)
    assert manifest == build_hydrograph_manifest(updated_utc=stamp)
    assert manifest["bucket"] == "gs://camels-us-mip"
    uris = [store["zarr_uri"] for store in manifest["stores"]]
    assert uris == sorted(uris) and len(uris) == len(set(uris))
    assert manifest["total_hydrograph_stores"] == len(uris)

    expected_rows: set[tuple[str, str, str]] = set()
    for lb in load_all_leaderboards()["leaderboards"]:
        for row in lb["rows"]:
            if str(row.get("zarr_path") or "").strip():
                expected_rows.add((lb["id"], row["entry_id"], row["zarr_path"]))
    listed_rows = {
        (ref["leaderboard_id"], ref["entry_id"], store["zarr_uri"])
        for store in manifest["stores"]
        for ref in store["referenced_by_rows"]
    }
    assert listed_rows == expected_rows
    assert manifest["total_zarr_backed_leaderboard_rows"] == len(expected_rows)
    for store in manifest["stores"]:
        assert store["zarr_uri"] == (
            f"gs://camels-us-mip/hydrographs/{store['leaderboard_id']}/{store['store_name']}"
        )
        assert all(ref["leaderboard_id"] == store["leaderboard_id"] for ref in store["referenced_by_rows"])

    # The CC-BY-NC Frame et al. (2023) archive rows have no store and are never listed.
    assert not any(store["store_name"] in {"t3_12.zarr", "t3_28.zarr"} for store in manifest["stores"])
    for regime in ("gauged", "pub", "pur"):
        assert (
            f"gs://camels-us-mip/hydrographs/{regime}__lagged_streamflow/naive_lag1_persistence_{regime}.zarr"
            in uris
        )


@pytest.mark.unit
def test_hydrograph_manifest_rejects_stores_outside_the_bucket(tmp_path: Path) -> None:
    """A row whose `zarr_path` points outside `<bucket>/hydrographs/<leaderboard_id>/` is an error."""
    lb_dir = tmp_path / "leaderboards"
    lb_dir.mkdir()
    for lbid in CANONICAL_LEADERBOARD_IDS:
        shutil.copy2(DEFAULT_LEADERBOARDS_DIR / f"{lbid}.json", lb_dir / f"{lbid}.json")
    lb_path = lb_dir / "gauged__single_camels.json"
    data = json.loads(lb_path.read_text(encoding="utf-8"))
    row = next(r for r in data["rows"] if str(r.get("zarr_path") or "").strip())
    row["zarr_path"] = f"gs://another-bucket/hydrographs/gauged__single_camels/{row['entry_id']}.zarr"
    lb_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 'gs://camels-us-mip/hydrographs/gauged__single_camels/"):
        build_hydrograph_manifest(leaderboards_dir=lb_dir, updated_utc="2026-01-01T00:00:00Z")

    out = tmp_path / "manifest_index.json"
    summary = write_hydrograph_manifest(out, updated_utc="2026-01-01T00:00:00Z")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == build_hydrograph_manifest(updated_utc="2026-01-01T00:00:00Z")
    assert summary["total_hydrograph_stores"] == written["total_hydrograph_stores"]


@pytest.mark.unit
def test_every_row_reports_the_14_canonical_metrics_in_canonical_order() -> None:
    """Store-backed rows carry all 14 Gauch et al. (2023) summaries (incl. logNSE / logKGE).

    Rows without a hydrograph store keep only the metrics their source reports, plus explicit
    `null` log-space metrics, always in the canonical metric order.
    """
    from metrics.evaluator import GAUCH_2023_METRICS

    for lb in load_all_leaderboards()["leaderboards"]:
        for row in lb["rows"]:
            g = row["gauch_2023_metrics"]
            where = (lb["id"], row["entry_id"])
            assert list(g) == [m for m in GAUCH_2023_METRICS if m in g], where
            assert "logNSE" in g and "logKGE" in g, where
            if str(row.get("zarr_path") or "").strip():
                assert list(g) == list(GAUCH_2023_METRICS), where
                for metric in GAUCH_2023_METRICS:
                    assert isinstance(g[metric], dict), (where, metric)
                    assert isinstance(g[metric]["median"], float | int), (where, metric)

    contributions = json.loads(
        (DEFAULT_LEADERBOARDS_DIR / "contributions_index.json").read_text(encoding="utf-8")
    )
    for entry_id, entry in contributions["entries"].items():
        g = entry.get("gauch_2023_metrics") or {}
        if g:
            assert list(g) == list(GAUCH_2023_METRICS), entry_id


@pytest.mark.unit
def test_ingesting_a_row_sets_code_provenance_and_refreshes_registry_counts(tmp_path: Path) -> None:
    """`upsert_leaderboard_entry` keeps table_2_registry.json consistent for ci_guardrails."""
    from leaderboards import default_code_provenance, refresh_registry_counts

    lb_dir = tmp_path / "leaderboards"
    shutil.copytree(DEFAULT_LEADERBOARDS_DIR, lb_dir)
    assert refresh_registry_counts(lb_dir) is False  # the committed registry is consistent

    registry = json.loads((lb_dir / "table_2_registry.json").read_text(encoding="utf-8"))
    fam = next(f for f in registry["families"] if f["family_id"] == "naive_lag1_persistence")
    before = fam["n_entries"]
    row = upsert_leaderboard_entry(
        "gauged__single_camels",
        {
            "entry_id": "naive_lag1_persistence_extra_test",
            "model_name": "Naive persistence (test row)",
            "family_id": "naive_lag1_persistence",
            "category": "Contribution",
            "primary_ranking_nse": 0.1,
        },
        leaderboards_dir=lb_dir,
        update_figures=False,
    )
    assert row["code_provenance"] == "Our Code"
    registry = json.loads((lb_dir / "table_2_registry.json").read_text(encoding="utf-8"))
    fam = next(f for f in registry["families"] if f["family_id"] == "naive_lag1_persistence")
    assert fam["n_entries"] == before + 1
    assert "gauged__single_camels" in fam["leaderboards_covered"]
    for key in ("total_families", "num_families", "n_families"):
        assert registry[key] == len(registry["families"])

    assert default_code_provenance({"category": "Archive*"}) == "Author Archive"
    assert default_code_provenance({"category": "Literature", "code_provenance": "X"}) == "X"
    with pytest.raises(ValueError, match="code_provenance"):
        default_code_provenance({"entry_id": "e", "category": "Mystery"})
