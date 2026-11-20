"""Unit and integration tests for dynamic manuscript figure generation (`leaderboards/generate_figures.py`)."""

from __future__ import annotations

import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from leaderboards import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_FIGURES_DIR,
    DEFAULT_LEADERBOARDS_DIR,
    classify_model_category,
    extract_publication_year,
    generate_all_paper_figures,
    generate_figure_1_water_year_coverage,
    upsert_leaderboard_entry,
)
from leaderboards import generate_figures as gf

SPATIAL_REGIMES = ("gauged", "pub", "pur")


def _copy_leaderboards(dst: Path) -> Path:
    """Copy the committed leaderboard JSON files into `dst` and return it."""
    dst.mkdir(parents=True, exist_ok=True)
    for lbid in CANONICAL_LEADERBOARD_IDS:
        shutil.copy2(DEFAULT_LEADERBOARDS_DIR / f"{lbid}.json", dst / f"{lbid}.json")
    shutil.copy2(DEFAULT_LEADERBOARDS_DIR / "tables_3_to_14.json", dst / "tables_3_to_14.json")
    return dst


def _is_finite_nse(value: Any) -> bool:
    """Independent (test-side) definition of a plottable NSE: finite number above the -999 sentinel."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > -900.0
    )


def _rows_by_spatial_regime(lb_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read every row of every per-leaderboard JSON file in `lb_dir`, grouped by spatial regime."""
    rows: dict[str, list[dict[str, Any]]] = {sp: [] for sp in SPATIAL_REGIMES}
    for lbid in CANONICAL_LEADERBOARD_IDS:
        spatial = lbid.split("__", 1)[0]
        data = json.loads((lb_dir / f"{lbid}.json").read_text(encoding="utf-8"))
        rows[spatial].extend(data.get("rows", []))
    return rows


def _row_is_plottable(row: dict[str, Any]) -> bool:
    return any(
        _is_finite_nse(row.get(key))
        for key in ("primary_ranking_nse", "ens_med_nse", "single_med_nse")
    )


@pytest.fixture(scope="module")
def generated_figures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Generate all figures once, into a temp dir, from a copy of the committed leaderboards."""
    root = tmp_path_factory.mktemp("mip_figures")
    lb_dir = _copy_leaderboards(root / "leaderboards")
    fig_dir = root / "figures"
    manifest = generate_all_paper_figures(leaderboards_dir=lb_dir, output_dir=fig_dir)
    return {"lb_dir": lb_dir, "fig_dir": fig_dir, "manifest": manifest}


@pytest.mark.unit
def test_model_category_and_publication_year_classifiers() -> None:
    """Verify deterministic classification of model category and publication year from JSON rows."""
    assert (
        classify_model_category(
            {
                "family_id": "kratzert_ealstm",
                "paradigm": "Entity-Aware Recurrent",
                "model_name": "Kratzert et al. (2019a) EA-LSTM (8-model ensemble)",
            }
        )
        == "Machine learning"
    )
    assert (
        classify_model_category(
            {
                "family_id": "song_dhbv11p",
                "paradigm": "Differentiable Hybrid Conceptual",
                "model_name": "δHBV 1.1p (4-model ensemble)",
            }
        )
        == "Hybrid (ML + Physics)"
    )
    assert (
        classify_model_category(
            {
                "family_id": "newman_sacsma",
                "paradigm": "Conceptual Operational",
                "model_name": "Newman et al. (2015, 2017) SAC-SMA",
            }
        )
        == "Conceptual"
    )
    assert (
        classify_model_category(
            {
                "family_id": "newman_vic",
                "paradigm": "Distributed Land-Surface Hydrology Model",
                "model_name": "VIC 4.1.2 (Basin-Calibrated)*",
            }
        )
        == "Process-based"
    )

    assert (
        extract_publication_year(
            {
                "model_name": "Kratzert et al. (2019a) LSTM (8-model ensemble)",
                "citation": "Kratzert et al. (2019a, 2019b, 2021)",
            }
        )
        == 2019
    )
    assert (
        extract_publication_year(
            {
                "model_name": "Community HydroLSTM (8-model ensemble)",
                "citation": "This Study (§3.2)",
            }
        )
        == 2026
    )


@pytest.mark.unit
def test_extract_publication_year_literature_vs_mip_rerun_rows() -> None:
    """'Literature'/'Archive*'/'MIP Re-run' rows use the original paper year; 'Contribution' rows use the MIP year."""
    literature_row = {
        "entry_id": "lit",
        "model_name": "δHBV 1.0 (4-model ensemble)",
        "citation": "Feng et al. (2022, 2023)",
        "category": "Literature",
        "publication_year": 2022,
    }
    assert extract_publication_year(literature_row) == 2022

    # 'MIP Re-run' entries of a Series-1 paper are placed at the original paper's publication year.
    rerun_row = {**literature_row, "category": "MIP Re-run", "citation": "Feng et al. (2022)"}
    assert extract_publication_year(rerun_row) == 2022
    for spelling in ("MIP Re-run", "MIP-Rerun", "MIP rerun"):
        assert extract_publication_year({**rerun_row, "category": spelling}) == 2022, spelling

    # Explicit `publication_year` on a 'MIP Re-run' row gives the original Series-1 paper year.
    assert extract_publication_year({**rerun_row, "publication_year": 2024}) == 2024
    # Even if a 'MIP Re-run' row also carries a 2026 `submitted_at` timestamp, the original paper year wins.
    assert (
        extract_publication_year(
            {**rerun_row, "publication_year": 2019, "submitted_at": "2026-04-12T06:22:17Z"}
        )
        == 2019
    )

    # 'Contribution' entries without a prior Series-1 paper use the MIP year (2026+).
    contrib_row = {
        "entry_id": "baseline_contrib",
        "model_name": "GRU (Daymet, 1-model)",
        "citation": "This Study (2026)",
        "category": "Contribution",
    }
    for spelling in ("Contribution", "Contributed", "community", "Community Submission"):
        assert extract_publication_year({**contrib_row, "category": spelling}) == 2026, spelling
    assert extract_publication_year({**contrib_row, "publication_year": 2027}) == 2027
    assert extract_publication_year({**contrib_row, "publication_year": 2019}) == 2026

    # User submissions are dated by `submitted_at` and never raise, even without any citation year.
    submission = {
        "entry_id": "sub",
        "model_name": "Community LSTM",
        "citation": "Community submission",
        "category": "Contribution",
        "submitted_at": "2027-03-04T10:00:00Z",
    }
    assert extract_publication_year(submission) == 2027
    assert extract_publication_year({**submission, "category": None, "is_user_submission": True}) == 2027
    assert extract_publication_year({k: v for k, v in submission.items() if k != "submitted_at"}) == 2026

    # Literature rows: explicit field, then model_name, then citation; 1990–2035 window.
    assert extract_publication_year({"model_name": "Old model", "citation": "Nash & Sutcliffe (1995)", "category": "Literature"}) == 1995
    assert extract_publication_year({"model_name": "X", "citation": "Y", "category": "Archive*", "publication_year": 2023}) == 2023
    assert extract_publication_year({"model_name": "X", "citation": "Y", "category": "Archive", "publication_year": "2023"}) == 2023
    assert extract_publication_year({"model_name": "Z (1985) legacy", "citation": "Z et al. (2001)", "category": "Literature"}) == 2001
    assert extract_publication_year({"model_name": "A", "citation": "B (2003)", "category": "Literature", "publication_year": True}) == 2003

    # 'Literature'/'Archive*' rows are literature/archive results even if the citation mentions this study.
    assert extract_publication_year({"model_name": "A", "citation": "B et al. (2018), This Study", "category": "Literature"}) == 2018


@pytest.mark.unit
def test_extract_publication_year_raises_for_literature_rows_without_year() -> None:
    """No silent default year: an undatable literature / uncategorized row raises, naming the entry."""
    with pytest.raises(ValueError, match="t_mystery"):
        extract_publication_year(
            {"entry_id": "t_mystery", "model_name": "Mystery model", "citation": "Someone", "category": "Literature"}
        )
    with pytest.raises(ValueError, match="t_nocat"):
        extract_publication_year({"entry_id": "t_nocat", "model_name": "Mystery model", "citation": ""})
    # A year outside the accepted 1990–2035 window does not count.
    with pytest.raises(ValueError, match="t_old"):
        extract_publication_year(
            {"entry_id": "t_old", "model_name": "Ancient (1970)", "citation": "A (1970)", "category": "Archive*"}
        )
    # "this study" rows are MIP results and never raise.
    assert extract_publication_year({"entry_id": "t_ts", "model_name": "New", "citation": "This Study"}) == 2026


@pytest.mark.unit
def test_format_leader_label_never_crashes_and_marks_mip_reruns() -> None:
    """Labels survive a missing NSE and attribute MIP re-runs to the MIP, not the cited paper."""
    rerun = {
        "model_name": "δHBV 1.1p (4-model ensemble)",
        "citation": "Song et al. (2024)",
        "category": "MIP Re-run",
        "primary_ranking_nse": None,
    }
    label_none = gf.format_leader_label(rerun, year=2024)
    assert label_none == "δHBV 1.1p Ens\n(MIP re-run of Song 2024: n/a)"
    assert gf.format_leader_label({**rerun, "primary_ranking_nse": 0.714}, 2024).endswith("(MIP re-run of Song 2024: 0.714)")
    assert gf.format_leader_label({**rerun, "primary_ranking_nse": float("nan")}, 2024).endswith(": n/a)")
    assert gf.format_leader_label({**rerun, "primary_ranking_nse": -999.0}, 2024).endswith(": n/a)")
    assert gf.format_leader_label({**rerun, "primary_ranking_nse": "bogus"}, 2024).endswith(": n/a)")

    literature = {
        "model_name": "Kratzert et al. (2019a) LSTM (8-model ensemble)",
        "citation": "Kratzert et al. (2019a)",
        "category": "Archive",
        "primary_ranking_nse": 0.759,
    }
    assert gf.format_leader_label(literature, 2019) == "LSTM Ens\n(Kratzert 2019a: 0.759)"

    starred = {
        "model_name": "Towler et al. (2023) NOAA NWM v2.1* (Regionalized)",
        "citation": "Towler et al. (2023)",
        "category": "Archive*",
        "asterisk": True,
        "primary_ranking_nse": 0.591,
    }
    assert gf.format_leader_label(starred, 2023) == "NOAA NWM v2.1*\n(Towler 2023: 0.591*)"

    # A name consisting only of "<Name> (<year>)" keeps its name instead of an empty first line.
    community = {
        "model_name": "Community Frontier Model (2026)",
        "citation": "Community et al. (2026)",
        "category": "Contribution",
        "primary_ranking_nse": 0.845,
    }
    assert gf.format_leader_label(community, 2026) == "Community Frontier Model\n(Community 2026: 0.845)"


@pytest.mark.unit
def test_resolve_primary_nse_excludes_null_nonfinite_and_sentinel_values() -> None:
    """Only finite, non-sentinel NSE values are plottable; everything else carries a reason."""
    assert gf.resolve_primary_nse({"primary_ranking_nse": 0.29}) == (0.29, None)
    assert gf.resolve_primary_nse({"primary_ranking_nse": -0.3}) == (-0.3, None)
    assert gf.resolve_primary_nse({"primary_ranking_nse": None, "ens_med_nse": 0.7}) == (0.7, None)
    assert gf.resolve_primary_nse({"primary_ranking_nse": -999.0, "single_med_nse": 0.6}) == (0.6, None)

    for row in (
        {},
        {"primary_ranking_nse": None, "ens_med_nse": None, "single_med_nse": None},
        {"primary_ranking_nse": -999.0, "ens_med_nse": None, "single_med_nse": None},
        {"primary_ranking_nse": float("nan")},
        {"primary_ranking_nse": float("inf")},
        {"primary_ranking_nse": "N/A"},
        {"primary_ranking_nse": True},
    ):
        nse, reason = gf.resolve_primary_nse(row)
        assert nse is None, row
        assert isinstance(reason, str) and reason, row
    assert "sentinel" in gf.resolve_primary_nse({"primary_ranking_nse": -999.0})[1]


@pytest.mark.unit
def test_compute_water_year_completeness_on_synthetic_array(tmp_path: Path) -> None:
    """The Figure-1 series are recomputed exactly from a (n_basins, n_days) array + datetime64 dates."""
    # 1980-10-01 .. 1982-12-31: WY1981 (365 d), WY1982 (365 d) and a partial WY1983 (92 d).
    dates = np.arange(
        np.datetime64("1980-10-01"), np.datetime64("1983-01-01"), dtype="datetime64[D]"
    )
    assert dates.shape[0] == 365 + 365 + 92
    months = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    wy = np.where(months >= 10, years + 1, years)

    q = np.full((3, dates.shape[0]), 1.5, dtype=np.float32)
    q[1, 0] = np.nan  # basin 1: a single missing day in WY1981
    q[2, wy == 1981] = np.nan  # basin 2: nothing in WY1981 ...
    idx_1982 = np.flatnonzero(wy == 1982)
    q[2, idx_1982[:100]] = np.nan  # ... and the first 100 days of WY1982 missing
    q[0, idx_1982[5]] = np.inf  # non-finite counts as missing

    res = gf.compute_water_year_completeness(q, dates)
    assert res["n_basins"] == 3
    assert res["water_years"] == [1981, 1982]
    assert res["partial_water_years_excluded"] == [1983]
    assert res["basins_with_100pct_complete"] == [1, 1]
    assert res["basins_with_any_valid_data"] == [2, 3]
    assert res["mean_daily_coverage_fraction"] == pytest.approx(
        [(1.0 + 364 / 365 + 0.0) / 3, (364 / 365 + 1.0 + 265 / 365) / 3]
    )
    assert res["newman_eval_water_years"] == [1990, 1999]
    assert res["newman_train_water_years"] == [2000, 2008]
    assert json.loads(json.dumps(res)) == res  # JSON-serializable plain Python values

    # datetime64[ns] dates (as stored in newman_eval_targets.zarr `date_all`) give identical results.
    res_ns = gf.compute_water_year_completeness(q, dates.astype("datetime64[ns]"))
    assert res_ns == res

    # Partial water years can be kept explicitly (statistics over the days present).
    res_partial = gf.compute_water_year_completeness(q, dates, include_partial_water_years=True)
    assert res_partial["water_years"] == [1981, 1982, 1983]
    assert res_partial["partial_water_years_excluded"] == []
    assert res_partial["basins_with_100pct_complete"] == [1, 1, 3]
    assert res_partial["basins_with_any_valid_data"] == [2, 3, 3]
    assert res_partial["mean_daily_coverage_fraction"][2] == pytest.approx(1.0)

    # Leap-year water years span 366 days (WY1984 contains 29 Feb 1984) and count as complete.
    leap_dates = np.arange(
        np.datetime64("1983-10-01"), np.datetime64("1984-10-01"), dtype="datetime64[D]"
    )
    assert leap_dates.shape[0] == 366
    res_leap = gf.compute_water_year_completeness(np.ones((2, 366)), leap_dates)
    assert res_leap["water_years"] == [1984]
    assert res_leap["basins_with_100pct_complete"] == [2]
    assert res_leap["partial_water_years_excluded"] == []

    # The computed block validates and is sufficient to draw Figure 1.
    assert gf.validate_water_year_completeness(res)["water_years"] == [1981, 1982]
    out = generate_figure_1_water_year_coverage({"water_year_completeness": res}, tmp_path)
    assert out["png"].is_file() and out["pdf"].is_file()

    with pytest.raises(ValueError, match="n_days"):
        gf.compute_water_year_completeness(q[:, :-1], dates)
    with pytest.raises(ValueError, match="2-D"):
        gf.compute_water_year_completeness(q[0], dates)
    with pytest.raises(ValueError, match="duplicate"):
        gf.compute_water_year_completeness(q[:, :2], np.array([dates[0], dates[0]]))
    with pytest.raises(ValueError, match="datetime64"):
        gf.compute_water_year_completeness(q[:, :2], np.array(["a", "b"]))


@pytest.mark.unit
def test_hard_coded_water_year_series_removed_and_missing_protocol_block_raises(tmp_path: Path) -> None:
    """Figure 1 comes only from `protocol.water_year_completeness`; no built-in fallback exists."""
    source = inspect.getsource(gf)
    assert "DEFAULT_WATER_YEAR_COMPLETENESS" not in source
    # The old hard-coded series values must not survive anywhere in the module.
    assert "[476, 478, 486, 496, 500" not in source
    assert "0.8970" not in source

    with pytest.raises(ValueError, match="water_year_completeness"):
        generate_figure_1_water_year_coverage({}, tmp_path)
    with pytest.raises(ValueError, match="water_year_completeness"):
        generate_figure_1_water_year_coverage({"n_basins": 531}, tmp_path)
    with pytest.raises(ValueError, match="dict"):
        generate_figure_1_water_year_coverage(None, tmp_path)  # type: ignore[arg-type]

    good = {
        "n_basins": 4,
        "water_years": [2001, 2002, 2003],
        "basins_with_100pct_complete": [4, 3, 2],
        "basins_with_any_valid_data": [4, 4, 3],
        "mean_daily_coverage_fraction": [1.0, 0.9, 0.7],
    }
    assert gf.validate_water_year_completeness(good)["n_basins"] == 4
    for broken, pattern in (
        ({**good, "water_years": [2001, 2002]}, "aligned"),
        ({k: v for k, v in good.items() if k != "mean_daily_coverage_fraction"}, "missing required"),
        ({**good, "mean_daily_coverage_fraction": [1.0, 0.9, 1.7]}, r"\[0, 1\]"),
        ({**good, "basins_with_100pct_complete": [4, 3, 9]}, "basin counts"),
        ({**good, "basins_with_any_valid_data": [4, float("nan"), 3]}, "finite"),
        ({**good, "water_years": [2003, 2002, 2001]}, "increasing"),
        ({**good, "n_basins": 0}, "positive integer"),
        ({**good, "newman_eval_water_years": [1999]}, "newman_eval_water_years"),
        ("not-a-dict", "JSON object"),
    ):
        with pytest.raises(ValueError, match=pattern):
            gf.validate_water_year_completeness(broken)

    # End-to-end: a leaderboards dir whose protocol block lacks the key fails loudly.
    lb_dir = _copy_leaderboards(tmp_path / "leaderboards")
    for path in [lb_dir / f"{lbid}.json" for lbid in CANONICAL_LEADERBOARD_IDS] + [
        lb_dir / "tables_3_to_14.json"
    ]:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.get("protocol", {}).pop("water_year_completeness", None)
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="water_year_completeness"):
        generate_all_paper_figures(leaderboards_dir=lb_dir, output_dir=tmp_path / "figures")


@pytest.mark.unit
def test_figure_1_is_drawn_from_the_committed_protocol_block(tmp_path: Path) -> None:
    """The committed protocol block is valid and sufficient to regenerate Figure 1."""
    from leaderboards import load_all_leaderboards

    protocol = load_all_leaderboards(DEFAULT_LEADERBOARDS_DIR)["protocol"]
    block = gf.validate_water_year_completeness(protocol["water_year_completeness"])
    assert block["n_basins"] == 531
    assert len(block["water_years"]) == len(block["mean_daily_coverage_fraction"]) >= 10
    out = generate_figure_1_water_year_coverage(protocol, tmp_path)
    assert out["png"].stat().st_size > 10_000
    assert out["pdf"].stat().st_size > 1_000


@pytest.mark.unit
def test_subpanel_draws_every_point_and_labels_leaders_with_unknown_years() -> None:
    """Axis limits follow the data (no cropping) and labels do not depend on INITIAL_LABEL_OFFSETS."""

    def _pt(eid: str, year: int, nse: float, provenance: str, regime: str = "Single CAMELS Forcing") -> dict[str, Any]:
        return {
            "entry_id": eid,
            "leaderboard_id": "gauged__single_camels",
            "year": year,
            "nse": nse,
            "model_category": "Machine learning",
            "input_regime": regime,
            "label": f"{eid}\n(x {year}: {nse:.3f})",
            "model_name": eid,
            "exclude_frontier": False,
            "category": "MIP Re-run" if provenance == "mip_rerun" else "Literature",
            "provenance": provenance,
        }

    models = [
        _pt("a", 2016, 0.05, "literature"),  # far below the 0.42 hint floor; unknown offset key
        _pt("b", 2020, 0.40, "literature"),
        _pt("c", 2020, 0.30, "literature"),  # non-leader
        _pt("d", 2031, 0.95, "mip_rerun"),  # above the 0.91 hint ceiling; unknown offset key
        _pt("e", 2016, -0.20, "mip_rerun", regime="Freeform Forcings"),  # negative NSE
    ]
    assert not any((0, m["input_regime"], m["year"]) in gf.INITIAL_LABEL_OFFSETS for m in models)

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=72)
    res = gf._plot_subpanel(
        fig, ax, panel_idx=0, models=models, title="synthetic",
        ylim_hint=gf.PANEL_YLIM_HINTS["gauged"], year_range=(2016, 2031),
    )
    plt.close(fig)

    assert res["n_drawn"] == len(models)
    assert [ldr["entry_id"] for ldr in res["leaders"]] == ["a", "b", "d", "e"]
    assert len(res["annotations"]) == len(res["leaders"])
    assert res["ylim"][0] <= -0.20 - gf.FIGURE_2_YLIM_PAD + 1e-12
    assert res["ylim"][1] >= 0.95 + gf.FIGURE_2_YLIM_PAD - 1e-12
    assert res["xlim"][0] < 2016 and res["xlim"][1] > 2031

    # Hint-only behaviour when the data fit inside the hint, and empty panels fall back to the hint.
    assert gf.compute_panel_ylim([0.5, 0.8], (0.42, 0.91)) == (0.42, 0.91)
    assert gf.compute_panel_ylim([], (0.42, 0.91)) == (0.42, 0.91)
    assert gf.compute_panel_ylim([0.29, 0.8], (0.42, 0.91)) == pytest.approx((0.27, 0.91))
    assert gf.compute_year_axis((2015, 2026)) == ((2014.4, 2026.8), list(range(2015, 2027)), 2026.35)


@pytest.mark.integration
def test_generated_manifest_counts_match_finite_nse_rows(generated_figures: dict[str, Any]) -> None:
    """Every row with a finite NSE is plotted; only non-finite rows are excluded (with a reason)."""
    manifest = generated_figures["manifest"]
    fig_dir: Path = generated_figures["fig_dir"]
    rows_by_regime = _rows_by_spatial_regime(generated_figures["lb_dir"])

    for fig_path in manifest["figures"].values():
        assert Path(fig_path).is_file() and Path(fig_path).stat().st_size > 10_000, fig_path
    assert json.loads((fig_dir / "figures_manifest.json").read_text(encoding="utf-8")) == manifest

    for spatial in SPATIAL_REGIMES:
        rows = rows_by_regime[spatial]
        plottable = [r for r in rows if _row_is_plottable(r)]
        unplottable_ids = {str(r["entry_id"]) for r in rows if not _row_is_plottable(r)}

        assert manifest["panel_entry_counts"][spatial] == len(plottable), spatial
        excluded = manifest["excluded_rows"][spatial]
        assert {e["entry_id"] for e in excluded} == unplottable_ids, spatial
        assert len(excluded) == len(unplottable_ids)
        for e in excluded:
            assert e["reason"] and e["leaderboard_id"].startswith(f"{spatial}__"), e
        assert manifest["panel_entry_counts"][spatial] + len(excluded) == len(rows)

        # No finite point can be cropped: all plotted NSEs / years lie inside the panel limits.
        y_lo, y_hi = manifest["panel_axis_limits"][spatial]["ylim"]
        x_lo, x_hi = manifest["panel_axis_limits"][spatial]["xlim"]
        for r in plottable:
            nse, reason = gf.resolve_primary_nse(r)
            assert reason is None and nse is not None
            assert y_lo < nse < y_hi, (spatial, r["entry_id"], nse, (y_lo, y_hi))
            assert x_lo < extract_publication_year(r) < x_hi, (spatial, r["entry_id"])

        # 'Contribution' results are never attributed to an earlier year; 'MIP Re-run' entries
        # are placed at the original publication year of the Series-1 paper they build from.
        for ldr in manifest["frontier_leaders"][spatial]:
            assert ldr["provenance"] in {"literature", "mip_rerun"}
            if gf.normalize_source_category(ldr) in gf.MIP_CONTRIBUTION_CATEGORIES:
                assert ldr["provenance"] == "mip_rerun" and ldr["year"] >= gf.MIP_STUDY_YEAR, ldr
            elif gf.normalize_source_category(ldr) == "MIP Re-run":
                assert ldr["provenance"] == "mip_rerun", ldr

    # Verify Single CAMELS Forcing leaderboards never contain a Freeform Forcings point,
    # and _build_legend_handles omits Freeform Forcings when not in active_regimes.
    from leaderboards import load_all_leaderboards

    all_lbs = load_all_leaderboards(generated_figures["lb_dir"])
    for spatial in SPATIAL_REGIMES:
        pts, _ = gf.partition_panel_rows(all_lbs, spatial)
        single_pts = [m for m in pts if m["leaderboard_id"] == f"{spatial}__single_camels"]
        assert all(m["input_regime"] == "Single CAMELS Forcing" for m in single_pts)
        assert all("nwm" not in m["model_name"].lower() for m in single_pts)
    _, single_regime_handles = gf._build_legend_handles(["Single CAMELS Forcing"])
    assert [h.get_label() for h in single_regime_handles] == ["Single CAMELS Forcing"]
    _, pur_regime_handles = gf._build_legend_handles(
        ["Single CAMELS Forcing", "Multiple CAMELS Forcings"]
    )
    assert "Freeform Forcings" not in [h.get_label() for h in pur_regime_handles]

    assert manifest["total_leaderboard_entries"] == sum(len(v) for v in rows_by_regime.values())
    assert manifest["mip_study_year"] == gf.MIP_STUDY_YEAR
    assert manifest["figure_1_source"] == "protocol.water_year_completeness"


@pytest.mark.unit
def test_repository_figures_exist_and_match_manifest() -> None:
    """Verify all generated paper figures exist in `figures/` and match `figures_manifest.json`."""
    manifest_path = DEFAULT_FIGURES_DIR / "figures_manifest.json"
    assert manifest_path.is_file(), f"Missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for fig_key, fig_path_str in manifest["figures"].items():
        p = Path(fig_path_str)
        assert p.is_file(), f"Missing generated figure {fig_key}: {p}"
        assert p.stat().st_size > 10_000, f"Generated figure {p} is unexpectedly small"


@pytest.mark.integration
def test_upsert_leaderboard_entry_automatically_updates_figures(tmp_path: Path) -> None:
    """Verify adding a new community model to a leaderboard JSON automatically updates the figures."""
    lb_dir = _copy_leaderboards(tmp_path / "leaderboards")
    fig_dir = tmp_path / "figures"

    upsert_leaderboard_entry(
        leaderboard_id="gauged__single_camels",
        entry={
            "entry_id": "community_new_leader_2026",
            "model_name": "Community Frontier Model (2026)",
            "family_id": "kratzert_lstm",
            "paradigm": "Deep Recurrent Network",
            "category": "Contribution",
            "forcing": "Daymet",
            "primary_ranking_nse": 0.845,
            "ens_med_nse": 0.845,
            "single_med_nse": None,
            "citation": "Community et al. (2026)",
        },
        leaderboards_dir=lb_dir,
        update_figures=True,
    )

    manifest_path = fig_dir / "figures_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (fig_dir / "figure_1_water_year_coverage.png").is_file()
    assert (fig_dir / "figure_2_leaderboard_progress_all_regimes.png").is_file()

    gauged_leaders = manifest["frontier_leaders"]["gauged"]
    single_2026 = [
        ldr
        for ldr in gauged_leaders
        if ldr["input_regime"] == "Single CAMELS Forcing" and ldr["year"] == 2026
    ]
    assert len(single_2026) == 1
    assert single_2026[0]["entry_id"] == "community_new_leader_2026"
    assert single_2026[0]["nse"] == pytest.approx(0.845)
    assert single_2026[0]["provenance"] == "mip_rerun"
