"""Native-resolution (531 x 3652) tests for `benchmarks/metrics_pipeline.py`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import FsspecStore

from benchmarks import (
    discover_hydrograph_zarr_entries,
    format_verification_markdown_table,
    load_manifest_entries,
    run_metrics_benchmark_pipeline,
    verify_single_entry_report,
)
from benchmarks.metrics_pipeline import main as benchmark_cli_main
from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths
from metrics import EVAL_END_DATE, EVAL_START_DATE, NUM_BASINS, NUM_EVAL_DAYS

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = resolve_mip_paths(DEFAULT_MIP_ID).splits_dir


def _load_canonical_basins_and_huc02() -> tuple[np.ndarray, np.ndarray]:
    """Load the 531 canonical USGS basin IDs and 2-digit HUC-02 codes."""
    spec_path = SPLITS_DIR / "splits_and_regimes.json"
    assert spec_path.exists(), f"Missing canonical split specification at {spec_path}"

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    basin_to_huc02: dict[str, str] = spec["basin_to_huc02"]
    basins = np.array(sorted(basin_to_huc02.keys()), dtype="<U8")
    huc_02 = np.array([basin_to_huc02[str(b)] for b in basins], dtype="<U2")
    assert basins.shape == (NUM_BASINS,)
    assert huc_02.shape == (NUM_BASINS,)
    return basins, huc_02


def _canonical_dates() -> np.ndarray:
    """Return the 3,652 daily timestamps over 1989-10-01 to 1999-09-30."""
    return pd.date_range(EVAL_START_DATE, EVAL_END_DATE, freq="D").values.astype(
        "datetime64[ns]"
    )


def _build_synthetic_qobs(basins: np.ndarray) -> np.ndarray:
    """Generate deterministic positive daily streamflow observations of shape (531, 3652)."""
    t = np.linspace(0.0, 20.0 * np.pi, NUM_EVAL_DAYS, dtype=np.float64)
    basin_idx = np.arange(len(basins), dtype=np.float64)[:, None]
    seasonal = 2.0 + 1.5 * np.sin(t[None, :] + 0.05 * basin_idx)
    peaks = np.where((np.arange(NUM_EVAL_DAYS)[None, :] % 45) == 0, 14.0, 0.0)
    scale = 1.0 + 0.002 * basin_idx
    return (scale * (seasonal + peaks)).astype(np.float32)


def _write_targets_and_hydrographs(
    bucket_root: Path,
    entries_with_scales: list[tuple[str, str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write canonical (531, 3652) targets and model hydrograph Zarr stores."""
    basins, huc_02 = _load_canonical_basins_and_huc02()
    dates = _canonical_dates()
    qobs = _build_synthetic_qobs(basins)

    targets_path = bucket_root / "observations" / "newman_eval_targets.zarr"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets_ds = xr.Dataset(
        data_vars={"qobs": (("basin", "date"), qobs)},
        coords={"basin": basins, "date": dates, "huc_02": ("basin", huc_02)},
    )
    targets_ds.to_zarr(targets_path, mode="w", consolidated=True, zarr_format=2)

    for leaderboard_id, model_id, scale in entries_with_scales:
        qsim = (scale * qobs).astype(np.float32)
        store_path = (
            bucket_root / "hydrographs" / leaderboard_id / f"{model_id}.zarr"
        )
        store_path.parent.mkdir(parents=True, exist_ok=True)
        sim_ds = xr.Dataset(
            data_vars={"qsim": (("basin", "date"), qsim)},
            coords={"basin": basins, "date": dates, "huc_02": ("basin", huc_02)},
            attrs={"model_id": model_id, "leaderboard_id": leaderboard_id},
        )
        sim_ds.to_zarr(store_path, mode="w", consolidated=True, zarr_format=2)

    return basins, huc_02, qobs


@pytest.mark.unit
def test_discover_and_run_benchmark_pipeline_across_leaderboards(
    tmp_path: Path,
) -> None:
    """Verify automatic Zarr discovery, leaderboard ranking, and lower-tail reporting."""
    _write_targets_and_hydrographs(
        tmp_path,
        [
            ("gauged__single_camels", "model_b", 1.15),
            ("gauged__single_camels", "model_a", 1.00),
            ("pub__multi_camels", "model_pub_1", 1.10),
        ],
    )

    discovered = discover_hydrograph_zarr_entries(bucket_uri=str(tmp_path))
    assert discovered == [
        ("gauged__single_camels", "model_a"),
        ("gauged__single_camels", "model_b"),
        ("pub__multi_camels", "model_pub_1"),
    ]

    out_json = tmp_path / "benchmark_report.json"
    report = run_metrics_benchmark_pipeline(
        bucket_uri=str(tmp_path),
        output_json=out_json,
    )

    assert out_json.exists()
    assert report["total_evaluated_entries"] == 3
    assert report["total_pred_nan_when_obs_valid"] == 0
    assert report["all_invariants_verified"] is True

    gauged_lb = report["leaderboards"]["gauged__single_camels"]
    assert gauged_lb["num_entries"] == 2
    assert gauged_lb["top_model_id"] == "model_a"
    assert gauged_lb["rows"][0]["rank"] == 1
    assert gauged_lb["rows"][0]["model_id"] == "model_a"
    assert gauged_lb["rows"][0]["median_nse"] == pytest.approx(1.0, abs=1e-4)
    assert gauged_lb["rows"][1]["rank"] == 2
    assert gauged_lb["rows"][1]["model_id"] == "model_b"
    assert gauged_lb["rows"][1]["median_nse"] < 1.0

    md_table = format_verification_markdown_table(report)
    assert "| `gauged__single_camels` | 1 | `model_a` |" in md_table
    assert "| `gauged__single_camels` | 2 | `model_b` |" in md_table
    assert "| `pub__multi_camels` | 1 | `model_pub_1` |" in md_table


@pytest.mark.unit
def test_load_manifest_entries_and_cli_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify manifest parsing (both list and contributions_index dict formats) and CLI."""
    _write_targets_and_hydrographs(
        tmp_path,
        [
            ("gauged__multi_camels", "t4_06", 1.00),
            ("pur__single_camels", "t11_01", 1.05),
        ],
    )

    # 1. Contributions-index dict format
    dict_manifest_path = tmp_path / "manifest_dict.json"
    dict_manifest_path.write_text(
        json.dumps(
            {
                "entries": {
                    "t4_06": {
                        "entry_id": "t4_06",
                        "target_leaderboard_tables": [
                            "table_4",
                            "gauged__multi_camels",
                        ],
                    },
                    "t11_01": {
                        "entry_id": "t11_01",
                        "leaderboard_id": "pur__single_camels",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    parsed = load_manifest_entries(dict_manifest_path)
    assert parsed == [
        ("gauged__multi_camels", "t4_06"),
        ("pur__single_camels", "t11_01"),
    ]

    # 2. Run CLI with --manifest-path and --output-json
    out_report_path = tmp_path / "cli_report.json"
    rc = benchmark_cli_main(
        [
            "--bucket-uri",
            str(tmp_path),
            "--manifest-path",
            str(dict_manifest_path),
            "--output-json",
            str(out_report_path),
        ]
    )
    assert rc == 0
    assert out_report_path.exists()
    saved_report = json.loads(out_report_path.read_text(encoding="utf-8"))
    assert saved_report["total_evaluated_entries"] == 2

    stdout_str = capsys.readouterr().out
    assert "`t4_06`" in stdout_str
    assert "`t11_01`" in stdout_str


@pytest.mark.unit
def test_verify_single_entry_report_rejects_invariant_violations(
    tmp_path: Path,
) -> None:
    """Verify verify_single_entry_report catches anti-masking, basin-count, and ordering violations."""
    _write_targets_and_hydrographs(
        tmp_path,
        [("gauged__single_camels", "m_ok", 1.00)],
    )
    report = run_metrics_benchmark_pipeline(bucket_uri=str(tmp_path))
    valid_entry = dict(report["entries"]["gauged__single_camels/m_ok"])

    # Valid entry passes
    verify_single_entry_report(valid_entry)

    # 1. pred_nan_when_obs_valid > 0 fails
    bad_nan = dict(valid_entry, pred_nan_when_obs_valid=2)
    with pytest.raises(ValueError, match="Anti-masking violation"):
        verify_single_entry_report(bad_nan)

    # 2. Incomplete basin coverage (< 530) fails
    bad_basins = dict(valid_entry, num_evaluated_basins=447)
    with pytest.raises(ValueError, match="Incomplete basin coverage"):
        verify_single_entry_report(bad_basins)

    # 3. Non-monotonic lower-tail NSE percentiles fail
    bad_order = dict(valid_entry, min_nse=0.90, p01_nse=0.50)
    with pytest.raises(
        ValueError, match="Lower-tail NSE percentiles are not monotonically ordered"
    ):
        verify_single_entry_report(bad_order)

    # 4. Missing HUC-02 region fails
    incomplete_huc = dict(valid_entry["regional_huc02"])
    del incomplete_huc["18"]
    bad_huc = dict(valid_entry, regional_huc02=incomplete_huc)
    with pytest.raises(KeyError, match="Missing USGS HUC-02 region '18'"):
        verify_single_entry_report(bad_huc)


@pytest.mark.unit
def test_discover_gcs_entries_and_negative_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify GCS discovery via gcsfs boundary and negative error cases."""
    with pytest.raises(FileNotFoundError):
        discover_hydrograph_zarr_entries(bucket_uri=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid leaderboard_id"):
        discover_hydrograph_zarr_entries(
            bucket_uri=str(tmp_path),
            leaderboard_ids=["not_a_leaderboard"],
        )

    _write_targets_and_hydrographs(
        tmp_path,
        [("pur__lagged_streamflow", "gcs_pur_model", 1.00)],
    )

    class _LocalMappedGCSFileSystem:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def exists(self, uri: str) -> bool:
            prefix = "gs://camels-us-mip/"
            assert uri.startswith(prefix)
            return (tmp_path / uri[len(prefix) :]).exists()

        def ls(self, uri: str, detail: bool = False) -> list[str]:
            del detail
            prefix = "gs://camels-us-mip/"
            assert uri.startswith(prefix)
            local_dir = tmp_path / uri[len(prefix) :]
            return [
                f"camels-us-mip/{uri[len(prefix):]}/{p.name}"
                for p in sorted(local_dir.iterdir())
            ]

    orig_from_url = FsspecStore.from_url

    def _mapped_from_url(
        url: str, storage_options: dict[str, Any] | None = None, **kwargs: Any
    ) -> FsspecStore:
        prefix = "gs://camels-us-mip/"
        if url.startswith(prefix):
            local_url = f"file://{tmp_path}/{url[len(prefix):]}"
            return orig_from_url(local_url, storage_options=None, **kwargs)
        return orig_from_url(url, storage_options=storage_options, **kwargs)

    monkeypatch.setattr(gcsfs, "GCSFileSystem", _LocalMappedGCSFileSystem)
    monkeypatch.setattr(FsspecStore, "from_url", _mapped_from_url)

    report = run_metrics_benchmark_pipeline(
        bucket_uri="gs://camels-us-mip",
        leaderboard_ids=["pur__lagged_streamflow"],
    )
    assert report["total_evaluated_entries"] == 1
    assert (
        report["leaderboards"]["pur__lagged_streamflow"]["top_model_id"]
        == "gcs_pur_model"
    )
