"""Canonical benchmark verification package for CAMELS-US MIP (`benchmarks/`).

Provides end-to-end verification and leaderboard table scoring across all model
hydrograph Zarr stores in `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`.
"""

from benchmarks.metrics_pipeline import (
    discover_hydrograph_zarr_entries,
    format_verification_markdown_table,
    load_manifest_entries,
    run_metrics_benchmark_pipeline,
    verify_single_entry_report,
)

__all__ = [
    "discover_hydrograph_zarr_entries",
    "format_verification_markdown_table",
    "load_manifest_entries",
    "run_metrics_benchmark_pipeline",
    "verify_single_entry_report",
]
