# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Safe multi-format hydrograph ingestion, sanity verification, and schema inference."""

from hydromip.io.ingest import (
    DEFAULT_PREFIX_STRIPS,
    EVAL_END_DATE,
    EVAL_START_DATE,
    EXPECTED_NUM_BASINS,
    EXPECTED_NUM_DAYS,
    FORBIDDEN_PICKLE_EXTS,
    IngestionRecipe,
    detect_file_format,
    ensure_hydromip_on_path,
    extract_file_metadata,
    ingest_hydrograph,
    normalize_basin_id,
    resolve_deterministic_recipe,
    run_physical_sanity_checks,
    safe_extract_zip,
    write_canonical_zarr_v2,
)
from hydromip.io.llm_schema_adapter import (
    DEFAULT_GEMINI_MODEL,
    build_hydromip_package_snippet,
    build_stuck_report,
    infer_recipe_from_metadata,
)

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_PREFIX_STRIPS",
    "EVAL_END_DATE",
    "EVAL_START_DATE",
    "EXPECTED_NUM_BASINS",
    "EXPECTED_NUM_DAYS",
    "FORBIDDEN_PICKLE_EXTS",
    "IngestionRecipe",
    "build_hydromip_package_snippet",
    "build_stuck_report",
    "detect_file_format",
    "ensure_hydromip_on_path",
    "extract_file_metadata",
    "infer_recipe_from_metadata",
    "ingest_hydrograph",
    "normalize_basin_id",
    "resolve_deterministic_recipe",
    "run_physical_sanity_checks",
    "safe_extract_zip",
    "write_canonical_zarr_v2",
]
