"""CAMELS-US MIP leaderboards package (`leaderboards/`).

Maintains the 12 canonical leaderboard JSON files (`<leaderboard_id>.json`),
the master multi-table bundle (`tables_3_to_14.json`), the model family registry
(`table_2_registry.json`), the Stage-1 author-split confirmation appendix
(`appendix_stage1_confirmation_runs.json`, rendered to `.md` and `.csv`), and the
automated scripts triggered upon each community submission to update and re-rank the
leaderboard JSON files. The data files live in `mips/<mip_id>/leaderboards/`.
"""

from leaderboards.generate_figures import (
    DEFAULT_FIGURES_DIR,
    classify_model_category,
    extract_publication_year,
    generate_all_paper_figures,
    generate_figure_1_water_year_coverage,
    generate_figure_2_leaderboard_progress,
)
from leaderboards.update_leaderboard import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_LEADERBOARDS_DIR,
    HYDROGRAPH_MANIFEST_NAME,
    build_hydrograph_manifest,
    clean_model_name,
    default_code_provenance,
    ensure_per_leaderboard_files,
    find_duplicate_metric_rows,
    format_row_metrics_from_evaluator,
    load_all_leaderboards,
    load_leaderboard,
    normalize_legacy_leaderboards,
    normalize_legacy_row,
    rank_leaderboard_rows,
    recalculate_all_self_run_models,
    reconcile_archived_rows,
    remove_leaderboard_entry,
    render_stage1_appendix_csv,
    refresh_registry_counts,
    render_stage1_appendix_markdown,
    resolve_row_store_id,
    sync_master_tables_from_per_leaderboard_files,
    trigger_submission_from_gcs,
    upsert_leaderboard_entry,
    validate_stage1_appendix,
    write_hydrograph_manifest,
    write_stage1_appendix,
)

__all__ = [
    "CANONICAL_LEADERBOARD_IDS",
    "DEFAULT_FIGURES_DIR",
    "DEFAULT_LEADERBOARDS_DIR",
    "HYDROGRAPH_MANIFEST_NAME",
    "build_hydrograph_manifest",
    "classify_model_category",
    "clean_model_name",
    "default_code_provenance",
    "ensure_per_leaderboard_files",
    "extract_publication_year",
    "find_duplicate_metric_rows",
    "format_row_metrics_from_evaluator",
    "generate_all_paper_figures",
    "generate_figure_1_water_year_coverage",
    "generate_figure_2_leaderboard_progress",
    "load_all_leaderboards",
    "load_leaderboard",
    "normalize_legacy_leaderboards",
    "normalize_legacy_row",
    "rank_leaderboard_rows",
    "recalculate_all_self_run_models",
    "reconcile_archived_rows",
    "refresh_registry_counts",
    "remove_leaderboard_entry",
    "render_stage1_appendix_csv",
    "render_stage1_appendix_markdown",
    "resolve_row_store_id",
    "sync_master_tables_from_per_leaderboard_files",
    "trigger_submission_from_gcs",
    "upsert_leaderboard_entry",
    "validate_stage1_appendix",
    "write_hydrograph_manifest",
    "write_stage1_appendix",
]
