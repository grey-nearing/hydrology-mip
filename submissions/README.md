# Community Submission Manifests (`submissions/`)

This directory holds reviewed submission manifest JSON files organized by target leaderboard:

```text
submissions/<leaderboard_id>/<entry_id>.json
```

When a manifest PR or automated issue submission targets `main`, GitHub Actions runs
[`hydromip/tools/process_submission.py`](../hydromip/tools/process_submission.py) to:

1. Validate the DOI/archive URL (`https://` allowlisted hosts, SSRF protection, zero `.pkl` payloads).
2. Ingest arbitrary tabular or gridded predictions via `hydromip.io.ingest` into standard `(531, 3652)` consolidated Zarr v2 (`mm/day`, `1989-10-01` to `1999-09-30`, sorted 8-digit USGS IDs).
3. Compute all 14 Gauch et al. (2023) deterministic metrics (including `logNSE` and `logKGE`) via `metrics.evaluator`.
4. Atomically update `mips/camels_us_daily/leaderboards/*.json`, `table_2_registry.json`, `tables_3_to_14.json`, and paper figures upon merge.

Manifests may be generated interactively at <https://hydrologymip.org/#contribute>.
