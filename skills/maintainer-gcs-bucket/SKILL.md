---
name: maintainer-gcs-bucket
description: >-
  Maintainer-only runbook for the private HydrologyMIP archive bucket gs://camels-us-mip/: layout,
  access with standard Google Cloud authentication, invariants (one store per row, no orphans, no
  non-redistributable or path-leaking stores), and safe procedures to upload, verify, back up,
  delete, and re-index stores, rebuild targets, and relate the bucket to the Zenodo release
  archives. Use only when you have maintainer write access and must change or audit the bucket.
license: Apache-2.0
metadata:
  audience: maintainer
---

# Maintainer runbook: `gs://camels-us-mip/`

> **Maintainers only.** Contributors never need this bucket. They work with `./outputs/` and
> `./local_bucket/` (see `skills/contribute-model/SKILL.md`).

All paths are relative to the repository root.

## 1. Facts

- Private bucket with public access prevention enforced and uniform bucket-level access. Do not
  change its visibility without an explicit decision by the project lead.
- Each MIP declares its storage in `protocol.json` (`storage.bucket_uri`,
  `storage.targets_zarr_uri`, `storage.hydrographs_prefix`). For `camels_us_daily` that is this
  bucket.
- The bucket has **no object versioning**. A delete or overwrite cannot be undone, so back up first.

## 2. Layout

```text
gs://camels-us-mip/
├── README.md
├── splits/                          # mirror of mips/camels_us_daily/splits/
├── observations/newman_eval_targets.zarr
├── stage1_author_reproductions/     # <model>.zarr, <model>_results.json, manifest_index.json
└── hydrographs/
    ├── manifest_index.json          # generated; one entry per store-backed leaderboard row
    └── <leaderboard_id>/<entry_id>.zarr
```

Every store lives directly under its leaderboard folder. There are no other partitions.

## 3. Access

Use standard Google Cloud authentication only:

```bash
gcloud auth login
gcloud auth application-default login     # used by gcsfs/xarray in this repo
# or a service account: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

`HYDROMIP_GCS_ANON=1` forces anonymous access and only works for public buckets. CI uses it to prove
that tests never need credentials.

## 4. Invariants (check before and after every change)

1. **Bijection.** Every store-backed row (`zarr_path`) points to exactly one existing store, and
   every store is referenced by at least one row. No orphans. An ensemble row and its `_single` row
   may share a store.
2. **Licensing.** Only stores we may redistribute (`DATA_LICENSE.md`). Stores under
   non-commercial or other non-redistributable licenses do not belong in this bucket.
3. **No path leaks.** Store attributes never contain absolute local paths. `data_source` holds a
   file name only.
4. **Stability.** A store referenced by a published row is not overwritten. A corrected run gets a
   new entry ID, or a recorded issue explains the replacement.

## 5. Procedures

```bash
# Upload a vetted store (validated with models.zarr_writer.validate_hydrograph_dataset first)
gcloud storage cp -r ./local_bucket/hydrographs/<leaderboard_id>/<entry_id>.zarr \
  gs://camels-us-mip/hydrographs/<leaderboard_id>/

# Verify it scores from the bucket
mip-score-model --model-id <entry_id> --leaderboard-id <leaderboard_id> \
  --output-json ./outputs/<entry_id>_official_score.json

# Back up before any delete or overwrite (no object versioning)
gcloud storage cp -r gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr ./outputs/backup/

# Regenerate and publish the hydrograph manifest from the leaderboard rows
mip-update-leaderboard --render-hydrograph-manifest ./outputs/manifest_index.json --bucket-uri gs://camels-us-mip
gcloud storage cp ./outputs/manifest_index.json gs://camels-us-mip/hydrographs/manifest_index.json

# Rebuild the targets and compare them with the canonical store (never overwrite it without review)
python -m metrics.build_eval_targets --camels-dir /path/to/basin_dataset_public_v1p2 \
  --splits-dir mips/camels_us_daily/splits --output ./outputs/newman_eval_targets.zarr \
  --verify-against gs://camels-us-mip/observations/newman_eval_targets.zarr
```

Full-bucket check: `benchmark-mip-metrics --output-json ./outputs/benchmark_report.json`. A partial
`Archive*` store stops a whole-bucket run, so pass `--manifest-path` with the
`(leaderboard_id, model_id)` entries to check, or filter with `--leaderboard-id`. Write reports to
`./outputs/`, never into `leaderboards/` or `mips/*/leaderboards/`.

## 6. Zenodo archives

- `10.5281/zenodo.24383448` is the frozen v1.0 release archive of HydrologyMIP.
- `10.5281/zenodo.23641695` remains the archive cited by the ERA rows.
- A later release gets a new Zenodo version and a new git tag. Never move or delete a published tag.

## Stop and ask the project lead if

- a store must be deleted, moved, or overwritten;
- a store's license is unclear;
- someone proposes changing the bucket's visibility or access policy.
