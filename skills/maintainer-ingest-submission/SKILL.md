---
name: maintainer-ingest-submission
description: >-
  Maintainer-only workflow to turn an accepted HydrologyMIP submission (model PR or
  hydrograph-submission issue) into official leaderboard entries: triage, store validation,
  standard entry IDs, upload to the archive bucket, official scoring with mip-update-leaderboard,
  completing Table 2 and contributions_index metadata, regenerating tables, figures, and the
  manifest, and closing the loop. Use only when you have maintainer write access and are ingesting
  a contribution.
license: Apache-2.0
metadata:
  audience: maintainer
---

# Maintainer workflow: ingest a submission

> **Maintainers only.** All paths are relative to the repository root. Read
> `skills/maintainer-gcs-bucket/SKILL.md` and `skills/leaderboard-norms/SKILL.md` first.

## 1. Triage

- Every field of the issue form or PR template is filled in, every attestation is checked, and CI
  is green (`CI Gate`).
- Check the protocol answers: training only in `1999-10-01`..`2008-09-30`; standard folds for
  PUB/PUR; all 531 basins and 3,652 days.
- Check the licenses: hydrographs under CC-BY-4.0, and code under a permissive license (or the
  submission is marked closed-source).
- Decide the leaderboard(s), the provenance category, the single primary citation, and whether
  this is a new family. If any answer is unclear, ask in the issue before you continue.
- For a code PR: merge the code first, then run the model yourself
  (`python -m models.run_model ...`), or use the contributor's archived store.

## 2. Validate the store locally

```bash
# download and unzip the contributor's Zenodo archive into ./local_bucket/hydrographs/<leaderboard_id>/
python -c "import xarray as xr; from models.zarr_writer import validate_hydrograph_dataset as v; v(xr.open_zarr('<path>.zarr'))"
mip-score-model --model-id <store_name> --leaderboard-id <leaderboard_id> --bucket-uri ./local_bucket \
  --output-json ./outputs/<store_name>_check.json
```

The local score must match the contributor's JSON. If the attributes need fixing (for example
`entry_id`, `family_id`, or `provenance_category`), rebuild the store with
`models.zarr_writer.build_hydrograph_dataset` from its `qsim` and `qsim_members`. Never edit
`.zattrs` by hand.

## 3. Choose the entry ID

Use `<family_id>_<spatial_protocol>_<forcing>`, or the default from `run_model.py`. It must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. The store name must equal the entry ID. For an N ≥ 2 ensemble,
the single-model row uses `<entry_id>_single` and points to the same store.

## 4. Upload, score officially, and add the row

```bash
gcloud storage cp -r ./local_bucket/hydrographs/<leaderboard_id>/<entry_id>.zarr \
  gs://camels-us-mip/hydrographs/<leaderboard_id>/

mip-update-leaderboard --model-id <entry_id> --leaderboard-id <leaderboard_id> \
  --model-name "<Display Name>" --family-id <family_id> --forcing "<Daymet | Daymet, Maurer, NLDAS | ...>" \
  --category Contribution --code-provenance "Our Code" --citation "<Author et al. (YYYY)>" \
  --code-url <code URL> --data-archive-url https://doi.org/<zenodo DOI> \
  --bucket-uri gs://camels-us-mip
```

What the command does:

- scores the store from the bucket;
- inserts the row into `mips/camels_us_daily/leaderboards/<leaderboard_id>.json` and re-ranks it;
- sets `code_provenance`: the value of `--code-provenance`, or else one derived from `--category`
  (`Contribution` and `MIP Re-run` give `Our Code`, `Archive*` gives `Author Archive`, and
  `Literature*` gives `Published Paper`);
- recounts the family counts in `table_2_registry.json`;
- re-syncs `tables_3_to_14.json` and redraws the figures.

`--is-ensemble` is only a cross-check; the store is the authority.

## 5. Complete the metadata the CLI does not write

- **New family:** CI requires every row's `family_id` to exist in `table_2_registry.json`. For a new
  family, add it with
  `leaderboards.update_leaderboard.upsert_leaderboard_entry(leaderboard_id, entry, family_record=...)`.
  The counts (`n_entries`, `leaderboards_covered`, family totals) are recomputed automatically.
- **Contributions index:** the CLI does not update `contributions_index.json`. Pass
  `contribution_record=...` to `upsert_leaderboard_entry`.
- **Row fields the CLI leaves empty,** such as `model_category` and `publication_year`: set them
  through the same reviewed `upsert_leaderboard_entry` call.
- Use the field lists in `skills/data-schemas/SKILL.md` and the notes in `leaderboards/README.md`.
  Do not edit the JSON files by hand.

## 6. Regenerate and check

```bash
bash leaderboards/on_community_commit.sh       # re-sync tables_3_to_14.json and figures for every MIP
mip-update-leaderboard --render-hydrograph-manifest ./outputs/manifest_index.json --bucket-uri gs://camels-us-mip
gcloud storage cp ./outputs/manifest_index.json gs://camels-us-mip/hydrographs/manifest_index.json
python -m hydromip.tools.ci_guardrails --mode all   # includes the registry bijection checks
pytest leaderboards/tests
```

## 7. Close the loop

- Open a PR with the JSON and figure changes that references the issue, and get it reviewed and
  merged.
- Reply on the issue with the leaderboard row and its rank, then close it.

## Stop and ask the project lead if

- the official score differs from the contributor's local score;
- the submission needs a new leaderboard, regime, or provenance category;
- a protocol answer, license, or citation is doubtful.
