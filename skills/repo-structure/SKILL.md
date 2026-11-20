---
name: repo-structure
description: >-
  Map of the HydrologyMIP (camels-us-mip) repository: the shared hydromip core, self-contained
  benchmarks under mips/<mip_id>/, the model zoo, scoring, leaderboard tooling, the web portal,
  CI guardrails, and the two documentation layers. Use when you need to find where code or data
  lives, decide where a new file belongs, or check which paths you must never edit.
license: Apache-2.0
metadata:
  audience: contributor
---

# Repository structure

HydrologyMIP keeps shared code in a small number of packages and puts everything that belongs to
one benchmark under `mips/<mip_id>/`. The human-readable tree is in the root `README.md`. This skill
adds the rules an agent needs. All paths in this skill are relative to the repository root.

## Map

| Path | What it holds | Who may edit it |
| :--- | :--- | :--- |
| `hydromip/` | Shared core: protocol schema and discovery, path resolution, cross-validation checks, metric suites, Zarr I/O, and `tools/ci_guardrails.py`. See `hydromip/README.md`. | Core changes need maintainer agreement in an issue first |
| `mips/<mip_id>/` | One benchmark: `protocol.json`, `adapter.py`, `splits/`, `leaderboards/`, `figures/`, `data/`, optional `era_experiments/`. See `mips/README.md`. | `protocol.json` and `splits/` of an active MIP are frozen. `leaderboards/` and `figures/` are generated. |
| `models/` | Model zoo: `run_model.py` (central runner and `MODEL_REGISTRY`), `catalog.json` (mirror of the registry), `camels_data.py` (data loader), `zarr_writer.py` (store writer), and one folder per model. See `models/README.md`. | Contributors add `models/<name>/` plus registry, catalog, and test entries |
| `metrics/` | Scoring: `evaluator.py`, the `mip-score-model` CLI, and `build_eval_targets.py` | Bug fixes only, with tests |
| `leaderboards/` | Maintainer tools that update leaderboard JSON and draw figures. Holds code only, never data. | Maintainers |
| `benchmarks/` | End-to-end verification of every store in the bucket | Maintainers |
| `web/` | Portal UI (`web/portal/`), stateless preview scorer API (`web/scorer_api/`), Cloudflare Worker bundler (`web/worker/`), and tests (`web/tests/`). | Maintainers (or PRs touching portal UI/scorer) |
| `skills/` | Agent skills (this folder). `.agents/skills` and `.claude/skills` are symlinks to it. | Follow `skills/docs-style/SKILL.md` |
| `.github/` | CI workflows, PR and issue templates, `CODEOWNERS` | Maintainers |
| Root files | `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `LICENSE`, `NOTICE`, `DATA_LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, `.zenodo.json`, `pyproject.toml` | Docs: anyone. Legal and metadata files: maintainers. |

## Rules

1. **Per-MIP data lives only in `mips/<mip_id>/`.** Shared packages never store leaderboard tables,
   split files, or figures.
2. **No large or binary data in git.** Git ignores `.npz`, `.npy`, `.zarr/`, `.zip`, `.nc`, `.pt`,
   `outputs/`, and `hydrographs/`. Keep data bundles in `mips/<mip_id>/data/` (gitignored), model
   output in `./outputs/`, and local scoring stores in `./local_bucket/`. CI rejects any file
   over 5 MiB.
3. **Model code lives once**, in `models/<name>/`. It declares the MIPs and regimes it supports in
   `MODEL_REGISTRY` and `models/catalog.json`. It never contains a copy of the runner, the loader, or
   the writer.
4. **No new top-level folders** such as `scripts/` or `tools/`. Helpers belong in
   `<package>/tools/`.
5. **Generated files are never hand-edited:** `mips/*/leaderboards/*.json` (including
   `tables_3_to_14.json`, `table_2_registry.json`, `contributions_index.json`, the Stage-1 appendix
   files) and `mips/*/figures/*`.
6. **Keep things in sync.** When you add a package, test folder, or MIP, update
   `pyproject.toml` (package data, `testpaths`), the CI workflow if needed, and the docs.

## Sources of truth

| Fact | Canonical source |
| :--- | :--- |
| Dates, spatial regimes, leaderboards of a MIP | `mips/<mip_id>/protocol.json` |
| Registered models and their allowed regimes | `MODEL_REGISTRY` in `models/run_model.py` (mirrored in `models/catalog.json`; a test checks they match) |
| Hydrograph store schema | `models/zarr_writer.py` |
| The 14 daily metrics | `hydromip/metrics/deterministic_daily.py` and the suite in `hydromip/metrics/registry.py` (re-exported by `metrics/metrics.py`) |
| Model families (Table 2) | `mips/camels_us_daily/leaderboards/table_2_registry.json` |
| Leaderboard rows | `mips/camels_us_daily/leaderboards/<leaderboard_id>.json` and `tables_3_to_14.json` |

## Two documentation layers

- **Layer 1, for humans:** `README.md`, `CONTRIBUTING.md`, and the package and MIP READMEs. They
  describe what exists and how to use it.
- **Layer 2, for agents:** `AGENTS.md` (always loaded) and `skills/<name>/SKILL.md` (loaded when
  needed). They describe how to act and what to check. They link to Layer 1 for facts.

## Verify

```bash
python -m hydromip.tools.ci_guardrails --mode all
```

## Stop and ask a human if

- the change needs a new top-level folder, or a change to `hydromip/`, `protocol.json`, or `splits/`;
- you are unsure whether a file is generated;
- you would have to edit `.github/` workflows or security guardrails.
