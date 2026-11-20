# AGENTS.md — HydrologyMIP (`camels-us-mip`)

Rules for AI coding agents working in this repository (Claude Code, Codex, Gemini CLI, GitHub
Copilot, Cursor, and others). Humans: start with [README.md](README.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Project in one paragraph

HydrologyMIP (<https://hydrologymip.org>) is an open platform for fair comparison of hydrological
models. Each benchmark (a "MIP") lives in `mips/<mip_id>/`. The active benchmark is
`mips/camels_us_daily/`: daily streamflow for 531 CAMELS-US basins, with 12 leaderboards
(3 spatial regimes × 4 input regimes). Its machine-readable protocol is
`mips/camels_us_daily/protocol.json`. That file is the source of truth for dates, regimes, and
leaderboards. The Python import package is `hydromip`.

## Commands

| Task | Command |
| :--- | :--- |
| Install (Python 3.11+) | `pip install -e ".[dev]"` (CPU-only PyTorch: install `torch` from the PyTorch CPU index first) |
| Lint | `ruff check .` |
| Repository guardrails | `python -m hydromip.tools.ci_guardrails --mode all` |
| Tests | `pytest` (one suite: `pytest models/tests`) |
| List models | `python -m models.run_model --list-models` |
| Run a model (writes `./outputs/<leaderboard_id>/<model_id>.zarr`) | `python -m models.run_model --model <key> --spatial-regime gauged --forcings daymet` |
| Score a local store | `mip-score-model --model-id <id> --leaderboard-id <lb> --bucket-uri ./local_bucket` |
| Input data (not stored in git) | see `mips/camels_us_daily/data/README.md` |

## Hard rules (a PR that breaks one of these is rejected)

1. **Never change** the splits, dates, basin lists, or `protocol.json` of an active MIP.
2. **Every model run goes through `models/run_model.py`.** Model code never reads split files and
   never hard-codes dates or basins. It receives them through `run_fold(...)`.
3. **No test-period leakage.** Fit scalers, loss weights, early stopping, hyperparameter search,
   architecture choice, and donor selection on the training basins and the training period only.
4. **Missing data gives NaN or a loud error.** No imputation, no silent masking, no fallback data
   paths, and no `try`/`except` that hides a failure. Validate inputs and raise instead.
5. **Hydrograph stores** have shape `(531, 3652)`, use `basin`/`date` coordinates, and are written
   only with `models/zarr_writer.py`.
6. **Never hand-edit generated files:** the leaderboard JSONs, `tables_3_to_14.json`,
   `table_2_registry.json`, `contributions_index.json`, the Stage-1 appendix, or `mips/*/figures/*`.
   Maintainers regenerate them with the tools in `leaderboards/`.
7. **Never write to `gs://camels-us-mip/`.** It is maintainer-only. Use local paths
   (`./outputs/...`, `./local_bucket/...`).
8. **Never commit** secrets, credentials, machine-specific paths, data bundles, Zarr stores, model
   checkpoints, or any file larger than 5 MiB.
9. **Tests** run at the native size of 531 basins × 3,652 days. Mock only network or cloud
   boundaries.
10. **Do not push to `main`, merge, tag, or release.** Open a pull request, fill in the template,
    and say that AI tools helped write it.

## Skills: load the one that matches your task

Each skill is a folder with a `SKILL.md`. Tools that support Agent Skills find them through
`.agents/skills/` and `.claude/skills/`, which both link to `skills/`. If your tool does not, open
the file directly.

| Task | Skill file | Audience |
| :--- | :--- | :--- |
| Understand the layout, or decide where a file belongs | `skills/repo-structure/SKILL.md` | everyone |
| Any change that touches data, splits, training, scoring, or stores | `skills/protocol-guardrails/SKILL.md` | everyone |
| Add a runnable model under `models/<name>/` | `skills/contribute-model/SKILL.md` | contributors |
| Package and submit hydrographs from code that runs elsewhere | `skills/contribute-hydrographs/SKILL.md` | contributors |
| Add a new benchmark under `mips/<mip_id>/` | `skills/contribute-mip/SKILL.md` | contributors |
| Provenance, citation, ensemble rows, leaderboard routing | `skills/leaderboard-norms/SKILL.md` | contributors, reviewers |
| Zarr, splits, leaderboard-row, catalog, and protocol schemas | `skills/data-schemas/SKILL.md` | everyone |
| Write or run tests | `skills/testing/SKILL.md` | everyone |
| Write or edit documentation or skills | `skills/docs-style/SKILL.md` | everyone |
| Review a PR, or check your own PR before opening it | `skills/pr-review/SKILL.md` | reviewers, contributors |
| Bucket layout and safe write procedures | `skills/maintainer-gcs-bucket/SKILL.md` | maintainers only |
| Ingest a submission and update leaderboards and figures | `skills/maintainer-ingest-submission/SKILL.md` | maintainers only |
| Forensic model audits, registry bijection, and benchmark runs | `skills/maintainer-audit/SKILL.md` | maintainers only |

## Where the facts live (do not copy them into new places)

- Human docs: [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md),
  [hydromip/README.md](hydromip/README.md), [mips/README.md](mips/README.md),
  [mips/camels_us_daily/README.md](mips/camels_us_daily/README.md),
  [models/README.md](models/README.md), [metrics/README.md](metrics/README.md),
  [leaderboards/README.md](leaderboards/README.md), [benchmarks/README.md](benchmarks/README.md).
- Machine-readable sources of truth: `mips/<mip_id>/protocol.json`, `MODEL_REGISTRY` in
  `models/run_model.py` (mirrored in `models/catalog.json`), `models/zarr_writer.py`,
  `mips/<mip_id>/leaderboards/*.json`.

## When docs and code disagree

The code and `protocol.json` win. Do not change the protocol to match a document. Report the
mismatch in your PR, or fix the document.
