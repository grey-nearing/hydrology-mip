---
name: pr-review
description: >-
  Review checklist for HydrologyMIP pull requests from humans or AI agents: scope, CI status,
  protocol and anti-leakage rules, data integrity, tests, code quality, licensing and provenance of
  contributed code and data, generated files, documentation, and AI-assistance disclosure, with a
  severity-ranked output format. Use when reviewing a PR, or when checking your own change before
  opening one.
license: Apache-2.0
metadata:
  audience: reviewer
---

# Pull request review

Load the skills that match the change:

- always `skills/protocol-guardrails/SKILL.md`, `skills/testing/SKILL.md`, and
  `skills/docs-style/SKILL.md`;
- for a model PR, also `skills/contribute-model/SKILL.md` and `skills/leaderboard-norms/SKILL.md`;
- for a new benchmark, also `skills/contribute-mip/SKILL.md`.

All paths are relative to the repository root.

## 1. Scope and CI

- [ ] The PR has one purpose and matches its description and template. No unrelated edits.
- [ ] `CI Gate` is green: ruff, guardrails, docs and registry checks, packaging, and the test shards.
- [ ] Review comments stay within the PR's scope. Record follow-up ideas as issues instead of
      growing the PR.

## 2. Protocol and leakage

- [ ] Model code gets basins, folds, and periods only through `run_fold(...)`. No split files are
      read and no canonical dates are hard-coded.
- [ ] Scalers, loss weights, early stopping, and hyperparameter choices use training basins and the
      training period only. Held-out PUB/PUR basins are never used within their fold.
- [ ] Spin-up discharge stays masked. Test-basin discharge is used only by models registered for
      `lagged_streamflow`.
- [ ] `allowed_forcing_regimes` and `allowed_spatial_regimes` are honest. The registry and the
      catalog match.
- [ ] No change to `protocol.json` or `splits/` of an active MIP.

## 3. Data integrity

- [ ] No imputation, zero-filling, forward-filling, interpolation, or silent masking. No fallback
      data paths or "try several keys" logic.
- [ ] No `try`/`except`, `contextlib.suppress`, `warnings.catch_warnings`, or `ignore_errors=True`.
      Preconditions are checked explicitly and fail with clear errors.
- [ ] `np.load(..., allow_pickle=False)` and `torch.load(..., weights_only=True)`. No hard-coded
      `/tmp` or machine paths.
- [ ] Stores are written only with `models/zarr_writer.py`. Nothing is written to
      `gs://camels-us-mip/` from contributor code.

## 4. Tests

- [ ] New behavior has tests. Store, split, and scoring code is tested at 531 × 3,652.
- [ ] Negative tests exist for missing stores, wrong shapes or IDs, and NaN predictions on valid
      observation days.
- [ ] Mocks sit only at network or cloud boundaries. Only registered markers are used.

## 5. Code quality

- [ ] Public functions have type annotations and docstrings. No dead code, debug prints, or
      commented-out blocks.
- [ ] The change reuses the shared helpers (`models.camels_data`, `models.zarr_writer`,
      `hydromip.evaluation`, `hydromip.splits`) instead of copying them.
- [ ] Every new `.py` file starts with the copyright line and `# SPDX-License-Identifier: Apache-2.0`.

## 6. Licensing and provenance

- [ ] Contributed code is the author's own or uses a permissive license (MIT, BSD, Apache-2.0)
      credited in `THIRD_PARTY_NOTICES.md`. No GPL, LGPL, or non-commercial code is copied in.
- [ ] Hydrographs are offered under CC-BY-4.0, or their upstream license is recorded in
      `DATA_LICENSE.md`.
- [ ] The entry has one canonical citation and one correct provenance category
      (`skills/leaderboard-norms/SKILL.md`). For a MIP Re-run, a Stage-1 reproduction is included.
- [ ] The PR template's contributor attestation is checked, and AI assistance is disclosed.

## 7. Generated files, data, and docs

- [ ] No hand edits to leaderboard JSONs, `tables_3_to_14.json`, `table_2_registry.json`,
      `contributions_index.json`, the Stage-1 appendix, or figures. Only a maintainer's tooling
      commit may change them.
- [ ] No binaries, Zarr stores, data bundles, checkpoints, secrets, or files larger than 5 MiB.
- [ ] Docs and skills changed where behavior changed (README, `models/README.md`, the model's
      README, `AGENTS.md` if a skill was added). All relative links resolve.

## Output format

Report findings by severity (**blocker**, **major**, **minor**, **nit**). Give `file:line` and a
concrete fix for each. End with a verdict: *approve*, *request changes*, or *needs maintainer
decision*.
