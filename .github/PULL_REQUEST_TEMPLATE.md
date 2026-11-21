## Summary

<!-- Briefly describe what this PR adds or fixes (e.g., new community model contribution, new MIP plugin under mips/<mip_id>/, bug fix, or documentation update). -->

- **Contribution Type:** <!-- [ ] New Model (`Contribution`) | [ ] Benchmark Re-run (`MIP Re-run`) | [ ] New MIP (`mips/<mip_id>/`) | [ ] Core / Bug Fix / Docs -->
- **Target MIP & Leaderboard(s):** <!-- e.g., `camels_us_daily` -> `gauged__single_camels`, `pub__single_camels`, or `N/A` -->
- **Source Code URL (`code_url`):** <!-- Required for open-source model contributions; write "Closed-source (hydrograph-only)" if applicable -->
- **Public Hydrograph Archive DOI / URI (`archive_url`):** <!-- Zenodo / HydroShare DOI or Zarr URI if applicable -->
- **AI Tools Used (if any):** <!-- e.g., "None" or "Claude Code / Codex / Gemini CLI used for scaffolding; all code and tests reviewed by author" -->

> **Note on Automated Review Gating:** Maintainer review (`@grey-nearing`) is gated on passing automated CI checks. While `CI Gate` is still running or if any check fails, external PRs are automatically held in **Draft** so maintainer review is only requested once all tests and protocol guardrails pass.

---

## Mandatory Verification Checklist (Human & AI-Agent Contributors)

### 1. Code Quality & Repository Guardrails (All PRs)
- [ ] `ruff check .` passes locally with zero warnings or errors.
- [ ] `python -m hydromip.tools.ci_guardrails --mode all` passes with zero violations.
- [ ] `pytest` passes across all affected native-resolution (`531 x 3652`) test suites.
- [ ] Zero `try`/`except` error-swallowing blocks, zero `contextlib.suppress` / `warnings.catch_warnings`, zero `allow_pickle=True`, and zero silent fallback paths.
- [ ] Zero large binary blobs (`> 5 MiB`), `.npz`, `.npy`, `.pt`, or `.zarr` directories committed to Git.

### 2. HydrologyMIP Benchmarking Protocol (If Adding or Updating a Model)
- [ ] **Strict Temporal Anti-Leakage:** Training, hyperparameter tuning, early stopping, and architecture selection used **only** `1999-10-01` to `2008-09-30` (never inspecting the `1989-10-01` to `1999-09-30` evaluation decade).
- [ ] **Strict Spatial Anti-Leakage (`pub_12fold` / `pur_18huc02`):** Feature/target normalization statistics and donor weights were fit strictly on `train_basins` inside each fold.
- [ ] **Centralized Splits & Dates:** `models/<model_name>/` never loads split files or hardcodes Newman dates internally; all splits and periods are injected by `models/run_model.py`.
- [ ] **Complete 531-Basin Coverage:** Predictions cover all `531` benchmark basins (`3,652` days, `float32` in `mm/day`) with `pred_nan_when_obs_valid == 0`.
- [ ] **Ensemble vs. Single-Seed Integrity:** If `N >= 2` seeds were run, `qsim == mean(qsim_members)` and single-model skill is reported across all `N` seeds (never `seed 0` alone).
- [ ] **Stage-1 Reproduction (for `MIP Re-run` only):** `reproduce_author_metrics.py` reproduces the cited paper's reported metrics on the authors' original split/forcing, and only author-defined configurations were run.
- [ ] **Registry & Generated-File Hygiene:** Registered runnable models in `MODEL_REGISTRY` and `models/catalog.json` with resolving relative Markdown links, and **did not hand-edit** generated leaderboard/figure files (`mips/*/leaderboards/*.json`, `tables_3_to_14.json`, `table_2_registry.json`, `contributions_index.json`, or `mips/*/figures/*` — maintainers regenerate those upon ingestion via `mip-update-leaderboard` and `leaderboards/on_community_commit.sh`).

### 3. Licensing, Right-to-Submit & AI-Assistance Attestations (All PRs)
- [ ] **Contributor License & DCO ([CONTRIBUTING.md](../CONTRIBUTING.md), [DATA_LICENSE.md](../DATA_LICENSE.md)):** I confirm I have the right to submit this work under the Developer Certificate of Origin 1.1 (`git commit -s`), licensing contributed code under [Apache-2.0](../LICENSE) and contributed hydrographs/data under `CC-BY-4.0` (with any compatible permissive third-party code credited in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
- [ ] **AI-Assistance Disclosure ([AGENTS.md](../AGENTS.md)):** I disclosed above which AI coding tools/models were used (if any), and I have personally reviewed and verified every line of submitted code, documentation, and test logic.
