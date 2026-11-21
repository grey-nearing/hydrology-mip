---
name: maintainer-audit
description: >-
  Maintainer-only procedures for auditing HydrologyMIP models and leaderboards: the six-point
  forensic model audit, end-to-end bijection across leaderboard rows, archive stores, the
  manifest, registries, the Stage-1 appendix, and the manuscript tables, re-scoring and
  reconciliation tools, the full-bucket benchmark run, clean-slate rebuild rules, and the release
  checklist. Use only when verifying MIP Re-runs, reconciling leaderboards, or preparing a release.
license: Apache-2.0
metadata:
  audience: maintainer
---

# Maintainer audits and releases

> **Maintainers only.** All paths are relative to the repository root.

## 1. Six-point forensic audit (for every MIP Re-run, and before every release)

1. **Store provenance.** Read the store attributes: `git_commit`, `git_dirty`,
   `runner_kwargs_json`, `seeds`, `fold_ids`, `created_utc`, `data_source`. Confirm the store came
   from `models/run_model.py` with the registry defaults. Confirm that no keyword overrides
   shortened training (epochs, batches per epoch, sequence or target length, stride). A Stage-1
   record is not proof that a Stage-2 store was made the same way.
2. **Training budget.** Compare the total gradient updates per seed (epochs × batches per epoch)
   and the training and evaluation window strides with the original authors' code.
3. **Line-by-line fidelity.** Compare our implementation with the authors' upstream repository:
   - input and target transforms;
   - static attribute list;
   - layers, readout head, and dropout;
   - loss;
   - optimizer, learning-rate schedule, batch size, and clipping;
   - sequence length, lagged inputs, and spin-up;
   - number of seeds and fine-tuning stages.
   Record every remaining difference in the model README and the row's `notes`.
4. **Sanity checks across tables** (same model family):
   - ensemble NSE ≥ single-seed mean;
   - gauged > PUB > PUR;
   - three forcings ≥ one forcing;
   - with lagged streamflow > without.
   Explain or fix any violation.
5. **Authorship gate.** The paper defined and ran every included configuration ("if they didn't run
   it, we don't run it"). No re-run duplicates an existing `Archive` or `Literature` entry for the
   same configuration.
6. **Bijection.** Rows in `mips/camels_us_daily/leaderboards/*__*.json` and `tables_3_to_14.json`
   must match each other and the following, with zero orphans and zero missing items:
   - the stores under `gs://camels-us-mip/hydrographs/` and `hydrographs/manifest_index.json`;
   - the families in `table_2_registry.json` (`n_entries`, `leaderboards_covered`, `best_nse`);
   - `contributions_index.json` and `appendix_stage1_confirmation_runs.json`;
   - the manuscript's Table 2, Tables 3–14, and appendices.

## 2. Tools

```bash
python -m hydromip.tools.ci_guardrails --mode all                 # registry and MIP JSON bijection, docs, code rules
mip-update-leaderboard --reconcile-archived --bucket-uri gs://camels-us-mip --output-json ./outputs/reconcile.json   # report only
mip-update-leaderboard --reconcile-archived --apply --bucket-uri gs://camels-us-mip  # rewrite those rows from the evaluator
mip-update-leaderboard --recalculate-all-self-run --bucket-uri gs://camels-us-mip --output-json ./outputs/recalc.json
mip-update-leaderboard --render-stage1-appendix                   # rebuild the .md/.csv appendix from its JSON
mip-update-leaderboard --render-hydrograph-manifest ./outputs/manifest_index.json --bucket-uri gs://camels-us-mip
benchmark-mip-metrics --bucket-uri gs://camels-us-mip --output-json ./outputs/benchmark_report.json
```

Write every report to `./outputs/`. Never write a report into `leaderboards/` or
`mips/*/leaderboards/`, where it could overwrite official files. A partial `Archive*` store stops a
whole-bucket `benchmark-mip-metrics` run, so give it `--manifest-path` or `--leaderboard-id`
filters.

## 3. Clean-slate rebuild rule

For a major rebuild of the registry or leaderboards, do not patch JSON files piece by piece:

1. Rebuild `table_2_registry.json`, the 12 leaderboard files, `tables_3_to_14.json`, and the Stage-1
   appendix in one deterministic pass. Use the verified per-model specifications and fresh
   evaluator runs on the archive stores.
2. Regenerate all figures (`bash leaderboards/on_community_commit.sh`).
3. Run the guardrails and the `leaderboards/` tests.

## 4. Release checklist

- [ ] The audit above passes, and `CI Gate` is green on `main`.
- [ ] The manifest is published, and the bucket invariants in
      `skills/maintainer-gcs-bucket/SKILL.md` hold.
- [ ] `CITATION.cff`, `.zenodo.json`, and `pyproject.toml` versions and authors agree. The guardrails
      check this.
- [ ] Ship a **new** version tag (for example `v1.0.1`). Never move or delete a published tag. Create
      a new Zenodo version.

## Stop and ask the project lead if

- an audit finds a protocol violation in a published row;
- fixing a problem would change a published rank or number;
- the manuscript and the leaderboards disagree.
