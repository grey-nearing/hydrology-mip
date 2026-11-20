---
name: leaderboard-norms
description: >-
  HydrologyMIP rules for what may appear on a leaderboard and how it is labeled: one canonical
  citation per entry, the provenance vocabulary (row category plus code_provenance, Table 2
  families, Zarr attributes), architecture classes, literature and archive priority, "if they
  didn't run it, we don't run it", ensemble versus single-model rows, routing to the 12 CAMELS-US
  leaderboards, asterisk flags, and Stage-1 author reproductions. Use when classifying, routing,
  naming, or reviewing any model entry or leaderboard row.
license: Apache-2.0
metadata:
  audience: contributor
---

# Leaderboard and provenance norms

Contributors need these rules to label a submission correctly. Reviewers use them to accept or
reject it. Maintainers write the leaderboard rows with the tools in `leaderboards/`; nobody edits the
JSON by hand. All paths are relative to the repository root. The data files are documented in
`mips/camels_us_daily/leaderboards/README.md`.

## 1. One canonical citation per entry

- Every family in `table_2_registry.json` and every leaderboard row cites exactly **one** paper
  (`citation`, `paper_url`, `publication_year`). Never join several citations in one string.
- Which paper to cite:
  1. the first paper that ran this model configuration on the Newman et al. 531-basin split;
  2. otherwise, the first paper that ran the model on CAMELS-US;
  3. otherwise (a general ML architecture never run on CAMELS-US), the paper that introduced the
     architecture.
- Distinct configurations or versions from different papers are separate families and separate
  rows.

## 2. Provenance vocabulary (two levels)

| Where | Field | Allowed values |
| :--- | :--- | :--- |
| Leaderboard row | `category` | `Literature`, `Literature*`, `Archive`, `Archive*`, `MIP Re-run`, `Contribution` |
| Leaderboard row | `code_provenance` | `Our Code` (re-implementation in this repo), `Author Archive` (authors' archived hydrographs), `Published Paper` (numbers from the paper) |
| Table 2 family | `provenance_category` | the row values, but `MIP Re-run` is written `MIP Re-run (Our Code)` or `MIP Re-run (Author Code)` |
| Zarr store attribute | `provenance_category` | `Literature`, `MIP Re-run`, `Archive`, `Archive*`, `Contribution` (enforced by `models/zarr_writer.py`) |

What the categories mean:

- **`Literature`**: metrics reported in a paper on the canonical 531-basin split, with no public
  daily hydrographs. Usually only NSE is known; the other metrics show `---`.
- **`Literature*`**: like `Literature`, but on a slightly different evaluation window, such as water
  years 1991–1999.
- **`Archive`**: daily hydrographs archived by the authors, covering all 531 basins and the
  canonical evaluation window, and scored here by `metrics/evaluator.py`.
- **`Archive*`**: archived hydrographs that cover fewer than 531 basins or a different window.
- **`MIP Re-run`**: a published model re-implemented (`Our Code`) or re-executed from the authors'
  code, run on the MIP protocol with `models/run_model.py`. It needs a Stage-1 reproduction
  (section 6).
- **`Contribution`**: a new model contributed to HydrologyMIP, with full `(531, 3652)` hydrographs.

Architecture class (`model_category` on rows, `category` in Table 2): `Machine learning`,
`Hybrid (ML + Physics)`, `Conceptual`, `Process-based`, or `Statistical`.

## 3. Priority and authorship

- If a model was already published on the canonical split for a configuration, the leaderboard
  keeps the original `Literature` or `Archive` values. A re-run can be added as a separate,
  competing entry. It never replaces the original.
- **If they didn't run it, we don't run it.** For published models by outside authors, run only the
  forcing and spatial configurations that the original paper ran. The only exceptions are a project
  maintainer who co-authored that paper, or explicit maintainer approval recorded in an issue.

## 4. Ensemble and single-model rows

- When a store holds N ≥ 2 seeds (`qsim_members`), the model appears as **two competing rows** that
  point to the same store:
  - **Ensemble row:** `entry_id = "<base>"`, `is_ensemble: true`, `num_members: N`. It is scored on
    the mean hydrograph `qsim`. Its name may end in `(<N>-model ensemble)`, and then N must equal
    `num_members`.
  - **Single-model row:** `entry_id = "<base>_single"`, `is_ensemble: false`. It is scored as the
    mean ± standard deviation over **all N members**, never seed 0 alone. It never carries an
    ensemble suffix; `(single model)` is allowed.
- The evaluator reads the ensemble status from the store, never from the model name or a command-
  line claim.

## 5. Routing to the 12 CAMELS-US leaderboards

| Spatial regime | `single_camels` (one of Daymet, Maurer, NLDAS) | `multi_camels` (2–3 products) | `lagged_streamflow` | `freeform` |
| :--- | :--- | :--- | :--- | :--- |
| `gauged` | `gauged__single_camels` (Table 3) | `gauged__multi_camels` (4) | `gauged__lagged_streamflow` (5) | `gauged__freeform` (6) |
| `pub_12fold` | `pub__single_camels` (7) | `pub__multi_camels` (8) | `pub__lagged_streamflow` (9) | `pub__freeform` (10) |
| `pur_18huc02` | `pur__single_camels` (11) | `pur__multi_camels` (12) | `pur__lagged_streamflow` (13) | `pur__freeform` (14) |

- `lagged_streamflow`: the model uses observed discharge of the target basins at test time (lagged
  values, data integration, or data assimilation). In PUB and PUR these basins are still held out
  during training.
- `freeform`: non-CAMELS forcings (for example ERA5-Land or AORC), extra attributes beyond the 27
  standard CAMELS attributes, flow-derived features, or foundation-model pretraining on target
  gauges.
- Out of scope: sub-daily models that need hourly data for all basins. Also out of scope:
  retrospective archives that were neither calibrated on the canonical training period nor
  evaluated with a uniform split, such as the NOAA National Water Model retrospectives.
- Any row with fewer than 531 basins or a non-standard window sets `asterisk: true`, gives a
  specific `asterisk_reason`, and ends its display name with `*`.

## 6. Stage-1 author reproduction (MIP Re-run only)

- Before a re-run goes on the leaderboards, `models/<key>/reproduce_author_metrics.py` must
  reproduce the paper's numbers with the paper's own split, basins, forcing, and hyperparameters.
  The results go in `stage1_author_reproduction_results.json` and in
  `mips/camels_us_daily/leaderboards/appendix_stage1_confirmation_runs.json`.
- Every target number names its table, figure, or page in the paper.
- Runs that were smoke-tested with shortened training, or that fail Stage 1, stay off the
  leaderboards until they are fixed.

## Stop and ask a human if

- a model fits more than one category or leaderboard;
- you cannot tell which paper is canonical;
- an outside author's configuration is not clearly stated in their paper.
