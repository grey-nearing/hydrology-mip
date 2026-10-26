---
name: model-compliance-and-leaderboard-norms
description: >-
  Authoritative rules and compliance norms for model organization, provenance
  classification, single-citation attribution, Literature vs. Archive vs. MIP
  Re-run separation, competing Ensemble vs. Single-Model leaderboard rows
  (scored across all N seeds), Stage-1 author-protocol confirmation runs,
  12-leaderboard spatial/forcing routing, and clean-slate leaderboard JSON
  replacement in camels-us-mip. Use whenever auditing, adding, modifying,
  scoring, or rebuilding models and leaderboards.
---

# Model Organization, Provenance & Leaderboard Compliance Norms (`camels-us-mip`)

Every model family in [`leaderboards/table_2_registry.json`](../leaderboards/table_2_registry.json), every leaderboard row across the 12 canonical leaderboards (`leaderboards/<leaderboard_id>.json` and [`leaderboards/tables_3_to_14.json`](../leaderboards/tables_3_to_14.json)), every Stage-1 confirmation run in [`leaderboards/appendix_stage1_confirmation_runs.json`](../leaderboards/appendix_stage1_confirmation_runs.json), and every runnable model directory in [`models/`](../models/) must strictly comply with the following ten rules.

---

## 1. Strictly One Canonical Citation per Model Entry

1. **Single-Paper Attribution Rule:**
   - Every model entry in `leaderboards/table_2_registry.json` and every leaderboard row in `leaderboards/*__*.json` must cite **strictly one canonical paper** (`citation`, `paper_url`, `publication_year`).
   - **How to Choose the Canonical Citation:**
     1. Cite the paper that **first ran that model configuration over the Newman et al. (2017) 531-basin split**.
     2. If no prior paper ran the model over the Newman split, cite the **first paper to run that model on CAMELS-US data**.
     3. If the model is a general machine-learning or time-series architecture never previously run on CAMELS-US (e.g., `TiDE`), cite the **foundational architecture paper** (e.g., Das et al., 2023) and mark its provenance as an adaptation (`MIP Re-run (Our Code)`).
   - **Never Concatenate Multiple Citations:** Strings such as `"Newman et al. (2015, 2017); Nearing et al. (2020)"`, `"Mizukami et al. (2019), Rakovec et al. (2019)"`, or `"Addor et al. (2017), Thébault et al. (2026)"` are strictly forbidden.
2. **Splitting Distinct Papers or Configurations into Separate Model Entries:**
   - When different papers introduced genuinely distinct configurations, parameterizations, or versions of a base model, they **must** be represented as **separate model entries** in `table_2_registry.json` and on the leaderboards:
     - `newman_sacsma` (`Newman et al., 2017`, basin-calibrated `Archive*`) vs. `nearing_sacsma` (`Nearing et al., 2020`, re-implemented DDS & donor-transfer `MIP Re-run (Our Code)`).
     - `newman_vic` (`Newman et al., 2017`, basin-calibrated VIC 4.1.2 `Archive*`) vs. `mizukami_vic_mpr` (`Mizukami et al., 2017`, CONUS MPR regionalization `Archive`).
     - `mizukami_mhm` (`Mizukami et al., 2019`, basin-calibrated mHM `Archive*`) vs. `rakovec_mhm_mpr` (`Rakovec et al., 2019`, CONUS MPR regionalization `Archive*`).
     - `kratzert2019b_lstm` (`Kratzert et al., 2019b`, 270d single-forcing Maurer `Archive`) vs. `kratzert2019a_pub_lstm` (`Kratzert et al., 2019a`, 270d 12-fold PUB Maurer `Archive`) vs. `kratzert2021_lstm_archive` & `kratzert2021_lstm_rerun` (`Kratzert et al., 2021`, 365d single- and multi-forcing synergy LSTM).
     - `hoedt_mclstm_lit` & `hoedt_mclstm_rerun` (`Hoedt et al., 2021`, Maurer `Literature` and clean-room `MIP Re-run (Our Code)`) vs. `frame2023_mclstm_archive` (`Frame et al., 2023`, NLDAS & Daymet `Archive`).

---

## 2. Strictly One Provenance Category per Model Entry

1. **No Mixed Provenance Strings:**
   - Every entry in `leaderboards/table_2_registry.json` and every row in `leaderboards/*__*.json` must declare **strictly one** provenance category—never a compound string like `"Archive & MIP Re-run"` or `"Literature & MIP Re-run"`:
     - **`Archive`**: Publicly archived daily hydrographs (`(531, 3652)` over `1989-10-01` to `1999-09-30` on all 531 canonical basins) produced by the cited authors and scored directly by `metrics/evaluator.py`.
     - **`Archive*`**: Publicly archived daily hydrographs produced by the cited authors that cover fewer than 531 basins (e.g., 447, 489, 524, or 530 basins) or a non-standard time window (`asterisk: true`).
     - **`Literature`**: Published paper metrics reported directly on the canonical 531-basin Newman split (`1989-10-01` to `1999-09-30`) without a public daily hydrograph archive (only `NSE` is populated; the remaining 13 Gauch et al. 2023 metrics are `None` / `"---"`).
     - **`Literature*`**: Published paper metrics reported on the 531 Newman basins over a slightly modified evaluation window (e.g., the 9-water-year `WY1991–1999` window in `Feng et al., 2020` and `Jamaat et al., 2025`, `asterisk: true`).
     - **`MIP Re-run (Our Code)`**: Clean-room re-implementation (or daily/CAMELS domain adaptation) written in `models/<model_name>/`, verified via Stage-1 author-protocol reproduction, and executed on the standardized MIP protocol via `models/run_model.py`.
     - **`MIP Re-run (Author Code)`**: Direct execution of the original authors' unmodified open-source repository on the standardized MIP protocol.
     - **`Contribution`**: A new model architecture contributed directly to the CAMELS-US MIP with full `(531, 3652)` Zarr hydrographs in `gs://camels-us-mip/hydrographs/`.

---

## 3. Literature & Archive Priority Rule vs. MIP Re-runs

1. **Never Replace a Valid Newman-Split Literature or Archive Entry with a MIP Re-run:**
   - If a model was published on the Newman splits by a previous author in a given configuration (`gauged`, `pub`, or `pur` with a specific forcing), **we report the original `Literature` or `Archive` values** for that configuration and do **not** replace them with a MIP re-run. For `Literature` entries, we accept missing secondary statistics (`"---"` for non-NSE metrics) in order to preserve 100% fidelity to the original authors' published numbers.
2. **Separate Model Entry for Additional or Competing MIP Re-runs:**
   - When we include a verified clean-room `MIP Re-run (Our Code)` alongside a `Literature` entry on the same leaderboard to provide all 14 Gauch et al. (2023) hydrograph metrics, or when we run authorized expansions of a model, those runs **must live in a separate model entry** (`<family_id>_rerun`) with the same model name and canonical citation, marked `MIP Re-run (Our Code)`.
   - The `Literature`/`Archive` entry and the `MIP Re-run (Our Code)` entry are treated as separate, competing entries on the leaderboard and in Table 2.
3. **No Unauthored Configurations for External `MIP Re-run` Models ("If They Didn't Run It, We Don't Run It"):**
   - For `MIP Re-run` models from external authors, **only evaluate and include configurations (single-forcing vs. multi-forcing, Gauged vs. PUB/PUR) that the original authors actually defined and ran in their paper**.
   - Never invent ad-hoc multi-forcing adaptations (such as feeding 3-product mean precipitation into a single-input conceptual bucket) or run unauthored spatial splits for an external group's model.
   - The **only** exception is when Grey Nearing was an author or co-author on the original paper (e.g., Kratzert et al., 2019/2021; Nearing et al., 2020; Hoedt et al., 2021; Frame et al., 2023) or explicitly authorizes a specific evaluation.

---

## 4. Ensemble vs. Single-Model Runs as Separate, Competing Leaderboard Rows

1. **Ensembling Is Part of the Model Definition:**
   - From the perspective of the CAMELS-US MIP, the ensembling strategy ($N$-seed averaging vs. single-seed prediction) is an algorithmic component of the model.
   - Whenever a model architecture is evaluated with $N \ge 2$ random seeds (stored in `qsim_members` of shape `(N, 531, 3652)`), it appears as **two separate, competing rows** on the same leaderboard:
     1. **Ensemble Row (`is_ensemble: true`)**:
        - `entry_id`: `"<base_entry_id>"`
        - `model_name`: `"<Base Model Name> (<N>-model ensemble)"` (where `<N>` strictly equals `num_members`)
        - Scored and ranked on the ensemble-mean daily hydrograph `qsim = np.mean(qsim_members, axis=0)` across all 14 Gauch et al. (2023) metrics.
     2. **Single-Model Row (`is_ensemble: false`)**:
        - `entry_id`: `"<base_entry_id>_single"`
        - `model_name`: `"<Base Model Name>"` (with **no** `(N-model ensemble)` suffix)
        - Scored and ranked using the **mean ($\pm$ std) across all $N$ individual seed members** (`qsim_members[0 .. N-1]`) for **all 14 Gauch et al. (2023) metrics** and `nse_le_0`.
2. **Strict Prohibition on `seed 0` (`member[0]`) Single-Model Scoring:**
   - **Never** score or rank a Single-Model row from an $N$-seed Zarr store using `member[0]` (`seed 0`) alone. Scoring `seed 0` alone introduces random-seed luck noise and can cause a single seed to artificially outrank its own ensemble mean (e.g., `0.824 > 0.820`). Every Single-Model row backed by `qsim_members` ($N \ge 2$) must aggregate per-member metrics across all $N$ members (`evaluation_mode="single_member_mean"` in `metrics/evaluator.py`).
3. **Ensemble Display Suffix Hygiene:**
   - A row may include `(<N>-model ensemble)` in `model_name` **if and only if** `is_ensemble is True` and `num_members == N >= 2`.
   - Any row with `is_ensemble is False` or `num_members == 1` must **never** carry an `(<N>-model ensemble)` suffix.

---

## 5. Stage-1 Author-Protocol Confirmation Runs (`reproduce_author_metrics.py`)

1. **Purpose of Stage 1:**
   - Before any clean-room implementation (`MIP Re-run (Our Code)`) is run on the standardized Stage-2 MIP protocol (`1999–2008` train / `1989–1999` eval on all 531 basins), we run **Stage 1** (`models/<model_name>/reproduce_author_metrics.py`) to verify that our code reproduces the original paper's reported metrics when configured with the **original paper's exact temporal split, basin subset, forcing product, and hyperparameters**.
2. **Stage-1 Integrity Rules:**
   - **Never Overwrite Stage 1 with Stage-2 Newman-Split Runs:** If the original paper used a non-Newman temporal split (e.g., `Split-2C` `1995–2005` train / `2005–2014` test in Poudel & Steinschneider, 2026; `Split-3` `1980–1995` train / `1995–2010` test in Feng et al., 2022 and Song et al., 2024; or 471 basins in Yin et al., 2022), `reproduce_author_metrics.py` and `stage1_author_reproduction_results.json` **must** run and report that exact author split—never the Stage-2 Newman split.
   - **100% Verified Author Targets:** Every `author_reported_median_nse` (or `mean_nse` / `kge`) target in `reproduce_author_metrics.py` and `leaderboards/appendix_stage1_confirmation_runs.json` must be verified against a specific Table, Figure, or page number in the cited paper.
   - **Never Include Broken or Subsampled Runs in Leaderboards:** Any MIP re-run that was smoke-tested with subsampled epochs/batches (e.g., `< 2%` of batches per epoch) or that fails Stage-1 verification must be excluded from the Stage-2 leaderboards until properly trained.

---

## 6. Strict 12-Leaderboard Taxonomy & Anti-Leakage Routing Rules

Every leaderboard row must be placed in the exact leaderboard matching its spatial generalization regime and input/forcing regime:

| Spatial Regime | Single CAMELS Forcing (`single_camels`: Daymet, Maurer, or NLDAS) | Multi-CAMELS Forcing (`multi_camels`: $\ge 2$ of Daymet, Maurer, NLDAS) | Lagged Streamflow / Data Assimilation (`lagged_streamflow`: $Q(t-k)$, AR, DA) | Freeform / External Inputs (`freeform`: NLDAS-2, AORC, ERA5, or auxiliary flow quantiles) |
| :--- | :---: | :---: | :---: | :---: |
| **Gauged (`gauged`)** | **Table 3** (`gauged__single_camels`) | **Table 4** (`gauged__multi_camels`) | **Table 5** (`gauged__lagged_streamflow`) | **Table 6** (`gauged__freeform`) |
| **12-Fold PUB (`pub_12fold`)** | **Table 7** (`pub__single_camels`) | **Table 8** (`pub__multi_camels`) | **Table 9** (`pub__lagged_streamflow`) | **Table 10** (`pub__freeform`) |
| **18-Fold PUR (`pur_18huc02`)** | **Table 11** (`pur__single_camels`) | **Table 12** (`pur__multi_camels`) | **Table 13** (`pur__lagged_streamflow`) | **Table 14** (`pur__freeform`) |

1. **Daily Core CAMELS Scope Only:**
   - The CAMELS-US MIP evaluates daily (`1D`) streamflow prediction using the core daily CAMELS-US dataset. Sub-daily models requiring hourly forcing and streamflow data that do not exist for all 531 CAMELS basins (e.g., `sMTS-LSTM`, Gauch et al., 2021a) are out of scope and excluded from the repository, leaderboards, and GCS bucket.
2. **Definition of Lagged-Streamflow PUB (`T9`) & PUR (`T13`):**
   - Following Nearing et al. (2022, *HESS* 26, 5493–5513, Section 2.4 & 3.3), `pub__lagged_streamflow` (`Table 9`) and `pur__lagged_streamflow` (`Table 13`) evaluate models where the target basins or regions are **held out (ungauged) during model training** (`1999–2008`), while lagged observed streamflow $Q(t-k)$ becomes available during inference (`1989–1999`) via autoregression, data integration, or data assimilation.
   - Standard simulation leaderboards (`T7`, `T8`, `T10`, `T11`, `T12`, `T14`) strictly prohibit any observed streamflow inputs during testing.
3. **Exclusion of Non-Conforming External Retrospective Archives (`NOAA NWM`):**
   - External retrospective archives that were neither calibrated on the Newman `WY2000–2008` training period nor evaluated with a uniform gauged/ungauged split across the 531 CAMELS basins (such as `NOAA NWM v2.0, v2.1, v3.0`, which were calibrated on `WY2008–2013` / `2007–2019` across 480 of 530 mapped basins) are excluded from all leaderboards, the repository, and the GCS bucket.
4. **Auxiliary Non-CAMELS Static Attributes Route to `freeform` (`T6`):**
   - Models that ingest hydrograph-derived flow quantiles (`Q5`, `Q95`) or external datasets not included in the standard 27 Kratzert et al. (2019b) static catchment attributes belong in `gauged__freeform` (`T6`), not `gauged__single_camels` (`T3`).
5. **Mandatory Asterisk (`*`) Flagging (`asterisk: true`):**
   - Any `Archive*` row with fewer than 531 evaluated basins (e.g., `530`, `501`, `498`, `492`, or `479` of `531` basins) or any `Literature*` / `Archive*` row evaluated on a non-standard time window (e.g., `WY1991–1999` or Split-3) **must** set `asterisk: true`, provide an exact model-specific `asterisk_reason` footnote, and carry a trailing `*` in its provenance badge and display name.

---

## 7. Complete Clean-Slate Leaderboard JSON Replacement Rule

Whenever executing a major registry or leaderboard rebuild:
1. **Never Incrementally Patch Stale JSON Files:**
   - Do not mutate existing `leaderboards/*.json` files in place via ad-hoc regexes or partial row edits that risk leaving behind orphaned, mis-cited, or mis-scored legacy entries.
2. **Rebuild Atomically from the Verified Spec & GCS Zarr Stores:**
   - Construct `leaderboards/table_2_registry.json`, all 12 `leaderboards/<leaderboard_id>.json` files, `leaderboards/tables_3_to_14.json`, and `leaderboards/appendix_stage1_confirmation_runs.{json,md}` **from scratch** in a single deterministic pass from:
     1. The verified per-model specifications (`audit_reports/round2_specs/*.json`),
     2. Live cloud evaluations of `gs://camels-us-mip/hydrographs/<leaderboard_id>/<entry_id>.zarr` via `metrics/evaluator.py`, and
     3. Active `Contribution` entries (e.g., ongoing ERA and TFT entries).
   - After writing the clean JSON files, always regenerate all paper figures (`generate_all_paper_figures()`) and run the automated compliance verification suite.
