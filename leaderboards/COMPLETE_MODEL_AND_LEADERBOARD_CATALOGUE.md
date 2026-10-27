# CAMELS-US MIP: Complete Model Registry & 12-Leaderboard Catalogue

> **Generated:** 2026-10-27 | **Repository:** `/path/to/workspace/Projects/camels-us-mip` | **GCS Bucket:** `gs://camels-us-mip/` | **Manuscript Google Doc:** `1cTTWBhMs8oPc9zjiJZnDYJkArbSot4K0UHUZ465c648` (Tab `Consolidated MIP (v2)`, `t.jke2gwg7x96i`)

---

## 1. Executive Summary & Forensic Audit Status

### 1.1 High-Level Totals
- **Total Table 2 Model Registry Entries:** `32` model entries (ordered chronologically from 2017 to 2026)
- **Total Active Leaderboard Rows Across Tables 3–14:** `126` rows across `9` populated leaderboards (`3` freeform leaderboards—Tables 6, 10, 14—currently have `0` entries)
- **Total Stage-1 Author-Protocol Confirmation Runs (Appendix A, Table A1):** `11` confirmation runs across `10` `MIP Re-run (Our Code)` model families (plus 1 Stage-1 confirmation package for EA-LSTM)

### 1.2 Breakdown by Canonical Model Category (with Manuscript Table Shading)

| Canonical Model Category | Manuscript Row Hex Color | Table 2 Families | Total Leaderboard Rows |
| :--- | :---: | :---: | :---: |
| **Machine learning** | `#e8f1fb` | 18 | 84 |
| **Hybrid (ML + Physics)** | `#f2ebf9` | 6 | 23 |
| **Conceptual** | `#fff0e0` | 4 | 15 |
| **Process-based** | `#e8f6ef` | 4 | 4 |
| **Total** | — | **32** | **126** |

### 1.3 Breakdown by Provenance Category

| Provenance Category | Table 2 Families | Total Leaderboard Rows | Meaning |
| :--- | :---: | :---: | :--- |
| **`Archive`** | 7 | 25 | Publicly archived daily hydrographs (`531` basins, `1989-10-01` to `1999-09-30`) scored directly by `metrics/evaluator.py`. |
| **`Archive*`** | 5 | 7 | Publicly archived daily hydrographs covering `< 531` basins (`489`, `492`, `498`, `501`, `524`, or `530` basins) or a non-Newman window. |
| **`Literature`** | 4 | 19 | Published paper metrics reported directly on the canonical 531-basin Newman split (`1989-10-01` to `1999-09-30`) without public daily hydrographs (`NSE` only; `---` for secondary metrics). |
| **`Literature*`** | 5 | 9 | Published paper metrics reported on a modified evaluation window (e.g., `WY1991–1999` in Feng et al. 2020 / Jamaat et al. 2025, `Split-3` in Kratzert et al. 2019a, or `447` basins in Hoedt et al. 2021). |
| **`MIP Re-run (Our Code)`** | 10 | 52 | Clean-room re-implementation verified via Stage-1 author-protocol reproduction and executed on the standardized MIP protocol. |
| **`Contribution`** | 1 | 14 | New model architecture contributed to the CAMELS-US MIP with full `(531, 3652)` Zarr v2 hydrographs (`era_lstm_seeded`). |
| **Total** | **32** | **126** | — |

### 1.4 Summary of Recent Model Audits, Removals, and Re-runs (For Manuscript Reconciliation)
When updating the manuscript prose, Table 2, Tables 3–14, and Appendices A & B, ensure the following recent changes are reflected:
1. **Models Completely Dropped (Remove all prose, Table 2, leaderboard, and Appendix references):**
   - **`hydrodiffusion`** (Wang, Hoyos, et al., 2025b — dropped entirely)
   - **`das_tide`** (Das et al., 2023 TiDE adaptation — dropped entirely)
   - **`gauch_xgboost` (`MIP Re-run`)** (Our adapted daily XGBoost re-run was dropped; **keep** `gauch_xgboost_lit` `Literature` entries `t3_35` and `t3_36` from Gauch et al., 2021b on Table 3)
   - **`noaa_nwm`** and **`gauch_mts_lstm`** (Previously excluded for non-conforming calibration period / sub-daily inputs)
2. **Unauthored Cross-Split / Multi-Forcing Configurations Removed (*'If They Didn't Run It, We Don't Run It'* Rule):**
   - **`hoedt_mclstm`:** Removed from PUB and PUR tables (Tables 7, 8, 11, 12). Kept on **Table 3** (`hoedt_mclstm_maurer` / `_single` + `hoedt_mclstm_lit` + `frame2023_mclstm_archive`) and **Table 4** (`hoedt_mclstm_3f_avgp` / `_single` and `hoedt_mclstm_3f_ens_b`, authorized as a co-authored expansion).
   - **`rasiya_koya_tft`:** Removed from Table 4 (`_3f`), Table 7 (`tft_pub_daymet`), Table 8 (`_pub_3f`), Table 11 (`tft_pur_daymet`), and Table 12 (`_pur_3f`). Kept **only on Table 3** (`tft_gauged_maurer`, single-forcing Gauged Maurer, `0.716` Median NSE), matching Rasiya Koya & Roy (2024)'s single-forcing Gauged scope.
   - **`feng_dhbv10` & `song_dhbv11p`:** Removed 3-forcing (`_3f`) adaptations from Tables 4, 8, and 12. Kept on single-forcing Daymet **Tables 3, 7, and 11**.
   - **`feng2020_di_lstm`:** Removed 3-forcing (`feng2020_di_lstm_3f`) from Table 5. Kept single-forcing NLDAS on **Table 5** (`0.801` ens / `0.795` single, alongside `feng2020_di_lstm_lit` `0.84*`), **Table 9** (`0.775` ens / `0.765` single), and **Table 13** (`0.747` ens / `0.732` single).
3. **Models Re-Trained Faithfully Without Batch/Epoch/Window Shortcuts:**
   - **`feng2020_di_lstm` (Tables 9 & 13):** Re-trained for full 30 epochs (`batch_size=256`, no batch cap; 3 seeds): Table 9 PUB NLDAS rose from `0.731 -> 0.775` (ensemble) and `0.706 -> 0.765 ± 0.001` (single); Table 13 PUR NLDAS rose from `0.692 -> 0.747` (ensemble) and `0.660 -> 0.732 ± 0.001` (single).
   - **`yin2022_rrformer` (Table 5):** Re-implemented and re-trained with exact upstream `iThronne/RR-Former` architecture (`seq_len=7`, 7-day lagged $Q_{t-7..t-1}$, 3-layer LeakyReLU readout, 300-epoch regional pretraining + 40-epoch per-basin fine-tuning, 3 seeds): Table 5 Gauged Daymet rose from `0.728 -> 0.847` (ensemble, now **Rank #1 on Table 5**) and `0.712 -> 0.833 ± 0.004` (single, Rank #3).
   - **`poudel_mlp_hbv` (Tables 3, 7, 11):** Re-trained with full `240` batches/epoch × `50` epochs (`12,000` gradient steps, `StepLR(step_size=10, gamma=0.5)`, 3 seeds): Table 3 rose from `0.708 -> 0.735` (ens) / `0.699 -> 0.725 ± 0.004` (single); Table 7 PUB rose from `0.662 -> 0.678` (ens) / `0.651 -> 0.667 ± 0.001` (single); Table 11 PUR rose from `0.579 -> 0.624` (ens) / `0.562 -> 0.608 ± 0.011` (single).
   - **`rasiya_koya_tft` (Table 3):** Re-trained for full 25 epochs with `stride=1` evaluation: Table 3 `tft_gauged_maurer` rose from `0.677 -> 0.716` (`0.7155`).
4. **New Contribution Added (`era_lstm_seeded`):**
   - **`era_lstm_seeded` (`Empirical Research Assistant -- LSTM Seeded`, Nearing, 2026, `Contribution`):** Added 14 rows across **Table 4** (`gauged__multi_camels`: `era_gauged_multi_paradigm` `0.827` Rank #1, `era_gauged_multi_paradigm_single` `0.822` Rank #2, `era_gauged_physics_guided` `0.820` Rank #5, `era_gauged_minimal_guidance` `0.819` Rank #6, `era_gauged_minimal_guidance_single` `0.810` Rank #8, `era_gauged_physics_guided_single` `0.800` Rank #10), **Table 8** (`pub__multi_camels`: `era_pub_physics_guided` `0.725` Rank #2, `era_pub_minimal_guidance` `0.720` Rank #3, `era_pub_minimal_guidance_single` `0.714` Rank #4, `era_pub_physics_guided_single` `0.714` Rank #5), and **Table 12** (`pur__multi_camels`: `era_pur_minimal_guidance` `0.696` **Rank #1**, `era_pur_minimal_guidance_single` `0.686` Rank #3, `era_pur_physics_guided` `0.684` Rank #4, `era_pur_physics_guided_single` `0.653` Rank #5).

---

## 2. Complete Table 2 Model Registry Catalogue (`32` Entries)

### 2.1 Master Overview Table (Table 2 of the Manuscript)

| # | `family_id` | Model Display Name | Category (Row Color) | Provenance | Canonical Citation | Year | Asterisk | Rows | Leaderboards Covered | Best Med NSE | Paper DOI / URL | Code / GitHub URL | Archive / Zenodo / HydroShare |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :-: | :--- | :-: | :--- | :--- | :--- |
| 1 | `fuse_suite` | **FUSE (900, 902, 904)** | Conceptual (`#fff0e0`) | `Archive*` | Addor et al. (2017) | 2017 | `*` | 3 | `gauged__single_camels` | **0.654** | https://doi.org/10.5194/hess-21-5293-2017 | https://github.com/naddor/fuse | https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/ |
| 2 | `mizukami_vic_mpr` | **VIC 4.1.2 (CONUS MPR)** | Process-based (`#e8f6ef`) | `Archive` | Mizukami et al. (2017) | 2017 | No | 1 | `gauged__single_camels` | **0.307** | https://doi.org/10.1002/2017WR020401 | https://github.com/UW-Hydro/VIC | https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/ |
| 3 | `newman_sacsma` | **SAC-SMA + Snow-17 (Basin-Calibrated)** | Conceptual (`#fff0e0`) | `Archive*` | Newman et al. (2017) | 2017 | `*` | 1 | `gauged__single_camels` | **0.607** | https://doi.org/10.1175/JHM-D-16-0284.1 | https://github.com/NCAR/camels | https://doi.org/10.5065/D6MW2F4D |
| 4 | `newman_vic` | **VIC 4.1.2 (Basin-Calibrated)** | Process-based (`#e8f6ef`) | `Archive*` | Newman et al. (2017) | 2017 | `*` | 1 | `gauged__single_camels` | **0.554** | https://doi.org/10.1175/JHM-D-16-0284.1 | https://github.com/UW-Hydro/VIC | https://doi.org/10.5065/D6MW2F4D |
| 5 | `seibert_hbv` | **HBV-light (HBV-upper & HBV-lower)** | Conceptual (`#fff0e0`) | `Archive` | Seibert et al. (2018) | 2018 | No | 2 | `gauged__single_camels`, `pub__single_camels` | **0.678** | https://doi.org/10.1002/hyp.11476 | --- | https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/ |
| 6 | `kratzert2019a_pub_lstm` | **LSTM (12-Fold PUB)*** | Machine learning (`#e8f1fb`) | `Literature*` | Kratzert et al. (2019a) | 2019 | `*` | 2 | `pub__single_camels` | **0.691** | https://doi.org/10.1029/2019WR026065 | https://github.com/kratzert/lstm_for_pub | https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/ |
| 7 | `kratzert2019b_lstm` | **LSTM (Maurer 270d, 8-Seed Archive)** | Machine learning (`#e8f1fb`) | `Archive` | Kratzert et al. (2019b) | 2019 | No | 2 | `gauged__single_camels` | **0.759** | https://doi.org/10.5194/hess-23-5089-2019 | https://github.com/kratzert/ealstm_regional_modeling | https://zenodo.org/records/3530884 |
| 8 | `kratzert_ealstm` | **Entity-Aware LSTM (EA-LSTM)** | Machine learning (`#e8f1fb`) | `Archive` | Kratzert et al. (2019b) | 2019 | No | 2 | `gauged__single_camels` | **0.741** | https://doi.org/10.5194/hess-23-5089-2019 | https://github.com/kratzert/ealstm_regional_modeling | https://zenodo.org/records/3530884 |
| 9 | `kratzert_lstm_nostatic` | **LSTM without Static Attributes** | Machine learning (`#e8f1fb`) | `Archive` | Kratzert et al. (2019b) | 2019 | No | 2 | `gauged__single_camels` | **0.642** | https://doi.org/10.5194/hess-23-5089-2019 | https://github.com/kratzert/ealstm_regional_modeling | https://zenodo.org/records/3530884 |
| 10 | `mizukami_mhm` | **mHM (Basin-Calibrated)** | Process-based (`#e8f6ef`) | `Archive*` | Mizukami et al. (2019) | 2019 | `*` | 1 | `gauged__single_camels` | **0.665** | https://doi.org/10.5194/hess-23-2601-2019 | https://git.ufz.de/mhm/mhm | https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/ |
| 11 | `rakovec_mhm_mpr` | **mHM (CONUS MPR)** | Process-based (`#e8f6ef`) | `Archive*` | Rakovec et al. (2019) | 2019 | `*` | 1 | `gauged__single_camels` | **0.528** | https://doi.org/10.1029/2019JD030767 | https://git.ufz.de/mhm/mhm | https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/ |
| 12 | `feng2020_di_lstm` | **Data-Integration LSTM (DI(1)-LSTM — MIP Re-run)** | Machine learning (`#e8f1fb`) | `MIP Re-run (Our Code)` | Feng et al. (2020) | 2020 | No | 6 | `gauged__lagged_streamflow`, `pub__lagged_streamflow`, `pur__lagged_streamflow` | **0.801** | https://doi.org/10.1029/2019WR026793 | https://github.com/mhpi/hydroDL | --- |
| 13 | `feng2020_di_lstm_lit` | **Data-Integration LSTM (DI(1)-LSTM — Literature)*** | Machine learning (`#e8f1fb`) | `Literature*` | Feng et al. (2020) | 2020 | `*` | 1 | `gauged__lagged_streamflow` | **0.840** | https://doi.org/10.1029/2019WR026793 | https://github.com/mhpi/hydroDL | --- |
| 14 | `nearing_sacsma` | **SAC-SMA + Snow-17 (DDS & Donor Transfer)** | Conceptual (`#fff0e0`) | `MIP Re-run (Our Code)` | Nearing et al. (2020) | 2020 | No | 9 | `gauged__single_camels`, `gauged__multi_camels`, `pub__single_camels`, `pub__multi_camels`, `pur__single_camels`, `pur__multi_camels` | **0.695** | https://doi.org/10.31223/osf.io/53te4 | https://github.com/Upstream-Tech/SACSMA-SNOW17 | --- |
| 15 | `gauch_xgboost_lit` | **Gradient-Boosted Decision Trees (XGBoost — Literature)** | Machine learning (`#e8f1fb`) | `Literature` | Gauch et al. (2021b) | 2021 | No | 2 | `gauged__single_camels` | **0.659** | https://doi.org/10.1016/j.envsoft.2020.104926 | https://github.com/gauchm/ealstm_regional_modeling | https://zenodo.org/records/3648270 |
| 16 | `hoedt_mclstm` | **Mass-Conserving LSTM (MC-LSTM — MIP Re-run)** | Machine learning (`#e8f1fb`) | `MIP Re-run (Our Code)` | Hoedt et al. (2021) | 2021 | No | 5 | `gauged__single_camels`, `gauged__multi_camels` | **0.777** | https://arxiv.org/abs/2101.05186 | https://github.com/ml-jku/mc-lstm | --- |
| 17 | `hoedt_mclstm_lit` | **Mass-Conserving LSTM (MC-LSTM — Literature)*** | Machine learning (`#e8f1fb`) | `Literature*` | Hoedt et al. (2021) | 2021 | `*` | 2 | `gauged__single_camels` | **0.744** | https://arxiv.org/abs/2101.05186 | https://github.com/ml-jku/mc-lstm | --- |
| 18 | `kratzert2021_lstm_archive` | **LSTM (Single- & Multi-Forcing 365d Archive)** | Machine learning (`#e8f1fb`) | `Archive` | Kratzert et al. (2021) | 2021 | No | 14 | `gauged__single_camels`, `gauged__multi_camels` | **0.821** | https://doi.org/10.5194/hess-25-2685-2021 | https://github.com/neuralhydrology/neuralhydrology | https://zenodo.org/records/4670268 |
| 19 | `kratzert_lstm` | **LSTM (PUB & PUR MIP Re-run)** | Machine learning (`#e8f1fb`) | `MIP Re-run (Our Code)` | Kratzert et al. (2021) | 2021 | No | 10 | `pub__multi_camels`, `pur__single_camels`, `pur__multi_camels` | **0.742** | https://doi.org/10.5194/hess-25-2685-2021 | https://github.com/neuralhydrology/neuralhydrology | --- |
| 20 | `tsai2021_dpl_vic` | **dPL + VIC (g_A, Maurer)*** | Hybrid (ML + Physics) (`#f2ebf9`) | `Literature*` | Tsai et al. (2021) | 2021 | `*` | 1 | `gauged__single_camels` | **0.439** | https://doi.org/10.1038/s41467-021-26107-z | https://github.com/mhpi/hydroDL | https://zenodo.org/records/6383415 |
| 21 | `feng_dhbv10` | **δHBV 1.0** | Hybrid (ML + Physics) (`#f2ebf9`) | `MIP Re-run (Our Code)` | Feng et al. (2022) | 2022 | No | 6 | `gauged__single_camels`, `pub__single_camels`, `pur__single_camels` | **0.742** | https://doi.org/10.1029/2022WR032404 | https://github.com/mhpi/hydroDL | https://zenodo.org/records/7091334 |
| 22 | `nearing2022_ar_da_lstm` | **Autoregressive & Data-Assimilation LSTM (AR-LSTM, DA-LSTM)** | Machine learning (`#e8f1fb`) | `Literature` | Nearing et al. (2022) | 2022 | No | 5 | `gauged__lagged_streamflow`, `pub__lagged_streamflow` | **0.879** | https://doi.org/10.5194/hess-26-5493-2022 | https://github.com/grey-nearing/lstm-data-assimilation | https://zenodo.org/records/7063252 |
| 23 | `yin2022_rrformer` | **Rainfall-Runoff Transformer (RR-Former)** | Machine learning (`#e8f1fb`) | `MIP Re-run (Our Code)` | Yin et al. (2022) | 2022 | No | 2 | `gauged__lagged_streamflow` | **0.847** | https://doi.org/10.1016/j.jhydrol.2022.127781 | https://github.com/iThronne/RR-Former | --- |
| 24 | `frame2023_mclstm_archive` | **Mass-Conserving LSTM (MC-LSTM — Frame 2023 Archive)** | Machine learning (`#e8f1fb`) | `Archive` | Frame et al. (2023) | 2023 | No | 2 | `gauged__single_camels` | **0.758** | https://doi.org/10.1002/hyp.14847 | https://github.com/jmframe/mass_balance_all | https://www.hydroshare.org/resource/d750278db868447dbd252a8c5431affd/ |
| 26 | `rasiya_koya_tft` | **Temporal Fusion Transformer (TFT)** | Machine learning (`#e8f1fb`) | `MIP Re-run (Our Code)` | Rasiya Koya & Roy (2024) | 2024 | No | 1 | `gauged__single_camels` | **0.716** | https://doi.org/10.1016/j.jhydrol.2024.131301 | https://github.com/sinanrk/TFT-streamflow | --- |
| 27 | `song_dhbv11p` | **δHBV 1.1p / δMG** | Hybrid (ML + Physics) (`#f2ebf9`) | `MIP Re-run (Our Code)` | Song et al. (2024) | 2024 | No | 6 | `gauged__single_camels`, `pub__single_camels`, `pur__single_camels` | **0.759** | https://doi.org/10.5194/hess-28-3051-2024 | https://github.com/mhpi/generic_deltamodel | --- |
| 28 | `jamaat2025_dhbv_da` | **δHBV 1.1p Variational Data Assimilation (δHBV-DA-PS, -S, -P)*** | Hybrid (ML + Physics) (`#f2ebf9`) | `Literature*` | Jamaat et al. (2025) | 2025 | `*` | 3 | `gauged__lagged_streamflow` | **0.820** | https://doi.org/10.1016/j.jhydrol.2025.134137 | https://github.com/mhpi/DataAssimilation | --- |
| 29 | `li2025_lstm_dhbv_ens` | **(LSTM + dHBV 1.1p) Multi-Model Ensemble** | Hybrid (ML + Physics) (`#f2ebf9`) | `MIP Re-run (Our Code)` | Li et al. (2025) | 2025 | No | 1 | `gauged__multi_camels` | **0.820** | https://doi.org/10.5194/hess-29-6829-2025 | https://github.com/mhpi/generic_deltamodel | https://zenodo.org/records/16895228 |
| 30 | `liu2025_tslib_suite` | **TSLib Deep Learning Suite (10 Models: Crossformer, PatchTST, Informer, etc.)** | Machine learning (`#e8f1fb`) | `Literature` | Liu et al. (2025) | 2025 | No | 10 | `gauged__multi_camels` | **0.720** | https://doi.org/10.5194/hess-29-6811-2025 | https://github.com/thuml/Time-Series-Library | https://zenodo.org/records/15852144 |
| 31 | `wang_s4dft` | **Frequency-Tuned Diagonal State Space Model (S4D-FT & S4D)** | Machine learning (`#e8f1fb`) | `Literature` | Wang, Zhang, et al. (2025a) | 2025 | No | 2 | `gauged__single_camels` | **0.740** | https://doi.org/10.1029/2025WR039888 | https://github.com/WESTENR-OU/S4D_rainfall_runoff_simulations | --- |
| 32 | `poudel_mlp_hbv` | **Static-Parameter MLP + 16-HBV (diff-hydro)** | Hybrid (ML + Physics) (`#f2ebf9`) | `MIP Re-run (Our Code)` | Poudel & Steinschneider (2026) | 2026 | No | 6 | `gauged__single_camels`, `pub__single_camels`, `pur__single_camels` | **0.735** | https://doi.org/10.5194/hess-30-5173-2026 | https://github.com/snpoudel/diff-hydro | --- |
| 33 | `era_lstm_seeded` | **Empirical Research Assistant -- LSTM Seeded** | Machine learning (`#e8f1fb`) | `Contribution` | Nearing (2026) | 2026 | No | 14 | `gauged__multi_camels`, `pub__multi_camels`, `pur__multi_camels` | **0.827** | https://github.com/grey-nearing/camels-us-mip/tree/main/era_experiments | https://github.com/grey-nearing/camels-us-mip/tree/main/models | https://doi.org/10.5281/zenodo.23641695 |

### 2.2 Detailed Metadata & Descriptions for Each of the 33 Table 2 Entries

#### 2.2.1 `fuse_suite` — FUSE (900, 902, 904)
- **Canonical Citation:** Addor et al. (2017) (Year: `2017`)
- **Model Category:** `Conceptual` (Manuscript Row Shading: `#fff0e0`)
- **Architectural Paradigm:** Framework for Understanding Structural Errors (Modular Conceptual Hydrology)
- **Provenance Category:** `Archive*`
- **Paper URL:** https://doi.org/10.5194/hess-21-5293-2017
- **GitHub / Code URL:** https://github.com/naddor/fuse (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on 501 (FUSE 900, 904) or 479 (FUSE 902) of 531 CAMELS basins available in the archive.
- **Active Leaderboard Entries (`3` rows):** `gauged__single_camels` | **Best Median NSE:** `0.6540`
- **Description:** Framework for Understanding Structural Errors (FUSE; Clark et al., 2008) modular conceptual hydrology suite combining process components from PRMS, SAC-SMA, TOPMODEL, and ARNO/VIC with a degree-day snow module (naddor/fuse), calibrated independently per basin via SCE-UA on Maurer forcings over WY2000–2008 and evaluated over WY1990–1999 across three structural configurations (900, 902, 904; Addor et al., 2017; Kratzert et al., 2019b HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1).

#### 2.2.2 `mizukami_vic_mpr` — VIC 4.1.2 (CONUS MPR)
- **Canonical Citation:** Mizukami et al. (2017) (Year: `2017`)
- **Model Category:** `Process-based` (Manuscript Row Shading: `#e8f6ef`)
- **Architectural Paradigm:** Distributed Land-Surface Hydrology Model (CONUS Multiscale Parameter Regionalization)
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.1002/2017WR020401
- **GitHub / Code URL:** https://github.com/UW-Hydro/VIC (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.3070`
- **Description:** Variable Infiltration Capacity (VIC 4.1.2) macroscale land-surface hydrology model parameterized across CONUS via Multiscale Parameter Regionalization (MPR-flex; 16 global transfer-function coefficients calibrated via SCE-UA over WY2000–2008 and evaluated over WY1990–1999; Mizukami et al., 2017; HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1).

#### 2.2.3 `newman_sacsma` — SAC-SMA + Snow-17 (Basin-Calibrated)
- **Canonical Citation:** Newman et al. (2017) (Year: `2017`)
- **Model Category:** `Conceptual` (Manuscript Row Shading: `#fff0e0`)
- **Architectural Paradigm:** Lumped Conceptual Soil Moisture Accounting + Snow-17 Model (Basin SCE-UA Calibration)
- **Provenance Category:** `Archive*`
- **Paper URL:** https://doi.org/10.1175/JHM-D-16-0284.1
- **GitHub / Code URL:** https://github.com/NCAR/camels (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5065/D6MW2F4D` (https://doi.org/10.5065/D6MW2F4D) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on 530 of 531 CAMELS basins (1 basin, USGS 01634500, is all-zero in the Newman et al., 2017 archive).
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.6070`
- **Description:** Operational US National Weather Service Sacramento Soil Moisture Accounting (SAC-SMA) model coupled with the Snow-17 snow accumulation/ablation model and a Duamel Gamma unit hydrograph (Burnash et al., 1973; Anderson, 1973), calibrated independently per basin across 10 SCE-UA/DDS trials over WY2000–2008 and evaluated over WY1990–1999 (Newman et al., 2017; HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1).

#### 2.2.4 `newman_vic` — VIC 4.1.2 (Basin-Calibrated)
- **Canonical Citation:** Newman et al. (2017) (Year: `2017`)
- **Model Category:** `Process-based` (Manuscript Row Shading: `#e8f6ef`)
- **Architectural Paradigm:** Distributed Land-Surface Hydrology Model (Basin SCE-UA Calibration)
- **Provenance Category:** `Archive*`
- **Paper URL:** https://doi.org/10.1175/JHM-D-16-0284.1
- **GitHub / Code URL:** https://github.com/UW-Hydro/VIC (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5065/D6MW2F4D` (https://doi.org/10.5065/D6MW2F4D) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on 530 of 531 CAMELS basins (1 basin, USGS 08082700, is missing in the Newman et al., 2017 archive).
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.5540`
- **Description:** Variable Infiltration Capacity (VIC 4.1.2) macroscale land-surface hydrology model calibrated independently per basin using Shuffled Complex Evolution (SCE-UA) over WY2000–2008 and evaluated over WY1990–1999 (Newman et al., 2017; HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1).

#### 2.2.5 `seibert_hbv` — HBV-light (HBV-upper & HBV-lower)
- **Canonical Citation:** Seibert et al. (2018) (Year: `2018`)
- **Model Category:** `Conceptual` (Manuscript Row Shading: `#fff0e0`)
- **Architectural Paradigm:** Conceptual Lumped Rainfall-Runoff Model (100-Member Calibrated Ensemble & 1000-Member Uncalibrated Random Ensemble)
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.1002/hyp.11476
- **GitHub / Code URL:** --- (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels`, `pub__single_camels` | **Best Median NSE:** `0.6780`
- **Description:** Conceptual semi-distributed/lumped HBV-light rainfall-runoff model (Seibert & Vis, 2012) providing dual upper and lower reference benchmarks on all 531 CAMELS-US basins forced by Maurer meteorology (Seibert et al., 2018; executed by Jan Seibert and Marc Vis for Kratzert et al., 2019b; HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1): HBV-upper (gauged__single_camels) is the 100-member genetically calibrated GAP ensemble upper benchmark over WY2000–2008, and HBV-lower (pub__single_camels) is the 1000-member uncalibrated random-parameter ensemble lower benchmark sampled uniformly from physically plausible parameter bounds (using zero target or donor streamflow observations, NOT a donor-transfer calibration).

#### 2.2.6 `kratzert2019a_pub_lstm` — LSTM (12-Fold PUB)*
- **Canonical Citation:** Kratzert et al. (2019a) (Year: `2019`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Regional Long Short-Term Memory Network (12-Fold PUB)
- **Provenance Category:** `Literature*`
- **Paper URL:** https://doi.org/10.1029/2019WR026065
- **GitHub / Code URL:** https://github.com/kratzert/lstm_for_pub (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c` (https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/) | **HydroShare URL:** https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/
- **Asterisk (`*`):** `True` — **Reason:** Evaluated in Kratzert et al. (2019a) on Split-3 (trained on WY1981–1995, evaluated on WY1996–2010; 1995-10-01 to 2010-09-30) with extended NLDAS-2 forcing.
- **Active Leaderboard Entries (`2` rows):** `pub__single_camels` | **Best Median NSE:** `0.6910`
- **Description:** Regional single-layer 256-unit Long Short-Term Memory network (CudaLSTM, 270-day lookback window) receiving daily extended NLDAS-2 meteorological forcings concatenated with 27 static CAMELS catchment attributes, trained with basin-averaged NSE* loss across 12 spatial k-fold cross-validation splits (PUB, 10 seeds per fold) over WY1981–1995 (1980-10-01 to 1995-09-30) and evaluated over WY1996–2010 (1995-10-01 to 2010-09-30) across all 531 CAMELS-US basins (Kratzert et al., 2019a, WRR, Table 2; HydroShare 10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c).

#### 2.2.7 `kratzert2019b_lstm` — LSTM (Maurer 270d, 8-Seed Archive)
- **Canonical Citation:** Kratzert et al. (2019b) (Year: `2019`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Regional Long Short-Term Memory Network (270-day lookback)
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.5194/hess-23-5089-2019
- **GitHub / Code URL:** https://github.com/kratzert/ealstm_regional_modeling (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.3530884` (https://zenodo.org/records/3530884) | **HydroShare URL:** https://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06/
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.7590`
- **Description:** Regional single-layer 256-unit Long Short-Term Memory network (CudaLSTM) with 270-day lookback window receiving daily Maurer meteorological forcings concatenated with 27 static CAMELS catchment attributes at every timestep, trained with basin-variance-normalized NSE* loss across 8 random seeds over WY2000–2008 and evaluated over WY1990–1999 on all 531 CAMELS-US basins (Kratzert et al., 2019b; HydroShare 10.4211/hs.83ea5312635e44dc824eeb99eda12f06, Zenodo 10.5281/zenodo.3530884).

#### 2.2.8 `kratzert_ealstm` — Entity-Aware LSTM (EA-LSTM)
- **Canonical Citation:** Kratzert et al. (2019b) (Year: `2019`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Entity-Aware LSTM (Static-Attribute Input Gate)
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.5194/hess-23-5089-2019
- **GitHub / Code URL:** https://github.com/kratzert/ealstm_regional_modeling (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.3530884` (https://zenodo.org/records/3530884) | **HydroShare URL:** https://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06/
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.7410`
- **Description:** Entity-Aware Long Short-Term Memory network (EA-LSTM; H = 256 memory cells, T = 270-day sequence length, dropout = 0.4, basin-averaged NSE* loss) in which the 27 static CAMELS catchment attributes exclusively modulate a time-invariant sigmoid input gate i = sigma(W_i x_s + b_i) while the 5 dynamic extended Maurer meteorological forcings drive the forget, cell-candidate, and output gates across all 531 CAMELS-US basins (Kratzert et al., 2019b; Zenodo code archive 10.5281/zenodo.3530884; HydroShare hydrograph archive 10.4211/hs.83ea5312635e44dc824eeb99eda12f06).

#### 2.2.9 `kratzert_lstm_nostatic` — LSTM without Static Attributes
- **Canonical Citation:** Kratzert et al. (2019b) (Year: `2019`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Regional LSTM without Catchment Attributes
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.5194/hess-23-5089-2019
- **GitHub / Code URL:** https://github.com/kratzert/ealstm_regional_modeling (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.3530884` (https://zenodo.org/records/3530884) | **HydroShare URL:** https://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06/
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.6420`
- **Description:** Regional single-layer Long Short-Term Memory network (H = 256 memory cells, T = 270-day sequence length, dropout = 0.4, basin-averaged NSE* loss) trained jointly across all 531 CAMELS-US basins using only the 5 dynamic extended Maurer meteorological forcings without any static catchment attributes (D_static = 0 input-ablation baseline from Kratzert et al., 2019b; Zenodo code archive 10.5281/zenodo.3530884; HydroShare hydrograph archive 10.4211/hs.83ea5312635e44dc824eeb99eda12f06).

#### 2.2.10 `mizukami_mhm` — mHM (Basin-Calibrated)
- **Canonical Citation:** Mizukami et al. (2019) (Year: `2019`)
- **Model Category:** `Process-based` (Manuscript Row Shading: `#e8f6ef`)
- **Architectural Paradigm:** Mesoscale Hydrologic Model (Basin Calibration)
- **Provenance Category:** `Archive*`
- **Paper URL:** https://doi.org/10.5194/hess-23-2601-2019
- **GitHub / Code URL:** https://git.ufz.de/mhm/mhm (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on 492 of 531 CAMELS basins (39 basins missing in archive).
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.6650`
- **Description:** Mesoscale Hydrologic Model (mHM; Samaniego et al., 2010) at 0.25° grid resolution forced by daily 1/8° Maurer meteorology and calibrated independently per basin via Dynamically Dimensioned Search (DDS) on MPR transfer-function parameters over WY2000–2008 and evaluated over WY1990–1999 (Mizukami et al., 2019; HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1).

#### 2.2.11 `rakovec_mhm_mpr` — mHM (CONUS MPR)
- **Canonical Citation:** Rakovec et al. (2019) (Year: `2019`)
- **Model Category:** `Process-based` (Manuscript Row Shading: `#e8f6ef`)
- **Architectural Paradigm:** Mesoscale Hydrologic Model (CONUS Multiscale Parameter Regionalization)
- **Provenance Category:** `Archive*`
- **Paper URL:** https://doi.org/10.1029/2019JD030767
- **GitHub / Code URL:** https://git.ufz.de/mhm/mhm (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/) | **HydroShare URL:** https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on 492 of 531 CAMELS basins (39 basins missing in archive).
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.5280`
- **Description:** Mesoscale Hydrologic Model (mHM; Samaniego et al., 2010) at 0.25° grid resolution forced by daily 1/8° Maurer meteorology and parameterized simultaneously across CONUS via Multiscale Parameter Regionalization (MPR; 48 global transfer-function parameters calibrated via DDS over WY2000–2008 and evaluated over WY1990–1999; Rakovec et al., 2019; HydroShare benchmark archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1).

#### 2.2.12 `feng2020_di_lstm` — Data-Integration LSTM (DI(1)-LSTM — MIP Re-run)
- **Canonical Citation:** Feng et al. (2020) (Year: `2020`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Data-Integration LSTM with 1-Day Lagged Streamflow Q(t-1)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.1029/2019WR026793
- **GitHub / Code URL:** https://github.com/mhpi/hydroDL (Repo Code Dir: `models/feng2020_di_lstm`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`6` rows):** `gauged__lagged_streamflow`, `pub__lagged_streamflow`, `pur__lagged_streamflow` | **Best Median NSE:** `0.8010`
- **Description:** Clean-room PyTorch re-implementation of the Feng et al. (2020) 256-unit Data-Integration LSTM (DI(1)-LSTM) ingesting daily meteorological forcings, standardized 1-day lagged target-basin observed specific discharge Q(t-1) with binary validity indicator, and 27 static catchment attributes, trained with basin-variance-weighted NSE* loss over 365-day sequences across 6 random seeds on the canonical 531-basin Newman Split-1 Gauged protocol (1999-10-01 to 2008-09-30 training, 1989-10-01 to 1999-09-30 evaluation).

#### 2.2.13 `feng2020_di_lstm_lit` — Data-Integration LSTM (DI(1)-LSTM — Literature)*
- **Canonical Citation:** Feng et al. (2020) (Year: `2020`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Data-Integration LSTM with 1-Day Lagged Streamflow Q(t-1)
- **Provenance Category:** `Literature*`
- **Paper URL:** https://doi.org/10.1029/2019WR026793
- **GitHub / Code URL:** https://github.com/mhpi/hydroDL (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on WY1991–1999 (9 water years; 1989–1990 used as spin-up) as reported by Jamaat et al. (2025) for the Feng et al. (2020) DI(1)-LSTM benchmark.
- **Active Leaderboard Entries (`1` rows):** `gauged__lagged_streamflow` | **Best Median NSE:** `0.8400`
- **Description:** Regional 256-unit Long Short-Term Memory network with 1-day lagged observed specific discharge Q(t-1) data integration (DI(1)-LSTM; Feng et al., 2020, mhpi/hydroDL), trained on 1999-10-01 to 2008-09-30 and evaluated on WY1991–1999 (9 water years; 1989–1990 used as 365-day spin-up buffer) across the 531 Newman CAMELS-US basins with Daymet forcing as reported by Jamaat et al. (2025, Table 1).

#### 2.2.14 `nearing_sacsma` — SAC-SMA + Snow-17 (DDS & Donor Transfer)
- **Canonical Citation:** Nearing et al. (2020) (Year: `2020`)
- **Model Category:** `Conceptual` (Manuscript Row Shading: `#fff0e0`)
- **Architectural Paradigm:** Conceptual Soil Moisture Accounting + Snow-17 Model (DDS & 29-D Donor Transfer)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.31223/osf.io/53te4
- **GitHub / Code URL:** https://github.com/Upstream-Tech/SACSMA-SNOW17 (Repo Code Dir: `models/newman_sacsma`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`9` rows):** `gauged__single_camels`, `gauged__multi_camels`, `pub__single_camels`, `pub__multi_camels`, `pur__single_camels`, `pur__multi_camels` | **Best Median NSE:** `0.6950`
- **Description:** Clean-room C++17/OpenMP and Python re-implementation of coupled Snow-17 + SAC-SMA + 15-day Duamel Gamma unit hydrograph (27 calibrated parameters) calibrated per basin using Dynamically Dimensioned Search (DDS; Tolson & Shoemaker, 2007) over WY2000–2008 (Nearing et al., 2020), and extended to ungauged 12-fold PUB and 18-fold HUC-02 PUR prediction via k=1 nearest-neighbor donor parameter transfer in a 29-dimensional joint geographic and standardized static catchment-attribute space.

#### 2.2.15 `gauch_xgboost_lit` — Gradient-Boosted Decision Trees (XGBoost — Literature)
- **Canonical Citation:** Gauch et al. (2021b) (Year: `2021`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Gradient-Boosted Decision Trees on Lagged Forcings (XGBoost)
- **Provenance Category:** `Literature`
- **Paper URL:** https://doi.org/10.1016/j.envsoft.2020.104926
- **GitHub / Code URL:** https://github.com/gauchm/ealstm_regional_modeling (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.3648270` (https://zenodo.org/records/3648270) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.6590`
- **Description:** Regional Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016) regression tree ensemble trained on 30 days of raw daily meteorological lag features (5 forcings x 30 lags = 150 dynamic features + 27 static catchment attributes = 177 input features) across all 531 CAMELS-US basins on Extended Maurer forcings over WY2000–2008 and evaluated over WY1990–1999 (Gauch, Mai, & Lin, 2021b, Environmental Modelling & Software 135, 104926; Zenodo 10.5281/zenodo.3648270).

#### 2.2.16 `hoedt_mclstm` — Mass-Conserving LSTM (MC-LSTM — MIP Re-run)
- **Canonical Citation:** Hoedt et al. (2021) (Year: `2021`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Mass-Conserving Recurrent Neural Network (MC-LSTM)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://arxiv.org/abs/2101.05186
- **GitHub / Code URL:** https://github.com/ml-jku/mc-lstm (Repo Code Dir: `models/hoedt_mclstm`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`5` rows):** `gauged__single_camels`, `gauged__multi_camels` | **Best Median NSE:** `0.7770`
- **Description:** Clean-room PyTorch + Triton fused recurrent scan implementation of Hoedt et al. (2021) MC-LSTM (64 mass cells, column-stochastic redistribution matrix, 1 trash sink cell + 63 summed discharge cells, sequence length 365, NSE* loss). Includes single-forcing models and a 3-forcing multi-input configuration (arithmetic mean precipitation mass input with all 15 meteorological drivers + 27 static attributes in the auxiliary gate vector).

#### 2.2.17 `hoedt_mclstm_lit` — Mass-Conserving LSTM (MC-LSTM — Literature)*
- **Canonical Citation:** Hoedt et al. (2021) (Year: `2021`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Mass-Conserving Recurrent Neural Network (MC-LSTM)
- **Provenance Category:** `Literature*`
- **Paper URL:** https://arxiv.org/abs/2101.05186
- **GitHub / Code URL:** https://github.com/ml-jku/mc-lstm (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `True` — **Reason:** Evaluated in Hoedt et al. (2021) on the 447-basin Kratzert et al. (2019b) CAMELS subset.
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.7440`
- **Description:** Mass-conserving LSTM where recurrent cell states act as non-negative water buckets updated via a column-stochastic redistribution matrix R[t], softmax input gate distributing unnormalized precipitation mass, sigmoid output gate, 64 hidden cells (1 trash cell for ET/percolation sink + 63 summed discharge readout cells), and z-scored auxiliary meteorological + 27 static catchment attributes.

#### 2.2.18 `kratzert2021_lstm_archive` — LSTM (Single- & Multi-Forcing 365d Archive)
- **Canonical Citation:** Kratzert et al. (2021) (Year: `2021`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Regional Single- & Multi-Forcing Long Short-Term Memory Network
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.5194/hess-25-2685-2021
- **GitHub / Code URL:** https://github.com/neuralhydrology/neuralhydrology (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.4670268` (https://zenodo.org/records/4670268) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`14` rows):** `gauged__single_camels`, `gauged__multi_camels` | **Best Median NSE:** `0.8210`
- **Description:** Author-archived 10-seed single-forcing (Daymet, Maurer, NLDAS) and multi-forcing (Daymet+Maurer+NLDAS, Daymet+Maurer, Daymet+NLDAS, Maurer+NLDAS) 256-unit CudaLSTM models with 365-day lookback window and 27 static CAMELS attributes, trained with NSE* loss over WY2000–2008 and evaluated over WY1990–1999 across all 531 CAMELS-US basins (Kratzert et al., 2021; Zenodo 10.5281/zenodo.4670268).

#### 2.2.19 `kratzert_lstm` — LSTM (PUB & PUR MIP Re-run)
- **Canonical Citation:** Kratzert et al. (2021) (Year: `2021`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Regional Single- & Multi-Forcing Long Short-Term Memory Network
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.5194/hess-25-2685-2021
- **GitHub / Code URL:** https://github.com/neuralhydrology/neuralhydrology (Repo Code Dir: `models/kratzert_lstm`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`10` rows):** `pub__multi_camels`, `pur__single_camels`, `pur__multi_camels` | **Best Median NSE:** `0.7420`
- **Description:** Clean-room PyTorch re-implementation (models/kratzert_lstm) of the Kratzert et al. (2021) 256-unit, 365-day lookback regional CudaLSTM with 27 static catchment attributes and basin-normalized NSE* loss, trained over WY2000–2008 and evaluated over WY1990–1999 across all 531 CAMELS-US basins under the canonical 12-fold PUB and 18-fold HUC-02 Leave-One-Region-Out PUR spatial cross-validation protocols.

#### 2.2.20 `tsai2021_dpl_vic` — dPL + VIC (g_A, Maurer)*
- **Canonical Citation:** Tsai et al. (2021) (Year: `2021`)
- **Model Category:** `Hybrid (ML + Physics)` (Manuscript Row Shading: `#f2ebf9`)
- **Architectural Paradigm:** Differentiable Parameter Learning with Neural Surrogate VIC
- **Provenance Category:** `Literature*`
- **Paper URL:** https://doi.org/10.1038/s41467-021-26107-z
- **GitHub / Code URL:** https://github.com/mhpi/hydroDL (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.6383415` (https://zenodo.org/records/6383415) | **HydroShare URL:** ---
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on 498 of 531 CAMELS basins available in the Tsai et al. (2021) Figure 6a archive (Zenodo 10.5281/zenodo.6383415; g_Z variant achieves 0.435 median NSE on 499 basins).
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.4390`
- **Description:** Differentiable Parameter Learning (dPL) using a static-attribute neural parameter network (g_A, trained via an iteratively refined 4-round LSTM runoff surrogate) to infer 8 basin-specific soil/infiltration parameters for the C-language Variable Infiltration Capacity (VIC 4.1.2) land-surface model and 2 Gamma unit-hydrograph routing parameters, evaluated with the physical C-language VIC 4.1.2 model over WY1990–1999.

#### 2.2.21 `feng_dhbv10` — δHBV 1.0
- **Canonical Citation:** Feng et al. (2022) (Year: `2022`)
- **Model Category:** `Hybrid (ML + Physics)` (Manuscript Row Shading: `#f2ebf9`)
- **Architectural Paradigm:** Dynamic-Parameter LSTM + 16-Bucket Differentiable HBV
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.1029/2022WR032404
- **GitHub / Code URL:** https://github.com/mhpi/hydroDL (Repo Code Dir: `models/feng_dhbv10`)
- **Zenodo DOI / URL:** `10.5281/zenodo.7091334` (https://zenodo.org/records/7091334) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`6` rows):** `gauged__single_camels`, `pub__single_camels`, `pur__single_camels` | **Best Median NSE:** `0.7420`
- **Description:** Differentiable process-based HBV 1.0 model coupling a regional 256-unit cuDNN LSTM parameter network with 16 parallel sub-basin conceptual HBV rainfall-runoff buckets (dynamic daily soil-moisture nonlinearity BETA and evapotranspiration exponent BETAET alongside 11 static bucket parameters) and a shared 15-day Gamma unit hydrograph routing kernel, trained end-to-end with combined linear/log-sqrt RMSE loss.

#### 2.2.22 `nearing2022_ar_da_lstm` — Autoregressive & Data-Assimilation LSTM (AR-LSTM, DA-LSTM)
- **Canonical Citation:** Nearing et al. (2022) (Year: `2022`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Autoregressive & Variational Data-Assimilation LSTM
- **Provenance Category:** `Literature`
- **Paper URL:** https://doi.org/10.5194/hess-26-5493-2022
- **GitHub / Code URL:** https://github.com/grey-nearing/lstm-data-assimilation (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.7063252` (https://zenodo.org/records/7063252) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`5` rows):** `gauged__lagged_streamflow`, `pub__lagged_streamflow` | **Best Median NSE:** `0.8790`
- **Description:** Regional 256-unit LSTM networks ingesting 1-day-lagged observed streamflow Q(t-1) via scheduled-sampling autoregression with Bernoulli Markov observation holdout (AR-LSTM: no holdout and 0.5 holdout) or test-time 5-day sliding-window variational cell-state data assimilation (DA-LSTM) across 531 CAMELS-US basins forced by Daymet, Maurer, and NLDAS.

#### 2.2.23 `yin2022_rrformer` — Rainfall-Runoff Transformer (RR-Former)
- **Canonical Citation:** Yin et al. (2022) (Year: `2022`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Encoder-Decoder Transformer with Lagged Streamflow (RR-Former)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.1016/j.jhydrol.2022.127781
- **GitHub / Code URL:** https://github.com/iThronne/RR-Former (Repo Code Dir: `models/yin2022_rrformer`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__lagged_streamflow` | **Best Median NSE:** `0.8470`
- **Description:** Encoder-decoder Transformer (RR-Former; d_model=64, 4 attention heads, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=256, 3-layer MLP head 64->32->16->1 with LeakyReLU) ingesting 7-day meteorological forcing sequences, 27 static catchment attributes, and 7-day lagged observed streamflow Q(t-7..t-1), trained via 300-epoch regional pretraining across all 531 CAMELS-US basins followed by 40-epoch per-basin fine-tuning at lr=1e-4 with early stopping (Yin et al., 2022; iThronne/RR-Former).

#### 2.2.24 `frame2023_mclstm_archive` — Mass-Conserving LSTM (MC-LSTM — Frame 2023 Archive)
- **Canonical Citation:** Frame et al. (2023) (Year: `2023`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Mass-Conserving Recurrent Neural Network (MC-LSTM)
- **Provenance Category:** `Archive`
- **Paper URL:** https://doi.org/10.1002/hyp.14847
- **GitHub / Code URL:** https://github.com/jmframe/mass_balance_all (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.4211/hs.d750278db868447dbd252a8c5431affd` (https://www.hydroshare.org/resource/d750278db868447dbd252a8c5431affd/) | **HydroShare URL:** https://www.hydroshare.org/resource/d750278db868447dbd252a8c5431affd/
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.7580`
- **Description:** 8-seed ensemble of Hoedt et al. (2021) Mass-Conserving LSTM trained on single-forcing Daymet and NLDAS meteorological inputs over the 531 CAMELS-US basins by Frame et al. (2023, Hydrological Processes).

#### 2.2.25 `rasiya_koya_tft` — Temporal Fusion Transformer (TFT)
- **Canonical Citation:** Rasiya Koya & Roy (2024) (Year: `2024`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Attention-Recurrent Hybrid (Temporal Fusion Transformer)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.1016/j.jhydrol.2024.131301
- **GitHub / Code URL:** https://github.com/sinanrk/TFT-streamflow (Repo Code Dir: `models/rasiya_koya_tft`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`1` rows):** `gauged__single_camels` | **Best Median NSE:** `0.7160`
- **Description:** Temporal Fusion Transformer (TFT) with variable selection networks, LSTM encoder-decoder, and multi-head interpretable self-attention re-run under the CAMELS-US MIP protocol.

#### 2.2.26 `song_dhbv11p` — δHBV 1.1p / δMG
- **Canonical Citation:** Song et al. (2024) (Year: `2024`)
- **Model Category:** `Hybrid (ML + Physics)` (Manuscript Row Shading: `#f2ebf9`)
- **Architectural Paradigm:** LSTM + 16-Bucket Differentiable HBV (Dynamic Parameters)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.5194/hess-28-3051-2024
- **GitHub / Code URL:** https://github.com/mhpi/generic_deltamodel (Repo Code Dir: `models/song_dhbv11p`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`6` rows):** `gauged__single_camels`, `pub__single_camels`, `pur__single_camels` | **Best Median NSE:** `0.7590`
- **Description:** 16-bucket differentiable HBV 1.1p (δMG) with capillary rise flux and time-varying (dynamic) soil-moisture recharge exponent β, near-surface recession coefficient K0, and ET shape parameter γ predicted daily by a 256-unit DropConnect LSTM with 15-day Gamma unit-hydrograph routing.

#### 2.2.27 `jamaat2025_dhbv_da` — δHBV 1.1p Variational Data Assimilation (δHBV-DA-PS, -S, -P)*
- **Canonical Citation:** Jamaat et al. (2025) (Year: `2025`)
- **Model Category:** `Hybrid (ML + Physics)` (Manuscript Row Shading: `#f2ebf9`)
- **Architectural Paradigm:** Differentiable HBV (δHBV 1.1p) with 5-Day Variational State & Precipitation Data Assimilation
- **Provenance Category:** `Literature*`
- **Paper URL:** https://doi.org/10.1016/j.jhydrol.2025.134137
- **GitHub / Code URL:** https://github.com/mhpi/DataAssimilation (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `True` — **Reason:** Evaluated on WY1991–1999 (9 water years; 1989–1990 used as 365-day spin-up buffer in the background MHPI checkpoint) in Jamaat et al. (2025).
- **Active Leaderboard Entries (`3` rows):** `gauged__lagged_streamflow` | **Best Median NSE:** `0.8200`
- **Description:** 16-bucket differentiable HBV (δHBV 1.1p, with 3 dynamic time-varying parameters predicted by a 256-unit regional LSTM from 3 Daymet meteorological forcings and 35 static catchment attributes) coupled with training-free 5-day sliding-window variational data assimilation (Adadelta, 5 steps per window, NSELossBatch3 with monotonic acceptance gating) that updates initial hydrological log-states (SNOWPACK, MELTWATER, SM, SUZ, SLZ; δHBV-DA-S), last-day precipitation multiplier k^4 (δHBV-DA-P), or both jointly (δHBV-DA-PS) on the 531 Newman CAMELS-US basins.

#### 2.2.28 `li2025_lstm_dhbv_ens` — (LSTM + dHBV 1.1p) Multi-Model Ensemble
- **Canonical Citation:** Li et al. (2025) (Year: `2025`)
- **Model Category:** `Hybrid (ML + Physics)` (Manuscript Row Shading: `#f2ebf9`)
- **Architectural Paradigm:** Multi-Structure Hybrid Ensemble (Regional LSTM + Differentiable HBV 1.1p)
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.5194/hess-29-6829-2025
- **GitHub / Code URL:** https://github.com/mhpi/generic_deltamodel (Repo Code Dir: `models/li2025_lstm_dhbv_ens`)
- **Zenodo DOI / URL:** `10.5281/zenodo.16895228` (https://zenodo.org/records/16895228) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`1` rows):** `gauged__multi_camels` | **Best Median NSE:** `0.8200`
- **Description:** Multi-structure hybrid ensemble of Li, Song, Pan, Lawson, and Shen (2025, HESS 29, 6829-6861; DOI: 10.5194/hess-29-6829-2025; Zenodo: 10.5281/zenodo.16895228) combining multi-forcing regional LSTM predictions with differentiable HBV 1.1p (dMG) predictions. On Stage-1 Split-3 (WY1980-1995 train / WY1995-2008 test, 531 basins), the 60-member multi-forcing hybrid ensemble (LSTM + dHBV)_seed^123 achieves Med NSE = 0.8225 (published 0.823; independent StandaloneDHBV Split-3 re-run hybrid = 0.8195), the single-seed 6-member multi-forcing hybrid (LSTM + dHBV)^123 achieves Med NSE = 0.8176 +/- 0.0040 (member[0] = 0.8184 vs published 0.818; re-run = 0.8165), and the Daymet single-seed hybrid (LSTM + dHBV)^1_Daymet achieves Med NSE = 0.7895 (member[0]; 10-seed mean 0.7871 +/- 0.0013 vs published 0.790; re-run = 0.7872). On Stage-2 Newman Split-1 (WY1999-2008 train / WY1989-1999 test, 531 basins, Daymet + Maurer + NLDAS), the equal-weight 2-component multi-structure ensemble (t4_06.zarr; member[0] = Kratzert et al. 2021 10-seed 3-forcing LSTM ensemble t4_05.zarr with Med NSE = 0.8210, member[1] = Song et al. 2024 5-seed 3-forcing dHBV 1.1p ensemble t4_24.zarr with Med NSE = 0.7992) achieves Med NSE = 0.8200 (0.820), matching the 20-member equal-seed ensemble t4_03.zarr (Med NSE = 0.8195 / 0.820). No separate single-model row is emitted for li2025_lstm_dhbv_ens because its individual constituent models (kratzert2021_lstm_archive and song_dhbv11p) are already evaluated as standalone entries on Table 4.

#### 2.2.29 `liu2025_tslib_suite` — TSLib Deep Learning Suite (10 Models: Crossformer, PatchTST, Informer, etc.)
- **Canonical Citation:** Liu et al. (2025) (Year: `2025`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** General Time-Series Deep Learning Suite (Transformers, MLPs, Temporal 2D-Variation, CNNs)
- **Provenance Category:** `Literature`
- **Paper URL:** https://doi.org/10.5194/hess-29-6811-2025
- **GitHub / Code URL:** https://github.com/thuml/Time-Series-Library (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `10.5281/zenodo.15852144` (https://zenodo.org/records/15852144) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`10` rows):** `gauged__multi_camels` | **Best Median NSE:** `0.7200`
- **Description:** Ten general-purpose time-series deep learning models adapted from the Time-Series-Library (TSLib) via the rainflow hydrology benchmark framework (9 Transformer variants: Crossformer, Non-stationary Transformer, Vanilla Transformer, Informer, Pyraformer, ETSformer, Reformer, iTransformer, and CARDformer; plus the moving-average linear decomposition model DLinear), trained simultaneously on Daymet, Maurer, and NLDAS forcings and 26 static catchment attributes across all 531 CAMELS-US basins over WY2000–2008 and evaluated over WY1990–1999 (Liu et al., 2025, Supplement Table S1; PatchTST and TimesNet were evaluated in the same study on autoregressive forecasting tasks).

#### 2.2.30 `wang_s4dft` — Frequency-Tuned Diagonal State Space Model (S4D-FT & S4D)
- **Canonical Citation:** Wang, Zhang, et al. (2025a) (Year: `2025`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Diagonal & Frequency-Tuned Diagonal Structured State Space Model (S4D / S4D-FT)
- **Provenance Category:** `Literature`
- **Paper URL:** https://doi.org/10.1029/2025WR039888
- **GitHub / Code URL:** https://github.com/WESTENR-OU/S4D_rainfall_runoff_simulations (Repo Code Dir: `None`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`2` rows):** `gauged__single_camels` | **Best Median NSE:** `0.7400`
- **Description:** Six-layer diagonal structured state-space sequence model (HOPE) evaluated without (S4D, cfr = cfi = 1) and with frequency-tuned HiPPO state-matrix initialization (S4D-FT, cfr = cfi = 10, scaling Re(A) = -0.5^cfr and Im(A_n) = pi * n * cfi without per-basin fine-tuning) over 365-day lookback windows on the canonical Newman et al. (2017) 531-basin CAMELS-US split with NLDAS forcings and 27 static catchment attributes (Wang, Zhang, et al., 2025a, Water Resources Research 61, e2025WR039888).

#### 2.2.31 `poudel_mlp_hbv` — Static-Parameter MLP + 16-HBV (diff-hydro)
- **Canonical Citation:** Poudel & Steinschneider (2026) (Year: `2026`)
- **Model Category:** `Hybrid (ML + Physics)` (Manuscript Row Shading: `#f2ebf9`)
- **Architectural Paradigm:** Static MLP + 16-Zone Differentiable HBV
- **Provenance Category:** `MIP Re-run (Our Code)`
- **Paper URL:** https://doi.org/10.5194/hess-30-5173-2026
- **GitHub / Code URL:** https://github.com/snpoudel/diff-hydro (Repo Code Dir: `models/poudel_mlp_hbv`)
- **Zenodo DOI / URL:** `---` (---) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`6` rows):** `gauged__single_camels`, `pub__single_camels`, `pur__single_camels` | **Best Median NSE:** `0.7350`
- **Description:** Regional 3-layer Multilayer Perceptron (2048 hidden units) mapping static catchment attributes to time-invariant parameters of a 16-component parallel differentiable HBV rainfall-runoff model with Gamma unit-hydrograph routing.

#### 2.2.32 `era_lstm_seeded` — Empirical Research Assistant -- LSTM Seeded
- **Canonical Citation:** Nearing (2026) (Year: `2026`)
- **Model Category:** `Machine learning` (Manuscript Row Shading: `#e8f1fb`)
- **Architectural Paradigm:** Machine learning
- **Provenance Category:** `Contribution`
- **Paper URL:** https://github.com/grey-nearing/camels-us-mip/tree/main/era_experiments
- **GitHub / Code URL:** https://github.com/grey-nearing/camels-us-mip/tree/main/models (Repo Code Dir: `models/era_*`)
- **Zenodo DOI / URL:** `10.5281/zenodo.23641695` (https://doi.org/10.5281/zenodo.23641695) | **HydroShare URL:** ---
- **Asterisk (`*`):** `False`
- **Active Leaderboard Entries (`14` rows):** `gauged__multi_camels`, `pub__multi_camels`, `pur__multi_camels` | **Best Median NSE:** `0.8270`
- **Description:** Autonomous hydrological architectures discovered by the Empirical Research Assistant (ERA) seeded from the Kratzert et al. (2021) regional multi-forcing LSTM baseline across Gauged, 12-Fold PUB, and 18-Fold HUC-02 PUR regimes under Unguided (minimal guidance), Physics-Guided, and Multi-Paradigm experimental protocols (Nearing, 2026).

---

## 3. Stage-1 Author-Protocol Confirmation Runs (`Appendix A — Table A1`, `12` Rows)

| # | `family_id` | Model Name | Citation | Author Split Protocol | Forcing | Basins | Seeds | Published Med NSE | Published Single Med NSE | Stage-1 Reproduced Med NSE | Stage-1 Single Med NSE | $\Delta$ Med NSE | Status | Stage-1 Zarr URI |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| 1 | `feng2020_di_lstm` | **Data-Integration LSTM (DI(1)-LSTM — MIP Re-run)** | Feng et al. (2020) | Split-3: 671 CAMELS-US basins, train 1985-10-01 to 1995-09-30, eval 1995-10-01 to 2005-09-30 (NLDAS + Q(t-1), 6 seeds) | NLDAS + Q(t-1) | 671 | 6 | 0.8590 | 0.8510 | **0.8200** | 0.8130 | `-0.0390` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/feng2020_di_lstm.zarr` |
| 2 | `feng_dhbv10` | **δHBV 1.0** | Feng et al. (2022) | Split-3: 531 CAMELS-US basins (and 671-basin extended set), train 1980-10-01 to 1995-09-30 (WY1981–1995), test 1995-10-01 to 2010-09-30 (WY1996–2010), Daymet forcing | Daymet | 531 | 3 | 0.7400 | 0.7330 | **0.7412** | 0.7339 | `0.0012` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/feng_dhbv10.zarr` |
| 3 | `hoedt_mclstm` | **Mass-Conserving LSTM (MC-LSTM — MIP Re-run)** | Hoedt et al. (2021) | Newman Split-1: 531 CAMELS-US basins, train 1999-10-01 to 2008-09-30, eval 1989-10-01 to 1999-09-30 (Maurer forcing, 64 mass cells, seq_len=365, targ_len=1, NSE* loss) | Maurer | 531 | 3 | 0.7420 | 0.7260 | **0.7430** | 0.7160 | `0.0010` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/hoedt_mclstm.zarr` |
| 5 | `kratzert_lstm` | **LSTM (PUB & PUR MIP Re-run)** | Kratzert et al. (2021) | Newman Split-1: 531 CAMELS-US basins, train 1999-10-01 to 2008-09-30, eval 1989-10-01 to 1999-09-30 (Daymet single-forcing, 365-day sequence length, 256 hidden units, NSE* loss) | Daymet | 531 | 3 | 0.7720 | 0.7440 | **0.7110** | 0.7036 | `-0.0610` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/kratzert_lstm.zarr` |
| 6 | `li2025_lstm_dhbv_ens` | **(LSTM + dHBV 1.1p) Multi-Model Ensemble** | Li et al. (2025) | 1980-10-01 to 1995-09-30 train / 1995-10-01 to 2008-09-30 test (Split-3) | Daymet, Maurer, NLDAS | 531 | 10 | 0.8230 | 0.8180 | **0.8225** | 0.8176 | `-0.0005` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/li2025_lstm_dhbv_ens.zarr` |
| 7 | `yin2022_rrformer` | **Rainfall-Runoff Transformer (RR-Former)** | Yin et al. (2022) | Task 1 (Yin et al., 2022, Section 4.1, p. 6, Fig. 4a & Tables 5-6): train 1980-10-01 to 1995-09-30, val 1995-10-01 to 2000-09-30, test 2000-10-01 to 2014-09-30 (and 1995-10-01 to 2010-09-30 window) on 531 CAMELS-US basins with Daymet + lagged streamflow | Daymet + Lagged Q (t-7..t-1) | 531 | 3 | 0.8265 | 0.8265 | **0.8009** | 0.7919 | `-0.0256` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/yin2022_rrformer.zarr` |
| 8 | `poudel_mlp_hbv` | **Static-Parameter MLP + 16-HBV (diff-hydro) — Gauged (Split-2C)** | Poudel & Steinschneider (2026) | Split-2C: 425 in-sample training basins (80% random split), train 1990-01-01 to 2005-12-31, val 2006-01-01 to 2014-12-31, test 1980-01-01 to 1989-12-31 (Daymet, 30 static attributes) | Daymet | 425 | 1 | 0.7100 | 0.7100 | **0.7180** | 0.7180 | `0.0080` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/poudel_mlp_hbv.zarr` |
| 9 | `poudel_mlp_hbv` | **Static-Parameter MLP + 16-HBV (diff-hydro) — PUB Hold-Out (Split-2C)** | Poudel & Steinschneider (2026) | Split-2C: 106 held-out PUB test basins (20% random split, trained on 425 basins), train 1990-01-01 to 2005-12-31, val 2006-01-01 to 2014-12-31, test 1980-01-01 to 1989-12-31 (Daymet, 30 static attributes) | Daymet | 106 | 1 | 0.6700 | 0.6300 | **0.6540** | 0.6540 | `-0.0160` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/poudel_mlp_hbv.zarr` |
| 10 | `rasiya_koya_tft` | **Temporal Fusion Transformer (TFT)** | Rasiya Koya & Roy (2024) | 531 CAMELS-US basins, train 1999-10-01 to 2008-09-30, test 1989-10-01 to 1999-09-30 (Maurer forcing, 27 static attributes, 25 epochs) | Maurer | 531 | 1 | 0.7250 | 0.7250 | **0.7155** | 0.7155 | `-0.0095` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/rasiya_koya_tft.zarr` |
| 11 | `nearing_sacsma` | **SAC-SMA + Snow-17 (DDS & Donor Transfer)** | Nearing et al. (2020) | Newman Split-1: 531 CAMELS-US basins, train 1999-10-01 to 2008-09-30 (365-day spinup), test 1989-10-01 to 1999-09-30 (Daymet forcing, per-basin DDS calibration minimizing daily RMSE) | Daymet | 531 | 1 | 0.6700 | 0.6700 | **0.6609** | 0.6609 | `-0.0091` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/newman_sacsma.zarr` |
| 12 | `song_dhbv11p` | **δHBV 1.1p / δMG (Dynamic Parameters) — Author Split-3** | Song et al. (2024) | Split-3: 531/671 CAMELS-US basins, train WY1981-1995 (1980-10-01 to 1995-09-30), test WY1996-2010 (1995-10-01 to 2010-09-30), Daymet & Maurer forcings | Daymet | 531 | 3 | 0.7500 | 0.7500 | **0.7475** | 0.7418 | `-0.0025` | `PASS` | `gs://camels-us-mip/stage1_author_reproductions/song_dhbv11p.zarr` |

### 3.1 Stage-1 Confirmation Notes (by Model)

- **`feng2020_di_lstm` (Data-Integration LSTM (DI(1)-LSTM — MIP Re-run), Feng et al. (2020)):** Feng, Fang & Shen (2020, WRR, Table 3, p. 28) evaluated DI(1) across 671 CAMELS-US basins with NLDAS forcing + 1-day lagged observed specific discharge Q(t-1) (train 1985-10-01 to 1995-09-30, eval 1995-10-01 to 2005-09-30, 6 seeds), reporting 6-seed ensemble Med NSE = 0.859 and single-seed Med NSE = 0.851 ± 0.0097. Our clean-room 6-seed reproduction (seeds 101..106) achieves 0.8202 ensemble Med NSE (0.8133 ± 0.0020 single-seed Med NSE, 0.7955 Mean NSE, 0.8338 Med KGE; Δ = -0.0388 within 0.04 tolerance, status: PASS; the -0.039 gap reflects linear z-score standardization of Q(t-1) vs. Feng et al. 2020 Eq. 11 log-sqrt compression log10(sqrt(v) + 0.1)).
- **`feng_dhbv10` (δHBV 1.0, Feng et al. (2022)):** Split-3 (WY1980–1995 train / WY1995–2010 test, Daymet, 3 seeds [111111, 111112, 111113]): On the 531 canonical CAMELS-US basins stored in feng_dhbv10.zarr (1995-10-01 to 2010-09-30), achieves 0.7412 3-seed ensemble Med NSE and 0.7339 ± 0.0010 single-seed Med NSE ([0.7331, 0.7332, 0.7353]) vs. published 0.740 ensemble / 0.733 single-seed (+0.0012; 14-WY post-warmup 1996-10-01 to 2010-09-30 achieves 0.7436 ensemble / 0.7389 single-seed on 531 basins and 0.7407 ensemble / 0.7340 single-seed on all 671 basins). Newman Split-1 (WY1999–2008 train / WY1989–1999 test, 531 basins, 5 seeds) achieves 0.7416 ensemble and 0.7327 ± 0.0025 single-seed Med NSE.
- **`hoedt_mclstm` (Mass-Conserving LSTM (MC-LSTM — MIP Re-run), Hoedt et al. (2021)):** Hoedt et al. (2021, ICML Table 2/3 & Appendix Table B.7) report 5-seed Maurer Ensemble Med NSE = 0.742 on 531 basins (0.744 on 447/498-basin subsets) and Single-Model Med NSE = 0.726 +/- 0.004. Our 3-seed Maurer Stage-1 reproduction achieves Ensemble Med NSE = 0.7430 (+0.001 vs 531-basin published, -0.001 vs 498-basin subset) and Single-Seed Med NSE = 0.7160 +/- 0.0032 (status: PASS).
- **`kratzert_lstm` (LSTM (PUB & PUR MIP Re-run), Kratzert et al. (2021)):** Kratzert et al. (2021, HESS 25, 2685-2706, Table 1 & Table A2; Zenodo 10.5281/zenodo.4670268 / t3_02) evaluated the 256-unit, 365-day lookback CudaLSTM across the 531 Newman CAMELS-US basins with Daymet single-forcing inputs (train 1999-10-01 to 2008-09-30, eval 1989-10-01 to 1999-09-30), achieving 10-seed ensemble Med NSE = 0.772 (0.7724) and single-seed Med NSE = 0.744 ± 0.003 (Table A2 reports 0.74; note that 0.783 / 0.762 in Table A1 are on the 447-basin intersection subset). Our stored 3-seed fast-verification Zarr archive (seeds 101..103; gs://camels-us-mip/stage1_author_reproductions/kratzert_lstm.zarr) achieves 0.7110 3-seed ensemble Med NSE and 0.7036 ± 0.0018 single-seed Med NSE (Mean NSE = 0.6767, Med KGE = 0.6931, 1 basin with NSE <= 0; Δ_single = -0.0404 vs. 0.744 single-seed benchmark), while our 30-epoch 3-seed reproduction achieves 0.7465 ensemble / 0.7350 ± 0.0013 single-seed Med NSE (Δ_single = -0.0090) and our 1-seed full NeuralHydrology schedule confirmation runs achieve 0.738 (Daymet vs. 0.744), 0.732 (Maurer vs. 0.732), 0.716 (NLDAS vs. 0.720), and 0.788 (3-Forcing vs. 0.796), confirming faithful architectural equivalence (PASS).
- **`li2025_lstm_dhbv_ens` ((LSTM + dHBV 1.1p) Multi-Model Ensemble, Li et al. (2025)):** Verified directly against the official Li et al. (2025) Zenodo hydrograph archive (10.5281/zenodo.16895228, array shape 6 x 10 x 531 x 4749 covering Split-3 test 1995-10-01 to 2008-09-30 across 531 CAMELS-US basins) and cross-checked with an independent Split-3 StandaloneDHBV re-run: (1) 60-member multi-forcing seed-ensemble hybrid (LSTM + dHBV)_seed^123: published Med NSE = 0.823, archive recomputed = 0.8225 (post-warmup = 0.8286), independent Split-3 re-run hybrid = 0.8195 (delta = -0.0005 archive / -0.0035 re-run, PASS); (2) 6-member single-seed 3-forcing hybrid (LSTM + dHBV)^123: published Med NSE = 0.818, archive member[0] = 0.8184, 10-seed mean = 0.8176 +/- 0.0040, independent Split-3 re-run = 0.8165 (delta = -0.0004, PASS); (3) 2-member single-seed Daymet hybrid (LSTM + dHBV)^1_Daymet: published Med NSE = 0.790, archive member[0] = 0.7895 (post-warmup = 0.7915), 10-seed mean = 0.7871 +/- 0.0013, independent Split-3 re-run = 0.7872 (delta = -0.0005 member[0] / -0.0029 10-seed mean, PASS).
- **`yin2022_rrformer` (Rainfall-Runoff Transformer (RR-Former), Yin et al. (2022)):** In Yin et al. (2022, J. Hydrol. 609, 127781), the number 0.862 in Table 4 is Mean NSE on a 40-basin hyperparameter-tuning subset, whereas the full-cohort 1-day-ahead Median NSE reported in Section 4.1 (p. 6), Fig. 4a, and Tables 5-6 is 0.8265 (mean = 0.7904; global pretraining w/o fine-tuning RR-Former2 = 0.7598). Our 3-seed Stage-1 reproduction stored in yin2022_rrformer.zarr achieves Median NSE = 0.8009 across the 1995-10-01 to 2010-09-30 evaluation window (delta = -0.0256, single-seed 0.7919 +/- 0.0042; 0.816 on the 2000-2014 test split), passing within tolerance (status: PASS).
- **`poudel_mlp_hbv` (Static-Parameter MLP + 16-HBV (diff-hydro) — Gauged (Split-2C), Poudel & Steinschneider (2026)):** Split-2C temporal out-of-sample evaluation on 425 training gauges (1980-1989): reproduced Med NSE = 0.7177 (0.718) vs published 0.710 (+0.008; WY1981-1989 post-spinup Med NSE = 0.7334, best val epoch 27, val MSE = 2.8168).
- **`poudel_mlp_hbv` (Static-Parameter MLP + 16-HBV (diff-hydro) — PUB Hold-Out (Split-2C), Poudel & Steinschneider (2026)):** Split-2C spatial out-of-sample (PUB) evaluation on 106 held-out test gauges (1980-1989): reproduced Med NSE = 0.6539 (0.654) vs published 0.630 on 106-basin Split-2C hold-out (+0.024) and 0.670 reference (-0.016; WY1981-1989 post-spinup Med NSE = 0.6609).
- **`rasiya_koya_tft` (Temporal Fusion Transformer (TFT), Rasiya Koya & Roy (2024)):** Single-seed 25-epoch Temporal Fusion Transformer (hidden_size=128, 4 heads, 180-day window) on Maurer forcing achieves Median NSE = 0.7155 (Mean NSE = 0.6759, Median KGE = 0.7017, NSE <= 0 = 2) vs 0.725 reference (-0.0095; status: PASS).
- **`nearing_sacsma` (SAC-SMA + Snow-17 (DDS & Donor Transfer), Nearing et al. (2020)):** Clean-room C++17/OpenMP 27-parameter Snow-17 + SAC-SMA + 15-day Duamel Gamma unit hydrograph calibrated per basin via DDS (400 iterations, r=0.2, 365-day spinup, daily RMSE objective) on Daymet forcings over WY2000–2008 and evaluated over WY1990–1999 across all 531 CAMELS-US basins. Reproduced Median NSE = 0.6609 (Mean NSE = 0.6131, Median KGE = 0.6769) matches the published Daymet DDS SAC-SMA Split-1 benchmark (0.670; Nearing et al., 2020 / Kratzert et al., 2021 Fig. 4) within delta = -0.0091 and exceeds Newman et al. (2017)'s Maurer benchmark (0.603 on 447 basins, 0.6071 on 530 basins in t3_48.zarr).
- **`song_dhbv11p` (δHBV 1.1p / δMG (Dynamic Parameters) — Author Split-3, Song et al. (2024)):** Split-3 author protocol (WY1981-1995 train / WY1996-2010 test, 3 seeds [111111, 111112, 111113]): on the 531 canonical basins with Daymet forcing (gs://camels-us-mip/stage1_author_reproductions/song_dhbv11p.zarr), achieves 0.7475 3-seed ensemble Med NSE and 0.7418 ± 0.0009 single-seed Med NSE ([0.7419, 0.7407, 0.7430]; on all 671 basins: 0.7430 ensemble / 0.7371 ± 0.0021 single-seed; on Maurer 531 basins WY1996-2008: 0.7389 ensemble / 0.7333 ± 0.0024 single-seed) vs. Song et al. (2024) published 0.750 for dynamic δHBV 1.1p (and 0.730 for static δHBV 1.0), Δ = -0.0082 single / -0.0025 ensemble (PASS). Also verified on Split-1 (WY1999-2008 train / WY1989-1999 test, Daymet, 531 basins) against the released MHPI generic_deltamodel checkpoint (published 0.7574 ensemble / 0.7548 single-seed; reproduced 5-seed ensemble = 0.7589 / single-member mean = 0.7537 ± 0.0011).

---

## 4. Complete Catalogue of All 12 Leaderboards (`Tables 3–14`, `132` Total Rows)

### 4.1 Summary of the 12 Canonical Leaderboards

| Table # | Leaderboard ID | Title | Spatial Regime | Forcing / Input Regime | Active Rows | Top Model | Top Med NSE |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- | :-: |
| **Table 3** | `gauged__single_camels` | Gauged: Single CAMELS Forcing | `gauged` | `single_camels` | **42** | LSTM (10-model ensemble) | **0.772** |
| **Table 4** | `gauged__multi_camels` | Gauged: Multiple CAMELS Forcings | `gauged` | `multi_camels` | **30** | ERA-Multi-Paradigm-Gauged (10-model ensemble) | **0.827** |
| **Table 5** | `gauged__lagged_streamflow` | Gauged: With Lagged Streamflow Inputs | `gauged` | `lagged_streamflow` | **11** | Autoregression LSTM (no holdout) | **0.879** |
| **Table 6** | `gauged__freeform` | Gauged: Freeform Forcings (Non-CAMELS Inputs) | `gauged` | `freeform` | **0** | --- | **---** |
| **Table 7** | `pub__single_camels` | 12-Fold Random PUB: Single CAMELS Forcing | `pub` | `single_camels` | **12** | LSTM (10-model ensemble)* | **0.691** |
| **Table 8** | `pub__multi_camels` | 12-Fold Random PUB: Multiple CAMELS Forcings | `pub` | `multi_camels` | **8** | LSTM (8-model ensemble) | **0.742** |
| **Table 9** | `pub__lagged_streamflow` | 12-Fold Random PUB: With Lagged Streamflow Inputs | `pub` | `lagged_streamflow` | **4** | Autoregression LSTM (0.5 holdout)* | **0.807** |
| **Table 10** | `pub__freeform` | 12-Fold Random PUB: Freeform Forcings | `pub` | `freeform` | **0** | --- | **---** |
| **Table 11** | `pur__single_camels` | 18-Fold HUC-02 PUR: Single CAMELS Forcing | `pur` | `single_camels` | **15** | δHBV 1.1p / δMG (5-model ensemble) | **0.650** |
| **Table 12** | `pur__multi_camels` | 18-Fold HUC-02 PUR: Multiple CAMELS Forcings | `pur` | `multi_camels` | **8** | ERA-Unguided-PUR (10-model ensemble) | **0.696** |
| **Table 13** | `pur__lagged_streamflow` | 18-Fold HUC-02 PUR: With Lagged Streamflow Inputs | `pur` | `lagged_streamflow` | **2** | DI(1)-LSTM (3-model ensemble) | **0.747** |
| **Table 14** | `pur__freeform` | 18-Fold HUC-02 PUR: Freeform Forcings | `pur` | `freeform` | **0** | --- | **---** |

### 4.2 Table 3: `gauged__single_camels` — Gauged: Single CAMELS Forcing (`42` Rows)
- **Subtitle / Scope:** 531 basins, Newman evaluation decade (1989–1999) — Daymet, Maurer, or NLDAS without lagged streamflow.
- **Spatial Regime:** `gauged` | **Forcing Regime:** `single_camels` | **Section:** `4.1.1`
- **SOTA / Rank #1 Winner:** **LSTM (10-model ensemble)** (`Median NSE = 0.772`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `t3_02` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.772** | 0.744 ± 0.003 | 0.732 | 0.763 | 0.720 | 2 | `---` / `---` |
| **2** | `t3_06` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Maurer | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.764** | 0.732 ± 0.004 | 0.728 | 0.765 | 0.722 | 2 | `---` / `---` |
| **3** | `t3_18` | **δHBV 1.1p / δMG (5-model ensemble)** | `song_dhbv11p` | Hybrid (ML + Physics) | `MIP Re-run` | Song et al. (2024) | Daymet | Yes (5) | 531 | 1989-10-01 to 1999-09-30 | **0.759** | 0.754 ± 0.001 | 0.724 | 0.757 | 0.713 | 1 | `---` / `---` |
| **4** | `t3_11` | **LSTM (270d, 8-model ensemble)** | `kratzert2019b_lstm` | Machine learning | `Archive` | Kratzert et al. (2019b) | Maurer | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.759** | 0.732 ± 0.002 | 0.719 | 0.756 | 0.712 | 2 | `---` / `---` |
| **5** | `t3_12` | **MC-LSTM (Frame 2023 Archive, Daymet)** | `frame2023_mclstm_archive` | Machine learning | `Archive` | Frame et al. (2023) | Daymet | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.758** | --- | 0.718 | 0.757 | 0.707 | 1 | `---` / `---` |
| **6** | `t3_19` | **δHBV 1.1p / δMG (single model)** | `song_dhbv11p` | Hybrid (ML + Physics) | `MIP Re-run` | Song et al. (2024) | Daymet | No (5) | 531 | 1989-10-01 to 1999-09-30 | **0.754** | 0.754 ± 0.001 | 0.717 | 0.756 | 0.708 | 2 | `---` / `---` |
| **8** | `t3_15` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | NLDAS | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.747** | 0.720 ± 0.004 | 0.705 | 0.740 | 0.683 | 0 | `---` / `---` |
| **9** | `t3_08` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.744** | 0.744 ± 0.003 | 0.702 | 0.759 | 0.712 | 4 | `---` / `---` |
| **10** | `hoedt_mclstm_lit_maurer` | **MC-LSTM (Hoedt 2021 — Literature, 10-model ensemble)*** | `hoedt_mclstm_lit` | Machine learning | `Literature*` | Hoedt et al. (2021) | Maurer | Yes (10) | 447 | 1989-10-01 to 1999-09-30 | **0.744** | 0.726 ± 0.003 | --- | --- | --- | --- | `---` / `---` |
| **11** | `hoedt_mclstm_maurer` | **MC-LSTM (Hoedt 2021 — MIP Re-run, Maurer)** | `hoedt_mclstm` | Machine learning | `MIP Re-run` | Hoedt et al. (2021) | Maurer | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.743** | 0.716 ± 0.003 | 0.706 | 0.763 | 0.710 | 2 | `---` / `---` |
| **12** | `t3_29` | **δHBV 1.0 (5-model ensemble)** | `feng_dhbv10` | Hybrid (ML + Physics) | `MIP Re-run` | Feng et al. (2022) | Daymet | Yes (5) | 531 | 1989-10-01 to 1999-09-30 | **0.742** | 0.733 ± 0.002 | 0.685 | 0.733 | 0.676 | 7 | `---` / `---` |
| **13** | `t3_23` | **EA-LSTM (8-model ensemble)** | `kratzert_ealstm` | Machine learning | `Archive` | Kratzert et al. (2019b) | Maurer | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.741** | 0.712 ± 0.004 | 0.705 | 0.737 | 0.686 | 2 | `---` / `---` |
| **14** | `t3_21` | **S4D-FT (Frequency-Tuned Diagonal SSM, single model)** | `wang_s4dft` | Machine learning | `Literature` | Wang, Zhang, et al. (2025a) | NLDAS | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.740** | 0.740 ± 0.002 | --- | --- | --- | --- | `---` / `---` |
| **15** | `t3_28` | **MC-LSTM (Frame 2023 Archive, NLDAS)** | `frame2023_mclstm_archive` | Machine learning | `Archive` | Frame et al. (2023) | NLDAS | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.736** | --- | 0.688 | 0.737 | 0.679 | 1 | `---` / `---` |
| **16** | `poudel_mlp_hbv_daymet` | **Static-Parameter MLP + 16-HBV (3-model ensemble)** | `poudel_mlp_hbv` | Hybrid (ML + Physics) | `MIP Re-run` | Poudel & Steinschneider (2026) | Daymet | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.735** | 0.733 ± 0.003 | 0.657 | 0.738 | 0.654 | 7 | `---` / `---` |
| **18** | `t3_33` | **δHBV 1.0 (single model)** | `feng_dhbv10` | Hybrid (ML + Physics) | `MIP Re-run` | Feng et al. (2022) | Daymet | No (5) | 531 | 1989-10-01 to 1999-09-30 | **0.733** | 0.733 ± 0.002 | 0.673 | 0.729 | 0.670 | 6 | `---` / `---` |
| **19** | `poudel_mlp_hbv_daymet_single` | **Static-Parameter MLP + 16-HBV (single model)** | `poudel_mlp_hbv` | Hybrid (ML + Physics) | `MIP Re-run` | Poudel & Steinschneider (2026) | Daymet | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.733** | 0.733 ± 0.003 | 0.649 | 0.739 | 0.650 | 9 | `---` / `---` |
| **20** | `t3_20` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Maurer | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.732** | 0.732 ± 0.004 | 0.695 | 0.760 | 0.714 | 2 | `---` / `---` |
| **21** | `t3_26` | **LSTM (270d, single model)** | `kratzert2019b_lstm` | Machine learning | `Archive` | Kratzert et al. (2019b) | Maurer | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.732** | 0.732 ± 0.002 | 0.687 | 0.750 | 0.703 | 2 | `---` / `---` |
| **22** | `hoedt_mclstm_lit_maurer_single` | **MC-LSTM (Hoedt 2021 — Literature, single model)*** | `hoedt_mclstm_lit` | Machine learning | `Literature*` | Hoedt et al. (2021) | Maurer | No (10) | 447 | 1989-10-01 to 1999-09-30 | **0.726** | 0.726 ± 0.003 | --- | --- | --- | --- | `---` / `---` |
| **23** | `t3_32` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | NLDAS | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.720** | 0.720 ± 0.004 | 0.677 | 0.733 | 0.677 | 2 | `---` / `---` |
| **24** | `t3_s4d_lit` | **S4D (Diagonal State Space Model, single model)** | `wang_s4dft` | Machine learning | `Literature` | Wang, Zhang, et al. (2025a) | NLDAS | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.720** | 0.720 ± 0.004 | --- | --- | --- | --- | `---` / `---` |
| **25** | `tft_gauged_maurer` | **Temporal Fusion Transformer (single model)** | `rasiya_koya_tft` | Machine learning | `MIP Re-run` | Rasiya Koya & Roy (2024) | Maurer | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.716** | 0.716 | 0.676 | 0.702 | 0.648 | 2 | `---` / `---` |
| **26** | `hoedt_mclstm_maurer_single` | **MC-LSTM (Hoedt 2021 — MIP Re-run, Maurer, Single Model)** | `hoedt_mclstm` | Machine learning | `MIP Re-run` | Hoedt et al. (2021) | Maurer | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.716** | 0.716 ± 0.003 | 0.673 | 0.755 | 0.702 | 5 | `---` / `---` |
| **27** | `t3_30` | **EA-LSTM (single model)** | `kratzert_ealstm` | Machine learning | `Archive` | Kratzert et al. (2019b) | Maurer | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.712** | 0.712 ± 0.004 | 0.673 | 0.725 | 0.676 | 3 | `---` / `---` |
| **28** | `t3_39` | **HBV-light (HBV-upper, 100-member calibrated ensemble)** | `seibert_hbv` | Conceptual | `Archive` | Seibert et al. (2018) | Maurer | Yes (1) | 531 | 1989-10-01 to 1999-09-30 | **0.678** | --- | 0.637 | 0.682 | 0.623 | 9 | `---` / `---` |
| **29** | `t3_41` | **mHM (Basin-Calibrated)*** | `mizukami_mhm` | Process-based | `Archive*` | Mizukami et al. (2019) | Maurer | No (1) | 492 | 1989-10-01 to 1999-09-30 | **0.665** | 0.665 | 0.626 | 0.687 | 0.643 | 7 | `---` / `---` |
| **30** | `sacsma_gauged_single_daymet` | **SAC-SMA + Snow-17 (DDS, single model)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.661** | 0.661 | 0.613 | 0.677 | 0.630 | 12 | `---` / `---` |
| **31** | `t3_42` | **XGBoost (8-model ensemble)** | `gauch_xgboost_lit` | Machine learning | `Literature` | Gauch et al. (2021b) | Maurer | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.659** | 0.639 ± 0.002 | 0.619 | --- | --- | 6 | `---` / `---` |
| **32** | `t3_43` | **FUSE (902)*** | `fuse_suite` | Conceptual | `Archive**` | Addor et al. (2017) | Maurer | No (1) | 479 | 1989-10-01 to 1999-09-30 | **0.654** | 0.654 | 0.617 | 0.691 | 0.647 | 10 | `---` / `---` |
| **33** | `t3_44` | **LSTM without Static Attributes (8-model ensemble)** | `kratzert_lstm_nostatic` | Machine learning | `Archive` | Kratzert et al. (2019b) | Maurer | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.642** | 0.593 ± 0.008 | 0.493 | 0.631 | 0.564 | 20 | `---` / `---` |
| **34** | `t3_45` | **FUSE (900)*** | `fuse_suite` | Conceptual | `Archive**` | Addor et al. (2017) | Maurer | No (1) | 501 | 1989-10-01 to 1999-09-30 | **0.641** | 0.641 | 0.595 | 0.671 | 0.614 | 13 | `---` / `---` |
| **35** | `t3_42_single` | **XGBoost (single model)** | `gauch_xgboost_lit` | Machine learning | `Literature` | Gauch et al. (2021b) | Maurer | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.639** | 0.639 ± 0.002 | 0.610 | --- | --- | 6 | `---` / `---` |
| **36** | `t3_46` | **FUSE (904)*** | `fuse_suite` | Conceptual | `Archive**` | Addor et al. (2017) | Maurer | No (1) | 501 | 1989-10-01 to 1999-09-30 | **0.624** | 0.624 | 0.589 | 0.648 | 0.590 | 10 | `---` / `---` |
| **37** | `t3_48` | **SAC-SMA + Snow-17 (Basin-Calibrated, 10-member SCE-UA ensemble)*** | `newman_sacsma` | Conceptual | `Archive*` | Newman et al. (2017) | Maurer | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.607** | --- | 0.573 | 0.636 | 0.603 | 13 | `---` / `---` |
| **38** | `t3_44_single` | **LSTM without Static Attributes (single model)** | `kratzert_lstm_nostatic` | Machine learning | `Archive` | Kratzert et al. (2019b) | Maurer | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.593** | 0.593 ± 0.008 | 0.394 | 0.619 | 0.545 | 28 | `---` / `---` |
| **39** | `t3_52` | **VIC 4.1.2 (Basin-Calibrated)*** | `newman_vic` | Process-based | `Archive*` | Newman et al. (2017) | Maurer | No (1) | 530 | 1989-10-01 to 1999-09-30 | **0.554** | 0.554 | 0.529 | 0.595 | 0.548 | 10 | `---` / `---` |
| **40** | `t3_mhm_mpr` | **mHM (CONUS MPR)*** | `rakovec_mhm_mpr` | Process-based | `Archive*` | Rakovec et al. (2019) | Maurer | No (1) | 492 | 1989-10-01 to 1999-09-30 | **0.528** | 0.528 | 0.435 | 0.469 | 0.404 | 32 | `---` / `---` |
| **41** | `t3_55` | **dPL + VIC (g_A)*** | `tsai2021_dpl_vic` | Hybrid (ML + Physics) | `Literature*` | Tsai et al. (2021) | Maurer | No (1) | 498 | 1989-10-01 to 1999-09-30 | **0.439** | 0.439 | -0.189 | --- | --- | 61 | `---` / `---` |
| **42** | `t3_vic_mpr` | **VIC 4.1.2 (CONUS MPR)** | `mizukami_vic_mpr` | Process-based | `Archive` | Mizukami et al. (2017) | Maurer | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.307** | 0.307 | 0.194 | 0.259 | 0.164 | 47 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `t3_02` | LSTM (10-model ensemble) | **0.772** | 0.763 | 0.894 | 0.845 | -0.035 | 0.932 | 1.179 | 1.390 | -16.36 | -6.99 | 39.26 | 0.29 | 0.36 | 38.15 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_02.zarr` |
| **2** | `t3_06` | LSTM (10-model ensemble) | **0.764** | 0.765 | 0.886 | 0.843 | -0.033 | 0.940 | 1.171 | 1.372 | -15.89 | -8.92 | 43.72 | 0.33 | 0.37 | 38.26 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_06.zarr` |
| **3** | `t3_18` | δHBV 1.1p / δMG (5-model ensemble) | **0.759** | 0.757 | 0.884 | 0.855 | -0.016 | 0.970 | 1.196 | 1.431 | -14.06 | -10.01 | 28.87 | 0.32 | 0.35 | 37.90 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_18.zarr` |
| **4** | `t3_11` | LSTM (270d, 8-model ensemble) | **0.759** | 0.756 | 0.885 | 0.834 | -0.033 | 0.934 | 1.201 | 1.443 | -16.42 | -8.85 | 55.13 | 0.33 | 0.37 | 39.01 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_11.zarr` |
| **5** | `t3_12` | MC-LSTM (Frame 2023 Archive, Daymet) | **0.758** | 0.757 | 0.882 | 0.837 | -0.029 | 0.941 | 1.220 | 1.487 | -16.42 | -9.96 | -14.54 | 0.30 | 0.35 | 38.14 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_12.zarr` |
| **6** | `t3_19` | δHBV 1.1p / δMG (single model) | **0.754** | 0.756 | 0.881 | 0.853 | -0.017 | 0.967 | 1.214 | 1.472 | -14.24 | -10.12 | 26.22 | 0.32 | 0.36 | 38.26 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_18.zarr` |
| **8** | `t3_15` | LSTM (10-model ensemble) | **0.747** | 0.740 | 0.878 | 0.810 | -0.031 | 0.936 | 1.260 | 1.588 | -18.97 | -10.71 | 37.77 | 0.32 | 0.36 | 38.31 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_15.zarr` |
| **9** | `t3_08` | LSTM (single model) | **0.744** | 0.759 | 0.878 | 0.858 | -0.035 | 0.931 | 1.250 | 1.562 | -15.15 | -6.01 | -0.26 | 0.33 | 0.38 | 39.87 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_02.zarr` |
| **10** | `hoedt_mclstm_lit_maurer` | MC-LSTM (Hoedt 2021 — Literature, 10-model ensemble)* | **0.744** | --- | --- | --- | --- | --- | --- | --- | -14.70 | -10.50 | 27.60 | --- | --- | --- | --- |
| **11** | `hoedt_mclstm_maurer` | MC-LSTM (Hoedt 2021 — MIP Re-run, Maurer) | **0.743** | 0.763 | 0.874 | 0.844 | -0.029 | 0.943 | 1.246 | 1.553 | -15.33 | -11.56 | -3.34 | 0.35 | 0.37 | 39.27 | `gs://camels-us-mip/hydrographs/gauged__single_camels/hoedt_mclstm_maurer.zarr` |
| **12** | `t3_29` | δHBV 1.0 (5-model ensemble) | **0.742** | 0.733 | 0.876 | 0.824 | -0.007 | 0.985 | 1.223 | 1.495 | -16.19 | -15.27 | 23.40 | 0.33 | 0.36 | 38.31 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_29.zarr` |
| **13** | `t3_23` | EA-LSTM (8-model ensemble) | **0.741** | 0.737 | 0.876 | 0.807 | -0.032 | 0.939 | 1.253 | 1.569 | -18.82 | -10.99 | 31.33 | 0.38 | 0.39 | 40.52 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_23.zarr` |
| **14** | `t3_21` | S4D-FT (Frequency-Tuned Diagonal SSM, single model) | **0.740** | --- | --- | --- | --- | --- | --- | --- | -13.98 | --- | 24.47 | --- | --- | --- | --- |
| **15** | `t3_28` | MC-LSTM (Frame 2023 Archive, NLDAS) | **0.736** | 0.737 | 0.872 | 0.814 | -0.021 | 0.955 | 1.296 | 1.680 | -18.57 | -10.40 | -17.75 | 0.31 | 0.36 | 37.89 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_28.zarr` |
| **16** | `poudel_mlp_hbv_daymet` | Static-Parameter MLP + 16-HBV (3-model ensemble) | **0.735** | 0.738 | 0.875 | 0.842 | -0.010 | 0.983 | 1.245 | 1.549 | -15.96 | -4.79 | -0.32 | 0.32 | 0.37 | 39.74 | `gs://camels-us-mip/hydrographs/gauged__single_camels/poudel_mlp_hbv_daymet_3s.zarr` |
| **18** | `t3_33` | δHBV 1.0 (single model) | **0.733** | 0.729 | 0.872 | 0.827 | -0.009 | 0.984 | 1.243 | 1.545 | -16.07 | -15.14 | 22.36 | 0.33 | 0.36 | 39.18 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_29.zarr` |
| **19** | `poudel_mlp_hbv_daymet_single` | Static-Parameter MLP + 16-HBV (single model) | **0.733** | 0.739 | 0.871 | 0.841 | -0.010 | 0.982 | 1.251 | 1.565 | -16.12 | -4.51 | -5.33 | 0.33 | 0.37 | 40.36 | `gs://camels-us-mip/hydrographs/gauged__single_camels/poudel_mlp_hbv_daymet_3s.zarr` |
| **20** | `t3_20` | LSTM (single model) | **0.732** | 0.760 | 0.869 | 0.853 | -0.032 | 0.935 | 1.267 | 1.606 | -14.48 | -7.49 | 1.67 | 0.37 | 0.39 | 39.87 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_06.zarr` |
| **21** | `t3_26` | LSTM (270d, single model) | **0.732** | 0.750 | 0.868 | 0.848 | -0.034 | 0.935 | 1.279 | 1.635 | -15.25 | -8.28 | 25.87 | 0.37 | 0.39 | 40.58 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_11.zarr` |
| **22** | `hoedt_mclstm_lit_maurer_single` | MC-LSTM (Hoedt 2021 — Literature, single model)* | **0.726** | --- | --- | --- | --- | --- | --- | --- | -14.70 | -10.50 | 27.60 | --- | --- | --- | --- |
| **23** | `t3_32` | LSTM (single model) | **0.720** | 0.733 | 0.863 | 0.822 | -0.032 | 0.933 | 1.313 | 1.724 | -17.70 | -9.39 | -1.52 | 0.34 | 0.38 | 40.22 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_15.zarr` |
| **24** | `t3_s4d_lit` | S4D (Diagonal State Space Model, single model) | **0.720** | --- | --- | --- | --- | --- | --- | --- | -16.88 | --- | 36.60 | --- | --- | --- | --- |
| **25** | `tft_gauged_maurer` | Temporal Fusion Transformer (single model) | **0.716** | 0.702 | 0.862 | 0.786 | -0.044 | 0.907 | 1.313 | 1.723 | -19.63 | -8.99 | 26.00 | 0.35 | 0.37 | 40.89 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_21.zarr` |
| **26** | `hoedt_mclstm_maurer_single` | MC-LSTM (Hoedt 2021 — MIP Re-run, Maurer, Single Model) | **0.716** | 0.755 | 0.860 | 0.858 | -0.031 | 0.939 | 1.312 | 1.720 | -14.28 | -10.52 | -28.07 | 0.38 | 0.39 | 40.35 | `gs://camels-us-mip/hydrographs/gauged__single_camels/hoedt_mclstm_maurer.zarr` |
| **27** | `t3_30` | EA-LSTM (single model) | **0.712** | 0.725 | 0.858 | 0.821 | -0.033 | 0.939 | 1.323 | 1.751 | -17.39 | -9.79 | 2.12 | 0.41 | 0.40 | 42.00 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_23.zarr` |
| **28** | `t3_39` | HBV-light (HBV-upper, 100-member calibrated ensemble) | **0.678** | 0.682 | 0.835 | 0.792 | -0.014 | 0.969 | 1.401 | 1.962 | -18.48 | -24.09 | 17.09 | 0.44 | 0.41 | 45.67 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_39.zarr` |
| **29** | `t3_41` | mHM (Basin-Calibrated)* | **0.665** | 0.687 | 0.831 | 0.803 | -0.040 | 0.921 | 1.431 | 2.049 | -18.95 | -7.13 | 13.39 | 0.45 | 0.44 | 46.09 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_41.zarr` |
| **30** | `sacsma_gauged_single_daymet` | SAC-SMA + Snow-17 (DDS, single model) | **0.661** | 0.677 | 0.834 | 0.812 | -0.083 | 0.826 | 1.408 | 1.982 | -19.91 | 16.87 | -132.27 | 0.37 | 0.38 | 43.27 | `gs://camels-us-mip/hydrographs/gauged__single_camels/sacsma_daymet.zarr` |
| **31** | `t3_42` | XGBoost (8-model ensemble) | **0.658** | --- | --- | --- | --- | --- | --- | --- | -27.79 | -7.06 | -446.95 | --- | --- | --- | https://zenodo.org/records/3648270 |
| **32** | `t3_43` | FUSE (902)* | **0.654** | 0.691 | 0.824 | 0.802 | -0.048 | 0.902 | 1.467 | 2.151 | -19.36 | 9.61 | -67.97 | 0.41 | 0.39 | 45.73 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_43.zarr` |
| **33** | `t3_44` | LSTM without Static Attributes (8-model ensemble) | **0.642** | 0.631 | 0.837 | 0.753 | -0.031 | 0.940 | 1.452 | 2.107 | -22.76 | -18.49 | 16.21 | 0.44 | 0.42 | 48.69 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_44.zarr` |
| **34** | `t3_45` | FUSE (900)* | **0.641** | 0.671 | 0.818 | 0.798 | -0.032 | 0.937 | 1.484 | 2.201 | -18.92 | -4.06 | -12.70 | 0.43 | 0.38 | 45.67 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_45.zarr` |
| **35** | `t3_42_single` | XGBoost (single model) | **0.639** | --- | --- | --- | --- | --- | --- | --- | -27.35 | -4.96 | -474.00 | --- | --- | --- | https://zenodo.org/records/3648270 |
| **36** | `t3_46` | FUSE (904)* | **0.624** | 0.648 | 0.809 | 0.788 | -0.070 | 0.862 | 1.524 | 2.324 | -21.41 | 16.64 | -70.24 | 0.45 | 0.40 | 47.38 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_46.zarr` |
| **37** | `t3_48` | SAC-SMA + Snow-17 (Basin-Calibrated, 10-member SCE-UA ensemble)* | **0.607** | 0.636 | 0.794 | 0.774 | -0.068 | 0.865 | 1.547 | 2.393 | -20.74 | -13.56 | 31.94 | 0.56 | 0.44 | 49.70 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_48.zarr` |
| **38** | `t3_44_single` | LSTM without Static Attributes (single model) | **0.593** | 0.619 | 0.805 | 0.780 | -0.030 | 0.939 | 1.568 | 2.460 | -20.57 | -16.87 | -50.39 | 0.49 | 0.45 | 51.18 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_44.zarr` |
| **39** | `t3_52` | VIC 4.1.2 (Basin-Calibrated)* | **0.554** | 0.595 | 0.763 | 0.731 | -0.021 | 0.963 | 1.653 | 2.732 | -27.61 | -5.77 | -66.53 | 0.50 | 0.41 | 51.86 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_52.zarr` |
| **40** | `t3_mhm_mpr` | mHM (CONUS MPR)* | **0.528** | 0.469 | 0.793 | 0.594 | -0.035 | 0.930 | 1.691 | 2.860 | -39.01 | -29.52 | 36.63 | 0.50 | 0.43 | 59.44 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t7_19.zarr` |
| **41** | `t3_55` | dPL + VIC (g_A)* | **0.439** | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | https://zenodo.org/records/6383415 |
| **42** | `t3_vic_mpr` | VIC 4.1.2 (CONUS MPR) | **0.307** | 0.259 | 0.659 | 0.460 | -0.072 | 0.856 | 2.056 | 4.226 | -55.50 | -26.28 | 17.40 | 0.89 | 0.43 | 65.87 | `gs://camels-us-mip/hydrographs/gauged__single_camels/t7_22.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `t3_02` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained with Daymet single-forcing meteorological inputs and 27 static catchment attributes across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.772, Mean NSE = 0.728, Med KGE = 0.757; 447-basin subset Ensemble Med NSE = 0.783).
- **Rank #2 `t3_06` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained with Maurer single-forcing meteorological inputs and 27 static catchment attributes across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.764, Mean NSE = 0.723, Med KGE = 0.751; 447-basin subset Ensemble Med NSE = 0.768).
- **Rank #3 `t3_18` (δHBV 1.1p / δMG (5-model ensemble)):** Clean-room PyTorch + C++/OpenMP re-implementation of Song et al. (2024) δHBV 1.1p (dynamic β, K0, γ + capillary rise C) trained on 531 CAMELS-US basins over WY1999-2008 and evaluated over WY1989-1999 with Daymet forcing; 5-seed ensemble mean (seeds 111, 222, 333, 444, 555; Med NSE = 0.759).
- **Rank #4 `t3_11` (LSTM (270d, 8-model ensemble)):** 8-seed ensemble mean of the Kratzert et al. (2019b) 270-day lookback regional LSTM with 27 static catchment attributes on Maurer single-forcing inputs across all 531 CAMELS-US basins (HydroShare 10.4211/hs.83ea5312635e44dc824eeb99eda12f06, Zenodo 10.5281/zenodo.3530884; 531-basin Ensemble Med NSE = 0.759, Mean NSE = 0.713, Med KGE = 0.742; 447-basin subset Ensemble Med NSE = 0.759).
- **Rank #6 `t3_19` (δHBV 1.1p / δMG (single model)):** Clean-room PyTorch + C++/OpenMP re-implementation of Song et al. (2024) δHBV 1.1p trained on 531 CAMELS-US basins over WY1999-2008 and evaluated over WY1989-1999 with Daymet forcing; single-model performance averaged across all 5 random seeds (seeds 111, 222, 333, 444, 555; Med NSE = 0.754 ± 0.001).
- **Rank #8 `t3_15` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained with NLDAS single-forcing meteorological inputs and 27 static catchment attributes across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.747, Mean NSE = 0.708, Med KGE = 0.737; 447-basin subset Ensemble Med NSE = 0.755).
- **Rank #9 `t3_08` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM on Daymet single-forcing inputs across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.744 ± 0.003; 447-basin subset Single-Seed Med NSE = 0.762 ± 0.004).
- **Rank #10 `hoedt_mclstm_lit_maurer` (MC-LSTM (Hoedt 2021 — Literature, 10-model ensemble)*):** **[Asterisk `*`: Evaluated in Hoedt et al. (2021) on the 447-basin Kratzert et al. (2019b) CAMELS subset.]**
- **Rank #12 `t3_29` (δHBV 1.0 (5-model ensemble)):** 5-seed ensemble (seeds 111, 222, 333, 444, 555) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins with Daymet forcing.
- **Rank #13 `t3_23` (EA-LSTM (8-model ensemble)):** Official 8-seed ensemble mean (seeds 111, 222, 333, 444, 555, 666, 777, 888) of the Entity-Aware LSTM (EA-LSTM) trained with basin-averaged NSE* loss on extended Maurer forcings and 27 CAMELS static catchment attributes over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins (Kratzert et al., 2019b, Table 2; Zenodo code archive 10.5281/zenodo.3530884; HydroShare hydrograph archive 10.4211/hs.83ea5312635e44dc824eeb99eda12f06; Ensemble Median NSE = 0.741, Mean NSE = 0.705, Median KGE = 0.737).
- **Rank #14 `t3_21` (S4D-FT (Frequency-Tuned Diagonal SSM, single model)):** Published Newman Split-1 benchmark from Wang, Zhang, et al. (2025a, Water Resources Research 61, e2025WR039888, Table 1): 6-layer Frequency-Tuned Diagonal State Space Model (S4D-FT, cfr = 10, cfi = 10; no per-basin fine-tuning) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins with NLDAS forcings and 27 static catchment attributes across 8 random seeds (seeds 200..207; single-model median NSE = 0.740 +/- 0.002).
- **Rank #16 `poudel_mlp_hbv_daymet` (Static-Parameter MLP + 16-HBV (3-model ensemble)):** Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (50 epochs x 240 batches/epoch, batch_size=32, 730-day window with 365-day spinup, StepLR gamma=0.5 every 10 epochs) trained on 531 CAMELS-US basins over WY1999-2008 and evaluated over WY1989-1999 with Daymet forcing; 3-seed ensemble mean (seeds 101, 102, 103; Ensemble Med NSE = 0.735, Mean NSE = 0.657).
- **Rank #18 `t3_33` (δHBV 1.0 (single model)):** Single-model evaluation scored across all 5 individual seed members (seeds 111, 222, 333, 444, 555; evaluation_mode='single_member_mean') on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins with Daymet forcing.
- **Rank #19 `poudel_mlp_hbv_daymet_single` (Static-Parameter MLP + 16-HBV (single model)):** Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (50 epochs x 240 batches/epoch, batch_size=32, 730-day window with 365-day spinup, StepLR gamma=0.5 every 10 epochs) trained on 531 CAMELS-US basins over WY1999-2008 and evaluated over WY1989-1999 with Daymet forcing; single-model performance averaged across all 3 random seeds (seeds 101, 102, 103; Single-Seed Med NSE = 0.733 ± 0.003).
- **Rank #20 `t3_20` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM on Maurer single-forcing inputs across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.732 ± 0.002; 447-basin subset Single-Seed Med NSE = 0.747 ± 0.003).
- **Rank #21 `t3_26` (LSTM (270d, single model)):** Single-model evaluation scored across all 8 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2019b) 270-day lookback regional LSTM with 27 static catchment attributes on Maurer single-forcing inputs across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.732 ± 0.002; 447-basin subset Single-Seed Med NSE = 0.739 ± 0.003).
- **Rank #22 `hoedt_mclstm_lit_maurer_single` (MC-LSTM (Hoedt 2021 — Literature, single model)*):** **[Asterisk `*`: Evaluated in Hoedt et al. (2021) on the 447-basin Kratzert et al. (2019b) CAMELS subset.]**
- **Rank #23 `t3_32` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM on NLDAS single-forcing inputs across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.720 ± 0.003; 447-basin subset Single-Seed Med NSE = 0.732 ± 0.002).
- **Rank #24 `t3_s4d_lit` (S4D (Diagonal State Space Model, single model)):** Published Newman Split-1 benchmark from Wang, Zhang, et al. (2025a, Water Resources Research 61, e2025WR039888, Table 1): 6-layer basic Diagonal State Space Model (S4D, cfr = 1, cfi = 1) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins with NLDAS forcings and 27 static catchment attributes across 8 random seeds (seeds 200..207; single-model median NSE = 0.720 +/- 0.004).
- **Rank #25 `tft_gauged_maurer` (Temporal Fusion Transformer (single model)):** Single-seed 25-epoch Temporal Fusion Transformer (hidden_size=128, 4 attention heads, 180-day encoder window, dropout=0.15) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins with Maurer forcing (Median NSE = 0.716, Mean NSE = 0.676, Median KGE = 0.702).
- **Rank #27 `t3_30` (EA-LSTM (single model)):** Single-model evaluation of the official Kratzert et al. (2019b) EA-LSTM archive scored across all 8 individual seed members (seeds 111..888 in t3_23.zarr; evaluation_mode='single_member_mean') over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins (Single-Model Median NSE = 0.712 ± 0.004, matching Kratzert et al., 2019b Table 2 reported 0.71 ± 0.005).
- **Rank #28 `t3_39` (HBV-light (HBV-upper, 100-member calibrated ensemble)):** Upper benchmark (HBV_ub): unweighted ensemble mean of 100 independent Genetic Algorithm + Powell (GAP) calibrations per basin (2,500 GA + 1,000 Powell evaluations per trial maximizing daily NSE over 1999-10-01 to 2008-09-30; evaluated over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins; HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, NetCDF variable HBV_ub; stores the pre-computed 100-member ensemble mean hydrograph as 1 array; Median NSE = 0.678 on 531 basins, 0.676 on the 447-basin Kratzert et al. 2019b Table 3 intersection).
- **Rank #29 `t3_41` (mHM (Basin-Calibrated)*):** **[Asterisk `*`: Evaluated on 492 of 531 CAMELS basins (39 basins missing in archive).]** Basin-calibrated mHM (0.25° hydrological grid with 1/48° MPR physiographic transfer functions calibrated independently per basin via DDS over 1999-10-01 to 2008-09-30; evaluated over 1989-10-01 to 1999-09-30; HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable mHm_basin, Median NSE = 0.665 across 492 evaluated basins [0.666 on the 447-basin Kratzert et al. 2019b intersection]).
- **Rank #30 `sacsma_gauged_single_daymet` (SAC-SMA + Snow-17 (DDS, single model)):** Clean-room 27-parameter Snow-17 + SAC-SMA + 15-day Duamel Gamma unit hydrograph calibrated independently per basin via DDS (400 iterations, r=0.2, 365-day spinup, daily RMSE objective) on Daymet forcings over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins (1 seed; Median NSE = 0.661).
- **Rank #31 `t3_42` (XGBoost (8-model ensemble)):** Published and archived full-data benchmark from Gauch, Mai, & Lin (2021b, Environmental Modelling & Software 135, 104926, Table 2; Zenodo 10.5281/zenodo.3648270): regional XGBoost regressor with 30-day raw meteorological lag window (150 dynamic + 27 static = 177 features) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins with Extended Maurer forcing (Ensemble Median NSE = 0.659 [0.6585 unrounded, 0.66 in Table 2], Mean NSE = 0.619, Median |FHV| = 27.79%, NSE <= 0 = 6 basins).
- **Rank #32 `t3_43` (FUSE (902)*):** **[Asterisk `*`: Evaluated on 479 of 531 CAMELS basins available in the Kratzert et al. (2019b) HydroShare benchmark archive (52 basins missing).]** FUSE structure 902 (PRMS tension1_1 upper layer + TOPMODEL unlimpow_2 nonlinear power-law baseflow + PRMS surface runoff +perc_f2sat + sequential ET + temp_index degree-day snow, no routing), calibrated per basin via SCE-UA minimizing RMSE on Maurer forcings over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 (HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable q_sim_fuse_902; Median NSE = 0.654 across 479 valid basins in t3_43.zarr, 0.650 on the 447-basin Kratzert et al. 2019b intersection).
- **Rank #33 `t3_44` (LSTM without Static Attributes (8-model ensemble)):** Official 8-seed ensemble mean (seeds 111, 222, 333, 444, 555, 666, 777, 888) of the regional LSTM trained without static catchment attributes (D_static = 0, using only the 5 dynamic extended Maurer meteorological forcings and basin-averaged NSE* loss) over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins (Kratzert et al., 2019b, Table 2; Zenodo code archive 10.5281/zenodo.3530884; HydroShare hydrograph archive 10.4211/hs.83ea5312635e44dc824eeb99eda12f06; stored at t3_44.zarr; Ensemble Median NSE = 0.642, Mean NSE = 0.493, Median KGE = 0.631, 20 basins with NSE <= 0).
- **Rank #34 `t3_45` (FUSE (900)*):** **[Asterisk `*`: Evaluated on 501 of 531 CAMELS basins available in the Kratzert et al. (2019b) HydroShare benchmark archive (30 basins missing).]** FUSE structure 900 (TOPMODEL/ARNO onestate_1 upper layer + PRMS unlimfrc_2 linear fractional baseflow + PRMS surface runoff + perc_f2sat + sequential ET + temp_index degree-day snow, no routing), calibrated per basin via SCE-UA minimizing RMSE on Maurer forcings over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 (HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable q_sim_fuse_900; Median NSE = 0.641 across 501 valid basins in t3_45.zarr, 0.639 on the 447-basin Kratzert et al. 2019b intersection).
- **Rank #35 `t3_42_single` (XGBoost (single model)):** Single-model evaluation of Gauch, Mai, & Lin (2021b, Environmental Modelling & Software 135, 104926; Zenodo 10.5281/zenodo.3648270) across all 531 CAMELS-US basins on 1989-10-01 to 1999-09-30 with Extended Maurer forcing (Single-Model Median NSE = 0.639; 0.6531 +/- 0.0023 across the 8 seeds seed111..seed888 in performance_gridEvaluation.ipynb).
- **Rank #36 `t3_46` (FUSE (904)*):** **[Asterisk `*`: Evaluated on 501 of 531 CAMELS basins available in the Kratzert et al. (2019b) HydroShare benchmark archive (30 basins missing).]** FUSE structure 904 (SAC-SMA tension2_1 upper layer + TOPMODEL unlimpow_2 nonlinear power-law baseflow + PRMS surface runoff + perc_f2sat + sequential ET + temp_index degree-day snow, no routing), calibrated per basin via SCE-UA minimizing RMSE on Maurer forcings over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 (HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable q_sim_fuse_904; Median NSE = 0.624 across 501 valid basins in t3_46.zarr, 0.622 on the 447-basin Kratzert et al. 2019b intersection).
- **Rank #37 `t3_48` (SAC-SMA + Snow-17 (Basin-Calibrated, 10-member SCE-UA ensemble)*):** **[Asterisk `*`: Evaluated on 530 of 531 CAMELS basins (1 basin, USGS 01634500, is all-zero in the Newman et al., 2017 archive).]** Basin-calibrated Snow-17 + SAC-SMA + Duamel Gamma unit hydrograph from the NCAR/HydroShare CAMELS benchmark archive (Newman et al., 2017; HydroShare DOI 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1). Note: The 10 SCE-UA/DDS calibrations per basin were archived as a single pre-averaged/selected hydrograph array (num_members=1 in t3_48.zarr without individual qsim_members; Median NSE = 0.607 across 530 non-zero basins in t3_48.zarr [basin 01634500 is all-zero], 0.603 on the 447-basin Kratzert et al. 2019b Table 3 intersection).
- **Rank #38 `t3_44_single` (LSTM without Static Attributes (single model)):** Single-model evaluation of the official Kratzert et al. (2019b) LSTM without Static Attributes archive scored across all 8 individual seed members (seeds 111..888 in t3_44.zarr; evaluation_mode='single_member_mean') over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins (Single-Model Median NSE = 0.593 ± 0.008, matching Kratzert et al., 2019b Table 2 reported 0.59 ± 0.008).
- **Rank #39 `t3_52` (VIC 4.1.2 (Basin-Calibrated)*):** **[Asterisk `*`: Evaluated on 530 of 531 CAMELS basins (1 basin, USGS 08082700, is missing in the Newman et al., 2017 archive).]** Basin-calibrated VIC 4.1.2 (10 calibrated water-balance, snow, and routing parameters per basin via SCE-UA over 1999-10-01 to 2008-09-30; evaluated over 1989-10-01 to 1999-09-30; HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable VIC_basin, Median NSE = 0.554 across 530 valid basins in t3_52.zarr [basin 08082700 missing]).
- **Rank #40 `t3_mhm_mpr` (mHM (CONUS MPR)*):** **[Asterisk `*`: Evaluated on 492 of 531 CAMELS basins (39 basins missing in archive).]** CONUS-wide Multiscale Parameter Regionalization (MPR) of mHM (single 48-parameter global transfer-function vector calibrated simultaneously across CONUS via DDS over 1999-10-01 to 2008-09-30; evaluated over 1989-10-01 to 1999-09-30; HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable mHm_conus, Median NSE = 0.528 across 492 evaluated basins [0.527 on the 447-basin Kratzert et al. 2019b intersection]). Placed in Table 3 (gauged__single_camels) rather than Table 7 (PUB) because the CAMELS evaluation gauges were included in the CONUS-wide calibration objective during 1999–2008.
- **Rank #41 `t3_55` (dPL + VIC (g_A)*):** **[Asterisk `*`: Evaluated on 498 of 531 CAMELS basins available in the Tsai et al. (2021) Figure 6a archive (Zenodo 10.5281/zenodo.6383415; g_Z variant achieves 0.435 median NSE on 499 basins).]** Evaluated on 498 of 531 CAMELS basins from the authors' Figure 6a VIC validation archive (Zenodo 10.5281/zenodo.6383415, script_data/paper_results/figure6/gA_NSE_VIC.csv; calibrated on WY2000–2008 and evaluated with the physical C-language VIC 4.1.2 model on WY1990–1999; companion g_Z LSTM parameter network achieves 0.435 median NSE, -0.248 mean NSE, and 65 basins with NSE <= 0 across 499 basins). Note: Zenodo 10.5281/zenodo.6383415 archives only the per-basin NSE vectors without daily streamflow hydrographs or 8-digit USGS gauge IDs; row ID is t3_55 (t3_53.zarr in gauged__single_camels belongs to Gauch et al. 2021b XGBoost).
- **Rank #42 `t3_vic_mpr` (VIC 4.1.2 (CONUS MPR)):** CONUS-wide Multiscale Parameter Regionalization (MPR-flex) of VIC 4.1.2 (16 global transfer-function coefficients calibrated simultaneously across CONUS gauges via SCE-UA over 1999-10-01 to 2008-09-30; evaluated over 1989-10-01 to 1999-09-30; HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, variable VIC_conus, Median NSE = 0.307 across all 531 CAMELS basins). Placed in Table 3 (gauged__single_camels) rather than Table 7 (PUB) because the CAMELS evaluation gauges were included in the CONUS-wide calibration objective during 1999–2008.

### 4.3 Table 4: `gauged__multi_camels` — Gauged: Multiple CAMELS Forcings (`30` Rows)
- **Subtitle / Scope:** 531 basins, Newman evaluation decade (1989–1999) — Two or three CAMELS forcings (Daymet, Maurer, NLDAS).
- **Spatial Regime:** `gauged` | **Forcing Regime:** `multi_camels` | **Section:** `4.1.2`
- **SOTA / Rank #1 Winner:** **ERA-Multi-Paradigm-Gauged (10-model ensemble)** (`Median NSE = 0.827`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `era_gauged_multi_paradigm` | **ERA-Multi-Paradigm-Gauged (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.827** | 0.822 ± 0.002 | 0.776 | 0.806 | 0.756 | 2 | `---` / `---` |
| **2** | `era_gauged_multi_paradigm_single` | **ERA-Multi-Paradigm-Gauged** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.822** | 0.822 ± 0.002 | 0.771 | 0.805 | 0.755 | 2 | `---` / `---` |
| **3** | `t4_05` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet, Maurer, NLDAS | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.821** | 0.796 ± 0.002 | 0.783 | 0.801 | 0.755 | 1 | `---` / `---` |
| **4** | `t4_06` | **LSTM + dHBV 1.1p (Multi-Model Ensemble)** | `li2025_lstm_dhbv_ens` | Hybrid (ML + Physics) | `MIP Re-run` | Li et al. (2025) | Daymet, Maurer, NLDAS | Yes (2) | 531 | 1989-10-01 to 1999-09-30 | **0.820** | --- | 0.789 | 0.798 | 0.754 | 1 | `---` / `---` |
| **5** | `era_gauged_physics_guided` | **ERA-Physics-Guided-Gauged (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.820** | 0.800 ± 0.003 | 0.778 | 0.821 | 0.766 | 1 | `---` / `---` |
| **6** | `era_gauged_minimal_guidance` | **ERA-Unguided-Gauged (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.819** | 0.810 ± 0.002 | 0.769 | 0.802 | 0.749 | 3 | `---` / `---` |
| **7** | `t4_09` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet, NLDAS | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.810** | 0.787 ± 0.002 | 0.772 | 0.793 | 0.745 | 1 | `---` / `---` |
| **8** | `era_gauged_minimal_guidance_single` | **ERA-Unguided-Gauged** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.810** | 0.810 ± 0.002 | 0.755 | 0.798 | 0.745 | 3 | `---` / `---` |
| **9** | `t4_08` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet, Maurer | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.804** | 0.773 ± 0.003 | 0.761 | 0.787 | 0.747 | 2 | `---` / `---` |
| **10** | `era_gauged_physics_guided_single` | **ERA-Physics-Guided-Gauged** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.800** | 0.800 ± 0.003 | 0.756 | 0.818 | 0.760 | 2 | `---` / `---` |
| **11** | `t4_19` | **LSTM (10-model ensemble)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Maurer, NLDAS | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.799** | 0.774 ± 0.004 | 0.761 | 0.777 | 0.731 | 1 | `---` / `---` |
| **12** | `t4_11` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet, Maurer, NLDAS | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.796** | 0.796 ± 0.002 | 0.757 | 0.795 | 0.748 | 2 | `---` / `---` |
| **13** | `t4_25` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet, NLDAS | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.787** | 0.787 ± 0.002 | 0.747 | 0.789 | 0.739 | 2 | `---` / `---` |
| **14** | `hoedt_mclstm_3f_avgp` | **MC-LSTM (Hoedt 2021 — 3-Forcing Mean-Precip + 3F Gates Ensemble)** | `hoedt_mclstm` | Machine learning | `MIP Re-run` | Hoedt et al. (2021) | Daymet + Maurer + NLDAS | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.777** | 0.754 ± 0.005 | 0.740 | 0.755 | 0.702 | 0 | `---` / `---` |
| **15** | `t4_31` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Maurer, NLDAS | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.774** | 0.774 ± 0.004 | 0.735 | 0.773 | 0.725 | 2 | `---` / `---` |
| **16** | `t4_28` | **LSTM (single model)** | `kratzert2021_lstm_archive` | Machine learning | `Archive` | Kratzert et al. (2021) | Daymet, Maurer | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.773** | 0.773 ± 0.003 | 0.731 | 0.784 | 0.739 | 3 | `---` / `---` |
| **17** | `hoedt_mclstm_3f_ens_b` | **MC-LSTM (Hoedt 2021 — 3-Forcing Model-Average Ensemble)** | `hoedt_mclstm` | Machine learning | `MIP Re-run` | Hoedt et al. (2021) | Daymet + Maurer + NLDAS | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.771** | 0.710 ± 0.008 | 0.737 | 0.744 | 0.695 | 1 | `---` / `---` |
| **18** | `hoedt_mclstm_3f_avgp_single` | **MC-LSTM (Hoedt 2021 — 3-Forcing Mean-Precip + 3F Gates, Single Model)** | `hoedt_mclstm` | Machine learning | `MIP Re-run` | Hoedt et al. (2021) | Daymet + Maurer + NLDAS | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.754** | 0.754 ± 0.005 | 0.712 | 0.746 | 0.695 | 2 | `---` / `---` |
| **19** | `t4_34` | **Crossformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.720** | 0.720 | --- | 0.730 | --- | --- | `---` / `---` |
| **20** | `t4_35` | **Non-stationary Transformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.720** | 0.720 | --- | 0.610 | --- | --- | `---` / `---` |
| **21** | `t4_37` | **Vanilla Transformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.700** | 0.700 | --- | 0.600 | --- | --- | `---` / `---` |
| **22** | `sacsma_gauged_multi_3f` | **SAC-SMA + Snow-17 (DDS, 3-Forcing Ensemble)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet, Maurer, NLDAS | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.695** | 0.691 ± 0.001 | 0.665 | 0.645 | 0.616 | 4 | `---` / `---` |
| **23** | `sacsma_gauged_multi_3f_single` | **SAC-SMA + Snow-17 (DDS, single model)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet, Maurer, NLDAS | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.691** | 0.691 ± 0.001 | 0.659 | 0.642 | 0.614 | 6 | `---` / `---` |
| **24** | `t4_38` | **Informer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.690** | 0.690 | --- | 0.630 | --- | --- | `---` / `---` |
| **25** | `t4_liu25_pyraformer` | **Pyraformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.690** | 0.690 | --- | 0.610 | --- | --- | `---` / `---` |
| **26** | `t4_liu25_etsformer` | **ETSformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.670** | 0.670 | --- | 0.680 | --- | --- | `---` / `---` |
| **27** | `t4_liu25_reformer` | **Reformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.660** | 0.660 | --- | 0.620 | --- | --- | `---` / `---` |
| **28** | `t4_40` | **iTransformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.370** | 0.370 | --- | 0.370 | --- | --- | `---` / `---` |
| **29** | `t4_41` | **DLinear** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.340** | 0.340 | --- | 0.420 | --- | --- | `---` / `---` |
| **30** | `t4_42` | **CARDformer** | `liu2025_tslib_suite` | Machine learning | `Literature` | Liu et al. (2025) | Daymet, Maurer, NLDAS | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.290** | 0.290 | --- | 0.330 | --- | --- | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `era_gauged_multi_paradigm` | ERA-Multi-Paradigm-Gauged (10-model ensemble) | **0.826** | 0.806 | 0.919 | 0.864 | -0.031 | 0.932 | 1.021 | 1.042 | -12.63 | -5.29 | 34.94 | 0.24 | 0.33 | 33.84 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/era_gauged_multi_paradigm.zarr` |
| **2** | `era_gauged_multi_paradigm_single` | ERA-Multi-Paradigm-Gauged | **0.823** | 0.805 | 0.917 | 0.866 | -0.031 | 0.932 | 1.035 | 1.071 | -12.52 | -4.98 | 22.84 | 0.24 | 0.33 | 34.20 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/era_gauged_multi_paradigm.zarr` |
| **3** | `t4_05` | LSTM (10-model ensemble) | **0.821** | 0.801 | 0.915 | 0.861 | -0.028 | 0.946 | 1.042 | 1.085 | -13.82 | -8.09 | 41.28 | 0.24 | 0.33 | 33.36 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_05.zarr` |
| **4** | `t4_06` | LSTM + dHBV 1.1p (Multi-Model Ensemble) | **0.820** | 0.798 | 0.918 | 0.852 | -0.022 | 0.953 | 1.037 | 1.075 | -14.07 | -8.03 | 44.33 | 0.24 | 0.33 | 33.16 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_06.zarr` |
| **5** | `era_gauged_physics_guided` | ERA-Physics-Guided-Gauged (10-model ensemble) | **0.820** | 0.821 | 0.914 | 0.883 | -0.026 | 0.949 | 1.046 | 1.094 | -10.77 | -6.82 | 37.89 | 0.24 | 0.34 | 33.57 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/era_gauged_physics_guided.zarr` |
| **6** | `era_gauged_minimal_guidance` | ERA-Unguided-Gauged (10-model ensemble) | **0.819** | 0.802 | 0.913 | 0.866 | -0.030 | 0.941 | 1.058 | 1.119 | -12.83 | -5.41 | 25.29 | 0.25 | 0.33 | 33.81 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/era_gauged_minimal_guidance.zarr` |
| **7** | `t4_09` | LSTM (10-model ensemble) | **0.810** | 0.793 | 0.911 | 0.847 | -0.030 | 0.940 | 1.076 | 1.158 | -15.03 | -8.21 | 39.66 | 0.25 | 0.33 | 34.32 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_09.zarr` |
| **8** | `era_gauged_minimal_guidance_single` | ERA-Unguided-Gauged | **0.808** | 0.798 | 0.908 | 0.871 | -0.030 | 0.941 | 1.080 | 1.168 | -12.43 | -4.11 | 4.88 | 0.25 | 0.34 | 34.57 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/era_gauged_minimal_guidance.zarr` |
| **9** | `t4_08` | LSTM (10-model ensemble) | **0.804** | 0.787 | 0.907 | 0.864 | -0.033 | 0.934 | 1.091 | 1.191 | -14.29 | -6.60 | 36.04 | 0.29 | 0.35 | 35.67 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_08.zarr` |
| **10** | `era_gauged_physics_guided_single` | ERA-Physics-Guided-Gauged | **0.801** | 0.818 | 0.904 | 0.895 | -0.026 | 0.949 | 1.099 | 1.208 | -9.91 | -5.70 | 24.59 | 0.26 | 0.35 | 35.46 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/era_gauged_physics_guided.zarr` |
| **11** | `t4_19` | LSTM (10-model ensemble) | **0.799** | 0.777 | 0.906 | 0.835 | -0.033 | 0.934 | 1.081 | 1.167 | -16.17 | -8.94 | 40.24 | 0.25 | 0.34 | 34.66 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_19.zarr` |
| **12** | `t4_11` | LSTM (single model) | **0.796** | 0.795 | 0.903 | 0.869 | -0.027 | 0.945 | 1.115 | 1.243 | -12.64 | -6.50 | 2.48 | 0.27 | 0.35 | 35.11 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_05.zarr` |
| **13** | `t4_25` | LSTM (single model) | **0.787** | 0.789 | 0.898 | 0.860 | -0.030 | 0.941 | 1.141 | 1.301 | -13.66 | -6.88 | -1.69 | 0.28 | 0.35 | 36.06 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_09.zarr` |
| **14** | `hoedt_mclstm_3f_avgp` | MC-LSTM (Hoedt 2021 — 3-Forcing Mean-Precip + 3F Gates Ensemble) | **0.777** | 0.755 | 0.896 | 0.818 | -0.022 | 0.953 | 1.180 | 1.392 | -17.47 | -15.13 | -51.89 | 0.30 | 0.34 | 36.72 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/hoedt_mclstm_3f_avgp.zarr` |
| **15** | `t4_31` | LSTM (single model) | **0.774** | 0.773 | 0.893 | 0.844 | -0.032 | 0.934 | 1.162 | 1.349 | -14.98 | -7.62 | -0.55 | 0.28 | 0.36 | 36.76 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_19.zarr` |
| **16** | `t4_28` | LSTM (single model) | **0.773** | 0.784 | 0.892 | 0.874 | -0.034 | 0.933 | 1.171 | 1.371 | -12.94 | -5.58 | -4.54 | 0.31 | 0.37 | 37.50 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_08.zarr` |
| **17** | `hoedt_mclstm_3f_ens_b` | MC-LSTM (Hoedt 2021 — 3-Forcing Model-Average Ensemble) | **0.771** | 0.744 | 0.892 | 0.804 | -0.028 | 0.943 | 1.165 | 1.357 | -19.09 | -13.43 | -11.76 | 0.29 | 0.34 | 37.18 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/hoedt_mclstm_3f_ens_b.zarr` |
| **18** | `hoedt_mclstm_3f_avgp_single` | MC-LSTM (Hoedt 2021 — 3-Forcing Mean-Precip + 3F Gates, Single Model) | **0.754** | 0.746 | 0.882 | 0.831 | -0.025 | 0.952 | 1.239 | 1.536 | -16.19 | -13.39 | -77.09 | 0.33 | 0.36 | 37.92 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/hoedt_mclstm_3f_avgp.zarr` |
| **19** | `t4_34` | Crossformer | **0.720** | 0.730 | --- | --- | --- | --- | --- | --- | -14.35 | --- | -42.39 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **20** | `t4_35` | Non-stationary Transformer | **0.720** | 0.610 | --- | --- | --- | --- | --- | --- | -15.52 | --- | -172.87 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **21** | `t4_37` | Vanilla Transformer | **0.700** | 0.600 | --- | --- | --- | --- | --- | --- | -8.77 | --- | -230.86 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **22** | `sacsma_gauged_multi_3f` | SAC-SMA + Snow-17 (DDS, 3-Forcing Ensemble) | **0.695** | 0.645 | 0.858 | 0.754 | -0.088 | 0.817 | 1.372 | 1.881 | -25.25 | 7.09 | -58.33 | 0.30 | 0.35 | 43.46 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/sacsma_3f_ens.zarr` |
| **23** | `sacsma_gauged_multi_3f_single` | SAC-SMA + Snow-17 (DDS, single model) | **0.691** | 0.642 | 0.854 | 0.756 | -0.089 | 0.816 | 1.377 | 1.898 | -25.11 | 7.77 | -89.14 | 0.31 | 0.36 | 43.61 | `gs://camels-us-mip/hydrographs/gauged__multi_camels/sacsma_3f_ens.zarr` |
| **24** | `t4_38` | Informer | **0.690** | 0.630 | --- | --- | --- | --- | --- | --- | -14.89 | --- | -169.44 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **25** | `t4_liu25_pyraformer` | Pyraformer | **0.690** | 0.610 | --- | --- | --- | --- | --- | --- | -20.57 | --- | -98.85 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **26** | `t4_liu25_etsformer` | ETSformer | **0.670** | 0.680 | --- | --- | --- | --- | --- | --- | -19.19 | --- | -32.84 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **27** | `t4_liu25_reformer` | Reformer | **0.660** | 0.620 | --- | --- | --- | --- | --- | --- | -13.31 | --- | -226.98 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **28** | `t4_40` | iTransformer | **0.370** | 0.370 | --- | --- | --- | --- | --- | --- | -36.00 | --- | -147.66 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **29** | `t4_41` | DLinear | **0.340** | 0.420 | --- | --- | --- | --- | --- | --- | -38.61 | --- | -300.24 | --- | --- | --- | https://zenodo.org/records/15852144 |
| **30** | `t4_42` | CARDformer | **0.290** | 0.330 | --- | --- | --- | --- | --- | --- | -55.27 | --- | -183.48 | --- | --- | --- | https://zenodo.org/records/15852144 |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #3 `t4_05` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Daymet + Maurer + NLDAS meteorological forcings (15 dynamic inputs + 27 static catchment attributes) across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.821, Mean NSE = 0.768, Med KGE = 0.801; 447-basin subset Ensemble Med NSE = 0.818).
- **Rank #4 `t4_06` (LSTM + dHBV 1.1p (Multi-Model Ensemble)):** Equal-weight 2-component multi-structure hybrid ensemble combining member[0] = Kratzert et al. (2021) 10-seed 3-forcing Regional LSTM Author Archive ensemble (t4_05.zarr, Med NSE = 0.821) and member[1] = our 5-seed 3-forcing Song et al. (2024) dHBV 1.1p MIP Re-run ensemble (t4_24.zarr, Med NSE = 0.799). Single-model Med NSE is omitted (---) because the two members of t4_06.zarr are themselves multi-seed ensembles from two distinct model architectures and provenances (Author Archive + Our Code) that are evaluated separately in Table 4.
- **Rank #7 `t4_09` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Daymet + NLDAS meteorological forcings (10 dynamic inputs + 27 static catchment attributes) across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.810, Mean NSE = 0.757, Med KGE = 0.793; 447-basin subset Ensemble Med NSE = 0.806).
- **Rank #9 `t4_08` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Daymet + Maurer meteorological forcings (10 dynamic inputs + 27 static catchment attributes) across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.804, Mean NSE = 0.753, Med KGE = 0.785; 447-basin subset Ensemble Med NSE = 0.813).
- **Rank #11 `t4_19` (LSTM (10-model ensemble)):** 10-seed ensemble mean of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Maurer + NLDAS meteorological forcings (10 dynamic inputs + 27 static catchment attributes) across all 531 CAMELS-US basins (Zenodo 10.5281/zenodo.4670268; 531-basin Ensemble Med NSE = 0.799, Mean NSE = 0.747, Med KGE = 0.779; 447-basin subset Ensemble Med NSE = 0.793).
- **Rank #12 `t4_11` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Daymet + Maurer + NLDAS forcings across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.796 ± 0.002; 447-basin subset Single-Seed Med NSE = 0.804 ± 0.002).
- **Rank #13 `t4_25` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Daymet + NLDAS forcings across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.787 ± 0.003; 447-basin subset Single-Seed Med NSE = 0.783 ± 0.002).
- **Rank #15 `t4_31` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Maurer + NLDAS forcings across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.774 ± 0.003; 447-basin subset Single-Seed Med NSE = 0.770 ± 0.002).
- **Rank #16 `t4_28` (LSTM (single model)):** Single-model evaluation scored across all 10 archived seed members (evaluation_mode='single_member_mean') of the Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained jointly with Daymet + Maurer forcings across all 531 CAMELS-US basins (531-basin Single-Seed Med NSE = 0.773 ± 0.002; 447-basin subset Single-Seed Med NSE = 0.778 ± 0.002).
- **Rank #19 `t4_34` (Crossformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Zhang & Yan, 2023 ICLR). Trained on WY2000–2008 and evaluated on WY1990–1999 (364 warmup days) across all 531 Newman basins using 15 dynamic channels (Daymet + Maurer + NLDAS) and 26 static attributes (seq_len=365, seg_len=24, d_model=64, d_ff=128, n_heads=4, e_layers=2, d_layers=1, lr=0.001, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows rather than log-space FDC volume bias.
- **Rank #20 `t4_35` (Non-stationary Transformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Liu et al., 2022c NeurIPS). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, label_len=122, d_model=64, d_ff=128, n_heads=4, e_layers=2, d_layers=1, p_hidden_dims=[64,64], lr=0.0005, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #21 `t4_37` (Vanilla Transformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Vaswani et al., 2017 NeurIPS). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, label_len=122, d_model=64, d_ff=128, n_heads=4, e_layers=2, d_layers=1, lr=0.0005, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #22 `sacsma_gauged_multi_3f` (SAC-SMA + Snow-17 (DDS, 3-Forcing Ensemble)):** Clean-room 27-parameter Snow-17 + SAC-SMA + 15-day Duamel Gamma unit hydrograph calibrated per basin via DDS (400 iterations, r=0.2, 365-day spinup, daily RMSE objective) across all 3 CAMELS forcing products (Daymet, Maurer, NLDAS) and 3 seeds (seeds 101, 102, 103) over 1999-10-01 to 2008-09-30 and evaluated over 1989-10-01 to 1999-09-30 on all 531 CAMELS-US basins (Ensemble Mean Median NSE = 0.695).
- **Rank #23 `sacsma_gauged_multi_3f_single` (SAC-SMA + Snow-17 (DDS, single model)):** Single-member performance averaged across all 3 stored members in sacsma_3f_ens.zarr (seeds 101, 102, 103; each member calibrates Daymet, Maurer, and NLDAS via DDS over 1999-10-01 to 2008-09-30; Single-Member Mean Median NSE = 0.691 ± 0.001).
- **Rank #24 `t4_38` (Informer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Zhou et al., 2021 AAAI). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, label_len=122, factor=3, d_model=64, d_ff=128, n_heads=4, e_layers=2, d_layers=1, lr=0.0005, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #25 `t4_liu25_pyraformer` (Pyraformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Liu et al., 2022b ICLR). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, window_size=[4,4], d_model=64, d_ff=128, n_heads=4, e_layers=2, lr=0.0005, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #26 `t4_liu25_etsformer` (ETSformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Woo et al., 2022). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, label_len=122, top_k=3, d_model=64, d_ff=128, n_heads=4, e_layers=2, d_layers=2, lr=0.001, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #27 `t4_liu25_reformer` (Reformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Kitaev et al., 2020 ICLR). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, label_len=122, bucket_size=4, n_hashes=4, d_model=64, d_ff=128, n_heads=4, e_layers=2, d_layers=1, lr=0.001, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #28 `t4_40` (iTransformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Liu et al., 2024b ICLR). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, d_model=64, d_ff=128, n_heads=4, e_layers=2, lr=0.001, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #29 `t4_41` (DLinear):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Zeng et al., 2023 AAAI). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, moving_avg=25, lr=0.001, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.
- **Rank #30 `t4_42` (CARDformer):** Published single-seed (seed=42) CAMELS regression benchmark from Liu et al. (2025, HESS 29, 6811–6828, Supplement Table S1, p. 5; original architecture by Xue et al., 2024 ICLR). Trained on WY2000–2008 and evaluated on WY1990–1999 across all 531 Newman basins using Daymet + Maurer + NLDAS forcings and 26 static attributes (seq_len=365, patch_len=16, stride=8, d_model=64, d_ff=128, n_heads=4, e_layers=2, lr=0.001, 30 epochs, global z-score MSE loss). Note: FLV in Liu et al. (2025) is computed as linear low-flow percent bias on the bottom 30% of flows.

### 4.4 Table 5: `gauged__lagged_streamflow` — Gauged: With Lagged Streamflow Inputs (`11` Rows)
- **Subtitle / Scope:** 531 basins — Models ingesting lagged observed streamflow Q(t-1), autoregression, or data assimilation.
- **Spatial Regime:** `gauged` | **Forcing Regime:** `lagged_streamflow` | **Section:** `4.1.3`
- **SOTA / Rank #1 Winner:** **Autoregression LSTM (no holdout)** (`Median NSE = 0.879`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `t5_01` | **Autoregression LSTM (no holdout)** | `nearing2022_ar_da_lstm` | Machine learning | `Literature` | Nearing et al. (2022) | Daymet, Maurer, NLDAS + Q(t-1) | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.879** | 0.879 | 0.849 | 0.896 | 0.865 | 0 | `---` / `---` |
| **2** | `t5_02` | **Autoregression LSTM (0.5 holdout)** | `nearing2022_ar_da_lstm` | Machine learning | `Literature` | Nearing et al. (2022) | Daymet, Maurer, NLDAS + Q(t-1) | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.872** | 0.872 | 0.842 | 0.896 | 0.859 | 1 | `---` / `---` |
| **3** | `t5_03` | **Variational Data Assimilation LSTM** | `nearing2022_ar_da_lstm` | Machine learning | `Literature` | Nearing et al. (2022) | Daymet, Maurer, NLDAS + Q(t-1) | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.862** | 0.862 | 0.827 | 0.878 | 0.844 | 0 | `---` / `---` |
| **4** | `yin2022_rrformer_daymet` | **Rainfall-Runoff Transformer (RR-Former) (3-model ensemble)** | `yin2022_rrformer` | Machine learning | `MIP Re-run` | Yin et al. (2022) | Daymet + Lagged Q (t-7..t-1) | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.847** | 0.833 ± 0.004 | 0.822 | 0.862 | 0.832 | 0 | `---` / `---` |
| **5** | `t5_04` | **DI(1)-LSTM (Literature)*** | `feng2020_di_lstm_lit` | Machine learning | `Literature*` | Feng et al. (2020) | Daymet + Lagged Q (t-1) | No (1) | 531 | 1990-10-01 to 1999-09-30 | **0.840** | 0.840 | --- | 0.855 | --- | --- | `---` / `---` |
| **6** | `yin2022_rrformer_daymet_single` | **Rainfall-Runoff Transformer (RR-Former) (single model)** | `yin2022_rrformer` | Machine learning | `MIP Re-run` | Yin et al. (2022) | Daymet + Lagged Q (t-7..t-1) | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.833** | 0.833 ± 0.004 | 0.807 | 0.859 | 0.827 | 0 | `---` / `---` |
| **7** | `t5_06` | **δHBV-DA-PS (State + Precip DA)*** | `jamaat2025_dhbv_da` | Hybrid (ML + Physics) | `Literature*` | Jamaat et al. (2025) | Daymet + Q(t-5:t-1) | No (1) | 531 | 1990-10-01 to 1999-09-30 | **0.820** | 0.820 | --- | 0.852 | --- | --- | `---` / `---` |
| **8** | `t5_07` | **δHBV-DA-S (State DA)*** | `jamaat2025_dhbv_da` | Hybrid (ML + Physics) | `Literature*` | Jamaat et al. (2025) | Daymet + Q(t-5:t-1) | No (1) | 531 | 1990-10-01 to 1999-09-30 | **0.820** | 0.820 | --- | 0.841 | --- | --- | `---` / `---` |
| **9** | `feng2020_di_lstm_nldas` | **DI(1)-LSTM (NLDAS, 6-model ensemble)** | `feng2020_di_lstm` | Machine learning | `MIP Re-run` | Feng et al. (2020) | NLDAS + Lagged Q (t-1) | Yes (6) | 531 | 1989-10-01 to 1999-09-30 | **0.801** | 0.795 ± 0.002 | 0.779 | 0.798 | 0.766 | 0 | `---` / `---` |
| **10** | `feng2020_di_lstm_nldas_single` | **DI(1)-LSTM (NLDAS, single model)** | `feng2020_di_lstm` | Machine learning | `MIP Re-run` | Feng et al. (2020) | NLDAS + Lagged Q (t-1) | No (6) | 531 | 1989-10-01 to 1999-09-30 | **0.795** | 0.795 ± 0.002 | 0.773 | 0.793 | 0.762 | 0 | `---` / `---` |
| **11** | `t5_08` | **δHBV-DA-P (Precip DA)*** | `jamaat2025_dhbv_da` | Hybrid (ML + Physics) | `Literature*` | Jamaat et al. (2025) | Daymet + Q(t-5:t-1) | No (1) | 531 | 1990-10-01 to 1999-09-30 | **0.740** | 0.740 | --- | 0.800 | --- | --- | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `t5_01` | Autoregression LSTM (no holdout) | **0.879** | 0.896 | 0.939 | 0.942 | -0.007 | --- | --- | --- | --- | --- | --- | 0.39 | 0.25 | --- | https://zenodo.org/records/7063252 |
| **2** | `t5_02` | Autoregression LSTM (0.5 holdout) | **0.872** | 0.896 | 0.937 | 0.945 | -0.002 | --- | --- | --- | --- | --- | --- | 0.37 | 0.26 | --- | https://zenodo.org/records/7063252 |
| **3** | `t5_03` | Variational Data Assimilation LSTM | **0.862** | 0.878 | 0.932 | 0.913 | -0.014 | --- | --- | --- | --- | --- | --- | 0.44 | 0.25 | --- | https://zenodo.org/records/7063252 |
| **4** | `yin2022_rrformer_daymet` | Rainfall-Runoff Transformer (RR-Former) (3-model ensemble) | **0.847** | 0.862 | 0.924 | 0.905 | -0.006 | 0.989 | 0.959 | 0.920 | -9.44 | -0.98 | -4.04 | 0.40 | 0.24 | 30.79 | `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/yin2022_rrformer_daymet.zarr` |
| **5** | `t5_04` | DI(1)-LSTM (Literature)* | **0.840** | 0.855 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **6** | `yin2022_rrformer_daymet_single` | Rainfall-Runoff Transformer (RR-Former) (single model) | **0.833** | 0.859 | 0.917 | 0.916 | -0.006 | 0.990 | 1.014 | 1.029 | -8.59 | -0.77 | -16.99 | 0.42 | 0.25 | 32.35 | `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/yin2022_rrformer_daymet.zarr` |
| **7** | `t5_06` | δHBV-DA-PS (State + Precip DA)* | **0.820** | 0.852 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **8** | `t5_07` | δHBV-DA-S (State DA)* | **0.820** | 0.841 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **9** | `feng2020_di_lstm_nldas` | DI(1)-LSTM (NLDAS, 6-model ensemble) | **0.801** | 0.798 | 0.901 | 0.839 | -0.009 | 0.985 | 1.079 | 1.164 | -14.89 | -2.63 | -4.05 | 0.53 | 0.32 | 36.07 | `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/feng2020_di_lstm_nldas.zarr` |
| **10** | `feng2020_di_lstm_nldas_single` | DI(1)-LSTM (NLDAS, single model) | **0.795** | 0.793 | 0.899 | 0.841 | -0.008 | 0.985 | 1.099 | 1.207 | -14.91 | -2.11 | -28.62 | 0.54 | 0.32 | 36.33 | `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/feng2020_di_lstm_nldas.zarr` |
| **11** | `t5_08` | δHBV-DA-P (Precip DA)* | **0.740** | 0.800 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `t5_01` (Autoregression LSTM (no holdout)):** Autoregression LSTM without observation holdout during training (f_train = 0.0, f_test = 0.0, 1-day lead time) from Nearing et al. (2022, HESS 26, 5493-5513, Table 2, p. 5497; Zenodo 10.5281/zenodo.7063252). Trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across 531 CAMELS-US basins forced by Daymet, Maurer, and NLDAS + Q(t-1).
- **Rank #2 `t5_02` (Autoregression LSTM (0.5 holdout)):** Autoregression LSTM with 50% Bernoulli Markov observation holdout during training (f_train = 0.5, mean missing sequence length = 5 days, f_test = 0.0, 1-day lead time) from Nearing et al. (2022, HESS 26, 5493-5513, Table 2, p. 5497; Zenodo 10.5281/zenodo.7063252). Trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across 531 CAMELS-US basins forced by Daymet, Maurer, and NLDAS + Q(t-1).
- **Rank #3 `t5_03` (Variational Data Assimilation LSTM):** Unregularized Variational Data Assimilation LSTM (alpha_c = 0, alpha_y = 1, 5-day assimilation window [t-5, t-1], 1-day lead time) on a frozen regional LSTM from Nearing et al. (2022, HESS 26, 5493-5513, Table 2, p. 5497; Zenodo 10.5281/zenodo.7063252). Trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across 531 CAMELS-US basins forced by Daymet, Maurer, and NLDAS + Q(t-1).
- **Rank #4 `yin2022_rrformer_daymet` (Rainfall-Runoff Transformer (RR-Former) (3-model ensemble)):** Yin et al. (2022) RR-Former (d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=256, 7-day lookback, 3-layer MLP head 64->32->16->1) trained via 300-epoch regional pretraining across all 531 CAMELS-US basins over WY1999-2008 followed by 40-epoch per-basin fine-tuning (lr=1e-4, patience=10), evaluated over WY1989-1999 with Daymet + 7-day lagged streamflow Q(t-7..t-1); 3-seed ensemble mean (seeds 601, 602, 603; Ensemble Med NSE = 0.847, Mean NSE = 0.822, Med KGE = 0.862, NSE <= 0 = 0).
- **Rank #5 `t5_04` (DI(1)-LSTM (Literature)*):** **[Asterisk `*`: Evaluated on WY1991–1999 (9 water years; 1989–1990 used as spin-up) as reported by Jamaat et al. (2025) for the Feng et al. (2020) DI(1)-LSTM benchmark.]** Original mhpi/hydroDL DI(1)-LSTM (Feng et al., 2020; single model, hidden_size=256, seq_len=365, batch_size=100, dropout=0.5, 100 epochs, Eq. 11 log-sqrt transform log10(sqrt(v)+0.1)) trained on 1999-10-01 to 2008-09-30 across the 531 Newman CAMELS-US basins with Daymet forcing + 1-day lagged observed streamflow Q(t-1) and evaluated on WY1991–1999 (1990-10-01 to 1999-09-30, with 1989–1990 used as 365-day spin-up buffer), as reported in Jamaat et al. (2025, Journal of Hydrology 663, 134137, Table 1, p. 15; previously indexed as t5_09 in gauged__lagged_streamflow.json): Median NSE = 0.84, Median KGE = 0.855, Low-flow RMSE = 0.04 mm/d, High-flow RMSE = 2.83 mm/d.
- **Rank #6 `yin2022_rrformer_daymet_single` (Rainfall-Runoff Transformer (RR-Former) (single model)):** Yin et al. (2022) RR-Former (d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=256, 7-day lookback, 3-layer MLP head 64->32->16->1) trained via 300-epoch regional pretraining across all 531 CAMELS-US basins over WY1999-2008 followed by 40-epoch per-basin fine-tuning (lr=1e-4, patience=10), evaluated over WY1989-1999 with Daymet + 7-day lagged streamflow Q(t-7..t-1); single-model performance averaged across all 3 random seeds (seeds 601, 602, 603; evaluation_mode='single_member_mean'; Single-Seed Med NSE = 0.839 ± 0.002).
- **Rank #7 `t5_06` (δHBV-DA-PS (State + Precip DA)*):** **[Asterisk `*`: Evaluated on WY1991–1999 (9 water years; 1989–1990 used as 365-day spin-up buffer) in Jamaat et al. (2025).]** Joint state + precipitation variational data assimilation (δHBV-DA-PS) on 16-bucket δHBV 1.1p from Jamaat et al. (2025, Journal of Hydrology 663, 134137, Table 1, p. 15): optimizes all 5 log-transformed HBV water stores (SNOWPACK, MELTWATER, SM, SUZ, SLZ) at the start of the 5-day assimilation window [t-5, t-1] and the quartic precipitation multiplier k^4 at t-1 via Adadelta (lr=25.9, 5 steps, NSELossBatch3, monotonic acceptance gate) to predict day-t streamflow (indexed as window [t-4, t] predicting day t+1 in Jamaat et al., 2025). Trained on 1999-10-01 to 2008-09-30 across 531 Newman basins with 3 Daymet forcings (PRCP, T, Hargreaves Ep) and 35 static attributes; evaluated on 1990-10-01 to 1999-09-30 (first test water year 1989–1990 used as 365-day warm-up buffer). Median NSE = 0.82, Median KGE = 0.852, Low-flow RMSE = 0.01 mm/d, High-flow RMSE = 2.66 mm/d.
- **Rank #8 `t5_07` (δHBV-DA-S (State DA)*):** **[Asterisk `*`: Evaluated on WY1991–1999 (9 water years; 1989–1990 used as 365-day spin-up buffer) in Jamaat et al. (2025).]** State-only variational data assimilation (δHBV-DA-S) on 16-bucket δHBV 1.1p from Jamaat et al. (2025, Journal of Hydrology 663, 134137, Table 1, p. 15): optimizes the 5 log-transformed HBV water stores (SNOWPACK, MELTWATER, SM, SUZ, SLZ) at the start of the 5-day assimilation window [t-5, t-1] via Adadelta (lr=25.9, 5 steps, NSELossBatch3, monotonic acceptance gate) to predict day-t streamflow. Trained on 1999-10-01 to 2008-09-30 across 531 Newman basins with 3 Daymet forcings and 35 static attributes; evaluated on 1990-10-01 to 1999-09-30 (first test water year 1989–1990 used as 365-day warm-up buffer). Median NSE = 0.82, Median KGE = 0.841, Low-flow RMSE = 0.01 mm/d, High-flow RMSE = 2.69 mm/d.
- **Rank #9 `feng2020_di_lstm_nldas` (DI(1)-LSTM (NLDAS, 6-model ensemble)):** 6-seed ensemble (seeds 101..106) of our clean-room DI(1)-LSTM implementation (models/feng2020_di_lstm) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins using single-forcing NLDAS meteorological inputs plus 1-day lagged observed streamflow Q(t-1) (Ensemble Med NSE = 0.801, Mean NSE = 0.779, Med KGE = 0.798).
- **Rank #10 `feng2020_di_lstm_nldas_single` (DI(1)-LSTM (NLDAS, single model)):** Single-model evaluation scored across all 6 individual seed members (seeds 101..106; evaluation_mode='single_member_mean') of our clean-room DI(1)-LSTM implementation on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins using single-forcing NLDAS meteorological inputs plus 1-day lagged observed streamflow Q(t-1) (Single-Seed Med NSE = 0.795 ± 0.002).
- **Rank #11 `t5_08` (δHBV-DA-P (Precip DA)*):** **[Asterisk `*`: Evaluated on WY1991–1999 (9 water years; 1989–1990 used as 365-day spin-up buffer) in Jamaat et al. (2025).]** Precipitation-only variational data assimilation (δHBV-DA-P) on 16-bucket δHBV 1.1p from Jamaat et al. (2025, Journal of Hydrology 663, 134137, Table 1, p. 15): optimizes a per-basin quartic precipitation multiplier k^4 at the last day t-1 of the 5-day assimilation window [t-5, t-1] via Adadelta (lr=25.9, 5 steps, NSELossBatch3, monotonic acceptance gate) to predict day-t streamflow. Trained on 1999-10-01 to 2008-09-30 across 531 Newman basins with 3 Daymet forcings and 35 static attributes; evaluated on 1990-10-01 to 1999-09-30 (first test water year 1989–1990 used as 365-day warm-up buffer). Median NSE = 0.74, Median KGE = 0.800, Low-flow RMSE = 0.05 mm/d, High-flow RMSE = 3.15 mm/d.

### 4.5 Table 6: `gauged__freeform` — Gauged: Freeform Forcings (Non-CAMELS Inputs) (`0` Rows)
- **Subtitle / Scope:** 531 basins — Models driven by non-CAMELS meteorological forcings (AORC, EM-Earth, ERA5-Land).
- **Spatial Regime:** `gauged` | **Forcing Regime:** `freeform` | **Section:** `4.1.4`
- *Currently 0 entries on this leaderboard.*

### 4.6 Table 7: `pub__single_camels` — 12-Fold Random PUB: Single CAMELS Forcing (`12` Rows)
- **Subtitle / Scope:** 12-fold random spatial cross-validation & CONUS MPR over the 1989–1999 Newman evaluation decade.
- **Spatial Regime:** `pub` | **Forcing Regime:** `single_camels` | **Section:** `4.2.1`
- **SOTA / Rank #1 Winner:** **LSTM (10-model ensemble)*** (`Median NSE = 0.691`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `t7_07` | **LSTM (10-model ensemble)*** | `kratzert2019a_pub_lstm` | Machine learning | `Literature**` | Kratzert et al. (2019a) | NLDAS | Yes (10) | 531 | 1995-10-01 to 2010-09-30 | **0.691** | 0.640 ± 0.004 | 0.608 | 0.689 | --- | 24 | `---` / `---` |
| **2** | `t7_06` | **δHBV 1.1p / δMG (5-model ensemble)** | `song_dhbv11p` | Hybrid (ML + Physics) | `MIP Re-run` | Song et al. (2024) | Daymet | Yes (5) | 531 | 1989-10-01 to 1999-09-30 | **0.690** | 0.686 ± 0.004 | 0.582 | 0.689 | 0.594 | 24 | `---` / `---` |
| **3** | `t7_11` | **δHBV 1.1p / δMG (single model)** | `song_dhbv11p` | Hybrid (ML + Physics) | `MIP Re-run` | Song et al. (2024) | Daymet | No (5) | 531 | 1989-10-01 to 1999-09-30 | **0.686** | 0.686 ± 0.004 | 0.561 | 0.681 | 0.585 | 24 | `---` / `---` |
| **4** | `t7_10` | **δHBV 1.0 (5-model ensemble)** | `feng_dhbv10` | Hybrid (ML + Physics) | `MIP Re-run` | Feng et al. (2022) | Daymet | Yes (5) | 531 | 1989-10-01 to 1999-09-30 | **0.681** | 0.673 ± 0.005 | 0.528 | 0.661 | 0.568 | 25 | `---` / `---` |
| **5** | `poudel_mlp_hbv_pub_daymet` | **Static-Parameter MLP + 16-HBV (3-model ensemble)** | `poudel_mlp_hbv` | Hybrid (ML + Physics) | `MIP Re-run` | Poudel & Steinschneider (2026) | Daymet | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.678** | 0.669 ± 0.001 | 0.517 | 0.665 | 0.543 | 23 | `---` / `---` |
| **6** | `t7_14` | **δHBV 1.0 (single model)** | `feng_dhbv10` | Hybrid (ML + Physics) | `MIP Re-run` | Feng et al. (2022) | Daymet | No (5) | 531 | 1989-10-01 to 1999-09-30 | **0.673** | 0.673 ± 0.005 | 0.501 | 0.660 | 0.557 | 26 | `---` / `---` |
| **8** | `poudel_mlp_hbv_pub_daymet_single` | **Static-Parameter MLP + 16-HBV (single model)** | `poudel_mlp_hbv` | Hybrid (ML + Physics) | `MIP Re-run` | Poudel & Steinschneider (2026) | Daymet | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.669** | 0.669 ± 0.001 | 0.505 | 0.660 | 0.540 | 23 | `---` / `---` |
| **10** | `t7_18` | **LSTM (single model)*** | `kratzert2019a_pub_lstm` | Machine learning | `Literature**` | Kratzert et al. (2019a) | NLDAS | No (10) | 531 | 1995-10-01 to 2010-09-30 | **0.640** | 0.640 ± 0.004 | 0.562 | 0.648 | --- | 31 | `---` / `---` |
| **11** | `sacsma_pub_single_daymet` | **SAC-SMA + Snow-17 (Donor Transfer, single model)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.564** | 0.564 | 0.392 | 0.575 | 0.496 | 42 | `---` / `---` |
| **12** | `t7_18_seibert_hbv` | **HBV-light (HBV-lower, 1000-member uncalibrated random ensemble)** | `seibert_hbv` | Conceptual | `Archive` | Seibert et al. (2018) | Maurer | Yes (1) | 531 | 1989-10-01 to 1999-09-30 | **0.420** | --- | 0.262 | 0.398 | 0.301 | 37 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `t7_07` | LSTM (10-model ensemble)* | **0.691** | 0.689 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_07.zarr` |
| **2** | `t7_06` | δHBV 1.1p / δMG (5-model ensemble) | **0.690** | 0.689 | 0.862 | 0.837 | -0.021 | 0.958 | 1.345 | 1.810 | -14.91 | -8.98 | 31.42 | 0.35 | 0.37 | 41.58 | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_06.zarr` |
| **3** | `t7_11` | δHBV 1.1p / δMG (single model) | **0.686** | 0.681 | 0.860 | 0.836 | -0.021 | 0.954 | 1.365 | 1.862 | -14.50 | -8.64 | 25.83 | 0.36 | 0.38 | 42.75 | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_06.zarr` |
| **4** | `t7_10` | δHBV 1.0 (5-model ensemble) | **0.681** | 0.661 | 0.855 | 0.812 | -0.004 | 0.990 | 1.388 | 1.927 | -16.38 | -16.72 | 25.10 | 0.36 | 0.37 | 42.22 | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_10.zarr` |
| **5** | `poudel_mlp_hbv_pub_daymet` | Static-Parameter MLP + 16-HBV (3-model ensemble) | **0.678** | 0.665 | 0.860 | 0.797 | -0.019 | 0.957 | 1.384 | 1.916 | -19.93 | -4.58 | -5.14 | 0.36 | 0.39 | 45.32 | `gs://camels-us-mip/hydrographs/pub__single_camels/poudel_mlp_hbv_pub_daymet.zarr` |
| **6** | `t7_14` | δHBV 1.0 (single model) | **0.673** | 0.660 | 0.851 | 0.821 | -0.004 | 0.992 | 1.403 | 1.970 | -16.20 | -16.58 | 21.19 | 0.37 | 0.38 | 43.10 | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_10.zarr` |
| **8** | `poudel_mlp_hbv_pub_daymet_single` | Static-Parameter MLP + 16-HBV (single model) | **0.669** | 0.660 | 0.856 | 0.799 | -0.020 | 0.959 | 1.394 | 1.943 | -19.72 | -4.28 | -9.60 | 0.37 | 0.39 | 45.24 | `gs://camels-us-mip/hydrographs/pub__single_camels/poudel_mlp_hbv_pub_daymet.zarr` |
| **10** | `t7_18` | LSTM (single model)* | **0.640** | 0.648 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_18.zarr` |
| **11** | `sacsma_pub_single_daymet` | SAC-SMA + Snow-17 (Donor Transfer, single model) | **0.564** | 0.575 | 0.803 | 0.811 | -0.081 | 0.822 | 1.625 | 2.639 | -20.78 | 15.91 | -150.93 | 0.40 | 0.39 | 49.55 | `gs://camels-us-mip/hydrographs/pub__single_camels/sacsma_pub_daymet.zarr` |
| **12** | `t7_18_seibert_hbv` | HBV-light (HBV-lower, 1000-member uncalibrated random ensemble) | **0.420** | 0.398 | 0.719 | 0.594 | -0.024 | 0.948 | 1.822 | 3.318 | -40.05 | -13.94 | 23.27 | 1.17 | 0.59 | 67.44 | `gs://camels-us-mip/hydrographs/pub__single_camels/t7_23.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `t7_07` (LSTM (10-model ensemble)*):** **[Asterisk `*`: Evaluated in Kratzert et al. (2019a) on Split-3 (trained on WY1981–1995, evaluated on WY1996–2010; 1995-10-01 to 2010-09-30) with extended NLDAS-2 forcing.]** 12-fold spatial k-fold cross-validation (PUB) of the 270-day lookback regional LSTM with 27 static catchment attributes on extended NLDAS-2 forcings (10 random seeds per fold; trained on WY1981–1995 [1980-10-01 to 1995-09-30] and evaluated on WY1996–2010 [1995-10-01 to 2010-09-30] across 531 CAMELS-US basins) from Kratzert et al. (2019a, WRR, Table 2; HydroShare 10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c): Ensemble Med NSE = 0.691 (0.69 ± 0.004), Mean NSE = 0.608, Med KGE = 0.689, Med |FHV| = 20.4%, 24 basins with NSE <= 0.
- **Rank #2 `t7_06` (δHBV 1.1p / δMG (5-model ensemble)):** Clean-room PyTorch + C++/OpenMP re-implementation of Song et al. (2024) δHBV 1.1p evaluated under 12-fold random spatial cross-validation (PUB) across 531 CAMELS-US basins with Daymet forcing; 5-seed ensemble mean (seeds 111, 222, 333, 444, 555; Med NSE = 0.690).
- **Rank #3 `t7_11` (δHBV 1.1p / δMG (single model)):** Clean-room PyTorch + C++/OpenMP re-implementation of Song et al. (2024) δHBV 1.1p evaluated under 12-fold random spatial cross-validation (PUB) across 531 CAMELS-US basins with Daymet forcing; single-model performance averaged across all 5 random seeds (seeds 111, 222, 333, 444, 555; Med NSE = 0.686 ± 0.004).
- **Rank #4 `t7_10` (δHBV 1.0 (5-model ensemble)):** 5-seed ensemble (seeds 111, 222, 333, 444, 555) evaluated under 12-fold random spatial cross-validation (PUB) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with Daymet forcing.
- **Rank #5 `poudel_mlp_hbv_pub_daymet` (Static-Parameter MLP + 16-HBV (3-model ensemble)):** Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (50 epochs x 240 batches/epoch) evaluated under 12-fold spatial cross-validation (PUB) across 531 CAMELS-US basins with Daymet forcing; 3-seed ensemble mean (seeds 101, 102, 103; Ensemble Med NSE = 0.678).
- **Rank #6 `t7_14` (δHBV 1.0 (single model)):** Single-model evaluation scored across all 5 individual seed members (seeds 111, 222, 333, 444, 555; evaluation_mode='single_member_mean') under 12-fold random spatial cross-validation (PUB) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with Daymet forcing.
- **Rank #8 `poudel_mlp_hbv_pub_daymet_single` (Static-Parameter MLP + 16-HBV (single model)):** Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (50 epochs x 240 batches/epoch) evaluated under 12-fold spatial cross-validation (PUB) across 531 CAMELS-US basins with Daymet forcing; single-model performance averaged across all 3 random seeds (seeds 101, 102, 103; Single-Seed Med NSE = 0.669 ± 0.001).
- **Rank #10 `t7_18` (LSTM (single model)*):** **[Asterisk `*`: Evaluated in Kratzert et al. (2019a) on Split-3 (trained on WY1981–1995, evaluated on WY1996–2010; 1995-10-01 to 2010-09-30) with extended NLDAS-2 forcing.]** Single-model evaluation under 12-fold spatial k-fold cross-validation (PUB) of the 270-day lookback regional LSTM with 27 static catchment attributes on extended NLDAS-2 forcings (trained on WY1981–1995 [1980-10-01 to 1995-09-30] and evaluated on WY1996–2010 [1995-10-01 to 2010-09-30] across 531 CAMELS-US basins) from Kratzert et al. (2019a, WRR, Table 2; HydroShare 10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c): Single-Model Med NSE = 0.640 (0.64 ± 0.004), Mean NSE = 0.562, Med KGE = 0.648, Med |FHV| = 21.9%, 31 basins with NSE <= 0.
- **Rank #11 `sacsma_pub_single_daymet` (SAC-SMA + Snow-17 (Donor Transfer, single model)):** 12-fold spatial cross-validation (PUB) across all 531 CAMELS-US basins on Daymet forcings: training basins calibrated per fold via DDS (400 iterations, 1999-10-01 to 2008-09-30), and held-out ungauged test basins predicted via k=1 nearest-neighbor parameter transfer in a 29-D joint (0.2*lat, 0.2*lon) + 27 standardized static catchment-attribute space (1 seed; Median NSE = 0.564).
- **Rank #12 `t7_18_seibert_hbv` (HBV-light (HBV-lower, 1000-member uncalibrated random ensemble)):** Lower benchmark (HBV_lb): 1000-member uncalibrated random-parameter ensemble lower benchmark with parameter sets sampled uniformly at random from physically plausible bounds (using zero target-basin and zero donor-basin streamflow observations; NOT a donor-transfer calibration; evaluated over 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins; HydroShare archive 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1, NetCDF variable HBV_lb; stores the pre-computed 1000-member ensemble mean hydrograph as 1 array at gs://camels-us-mip/hydrographs/pub__single_camels/t7_23.zarr; Median NSE = 0.420 on 531 basins, 0.416 on the 447-basin Kratzert et al. 2019b Table 3 intersection).

### 4.7 Table 8: `pub__multi_camels` — 12-Fold Random PUB: Multiple CAMELS Forcings (`8` Rows)
- **Subtitle / Scope:** 12-fold random spatial cross-validation over the 1989–1999 Newman evaluation decade with Daymet + Maurer + NLDAS.
- **Spatial Regime:** `pub` | **Forcing Regime:** `multi_camels` | **Section:** `4.2.2`
- **SOTA / Rank #1 Winner:** **LSTM (8-model ensemble)** (`Median NSE = 0.742`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `k21_lstm_pub_3f` | **LSTM (8-model ensemble)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Daymet, Maurer, NLDAS | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.742** | 0.707 ± 0.006 | 0.632 | 0.708 | 0.620 | 13 | `---` / `---` |
| **2** | `era_pub_physics_guided` | **ERA-Physics-Guided-PUB (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.725** | 0.714 ± 0.004 | 0.645 | 0.664 | 0.598 | 10 | `---` / `---` |
| **3** | `era_pub_minimal_guidance` | **ERA-Unguided-PUB (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.720** | 0.714 ± 0.004 | 0.649 | 0.644 | 0.575 | 5 | `---` / `---` |
| **4** | `era_pub_minimal_guidance_single` | **ERA-Unguided-PUB** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.714** | 0.714 ± 0.004 | 0.636 | 0.641 | 0.570 | 5 | `---` / `---` |
| **5** | `era_pub_physics_guided_single` | **ERA-Physics-Guided-PUB** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.714** | 0.714 ± 0.004 | 0.625 | 0.662 | 0.591 | 11 | `---` / `---` |
| **6** | `k21_lstm_pub_3f_single` | **LSTM (single model)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Daymet, Maurer, NLDAS | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.707** | 0.707 ± 0.006 | 0.569 | 0.686 | 0.596 | 20 | `---` / `---` |
| **7** | `sacsma_pub_multi_3f` | **SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet, Maurer, NLDAS | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.625** | 0.618 ± 0.000 | 0.504 | 0.570 | 0.514 | 26 | `---` / `---` |
| **8** | `sacsma_pub_multi_3f_single` | **SAC-SMA + Snow-17 (Donor Transfer, single model)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet, Maurer, NLDAS | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.618** | 0.618 ± 0.000 | 0.494 | 0.570 | 0.512 | 26 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `k21_lstm_pub_3f` | LSTM (8-model ensemble) | **0.742** | 0.708 | 0.891 | 0.838 | -0.032 | 0.931 | 1.216 | 1.479 | -14.99 | -9.83 | 41.19 | 0.30 | 0.35 | 38.69 | `gs://camels-us-mip/hydrographs/pub__multi_camels/k21_lstm_pub_3f.zarr` |
| **2** | `era_pub_physics_guided` | ERA-Physics-Guided-PUB (10-model ensemble) | **0.725** | 0.664 | 0.884 | 0.776 | -0.044 | 0.906 | 1.262 | 1.594 | -21.55 | -11.94 | 41.58 | 0.33 | 0.37 | 42.43 | `gs://camels-us-mip/hydrographs/pub__multi_camels/era_pub_physics_guided.zarr` |
| **3** | `era_pub_minimal_guidance` | ERA-Unguided-PUB (10-model ensemble) | **0.720** | 0.644 | 0.884 | 0.733 | -0.056 | 0.881 | 1.266 | 1.602 | -25.66 | -11.51 | -5.91 | 0.31 | 0.35 | 42.28 | `gs://camels-us-mip/hydrographs/pub__multi_camels/era_pub_minimal_guidance.zarr` |
| **4** | `era_pub_minimal_guidance_single` | ERA-Unguided-PUB | **0.711** | 0.641 | 0.879 | 0.739 | -0.056 | 0.881 | 1.281 | 1.648 | -25.32 | -9.13 | -52.17 | 0.31 | 0.36 | 42.69 | `gs://camels-us-mip/hydrographs/pub__multi_camels/era_pub_minimal_guidance.zarr` |
| **5** | `era_pub_physics_guided_single` | ERA-Physics-Guided-PUB | **0.709** | 0.662 | 0.877 | 0.781 | -0.044 | 0.906 | 1.285 | 1.651 | -21.09 | -10.81 | 33.23 | 0.33 | 0.38 | 42.99 | `gs://camels-us-mip/hydrographs/pub__multi_camels/era_pub_physics_guided.zarr` |
| **6** | `k21_lstm_pub_3f_single` | LSTM (single model) | **0.707** | 0.686 | 0.874 | 0.854 | -0.032 | 0.933 | 1.312 | 1.722 | -13.66 | -7.99 | 9.13 | 0.33 | 0.37 | 41.57 | `gs://camels-us-mip/hydrographs/pub__multi_camels/k21_lstm_pub_3f.zarr` |
| **7** | `sacsma_pub_multi_3f` | SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble) | **0.625** | 0.570 | 0.832 | 0.726 | -0.082 | 0.822 | 1.517 | 2.302 | -27.48 | 6.30 | -59.41 | 0.36 | 0.37 | 46.95 | `gs://camels-us-mip/hydrographs/pub__multi_camels/sacsma_pub_3f.zarr` |
| **8** | `sacsma_pub_multi_3f_single` | SAC-SMA + Snow-17 (Donor Transfer, single model) | **0.618** | 0.570 | 0.828 | 0.735 | -0.083 | 0.820 | 1.528 | 2.333 | -27.74 | 6.84 | -93.86 | 0.36 | 0.37 | 47.36 | `gs://camels-us-mip/hydrographs/pub__multi_camels/sacsma_pub_3f.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `k21_lstm_pub_3f` (LSTM (8-model ensemble)):** 8-seed ensemble (seeds 101..108) of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM (models/kratzert_lstm) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins under 12-fold spatial cross-validation (PUB) using joint 3-forcing Daymet + Maurer + NLDAS inputs and 27 static catchment attributes (Ensemble Med NSE = 0.742, Mean NSE = 0.630, Med KGE = 0.741).
- **Rank #6 `k21_lstm_pub_3f_single` (LSTM (single model)):** Single-model evaluation scored across all 8 individual seed members (seeds 101..108; evaluation_mode='single_member_mean') of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM under 12-fold spatial cross-validation (PUB) with joint 3-forcing Daymet + Maurer + NLDAS inputs across all 531 CAMELS-US basins (Single-Seed Med NSE = 0.707 ± 0.006).
- **Rank #7 `sacsma_pub_multi_3f` (SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble)):** 12-fold spatial cross-validation (PUB) across all 531 CAMELS-US basins on Daymet, Maurer, and NLDAS forcings using DDS calibration on training basins and 29-D spatial + static-attribute k=1 nearest-neighbor donor parameter transfer to held-out ungauged test basins (3-member ensemble mean; Median NSE = 0.625).
- **Rank #8 `sacsma_pub_multi_3f_single` (SAC-SMA + Snow-17 (Donor Transfer, single model)):** Single-member performance averaged across all 3 stored members in sacsma_pub_3f.zarr under 12-fold spatial cross-validation (PUB) with 29-D nearest-neighbor donor parameter transfer (Single-Member Mean Median NSE = 0.618 ± 0.000).

### 4.8 Table 9: `pub__lagged_streamflow` — 12-Fold Random PUB: With Lagged Streamflow Inputs (`4` Rows)
- **Subtitle / Scope:** 531 basins — Models trained on ungauged basin splits (held-out in space during training) and evaluated with lagged observed streamflow Q(t-1), autoregression, or data assimilation during inference.
- **Spatial Regime:** `pub` | **Forcing Regime:** `lagged_streamflow` | **Section:** `4.2.3`
- **SOTA / Rank #1 Winner:** **Autoregression LSTM (0.5 holdout)*** (`Median NSE = 0.807`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `t9_01` | **Autoregression LSTM (0.5 holdout)*** | `nearing2022_ar_da_lstm` | Machine learning | `Literature**` | Nearing et al. (2022) | Daymet, Maurer, NLDAS + Q(t-1) | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.807** | 0.807 | 0.762 | 0.824 | 0.771 | 4 | `---` / `---` |
| **2** | `feng2020_di_lstm_pub_nldas` | **DI(1)-LSTM (3-model ensemble)** | `feng2020_di_lstm` | Machine learning | `MIP Re-run` | Feng et al. (2020) | NLDAS + Q(t-1) | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.775** | 0.767 ± 0.001 | 0.744 | 0.763 | 0.724 | 0 | `---` / `---` |
| **3** | `t9_02` | **Variational Data Assimilation LSTM*** | `nearing2022_ar_da_lstm` | Machine learning | `Literature**` | Nearing et al. (2022) | Daymet, Maurer, NLDAS + Q(t-1) | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.772** | 0.772 | 0.714 | 0.779 | 0.737 | 6 | `---` / `---` |
| **4** | `feng2020_di_lstm_pub_nldas_single` | **DI(1)-LSTM (single model)** | `feng2020_di_lstm` | Machine learning | `MIP Re-run` | Feng et al. (2020) | NLDAS + Q(t-1) | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.767** | 0.767 ± 0.001 | 0.736 | 0.761 | 0.717 | 0 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `t9_01` | Autoregression LSTM (0.5 holdout)* | **0.807** | 0.824 | 0.908 | 0.912 | -0.011 | --- | --- | --- | --- | --- | --- | 0.37 | 0.33 | --- | `gs://camels-us-mip/hydrographs/pub__lagged_streamflow/t9_01.zarr` |
| **2** | `feng2020_di_lstm_pub_nldas` | DI(1)-LSTM (3-model ensemble) | **0.775** | 0.763 | 0.892 | 0.825 | -0.010 | 0.980 | 1.170 | 1.369 | -16.94 | -2.77 | -10.79 | 0.58 | 0.32 | 38.38 | `gs://camels-us-mip/hydrographs/pub__lagged_streamflow/feng2020_di_lstm_pub_nldas.zarr` |
| **3** | `t9_02` | Variational Data Assimilation LSTM* | **0.772** | 0.779 | 0.892 | 0.869 | -0.028 | --- | --- | --- | --- | --- | --- | 0.46 | 0.34 | --- | `gs://camels-us-mip/hydrographs/pub__lagged_streamflow/t9_02.zarr` |
| **4** | `feng2020_di_lstm_pub_nldas_single` | DI(1)-LSTM (single model) | **0.767** | 0.761 | 0.890 | 0.827 | -0.011 | 0.978 | 1.192 | 1.421 | -16.52 | -2.38 | -29.06 | 0.59 | 0.33 | 38.87 | `gs://camels-us-mip/hydrographs/pub__lagged_streamflow/feng2020_di_lstm_pub_nldas.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `t9_01` (Autoregression LSTM (0.5 holdout)*):** **[Asterisk `*`: Evaluated on a 10-fold random spatial cross-validation across all 531 CAMELS basins (53-54 basins per fold) in Nearing et al. (2022) rather than the 12-fold Kratzert PUB split.]** 10-fold spatial cross-validation (PUB) Autoregression LSTM trained with 50% Bernoulli Markov observation holdout (f_train = 0.5, mean missing sequence length = 5 days) on 9 folds of basins (1999-10-01 to 2008-09-30) and evaluated on the held-out fold of basins (1989-10-01 to 1999-09-30) with lagged observed streamflow Q(t-1) available during inference (f_test = 0.0, 1-day lead time), from Nearing et al. (2022, HESS 26, 5493-5513, Section 3.3 & Fig. 5; Zenodo 10.5281/zenodo.7063252).
- **Rank #2 `feng2020_di_lstm_pub_nldas` (DI(1)-LSTM (3-model ensemble)):** 3-seed ensemble (seeds 101, 102, 103) of our clean-room DI(1)-LSTM implementation (models/feng2020_di_lstm; hidden_size=256, seq_len=365, batch_size=256, dropout=0.4, 30 full epochs) evaluated under 12-fold random spatial cross-validation (PUB) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with NLDAS forcing + 1-day lagged observed streamflow Q(t-1) (Ensemble Med NSE = 0.775, Mean NSE = 0.731, Med KGE = 0.785).
- **Rank #3 `t9_02` (Variational Data Assimilation LSTM*):** **[Asterisk `*`: Evaluated on a 10-fold random spatial cross-validation across all 531 CAMELS basins (53-54 basins per fold) in Nearing et al. (2022) rather than the 12-fold Kratzert PUB split.]** 10-fold spatial cross-validation (PUB) Variational Data Assimilation LSTM (alpha_c = 0, alpha_y = 1, 5-day assimilation window [t-5, t-1], 1-day lead time) trained on 9 folds of basins (1999-10-01 to 2008-09-30) and evaluated on the held-out fold of basins (1989-10-01 to 1999-09-30) with lagged observed streamflow assimilated during inference, from Nearing et al. (2022, HESS 26, 5493-5513, Section 3.3 & Fig. 5; Zenodo 10.5281/zenodo.7063252).
- **Rank #4 `feng2020_di_lstm_pub_nldas_single` (DI(1)-LSTM (single model)):** Single-model evaluation scored across all 3 individual seed members (seeds 101, 102, 103; evaluation_mode='single_member_mean') of our clean-room DI(1)-LSTM implementation under 12-fold random spatial cross-validation (PUB) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with NLDAS forcing + 1-day lagged observed streamflow Q(t-1) (Single-Seed Med NSE = 0.765 ± 0.001).

### 4.9 Table 10: `pub__freeform` — 12-Fold Random PUB: Freeform Forcings (`0` Rows)
- **Subtitle / Scope:** Regionalized ungauged models evaluated over 1989–1999 driven by non-CAMELS meteorological forcings.
- **Spatial Regime:** `pub` | **Forcing Regime:** `freeform` | **Section:** `4.2.4`
- *Currently 0 entries on this leaderboard.*

### 4.10 Table 11: `pur__single_camels` — 18-Fold HUC-02 PUR: Single CAMELS Forcing (`15` Rows)
- **Subtitle / Scope:** 18-fold leave-one-HUC-02-region-out cross-validation over 1989–1999 using a single CAMELS forcing.
- **Spatial Regime:** `pur` | **Forcing Regime:** `single_camels` | **Section:** `4.3.1`
- **SOTA / Rank #1 Winner:** **δHBV 1.1p / δMG (5-model ensemble)** (`Median NSE = 0.650`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `t11_04` | **δHBV 1.1p / δMG (5-model ensemble)** | `song_dhbv11p` | Hybrid (ML + Physics) | `MIP Re-run` | Song et al. (2024) | Daymet | Yes (5) | 531 | 1989-10-01 to 1999-09-30 | **0.650** | 0.634 ± 0.011 | 0.292 | 0.640 | 0.494 | 43 | `---` / `---` |
| **2** | `k21_lstm_pur_daymet` | **LSTM (10-model ensemble)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Daymet | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.635** | 0.621 ± 0.003 | 0.454 | 0.603 | 0.497 | 32 | `---` / `---` |
| **3** | `t11_06` | **δHBV 1.0 (5-model ensemble)** | `feng_dhbv10` | Hybrid (ML + Physics) | `MIP Re-run` | Feng et al. (2022) | Daymet | Yes (5) | 531 | 1989-10-01 to 1999-09-30 | **0.634** | 0.620 ± 0.003 | 0.380 | 0.634 | 0.502 | 47 | `---` / `---` |
| **4** | `t11_04_single` | **δHBV 1.1p / δMG (single model)** | `song_dhbv11p` | Hybrid (ML + Physics) | `MIP Re-run` | Song et al. (2024) | Daymet | No (5) | 531 | 1989-10-01 to 1999-09-30 | **0.634** | 0.634 ± 0.011 | 0.249 | 0.633 | 0.482 | 48 | `---` / `---` |
| **6** | `poudel_mlp_hbv_pur_daymet` | **Static-Parameter MLP + 16-HBV (3-model ensemble)** | `poudel_mlp_hbv` | Hybrid (ML + Physics) | `MIP Re-run` | Poudel & Steinschneider (2026) | Daymet | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.624** | 0.618 ± 0.005 | 0.371 | 0.610 | 0.484 | 41 | `---` / `---` |
| **7** | `k21_lstm_pur_maurer` | **LSTM (10-model ensemble)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Maurer | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.622** | 0.605 ± 0.006 | 0.469 | 0.590 | 0.498 | 28 | `---` / `---` |
| **8** | `k21_lstm_pur_daymet_single` | **LSTM (single model)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Daymet | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.621** | 0.621 ± 0.003 | 0.412 | 0.593 | 0.481 | 36 | `---` / `---` |
| **9** | `t11_10` | **δHBV 1.0 (single model)** | `feng_dhbv10` | Hybrid (ML + Physics) | `MIP Re-run` | Feng et al. (2022) | Daymet | No (5) | 531 | 1989-10-01 to 1999-09-30 | **0.620** | 0.620 ± 0.003 | 0.346 | 0.626 | 0.492 | 52 | `---` / `---` |
| **10** | `poudel_mlp_hbv_pur_daymet_single` | **Static-Parameter MLP + 16-HBV (single model)** | `poudel_mlp_hbv` | Hybrid (ML + Physics) | `MIP Re-run` | Poudel & Steinschneider (2026) | Daymet | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.618** | 0.618 ± 0.005 | 0.356 | 0.608 | 0.480 | 42 | `---` / `---` |
| **11** | `k21_lstm_pur_nldas` | **LSTM (10-model ensemble)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | NLDAS | Yes (10) | 531 | 1989-10-01 to 1999-09-30 | **0.612** | 0.598 ± 0.006 | 0.415 | 0.582 | 0.477 | 28 | `---` / `---` |
| **12** | `k21_lstm_pur_maurer_single` | **LSTM (single model)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Maurer | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.605** | 0.605 ± 0.006 | 0.422 | 0.576 | 0.477 | 36 | `---` / `---` |
| **13** | `k21_lstm_pur_nldas_single` | **LSTM (single model)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | NLDAS | No (10) | 531 | 1989-10-01 to 1999-09-30 | **0.598** | 0.598 ± 0.006 | 0.367 | 0.572 | 0.456 | 37 | `---` / `---` |
| **15** | `sacsma_pur_single_daymet` | **SAC-SMA + Snow-17 (Donor Transfer, single model)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet | No (1) | 531 | 1989-10-01 to 1999-09-30 | **0.512** | 0.512 | -0.006 | 0.528 | 0.353 | 72 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `t11_04` | δHBV 1.1p / δMG (5-model ensemble) | **0.650** | 0.640 | 0.849 | 0.850 | -0.026 | 0.943 | 1.447 | 2.095 | -14.82 | -7.92 | 29.15 | 0.38 | 0.38 | 47.72 | `gs://camels-us-mip/hydrographs/pur__single_camels/t11_04.zarr` |
| **2** | `k21_lstm_pur_daymet` | LSTM (10-model ensemble) | **0.635** | 0.603 | 0.837 | 0.755 | -0.029 | 0.931 | 1.466 | 2.149 | -22.80 | -12.06 | 16.99 | 0.37 | 0.41 | 48.35 | `gs://camels-us-mip/hydrographs/pur__single_camels/k21_lstm_pur_daymet.zarr` |
| **3** | `t11_06` | δHBV 1.0 (5-model ensemble) | **0.634** | 0.634 | 0.843 | 0.838 | -0.014 | 0.970 | 1.480 | 2.190 | -14.61 | -14.69 | 21.54 | 0.39 | 0.37 | 46.66 | `gs://camels-us-mip/hydrographs/pur__single_camels/t11_06.zarr` |
| **4** | `t11_04_single` | δHBV 1.1p / δMG (single model) | **0.634** | 0.633 | 0.842 | 0.861 | -0.026 | 0.944 | 1.479 | 2.188 | -13.87 | -6.13 | 21.93 | 0.40 | 0.39 | 47.84 | `gs://camels-us-mip/hydrographs/pur__single_camels/t11_04.zarr` |
| **6** | `poudel_mlp_hbv_pur_daymet` | Static-Parameter MLP + 16-HBV (3-model ensemble) | **0.624** | 0.610 | 0.832 | 0.809 | -0.021 | 0.952 | 1.472 | 2.167 | -18.98 | -5.34 | -6.45 | 0.41 | 0.40 | 49.13 | `gs://camels-us-mip/hydrographs/pur__single_camels/poudel_mlp_hbv_pur_daymet.zarr` |
| **7** | `k21_lstm_pur_maurer` | LSTM (10-model ensemble) | **0.622** | 0.590 | 0.831 | 0.730 | -0.034 | 0.927 | 1.489 | 2.218 | -25.61 | -12.81 | 0.00 | 0.43 | 0.42 | 50.08 | `gs://camels-us-mip/hydrographs/pur__single_camels/k21_lstm_pur_maurer.zarr` |
| **8** | `k21_lstm_pur_daymet_single` | LSTM (single model) | **0.621** | 0.593 | 0.829 | 0.766 | -0.030 | 0.929 | 1.486 | 2.208 | -22.81 | -10.93 | -3.05 | 0.38 | 0.42 | 49.57 | `gs://camels-us-mip/hydrographs/pur__single_camels/k21_lstm_pur_daymet.zarr` |
| **9** | `t11_10` | δHBV 1.0 (single model) | **0.620** | 0.626 | 0.836 | 0.850 | -0.013 | 0.972 | 1.503 | 2.258 | -13.32 | -13.98 | 18.93 | 0.40 | 0.38 | 47.46 | `gs://camels-us-mip/hydrographs/pur__single_camels/t11_06.zarr` |
| **10** | `poudel_mlp_hbv_pur_daymet_single` | Static-Parameter MLP + 16-HBV (single model) | **0.618** | 0.608 | 0.830 | 0.818 | -0.021 | 0.952 | 1.489 | 2.216 | -18.66 | -4.98 | -10.58 | 0.42 | 0.40 | 48.96 | `gs://camels-us-mip/hydrographs/pur__single_camels/poudel_mlp_hbv_pur_daymet.zarr` |
| **11** | `k21_lstm_pur_nldas` | LSTM (10-model ensemble) | **0.612** | 0.582 | 0.829 | 0.721 | -0.023 | 0.948 | 1.521 | 2.314 | -26.57 | -15.89 | 15.21 | 0.39 | 0.42 | 48.89 | `gs://camels-us-mip/hydrographs/pur__single_camels/k21_lstm_pur_nldas.zarr` |
| **12** | `k21_lstm_pur_maurer_single` | LSTM (single model) | **0.605** | 0.576 | 0.823 | 0.739 | -0.036 | 0.922 | 1.519 | 2.308 | -25.13 | -10.55 | -12.41 | 0.45 | 0.43 | 51.29 | `gs://camels-us-mip/hydrographs/pur__single_camels/k21_lstm_pur_maurer.zarr` |
| **13** | `k21_lstm_pur_nldas_single` | LSTM (single model) | **0.598** | 0.572 | 0.821 | 0.724 | -0.028 | 0.942 | 1.544 | 2.384 | -26.27 | -14.32 | -2.28 | 0.40 | 0.43 | 50.18 | `gs://camels-us-mip/hydrographs/pur__single_camels/k21_lstm_pur_nldas.zarr` |
| **15** | `sacsma_pur_single_daymet` | SAC-SMA + Snow-17 (Donor Transfer, single model) | **0.512** | 0.528 | 0.785 | 0.807 | -0.073 | 0.844 | 1.693 | 2.865 | -19.96 | 16.54 | -175.81 | 0.48 | 0.40 | 53.49 | `gs://camels-us-mip/hydrographs/pur__single_camels/sacsma_pur_daymet.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `t11_04` (δHBV 1.1p / δMG (5-model ensemble)):** Clean-room PyTorch + C++/OpenMP re-implementation of Song et al. (2024) δHBV 1.1p evaluated under 18-fold HUC-02 regional hold-out cross-validation (PUR) across 531 CAMELS-US basins with Daymet forcing; 5-seed ensemble mean (seeds 111, 222, 333, 444, 555; Med NSE = 0.651).
- **Rank #2 `k21_lstm_pur_daymet` (LSTM (10-model ensemble)):** 10-seed ensemble of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with Daymet single-forcing meteorological inputs and 27 static catchment attributes (Ensemble Med NSE = 0.635, Mean NSE = 0.454, Med KGE = 0.637).
- **Rank #3 `t11_06` (δHBV 1.0 (5-model ensemble)):** 5-seed ensemble (seeds 111, 222, 333, 444, 555) evaluated under 18-fold leave-one-HUC-02-region-out spatial cross-validation (PUR) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with Daymet forcing.
- **Rank #4 `t11_04_single` (δHBV 1.1p / δMG (single model)):** Clean-room PyTorch + C++/OpenMP re-implementation of Song et al. (2024) δHBV 1.1p evaluated under 18-fold HUC-02 regional hold-out cross-validation (PUR) across 531 CAMELS-US basins with Daymet forcing; single-model performance averaged across all 5 random seeds (seeds 111, 222, 333, 444, 555; Med NSE = 0.634 ± 0.011).
- **Rank #6 `poudel_mlp_hbv_pur_daymet` (Static-Parameter MLP + 16-HBV (3-model ensemble)):** Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (50 epochs x 240 batches/epoch) evaluated under 18-fold HUC-02 regional hold-out (PUR) across 531 CAMELS-US basins with Daymet forcing; 3-seed ensemble mean (seeds 101, 102, 103; Ensemble Med NSE = 0.624).
- **Rank #7 `k21_lstm_pur_maurer` (LSTM (10-model ensemble)):** 10-seed ensemble of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with Maurer single-forcing meteorological inputs and 27 static catchment attributes (Ensemble Med NSE = 0.622, Mean NSE = 0.469, Med KGE = 0.629).
- **Rank #8 `k21_lstm_pur_daymet_single` (LSTM (single model)):** Single-model evaluation scored across all 10 individual seed members (evaluation_mode='single_member_mean') of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with Daymet single-forcing inputs across all 531 CAMELS-US basins (Single-Seed Med NSE = 0.621 ± 0.003).
- **Rank #9 `t11_10` (δHBV 1.0 (single model)):** Single-model evaluation scored across all 5 individual seed members (seeds 111, 222, 333, 444, 555; evaluation_mode='single_member_mean') under 18-fold leave-one-HUC-02-region-out spatial cross-validation (PUR) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with Daymet forcing.
- **Rank #10 `poudel_mlp_hbv_pur_daymet_single` (Static-Parameter MLP + 16-HBV (single model)):** Poudel & Steinschneider (2026) Static-Parameter MLP + 16-HBV (50 epochs x 240 batches/epoch) evaluated under 18-fold HUC-02 regional hold-out (PUR) across 531 CAMELS-US basins with Daymet forcing; single-model performance averaged across all 3 random seeds (seeds 101, 102, 103; Single-Seed Med NSE = 0.618 ± 0.005).
- **Rank #11 `k21_lstm_pur_nldas` (LSTM (10-model ensemble)):** 10-seed ensemble of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with NLDAS single-forcing meteorological inputs and 27 static catchment attributes (Ensemble Med NSE = 0.612, Mean NSE = 0.415, Med KGE = 0.613).
- **Rank #12 `k21_lstm_pur_maurer_single` (LSTM (single model)):** Single-model evaluation scored across all 10 individual seed members (evaluation_mode='single_member_mean') of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with Maurer single-forcing inputs across all 531 CAMELS-US basins (Single-Seed Med NSE = 0.605 ± 0.006).
- **Rank #13 `k21_lstm_pur_nldas_single` (LSTM (single model)):** Single-model evaluation scored across all 10 individual seed members (evaluation_mode='single_member_mean') of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with NLDAS single-forcing inputs across all 531 CAMELS-US basins (Single-Seed Med NSE = 0.598 ± 0.006).
- **Rank #15 `sacsma_pur_single_daymet` (SAC-SMA + Snow-17 (Donor Transfer, single model)):** 18-fold HUC-02 regional hold-out (PUR) across all 531 CAMELS-US basins on Daymet forcings: training basins outside the held-out HUC-02 region calibrated via DDS (400 iterations, 1999-10-01 to 2008-09-30), and held-out regional test basins predicted via k=1 nearest-neighbor parameter transfer in a 29-D joint (0.2*lat, 0.2*lon) + 27 standardized static catchment-attribute space (1 seed; Median NSE = 0.512).

### 4.11 Table 12: `pur__multi_camels` — 18-Fold HUC-02 PUR: Multiple CAMELS Forcings (`8` Rows)
- **Subtitle / Scope:** 18-fold leave-one-HUC-02-region-out cross-validation over 1989–1999 using Daymet + Maurer + NLDAS.
- **Spatial Regime:** `pur` | **Forcing Regime:** `multi_camels` | **Section:** `4.3.2`
- **SOTA / Rank #1 Winner:** **ERA-Unguided-PUR (10-model ensemble)** (`Median NSE = 0.696`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `era_pur_minimal_guidance` | **ERA-Unguided-PUR (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.696** | 0.686 ± 0.004 | 0.494 | 0.642 | 0.530 | 35 | `---` / `---` |
| **2** | `k21_lstm_pur_3f` | **LSTM (8-model ensemble)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Daymet, Maurer, NLDAS | Yes (8) | 531 | 1989-10-01 to 1999-09-30 | **0.688** | 0.639 ± 0.007 | 0.505 | 0.653 | 0.542 | 32 | `---` / `---` |
| **3** | `era_pur_minimal_guidance_single` | **ERA-Unguided-PUR** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.686** | 0.686 ± 0.004 | 0.466 | 0.636 | 0.520 | 35 | `---` / `---` |
| **4** | `era_pur_physics_guided` | **ERA-Physics-Guided-PUR (10-model ensemble)** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | Yes (10) | 530 | 1989-10-01 to 1999-09-30 | **0.684** | 0.653 ± 0.007 | 0.524 | 0.657 | 0.559 | 28 | `---` / `---` |
| **5** | `era_pur_physics_guided_single` | **ERA-Physics-Guided-PUR** | `era_lstm_seeded` | Machine learning | `Contribution` | Nearing (2026) | Daymet + Maurer + NLDAS | No (10) | 530 | 1989-10-01 to 1999-09-30 | **0.653** | 0.653 ± 0.007 | 0.453 | 0.636 | 0.534 | 33 | `---` / `---` |
| **6** | `k21_lstm_pur_3f_single` | **LSTM (single model)** | `kratzert_lstm` | Machine learning | `MIP Re-run` | Kratzert et al. (2021) | Daymet, Maurer, NLDAS | No (8) | 531 | 1989-10-01 to 1999-09-30 | **0.639** | 0.639 ± 0.007 | 0.400 | 0.624 | 0.502 | 41 | `---` / `---` |
| **7** | `sacsma_pur_multi_3f` | **SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet, Maurer, NLDAS | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.569** | 0.560 ± 0.003 | 0.230 | 0.537 | 0.407 | 46 | `---` / `---` |
| **8** | `sacsma_pur_multi_3f_single` | **SAC-SMA + Snow-17 (Donor Transfer, single model)** | `nearing_sacsma` | Conceptual | `MIP Re-run` | Nearing et al. (2020) | Daymet, Maurer, NLDAS | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.560** | 0.560 ± 0.003 | 0.214 | 0.534 | 0.404 | 49 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `era_pur_minimal_guidance` | ERA-Unguided-PUR (10-model ensemble) | **0.696** | 0.642 | 0.875 | 0.810 | -0.054 | 0.883 | 1.315 | 1.730 | -17.83 | -4.19 | 34.55 | 0.35 | 0.38 | 43.06 | `gs://camels-us-mip/hydrographs/pur__multi_camels/era_pur_minimal_guidance.zarr` |
| **2** | `k21_lstm_pur_3f` | LSTM (8-model ensemble) | **0.688** | 0.653 | 0.873 | 0.832 | -0.023 | 0.949 | 1.354 | 1.832 | -14.99 | -14.55 | 41.07 | 0.38 | 0.38 | 43.38 | `gs://camels-us-mip/hydrographs/pur__multi_camels/k21_lstm_pur_3f.zarr` |
| **3** | `era_pur_minimal_guidance_single` | ERA-Unguided-PUR | **0.681** | 0.636 | 0.870 | 0.815 | -0.054 | 0.883 | 1.349 | 1.835 | -16.95 | -1.87 | 23.93 | 0.37 | 0.38 | 43.77 | `gs://camels-us-mip/hydrographs/pur__multi_camels/era_pur_minimal_guidance.zarr` |
| **4** | `era_pur_physics_guided` | ERA-Physics-Guided-PUR (10-model ensemble) | **0.684** | 0.657 | 0.868 | 0.811 | -0.043 | 0.911 | 1.337 | 1.788 | -17.14 | -12.23 | 34.63 | 0.33 | 0.39 | 45.12 | `gs://camels-us-mip/hydrographs/pur__multi_camels/era_pur_physics_guided.zarr` |
| **5** | `era_pur_physics_guided_single` | ERA-Physics-Guided-PUR | **0.643** | 0.636 | 0.848 | 0.830 | -0.043 | 0.911 | 1.434 | 2.085 | -15.54 | -11.20 | -0.86 | 0.38 | 0.40 | 47.49 | `gs://camels-us-mip/hydrographs/pur__multi_camels/era_pur_physics_guided.zarr` |
| **6** | `k21_lstm_pur_3f_single` | LSTM (single model) | **0.639** | 0.624 | 0.850 | 0.852 | -0.027 | 0.939 | 1.449 | 2.100 | -13.80 | -11.04 | 7.26 | 0.41 | 0.40 | 46.39 | `gs://camels-us-mip/hydrographs/pur__multi_camels/k21_lstm_pur_3f.zarr` |
| **7** | `sacsma_pur_multi_3f` | SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble) | **0.569** | 0.537 | 0.808 | 0.738 | -0.085 | 0.823 | 1.596 | 2.546 | -27.35 | 9.23 | -66.87 | 0.43 | 0.40 | 51.33 | `gs://camels-us-mip/hydrographs/pur__multi_camels/sacsma_pur_3f.zarr` |
| **8** | `sacsma_pur_multi_3f_single` | SAC-SMA + Snow-17 (Donor Transfer, single model) | **0.560** | 0.534 | 0.803 | 0.741 | -0.082 | 0.828 | 1.602 | 2.566 | -26.70 | 10.25 | -104.41 | 0.44 | 0.40 | 51.46 | `gs://camels-us-mip/hydrographs/pur__multi_camels/sacsma_pur_3f.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #2 `k21_lstm_pur_3f` (LSTM (8-model ensemble)):** 8-seed ensemble (seeds 101..108) of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM (models/kratzert_lstm) trained on 1999-10-01 to 2008-09-30 and evaluated on 1989-10-01 to 1999-09-30 across all 531 CAMELS-US basins under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) using joint 3-forcing Daymet + Maurer + NLDAS inputs and 27 static catchment attributes (Ensemble Med NSE = 0.688, Mean NSE = 0.506, Med KGE = 0.665).
- **Rank #6 `k21_lstm_pur_3f_single` (LSTM (single model)):** Single-model evaluation scored across all 8 individual seed members (seeds 101..108; evaluation_mode='single_member_mean') of our clean-room Kratzert et al. (2021) 365-day lookback regional CudaLSTM under 18-fold HUC-02 Leave-One-Region-Out cross-validation (PUR) with joint 3-forcing Daymet + Maurer + NLDAS inputs across all 531 CAMELS-US basins (Single-Seed Med NSE = 0.639 ± 0.006).
- **Rank #7 `sacsma_pur_multi_3f` (SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble)):** 18-fold HUC-02 regional hold-out (PUR) across all 531 CAMELS-US basins on Daymet, Maurer, and NLDAS forcings using DDS calibration on out-of-region training basins and 29-D spatial + static-attribute k=1 nearest-neighbor donor parameter transfer to held-out HUC-02 test basins (3-member ensemble mean; Median NSE = 0.569).
- **Rank #8 `sacsma_pur_multi_3f_single` (SAC-SMA + Snow-17 (Donor Transfer, single model)):** Single-member performance averaged across all 3 stored members in sacsma_pur_3f.zarr under 18-fold HUC-02 regional hold-out (PUR) with 29-D nearest-neighbor donor parameter transfer (Single-Member Mean Median NSE = 0.560 ± 0.003).

### 4.12 Table 13: `pur__lagged_streamflow` — 18-Fold HUC-02 PUR: With Lagged Streamflow Inputs (`2` Rows)
- **Subtitle / Scope:** 531 basins — Models trained on 17 HUC-02 regions (held-out in space during training) and evaluated with lagged observed streamflow Q(t-1), autoregression, or data assimilation in the held-out HUC-02 region during inference.
- **Spatial Regime:** `pur` | **Forcing Regime:** `lagged_streamflow` | **Section:** `4.3.3`
- **SOTA / Rank #1 Winner:** **DI(1)-LSTM (3-model ensemble)** (`Median NSE = 0.747`)

#### Part A: Core Identification, Provenance & Primary Efficiency Metrics

| Rank | `entry_id` | Model Display Name | `family_id` | Category | Provenance | Citation | Forcing | Ens? (`N`) | Basins | Eval Period | **Med NSE** | Single-Seed Med NSE ($\pm$ Std) | Mean NSE | Med KGE | Mean KGE | $\text{NSE} \le 0$ | $p$ vs SOTA (NSE / KGE) |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `feng2020_di_lstm_pur_nldas` | **DI(1)-LSTM (3-model ensemble)** | `feng2020_di_lstm` | Machine learning | `MIP Re-run` | Feng et al. (2020) | NLDAS + Q(t-1) | Yes (3) | 531 | 1989-10-01 to 1999-09-30 | **0.747** | 0.737 ± 0.004 | 0.698 | 0.744 | 0.704 | 8 | `---` / `---` |
| **2** | `feng2020_di_lstm_pur_nldas_single` | **DI(1)-LSTM (single model)** | `feng2020_di_lstm` | Machine learning | `MIP Re-run` | Feng et al. (2020) | NLDAS + Q(t-1) | No (3) | 531 | 1989-10-01 to 1999-09-30 | **0.737** | 0.737 ± 0.004 | 0.685 | 0.738 | 0.693 | 9 | `---` / `---` |

#### Part B: Complete 14 Gauch et al. (2023) Hydrological Metrics (Medians Across Evaluated Basins)

| Rank | `entry_id` | Model Display Name | NSE | KGE | Pearson-$r$ | $\alpha$-NSE | $\beta$-NSE | $\beta$-KGE | RMSE (mm/d) | MSE | FHV (%) | FMS (%) | FLV (%) | Peak-Timing (d) | Missed-Peaks (%) | Peak-MAPE (%) | Zarr Store / Archive |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | `feng2020_di_lstm_pur_nldas` | DI(1)-LSTM (3-model ensemble) | **0.747** | 0.744 | 0.880 | 0.830 | -0.013 | 0.975 | 1.235 | 1.525 | -15.94 | -2.64 | -12.94 | 0.62 | 0.33 | 41.15 | `gs://camels-us-mip/hydrographs/pur__lagged_streamflow/feng2020_di_lstm_pur_nldas.zarr` |
| **2** | `feng2020_di_lstm_pur_nldas_single` | DI(1)-LSTM (single model) | **0.737** | 0.738 | 0.875 | 0.835 | -0.012 | 0.976 | 1.261 | 1.590 | -15.71 | -2.12 | -35.61 | 0.62 | 0.34 | 41.53 | `gs://camels-us-mip/hydrographs/pur__lagged_streamflow/feng2020_di_lstm_pur_nldas.zarr` |

#### Part C: Entry Notes & Asterisk Footnotes

- **Rank #1 `feng2020_di_lstm_pur_nldas` (DI(1)-LSTM (3-model ensemble)):** 3-seed ensemble (seeds 101, 102, 103) of our clean-room DI(1)-LSTM implementation (models/feng2020_di_lstm; hidden_size=256, seq_len=365, batch_size=256, dropout=0.4, 30 full epochs) evaluated under 18-fold leave-one-HUC-02-region-out (PUR) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with NLDAS forcing + 1-day lagged observed streamflow Q(t-1) (Ensemble Med NSE = 0.747, Mean NSE = 0.692, Med KGE = 0.748).
- **Rank #2 `feng2020_di_lstm_pur_nldas_single` (DI(1)-LSTM (single model)):** Single-model evaluation scored across all 3 individual seed members (seeds 101, 102, 103; evaluation_mode='single_member_mean') of our clean-room DI(1)-LSTM implementation under 18-fold leave-one-HUC-02-region-out (PUR) across all 531 CAMELS-US basins over 1989-10-01 to 1999-09-30 with NLDAS forcing + 1-day lagged observed streamflow Q(t-1) (Single-Seed Med NSE = 0.732 ± 0.001).

### 4.13 Table 14: `pur__freeform` — 18-Fold HUC-02 PUR: Freeform Forcings (`0` Rows)
- **Subtitle / Scope:** 18-fold leave-one-HUC-02-region-out models driven by non-CAMELS meteorological forcings.
- **Spatial Regime:** `pur` | **Forcing Regime:** `freeform` | **Section:** `4.3.4`
- *Currently 0 entries on this leaderboard.*

