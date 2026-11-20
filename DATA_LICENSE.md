# Data License — CAMELS-US Model Intercomparison Project (`gs://camels-us-mip/`)

This document specifies the licensing terms for all **datasets, basin/fold split definitions, leaderboard tables, figures, and daily hydrograph Zarr v2 stores** published by the **CAMELS-US Model Intercomparison Project (`camels-us-mip`)**, both in this Git repository (`splits/`, `leaderboards/`, `figures/`) and in the companion public cloud archive `gs://camels-us-mip/` (and its Zenodo deposit).

Executable software and source code are licensed separately under the **Apache License, Version 2.0** (see [`LICENSE`](./LICENSE), [`NOTICE`](./NOTICE), and [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)).

---

## 1. First-Party Data, Splits, Leaderboards, Figures, and MIP / ERA Hydrographs (`CC-BY-4.0`)

All first-party data artifacts produced by the CAMELS-US MIP—including:
- the canonical 531-basin list, 12-fold PUB splits, and 18-fold HUC-02 PUR splits (`splits/` and `gs://camels-us-mip/splits/`),
- all leaderboard JSON, CSV, and Markdown tables (`leaderboards/*`),
- all manuscript figures (`figures/*`),
- all `MIP Re-run` and `Contribution` (`era_*`) consolidated Zarr v2 hydrograph stores in `gs://camels-us-mip/hydrographs/`,
- all Stage-1 author-protocol confirmation Zarr stores in `gs://camels-us-mip/stage1_author_reproductions/`, and
- `gs://camels-us-mip/hydrographs/manifest_index.json`

are licensed under the **Creative Commons Attribution 4.0 International License (`CC-BY-4.0`)** (`https://creativecommons.org/licenses/by/4.0/`).

**Copyright 2025–2026 Google LLC, Grey Nearing, Daniel Klotz, and the CAMELS-US MIP Contributors.**

You are free to copy, redistribute, remix, transform, and build upon these datasets for any purpose, even commercially, provided that you give appropriate credit:

> Nearing, G. S., Klotz, D., & Ebel, P. (2026). *HydroMIP: A CAMELS-US Model Intercomparison Project* (`hydromip.org`). Licensed under CC BY 4.0.

When using simulated hydrographs from any model family in `gs://camels-us-mip/`, please also cite the original publication introducing that model architecture (recorded in `leaderboards/table_2_registry.json` and in each Zarr store's `.zattrs`).

---

## 2. Third-Party Hydrograph Archives & Target Observations in `gs://camels-us-mip/`

Each `.zarr` directory in `gs://camels-us-mip/` is an independent dataset in a storage aggregate (`CC-BY-4.0` / `CC-BY-NC-4.0` Section 2(a)(1); `GPL-3.0` Section 5). Third-party author hydrograph archives that have been standardized by the MIP (subset to the 531 Newman basins, converted to $\text{mm day}^{-1}$, aligned to `1989-10-01`–`1999-09-30`, and written as consolidated Zarr v2) retain their original upstream licenses as detailed below:

| Zarr Store(s) in `gs://camels-us-mip/` | Model Family (`family_id`) | Upstream Source & DOI | Governing Data License | Required Attribution & Modification Statement |
| :--- | :--- | :--- | :--- | :--- |
| `observations/newman_eval_targets.zarr` (and `qobs` in `stage1_author_reproductions/*.zarr`) | *Target Observations* | USGS NWIS & NCAR GDEX CAMELS-US v1.2 (Newman et al., 2015; Addor et al., 2017; `https://doi.org/10.5065/D6MW2F4D`, `https://doi.org/10.5065/D6G73C3Q`) | **U.S. Government Public Domain (`17 U.S.C. § 105`) & `CC-BY-4.0`** | USGS NWIS streamflow observations are in the U.S. Public Domain (`17 U.S.C. § 105`); CAMELS-US packaging and basin areas (Newman et al., 2015; Addor et al., 2017) are licensed under `CC-BY-4.0` / UCAR Terms of Use. Converted to $\text{mm day}^{-1}$ across 531 basins by the CAMELS-US MIP. |
| `hydrographs/gauged__single_camels/t3_02.zarr`, `t3_06.zarr`, `t3_15.zarr`; `hydrographs/gauged__multi_camels/t4_05.zarr`, `t4_08.zarr`, `t4_09.zarr`, `t4_19.zarr` (and member 0 of `t4_06.zarr`) | `kratzert2021_lstm_archive` (and `li2025_lstm_dhbv_ens`) | Kratzert, Klotz, Hochreiter, & Nearing (2021), Zenodo `https://doi.org/10.5281/zenodo.4670268` | **`CC-BY-4.0`** | Contains data from Kratzert et al. (2021, *HESS*, `10.5281/zenodo.4670268`), licensed under `CC-BY-4.0`; reformatted to consolidated Zarr v2 (`531` basins, `1989-10-01`–`1999-09-30`) by the CAMELS-US MIP. |
| `hydrographs/gauged__single_camels/t3_11.zarr`, `t3_23.zarr`, `t3_44.zarr` | `kratzert2019b_lstm`, `kratzert_ealstm`, `kratzert_lstm_nostatic` | Kratzert et al. (2019b), HydroShare `https://doi.org/10.4211/hs.83ea5312635e44dc824eeb99eda12f06` & Zenodo `https://doi.org/10.5281/zenodo.3530884` | **`CC-BY-4.0`** | Contains data from Kratzert et al. (2019b, *HESS*, `10.4211/hs.83ea5312635e44dc824eeb99eda12f06`), licensed under `CC-BY-4.0`; reformatted to consolidated Zarr v2 by the CAMELS-US MIP. |
| `hydrographs/gauged__single_camels/t3_39.zarr`, `t3_41.zarr`, `t3_43.zarr`, `t3_45.zarr`, `t3_46.zarr`, `t3_48.zarr`, `t3_52.zarr`, `t7_19.zarr`, `t7_22.zarr`; `hydrographs/pub__single_camels/t7_23.zarr` | `seibert_hbv`, `mizukami_mhm`, `fuse_suite`, `newman_sacsma`, `newman_vic`, `rakovec_mhm_mpr`, `mizukami_vic_mpr` | *CAMELS benchmark models* archive curated by Kratzert et al. (2019b), HydroShare `https://doi.org/10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` & NCAR GDEX `https://doi.org/10.5065/D6MW2F4D` | **`CC-BY-4.0`** | Contains simulations by Addor et al. (2017), Newman et al. (2017), Mizukami et al. (2017, 2019), Seibert et al. (2018), and Rakovec et al. (2019) from the *CAMELS benchmark models* archive (Kratzert et al., 2019b, `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1`), licensed under `CC-BY-4.0`; reformatted to consolidated Zarr v2 by the CAMELS-US MIP. |
| `hydrographs/gauged__single_camels/t3_12.zarr`, `t3_28.zarr` | `frame2023_mclstm_archive` | Frame, Kratzert, Gupta, Sorooshian, & Nearing (2022, 2023), HydroShare `https://doi.org/10.4211/hs.d750278db868447dbd252a8c5431affd` | **`CC-BY-NC-4.0`** | Contains data from *MC-LSTM papers, model runs* (Frame et al., 2022, 2023, `10.4211/hs.d750278db868447dbd252a8c5431affd`), licensed under `CC-BY-NC-4.0` (non-commercial research and benchmarking use with attribution); reformatted to consolidated Zarr v2 by the CAMELS-US MIP. Commercial users requiring `CC-BY-4.0` `MC-LSTM` hydrographs should use `hoedt_mclstm_maurer.zarr`, `hoedt_mclstm_3f_avgp.zarr`, and `hoedt_mclstm_3f_ens_b.zarr`. |

---

## 3. Complete Summary of All 58 Active Stage-2 Hydrograph Zarr Stores by License

| License | Store Count | Stores in `gs://camels-us-mip/hydrographs/` |
| :--- | :---: | :--- |
| **`CC-BY-4.0` (First-Party `MIP Re-run` & `ERA Contribution`)** | **36** | `feng2020_di_lstm_nldas.zarr`, `feng2020_di_lstm_pub_nldas.zarr`, `feng2020_di_lstm_pur_nldas.zarr`, `yin2022_rrformer_daymet.zarr`, `hoedt_mclstm_maurer.zarr`, `hoedt_mclstm_3f_avgp.zarr`, `hoedt_mclstm_3f_ens_b.zarr`, `sacsma_daymet.zarr`, `sacsma_3f_ens.zarr`, `sacsma_pub_daymet.zarr`, `sacsma_pub_3f.zarr`, `sacsma_pur_daymet.zarr`, `sacsma_pur_3f.zarr`, `t3_18.zarr`, `t3_21.zarr`, `t3_29.zarr`, `t4_06.zarr`, `poudel_mlp_hbv_daymet_3s.zarr`, `poudel_mlp_hbv_pub_daymet.zarr`, `poudel_mlp_hbv_pur_daymet.zarr`, `k21_lstm_pub_3f.zarr`, `t7_06.zarr`, `t7_10.zarr`, `k21_lstm_pur_3f.zarr`, `k21_lstm_pur_daymet.zarr`, `k21_lstm_pur_maurer.zarr`, `k21_lstm_pur_nldas.zarr`, `t11_04.zarr`, `t11_06.zarr`, `era_gauged_minimal_guidance.zarr`, `era_gauged_multi_paradigm.zarr`, `era_gauged_physics_guided.zarr`, `era_pub_minimal_guidance.zarr`, `era_pub_physics_guided.zarr`, `era_pur_minimal_guidance.zarr`, `era_pur_physics_guided.zarr` |
| **`CC-BY-4.0` (External Author Archives)** | **20** | `t3_02.zarr`, `t3_06.zarr`, `t3_11.zarr`, `t3_15.zarr`, `t3_23.zarr`, `t3_39.zarr`, `t3_41.zarr`, `t3_43.zarr`, `t3_44.zarr`, `t3_45.zarr`, `t3_46.zarr`, `t3_48.zarr`, `t3_52.zarr`, `t4_05.zarr`, `t4_08.zarr`, `t4_09.zarr`, `t4_19.zarr`, `t7_19.zarr`, `t7_22.zarr`, `t7_23.zarr` |
| **`CC-BY-NC-4.0` (External Non-Commercial Archive: Frame et al., 2023)** | **2** | `t3_12.zarr`, `t3_28.zarr` |
| **Total Active Stage-2 Hydrograph Stores** | **58** | — |

---

## 4. Policy on Community Submissions & External Archives

1. **Community Contributions (`hydromip.org`):** Community-contributed hydrograph stores submitted to the MIP leaderboards are licensed under **`CC-BY-4.0`** with scholarly attribution to the contributing authors.
2. **Literature-Only Entries:** For `Literature` and `Literature*` entries (`kratzert2019a_pub_lstm`, `feng2020_di_lstm_lit`, `gauch_xgboost_lit`, `hoedt_mclstm_lit`, `tsai2021_dpl_vic`, `nearing2022_ar_da_lstm`, `jamaat2025_dhbv_da`, `liu2025_tslib_suite`, `wang_s4dft`), **zero hydrograph Zarr stores and zero source code files** are redistributed; only published scalar summary statistics (`17 U.S.C. § 102(b)`) are reported in `leaderboards/*.json` with full academic citations.
3. **No Warranty:** All datasets and hydrograph archives are provided "AS IS", without warranty of any kind, express or implied, in accordance with `CC-BY-4.0` Section 5 and the respective upstream data licenses.
