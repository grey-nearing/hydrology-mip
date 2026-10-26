# Open-Source Code and Hydrograph Archive Licensing Assessment

This document provides a complete licensing audit for every third-party model codebase, source file, and external hydrograph archive used in the **CAMELS-US Model Intercomparison Project (`camels-us-mip`)**. It covers both:

- **Part A — Model Code Repositories and Source Code (21 items):** All third-party and first-party model implementations used in or copied into [`grey-nearing/camels-us-mip/models/`](https://github.com/grey-nearing/camels-us-mip).
- **Part B — Hydrograph Data Repositories and External Archives (10 items):** All external streamflow simulation and observation archives ingested into the public Google Cloud Storage bucket (`gs://camels-us-mip/`) and Zenodo archive.

---

## Status Classification Key

Every code repository and hydrograph archive is assigned one of three public-release readiness ratings:

| Status Rating | Meaning | Impact on Public Release |
| :--- | :--- | :--- |
| **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Licensed under a standard permissive open-source software license (`MIT`, `BSD-3-Clause`, `Apache-2.0`), an open data license (`CC-BY-4.0`, `CC0-1.0`), or U.S. Government Public Domain (`17 U.S.C. § 105`). | Safe for immediate public redistribution in `grey-nearing/camels-us-mip` and `gs://camels-us-mip/`, provided original copyright notices, license files, and scholarly citations are retained. |
| **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** | Licensed under a reciprocal copyleft license (`GPL-3.0`, `LGPL-3.0`) or a non-commercial academic license (`CC-BY-NC-4.0`, `LicenseRef-PSU-NonCommercial`, `LicenseRef-UCAR-WRF-Hydro`). | Can be redistributed publicly only if license conditions are respected: `GPL-3.0` code/data must keep its `GPL-3.0` license notice and cannot be relicensed under `MIT`/`Apache-2.0`; non-commercial (`NC`) code/data must either be clearly labeled as non-commercial, replaced by a clean-room implementation, or relicensed by the original authors. |
| **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`** | The upstream public GitHub repository has no `LICENSE` or `COPYING` file. Under international copyright law, source code posted on GitHub without a license defaults to **All Rights Reserved**. | Verbatim source code from these repositories **must not** be redistributed in a public GitHub repository until either (a) the upstream authors add an open-source license or grant written permission, or (b) the code in `models/` is replaced with a clean-room implementation written from the published mathematical equations. |

---

## Executive Summary & Pre-Release Action Checklist

Out of **31 total resources** audited (21 code repositories in Part A and 10 hydrograph archives in Part B):

- **22 resources are `GREEN`** (permissive open-source or open-data licenses, U.S. Government Public Domain, or first-party code authored by this study).
- **6 resources are `YELLOW`** (copyleft `GPL-3.0` or non-commercial `CC-BY-NC-4.0` / PSU Non-Commercial licenses; in several cases, we already use a clean-room implementation or only ingest hydrographs).
- **3 resources in Part A are `RED`** (upstream GitHub repositories missing a `LICENSE` file where local files in `models/` currently adapt or wrap upstream code).

### Required Actions Before Making `grey-nearing/camels-us-mip` and `gs://camels-us-mip/` Public

1. **Add a Root `LICENSE` File to `grey-nearing/camels-us-mip` (`Part A #21`):**
   - Commit a root `LICENSE` file (`Apache-2.0` or `MIT`) and a `THIRD_PARTY_NOTICES.md` file to the repository root.
2. **Resolve the 3 `RED` Code Items in `models/` (`Part A #7, #11, #13, #14`):**
   - **Wang et al. (2025) S4D / S4D-FT (`models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_s4dft.py`, `Part A #7`):** The underlying `state-spaces/s4` library (`S4DKernel`) is `Apache-2.0` (`GREEN`), and the training setup is adapted from `kratzert/lstm_for_pub` (`Apache-2.0`), but `WESTENR-OU/S4D_rainfall_runoff_simulations` has no `LICENSE` file. Either obtain written permission / a license file from Y. Wang et al., or keep only `Apache-2.0` `state-spaces/s4` layers inside our own standard `camels-us-mip` training wrapper.
   - **Poudel & Steinschneider (2026) MLP + 16HBV (`models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_hbv_poudel.py`, `Part A #11`):** `snpoudel/diff-hydro` has no `LICENSE` file on GitHub (though the HESS 2026 paper is `CC-BY-4.0`). Either ask Shashank Poudel & Scott Steinschneider to add a `LICENSE` file to `snpoudel/diff-hydro`, or replace `model_hbv_poudel.py` with a clean-room implementation coupling a PyTorch `nn.Sequential` static-attribute MLP to our first-party `StandaloneDHBV` CUDA HBV routing kernel.
   - **Wang et al. (2025) HydroDiffusion (`models/category_ii_tier2_s4d_hy2dl_tide_diffusion/driver_hydrodiffusion_*.py`, `Part A #13`):** `yhwang08/HydroDiffusion` has no `LICENSE` file on GitHub. Currently, `camels-us-mip` only contains our own clean driver scripts (`driver_hydrodiffusion_train.py`, `driver_hydrodiffusion_eval.py`) and does **not** copy `yhwang08/HydroDiffusion/papercode/`. Do not copy the upstream `papercode/` directory into `grey-nearing/camels-us-mip` unless the authors add an open-source license; instead, instruct users to clone `yhwang08/HydroDiffusion` externally or implement the DDIM diffusion head cleanly over `Apache-2.0` `state-spaces/s4`.
   - **Newman SAC-SMA + Snow-17 Wrapper (`models/category_ii_sacsma/optimize_single_basin.py`, `Part A #14`):** The core NOAA NWSRFS Fortran routines (`sac1.f`, `ex_sac1.f`, `snow19/*.f`, `duamel.f`) are U.S. Government Public Domain (`GREEN`), and the NCAR Fortran driver (`sacsma_source/LICENSE`) is `GPL-3.0-only` (`YELLOW`), but the `Upstream-Tech/SACSMA-SNOW17` Python wrapper (`optimize_single_basin.py`, `model/`, `optimizer/`) has no root license file. Replace `optimize_single_basin.py` with a clean-room Python `f2py` + DDS calibration script calling the public-domain NOAA Fortran routines directly.
3. **Preserve Clean-Room Separation and License Labels for `YELLOW` Resources:**
   - **MHPI $\delta\text{HBV 1.0}$ / $\delta\text{HBV 1.1p}$ (`Part A #4`, `Part B #3`):** Upstream `mhpi/*` repositories carry the Pennsylvania State University Non-Commercial Software License. Our local implementation in `models/category_ii_dhbv/dhbv_standalone_core.py` is already a 100% clean-room custom CUDA C++ and PyTorch re-implementation written from the published equations. Do not copy upstream `mhpi/hydroDL` or `mhpi/generic_deltamodel` files into `grey-nearing/camels-us-mip`.
   - **Frame et al. (2022, 2023) MC-LSTM HydroShare Archive (`Part B #5`):** HydroShare resource `d750278db868447dbd252a8c5431affd` is currently set to `CC-BY-NC-4.0`. Either label the archived `frame22_*` / `frame23_*` Zarr stores in `gs://camels-us-mip/` as `CC-BY-NC-4.0`, or have co-authors Jonathan Frame and Grey Nearing update the HydroShare resource license to `CC-BY-4.0` (matching all other Kratzert/Nearing/Frame resources).
   - **Knoben et al. (2020) MARRMoT 36-Model Hydrograph Archive (`Part B #8`):** The University of Bristol dataset (`10.5523/bris.2zutxh2qeep6y2cy6scwgk9eqj`) explicitly licenses its simulated hydrographs under `GPL-3.0`. Include the `GPL-3.0` notice in the Zarr metadata and `gs://camels-us-mip/README.md` for the `marrmot_*` stores.
4. **Fix Placeholder / Mismatched DOIs and Repository URLs in `leaderboards/table_2_registry.json`:**
   - Update 9 entries in `leaderboards/table_2_registry.json` where the recorded Zenodo DOI or GitHub URL points to an unrelated record or a moved repository (documented in full in the [Registry URL & DOI Corrections Table](#corrections-needed-in-leaderboardstable_2_registryjson) at the end of this report).

---

## Part A — Model Code Repositories & Upstream Source Code (21 Items)

### Part A Summary Table

| # | Resource / Model Family & Key Citations | Upstream URL / DOI | Local Path in `camels-us-mip` | Exact License Name & SPDX Identifier | Redistribution Status | Required Action Before Public Release |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | **`neuralhydrology/neuralhydrology`** (`CudaLSTM`, `EA-LSTM`, `MC-LSTM`, `TFT`; Kratzert et al., 2022) | `https://github.com/neuralhydrology/neuralhydrology` (`10.21105/joss.04050`, `10.5281/zenodo.7063259`) | `models/category_ii_neuralhydrology/` (`models/kratzert_lstm/`, `kratzert_ealstm/`, `hoedt_mclstm/`) | BSD 3-Clause "New" or "Revised" License (`BSD-3-Clause`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain `BSD-3-Clause` copyright notice (`Copyright (c) 2021, NeuralHydrology`) in third-party notices. |
| **A2** | **`kratzert/ealstm_regional_modeling`, `kratzert/lstm_for_pub`, `kratzert/multiple_forcing`** (Kratzert et al., 2019a, 2019b, 2021) | `https://github.com/kratzert/ealstm_regional_modeling`, `https://github.com/kratzert/lstm_for_pub`, `https://github.com/kratzert/multiple_forcing` | `models/category_ii_neuralhydrology/` (`models/kratzert_lstm/`, `models/kratzert_ealstm/`) | Apache License 2.0 (`Apache-2.0`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain `Apache-2.0` license and copyright notice (`Copyright 2019 Frederik Kratzert`). |
| **A3** | **`ml-jku/mc-lstm` & `jmframe/mclstm_2021_extrapolate`** (Hoedt et al., 2021; Frame et al., 2022, 2023) | `https://github.com/ml-jku/mc-lstm`, `https://github.com/jmframe/mclstm_2021_extrapolate` (`10.5281/zenodo.5165216`) | `models/category_ii_neuralhydrology/` (`models/hoedt_mclstm/`) | MIT License (`MIT`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain `MIT` copyright notice (`Copyright (c) 2021 Institute for Machine Learning, JKU Linz`). |
| **A4** | **`mhpi/hydroDL`, `mhpi/generic_deltamodel`, `mhpi/dhbv2`, `mhpi/dPLHBVrelease`, `mhpi/dPL`** (Feng et al., 2022, 2023 $\delta\text{HBV 1.0}$; Song et al., 2024, 2025 $\delta\text{HBV 1.1p}$; Li et al., 2025; Tsai et al., 2021; Liu et al., 2024) | `https://github.com/mhpi/hydroDL`, `https://github.com/mhpi/generic_deltamodel`, `https://github.com/mhpi/dhbv2`, `https://github.com/mhpi/dPLHBVrelease`, `10.5281/zenodo.5227738` | `models/feng_dhbv10/dhbv_standalone_core.py`, `models/song_dhbv11p/dhbv_standalone_core.py` | Upstream: Pennsylvania State University Non-Commercial Software License (`LicenseRef-PSU-NonCommercial`). Local `dhbv_standalone_core.py`: First-party clean-room CUDA/PyTorch implementation. | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (Upstream `mhpi/*` repos); **`GREEN`** for our clean-room `dhbv_standalone_core.py` | Do **not** copy verbatim files from `mhpi/*` into `grey-nearing/camels-us-mip`. Keep only our clean-room CUDA/PyTorch implementation (`dhbv_standalone_core.py`) in `models/`. |
| **A5** | **`sinanrk/TFT-streamflow` (`TFT-Runoff`)** (Rasiya Koya & Roy, 2025) | `https://github.com/sinanrk/TFT-streamflow` (listed as `sinanrk/TFT-Runoff` in `table_2_registry.json`) | `models/category_ii_ml_and_conceptual/train_tft_and_xgboost_gauged.py` (`models/rasiya_koya_tft/`) | MIT License (`MIT`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Update repo URL in `table_2_registry.json` to `https://github.com/sinanrk/TFT-streamflow` and retain `MIT` notice if any upstream code is used (local script is clean-room). |
| **A6** | **`thuml/Time-Series-Library` & Liu et al. (2025) `rainflow`** (`TSLib` 10-model benchmark: `PatchTST`, `iTransformer`, `TimeMixer`, `TimesNet`, `DLinear`, `FEDformer`, `Autoformer`, `Informer`, `Crossformer`, `Mamba`; Liu et al., 2025) | `https://github.com/thuml/Time-Series-Library` & `https://doi.org/10.5281/zenodo.15852144` (`10.5281/zenodo.17863071`) | None in `camels-us-mip/models/` (Literature published-table-only entry; staged in `<external_staging_dir>/step1_published_split1/downloads/liu2025/`) | `thuml/Time-Series-Library`: MIT License (`MIT`). Liu et al. (2025) `HESS_paper.zip`: `CC-BY-NC-4.0`. | **`GREEN — Permissive`** (`thuml/Time-Series-Library`); **`YELLOW — Non-Commercial`** (Liu et al. 2025 `rainflow` wrapper) | Do not copy `downloads/liu2025/` into `grey-nearing/camels-us-mip` (not needed since Literature only cites published metrics). |
| **A7** | **`state-spaces/s4` & `WESTENR-OU/S4D_rainfall_runoff_simulations`** (Gu et al., 2022; Wang et al., 2025 `S4D` & `S4D-FT`) | `https://github.com/state-spaces/s4`, `https://github.com/WESTENR-OU/S4D_rainfall_runoff_simulations` (`10.1029/2025WR039888`) | `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_s4dft.py`, `driver_s4dft.py` (`models/wang_s4dft/`) | `state-spaces/s4`: Apache License 2.0 (`Apache-2.0`). `WESTENR-OU/S4D_rainfall_runoff_simulations`: **No explicit license file in upstream repo — All Rights Reserved by default under copyright law**. | **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`** (for `WESTENR-OU` wrapper); **`GREEN`** for `state-spaces/s4` | Request an open-source license / written permission from `WESTENR-OU` (Y. Wang et al.), or ensure `models/wang_s4dft/` uses only `Apache-2.0` `state-spaces/s4` primitives (`S4DKernel`) inside our own clean-room hydrological training wrapper. |
| **A8** | **`kshitijtayal/Geo-ViT-LSTM`** (Tayal et al., 2024 `Geo-ViT-LSTM`) | `https://github.com/kshitijtayal/Geo-ViT-LSTM` (HTTP 404 — repository not publicly released; Paper DOI: `10.1145/3678717.3691324`) | None in `camels-us-mip/models/` (Literature published-table-only entry) | No public repository released (Not Applicable — 0 lines of code or data copied) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** (No code or data copied; factual citation only) | Remove or annotate the dead `github.com/kshitijtayal/Geo-ViT-LSTM` URL in `leaderboards/table_2_registry.json`. |
| **A9** | **`gauchm/ealstm_regional_modeling` (`proper-care-and-feeding`) & `dmlc/xgboost`** (Gauch et al., 2021 `XGBoost`) | `https://github.com/gauchm/ealstm_regional_modeling` (`10.5281/zenodo.3648270`), `https://github.com/dmlc/xgboost` | `models/category_ii_ml_and_conceptual/train_tft_and_xgboost_gauged.py` (`models/gauch_xgboost/`) | Apache License 2.0 (`Apache-2.0`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Update `github_url` in `table_2_registry.json` from `martin-gauch/proper-care-and-feeding` to `https://github.com/gauchm/ealstm_regional_modeling`. |
| **A10** | **`google-research/google-research` (`tide`)** (Das et al., 2023 `TiDE`) | `https://github.com/google-research/google-research/tree/master/tide` | `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_tide.py`, `driver_tide.py` (`models/das_tide/`) | Apache License 2.0 (`Apache-2.0`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain `Apache-2.0` notice and citation to Das et al. (2023). |
| **A11** | **`snpoudel/diff-hydro` (`poudel2026_mlp_hbv`)** (Poudel & Steinschneider, 2026 `MLP + 16HBV`) | `https://github.com/snpoudel/diff-hydro` (Paper DOI: `10.5194/hess-30-5173-2026`) | `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_hbv_poudel.py`, `driver_mlp16hbv.py` (`models/poudel_mlp_hbv/`) | **No explicit license file in upstream repo — All Rights Reserved by default under copyright law** (HESS 2026 article is `CC-BY-4.0`) | **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`** | Ask Shashank Poudel & Scott Steinschneider to add a `LICENSE` file to `snpoudel/diff-hydro` or grant permission, **or** replace `model_hbv_poudel.py` with a clean-room implementation linking an `nn.Sequential` MLP to our `dhbv_standalone_core.py` CUDA HBV kernel. |
| **A12** | **`eduardoAcunaEspinoza/Hy2DL` & `KIT-HYD/Hy2DL` (`hy2dl_hbv`)** (Acuña Espinoza et al., 2024 `LSTM + HBV`) | `https://github.com/eduardoAcunaEspinoza/Hy2DL` (`10.5281/zenodo.22083944`), legacy `https://github.com/KIT-HYD/Hy2DL` (`10.5281/zenodo.17251944`) | `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_hy2dl_hbv.py`, `driver_hy2dl.py` (`models/hy2dl_hbv/`) | Active canonical repo (`eduardoAcunaEspinoza/Hy2DL`): BSD 3-Clause License (`BSD-3-Clause`). Legacy v1.0 mirror (`KIT-HYD/Hy2DL`): `GPL-3.0-only`. | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** (under active canonical repo `eduardoAcunaEspinoza/Hy2DL` `BSD-3-Clause`) | Cite the canonical repository `https://github.com/eduardoAcunaEspinoza/Hy2DL` (`BSD-3-Clause`, `Copyright (c) 2024, Eduardo Acuña Espinoza`) in `model_hy2dl_hbv.py` and `table_2_registry.json`. |
| **A13** | **`yhwang08/HydroDiffusion` (`hydrodiffusion`)** (Wang, Yu, Zhang & Erichson, 2025 `HydroDiffusion`) | `https://github.com/yhwang08/HydroDiffusion` (`arXiv:2512.12183`) | `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/driver_hydrodiffusion_train.py`, `driver_hydrodiffusion_eval.py` (`models/hydrodiffusion/`; upstream clone only in `<external_staging_dir>/step4_tier2/repos/HydroDiffusion/`) | **No explicit license file in upstream repo — All Rights Reserved by default under copyright law** (bundles `state-spaces/s4` `Apache-2.0` in `papercode/src/`) | **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`** (for upstream `yhwang08/HydroDiffusion`); **`GREEN`** for our own `driver_hydrodiffusion_*.py` scripts | Do **not** copy `yhwang08/HydroDiffusion/papercode/` into `grey-nearing/camels-us-mip` without author permission or a clean-room DDIM diffusion head built over `Apache-2.0` `state-spaces/s4`. |
| **A14** | **`sacsma_upstream` (`Upstream-Tech/SACSMA-SNOW17` / NOAA NWSRFS / NCAR `newman_sacsma`)** (Newman et al., 2015, 2017; Ben-Shalom et al., 2022) | `https://github.com/Upstream-Tech/SACSMA-SNOW17` | `models/category_ii_sacsma/optimize_single_basin.py`, `run_pipeline_updated.py` (`models/newman_sacsma/`; upstream clone in `<external_staging_dir>/sacsma_upstream/`) | Core NOAA NWSRFS Fortran (`sac1.f`, `snow19/*.f`, `duamel.f`): **U.S. Government Public Domain (`17 U.S.C. § 105`)**. NCAR `sacsma_source/LICENSE`: **GNU General Public License v3.0 (`GPL-3.0-only`)**. `Upstream-Tech` Python wrapper: **No root license file**. | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (`sacsma_source` `GPL-3.0`) / **`RED`** (`Upstream-Tech` Python wrapper `optimize_single_basin.py`) | Replace `Upstream-Tech`'s `optimize_single_basin.py` wrapper with our own clean-room Python `f2py` + DDS calibration script calling the public-domain NOAA NWSRFS Fortran subroutines directly (making `models/newman_sacsma/` 100% `GREEN`). |
| **A15** | **`NCAR/summa`** (Clark et al., 2015; Farahani et al., 2025 `SUMMA`) | `https://github.com/NCAR/summa` | None in `camels-us-mip/models/` (cloned only in `<external_staging_dir>/scratch_summa_test/`) | GNU General Public License v3.0 or later (`GPL-3.0-or-later`) | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (if copied into repo); **`GREEN`** as currently excluded from `camels-us-mip/models/` | Do not copy `scratch_summa_test/` into `grey-nearing/camels-us-mip` (or keep isolated with `COPYING` / `LICENSE.txt` under `GPL-3.0`). |
| **A16** | **`UW-Hydro/VIC`** (Liang et al., 1994; Newman et al., 2017; Mizukami et al., 2017 `VIC 4.1.2` / `VIC 5`) | `https://github.com/UW-Hydro/VIC` | None in `camels-us-mip/models/` (Literature benchmark hydrographs only) | MIT License (`MIT`) (VIC 5; historical VIC 4.1.2 was `GPL-2.0`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | No code copied into `camels-us-mip/models/`; if ever copied, retain `UW-Hydro/VIC` `MIT` license notice. |
| **A17** | **`ufz/mhm` (`git.ufz.de/mhm/mhm`)** (Samaniego et al., 2010; Mizukami et al., 2019; Rakovec et al., 2019 `mHM`) | `https://git.ufz.de/mhm/mhm` | None in `camels-us-mip/models/` (Literature benchmark hydrographs only) | GNU General Public License v3.0 / Lesser General Public License v3.0 (`GPL-3.0-only` / `LGPL-3.0`) | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (for Fortran source code); **`GREEN`** as currently excluded from `camels-us-mip/models/` | Do not copy `mHM` Fortran source code into `grey-nearing/camels-us-mip`. |
| **A18** | **`naddor/fuse` & `CyrilThebault/FUSE-MMComparison-paper`** (Clark et al., 2008; Addor et al., 2017; Thébault et al., 2026 `FUSE`) | `https://github.com/naddor/fuse`, `https://github.com/CyrilThebault/FUSE-MMComparison-paper` (`10.5281/zenodo.20500212`) | None in `camels-us-mip/models/` (cloned only in `<external_staging_dir>/step2_process_archives/fuse78_thebault/`) | GNU General Public License v3.0 (`GPL-3.0-only`) | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (for Fortran/R code); **`GREEN`** as currently excluded from `camels-us-mip/models/` | Do not copy `naddor/fuse` or `fuse78_thebault/` source code into `grey-nearing/camels-us-mip`. |
| **A19** | **`wknoben/MARRMoT`** (Knoben et al., 2019, 2020 `MARRMoT` 36-model suite) | `https://github.com/wknoben/MARRMoT` (`10.5281/zenodo.2482542`) | None in `camels-us-mip/models/` (Archive* archive hydrographs only) | GNU General Public License v3.0 (`GPL-3.0-only`) | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (for MATLAB code); **`GREEN`** as currently excluded from `camels-us-mip/models/` | Do not copy `wknoben/MARRMoT` MATLAB source code into `grey-nearing/camels-us-mip`. |
| **A20** | **`NCAR/wrf_hydro_nwm_public`** (NOAA National Water Model / `WRF-Hydro`) | `https://github.com/NCAR/wrf_hydro_nwm_public` | None in `camels-us-mip/models/` (Archive* retrospective streamflows extracted from AWS Open Data) | UCAR WRF-Hydro Custom Non-Sale License (`LicenseRef-UCAR-WRF-Hydro`) | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** (for WRF-Hydro source code); **`GREEN`** as currently excluded from `camels-us-mip/models/` | Do not copy `NCAR/wrf_hydro_nwm_public` source code into `grey-nearing/camels-us-mip`. |
| **A21** | **First-Party Study Code (`dhbv_standalone_core`, `era_*`, `metrics/`, `frontend/`, `splits/`)** (This study) | `https://github.com/grey-nearing/camels-us-mip` | `models/feng_dhbv10/dhbv_standalone_core.py`, `models/era_*/`, `metrics/`, `frontend/`, `splits/` | First-party code authored by this study (Grey Nearing / Google LLC); imports `neuralhydrology` (`BSD-3-Clause`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Add a root `LICENSE` file (`Apache-2.0` or `MIT`) to `grey-nearing/camels-us-mip/LICENSE` before making the repository public. |

---

### Part A Detailed Per-Resource Audit

#### A1. `neuralhydrology/neuralhydrology` (`CudaLSTM`, `EA-LSTM`, `MC-LSTM`, `TFT`)
- **Resource / Model Family & Citation:** NeuralHydrology Python library (Kratzert, Gauch, Nearing, & Klotz, 2022, *Journal of Open Source Software*, 7(71), 4050, `10.21105/joss.04050`). Used for our MIP Re-run and Contribution runs of `CudaLSTM`, `EA-LSTM`, `MC-LSTM`, and `TFT`.
- **Upstream URL / DOI:** `https://github.com/neuralhydrology/neuralhydrology` (`https://doi.org/10.21105/joss.04050`, `https://doi.org/10.5281/zenodo.7063259`).
- **Local Path in `camels-us-mip`:** `models/kratzert_lstm/`, `models/kratzert_ealstm/`, `models/hoedt_mclstm/`.
- **Exact License Name / SPDX Identifier:** **BSD 3-Clause "New" or "Revised" License (`BSD-3-Clause`)**, `Copyright (c) 2021, NeuralHydrology`.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Retain the `BSD-3-Clause` license text and copyright notice in `THIRD_PARTY_NOTICES.md`.

#### A2. `kratzert/ealstm_regional_modeling`, `kratzert/lstm_for_pub`, & `kratzert/multiple_forcing`
- **Resource / Model Family & Citation:** Original LSTM, EA-LSTM, and Multi-Forcing LSTM training configurations and implementations (Kratzert et al., 2019a, *WRR*, `10.1029/2019WR026065`; Kratzert et al., 2019b, *HESS*, `10.5194/hess-23-5089-2019`; Kratzert et al., 2021, *HESS*, `10.5194/hess-25-2685-2021`).
- **Upstream URL / DOI:** `https://github.com/kratzert/ealstm_regional_modeling`, `https://github.com/kratzert/lstm_for_pub`, and `https://github.com/kratzert/multiple_forcing`.
- **Local Path in `camels-us-mip`:** `models/category_ii_neuralhydrology/` (target layout: `models/kratzert_lstm/`, `models/kratzert_ealstm/`).
- **Exact License Name / SPDX Identifier:** **Apache License 2.0 (`Apache-2.0`)**, `Copyright 2019 Frederik Kratzert`.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Retain `Apache-2.0` copyright and license notice in `THIRD_PARTY_NOTICES.md`.

#### A3. `ml-jku/mc-lstm` & `jmframe/mclstm_2021_extrapolate` (`MC-LSTM`)
- **Resource / Model Family & Citation:** Mass-Conserving LSTM (`MC-LSTM`) architecture and CAMELS extrapolation experiments (Hoedt et al., 2021, *ICML*, `http://proceedings.mlr.press/v139/hoedt21a.html`; Frame et al., 2022, *HESS*, `10.5194/hess-26-3377-2022`; Frame et al., 2023, *JHM*, `10.1175/JHM-D-22-0083.1`).
- **Upstream URL / DOI:** `https://github.com/ml-jku/mc-lstm` and `https://github.com/jmframe/mclstm_2021_extrapolate` (`https://doi.org/10.5281/zenodo.5165216`).
- **Local Path in `camels-us-mip`:** `models/category_ii_neuralhydrology/` (target layout: `models/hoedt_mclstm/`).
- **Exact License Name / SPDX Identifier:** **MIT License (`MIT`)**, `Copyright (c) 2021 Institute for Machine Learning, JKU Linz` (and `MIT` on `jmframe/mclstm_2021_extrapolate`).
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Retain `MIT` copyright and permission notice in `THIRD_PARTY_NOTICES.md`.

#### A4. `mhpi/hydroDL`, `mhpi/generic_deltamodel`, `mhpi/dhbv2`, `mhpi/dPLHBVrelease`, & `mhpi/dPL` ($\delta\text{HBV 1.0}$, $\delta\text{HBV 1.1p}$, `DI-LSTM`, `dPL+VIC`, `Modified Transformer`)
- **Resource / Model Family & Citation:** Differentiable HBV and hybrid physics-informed models from the Penn State MHPI group (Feng et al., 2020, *WRR*, `10.1029/2019WR026793`; Tsai et al., 2021, *Nature Communications*, `10.1038/s41467-021-26107-z`; Feng et al., 2022, *WRR*, `10.1029/2022WR032404`; Feng et al., 2023, *WRR*, `10.1029/2022WR033893`; Liu et al., 2024, *GMD*, `10.5194/gmd-17-4047-2024`; Song et al., 2024, *PNAS*, `10.1073/pnas.2317429121`; Song et al., 2025, *WRR*, `10.1029/2024WR038928`; Li et al., 2025, *WRR*, `10.1029/2024WR038343`).
- **Upstream URL / DOI:** `https://github.com/mhpi/hydroDL`, `https://github.com/mhpi/generic_deltamodel`, `https://github.com/mhpi/dhbv2`, `https://github.com/mhpi/dPLHBVrelease`, and Zenodo `https://doi.org/10.5281/zenodo.5227738` (Note: `https://github.com/mhpi/dPL` returns HTTP 404 on GitHub; the archived release is on Zenodo at `10.5281/zenodo.5227738`).
- **Local Path in `camels-us-mip`:** `models/feng_dhbv10/dhbv_standalone_core.py` and `models/song_dhbv11p/dhbv_standalone_core.py`. Note that `dPL+VIC` (Tsai et al., 2021), `DI-LSTM` (Feng et al., 2020), `LSTM+dHBV` (Li et al., 2025), and `Modified Transformer` (Liu et al., 2024) are Literature Literature entries with no code in `camels-us-mip/models/`.
- **Exact License Name / SPDX Identifier:**
  - **Upstream `mhpi/*` repositories:** **Pennsylvania State University Non-Commercial Software License Agreement (`LicenseRef-PSU-NonCommercial` / Zenodo `other-nc`)**, `Copyright (c) 2020–2025 The Pennsylvania State University ("PSU")`. Grants rights to reproduce, modify, and prepare derivative works **for non-commercial purposes only**.
  - **Local `models/feng_dhbv10/dhbv_standalone_core.py`:** **First-party clean-room implementation** (`hbv_forward_kernel`, `hbv_backward_kernel`, `StandaloneDHBV`, `CuDNNLSTMBackbone`) written from scratch in CUDA C++ and PyTorch from the published mathematical equations in Feng et al. (2022, 2023) and Song et al. (2025).
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** for upstream `mhpi/*` repositories; **`GREEN — Permissive`** for our clean-room `dhbv_standalone_core.py`.
- **Required Action Before Public Release:** Do **not** copy any source files from `<external_staging_dir>/step3a_dhbv/hydroDL/` or `generic_deltamodel/` into `grey-nearing/camels-us-mip`. Publish only our clean-room CUDA/PyTorch implementation (`dhbv_standalone_core.py`).

#### A5. `sinanrk/TFT-streamflow` (`TFT-Runoff`)
- **Resource / Model Family & Citation:** Temporal Fusion Transformer (`TFT`) for CAMELS-US rainfall-runoff modeling (Rasiya Koya & Roy, 2025, *Environmental Modelling & Software*, 185, 106299, `10.1016/j.envsoft.2024.106299`).
- **Upstream URL / DOI:** `https://github.com/sinanrk/TFT-streamflow` (Note: `leaderboards/table_2_registry.json` lists `https://github.com/sinanrk/TFT-Runoff`, which returns HTTP 404; the author's active public repository is `sinanrk/TFT-streamflow`).
- **Local Path in `camels-us-mip`:** `models/category_ii_ml_and_conceptual/train_tft_and_xgboost_gauged.py` (target layout: `models/rasiya_koya_tft/`), which implements a clean-room PyTorch `TemporalFusionTransformerHydro` module alongside `neuralhydrology`'s `TFT`.
- **Exact License Name / SPDX Identifier:** **MIT License (`MIT`)**, `Copyright (c) 2023 Sinan Rasiya Koya`.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Update `github_url` in `leaderboards/table_2_registry.json` from `https://github.com/sinanrk/TFT-Runoff` to `https://github.com/sinanrk/TFT-streamflow`.

#### A6. `thuml/Time-Series-Library` & Liu et al. (2025) `rainflow` Benchmark (`TSLib` 10-Model Suite)
- **Resource / Model Family & Citation:** Ten deep-learning time-series architectures (`PatchTST`, `iTransformer`, `TimeMixer`, `TimesNet`, `DLinear`, `FEDformer`, `Autoformer`, `Informer`, `Crossformer`, `Mamba`) benchmarked on CAMELS-US (Liu et al., 2025, *HESS*, 29, 6977–7004, `10.5194/hess-29-6977-2025`; built on `thuml/Time-Series-Library` by Wu et al., 2023).
- **Upstream URL / DOI:** `https://github.com/thuml/Time-Series-Library` and Liu et al. (2025) Zenodo code/results archive `https://doi.org/10.5281/zenodo.15852144` (`10.5281/zenodo.17863071`).
- **Local Path in `camels-us-mip`:** Not copied into `camels-us-mip/models/` (Literature published-table-only entry; staged locally in `<external_staging_dir>/step1_published_split1/downloads/liu2025/`).
- **Exact License Name / SPDX Identifier:**
  - `thuml/Time-Series-Library`: **MIT License (`MIT`)**, `Copyright (c) 2021 THUML @ Tsinghua University`.
  - Liu et al. (2025) `HESS_paper.zip` (`rainflow` wrapper on Zenodo `10.5281/zenodo.15852144`): **Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC-4.0`)**.
- **Redistribution Status:** **`GREEN — Permissive (`MIT`)`** for `thuml/Time-Series-Library`; **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`CC-BY-NC-4.0`)`** for Liu et al. (2025)'s `rainflow` wrapper scripts.
- **Required Action Before Public Release:** Keep `downloads/liu2025/` out of `grey-nearing/camels-us-mip` (no code or hydrographs from Liu et al. 2025 are redistributed; only published summary metrics from Table 2 of their HESS paper are cited).

#### A7. `state-spaces/s4` & `WESTENR-OU/S4D_rainfall_runoff_simulations` (`S4D` & `S4D-FT`)
- **Resource / Model Family & Citation:** Diagonal State Space (`S4D`) and Fourier-Transform Diagonal State Space (`S4D-FT` / `HOPE`) models for rainfall-runoff simulation (Gu et al., 2022, *NeurIPS*; Wang et al., 2025, *Water Resources Research*, 61, e2025WR039888, `10.1029/2025WR039888`).
- **Upstream URL / DOI:** `https://github.com/state-spaces/s4` and `https://github.com/WESTENR-OU/S4D_rainfall_runoff_simulations` (`https://doi.org/10.1029/2025WR039888`).
- **Local Path in `camels-us-mip`:** `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_s4dft.py` and `driver_s4dft.py` (target layout: `models/wang_s4dft/`).
- **Exact License Name / SPDX Identifier:**
  - `state-spaces/s4`: **Apache License 2.0 (`Apache-2.0`)**.
  - `WESTENR-OU/S4D_rainfall_runoff_simulations`: **No explicit license file in upstream repo — All Rights Reserved by default under copyright law** (note: its `README.md` states the repository is adapted from `kratzert/lstm_for_pub` [`Apache-2.0`] and bundles `state-spaces/s4` [`Apache-2.0`]).
- **Redistribution Status:** **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`** (for `WESTENR-OU/S4D_rainfall_runoff_simulations`); **`GREEN — Permissive (`Apache-2.0`)`** for `state-spaces/s4`.
- **Required Action Before Public Release:** `model_s4dft.py` currently ports the `HOPE` wrapper from `WESTENR-OU/S4D_rainfall_runoff_simulations` alongside `S4DKernel` from `state-spaces/s4`. Before making `grey-nearing/camels-us-mip` public, either:
  1. Ask the authors (`WESTENR-OU` / Y. Wang et al.) to add an open-source `LICENSE` file (e.g., `Apache-2.0` matching `state-spaces/s4` and `kratzert/lstm_for_pub`) or provide written permission, **or**
  2. Refactor `model_s4dft.py` so it uses only `Apache-2.0` `state-spaces/s4` layers inside our own first-party `camels-us-mip` model wrapper.

#### A8. `kshitijtayal/Geo-ViT-LSTM` (`Geo-ViT-LSTM`)
- **Resource / Model Family & Citation:** Vision Transformer + LSTM (`Geo-ViT-LSTM`) for regional hydrology (Tayal et al., 2024, *ACM SIGSPATIAL*, `10.1145/3678717.3691324`).
- **Upstream URL / DOI:** Listed as `https://github.com/kshitijtayal/Geo-ViT-LSTM` in `leaderboards/table_2_registry.json` (returns HTTP 404 — no public repository was released by the authors).
- **Local Path in `camels-us-mip`:** None (Literature published-table-only entry; 0 lines of code and 0 hydrographs copied).
- **Exact License Name / SPDX Identifier:** **No public repository released (Not Applicable — no code or data copied)**.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** (for our repository release, since only factual summary numbers from the published paper are cited).
- **Required Action Before Public Release:** Remove or annotate the dead `https://github.com/kshitijtayal/Geo-ViT-LSTM` URL in `leaderboards/table_2_registry.json`.

#### A9. `gauchm/ealstm_regional_modeling` (`proper-care-and-feeding`) & `dmlc/xgboost` (`XGBoost`)
- **Resource / Model Family & Citation:** Gradient-boosted decision trees (`XGBoost`) calibrated for CAMELS-US streamflow prediction (Gauch, Mai, & Lin, 2021, *Environmental Modelling & Software*, 135, 104926, *"The Proper Care and Feeding of CAMELS: How Limited Training Data Affects Streamflow Prediction"*, `10.1016/j.envsoft.2020.104926`).
- **Upstream URL / DOI:** `https://github.com/gauchm/ealstm_regional_modeling` (`https://doi.org/10.5281/zenodo.3648270`) and `https://github.com/dmlc/xgboost` (Note: `leaderboards/table_2_registry.json` lists a dead slug `https://github.com/martin-gauch/proper-care-and-feeding`).
- **Local Path in `camels-us-mip`:** `models/category_ii_ml_and_conceptual/train_tft_and_xgboost_gauged.py` (target layout: `models/gauch_xgboost/`), which is our own clean Python script calling the `xgboost` library.
- **Exact License Name / SPDX Identifier:** **Apache License 2.0 (`Apache-2.0`)** on both `gauchm/ealstm_regional_modeling` and `dmlc/xgboost`.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Update `github_url` in `leaderboards/table_2_registry.json` from `https://github.com/martin-gauch/proper-care-and-feeding` to `https://github.com/gauchm/ealstm_regional_modeling`.

#### A10. `google-research/google-research` (`tide`) (`TiDE`)
- **Resource / Model Family & Citation:** Time-series Dense Encoder (`TiDE`) architecture (Das et al., 2023, *TMLR*, `arXiv:2304.08424`), evaluated in pure meteorological simulation mode (`tide_sim`).
- **Upstream URL / DOI:** `https://github.com/google-research/google-research/tree/master/tide`.
- **Local Path in `camels-us-mip`:** `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_tide.py` and `driver_tide.py` (target layout: `models/das_tide/`).
- **Exact License Name / SPDX Identifier:** **Apache License 2.0 (`Apache-2.0`)**, `Copyright 2023 The Google Research Authors`.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Retain `Apache-2.0` attribution and citation to Das et al. (2023).

#### A11. `snpoudel/diff-hydro` (`poudel2026_mlp_hbv` / `MLP + 16HBV`)
- **Resource / Model Family & Citation:** Static-attribute Multi-Layer Perceptron (`MLP`) parameterizing 16 parallel differentiable HBV sub-basins (`MLP + 16HBV`; Poudel & Steinschneider, 2026, *HESS*, 30, 5173–5194, `10.5194/hess-30-5173-2026`).
- **Upstream URL / DOI:** `https://github.com/snpoudel/diff-hydro` (Paper DOI: `https://doi.org/10.5194/hess-30-5173-2026`).
- **Local Path in `camels-us-mip`:** `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_hbv_poudel.py` and `driver_mlp16hbv.py` (target layout: `models/poudel_mlp_hbv/`).
- **Exact License Name / SPDX Identifier:** **No explicit license file in upstream repo (`snpoudel/diff-hydro`) — All Rights Reserved by default under copyright law** (note: the published HESS 2026 paper and equations are licensed under `CC-BY-4.0`).
- **Redistribution Status:** **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`**
- **Required Action Before Public Release:** `model_hbv_poudel.py` is currently a TorchScript port of `snpoudel/diff-hydro/models/multi_hbv.py`. Before making `grey-nearing/camels-us-mip` public, either:
  1. Contact Shashank Poudel & Scott Steinschneider to add a permissive open-source `LICENSE` file to `snpoudel/diff-hydro` (or provide written permission), **or**
  2. Replace `model_hbv_poudel.py` with a clean-room implementation connecting a standard PyTorch `nn.Sequential` static-attribute MLP to our first-party `StandaloneDHBV` CUDA HBV routing kernel (`dhbv_standalone_core.py`).

#### A12. `eduardoAcunaEspinoza/Hy2DL` & `KIT-HYD/Hy2DL` (`hy2dl_hbv`)
- **Resource / Model Family & Citation:** Hybrid hydrological deep learning (`Hy2DL` `LSTM + HBV`, Acuña Espinoza et al., 2024, *HESS*, 28, 2705–2719, `10.5194/hess-28-2705-2024`).
- **Upstream URL / DOI:** Active canonical repository `https://github.com/eduardoAcunaEspinoza/Hy2DL` (`https://doi.org/10.5281/zenodo.22083944`) and deprecated v1.0 repository `https://github.com/KIT-HYD/Hy2DL` (`https://doi.org/10.5281/zenodo.17251944`).
- **Local Path in `camels-us-mip`:** `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/model_hy2dl_hbv.py` and `driver_hy2dl.py` (target layout: `models/hy2dl_hbv/`).
- **Exact License Name / SPDX Identifier:**
  - Active canonical repository (`eduardoAcunaEspinoza/Hy2DL` v2.0+): **BSD 3-Clause "New" or "Revised" License (`BSD-3-Clause`)**, `Copyright (c) 2024, Eduardo Acuña Espinoza`.
  - Deprecated v1.0 repository (`KIT-HYD/Hy2DL`): **GNU General Public License v3.0 (`GPL-3.0-only`)**.
- **Redistribution Status:** **`GREEN — Permissive (`BSD-3-Clause`)`** under the original author's active canonical repository `eduardoAcunaEspinoza/Hy2DL`.
- **Required Action Before Public Release:** Update the upstream reference in `model_hy2dl_hbv.py` and `leaderboards/table_2_registry.json` from `KIT-HYD/Hy2DL` to `https://github.com/eduardoAcunaEspinoza/Hy2DL` (`10.5281/zenodo.22083944`) and include the `BSD-3-Clause` copyright notice of Eduardo Acuña Espinoza in `THIRD_PARTY_NOTICES.md`.

#### A13. `yhwang08/HydroDiffusion` (`hydrodiffusion`)
- **Resource / Model Family & Citation:** Score-based generative diffusion model with State-Space backbone (`HydroDiffusion`; Wang, Yu, Zhang, & Erichson, 2025, `arXiv:2512.12183`).
- **Upstream URL / DOI:** `https://github.com/yhwang08/HydroDiffusion` (`https://arxiv.org/abs/2512.12183`).
- **Local Path in `camels-us-mip`:** `models/category_ii_tier2_s4d_hy2dl_tide_diffusion/driver_hydrodiffusion_train.py` and `driver_hydrodiffusion_eval.py` (target layout: `models/hydrodiffusion/`). The upstream `yhwang08/HydroDiffusion` repository is cloned only in `<external_staging_dir>/step4_tier2/repos/HydroDiffusion/` and is **not** copied into `camels-us-mip/models/`.
- **Exact License Name / SPDX Identifier:** **No explicit license file in upstream repo (`yhwang08/HydroDiffusion`) — All Rights Reserved by default under copyright law** (note: `papercode/src/` inside `yhwang08/HydroDiffusion` is copied from `state-spaces/s4` under `Apache-2.0`).
- **Redistribution Status:** **`RED — Missing License / Needs Author Permission or Clean-Room Re-implementation Before Public Release`** (for upstream `yhwang08/HydroDiffusion` code); **`GREEN`** for our own `driver_hydrodiffusion_*.py` scripts.
- **Required Action Before Public Release:** Do **not** vendor `yhwang08/HydroDiffusion/papercode/` into `grey-nearing/camels-us-mip` unless the authors add an open-source license or grant written permission. Either keep only our clean driver scripts (which clone or import the upstream repo at runtime) or write a clean-room DDIM diffusion head on top of `Apache-2.0` `state-spaces/s4`.

#### A14. `sacsma_upstream` (`Upstream-Tech/SACSMA-SNOW17` / NOAA NWSRFS / NCAR `newman_sacsma`)
- **Resource / Model Family & Citation:** Sacramento Soil Moisture Accounting (`SAC-SMA`) coupled with `Snow-17` and unit-hydrograph routing (Newman et al., 2015, *HESS*, `10.5194/hess-19-209-2015`; Newman et al., 2017, *JHM*, `10.1175/JHM-D-16-0284.1`; Python/f2py wrapper by Ben-Shalom et al., 2022, `Upstream-Tech/SACSMA-SNOW17`).
- **Upstream URL / DOI:** `https://github.com/Upstream-Tech/SACSMA-SNOW17`.
- **Local Path in `camels-us-mip`:** `models/category_ii_sacsma/optimize_single_basin.py` and `run_pipeline_updated.py` (target layout: `models/newman_sacsma/`; full upstream clone in `<external_staging_dir>/sacsma_upstream/`).
- **Exact License Name / SPDX Identifier:**
  1. **Core NOAA NWS Hydrologic Research Laboratory Fortran subroutines** (`sac1.f`, `ex_sac1.f`, `snow19/sn19.f`, `snow19/aesc19.f`, `snow19/melt19.f`, `snow19/rout19.f`, `snow19/updt19.f`, `snow19/zero19.f`, `duamel.f` by Eric Anderson & Victor Koren, NOAA/NWS): **U.S. Government Public Domain (`17 U.S.C. § 105`)**.
  2. **NCAR lumped Fortran driver wrapper** (`sacsma_source/driver/multi_driver.f`, `sacsma_source/LICENSE` by Andy Newman & Pablo Mendoza, NCAR): **GNU General Public License v3.0 (`GPL-3.0-only`)**.
  3. **`Upstream-Tech/SACSMA-SNOW17` Python/f2py wrapper** (`model/`, `optimizer/`, `optimize_single_basin.py`): **No root `LICENSE` file in `Upstream-Tech/SACSMA-SNOW17`** (bundles `sacsma_source/LICENSE` = `GPL-3.0-only`).
- **Redistribution Status:** **`YELLOW — Copyleft (`GPL-3.0-only`)`** for `sacsma_source/`; **`RED — Missing License`** for the `Upstream-Tech` Python wrapper (`optimize_single_basin.py`); **`GREEN — Public Domain`** for the core NOAA NWSRFS Fortran physics routines.
- **Required Action Before Public Release:** Replace `Upstream-Tech`'s `optimize_single_basin.py` wrapper with our own clean-room Python `f2py` + Dynamically Dimensioned Search (DDS) calibration driver calling the public-domain NOAA NWSRFS Fortran subroutines (`sac1.f`, `snow19/*.f`, `duamel.f`) directly. That eliminates both the `GPL-3.0` NCAR wrapper and the unlicensed `Upstream-Tech` wrapper, making `models/newman_sacsma/` 100% **`GREEN`**.

#### A15. `NCAR/summa` (`SUMMA`)
- **Resource / Model Family & Citation:** Structure for Unifying Multiple Modeling Alternatives (`SUMMA`; Clark et al., 2015, *WRR*, `10.1002/2015WR017198`; Farahani et al., 2025, *HESS*, `10.5194/hess-29-4515-2025`).
- **Upstream URL / DOI:** `https://github.com/NCAR/summa`.
- **Local Path in `camels-us-mip`:** None in `camels-us-mip/models/` (cloned only in `<external_staging_dir>/scratch_summa_test/`; Farahani et al. 2025 is evaluated as a Archive* archive entry).
- **Exact License Name / SPDX Identifier:** **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)**, `Copyright (C) 2014–2017 NCAR/RAL` (`COPYING`, `LICENSE.txt`, `header.license`).
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`GPL-3.0-or-later`)`** (if source code is copied); **`GREEN`** as currently excluded from `camels-us-mip/models/`.
- **Required Action Before Public Release:** Do not copy `scratch_summa_test/` into `grey-nearing/camels-us-mip`.

#### A16. `UW-Hydro/VIC` (`VIC 4.1.2` / `VIC 5`)
- **Resource / Model Family & Citation:** Variable Infiltration Capacity (`VIC`) macroscale hydrologic model (Liang et al., 1994, *JGR*; Newman et al., 2017, *JHM*, `10.1175/JHM-D-16-0284.1`; Mizukami et al., 2017, *WRR*, `10.1002/2017WR020401`).
- **Upstream URL / DOI:** `https://github.com/UW-Hydro/VIC`.
- **Local Path in `camels-us-mip`:** None in `camels-us-mip/models/` (Literature benchmark hydrographs ingested from NCAR GDEX and Kratzert et al., 2019b HydroShare).
- **Exact License Name / SPDX Identifier:** **MIT License (`MIT`)**, `Copyright (c) 2016, UW Hydro, Computational Hydrology group, University of Washington` (Note: historical VIC 4.1.2 releases prior to VIC 5 were `GPL-2.0`).
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** None (no VIC source code is copied into `camels-us-mip/models/`).

#### A17. `ufz/mhm` (`git.ufz.de/mhm/mhm`) (`mHM`)
- **Resource / Model Family & Citation:** Mesoscale Hydrologic Model (`mHM`; Samaniego et al., 2010, *WRR*, `10.1029/2008WR007327`; Mizukami et al., 2019, *WRR*, `10.1029/2019WR025185`; Rakovec et al., 2019, *JHM*, `10.1175/JHM-D-19-0024.1`).
- **Upstream URL / DOI:** `https://git.ufz.de/mhm/mhm`.
- **Local Path in `camels-us-mip`:** None in `camels-us-mip/models/` (Literature benchmark hydrographs ingested from Kratzert et al., 2019b HydroShare).
- **Exact License Name / SPDX Identifier:** **GNU General Public License v3.0 / GNU Lesser General Public License v3.0 (`GPL-3.0-only` / `LGPL-3.0`)**, `Copyright (C) 2005–2024 Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ`.
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`GPL-3.0-only` / `LGPL-3.0`)`** (for `mHM` Fortran source code); **`GREEN`** as currently excluded from `camels-us-mip/models/`.
- **Required Action Before Public Release:** Do not copy `mHM` Fortran source code into `grey-nearing/camels-us-mip`.

#### A18. `naddor/fuse` & `CyrilThebault/FUSE-MMComparison-paper` (`FUSE`)
- **Resource / Model Family & Citation:** Framework for Understanding Structural Errors (`FUSE`; Clark et al., 2008, *WRR*, `10.1029/2007WR006735`; Addor et al., 2017, *HESS*, `10.5194/hess-21-5293-2017`; Thébault et al., 2026, *HESS*, `10.5194/hess-30-663-2026`).
- **Upstream URL / DOI:** `https://github.com/naddor/fuse` and `https://github.com/CyrilThebault/FUSE-MMComparison-paper` (`https://doi.org/10.5281/zenodo.20500212`).
- **Local Path in `camels-us-mip`:** None in `camels-us-mip/models/` (cloned only in `<external_staging_dir>/step2_process_archives/fuse78_thebault/`).
- **Exact License Name / SPDX Identifier:** **GNU General Public License v3.0 (`GPL-3.0-only`)** on both `naddor/fuse` and `CyrilThebault/FUSE-MMComparison-paper`.
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`GPL-3.0-only`)`** (for Fortran/R code); **`GREEN`** as currently excluded from `camels-us-mip/models/`.
- **Required Action Before Public Release:** Keep `naddor/fuse` and `fuse78_thebault/` source code out of `grey-nearing/camels-us-mip`.

#### A19. `wknoben/MARRMoT` (`MARRMoT` 36-Model Suite)
- **Resource / Model Family & Citation:** Modular Assessment of Rainfall-Runoff Models Toolbox (`MARRMoT`; Knoben et al., 2019, *GMD*, `10.5194/gmd-12-2463-2019`; Knoben et al., 2020, *WRR*, `10.1029/2019WR025975`).
- **Upstream URL / DOI:** `https://github.com/wknoben/MARRMoT` (`https://doi.org/10.5281/zenodo.2482542`).
- **Local Path in `camels-us-mip`:** None in `camels-us-mip/models/` (Archive* simulated hydrographs extracted from the University of Bristol archive).
- **Exact License Name / SPDX Identifier:** **GNU General Public License v3.0 (`GPL-3.0-only`)**.
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`GPL-3.0-only`)`** (for MATLAB source code); **`GREEN`** as currently excluded from `camels-us-mip/models/`.
- **Required Action Before Public Release:** Do not copy `MARRMoT` MATLAB source code into `grey-nearing/camels-us-mip`.

#### A20. `NCAR/wrf_hydro_nwm_public` (NOAA National Water Model / `WRF-Hydro`)
- **Resource / Model Family & Citation:** NOAA National Water Model (`NWM v2.0`, `v2.1`, `v3.0`) based on the NCAR `WRF-Hydro` modeling system (Gochis et al., 2020; Frame et al., 2021, *JAWRA*, `10.1111/1752-1688.12964`; Towler et al., 2023, *JHM*, `10.1175/JHM-D-22-0125.1`).
- **Upstream URL / DOI:** `https://github.com/NCAR/wrf_hydro_nwm_public`.
- **Local Path in `camels-us-mip`:** None in `camels-us-mip/models/` (Archive* retrospective streamflows extracted from AWS Open Data Registry).
- **Exact License Name / SPDX Identifier:** **UCAR WRF-Hydro Custom Non-Sale License (`LicenseRef-UCAR-WRF-Hydro`)** (`LICENSE.txt`: grants a non-exclusive, royalty-free license to use, publish, distribute, modify, and copy WRF-Hydro, subject to the condition: *"You shall not sell, license or transfer for a fee the Software, or any work that in any manner contains the Software"*).
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`LicenseRef-UCAR-WRF-Hydro`)`** (if WRF-Hydro source code is copied); **`GREEN`** as currently excluded from `camels-us-mip/models/`.
- **Required Action Before Public Release:** Do not copy `NCAR/wrf_hydro_nwm_public` source code into `grey-nearing/camels-us-mip`.

#### A21. First-Party Study Code (`dhbv_standalone_core`, `era_*`, `metrics/`, `frontend/`, `splits/`)
- **Resource / Model Family & Citation:** First-party models, evaluation harnesses, splits, and interactive web dashboard developed for this study:
  - MIP Re-run Clean-Room Differentiable HBV CUDA/PyTorch Core (`models/feng_dhbv10/dhbv_standalone_core.py`, `models/song_dhbv11p/dhbv_standalone_core.py`).
  - Contribution ERA Winning Architectures (`models/era_*/`).
  - Core MIP Evaluation Harness, Metrics, Splits, and Dashboard (`metrics/`, `splits/`, `frontend/`).
- **Upstream URL / DOI:** `https://github.com/grey-nearing/camels-us-mip`.
- **Local Path in `camels-us-mip`:** `models/feng_dhbv10/dhbv_standalone_core.py`, `models/era_*/`, `metrics/`, `splits/`, `frontend/`.
- **Exact License Name / SPDX Identifier:** **First-Party Original Work (Grey Nearing / Google LLC)**, intended for release under **Apache License 2.0 (`Apache-2.0`)** or **MIT License (`MIT`)** (currently no root `LICENSE` file is committed in ``).
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Add a root `LICENSE` file (`Apache-2.0` or `MIT`) to `LICENSE`.

---

## Part B — Hydrograph Data Repositories & External Archives (`gs://camels-us-mip/`, 10 Items)

### Part B Summary Table

| # | Hydrograph Archive / Dataset & Key Citations | Upstream URL / Canonical DOI | Local / GCS Path in `gs://camels-us-mip/` | Exact License Name & SPDX Identifier | Redistribution Status | Required Action Before Public Release |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | **NCAR/UCAR GDEX CAMELS-US v1.2 / v2.0 Observations, Forcings, Attributes, & Newman (2015, 2017) SAC-SMA / VIC Benchmark Hydrographs** | `https://doi.org/10.5065/D6MW2F4D`, `https://doi.org/10.5065/D6G73C3Q`, `https://doi.org/10.5281/zenodo.15529996` (`https://gdex.ucar.edu/dataset/camels.html`) | `gs://camels-us-mip/observations/newman_eval_targets.zarr/`, `gs://camels-us-mip/splits/`, `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`protocol/newman_eval_targets.npz`) | Creative Commons Attribution 4.0 International (`CC-BY-4.0`) & UCAR Terms of Use; USGS observed discharge is **U.S. Government Public Domain (`17 U.S.C. § 105`)** | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain `CC-BY-4.0` attribution to Newman et al. (2015, 2017) and Addor et al. (2017) (`10.5065/D6MW2F4D`, `10.5065/D6G73C3Q`). |
| **B2** | **Kratzert et al. (2019a, 2019b, 2021) HydroShare & Zenodo Benchmark & LSTM/EA-LSTM Hydrograph Archives** (`SAC-SMA`, `VIC`, `FUSE`, `HBV`, `mHM`, `LSTM`, `EA-LSTM`, `Multi-Forcing LSTM`) | HydroShare `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1`, `10.4211/hs.83ea5312635e44dc824eeb99eda12f06`, `10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c`, `10.4211/hs.17c896843cf940339c3c3496d0c1c077`; Zenodo `10.5281/zenodo.4670268` | `gs://camels-us-mip/hydrographs/{gauged,pub,pur}__{single,multi}_camels/` (`step1_published_split1/hydrographs_kratzert2019b_*.npz`, `hydrographs_kratzert2021_*.npz`) | Creative Commons Attribution 4.0 International (`CC-BY-4.0`) across all HydroShare and Zenodo archives | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain `CC-BY-4.0` attribution to Kratzert et al. (2019a, 2019b, 2021) and benchmark model authors; update `table_2_registry.json` DOI for Kratzert et al. (2021) to `10.5281/zenodo.4670268`. |
| **B3** | **MHPI ($\delta\text{HBV 1.0}$, $\delta\text{HBV 1.1p}$, Li et al. 2025 `LSTM+dHBV`, Feng et al. 2020 `DI-LSTM`, Tsai et al. 2021 `dPL+VIC`) Zenodo & S3 Archives** | Zenodo `10.5281/zenodo.16895228` (Li et al. 2025), `10.5281/zenodo.7091334` / `7943626` (Feng et al. 2022), `10.5281/zenodo.3993880` (Feng et al. 2020), `10.5281/zenodo.5227738` (Tsai et al. 2021) | `gs://camels-us-mip/hydrographs/{gauged,pub,pur}__{single,multi}_camels/` (first-party `StandaloneDHBV` runs in `hydrographs/category_ii/t3_*.npz`, `t4_*.npz`; Li et al. 2025 in `step3a_dhbv/li2025_ensemble_hydrographs_split3_test.npz`) | Zenodo `16895228` (Li 2025), `7091334` (Feng 2022), `3993880` (Feng 2020): **`CC-BY-4.0`**. Zenodo `5227738` (Tsai 2021) & MHPI S3 checkpoints: **PSU Non-Commercial (`other-nc`)**. | **`GREEN — Permissive (`CC-BY-4.0` / First-Party Runs)`** for all hydrographs in `gs://camels-us-mip/`; **`YELLOW — Non-Commercial`** for `10.5281/zenodo.5227738` & MHPI S3 | Fix DOIs in `table_2_registry.json` (`7091372` $\rightarrow$ `16895228` / `7091334`; `5165216` $\rightarrow$ `5227738`); retain `CC-BY-4.0` attribution for Li et al. (2025) and Feng et al. (2020, 2022); do not redistribute raw Tsai et al. (2021) archive files. |
| **B4** | **Nearing et al. (2022) AR & DA LSTM Zenodo Archive** | `https://doi.org/10.5281/zenodo.7063252` & `https://doi.org/10.5281/zenodo.7063259` (Paper DOI: `10.5194/hess-26-5493-2022`) | `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/` | `10.5281/zenodo.7063259`: `BSD-3-Clause`; HESS paper: `CC-BY-4.0`; `10.5281/zenodo.7063252`: `other-open` (first-party work by lead author Grey Nearing) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Fix DOI typo in `table_2_registry.json` from `10.5281/zenodo.7152656` to `10.5281/zenodo.7063252`. |
| **B5** | **Hoedt et al. (2021) & Frame et al. (2022, 2023) `MC-LSTM` Zenodo & HydroShare Archives** | Hoedt et al. (2021) Zenodo: `https://doi.org/10.5281/zenodo.8138359`; Frame et al. (2022) Zenodo: `https://doi.org/10.5281/zenodo.5165216`; Frame et al. (2022, 2023) HydroShare: `https://doi.org/10.4211/hs.d750278db868447dbd252a8c5431affd` | `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`step1_published_split1/hydrographs_frame_mclstm_archive.npz` and our own NeuralHydrology `MC-LSTM` re-runs) | Hoedt et al. (2021) (`8138359`): **`CC-BY-4.0`**. Frame et al. (2022) (`5165216`): **`MIT`**. Frame et al. (2022, 2023) HydroShare (`d750278db868447dbd252a8c5431affd`): **`CC-BY-NC-4.0`**. | **`GREEN — Permissive`** (Hoedt et al. 2021 & our `MC-LSTM` re-runs); **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`CC-BY-NC-4.0`)`** (Frame et al. HydroShare archive) | Fix DOI in `table_2_registry.json` from `10.5281/zenodo.5040168` to `10.5281/zenodo.8138359`. Label `frame22_*` / `frame23_*` Zarr stores as `CC-BY-NC-4.0`, or have co-authors Jonathan Frame / Grey Nearing update HydroShare `d750278db868447dbd252a8c5431affd` to `CC-BY-4.0`. |
| **B6** | **Gauch et al. (2021) `XGBoost` & `MTS-LSTM` Zenodo Archives** | `https://doi.org/10.5281/zenodo.3648270` (`XGBoost`), `https://doi.org/10.5281/zenodo.4071885` (`4071886`) & `https://doi.org/10.5281/zenodo.4072700` (`4072701`) (`MTS-LSTM`) | `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`hydrographs/category_ii/t3_53.npz` & `step3b_neuralhydrology/archives/gauch2021/`) | Creative Commons Attribution 4.0 International (`CC-BY-4.0`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Fix DOI in `table_2_registry.json` from `10.5281/zenodo.4539986` to `10.5281/zenodo.3648270` and retain `CC-BY-4.0` attribution. |
| **B7** | **Thébault et al. (2026) / Addor et al. (2017) `FUSE` 78-Model & 3-Model Archives** | HydroShare `https://doi.org/10.4211/hs.53063da8fc894149b5b25f3201d1e8e8` (`Eval_FUSE.Rdata`), HydroShare `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (`FUSE 900/902/904`), Zenodo `https://doi.org/10.5281/zenodo.20500212` | `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`FUSE 900/902/904` daily hydrographs; `fuse78_per_basin_metrics.npz` summary metrics) | HydroShare data archives (`53063da8...` & `474ecc37...`): **`CC-BY-4.0`**. Zenodo code archive (`20500212`): **`GPL-3.0-only`**. | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** (for all `FUSE` hydrographs and evaluation metric data) | Fix DOI in `table_2_registry.json` from `10.5281/zenodo.14019252` to `10.4211/hs.53063da8fc894149b5b25f3201d1e8e8` (`10.5281/zenodo.20500212`) and retain `CC-BY-4.0` attribution. |
| **B8** | **Knoben et al. (2020) `MARRMoT` 36-Model Hydrograph Archive** | University of Bristol `https://doi.org/10.5523/bris.2zutxh2qeep6y2cy6scwgk9eqj` (`https://data.bris.ac.uk/datasets/2zutxh2qeep6y2cy6scwgk9eqj/`) & Zenodo `https://doi.org/10.5281/zenodo.2482542` | `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`step2_process_archives/marrmot_bristol/simq_531/m_*_simq531.npz`) | GNU General Public License v3.0 (`GPL-3.0-only`) (explicitly stated in dataset `readme.txt`) | **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`** | Fix DOI in `table_2_registry.json` from `10.5281/zenodo.3746544` to `10.5523/bris.2zutxh2qeep6y2cy6scwgk9eqj`. Include the `GPL-3.0` notice and attribution to Knoben et al. (2020) on the `marrmot_*` Zarr stores in `gs://camels-us-mip/`. |
| **B9** | **Farahani et al. (2025) `SUMMA-mizuRoute` LSE Zenodo Archive** | `https://doi.org/10.5281/zenodo.16422768` (Paper DOI: `10.5194/hess-29-4515-2025`) | `gs://camels-us-mip/hydrographs/gauged__freeform/` (published-summary-only entry; no daily hydrographs in upstream Zenodo archive) | Creative Commons Attribution 4.0 International (`CC-BY-4.0`) | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Fix DOI in `table_2_registry.json` from `10.5281/zenodo.13892055` to `10.5281/zenodo.16422768` and retain `CC-BY-4.0` attribution. |
| **B10** | **NOAA National Water Model (`NWM v2.0`, `v2.1`, `v3.0`) Retrospective Archive on AWS Open Data Registry** | `https://registry.opendata.aws/nwm-archive/` (`s3://noaa-nwm-retrospective-3-0-pds`, `s3://noaa-nwm-retrospective-2-1-zarr-pds`, `s3://noaa-nwm-retro-v2-zarr-pds`) | `gs://camels-us-mip/hydrographs/{gauged,pub,pur}__freeform/`, `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`step2_process_archives/hydrographs_nwm_*.npz`, `nwm_postprocessed_hydrographs.npz`) | **U.S. Government Public Domain (`17 U.S.C. § 105`)** / AWS Open Data Registry ("No restrictions on the use of the data") | **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`** | Retain citation to NOAA Office of Water Prediction, Frame et al. (2021), and Towler et al. (2023). |

---

### Part B Detailed Per-Resource Audit

#### B1. NCAR/UCAR GDEX CAMELS-US v1.2 & v2.0 Observations, Forcings, Attributes, & Newman (2015, 2017) SAC-SMA / VIC Benchmark Hydrographs
- **Resource & Citation:** CAMELS-US hydrometeorological time series, catchment attributes, and calibrated `SAC-SMA` + `Snow-17` and `VIC` baseline simulations (Newman et al., 2015, *HESS*, `10.5194/hess-19-209-2015`; Newman et al., 2017, *JHM*, `10.1175/JHM-D-16-0284.1`; Addor et al., 2017, *HESS*, `10.5194/hess-21-5293-2017`).
- **Upstream URL / DOI:** `https://gdex.ucar.edu/dataset/camels.html`, `https://doi.org/10.5065/D6MW2F4D` (Newman et al., 2015), `https://doi.org/10.5065/D6G73C3Q` (Addor et al., 2017), and Zenodo mirror `https://doi.org/10.5281/zenodo.15529996`.
- **Local / GCS Path:** `gs://camels-us-mip/observations/newman_eval_targets.zarr/`, `gs://camels-us-mip/splits/`, `gs://camels-us-mip/hydrographs/gauged__single_camels/`, and `protocol/newman_eval_targets.npz`.
- **Exact License Name / SPDX Identifier:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** and **UCAR Terms of Use** (`https://www.ucar.edu/terms-of-use`). Underlying USGS NWIS observed streamflow gauges are a work of the U.S. Federal Government in the **Public Domain (`17 U.S.C. § 105`)**.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Include full citations and DOIs (`10.5065/D6MW2F4D`, `10.5065/D6G73C3Q`) in `gs://camels-us-mip/README.md` and Zenodo metadata.

#### B2. Kratzert et al. (2019a, 2019b, 2021) HydroShare & Zenodo Benchmark & LSTM/EA-LSTM Hydrograph Archives
- **Resource & Citation:**
  1. **CAMELS Hydrology Benchmark Models Archive:** Simulated daily hydrographs for `SAC-SMA` (Newman et al., 2015), `VIC` basin & regional (Newman et al., 2017; Mizukami et al., 2017), `FUSE 900/902/904` (Addor et al., 2017), `HBV` lower/upper/ensemble (Seibert et al., 2018), and `mHM` basin & CONUS (Mizukami et al., 2019; Rakovec et al., 2019), curated by Kratzert et al. (2019b).
  2. **Kratzert et al. (2019a, 2019b) `LSTM` & `EA-LSTM` Archive:** Single-basin and regional `LSTM` and `EA-LSTM` ensemble and per-seed hydrographs.
  3. **Kratzert et al. (2021) Multi-Forcing `LSTM` Archive:** Single-, dual-, and triple-forcing (`Daymet` + `Maurer` + `NLDAS`) `LSTM` hydrographs.
- **Upstream URL / DOI:**
  - HydroShare Benchmark Models: `https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/` (`10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1`)
  - HydroShare LSTM/EA-LSTM Models: `https://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06/` (`10.4211/hs.83ea5312635e44dc824eeb99eda12f06`)
  - HydroShare Extended NLDAS & Maurer Forcings: `10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c` and `10.4211/hs.17c896843cf940339c3c3496d0c1c077`
  - Zenodo Multi-Forcing LSTM (Kratzert et al., 2021): `https://doi.org/10.5281/zenodo.4670268` (listed as `10.5281/zenodo.4071885` in `leaderboards/table_2_registry.json`, which is Gauch, Kratzert et al., 2021 `MTS-LSTM`).
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__single_camels/`, `gs://camels-us-mip/hydrographs/gauged__multi_camels/`, `gs://camels-us-mip/hydrographs/pub__single_camels/`, `gs://camels-us-mip/hydrographs/pur__single_camels/` (`<external_staging_dir>/step1_published_split1/hydrographs_kratzert2019b_*.npz`, `hydrographs_kratzert2021_lstm_per_seed.npz`).
- **Exact License Name / SPDX Identifier:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** on every HydroShare resource and Zenodo record.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Retain `CC-BY-4.0` citations to Kratzert et al. (2019a, 2019b, 2021) and all contributing process-model authors, and update `table_2_registry.json` to cite `10.5281/zenodo.4670268` for Kratzert et al. (2021).

#### B3. MHPI ($\delta\text{HBV 1.0}$, $\delta\text{HBV 1.1p}$, Li et al. 2025 `LSTM+dHBV`, Feng et al. 2020 `DI-LSTM`, Tsai et al. 2021 `dPL+VIC`) Zenodo & S3 Archives
- **Resource & Citation:** Published model outputs and archives from the Penn State MHPI group (Feng et al., 2020, 2022, 2023; Tsai et al., 2021; Song et al., 2024, 2025; Li et al., 2025).
- **Upstream URL / DOI:**
  - Li et al. (2025) $\text{LSTM} + \delta\text{HBV}$ Zenodo archive: `https://doi.org/10.5281/zenodo.16895228` (`CC-BY-4.0`).
  - Feng et al. (2022) $\delta\text{HBV 1.0}$ Zenodo archive: `https://doi.org/10.5281/zenodo.7091334` / `10.5281/zenodo.7943626` (`CC-BY-4.0`).
  - Feng et al. (2020) `DI-LSTM` (`MHPI-hydroDL`) Zenodo archive: `https://doi.org/10.5281/zenodo.3993880` (`CC-BY-4.0`).
  - Tsai et al. (2021) `dPL + VIC` Zenodo archive: `https://doi.org/10.5281/zenodo.5227738` / `10.5281/zenodo.6383415` (`other-nc` / PSU Non-Commercial License).
  - MHPI S3 pretrained model weights (`https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/models/`): governed by `mhpi` PSU Non-Commercial Software License.
  - *(Note: `leaderboards/table_2_registry.json` currently lists `10.5281/zenodo.7091372` and `10.5281/zenodo.5165216` for these entries; `7091372` is an unrelated Zenodo math record and `5165216` is `jmframe/mclstm_2021_extrapolate`.)*
- **Local / GCS Path:**
  - MIP Re-run $\delta\text{HBV 1.0}$ and $\delta\text{HBV 1.1p}$ hydrographs in `gs://camels-us-mip/hydrographs/{gauged,pub,pur}__{single,multi}_camels/` (`hydrographs/category_ii/t3_18.npz`, `t3_19.npz`, `t3_29.npz`, `t3_33.npz`, `t4_24.npz`, `t4_29.npz`, `t4_32.npz`, `t4_33.npz`, etc.) were generated from scratch by our first-party `StandaloneDHBV` CUDA implementation.
  - Li et al. (2025) (`10.5281/zenodo.16895228`, `CC-BY-4.0`) is staged in `<external_staging_dir>/step3a_dhbv/li2025_ensemble_hydrographs_split3_test.npz`.
  - No daily hydrographs from Tsai et al. (2021) (`10.5281/zenodo.5227738`) are copied into `gs://camels-us-mip/` (published-table-only entry).
- **Exact License Name / SPDX Identifier:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** for Zenodo `16895228`, `7091334`, and `3993880`; **Pennsylvania State University Non-Commercial Software License (`LicenseRef-PSU-NonCommercial` / `other-nc`)** for Zenodo `5227738` and MHPI S3 checkpoints.
- **Redistribution Status:** **`GREEN — Permissive (`CC-BY-4.0` / First-Party Re-Runs)`** for all hydrographs stored in `gs://camels-us-mip/`; **`YELLOW — Non-Commercial`** for `10.5281/zenodo.5227738` and MHPI S3 checkpoints.
- **Required Action Before Public Release:** Fix the Zenodo DOIs in `leaderboards/table_2_registry.json` (`7091372` $\rightarrow$ `10.5281/zenodo.16895228` and `10.5281/zenodo.7091334`; `5165216` $\rightarrow$ `10.5281/zenodo.5227738`), retain `CC-BY-4.0` attribution for Li et al. (2025) and Feng et al. (2020, 2022), and do not redistribute raw files from `10.5281/zenodo.5227738` or MHPI S3 in `gs://camels-us-mip/`.

#### B4. Nearing et al. (2022) AR & DA LSTM Zenodo Archive
- **Resource & Citation:** Autoregressive (`AR-LSTM`) and variational Data-Assimilation (`DA-LSTM`) streamflow models (Nearing, Kratzert, Frame, Pelissier, Klotz, Frame, & Gupta, 2022, *HESS*, 26, 5493–5513, `10.5194/hess-26-5493-2022`).
- **Upstream URL / DOI:** `https://doi.org/10.5281/zenodo.7063252` (`grey-nearing/lstm-data-assimilation` v1.0) and `https://doi.org/10.5281/zenodo.7063259` (`NeuralHydrology` v1.3.0). *(Note: `leaderboards/table_2_registry.json` lists `10.5281/zenodo.7152656`, which is an unrelated LoRa wireless dataset.)*
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/`.
- **Exact License Name / SPDX Identifier:** **`BSD-3-Clause`** (`10.5281/zenodo.7063259`), **`CC-BY-4.0`** (HESS article `10.5194/hess-26-5493-2022`), and first-party author work by Grey Nearing (`10.5281/zenodo.7063252`).
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Fix the DOI typo in `leaderboards/table_2_registry.json` from `10.5281/zenodo.7152656` to `10.5281/zenodo.7063252`.

#### B5. Hoedt et al. (2021) & Frame et al. (2022, 2023) `MC-LSTM` Zenodo & HydroShare Archives
- **Resource & Citation:** Mass-Conserving LSTM (`MC-LSTM`) CAMELS-US simulations (Hoedt et al., 2021, *ICML*; Frame et al., 2022, *HESS*, `10.5194/hess-26-3377-2022`; Frame et al., 2023, *JHM*, `10.1175/JHM-D-22-0083.1`).
- **Upstream URL / DOI:**
  - Hoedt et al. (2021) Zenodo archive: `https://doi.org/10.5281/zenodo.8138359` (`CC-BY-4.0`). *(Note: `leaderboards/table_2_registry.json` lists `10.5281/zenodo.5040168`, which is an unrelated linguistics thesis.)*
  - Frame et al. (2022) Zenodo code archive: `https://doi.org/10.5281/zenodo.5165216` (`jmframe/mclstm_2021_extrapolate`, `MIT`).
  - Frame et al. (2022, 2023) HydroShare hydrograph archive: `https://www.hydroshare.org/resource/d750278db868447dbd252a8c5431affd/` (`10.4211/hs.d750278db868447dbd252a8c5431affd`, `CC-BY-NC-4.0`).
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`<external_staging_dir>/step1_published_split1/hydrographs_frame_mclstm_archive.npz` and our own first-party `NeuralHydrology` `MC-LSTM` re-runs in `step3b_neuralhydrology/`).
- **Exact License Name / SPDX Identifier:**
  - Hoedt et al. (2021) (`10.5281/zenodo.8138359`): **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)**.
  - Frame et al. (2022, 2023) HydroShare (`10.4211/hs.d750278db868447dbd252a8c5431affd`): **Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC-4.0`)**.
- **Redistribution Status:** **`GREEN — Permissive (`CC-BY-4.0`)`** for Hoedt et al. (2021) and our own `MC-LSTM` re-runs; **`YELLOW — Copyleft / ShareAlike / Non-Commercial (`CC-BY-NC-4.0`)`** for the Frame et al. (2022, 2023) HydroShare hydrograph archive (`d750278db868447dbd252a8c5431affd`).
- **Required Action Before Public Release:**
  1. Update `leaderboards/table_2_registry.json` DOI from `10.5281/zenodo.5040168` to `10.5281/zenodo.8138359` (and `10.4211/hs.d750278db868447dbd252a8c5431affd`).
  2. Since Grey Nearing is a co-author on Frame et al. (2022, 2023), either ask lead author Jonathan Frame to switch the HydroShare resource license on `d750278db868447dbd252a8c5431affd` from `CC-BY-NC-4.0` to `CC-BY-4.0`, or clearly mark the `frame22_*` / `frame23_*` Zarr stores in `gs://camels-us-mip/` as `CC-BY-NC-4.0`.

#### B6. Gauch et al. (2021) `XGBoost` Zenodo Archive
- **Resource & Citation:**
  1. Gauch, Mai, & Lin (2021, *EMS*, `10.1016/j.envsoft.2020.104926`): *"The Proper Care and Feeding of CAMELS"* (`XGBoost` and `LSTM` data/code).
- **Upstream URL / DOI:**
  - Gauch et al. (2021) `XGBoost` Zenodo archive: `https://doi.org/10.5281/zenodo.3648270` *(listed as `10.5281/zenodo.4539986` in `leaderboards/table_2_registry.json`, which is an unrelated Alpine catchment dataset)*.
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`hydrographs/category_ii/t3_53.npz`).
- **Exact License Name / SPDX Identifier:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** (`3648270`).
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Fix the Zenodo DOI in `leaderboards/table_2_registry.json` from `10.5281/zenodo.4539986` to `10.5281/zenodo.3648270` and retain `CC-BY-4.0` attribution.

#### B7. Thébault et al. (2026) / Addor et al. (2017) `FUSE` 78-Model & 3-Model Archives
- **Resource & Citation:**
  1. Thébault, Knoben, Clark, & Perrin (2026, *HESS*, 30, 663–688, `10.5194/hess-30-663-2026`): 78 `FUSE` model structures evaluated across 558 CAMELS-US basins.
  2. Addor et al. (2017, *HESS*, `10.5194/hess-21-5293-2017`): `FUSE 900`, `902`, and `904` daily hydrographs on the Newman 531-basin split (archived in Kratzert et al., 2019b HydroShare `474ecc37e7db45baa425cdb4fc1b61e1`).
- **Upstream URL / DOI:**
  - Thébault et al. (2026) HydroShare data archive (`Eval_FUSE.Rdata`): `https://www.hydroshare.org/resource/53063da8fc894149b5b25f3201d1e8e8/` (`10.4211/hs.53063da8fc894149b5b25f3201d1e8e8`).
  - Thébault et al. (2026) Zenodo code archive: `https://doi.org/10.5281/zenodo.20500212` *(Note: `leaderboards/table_2_registry.json` lists `10.5281/zenodo.14019252`, which is an unrelated medical record).*
  - Addor et al. (2017) / Kratzert et al. (2019b) HydroShare: `https://doi.org/10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1`.
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__single_camels/` (`FUSE 900/902/904` daily hydrographs; per-basin `FUSE-78` evaluation metrics in `<external_staging_dir>/step2_process_archives/results/fuse78_per_basin_metrics.npz` — note that Thébault et al. 2026 only published per-basin metric tables in `Eval_FUSE.Rdata`, not daily streamflow time series).
- **Exact License Name / SPDX Identifier:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** on both HydroShare data archives (`53063da8fc894149b5b25f3201d1e8e8` and `474ecc37e7db45baa425cdb4fc1b61e1`); **GNU General Public License v3.0 (`GPL-3.0-only`)** on the Zenodo code archive (`10.5281/zenodo.20500212`).
- **Redistribution Status:** **`GREEN — Permissive (`CC-BY-4.0`)`** for all hydrograph and evaluation metric data in `gs://camels-us-mip/`.
- **Required Action Before Public Release:** Fix the DOI in `leaderboards/table_2_registry.json` from `10.5281/zenodo.14019252` to `10.4211/hs.53063da8fc894149b5b25f3201d1e8e8` (`10.5281/zenodo.20500212`) and retain `CC-BY-4.0` attribution.

#### B8. Knoben et al. (2020) `MARRMoT` 36-Model Hydrograph Archive
- **Resource & Citation:** Daily streamflow simulations for 36 conceptual rainfall-runoff models (`m_01` through `m_37`, excluding `m_04`) calibrated across 559 CAMELS-US basins (Knoben, Freer, Peel, Fowler, & Woods, 2020, *Water Resources Research*, 56, e2019WR025975, `10.1029/2019WR025975`).
- **Upstream URL / DOI:** University of Bristol `data.bris` repository `https://doi.org/10.5523/bris.2zutxh2qeep6y2cy6scwgk9eqj` (`https://data.bris.ac.uk/datasets/2zutxh2qeep6y2cy6scwgk9eqj/`) and MARRMoT v1.0 Zenodo `https://doi.org/10.5281/zenodo.2482542`. *(Note: `leaderboards/table_2_registry.json` lists `10.5281/zenodo.3746544`, which returns HTTP 404.)*
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__single_camels/` (staged in `<external_staging_dir>/step2_process_archives/marrmot_bristol/simq_531/m_*_simq531.npz`).
- **Exact License Name / SPDX Identifier:** **GNU General Public License v3.0 (`GPL-3.0-only`)** — explicitly specified in `step2_process_archives/marrmot_bristol/readme.txt`:
  > *"All data included in this package are created in this study and provided here under the GNU General Public License version 3.0."*
- **Redistribution Status:** **`YELLOW — Copyleft / ShareAlike / Non-Commercial (GPL/LGPL/CC-BY-NC/CC-BY-SA)`**
- **Required Action Before Public Release:**
  1. Fix the DOI in `leaderboards/table_2_registry.json` from `10.5281/zenodo.3746544` to `10.5523/bris.2zutxh2qeep6y2cy6scwgk9eqj`.
  2. Include the `GPL-3.0` license notice and attribution to Knoben et al. (2020) in the Zarr metadata (`attrs["license"] = "GPL-3.0-only"`) and `gs://camels-us-mip/README.md` for the 36 `marrmot_*` Zarr stores. Under `GPL-3.0` Section 5 ("Aggregate"), hosting independent `GPL-3.0` Zarr data stores alongside `CC-BY-4.0` Zarr stores in the same storage bucket is mere aggregation and does not affect the license of other models' hydrographs.

#### B9. Farahani et al. (2025) `SUMMA-mizuRoute` LSE Zenodo Archive
- **Resource & Citation:** Large-sample ensemble (`LSE`) `SUMMA-mizuRoute` simulations over CAMELS-US (Farahani et al., 2025, *HESS*, 29, 4515–4539, `10.5194/hess-29-4515-2025`).
- **Upstream URL / DOI:** `https://doi.org/10.5281/zenodo.16422768` *(Note: `leaderboards/table_2_registry.json` lists `10.5281/zenodo.13892055`, which is an unrelated `CC0-1.0` dataset).*
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__freeform/` (Note: per `<external_staging_dir>/step2_process_archives/README.md`, `10.5281/zenodo.16422768` contains basin attributes, parameter sets, and trained emulators, but no daily streamflow time series were uploaded by the authors; therefore `summa_farahani` is a published-summary-only entry with no hydrograph Zarr store redistributed).
- **Exact License Name / SPDX Identifier:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** on `10.5281/zenodo.16422768`.
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Fix the DOI in `leaderboards/table_2_registry.json` from `10.5281/zenodo.13892055` to `10.5281/zenodo.16422768` and retain `CC-BY-4.0` attribution.

#### B10. NOAA National Water Model (`NWM v2.0`, `v2.1`, `v3.0`) Retrospective Archive on AWS Open Data Registry
- **Resource & Citation:** NOAA Office of Water Prediction National Water Model (`NWM`) CONUS Retrospective v2.0, v2.1, and v3.0 hourly streamflow archives mapped to CAMELS-US USGS gauges (Frame et al., 2021, *JAWRA*, `10.1111/1752-1688.12964`; Towler et al., 2023, *JHM*, `10.1175/JHM-D-22-0125.1`).
- **Upstream URL / DOI:** `https://registry.opendata.aws/nwm-archive/` (`s3://noaa-nwm-retrospective-3-0-pds`, `s3://noaa-nwm-retrospective-2-1-zarr-pds`, `s3://noaa-nwm-retro-v2-zarr-pds`).
- **Local / GCS Path:** `gs://camels-us-mip/hydrographs/gauged__freeform/`, `gs://camels-us-mip/hydrographs/gauged__single_camels/`, `gs://camels-us-mip/hydrographs/pub__freeform/`, `gs://camels-us-mip/hydrographs/pur__freeform/` (`<external_staging_dir>/step2_process_archives/hydrographs_nwm_*.npz` and `step3b_neuralhydrology/results/nwm_postprocessed_hydrographs.npz`).
- **Exact License Name / SPDX Identifier:** **U.S. Government Public Domain (`17 U.S.C. § 105`)** / AWS Open Data Sponsorship Program (`License: "No restrictions on the use of the data"` in `https://github.com/awslabs/open-data-registry/blob/main/datasets/nwm-archive.yaml`).
- **Redistribution Status:** **`GREEN — Permissive (MIT/BSD/Apache/CC-BY/Public Domain)`**
- **Required Action Before Public Release:** Retain citation to NOAA Office of Water Prediction, Frame et al. (2021), and Towler et al. (2023).

---

## Corrections Needed in `leaderboards/table_2_registry.json`

During this audit, we verified every `github_url` and `zenodo_doi` in `leaderboards/table_2_registry.json` against the live GitHub, Zenodo, HydroShare, and DataCite APIs and cross-checked them against the actual download/ingestion scripts in `<external_staging_dir>/`. The following 12 fields in `leaderboards/table_2_registry.json` should be updated so that all public links resolve to the exact canonical repositories and data archives:

| Entry / Citation in `table_2_registry.json` | Field | Current Value in `table_2_registry.json` | Problem with Current Value | Verified Canonical Replacement | Canonical License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kratzert et al. (2021)** (`Multi-forcing LSTM`) | `zenodo_doi` | `10.5281/zenodo.4071885` | Points to Gauch, Kratzert et al. (2021) `MTS-LSTM` record | `10.5281/zenodo.4670268` | `CC-BY-4.0` |
| **Feng et al. (2022, 2023)** (`dHBV 1.0`) | `zenodo_doi` | `10.5281/zenodo.7091372` | Unrelated Zenodo math record (*"Weakly Nonlinear Dynamics..."*) | `10.5281/zenodo.7091334` (`10.5281/zenodo.7943626`) | `CC-BY-4.0` |
| **Li et al. (2025)** (`LSTM + dHBV`) | `zenodo_doi` | `10.5281/zenodo.7091372` | Unrelated Zenodo math record | `10.5281/zenodo.16895228` | `CC-BY-4.0` |
| **Tsai et al. (2021)** (`dPL + VIC`) | `github_url` & `zenodo_doi` | `https://github.com/mhpi/dPL` & `10.5281/zenodo.5165216` | `mhpi/dPL` is 404; `5165216` is `jmframe/mclstm_2021_extrapolate` | `https://github.com/mhpi/hydroDL` & `10.5281/zenodo.5227738` (`10.5281/zenodo.6383415`) | `LicenseRef-PSU-NonCommercial` (`other-nc`) |
| **Nearing et al. (2022)** (`AR & DA LSTM`) | `zenodo_doi` | `10.5281/zenodo.7152656` | Unrelated Zenodo record (*"Outdoor LoRa Gateways..."*) | `10.5281/zenodo.7063252` (`10.5281/zenodo.7063259`) | `BSD-3-Clause` / `CC-BY-4.0` |
| **Hoedt et al. (2021) / Frame et al. (2022, 2023)** (`MC-LSTM`) | `zenodo_doi` | `10.5281/zenodo.5040168` | Unrelated Zenodo linguistics record | `10.5281/zenodo.8138359` (Hoedt 2021), `10.5281/zenodo.5165216` (Frame 2022 code), `10.4211/hs.d750278db868447dbd252a8c5431affd` (Frame 2022/2023 hydrographs) | `CC-BY-4.0` / `MIT` / `CC-BY-NC-4.0` |
| **Rasiya Koya & Roy (2025)** (`TFT`) | `github_url` | `https://github.com/sinanrk/TFT-Runoff` | HTTP 404 on GitHub | `https://github.com/sinanrk/TFT-streamflow` | `MIT` |
| **Gauch et al. (2021)** (`XGBoost`) | `github_url` & `zenodo_doi` | `https://github.com/martin-gauch/proper-care-and-feeding` & `10.5281/zenodo.4539986` | Repo URL is 404; `4539986` is an unrelated Alpine catchment record | `https://github.com/gauchm/ealstm_regional_modeling` & `10.5281/zenodo.3648270` | `Apache-2.0` (code) / `CC-BY-4.0` (data) |
| **Acuña Espinoza et al. (2024)** (`Hy2DL`) | `github_url` & `zenodo_doi` | `https://github.com/KIT-HYD/Hy2DL` & `10.5281/zenodo.11181938` | `KIT-HYD/Hy2DL` is deprecated (`GPL-3.0`); `11181938` is 404 | `https://github.com/eduardoAcunaEspinoza/Hy2DL` & `10.5281/zenodo.22083944` (`10.5281/zenodo.17251944`) | `BSD-3-Clause` |
| **Thébault et al. (2026)** (`FUSE-78`) | `zenodo_doi` | `10.5281/zenodo.14019252` | Unrelated Zenodo medical record | `10.4211/hs.53063da8fc894149b5b25f3201d1e8e8` (data) & `10.5281/zenodo.20500212` (code) | `CC-BY-4.0` (data) / `GPL-3.0-only` (code) |
| **Knoben et al. (2020)** (`MARRMoT 36-Model Suite`) | `zenodo_doi` | `10.5281/zenodo.3746544` | HTTP 404 on Zenodo | `10.5523/bris.2zutxh2qeep6y2cy6scwgk9eqj` (hydrographs) & `10.5281/zenodo.2482542` (code) | `GPL-3.0-only` |
| **Farahani et al. (2025)** (`SUMMA-mizuRoute LSE`) | `zenodo_doi` | `10.5281/zenodo.13892055` | Points to an unrelated record | `10.5281/zenodo.16422768` | `CC-BY-4.0` |
| **Tayal et al. (2024)** (`Geo-ViT-LSTM`) | `github_url` | `https://github.com/kshitijtayal/Geo-ViT-LSTM` | HTTP 404 (no public repo released by authors) | `null` (cite paper DOI `10.1145/3678717.3691324`) | N/A (published table only) |
