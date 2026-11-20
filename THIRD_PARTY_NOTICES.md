# Third-Party Notices and Model Provenance

This file documents the third-party software from which portions of **HydrologyMIP** (`https://hydrologymip.org`; repository `camels-us-mip`: the shared `hydromip/` evaluation core, the model zoo in `models/`, the web portal, and the registered MIPs under `mips/`, starting with the **CAMELS-US Daily** MIP in `mips/camels_us_daily/`) are derived, together with the copyright notices and full license texts required by their respective licenses, as well as the academic provenance of the independent model implementations in `models/`.

The software in this repository is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE)). The derived portions identified in Section 1 remain subject to their permissive upstream licenses (`BSD-3-Clause`, `Apache-2.0`, `MIT`), all of which are compatible with the Apache License 2.0.

First-party and third-party **datasets and hydrograph Zarr stores** in `gs://camels-us-mip/` are governed by [`DATA_LICENSE.md`](./DATA_LICENSE.md).

---

## Summary of Licensing & Provenance Tiers

| Tier | Scope | License (SPDX Identifier) | Location |
| :--- | :--- | :--- | :--- |
| **Tier 1 — First-Party Original Code & ERA Architectures** | Shared HydrologyMIP core (`hydromip/` Python package: protocol registry, splits, paths, evaluation/Zarr I/O, leaderboard ranking, metric suites), MIP adapters (`mips/*/adapter.py`), MIP evaluation pipeline (`metrics/evaluator.py`, `metrics/cli.py`, `metrics/build_eval_targets.py`), leaderboard and figure tooling (`leaderboards/*.py`), runner/Zarr infrastructure (`models/run_model.py`, `models/camels_data.py`, `models/zarr_writer.py`), web platform (`web/`), benchmarks (`benchmarks/`), ERA notebooks (`mips/camels_us_daily/era_experiments/`), 7 ERA model directories (`models/era_*/`), `models/li2025_lstm_dhbv_ens/`, and the lag-1 persistence baseline `models/naive_lag1_persistence/` | **`Apache-2.0`** | Git repository (`camels-us-mip`) |
| **Tier 2 — Permissive Third-Party Code & Co-Authored Upstream Adaptations** | `neuralhydrology` (`BSD-3-Clause`), `kratzert/*` (`Apache-2.0`), `ml-jku/mc-lstm` (`MIT`), `jmframe/mclstm_2021_extrapolate` (`MIT`), `sinanrk/TFT-streamflow` (`MIT`), NOAA NWS `Snow-17`/`SAC-SMA` equations (U.S. Public Domain, `17 U.S.C. § 105`) | **`BSD-3-Clause`, `MIT`, `Apache-2.0`, Public Domain** | `hydromip/metrics/deterministic_daily.py` (re-exported by `metrics/metrics.py`), `models/kratzert_lstm/`, `models/kratzert_ealstm/`, `models/hoedt_mclstm/`, `models/rasiya_koya_tft/` |
| **Tier 3 — Independent First-Party Implementations of Published Mathematical Models** | Independent first-party C++/OpenMP, CUDA, and PyTorch implementations written from published mathematical equations and parameter bounds with no upstream source code copied | **`Apache-2.0`** (with academic citation) | `models/_dhbv_common/` (shared δHBV core), `models/feng_dhbv10/`, `models/song_dhbv11p/`, `models/feng2020_di_lstm/`, `models/newman_sacsma/`, `models/poudel_mlp_hbv/`, `models/yin2022_rrformer/`, `models/shahriar2026_mamba3/` |
| **Tier 4 — Datasets, Splits, Leaderboards, & Hydrograph Zarr Stores** | First-party data (`CC-BY-4.0`) and external author-archived hydrographs (`CC-BY-4.0`, U.S. Public Domain `17 U.S.C. § 105`, `CC-BY-NC-4.0` for Frame et al. 2023) | **See [`DATA_LICENSE.md`](./DATA_LICENSE.md)** | `mips/camels_us_daily/splits/`, `mips/camels_us_daily/leaderboards/*.{json,csv,md}`, `mips/camels_us_daily/figures/`, and `gs://camels-us-mip/` |

---

## 1. Permissive Third-Party Code & Co-Authored Upstream Libraries Incorporated into This Repository (Tier 2)

| Component | Upstream Repository & Citation | Upstream License | Where Used in `camels-us-mip` |
| :--- | :--- | :--- | :--- |
| **NeuralHydrology** | `https://github.com/neuralhydrology/neuralhydrology` (Kratzert, Gauch, Nearing, & Klotz, 2022, *JOSS*, 7(71), 4050, `https://doi.org/10.21105/joss.04050`; Gauch et al., 2023, *WRR*, `https://doi.org/10.1029/2022WR033918`) | **`BSD-3-Clause`** | `hydromip/metrics/deterministic_daily.py` (derived from `neuralhydrology/evaluation/metrics.py`; re-exported unchanged by `metrics/metrics.py`); `models/hoedt_mclstm/model.py` (`_Gate`, `_NormalizedGate`, and MC-LSTM cell structure derived from `neuralhydrology/modelzoo/mclstm.py`); `models/kratzert_ealstm/model.py` and `models/kratzert_lstm/model.py` (EA-LSTM gate parameterization and basin-averaged `NSE*` loss follow `neuralhydrology/modelzoo/ealstm.py` and `neuralhydrology/training/loss.py`) |
| **`ealstm_regional_modeling`, `lstm_for_pub`, `multiple_forcing`** | `https://github.com/kratzert/ealstm_regional_modeling` (`https://doi.org/10.5281/zenodo.3530884`), `https://github.com/kratzert/lstm_for_pub`, `https://github.com/kratzert/multiple_forcing` (`https://doi.org/10.5281/zenodo.4670268`) (Kratzert et al., 2019a, 2019b, 2021) | **`Apache-2.0`** | `models/kratzert_ealstm/model.py` (`_DynamicGates` initialization follows `papercode/ealstm.py`); training-protocol schedule in `models/kratzert_lstm/model.py` |
| **`ml-jku/mc-lstm` & `jmframe/mclstm_2021_extrapolate`** | `https://github.com/ml-jku/mc-lstm` (`https://doi.org/10.5281/zenodo.8138359`, Hoedt et al., 2021, *ICML*) and `https://github.com/jmframe/mclstm_2021_extrapolate` (`https://doi.org/10.5281/zenodo.5165216`, Frame et al., 2022, 2023) | **`MIT`** | Lineage of the Mass-Conserving LSTM (`MC-LSTM`) module used in `models/hoedt_mclstm/model.py` |
| **`sinanrk/TFT-streamflow`** | `https://github.com/sinanrk/TFT-streamflow` (Rasiya Koya & Roy, 2024, *J. Hydrol.*, `https://doi.org/10.1016/j.jhydrol.2024.131301`) | **`MIT`** | Referenced for hyperparameter configuration in `models/rasiya_koya_tft/` |

### 1.1 NeuralHydrology — BSD 3-Clause License

```text
BSD 3-Clause License

Copyright (c) 2021, NeuralHydrology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### 1.2 Frederik Kratzert (`ealstm_regional_modeling`, `lstm_for_pub`, `multiple_forcing`) — Apache License 2.0

```text
Copyright 2019-2021 Frederik Kratzert, Daniel Klotz, Grey Nearing

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### 1.3 `ml-jku/mc-lstm` & `jmframe/mclstm_2021_extrapolate` — MIT License

```text
MIT License

Copyright (c) 2021 Institute for Machine Learning, Johannes Kepler University Linz
Copyright (c) 2021 Jonathan M. Frame

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.4 `sinanrk/TFT-streamflow` — MIT License

```text
MIT License

Copyright (c) 2024 Sinan Rasiya Koya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.5 U.S. Government Public-Domain Hydrological Equations (`17 U.S.C. § 105`)

The Sacramento Soil Moisture Accounting (`SAC-SMA`, Burnash, Ferral, & McGuire, 1973), `Snow-17` snow accumulation/ablation (Anderson, 1973, 2006), and Duamel unit-hydrograph routing algorithms implemented in `models/newman_sacsma/model.py` are U.S. National Weather Service (NWS) Hydrologic Research Laboratory formulations in the **U.S. Government Public Domain (`17 U.S.C. § 105`)**. `models/newman_sacsma/` is an independent C++17/OpenMP and NumPy implementation of the published equations and does not include the NCAR Fortran wrapper (`GPL-3.0-only`) or `Upstream-Tech/SACSMA-SNOW17` (no license).

---

## 2. Independent Implementations & Reference-Only Repositories — No Upstream Code Included (Tier 3)

In accordance with `17 U.S.C. § 102(b)` (the idea/expression dichotomy), the model directories listed below implement mathematical equations, flux laws, neural network architectures, parameter bounds, and training schedules published in peer-reviewed journals. **No source code from any non-commercial, copyleft, or unlicensed upstream repository is included in this repository.** They are listed below for scientific attribution:

| Model Family & Citation | Upstream Reference Repository & License | MIP Implementation & Status |
| :--- | :--- | :--- |
| **$\delta\text{HBV 1.0}$** (Feng et al., 2022, *WRR*, `10.1029/2022WR032404`; Feng et al., 2023, *HESS*, `10.5194/hess-27-2357-2023`) | `https://github.com/mhpi/hydroDL` (`LicenseRef-PSU-NonCommercial`) | `models/feng_dhbv10/` — First-party CUDA C++ (`hbv_forward_kernel`, `hbv_backward_kernel`) and PyTorch implementation from published equations, built on the shared first-party δHBV core `models/_dhbv_common/core.py` (`Apache-2.0`) |
| **$\delta\text{HBV 1.1p}$ / $\delta\text{MG}$** (Song et al., 2024, *HESS*, `10.5194/hess-28-3051-2024`, `CC-BY-4.0`; Song et al., 2025, *WRR*, `10.1029/2024WR038928`) | `https://github.com/mhpi/generic_deltamodel` (`LicenseRef-PSU-NonCommercial`) | `models/song_dhbv11p/` — First-party CUDA C++ and PyTorch implementation from published equations, built on the shared first-party δHBV core `models/_dhbv_common/core.py` (`Apache-2.0`) |
| **$\text{LSTM} + \delta\text{HBV 1.1p}$ Ensemble** (Li et al., 2025, *HESS*, `10.5194/hess-29-6829-2025`, `CC-BY-4.0`) | `https://github.com/mhpi/generic_deltamodel` (`LicenseRef-PSU-NonCommercial`) | `models/li2025_lstm_dhbv_ens/` — First-party 20-line NumPy 50/50 ensemble combiner (`Apache-2.0`) |
| **`DI(1)-LSTM`** (Feng, Fang, & Shen, 2020, *WRR*, `10.1029/2019WR026793`) | `https://github.com/mhpi/hydroDL` (`LicenseRef-PSU-NonCommercial`) | `models/feng2020_di_lstm/` — First-party PyTorch lagged-discharge feature builder over `models.kratzert_lstm.CudaLSTM` (`Apache-2.0`) |
| **`SAC-SMA + Snow-17` (DDS & 29-D Donor Transfer)** (Nearing et al., 2020, `10.31223/osf.io/53te4`; Newman et al., 2017, `10.1175/JHM-D-16-0284.1`; Tolson & Shoemaker, 2007) | `https://github.com/Upstream-Tech/SACSMA-SNOW17` (no license); NCAR Fortran wrapper (`GPL-3.0-only`); NOAA NWS equations (U.S. Public Domain) | `models/newman_sacsma/` — First-party C++17/OpenMP (`_SACSMA_OMP_CPP`) and NumPy implementation of public-domain NWS equations (`Apache-2.0`) |
| **`MLP + 16-HBV`** (Poudel & Steinschneider, 2026, *HESS*, `10.5194/hess-30-5173-2026`, `CC-BY-4.0`) | `https://github.com/snpoudel/diff-hydro` (no license file; HESS paper is `CC-BY-4.0`) | `models/poudel_mlp_hbv/` — First-party C++/OpenMP, CUDA, FFT Gamma UH routing, and PyTorch `nn.Sequential` implementation from the `CC-BY-4.0` HESS paper (`Apache-2.0`) |
| **`RR-Former`** (Yin et al., 2022, *J. Hydrol.*, `10.1016/j.jhydrol.2022.127781`) | `https://github.com/iThronne/RR-Former` (no license file) | `models/yin2022_rrformer/` — First-party PyTorch 2.0 (`F.scaled_dot_product_attention`) and closed-form L2 ridge readout implementation from Yin et al. (2022) (`Apache-2.0`) |
| **`TFT`** (Rasiya Koya & Roy, 2024, *J. Hydrol.*, `10.1016/j.jhydrol.2024.131301`; Lim et al., 2021) | `https://github.com/sinanrk/TFT-streamflow` (`MIT`) | `models/rasiya_koya_tft/` — First-party PyTorch VSN/GRN/LSTM/MHA implementation (`Apache-2.0`) |
| **`Mamba-3 Static-Dynamic Adapter`** (Shahriar, 2026, *SSRN* 7552815, `10.2139/ssrn.7552815`) | `https://doi.org/10.5281/zenodo.23087719` (`MIT`) | `models/shahriar2026_mamba3/` — First-party pure-PyTorch selective state-space + static-dynamic adapter implementation (`Apache-2.0`); the sequence mixer `models/shahriar2026_mamba3/mamba3.py` is Copyright 2026 Google LLC under `Apache-2.0` |
| **Literature-Only Entries** (`kratzert2019a_pub_lstm`, `feng2020_di_lstm_lit`, `gauch_xgboost_lit`, `hoedt_mclstm_lit`, `tsai2021_dpl_vic`, `nearing2022_ar_da_lstm`, `jamaat2025_dhbv_da`, `liu2025_tslib_suite`, `wang_s4dft`) | Various (`Apache-2.0`, `MIT`, `CC-BY-NC-4.0`, `LicenseRef-PSU-NonCommercial`, or no license) | **Literature-only summary rows in `mips/camels_us_daily/leaderboards/*.json`** (`code_dir: null`; zero source code files and zero hydrograph Zarr stores redistributed) |
| **Archive-Only Entries** (`fuse_suite`, `mizukami_vic_mpr`, `newman_sacsma`, `newman_vic`, `seibert_hbv`, `kratzert2019b_lstm`, `kratzert_ealstm`, `kratzert_lstm_nostatic`, `mizukami_mhm`, `rakovec_mhm_mpr`, `kratzert2021_lstm_archive`, `frame2023_mclstm_archive`) | Author hydrograph archives on HydroShare, Zenodo, and NCAR GDEX | **External author hydrograph archives in `gs://camels-us-mip/`** (`code_dir: null` except `kratzert_ealstm` Stage-1 runner; see [`DATA_LICENSE.md`](./DATA_LICENSE.md)) |

---

## 3. Runtime Dependencies (Not Redistributed)

`camels-us-mip` imports the following open-source Python packages at runtime (installed by the user via `pip` from `pyproject.toml` — core dependencies plus the `portal`, `models`, and `dev` extras — and **not** redistributed in this repository):

| Package | License |
| :--- | :--- |
| `torch` | `BSD-3-Clause` |
| `numpy`, `pandas`, `scipy`, `dask`, `fsspec`, `gcsfs`, `uvicorn`, `httpx` | `BSD-3-Clause` |
| `xarray`, `python-multipart`, `ninja` | `Apache-2.0` |
| `zarr`, `fastapi`, `pydantic`, `pytest`, `pytest-xdist`, `pyyaml`, `ruff` | `MIT` |
| `matplotlib` | `PSF-2.0` / Matplotlib License (BSD-compatible) |

The web interface (`web/portal/index.html`) loads the IBM Plex Sans, JetBrains Mono, and Newsreader typefaces at page-load time from Google Fonts (all `OFL-1.1`, SIL Open Font License 1.1); no font files are redistributed in this repository.
