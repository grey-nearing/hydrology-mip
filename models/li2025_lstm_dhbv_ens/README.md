# Li et al. (2025) — $(\text{LSTM} + \delta\text{HBV 1.1p})$ Multi-Model Ensemble

## Canonical Metadata & Provenance

| Field | Value |
| :--- | :--- |
| **Directory / Family ID** | `models/li2025_lstm_dhbv_ens` (`li2025_lstm_dhbv_ens`) |
| **Canonical Model Name** | `(LSTM + dHBV 1.1p) Multi-Model Ensemble` (`LSTM + dHBV 1.1p (Multi-Model Ensemble)`) |
| **Model Category** | Differentiable / Hybrid |
| **Architecture Type** | Multi-Structure Hybrid Ensemble (Regional LSTM + Differentiable HBV 1.1p) |
| **Provenance** | `MIP Re-run (Our Code)` |
| **Canonical Citation** | Li, B., Song, Y., Pan, M., Lawson, K., & Shen, C. (2025), *Hydrology and Earth System Sciences*, 29, 6829–6861 |
| **Paper DOI / URL** | [https://doi.org/10.5194/hess-29-6829-2025](https://doi.org/10.5194/hess-29-6829-2025) |
| **Upstream Reference Repo** | [https://github.com/mhpi/generic_deltamodel](https://github.com/mhpi/generic_deltamodel) |
| **Zenodo Archive** | [https://doi.org/10.5281/zenodo.16895228](https://doi.org/10.5281/zenodo.16895228) (`10.5281/zenodo.16895228`) |

---

## Model Architecture

Li et al. (2025) introduced a multi-structure, multi-forcing, multi-seed ensemble combining two distinct regional deep-learning/differentiable hydrological model paradigms across the three standard CAMELS-US meteorological forcing products (Daymet, Maurer, and NLDAS):

1. **Pure Data-Driven Regional LSTM ($\widehat{Q}_{\text{LSTM}}$):** Entity-aware / multi-forcing regional LSTM models trained on CAMELS-US basins (`kratzert2021_lstm_archive` / `kratzert_lstm`).
2. **Differentiable Process-Based Model ($\widehat{Q}_{\delta\text{HBV 1.1p}}$):** The dynamic-parameter differentiable HBV 1.1p ($\delta\text{MG}$) architecture of Song et al. (2024) (`models/song_dhbv11p`), where a 256-unit DropConnect LSTM parameter estimation network outputs 14 static HBV parameters per sub-grid bucket ($N_{\text{mul}} = 16$), 2 Gamma unit hydrograph routing parameters, and 3 daily time-varying parameters ($\beta_t, K_{0,t}, \gamma_t$) routed through an explicit differentiable HBV snow, soil moisture, capillary rise ($C$), two-zone groundwater, and 15-day Gamma unit hydrograph core.
3. **Multi-Structure Hybrid Combination:** Equal-weight arithmetic averaging across the two model families:
   $$\widehat{Q}_{\text{LSTM}+\delta\text{HBV}}(b, t) = \frac{1}{2}\!\left(\widehat{Q}_{\text{LSTM}}(b, t) + \widehat{Q}_{\delta\text{HBV 1.1p}}(b, t)\right)$$
   In Li et al. (2025), $(\text{LSTM} + \delta\text{HBV})_{\text{seed}}^{123}$ averages 60 individual daily hydrographs (3 forcing products $\times$ 2 model structures $\times$ 10 random seeds), while $(\text{LSTM} + \delta\text{HBV})^{123}$ averages 6 hydrographs from a single random seed across 3 forcings and 2 structures, and $(\text{LSTM} + \delta\text{HBV})^1_{\text{Daymet}}$ averages 1 Daymet LSTM and 1 Daymet $\delta\text{HBV 1.1p}$ member.

---

## Stage-1 Author Protocol Reproduction

Li et al. (2025, *HESS* 29, 6829–6861, Section 2.3 & Figure 1a) evaluated temporal generalization on **Split-3** (`1980-10-01` to `1995-09-30` training; `1995-10-01` to `2008-09-30` evaluation across 531 CAMELS-US basins) and archived all 60 daily member hydrographs on Zenodo (`10.5281/zenodo.16895228`, array shape `6 × 10 × 531 × 4749`). We verified the official Zenodo archive directly and also re-trained all three single-forcing $\delta\text{HBV 1.1p}$ models from scratch on Split-3 to verify our clean-room implementation (`models/song_dhbv11p`).

| Configuration | Split (Train / Test) | Forcing & Basins | Published Med NSE | Archive Recomputed Med NSE | Independent Re-run Hybrid Med NSE | $\Delta$ (Archive) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$(\text{LSTM} + \delta\text{HBV})_{\text{seed}}^{123}$** (60-member hybrid) | WY1980–1995 / WY1995–2008 | Daymet, Maurer, NLDAS (531) | 0.823 | **0.823** (`0.8225`; post-warmup `0.8286`) | **0.820** (`0.8195`) | `-0.0005` | **PASS** |
| **$(\text{LSTM} + \delta\text{HBV})^{123}$** (6-member 1-seed hybrid) | WY1980–1995 / WY1995–2008 | Daymet, Maurer, NLDAS (531) | 0.818 | **0.818** (`0.8184` member 0; `0.8176 ± 0.0040` 10-seed mean) | **0.817** (`0.8165`) | `-0.0004` | **PASS** |
| **$(\text{LSTM} + \delta\text{HBV})^1_{\text{Daymet}}$** (2-member 1-seed hybrid) | WY1980–1995 / WY1995–2008 | Daymet (531) | 0.790 | **0.790** (`0.7895` member 0; `0.7871 ± 0.0013` 10-seed mean) | **0.787** (`0.7872`) | `-0.0005` | **PASS** |
| **$\text{LSTM}_{\text{seed}}^{123}$** (30-member constituent) | WY1980–1995 / WY1995–2008 | Daymet, Maurer, NLDAS (531) | 0.818 | **0.814** (`0.8141`; post-warmup `0.8226`) | — | `-0.0039` | **PASS** |
| **$\delta\text{HBV}_{\text{seed}}^{123}$** (30-member constituent) | WY1980–1995 / WY1995–2008 | Daymet, Maurer, NLDAS (531) | 0.798 | **0.795** (`0.7945`; post-warmup `0.8005`) | **0.791** (`0.7913`) | `-0.0035` | **PASS** |

---

## Stage-2 CAMELS-US MIP Leaderboard Summary

On the standardized CAMELS-US MIP Split-1 temporal window (`1999-10-01` to `2008-09-30` training; `1989-10-01` to `1999-09-30` evaluation across 531 basins and 3,652 days), `t4_06.zarr` combines the two multi-forcing constituent ensembles on Table 4 with equal model-family weights (`num_members = 2`):
- **Member 0 (`t4_05.zarr`):** Kratzert et al. (2021) 10-seed 3-forcing regional LSTM ensemble (`Med NSE = 0.8210`).
- **Member 1 (`t4_24.zarr`):** Song et al. (2024) 5-seed 3-forcing $\delta\text{HBV 1.1p}$ ensemble (`Med NSE = 0.7992`).

An additional 20-member seed-level ensemble store (`t4_03.zarr`, stacking 10 LSTM members from `t4_05.zarr` and 10 $\delta\text{HBV 1.1p}$ members from `t4_23.zarr`) is also preserved on GCS and yields identical rounded performance (`Med NSE = 0.820`). Because the individual constituent models (`kratzert2021_lstm_archive` and `song_dhbv11p`) are already evaluated as standalone entries on Table 4, no duplicate single-model row is emitted for `li2025_lstm_dhbv_ens`.

| Table / Leaderboard | Row ID | Evaluation Mode | Forcings | GCS Zarr Store | Med NSE | Mean NSE | Med KGE | Med $|\text{FHV}|$ (%) | $\text{NSE} \le 0$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **T4** (`gauged__multi_camels`) | `t4_06` | 2-Component Multi-Model Ensemble | Daymet, Maurer, NLDAS | `gauged__multi_camels/t4_06.zarr` | **0.820** (`0.8200`) | 0.789 | 0.798 | 14.36 | 1 |
| *(Auxiliary 20-member store)* | `t4_03` | 20-Member Multi-Model Ensemble | Daymet, Maurer, NLDAS | `gauged__multi_camels/t4_03.zarr` | **0.820** (`0.8195`) | 0.786 | 0.801 | 14.22 | 1 |

---

## Build Prerequisites

The $\delta\text{HBV 1.1p}$ component uses the shared dHBV core (`models/_dhbv_common/core.py`), which contains C++/OpenMP (CPU) and CUDA (GPU) kernels. They are JIT-compiled with `torch.utils.cpp_extension.load_inline` on the first call, which takes a few minutes. The build is cached in `~/.cache/torch_extensions`. Set `TORCH_EXTENSIONS_DIR` to change the cache location, for example to a separate directory for each concurrent process. You need:
- **C++ compiler:** Linux `g++` with OpenMP. On macOS (Apple clang) the C++ kernels build without `-fopenmp` and run single-threaded.
- **`ninja`:** `pip install ninja`. torch cannot load C++ extensions without it.
- **GPU only:** an `nvcc` whose version matches `torch.version.cuda`, with `CUDA_HOME` pointing at that toolkit.

---

## Reproducing Results

```bash
# Print the tracked Stage-1 author reproduction report
python -m models.li2025_lstm_dhbv_ens.reproduce_author_metrics --mode verify

# Re-score the Stage-2 multi-model ensemble Zarr store t4_06 from GCS (needs GCS read credentials)
python -m models.li2025_lstm_dhbv_ens.reproduce_author_metrics --mode verify --verify-gcs
```

- `--verify-gcs` and `--mode evaluate_stage2` read from `gs://camels-us-mip`, which is not public, so they need GCS read credentials (Application Default Credentials).
- `--upload` (formerly `--write-gcs`) rebuilds `t4_06.zarr` from `t4_05.zarr` and `t4_24.zarr` and writes it back to the bucket before scoring. It is opt-in and maintainer-only, because it needs GCS write access.
- Any report the script writes goes to `--out`. The default is `models/li2025_lstm_dhbv_ens/stage1_author_reproduction_results.local.json`, which is not tracked by git. The tracked `stage1_author_reproduction_results.json` is overwritten only if `--out` points at it explicitly.
