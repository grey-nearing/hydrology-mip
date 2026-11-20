# Song et al. (2024) — Differentiable HBV 1.1p ($\delta\text{HBV 1.1p}$ / $\delta\text{MG}$)

## Canonical Metadata & Provenance

| Field | Value |
| :--- | :--- |
| **Directory / Family ID** | `models/song_dhbv11p` (`song_dhbv11p`) |
| **Canonical Model Name** | `δHBV 1.1p / δMG` (`Song et al. (2024) δHBV 1.1p (Dynamic Parameters)`) |
| **Model Category** | Differentiable / Hybrid |
| **Provenance** | `MIP Re-run (Our Code)` (Clean-room PyTorch + C++/OpenMP autograd re-implementation) |
| **Canonical Citation** | Song et al. (2024), *Hydrology and Earth System Sciences*, 28(13), 3051–3077 |
| **Paper DOI / URL** | [https://doi.org/10.5194/hess-28-3051-2024](https://doi.org/10.5194/hess-28-3051-2024) |
| **Upstream Reference Repo** | [https://github.com/mhpi/generic_deltamodel](https://github.com/mhpi/generic_deltamodel) |

---

## Model Architecture

$\delta\text{HBV 1.1p}$ (also denoted $\delta\text{MG}$ in the MHPI `generic_deltamodel` framework; Song et al., 2024) extends the differentiable HBV 1.0 architecture of Feng et al. (2022), which has dynamic $\beta_t$ and $\gamma_t$ only, by introducing:

1. **Capillary Rise Flux ($C$):** An upward moisture flux from the lower groundwater zone ($S_{\text{LZ}}$) back into the root-zone soil moisture store ($S_{\text{soil}}$) when soil moisture is below field capacity ($\text{FC}$):
   $$R_{\text{up},t} = \min\!\left(S_{\text{LZ},t},\; C \cdot S_{\text{LZ},t} \cdot \left(1 - \min\!\left(1,\; \frac{S_{\text{soil},t}}{\text{FC}}\right)\right)\right), \quad C \in [0, 1]\text{ day}^{-1}$$
2. **Dynamic (Time-Varying) Parameterization ($\beta_t, K_{0,t}, \gamma_t$):** At each daily timestep $t$ and across each of the $N_{\text{mul}} = 16$ sub-grid HBV response units, the LSTM parameter network outputs three time-varying physical parameters (`dynamic_params = ["parBETA", "parK0", "parBETAET"]`):
   - $\beta_t \in [1.0, 6.0]$ (`parBETA`): Nonlinear soil moisture recharge exponent in $(S_{\text{soil},t}/\text{FC})^{\beta_t}$.
   - $K_{0,t} \in [0.05, 0.9]\text{ day}^{-1}$ (`parK0`): Near-surface rapid runoff recession coefficient above threshold $\text{UZL}$.
   - $\gamma_t = \beta_{\text{ET},t} \in [0.3, 5.0]$ (`parBETAET`): Nonlinear evapotranspiration shape exponent in
     $$E_{\text{act},t} = \min\!\left(S_{\text{soil},t},\; \text{PET}_t \cdot \min\!\left(\left(\frac{S_{\text{soil},t}}{\text{FC}\cdot\text{LP}}\right)^{\gamma_t},\; 1.0\right)\right)$$
3. **Parameter Estimation Network ($g_{\text{NN}}$):** `StandaloneDHBV11p` in `models/_dhbv_common/core.py` (trained by `models/song_dhbv11p/model.py`): a `Linear + ReLU` input layer feeding a 1-layer, 256-unit DropConnect LSTM (`dropout=0.5`). Its inputs are the z-score normalized **raw** CAMELS forcing channels (`prcp`, `srad`, `tmax`, `tmin`, `vp`; precipitation is not transformed, unlike `feng_dhbv10`), plus the derived daily mean temperature and Hargreaves PET, concatenated with 27 static CAMELS-US catchment attributes. A single linear readout produces, at every timestep, $16 \times 14 + 2 = 226$ sigmoid-activated outputs:
   - **Static parameters:** the 11 static HBV parameters per sub-bucket (`TT` $\in [-2.5, 2.5]$, `CFMAX`, `CFR`, `CWH`, `FC`, `LP`, `K1`, `K2`, `UZL`, `PERC`, `C`) and 2 basin-wide Gamma unit hydrograph routing parameters ($a \in [0, 2.9]$, $b \in [0, 6.5]$). Static parameters and routing parameters are read from the LSTM output at the **last timestep** of the input sequence.
   - **Dynamic parameters:** for the $16 \times 3 = 48$ entries of $(\beta_t, K_{0,t}, \gamma_t)$, the LSTM output at every timestep $t$ is used.
   - **Note on static parameters (faithful to the MHPI implementation):** Because static and routing parameters come from the final timestep (`StandaloneDHBV._descale_phy` and `forward`), they are non-causal within a sequence and depend on its length: training uses warm-up + `rho`-day windows, whereas evaluation feeds the full spinup + test sequence.
4. **Routing & Objective Function:** Sub-grid runoff averaged across all $N_{\text{mul}} = 16$ buckets is routed through a 15-day causal Gamma unit hydrograph (`uh_gamma_fast` and `uh_conv_fast`). Models are trained end-to-end for 50 epochs using `Adadelta(lr=1.0, rho=0.9, eps=1e-6)` (hardcoded in the trainer; the runner's `lr` argument is currently ignored) on `batch_size=100` random basin-time windows of length $\rho = 365$ days preceded by $W = 365$ warmup spinup days, minimizing the basin-std normalized squared error loss (`NseBatchLoss`, $\epsilon = 0.1$).

---

## Stage-1 Author Protocol Reproduction

Song et al. (2024, *HESS* 28, 3051–3077, Section 2.4 & Figure 2) evaluated $\delta\text{HBV}$ on **Split-3** (`1980-10-01` to `1995-09-30` training; `1995-10-01` to `2010-09-30` testing), reporting a single-seed median NSE of **0.750** for dynamic $\delta\text{HBV 1.1p}$ (and **0.730** for static $\delta\text{HBV 1.0}$). In addition, the upstream MHPI `generic_deltamodel` repository released a pre-trained Split-1 (`1999-10-01` to `2008-09-30` train / `1989-10-01` to `1999-09-30` test, Daymet, 531 basins) benchmark achieving **0.7574** (10-seed ensemble) and **0.7548** (single seed).

| Verification Protocol | Split (Train / Test) | Forcing & Basins | Published Med NSE | Reproduced 1-Seed Med NSE | Reproduced Ensemble Med NSE | $\Delta$ (1-Seed) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Author Split-3 (Maurer 531)** | WY1980–1995 / WY1995–2008 | Maurer (531 basins) | 0.750 (0.730 $\delta\text{HBV 1.0}$) | **0.733** (`0.7333 ± 0.0024`, 3 seeds) | **0.739** (`0.7389`, 3 seeds) | `-0.017` | **PASS** |
| **Author Split-3 (Daymet 671)** | WY1980–1995 / WY1995–2010 | Daymet (671 basins) | 0.750 | **0.737** (`0.7371 ± 0.0021`, 3 seeds) | **0.743** (`0.7430`, 3 seeds) | `-0.013` | **PASS** |
| **Author Split-3 (Daymet 531)** | WY1980–1995 / WY1995–2010 | Daymet (531 basins) | 0.750 | **0.742** (`0.7418 ± 0.0009`, 3 seeds) | **0.748** (`0.7475`, 3 seeds) | `-0.008` | **PASS** |
| **Split-1 MHPI Checkpoint** | WY1999–2008 / WY1989–1999 | Daymet (531 basins) | 0.757 (`0.7574` ens / `0.7548` 1-seed) | **0.754** (`0.7537 ± 0.0011`, 5 seeds) | **0.759** (`0.7589`, 5 seeds) | `-0.001` (`+0.002` ens) | **PASS** |

---

## Stage-2 CAMELS-US MIP Leaderboard Summary

All Stage-2 runs use the standardized CAMELS-US MIP Split-1 temporal window (`1999-10-01` to `2008-09-30` training; `1989-10-01` to `1999-09-30` evaluation across 531 basins and 3,652 days) with $M = 5$ independent random seeds (`seeds = [111, 222, 333, 444, 555]`). Both the 5-member ensemble mean (`evaluation_mode: "ensemble"`) and the mean across all 5 individual members (`evaluation_mode: "single_member_mean"`) are evaluated from the same 5-member GCS Zarr v2 stores (`qsim_members` shape `(5, 531, 3652)`).

| Table / Leaderboard | Row ID | Evaluation Mode | Forcings | Zarr Store | Med NSE | Mean NSE | Med KGE | Med $|\text{FHV}|$ (%) | $\text{NSE} \le 0$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** (`gauged__single_camels`) | `t3_18` | 5-Seed Ensemble | Daymet | `gauged__single_camels/t3_18.zarr` | **0.759** | 0.724 | 0.757 | 15.06 | 1 |
| **S1** (`gauged__single_camels`) | `t3_19` | Single-Model (5-seed mean) | Daymet | `gauged__single_camels/t3_18.zarr` | **0.754** (`±0.001`) | 0.717 | 0.756 | 15.08 | 2 |
| **S2** (`gauged__multi_camels`) | `t4_24` | 5-Seed Ensemble | Daymet, Maurer, NLDAS | `gauged__multi_camels/t4_24.zarr` | **0.799** | 0.763 | 0.778 | 13.98 | 2 |
| **S4** (`pub__single_camels`) | `t7_06` | 5-Seed Ensemble | Daymet | `pub__single_camels/t7_06.zarr` | **0.690** | 0.582 | 0.689 | 20.67 | 24 |
| **S4** (`pub__single_camels`) | `t7_11` | Single-Model (5-seed mean) | Daymet | `pub__single_camels/t7_06.zarr` | **0.686** (`±0.004`) | 0.561 | 0.681 | 20.74 | 24 |
| **S7** (`pur__single_camels`) | `t11_04` | 5-Seed Ensemble | Daymet | `pur__single_camels/t11_04.zarr` | **0.650** | 0.292 | 0.640 | 25.04 | 43 |
| **S7** (`pur__single_camels`) | `t11_04_single` | Single-Model (5-seed mean) | Daymet | `pur__single_camels/t11_04.zarr` | **0.634** (`±0.011`) | 0.249 | 0.633 | 24.93 | 48 |

---

## Build Prerequisites

The shared dHBV core (`models/_dhbv_common/core.py`, also used by `feng_dhbv10`) contains C++/OpenMP (CPU) and CUDA (GPU) kernels. They are JIT-compiled with `torch.utils.cpp_extension.load_inline` on the first call, which takes a few minutes. The build is cached in `~/.cache/torch_extensions`. Set `TORCH_EXTENSIONS_DIR` to change the cache location, for example to a separate directory for each concurrent process. You need:
- **C++ compiler:** Linux `g++` with OpenMP. On macOS (Apple clang) the C++ kernels build without `-fopenmp` and run single-threaded.
- **`ninja`:** `pip install ninja`. torch cannot load C++ extensions without it.
- **GPU only:** an `nvcc` whose version matches `torch.version.cuda`, with `CUDA_HOME` pointing at that toolkit.

---

## Reproducing Results

```bash
# Print the tracked Stage-1 author reproduction report
python -m models.song_dhbv11p.reproduce_author_metrics --mode verify

# Re-score the canonical Split-1 Zarr store t3_18 from GCS (needs GCS read credentials)
python -m models.song_dhbv11p.reproduce_author_metrics --mode verify --verify-gcs

# Re-train and evaluate on Split-3 from scratch
python -m models.song_dhbv11p.reproduce_author_metrics --mode train_and_eval --split split3 --forcing daymet --out outputs/song_stage1.json

# Run the Stage-2 standardized MIP fold evaluation
python -m models.run_model --model song_dhbv11p --spatial-regime gauged --forcings daymet --seeds 111 222 333 444 555
```

- `reproduce_author_metrics.py` writes its report to `--out`. The default is `models/song_dhbv11p/stage1_author_reproduction_results.local.json`, which is not tracked by git. The tracked `stage1_author_reproduction_results.json` is overwritten only if `--out` points at it explicitly.
- `--verify-gcs` reads from `gs://camels-us-mip`, which is not public, so it needs GCS read credentials (Application Default Credentials).
- The Split-3 `train_and_eval` run uses 274 spinup days, because the data bundle starts on `1980-01-01` and training starts on `1980-10-01`.
