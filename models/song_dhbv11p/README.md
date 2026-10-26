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

$\delta\text{HBV 1.1p}$ (also denoted $\delta\text{MG}$ in the MHPI `generic_deltamodel` framework; Song et al., 2024) extends the static differentiable HBV 1.0 architecture of Feng et al. (2022) by introducing:

1. **Capillary Rise Flux ($C$):** An upward moisture flux from the upper groundwater zone ($S_{\text{UZ}}$) back into the root-zone soil moisture store ($S_{\text{soil}}$) when soil moisture is below field capacity ($\text{FC}$):
   $$R_{\text{up},t} = \min\!\left(S_{\text{UZ},t},\; C \cdot S_{\text{UZ},t} \cdot \max\!\left(0,\; 1 - \frac{S_{\text{soil},t}}{\text{FC}}\right)\right), \quad C \in [0.0001, 1.0]\text{ day}^{-1}$$
2. **Dynamic (Time-Varying) Parameterization ($\beta_t, K_{0,t}, \gamma_t$):** At each daily timestep $t$ and across each of the $N_{\text{mul}} = 16$ sub-grid HBV response units, the LSTM parameter network outputs three time-varying physical parameters:
   - $\beta_t \in [1.0, 6.0]$ (`parBETA`): Nonlinear soil moisture recharge exponent in $(S_{\text{soil},t}/\text{FC})^{\beta_t}$.
   - $K_{0,t} \in [0.05, 0.99]\text{ day}^{-1}$ (`parK0`): Near-surface rapid runoff recession coefficient above threshold $\text{UZL}$.
   - $\gamma_t = \beta_{\text{ET},t} \in [0.3, 5.0]$ (`parBETAET`): Nonlinear evapotranspiration shape exponent in
     $$E_{\text{act},t} = \min\!\left(S_{\text{soil},t},\; \text{PET}_t \cdot \min\!\left(\left(\frac{S_{\text{soil},t}}{\text{FC}\cdot\text{LP}}\right)^{\gamma_t},\; 1.0\right)\right)$$
3. **Parameter Estimation Network ($g_{\text{NN}}$):** A 1-layer, 256-unit DropConnect LSTM (`dropout=0.5`) receiving z-score normalized dynamic meteorological forcings (with precipitation pre-transformed via $\log_{10}(\sqrt{P_t} + 0.1)$) concatenated with 27 static CAMELS-US catchment attributes. The network outputs:
   - **Static head:** $16 \times 14 + 2 = 226$ time-averaged sigmoid-activated outputs mapped to the 14 static HBV bounds per sub-bucket (`TT`, `CFMAX`, `CFR`, `CWH`, `FC`, `LP`, `K0` baseline, `K1`, `K2`, `UZL`, `PERC`, `BETA` baseline, `C`, `BETAET` baseline) and 2 basin-wide Gamma unit hydrograph routing parameters ($\alpha_{\text{UH}} \in [1.0, 2.9]$, $\theta_{\text{UH}} \in [0.01, 6.5]$).
   - **Dynamic head:** $16 \times 3 = 48$ daily sigmoid-activated outputs overriding $(\beta_t, K_{0,t}, \gamma_t)$ at every timestep $t$.
4. **Routing & Objective Function:** Sub-grid runoff averaged across all $N_{\text{mul}} = 16$ buckets is routed through a 15-day causal Gamma unit hydrograph (`uh_gamma_fast` and `uh_conv_fast`). Models are trained end-to-end for 50 epochs using `Adadelta(lr=1.0, rho=0.9, eps=1e-6)` on `batch_size=100` random basin-time windows of length $\rho = 365$ days preceded by $W = 365$ warmup spinup days, minimizing the basin-std normalized squared error loss (`NseBatchLoss`, $\epsilon = 0.1$).

---

## Stage-1 Author Protocol Reproduction

Song et al. (2024, *HESS* 28, 3051–3077, Section 2.4 & Figure 2) evaluated $\delta\text{HBV}$ on **Split-3** (`1980-10-01` to `1995-09-30` training; `1995-10-01` to `2010-09-30` testing), reporting a single-seed median NSE of **0.750** for dynamic $\delta\text{HBV 1.1p}$ (and **0.730** for static $\delta\text{HBV 1.0}$). In addition, the upstream MHPI `generic_deltamodel` repository released a pre-trained Split-1 (`1999-10-01` to `2008-09-30` train / `1989-10-01` to `1999-09-30` test, Daymet, 531 basins) benchmark achieving **0.7574** (10-seed ensemble) and **0.7548** (single seed).

| Verification Protocol | Split (Train / Test) | Forcing & Basins | Published Med NSE | Reproduced 1-Seed Med NSE | Reproduced Ensemble Med NSE | $\Delta$ (1-Seed) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Author Split-3 (Maurer 531)** | WY1980–1995 / WY1995–2010 | Maurer (531 basins) | 0.750 (0.730 $\delta\text{HBV 1.0}$) | **0.733** (`0.7333`) | **0.746** (`0.7460`, 2 seeds) | `-0.017` | **PASS** |
| **Author Split-3 (Daymet 671)** | WY1980–1995 / WY1995–2009 | Daymet (670 basins) | 0.750 | **0.733** (`0.7330`) | **0.744** (`0.7440`, 2 seeds) | `-0.017` | **PASS** |
| **Author Split-3 (Daymet 531)** | WY1980–1995 / WY1995–2010 | Daymet (531 basins) | 0.750 | **0.741** (`0.7407`) | **0.756** (`0.7556`, 2 seeds) | `-0.009` | **PASS** |
| **Split-1 MHPI Checkpoint** | WY1999–2008 / WY1989–1999 | Daymet (531 basins) | 0.757 (`0.7574` ens / `0.7548` 1-seed) | **0.754** (`0.7537 ± 0.0011`, 5 seeds) | **0.759** (`0.7589`, 5 seeds) | `-0.001` (`+0.002` ens) | **PASS** |

---

## Stage-2 CAMELS-US MIP Leaderboard Summary

All Stage-2 runs use the standardized CAMELS-US MIP Split-1 temporal window (`1999-10-01` to `2008-09-30` training; `1989-10-01` to `1999-09-30` evaluation across 531 basins and 3,652 days) with $M = 5$ independent random seeds (`seeds = [111, 222, 333, 444, 555]`). Both the 5-member ensemble mean (`evaluation_mode: "ensemble"`) and the mean across all 5 individual members (`evaluation_mode: "single_member_mean"`) are evaluated from the same 5-member GCS Zarr v2 stores (`qsim_members` shape `(5, 531, 3652)`).

| Table / Leaderboard | Row ID | Evaluation Mode | Forcings | GCS Zarr Store | Med NSE | Mean NSE | Med KGE | Med $|\text{FHV}|$ (%) | $\text{NSE} \le 0$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **T3** (`gauged__single_camels`) | `t3_18` | 5-Seed Ensemble | Daymet | `gauged__single_camels/t3_18.zarr` | **0.759** | 0.724 | 0.757 | 15.06 | 1 |
| **T3** (`gauged__single_camels`) | `t3_19` | Single-Model (5-seed mean) | Daymet | `gauged__single_camels/t3_18.zarr` | **0.754** (`±0.001`) | 0.719 | 0.753 | 15.35 | 3 |
| **T4** (`gauged__multi_camels`) | `t4_24` | 5-Seed Ensemble | Daymet, Maurer, NLDAS | `gauged__multi_camels/t4_24.zarr` | **0.799** | 0.763 | 0.778 | 13.98 | 2 |
| **T4** (`gauged__multi_camels`) | `t4_29` | Single-Model (5-seed mean) | Daymet, Maurer, NLDAS | `gauged__multi_camels/t4_24.zarr` | **0.793** (`±0.002`) | 0.756 | 0.774 | 14.07 | 3 |
| **T7** (`pub__single_camels`) | `t7_06` | 5-Seed Ensemble | Daymet | `pub__single_camels/t7_06.zarr` | **0.690** | 0.582 | 0.689 | 20.67 | 24 |
| **T7** (`pub__single_camels`) | `t7_11` | Single-Model (5-seed mean) | Daymet | `pub__single_camels/t7_06.zarr` | **0.686** (`±0.004`) | 0.547 | 0.681 | 20.78 | 23 |
| **T8** (`pub__multi_camels`) | `t8_09` | 5-Seed Ensemble | Daymet, Maurer, NLDAS | `pub__multi_camels/t8_09.zarr` | **0.739** | 0.621 | 0.709 | 20.72 | 19 |
| **T8** (`pub__multi_camels`) | `t8_11` | Single-Model (5-seed mean) | Daymet, Maurer, NLDAS | `pub__multi_camels/t8_09.zarr` | **0.727** (`±0.002`) | 0.576 | 0.686 | 21.48 | 27 |
| **T11** (`pur__single_camels`) | `t11_04` | 5-Seed Ensemble | Daymet | `pur__single_camels/t11_04.zarr` | **0.651** | 0.292 | 0.641 | 25.04 | 43 |
| **T11** (`pur__single_camels`) | `t11_04_single` | Single-Model (5-seed mean) | Daymet | `pur__single_camels/t11_04.zarr` | **0.634** (`±0.011`) | 0.350 | 0.630 | 24.44 | 42 |
| **T12** (`pur__multi_camels`) | `t12_07` | 5-Seed Ensemble | Daymet, Maurer, NLDAS | `pur__multi_camels/t12_07.zarr` | **0.691** | 0.368 | 0.664 | 22.95 | 35 |
| **T12** (`pur__multi_camels`) | `t12_07_single` | Single-Model (5-seed mean) | Daymet, Maurer, NLDAS | `pur__multi_camels/t12_07.zarr` | **0.677** (`±0.003`) | 0.312 | 0.625 | 25.69 | 49 |

---

## Reproducing Results

```bash
# Verify Stage-1 author reproduction report
python3 models/song_dhbv11p/reproduce_author_metrics.py --mode verify

# Run Stage-2 standardized MIP fold evaluation
python3 models/run_model.py \
  --model-dir models/song_dhbv11p \
  --regime gauged \
  --forcings single \
  --seeds 111 222 333 444 555
```
