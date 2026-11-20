# Differentiable HBV 1.0 ($\delta\text{HBV 1.0}$) — Feng et al. (2022)

## 1. Metadata & Provenance
- **Family ID**: `feng_dhbv10` (Table 2 Entry #13)
- **Model Name**: $\delta\text{HBV 1.0}$ (Dynamic-Parameter LSTM + 16-Bucket Differentiable HBV)
- **Category**: Differentiable / Hybrid
- **Provenance Category**: `MIP Re-run (Our Code)`
- **Canonical Citation**: Feng et al. (2022) — Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. *Water Resources Research*, 58(10), e2022WR032404.
- **Paper URL**: https://doi.org/10.1029/2022WR032404
- **Upstream GitHub Repository**: https://github.com/mhpi/hydroDL (and successor https://github.com/mhpi/generic_deltamodel)
- **Archive DOI**: `10.5281/zenodo.7091334` (Feng et al., 2022 source code archive; see also `10.5281/zenodo.7943626` for Feng et al., 2023 simulation outputs)
- **MIP Implementation Directory**: `models/feng_dhbv10/`

---

## 2. Model Architecture & Hydrological Formulation
$\delta\text{HBV 1.0}$ (network `StandaloneDHBV(version="1_0")` in `models/_dhbv_common/core.py`, trained by `DHBV10Trainer` in `models/feng_dhbv10/model.py`) couples a regional parameter network (a `Linear + ReLU` input layer feeding a single-layer 256-unit LSTM with DropConnect weight dropout `dr = 0.5` on the input and recurrent weights, followed by a linear readout) with $N_{\text{mul}} = 16$ parallel sub-basin conceptual HBV rainfall-runoff buckets routed through a shared 15-day Gamma unit hydrograph (`routing_len = 15`):

1. **LSTM Parameterization Head**:
   - **Inputs**: All raw CAMELS forcing channels of each forcing product (`prcp`, `srad`, `tmax`, `tmin`, `vp`), with every precipitation channel transformed as $\log_{10}(\sqrt{P_t} + 0.1)$, plus the derived daily mean temperature (`Tmean`) and Hargreaves (1985) `PET`. There is no `dayl` channel. These are z-scored with training-basin, post-spinup statistics and concatenated with the 27 standardized static catchment attributes at every daily timestep.
   - **Static HBV Bucket Parameters** ($11 \times 16 = 176$, read from the LSTM output at the **last timestep** of the input sequence and held time-invariant across the simulation; see the note below):
     - Degree-day snow routine: `TT` $\in [-2.5, 2.5]\;^\circ\text{C}$, `CFMAX` $\in [0.5, 10.0]\;\text{mm}\;^\circ\text{C}^{-1}\;\text{d}^{-1}$, `CFR` $\in [0.0, 0.1]$, `CWH` $\in [0.0, 0.2]$.
     - Soil moisture routine: `FC` $\in [50, 1000]\;\text{mm}$, `LP` $\in [0.2, 1.0]$.
     - Upper & lower response reservoirs: `K0` $\in [0.05, 0.9]\;\text{d}^{-1}$, `K1` $\in [0.01, 0.5]\;\text{d}^{-1}$, `K2` $\in [0.001, 0.2]\;\text{d}^{-1}$, `PERC` $\in [0.0, 10.0]\;\text{mm}\,\text{d}^{-1}$, `UZL` $\in [0.0, 100.0]\;\text{mm}$.
   - **Dynamic Time-Varying HBV Parameters** ($2 \times 16 = 32$, `dynamic_params = ["parBETA", "parBETAET"]`, predicted at every daily timestep $t$):
     - Soil moisture runoff nonlinearity exponent `BETA` ($\beta_t \in [1.0, 6.0]$).
     - Evapotranspiration shape exponent `BETAET` ($\gamma_t \in [0.3, 5.0]$), modifying potential evapotranspiration demand via $\text{ET}_t = \min\!\left(S_{\text{sm}, t},\; \text{PET}_t \cdot \min\!\left((S_{\text{sm}, t} / (\text{FC} \cdot \text{LP}))^{\gamma_t}, 1.0\right)\right)$.
   - **Shared Routing Parameters** ($2$, static per basin, also read from the last LSTM timestep):
     - Gamma unit hydrograph parameters $a \in [0, 2.9]$ and $b \in [0, 6.5]$ (`route_bounds`), discretized over a 15-day causal impulse response kernel via numerical quadrature.
   - **Note on static parameters (faithful to Feng et al., 2022):** The static HBV and routing parameters are taken from the LSTM output at the final timestep of whatever sequence the network sees (`StandaloneDHBV._descale_phy` and `forward` in `models/_dhbv_common/core.py`). They are therefore non-causal within a sequence and depend on its length: training uses warm-up + `rho`-day windows, whereas evaluation feeds the full spinup + test sequence. This matches the upstream MHPI implementation.
2. **Training Objective (`RmseCombLoss`)**:
   - Trained end-to-end with Adadelta (`lr = 1.0`, `rho = 0.9`, `eps = 1e-6`; the optimizer settings are hardcoded in `DHBV10Trainer.fit_and_predict`, so the runner's `lr` argument is currently ignored), 365-day warm-up + 365-day active sequence (`rho = 365`), `batch_size = 100`, and combined linear/log-sqrt RMSE loss ($\alpha = 0.25$):
     $$\mathcal{L} = 0.75\,\text{RMSE}(Q, \hat{Q}) + 0.25\,\text{RMSE}\!\left(\log_{10}(\sqrt{Q + 10^{-6}} + 0.1),\; \log_{10}(\sqrt{\hat{Q} + 10^{-6}} + 0.1)\right)$$

---

## 3. Stage-1 Author-Protocol Confirmation (`Split-3`: `WY1980–1995` Train / `WY1995–2010` Test)
Feng et al. (2022) trained $\delta\text{HBV 1.0}$ on **Split-3** (`1980-10-01` to `1995-09-30` training, `1995-10-01` to `2010-09-30` evaluation) across all 671 CAMELS-US basins using Daymet forcing, reporting a single-seed Median NSE of **`0.733`** (`0.726` with static parameters) and an ensemble Median NSE of **`0.740`** (Song et al., 2025 Table 1).

Our 3-seed Split-3 confirmation run (`seeds = [111111, 111112, 111113]`, 50 epochs) achieves a clean **`PASS`** across both the 14-water-year MHPI post-warmup evaluation window (`1996-10-01` to `2010-09-30`, where the first test year `WY1996` serves as internal state spin-up) and the full 15-water-year window (`1995-10-01` to `2010-09-30` with external pre-test spin-up):

| Evaluation Window & Basin Set (Split-3, Daymet) | Basins | Published Single / Ens Med NSE | Reproduced Single Med NSE (3 seeds) | Reproduced 3-Seed Ens Med NSE | $\Delta$ Single / Ens | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **14-WY Post-Warmup (`1996-10-01` to `2010-09-30`, All 671)** | 671 | `0.733` / `0.740` | **`0.7340 ± 0.0009`** (`[0.7347, 0.7327, 0.7347]`) | **`0.7407`** | `+0.0010` / `+0.0007` | **PASS** |
| **15-WY Full Window (`1995-10-01` to `2010-09-30`, All 671)** | 671 | `0.733` / `0.740` | **`0.7280 ± 0.0008`** (`[0.7273, 0.7277, 0.7291]`) | **`0.7376`** | `-0.0050` / `-0.0024` | **PASS** |
| **14-WY Post-Warmup (`1996-10-01` to `2010-09-30`, 531 Subset)** | 531 | `0.733` / `0.740` | **`0.7389 ± 0.0015`** (`[0.7375, 0.7381, 0.7410]`) | **`0.7436`** | `+0.0059` / `+0.0036` | **PASS** |
| **15-WY Full Window (`1995-10-01` to `2010-09-30`, 531 Subset)** | 531 | `0.733` / `0.740` | **`0.7339 ± 0.0010`** (`[0.7331, 0.7332, 0.7353]`) | **`0.7412`** | `+0.0009` / `+0.0012` | **PASS** |

---

## 4. Stage-2 CAMELS-US MIP Protocol Results (`Split-1`: `WY1999–2008` Train / `WY1989–1999` Test)
Under the standardized CAMELS-US MIP protocol (`1999-10-01` to `2008-09-30` training, `1989-10-01` to `1999-09-30` evaluation, 531 canonical basins, 5 random seeds `[111, 222, 333, 444, 555]`), `feng_dhbv10` is evaluated across all 3 single-forcing Daymet leaderboards (Supplement Tables S1, S4, and S7), emitting both the 5-model ensemble row (`evaluation_mode="ensemble"`) and the competing single-model row (`evaluation_mode="single_member_mean"` scored across all 5 seeds):

| Leaderboard | Row ID | Mode | Forcing | Seeds | Med NSE | Zarr Store |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Table S1** (`gauged__single_camels`) | `t3_29` | 5-model ensemble | Daymet | 5 | **0.742** (`0.7416`) | `gauged__single_camels/t3_29.zarr` |
| **Table S1** (`gauged__single_camels`) | `t3_33` | Single model (mean of 5 seeds) | Daymet | 5 | **0.733 ± 0.002** (`0.7327`) | `gauged__single_camels/t3_29.zarr` |
| **Table S4** (`pub__single_camels`) | `t7_10` | 5-model ensemble | Daymet | 5 | **0.681** (`0.6806`) | `pub__single_camels/t7_10.zarr` |
| **Table S4** (`pub__single_camels`) | `t7_14` | Single model (mean of 5 seeds) | Daymet | 5 | **0.673 ± 0.005** (`0.6735`) | `pub__single_camels/t7_10.zarr` |
| **Table S7** (`pur__single_camels`) | `t11_06` | 5-model ensemble | Daymet | 5 | **0.634** (`0.6336`) | `pur__single_camels/t11_06.zarr` |
| **Table S7** (`pur__single_camels`) | `t11_10` | Single model (mean of 5 seeds) | Daymet | 5 | **0.620 ± 0.003** (`0.6200`) | `pur__single_camels/t11_06.zarr` |

---

## 5. Build Prerequisites
The shared dHBV core (`models/_dhbv_common/core.py`, also used by `song_dhbv11p`) contains C++/OpenMP (CPU) and CUDA (GPU) kernels. They are JIT-compiled with `torch.utils.cpp_extension.load_inline` on the first call, which takes a few minutes. The build is cached in `~/.cache/torch_extensions`. Set `TORCH_EXTENSIONS_DIR` to change the cache location, for example to a separate directory for each concurrent process. You need:
- **C++ compiler:** Linux `g++` with OpenMP. On macOS (Apple clang) the C++ kernels build without `-fopenmp` and run single-threaded.
- **`ninja`:** `pip install ninja`. torch cannot load C++ extensions without it.
- **GPU only:** an `nvcc` whose version matches `torch.version.cuda`, with `CUDA_HOME` pointing at that toolkit.

---

## 6. Reproduction Commands
```bash
# Print the tracked Stage-1 report
python -m models.feng_dhbv10.reproduce_author_metrics --mode verify

# Re-train and evaluate on Split-3 from scratch
python -m models.feng_dhbv10.reproduce_author_metrics --mode train_and_eval --out outputs/feng_stage1.json
```
- `--mode train_and_eval` writes to `--out`. The default is `models/feng_dhbv10/stage1_author_reproduction_results.local.json`, which is not tracked by git. The tracked `stage1_author_reproduction_results.json` is overwritten only if `--out` points at it explicitly.
- The Split-3 run uses 274 spinup days, because the data bundle starts on `1980-01-01` and training starts on `1980-10-01`.
