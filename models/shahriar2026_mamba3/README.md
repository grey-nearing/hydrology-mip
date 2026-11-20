# Shahriar (2026) Mamba-3 Static-Dynamic Adapter

Reference:
> Shahriar, M. (2026). Can Mamba match LSTM for daily streamflow prediction? A parameter- and depth-matched benchmark on 671 CAMELS basins. *SSRN*, 7552815. https://doi.org/10.2139/ssrn.7552815
>
> Upstream Code & Zenodo Archive (`CC-BY-4.0` / `MIT`): https://doi.org/10.5281/zenodo.23087719

## Architecture & Upstream Training Parity
- **Backbone (`Mamba3Regressor`, $320{,}641$ trainable parameters)**:
  - Dynamic embedding: `Linear(5, 112)` projecting standardized Daymet daily forcings `['dayl', 'prcp', 'srad', 'tmean', 'vp']` to hidden dimension $d = 112$.
  - 2 stacked selective state-space layers (`d_model=112`, `d_state=32`, `d_conv=4`, `expand=2`, pre-`LayerNorm`, residual dropout $p = 0.10$), each followed immediately (`block(x)` then `adapter(x, x_stat)`) by a **Static-Dynamic Adapter** (`Linear(27 -> 64) + SiLU + Linear(64 -> 224)` shifting/scaling `[gamma, beta]` initialized to zeros so $x \leftarrow x + \tanh(\gamma) \odot \text{LN}(x) + 0.1\,\beta$).
  - Readout: final `LayerNorm(112) + Linear(112, 1)` mapped from standardized target space back to daily specific discharge (`mm/day`) and clipped non-negative at inference.
- **Exact `hydroDL` Training Pipeline (`runner.py`)**:
  - Linear Gaussian z-score standardization on all dynamic forcings, 27 static CAMELS catchment attributes, and specific discharge (`mm/day`) fit strictly on training-period observations of `train_basins`.
  - Sequence sampling: `rho = 365` scored days preceded by `365` meteorological spin-up days (`730`-day forward window; continuous full-period rollout evaluation with the leading `365` spin-up days discarded).
  - Exact `hydroDL` epoch iteration formula: $\text{nIterEp} = \lceil \ln(0.01) / \ln(1 - B \cdot \rho / (N_{\text{basins}} \cdot (N_{\text{days}} - \rho))) \rceil$ (`361` steps/epoch on 671-basin Split-3 = $10{,}830$ updates; `170` steps/epoch on 531-basin Newman Split-1 = $5{,}100$ updates at `batch_size = 64`).
  - Optimizer & loss: per-basin variance-weighted `NSELossBatch` ($\epsilon = 0.1$ on training-standardized target std), `AdamW` ($\text{lr} = 3 \times 10^{-3} \to 3 \times 10^{-4}$ cosine decay after 5-epoch linear warmup, $\beta = (0.9, 0.95)$, decoupled weight decay $10^{-3}$ excluding bias/norm/SSM `A_log`/`D` parameters, gradient norm clip `0.5`, single author seed `111111`, `30` epochs).

## Stage-1 Author-Protocol Confirmation Run (`status: "PASS"`)
- **Author Protocol (Split-3)**: `671` CAMELS-US basins (evaluated on both all `671` basins and the canonical `531`-basin Newman subset), trained on `1980-10-01` to `1995-09-30` (`WY1981–1995`), evaluated on `1995-10-01` to `2010-09-30` (`WY1996–2010`), Daymet forcings, single seed `111111`, 30 epochs × 361 steps/epoch (`10,830` gradient updates).
- **Published Target**: Median NSE = **`0.7272`** (`671` basins) and **`0.7307`** (`531` basins).
- **Zenodo Archive Verification (`pred_daily_m3s.npy` converted cleanly via $\times 86.4 / A_{\text{GAGES-II, km}^2}$ to `mm/day`)**:
  - `671` basins: Median NSE = **`0.7272`**, Mean NSE = `0.6616`, Median KGE = `0.7636`, `NSE <= 0` = `13` ($\Delta = 0.0000$).
  - `531` basins: Median NSE = **`0.7307`**, Mean NSE = `0.6714`, Median KGE = `0.7689`, `NSE <= 0` = `6` ($\Delta = 0.0000$).
- **Clean-Room Re-Training From Scratch (`gs://camels-us-mip/stage1_author_reproductions/shahriar2026_mamba3.zarr`)**:
  - `671` basins: Median NSE = **`0.7181`**, Mean NSE = `0.6507`, Median KGE = `0.7552`, `NSE <= 0` = `16` ($\Delta = -0.0091$).
  - `531` basins: Median NSE = **`0.7235`**, Mean NSE = `0.6632`, Median KGE = `0.7659`, `NSE <= 0` = `9` ($\Delta = -0.0072$, `status: "PASS"`).
  - *(MIP standardization note)*: Clean-room runs use the standardized 27 Kratzert et al. (2019b) static catchment attributes from `camels_{clim,topo,soil,geol,vege}.txt` (excluding `gauge_lat`/`gauge_lon` used in Shahriar's 17-attribute subset) and `mamba_ssm` CUDA selective scan (`Mamba` with matching `d_model=112, d_state=32, d_conv=4, expand=2`).

## Stage-2 CAMELS-US MIP Protocol (`gauged__single_camels`, Table 3)
- **Zarr Store**: `gs://camels-us-mip/hydrographs/gauged__single_camels/shahriar2026_mamba3_daymet.zarr` (`is_ensemble = false`, `num_members = 1`, `seed = 111111`, 30 epochs × 170 steps = `5,100` gradient updates, trained `1999-10-01` to `2008-09-30` with `365`-day spin-up, evaluated `1989-10-01` to `1999-09-30` across all `531` canonical basins in `mm/day`).
- **Leaderboard Row (`shahriar2026_mamba3_daymet`, `Mamba-3 Static-Dynamic Adapter`)**:
  - **Single-Model Median NSE**: **`0.718`** (`0.7182`), Mean NSE = **`0.675`** (`0.6752`), `NSE <= 0` = **`6`** (`531 / 531` basins, `0` NaNs).
  - **Secondary Gauch et al. (2023) Metrics**: Median KGE = **`0.737`** (`0.7374`), Median $|\text{FHV}|$ = **`15.20%`** (FHV = `-14.06%`), FMS = `-7.27%`, FLV = `-57.34%`, Alpha-NSE = `0.8641`, Beta-NSE = `-0.0291`, Pearson-$r$ = `0.8641`, Beta-KGE = `0.9390`, RMSE = `1.2958 mm/day`, Peak-Timing = `0.3333 days`.
