# Shahriar (2026) Mamba-3 Static-Dynamic Adapter

Reference:
> Shahriar, M. (2026). Can Mamba match LSTM for daily streamflow prediction? A parameter- and depth-matched benchmark on 671 CAMELS basins. *SSRN*, 7552815. https://doi.org/10.2139/ssrn.7552815
>
> Upstream Code & Zenodo Archive (`CC-BY-4.0` / `MIT`): https://doi.org/10.5281/zenodo.23087719

## Architecture (`model.py`, `MambaStreamflowModelAdapter`)
Configuration matches the Zenodo `Tier1_Ep30_Variant-adapter_V3_head32_H80_L3_dConv4` run read by `reproduce_author_metrics.py`. With the `runner.py` `RUNNER_DEFAULTS` (5 dynamic inputs, 27 static attributes) the model has **208,706 trainable parameters**.
- **Input embedding**: `Linear(5, 80)` followed by `RMSNorm(80)`, applied to standardized Daymet daily forcings `['dayl', 'prcp', 'srad', 'tmean', 'vp']` (`tmean = 0.5 * (tmax + tmin)`; `dayl` from the CAMELS slice `x_aux`). Hidden size $H = 80$.
- **3 layers**, each applying, in order:
  1. A pre-norm residual Mamba-3 block: $x \leftarrow x + s \cdot \text{Dropout}_{0.11}(\text{Mamba3}(\text{RMSNorm}(x)))$, with a learnable scalar residual scale $s$ initialized to `0.1`. `Mamba3(d_model=80, d_state=64, expand=2, headdim=32)` (SISO, `ngroups=1`, `rope_fraction=0.5`) uses exponential-trapezoidal discretization and a rotary (complex-valued) state, with no causal `conv1d` (the `d_conv=4` argument is accepted and ignored).
  2. A sigmoid static-gated bottleneck adapter: $x \leftarrow x + \text{Adapter}(x) \odot \sigma(\text{Gate}(x_{\text{stat}}))$, where `Adapter = Linear(80, 64) + GELU + Dropout(0.11) + Linear(64, 80)` and `Gate = Linear(27, 64) + GELU + Linear(64, 80)`.
- **Readout**: `Linear(80, 1)` in standardized target space. It is inverse-scaled to daily specific discharge (`mm/day`) and clipped non-negative at inference.
- **Implementation**: `mamba3.py` is a pure-PyTorch Mamba-3 (chunked scan). It does not depend on `mamba_ssm` and uses no custom CUDA or Triton kernels, so it runs on CPU and GPU.

## Training (`runner.py`, `hydroDL` protocol)
- Linear Gaussian z-score standardization of the dynamic forcings, the 27 static CAMELS attributes, and specific discharge (`mm/day`). Scalers are fit only on the training-period target days of `train_basins`.
- Sequence sampling: mini-batches of `batch_size = 128` windows, each `rho = 365` target days preceded by a `365`-day warm-up buffer (`730`-day forward window; loss on the trailing `365` days only). Evaluation is a continuous full-period rollout with the leading spin-up days discarded.
- Iterations per epoch follow `hydroDL`: $\text{nIterEp} = \lceil \ln(0.01) / \ln(1 - B \rho / (N_{\text{basins}} T)) \rceil$, where $T$ is the number of training target days. At $B = 128$ this gives `361` steps/epoch for the 671-basin Split-3 period (`T = 5,478`; `10,830` updates over 30 epochs) and `170` steps/epoch for the 531-basin Stage-2 period (`T = 3,288`; `5,100` updates).
- Loss: per-basin variance-weighted `NSELossBatch` ($\epsilon = 0.1$ added to each basin's standardized target std). A non-finite loss raises `FloatingPointError`; there is no step skipping or NaN replacement of predictions.
- Optimizer: `AdamW` with $\text{lr} = 3 \times 10^{-3}$, 5-epoch linear warmup then cosine decay to $3 \times 10^{-4}$, $\beta = (0.9, 0.95)$, decoupled weight decay $10^{-3}$ on 2-D weights only (no decay on biases, norm weights, residual scales, or the SSM `D`/`dt_bias`/`B`-bias/`C`-bias parameters; Mamba-3's `A` is input-dependent, projected through `in_proj`), gradient-norm clip `0.5`, single seed `111111`, `30` epochs.

## Stage-1 Author-Protocol Confirmation Run (`status: "PASS"`)
- **Author Protocol (Split-3)**: `671` CAMELS-US basins (scored on all `671` and on the canonical `531`-basin Newman subset), trained on `1980-10-01` to `1995-09-30` (`WY1981–1995`), evaluated on `1995-10-01` to `2010-09-30` (`WY1996–2010`), Daymet forcings, single seed `111111`, 30 epochs × 361 steps/epoch (`10,830` gradient updates). The clean-room run uses a `274`-day spin-up (`1980-01-01` to `1980-09-30`, the forcing available before the training period).
- **Published Target**: Median NSE = **`0.7272`** (`671` basins) and **`0.7307`** (`531` basins).
- **Zenodo Archive Verification** (`pred_daily_m3s.npy` converted to `mm/day` as $q \times 86.4 / A$, where $A$ is the bundle `area_km2`, i.e. the catchment area from the Daymet forcing-file header):
  - `671` basins: Median NSE = **`0.7272`**, Mean NSE = `0.6616`, Median KGE = `0.7636`, `NSE <= 0` = `13` ($\Delta = 0.0000$).
  - `531` basins: Median NSE = **`0.7307`**, Mean NSE = `0.6714`, Median KGE = `0.7689`, `NSE <= 0` = `6` ($\Delta = 0.0000$).
- **Clean-Room Re-Training From Scratch (`gs://camels-us-mip/stage1_author_reproductions/shahriar2026_mamba3.zarr`)**:
  - `671` basins: Median NSE = **`0.7181`**, Mean NSE = `0.6507`, Median KGE = `0.7552`, `NSE <= 0` = `16` ($\Delta = -0.0091$).
  - `531` basins: Median NSE = **`0.7235`**, Mean NSE = `0.6632`, Median KGE = `0.7659`, `NSE <= 0` = `9` ($\Delta = -0.0072$, `status: "PASS"`).
  - *(MIP standardization note)*: Clean-room runs use the standardized 27 Kratzert et al. (2019a) static catchment attributes from `camels_{clim,topo,soil,geol,vege}.txt` (excluding `gauge_lat`/`gauge_lon` used in Shahriar's 17-attribute subset) and the pure-PyTorch `mamba3.py` described above.
- **Re-running**: `python -m models.shahriar2026_mamba3.reproduce_author_metrics` writes results to the untracked `stage1_author_reproduction_results.local.json` by default (`--out` to choose another path; the tracked `stage1_author_reproduction_results.json` is only overwritten if `--out` points at it). The Stage-1 Zarr store is only written to GCS with `--upload`.

## Stage-2 CAMELS-US MIP Protocol (`gauged__single_camels`, Table 3)
- **Zarr Store**: `gs://camels-us-mip/hydrographs/gauged__single_camels/shahriar2026_mamba3_daymet.zarr` (`is_ensemble = false`, `num_members = 1`, `seed = 111111`, 30 epochs × 170 steps = `5,100` gradient updates, trained `1999-10-01` to `2008-09-30` with `365`-day spin-up, evaluated `1989-10-01` to `1999-09-30` across all `531` canonical basins in `mm/day`).
- **Leaderboard Row (`shahriar2026_mamba3_daymet`, `Mamba-3 Static-Dynamic Adapter`)**:
  - **Single-Model Median NSE**: **`0.718`** (`0.7182`), Mean NSE = **`0.675`**, `NSE <= 0` = **`6`** (`531 / 531` basins, `0` NaNs).
  - **Secondary Gauch et al. (2023) Metrics (basin medians)**: Median KGE = **`0.737`** (`0.7374`), logNSE = `0.6998`, logKGE = `0.5367`, Median $|\text{FHV}|$ = **`15.20%`** (FHV = `-14.06%`), FMS = `-7.27%`, FLV = `-57.34%`, Alpha-NSE = `0.8641`, Beta-NSE = `-0.0291`, Pearson-$r$ = `0.8641`, Beta-KGE = `0.9390`, RMSE = `1.2958 mm/day`, Peak-Timing = `0.3333 days`. The equal Alpha-NSE and Pearson-$r$ medians are a coincidence at four decimals (re-scored from the store with `metrics.evaluator`; interquartile ranges `0.7506–0.9595` and `0.7993–0.9047`).
