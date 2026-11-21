# Feng et al. (2020) Data-Integration Lagged-Discharge LSTM (`DI(1)-LSTM`)

Clean-room PyTorch implementation of the regional Data-Integration Long Short-Term Memory network (`DI(1)`) from **Feng, Fang, & Shen (2020)** (*Water Resources Research*, 56(9), e2019WR026793, <https://doi.org/10.1029/2019WR026793>), matching `mhpi/hydroDL` (`flow_regime == 0`).

## Architecture & Training Protocol

- **Meteorological Inputs (NLDAS, 5 channels):** `['dayl', 'prcp', 'srad', 'tmax', 'vp']`, where `prcp` is compressed with Eq. (11) `transNorm`: $\log_{10}(\sqrt{\max(x, 0)} + 0.1)$, then z-score standardized on training-split target-window statistics.
- **Lagged Observed Streamflow (`DI(1)`):** 1-day lagged specific discharge $Q_{t-1}$ normalized by training-split basin mean precipitation `basinNorm` ($Q / \bar{P}_{\text{train}}$; out-of-sample PUB/PUR basins fall back to the training global mean $\bar{P}$ with zero eval-period or test-basin leakage), compressed symmetrically with Eq. (11) `transNorm`, z-scored using training target `(y_mean, y_std)`, and paired with a binary validity flag `q_lag_valid` $\in \{0, 1\}$.
- **Static Attributes:** 27 CAMELS catchment attributes z-score standardized on training basins.
- **Network:** `Linear(nx, 256) -> ReLU -> CudnnLstm(256, dr=0.5) -> Linear(256, 1)`.
- **Objective & Optimization:** `RmseLoss` in z-scored transformed space, `Adadelta(lr=1.0, rho=0.9, eps=1e-6)`, `rho=365`, `batch_size=100`, `epochs=300`, `hydroDL` iteration schedule `nIterEp`.

## Stage-1 Author-Protocol Confirmation

- **Protocol:** 671 CAMELS-US basins, NLDAS forcing + $Q_{t-1}$, train `1985-10-01`–`1995-09-30`, evaluate `1995-10-01`–`2005-09-30`, 6 seeds (`101..106`).
- **Results (`models/feng2020_di_lstm/stage1_author_reproduction_results.json`):**
  - **6-seed Ensemble Median NSE:** **0.8652** (published `0.859`, $\Delta = +0.0062 \le 0.01$, **PASS**; Mean NSE `0.8290`, Med KGE `0.8952`, `NSE <= 0`: `2/671`, `0` NaNs).
  - **Single-seed Median NSE:** **0.8581 ± 0.0028** (`[0.8561, 0.8594, 0.8555, 0.8547, 0.8625, 0.8604]`; published `0.851`, $\Delta = +0.0071 \le 0.01$).
  - **Stage-1 Zarr:** `gs://camels-us-mip/stage1_author_reproductions/feng2020_di_lstm.zarr`.

## Stage-2 CAMELS-US MIP Protocol (`NLDAS + Q(t-1)`, 531 basins)

Train `1999-10-01`–`2008-09-30`, blind evaluation `1989-10-01`–`1999-09-30`:

- **Gauged (`gauged__lagged_streamflow`, 6 seeds `101..106`):**
  - Ensemble Median NSE: **0.8333** (`mean_nse=0.7107`, `med_kge=0.8669`, `logNSE=0.9726`, `logKGE=0.9622`, `|FHV|=7.37%`).
  - Single-Model Median NSE: **0.8251 ± 0.0061** (`mean_nse=0.6809`, `med_kge=0.8503`, `logNSE=0.9703`, `logKGE=0.9446`, `|FHV|=9.15%`).
- **12-Fold Random PUB (`pub__lagged_streamflow`, 3 seeds `101..103`):**
  - Ensemble Median NSE: **0.7866** (`mean_nse=0.6166`, `med_kge=0.7975`, `logNSE=0.9643`, `logKGE=0.9448`, `|FHV|=12.70%`).
  - Single-Model Median NSE: **0.7790 ± 0.0074** (`mean_nse=0.5914`, `med_kge=0.7792`, `logNSE=0.9623`, `logKGE=0.9286`, `|FHV|=14.06%`).
- **18-Fold HUC-02 PUR (`pur__lagged_streamflow`, 3 seeds `101..103`):**
  - Ensemble Median NSE: **0.7424** (`mean_nse=0.2995`, `med_kge=0.7448`, `logNSE=0.9576`, `logKGE=0.9230`, `|FHV|=16.33%`).
  - Single-Model Median NSE: **0.7344 ± 0.0020** (`mean_nse=0.2272`, `med_kge=0.7377`, `logNSE=0.9556`, `logKGE=0.9122`, `|FHV|=17.19%`).

## Usage

```bash
python -m models.feng2020_di_lstm.reproduce_author_metrics --smoke
python -m models.run_model --model feng2020_di_lstm --spatial-regime gauged --forcings nldas --forcing-regime lagged_streamflow
```
