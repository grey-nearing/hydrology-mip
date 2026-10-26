# Feng, Fang & Shen (2020) Data-Integration Lagged-Discharge LSTM (`DI(1)-LSTM`)

## Canonical Reference
> Feng, D., Fang, K., & Shen, C. (2020). Enhancing Streamflow Forecast and Extracting Insights Using Long-Short Term Memory Networks With Data Integration at Continental Scales. *Water Resources Research*, 56(9), e2019WR026793. https://doi.org/10.1029/2019WR026793

- **Original Author Code**: [`https://github.com/mhpi/hydroDL`](https://github.com/mhpi/hydroDL)
- **Provenance**:
  - `feng2020_di_lstm_lit` (`Literature*`): Published Daymet + $Q(t-1)$ benchmark evaluated on `WY1991–1999` (9 water years; `1989–1990` used as 365-day spin-up) across all 531 Newman basins as reported by Jamaat et al. (2025, *J. Hydrol.*, Table 1, p. 15) for the Feng et al. (2020) `DI(1)-LSTM` benchmark (`Single-Model Median NSE = 0.840`, `Median KGE = 0.855`).
  - `feng2020_di_lstm_rerun` (`MIP Re-run (Our Code)`): Clean-room PyTorch implementation in `models/feng2020_di_lstm/` evaluated on the canonical 531-basin Newman Split-1 Gauged protocol (`1999-10-01` to `2008-09-30` training; `1989-10-01` to `1999-09-30` evaluation) in **Table 5 (`gauged__lagged_streamflow`)**.

## Architecture (`models/feng2020_di_lstm/model.py`)
- **Model**: 1-layer 256-unit LSTM (`DILSTM`) with output dropout `0.4`, initial forget-gate bias `3.0`, and linear readout head.
- **Dynamic Inputs**: Daily meteorological forcings plus two lagged-discharge Data-Integration (`DI(1)`) channels at day $t$:
  1. Standardized 1-day-lagged target-basin specific discharge $\tilde{Q}_{t-1} = (Q_{t-1} - \mu_Q) / \sigma_Q$ when $Q_{t-1}$ is valid, and $0.0$ when missing.
  2. Binary validity indicator $v_{t-1} \in \{0.0, 1.0\}$ (`lag_dropout_prob = 0.02` during training).
- **Static Inputs**: 27 CAMELS catchment attributes standardized on the training period and broadcast across time.
- **Training Objective**: Basin-variance-weighted $\text{NSE}^*$ loss (`nse_eps = 0.1`) over sequence length `seq_len = 365` days (`targ_len = 183` post-warmup target steps), trained for `30` epochs (`batch_size = 256`, `clip_grad_norm = 1.0`, `6` random seeds `101..106`).

## Spatial Scope: Gauged Lagged-Streamflow (`Table 5`) Only
`DI(1)-LSTM` ingests the **target basin's own 1-day-lagged observed streamflow $Q(t-1)$** at every timestep $t$. It is therefore **only valid for Gauged Lagged-Streamflow (`gauged__lagged_streamflow`, Table 5)**.

In ungauged Prediction in Ungauged Basins (`pub_12fold`, Table 9) and Prediction in Ungauged Regions (`pur_18huc02`, Table 13), held-out test basins have no in-basin stream gauge ($Q_{\text{target}}(t-1)$ is unavailable at test time, and feeding $Q_{\text{target}}(t-1)$ into held-out test basins constitutes direct test-time target leakage). Attempting to call `run_pub`, `run_pur`, or `run_fold` with non-gauged spatial folds raises:
```python
NotImplementedError(
    "DI(1)-LSTM requires target-basin lagged observed streamflow Q(t-1) and is only valid "
    "for Gauged Lagged-Streamflow (Table 5). Ungauged PUB/PUR requires an explicit "
    "donor-gauge transfer mechanism."
)
```

## Stage-1 Author-Protocol Confirmation (`reproduce_author_metrics.py`)
- **Author Split**: 671 CAMELS-US basins, single-forcing `NLDAS` + $Q(t-1)$, trained on `1985-10-01` to `1995-09-30` and evaluated on `1995-10-01` to `2005-09-30` across 6 random seeds (`[101, 102, 103, 104, 105, 106]`).
- **Published Benchmark (Feng et al., 2020, Table 3, p. 28)**: 6-seed ensemble Median NSE = `0.859`, single-seed Median NSE = `0.851 ± 0.0097`.
- **Stage-1 Reproduction**: 6-seed ensemble Median NSE = `0.820` (`0.8202`), single-seed Median NSE = `0.813 ± 0.002` (`0.8133 ± 0.0020`), Median KGE = `0.834` (`status: PASS` within `0.04` tolerance).
