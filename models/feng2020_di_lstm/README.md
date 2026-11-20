# Feng, Fang & Shen (2020) Data-Integration Lagged-Discharge LSTM (`DI(1)-LSTM`)

## Canonical Reference
> Feng, D., Fang, K., & Shen, C. (2020). Enhancing Streamflow Forecast and Extracting Insights Using Long-Short Term Memory Networks With Data Integration at Continental Scales. *Water Resources Research*, 56(9), e2019WR026793. https://doi.org/10.1029/2019WR026793

- **Original Author Code**: [`https://github.com/mhpi/hydroDL`](https://github.com/mhpi/hydroDL)
- **Provenance**:
  - `feng2020_di_lstm_lit` (`Literature*`): Published Daymet + $Q(t-1)$ benchmark evaluated on `WY1991–1999` (9 water years; `1989–1990` used as 365-day spin-up) across all 531 Newman basins as reported by Jamaat et al. (2025, *J. Hydrol.*, Table 1, p. 15) for the Feng et al. (2020) `DI(1)-LSTM` benchmark (`Single-Model Median NSE = 0.840`, `Median KGE = 0.855`).
  - `feng2020_di_lstm_rerun` (`MIP Re-run (Our Code)`): Clean-room PyTorch implementation in `models/feng2020_di_lstm/` evaluated on the canonical 531-basin Newman Split-1 Gauged protocol (`1999-10-01` to `2008-09-30` training; `1989-10-01` to `1999-09-30` evaluation) in **Table 5 (`gauged__lagged_streamflow`)**, plus the PUB/PUR lagged-streamflow rows in Tables 9 and 13 (see *Scope* below).

## Architecture (`models/feng2020_di_lstm/model.py`)
- **Model**: 1-layer 256-unit LSTM (`DILSTM`) with output dropout `0.4`, initial forget-gate bias `3.0`, and linear readout head.
- **Dynamic Inputs**: Daily meteorological forcings plus two lagged-discharge Data-Integration (`DI(1)`) channels at day $t$:
  1. Standardized 1-day-lagged target-basin specific discharge $\tilde{Q}_{t-1} = (Q_{t-1} - \mu_Q) / \sigma_Q$ when $Q_{t-1}$ is valid, and $0.0$ when missing.
  2. Binary validity indicator $v_{t-1} \in \{0.0, 1.0\}$ (`lag_dropout_prob = 0.02` during training).
- **Static Inputs**: 27 CAMELS catchment attributes standardized on the training period and broadcast across time.
- **Training Objective**: Basin-variance-weighted $\text{NSE}^*$ loss (`nse_eps = 0.1`) over sequence length `seq_len = 365` days (`targ_len = 183` post-warmup target steps), trained for `30` epochs (`batch_size = 256`, `clip_grad_norm = 1.0`, `6` random seeds `101..106`).

## Scope: Lagged-Streamflow Leaderboards Only
`DI(1)-LSTM` takes the **target basin's own observed streamflow from the previous day, $Q(t-1)$**, as an input at every timestep $t$. Lagged values are offset by one day, so $Q(t)$ never leaks into its own prediction, and spin-up discharge in the evaluation slice is masked. The model is therefore valid **only on the `lagged_streamflow` forcing-regime leaderboards**:

| Leaderboard | Table | Entry IDs |
|---|---|---|
| `gauged__lagged_streamflow` | 5 | `feng2020_di_lstm_nldas`, `feng2020_di_lstm_nldas_single` |
| `pub__lagged_streamflow` | 9 | `feng2020_di_lstm_pub_nldas`, `feng2020_di_lstm_pub_nldas_single` |
| `pur__lagged_streamflow` | 13 | `feng2020_di_lstm_pur_nldas`, `feng2020_di_lstm_pur_nldas_single` |

The PUB (`pub_12fold`) and PUR (`pur_18huc02`) rows follow a data-assimilation definition of "ungauged". The network weights are trained only on the fold's training basins. At prediction time, however, the held-out basins' own past streamflow $Q(t-1)$ is assumed to be available as an input, as it would be for a newly installed gauge with a short record. These rows answer a different question from the `single_camels` / `multi_camels` PUB/PUR tables and must not be compared with them directly.

`run_fold` itself has no spatial-regime guard: it accepts and ignores the `spatial_regime` and `forcing_regime` keywords. The restriction to `lagged_streamflow` leaderboards is enforced in `models/run_model.py` through the model registry.

## Stage-1 Author-Protocol Confirmation (`reproduce_author_metrics.py`)
- **Author Split**: 671 CAMELS-US basins, single-forcing `NLDAS` + $Q(t-1)$, trained on `1985-10-01` to `1995-09-30` and evaluated on `1995-10-01` to `2005-09-30` across 6 random seeds (`[101, 102, 103, 104, 105, 106]`).
- **Published Benchmark (Feng et al., 2020, Table 3, p. 28)**: 6-seed ensemble Median NSE = `0.859`, single-seed Median NSE = `0.851 ± 0.0097`.
- **Stage-1 Reproduction**: 6-seed ensemble Median NSE = `0.820` (`0.8202`), single-seed Median NSE = `0.813 ± 0.002` (`0.8133 ± 0.0020`), Median KGE = `0.834` (`status: PASS` within `0.04` tolerance).
