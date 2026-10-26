# Kratzert et al. (2019b) — Entity-Aware LSTM (`kratzert_ealstm`) & LSTM without Static Attributes (`kratzert_lstm_nostatic`)

## Canonical Reference & Links
- **Citation**: Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. (2019b). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *Hydrology and Earth System Sciences*, 23(12), 5089–5110.
- **Paper URL**: https://doi.org/10.5194/hess-23-5089-2019
- **Official Code Repository**: https://github.com/kratzert/ealstm_regional_modeling (Zenodo archive: `10.5281/zenodo.3530884`, Apache-2.0)
- **Official Hydrograph Archive**: HydroShare `10.4211/hs.83ea5312635e44dc824eeb99eda12f06` (`https://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06/`, CC-BY-4.0)
- **Table 2 & Table 3 Provenance Category**: `Archive` (official 8-seed author hydrograph archives from Kratzert et al., 2019b)

## Architecture
- **Entity-Aware LSTM (`kratzert_ealstm`)**: Single-layer recurrent network ($H = 256$ memory cells, sequence length $T = 270$ days, dropout $p = 0.4$, linear readout head, basin-averaged $\text{NSE}^*$ loss with $\epsilon = 0.1$) in which the input gate $\mathbf{i} = \sigma(\mathbf{W}_i \mathbf{x}_s + \mathbf{b}_i) \in (0, 1)^{256}$ depends exclusively on the 27 static CAMELS catchment attributes $\mathbf{x}_s \in \mathbb{R}^{27}$ and remains fixed over time for each basin, while the forget gate $\mathbf{f}[t]$, cell candidate $\mathbf{g}[t]$, and output gate $\mathbf{o}[t]$ depend only on the 5 dynamic meteorological forcings $\mathbf{x}_d[t] \in \mathbb{R}^5$ (`prcp`, `tmin`, `tmax`, `srad`, `vp`) and the previous hidden state $\mathbf{h}[t-1]$.
- **LSTM without Static Attributes (`kratzert_lstm_nostatic`)**: Companion regional ablation baseline from Kratzert et al. (2019b) using a standard single-layer LSTM ($H = 256$, $T = 270$ days, dropout $p = 0.4$, $\text{NSE}^*$ loss) trained jointly across all 531 basins using only the 5 dynamic Maurer meteorological forcings without any static catchment attributes ($D_{\text{static}} = 0$).

## Official Author Archive Used in Table 3 (`gauged__single_camels`)
Per the CAMELS-US MIP Archive Priority Rule, **Table 3 (`gauged__single_camels`) uses the official 8-seed Kratzert et al. (2019b) author archive** (`10.5281/zenodo.3530884` / HydroShare `10.4211/hs.83ea5312635e44dc824eeb99eda12f06`, seeds `111, 222, 333, 444, 555, 666, 777, 888`, trained on `1999-10-01` to `2008-09-30` and evaluated on `1989-10-01` to `1999-09-30` across all 531 CAMELS-US basins with extended Maurer forcings):

| Leaderboard | Row ID | Family ID | Model Display Name | Mode | Members | Med NSE | Mean NSE | Med KGE | Med \|FHV\| (%) | `NSE <= 0` | Zarr Path |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Table 3 (`gauged__single_camels`)** | `t3_23` | `kratzert_ealstm` | `EA-LSTM (8-model ensemble)` | `ensemble` | 8 | **`0.741`** (`0.7409`) | `0.705` | `0.737` | `19.14` | `2` | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_23.zarr` |
| **Table 3 (`gauged__single_camels`)** | `t3_30` | `kratzert_ealstm` | `EA-LSTM (single model)` | `single_member_mean` | 8 | **`0.712 ± 0.004`** (`0.7123`) | `0.673` | `0.728` | `18.30` | `3` | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_23.zarr` |
| **Table 3 (`gauged__single_camels`)** | `t3_44` | `kratzert_lstm_nostatic` | `LSTM without Static Attributes (8-model ensemble)` | `ensemble` | 8 | **`0.642`** (`0.6421`) | `0.493` | `0.631` | `26.83` | `20` | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_44.zarr` |
| **Table 3 (`gauged__single_camels`)** | `t3_44_single` | `kratzert_lstm_nostatic` | `LSTM without Static Attributes (single model)` | `single_member_mean` | 8 | **`0.593 ± 0.008`** (`0.5933`) | `0.394` | `0.579` | `27.32` | `28` | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_44.zarr` |

## Local Clean-Room Implementation (`models/kratzert_ealstm/`)
This directory also provides a clean-room PyTorch implementation of `EA-LSTM` (`model.py`, `runner.py`, `reproduce_author_metrics.py`) with fused cuDNN execution (`EALSTM._forward_cudnn`) and TorchScript CPU recurrence (`_ealstm_recurrence`):
```bash
python -m models.kratzert_ealstm.reproduce_author_metrics --mode verify
```
