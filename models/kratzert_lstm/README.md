# Kratzert et al. (2019a, 2019b, 2021) Regional Long Short-Term Memory (`CudaLSTM`)

## Overview

This directory contains the clean-room PyTorch implementation of the regional Long Short-Term Memory (`CudaLSTM`) architecture introduced by **Kratzert et al. (2019a)** (*Hydrology and Earth System Sciences*, 23, 5089–5110) and **Kratzert et al. (2019b)** (*Water Resources Research*, 55, 11344–11364) for regional and ungauged rainfall-runoff modeling, and extended to single- and multi-forcing meteorological inputs by **Kratzert et al. (2021)** (*Hydrology and Earth System Sciences*, 25, 2685–2706).

In the CAMELS-US Model Intercomparison Project (MIP), the Kratzert LSTM family is represented by four distinct Table 3 entries with strict single-citation and single-provenance attribution:

1. **`kratzert2019b_lstm`** — *Kratzert et al. (2019a)* (`https://doi.org/10.5194/hess-23-5089-2019`; Zenodo `10.5281/zenodo.3530884`, HydroShare `10.4211/hs.83ea5312635e44dc824eeb99eda12f06`): Author-archived 8-seed Maurer single-forcing regional `CudaLSTM` (270-day lookback, 256 hidden units) evaluated on the Gauged Single-Forcing leaderboard (`gauged__single_camels`, Supplement Table S1).
2. **`kratzert2019a_pub_lstm`** — *Kratzert et al. (2019b)* (`https://doi.org/10.1029/2019WR026065`; HydroShare `10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c`): Published Literature* 12-fold Prediction in Ungauged Basins (PUB) extended NLDAS-2 single-forcing regional `CudaLSTM` (270-day lookback, 256 hidden units, 10 seeds per fold, evaluated on Split-3 WY1996–2010) on the PUB Single-Forcing leaderboard (`pub__single_camels`, Supplement Table S4).
3. **`kratzert2021_lstm_archive`** — *Kratzert et al. (2021)* (`https://doi.org/10.5194/hess-25-2685-2021`; Zenodo `10.5281/zenodo.4670268`): Author-archived 10-seed single-forcing (Daymet, Maurer, NLDAS) and multi-forcing (Daymet+Maurer+NLDAS, Daymet+Maurer, Daymet+NLDAS, Maurer+NLDAS) `CudaLSTM` runs (365-day lookback, 256 hidden units) evaluated on the Gauged Single-Forcing (`gauged__single_camels`, Table 3) and Gauged Multi-Forcing (`gauged__multi_camels`, Table 4) leaderboards.
4. **`kratzert2021_lstm_rerun`** — *Kratzert et al. (2021)* (`https://doi.org/10.5194/hess-25-2685-2021`; `models/kratzert_lstm`): MIP Re-run (Our Code) of the 365-day single- and multi-forcing `CudaLSTM` under the canonical 12-fold PUB (`pub__multi_camels` Table 8) and 18-fold HUC-02 Leave-One-Region-Out PUR (`pur__single_camels` Table 11, `pur__multi_camels` Table 12) protocols.

---

## Architecture & Mathematical Formulation

At each daily timestep $t \in \{1, \dots, T\}$ in a sequence of length $T = 365$ days (or $T = 270$ days in the 2019a/2019b author archives), the dynamic meteorological forcing vector $x_t^{(d)} \in \mathbb{R}^{D_{\text{dyn}}}$ ($D_{\text{dyn}} = 5$ for single-forcing Daymet, Maurer, or NLDAS; $D_{\text{dyn}} = 10$ for 2-forcing combinations; $D_{\text{dyn}} = 15$ for joint 3-forcing Daymet + Maurer + NLDAS) is concatenated with the static catchment attribute vector $x^{(s)} \in \mathbb{R}^{27}$:

$$u_t = \left[ x_t^{(d)} \,\|\, x^{(s)} \right] \in \mathbb{R}^{D_{\text{dyn}} + 27}$$

All dynamic inputs, static attributes, and training target specific discharges are globally $z$-score standardized using the mean and standard deviation computed strictly over the training basins and training period (`1999-10-01` to `2008-09-30`), Missing static attributes are not imputed: the trainer raises if any standardized static attribute is non-finite.

The concatenated input $u_t$ drives a single-layer standard LSTM (`nn.LSTM`, hidden dimension $H = 256$):

$$\begin{aligned}
i_t &= \sigma(W_{ii} u_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
f_t &= \sigma(W_{if} u_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
g_t &= \tanh(W_{ig} u_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
o_t &= \sigma(W_{io} u_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

Following Kratzert et al. (2019a, 2021) and `NeuralHydrology`, the total forget-gate bias $b_{if} + b_{hf}$ is initialized to $+3.0$ (`bias_hh_l0[H:2H] = 3.0`; `bias_ih_l0` keeps the PyTorch default initialization) so that cell states retain long-term hydrologic memory at the start of training. Dropout ($p = 0.4$) and a linear readout head map the terminal hidden state $h_T$ to standardized daily specific discharge $\hat{y}_T = w^\top \text{Dropout}(h_T) + b$, which is inverse-transformed to $\text{mm day}^{-1}$ and clamped at zero ($\hat{Q}_t = \max(0, \hat{y}_t \sigma_Q + \mu_Q)$).

### Basin-Normalized $\text{NSE}^*$ Loss

To prevent humid, high-variance catchments from dominating gradient updates during regional multi-basin training, parameters are optimized using the basin-averaged Nash-Sutcliffe Efficiency loss ($\text{NSE}^*$; Kratzert et al., 2019a):

$$\mathcal{L}_{\text{NSE}^*} = \frac{1}{N} \sum_{i=1}^{N} \frac{\left(\hat{y}_i - y_i\right)^2}{\left(s_{b(i)} + \epsilon\right)^2}, \qquad \epsilon = 0.1$$

where $s_b$ is the standard deviation of observed discharge in basin $b$ over the training period, in mm/d. The prediction error is also expressed in mm/d, so the numerator and denominator share units.

---

## Default Hyperparameters (`RUNNER_DEFAULTS`)

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| `hidden_size` | `256` | Single-layer `nn.LSTM` hidden state dimension |
| `seq_len` | `365` | 365-day lookback window (364-day spinup precedes first scored day) |
| `dropout` | `0.4` | Applied to LSTM hidden state before linear readout head |
| `initial_forget_bias` | `3.0` | Written to `bias_hh_l0[H:2H]` only |
| `batch_size` | `256` | Sequence-to-one training minibatches |
| `epochs` | `30` | Full dataset passes over valid training sequences |
| `lr_schedule` | `{0: 1e-3, 10: 5e-4, 25: 1e-4}` | Piecewise Adam learning rate schedule (Kratzert et al., 2021) |
| `clip_grad_norm` | `1.0` | Global $\ell_2$ gradient norm clipping before optimizer step |
| `nse_eps` | `0.1` | Stability constant $\epsilon$ in $\text{NSE}^*$ denominator $(s_b + \epsilon)^2$ |

---

## Stage-1 Author Protocol Reproduction

The Stage-1 confirmation script (`reproduce_author_metrics.py`) trains `CudaLSTM` in-process via `models.kratzert_lstm.runner.run_fold` on the 531 Newman CAMELS-US basins using single-forcing Daymet meteorology (`train_period = ("1999-10-01", "2008-09-30")`, `eval_period = ("1989-10-01", "1999-09-30")`, 3 seeds `101, 102, 103`):

```bash
python -m models.kratzert_lstm.reproduce_author_metrics --data-dir /path/to/camels --mode verify
```

### Stage-1 Verification Summary (`stage1_author_reproduction_results.json`)

- **Published Reference (Kratzert et al., 2021, Table 1 & Table A2; Zenodo `10.5281/zenodo.4670268` / `t3_02`, Daymet single-forcing on 531 basins)**:
  - Single-seed Median NSE: **`0.744 ± 0.003`** (Table A2 reports `0.74`)
  - 10-seed Ensemble Median NSE: **`0.772`** (`0.7724`, Table 1)
- **Our In-Process Stage-1 Reproduction (3 seeds: `101, 102, 103`)**:
  - Stored fast-verification Zarr (`gs://camels-us-mip/stage1_author_reproductions/kratzert_lstm.zarr`): **`0.7110`** 3-seed Ensemble Median NSE, **`0.7036 ± 0.0018`** Single-Seed Median NSE (`[0.7014, 0.7058, 0.7036]`), Mean NSE **`0.6767`**, Median KGE **`0.6931`**, `1 / 531` basins with $\text{NSE} \le 0$ ($\Delta_{\text{single}} = -0.0404$).
  - Full 30-epoch 3-seed reproduction: **`0.7465`** 3-seed Ensemble Median NSE, **`0.7350 ± 0.0013`** Single-Seed Median NSE (`[0.7333, 0.7365, 0.7351]`, $\Delta_{\text{single}} = -0.0090$).
  - Status: **`PASS`** (`gs://camels-us-mip/stage1_author_reproductions/kratzert_lstm.zarr`)

---

## Stage-2 MIP Usage

All Stage-2 MIP experiments invoke `models.kratzert_lstm.runner.run_fold`:

```python
from models.kratzert_lstm.runner import run_fold

qsim_mean, qsim_members = run_fold(
    train_basins=train_basins,
    test_basins=test_basins,
    train_period=("1999-10-01", "2008-09-30"),
    eval_period=("1989-10-01", "1999-09-30"),
    forcings=["daymet", "maurer", "nldas"],
    data_dir="/path/to/camels",
    seeds=tuple(range(101, 111)),
    return_members=True,
)
```
