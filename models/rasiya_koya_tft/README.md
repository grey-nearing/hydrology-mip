# Temporal Fusion Transformer (`models/rasiya_koya_tft/`)

Clean-room PyTorch implementation of the Temporal Fusion Transformer (TFT; Lim et al., 2021) as applied to CAMELS-US rainfall-runoff modeling by Rasiya Koya & Roy (2024, *Journal of Hydrology*).

## Architecture & Protocol
- **Model Family ID:** `rasiya_koya_tft`
- **Category:** `MIP Re-run` (`Machine learning`)
- **Architecture:** Variable Selection Networks (VSN) over static catchment attributes and daily meteorological forcings, Gated Residual Networks (GRN) with ELU activations and LayerNorm, 1-layer LSTM local sequence encoder (`hidden_size = 128`, `dropout = 0.2`), 4-head interpretable multi-head self-attention over the 365-day lookback window, and a linear discharge projection head trained with basin-variance-normalized $\text{NSE}^*$ loss.
- **Stage-1 Author Protocol Reproduction (`gs://camels-us-mip/stage1_author_reproductions/rasiya_koya_tft.zarr`):**
  - Evaluated on the 531 Newman benchmark basins with Extended Maurer forcing (`1999-10-01` to `2008-09-30` training, `1989-10-01` to `1999-09-30` evaluation, 365-day spin-up from `1988-10-01`).
  - Single-model Median NSE: **`0.716`** (vs. `0.731` for single-seed Maurer LSTM on the identical protocol).
- **Stage-2 MIP Leaderboard Entry (`Table 3: gauged__single_camels`):**
  - `entry_id`: `tft_gauged_maurer` (`gs://camels-us-mip/hydrographs/gauged__single_camels/t3_21.zarr`)
  - Single-model Median NSE: **`0.716`** (Mean NSE `0.629`, Median KGE `0.732`).
