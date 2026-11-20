# ERA-Unguided-PUB (10-model ensemble) (`era_pub_minimal_guidance`)

## Overview & Provenance
- **Model Family**: `Empirical Research Assistance (ERA)` (`era_lstm_seeded`)
- **Citation**: `Nearing et al. (2026)`
- **Target Leaderboard**: `pub__multi_camels` (`pub_12fold`, `Daymet + Maurer + NLDAS`)
- **Zenodo Hydrograph Archive DOI**: [`10.5281/zenodo.23639387`](https://doi.org/10.5281/zenodo.23639387)
- **ERA Starting Notebook**: [`mips/camels_us_daily/era_experiments/era_pub_minimal_guidance.ipynb`](../../mips/camels_us_daily/era_experiments/era_pub_minimal_guidance.ipynb)
- **ERA Winner Notebook**: [`mips/camels_us_daily/era_experiments/era_pub_minimal_guidance_winner.ipynb`](../../mips/camels_us_daily/era_experiments/era_pub_minimal_guidance_winner.ipynb) (Winning Tree-Search Node: **Node 82**)
- **Model Code**: [`model.py`](./model.py) · [`runner.py`](./runner.py)

## Frozen Architecture Summary (Code-Verified)
`EALSTM` (1 network per seed per fold, 10 seeds across canonical 12-fold PUB):
- **Backbone:** 1-layer Entity-Aware LSTM (`hidden_size = 256`) with a 2-layer LayerNorm+SiLU static bottleneck embedding (27 CAMELS attributes + `lat`, `lon`, `log1p(area)`) concatenated at each step over 26 dynamic channels, exponential moving average (EMA) weight averaging, and continuous evaluation rollout clipped at `0.0 mm/day`.
- **Loss & Protocol:** Basin-scaled Huber loss (`delta = 1.0`) plus `0.10 * MSE(log1p(relu(y_hat)), log1p(relu(y_true)))` with AdamW (`weight_decay = 0.01`). Architecture selected via 256-iteration tree search in which every candidate was trained on all 12 canonical PUB folds (weights fit on `2003-10-01`–`2008-09-30`) and ranked by the median NSE of the stitched out-of-fold predictions on `1999-10-01`–`2003-09-30`; the winner was then retrained across all 12 PUB folds on the full Newman training window.

## Packaging Notes & Known Deviations (Code-Verified)
- **`t_range` feature is not a temperature range (copied bug-for-bug from the notebook).** `runner.py` (`_prepare_features`, L53–L54) computes `t_range = x_dyn[:, :, 6:9] - x_dyn[:, :, 9:12]`, exactly as in `era_pub_minimal_guidance_winner.ipynb` cell 5 L22. With the canonical product-major forcing layout (`prcp, srad, tmax, tmin, vp` per product), this yields `[srad_maurer - vp_maurer, tmax_maurer - prcp_nldas, tmin_maurer - srad_nldas]`, not `tmax - tmin`. The published hydrographs were produced with this feature, so it is kept as-is.
- **Spatial selection caveat.** The PUB test basins are ungauged with respect to weight training, but their WY2000–2003 streamflow was used to rank candidate architectures during the tree search (see [`mips/camels_us_daily/era_experiments/README.md`](../../mips/camels_us_daily/era_experiments/README.md)).
- **Train vs. inference protocol:** training samples random `365`-day windows (`seq_len = 365`), but inference runs **one stateful pass** over the 365-day spin-up plus the full 3,652-day evaluation decade (about 4,017 days), discarding the first 365 predicted days. Predictions are therefore not windowed sequence-to-one outputs.

## Benchmark Performance (531 Newman Basins, 12-Fold PUB, Water Years 1990–1999)
| Metric | 10-Seed Ensemble (`era_pub_minimal_guidance`) | Single-Seed Realization (`era_pub_minimal_guidance_single`) |
| :--- | :---: | :---: |
| **Median NSE** | **`0.720`** | `0.714 ± 0.004` |
| **Mean NSE** | `0.649` | `0.636` |
| **Median KGE** | `0.644` | `0.641` |
| **Basins with `NSE <= 0`** | `5 / 531` | `5 / 531` |
