# Naive Lag-1 Streamflow Persistence (`models/naive_lag1_persistence/`)

Zero-parameter deterministic lower benchmark for all three **With Lagged Streamflow** leaderboards (`gauged__lagged_streamflow`, `pub__lagged_streamflow`, `pur__lagged_streamflow`):

$$\hat{Q}(t) = Q_{\text{obs}}(t - 1)$$

## Overview
- **Model Family ID:** `naive_lag1_persistence`
- **Category:** `Contribution` (`Statistical`)
- **Parameters:** `0` (no training, identical predictions across Gauged, 12-Fold PUB, and 18-Fold HUC-02 PUR)
- **Missing-day handling:** Strictly causal carry-forward of the most recent valid prior observation $Q_{\text{obs}}(t')$ ($t' < t$) on the 2 isolated pre-window missing days across the 531 Newman basins (`pred_nan_when_obs_valid = 0`).
- **Benchmark Performance (531 Newman basins, Water Years 1990–1999):**
  - Median NSE: **`0.454`** (Mean NSE: `0.412`, `86 / 531` basins with `NSE <= 0`)
  - Median KGE: **`0.727`** (Mean KGE: `0.706`)
  - Median Peak-Timing Error: **`1.00 d`** (exact 1-day phase shift)
