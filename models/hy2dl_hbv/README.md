# Dynamic-Parameter LSTM + 15-HBV (`Hy2DL`) — Acuña Espinoza et al. (2024)

## Reference & Provenance
- **Canonical Citation**: Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., & Ehret, U. (2024). To bucket or not to bucket? Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization. *Hydrology and Earth System Sciences*, 28(12), 2705–2719. https://doi.org/10.5194/hess-28-2705-2024
- **Upstream Software Repository**: https://github.com/KIT-HYD/Hy2DL (Apache-2.0 License)
- **MIP Provenance Category**: `MIP Re-run (Our Code)` (`models/hy2dl_hbv/`)

## Model Architecture
- **Hybrid Conceptual–Deep Learning Formulation**: A 1-layer regional LSTM network (`hidden_size = 256`, initial forget-gate bias `3.0`, linear readout head of size `210`) maps daily meteorological forcings (`prcp, srad, tmax, tmin, vp`, plus derived `tmean` and Hamon `pet`) and 27 static CAMELS catchment attributes to the parameters of $M = 16$ parallel uncalibrated HBV conceptual snow–soil–response buckets and a shared 15-day Gamma unit hydrograph routing kernel (`routing_len = 15`).
- **Dynamic vs. Static Parameterization**: Following the `Hy2DL` CAMELS-US hybrid specification (`HybridModel_US.ipynb` / `hybrid_camels_us.yml`, building on Feng et al., 2022), two soil-moisture nonlinearity exponents (`BETA` $\in [1.0, 6.0]$ and `BETAET` $\in [0.3, 5.0]$) are predicted dynamically at every active simulation day, whereas the remaining 11 HBV bucket parameters (`FC`, `K0`, `K1`, `K2`, `LP`, `PERC`, `UZL`, `TT`, `CFMAX`, `CFR`, `CWH`) and the 2 Gamma unit hydrograph parameters are time-invariant (extracted from the final step of the LSTM sequence).
- **Spin-up & Training Objective**: Each training sequence spans 730 days (`365`-day `torch.no_grad()` spin-up with `BETA` and `BETAET` held constant at day 365, followed by a `365`-day active simulation window). Trained for 20 epochs (`450` batches/epoch, `batch_size = 256`, `Adam` with `lr = 1e-3` and gradient-norm clipping at `1.0`) using the `Hy2DL` combined root-transformed RMSE loss:
  $$\mathcal{L} = 0.75\,\text{RMSE}(Q, \hat{Q}) + 0.25\,\text{RMSE}\!\left(\log_{10}(\sqrt{Q + 10^{-6}} + 0.1),\; \log_{10}(\sqrt{\hat{Q} + 10^{-6}} + 0.1)\right)$$

## Stage-1 Author Benchmark Context & Confirmation
- Acuña Espinoza et al. (2024) conducted their primary manuscript experiments on CAMELS-GB (`Med NSE = 0.87` on the 60-basin GB subset and `0.84` across 669 GB basins with `LSTM+SHM`), while citing Feng et al. (2022)'s `0.733` CAMELS-US Split-3 benchmark in their Section 3.4 discussion and releasing the CAMELS-US `LSTM + HBV` configuration in the companion `KIT-HYD/Hy2DL` repository.
- On the 531-basin CAMELS-US Newman Split-1 protocol (`1999-10-01` to `2008-09-30` training, `1989-10-01` to `1999-09-30` evaluation, Daymet forcing, 3 seeds `101, 102, 103`), our reproduction achieves a 3-seed ensemble Median NSE of **`0.750`** and a single-model Median NSE of **`0.730 ± 0.001`** (`0.734 ± 0.004` across individual seed medians `[0.7398, 0.7312, 0.7319]`), `status: "PASS"`.

## Stage-2 CAMELS-US MIP Leaderboards Covered
- **Table 3 (`gauged__single_camels`)**: `hy2dl_hbv_daymet` (3-model ensemble) and `hy2dl_hbv_daymet_single` (single model, `evaluation_mode="single_member_mean"`).
- **Table 7 (`pub__single_camels`)**: `hy2dl_hbv_pub_daymet` (3-model ensemble) and `hy2dl_hbv_pub_daymet_single` (single model, `evaluation_mode="single_member_mean"`).
- **Table 11 (`pur__single_camels`)**: `hy2dl_hbv_pur_daymet` (3-model ensemble) and `hy2dl_hbv_pur_daymet_single` (single model, `evaluation_mode="single_member_mean"`).
