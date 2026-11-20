# Yin et al. (2022) RR-Former

Reference:
> Yin, H., Guo, Z., Zhang, X., Chen, J., & Zhang, Y. (2022). RR-Former: Rainfall-runoff modeling based on Transformer. *Journal of Hydrology*, 609, 127781. https://doi.org/10.1016/j.jhydrol.2022.127781

## Architecture
- **Model**: Encoder-decoder Transformer (`d_model=64`, `nhead=4`, `num_encoder_layers=4`, `num_decoder_layers=4`, `dim_feedforward=256`, `dropout=0.1`) with meteorological forcings, 27 static catchment attributes, and lagged observed specific discharge (`[q_lag_std, q_lag_valid]`).
- **Pretraining + Fine-Tuning**: Global regional pretraining across all training basins followed by basin-specific ridge-regularized readout fine-tuning ($\lambda = 10.0$) anchored to the global parameters for gauged basins.

## Stage-1 Author-Protocol Confirmation Run (`status: "PASS"`)
- **Published Target (`author_published_med_nse = 0.8265`)**: In Yin et al. (2022), Section 4.1 (`p. 6`), Abstract, Fig. 4a, and Tables 5–6 report a 1-day-ahead **Median NSE of `0.8265`** (Mean NSE = `0.7904`, and `0.7598` Median NSE for global pretraining without fine-tuning, `RR-Former2`) on the `1980–1995` train / `2000–2014` test split with Daymet forcings and lagged streamflow. Note that the number `0.862` in Table 4 is **Mean NSE on a 40-basin hyperparameter-tuning subset** (and matches the Feng et al., 2020 `DI(1)` LSTM benchmark), not the full-cohort Median NSE.
- **Stage-1 Reproduction**: Our 3-seed Stage-1 reproduction (`seeds=[601, 602, 603]`, trained `1980-10-01` to `1995-09-30` on 531 CAMELS-US basins with Daymet + lagged streamflow) achieves **Median NSE = `0.816`** on the `2000–2014` test split ($\Delta = -0.0105$) and **`0.8009`** across the full `1995-10-01` to `2010-09-30` evaluation window ($\Delta = -0.0256$, single-seed `0.7919 ± 0.0042`, Mean NSE = `0.7706`, Median KGE = `0.8375`), passing within tolerance (`status: "PASS"`).

## Stage-2 CAMELS-US MIP Protocol (`gauged__lagged_streamflow`, Table 5)
- **Zarr Store**: `gs://camels-us-mip/hydrographs/gauged__lagged_streamflow/yin2022_rrformer_daymet.zarr` (`num_members = 3`, `seeds = [601, 602, 603]` per the store's `.zattrs`, trained `1999-10-01` to `2008-09-30`, evaluated `1989-10-01` to `1999-09-30` across 531 basins).
- **Competing Rows**:
  - `yin2022_rrformer_daymet` (`Rainfall-Runoff Transformer (RR-Former) (3-model ensemble)`, `is_ensemble: true`, `evaluation_mode: "ensemble"`): Median NSE = `0.802`, Mean NSE = `0.773`, Median KGE = `0.828`.
  - `yin2022_rrformer_daymet_single` (`Rainfall-Runoff Transformer (RR-Former) (single model)`, `is_ensemble: false`, `evaluation_mode: "single_member_mean"`): Single-Model Median NSE = `0.797 ± 0.003` across all 3 seeds.
