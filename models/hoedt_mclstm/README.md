# Mass-Conserving LSTM (MC-LSTM) — Hoedt et al. (2021)

## 1. Canonical Provenance & References

- **Primary Publication**: Hoedt, P.-J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., Hochreiter, S., & Klambauer, G. (2021). *MC-LSTM: Mass-Conserving LSTM*. Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:4275–4286. [https://arxiv.org/abs/2101.05186](https://arxiv.org/abs/2101.05186) / [https://proceedings.mlr.press/v139/hoedt21a.html](https://proceedings.mlr.press/v139/hoedt21a.html)
- **Official Author Code Repository**: [https://github.com/ml-jku/mc-lstm](https://github.com/ml-jku/mc-lstm) (`hydrology/mclstm/` and `NeuralHydrology` integration)
- **Official Author Data Archive (Maurer 531-Basin Ensemble Runs)**: Hoedt et al. (2021), Zenodo DOI: [`10.5281/zenodo.5074790`](https://doi.org/10.5281/zenodo.5074790)
- **External Benchmark Archive (Daymet & NLDAS Single-Forcing Runs)**: Frame, J. M., Kratzert, F., Gupta, H. V., Ullrich, P., & Nearing, G. S. (2023). *On strictly enforced mass conservation constraints for modelling the rainfall–runoff process*. Hydrological Processes, 37(3), e14847. [https://doi.org/10.1002/hyp.14847](https://doi.org/10.1002/hyp.14847)
  - Code: [https://github.com/jmframe/mclstm_camels_2022](https://github.com/jmframe/mclstm_camels_2022)
  - HydroShare Archive: [`10.4211/hs.d750278db868447dbd252a8c5431affd`](https://doi.org/10.4211/hs.d750278db868447dbd252a8c5431affd)

---

## 2. Model Overview & Architecture

The Mass-Conserving LSTM (MC-LSTM; Hoedt et al., 2021) modifies the recurrent cell state update of the standard LSTM so that the cell states $\mathbf{c}[t] \in \mathbb{R}_{\ge 0}^K$ act as non-negative physical mass buckets whose sum is conserved across time up to incoming mass fluxes $\mathbf{x}_m[t]$ and outgoing mass fluxes (discharges and sink losses).

At each daily timestep $t$:
1. **Input Gate ($\mathbf{i}[t]$)**: Distributes incoming daily precipitation mass $x_m[t] \ge 0$ (`PRCP(mm/day)`, kept non-negative in $\text{mm day}^{-1}$ rather than centered to zero mean) across the $K=64$ internal mass states via a column-normalized softmax:
   $$\mathbf{i}[t] = \operatorname{softmax}(\mathbf{W}_i x_m[t] + \mathbf{U}_i (\mathbf{c}[t-1] / \tau_c) + \mathbf{V}_i \mathbf{a}[t] + \mathbf{b}_i)$$
2. **Output Gate ($\mathbf{o}[t]$)**: Determines the fraction of mass leaving each cell state at step $t$:
   $$\mathbf{o}[t] = \sigma(\mathbf{W}_o x_m[t] + \mathbf{U}_o (\mathbf{c}[t-1] / \tau_c) + \mathbf{V}_o \mathbf{a}[t] + \mathbf{b}_o)$$
3. **Redistribution Matrix ($\mathbf{R}[t] \in \mathbb{R}^{K \times K}$)**: A column-stochastic matrix ($\sum_{k=1}^K R_{kj}[t] = 1$, $R_{kj}[t] \ge 0$) computed via column-wise softmax that redistributes retained mass across internal buckets without creating or destroying mass:
   $$\mathbf{c}^+[t] = (\mathbf{1} - \mathbf{o}[t]) \odot \mathbf{c}[t-1] + \mathbf{i}[t] x_m[t], \qquad \mathbf{c}[t] = \mathbf{R}[t]\,\mathbf{c}^+[t], \qquad \mathbf{h}[t] = \mathbf{o}[t] \odot \mathbf{c}^+[t]$$
4. **Auxiliary Inputs ($\mathbf{a}[t]$)**: The remaining 4 meteorological forcing variables (`SRAD`, `Tmax`, `Tmin`, `Vp`) and 27 static catchment attributes from Kratzert et al. (2019), standardized via z-score over the training period/basins, drive the gates ($\mathbf{i}[t], \mathbf{o}[t], \mathbf{R}[t]$) without entering the conserved mass state directly.
5. **Streamflow Readout & Trash Cell**: Following Hoedt et al. (2021, Section 4.3), one of the $K=64$ outgoing mass fluxes (`h[t, 0]`) acts as an unaccounted mass sink ("trash cell" for evapotranspiration, deep percolation, and forcing bias), while simulated daily specific discharge is the exact sum of the remaining $K-1 = 63$ outgoing mass fluxes:
   $$\hat{y}[t] = \sum_{k=1}^{K-1} h_k[t]$$

---

## 3. Clean-Room Implementation (`models/hoedt_mclstm/`)

- [`model.py`](model.py): Clean-room PyTorch + Triton fused recurrent scan implementation (`MassConservingLSTM`, `_MCLSTMScanFn`, `_mclstm_fwd_kernel`, `_mclstm_bwd_kernel`) implementing the exact column-stochastic redistribution matrix, input softmax gate, sigmoid output gate, 64 hidden cells, trash cell (`h[..., 0]`), and sum readout (`h[..., 1:].sum(dim=-1)`). Trained with the basin-std-normalized $\text{NSE}^*$ loss (Kratzert et al., 2019), sequence length 365 days (`targ_len=1` last-step supervision), AdamW ($\text{lr}=10^{-3}$ decayed across 35 epochs), and gradient clipping at 1.0.
- [`runner.py`](runner.py): Supports both single-forcing configurations (`maurer`, `daymet`, `nldas`) and multi-forcing configurations (`mode="mean_precip_3f_gates"` for Table 4/8/12 where the conserved mass input $x_m[t]$ is the arithmetic mean of Daymet, Maurer, and NLDAS precipitation while all 15 meteorological drivers enter the auxiliary vector $\mathbf{a}[t]$; and `mode="3f_single_ensemble"` where independent Daymet, Maurer, and NLDAS MC-LSTM models are averaged).
- [`reproduce_author_metrics.py`](reproduce_author_metrics.py): Stage-1 author-protocol verification script.

---

## 4. Stage-1 Author Protocol Reproduction

Hoedt et al. (2021, Table 2/3 & Appendix B, Table B.7) trained MC-LSTM on the 531 Newman CAMELS-US basins using **Maurer (extended)** forcings (`train: 1999-10-01..2008-09-30`, `eval: 1989-10-01..1999-09-30`), reporting:
- **Published Ensemble Median NSE**: **0.744** (447/498-basin subset, Table 2/3) / **0.742** (all 531 basins, Appendix Table B.7)
- **Published Single-Model Median NSE**: **0.726 ± 0.003** (498 basins) / **0.726 ± 0.004** (531 basins)

Our 3-seed Stage-1 reproduction (`seeds = [301, 302, 303]`, stored in [`stage1_author_reproduction_results.json`](stage1_author_reproduction_results.json) and `gs://camels-us-mip/stage1_author_reproductions/hoedt_mclstm.zarr`) achieves:
- **Stage-1 Ensemble Median NSE (531 basins)**: **0.7430** (`abs_diff = 0.0010` vs. published `0.744` / `0.0010` vs. `0.742`, well within `tolerance = 0.02`, **`status: "PASS"`**)
- **Stage-1 Single-Seed Median NSE (531 basins)**: **0.7160 ± 0.0032** across seeds `[0.7119, 0.7165, 0.7196]` (`abs_diff = 0.0100` vs. published `0.726 ± 0.004`, **`status: "PASS"`**)
- **Completeness**: `531 / 531` basins scored, `pred_nan_when_obs_valid = 0`.

---

## 5. Stage-2 CAMELS-US MIP Leaderboard Entries

In the CAMELS-US MIP manuscript, MC-LSTM is represented by **three strictly separated provenance entries** in Table 2:
1. **`hoedt_mclstm_lit`** (*Literature*, Hoedt et al., 2021): Maurer 531-basin published scores (`ens_med_nse = 0.742`, `single_med_nse = 0.726 ± 0.004` in Table 3).
2. **`frame2023_mclstm_archive`** (*Archive*, Frame et al., 2023): Pre-computed 8-seed ensemble hydrographs from HydroShare (`10.4211/hs.d750278db868447dbd252a8c5431affd`) for Daymet (`t3_12`, Med NSE = `0.758`) and NLDAS (`t3_28`, Med NSE = `0.736`) in Table 3.
3. **`hoedt_mclstm_rerun`** (*MIP Re-run (Our Code)*, Hoedt et al., 2021): Complete 531-basin Zarr v2 hydrographs across all 6 CAMELS-trained leaderboards:
   - **Table 3 (`gauged__single_camels`, Maurer)**: Ensemble Med NSE = **0.743** (`n=3`), Single-Model Med NSE = **0.716 ± 0.003**
   - **Table 4 (`gauged__multi_camels`, 3-Forcing)**: Mean-Precip + 3F Gates (`hoedt_mclstm_3f_avgp`) Ensemble Med NSE = **0.777** (`n=8`), Single-Model Med NSE = **0.754 ± 0.005**; 3-Forcing Model-Average Ensemble (`hoedt_mclstm_3f_ens_b`) Med NSE = **0.771** (`n=3`)
   - **Table 7 (`pub__single_camels`, 5-Fold Random PUB)**: Single-Model Med NSE = **0.648** (Daymet), **0.621** (Maurer), **0.585** (NLDAS)
   - **Table 8 (`pub__multi_camels`, 5-Fold Random PUB)**: Mean-Precip + 3F Gates Ensemble Med NSE = **0.685** (`n=3`), Single-Model Med NSE = **0.659 ± 0.004**; 3-Forcing Model-Average Ensemble Med NSE = **0.683** (`n=3`)
   - **Table 11 (`pur__single_camels`, 7-Fold HUC-02 PUR)**: Single-Model Med NSE = **0.560** (Daymet), **0.510** (Maurer), **0.516** (NLDAS)
   - **Table 12 (`pur__multi_camels`, 7-Fold HUC-02 PUR)**: Mean-Precip + 3F Gates Ensemble Med NSE = **0.608** (`n=3`), Single-Model Med NSE = **0.578 ± 0.005**; 3-Forcing Model-Average Ensemble Med NSE = **0.602** (`n=3`)
