# Mass-Conserving LSTM (MC-LSTM) — Hoedt et al. (2021)

## 1. Provenance & References

- **Primary Publication**: Hoedt, P.-J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., Hochreiter, S., & Klambauer, G. (2021). *MC-LSTM: Mass-Conserving LSTM*. Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:4275–4286. [https://arxiv.org/abs/2101.05186](https://arxiv.org/abs/2101.05186) / [https://proceedings.mlr.press/v139/hoedt21a.html](https://proceedings.mlr.press/v139/hoedt21a.html)
- **Official Author Code Repository**: [https://github.com/ml-jku/mc-lstm](https://github.com/ml-jku/mc-lstm) (`hydrology/mclstm/` and `NeuralHydrology` integration)
- **Official Author Data Archive (Maurer 531-Basin Ensemble Runs)**: Hoedt et al. (2021), Zenodo DOI: [`10.5281/zenodo.5074790`](https://doi.org/10.5281/zenodo.5074790)
- **External Benchmark Archive (Daymet & NLDAS Single-Forcing Runs)**: Frame, J. M., Kratzert, F., Gupta, H. V., Ullrich, P., & Nearing, G. S. (2023). *On strictly enforced mass conservation constraints for modelling the rainfall–runoff process*. Hydrological Processes, 37(3), e14847. [https://doi.org/10.1002/hyp.14847](https://doi.org/10.1002/hyp.14847)
  - Code: [https://github.com/jmframe/mclstm_camels_2022](https://github.com/jmframe/mclstm_camels_2022)
  - HydroShare Archive: [`10.4211/hs.d750278db868447dbd252a8c5431affd`](https://doi.org/10.4211/hs.d750278db868447dbd252a8c5431affd)

---

## 2. Model Overview & Architecture

The Mass-Conserving LSTM (MC-LSTM; Hoedt et al., 2021) modifies the recurrent cell state update of the standard LSTM so that the cell states $\mathbf{c}[t] \in \mathbb{R}_{\ge 0}^K$ act as non-negative physical mass buckets whose sum is conserved across time up to incoming mass fluxes $\mathbf{x}_m[t]$ and outgoing mass fluxes (discharges and sink losses).

At each daily timestep $t$ (exactly as implemented in `MCLSTM.forward`, following NeuralHydrology's `mclstm`):
1. **Gate features**: every gate sees the same feature vector $\mathbf{f}[t] = [\,x_m[t],\ \mathbf{a}[t],\ \mathbf{c}[t-1] / (\lVert\mathbf{c}[t-1]\rVert_1 + 10^{-5})\,]$, i.e. the mass input, the auxiliary inputs and the L1-normalized previous cell state.
2. **Input Gate ($\mathbf{i}[t]$)**: distributes incoming daily precipitation mass $x_m[t] \ge 0$ (`prcp(mm/day)`, kept in raw $\text{mm day}^{-1}$ rather than standardized) across the $K=64$ mass cells with an L1-normalized sigmoid ("normalized sigmoid"):
   $$\mathbf{i}[t] = \frac{\sigma(\mathbf{W}_i \mathbf{f}[t] + \mathbf{b}_i)}{\lVert\sigma(\mathbf{W}_i \mathbf{f}[t] + \mathbf{b}_i)\rVert_1}$$
3. **Redistribution Matrix ($\mathbf{R}[t] \in \mathbb{R}^{K \times K}$)**: a row-stochastic matrix ($\sum_{j} R_{kj}[t] = 1$, $R_{kj}[t] \ge 0$) obtained by L1-normalizing a ReLU ("normalized ReLU") of $\mathbf{W}_r \mathbf{f}[t] + \mathbf{b}_r$ row-wise; it moves stored mass between cells without creating or destroying it.
4. **Output Gate ($\mathbf{o}[t]$)**: the fraction of mass leaving each cell, $\mathbf{o}[t] = \sigma(\mathbf{W}_o \mathbf{f}[t] + \mathbf{b}_o)$, with $\mathbf{b}_o$ initialized to `initial_forget_bias = 3.0`. All gate weights use orthogonal initialization.
5. **Mass update**:
   $$\mathbf{m}[t] = \mathbf{R}[t]^{\top}\mathbf{c}[t-1] + x_m[t]\,\mathbf{i}[t], \qquad \mathbf{h}[t] = \mathbf{o}[t] \odot \mathbf{m}[t], \qquad \mathbf{c}[t] = (\mathbf{1} - \mathbf{o}[t]) \odot \mathbf{m}[t]$$
6. **Auxiliary Inputs ($\mathbf{a}[t]$)**: all non-precipitation dynamic forcings of the selected product(s) and the 27 static catchment attributes of Kratzert et al. (2019a), z-scored over training basins × training target days. They drive the gates but never enter the conserved mass state.
7. **Streamflow Readout & Trash Cell**: following Hoedt et al. (2021, Section 4.3), the outgoing flux of cell 0 (`h[t, 0]`) is an unaccounted mass sink ("trash cell" for evapotranspiration, deep percolation and forcing bias). Simulated discharge is the sum of the remaining $K-1 = 63$ outgoing fluxes:
   $$\hat{y}[t] = \sum_{k=1}^{K-1} h_k[t]$$

---

## 3. Clean-Room Implementation (`models/hoedt_mclstm/`)

- [`model.py`](model.py): Clean-room pure-PyTorch implementation (no Triton or custom CUDA kernels; the recurrence is a Python loop over time steps) with the cell class `MCLSTM` and the trainer `MCLSTMTrainer`. The model has 64 cells including the trash cell (`h[..., 0]`) and the sum readout (`h[..., 1:].sum(dim=-1)`), using the gates described above. Training uses the basin-averaged $\text{NSE}^*$ loss (Kratzert et al., 2019a; $\epsilon = 0.1$) on the **unnormalized** mm/d discharge, sequence length 365 days with sequence-to-one (last-step) supervision, batch size 256, and **Adam** with the Hoedt et al. (2021) step schedule `{0: 1e-2, 20: 5e-3, 25: 1e-3}` over **30 epochs**. Gradients are clipped at norm 1.0, there is no early stopping, and inference is windowed sequence-to-one for every evaluation day.
- [`runner.py`](runner.py): `run_fold` takes only the keys of `RUNNER_DEFAULTS` (`seq_len`, `batch_size`, `epochs`, `lr`, `lr_schedule`, `hidden_size`, `initial_forget_bias`, `nse_eps`, `clip_grad_norm`, `n_batches_per_epoch`, `eval_batch_size`, `device`, `return_members`); any other keyword raises. The precipitation column(s) are found by name (`prcp(mm/day)_<product>`). With one forcing product (`daymet`, `maurer` or `nldas`), that product's precipitation is the mass input. With several products (for example `--forcings daymet maurer nldas`), the mass input is the arithmetic mean of their precipitation, and every non-precipitation forcing of every product enters $\mathbf{a}[t]$. This is the "mean-precip + 3F gates" configuration (`hoedt_mclstm_3f_avgp`). The runner has no mode that averages three independently trained single-forcing models; the `hoedt_mclstm_3f_ens_b` model-average archive was assembled outside `run_model` from the single-forcing runs.
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

In the CAMELS-US MIP manuscript, MC-LSTM is represented by **three strictly separated provenance entries** in Table 3:
1. **`hoedt_mclstm_lit`** (*Literature*, Hoedt et al., 2021): Maurer 531-basin published scores (`ens_med_nse = 0.742`, `single_med_nse = 0.726 ± 0.004` in Supplement Table S1).
2. **`frame2023_mclstm_archive`** (*Archive*, Frame et al., 2023): Pre-computed 8-seed ensemble hydrographs from HydroShare (`10.4211/hs.d750278db868447dbd252a8c5431affd`) for Daymet (`t3_12`, Med NSE = `0.758`) and NLDAS (`t3_28`, Med NSE = `0.736`) in Supplement Table S1.
3. **`hoedt_mclstm`** (*MIP Re-run*, Hoedt et al., 2021): Complete 531-basin Zarr v2 hydrographs on the Gauged leaderboards:
   - **Table S1 (`gauged__single_camels`, Maurer)**: Ensemble Med NSE = **0.743** (`n=3`, `hoedt_mclstm_maurer`), Single-Model Med NSE = **0.716 ± 0.003** (`hoedt_mclstm_maurer_single`)
   - **Table S2 (`gauged__multi_camels`, 3-Forcing)**: Mean-Precip + 3F Gates (`hoedt_mclstm_3f_avgp`) Ensemble Med NSE = **0.777** (`n=8`), Single-Model Med NSE = **0.754 ± 0.005** (`hoedt_mclstm_3f_avgp_single`); 3-Forcing Model-Average Ensemble (`hoedt_mclstm_3f_ens_b`) Med NSE = **0.771** (`n=3`)
