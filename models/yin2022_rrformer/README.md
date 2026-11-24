# Yin et al. (2022) Rainfall-Runoff Transformer (`RR-Former`)

Independent clean-room PyTorch implementation of **RR-Former** from **Yin, Guo, Zhang, Chen, & Zhang (2022)** (*Journal of Hydrology*, 609, 127781, <https://doi.org/10.1016/j.jhydrol.2022.127781>), verified to float32 machine precision against the sole official public release (<https://github.com/iThronne/RR-Former>, `v0.0.1`, commit `add254f`, May 2023).

## Architecture & Two-Stage Training Protocol

- **Encoder-Decoder Transformer (`467,265` parameters for `pred_len=7`; `466,497` for causal `pred_len=1`):** `d_model=64`, `nhead=4`, `num_encoder_layers=4`, `num_decoder_layers=4`, `dim_feedforward=256`, `dropout=0.1` applied exclusively to feed-forward outputs before residual addition (Post-LayerNorm), `decode_mode="NAR"`, learnable `src_pos_emb` / `tgt_pos_emb`, historical-only decoder self-attention mask (bidirectional over `past_len` historical runoff tokens, masking future queries), linear readout `Linear(64, 1)`, `use_static=False`, `use_future_fea=True`.
- **Inputs:** Daymet 5 dynamic meteorological channels (`prcp, srad, tmax, tmin, vp`) over `[t_0 - past_len .. t_0 + pred_len - 1]` and single-channel standardized lagged discharge history `Q(t_0 - past_len .. t_0 - 1)` followed by zeros over target positions.
- **Upstream `v0.0.1` Bug Fixes:** Reproduces `iThronne/RR-Former` weights, forward pass (`≤ 4.77e-7`), `NSELoss` (`0.00`), gradients (`≤ 9.69e-8`), and LR schedule (`1.08e-19`) while fixing two concrete defects in `v0.0.1`: (1) `models/__init__.py` importing a missing `models.dnn` module, and (2) `utils/eval_model.py` passing `(y_hat, y_seq_future)` in reversed order to `calc_nse_torch(obs, sim)` during validation checkpoint selection.
- **Stage A (Global Pretraining):** Full `200` epochs (zero batch caps, zero early stopping), `NSELoss`, `Adam(lr=1e-3)`, `batch_size=512`, global z-score normalization, 50-epoch linear warmup + `0.99` exponential decay, selecting the best global checkpoint by validation NSE.
- **Stage B (Full-Network Per-Basin Fine-Tuning):** Every basin initializes from the best global pretrained checkpoint and fine-tunes **all Transformer weights** for full `200` epochs via `Adam(lr=1e-3)` and `MSELoss` with per-basin training z-score normalization and the same warmup/decay schedule, selecting the best per-basin validation checkpoint.

## Stage-1 Author-Protocol Confirmation Results (`PASS`)

- **Protocol (`models/yin2022_rrformer/reproduce_author_metrics.py`):** 671 CAMELS-US basins present in `camels_671_1980_2014.npz` (`06775500` and `06846500` absent), Daymet 5 dynamic channels, `use_static=False`, `past_len=15`, `pred_len=7`, train `1980-10-01`–`1995-09-30`, val `1995-10-01`–`2000-09-30`, test `2000-10-01`–`2014-09-30`, full 200-epoch global pretraining + full 200-epoch per-basin fine-tuning (seed `603`).
- **Results (`models/yin2022_rrformer/stage1_author_reproduction_results.json`):** Matches Yin et al. (2022, Tables 5–6) within `±0.01` median NSE across **all seven forecast lead days**:
  - **Day-1 Median NSE (`nse1_median`, Headline Gate):** **`0.8175`** vs. published **`0.8265`** (`Δ = -0.0090`, **`PASS`**).
  - **Days 2–7 Median NSE:** `[0.7783, 0.7644, 0.7494, 0.7438, 0.7338, 0.7221]` vs. published `[0.7832, 0.7674, 0.7582, 0.7484, 0.7423, 0.7282]` (all deltas `-0.0030` to `-0.0088`).
  - **7-Day Average Mean NSE:** **`0.7161`** vs. published **`0.7187`** (`Δ = -0.0026`).

## Stage-2 CAMELS-US MIP Benchmark (`gauged__lagged_streamflow`)

- **Protocol:** 531 Newman CAMELS-US basins, Daymet 5 dynamic channels (`use_static=False`), strictly causal `past_len=15`, `pred_len=1` (uses `Q(t-15..t-1)` and forcings only through day `t`), 3 seeds `(601, 602, 603)`, full 200-epoch pretraining + 200-epoch per-basin fine-tuning strictly inside `WY2000–2008` (chronological 2-year internal validation carve), blind evaluation over `WY1990–1999`.
- **Results:** 3-model ensemble median NSE = **`0.8003`** (mean `0.7705`, median KGE `0.8234`, `0` basins with `NSE ≤ 0`); single-model average median NSE = **`0.7633 ± 0.0215`** (`0.7576`, `0.7402`, `0.7920` per seed).

## Usage

```bash
python -m models.yin2022_rrformer.reproduce_author_metrics --smoke
```
