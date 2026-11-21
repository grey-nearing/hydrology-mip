# Yin et al. (2022) Rainfall-Runoff Transformer (`RR-Former`)

Independent clean-room PyTorch implementation of **RR-Former** from **Yin, Guo, Zhang, Chen, & Zhang (2022)** (*Journal of Hydrology*, 609, 127781, <https://doi.org/10.1016/j.jhydrol.2022.127781>).

## Architecture & Two-Stage Training Protocol

- **Encoder-Decoder Transformer:** `d_model=64`, `nhead=4`, `num_encoder_layers=4`, `num_decoder_layers=4`, `dim_feedforward=256`, `dropout=0.1`, `decode_mode="NAR"`, learnable `src_pos_emb` / `tgt_pos_emb`, causal + past-only key self-attention mask, linear head `Linear(64, 1)`, `use_static=False`, `use_future_fea=True`.
- **Inputs:** Daymet 5 dynamic meteorological channels (`prcp, srad, tmax, tmin, vp`) over `[t_0 - past_len .. t_0 + pred_len - 1]` and standardized lagged discharge history `[q_std, q_valid]` over `[t_0 - past_len .. t_0 - 1]` followed by zero target queries.
- **Stage A (Global Pretraining):** `NSELoss`, `Adam(lr=1e-3)`, `batch_size=512`, linear warmup (`0.25 * epochs`) + `0.99` exponential decay, selecting the best global checkpoint by validation NSE.
- **Stage B (Full-Network Per-Basin Fine-Tuning):** Every basin initializes from the best global pretrained checkpoint and fine-tunes **all Transformer weights** via `Adam(lr=1e-3)` and `MSELoss` with warmup + `0.99` decay and per-basin validation NSE early stopping.

## Stage-1 Author-Protocol Probe Results

- **Protocol (`models/yin2022_rrformer/reproduce_author_metrics.py`):** 671 CAMELS-US basins present in `camels_671_1980_2014.npz` (`06775500` and `06846500` absent), Daymet 5 dynamic channels, `use_static=False`, `past_len=15`, `pred_len=7`, train `1980-10-01`–`1995-09-30`, val `1995-10-01`–`2000-09-30`, test `2000-10-01`–`2014-09-30`, global pretraining (`epochs=120`, `n_batches_per_epoch=1024`) + per-basin full-network fine-tuning (`finetune_epochs=200`), seed `601` (`10,586.2 s` on 2x NVIDIA T4).
- **Results (`models/yin2022_rrformer/stage1_author_reproduction_results.json`):**
  - **Day-1 Median NSE (`nse1_median`):** **0.8294** (mean `0.7881`; published `0.8265` median / `0.7904` mean, $\Delta = +0.0029$).
  - **7-Day Mean NSE (`test_nse_mean_median`):** **0.7723** (mean `0.7206`; per-lead medians Days 1–7: `[0.8294, 0.7896, 0.7731, 0.7655, 0.7576, 0.7552, 0.7422]`).
  - **Gate Status:** **`FAIL`** (`|0.7723 - 0.8265| = 0.0542 > 0.01` hard gate on 7-day `test_nse_mean`, with global pretraining `best_val_nse` reaching `0.6890` before early stop at epoch `79`). Per the Stage-1 hard gate, Stage-2 re-training was halted and existing `gauged__lagged_streamflow.json` RR-Former rows were left unchanged pending lead review.

## Usage

```bash
python -m models.yin2022_rrformer.reproduce_author_metrics --smoke
```
