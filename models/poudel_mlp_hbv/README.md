# Poudel & Steinschneider (2026) — Static-Parameter MLP + 16-HBV (`poudel_mlp_hbv`)

## Reference & Links
- **Citation**: Poudel, S., & Steinschneider, S. (2026). An argument for parsimony in differentiable hydrologic models. *Hydrology and Earth System Sciences*, 30(15), 5173–5193.
- **Paper URL**: https://doi.org/10.5194/hess-30-5173-2026
- **Official Code Repository**: https://github.com/snpoudel/diff-hydro
- **Hydrograph Archive DOI**: `""` (no published daily hydrograph archive; preprocessed input mirror at `https://zenodo.org/records/15529996`)
- **Provenance Category**: `MIP Re-run (Our Code)`

## Architecture
- **Parameter Network**: 3-layer Multilayer Perceptron (MLP; hidden dimension $2048$ with ReLU activations, dropout, and sigmoid output layer) mapping static catchment attributes to time-invariant physical parameters for $N_z = 16$ parallel HBV conceptual sub-units per basin.
- **Differentiable 16-Bucket HBV Core**: Degree-day snow accumulation/melt and liquid-water retention/refreezing, nonlinear Beta soil-moisture accounting, Hamon potential evapotranspiration, two-layer upper/lower response reservoirs (quickflow, interflow, percolation, baseflow), and causal two-parameter Gamma unit-hydrograph routing.

## Stage-1 Author-Protocol Confirmation (`Split-2C`)
Poudel & Steinschneider (2026) evaluated `MLP + 16HBV` on **Split-2C** across the 531 CAMELS-US basins using an 80%/20% random spatial split into **425 in-sample training gauges** and **106 held-out PUB test gauges** (`camels_test_basins.csv`), driven by Daymet daily forcings (`1990-01-01` to `2005-12-31` training, `2006-01-01` to `2014-12-31` validation, `1980-01-01` to `1989-12-31` evaluation):

| Split-2C Regime (`1980-01-01` to `1989-12-31`, Daymet) | Basins | Author Published Med NSE | Stage-1 Reproduced Med NSE | Post-Spinup (`WY1981–1989`) Med NSE | $\Delta$ Med NSE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gauged (OOS in Time)** | 425 | `0.710` (`0.71`) | **`0.7177` (`0.718`)** | `0.7334` | `+0.0077` (`+0.008`) | **PASS** |
| **PUB Spatial Hold-Out (OOS in Space)** | 106 | `0.630` (`0.63`; `0.67` ref) | **`0.6539` (`0.654`)** | `0.6609` | `+0.0239` (vs `0.63`) / `-0.0161` (vs `0.67`) | **PASS** |

Verify the Stage-1 confirmation report via:
```bash
# Print the tracked Stage-1 report
python -m models.poudel_mlp_hbv.reproduce_author_metrics --mode verify

# Re-train and evaluate on Split-2C from scratch
python -m models.poudel_mlp_hbv.reproduce_author_metrics --mode train_and_eval --out outputs/poudel_stage1.json
```
- `--mode train_and_eval` writes to `--out`. The default is `models/poudel_mlp_hbv/stage1_author_reproduction_results.local.json`, which is not tracked by git. The tracked `stage1_author_reproduction_results.json` is overwritten only if `--out` points at it explicitly.
- Evaluation uses `eval_spinup_days=0`: the `1980-01-01` evaluation start is the first day of the data bundle, so there is no pre-evaluation spinup.

## Build Prerequisites
The HBV kernels in `model.py` come in two forms: C++/OpenMP kernels, used for all inference and for training on CPU, and CUDA kernels, used for training on GPU. Inference always uses the CPU OpenMP kernel, so a GPU host needs both a working `nvcc` and an OpenMP-capable `g++`. The kernels are JIT-compiled with `torch.utils.cpp_extension.load_inline` on the first call, which takes a few minutes. The build is cached in `~/.cache/torch_extensions`. Set `TORCH_EXTENSIONS_DIR` to change the cache location, for example to a separate directory for each concurrent process. You need:
- **C++ compiler:** Linux `g++` with OpenMP. On macOS (Apple clang) the C++ kernels build without `-fopenmp` and run single-threaded.
- **`ninja`:** `pip install ninja`. torch cannot load C++ extensions without it.
- **GPU only:** an `nvcc` whose version matches `torch.version.cuda`, with `CUDA_HOME` pointing at that toolkit.

## Stage-2 CAMELS-US MIP Protocol Results (Supplement Tables S1, S4, and S7)
Evaluated on all 531 basins over `1989-10-01` to `1999-09-30` (trained on `1999-10-01` to `2008-09-30`, Daymet forcing, 3 random seeds `[101, 102, 103]`):

| Leaderboard | Row ID | Mode | Members | Med NSE | Mean NSE | Med KGE | Med \|FHV\| (%) | `NSE <= 0` | Zarr Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Table S1 (`gauged__single_camels`)** | `poudel_mlp_hbv_daymet` | `ensemble` | 3 | **`0.735`** | `0.657` | `0.738` | `16.77` | `7` | `gs://camels-us-mip/hydrographs/gauged__single_camels/poudel_mlp_hbv_daymet_3s.zarr` |
| **Table S1 (`gauged__single_camels`)** | `poudel_mlp_hbv_daymet_single` | `single_member_mean` | 3 | **`0.733 ± 0.003`** | `0.649` | `0.739` | `17.17` | `9` | `gs://camels-us-mip/hydrographs/gauged__single_camels/poudel_mlp_hbv_daymet_3s.zarr` |
| **Table S4 (`pub__single_camels`)** | `poudel_mlp_hbv_pub_daymet` | `ensemble` | 3 | **`0.678`** | `0.517` | `0.665` | `23.93` | `23` | `gs://camels-us-mip/hydrographs/pub__single_camels/poudel_mlp_hbv_pub_daymet.zarr` |
| **Table S4 (`pub__single_camels`)** | `poudel_mlp_hbv_pub_daymet_single` | `single_member_mean` | 3 | **`0.669 ± 0.001`** | `0.505` | `0.660` | `24.12` | `23` | `gs://camels-us-mip/hydrographs/pub__single_camels/poudel_mlp_hbv_pub_daymet.zarr` |
| **Table S7 (`pur__single_camels`)** | `poudel_mlp_hbv_pur_daymet` | `ensemble` | 3 | **`0.624`** | `0.371` | `0.610` | `26.73` | `41` | `gs://camels-us-mip/hydrographs/pur__single_camels/poudel_mlp_hbv_pur_daymet.zarr` |
| **Table S7 (`pur__single_camels`)** | `poudel_mlp_hbv_pur_daymet_single` | `single_member_mean` | 3 | **`0.618 ± 0.005`** | `0.356` | `0.608` | `26.74` | `42` | `gs://camels-us-mip/hydrographs/pur__single_camels/poudel_mlp_hbv_pur_daymet.zarr` |
