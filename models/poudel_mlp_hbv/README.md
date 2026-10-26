# Poudel & Steinschneider (2026) — Static-Parameter MLP + 16-HBV (`poudel_mlp_hbv`)

## Canonical Reference & Links
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
python models/poudel_mlp_hbv/reproduce_author_metrics.py --mode verify
```

## Stage-2 CAMELS-US MIP Protocol Results (Tables 3, 7, and 11)
Evaluated on all 531 canonical basins over `1989-10-01` to `1999-09-30` (trained on `1999-10-01` to `2008-09-30`, Daymet forcing, 3 random seeds `[101, 102, 103]`):

| Leaderboard | Row ID | Mode | Members | Med NSE | Mean NSE | Med KGE | Med \|FHV\| (%) | `NSE <= 0` | Zarr Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Table 3 (`gauged__single_camels`)** | `poudel_mlp_hbv_daymet` | `ensemble` | 3 | **`0.708`** | `0.670` | `0.676` | `21.52` | `1` | `gs://camels-us-mip/hydrographs/gauged__single_camels/poudel_mlp_hbv_daymet_3s.zarr` |
| **Table 3 (`gauged__single_camels`)** | `poudel_mlp_hbv_daymet_single` | `single_member_mean` | 3 | **`0.698 ± 0.005`** | `0.656` | `0.665` | `21.74` | `3` | `gs://camels-us-mip/hydrographs/gauged__single_camels/poudel_mlp_hbv_daymet_3s.zarr` |
| **Table 7 (`pub__single_camels`)** | `poudel_mlp_hbv_pub_daymet` | `ensemble` | 3 | **`0.669`** | `0.572` | `0.635` | `24.02` | `17` | `gs://camels-us-mip/hydrographs/pub__single_camels/poudel_mlp_hbv_pub_daymet.zarr` |
| **Table 7 (`pub__single_camels`)** | `poudel_mlp_hbv_pub_daymet_single` | `single_member_mean` | 3 | **`0.661 ± 0.002`** | `0.556` | `0.625` | `24.22` | `20` | `gs://camels-us-mip/hydrographs/pub__single_camels/poudel_mlp_hbv_pub_daymet.zarr` |
| **Table 11 (`pur__single_camels`)** | `poudel_mlp_hbv_pur_daymet` | `ensemble` | 3 | **`0.627`** | `0.419` | `0.596` | `25.52` | `33` | `gs://camels-us-mip/hydrographs/pur__single_camels/poudel_mlp_hbv_pur_daymet.zarr` |
| **Table 11 (`pur__single_camels`)** | `poudel_mlp_hbv_pur_daymet_single` | `single_member_mean` | 3 | **`0.614 ± 0.001`** | `0.399` | `0.586` | `25.66` | `37` | `gs://camels-us-mip/hydrographs/pur__single_camels/poudel_mlp_hbv_pur_daymet.zarr` |
