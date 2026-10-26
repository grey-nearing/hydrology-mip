# SAC-SMA + Snow-17 (`newman_sacsma` & `nearing_sacsma`)

In accordance with the CAMELS-US MIP single-citation and single-provenance attribution norms, the coupled US National Weather Service **Snow-17** snow accumulation/ablation model (Anderson, 1973, 2006) and **Sacramento Soil Moisture Accounting (SAC-SMA)** conceptual rainfall-runoff model (Burnash et al., 1973; Burnash, 1995) routed with a Duamel Gamma unit hydrograph is represented by two distinct entries in `leaderboards/table_2_registry.json`:

1. **Entry #1 — `newman_sacsma`: `SAC-SMA + Snow-17 (Basin-Calibrated)` (`Archive*`)**
   - **Canonical Citation**: Newman, A. J., Mizukami, N., Clark, M. P., Wood, A. W., Littell, J. S., & Nijssen, B. (2017). Benchmarking of a physically based hydrologic model. *Journal of Hydrometeorology*, 18(8), 2215–2225.
   - **Paper URL**: https://doi.org/10.1175/JHM-D-16-0284.1
   - **Official Code / Dataset Repository**: https://github.com/NCAR/camels
   - **Data / Archive DOIs**: `10.5065/D6MW2F4D` (NCAR GDEX CAMELS-US dataset); `10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1` (HydroShare benchmark archive, Kratzert et al., 2019b)
   - **Provenance Category**: `Archive*` (`code_provenance: "Author Archive"`, `asterisk: true`, evaluated on 516 of 531 CAMELS basins available in the Newman et al. 2017 archive; 530 non-zero basins in `t3_48.zarr` where `01634500` is all-zero)
   - **Leaderboards**: `gauged__single_camels` (`id: "t3_48"`)

2. **Entry #2 — `nearing_sacsma`: `SAC-SMA + Snow-17 (DDS & Donor Transfer)` (`MIP Re-run (Our Code)`)**
   - **Canonical Citation**: Nearing, G., Sampson, A., Kratzert, F., & Frame, J. (2020). Post-Processing a Conceptual Rainfall-Runoff Model with an LSTM. *EarthArXiv Preprint*.
   - **Paper URL**: https://doi.org/10.31223/osf.io/53te4
   - **Reference Code Repository**: https://github.com/Upstream-Tech/SACSMA-SNOW17
   - **MIP Clean-Room Code Directory**: `models/newman_sacsma`
   - **Provenance Category**: `MIP Re-run (Our Code)` (`code_provenance: "Our Code"`, `asterisk: false`, evaluated on all 531 canonical CAMELS-US basins)
   - **Leaderboards**: `gauged__single_camels`, `gauged__multi_camels`, `pub__single_camels`, `pub__multi_camels`, `pur__single_camels`, `pur__multi_camels`

---

## Clean-Room Architecture & Parameterization (`models/newman_sacsma/model.py`)

To avoid licensing restrictions in third-party wrappers (`Upstream-Tech/SACSMA-SNOW17` has no open-source license and NCAR's Fortran driver is `GPL-3.0-only`), `models/newman_sacsma/model.py` implements a 100% first-party, clean-room C++17/OpenMP (`_SACSMA_OMP_CPP`) and NumPy realization directly from the public-domain mathematical equations:

1. **Snow-17 Accumulation & Ablation (10 calibrated parameters: `scf`, `mfmax`, `mfmin`, `uadj`, `si`, `pxtemp`, `nmf`, `tipm`, `mbase`, `plwhc`)**:
   - Rain/snow partitioning via smooth logistic transition around `pxtemp`, internal snowpack energy deficit tracking via antecedent temperature index (`tipm`, `nmf`), seasonally varying degree-day melt factor (`mfmin`, `mfmax`) above `mbase`, power-law areal snow depletion curve $A_s = \min(1, (\text{SWE}/\text{si})^{0.75})$, rain-on-snow wind enhancement $(1 + 0.1\,\text{uadj}\,P_{\text{rain}})$, and liquid-water holding capacity (`plwhc`).
2. **SAC-SMA Soil Moisture Accounting (15 calibrated parameters: `uztwm`, `uzfwm`, `lztwm`, `lzfpm`, `lzfsm`, `adimp`, `uzk`, `lzpk`, `lzsk`, `zperc`, `rexp`, `pctim`, `pfree`, `riva`, `side`, with fixed `rserv = 0.30`)**:
   - Six soil-moisture states (`uztwc`, `uzfwc`, `lztwc`, `lzfpc`, `lzfsc`, `adimc`) across permanent impervious (`pctim`), variable additional impervious (`adimp`), and pervious ($1 - \text{pctim} - \text{adimp}$) areas, forced by `1.15 × Hargreaves–Samani (1985)` potential evapotranspiration, nonlinear lower-zone percolation (`zperc`, `rexp`, `pfree`), exponential upper-zone interflow (`uzk`), and primary/supplemental baseflow (`lzpk`, `lzsk`, `side`).
3. **Duamel Gamma Unit-Hydrograph Routing (2 calibrated parameters: `uh_shape`, `uh_scale`)**:
   - Convolves fast channel inflow (impervious + surface + interflow runoff) with a normalized 15-day discrete Gamma unit hydrograph and adds observable lower-zone baseflow.
4. **DDS Basin Calibration & 29-D Nearest-Neighbor Donor Transfer (`PUB` / `PUR`)**:
   - **Gauged**: Each basin's 27 parameters are calibrated independently on `1999-10-01` to `2008-09-30` (`365`-day spinup) using vectorized Dynamically Dimensioned Search (DDS; Tolson & Shoemaker, 2007; `n_iter = 400`, $r = 0.2$, daily RMSE objective).
   - **Ungauged (`12-fold PUB` & `18-fold HUC-02 PUR`)**: Each held-out test basin $j \notin \mathcal{B}_{\text{train}}$ inherits the DDS-calibrated parameter vector of its $k=1$ nearest training donor basin in a 29-dimensional feature space combining scaled coordinates $[0.2\,\text{lat}_{\text{deg}},\, 0.2\,\text{lon}_{\text{deg}}]$ and the 27 z-score standardized static CAMELS catchment attributes (standardized strictly on $\mathcal{B}_{\text{train}}$, with zero test-basin streamflow leakage).

---

## Stage-1 Author-Protocol Confirmation (`Newman Split-1`)

| Benchmark / Run | Family ID | Forcing | Basins | Members | Published Med NSE | Reproduced Med NSE | $\Delta$ Med NSE | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Newman et al. (2017) Archive (`t3_48`)** | `newman_sacsma` | Daymet / Maurer | 516 (530 in Zarr; 447 mask) | 10 (archived as 1 mean) | `0.603` (447 mask) | **`0.6071`** (`0.6028` on 447) | `-0.0002` | **PASS** |
| **Nearing et al. (2020) / Kratzert et al. (2021) DDS** | `nearing_sacsma` | Daymet | 531 | 1 | `0.670` | **`0.6609`** | `-0.0091` | **PASS** |
| **Nearing et al. (2020) / Kratzert et al. (2021) 3-Forcing** | `nearing_sacsma` | D+M+N | 531 | 3 | `0.705` (30-mem Lit.) | **`0.6954`** | `-0.0096` | **PASS** |

Verify Stage-1 confirmation via:
```bash
python models/newman_sacsma/reproduce_author_metrics.py --mode verify
```

---

## Stage-2 CAMELS-US MIP Protocol Results (Tables 3, 4, 7, 8, 11, and 12)

| Leaderboard | Row ID | Family ID | Display Name | Mode | Members | Med NSE | Mean NSE | Med KGE | Med \|FHV\| (%) | `NSE <= 0` | Zarr Path |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Table 3 (`gauged__single_camels`)** | `t3_48` | `newman_sacsma` | `SAC-SMA + Snow-17 (Basin-Calibrated, 10-member SCE-UA ensemble)*` | `ensemble` | 10 | **`0.607`** | `0.573` | `0.636` | `21.87` | `13` | `gs://camels-us-mip/hydrographs/gauged__single_camels/t3_48.zarr` |
| **Table 3 (`gauged__single_camels`)** | `sacsma_gauged_single_daymet` | `nearing_sacsma` | `SAC-SMA + Snow-17 (DDS, single model)` | `ensemble` | 1 | **`0.661`** | `0.613` | `0.677` | `20.52` | `12` | `gs://camels-us-mip/hydrographs/gauged__single_camels/sacsma_daymet.zarr` |
| **Table 4 (`gauged__multi_camels`)** | `sacsma_gauged_multi_3f` | `nearing_sacsma` | `SAC-SMA + Snow-17 (DDS, 3-Forcing Ensemble)` | `ensemble` | 3 | **`0.695`** | `0.665` | `0.645` | `25.59` | `4` | `gs://camels-us-mip/hydrographs/gauged__multi_camels/sacsma_3f_ens.zarr` |
| **Table 4 (`gauged__multi_camels`)** | `sacsma_gauged_multi_3f_single` | `nearing_sacsma` | `SAC-SMA + Snow-17 (DDS, single model)` | `single_member_mean` | 3 | **`0.691 ± 0.001`** | `0.661` | `0.651` | `24.64` | `5` | `gs://camels-us-mip/hydrographs/gauged__multi_camels/sacsma_3f_ens.zarr` |
| **Table 7 (`pub__single_camels`)** | `sacsma_pub_single_daymet` | `nearing_sacsma` | `SAC-SMA + Snow-17 (Donor Transfer, single model)` | `ensemble` | 1 | **`0.564`** | `0.392` | `0.575` | `27.02` | `42` | `gs://camels-us-mip/hydrographs/pub__single_camels/sacsma_pub_daymet.zarr` |
| **Table 8 (`pub__multi_camels`)** | `sacsma_pub_multi_3f` | `nearing_sacsma` | `SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble)` | `ensemble` | 3 | **`0.625`** | `0.504` | `0.570` | `30.22` | `26` | `gs://camels-us-mip/hydrographs/pub__multi_camels/sacsma_pub_3f.zarr` |
| **Table 8 (`pub__multi_camels`)** | `sacsma_pub_multi_3f_single` | `nearing_sacsma` | `SAC-SMA + Snow-17 (Donor Transfer, single model)` | `single_member_mean` | 3 | **`0.618 ± 0.000`** | `0.494` | `0.578` | `29.41` | `28` | `gs://camels-us-mip/hydrographs/pub__multi_camels/sacsma_pub_3f.zarr` |
| **Table 11 (`pur__single_camels`)** | `sacsma_pur_single_daymet` | `nearing_sacsma` | `SAC-SMA + Snow-17 (Donor Transfer, single model)` | `ensemble` | 1 | **`0.512`** | `-0.006` | `0.528` | `30.69` | `72` | `gs://camels-us-mip/hydrographs/pur__single_camels/sacsma_pur_daymet.zarr` |
| **Table 12 (`pur__multi_camels`)** | `sacsma_pur_multi_3f` | `nearing_sacsma` | `SAC-SMA + Snow-17 (Donor Transfer, 3-Forcing Ensemble)` | `ensemble` | 3 | **`0.569`** | `0.230` | `0.537` | `33.97` | `46` | `gs://camels-us-mip/hydrographs/pur__multi_camels/sacsma_pur_3f.zarr` |
| **Table 12 (`pur__multi_camels`)** | `sacsma_pur_multi_3f_single` | `nearing_sacsma` | `SAC-SMA + Snow-17 (Donor Transfer, single model)` | `single_member_mean` | 3 | **`0.560 ± 0.003`** | `0.214` | `0.541` | `33.15` | `48` | `gs://camels-us-mip/hydrographs/pur__multi_camels/sacsma_pur_3f.zarr` |
