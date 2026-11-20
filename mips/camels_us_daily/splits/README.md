# CAMELS-US MIP Canonical Splits

This directory contains the basin lists, time periods, and spatial cross-validation folds for the CAMELS-US Model Intercomparison Project (MIP).

## Time Periods

All models evaluated in the standard protocol use the Newman et al. (2017) and Kratzert et al. (2019b) split:

- **Training / Calibration Period**: `1999-10-01` to `2008-09-30` (9 water years; 3,288 days). Models that use a 1-year spin-up or lookback window start on `1998-10-01`.
- **Evaluation / Testing Period**: `1989-10-01` to `1999-09-30` (10 water years; 3,652 days). Models that use a 1-year spin-up or lookback window start on `1988-10-01`.

## Basin Lists

- `basins_531.txt` and `basins_531.json`: The 531 USGS 8-digit gauge IDs from Newman et al. (2017), sorted in ascending order.
- `basins_447_kratzert2019b.txt` and `basins_447_kratzert2019b.json`: The 447-basin subset from Kratzert et al. (2019b) where all benchmark process models have complete simulations.
- `splits_and_regimes.json`: Machine-readable protocol metadata, including time windows, spatial regime definitions, and the mapping from each of the 531 gauges to its 2-digit USGS hydrologic region (`huc_02`).

## Spatial Cross-Validation Regimes

1. **Gauged (`gauged`)**: Train on all 531 basins during the training period (`1999-10-01` to `2008-09-30`) and evaluate on all 531 basins during the testing period (`1989-10-01` to `1999-09-30`).
2. **Prediction in Ungauged Basins (`pub_12fold`)**: 12-fold random cross-validation across the 531 basins (`random_state=42`). Each fold holds out 44 or 45 test basins and trains on the remaining 486 or 487 basins. Out-of-sample test predictions from all 12 folds are combined into a single 531-basin array on the testing period (`1989-10-01` to `1999-09-30`).
   - `pub_12fold_random_splits.json`: Complete fold definitions in JSON format.
   - `pub_12fold/fold_{00..11}_train.txt` and `pub_12fold/fold_{00..11}_test.txt`: Plain-text lists of training and testing gauge IDs for each fold.
3. **Prediction in Ungauged Regions (`pur_18fold`)**: 18-fold leave-one-region-out cross-validation across the 18 USGS 2-digit hydrologic regions (`huc_02` codes `01` through `18`). Each fold holds out all basins in one `huc_02` region for testing and trains on all basins in the remaining 17 regions. Out-of-sample test predictions from all 18 folds are combined into a single 531-basin array on the testing period (`1989-10-01` to `1999-09-30`).
   - `pur_18fold_huc02_splits.json`: Complete regional fold definitions in JSON format.
   - `pur_18fold/huc_{01..18}_train.txt` and `pur_18fold/huc_{01..18}_test.txt`: Plain-text lists of training and testing gauge IDs for each regional fold.

## 12-Fold Random PUB Summary

| Fold | Train Basins | Test Basins | First 5 Test Basins |
| :--- | :---: | :---: | :--- |
| `fold_00` | 486 | 45 | `01047000, 01057000, 01123000, 01134500, 01415000` |
| `fold_01` | 486 | 45 | `01022500, 01137500, 01170100, 01333000, 01435000` |
| `fold_02` | 486 | 45 | `01144000, 01181000, 01484100, 01539000, 01632000` |
| `fold_03` | 487 | 44 | `01055000, 01162500, 01169000, 01350000, 01350080` |
| `fold_04` | 487 | 44 | `01052500, 01073000, 01365000, 01414500, 01451800` |
| `fold_05` | 487 | 44 | `01440400, 01567500, 01644000, 02013000, 02015700` |
| `fold_06` | 487 | 44 | `01466500, 02056900, 02059500, 02152100, 02177000` |
| `fold_07` | 487 | 44 | `01518862, 01594950, 02017500, 02096846, 02112120` |
| `fold_08` | 487 | 44 | `01054200, 01139000, 01411300, 01413500, 01434025` |
| `fold_09` | 487 | 44 | `01078000, 01142500, 01487000, 01516500, 01583500` |
| `fold_10` | 487 | 44 | `01031500, 01139800, 01439500, 01543500, 01547700` |
| `fold_11` | 487 | 44 | `01187300, 01195100, 01543000, 01544500, 01557500` |

## 18-Fold Leave-One-HUC02-Out PUR Summary

| Fold | HUC-02 | Train Basins | Test Basins | First 5 Test Basins |
| :--- | :---: | :---: | :---: | :--- |
| `huc_01` | `01` | 508 | 23 | `01022500, 01031500, 01047000, 01052500, 01054200` |
| `huc_02` | `02` | 462 | 69 | `01333000, 01350000, 01350080, 01350140, 01365000` |
| `huc_03` | `03` | 452 | 79 | `02046000, 02051500, 02053200, 02053800, 02055100` |
| `huc_04` | `04` | 502 | 29 | `04015330, 04024430, 04027000, 04040500, 04043050` |
| `huc_05` | `05` | 496 | 35 | `03010655, 03011800, 03015500, 03021350, 03026500` |
| `huc_06` | `06` | 515 | 16 | `03439000, 03455500, 03456500, 03460000, 03463300` |
| `huc_07` | `07` | 502 | 29 | `05291000, 05362000, 05393500, 05399500, 05408000` |
| `huc_08` | `08` | 524 | 7 | `07291000, 07359610, 07362100, 07362587, 07375000` |
| `huc_09` | `09` | 529 | 2 | `05057200, 05120500` |
| `huc_10` | `10` | 482 | 49 | `06221400, 06224000, 06278300, 06280300, 06289000` |
| `huc_11` | `11` | 509 | 22 | `07057500, 07060710, 07066000, 07083000, 07142300` |
| `huc_12` | `12` | 499 | 32 | `08023080, 08050800, 08066200, 08066300, 08070000` |
| `huc_13` | `13` | 524 | 7 | `08267500, 08269000, 08271000, 08324000, 08377900` |
| `huc_14` | `14` | 516 | 15 | `09035800, 09035900, 09047700, 09065500, 09066000` |
| `huc_15` | `15` | 517 | 14 | `09386900, 09404450, 09430600, 09447800, 09484600` |
| `huc_16` | `16` | 526 | 5 | `10234500, 10244950, 10336645, 10336660, 10343500` |
| `huc_17` | `17` | 459 | 72 | `12010000, 12013500, 12020000, 12025700, 12035000` |
| `huc_18` | `18` | 505 | 26 | `10259000, 11124500, 11141280, 11143000, 11148900` |
