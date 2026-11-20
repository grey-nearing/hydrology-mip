# CAMELS-US Daily Splits (`mips/camels_us_daily/splits/`)

This folder holds the basin lists, the time periods, and the spatial cross-validation folds of
the CAMELS-US Daily MIP. These files define the benchmark. **Never change them.** Model code never
reads them: `models/run_model.py` reads them and passes the basins and periods to each model.

## Files

| File | Content |
| :--- | :--- |
| `basins_531.txt`, `basins_531.json` | The 531 benchmark basins (8-digit USGS gauge IDs, Newman et al., 2017), sorted. |
| `basins_671.txt`, `basins_671.json` | All 671 CAMELS-US basins. |
| `basins_447_kratzert2019b.txt`, `basins_447_kratzert2019b.json` | The 447-basin subset of Kratzert et al. (2019a), where all benchmark process models of that study have complete simulations. |
| `splits_and_regimes.json` | Machine-readable protocol facts: periods, units, regime descriptions, the map `basin_to_huc02` from each of the 531 basins to its two-digit USGS region, and the test basins of every fold (`pub_12fold_test_basins`, `pur_18huc02_test_basins`). |
| `pub_12fold_random_splits.json` | The 12 PUB folds (regime `pub_12fold`). |
| `pub_12fold/fold_{00..11}_{train,test}.txt` | The same 12 PUB folds as plain-text basin lists. |
| `pur_18fold_huc02_splits.json` | The 18 PUR folds (regime `pur_18huc02`). |
| `pur_18fold/huc_{01..18}_{train,test}.txt` | The same 18 PUR folds as plain-text basin lists. |

The PUR regime ID is `pur_18huc02`. Use it in `protocol.json`, on the command line
(`--spatial-regime pur_18huc02`), and in stores. The PUR file and folder keep the older name
`pur_18fold`. Both names mean the same 18 folds.

## Time periods

All regimes use the periods of Newman et al. (2017) and Kratzert et al. (2019a):

- **Training period:** 1999-10-01 to 2008-09-30 (water years 2000 to 2008; 3,288 days). With a
  one-year spin-up, model inputs start on 1998-10-01.
- **Evaluation period:** 1989-10-01 to 1999-09-30 (water years 1990 to 1999; 3,652 days). With a
  one-year spin-up, model inputs start on 1988-10-01.

## Spatial regimes

1. **Gauged (`gauged`).** Train on all 531 basins in the training period. Predict all 531 basins in
   the evaluation period. There is no split file.
2. **Prediction in ungauged basins (`pub_12fold`).** A random 12-fold split of the 531 basins
   (`random_state=42`). Each fold tests 44 or 45 basins and trains on the other 486 or 487. The
   test predictions of all 12 folds are joined into one 531-basin hydrograph for the evaluation
   period.
3. **Prediction in ungauged regions (`pur_18huc02`).** An 18-fold split by the 18 two-digit USGS
   hydrologic regions (`huc_02` codes `01` to `18`). Each fold tests all basins of one region and
   trains on the basins of the other 17 regions. The test predictions of all 18 folds are joined
   into one 531-basin hydrograph for the evaluation period.

In every PUB and PUR fold, the training and test basins do not overlap, together they cover all
531 basins, and every basin is tested in exactly one fold. In PUR, no region appears in both the
training and the test set of a fold. `models/run_model.py` and `mips/camels_us_daily/adapter.py`
check all of this before they use the folds.

## Split file format

Both split JSON files have a `folds` object. Each fold has these keys:

| Key | Meaning |
| :--- | :--- |
| `fold_index` | The position of the fold, starting at 0. |
| `n_train`, `n_test` | The number of training and test basins. |
| `train_basins`, `test_basins` | The basin IDs. |
| `train_indices`, `test_indices` | The positions of those basins in `basins_531.txt`. |
| `huc_02` | PUR only: the held-out region. |

The fold IDs are `fold_00` to `fold_11` (PUB) and `huc_01` to `huc_18` (PUR). The files also
record how they were made (`construction`) and the periods. `pub_12fold_random_splits.json` also
holds two reference blocks, `kratzert2019a_wrr_multiseed_test_basins` and
`huc_stratified_12fold_seed20260410`. They are not benchmark folds; the runner and the adapter
read only `folds`.

## 12-fold random PUB folds

| Fold | Train basins | Test basins | First 5 test basins |
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

## 18-fold leave-one-region-out PUR folds (`pur_18huc02`)

| Fold | HUC-02 | Train basins | Test basins | First 5 test basins |
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

## Load the folds in Python

```python
from mips.camels_us_daily import load_splits_for_regime

for regime in ("gauged", "pub_12fold", "pur_18huc02"):
    folds = load_splits_for_regime(regime)
    print(regime, len(folds), len(folds[0].test_entities))
# gauged 1 531
# pub_12fold 12 45
# pur_18huc02 18 23
```

`load_splits_for_regime` checks every fold before it returns it (see
[`../README.md`](../README.md#how-to-take-part)).
