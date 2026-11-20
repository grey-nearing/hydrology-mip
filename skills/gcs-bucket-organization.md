---
name: gcs-bucket-organization
description: >-
  Complete architecture, directory hierarchy, Zarr dataset schemas, and
  Python/CLI access patterns for the private gs://camels-us-mip/ Google Cloud
  Storage bucket (splits, observations, and hydrographs across all 12
  leaderboards and community submissions). Use whenever reading from, writing
  to, scoring, or documenting datasets in gs://camels-us-mip/.
---

# `gs://camels-us-mip/` GCS Bucket Organization & Runbook

This skill documents the canonical layout, Zarr schemas, and access conventions for the private **`gs://camels-us-mip/`** Google Cloud Storage bucket used by the CAMELS-US Model Intercomparison Project (`camels-us-mip` and `hydromip.org`).

- **Bucket Root:** `gs://camels-us-mip/`
- **Bucket Root Documentation:** `gs://camels-us-mip/README.md`
- **Storage Class & Region:** Standard (`us-central1`)
- **Access Control:** Private (`public_access_prevention: enforced`, `uniform_bucket_level_access: true`). When the manuscript is published, this bucket will be tarred and zipped (`tar.gz`) and deposited in a permanent DOI repository on Zenodo.

---

## 1. Top-Level Partition Summary

`gs://camels-us-mip/` is divided into four top-level data partitions plus root documentation:

| Partition | Purpose | Format |
| :--- | :--- | :--- |
| **`splits/`** | Canonical 531-basin list, 447-basin intersection list, master protocol specification (`splits_and_regimes.json`), 12-fold random PUB split lists (`pub_12fold/`), and 18-fold HUC-02 PUR split lists (`pur_18fold/`). | `.json` & `.txt` (exact mirror of `splits/` in GitHub repo) |
| **`observations/`** | Canonical observed USGS daily specific discharge (`mm/day`) across all 531 Newman benchmark catchments (`newman_eval_targets.zarr`). | Cloud-native Zarr (`xarray.Dataset`) |
| **`stage1_author_reproductions/`** | Stage-1 author-protocol confirmation runs for every self-run model family (`<model_name>.zarr` with `qsim`, `qsim_members`, and `qobs` on the author's evaluation period, plus `<model_name>_results.json` and `manifest_index.json`). | Cloud-native Zarr (`xarray.Dataset`) & `.json` |
| **`hydrographs/`** | Stage-2 standardized daily model hydrographs (`(531, 3652)` in `mm/day` over `1989-10-01` to `1999-09-30`) organized by the 12 canonical leaderboards (`<spatial_regime>__<forcing_regime>`) plus `community/`. | Cloud-native Zarr (`xarray.Dataset`) |

---

## 2. Full Directory Hierarchy

```text
gs://camels-us-mip/
│
├── README.md                                            # Bucket architecture, coordinate schemas & Zenodo packaging guide
│
├── splits/                                              # Canonical Frozen Spatial & Temporal Split Definitions
│   ├── README.md
│   ├── basins_531.txt                                   # Sorted 531 USGS 8-digit gauge IDs (Newman et al., 2015, 2017)
│   ├── basins_531.json
│   ├── basins_447_kratzert2019b.txt                     # 447-basin process-model intersection subset (Kratzert et al., 2019b)
│   ├── basins_447_kratzert2019b.json
│   ├── splits_and_regimes.json                          # Master protocol (train 1999-10-01..2008-09-30, eval 1989-10-01..1999-09-30)
│   ├── pub_12fold_random_splits.json                    # Canonical 12-fold random PUB split dictionary
│   ├── pub_12fold/                                      # Per-fold plain-text basin lists:
│   │   ├── fold_00_train.txt, fold_00_test.txt          #   486 train / 45 test basins (folds 00..02)
│   │   └── ... (fold_00 .. fold_11)                     #   487 train / 44 test basins (folds 03..11)
│   ├── pur_18fold_huc02_splits.json                     # Canonical 18-fold Leave-One-HUC02-Out PUR split dictionary
│   └── pur_18fold/                                      # Per-region plain-text basin lists:
│       ├── huc_01_train.txt, huc_01_test.txt            #   Hold out HUC-02 region 01 .. 18
│       └── ... (huc_01 .. huc_18)
│
├── observations/                                        # Ground-Truth Observed Streamflow Targets
│   └── newman_eval_targets.zarr/                        # Standardized observed streamflow Zarr store (531 basins x 3652 eval days)
│
├── stage1_author_reproductions/                         # Stage-1 Author-Protocol Confirmation Hydrographs & Metrics
│   ├── manifest_index.json                              # Index of all 11 Stage-1 Zarr stores (12 confirmation runs in Table A1)
│   ├── <model_name>.zarr/                               # Consolidated Zarr v2 store (qsim, qsim_members, qobs, basin, date, huc_02, member)
│   └── <model_name>_results.json                        # Full Stage-1 author reproduction report & per-basin metrics
│
└── hydrographs/                                         # Stage-2 Standardized Model Hydrograph Zarr Stores (61 stores; 531 basins x 3652 eval days)
    ├── manifest_index.json                              # Index of all 61 Stage-2 Zarr stores backing the 104 Zarr-backed rows across Tables 3–14
    ├── gauged__single_camels/                           # Table 3:  Gauged Simulation — Single CAMELS Forcing
    │   └── <model_id>.zarr/
    ├── gauged__multi_camels/                            # Table 4:  Gauged Simulation — Multiple CAMELS Forcings
    │   └── <model_id>.zarr/
    ├── gauged__lagged_streamflow/                       # Table 5:  Gauged Simulation — With Lagged Streamflow
    │   └── <model_id>.zarr/
    ├── gauged__freeform/                                # Table 6:  Gauged Simulation — Freeform Forcings
    │   └── <model_id>.zarr/
    ├── pub__single_camels/                              # Table 7:  12-Fold Random PUB — Single CAMELS Forcing
    │   └── <model_id>.zarr/
    ├── pub__multi_camels/                               # Table 8:  12-Fold Random PUB — Multiple CAMELS Forcings
    │   └── <model_id>.zarr/
    ├── pub__lagged_streamflow/                          # Table 9:  12-Fold Random PUB — With Lagged Streamflow
    │   └── <model_id>.zarr/
    ├── pub__freeform/                                   # Table 10: 12-Fold Random PUB — Freeform Forcings
    │   └── <model_id>.zarr/
    ├── pur__single_camels/                              # Table 11: 18-Fold HUC-02 PUR — Single CAMELS Forcing
    │   └── <model_id>.zarr/
    ├── pur__multi_camels/                               # Table 12: 18-Fold HUC-02 PUR — Multiple CAMELS Forcings
    │   └── <model_id>.zarr/
    ├── pur__lagged_streamflow/                          # Table 13: 18-Fold HUC-02 PUR — With Lagged Streamflow
    │   └── <model_id>.zarr/
    ├── pur__freeform/                                   # Table 14: 18-Fold HUC-02 PUR — Freeform Forcings
    │   └── <model_id>.zarr/
    └── community/                                       # Verified external community hydrograph uploads
        └── <leaderboard_id>/<model_id>.zarr/
```

---

## 3. Canonical Leaderboard IDs (`<leaderboard_id>`)

Every model hydrograph store is placed under its exact leaderboard partition directory in `gs://camels-us-mip/hydrographs/<leaderboard_id>/<model_id>.zarr`:

| Leaderboard ID (`<leaderboard_id>`) | Paper Table | Spatial Regime | Forcing Regime |
| :--- | :--- | :--- | :--- |
| **`gauged__single_camels`** | `table_3` | `gauged` (531 basins) | `single_forcing` (Daymet, Maurer, or NLDAS) |
| **`gauged__multi_camels`** | `table_4` | `gauged` (531 basins) | `multi_forcing` (2 or 3 of Daymet, Maurer, NLDAS) |
| **`gauged__lagged_streamflow`** | `table_5` | `gauged` (531 basins) | `lagged_streamflow` ($Q(t-1)$, AR, or DA) |
| **`gauged__freeform`** | `table_6` | `gauged` (531 basins) | `freeform` (Non-CAMELS forcings: AORC, ERA5-Land, etc.) |
| **`pub__single_camels`** | `table_7` | `pub_12fold` (12-fold random CV) | `single_forcing` |
| **`pub__multi_camels`** | `table_8` | `pub_12fold` (12-fold random CV) | `multi_forcing` |
| **`pub__lagged_streamflow`** | `table_9` | `pub_12fold` (12-fold random CV) | `lagged_streamflow` |
| **`pub__freeform`** | `table_10` | `pub_12fold` (12-fold random CV) | `freeform` |
| **`pur__single_camels`** | `table_11` | `pur_18huc02` (18-fold HUC-02 CV) | `single_forcing` |
| **`pur__multi_camels`** | `table_12` | `pur_18huc02` (18-fold HUC-02 CV) | `multi_forcing` |
| **`pur__lagged_streamflow`** | `table_13` | `pur_18huc02` (18-fold HUC-02 CV) | `lagged_streamflow` |
| **`pur__freeform`** | `table_14` | `pur_18huc02` (18-fold HUC-02 CV) | `freeform` |

---

## 4. Authentication & Access Patterns

### 4.1 CLI Access (`gcloud storage`)
On gLinux cloudtops where Context-Aware Access client certificates block default `gcloud storage` calls, pass:
```bash
CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=false \
CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=$HOME/.config/gcloud/application_default_credentials.json \
gcloud storage ls gs://camels-us-mip/
```

### 4.2 Python / Xarray Access (`xr.open_zarr`)
```python
import xarray as xr

store_uri = "gs://camels-us-mip/hydrographs/gauged__multi_camels/t4_06.zarr"
ds = xr.open_zarr(store_uri, consolidated=True)
# ds["qsim"] has dims ("basin", "date") and shape (531, 3652) in mm/day
```
