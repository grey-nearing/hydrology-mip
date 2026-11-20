# CAMELS-US Input Data (`mips/camels_us_daily/data/`)

This folder holds the input data for model runs. The data files are not stored in git. You
download the raw CAMELS-US files once and build one compiled file, the **bundle**. All model
runners read the data through `models.camels_data.load_camels_slice`.

## Overview

| File | What it is | In git? |
| :--- | :--- | :--- |
| `README.md` | This guide. | Yes |
| `camels_671_1980_2014.npz` | The compiled bundle (about 340 MB). You build it with `python -m models.camels_data build`. | No (git ignores `*.npz` files) |

The raw files serve two purposes:

| Purpose | Raw folders needed | Command |
| :--- | :--- | :--- |
| Run models | `basin_mean_forcing/{daymet,maurer_extended,nldas_extended}/`, `usgs_streamflow/`, and the attribute tables | `python -m models.camels_data build` (this page) |
| Score models on your own computer | `usgs_streamflow/` and `basin_mean_forcing/daymet/` | `python -m metrics.build_eval_targets` (see [`metrics/README.md`](../../../metrics/README.md)) |

## Prerequisites

- Python 3.11 or newer, with the repository installed: `pip install -e .` from the repository root.
- Free disk space for the raw files and the bundle.
- Run every command from the repository root.

## Step 1: Download the raw data

| Data | Where | DOI |
| :--- | :--- | :--- |
| CAMELS-US v1.2 time series: forcings and USGS streamflow (`basin_timeseries_v1p2_metForcing_obsFlow.zip`) | NCAR GDEX, <https://gdex.ucar.edu/dataset/camels.html> | [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D) |
| CAMELS-US attributes v2.0 (`camels_attributes_v2.0.zip`) | NCAR GDEX, same page | [10.5065/D6G73C3Q](https://doi.org/10.5065/D6G73C3Q) |
| Extended Maurer forcings | HydroShare | [10.4211/hs.17c896843cf940339c3c3496d0c1c077](https://doi.org/10.4211/hs.17c896843cf940339c3c3496d0c1c077) |
| Extended NLDAS forcings | HydroShare | [10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c](https://doi.org/10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c) |

1. Unzip the time series archive. It creates the folder `basin_dataset_public_v1p2/`.
2. Unzip the attributes archive next to it. You need the folder that holds the seven
   `camels_*.txt` files.
3. Put the extended Maurer and NLDAS folders inside `basin_dataset_public_v1p2/basin_mean_forcing/`
   with the names `maurer_extended/` and `nldas_extended/`.

You can place the data anywhere outside the repository. When you are done, the layout must look
like this (`camels_us/` is any folder name you like):

```text
camels_us/
├── basin_dataset_public_v1p2/
│   ├── basin_mean_forcing/
│   │   ├── daymet/<HUC02>/<basin>_lump_cida_forcing_leap.txt
│   │   ├── maurer_extended/<HUC02>/<basin>_lump_maurer_forcing_leap.txt
│   │   └── nldas_extended/<HUC02>/<basin>_lump_nldas_forcing_leap.txt
│   └── usgs_streamflow/<HUC02>/<basin>_streamflow_qc.txt
└── camels_attributes_v2.0/
    ├── camels_clim.txt
    ├── camels_geol.txt
    ├── camels_hydro.txt
    ├── camels_name.txt
    ├── camels_soil.txt
    ├── camels_topo.txt
    └── camels_vege.txt
```

`<HUC02>` is a two-digit region folder such as `01`, and `<basin>` is an 8-digit USGS gauge ID.
The build reads only `daymet/`, `maurer_extended/`, and `nldas_extended/`; other forcing folders
in the download are not used.

The build looks for the attribute tables in these places, in order:
`basin_dataset_public_v1p2/`, `basin_dataset_public_v1p2/camels_attributes_v2.0/`,
`camels_us/camels_attributes_v2.0/` (next to `basin_dataset_public_v1p2/`), and
`basin_dataset_public_v1p2/camels_attributes_v2.0/camels_attributes_v2.0/`. Use
`--attributes-dir` to point to another folder.

## Step 2: Build and check the bundle

```bash
python -m models.camels_data build \
  --raw-dir /path/to/camels_us/basin_dataset_public_v1p2 \
  --out mips/camels_us_daily/data/camels_671_1980_2014.npz

python -m models.camels_data validate \
  --bundle mips/camels_us_daily/data/camels_671_1980_2014.npz \
  --raw-dir /path/to/camels_us/basin_dataset_public_v1p2
```

`build` compiles all basins in the attribute tables (671) for 1980-01-01 to 2014-12-31.
`validate` checks the bundle's schema and prints its size, dates, and provenance. With
`--raw-dir`, it also checks the bundle against the raw files: forcings must match exactly, and
streamflow must match within 0.001 mm/day. Basins without raw files are skipped and counted.

`mip-run-model` reads `mips/camels_us_daily/data/` by default, so most users never pass
`--data-dir` (see [`models/README.md`](../../../models/README.md)).

## Argument reference

### `python -m models.camels_data build`

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--raw-dir PATH` | required | The raw `basin_dataset_public_v1p2` folder. It must contain `basin_mean_forcing/` and `usgs_streamflow/`. |
| `--out PATH` | `mips/camels_us_daily/data/camels_671_1980_2014.npz` | The bundle file to write. |
| `--attributes-dir PATH` | searched (see above) | The folder with the seven `camels_*.txt` attribute tables. |
| `--start DATE` | `1980-01-01` | The first day of the bundle. |
| `--end DATE` | `2014-12-31` | The last day of the bundle. |
| `--basins-file PATH` | all basins in the attribute tables | A text file with one gauge ID per line. |
| `--no-compress` | off | Write an uncompressed `.npz` file. |

### `python -m models.camels_data validate`

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--bundle PATH` | required | The bundle file to check. |
| `--raw-dir PATH` | none | A raw `basin_dataset_public_v1p2` folder to compare against. |

### `python -m models.camels_data convert-legacy`

Converts an older per-product `.npz` file to the bundle format. Most users never need it.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--legacy-npz PATH` | required | The old file. It must have `basins`, `dates`, `qobs`, `area_km2`, `dyn_daymet`, `dyn_maurer`, and `dyn_nldas`. |
| `--attributes-dir PATH` | required | The folder with the seven `camels_*.txt` attribute tables. |
| `--out PATH` | `mips/camels_us_daily/data/camels_671_1980_2014.npz` | The bundle file to write. |
| `--raw-dir PATH` | none | A raw folder to check the result against. |
| `--no-compress` | off | Write an uncompressed `.npz` file. |

## Where the loader looks for data

`--data-dir` of `mip-run-model` (default `mips/<mip>/data`) can point to any of these:

1. a bundle file (`.npz`);
2. a folder with `camels_671_1980_2014.npz`, or with exactly one other `.npz` file;
3. a raw `basin_dataset_public_v1p2` folder (it has `basin_mean_forcing/` and `usgs_streamflow/`);
4. a per-basin NetCDF folder with `time_series/<basin>.nc` and `attributes/camels_*.txt`.

The loader checks them in this order. A folder with more than one `.npz` file and no
`camels_671_1980_2014.npz` is an error.

## Bundle contents

The bundle (schema `camels-us-mip-bundle-v1`) is a NumPy `.npz` file with these arrays:

| Key | Shape and type | Content |
| :--- | :--- | :--- |
| `basins` | `(671,)` text | Sorted 8-digit USGS gauge IDs of all CAMELS-US basins. The 531 benchmark basins are a subset. |
| `dates` | `(12784,)` text | Every day from 1980-01-01 to 2014-12-31. |
| `forcing_daymet`, `forcing_maurer`, `forcing_nldas` | `(671, 12784, 5)` float32 | `prcp(mm/day)`, `srad(W/m2)`, `tmax(C)`, `tmin(C)`, `vp(Pa)` (names in `forcing_variable_names`). The extended Maurer data end on 2008-12-31; later values are missing, and a request for them fails. |
| `aux_daymet` | `(671, 12784, 2)` float32 | `dayl(s)` and `swe(mm)` from the Daymet files (names in `aux_variable_names`). |
| `qobs` | `(671, 12784)` float32 | USGS streamflow in mm/day: `cfs * 0.028316846592 * 86400 * 1000 / area_m2`, where `area_m2` is the basin area from the Daymet file header. Negative values and `-999` become missing (NaN). |
| `static_attributes` | `(671, 27)` float32 | The 27 basin attributes of Kratzert et al. (2019) (names in `static_attribute_names`). |
| `gauge_lat`, `gauge_lon`, `elev_mean`, `area_km2`, `huc_02` | `(671,)` | Gauge position, mean elevation, basin area (from the Daymet header), and HUC-02 region. |
| `schema_version`, `area_source`, `provenance` | text | The schema name, the source of the basin area, and a JSON record of how the bundle was built. |

The scoring targets (`newman_eval_targets.zarr`) use the same streamflow conversion, so training
targets and scoring targets match.

## Anti-leakage rules in the loader

`load_camels_slice` enforces these rules for every model:

- **Spin-up streamflow is hidden.** Streamflow inside the spin-up window (`qobs[:, :spinup_days]`)
  is returned as missing (NaN). It can never enter a loss, a target scaler, or a per-basin
  weight. This matters because the training spin-up year (1998-10-01 to 1999-09-30) lies inside
  the evaluation period (1989-10-01 to 1999-09-30).
- **Missing inputs stop the run.** A missing or non-finite forcing value or attribute in the
  requested window raises an error. Nothing is filled in, and there are no default coordinates or
  areas.
- **Bad streamflow becomes missing.** Negative streamflow and `-999` values become NaN.
- **Basins and dates come from the runner.** Model code never reads split files and never writes
  dates into code. `models/run_model.py` passes the basins and the periods to each model.
