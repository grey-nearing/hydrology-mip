# Canonical CAMELS-US Input Bundle (`data/`)

`models/run_model.py` and every `models/<model>/runner.py` read CAMELS-US forcings, static attributes,
and discharge exclusively through `models.camels_data.load_camels_slice`. By default the loader looks
for the canonical compiled bundle in this directory:

```
data/camels_671_1980_2014.npz        # gitignored (~340 MB); schema "camels-us-mip-bundle-v1"
```

| Key | Shape / dtype | Content |
| :--- | :--- | :--- |
| `basins` | `(671,) <U8` | Sorted 8-digit USGS gauge IDs (all CAMELS-US basins; the 531 Newman basins are a subset). |
| `dates` | `(12784,) <U10` | Contiguous daily dates `1980-01-01` .. `2014-12-31`. |
| `forcing_daymet` / `forcing_maurer` / `forcing_nldas` | `(671, 12784, 5) float32` | `prcp(mm/day), srad(W/m2), tmax(C), tmin(C), vp(Pa)` (`forcing_variable_names`). Maurer-extended ends `2008-12-31` (NaN afterwards; requesting it raises). |
| `aux_daymet` | `(671, 12784, 2) float32` | `dayl(s), swe(mm)` (optional). |
| `qobs` | `(671, 12784) float32` | USGS discharge in mm/day = `cfs * 0.028316846592 * 86400 * 1000 / area_m2`, with `area_m2` from the Daymet forcing-file header (same convention as `metrics/build_eval_targets.py` and the canonical `observations/newman_eval_targets.zarr`). `-999` / negative -> NaN. |
| `static_attributes` | `(671, 27) float32` | The 27 Kratzert et al. (2019) attributes (`static_attribute_names`). |
| `gauge_lat`, `gauge_lon`, `elev_mean`, `area_km2`, `huc_02` | `(671,)` | From the official `camels_attributes_v2.0` tables (`area_km2` = Daymet header area). |
| `provenance` | JSON string | Source, builder, build time, and raw cross-validation summary. |

## Rebuilding from the raw NCAR download

```bash
# basin_timeseries_v1p2_metForcing_obsFlow/ (basin_mean_forcing/, usgs_streamflow/) + camels_attributes_v2.0/
python -m models.camels_data build --raw-dir /path/to/basin_dataset_public_v1p2 \
    --out data/camels_671_1980_2014.npz
python -m models.camels_data validate --bundle data/camels_671_1980_2014.npz --raw-dir /path/to/basin_dataset_public_v1p2
```

`load_camels_slice` also reads a raw NCAR CAMELS-US v1.2 directory or a per-basin NetCDF layout directly;
see the module docstring of `models/camels_data.py`.

## Anti-leakage guarantees enforced by the loader

* Discharge inside any requested spin-up window (`qobs[:, :spinup_days]`) is returned as NaN, so it can
  never enter a loss, a target scaler, or a per-basin NSE weight. Under the Newman protocol the training
  spin-up (`1998-10-0x`..`1999-09-30`) lies inside the `1989-10-01`..`1999-09-30` evaluation decade.
* Non-finite forcings or attributes in the requested window raise; nothing is imputed and no coordinate
  defaults exist.
