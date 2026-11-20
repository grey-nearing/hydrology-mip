# CAMELS-US Daily Streamflow MIP (`mips/camels_us_daily/`)

Canonical benchmark package for the **531-basin CAMELS-US Daily Streamflow Model Intercomparison Project** (Newman et al., 2015, 2017; Kratzert et al., 2019, 2021; Nearing et al., 2026).

## Contents

- [`protocol.json`](./protocol.json): Machine-readable `hydromip-protocol-v1` specification (temporal windows, 3 spatial × 4 forcing = 12 leaderboards, Gauch et al. 2023 metric suite, GCS bucket URIs).
- [`adapter.py`](./adapter.py): Verified split & protocol adapter backed by `hydromip.splits.validate_cv_folds`.
- [`splits/`](./splits/README.md): Canonical 531-basin cohort, 12-fold random PUB (`pub_12fold`), and 18-fold Leave-One-HUC02-Out PUR (`pur_18huc02`) splits.
- [`leaderboards/`](./leaderboards/README.md): All 12 leaderboard JSON tables, `tables_3_to_14.json`, `table_2_registry.json`, `contributions_index.json`, and Stage-1 author-protocol confirmation runs.
- [`figures/`](./figures/README.md): Publication-ready manuscript & portal figures generated directly from `leaderboards/*.json`.
- [`era_experiments/`](./era_experiments/README.md): 14 canonical 256-iteration ERA prompt and winner notebooks across 7 tracks.
- [`data/`](./data/README.md): Raw NCAR CAMELS-US v1.2 compilation instructions and local gitignored bundle location.
