---
name: contribute-hydrographs
description: >-
  Package daily streamflow predictions from a model that runs outside this repository (open- or
  closed-source) into the canonical HydrologyMIP (531, 3652) Zarr store with models/zarr_writer.py,
  validate and self-score it locally without any Google Cloud account, deposit it on Zenodo, and
  submit it through the GitHub hydrograph issue form. Use when only hydrographs, not runnable code,
  are being contributed.
license: Apache-2.0
metadata:
  audience: contributor
---

# Contribute hydrographs from external code

Read `skills/protocol-guardrails/SKILL.md` first; the same protocol applies to code that runs
elsewhere. All paths are relative to the repository root. You never need write access to
`gs://camels-us-mip/`.

## 1. What a valid submission is

- One store per leaderboard, for example `gauged__single_camels` (see
  `skills/leaderboard-norms/SKILL.md` for routing).
- Predictions for **all 531 basins** in the order of `mips/camels_us_daily/splits/basins_531.txt`,
  and **all 3,652 days** from `1989-10-01` to `1999-09-30`. Daily specific discharge in mm/day,
  `float32`, with no NaN.
- The model was trained or calibrated only on `1999-10-01`..`2008-09-30`. For PUB and PUR, it used
  the canonical folds in `mips/camels_us_daily/splits/` and never saw a fold's test basins.
- Ensembles: also provide every member as `qsim_members` with shape `(N, 531, 3652)`. Then `qsim`
  must be the member mean.
- If your model produced m³/s or ft³/s, convert to mm/day with the CAMELS basin area before you
  package it.

## 2. Build and write the store

Always use `models/zarr_writer.py`. Never assemble Zarr attributes by hand. This example was checked
against the current code:

```python
import numpy as np

from mips.camels_us_daily.adapter import load_basin_to_huc02, load_canonical_basins
from models.zarr_writer import build_hydrograph_dataset, write_hydrograph_zarr

basins = load_canonical_basins()          # 531 sorted 8-digit USGS IDs
basin_to_huc02 = load_basin_to_huc02()    # {"01022500": "01", ...}
qsim = np.load("my_predictions.npy", allow_pickle=False).astype(np.float32)  # (531, 3652), rows = basins

ds = build_hydrograph_dataset(
    qsim,
    basins,
    basin_to_huc02,
    model_id="mymodel_gauged_daymet",     # becomes the store name and entry_id
    model_name="My Model",
    model_family="My Model Family",
    family_id="mymodel",
    provenance_category="Contribution",
    spatial_regime="gauged",              # "gauged" | "pub_12fold" | "pur_18huc02"
    forcing_regime="single_camels",       # "single_camels" | "multi_camels" | "lagged_streamflow" | "freeform"
    forcings_used=["daymet"],
    code_url="https://github.com/<you>/<repo>",
    # qsim_members=members,               # (N, 531, 3652) for an ensemble; qsim must equal the mean
)
leaderboard_id = ds.attrs["leaderboard_id"]   # derived from the two regimes, e.g. gauged__single_camels
write_hydrograph_zarr(
    ds,
    f"./local_bucket/hydrographs/{leaderboard_id}/mymodel_gauged_daymet.zarr",
    expected_basins=basins,
)
```

- The writer validates shape, coordinates (`basin`, `date`, `huc_02`), attributes, finiteness, and
  the ensemble mean, and raises on any problem.
- `archive_url` defaults to the official bucket location used later for registration. That is
  expected.
- For a closed-source model, set `code_url` to a public page that describes the model, such as the
  paper or a technical report. The issue form records that the code is closed.

## 3. Self-score locally

```bash
# once: rebuild the observation targets from the raw NCAR CAMELS-US v1.2 download
python -m metrics.build_eval_targets --camels-dir /path/to/basin_dataset_public_v1p2 \
  --splits-dir mips/camels_us_daily/splits \
  --output ./local_bucket/observations/newman_eval_targets.zarr
mip-score-model --model-id mymodel_gauged_daymet --leaderboard-id gauged__single_camels \
  --bucket-uri ./local_bucket --output-json ./outputs/mymodel_gauged_daymet_score.json
```

Check that `summary.num_evaluated_basins` is 531 and `summary.pred_nan_when_obs_valid` is 0.

Without the raw data, use the **Score** page on <https://hydrologymip.org>. It accepts a zipped Zarr
store (`.zarr.zip`) or a keyed `.npz` with `qsim`, `basin`, and `date`. It only scores; it stores
nothing.

## 4. Archive and submit

1. Zip the store: `cd local_bucket/hydrographs/<leaderboard_id> && zip -r mymodel_gauged_daymet.zarr.zip mymodel_gauged_daymet.zarr`.
2. Deposit the zip on Zenodo under **CC-BY-4.0** and note the DOI.
3. Open the **Community Model Submission** issue form
   (`.github/ISSUE_TEMPLATE/model_submission.yml`) — or submit via Contribute Step 4 on `https://hydrologymip.org/#contribute`. Give:
   - the DOI, model name, citation, architectural category, and target leaderboard(s);
   - the forcings and the number of ensemble members;
   - the score JSON;
   - the protocol attestations.
4. A maintainer validates the store, copies it into the official archive, computes the official
   score, and adds the leaderboard row.

## Do not

- write to `gs://camels-us-mip/` or ask for access to it;
- edit the leaderboard JSONs, `table_2_registry.json`, or the figures;
- drop or fill basins or days to get past the writer's checks.

## Stop and ask a human if

- your predictions do not cover every basin and day, or their units are unclear;
- the model used any discharge from `1989-10-01`..`1999-09-30` during training or selection;
- the model does not fit any of the 12 leaderboards.
