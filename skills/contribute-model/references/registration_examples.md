# Registration examples

A new model needs three entries. The name, family, provenance, runner, code URL, and regimes must be
identical in the registry and the catalog. `models/tests/test_run_model.py` and
`hydromip/tests/test_hydromip_core.py` check that they match.

## 1. `MODEL_REGISTRY` in `models/run_model.py`

```python
"<key>": ModelRegistryEntry(
    key="<key>",
    model_name="<Display Name>",
    model_family="<Model Family>",
    provenance_category="Contribution",          # or "MIP Re-run"
    runner_module="models.<key>.runner",
    code_url="https://github.com/<you>/<repo>",
    allowed_forcing_regimes=("single_camels", "multi_camels"),
    allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
    default_seeds=(101, 102, 103),
    default_kwargs={},
),
```

Which regimes to allow:

- `allowed_forcing_regimes`:
  - `single_camels` and/or `multi_camels` for models driven by CAMELS forcings;
  - `lagged_streamflow` **only** for models that read observed discharge of the test basins;
  - `freeform` for non-CAMELS inputs.
- `allowed_spatial_regimes`: leave out `pub_12fold` and `pur_18huc02` if the model learns per-basin
  parameters or embeddings that cannot transfer to unseen basins.

## 2. `models/catalog.json`

```json
{
  "key": "<key>",
  "model_name": "<Display Name>",
  "model_family": "<Model Family>",
  "provenance_category": "Contribution",
  "runner_module": "models.<key>.runner",
  "code_url": "https://github.com/<you>/<repo>",
  "supported_tasks": ["streamflow_daily_deterministic"],
  "supported_mips": ["camels_us_daily"],
  "supported_forcing_regimes": ["single_camels", "multi_camels"],
  "supported_spatial_regimes": ["gauged", "pub_12fold", "pur_18huc02"]
}
```

## 3. `EXPECTED_MODEL_DIRS` in `models/tests/test_run_model.py`

Add `"<key>"` to the tuple and update the expected count in the same file.

## Check

```bash
python -m models.run_model --list-models
pytest models/tests hydromip/tests
```
