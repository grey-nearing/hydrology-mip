# HydrologyMIP (`camels-us-mip`)

**HydrologyMIP** is an open platform for fair, reproducible comparison of hydrological models. Every
model is trained and tested on the same basins, data splits, and dates, and every score comes from
the same evaluation code. The live leaderboards are at <https://hydrologymip.org>.

The first benchmark is the **CAMELS-US Daily Streamflow MIP** (`mips/camels_us_daily/`):

- daily streamflow for 531 US river basins;
- 3 spatial regimes (gauged, ungauged basins, ungauged regions) × 4 input regimes = 12 leaderboards;
- 14 standard metrics.

New benchmarks plug in under `mips/<mip_id>/` and reuse the same models, metrics, and website.

## What is in this repository

| Folder | What it does | Guide |
| :--- | :--- | :--- |
| `hydromip/` | Shared core: benchmark protocol files, data splits and leakage checks, metric suites, Zarr input/output, and CI guardrails | [hydromip/README.md](hydromip/README.md) |
| `mips/` | One self-contained folder per benchmark (protocol, splits, leaderboards, figures, data instructions) | [mips/README.md](mips/README.md) |
| `models/` | The model zoo and the central runner that trains and evaluates every model the same way | [models/README.md](models/README.md) |
| `metrics/` | Scores one model on one leaderboard (`mip-score-model`) | [metrics/README.md](metrics/README.md) |
| `leaderboards/` | Maintainer tools that update the leaderboard tables and draw the figures | [leaderboards/README.md](leaderboards/README.md) |
| `benchmarks/` | End-to-end checks of every stored hydrograph | [benchmarks/README.md](benchmarks/README.md) |
| `frontend/` | The current website code; it will be replaced by a new `web/` folder | [frontend/README.md](frontend/README.md) |
| `skills/` | Step-by-step guides for AI coding agents | [skills/README.md](skills/README.md) |

```text
camels-us-mip/
├── README.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
├── AGENTS.md, CLAUDE.md, GEMINI.md     # rules for AI coding agents (CLAUDE.md and GEMINI.md import AGENTS.md)
├── LICENSE, NOTICE, DATA_LICENSE.md, THIRD_PARTY_NOTICES.md, CITATION.cff, .zenodo.json
├── pyproject.toml                      # package, command-line tools, test and lint settings
├── .github/                            # CI, pull request and issue templates, CODEOWNERS
├── .agents/skills, .claude/skills      # links to skills/ so agent tools find the skills
├── hydromip/  mips/  models/  metrics/  leaderboards/  benchmarks/  frontend/  skills/
└── mips/camels_us_daily/               # protocol.json, adapter.py, splits/, leaderboards/, figures/, data/, era_experiments/
```

## Quickstart

Run every command from the repository root. No Google Cloud account is needed.

### 1. Install (Python 3.11 or newer)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # everything; or ".[models]" for the model zoo, ".[portal]" for the website
```

For a CPU-only machine, install PyTorch from the CPU index first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### 2. Look at a benchmark in Python

```python
from hydromip import discover_mips, get_mip

print(list(discover_mips()))          # ['camels_us_daily']
proto = get_mip("camels_us_daily")
print(proto.name, proto.num_entities, len(proto.leaderboard_ids))
```

### 3. Get the input data

The data is not stored in git. Download CAMELS-US and build the input bundle as described in
[mips/camels_us_daily/data/README.md](mips/camels_us_daily/data/README.md).

### 4. Run a model

```bash
python -m models.run_model --list-models
# zero-parameter lower benchmark, finishes in minutes
python -m models.run_model --model naive_lag1_persistence --spatial-regime gauged \
  --forcings daymet --forcing-regime lagged_streamflow
# a regional LSTM on one forcing product, one seed
python -m models.run_model --model kratzert_lstm --spatial-regime gauged --forcings daymet --seeds 101
```

Results go to `./outputs/<leaderboard_id>/<model_id>.zarr`. See [models/README.md](models/README.md)
for all options.

### 5. Score a model locally

```bash
python -m metrics.build_eval_targets --camels-dir /path/to/basin_dataset_public_v1p2 \
  --splits-dir mips/camels_us_daily/splits \
  --output ./local_bucket/observations/newman_eval_targets.zarr
mip-score-model --model-id <model_id> --leaderboard-id <leaderboard_id> \
  --bucket-uri ./local_bucket --output-json ./outputs/<model_id>_score.json
```

The store must be at `./local_bucket/hydrographs/<leaderboard_id>/<model_id>.zarr`. Move it there,
or pass `--output-zarr` that path when you run the model. See [metrics/README.md](metrics/README.md).

### 6. Check your work

```bash
ruff check .
python -m hydromip.tools.ci_guardrails --mode all
pytest
```

## The CAMELS-US benchmark at a glance

| Item | Value |
| :--- | :--- |
| Basins | 531 CAMELS-US basins (Newman et al., 2015, 2017), listed in `mips/camels_us_daily/splits/basins_531.txt` |
| Training period | `1999-10-01` to `2008-09-30` |
| Evaluation period | `1989-10-01` to `1999-09-30` (3,652 days) |
| Spatial regimes | `gauged`; `pub_12fold` (12 random folds, ungauged basins); `pur_18huc02` (18 folds, one held-out USGS HUC-02 region each) |
| Input regimes | `single_camels`, `multi_camels`, `lagged_streamflow`, `freeform` |
| Metrics | 14 metrics from Gauch et al. (2023): NSE, MSE, RMSE, KGE, logNSE, logKGE, Alpha-NSE, Pearson-r, Beta-KGE, Beta-NSE, FHV, FMS, FLV, Peak-Timing |
| Leaderboard columns | 16 summary columns: the 531-basin median of all 14 metrics, plus the mean NSE and mean KGE |

The machine-readable protocol is [mips/camels_us_daily/protocol.json](mips/camels_us_daily/protocol.json),
and the rules in plain words are in [mips/camels_us_daily/README.md](mips/camels_us_daily/README.md).

## Contributing

We welcome new models, hydrographs from your own code (open or closed source), new benchmarks, and
fixes. Read [CONTRIBUTING.md](CONTRIBUTING.md). In short:

1. Run and score your model locally.
2. Open a pull request (for code) or a hydrograph submission issue (with a Zenodo DOI).
3. Wait for CI to pass. A maintainer then reviews it and adds the official score.

You never need write access to the project's cloud storage.

## For AI coding agents

Agents start with [AGENTS.md](AGENTS.md). It holds the hard rules and the list of skills in
[skills/](skills/README.md). Claude Code reads it through `CLAUDE.md`, and Gemini CLI through
`GEMINI.md`. Codex, GitHub Copilot, and Cursor read `AGENTS.md` directly. Skills are found through
`.agents/skills/` and `.claude/skills/`.

## Data and archives

- Raw inputs: NCAR CAMELS-US v1.2 (DOI [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D)) plus
  the extended Maurer and NLDAS forcings. See
  [mips/camels_us_daily/data/README.md](mips/camels_us_daily/data/README.md).
- HydrologyMIP v1.0 release archive (Zenodo):
  [10.5281/zenodo.24383448](https://doi.org/10.5281/zenodo.24383448).
- The official hydrograph archive bucket (`gs://camels-us-mip/`) is private to the maintainers.
  Leaderboard values are published in `mips/camels_us_daily/leaderboards/` and on
  <https://hydrologymip.org>.

## Citation

Nearing, G. S., Klotz, D., Ebel, P., & Gauch, M. (2026). *HydrologyMIP: A Rainfall-Runoff Model
Intercomparison Project.* Hydrology and Earth System Sciences (submitted). See
[CITATION.cff](CITATION.cff).

## License

- Code: Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- First-party data (splits, leaderboard tables, figures, and the project's hydrograph stores):
  CC-BY-4.0. Archived hydrographs from other authors keep their original licenses. See
  [DATA_LICENSE.md](DATA_LICENSE.md).

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md), and report security problems as described
in [SECURITY.md](SECURITY.md).
