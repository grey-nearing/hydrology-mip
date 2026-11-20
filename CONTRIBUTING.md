# Contributing to HydrologyMIP

Thank you for helping build a fair, open benchmark for hydrological models. This page explains how
to contribute. It takes about five minutes to read. Run every command from the repository root.

> **Using an AI coding agent?** Point it to [AGENTS.md](AGENTS.md). It loads the project rules and the
> step-by-step skills in [skills/](skills/README.md). You stay responsible for everything it writes.

## What you can contribute

| I want to… | Path | Human guide | Agent skill |
| :--- | :--- | :--- | :--- |
| Add a model that runs inside this repository | A | [models/README.md](models/README.md) | `skills/contribute-model/SKILL.md` |
| Submit predictions from my own code (open or closed source) | B | [Submit hydrographs](#b-submit-hydrographs) below | `skills/contribute-hydrographs/SKILL.md` |
| Propose a new benchmark (a new MIP) | C | [mips/README.md](mips/README.md) | `skills/contribute-mip/SKILL.md` |
| Fix a bug, improve the docs, or add tests | D | this page | `skills/pr-review/SKILL.md` |

## The rules of a fair benchmark

Please read these before you start. The full protocol is in
[mips/camels_us_daily/README.md](mips/camels_us_daily/README.md).

1. **Use the canonical basins, folds, and dates.** Never change them. The model trains on
   `1999-10-01`..`2008-09-30` and is evaluated on `1989-10-01`..`1999-09-30`.
2. **Tune only inside the training period.** Early stopping, hyperparameters, and model choice must
   never look at evaluation-period streamflow. In the PUB and PUR regimes, never fit anything on the
   held-out basins of a fold.
3. **Predict everything.** Give all 531 basins on all 3,652 evaluation days, in mm/day, with no
   missing values.
4. **One paper per entry.** Each entry cites exactly one canonical paper.
5. **Open science.** Share your code (or a public description for closed-source models) and archive
   your hydrographs openly under CC-BY-4.0.

## Set up

```bash
git clone https://github.com/<your-fork>/camels-us-mip.git && cd camels-us-mip
python -m venv .venv && source .venv/bin/activate        # Python 3.11 or newer
pip install -e ".[dev]"                                   # CPU only: first `pip install torch --index-url https://download.pytorch.org/whl/cpu`
ruff check .
python -m hydromip.tools.ci_guardrails --mode all
pytest
```

## Get the data

The input data is not stored in git. Follow
[mips/camels_us_daily/data/README.md](mips/camels_us_daily/data/README.md) to download:

- the NCAR CAMELS-US v1.2 dataset (<https://gdex.ucar.edu/dataset/camels.html>, DOI
  [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D));
- the extended Maurer forcings (HydroShare
  [10.4211/hs.17c896843cf940339c3c3496d0c1c077](https://doi.org/10.4211/hs.17c896843cf940339c3c3496d0c1c077));
- the extended NLDAS forcings (HydroShare
  [10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c](https://doi.org/10.4211/hs.0a68bfd7ddf642a8be9041d60f40868c)).

Then build the input bundle. The HydrologyMIP v1.0 release archive is on Zenodo
([10.5281/zenodo.24383448](https://doi.org/10.5281/zenodo.24383448)).

## Score your model locally (no Google Cloud account needed)

The official archive bucket is private to the maintainers. You never need it. The same scoring code
runs on a local folder:

```bash
# 1. Once: rebuild the observation targets from the raw NCAR download
python -m metrics.build_eval_targets --camels-dir /path/to/basin_dataset_public_v1p2 \
  --splits-dir mips/camels_us_daily/splits \
  --output ./local_bucket/observations/newman_eval_targets.zarr

# 2. Run a model into the local layout (path A); default output is ./outputs/<leaderboard_id>/<model_id>.zarr
python -m models.run_model --model <key> --spatial-regime gauged --forcings daymet \
  --output-zarr ./local_bucket/hydrographs/gauged__single_camels/<model_id>.zarr

# 3. Score it with all 14 metrics
mip-score-model --model-id <model_id> --leaderboard-id gauged__single_camels \
  --bucket-uri ./local_bucket --output-json ./outputs/<model_id>_score.json
```

You can also upload a zipped Zarr store or a keyed `.npz` to the **Score** page on
<https://hydrologymip.org>. It returns the metrics and does not store anything.

## Submit

### A. Model code (pull request)

1. Follow [models/README.md](models/README.md#adding-a-new-model): add `models/<key>/`, register it
   in `MODEL_REGISTRY` and `models/catalog.json`, and add tests. For a published model ("MIP
   Re-run"), also add a Stage-1 script that reproduces the paper.
2. Run it and score it locally (above).
3. Open a pull request from your fork. Fill in the template and attach your score JSON.

### B. Submit hydrographs

1. Package your predictions with `models/zarr_writer.py` (see
   `skills/contribute-hydrographs/SKILL.md` for a tested example). Score them locally.
2. Zip the store and deposit it on Zenodo under CC-BY-4.0.
3. Open a **Hydrograph Archive Submission** issue with the DOI. The issue form supports open- and
   closed-source models.

### C. New MIP

Open a **New MIP Proposal** issue first. After a maintainer agrees, follow
[mips/README.md](mips/README.md).

### D. Bug fixes, docs, and tests

Open a **Bug / Protocol / Leaderboard Report** issue or a pull request directly.

**After you submit:** the maintainers validate your store, compute the official score from the
archive, and add your leaderboard row. Please **do not** edit the leaderboard JSON files,
`table_2_registry.json`, `contributions_index.json`, or the figures by hand. The maintainers'
tools regenerate them.

## Pull request checklist

- [ ] `ruff check .`, `python -m hydromip.tools.ci_guardrails --mode all`, and `pytest` pass.
- [ ] The benchmark rules above are met. Model code never reads split files or hard-codes dates.
- [ ] No data, Zarr stores, checkpoints, secrets, or files larger than 5 MiB are committed.
- [ ] Docs are updated where behavior changed.
- [ ] Every item in the PR template is filled in.

CI must pass (`CI Gate`) before review. `@grey-nearing` reviews and approves every pull request.

## AI-assisted contributions

AI tools are welcome. You are responsible for every line you submit. Say in the PR description which
tools you used and for what. Agents must follow [AGENTS.md](AGENTS.md). They must never push to
`main`, merge, tag, or touch the archive bucket.

## License of your contribution (inbound = outbound)

- Code you contribute is licensed under the [Apache License 2.0](LICENSE), like the rest of the
  code.
- Hydrographs, tables, and other data you contribute are licensed under CC-BY-4.0, as described in
  [DATA_LICENSE.md](DATA_LICENSE.md).
- By contributing, you confirm that you have the right to submit the work under these licenses, as
  stated in the Developer Certificate of Origin 1.1 (<https://developercertificate.org>). Please sign
  off your commits with `git commit -s`.
- Only include third-party code under a compatible permissive license (MIT, BSD, Apache-2.0), and
  credit it in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Community and security

- Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
- Ask questions in a GitHub issue.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md). Do not open a
  public issue for them.

## For maintainers

Ingesting submissions, managing the archive bucket, audits, and releases are described in
`skills/maintainer-ingest-submission/SKILL.md`, `skills/maintainer-gcs-bucket/SKILL.md`, and
`skills/maintainer-audit/SKILL.md`.
