# Per-model README template

Use this layout for `models/<key>/README.md`. Keep it short and plain. Link to the leaderboards
instead of copying numbers, because copied numbers go out of date.

```markdown
# <Model Name> (`models/<key>/`)

One sentence: what the model is and who published it.

## Reference and links
- Paper: <full citation>, <DOI link>
- Upstream code: <URL> (license: <SPDX id>)
- Hydrograph archive: <DOI or "none">

## Category and provenance
- Leaderboard category: `Contribution` | `MIP Re-run`
- Code provenance: `Our Code` (re-implementation) | `Author Archive` | `Published Paper`
- Architecture class (Table 2): `Machine learning` | `Hybrid (ML + Physics)` | `Conceptual` | `Process-based` | `Statistical`

## Architecture
3–8 bullet points: inputs, core layers or equations, outputs, loss, training schedule.

## Run on the MIP protocol (Stage 2)
    python -m models.run_model --model <key> --spatial-regime gauged --forcings daymet
Allowed forcing regimes: ... Allowed spatial regimes: ...

## Stage 1: reproduce the paper (MIP Re-run only)
- Author protocol: <split, basins, forcing, hyperparameters>
- Verify: `python -m models.<key>.reproduce_author_metrics --mode verify`

## Results
See the rows for `<key>` in `mips/camels_us_daily/leaderboards/` (or on https://hydrologymip.org).

## Known differences from upstream
- ...
```
