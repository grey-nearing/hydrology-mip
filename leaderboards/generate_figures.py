"""Generate all CAMELS-US MIP paper figures directly from the leaderboard JSON files.

Reads the live leaderboard JSON files in `leaderboards/` (`<leaderboard_id>.json` and
`tables_3_to_14.json`) and writes publication-ready PNG and PDF figures into `figures/`:
- `figures/figure_1_water_year_coverage.png` & `.pdf`
- `figures/figure_2_leaderboard_progress_all_regimes.png` & `.pdf`
- `figures/figure_2a_leaderboard_progress_gauged.png`
- `figures/figure_2b_leaderboard_progress_pub.png`
- `figures/figure_2c_leaderboard_progress_pur.png`
- `figures/figures_manifest.json`

Invariants enforced by this module (and checked by `leaderboards/tests/test_generate_figures.py`):
- Figure 1 is drawn exclusively from `protocol["water_year_completeness"]` in the leaderboard
  JSON protocol block. There is no hard-coded fallback series; a missing or malformed block
  raises `ValueError`. `compute_water_year_completeness()` recomputes that block from the
  `newman_eval_targets.zarr` arrays (`qobs_all`, `date_all`).
- Every leaderboard row with a finite, non-sentinel `primary_ranking_nse` is drawn in Figure 2.
  Axis limits are derived from the plotted data, so no point is ever cropped away. Rows that
  cannot be plotted (null / non-numeric / non-finite / sentinel NSE) are listed, with a reason,
  under `excluded_rows` in `figures_manifest.json`.
- "MIP Re-run" results are placed at the publication year of the original paper
  the MIP re-run builds from (Series-1 reproduction paper year), whereas "Contribution"
  results (including new models contributed by this study)
  are dated no earlier than `MIP_STUDY_YEAR`. Both "MIP Re-run" and "Contribution" results are
  drawn with hollow markers so MIP-produced runs are visually distinguished from author-reported
  literature/archive numbers.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_LEADERBOARDS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIGURES_DIR: Path = REPO_ROOT / "figures"

#: Year of the CAMELS-US MIP study itself ("this study"). "Contribution" entries
#: without a prior Series-1 paper are dated at this year (or their `submitted_at` year).
MIP_STUDY_YEAR: int = 2026
#: Normalized source categories (see `normalize_source_category`) whose results were produced
#: by this MIP pipeline rather than taken from the cited paper (drawn with hollow markers).
MIP_RERUN_CATEGORIES: frozenset[str] = frozenset({"MIP Re-run", "Contribution"})
#: Normalized source categories for new MIP/community contributions (dated at `MIP_STUDY_YEAR`).
MIP_CONTRIBUTION_CATEGORIES: frozenset[str] = frozenset({"Contribution"})
#: Normalized source categories whose results are taken verbatim from the literature / archives.
LITERATURE_CATEGORIES: frozenset[str] = frozenset({"Literature", "Archive", "Archive*"})
#: Canonical publication year of the original paper each Series-1 (`MIP Re-run`) model family
#: builds from (verified against `models/<family>/reproduce_author_metrics.py` and Table 2).
SERIES1_FAMILY_PUBLICATION_YEARS: dict[str, int] = {
    "newman_sacsma": 2015,
    "kratzert_ealstm": 2019,
    "kratzert_lstm": 2021,
    "hoedt_mclstm": 2021,
    "gauch_xgboost": 2021,
    "gauch2021_xgboost": 2021,
    "feng_dhbv10": 2022,
    "das_tide": 2023,
    "das2023_tide": 2023,
    "rasiya_koya_tft": 2023,
    "nh_tft_rerun": 2023,
    "hy2dl_hbv": 2024,
    "song_dhbv11p": 2024,
    "wang_s4dft": 2025,
    "wang2025_s4d": 2025,
    "hydrodiffusion": 2025,
    "poudel_mlp_hbv": 2025,
    "poudel2026_mlp_hbv": 2025,
}
#: Accepted range for a publication year parsed from `model_name` / `citation` / `publication_year`.
PUBLICATION_YEAR_MIN: int = 1990
PUBLICATION_YEAR_MAX: int = 2035
_PUBLICATION_YEAR_PATTERN = re.compile(r"(?<!\d)(199[0-9]|20[0-2][0-9]|203[0-5])(?!\d)")
#: NSE values at or below this bound are legacy missing-value sentinels (e.g. -999.0), not
#: reported NSE values (mirrors `leaderboards.update_leaderboard._SENTINEL_NSE_BOUND`). Such rows
#: are excluded explicitly and reported in the manifest.
NSE_SENTINEL_FLOOR: float = -900.0
#: Earliest year shown on the Figure 2 x-axis (the axis always expands to fit the data).
FIGURE_2_MIN_YEAR: int = 2015
#: Padding (in NSE units) added around the plotted data when deriving Figure 2 y-limits.
FIGURE_2_YLIM_PAD: float = 0.02
#: Minimum y-extent per spatial regime; the axis always expands beyond these to fit the data.
PANEL_YLIM_HINTS: dict[str, tuple[float, float]] = {
    "gauged": (0.42, 0.91),
    "pub": (0.40, 0.80),
    "pur": (0.30, 0.76),
}
#: Newman et al. (2015/2017) benchmark water-year windows (protocol constants, not data).
NEWMAN_EVAL_WATER_YEARS: tuple[int, int] = (1990, 1999)
NEWMAN_TRAIN_WATER_YEARS: tuple[int, int] = (2000, 2008)
WATER_YEAR_COMPLETENESS_SERIES_KEYS: tuple[str, ...] = (
    "basins_with_100pct_complete",
    "basins_with_any_valid_data",
    "mean_daily_coverage_fraction",
)
REQUIRED_WATER_YEAR_COMPLETENESS_KEYS: tuple[str, ...] = (
    "n_basins",
    "water_years",
    *WATER_YEAR_COMPLETENESS_SERIES_KEYS,
)

MODEL_CATEGORY_COLORS: dict[str, str] = {
    "Machine learning": "#005AB5",
    "Hybrid (ML + Physics)": "#8E24AA",
    "Conceptual": "#D95F02",
    "Process-based": "#1B9E77",
}

FORCING_REGIME_MAP: dict[str, str] = {
    "single_camels": "Single CAMELS Forcing",
    "multi_camels": "Multiple CAMELS Forcings",
    "lagged_streamflow": "With Lagged Streamflow",
    "freeform": "Freeform Forcings",
}

REGIME_STYLES: dict[str, dict[str, Any]] = {
    "Single CAMELS Forcing": {
        "linestyle": "-",
        "marker": "o",
        "zorder_base": 4,
    },
    "Multiple CAMELS Forcings": {
        "linestyle": "--",
        "marker": "D",
        "zorder_base": 5,
    },
    "With Lagged Streamflow": {
        "linestyle": ":",
        "marker": "^",
        "zorder_base": 6,
    },
    "Freeform Forcings": {
        "linestyle": "-.",
        "marker": "s",
        "zorder_base": 3,
    },
}

# Initial offset hints (dx_pts, dy_pts) for known (panel_idx, regime, year) anchors.
# These are cosmetic only: any (panel, regime, year) key that is absent falls back to a
# data-derived default offset, so a label is always placed for every frontier leader.
INITIAL_LABEL_OFFSETS: dict[tuple[int, str, int], tuple[float, float]] = {
    (0, "Single CAMELS Forcing", 2015): (10.0, -24.0),
    (0, "Single CAMELS Forcing", 2017): (10.0, -26.0),
    (0, "Single CAMELS Forcing", 2018): (10.0, -24.0),
    (0, "Single CAMELS Forcing", 2019): (-68.0, 14.0),
    (0, "Single CAMELS Forcing", 2021): (10.0, -26.0),
    (0, "Single CAMELS Forcing", 2025): (10.0, -24.0),
    (0, "Multiple CAMELS Forcings", 2020): (10.0, -26.0),
    (0, "Multiple CAMELS Forcings", 2021): (-115.0, 12.0),
    (0, "Multiple CAMELS Forcings", 2025): (-125.0, -22.0),
    (0, "With Lagged Streamflow", 2020): (-95.0, 12.0),
    (0, "With Lagged Streamflow", 2022): (12.0, 10.0),
    (0, "Freeform Forcings", 2021): (-105.0, -26.0),
    (0, "Freeform Forcings", 2023): (10.0, -24.0),
    (1, "Single CAMELS Forcing", 2017): (10.0, -24.0),
    (1, "Single CAMELS Forcing", 2019): (-68.0, -26.0),
    (1, "Single CAMELS Forcing", 2024): (-115.0, -28.0),
    (1, "Multiple CAMELS Forcings", 2019): (-55.0, 16.0),
    (1, "Multiple CAMELS Forcings", 2022): (-95.0, 14.0),
    (1, "Multiple CAMELS Forcings", 2023): (-115.0, -28.0),
    (1, "Multiple CAMELS Forcings", 2024): (-95.0, 14.0),
    (1, "Multiple CAMELS Forcings", 2025): (-115.0, 14.0),
    (1, "Multiple CAMELS Forcings", 2026): (-125.0, 12.0),
    (1, "Freeform Forcings", 2021): (10.0, 10.0),
    (1, "Freeform Forcings", 2023): (10.0, -24.0),
    (2, "Single CAMELS Forcing", 2019): (10.0, -26.0),
    (2, "Single CAMELS Forcing", 2021): (-110.0, -30.0),
    (2, "Single CAMELS Forcing", 2022): (-95.0, -26.0),
    (2, "Single CAMELS Forcing", 2023): (-95.0, -28.0),
    (2, "Single CAMELS Forcing", 2024): (-105.0, -28.0),
    (2, "Single CAMELS Forcing", 2025): (-110.0, -28.0),
    (2, "Multiple CAMELS Forcings", 2019): (10.0, 8.0),
    (2, "Multiple CAMELS Forcings", 2021): (-115.0, 14.0),
    (2, "Multiple CAMELS Forcings", 2022): (-95.0, 12.0),
    (2, "Multiple CAMELS Forcings", 2024): (-105.0, 10.0),
    (2, "Multiple CAMELS Forcings", 2025): (-115.0, 14.0),
    (2, "Multiple CAMELS Forcings", 2026): (-125.0, 12.0),
    (2, "Freeform Forcings", 2023): (10.0, -24.0),
}


def classify_model_category(row: dict[str, Any]) -> str:
    """Map a leaderboard JSON row to one of the 4 canonical MIP model categories."""
    fid = str(row.get("family_id", "")).lower()
    paradigm = str(row.get("paradigm", "")).lower()
    name = str(row.get("model_name", "")).lower()

    if any(
        token in fid or token in name
        for token in ("dhbv", "δhbv", "mlp_hbv", "mlp-hbv", "16hbv", "dpl", "hy2dl")
    ):
        return "Hybrid (ML + Physics)"

    explicit = row.get("model_category")
    if explicit in MODEL_CATEGORY_COLORS:
        return str(explicit)
    if "hybrid" in paradigm and any(
        token in paradigm or token in name
        for token in ("hbv", "conceptual", "process", "surrogate", "multi-model")
    ):
        return "Hybrid (ML + Physics)"
    if any(token in fid or token in name for token in ("vic", "mhm", "nwm", "summa")):
        return "Process-based"
    if any(
        token in fid or token in name
        for token in ("sacsma", "sac-sma", "hbv", "fuse", "marrmot", "gr4j")
    ):
        return "Conceptual"
    if "conceptual" in paradigm:
        return "Conceptual"
    if "land-surface" in paradigm or "distributed process" in paradigm:
        return "Process-based"
    return "Machine learning"


def row_source_category(row: dict[str, Any]) -> str:
    """Return the leaderboard source category string of `row` as stored ('Literature', 'MIP Re-run', ...)."""
    return str(row.get("category") or "").strip()


def normalize_source_category(row: dict[str, Any]) -> str:
    """Normalize source category strings to 'Literature', 'MIP Re-run', 'Archive', 'Archive*', or 'Contribution'.

    Returns '' when the row carries no category.
    """
    raw = row_source_category(row).lower()
    raw = re.sub(r"^(?:category[_\s]*|cat\.?\s*)", "", raw).strip()
    raw = raw.strip("()").strip()
    raw = raw.replace("_asterisk", "*").replace(" *", "*")
    word_map = {
        "literature": "Literature",
        "lit": "Literature",
        "mip re-run": "MIP Re-run",
        "mip-rerun": "MIP Re-run",
        "mip rerun": "MIP Re-run",
        "re-run": "MIP Re-run",
        "rerun": "MIP Re-run",
        "archive*": "Archive*",
        "archive": "Archive",
        "contribution": "Contribution",
        "contributed": "Contribution",
        "contribution (era)": "Contribution",
        "contributed (era)": "Contribution",
        "community": "Contribution",
        "community submission": "Contribution",
        "community contribution": "Contribution",
    }
    return word_map.get(raw, raw)


def is_this_study_row(row: dict[str, Any]) -> bool:
    """True when the row's citation / name declares the result to be produced by this study."""
    text = f"{row.get('citation') or ''} {row.get('model_name') or ''}".lower()
    return "this study" in text


def is_mip_result_row(row: dict[str, Any]) -> bool:
    """True when the row's result was produced by this MIP pipeline rather than by the cited paper.

    That is: 'MIP Re-run' entries, 'Contribution' entries, user submissions
    (`is_user_submission` / `submitted_at`), or rows citing "this study". 'Literature'/'Archive'/'Archive*'
    rows are literature/archive results even if their citation also mentions this study.
    """
    category = normalize_source_category(row)
    if category in MIP_RERUN_CATEGORIES:
        return True
    if category in LITERATURE_CATEGORIES:
        return False
    if row.get("is_user_submission") or row.get("submitted_at"):
        return True
    return is_this_study_row(row)


def _coerce_year(value: Any) -> int | None:
    """Return `value` as an int publication year if it is an integral number in range, else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        year = int(value)
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)) or float(value) != int(value):
            return None
        year = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\s*\d{4}\s*", value):
        year = int(value)
    else:
        return None
    if PUBLICATION_YEAR_MIN <= year <= PUBLICATION_YEAR_MAX:
        return year
    return None


def _year_from_submitted_at(value: Any) -> int | None:
    """Return the calendar year of a `submitted_at` timestamp (ISO-8601 string, datetime, date)."""
    if value is None or isinstance(value, bool):
        return None
    year_attr = getattr(value, "year", None)
    if isinstance(year_attr, int):
        return _coerce_year(year_attr)
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return _coerce_year(int(value.astype("datetime64[Y]").astype(np.int64)) + 1970)
    m = re.match(r"\s*(\d{4})-\d{2}(?:-\d{2})?", str(value))
    return _coerce_year(int(m.group(1))) if m else None


def _detect_cited_year(row: dict[str, Any]) -> int | None:
    """Find the year a row cites: explicit `publication_year`, then `model_name`, then `citation`."""
    explicit = _coerce_year(row.get("publication_year"))
    if explicit is not None:
        return explicit
    for field in ("model_name", "citation"):
        matches = _PUBLICATION_YEAR_PATTERN.findall(str(row.get(field) or ""))
        if matches:
            return int(matches[0])
    return None


def _detect_series1_year(row: dict[str, Any]) -> int | None:
    """Find the original publication year of the Series-1 paper a 'MIP Re-run' builds from."""
    explicit = _coerce_year(row.get("publication_year"))
    if explicit is not None and explicit != MIP_STUDY_YEAR:
        return explicit
    years = [
        int(y)
        for field in ("model_name", "citation")
        for y in _PUBLICATION_YEAR_PATTERN.findall(str(row.get(field) or ""))
    ]
    prior_years = [y for y in years if y != MIP_STUDY_YEAR]
    if prior_years:
        return prior_years[0]
    family_id = str(row.get("family_id") or "").strip()
    if family_id in SERIES1_FAMILY_PUBLICATION_YEARS:
        return SERIES1_FAMILY_PUBLICATION_YEARS[family_id]
    if explicit is not None:
        return explicit
    if years:
        return years[0]
    return None


def extract_publication_year(row: dict[str, Any]) -> int:
    """Return the year at which a leaderboard JSON row is plotted on the Figure 2 x-axis.

    Rules:
    - 'MIP Re-run' entries (models for which we ran a Series-1 author reproduction experiment
      and then re-ran under the standardized MIP protocol) are plotted at the publication year of
      the original paper they build from (`_detect_series1_year`), falling back to `submitted_at`
      or `MIP_STUDY_YEAR` only if no original paper year is present.
    - 'Contribution' entries (new baseline/community models without a Series-1
      reproduction paper) are dated by `submitted_at` when present, otherwise
      `max(MIP_STUDY_YEAR, cited_year)`.
    - Literature / archive rows ('Literature'/'Archive'/'Archive*') use the explicit integer
      `publication_year` field if present, else the first 1990–2035 year found in `model_name`,
      else in `citation`.
    - A literature / uncategorized row with no year at all raises `ValueError` naming the entry:
      there is no silent default year.
    """
    cat = normalize_source_category(row)
    cited_year = _detect_cited_year(row)

    if cat == "MIP Re-run" or (
        is_mip_result_row(row)
        and cat not in MIP_CONTRIBUTION_CATEGORIES
        and _detect_series1_year(row) is not None
    ):
        series1_year = _detect_series1_year(row)
        if series1_year is not None:
            return series1_year
        submitted_year = _year_from_submitted_at(row.get("submitted_at"))
        return submitted_year if submitted_year is not None else MIP_STUDY_YEAR

    if is_mip_result_row(row):
        submitted_year = _year_from_submitted_at(row.get("submitted_at"))
        if submitted_year is not None:
            return submitted_year
        return MIP_STUDY_YEAR if cited_year is None else max(MIP_STUDY_YEAR, cited_year)

    if cited_year is not None:
        return cited_year

    raise ValueError(
        "Cannot determine the publication year of leaderboard entry "
        f"{row.get('entry_id')!r} (model_name={row.get('model_name')!r}, "
        f"citation={row.get('citation')!r}, category={row_source_category(row) or None!r}): "
        f"no integer 'publication_year' field and no {PUBLICATION_YEAR_MIN}–{PUBLICATION_YEAR_MAX} "
        "year in model_name or citation. Add 'publication_year' to the row, cite the source with "
        "its year, or mark the row as a MIP result (category 'MIP Re-run'/'Contribution', "
        "'submitted_at', or citation 'This Study')."
    )


_NUMERIC_STRING_PATTERN = re.compile(r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*")


def _is_real_number(value: Any) -> bool:
    """True for int / float / numpy numeric scalars (booleans excluded)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _finite_nse_or_none(value: Any) -> float | None:
    """Return `value` as a finite float NSE, or None if it is missing / non-numeric / non-finite."""
    if _is_real_number(value):
        nse = float(value)
    elif isinstance(value, str) and _NUMERIC_STRING_PATTERN.fullmatch(value):
        nse = float(value)
    else:
        return None
    if not math.isfinite(nse):
        return None
    return nse


def resolve_primary_nse(row: dict[str, Any]) -> tuple[float | None, str | None]:
    """Resolve the plottable primary NSE of a leaderboard row.

    Mirrors `leaderboards.update_leaderboard.rank_leaderboard_rows`: the first of
    `primary_ranking_nse`, `ens_med_nse`, `single_med_nse` that is a finite number above the
    legacy sentinel bound `NSE_SENTINEL_FLOOR` is used.

    Returns `(nse, None)` when the row is plottable, or `(None, reason)` when it must be excluded
    (every candidate value null, non-numeric, non-finite, or a sentinel such as -999.0).
    """
    problems: list[str] = []
    for key in ("primary_ranking_nse", "ens_med_nse", "single_med_nse"):
        raw = row.get(key)
        if raw is None:
            continue
        nse = _finite_nse_or_none(raw)
        if nse is None:
            problems.append(f"{key}={raw!r} is not a finite number")
            continue
        if nse <= NSE_SENTINEL_FLOOR:
            problems.append(
                f"{key}={nse!r} is a legacy missing-value sentinel (<= {NSE_SENTINEL_FLOOR})"
            )
            continue
        return nse, None
    if not problems:
        return None, "primary_ranking_nse, ens_med_nse and single_med_nse are all null"
    return None, "; ".join(problems) + "; no finite NSE available"


def format_leader_label(row: dict[str, Any], year: int) -> str:
    """Format a concise 2-line frontier leader label from a leaderboard JSON row.

    Never raises on a missing NSE (rendered as `n/a`). 'MIP Re-run' entries are labelled
    `MIP re-run of <Author> <paper year>` so the result is not attributed to the cited paper.
    """
    raw_name = str(row.get("model_name") or "").strip()
    citation = str(row.get("citation") or "").strip()
    nse = _finite_nse_or_none(row.get("primary_ranking_nse"))
    if nse is not None and nse <= NSE_SENTINEL_FLOOR:
        nse = None
    star = "*" if row.get("asterisk") or raw_name.endswith("*") else ""
    nse_str = f"{nse:.3f}{star}" if nse is not None else f"n/a{star}"

    year_paren_re = r"\(\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*\)"
    short_model = re.sub(
        r"^[A-Z][A-Za-z\s,&.\-]+?" + year_paren_re + r"\s*",
        "",
        raw_name,
    ).strip()
    if not short_model:
        # Names such as "Community Frontier Model (2026)" carry no author prefix: keep the name.
        short_model = re.sub(r"\s*" + year_paren_re, "", raw_name).strip() or raw_name

    short_model = re.sub(r"\(\d+-model ensemble\)", "Ens", short_model)
    short_model = re.sub(r"\([^,)]+,\s*\d+-model ensemble\)", "Ens", short_model)
    short_model = re.sub(r"\(3-Forcing Gauged Ensemble\)", "Ens", short_model)
    short_model = re.sub(r"\(\d+-calibration[^)]*ensemble\)", "Ens", short_model)
    short_model = re.sub(r"\s+Ensemble\s+Ens", " Ens", short_model)
    short_model = re.sub(r"\s*\(Regionalized\)", "", short_model)
    short_model = re.sub(r"\s*\(no holdout\)", "", short_model)
    short_model = re.sub(r"\s+Lagged Discharge\s+", " ", short_model)
    short_model = re.sub(r"\s+", " ", short_model).strip()

    author_re = r"^([A-Z][A-Za-z\-]+(?:\s+&\s+[A-Z][A-Za-z\-]+|\s+et\s+al\.)?)"
    m_author = re.match(author_re + r"\s*\((\d{4}[a-z]?)", raw_name)
    m_cit_auth = re.match(author_re, citation)
    cited_years = _PUBLICATION_YEAR_PATTERN.findall(citation)
    citation_is_this_study = "this study" in citation.lower()

    source_tag: str | None
    if m_author:
        source_tag = f"{m_author.group(1).replace(' et al.', '')} {m_author.group(2)}"
    elif m_cit_auth and cited_years and not citation.lower().startswith("this study"):
        source_tag = f"{m_cit_auth.group(1).replace(' et al.', '')} {cited_years[0]}"
    elif citation_is_this_study or not citation:
        source_tag = None
    elif m_cit_auth:
        source_tag = f"{m_cit_auth.group(1).replace(' et al.', '')} {year}"
    else:
        source_tag = f"{citation} {year}"

    if normalize_source_category(row) == "MIP Re-run":
        author_tag = f"MIP re-run of {source_tag}" if source_tag else "This study (MIP re-run)"
    elif source_tag is None:
        author_tag = "This study"
    else:
        author_tag = source_tag

    return f"{short_model}\n({author_tag}: {nse_str})"


def partition_panel_rows(
    all_leaderboards: dict[str, Any],
    spatial_regime: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the rows of `spatial_regime` ('gauged', 'pub', 'pur') into plotted points and exclusions.

    Returns `(points, excluded_rows)`. Every row with a finite, non-sentinel primary NSE becomes
    a point; every other row is listed in `excluded_rows` with its `entry_id` and a reason.
    Rows whose publication year cannot be determined raise `ValueError` (see
    `extract_publication_year`).
    """
    points: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for lb in all_leaderboards.get("leaderboards", []):
        lbid = str(lb.get("id", ""))
        if not lbid.startswith(f"{spatial_regime}__"):
            continue
        forcing_key = lbid.split("__", 1)[1]
        if forcing_key not in FORCING_REGIME_MAP:
            continue
        regime_label = FORCING_REGIME_MAP[forcing_key]
        for row in lb.get("rows", []):
            nse_val, reason = resolve_primary_nse(row)
            if nse_val is None:
                excluded.append(
                    {
                        "entry_id": str(row.get("entry_id", "")),
                        "leaderboard_id": lbid,
                        "model_name": str(row.get("model_name", "")),
                        "primary_ranking_nse": row.get("primary_ranking_nse"),
                        "reason": reason,
                    }
                )
                continue
            year = extract_publication_year(row)
            mcat = classify_model_category(row)
            label = format_leader_label(
                {**row, "primary_ranking_nse": nse_val},
                year=year,
            )
            exclude_frontier = "oracle" in str(row.get("model_name", "")).lower()
            points.append(
                {
                    "entry_id": str(row.get("entry_id", "")),
                    "leaderboard_id": lbid,
                    "year": year,
                    "nse": nse_val,
                    "model_category": mcat,
                    "input_regime": regime_label,
                    "label": label,
                    "model_name": str(row.get("model_name", "")),
                    "exclude_frontier": exclude_frontier,
                    "category": row_source_category(row),
                    "provenance": "mip_rerun" if is_mip_result_row(row) else "literature",
                }
            )
    return points, excluded


def extract_panel_models_from_leaderboards(
    all_leaderboards: dict[str, Any],
    spatial_regime: str,
) -> list[dict[str, Any]]:
    """Extract all plotted model points for `spatial_regime` ('gauged', 'pub', 'pur') from JSON."""
    return partition_panel_rows(all_leaderboards, spatial_regime)[0]


def compute_regime_leaders(
    models: list[dict[str, Any]],
    regime: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return `(leaders, non_leaders)` for a given `regime` directly from the JSON points."""
    pts = [m for m in models if m["input_regime"] == regime]
    if not pts:
        return [], []
    eligible = [m for m in pts if not m.get("exclude_frontier", False)]
    years = sorted({int(m["year"]) for m in eligible})
    leaders: list[dict[str, Any]] = []
    leader_ids: set[tuple[str, str]] = set()
    running_max = -np.inf
    for yr in years:
        y_pts = [m for m in eligible if int(m["year"]) == yr]
        best_at_y = max(y_pts, key=lambda x: (float(x["nse"]), x["model_name"]))
        if float(best_at_y["nse"]) > running_max + 1e-9:
            running_max = float(best_at_y["nse"])
            leaders.append(best_at_y)
            leader_ids.add((best_at_y["leaderboard_id"], best_at_y["entry_id"]))
    non_leaders = [
        m for m in pts if (m["leaderboard_id"], m["entry_id"]) not in leader_ids
    ]
    return leaders, non_leaders


def compute_panel_ylim(
    nses: list[float],
    ylim_hint: tuple[float, float],
    pad: float = FIGURE_2_YLIM_PAD,
) -> tuple[float, float]:
    """Derive a panel's y-limits so that every plotted NSE in `nses` lies strictly inside them.

    `ylim_hint` is the minimum extent of the axis (kept for visual consistency across panels);
    the lower limit drops to `min(nses) - pad` and the upper limit rises to `max(nses) + pad`
    whenever the data require it. Nothing is ever cropped.
    """
    finite = [float(v) for v in nses if math.isfinite(float(v))]
    if not finite:
        return (float(ylim_hint[0]), float(ylim_hint[1]))
    lo = min(float(ylim_hint[0]), min(finite) - pad)
    hi = max(float(ylim_hint[1]), max(finite) + pad)
    return (lo, hi)


def compute_year_axis(
    year_range: tuple[int, int],
) -> tuple[tuple[float, float], list[int], float]:
    """Return `(xlim, xticks, frontier_end)` for the publication-year axis covering `year_range`.

    The axis always spans at least `FIGURE_2_MIN_YEAR`..`MIP_STUDY_YEAR` and expands to fit
    earlier / later years so that no point is ever drawn outside the axes.
    """
    lo = min(FIGURE_2_MIN_YEAR, int(year_range[0]))
    hi = max(MIP_STUDY_YEAR, int(year_range[1]))
    span = hi - lo
    step = 1 if span <= 14 else (2 if span <= 30 else 5)
    xticks = list(range(lo, hi + 1, step))
    if xticks[-1] != hi:
        xticks.append(hi)
    return (lo - 0.6, hi + 0.8), xticks, hi + 0.35


def _get_text_bbox(ann: Any, renderer: Any) -> Any:
    patch = ann.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent(renderer)
    return ann.get_window_extent(renderer)


def _optimize_annotations(
    fig: Any,
    ax: Any,
    annotations: list[Any],
    point_coords_disp: list[tuple[float, float]],
    pad_px: float = 6.0,
    n_iter: int = 80,
) -> None:
    """Resolve annotation-annotation and annotation-border overlaps deterministically."""
    if not annotations:
        return
    px_per_pt = fig.dpi / 72.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)

    for it in range(n_iter):
        moved = False
        bboxes = [
            _get_text_bbox(ann, renderer).expanded(1.04, 1.10) for ann in annotations
        ]

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                b1, b2 = bboxes[i], bboxes[j]
                if b1.overlaps(b2):
                    overlap_x = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                    overlap_y = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                    c1x, c1y = 0.5 * (b1.x0 + b1.x1), 0.5 * (b1.y0 + b1.y1)
                    c2x, c2y = 0.5 * (b2.x0 + b2.x1), 0.5 * (b2.y0 + b2.y1)
                    dx = 1.0 if c1x >= c2x else -1.0
                    dy = 1.0 if c1y >= c2y else -1.0

                    if overlap_y * 2.2 < overlap_x:
                        shift_y_pt = 0.45 * (overlap_y + pad_px) * dy / px_per_pt
                        shift_x_pt = 0.08 * overlap_x * dx / px_per_pt
                    else:
                        shift_x_pt = 0.40 * (overlap_x + pad_px) * dx / px_per_pt
                        shift_y_pt = 0.30 * (overlap_y + pad_px) * dy / px_per_pt

                    x1, y1 = annotations[i].get_position()
                    x2, y2 = annotations[j].get_position()
                    annotations[i].set_position((x1 + shift_x_pt, y1 + shift_y_pt))
                    annotations[j].set_position((x2 - shift_x_pt, y2 - shift_y_pt))
                    moved = True

        marker_pad = 14.0
        for i, b in enumerate(bboxes):
            for px, py in point_coords_disp:
                if (b.x0 - marker_pad <= px <= b.x1 + marker_pad) and (
                    b.y0 - marker_pad <= py <= b.y1 + marker_pad
                ):
                    cx, cy = 0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1)
                    dy = 1.0 if cy >= py else -1.0
                    dx = 1.0 if cx >= px else -1.0
                    x1, y1 = annotations[i].get_position()
                    annotations[i].set_position((x1 + 1.5 * dx, y1 + 3.0 * dy))
                    moved = True

        for i, b in enumerate(bboxes):
            x1, y1 = annotations[i].get_position()
            dx_pt, dy_pt = 0.0, 0.0
            margin = 10.0
            if b.x0 < ax_bbox.x0 + margin:
                dx_pt = (ax_bbox.x0 + margin - b.x0) / px_per_pt
            elif b.x1 > ax_bbox.x1 - margin:
                dx_pt = -(b.x1 - (ax_bbox.x1 - margin)) / px_per_pt
            if b.y0 < ax_bbox.y0 + margin:
                dy_pt = (ax_bbox.y0 + margin - b.y0) / px_per_pt
            elif b.y1 > ax_bbox.y1 - margin:
                dy_pt = -(b.y1 - (ax_bbox.y1 - margin)) / px_per_pt
            if dx_pt != 0.0 or dy_pt != 0.0:
                annotations[i].set_position((x1 + dx_pt, y1 + dy_pt))
                moved = True

        fig.canvas.draw()
        if not moved and it > 10:
            break


def compute_water_year_completeness(
    qobs_all: np.ndarray,
    dates_all: np.ndarray,
    *,
    include_partial_water_years: bool = False,
    newman_eval_water_years: tuple[int, int] = NEWMAN_EVAL_WATER_YEARS,
    newman_train_water_years: tuple[int, int] = NEWMAN_TRAIN_WATER_YEARS,
) -> dict[str, Any]:
    """Compute the Figure-1 `protocol["water_year_completeness"]` block from observed streamflow.

    Args:
        qobs_all: `(n_basins, n_days)` observed daily discharge; missing days are NaN (any
            non-finite value is treated as missing). E.g. `newman_eval_targets.zarr["qobs_all"]`.
        dates_all: `(n_days,)` `datetime64` dates aligned with the columns of `qobs_all`, e.g.
            `newman_eval_targets.zarr["date_all"]`. Must be unique and free of NaT.
        include_partial_water_years: Water years for which `dates_all` does not contain every
            calendar day (Oct 1 – Sep 30) are dropped by default, because their completeness
            cannot be assessed; pass True to keep them (statistics then use the days present).
        newman_eval_water_years / newman_train_water_years: Benchmark windows copied verbatim
            into the returned block for the Figure-1 shading.

    Returns a JSON-serializable dict with keys `n_basins`, `water_years` (ascending ints),
    `basins_with_100pct_complete` (basins with no missing day in the water year),
    `basins_with_any_valid_data` (basins with >= 1 valid day in the water year),
    `mean_daily_coverage_fraction` (mean over basins of the fraction of valid days in the water
    year), `newman_eval_water_years`, `newman_train_water_years`, and
    `partial_water_years_excluded` (the water years dropped as incomplete calendars).
    A water year `Y` spans Oct 1 of `Y-1` through Sep 30 of `Y`.
    """
    q = np.asarray(qobs_all, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"qobs_all must be a 2-D (n_basins, n_days) array; got shape {q.shape}.")
    dates = np.asarray(dates_all)
    if not np.issubdtype(dates.dtype, np.datetime64):
        raise ValueError(
            f"dates_all must be a numpy datetime64 array (e.g. newman_eval_targets.zarr['date_all']); "
            f"got dtype {dates.dtype}."
        )
    days = dates.astype("datetime64[D]")
    if days.ndim != 1 or days.shape[0] != q.shape[1]:
        raise ValueError(
            f"dates_all must be 1-D with length n_days={q.shape[1]}; got shape {days.shape}."
        )
    if days.shape[0] == 0:
        raise ValueError("dates_all is empty; cannot compute water-year completeness.")
    if np.any(np.isnat(days)):
        raise ValueError("dates_all contains NaT values.")
    if np.unique(days).shape[0] != days.shape[0]:
        raise ValueError("dates_all contains duplicate calendar days.")

    cal_years = days.astype("datetime64[Y]").astype(np.int64) + 1970
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    water_years = np.where(months >= 10, cal_years + 1, cal_years)
    valid = np.isfinite(q)

    out_wys: list[int] = []
    complete: list[int] = []
    any_valid: list[int] = []
    mean_cov: list[float] = []
    partial: list[int] = []
    for wy in np.unique(water_years):
        wy_int = int(wy)
        mask = water_years == wy
        n_calendar_days = int(
            (np.datetime64(f"{wy_int}-10-01", "D") - np.datetime64(f"{wy_int - 1}-10-01", "D"))
            .astype(np.int64)
        )
        if int(mask.sum()) != n_calendar_days and not include_partial_water_years:
            partial.append(wy_int)
            continue
        v = valid[:, mask]
        out_wys.append(wy_int)
        complete.append(int(np.count_nonzero(v.all(axis=1))))
        any_valid.append(int(np.count_nonzero(v.any(axis=1))))
        mean_cov.append(float(np.mean(v.mean(axis=1))))

    return {
        "n_basins": int(q.shape[0]),
        "water_years": out_wys,
        "basins_with_100pct_complete": complete,
        "basins_with_any_valid_data": any_valid,
        "mean_daily_coverage_fraction": mean_cov,
        "newman_eval_water_years": [int(newman_eval_water_years[0]), int(newman_eval_water_years[1])],
        "newman_train_water_years": [
            int(newman_train_water_years[0]),
            int(newman_train_water_years[1]),
        ],
        "partial_water_years_excluded": partial,
    }


def _validate_water_year_window(value: Any, key: str, source: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(_coerce_int(v) is None for v in value)
    ):
        raise ValueError(
            f"{source}[{key!r}] must be a [first_water_year, last_water_year] pair of integers; "
            f"got {value!r}."
        )
    first, last = int(value[0]), int(value[1])
    if first > last:
        raise ValueError(f"{source}[{key!r}] must be ascending; got {value!r}.")
    return first, last


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return int(value) if float(value) == int(value) else None
    return None


def validate_water_year_completeness(
    wy_data: Any,
    source: str = "protocol['water_year_completeness']",
) -> dict[str, Any]:
    """Validate and normalize a `water_year_completeness` block; raise `ValueError` if malformed.

    Required keys: `n_basins` (positive int), `water_years` (ascending ints) and the three
    equal-length numeric series `basins_with_100pct_complete`, `basins_with_any_valid_data`
    (integers in `[0, n_basins]`) and `mean_daily_coverage_fraction` (floats in `[0, 1]`).
    Optional `newman_eval_water_years` / `newman_train_water_years` default to the Newman windows.
    """
    if not isinstance(wy_data, dict):
        raise ValueError(
            f"{source} must be a JSON object with keys {list(REQUIRED_WATER_YEAR_COMPLETENESS_KEYS)}; "
            f"got {type(wy_data).__name__}."
        )
    missing = [k for k in REQUIRED_WATER_YEAR_COMPLETENESS_KEYS if k not in wy_data]
    if missing:
        raise ValueError(
            f"{source} is missing required key(s) {missing}; required keys are "
            f"{list(REQUIRED_WATER_YEAR_COMPLETENESS_KEYS)}. Regenerate the block with "
            "leaderboards.generate_figures.compute_water_year_completeness(qobs_all, date_all)."
        )
    n_basins = _coerce_int(wy_data["n_basins"])
    if n_basins is None or n_basins <= 0:
        raise ValueError(f"{source}['n_basins'] must be a positive integer; got {wy_data['n_basins']!r}.")

    wys_raw = wy_data["water_years"]
    if not isinstance(wys_raw, (list, tuple)) or len(wys_raw) == 0:
        raise ValueError(f"{source}['water_years'] must be a non-empty list of integers.")
    wys = [_coerce_int(v) for v in wys_raw]
    if any(v is None for v in wys):
        raise ValueError(f"{source}['water_years'] must contain only integers; got {wys_raw!r}.")
    wys_int = [int(v) for v in wys if v is not None]
    if any(b <= a for a, b in zip(wys_int, wys_int[1:])):
        raise ValueError(f"{source}['water_years'] must be strictly increasing; got {wys_int!r}.")

    normalized: dict[str, Any] = {"n_basins": n_basins, "water_years": wys_int}
    for key in WATER_YEAR_COMPLETENESS_SERIES_KEYS:
        series_raw = wy_data[key]
        if not isinstance(series_raw, (list, tuple)):
            raise ValueError(f"{source}[{key!r}] must be a list of numbers; got {type(series_raw).__name__}.")
        if len(series_raw) != len(wys_int):
            raise ValueError(
                f"{source}[{key!r}] has {len(series_raw)} values but 'water_years' has "
                f"{len(wys_int)}; all series must be aligned with 'water_years'."
            )
        if not all(_is_real_number(v) for v in series_raw):
            raise ValueError(f"{source}[{key!r}] must contain only numbers; got {series_raw!r}.")
        series = np.asarray(series_raw, dtype=np.float64)
        if series.ndim != 1 or not np.all(np.isfinite(series)):
            raise ValueError(f"{source}[{key!r}] must be a 1-D list of finite numbers.")
        if key == "mean_daily_coverage_fraction":
            if np.any(series < 0.0) or np.any(series > 1.0):
                raise ValueError(f"{source}[{key!r}] values must lie in [0, 1]; got {series_raw!r}.")
            normalized[key] = [float(v) for v in series]
        else:
            if np.any(series < 0) or np.any(series > n_basins) or np.any(series != np.round(series)):
                raise ValueError(
                    f"{source}[{key!r}] values must be integer basin counts in [0, {n_basins}]; "
                    f"got {series_raw!r}."
                )
            normalized[key] = [int(v) for v in series]

    normalized["newman_eval_water_years"] = list(
        _validate_water_year_window(
            wy_data.get("newman_eval_water_years", list(NEWMAN_EVAL_WATER_YEARS)),
            "newman_eval_water_years",
            source,
        )
    )
    normalized["newman_train_water_years"] = list(
        _validate_water_year_window(
            wy_data.get("newman_train_water_years", list(NEWMAN_TRAIN_WATER_YEARS)),
            "newman_train_water_years",
            source,
        )
    )
    return normalized


def generate_figure_1_water_year_coverage(
    protocol: dict[str, Any],
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Path]:
    """Generate Figure 1 (`figure_1_water_year_coverage.png` and `.pdf`) from `protocol`.

    The data come exclusively from `protocol["water_year_completeness"]` (the protocol block of
    the leaderboard JSON files). Raises `ValueError` if that block is absent or malformed; there
    is no built-in fallback series.
    """
    if not isinstance(protocol, dict):
        raise ValueError(
            f"Figure 1 requires the leaderboard JSON protocol block as a dict; got {type(protocol).__name__}."
        )
    if "water_year_completeness" not in protocol:
        raise ValueError(
            "Figure 1 requires protocol['water_year_completeness'] in the leaderboard JSON protocol "
            "block (leaderboards/<leaderboard_id>.json -> protocol.water_year_completeness, mirrored "
            "in tables_3_to_14.json) with keys "
            f"{list(REQUIRED_WATER_YEAR_COMPLETENESS_KEYS)}. It is absent; regenerate it with "
            "leaderboards.generate_figures.compute_water_year_completeness(qobs_all, date_all) from "
            "newman_eval_targets.zarr and store it in the protocol block."
        )
    wy_data = validate_water_year_completeness(protocol["water_year_completeness"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_basins = int(wy_data["n_basins"])
    wys = np.asarray(wy_data["water_years"], dtype=np.int64)
    c_100 = np.asarray(wy_data["basins_with_100pct_complete"], dtype=np.float64)
    c_any = np.asarray(wy_data["basins_with_any_valid_data"], dtype=np.float64)
    m_daily = np.asarray(wy_data["mean_daily_coverage_fraction"], dtype=np.float64)
    eval_wy0, eval_wy1 = wy_data["newman_eval_water_years"]
    train_wy0, train_wy1 = wy_data["newman_train_water_years"]

    y_100 = c_100 / n_basins * 100.0
    y_any = c_any / n_basins * 100.0
    y_mean = m_daily * 100.0

    fig, ax = plt.subplots(figsize=(10.0, 4.4), dpi=300)
    ax.axvspan(
        eval_wy0 - 0.5,
        eval_wy1 + 0.5,
        color="#d9ecff",
        alpha=0.65,
        label=f"Newman Period Evaluation Decade (Water Years {eval_wy0}–{eval_wy1})",
    )
    ax.axvspan(
        train_wy0 - 0.5,
        train_wy1 + 0.5,
        color="#dff5e1",
        alpha=0.65,
        label=f"Newman Period Training / Calibration Window (Water Years {train_wy0}–{train_wy1})",
    )

    ax.plot(
        wys,
        y_mean,
        "-o",
        color="#1f4e78",
        lw=2.2,
        ms=4.5,
        label="Mean Daily Observation Coverage (%)",
    )
    ax.plot(
        wys,
        y_any,
        "-.d",
        color="#385723",
        lw=1.6,
        ms=3.8,
        label="Basins with Any Valid Data in Water Year (%)",
    )
    ax.plot(
        wys,
        y_100,
        ":^",
        color="#c55a11",
        lw=1.8,
        ms=4.0,
        label="Basins with 100% Complete Water Year (%)",
    )

    ax.set_xlim(float(wys.min()) - 0.5, float(wys.max()) + 0.5)
    ax.set_ylim(float(np.min(np.concatenate([y_100, y_any, y_mean]))) - 3.0, 101.2)
    ax.set_xlabel(
        f"Water Year (Water Year {int(wys.min())} = Oct {int(wys.min()) - 1} – Sep {int(wys.min())})",
        fontsize=10.5,
    )
    ax.set_ylabel(f"Percentage of {n_basins} CAMELS Basins (%)", fontsize=10.5)
    ax.set_title(
        f"Streamflow Observation Completeness by Water Year Across the {n_basins} CAMELS-US Benchmark Basins",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    plt.tight_layout()

    png_path = out_dir / "figure_1_water_year_coverage.png"
    pdf_path = out_dir / "figure_1_water_year_coverage.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _draw_point(ax: Any, pt: dict[str, Any], rstyle: dict[str, Any], is_leader: bool) -> None:
    """Draw one leaderboard point; hollow markers denote MIP-produced (MIP Re-run / Contribution) results."""
    x = float(pt["year"])
    y = float(pt["nse"])
    color = MODEL_CATEGORY_COLORS[pt["model_category"]]
    hollow = pt.get("provenance") == "mip_rerun"
    marker = rstyle["marker"]
    z = int(rstyle["zorder_base"])
    if is_leader and hollow:
        ax.scatter(
            x, y, marker=marker, s=175, facecolor="none", edgecolor="#0F172A",
            linewidth=1.0, alpha=1.0, zorder=z + 2,
        )
        ax.scatter(
            x, y, marker=marker, s=115, facecolor="white", edgecolor=color,
            linewidth=2.0, alpha=1.0, zorder=z + 3,
        )
    elif is_leader:
        ax.scatter(
            x, y, marker=marker, s=115, facecolor=color, edgecolor="#0F172A",
            linewidth=1.2, alpha=1.0, zorder=z + 3,
        )
    elif hollow:
        ax.scatter(
            x, y, marker=marker, s=46, facecolor="white", edgecolor=color,
            linewidth=1.3, alpha=0.85, zorder=2,
        )
    else:
        ax.scatter(
            x, y, marker=marker, s=46, facecolor=color, edgecolor="white",
            linewidth=0.7, alpha=0.58, zorder=2,
        )


def _plot_subpanel(
    fig: Any,
    ax: Any,
    panel_idx: int,
    models: list[dict[str, Any]],
    title: str,
    ylim_hint: tuple[float, float],
    year_range: tuple[int, int],
) -> dict[str, Any]:
    """Plot a single spatial-regime subpanel of Figure 2.

    Every point in `models` is drawn (axis limits are derived from the data, see
    `compute_panel_ylim` / `compute_year_axis`) and every frontier leader receives a label.
    Returns a dict with `leaders`, `annotations`, `leader_coords`, `n_drawn`, `ylim`, `xlim`.
    """
    ax.set_facecolor("#FAFBFC")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#CBD5E1", alpha=0.75, zorder=0)

    ylim = compute_panel_ylim([float(m["nse"]) for m in models], ylim_hint)
    xlim, xticks, frontier_end = compute_year_axis(year_range)
    y_mid = 0.5 * (ylim[0] + ylim[1])
    label_left_from_year = int(year_range[1]) - 2

    panel_leaders: list[dict[str, Any]] = []
    annotations: list[Any] = []
    leader_coords: list[tuple[float, float]] = []
    n_drawn = 0

    for regime, rstyle in REGIME_STYLES.items():
        leaders, non_leaders = compute_regime_leaders(models, regime)
        if not leaders and not non_leaders:
            continue

        for pt in non_leaders:
            _draw_point(ax, pt, rstyle, is_leader=False)
            n_drawn += 1

        if leaders:
            for i, ldr in enumerate(leaders):
                yr_curr = float(ldr["year"])
                nse_curr = float(ldr["nse"])
                mcat_curr = ldr["model_category"]
                yr_next = (
                    float(leaders[i + 1]["year"])
                    if i + 1 < len(leaders)
                    else frontier_end
                )
                ax.plot(
                    [yr_curr, yr_next],
                    [nse_curr, nse_curr],
                    linestyle=rstyle["linestyle"],
                    linewidth=2.5,
                    color=MODEL_CATEGORY_COLORS[mcat_curr],
                    alpha=0.92,
                    zorder=rstyle["zorder_base"],
                )
                if i + 1 < len(leaders):
                    nse_next = float(leaders[i + 1]["nse"])
                    mcat_next = leaders[i + 1]["model_category"]
                    ax.plot(
                        [yr_next, yr_next],
                        [nse_curr, nse_next],
                        linestyle=rstyle["linestyle"],
                        linewidth=2.2,
                        color=MODEL_CATEGORY_COLORS[mcat_next],
                        alpha=0.85,
                        zorder=rstyle["zorder_base"],
                    )

            for ldr in leaders:
                yr = int(ldr["year"])
                nse = float(ldr["nse"])
                mcat = ldr["model_category"]
                label = str(ldr["label"])
                panel_leaders.append(ldr)
                leader_coords.append((float(yr), nse))
                _draw_point(ax, ldr, rstyle, is_leader=True)
                n_drawn += 1
                dx, dy = INITIAL_LABEL_OFFSETS.get(
                    (panel_idx, regime, yr),
                    (
                        -95.0 if yr >= label_left_from_year else 10.0,
                        12.0 if nse < y_mid else -24.0,
                    ),
                )
                ann = ax.annotate(
                    label,
                    xy=(yr, nse),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.1,
                    fontweight="bold",
                    color="#0F172A",
                    bbox={
                        "boxstyle": "round,pad=0.26",
                        "facecolor": "white",
                        "edgecolor": MODEL_CATEGORY_COLORS[mcat],
                        "linewidth": 1.3,
                        "alpha": 0.95,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "connectionstyle": "arc3,rad=0.08",
                        "color": MODEL_CATEGORY_COLORS[mcat],
                        "linewidth": 1.1,
                    },
                    zorder=rstyle["zorder_base"] + 5,
                )
                annotations.append(ann)

    if n_drawn != len(models):
        raise RuntimeError(
            f"Panel {panel_idx} ({title!r}) drew {n_drawn} of {len(models)} points; "
            "every leaderboard row with a finite NSE must be plotted."
        )
    if len(annotations) != len(panel_leaders):
        raise RuntimeError(
            f"Panel {panel_idx} ({title!r}) placed {len(annotations)} labels for "
            f"{len(panel_leaders)} frontier leaders."
        )

    ax.set_title(
        title,
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
        loc="left",
        pad=8,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_ylabel(
        "531-Basin Median NSE\n(1989–1999 Newman Period)",
        fontsize=9.8,
        fontweight="bold",
        color="#1E293B",
    )
    ax.tick_params(axis="both", labelsize=9.2, labelbottom=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return {
        "leaders": panel_leaders,
        "annotations": annotations,
        "leader_coords": leader_coords,
        "n_drawn": n_drawn,
        "ylim": ylim,
        "xlim": xlim,
    }


def _build_legend_handles(
    active_regimes: Iterable[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Return `(category_and_provenance_handles, regime_handles)` for the Figure 2 legends.

    When `active_regimes` is provided, only forcing regimes present in `active_regimes` are
    included in `regime_handles` so single-regime or subset panels never display unused forcing
    regimes (such as Freeform Forcings) in the legend.
    """
    cat_handles = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="#0F172A",
            linewidth=0.8,
            label=cat,
        )
        for cat, color in MODEL_CATEGORY_COLORS.items()
    ]
    cat_handles.append(
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=8.0,
            markerfacecolor="#64748B",
            markeredgecolor="#0F172A",
            label="Filled: Literature / Archive",
        )
    )
    cat_handles.append(
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=8.0,
            markerfacecolor="white",
            markeredgecolor="#64748B",
            markeredgewidth=1.8,
            label="Hollow: MIP Re-run / Contribution",
        )
    )
    allowed_regimes = set(active_regimes) if active_regimes is not None else None
    regime_handles = [
        mlines.Line2D(
            [],
            [],
            color="#334155",
            linestyle=rstyle["linestyle"],
            linewidth=2.3,
            marker=rstyle["marker"],
            markersize=7.5,
            markerfacecolor="#64748B",
            markeredgecolor="#0F172A",
            label=rname,
        )
        for rname, rstyle in REGIME_STYLES.items()
        if allowed_regimes is None or rname in allowed_regimes
    ]
    return cat_handles, regime_handles


_CATEGORY_LEGEND_TITLE = "Model Category (Dot & Active Leader Color)  ·  Result Provenance (Marker Fill)"
_REGIME_LEGEND_TITLE = "Input Forcing Regime (Step-Line Frontier Trace & Marker Shape)"


def generate_figure_2_leaderboard_progress(
    all_leaderboards: dict[str, Any],
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Generate the 3-panel Figure 2 (`figure_2_leaderboard_progress_all_regimes.png`/`.pdf`)
    and the 3 single-panel regime figures (`figure_2a_...`, `figure_2b_...`, `figure_2c_...`).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_models: dict[str, list[dict[str, Any]]] = {}
    panel_excluded: dict[str, list[dict[str, Any]]] = {}
    for spatial in ("gauged", "pub", "pur"):
        panel_models[spatial], panel_excluded[spatial] = partition_panel_rows(
            all_leaderboards, spatial
        )
    all_years = [int(m["year"]) for pts in panel_models.values() for m in pts]
    year_range = (
        (min(all_years), max(all_years)) if all_years else (FIGURE_2_MIN_YEAR, MIP_STUDY_YEAR)
    )

    panel_titles = {
        "gauged": "Gauged Simulation Leaderboards — All Forcing Regimes",
        "pub": "Random Ungauged (12-Fold PUB) Leaderboards — All Forcing Regimes",
        "pur": "Regionally Ungauged (18-Fold HUC-02 PUR) Leaderboards — All Forcing Regimes",
    }
    panel_order = ("gauged", "pub", "pur")

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 15.8), dpi=300, sharex=True)
    fig.patch.set_facecolor("white")

    panel_results: dict[str, dict[str, Any]] = {}
    for p_idx, spatial in enumerate(panel_order):
        panel_results[spatial] = _plot_subpanel(
            fig,
            axes[p_idx],
            panel_idx=p_idx,
            models=panel_models[spatial],
            title=panel_titles[spatial],
            ylim_hint=PANEL_YLIM_HINTS[spatial],
            year_range=year_range,
        )
    axes[2].set_xlabel(
        "Publication Year of Contributing Model / Benchmark Study",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        labelpad=8,
    )

    all_active_regimes = {
        str(m.get("input_regime"))
        for pts in panel_models.values()
        for m in pts
        if m.get("input_regime")
    }
    cat_handles, regime_handles = _build_legend_handles(all_active_regimes)

    leg1 = fig.legend(
        handles=cat_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=6,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=8.6,
        title=_CATEGORY_LEGEND_TITLE,
        title_fontsize=9.5,
    )
    leg1.get_title().set_fontweight("bold")

    leg2 = fig.legend(
        handles=regime_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=max(1, len(regime_handles)),
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=9.3,
        title=_REGIME_LEGEND_TITLE,
        title_fontsize=9.5,
    )
    leg2.get_title().set_fontweight("bold")
    fig.add_artist(leg1)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.canvas.draw()
    for p_idx, spatial in enumerate(panel_order):
        res = panel_results[spatial]
        disp_coords = [axes[p_idx].transData.transform(pt) for pt in res["leader_coords"]]
        _optimize_annotations(
            fig, axes[p_idx], res["annotations"], disp_coords, pad_px=5.0, n_iter=60
        )

    fig2_png = out_dir / "figure_2_leaderboard_progress_all_regimes.png"
    fig2_pdf = out_dir / "figure_2_leaderboard_progress_all_regimes.pdf"
    fig.savefig(fig2_png, dpi=300)
    fig.savefig(fig2_pdf)
    plt.close(fig)

    # Also render standalone single-panel figures (2a, 2b, 2c)
    single_paths = {
        "gauged": out_dir / "figure_2a_leaderboard_progress_gauged.png",
        "pub": out_dir / "figure_2b_leaderboard_progress_pub.png",
        "pur": out_dir / "figure_2c_leaderboard_progress_pur.png",
    }
    for p_idx, spatial in enumerate(panel_order):
        s_fig, s_ax = plt.subplots(figsize=(13.0, 6.2), dpi=300)
        s_fig.patch.set_facecolor("white")
        s_res = _plot_subpanel(
            s_fig,
            s_ax,
            panel_idx=p_idx,
            models=panel_models[spatial],
            title=panel_titles[spatial],
            ylim_hint=PANEL_YLIM_HINTS[spatial],
            year_range=year_range,
        )
        s_ax.set_xlabel(
            "Publication Year of Contributing Model / Benchmark Study",
            fontsize=10.0,
            fontweight="bold",
            color="#0F172A",
            labelpad=6,
        )
        panel_active_regimes = {
            str(m.get("input_regime"))
            for m in panel_models[spatial]
            if m.get("input_regime")
        }
        s_cat_handles, s_regime_handles = _build_legend_handles(panel_active_regimes)
        s_leg1 = s_fig.legend(
            handles=s_cat_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=6,
            frameon=True,
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            fontsize=8.2,
            title=_CATEGORY_LEGEND_TITLE,
            title_fontsize=9.0,
        )
        s_leg1.get_title().set_fontweight("bold")
        s_leg2 = s_fig.legend(
            handles=s_regime_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.885),
            ncol=max(1, len(s_regime_handles)),
            frameon=True,
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            fontsize=8.8,
            title=_REGIME_LEGEND_TITLE,
            title_fontsize=9.0,
        )
        s_leg2.get_title().set_fontweight("bold")
        s_fig.add_artist(s_leg1)
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.78])
        s_fig.canvas.draw()
        s_disp = [s_ax.transData.transform(pt) for pt in s_res["leader_coords"]]
        _optimize_annotations(s_fig, s_ax, s_res["annotations"], s_disp, pad_px=5.0, n_iter=60)
        s_fig.savefig(single_paths[spatial], dpi=300)
        plt.close(s_fig)

    # Also render per-table figures (`figure_<leaderboard_id>.png`) for the 10 valid leaderboards.
    # If a leaderboard has 0 plottable rows (e.g. `pur__freeform`, Table 12), generate a placeholder
    # figure stating "Placeholder figure for Table X" to be replaced when a model is added.
    table_label_map = {
        "gauged__single_camels": ("Table 3", "Gauged: Single CAMELS Forcing"),
        "gauged__multi_camels": ("Table 4", "Gauged: Multiple CAMELS Forcings"),
        "gauged__lagged_streamflow": ("Table 5", "Gauged: With Lagged Streamflow"),
        "gauged__freeform": ("Table 6", "Gauged: Freeform Forcings"),
        "pub__single_camels": ("Table 7", "12-Fold PUB: Single CAMELS Forcing"),
        "pub__multi_camels": ("Table 8", "12-Fold PUB: Multiple CAMELS Forcings"),
        "pub__lagged_streamflow": ("Table 9", "12-Fold PUB: With Lagged Streamflow"),
        "pub__freeform": ("Table 10", "12-Fold PUB: Freeform Forcings"),
        "pur__single_camels": ("Table 11", "18-Fold HUC-02 PUR: Single CAMELS Forcing"),
        "pur__multi_camels": ("Table 12", "18-Fold HUC-02 PUR: Multiple CAMELS Forcings"),
        "pur__lagged_streamflow": ("Table 13", "18-Fold HUC-02 PUR: With Lagged Streamflow"),
        "pur__freeform": ("Table 14", "18-Fold HUC-02 PUR: Freeform Forcings"),
    }
    per_table_paths: dict[str, Path] = {}
    for lbid, (tbl_num, tbl_title) in table_label_map.items():
        sp_key, f_key = lbid.split("__", 1)
        expected_regime = FORCING_REGIME_MAP[f_key]
        p_idx = panel_order.index(sp_key)
        lb_pts = [
            m
            for m in panel_models[sp_key]
            if m.get("leaderboard_id") == lbid and m.get("input_regime") == expected_regime
        ]
        out_path = out_dir / f"figure_{lbid}.png"
        per_table_paths[lbid] = out_path

        t_fig, t_ax = plt.subplots(figsize=(13.0, 5.4), dpi=220)
        t_fig.patch.set_facecolor("white")
        if lb_pts:
            t_res = _plot_subpanel(
                t_fig,
                t_ax,
                panel_idx=p_idx,
                models=lb_pts,
                title=f"{tbl_title} — Leaderboard Progress Over Time",
                ylim_hint=PANEL_YLIM_HINTS[sp_key],
                year_range=year_range,
            )
            t_ax.set_xlabel(
                "Publication Year of Contributing Model / Benchmark Study",
                fontsize=10.0,
                fontweight="bold",
                color="#0F172A",
                labelpad=6,
            )
            t_leg = t_fig.legend(
                handles=cat_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=6,
                frameon=True,
                facecolor="#F8FAFC",
                edgecolor="#CBD5E1",
                fontsize=8.2,
                title=_CATEGORY_LEGEND_TITLE,
                title_fontsize=9.0,
            )
            t_leg.get_title().set_fontweight("bold")
            plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
            t_fig.canvas.draw()
            t_disp = [t_ax.transData.transform(pt) for pt in t_res["leader_coords"]]
            _optimize_annotations(t_fig, t_ax, t_res["annotations"], t_disp, pad_px=5.0, n_iter=25)
        else:
            t_ax.set_facecolor("#F8FAFC")
            t_ax.set_xlim(2014.4, 2026.8)
            t_ax.set_ylim(0.30, 0.80)
            t_ax.grid(True, linestyle="--", linewidth=0.7, color="#CBD5E1", alpha=0.8)
            t_ax.set_title(
                f"{tbl_title} — Leaderboard Progress Over Time",
                loc="left",
                fontsize=12.0,
                fontweight="bold",
                color="#0F172A",
                pad=10,
            )
            t_ax.set_ylabel(
                "531-Basin Median NSE\n(1989–1999 Newman Period)",
                fontsize=10.0,
                fontweight="bold",
                color="#0F172A",
            )
            t_ax.set_xlabel(
                "Publication Year of Contributing Model / Benchmark Study",
                fontsize=10.0,
                fontweight="bold",
                color="#0F172A",
            )
            t_ax.text(
                0.5,
                0.52,
                f"No entries yet for {tbl_title}\n\n"
                "This plot will populate automatically when the first model is added.",
                transform=t_ax.transAxes,
                ha="center",
                va="center",
                fontsize=13.0,
                fontweight="bold",
                color="#334155",
                bbox={
                    "boxstyle": "round,pad=0.9",
                    "facecolor": "#FFFBEB",
                    "edgecolor": "#D97706",
                    "linewidth": 1.8,
                },
            )
            plt.tight_layout()
        t_fig.savefig(out_path, dpi=220)
        plt.close(t_fig)

    return {
        "png": fig2_png,
        "pdf": fig2_pdf,
        "gauged_png": single_paths["gauged"],
        "pub_png": single_paths["pub"],
        "pur_png": single_paths["pur"],
        "per_table_pngs": per_table_paths,
        "counts": {spatial: int(panel_results[spatial]["n_drawn"]) for spatial in panel_order},
        "excluded": {spatial: panel_excluded[spatial] for spatial in panel_order},
        "axis_limits": {
            spatial: {
                "ylim": [float(v) for v in panel_results[spatial]["ylim"]],
                "xlim": [float(v) for v in panel_results[spatial]["xlim"]],
            }
            for spatial in panel_order
        },
        "leaders": {spatial: panel_results[spatial]["leaders"] for spatial in panel_order},
    }


def generate_all_paper_figures(
    leaderboards_dir: str | Path = DEFAULT_LEADERBOARDS_DIR,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate all paper figures directly from the leaderboard JSON files in `leaderboards_dir`."""
    lb_dir = Path(leaderboards_dir)
    if output_dir is None:
        out_dir = (
            DEFAULT_FIGURES_DIR
            if lb_dir.resolve() == DEFAULT_LEADERBOARDS_DIR.resolve()
            else lb_dir.parent / "figures"
        )
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from leaderboards.update_leaderboard import load_all_leaderboards

    all_lbs = load_all_leaderboards(lb_dir)
    protocol = all_lbs.get("protocol", {})

    fig1_res = generate_figure_1_water_year_coverage(
        protocol=protocol,
        output_dir=out_dir,
    )
    fig2_res = generate_figure_2_leaderboard_progress(
        all_leaderboards=all_lbs,
        output_dir=out_dir,
    )

    def _rel_path_str(p: str | Path) -> str:
        resolved = Path(p).resolve()
        repo_resolved = REPO_ROOT.resolve()
        if resolved.is_relative_to(repo_resolved):
            return str(resolved.relative_to(repo_resolved))
        return str(p)

    manifest = {
        "source_leaderboards_dir": _rel_path_str(lb_dir),
        "output_figures_dir": _rel_path_str(out_dir),
        "total_leaderboard_entries": int(all_lbs.get("total_entries", 0)),
        "mip_study_year": MIP_STUDY_YEAR,
        "panel_entry_counts": fig2_res["counts"],
        "excluded_rows": fig2_res["excluded"],
        "panel_axis_limits": fig2_res["axis_limits"],
        "figure_1_source": "protocol.water_year_completeness",
        "figures": {
            "figure_1_png": _rel_path_str(fig1_res["png"]),
            "figure_1_pdf": _rel_path_str(fig1_res["pdf"]),
            "figure_2_png": _rel_path_str(fig2_res["png"]),
            "figure_2_pdf": _rel_path_str(fig2_res["pdf"]),
            "figure_2a_gauged_png": _rel_path_str(fig2_res["gauged_png"]),
            "figure_2b_pub_png": _rel_path_str(fig2_res["pub_png"]),
            "figure_2c_pur_png": _rel_path_str(fig2_res["pur_png"]),
            **{
                f"figure_{lbid}_png": _rel_path_str(p)
                for lbid, p in fig2_res.get("per_table_pngs", {}).items()
            },
        },
        "frontier_leaders": {
            spatial: [
                {
                    "year": ldr["year"],
                    "nse": ldr["nse"],
                    "model_category": ldr["model_category"],
                    "input_regime": ldr["input_regime"],
                    "model_name": ldr["model_name"],
                    "leaderboard_id": ldr["leaderboard_id"],
                    "entry_id": ldr["entry_id"],
                    "category": ldr["category"],
                    "provenance": ldr["provenance"],
                }
                for ldr in ldr_list
            ]
            for spatial, ldr_list in fig2_res["leaders"].items()
        },
    }
    manifest_path = out_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mip-generate-figures",
        description=(
            "Generate all CAMELS-US MIP paper figures directly from the "
            "leaderboard JSON files in leaderboards/."
        ),
    )
    parser.add_argument(
        "--leaderboards-dir",
        default=str(DEFAULT_LEADERBOARDS_DIR),
        help="Path to the leaderboards/ directory containing the JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated figures (default: repo figures/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    manifest = generate_all_paper_figures(
        leaderboards_dir=args.leaderboards_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["figures"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
