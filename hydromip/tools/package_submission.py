# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Package arbitrary hydrograph files into canonical HydrologyMIP CAMELS-US Zarr v2 archives.

Exposed via both ``hydromip package`` and ``hydromip-package``. Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from hydromip.io.ingest import (
    EVAL_END_DATE,
    EVAL_START_DATE,
    IngestionRecipe,
    extract_file_metadata,
    ingest_hydrograph,
    resolve_deterministic_recipe,
    write_canonical_zarr_v2,
)
from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths

_REGIME_PREFIX_MAP: dict[str, str] = {
    "gauged": "gauged",
    "pub": "pub",
    "pub_12fold": "pub",
    "pur": "pur",
    "pur_18huc02": "pur",
}


def infer_forcing_regime(forcing_label: str) -> str:
    """Map a human-readable forcing string onto one of the 4 canonical forcing regimes."""
    low = forcing_label.strip().lower()
    if not low or low in ("single_camels", "multi_camels", "lagged_streamflow", "freeform"):
        return low or "single_camels"
    if "lagged" in low or "autoregress" in low or "assimilation" in low or "q(t" in low:
        return "lagged_streamflow"
    if "freeform" in low or "era5" in low or "aorc" in low or "em-earth" in low or "external" in low:
        return "freeform"
    camels_hits = sum(1 for tok in ("daymet", "maurer", "nldas") if tok in low)
    if camels_hits >= 2 or "+" in low or "," in low:
        return "multi_camels"
    if camels_hits == 1:
        return "single_camels"
    return "freeform"


def resolve_leaderboard_id(
    regime: str,
    forcing: str,
    explicit_leaderboard: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(leaderboard_id, spatial_prefix, forcing_regime)``."""
    if explicit_leaderboard and explicit_leaderboard.strip():
        lb_id = explicit_leaderboard.strip()
        if "__" not in lb_id:
            raise ValueError(f"Invalid --leaderboard {lb_id!r}; expected '<spatial>__<forcing>'.")
        sp, fr = lb_id.split("__", 1)
        if sp not in ("gauged", "pub", "pur"):
            raise ValueError(f"Unsupported spatial prefix {sp!r} in --leaderboard {lb_id!r}.")
        if fr not in ("single_camels", "multi_camels", "lagged_streamflow", "freeform"):
            raise ValueError(f"Unsupported forcing regime {fr!r} in --leaderboard {lb_id!r}.")
        return lb_id, sp, fr

    raw_reg = regime.strip().lower()
    if raw_reg not in _REGIME_PREFIX_MAP:
        raise ValueError(
            f"Unsupported --regime {regime!r}. Choose one of: {sorted(_REGIME_PREFIX_MAP)}."
        )
    sp = _REGIME_PREFIX_MAP[raw_reg]
    fr = infer_forcing_regime(forcing)
    return f"{sp}__{fr}", sp, fr


def load_canonical_camels_coordinates() -> tuple[list[str], list[str], list[str]]:
    """Load sorted 531 basin IDs, 3652 Newman dates, and matching HUC-02 codes."""
    paths = resolve_mip_paths(DEFAULT_MIP_ID)
    basins_file = paths.splits_dir / "basins_531.txt"
    if not basins_file.is_file():
        raise FileNotFoundError(f"Canonical splits file not found: {basins_file}")
    basins = [
        line.strip().zfill(8)
        for line in basins_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    huc_map: dict[str, str] = {}
    sr_file = paths.splits_dir / "splits_and_regimes.json"
    if sr_file.is_file():
        sr_doc = json.loads(sr_file.read_text(encoding="utf-8"))
        for b_id, h_code in (sr_doc.get("basin_to_huc02") or {}).items():
            huc_map[str(b_id).zfill(8)] = str(h_code).zfill(2)

    huc_02 = [huc_map.get(b, "01") for b in basins]
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(EVAL_START_DATE, EVAL_END_DATE, freq="D")]
    return basins, dates, huc_02


def _zip_directory(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(src_dir.parent)
                zf.write(file_path, arcname=rel.as_posix())


def package_hydrograph_submission(
    input_path: Path,
    output_path: Path,
    *,
    forcing: str = "Daymet",
    regime: str = "gauged",
    leaderboard_id: str | None = None,
    model_id: str | None = None,
    model_name: str | None = None,
    family_id: str | None = None,
    category: str = "Contribution",
    code_url: str = "",
    archive_url: str = "",
    citation: str = "Community Contribution (2026)",
    recipe_override: IngestionRecipe | None = None,
    allow_positional_order: bool = False,
) -> dict[str, Any]:
    """Ingest a user hydrograph file and package a canonical CAMELS-US Zarr v2 store."""
    src = Path(input_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input prediction file does not exist: {src}")

    lb_id, spatial_prefix, forcing_regime = resolve_leaderboard_id(regime, forcing, leaderboard_id)
    basins, dates, huc_02 = load_canonical_camels_coordinates()

    meta = extract_file_metadata(src)
    if recipe_override is not None:
        recipe = recipe_override
    else:
        recipe, issues = resolve_deterministic_recipe(meta)
        if recipe is None:
            reasons = "; ".join(issues or ["Ambiguous schema."])
            raise ValueError(
                f"Could not infer deterministic schema for '{src.name}': {reasons}"
            )
    if allow_positional_order and not recipe.allow_positional_order:
        recipe = recipe.model_copy(update={"allow_positional_order": True})

    ingested = ingest_hydrograph(src, recipe, basins, dates)
    qsim = ingested["qsim"]
    qsim_members = ingested["qsim_members"]

    clean_id = (model_id or src.stem.replace(".zarr", "") or "model_submission").strip()
    spatial_protocol = {"gauged": "gauged", "pub": "pub_12fold", "pur": "pur_18huc02"}[spatial_prefix]
    zarr_meta: dict[str, Any] = {
        "entry_id": clean_id,
        "model_id": clean_id,
        "model_name": (model_name or clean_id).strip(),
        "family_id": (family_id or clean_id).strip(),
        "model_family": (model_name or clean_id).strip(),
        "provenance_category": category.strip() or "Contribution",
        "leaderboard_id": lb_id,
        "spatial_regime": spatial_prefix,
        "spatial_protocol": spatial_protocol,
        "forcing_regime": forcing_regime,
        "forcings_used": forcing.strip(),
        "forcing_products": forcing.strip(),
        "train_period": "1999-10-01/2008-09-30",
        "test_period": f"{EVAL_START_DATE}/{EVAL_END_DATE}",
        "training_period": "1999-10-01 to 2008-09-30",
        "evaluation_period": f"{EVAL_START_DATE} to {EVAL_END_DATE}",
        "units": "mm/day",
        "code_url": code_url.strip() or "Closed-Source",
        "archive_url": archive_url.strip() or "pending-zenodo-doi",
        "citation": citation.strip(),
    }

    out = Path(output_path)
    write_zip = out.name.lower().endswith(".zip")
    if write_zip:
        with tempfile.TemporaryDirectory(prefix="hydromip_pkg_") as tmp_str:
            stage_zarr = Path(tmp_str) / f"{clean_id}.zarr"
            write_canonical_zarr_v2(
                qsim,
                basins,
                stage_zarr,
                zarr_meta,
                qsim_members=qsim_members,
                huc_02=huc_02,
                dates=dates,
            )
            _zip_directory(stage_zarr, out)
    else:
        write_canonical_zarr_v2(
            qsim,
            basins,
            out,
            zarr_meta,
            qsim_members=qsim_members,
            huc_02=huc_02,
            dates=dates,
        )

    return {
        "status": "ok",
        "output_path": str(out),
        "format": "zarr.zip" if write_zip else "zarr",
        "leaderboard_id": lb_id,
        "shape": list(qsim.shape),
        "num_members": int(qsim_members.shape[0]) if qsim_members is not None else 1,
        "notes": list(ingested.get("notes") or []),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``hydromip package`` / ``hydromip-package``."""
    parser = argparse.ArgumentParser(
        prog="hydromip package",
        description="Package daily streamflow predictions into a canonical HydrologyMIP Zarr v2 archive.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .nc/.zarr/.zarr.zip/.csv/.parquet/.npz file")
    parser.add_argument("--output", required=True, type=Path, help="Output .zarr.zip or .zarr directory path")
    parser.add_argument("--forcing", default="Daymet", help="Forcing product declaration (default: Daymet)")
    parser.add_argument(
        "--regime",
        default="gauged",
        choices=("gauged", "pub", "pur", "pub_12fold", "pur_18huc02"),
        help="Spatial generalization regime (default: gauged)",
    )
    parser.add_argument("--leaderboard", default=None, help="Explicit canonical leaderboard ID override")
    parser.add_argument("--model-id", default=None, help="Canonical model/entry identifier")
    parser.add_argument("--model-name", default=None, help="Human-readable model display name")
    parser.add_argument("--family-id", default=None, help="Model family identifier")
    parser.add_argument("--category", default="Contribution", help="Provenance category (default: Contribution)")
    parser.add_argument("--code-url", default="", help="Public code repository URL")
    parser.add_argument("--archive-url", default="", help="Permanent archive DOI/URL")
    parser.add_argument("--citation", default="Community Contribution (2026)", help="Canonical citation")
    parser.add_argument("--recipe-json", type=Path, default=None, help="Optional explicit IngestionRecipe JSON file")
    parser.add_argument(
        "--allow-positional-order",
        action="store_true",
        help="Allow canonical 531x3652 positional alignment when coordinates lack explicit labels",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    recipe_override = None
    if args.recipe_json is not None:
        recipe_override = IngestionRecipe.model_validate(
            json.loads(args.recipe_json.read_text(encoding="utf-8"))
        )

    summary = package_hydrograph_submission(
        args.input,
        args.output,
        forcing=args.forcing,
        regime=args.regime,
        leaderboard_id=args.leaderboard,
        model_id=args.model_id,
        model_name=args.model_name,
        family_id=args.family_id,
        category=args.category,
        code_url=args.code_url,
        archive_url=args.archive_url,
        citation=args.citation,
        recipe_override=recipe_override,
        allow_positional_order=args.allow_positional_order,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
