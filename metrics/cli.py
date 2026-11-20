"""Command-line interface (`mip-score-model`) for scoring one model and leaderboard."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from hydromip.paths import DEFAULT_MIP_ID
from hydromip.protocol import get_mip
from metrics.evaluator import (
    CANONICAL_LEADERBOARD_IDS,
    DEFAULT_BUCKET_URI,
    calculate_model_leaderboard_metrics,
)

#: The cloud Zarr evaluator in `metrics/evaluator.py` implements the CAMELS-US Daily
#: protocol (531 basins x 3,652 days); other registered MIPs need their own evaluator.
SUPPORTED_MIP_IDS: tuple[str, ...] = (DEFAULT_MIP_ID,)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for `mip-score-model`."""
    parser = argparse.ArgumentParser(
        prog="mip-score-model",
        description=(
            "Calculate standardized HydrologyMIP metrics for one model and one "
            "leaderboard directly from cloud Zarr hydrographs."
        ),
    )
    parser.add_argument(
        "--mip",
        default=DEFAULT_MIP_ID,
        type=str,
        help=f"Registered MIP identifier under mips/ (default: {DEFAULT_MIP_ID}).",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        type=str,
        help="Model identifier (<model_id>) in hydrographs/<leaderboard_id>/<model_id>.zarr.",
    )
    parser.add_argument(
        "--leaderboard-id",
        required=True,
        type=str,
        choices=list(CANONICAL_LEADERBOARD_IDS),
        help="Canonical leaderboard identifier (<spatial_regime>__<forcing_regime>).",
    )
    parser.add_argument(
        "--bucket-uri",
        default=None,
        type=str,
        help=(
            "Root URI of the Zarr bucket (default: the MIP protocol's storage.bucket_uri, "
            f"i.e. {DEFAULT_BUCKET_URI} for {DEFAULT_MIP_ID})."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        type=str,
        help="Optional path to save the full JSON metric dictionary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the `mip-score-model` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mip not in SUPPORTED_MIP_IDS:
        parser.error(
            f"--mip {args.mip!r} is not supported by this evaluator; supported: "
            f"{list(SUPPORTED_MIP_IDS)}."
        )
    protocol = get_mip(args.mip)
    if args.leaderboard_id not in protocol.leaderboard_ids:
        parser.error(
            f"--leaderboard-id {args.leaderboard_id!r} is not declared by MIP {args.mip!r}."
        )
    if args.bucket_uri is not None:
        bucket_uri = args.bucket_uri
    elif "bucket_uri" in protocol.storage:
        bucket_uri = str(protocol.storage["bucket_uri"])
    else:
        parser.error(
            f"MIP {args.mip!r} declares no storage.bucket_uri; pass --bucket-uri explicitly."
        )

    result = calculate_model_leaderboard_metrics(
        model_id=args.model_id,
        leaderboard_id=args.leaderboard_id,
        bucket_uri=bucket_uri,
    )

    if args.output_json is not None:
        output_path = Path(args.output_json)
        if output_path.parent != Path(""):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    cli_summary = {
        "model_id": result["model_id"],
        "leaderboard_id": result["leaderboard_id"],
        "hydrograph_uri": result["hydrograph_uri"],
        "summary": result["summary"],
        "regional_huc02": result["regional_huc02"],
    }
    print(json.dumps(cli_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
