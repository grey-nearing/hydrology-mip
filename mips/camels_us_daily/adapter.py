# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapter binding the ``camels_us_daily`` MIP splits, protocol, and data to ``hydromip``.

Zero ``try``/``except`` blocks. All split invariants are verified strictly via
``hydromip.splits.validate_cv_folds``.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydromip.paths import resolve_mip_paths
from hydromip.protocol import MipProtocol, get_mip
from hydromip.splits import CvFold, load_cv_splits_json, validate_cv_folds

MIP_ID: str = "camels_us_daily"


def load_protocol(repo_root: str | Path | None = None) -> MipProtocol:
    """Load the validated ``MipProtocol`` for ``camels_us_daily``."""
    return get_mip(mip_id=MIP_ID, repo_root=repo_root)


def load_canonical_basins(splits_dir: str | Path | None = None) -> list[str]:
    """Load and validate the 531 canonical sorted 8-digit USGS basin IDs."""
    sdir = (
        Path(splits_dir)
        if splits_dir is not None
        else resolve_mip_paths(MIP_ID).splits_dir
    )
    basins_file = sdir / "basins_531.txt"
    if not basins_file.is_file():
        raise FileNotFoundError(f"Canonical CAMELS-US basin file not found: {basins_file}")
    basins = [
        line.strip().zfill(8)
        for line in basins_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(basins) != 531 or len(set(basins)) != 531:
        raise ValueError(
            f"Expected 531 unique basins in {basins_file}, found {len(basins)} ({len(set(basins))} unique)."
        )
    if basins != sorted(basins):
        raise ValueError(f"Canonical basin list in {basins_file} must be lexicographically sorted.")
    return basins


def load_basin_to_huc02(splits_dir: str | Path | None = None) -> dict[str, str]:
    """Load the 531-basin USGS HUC-02 region mapping from ``splits_and_regimes.json``."""
    sdir = (
        Path(splits_dir)
        if splits_dir is not None
        else resolve_mip_paths(MIP_ID).splits_dir
    )
    spec_path = sdir / "splits_and_regimes.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Required splits spec not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if "basin_to_huc02" not in spec or not isinstance(spec["basin_to_huc02"], dict):
        raise KeyError(f"{spec_path} missing 'basin_to_huc02' mapping.")
    return {str(k).zfill(8): str(v).zfill(2) for k, v in spec["basin_to_huc02"].items()}


def load_splits_for_regime(
    spatial_regime: str,
    splits_dir: str | Path | None = None,
) -> list[CvFold]:
    """Load and verify canonical CV folds for ``gauged``, ``pub_12fold``, or ``pur_18huc02``."""
    sdir = (
        Path(splits_dir)
        if splits_dir is not None
        else resolve_mip_paths(MIP_ID).splits_dir
    )
    basins = load_canonical_basins(sdir)
    if spatial_regime == "gauged":
        folds = [
            CvFold(
                fold_id="gauged",
                fold_index=0,
                train_entities=tuple(basins),
                test_entities=tuple(basins),
            )
        ]
        validate_cv_folds(
            folds,
            basins,
            require_disjoint_train_test=False,
            require_exact_test_partition=True,
        )
        return folds

    if spatial_regime == "pub_12fold":
        return load_cv_splits_json(
            sdir / "pub_12fold_random_splits.json",
            basins,
            expected_num_folds=12,
            require_disjoint_train_test=True,
            require_exact_test_partition=True,
        )

    if spatial_regime == "pur_18huc02":
        basin_to_huc02 = load_basin_to_huc02(sdir)
        return load_cv_splits_json(
            sdir / "pur_18fold_huc02_splits.json",
            basins,
            expected_num_folds=18,
            require_disjoint_train_test=True,
            require_exact_test_partition=True,
            entity_to_group=basin_to_huc02,
        )

    raise ValueError(
        f"Unsupported spatial_regime {spatial_regime!r} for {MIP_ID}; "
        "expected 'gauged', 'pub_12fold', or 'pur_18huc02'."
    )
