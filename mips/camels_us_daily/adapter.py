# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapter binding the ``camels_us_daily`` MIP splits, protocol, and data to ``hydromip``.

Split files and fold counts come from ``mips/camels_us_daily/protocol.json``
(``spatial_regimes[*].split_file`` / ``num_folds``), so the protocol is the single source
of truth. Every regime is verified strictly via ``hydromip.splits.validate_cv_folds``:

* ``gauged``: one fold whose train and test sets are the full 531-basin cohort.
* ``pub_12fold``: disjoint train/test, train + test == 531 in every fold, and the test sets
  partition the 531 basins exactly.
* ``pur_18huc02``: as PUB, plus every fold holds out exactly one USGS HUC-02 region that
  never appears in its training set.

Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydromip.paths import resolve_mip_paths
from hydromip.protocol import MipProtocol, get_mip
from hydromip.splits import CvFold, load_cv_splits_json, validate_cv_folds

MIP_ID: str = "camels_us_daily"
NUM_CANONICAL_BASINS: int = 531
#: Spatial regimes whose folds must be leave-one-HUC02-region-out.
HUC02_GROUPED_REGIMES: frozenset[str] = frozenset({"pur_18huc02"})


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
    if len(basins) != NUM_CANONICAL_BASINS or len(set(basins)) != NUM_CANONICAL_BASINS:
        raise ValueError(
            f"Expected {NUM_CANONICAL_BASINS} unique basins in {basins_file}, found "
            f"{len(basins)} ({len(set(basins))} unique)."
        )
    if basins != sorted(basins):
        raise ValueError(f"Canonical basin list in {basins_file} must be lexicographically sorted.")
    malformed = [b for b in basins if len(b) != 8 or not b.isdigit()]
    if malformed:
        raise ValueError(f"Non 8-digit basin IDs in {basins_file}: {malformed[:5]}")
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
    mapping = {str(k).zfill(8): str(v).zfill(2) for k, v in spec["basin_to_huc02"].items()}
    canonical = set(load_canonical_basins(sdir))
    if set(mapping) != canonical:
        raise ValueError(f"'basin_to_huc02' keys in {spec_path} do not equal the canonical basins.")
    return mapping


def load_splits_for_regime(
    spatial_regime: str,
    splits_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[CvFold]:
    """Load and verify the canonical CV folds of one declared spatial regime.

    Args:
        spatial_regime: A spatial regime ``id`` declared in ``protocol.json``
            (``gauged``, ``pub_12fold``, or ``pur_18huc02``).
        splits_dir: Optional override of the splits directory (defaults to the MIP's).
        repo_root: Optional repository root used to load the protocol.
    """
    protocol = load_protocol(repo_root)
    regime = protocol.get_spatial_regime(spatial_regime)
    sdir = Path(splits_dir) if splits_dir is not None else protocol.paths.splits_dir
    basins = load_canonical_basins(sdir)

    if regime.split_file is None:
        if regime.num_folds != 1:
            raise ValueError(
                f"Spatial regime {regime.id!r} declares {regime.num_folds} folds without a split file."
            )
        folds = [
            CvFold(
                fold_id=regime.id,
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
            require_train_test_union_equals_cohort=True,
        )
        return folds

    grouped = regime.id in HUC02_GROUPED_REGIMES
    return load_cv_splits_json(
        sdir / regime.split_file,
        basins,
        expected_num_folds=regime.num_folds,
        require_disjoint_train_test=True,
        require_exact_test_partition=True,
        entity_to_group=load_basin_to_huc02(sdir) if grouped else None,
        require_train_test_union_equals_cohort=True,
        require_single_test_group=grouped,
    )
