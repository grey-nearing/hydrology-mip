# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical repository and per-MIP path resolution for the multi-MIP HydroMIP monorepo.

Every MIP lives in its own self-contained subdirectory under ``mips/<mip_id>/`` with:

* ``protocol.json`` — machine-readable MIP specification (task, metrics, splits, leaderboards, storage)
* ``splits/`` — canonical entity lists and cross-validation fold specifications
* ``leaderboards/`` — per-leaderboard JSON tables, family registry, and contributions index
* ``figures/`` — generated manuscript and dashboard figures
* ``data/`` — dataset instructions and optional locally compiled bundles (gitignored)
* ``era_experiments/`` — optional MIP-specific experiment/hillclimbing notebooks

Zero ``try``/``except`` blocks. All path checks validate existence explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MIP_ID: str = "camels_us_daily"
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MIPS_DIR: Path = REPO_ROOT / "mips"
MODELS_DIR: Path = REPO_ROOT / "models"


@dataclass(frozen=True)
class MipPaths:
    """Resolved filesystem directories for one registered MIP under ``mips/<mip_id>/``."""

    mip_id: str
    root: Path
    protocol_file: Path
    splits_dir: Path
    leaderboards_dir: Path
    figures_dir: Path
    data_dir: Path
    era_experiments_dir: Path


def get_repo_root(repo_root: str | Path | None = None) -> Path:
    """Return the resolved repository root directory."""
    if repo_root is None:
        return REPO_ROOT
    return Path(repo_root).resolve()


def resolve_mip_paths(
    mip_id: str = DEFAULT_MIP_ID,
    repo_root: str | Path | None = None,
) -> MipPaths:
    """Return canonical directory paths for ``mips/<mip_id>/``.

    Does not require every optional directory (such as ``era_experiments/``) to exist,
    so both existing and newly scaffolded MIPs can resolve their canonical layout.
    """
    clean_id = str(mip_id).strip()
    if not clean_id or "/" in clean_id or "\\" in clean_id or clean_id in (".", ".."):
        raise ValueError(f"Invalid mip_id: {mip_id!r}")
    root = get_repo_root(repo_root)
    mip_root = root / "mips" / clean_id
    return MipPaths(
        mip_id=clean_id,
        root=mip_root,
        protocol_file=mip_root / "protocol.json",
        splits_dir=mip_root / "splits",
        leaderboards_dir=mip_root / "leaderboards",
        figures_dir=mip_root / "figures",
        data_dir=mip_root / "data",
        era_experiments_dir=mip_root / "era_experiments",
    )


def resolve_mip_subdir(
    data_root: str | Path,
    subdir: str,
    mip_id: str = DEFAULT_MIP_ID,
    marker_filename: str | None = None,
) -> Path:
    """Resolve ``subdir`` (e.g. ``leaderboards``, ``splits``, ``figures``, ``data``) under ``data_root``.

    Supports both:
    1. Multi-MIP monorepo layout: ``<data_root>/mips/<mip_id>/<subdir>``
    2. Isolated test-fixture / flat layout: ``<data_root>/<subdir>`` (or ``<data_root>`` itself
       when ``marker_filename`` is directly inside ``<data_root>``).
    """
    root = Path(data_root).resolve()
    nested = root / "mips" / mip_id / subdir
    if nested.exists():
        return nested
    flat = root / subdir
    if flat.exists():
        return flat
    if marker_filename is not None and (root / marker_filename).exists():
        return root
    raise FileNotFoundError(
        f"MIP subdirectory '{subdir}' for mip_id={mip_id!r} not found under {root} "
        f"(checked {nested} and {flat})."
    )
