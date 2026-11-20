# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical repository and per-MIP path resolution for the multi-MIP HydrologyMIP monorepo.

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

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MIP_ID: str = "camels_us_daily"
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MIPS_DIR: Path = REPO_ROOT / "mips"
MODELS_DIR: Path = REPO_ROOT / "models"

#: Canonical per-MIP subdirectory names (the only names `resolve_mip_subdir` accepts).
MIP_SUBDIRS: tuple[str, ...] = ("splits", "leaderboards", "figures", "data", "era_experiments")
#: A MIP identifier is a single path component: alphanumerics, ``_`` and ``-`` only.
MIP_ID_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


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


def validate_mip_id(mip_id: str) -> str:
    """Return the stripped ``mip_id`` if it is a valid single path component, else raise."""
    clean_id = str(mip_id).strip()
    if MIP_ID_PATTERN.fullmatch(clean_id) is None:
        raise ValueError(
            f"Invalid mip_id: {mip_id!r} (expected ^[A-Za-z0-9][A-Za-z0-9_-]{{0,63}}$)."
        )
    return clean_id


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
    clean_id = validate_mip_id(mip_id)
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
    """Resolve ``subdir`` (``leaderboards``, ``splits``, ``figures``, ...) under ``data_root``.

    The layout is decided deterministically by ``data_root`` itself; there is no fallback
    from one layout to the other:

    1. Monorepo layout: if ``<data_root>/mips/`` exists (always true for a repository
       checkout or a deployed copy of it), the result is ``<data_root>/mips/<mip_id>/<subdir>``.
    2. Flat override layout: otherwise ``data_root`` is an override data directory for a
       single MIP (for example an isolated test fixture) and the result is
       ``<data_root>/<subdir>``. The flat layout is never applied to the repository root,
       so a missing per-MIP directory can never resolve to a code package such as the
       repository's ``leaderboards/`` Python package.

    Raises:
        ValueError: If ``mip_id`` or ``subdir`` is not a canonical identifier.
        FileNotFoundError: If the resolved directory, or ``marker_filename`` inside it,
            does not exist.
    """
    clean_id = validate_mip_id(mip_id)
    if subdir not in MIP_SUBDIRS:
        raise ValueError(f"Unknown MIP subdirectory {subdir!r}; expected one of {MIP_SUBDIRS}.")
    root = Path(data_root).resolve()
    if (root / "mips").is_dir():
        candidate = root / "mips" / clean_id / subdir
    elif root == REPO_ROOT.resolve():
        raise FileNotFoundError(
            f"Repository root {root} has no mips/ directory; refusing to resolve '{subdir}' "
            "against the repository root (it would alias a code package)."
        )
    else:
        candidate = root / subdir
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"MIP subdirectory '{subdir}' for mip_id={clean_id!r} not found: {candidate}"
        )
    if marker_filename is not None and not (candidate / marker_filename).is_file():
        raise FileNotFoundError(
            f"Required file '{marker_filename}' not found in {candidate} "
            f"(mip_id={clean_id!r}, subdir='{subdir}')."
        )
    return candidate
