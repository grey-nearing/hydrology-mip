# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-MIP leaderboard and figure directory resolution helpers.

Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

from pathlib import Path

from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths
from hydromip.protocol import MipProtocol, get_mip


def get_mip_leaderboards_dir(
    mip_id: str = DEFAULT_MIP_ID,
    repo_root: str | Path | None = None,
) -> Path:
    """Return the canonical ``leaderboards/`` directory for ``mip_id``."""
    return resolve_mip_paths(mip_id=mip_id, repo_root=repo_root).leaderboards_dir


def get_mip_figures_dir(
    mip_id: str = DEFAULT_MIP_ID,
    repo_root: str | Path | None = None,
) -> Path:
    """Return the canonical ``figures/`` directory for ``mip_id``."""
    return resolve_mip_paths(mip_id=mip_id, repo_root=repo_root).figures_dir


def get_mip_leaderboard_ids(
    mip_id: str = DEFAULT_MIP_ID,
    repo_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the ordered tuple of canonical leaderboard IDs declared by ``mip_id``."""
    proto: MipProtocol = get_mip(mip_id=mip_id, repo_root=repo_root)
    return proto.leaderboard_ids
