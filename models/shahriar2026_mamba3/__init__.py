# Copyright 2026 Google LLC, Grey Nearing, and the CAMELS-US MIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shahriar (2026) Mamba-3 Static-Dynamic Adapter streamflow model for CAMELS-US MIP."""

from models.shahriar2026_mamba3.model import MambaStreamflowModelAdapter
from models.shahriar2026_mamba3.runner import DEFAULT_SEEDS, run_fold

__all__ = ["DEFAULT_SEEDS", "MambaStreamflowModelAdapter", "run_fold"]
