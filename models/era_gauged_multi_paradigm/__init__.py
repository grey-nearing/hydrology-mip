# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""ERA Gauged Multi-Paradigm (Newman CV) model package."""

from models.era_gauged_multi_paradigm.model import (
    CausalAtrousGRU,
    CausalConvGRULSTM,
    DualPathwayFiLMGRULSTM,
    EntityAwareHydroLSTM,
    MultiScaleFiLMEALSTM,
    ResidualAtrousGRU,
)
from models.era_gauged_multi_paradigm.runner import RUNNER_DEFAULTS, run_fold

__all__ = [
    "CausalAtrousGRU",
    "CausalConvGRULSTM",
    "DualPathwayFiLMGRULSTM",
    "EntityAwareHydroLSTM",
    "MultiScaleFiLMEALSTM",
    "ResidualAtrousGRU",
    "RUNNER_DEFAULTS",
    "run_fold",
]
