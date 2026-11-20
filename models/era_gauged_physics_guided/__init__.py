# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""ERA Physics-Guided Gauged winning model package (Node 187 — PhysicsHybridEALSTM / d-EA-LSTM)."""

from models.era_gauged_physics_guided.model import DifferentiableSnowPack, PhysicsHybridEALSTM
from models.era_gauged_physics_guided.runner import run_fold

__all__ = ["DifferentiableSnowPack", "PhysicsHybridEALSTM", "run_fold"]
