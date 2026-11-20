# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Zero-parameter Naive Lag-1 Persistence lower benchmark for lagged-streamflow tracks."""

from models.naive_lag1_persistence.model import predict_lag1_persistence
from models.naive_lag1_persistence.runner import run_fold

__all__ = ["predict_lag1_persistence", "run_fold"]
