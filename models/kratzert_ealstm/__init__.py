# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Kratzert et al. (2019) Entity-Aware LSTM (EA-LSTM) regional rainfall-runoff model."""

from models.kratzert_ealstm.model import EALSTM
from models.kratzert_ealstm.runner import run_fold

__all__ = ["EALSTM", "run_fold"]
