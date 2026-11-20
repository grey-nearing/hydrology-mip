# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Li et al. (2025) (LSTM + dHBV 1.1p) Multi-Model Hybrid Ensemble architecture."""

from __future__ import annotations

import numpy as np


class Li2025LstmDhbvEnsemble:
    """Equal-weight multi-model ensemble combining CudaLSTM and dHBV 1.1p hydrographs (Li et al., 2025)."""

    @staticmethod
    def combine_hydrographs(q_lstm: np.ndarray, q_dhbv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(qsim_mean, qsim_members)` from `(n_basins, n_days)` LSTM and dHBV 1.1p predictions."""
        q_l = np.asarray(q_lstm, dtype=np.float32)
        q_d = np.asarray(q_dhbv, dtype=np.float32)
        if q_l.shape != q_d.shape:
            raise ValueError(f"Mismatched component shapes: {q_l.shape} vs {q_d.shape}")
        members = np.stack([q_l, q_d], axis=0).astype(np.float32)
        mean = (0.5 * (q_l.astype(np.float64) + q_d.astype(np.float64))).astype(np.float32)
        return mean, members
