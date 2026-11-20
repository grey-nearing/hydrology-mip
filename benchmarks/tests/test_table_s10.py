# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Supplement Table S10 planned paired model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.compute_table_s10 import (
    PLANNED_COMPARISONS,
    benjamini_hochberg_fdr,
    compute_table_s10,
)
from benchmarks.compute_table_s10 import main as table_s10_cli_main


def test_benjamini_hochberg_fdr_monotonic_and_bounded() -> None:
    q = benjamini_hochberg_fdr([0.001, 0.04, 0.02, 0.50])
    assert q.shape == (4,)
    assert q[0] <= q[2] <= q[1] <= q[3] <= 1.0
    assert abs(float(q[0]) - 0.004) < 1e-12


def test_compute_table_s10_all_25_comparisons_and_cli_json(tmp_path: Path) -> None:
    results = compute_table_s10(n_bootstrap=200, seed=42)
    assert len(results) == len(PLANNED_COMPARISONS) == 25
    for row in results:
        assert row["n"] == 531
        assert row["n_huc18"] == 18
        assert row["ci_lo"] <= row["ci_hi"]
        assert 0.0 <= row["p_basin"] <= row["q_basin"] <= 1.0
        assert 0.0 <= row["p_huc18"] <= row["q_huc18"] <= 1.0
        assert 0.0 <= row["p_tost_001"] <= 1.0

    out_json = tmp_path / "table_s10.json"
    rc = table_s10_cli_main(["--n-bootstrap", "50", "--out", str(out_json)])
    assert rc == 0
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(written) == 25
