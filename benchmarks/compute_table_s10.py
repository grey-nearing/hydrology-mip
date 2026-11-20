# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce Supplement Table S10 planned paired comparisons across 531 basins and 18 HUC-02 regions.

Zero ``try``/``except`` blocks. Runs two-sided paired Wilcoxon signed-rank tests across all
531 individual basins and across the 18 USGS HUC-02 regional medians, Benjamini-Hochberg
FDR adjustment across all planned comparisons, 5,000-resample HUC-02 cluster-bootstrap
95% confidence intervals, and two one-sided tests (TOST) for practical equivalence at a
+/-0.010 NSE margin across the 18 regional medians.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from hydromip.paths import DEFAULT_MIP_ID, resolve_mip_paths

DEFAULT_MIP_DIR: Path = resolve_mip_paths(DEFAULT_MIP_ID).root

# 25 planned comparisons centralized in Supplement Table S10 (Paper Section 7).
PLANNED_COMPARISONS: tuple[tuple[str, str, tuple[str, str], tuple[str, str]], ...] = (
    (
        "Finding 1: Forcing vs. Architecture",
        "Gauged 3F LSTM (10-ens) vs. Daymet LSTM (10-ens)",
        ("S2", "t4_05"),
        ("S1", "t3_02"),
    ),
    (
        "Finding 1: Forcing vs. Architecture",
        "Gauged 3F SAC-SMA (3-ens) vs. Daymet SAC-SMA (single)",
        ("S2", "sacsma_gauged_multi_3f"),
        ("S1", "sacsma_gauged_single_daymet"),
    ),
    (
        "Finding 1: Forcing vs. Architecture",
        "PUR 3F LSTM (8-ens) vs. Daymet LSTM (10-ens)",
        ("S8", "k21_lstm_pur_3f"),
        ("S7", "k21_lstm_pur_daymet"),
    ),
    (
        "Finding 1: Forcing vs. Architecture",
        "PUR 3F SAC-SMA (3-ens) vs. Daymet SAC-SMA (single)",
        ("S8", "sacsma_pur_multi_3f"),
        ("S7", "sacsma_pur_single_daymet"),
    ),
    (
        "Finding 1: Forcing vs. Architecture",
        "Gauged 3F ERA-Multi-Paradigm (40-net) vs. LSTM (10-ens)",
        ("S2", "era_gauged_multi_paradigm"),
        ("S2", "t4_05"),
    ),
    (
        "Finding 1: Forcing vs. Architecture",
        "Gauged 3F ERA-Multi-Paradigm (40-net) vs. LSTM + δHBV 1.1p",
        ("S2", "era_gauged_multi_paradigm"),
        ("S2", "t4_06"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged Daymet LSTM (10-ens) vs. δHBV 1.1p (5-ens)",
        ("S1", "t3_02"),
        ("S1", "t3_18"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged Daymet δHBV 1.1p (single) vs. LSTM (single)",
        ("S1", "t3_19"),
        ("S1", "t3_08"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged Daymet MLP + 16-HBV (3-ens) vs. SAC-SMA (single)",
        ("S1", "poudel_mlp_hbv_daymet"),
        ("S1", "sacsma_gauged_single_daymet"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged MLP + 16-HBV (3-ens) vs. HBV-upper (100-ens)",
        ("S1", "poudel_mlp_hbv_daymet"),
        ("S1", "t3_39"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged Daymet MLP + 16-HBV (single) vs. δHBV 1.0 (single)",
        ("S1", "poudel_mlp_hbv_daymet_single"),
        ("S1", "t3_33"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "PUB Daymet δHBV 1.0 (5-ens) vs. MLP + 16-HBV (3-ens)",
        ("S4", "t7_10"),
        ("S4", "poudel_mlp_hbv_pub_daymet"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "PUR Daymet δHBV 1.0 (5-ens) vs. MLP + 16-HBV (3-ens)",
        ("S7", "t11_06"),
        ("S7", "poudel_mlp_hbv_pur_daymet"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged Daymet δHBV 1.1p (5-ens) vs. δHBV 1.0 (5-ens)",
        ("S1", "t3_18"),
        ("S1", "t3_29"),
    ),
    (
        "Finding 2: Hybrid & Structural Ensembling",
        "Gauged Daymet δHBV 1.1p (5-ens) vs. MLP + 16-HBV (3-ens)",
        ("S1", "t3_18"),
        ("S1", "poudel_mlp_hbv_daymet"),
    ),
    (
        "Finding 3: Transformers & SSMs",
        "Gauged Maurer TFT (single) vs. Maurer LSTM (single)",
        ("S1", "tft_gauged_maurer"),
        ("S1", "t3_20"),
    ),
    (
        "Finding 3: Transformers & SSMs",
        "Gauged Daymet Mamba-3 (single) vs. Daymet LSTM (single)",
        ("S1", "shahriar2026_mamba3_daymet"),
        ("S1", "t3_08"),
    ),
    (
        "Finding 3: Transformers & SSMs",
        "Gauged Daymet RR-Former (3-ens, 15d Q) vs. NLDAS DI(1)-LSTM (6-ens, 1d Q)",
        ("S3", "yin2022_rrformer_daymet"),
        ("S3", "feng2020_di_lstm_nldas"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "3F LSTM: 12-Fold PUB (8-ens) vs. Gauged (10-ens)",
        ("S5", "k21_lstm_pub_3f"),
        ("S2", "t4_05"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "3F LSTM: 18-Fold PUR (8-ens) vs. 12-Fold PUB (8-ens)",
        ("S8", "k21_lstm_pur_3f"),
        ("S5", "k21_lstm_pub_3f"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "PUR Daymet δHBV 1.1p (5-ens) vs. Daymet LSTM (10-ens)",
        ("S7", "t11_04"),
        ("S7", "k21_lstm_pur_daymet"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "PUB 3F LSTM (8-ens) vs. ERA-Physics-Guided-PUB (10-ens)",
        ("S5", "k21_lstm_pub_3f"),
        ("S5", "era_pub_physics_guided"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "PUR 3F ERA-Unguided-PUR (10-ens) vs. LSTM (8-ens)",
        ("S8", "era_pur_minimal_guidance"),
        ("S8", "k21_lstm_pur_3f"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "PUR NLDAS DI(1)-LSTM (3-ens, 1d Q) vs. Unlagged NLDAS LSTM (10-ens)",
        ("S9", "feng2020_di_lstm_pur_nldas"),
        ("S7", "k21_lstm_pur_nldas"),
    ),
    (
        "Finding 4: Spatial Generalization & Lagged Q",
        "PUR NLDAS DI(1)-LSTM (3-ens, 1d Q) vs. Gauged Daymet SAC-SMA (single)",
        ("S9", "feng2020_di_lstm_pur_nldas"),
        ("S1", "sacsma_gauged_single_daymet"),
    ),
)

TABLE_FILES: dict[str, str] = {
    "S1": "gauged__single_camels.json",
    "S2": "gauged__multi_camels.json",
    "S3": "gauged__lagged_streamflow.json",
    "S4": "pub__single_camels.json",
    "S5": "pub__multi_camels.json",
    "S6": "pub__lagged_streamflow.json",
    "S7": "pur__single_camels.json",
    "S8": "pur__multi_camels.json",
    "S9": "pur__lagged_streamflow.json",
}


def benjamini_hochberg_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Compute Benjamini-Hochberg adjusted q-values preserving input order."""
    m = len(pvals)
    if m == 0:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(arr)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank_rev, idx in enumerate(order[::-1]):
        rank = m - rank_rev
        val = min(prev, float(arr[idx]) * m / rank)
        q[idx] = val
        prev = val
    return q


def _load_basin_to_huc02(splits_dir: Path) -> dict[str, str]:
    """Load the canonical 531-basin to HUC-02 region mapping."""
    huc_raw = json.loads((splits_dir / "pur_18fold_huc02_splits.json").read_text(encoding="utf-8"))
    basin_to_huc: dict[str, str] = {}
    folds = huc_raw.get("folds") or huc_raw.get("splits") or huc_raw
    for k, v in folds.items():
        bl = v if isinstance(v, list) else (v.get("test_basins") or v.get("basins") or [])
        for b in bl:
            basin_to_huc[str(b).zfill(8)] = str(k)
    return basin_to_huc


def _load_canonical_basins_and_entries(mip_dir: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
    """Load canonical sorted 531 basin IDs and leaderboard entries indexed by `(table_code, entry_id)`."""
    basins = [
        b.strip().zfill(8)
        for b in (mip_dir / "splits" / "basins_531.txt").read_text(encoding="utf-8").splitlines()
        if b.strip()
    ]
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    lb_dir = mip_dir / "leaderboards"
    for code, fname in TABLE_FILES.items():
        doc = json.loads((lb_dir / fname).read_text(encoding="utf-8"))
        for row in doc.get("rows", doc.get("entries", [])):
            entries[(code, str(row["entry_id"]))] = row
    return basins, entries


def _extract_basin_nse_map(row: dict[str, Any], canonical_basins: Sequence[str]) -> dict[str, float]:
    """Normalize `per_basin_nse` (list aligned to 531 basins or dict keyed by gauge ID)."""
    pb = row.get("per_basin_nse")
    if isinstance(pb, dict):
        return {str(k).zfill(8): float(v) for k, v in pb.items() if v is not None}
    if isinstance(pb, list) and len(pb) == len(canonical_basins):
        return {
            bid: float(v)
            for bid, v in zip(canonical_basins, pb)
            if v is not None
        }
    raise ValueError(f"Row {row.get('entry_id')!r} lacks a valid per_basin_nse vector.")


def compute_table_s10(
    *,
    mip_dir: Path = DEFAULT_MIP_DIR,
    n_bootstrap: int = 5000,
    seed: int = 42,
    tost_margin: float = 0.010,
) -> list[dict[str, Any]]:
    """Compute all 25 planned paired comparisons for Supplement Table S10."""
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")

    basin_to_huc = _load_basin_to_huc02(mip_dir / "splits")
    canonical_basins, entries = _load_canonical_basins_and_entries(mip_dir)
    hucs = sorted(set(basin_to_huc.values()))
    rng = np.random.default_rng(seed)

    results: list[dict[str, Any]] = []
    for group, label, key_a, key_b in PLANNED_COMPARISONS:
        row_a, row_b = entries[key_a], entries[key_b]
        pb_a = _extract_basin_nse_map(row_a, canonical_basins)
        pb_b = _extract_basin_nse_map(row_b, canonical_basins)
        common = sorted(set(pb_a.keys()) & set(pb_b.keys()))
        if not common:
            raise ValueError(f"No common basins for comparison {label!r}.")

        diffs = np.asarray([pb_a[b] - pb_b[b] for b in common], dtype=np.float64)
        med_diff = float(np.median(diffs))
        w_stat, p_basin = stats.wilcoxon(diffs, alternative="two-sided")

        huc_diffs: dict[str, list[float]] = {}
        for b, d in zip(common, diffs):
            h = basin_to_huc[b]
            huc_diffs.setdefault(h, []).append(float(d))

        huc_arrays = [np.asarray(huc_diffs[h], dtype=np.float64) for h in hucs if h in huc_diffs]
        huc_meds = np.asarray([float(np.median(arr)) for arr in huc_arrays], dtype=np.float64)
        _, p_huc = stats.wilcoxon(huc_meds, alternative="two-sided")

        n_h = len(huc_arrays)
        boot_meds = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n_h, size=n_h)
            sample = np.concatenate([huc_arrays[j] for j in idx])
            boot_meds[i] = float(np.median(sample))
        ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])

        _, p_lo = stats.wilcoxon(huc_meds + tost_margin, alternative="greater")
        _, p_hi = stats.wilcoxon(huc_meds - tost_margin, alternative="less")
        p_tost = float(max(p_lo, p_hi))

        results.append(
            {
                "group": group,
                "comparison": label,
                "model_a": f"{key_a[0]}:{key_a[1]}",
                "model_b": f"{key_b[0]}:{key_b[1]}",
                "n": len(common),
                "n_huc18": n_h,
                "med_a": float(np.median([pb_a[b] for b in common])),
                "med_b": float(np.median([pb_b[b] for b in common])),
                "med_diff": med_diff,
                "w_basin": float(w_stat),
                "p_basin": float(p_basin),
                "p_huc18": float(p_huc),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "p_tost_001": p_tost,
            }
        )

    q_basin = benjamini_hochberg_fdr([r["p_basin"] for r in results])
    q_huc = benjamini_hochberg_fdr([r["p_huc18"] for r in results])
    for r, qb, qh in zip(results, q_basin, q_huc):
        r["q_basin"] = float(qb)
        r["q_huc18"] = float(qh)

    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for computing Supplement Table S10."""
    parser = argparse.ArgumentParser(description="Compute HydrologyMIP Supplement Table S10 paired comparisons.")
    parser.add_argument("--mip-dir", type=Path, default=DEFAULT_MIP_DIR)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    results = compute_table_s10(
        mip_dir=args.mip_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Computed {len(results)} comparisons for Table S10:")
    for r in results:
        print(
            f"  {r['comparison'][:56]:56s} | Δ={r['med_diff']:+.4f} | "
            f"p_531={r['p_basin']:.2e} | p_huc={r['p_huc18']:.4f} | "
            f"q_huc={r['q_huc18']:.4f} | CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] | TOST={r['p_tost_001']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
