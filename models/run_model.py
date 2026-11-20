# Copyright 2026 Google LLC, Grey Nearing, Daniel Klotz, Patrick Ebel, Martin Gauch, and the HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Centralized deployment runner for the CAMELS-US Model Intercomparison Project (MIP).

This is the single entrypoint in `models/` that loads the canonical basin list
(`mips/<mip>/splits/basins_531.txt`), the canonical fold files (`pub_12fold_random_splits.json`,
`pur_18fold_huc02_splits.json` in the same directory),
and the canonical training/evaluation windows (`1999-10-01`..`2008-09-30` and `1989-10-01`..`1999-09-30`),
orchestrates fold execution (`gauged`, 12-fold `pub_12fold`, or 18-fold `pur_18huc02`) **once per random
seed**, stitches the out-of-fold predictions of every seed across all 531 basins, and writes one
consolidated Zarr v2 hydrograph store (`qsim` = seed mean, `qsim_members` = per-seed hydrographs) via
`models/zarr_writer.py`.

Model runner contract (`models/<model>/runner.py::run_fold`): called with ``seeds=(one_seed,)`` and must
return a float32 array of shape ``(len(test_basins), n_eval_days)`` in mm/day with rows ordered exactly
like ``test_basins``; a ``(1, len(test_basins), n_eval_days)`` array is also accepted.

Every registered model declares the forcing regimes and spatial protocols it may be run under
(``ModelRegistryEntry.allowed_forcing_regimes`` / ``allowed_spatial_regimes``). Models that read
observed discharge of the test basins as an input are only allowed on ``lagged_streamflow``
leaderboards, and models with per-basin embeddings only in the ``gauged`` regime.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hydromip.paths import DEFAULT_MIP_ID, REPO_ROOT, resolve_mip_paths
from models.camels_data import (
    DEFAULT_DATA_DIR,  # noqa: F401
    forcing_display_label,
    normalize_forcing_names,
    resolve_forcing_regime,
)
from models.zarr_writer import (
    CANONICAL_EVAL_END,
    CANONICAL_EVAL_START,
    CANONICAL_TRAIN_END,
    CANONICAL_TRAIN_START,
    EXPECTED_NUM_BASINS,
    EXPECTED_NUM_DAYS,
    VALID_FORCING_REGIMES,
    VALID_SPATIAL_REGIMES,
    build_hydrograph_dataset,
    default_store_uri,
    resolve_leaderboard_and_table_id,
    run_provenance_attrs,
    write_hydrograph_zarr,
)

DEFAULT_SPLITS_DIR: Path = resolve_mip_paths(DEFAULT_MIP_ID).splits_dir

SPLIT_FILES: dict[str, tuple[str, int]] = {
    "pub_12fold": ("pub_12fold_random_splits.json", 12),
    "pur_18huc02": ("pur_18fold_huc02_splits.json", 18),
}


@dataclass(frozen=True)
class FoldSpec:
    """Single spatial fold with disjoint training and test basin lists (identical in gauged mode)."""

    fold_id: str
    fold_index: int
    train_basins: tuple[str, ...]
    test_basins: tuple[str, ...]


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Metadata and module dispatch configuration for a registered MIP model."""

    key: str
    model_name: str
    model_family: str
    provenance_category: str
    runner_module: str
    code_url: str
    allowed_forcing_regimes: tuple[str, ...]
    allowed_spatial_regimes: tuple[str, ...]
    default_seeds: tuple[int, ...] = (123,)
    default_kwargs: dict[str, object] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "kratzert_lstm": ModelRegistryEntry(
        key="kratzert_lstm",
        model_name="CudaLSTM",
        model_family="LSTM",
        provenance_category="MIP Re-run",
        runner_module="models.kratzert_lstm.runner",
        code_url="https://github.com/neuralhydrology/neuralhydrology",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=tuple(range(101, 111)),
    ),
    "kratzert_ealstm": ModelRegistryEntry(
        key="kratzert_ealstm",
        model_name="EA-LSTM",
        model_family="EA-LSTM",
        provenance_category="MIP Re-run",
        runner_module="models.kratzert_ealstm.runner",
        code_url="https://github.com/neuralhydrology/neuralhydrology",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=tuple(range(201, 209)),
    ),
    "hoedt_mclstm": ModelRegistryEntry(
        key="hoedt_mclstm",
        model_name="MC-LSTM",
        model_family="MC-LSTM",
        provenance_category="MIP Re-run",
        runner_module="models.hoedt_mclstm.runner",
        code_url="https://github.com/ml-jku/mc-lstm",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=tuple(range(301, 311)),
    ),
    "rasiya_koya_tft": ModelRegistryEntry(
        key="rasiya_koya_tft",
        model_name="Temporal Fusion Transformer",
        model_family="Transformer",
        provenance_category="MIP Re-run",
        runner_module="models.rasiya_koya_tft.runner",
        code_url="https://github.com/sinanrk/TFT_hydrology",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(123,),
    ),
    "poudel_mlp_hbv": ModelRegistryEntry(
        key="poudel_mlp_hbv",
        model_name="MLP-HBV (16-zone)",
        model_family="Differentiable Process-Based",
        provenance_category="MIP Re-run",
        runner_module="models.poudel_mlp_hbv.runner",
        code_url="https://github.com/snpoudel/diff-hydro",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(101, 102, 103),
    ),
    "feng_dhbv10": ModelRegistryEntry(
        key="feng_dhbv10",
        model_name="δHBV 1.0",
        model_family="Differentiable Process-Based",
        provenance_category="MIP Re-run",
        runner_module="models.feng_dhbv10.runner",
        code_url="https://github.com/mhpi/hydroDL",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(111, 222, 333, 444, 555),
    ),
    "song_dhbv11p": ModelRegistryEntry(
        key="song_dhbv11p",
        model_name="δHBV 1.1p (Dynamic Parameters)",
        model_family="Differentiable Process-Based",
        provenance_category="MIP Re-run",
        runner_module="models.song_dhbv11p.runner",
        code_url="https://github.com/mhpi/generic_deltamodel",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(111, 222, 333, 444, 555),
    ),
    "li2025_lstm_dhbv_ens": ModelRegistryEntry(
        key="li2025_lstm_dhbv_ens",
        model_name="LSTM + δHBV 1.1p Ensemble",
        model_family="Hybrid DL + Conceptual Ensemble",
        provenance_category="MIP Re-run",
        runner_module="models.li2025_lstm_dhbv_ens.runner",
        code_url="https://github.com/mhpi/generic_deltamodel",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(101, 102, 103, 104, 105),
    ),
    "newman_sacsma": ModelRegistryEntry(
        key="newman_sacsma",
        model_name="Snow-17 + SAC-SMA",
        model_family="Conceptual Process-Based",
        provenance_category="MIP Re-run",
        runner_module="models.newman_sacsma.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/newman_sacsma",
        allowed_forcing_regimes=("single_camels",),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(123,),
    ),
    "era_gauged_minimal_guidance": ModelRegistryEntry(
        key="era_gauged_minimal_guidance",
        model_name="ERA-Unguided-Gauged",
        model_family="ERA-Unguided-Gauged",
        provenance_category="Contribution",
        runner_module="models.era_gauged_minimal_guidance.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_gauged_minimal_guidance",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("gauged",),
        default_seeds=(42, 88, 2024, 101, 102, 103, 104, 105, 106, 107),
    ),
    "era_pur_minimal_guidance": ModelRegistryEntry(
        key="era_pur_minimal_guidance",
        model_name="ERA-Unguided-PUR",
        model_family="ERA-Unguided-PUR",
        provenance_category="Contribution",
        runner_module="models.era_pur_minimal_guidance.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_pur_minimal_guidance",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("pur_18huc02",),
        default_seeds=(42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    ),
    "era_pub_minimal_guidance": ModelRegistryEntry(
        key="era_pub_minimal_guidance",
        model_name="ERA-Unguided-PUB",
        model_family="ERA-Unguided-PUB",
        provenance_category="Contribution",
        runner_module="models.era_pub_minimal_guidance.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_pub_minimal_guidance",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("pub_12fold",),
        default_seeds=(42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    ),
    "era_gauged_physics_guided": ModelRegistryEntry(
        key="era_gauged_physics_guided",
        model_name="ERA-Physics-Guided-Gauged",
        model_family="ERA-Physics-Guided-Gauged",
        provenance_category="Contribution",
        runner_module="models.era_gauged_physics_guided.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_gauged_physics_guided",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("gauged",),
        default_seeds=(100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
    ),
    "era_pub_physics_guided": ModelRegistryEntry(
        key="era_pub_physics_guided",
        model_name="ERA-Physics-Guided-PUB",
        model_family="ERA-Physics-Guided-PUB",
        provenance_category="Contribution",
        runner_module="models.era_pub_physics_guided.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_pub_physics_guided",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("pub_12fold",),
        default_seeds=(100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
    ),
    "era_pur_physics_guided": ModelRegistryEntry(
        key="era_pur_physics_guided",
        model_name="ERA-Physics-Guided-PUR",
        model_family="ERA-Physics-Guided-PUR",
        provenance_category="Contribution",
        runner_module="models.era_pur_physics_guided.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_pur_physics_guided",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("pur_18huc02",),
        default_seeds=(100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
    ),
    "era_gauged_multi_paradigm": ModelRegistryEntry(
        key="era_gauged_multi_paradigm",
        model_name="ERA-Multi-Paradigm-Gauged",
        model_family="ERA-Multi-Paradigm-Gauged",
        provenance_category="Contribution",
        runner_module="models.era_gauged_multi_paradigm.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/era_gauged_multi_paradigm",
        allowed_forcing_regimes=("multi_camels",),
        allowed_spatial_regimes=("gauged",),
        default_seeds=(100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
    ),
    "feng2020_di_lstm": ModelRegistryEntry(
        key="feng2020_di_lstm",
        model_name="DI(1) Lagged-Discharge LSTM",
        model_family="Data-Integration LSTM",
        provenance_category="MIP Re-run",
        runner_module="models.feng2020_di_lstm.runner",
        code_url="https://github.com/mhpi/hydroDL",
        allowed_forcing_regimes=("lagged_streamflow",),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(101, 102, 103, 104, 105, 106),
    ),
    "shahriar2026_mamba3": ModelRegistryEntry(
        key="shahriar2026_mamba3",
        model_name="Mamba-3 Static-Dynamic Adapter",
        model_family="Mamba State Space Model",
        provenance_category="MIP Re-run",
        runner_module="models.shahriar2026_mamba3.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/shahriar2026_mamba3",
        allowed_forcing_regimes=("single_camels", "multi_camels"),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(111111,),
    ),
    "yin2022_rrformer": ModelRegistryEntry(
        key="yin2022_rrformer",
        model_name="RR-Former",
        model_family="Encoder-Decoder Transformer",
        provenance_category="MIP Re-run",
        runner_module="models.yin2022_rrformer.runner",
        code_url="https://github.com/iThronne/RR-Former",
        allowed_forcing_regimes=("lagged_streamflow",),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(601, 602, 603),
    ),
    "naive_lag1_persistence": ModelRegistryEntry(
        key="naive_lag1_persistence",
        model_name="Naive Lag-1 Persistence",
        model_family="Statistical Lower Benchmark",
        provenance_category="Contribution",
        runner_module="models.naive_lag1_persistence.runner",
        code_url="https://github.com/grey-nearing/camels-us-mip/tree/main/models/naive_lag1_persistence",
        allowed_forcing_regimes=("lagged_streamflow",),
        allowed_spatial_regimes=("gauged", "pub_12fold", "pur_18huc02"),
        default_seeds=(0,),
        default_kwargs={"cold_start_value": 0.0},
    ),
}


# =============================================================================
# Canonical splits
# =============================================================================


def load_canonical_basins(splits_dir: str | Path = DEFAULT_SPLITS_DIR) -> list[str]:
    """Load and validate the 531 canonical sorted 8-digit USGS basin IDs from `<splits_dir>/basins_531.txt`."""
    splits_path = Path(splits_dir)
    txt_path = splits_path / "basins_531.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Canonical basin list not found at {txt_path}")

    basins = [
        line.strip().zfill(8)
        for line in txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(basins) != EXPECTED_NUM_BASINS:
        raise ValueError(f"Expected {EXPECTED_NUM_BASINS} basins in {txt_path}, found {len(basins)}.")
    if len(set(basins)) != EXPECTED_NUM_BASINS:
        raise ValueError(f"Duplicate basin IDs found in {txt_path}.")
    if basins != sorted(basins):
        raise ValueError(f"Canonical basin list {txt_path} must be sorted in ascending order.")
    bad = [b for b in basins if len(b) != 8 or not b.isdigit()]
    if bad:
        raise ValueError(f"Non 8-digit basin IDs in {txt_path}: {bad[:5]}")
    return basins


def load_basin_to_huc02(splits_dir: str | Path = DEFAULT_SPLITS_DIR) -> dict[str, str]:
    """Load the canonical 531 basin -> 2-digit HUC-02 region mapping."""
    splits_path = Path(splits_dir)
    json_path = splits_path / "splits_and_regimes.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Split metadata file not found at {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if "basin_to_huc02" not in payload:
        raise KeyError(f"{json_path} has no 'basin_to_huc02' mapping.")
    raw_map = payload["basin_to_huc02"]
    mapping = {str(k).strip().zfill(8): str(v).strip().zfill(2) for k, v in raw_map.items()}
    if len(mapping) != EXPECTED_NUM_BASINS:
        raise ValueError(f"Expected {EXPECTED_NUM_BASINS} basins in basin_to_huc02, found {len(mapping)}.")
    canonical = set(load_canonical_basins(splits_dir))
    if set(mapping) != canonical:
        raise ValueError(f"basin_to_huc02 keys in {json_path} do not equal the canonical 531 basins.")
    return mapping


def load_spatial_folds(
    spatial_regime: str,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> list[FoldSpec]:
    """Load and validate fold definitions for `gauged`, `pub_12fold`, or `pur_18huc02`.

    Only the canonical full split files are read; there is deliberately no fallback to any other file.
    """
    if spatial_regime not in VALID_SPATIAL_REGIMES:
        raise ValueError(f"Invalid spatial_regime {spatial_regime!r}; must be one of {VALID_SPATIAL_REGIMES}.")

    canonical_basins = load_canonical_basins(splits_dir)
    canonical_set = set(canonical_basins)
    splits_path = Path(splits_dir)

    if spatial_regime == "gauged":
        return [
            FoldSpec(
                fold_id="gauged",
                fold_index=0,
                train_basins=tuple(canonical_basins),
                test_basins=tuple(canonical_basins),
            )
        ]

    file_name, expected_n_folds = SPLIT_FILES[spatial_regime]
    split_file = splits_path / file_name
    if not split_file.is_file():
        raise FileNotFoundError(f"Canonical split file for {spatial_regime!r} not found at {split_file}.")
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if "folds" not in payload:
        raise KeyError(f"{split_file} has no 'folds' section.")
    folds_dict = payload["folds"]

    folds: list[FoldSpec] = []
    for idx, (fold_id, fold_data) in enumerate(sorted(folds_dict.items())):
        for key in ("train_basins", "test_basins"):
            if key not in fold_data:
                raise KeyError(f"Fold {fold_id!r} in {split_file} is missing {key!r}.")
        tr = tuple(str(b).strip().zfill(8) for b in fold_data["train_basins"])
        te = tuple(str(b).strip().zfill(8) for b in fold_data["test_basins"])
        fold_index = int(fold_data["fold_index"]) if "fold_index" in fold_data else idx
        folds.append(FoldSpec(fold_id=str(fold_id), fold_index=fold_index, train_basins=tr, test_basins=te))

    if len(folds) != expected_n_folds:
        raise ValueError(f"Spatial regime {spatial_regime!r} requires {expected_n_folds} folds, got {len(folds)}.")
    if len({f.fold_index for f in folds}) != len(folds):
        raise ValueError(f"Duplicate fold_index values in {split_file}.")

    seen_test_basins: set[str] = set()
    for f in folds:
        tr_set = set(f.train_basins)
        te_set = set(f.test_basins)
        if len(tr_set) != len(f.train_basins) or len(te_set) != len(f.test_basins):
            raise ValueError(f"Fold {f.fold_id!r} lists a basin more than once.")
        if not te_set:
            raise ValueError(f"Fold {f.fold_id!r} has an empty test_basins list.")
        if not tr_set.isdisjoint(te_set):
            overlap = sorted(tr_set & te_set)
            raise ValueError(f"Fold {f.fold_id!r} has overlapping train and test basins (first: {overlap[0]}).")
        if (tr_set | te_set) != canonical_set:
            raise ValueError(f"Fold {f.fold_id!r} train + test union does not equal the canonical 531 basins.")
        if not seen_test_basins.isdisjoint(te_set):
            dup = sorted(seen_test_basins & te_set)
            raise ValueError(f"Fold {f.fold_id!r} test basins overlap with an earlier fold (first: {dup[0]}).")
        seen_test_basins.update(te_set)

    if seen_test_basins != canonical_set:
        missing = sorted(canonical_set - seen_test_basins)
        raise ValueError(
            f"Out-of-fold test basins across {spatial_regime!r} do not cover all 531 basins "
            f"(missing {len(missing)}, first: {missing[0]})."
        )

    if spatial_regime == "pur_18huc02":
        basin_to_huc02 = load_basin_to_huc02(splits_dir)
        for f in folds:
            test_hucs = {basin_to_huc02[b] for b in f.test_basins}
            train_hucs = {basin_to_huc02[b] for b in f.train_basins}
            if len(test_hucs) != 1 or not test_hucs.isdisjoint(train_hucs):
                raise ValueError(
                    f"PUR fold {f.fold_id!r} is not a pure leave-one-HUC02-out fold "
                    f"(test HUCs {sorted(test_hucs)})."
                )
    return folds


# =============================================================================
# Model dispatch
# =============================================================================


def resolve_model_entry(model_name: str) -> ModelRegistryEntry:
    """Resolve a model directory name in `MODEL_REGISTRY`."""
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Registered models: {sorted(MODEL_REGISTRY.keys())}.")
    return MODEL_REGISTRY[key]


def load_model_fold_runner(entry: ModelRegistryEntry) -> Callable[..., np.ndarray]:
    """Import and return `run_fold` from the model's `runner.py` module."""
    mod = importlib.import_module(entry.runner_module)
    runner_fn = getattr(mod, "run_fold", None)
    if runner_fn is None or not callable(runner_fn):
        raise ValueError(f"Module {entry.runner_module!r} does not define a callable 'run_fold'.")
    return runner_fn


# =============================================================================
# Periods, forcings, stitching
# =============================================================================


def validate_periods(train_period: tuple[str, str], eval_period: tuple[str, str]) -> None:
    """Raise unless both periods are well formed, disjoint, and the eval window has 3652 days."""
    tr_a, tr_b = pd.Timestamp(train_period[0]), pd.Timestamp(train_period[1])
    ev_a, ev_b = pd.Timestamp(eval_period[0]), pd.Timestamp(eval_period[1])
    if tr_b < tr_a:
        raise ValueError(f"train_period {train_period!r} ends before it starts.")
    if ev_b < ev_a:
        raise ValueError(f"eval_period {eval_period!r} ends before it starts.")
    if not (ev_b < tr_a or tr_b < ev_a):
        raise ValueError(
            f"train_period {train_period!r} and eval_period {eval_period!r} overlap; "
            "training and evaluation windows must be disjoint."
        )
    n_eval = len(pd.date_range(ev_a, ev_b, freq="D"))
    if n_eval != EXPECTED_NUM_DAYS:
        raise ValueError(f"eval_period {eval_period!r} spans {n_eval} days; expected {EXPECTED_NUM_DAYS}.")


def resolve_forcing_regime_for_run(forcings: Sequence[str], forcing_regime: str | None) -> str:
    """Derive the forcing regime from the product list, or validate an explicit override."""
    derived = resolve_forcing_regime(forcings)
    if forcing_regime is None:
        return derived
    if forcing_regime not in VALID_FORCING_REGIMES:
        raise ValueError(f"Invalid forcing_regime {forcing_regime!r}; must be one of {VALID_FORCING_REGIMES}.")
    if forcing_regime in ("single_camels", "multi_camels") and forcing_regime != derived:
        raise ValueError(
            f"forcing_regime {forcing_regime!r} contradicts the forcing list {list(forcings)} ({derived!r})."
        )
    return forcing_regime


def _coerce_fold_prediction(pred: object, fold: FoldSpec, expected_days: int) -> np.ndarray:
    """Validate one `run_fold` output for a single seed and return it as float32 (n_test, n_days)."""
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    expected_shape = (len(fold.test_basins), expected_days)
    if arr.shape != expected_shape:
        raise ValueError(f"Fold {fold.fold_id!r} returned shape {arr.shape}, expected {expected_shape}.")
    if not np.all(np.isfinite(arr)):
        nan_cnt = int(np.isnan(arr).sum())
        inf_cnt = int(np.isinf(arr).sum())
        raise ValueError(
            f"Fold {fold.fold_id!r} returned non-finite predictions ({nan_cnt} NaNs, {inf_cnt} Infs)."
        )
    return arr


def stitch_fold_predictions(
    fold_results: Sequence[tuple[FoldSpec, np.ndarray]],
    canonical_basins: Sequence[str],
    expected_days: int = EXPECTED_NUM_DAYS,
) -> np.ndarray:
    """Verify 100% disjoint basin coverage across folds and stitch into `(531, expected_days)` float32."""
    basin_list = [str(b).strip().zfill(8) for b in canonical_basins]
    if len(basin_list) != EXPECTED_NUM_BASINS or basin_list != sorted(basin_list):
        raise ValueError("canonical_basins must be the 531 sorted 8-digit USGS gauge IDs.")
    basin_to_idx = {b: idx for idx, b in enumerate(basin_list)}

    stitched = np.full((EXPECTED_NUM_BASINS, expected_days), np.nan, dtype=np.float32)
    coverage_count = np.zeros(EXPECTED_NUM_BASINS, dtype=np.int64)

    for fold, pred in fold_results:
        pred_arr = _coerce_fold_prediction(pred, fold, expected_days)
        for row_i, b_id in enumerate(fold.test_basins):
            b_norm = str(b_id).strip().zfill(8)
            if b_norm not in basin_to_idx:
                raise ValueError(f"Fold {fold.fold_id!r} predicted unknown basin ID {b_norm!r}.")
            b_idx = basin_to_idx[b_norm]
            if coverage_count[b_idx] > 0:
                raise ValueError(
                    f"Basin {b_norm!r} was predicted more than once (duplicate in fold {fold.fold_id!r})."
                )
            stitched[b_idx, :] = pred_arr[row_i, :]
            coverage_count[b_idx] += 1

    missing_indices = np.where(coverage_count == 0)[0]
    if missing_indices.size > 0:
        missing_basins = [basin_list[int(i)] for i in missing_indices]
        raise ValueError(
            f"Incomplete basin coverage after fold stitching: {len(missing_basins)} of "
            f"{EXPECTED_NUM_BASINS} basins missing (first: {missing_basins[0]})."
        )
    return stitched


def stack_seed_members(
    seed_results: Mapping[int, Sequence[tuple[FoldSpec, np.ndarray]]],
    canonical_basins: Sequence[str],
    seeds: Sequence[int],
    expected_days: int = EXPECTED_NUM_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch every seed's folds and return ``(qsim_mean, qsim_members)`` as float32 arrays."""
    members = []
    for seed in seeds:
        if int(seed) not in seed_results:
            raise ValueError(f"No fold results recorded for seed {seed}.")
        members.append(stitch_fold_predictions(seed_results[int(seed)], canonical_basins, expected_days))
    members_arr = np.stack(members, axis=0).astype(np.float32)
    mean_arr = np.mean(members_arr.astype(np.float64), axis=0).astype(np.float32)
    return mean_arr, members_arr


def _git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """Return the current git commit hash, or None when git metadata is unavailable."""
    if not (repo_root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _git_dirty(repo_root: Path = REPO_ROOT) -> bool | None:
    """Return True when tracked files differ from HEAD, or None when git metadata is unavailable."""
    if not (repo_root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def validate_model_regimes(entry: ModelRegistryEntry, spatial_regime: str, forcing_regime: str) -> None:
    """Raise unless `entry` is registered for this spatial protocol and forcing regime.

    This is an anti-leakage guardrail: models that read observed test-basin discharge are only
    valid on ``lagged_streamflow`` leaderboards, and models with per-basin embeddings only in the
    ``gauged`` regime.
    """
    if spatial_regime not in entry.allowed_spatial_regimes:
        raise ValueError(
            f"Model {entry.key!r} is not registered for spatial_regime {spatial_regime!r}; "
            f"allowed: {entry.allowed_spatial_regimes}."
        )
    if forcing_regime not in entry.allowed_forcing_regimes:
        raise ValueError(
            f"Model {entry.key!r} is not registered for forcing_regime {forcing_regime!r}; "
            f"allowed: {entry.allowed_forcing_regimes}. Lagged-streamflow models need "
            "--forcing-regime lagged_streamflow."
        )


def default_output_uri(leaderboard_id: str, model_id: str, output_root: str | Path = "outputs") -> Path:
    """Local default store path ``<output_root>/<leaderboard_id>/<model_id>.zarr`` (never the shared bucket)."""
    return Path(output_root) / leaderboard_id / f"{model_id}.zarr"


def preflight_output_target(target_uri: str) -> None:
    """Fail fast, before any training, when the output store cannot be written.

    Local targets: the parent directory is created and must be writable. ``gs://`` targets: a small
    probe object is written next to the store and deleted again, so credential or permission errors
    surface immediately instead of after hours of training.
    """
    if target_uri.startswith("gs://"):
        import uuid

        import fsspec

        fs = fsspec.filesystem("gs")
        probe = f"{target_uri.rstrip('/')}.write_probe_{uuid.uuid4().hex}"
        fs.pipe(probe, b"")
        fs.rm(probe)
        return
    if "://" in target_uri:
        raise ValueError(f"Unsupported output URI scheme in {target_uri!r}; use a local path or gs://.")
    parent = Path(target_uri).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Output directory {parent} is not writable.")


def _limit_worker_threads(n_threads: int) -> None:
    """Process-pool initializer: cap BLAS/OpenMP/torch threads so N workers share the CPU cleanly."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(int(n_threads))
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        torch = importlib.import_module("torch")
        torch.set_num_threads(int(n_threads))


def _run_fold_seed_in_process(
    runner_module: str,
    fold: FoldSpec,
    seed: int,
    train_period: tuple[str, str],
    eval_period: tuple[str, str],
    forcings: Sequence[str],
    data_dir: str,
    merged_kwargs: dict[str, object],
) -> tuple[FoldSpec, int, np.ndarray]:
    """Module-level (picklable) task body for the process executor."""
    mod = importlib.import_module(runner_module)
    runner_fn = getattr(mod, "run_fold")
    pred = runner_fn(
        train_basins=fold.train_basins,
        test_basins=fold.test_basins,
        train_period=tuple(train_period),
        eval_period=tuple(eval_period),
        forcings=list(forcings),
        data_dir=data_dir,
        seeds=(int(seed),),
        fold_id=fold.fold_id,
        fold_index=fold.fold_index,
        **merged_kwargs,
    )
    return fold, int(seed), _coerce_fold_prediction(pred, fold, EXPECTED_NUM_DAYS)


def default_model_id(entry: ModelRegistryEntry, merged_kwargs: Mapping[str, object], spatial_regime: str, forcing_regime: str) -> str:
    """`<family>[_<model_kind|variant>]_<spatial_regime>_<forcing_regime>` so sub-model selectors never collide."""
    selector = merged_kwargs.get("model_kind", merged_kwargs.get("variant"))
    middle = f"_{str(selector).strip().lower()}" if selector not in (None, "") else ""
    return f"{entry.key}{middle}_{spatial_regime}_{forcing_regime}"


# =============================================================================
# Experiment driver
# =============================================================================


def run_model_experiment(
    model_name: str,
    spatial_regime: str,
    forcings: Sequence[str],
    data_dir: str | Path,
    *,
    output_zarr: str | Path | None = None,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    train_period: tuple[str, str] = (CANONICAL_TRAIN_START, CANONICAL_TRAIN_END),
    eval_period: tuple[str, str] = (CANONICAL_EVAL_START, CANONICAL_EVAL_END),
    seeds: Sequence[int] | None = None,
    max_workers: int = 1,
    executor: str = "thread",
    threads_per_worker: int = 1,
    fold_runner: Callable[..., np.ndarray] | None = None,
    model_id: str | None = None,
    forcing_regime: str | None = None,
    runner_kwargs: Mapping[str, object] | None = None,
) -> tuple[xr.Dataset, str]:
    """Run a registered model across all folds and seeds of `spatial_regime`; write its Zarr store.

    Every seed is executed as an independent full run of every fold (``run_fold(..., seeds=(seed,))``);
    the stitched per-seed hydrographs are stored as ``qsim_members`` and their arithmetic mean as
    ``qsim``. ``max_workers > 1`` parallelizes ``(fold, seed)`` tasks either with threads
    (``executor="thread"``; reproducible only if the runner uses seed-local RNGs) or with separate
    processes (``executor="process"``; each worker is capped to ``threads_per_worker`` BLAS/torch
    threads, which is the right choice for GIL-bound per-basin calibrations).

    The model must be registered for ``spatial_regime`` and the resolved forcing regime
    (``validate_model_regimes``). Without ``output_zarr`` the store is written locally to
    ``outputs/<leaderboard_id>/<model_id>.zarr``; the target is checked for write access before any
    fold runs. The canonical bucket URI is still recorded in the ``archive_url``/``data_url`` attrs.
    """
    entry = resolve_model_entry(model_name)
    norm_forcings = normalize_forcing_names(forcings)
    resolved_forcing_regime = resolve_forcing_regime_for_run(norm_forcings, forcing_regime)
    validate_periods(tuple(train_period), tuple(eval_period))
    if spatial_regime not in VALID_SPATIAL_REGIMES:
        raise ValueError(f"Invalid spatial_regime {spatial_regime!r}; must be one of {VALID_SPATIAL_REGIMES}.")
    if executor not in ("thread", "process"):
        raise ValueError(f"executor must be 'thread' or 'process', got {executor!r}.")
    if executor == "process" and fold_runner is not None:
        raise ValueError("executor='process' requires a registered runner module (fold_runner injection is thread-only).")

    canonical_basins = load_canonical_basins(splits_dir)
    basin_to_huc02 = load_basin_to_huc02(splits_dir)
    folds = load_spatial_folds(spatial_regime, splits_dir=splits_dir)

    resolved_seeds = tuple(int(s) for s in (seeds if seeds is not None else entry.default_seeds))
    if not resolved_seeds:
        raise ValueError("seeds sequence must contain at least one integer random seed.")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError(f"Duplicate seeds requested: {resolved_seeds}.")

    merged_kwargs: dict[str, object] = dict(entry.default_kwargs)
    if runner_kwargs is not None:
        merged_kwargs.update(runner_kwargs)

    validate_model_regimes(entry, spatial_regime, resolved_forcing_regime)
    leaderboard_id, table_id = resolve_leaderboard_and_table_id(
        spatial_regime=spatial_regime,
        forcing_regime=resolved_forcing_regime,
    )
    resolved_model_id = (
        model_id
        if model_id is not None
        else default_model_id(entry, merged_kwargs, spatial_regime, resolved_forcing_regime)
    )
    default_gcs_url = default_store_uri(leaderboard_id, resolved_model_id)
    target_uri = str(output_zarr) if output_zarr is not None else str(default_output_uri(leaderboard_id, resolved_model_id))
    preflight_output_target(target_uri)

    active_runner = fold_runner if fold_runner is not None else load_model_fold_runner(entry)

    def _execute_fold_seed(task: tuple[FoldSpec, int]) -> tuple[FoldSpec, int, np.ndarray]:
        fold, seed = task
        pred = active_runner(
            train_basins=fold.train_basins,
            test_basins=fold.test_basins,
            train_period=tuple(train_period),
            eval_period=tuple(eval_period),
            forcings=norm_forcings,
            data_dir=data_dir,
            seeds=(int(seed),),
            fold_id=fold.fold_id,
            fold_index=fold.fold_index,
            **merged_kwargs,
        )
        return fold, int(seed), _coerce_fold_prediction(pred, fold, EXPECTED_NUM_DAYS)

    tasks = [(fold, seed) for seed in resolved_seeds for fold in folds]
    if max_workers > 1 and len(tasks) > 1 and executor == "process":
        with ProcessPoolExecutor(
            max_workers=int(max_workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_worker_threads,
            initargs=(int(threads_per_worker),),
        ) as pool:
            futures = [
                pool.submit(
                    _run_fold_seed_in_process,
                    entry.runner_module,
                    fold,
                    seed,
                    tuple(train_period),
                    tuple(eval_period),
                    list(norm_forcings),
                    str(data_dir),
                    merged_kwargs,
                )
                for fold, seed in tasks
            ]
            results = [f.result() for f in futures]
    elif max_workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            results = list(pool.map(_execute_fold_seed, tasks))
    else:
        results = [_execute_fold_seed(task) for task in tasks]

    seed_results: dict[int, list[tuple[FoldSpec, np.ndarray]]] = {s: [] for s in resolved_seeds}
    for fold, seed, pred in results:
        seed_results[seed].append((fold, pred))

    qsim_mean, qsim_members = stack_seed_members(
        seed_results, canonical_basins, resolved_seeds, expected_days=EXPECTED_NUM_DAYS
    )

    provenance_attrs = run_provenance_attrs(
        seeds=resolved_seeds,
        fold_ids=[f.fold_id for f in folds],
        runner_kwargs=merged_kwargs,
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        data_source=str(data_dir),
    )

    ds = build_hydrograph_dataset(
        qsim=qsim_mean,
        basins=canonical_basins,
        basin_to_huc02=basin_to_huc02,
        model_id=resolved_model_id,
        model_name=entry.model_name,
        model_family=entry.model_family,
        family_id=entry.key,
        provenance_category=entry.provenance_category,
        spatial_regime=spatial_regime,
        forcing_regime=resolved_forcing_regime,
        forcings_used=forcing_display_label(norm_forcings),
        code_url=entry.code_url,
        qsim_members=qsim_members,
        is_ensemble=len(resolved_seeds) > 1,
        num_members=len(resolved_seeds),
        leaderboard_id=leaderboard_id,
        table_id=table_id,
        training_period=tuple(train_period),
        evaluation_period=tuple(eval_period),
        data_url=default_gcs_url,
        extra_attrs=provenance_attrs,
    )

    written_uri = write_hydrograph_zarr(ds=ds, target_uri=target_uri, expected_basins=canonical_basins)
    return ds, written_uri


# =============================================================================
# Command line
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Centralized HydrologyMIP model runner enforcing canonical splits and anti-leakage CV hygiene."
    )
    parser.add_argument(
        "--mip",
        type=str,
        default=DEFAULT_MIP_ID,
        help=f"Registered MIP identifier under mips/ (default: {DEFAULT_MIP_ID}).",
    )
    parser.add_argument("--list-models", action="store_true", help="List all registered model families and exit.")
    parser.add_argument(
        "--model", type=str, choices=sorted(MODEL_REGISTRY.keys()), help="Registered model directory name to run."
    )
    parser.add_argument(
        "--spatial-regime",
        type=str,
        default="gauged",
        choices=list(VALID_SPATIAL_REGIMES),
        help="Spatial evaluation protocol: gauged, pub_12fold, or pur_18huc02.",
    )
    parser.add_argument(
        "--forcings",
        type=str,
        nargs="+",
        default=["daymet"],
        help="Distinct CAMELS forcing products ('daymet', 'maurer', 'nldas'); one -> single_camels, "
        "two or three -> multi_camels.",
    )
    parser.add_argument(
        "--forcing-regime",
        type=str,
        default=None,
        choices=list(VALID_FORCING_REGIMES),
        help="Optional explicit forcing regime (needed only for lagged_streamflow / freeform submissions).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset bundle or raw directory (defaults to mips/<mip>/data).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory containing canonical split files (defaults to mips/<mip>/splits).",
    )
    parser.add_argument(
        "--output-zarr",
        type=str,
        default=None,
        help="Target Zarr path or gs:// URI (default: local ./outputs/<leaderboard_id>/<model_id>.zarr). "
        "Writable access is checked before training starts.",
    )
    parser.add_argument(
        "--model-id", type=str, default=None, help="Optional explicit entry/model id for attrs and store name."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional random seed list override.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel (fold, seed) workers. With --executor thread keep 1 unless the runner uses seed-local RNGs.",
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="thread",
        choices=["thread", "process"],
        help="How --max-workers > 1 parallelizes (fold, seed) tasks: threads or spawned processes.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="BLAS/OpenMP/torch threads per process worker (process executor only).",
    )
    parser.add_argument(
        "--runner-kwarg",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Extra keyword argument forwarded to run_fold, value parsed as JSON (repeatable).",
    )
    return parser


def _parse_runner_kwargs(items: Sequence[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--runner-kwarg expects KEY=JSON, got {item!r}.")
        key, raw = item.split("=", 1)
        out[key.strip()] = json.loads(raw)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for key, entry in sorted(MODEL_REGISTRY.items()):
            print(f"{key:30s} | {entry.provenance_category:21s} | {entry.model_name}")
        return 0

    if not args.model:
        parser.error("--model is required unless --list-models is specified.")

    mip_paths = resolve_mip_paths(args.mip)
    resolved_data_dir = args.data_dir if args.data_dir is not None else mip_paths.data_dir
    resolved_splits_dir = args.splits_dir if args.splits_dir is not None else mip_paths.splits_dir

    ds, written_uri = run_model_experiment(
        model_name=args.model,
        spatial_regime=args.spatial_regime,
        forcings=args.forcings,
        data_dir=resolved_data_dir,
        output_zarr=args.output_zarr,
        splits_dir=resolved_splits_dir,
        seeds=args.seeds,
        max_workers=args.max_workers,
        executor=args.executor,
        threads_per_worker=args.threads_per_worker,
        model_id=args.model_id,
        forcing_regime=args.forcing_regime,
        runner_kwargs=_parse_runner_kwargs(args.runner_kwarg),
    )
    print(
        f"Wrote consolidated Zarr hydrograph store ({ds.qsim.shape[0]} basins x {ds.qsim.shape[1]} days, "
        f"{int(ds.attrs['num_members'])} member(s)) to {written_uri}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
