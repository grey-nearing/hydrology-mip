# Copyright 2026 HydroMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""MIP-agnostic cross-validation split loading and strict anti-leakage hygiene verification.

Shared across all HydrologyMIP benchmarks (CAMELS-US, Caravan, probabilistic, soil moisture,
floods, forecasting). Enforces:

1. Every fold's ``train_entities`` and ``test_entities`` are strictly disjoint.
2. No fold contains duplicate entity IDs or IDs outside the canonical entity cohort.
3. For full-partition CV schemes (e.g. PUB, HUC-02 PUR, hydrologically separated PUR),
   the union of all fold ``test_entities`` forms an exact, disjoint partition of the
   target evaluation cohort.
4. For optional group-separated CV schemes (e.g. HUC-02 regions or HydroSHEDS river
   networks ``MAIN_BAS``), zero groups are shared between ``train_entities`` and
   ``test_entities`` in any fold.

Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CvFold:
    """One validated cross-validation fold with explicit train and test entity tuples."""

    fold_id: str
    fold_index: int
    train_entities: tuple[str, ...]
    test_entities: tuple[str, ...]


def validate_cv_folds(
    folds: Sequence[CvFold],
    canonical_entities: Sequence[str],
    *,
    require_disjoint_train_test: bool = True,
    require_exact_test_partition: bool = True,
    entity_to_group: Mapping[str, str | int] | None = None,
    require_train_test_union_equals_cohort: bool = False,
    require_single_test_group: bool = False,
) -> None:
    """Assert strict cross-validation hygiene invariants across ``folds``.

    Optional stricter invariants:

    * ``require_train_test_union_equals_cohort``: every fold's train and test entities
      together cover the whole canonical cohort (no entity silently dropped from a fold).
    * ``require_single_test_group`` (needs ``entity_to_group``): every fold's test set is
      exactly one group (leave-one-group-out, e.g. leave-one-HUC02-region-out PUR).
    """
    if not folds:
        raise ValueError("Cross-validation folds list must be non-empty.")
    if require_single_test_group and entity_to_group is None:
        raise ValueError("require_single_test_group=True requires entity_to_group.")

    canonical_list = list(canonical_entities)
    canonical_set = set(canonical_list)
    if len(canonical_set) != len(canonical_list):
        raise ValueError("canonical_entities contains duplicate IDs.")

    seen_test: list[str] = []
    seen_fold_ids: set[str] = set()
    seen_fold_indices: set[int] = set()
    for fold in folds:
        if fold.fold_id in seen_fold_ids:
            raise ValueError(f"Duplicate fold_id {fold.fold_id!r}.")
        seen_fold_ids.add(fold.fold_id)
        if fold.fold_index in seen_fold_indices:
            raise ValueError(f"Duplicate fold_index {fold.fold_index} (fold {fold.fold_id!r}).")
        seen_fold_indices.add(fold.fold_index)

        train_set = set(fold.train_entities)
        test_set = set(fold.test_entities)
        if len(train_set) != len(fold.train_entities):
            raise ValueError(f"Fold {fold.fold_id!r} contains duplicate train entities.")
        if len(test_set) != len(fold.test_entities):
            raise ValueError(f"Fold {fold.fold_id!r} contains duplicate test entities.")
        if not train_set:
            raise ValueError(f"Fold {fold.fold_id!r} has an empty train set.")
        if not test_set:
            raise ValueError(f"Fold {fold.fold_id!r} has an empty test set.")

        unknown_train = sorted(train_set - canonical_set)
        unknown_test = sorted(test_set - canonical_set)
        if unknown_train:
            raise ValueError(
                f"Fold {fold.fold_id!r} has train entities outside canonical cohort: {unknown_train[:5]}"
            )
        if unknown_test:
            raise ValueError(
                f"Fold {fold.fold_id!r} has test entities outside canonical cohort: {unknown_test[:5]}"
            )

        if require_disjoint_train_test:
            overlap = sorted(train_set & test_set)
            if overlap:
                raise ValueError(
                    f"CV leakage in fold {fold.fold_id!r}: {len(overlap)} entities appear in "
                    f"both train and test (e.g. {overlap[:5]})."
                )

        if entity_to_group is not None and require_disjoint_train_test:
            missing_grp = sorted((train_set | test_set) - set(entity_to_group.keys()))
            if missing_grp:
                raise KeyError(
                    f"entity_to_group missing {len(missing_grp)} entities (e.g. {missing_grp[:5]})."
                )
            train_groups = {entity_to_group[e] for e in train_set}
            test_groups = {entity_to_group[e] for e in test_set}
            shared_groups = train_groups & test_groups
            if shared_groups:
                sample = sorted(str(g) for g in shared_groups)[:5]
                raise ValueError(
                    f"Group separation violation in fold {fold.fold_id!r}: "
                    f"{len(shared_groups)} shared groups between train and test (e.g. {sample})."
                )
            if require_single_test_group and len(test_groups) != 1:
                raise ValueError(
                    f"Fold {fold.fold_id!r} must hold out exactly one group; its test set spans "
                    f"{len(test_groups)} groups ({sorted(str(g) for g in test_groups)[:5]})."
                )

        if require_train_test_union_equals_cohort:
            uncovered = sorted(canonical_set - (train_set | test_set))
            if uncovered:
                raise ValueError(
                    f"Fold {fold.fold_id!r} train + test do not cover the canonical cohort: "
                    f"{len(uncovered)} entities missing (e.g. {uncovered[:5]})."
                )

        seen_test.extend(fold.test_entities)

    if require_exact_test_partition:
        if len(seen_test) != len(set(seen_test)):
            raise ValueError("Test folds overlap: at least one entity appears in multiple test folds.")
        missing_eval = sorted(canonical_set - set(seen_test))
        if missing_eval:
            raise ValueError(
                f"Incomplete test partition: {len(missing_eval)} canonical entities never tested "
                f"(e.g. {missing_eval[:5]})."
            )


def load_cv_splits_json(
    split_json_path: str | Path,
    canonical_entities: Sequence[str],
    *,
    expected_num_folds: int | None = None,
    require_disjoint_train_test: bool = True,
    require_exact_test_partition: bool = True,
    entity_to_group: Mapping[str, str | int] | None = None,
    require_train_test_union_equals_cohort: bool = False,
    require_single_test_group: bool = False,
) -> list[CvFold]:
    """Load a canonical HydrologyMIP split JSON file and verify all CV hygiene invariants.

    Folds are ordered by fold id. A fold's ``fold_index`` is taken from the file when it
    declares one (and must then be unique); otherwise it is the fold's sorted position.
    """
    p = Path(split_json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Split JSON file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "folds" not in raw or not isinstance(raw["folds"], dict):
        raise ValueError(f"Split JSON {p} must contain a dict under top-level key 'folds'.")

    folds_dict: dict[str, dict[str, Sequence[str]]] = raw["folds"]
    if expected_num_folds is not None and len(folds_dict) != expected_num_folds:
        raise ValueError(
            f"Split JSON {p} defines {len(folds_dict)} folds; expected {expected_num_folds}."
        )

    folds: list[CvFold] = []
    for idx, (fold_id, fold_data) in enumerate(sorted(folds_dict.items())):
        if "train_basins" in fold_data and "test_basins" in fold_data:
            train_raw = fold_data["train_basins"]
            test_raw = fold_data["test_basins"]
        elif "train_entities" in fold_data and "test_entities" in fold_data:
            train_raw = fold_data["train_entities"]
            test_raw = fold_data["test_entities"]
        else:
            raise KeyError(
                f"Fold {fold_id!r} in {p} must specify 'train_basins'/'test_basins' "
                "or 'train_entities'/'test_entities'."
            )
        fold_index = int(fold_data["fold_index"]) if "fold_index" in fold_data else idx
        folds.append(
            CvFold(
                fold_id=str(fold_id),
                fold_index=fold_index,
                train_entities=tuple(str(x).strip() for x in train_raw),
                test_entities=tuple(str(x).strip() for x in test_raw),
            )
        )

    validate_cv_folds(
        folds,
        canonical_entities,
        require_disjoint_train_test=require_disjoint_train_test,
        require_exact_test_partition=require_exact_test_partition,
        entity_to_group=entity_to_group,
        require_train_test_union_equals_cohort=require_train_test_union_equals_cohort,
        require_single_test_group=require_single_test_group,
    )
    return folds
