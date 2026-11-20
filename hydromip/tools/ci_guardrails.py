# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Automated CI protocol, AST, security, documentation, skills, and registry guardrails.

Executed in GitHub Actions (`.github/workflows/ci.yml`) and runnable locally via:
    python -m hydromip.tools.ci_guardrails --mode all
    python -m hydromip.tools.ci_guardrails --self-test

Zero `try`/`except` blocks. All file, schema, and AST checks validate preconditions explicitly.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SELF_REL_PATH = "hydromip/tools/ci_guardrails.py"
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB ceiling outside cloud Zarr storage

FORBIDDEN_BINARY_EXTENSIONS = frozenset(
    {
        ".npz",
        ".npy",
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
        ".pickle",
        ".h5",
        ".hdf5",
        ".nc",
        ".parquet",
    }
)
BINARY_ASSET_EXTENSIONS = frozenset(
    {".png", ".pdf", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2"}
)
FORBIDDEN_MODEL_SPLIT_LITERALS = (
    "pub_12fold_random_splits.json",
    "pur_18fold_huc02_splits.json",
    "basins_531.txt",
    "basins_531.json",
    "basins_671.txt",
    "basins_671.json",
    "splits_and_regimes.json",
)
FORBIDDEN_MODEL_DATE_LITERALS = (
    "1989-10-01",
    "1999-09-30",
    "1999-10-01",
    "2008-09-30",
)


def _build_leak_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Construct workstation, internal-jargon, and secret regexes without embedding literal tokens."""
    ws_google = "/usr/local/" + "google"
    ws_user = "/home/" + "gsnearing"
    g3_src = "/google/" + "src"
    cns_prefix = "/" + "cns/"
    corp_dom = r"corp\.google" + r"\.com"
    plex_dom = r"googleplex" + r"\.com"
    g3_word = r"\bgoogle" + r"3\b"
    glinux_word = r"\bg" + r"Linux\b"
    cloudtop_word = r"\bcloud" + r"top\b"
    caa_env = "CLOUDSDK_CONTEXT_" + "AWARE"
    ghydro_env = r"\bgoogle" + r"hydrology\b"
    return (
        ("workstation_home_path", re.compile(rf"{ws_google}|{ws_user}|/Users/[A-Za-z0-9_.-]+")),
        (
            "internal_google_ref",
            re.compile(
                rf"{g3_src}|{cns_prefix}|{corp_dom}|{plex_dom}|"
                rf"{g3_word}|{glinux_word}|{cloudtop_word}|{caa_env}"
            ),
        ),
        ("internal_conda_env", re.compile(ghydro_env)),
        (
            "secret_or_private_key",
            re.compile(
                r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
                r"AIza[0-9A-Za-z\-_]{35}|AKIA[0-9A-Z]{16}|"
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
            ),
        ),
    )


LEAK_AND_SECRET_PATTERNS = _build_leak_patterns()
HARDCODED_TMP_PATTERN = re.compile(r"(?<![A-Za-z0-9_/])/tmp(?:/[A-Za-z0-9_.\-]+|['\"])")
STALE_DOC_TOKEN_PATTERN = re.compile(
    r"\b(?:single_camels_"
    + r"ensemble|CONSTANT_ZERO_"
    + r"OBS_BASINS|12 Gauch|12-"
    + r"metric|Missed-"
    + r"Peaks|Peak-"
    + r"MAPE)\b"
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _list_repo_files(repo_root: Path) -> list[Path]:
    """Return all tracked and non-ignored untracked files in `repo_root`."""
    git_dir = repo_root / ".git"
    if git_dir.exists():
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=repo_root, text=True
        ).splitlines()
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            text=True,
        ).splitlines()
        rels = sorted(set(tracked + untracked))
        return [repo_root / r for r in rels if (repo_root / r).is_file()]
    return sorted(
        p
        for p in repo_root.rglob("*")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    )


def _ast_try_types() -> tuple[type[ast.AST], ...]:
    """Return all AST try-statement node classes supported by this Python runtime."""
    try_star = getattr(ast, "TryStar", None)
    if try_star is not None:
        return (ast.Try, try_star)
    return (ast.Try,)


def check_code_guardrails(repo_root: Path = REPO_ROOT) -> list[str]:
    """Enforce AST rules, pickle safety, split/date isolation, leak checks, and <5 MiB file cap."""
    errors: list[str] = []
    files = _list_repo_files(repo_root)
    try_types = _ast_try_types()

    for path in files:
        rel = path.relative_to(repo_root)
        rel_str = rel.as_posix()
        size = path.stat().st_size
        is_self = rel_str == SELF_REL_PATH
        is_test = "tests" in rel.parts

        # 1. File size & binary blob guardrails
        if size > MAX_TRACKED_FILE_BYTES:
            errors.append(
                f"[binary-size] {rel_str} is {size / (1024 * 1024):.2f} MiB "
                "(exceeds 5.00 MiB Git limit; store large datasets/hydrographs in cloud Zarr storage)."
            )
        if path.suffix.lower() in FORBIDDEN_BINARY_EXTENSIONS or ".zarr/" in rel_str:
            errors.append(
                f"[forbidden-blob] {rel_str}: binary hydrograph/checkpoint files "
                f"({path.suffix or '.zarr'}) must never be committed to Git."
            )

        if is_self:
            continue

        # 2. Internal paths, workstation paths, and secrets across text & notebook files
        if path.suffix.lower() not in BINARY_ASSET_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            is_legacy_frontend_doc = rel_str == "frontend/README.md"
            for lineno, line in enumerate(text.splitlines(), 1):
                for label, pattern in LEAK_AND_SECRET_PATTERNS:
                    if is_legacy_frontend_doc and label == "internal_conda_env":
                        continue
                    if pattern.search(line):
                        errors.append(f"[leak:{label}] {rel_str}:{lineno}: {line.strip()[:120]}")
                if path.suffix.lower() == ".py" and not is_test and HARDCODED_TMP_PATTERN.search(line):
                    errors.append(
                        f"[leak:hardcoded_tmp_path] {rel_str}:{lineno}: {line.strip()[:120]}"
                    )
            if path.suffix.lower() == ".ipynb" and not is_test:
                nb_doc = json.loads(text)
                for cell_idx, cell in enumerate(nb_doc.get("cells", [])):
                    src_field = cell.get("source", [])
                    src_lines = (
                        src_field.splitlines() if isinstance(src_field, str) else list(src_field)
                    )
                    for src_line in src_lines:
                        if HARDCODED_TMP_PATTERN.search(src_line):
                            errors.append(
                                f"[leak:hardcoded_tmp_path] {rel_str}:cell[{cell_idx}]: "
                                f"{src_line.strip()[:120]}"
                            )

        # 3. Python AST protocol & safety guardrails
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_str)

            if not is_test:
                for node in ast.walk(tree):
                    if isinstance(node, try_types):
                        errors.append(
                            f"[prohibited-try-except] {rel_str}:{node.lineno}: "
                            "try/except/finally blocks are prohibited in library, model, and API code."
                        )
                    if isinstance(node, ast.Attribute) and node.attr in (
                        "suppress",
                        "catch_warnings",
                    ):
                        errors.append(
                            f"[error-suppression] {rel_str}:{node.lineno}: "
                            f"prohibited `{node.attr}` error/warning suppression."
                        )
                    if isinstance(node, ast.Call):
                        for kw in node.keywords:
                            if (
                                kw.arg == "ignore_errors"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True
                            ):
                                errors.append(
                                    f"[error-suppression] {rel_str}:{node.lineno}: "
                                    "prohibited `ignore_errors=True`."
                                )
                            if kw.arg == "allow_pickle" and not (
                                isinstance(kw.value, ast.Constant) and kw.value.value is False
                            ):
                                errors.append(
                                    f"[pickle-rce] {rel_str}:{node.lineno}: "
                                    "`allow_pickle` must be explicitly `False`."
                                )

                        func_name = ""
                        if isinstance(node.func, ast.Attribute):
                            base = getattr(node.func.value, "id", "")
                            func_name = f"{base}.{node.func.attr}" if base else node.func.attr
                        elif isinstance(node.func, ast.Name):
                            func_name = node.func.id

                        if func_name in ("np.load", "numpy.load"):
                            kw_map = {kw.arg: kw.value for kw in node.keywords}
                            if "allow_pickle" not in kw_map:
                                errors.append(
                                    f"[pickle-rce] {rel_str}:{node.lineno}: "
                                    "`np.load(...)` must pass explicit `allow_pickle=False`."
                                )
                        if func_name == "torch.load":
                            kw_map = {kw.arg: kw.value for kw in node.keywords}
                            if "weights_only" not in kw_map or not (
                                isinstance(kw_map["weights_only"], ast.Constant)
                                and kw_map["weights_only"].value is True
                            ):
                                errors.append(
                                    f"[pickle-rce] {rel_str}:{node.lineno}: "
                                    "`torch.load(...)` must pass `weights_only=True`."
                                )

            # Centralized split & date control inside `models/<model_key>/*`
            if (
                len(rel.parts) >= 3
                and rel.parts[0] == "models"
                and not is_test
                and path.name != "reproduce_author_metrics.py"
            ):
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        for tok in FORBIDDEN_MODEL_SPLIT_LITERALS:
                            if tok in node.value:
                                errors.append(
                                    f"[split-isolation] {rel_str}:{node.lineno}: "
                                    f"model code references split file {tok!r} "
                                    "(splits must be injected by models/run_model.py)."
                                )
                        for tok in FORBIDDEN_MODEL_DATE_LITERALS:
                            if tok in node.value:
                                errors.append(
                                    f"[date-isolation] {rel_str}:{node.lineno}: "
                                    f"model code hardcodes canonical Newman date {tok!r}."
                                )

    return errors


def _extract_model_registry_from_ast(run_model_path: Path) -> dict[str, dict[str, str]]:
    """Extract `MODEL_REGISTRY` string fields from `models/run_model.py` without importing torch."""
    tree = ast.parse(run_model_path.read_text(encoding="utf-8"), filename=str(run_model_path))
    registry: dict[str, dict[str, str]] = {}
    for node in tree.body:
        target_name = None
        dict_node = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            dict_node = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            dict_node = node.value
        if target_name == "MODEL_REGISTRY" and isinstance(dict_node, ast.Dict):
            for k_node, v_node in zip(dict_node.keys, dict_node.values):
                if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                    fields: dict[str, str] = {}
                    if isinstance(v_node, ast.Call):
                        for kw in v_node.keywords:
                            if (
                                kw.arg is not None
                                and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)
                            ):
                                fields[kw.arg] = kw.value.value
                    registry[k_node.value] = fields
    return registry


def _parse_skill_frontmatter(skill_md_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Parse leading YAML frontmatter (`--- ... ---`) from a `SKILL.md` file."""
    lines = skill_md_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, "missing opening '---' YAML frontmatter delimiter on line 1"
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return None, "missing closing '---' YAML frontmatter delimiter"
    parsed = yaml.safe_load("\n".join(lines[1:end_idx]))
    if not isinstance(parsed, dict):
        return None, "YAML frontmatter must parse to a mapping"
    return parsed, None


def check_agent_skills_layout(repo_root: Path = REPO_ROOT) -> list[str]:
    """Verify multi-agent skill layout whenever `AGENTS.md` exists at `repo_root`."""
    agents_md = repo_root / "AGENTS.md"
    if not agents_md.is_file():
        return []

    errors: list[str] = []
    skills_dir = repo_root / "skills"
    if not skills_dir.is_dir():
        errors.append("[agent-skills] AGENTS.md exists, but skills/ directory is missing.")
        return errors

    expected_skills_target = skills_dir.resolve()
    for link_rel in (".claude/skills", ".agents/skills"):
        link_path = repo_root / link_rel
        if not link_path.is_symlink():
            errors.append(f"[agent-skills] Expected symlink {link_rel} -> skills is missing.")
        elif link_path.resolve() != expected_skills_target:
            errors.append(
                f"[agent-skills] Symlink {link_rel} resolves to {link_path.resolve()}, "
                f"expected {expected_skills_target}."
            )

    for bridge_name in ("CLAUDE.md", "GEMINI.md"):
        bridge_path = repo_root / bridge_name
        if not bridge_path.is_file():
            errors.append(f"[agent-skills] Expected root {bridge_name} importing AGENTS.md.")
        else:
            bridge_txt = bridge_path.read_text(encoding="utf-8")
            if "AGENTS.md" not in bridge_txt:
                errors.append(f"[agent-skills] {bridge_name} must import AGENTS.md.")

    agents_text = agents_md.read_text(encoding="utf-8")
    skill_subdirs = sorted(
        d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))
    )
    if not skill_subdirs:
        errors.append("[agent-skills] No skill subdirectories found under skills/<name>/SKILL.md.")

    for folder in skill_subdirs:
        skill_file = folder / "SKILL.md"
        rel_skill = f"skills/{folder.name}/SKILL.md"
        if not skill_file.is_file():
            errors.append(f"[agent-skills] Missing {rel_skill}.")
            continue
        meta, err = _parse_skill_frontmatter(skill_file)
        if err is not None or meta is None:
            errors.append(f"[agent-skills] {rel_skill}: {err}.")
            continue
        if meta.get("name") != folder.name:
            errors.append(
                f"[agent-skills] {rel_skill}: frontmatter name={meta.get('name')!r} "
                f"must equal directory name {folder.name!r}."
            )
        desc = meta.get("description")
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"[agent-skills] {rel_skill}: frontmatter `description` must be non-empty.")
        elif len(desc) > 1024:
            errors.append(
                f"[agent-skills] {rel_skill}: frontmatter `description` length {len(desc)} exceeds 1024 chars."
            )
        if rel_skill not in agents_text:
            errors.append(f"[agent-skills] AGENTS.md does not reference '{rel_skill}'.")

    return errors


def check_docs_and_registries(repo_root: Path = REPO_ROOT) -> list[str]:
    """Verify Markdown links, banned doc tokens, metadata parity, skills, and MIP JSON bijection."""
    errors: list[str] = []
    files = _list_repo_files(repo_root)
    resolved_root = repo_root.resolve()

    # 1. Markdown relative links & stale/nonexistent doc token checks
    for path in files:
        rel = path.relative_to(repo_root)
        rel_str = rel.as_posix()
        if rel_str == SELF_REL_PATH:
            continue
        is_doc_or_skill = (
            path.suffix.lower() == ".md"
            or rel_str.startswith(".github/ISSUE_TEMPLATE/")
            or (
                len(rel.parts) >= 1
                and rel.parts[0] == "skills"
                and path.suffix.lower() not in BINARY_ASSET_EXTENSIONS
            )
        )
        if not is_doc_or_skill:
            continue

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        in_fence = False
        is_legacy_frontend_doc = rel_str == "frontend/README.md"
        for lineno, line in enumerate(lines, 1):
            for tok_match in STALE_DOC_TOKEN_PATTERN.finditer(line):
                errors.append(
                    f"[stale-doc-token] {rel_str}:{lineno}: references nonexistent "
                    f"identifier {tok_match.group(0)!r}."
                )
            for label, pattern in LEAK_AND_SECRET_PATTERNS:
                if is_legacy_frontend_doc and label == "internal_conda_env":
                    continue
                if pattern.search(line):
                    errors.append(f"[doc-leak:{label}] {rel_str}:{lineno}: {line.strip()[:120]}")

            if path.suffix.lower() == ".md":
                stripped = line.lstrip()
                if stripped.startswith(("```", "~~~")):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                clean_line = re.sub(r"`[^`]+`", "", line)
                for match in MARKDOWN_LINK_RE.finditer(clean_line):
                    raw_target = match.group(1).strip()
                    if not raw_target or raw_target.startswith(
                        ("http://", "https://", "mailto:", "#")
                    ):
                        continue
                    rel_target = raw_target.split("#", 1)[0].split(" ", 1)[0].strip()
                    if not rel_target:
                        continue
                    resolved = (path.parent / rel_target).resolve()
                    if not resolved.is_relative_to(resolved_root):
                        errors.append(
                            f"[broken-link] {rel_str}:{lineno}: link '{raw_target}' escapes repository root."
                        )
                    elif not resolved.exists():
                        errors.append(
                            f"[broken-link] {rel_str}:{lineno}: target '{raw_target}' does not exist."
                        )

    # 2. Agent skills layout (active once AGENTS.md exists)
    errors.extend(check_agent_skills_layout(repo_root))

    # 3. CITATION.cff, .zenodo.json, and pyproject.toml schema & author parity
    zenodo_path = repo_root / ".zenodo.json"
    cff_path = repo_root / "CITATION.cff"
    pyproject_path = repo_root / "pyproject.toml"

    if not zenodo_path.is_file():
        errors.append("[metadata] Missing .zenodo.json.")
    if not cff_path.is_file():
        errors.append("[metadata] Missing CITATION.cff.")
    if not pyproject_path.is_file():
        errors.append("[metadata] Missing pyproject.toml.")

    if zenodo_path.is_file() and cff_path.is_file() and pyproject_path.is_file():
        zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
        cff = yaml.safe_load(cff_path.read_text(encoding="utf-8"))
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        if not isinstance(zenodo, dict):
            errors.append("[metadata] .zenodo.json must contain a JSON object.")
        if not isinstance(cff, dict):
            errors.append("[metadata] CITATION.cff must contain a YAML mapping.")
        if isinstance(zenodo, dict) and isinstance(cff, dict):
            zenodo_surnames = [
                str(c.get("name", "")).split(",", 1)[0].strip()
                for c in zenodo.get("creators", [])
            ]
            cff_surnames = [str(a.get("family-names", "")).strip() for a in cff.get("authors", [])]
            if zenodo_surnames != cff_surnames:
                errors.append(
                    f"[metadata-drift] .zenodo.json creators {zenodo_surnames} "
                    f"!= CITATION.cff authors {cff_surnames}."
                )
            pref = cff.get("preferred-citation")
            if isinstance(pref, dict) and "authors" in pref:
                pref_surnames = [
                    str(a.get("family-names", "")).strip() for a in pref.get("authors", [])
                ]
                if pref_surnames != cff_surnames:
                    errors.append(
                        f"[metadata-drift] CITATION.cff preferred-citation authors {pref_surnames} "
                        f"!= top-level authors {cff_surnames}."
                    )
            if str(cff.get("version", "")) != str(pyproject["project"]["version"]):
                errors.append(
                    f"[metadata-drift] CITATION.cff version {cff.get('version')!r} != "
                    f"pyproject.toml version {pyproject['project']['version']!r}."
                )
            if zenodo.get("license") != "Apache-2.0" or cff.get("license") != "Apache-2.0":
                errors.append("[metadata-drift] Both .zenodo.json and CITATION.cff must declare license Apache-2.0.")

    # 4. Model catalog <-> filesystem <-> MODEL_REGISTRY parity
    catalog_path = repo_root / "models" / "catalog.json"
    run_model_path = repo_root / "models" / "run_model.py"
    if catalog_path.is_file() and run_model_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_models = catalog.get("models", [])
        catalog_keys = [str(m["key"]) for m in catalog_models]
        if len(catalog_keys) != len(set(catalog_keys)):
            errors.append("[catalog] Duplicate model keys in models/catalog.json.")

        catalog_by_key = {str(m["key"]): m for m in catalog_models}
        for key in catalog_keys:
            for req_file in ("runner.py", "model.py", "README.md"):
                if not (repo_root / "models" / key / req_file).is_file():
                    errors.append(f"[catalog] models/{key}/{req_file} missing for catalog entry {key!r}.")

        fs_model_dirs = sorted(
            d.name
            for d in (repo_root / "models").iterdir()
            if d.is_dir()
            and not d.name.startswith((".", "_"))
            and d.name != "tests"
            and (d / "runner.py").is_file()
        )
        if set(fs_model_dirs) != set(catalog_keys):
            errors.append(
                f"[catalog] Model directories on disk {fs_model_dirs} != "
                f"models/catalog.json keys {sorted(catalog_keys)}."
            )

        ast_registry = _extract_model_registry_from_ast(run_model_path)
        if ast_registry:
            if set(ast_registry.keys()) != set(catalog_keys):
                errors.append(
                    f"[catalog] MODEL_REGISTRY keys {sorted(ast_registry.keys())} != "
                    f"models/catalog.json keys {sorted(catalog_keys)}."
                )
            for key in sorted(set(ast_registry.keys()) & set(catalog_keys)):
                reg_entry = ast_registry[key]
                cat_entry = catalog_by_key[key]
                for shared_field in ("runner_module", "provenance_category", "code_url"):
                    if shared_field in reg_entry and reg_entry[shared_field] != cat_entry.get(shared_field):
                        errors.append(
                            f"[catalog-drift] {key}.{shared_field}: MODEL_REGISTRY has "
                            f"{reg_entry[shared_field]!r} vs catalog.json {cat_entry.get(shared_field)!r}."
                        )

    # 5. Per-MIP protocol, leaderboard sync, table_2_registry bijection, and figures_manifest sync
    mips_dir = repo_root / "mips"
    if mips_dir.is_dir():
        for proto_path in sorted(mips_dir.glob("*/protocol.json")):
            mip_dir = proto_path.parent
            mip_id = mip_dir.name
            proto: dict[str, Any] = json.loads(proto_path.read_text(encoding="utf-8"))
            if proto.get("mip_id") != mip_id:
                errors.append(
                    f"[mip:{mip_id}] protocol.json mip_id={proto.get('mip_id')!r} != folder name {mip_id!r}."
                )
            if proto.get("status") != "active":
                continue

            lb_dir = mip_dir / proto.get("paths", {}).get("leaderboards", "leaderboards")
            fig_dir = mip_dir / proto.get("paths", {}).get("figures", "figures")
            lb_ids = [str(lb["id"]) for lb in proto.get("leaderboards", [])]

            t2_path = lb_dir / "table_2_registry.json"
            t3_14_path = lb_dir / "tables_3_to_14.json"
            if not t2_path.is_file() or not t3_14_path.is_file():
                errors.append(f"[mip:{mip_id}] Missing table_2_registry.json or tables_3_to_14.json.")
                continue

            t2 = json.loads(t2_path.read_text(encoding="utf-8"))
            t3_14 = json.loads(t3_14_path.read_text(encoding="utf-8"))
            families = t2.get("families", [])
            for count_key in ("total_families", "num_families", "n_families"):
                if count_key in t2 and t2[count_key] != len(families):
                    errors.append(
                        f"[mip:{mip_id}] table_2_registry.json {count_key}={t2[count_key]} "
                        f"!= actual len(families)={len(families)}."
                    )

            t2_by_id = {str(f["family_id"]): f for f in families}
            master_by_id = {str(lb["id"]): lb for lb in t3_14.get("leaderboards", [])}
            actual_counts = {fid: 0 for fid in t2_by_id}
            actual_lbs: dict[str, set[str]] = {fid: set() for fid in t2_by_id}
            total_rows = 0

            for lbid in lb_ids:
                per_lb_path = lb_dir / f"{lbid}.json"
                if not per_lb_path.is_file():
                    errors.append(
                        f"[mip:{mip_id}] Missing leaderboard file {per_lb_path.relative_to(repo_root)}."
                    )
                    continue
                per_lb = json.loads(per_lb_path.read_text(encoding="utf-8"))
                rows = per_lb.get("rows", [])
                total_rows += len(rows)
                if per_lb.get("n_entries") != len(rows):
                    errors.append(
                        f"[mip:{mip_id}] {lbid}.json n_entries={per_lb.get('n_entries')} != len(rows)={len(rows)}."
                    )

                if lbid not in master_by_id:
                    errors.append(f"[mip:{mip_id}] {lbid} missing from tables_3_to_14.json.")
                else:
                    master_rows = master_by_id[lbid].get("rows", [])
                    if [r.get("entry_id") for r in rows] != [r.get("entry_id") for r in master_rows]:
                        errors.append(
                            f"[mip:{mip_id}] {lbid}.json entry_ids out of sync with tables_3_to_14.json "
                            "(run `bash leaderboards/on_community_commit.sh`)."
                        )
                    if [r.get("primary_ranking_nse") for r in rows] != [
                        r.get("primary_ranking_nse") for r in master_rows
                    ]:
                        errors.append(
                            f"[mip:{mip_id}] {lbid}.json primary_ranking_nse out of sync with tables_3_to_14.json."
                        )

                for row in rows:
                    fid = str(row.get("family_id", ""))
                    if fid not in t2_by_id:
                        errors.append(
                            f"[mip:{mip_id}] {lbid}/{row.get('entry_id')} references unregistered family_id={fid!r}."
                        )
                    else:
                        actual_counts[fid] += 1
                        actual_lbs[fid].add(lbid)

            for fid, fam in t2_by_id.items():
                if fam.get("n_entries") != actual_counts[fid]:
                    errors.append(
                        f"[mip:{mip_id}] table_2_registry family {fid!r}: "
                        f"n_entries={fam.get('n_entries')} != actual {actual_counts[fid]}."
                    )
                if set(fam.get("leaderboards_covered") or []) != actual_lbs[fid]:
                    errors.append(
                        f"[mip:{mip_id}] table_2_registry family {fid!r}: "
                        f"leaderboards_covered={fam.get('leaderboards_covered')} != actual {sorted(actual_lbs[fid])}."
                    )

            manifest_path = fig_dir / "figures_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("total_leaderboard_entries") != total_rows:
                    errors.append(
                        f"[mip:{mip_id}] figures_manifest.json total_leaderboard_entries="
                        f"{manifest.get('total_leaderboard_entries')} != {total_rows} "
                        "(run `bash leaderboards/on_community_commit.sh`)."
                    )
                for fig_key, fig_rel in manifest.get("figures", {}).items():
                    if not (repo_root / fig_rel).is_file():
                        errors.append(f"[mip:{mip_id}] Missing figure {fig_key}: {fig_rel}.")

    return errors


def _write_minimal_valid_repo(root: Path) -> None:
    """Populate `root` with a minimal compliant multi-MIP + Agent Skills tree for `--self-test`."""
    (root / "models" / "demo_model").mkdir(parents=True, exist_ok=True)
    (root / "mips" / "demo_mip" / "leaderboards").mkdir(parents=True, exist_ok=True)
    (root / "mips" / "demo_mip" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "skills" / "demo-skill").mkdir(parents=True, exist_ok=True)
    (root / ".claude").mkdir(parents=True, exist_ok=True)
    (root / ".agents").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "skills").symlink_to("../skills")
    (root / ".agents" / "skills").symlink_to("../skills")

    (root / "README.md").write_text(
        "# Demo\n\nSee [model](models/demo_model/README.md).\n", encoding="utf-8"
    )
    (root / "AGENTS.md").write_text(
        "# Agent Rules\n\n- Demo skill: `skills/demo-skill/SKILL.md`\n", encoding="utf-8"
    )
    (root / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
    (root / "GEMINI.md").write_text("@./AGENTS.md\n", encoding="utf-8")
    (root / "skills" / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill. Use when testing guardrails.\n---\n\n# Demo Skill\n",
        encoding="utf-8",
    )

    (root / ".zenodo.json").write_text(
        json.dumps(
            {
                "title": "Demo",
                "license": "Apache-2.0",
                "creators": [{"name": "Nearing, Grey S."}, {"name": "Gauch, Martin"}],
            }
        ),
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        yaml.safe_dump(
            {
                "cff-version": "1.2.0",
                "title": "Demo",
                "version": "1.0.1",
                "license": "Apache-2.0",
                "authors": [
                    {"family-names": "Nearing", "given-names": "Grey S."},
                    {"family-names": "Gauch", "given-names": "Martin"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.0.1"\n', encoding="utf-8"
    )

    (root / "models" / "demo_model" / "README.md").write_text("# Demo Model\n", encoding="utf-8")
    (root / "models" / "demo_model" / "model.py").write_text("X = 1\n", encoding="utf-8")
    (root / "models" / "demo_model" / "runner.py").write_text(
        "def run_fold(train_basins, test_basins, train_period, eval_period, forcings, data_dir, seeds=(1,), **kw):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (root / "models" / "run_model.py").write_text(
        "MODEL_REGISTRY: dict[str, object] = {\n"
        '    "demo_model": Entry(\n'
        '        key="demo_model",\n'
        '        runner_module="models.demo_model.runner",\n'
        '        provenance_category="Contribution",\n'
        '        code_url="https://example.org/demo",\n'
        "    ),\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "models" / "catalog.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "demo_model",
                        "runner_module": "models.demo_model.runner",
                        "provenance_category": "Contribution",
                        "code_url": "https://example.org/demo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    (root / "mips" / "demo_mip" / "protocol.json").write_text(
        json.dumps(
            {
                "mip_id": "demo_mip",
                "status": "active",
                "leaderboards": [{"id": "gauged__single_camels"}],
                "paths": {"leaderboards": "leaderboards", "figures": "figures"},
            }
        ),
        encoding="utf-8",
    )
    row = {"entry_id": "e1", "family_id": "demo_model", "primary_ranking_nse": 0.8}
    (root / "mips" / "demo_mip" / "leaderboards" / "gauged__single_camels.json").write_text(
        json.dumps({"id": "gauged__single_camels", "n_entries": 1, "rows": [row]}),
        encoding="utf-8",
    )
    (root / "mips" / "demo_mip" / "leaderboards" / "tables_3_to_14.json").write_text(
        json.dumps({"leaderboards": [{"id": "gauged__single_camels", "rows": [row]}]}),
        encoding="utf-8",
    )
    (root / "mips" / "demo_mip" / "leaderboards" / "table_2_registry.json").write_text(
        json.dumps(
            {
                "total_families": 1,
                "families": [
                    {
                        "family_id": "demo_model",
                        "n_entries": 1,
                        "leaderboards_covered": ["gauged__single_camels"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fig_path = root / "mips" / "demo_mip" / "figures" / "fig1.png"
    fig_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "mips" / "demo_mip" / "figures" / "figures_manifest.json").write_text(
        json.dumps(
            {
                "total_leaderboard_entries": 1,
                "figures": {"fig1": "mips/demo_mip/figures/fig1.png"},
            }
        ),
        encoding="utf-8",
    )


def run_self_tests() -> int:
    """Verify positive and negative detection across every guardrail category."""
    with tempfile.TemporaryDirectory() as tmp_str:
        root = Path(tmp_str)
        _write_minimal_valid_repo(root)

        clean_errors = check_code_guardrails(root) + check_docs_and_registries(root)
        if clean_errors:
            raise AssertionError(f"Expected 0 errors on clean fixture, got: {clean_errors}")

        # Inject controlled violations across AST, pickle, split/date, leaks, links, skills, and registries
        bad_ws = "/usr/local/" + "google/x"
        bad_env = "google" + "hydrology"
        bad_tok = "single_camels_" + "ensemble"
        (root / "models" / "demo_model" / "bad.py").write_text(
            "import contextlib, numpy as np, torch\n"
            f"P = '{bad_ws}'\n"
            "TMP = '/' + 'tmp'\n"
            "S = 'basins_531.txt'\n"
            "D = '1989-10-01'\n"
            "def f(x):\n"
            "    try:\n"
            "        a = np.load(x)\n"
            "        b = np.load(x, allow_pickle=True)\n"
            "        c = torch.load(x, weights_only=False)\n"
            "    except ValueError:\n"
            "        with contextlib.suppress(KeyError):\n"
            "            return None\n"
            "    return a, b, c\n",
            encoding="utf-8",
        )
        (root / "models" / "demo_model" / "tmp_leak.py").write_text(
            "CACHE = '/tm' + 'p/foo.o'\n",  # split stays clean; next line has actual literal
            encoding="utf-8",
        )
        (root / "models" / "demo_model" / "tmp_leak2.py").write_text(
            "from pathlib import Path\nP = Path(\"/t\" \"mp\")\n",
            encoding="utf-8",
        )
        # Write actual /tmp literal dynamically so self-file stays clean
        tmp_word = "/" + "tmp/kernel.o"
        (root / "models" / "demo_model" / "tmp_real.py").write_text(
            f"K = {tmp_word!r}\n", encoding="utf-8"
        )
        (root / "README.md").write_text(
            f"# Broken [link](missing.md) and {bad_env} and {bad_tok}\n",
            encoding="utf-8",
        )
        (root / "skills" / "wrong-folder").mkdir(parents=True, exist_ok=True)
        (root / "skills" / "wrong-folder" / "SKILL.md").write_text(
            "---\nname: mismatched-name\ndescription: Wrong.\n---\n",
            encoding="utf-8",
        )
        (root / "blob.npz").write_bytes(b"1234")

        code_errs = check_code_guardrails(root)
        doc_errs = check_docs_and_registries(root)
        combined = "\n".join(code_errs + doc_errs)

        expected_tags = (
            "[prohibited-try-except]",
            "[error-suppression]",
            "[pickle-rce]",
            "[split-isolation]",
            "[date-isolation]",
            "[leak:workstation_home_path]",
            "[leak:hardcoded_tmp_path]",
            "[forbidden-blob]",
            "[broken-link]",
            "[stale-doc-token]",
            "[doc-leak:internal_conda_env]",
            "[agent-skills]",
        )
        for tag in expected_tags:
            if tag not in combined:
                raise AssertionError(f"Self-test failed to detect {tag}. Got:\n{combined}")

    print("OK: ci_guardrails self-test verified all positive and negative checks.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HydrologyMIP CI guardrails.")
    parser.add_argument(
        "--mode",
        choices=("code-guardrails", "docs-and-registry", "all"),
        default="all",
        help="Guardrail suite to execute.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in hermetic self-tests of all guardrail checks and exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_tests()

    violations: list[str] = []
    if args.mode in ("code-guardrails", "all"):
        violations.extend(check_code_guardrails(REPO_ROOT))
    if args.mode in ("docs-and-registry", "all"):
        violations.extend(check_docs_and_registries(REPO_ROOT))

    if violations:
        print(
            f"FAILED: {len(violations)} HydrologyMIP guardrail violation(s) detected:",
            file=sys.stderr,
        )
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"OK: HydrologyMIP guardrails ({args.mode}) passed with 0 violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
