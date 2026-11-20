---
name: docs-style
description: >-
  Rules for writing HydrologyMIP documentation and agent skills: the two documentation layers and
  the no-duplication rule, plain-language style for non-experts and ESL readers, required README
  sections, branding, link and content rules checked by CI, and the skill format (frontmatter,
  folder name, AGENTS.md index). Use when adding, editing, or reviewing any README, CONTRIBUTING or
  SECURITY text, docstring, report, AGENTS.md, or SKILL.md.
license: Apache-2.0
metadata:
  audience: contributor
---

# Documentation and skill style

All paths are relative to the repository root.

## 1. Two layers, no duplication

| Layer | Files | Owns |
| :--- | :--- | :--- |
| 1. Humans | `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, package READMEs (`hydromip/`, `models/`, `metrics/`, `leaderboards/`, `benchmarks/`, `frontend/`), MIP READMEs (`mips/README.md`, `mips/<mip_id>/**/README.md`), per-model READMEs | What exists and how to use it: layout, protocol, commands, flags, schemas |
| 2. Agents | `AGENTS.md`, `skills/<name>/SKILL.md` (+ `references/`) | How to act: steps, MUST/NEVER rules, checks, when to stop and ask |

- A fact lives in **one** place. Skills link to the Layer-1 file instead of copying tables or
  numbers.
- Machine-readable files win over prose: `protocol.json`, `MODEL_REGISTRY` / `models/catalog.json`,
  `table_2_registry.json`, `tables_3_to_14.json`. **Do not write counts** of models, families, rows,
  or stores in prose. Point to the JSON file instead.
- `CLAUDE.md` and `GEMINI.md` contain only an import of `AGENTS.md`. Never add rules to them.

## 2. Writing style (Layer 1)

- Describe only what exists now and how to use it. No development history, no "we used to", no
  design debates, no internal engineering rules (those belong in skills).
- Write for a high-school reading level, non-experts, and readers whose first language is not
  English. Use short sentences, active voice, and common words. Define a technical term the first
  time you use it (for example NSE, PUB, PUR, HUC-02).
- Be concise. Prefer tables, bullet lists, and copy-paste commands to long paragraphs.
- Always give units and formats: mm/day, km², dates as `YYYY-MM-DD`, 8-digit USGS gauge IDs.
- Run every command from the repository root, and say so.

## 3. Required sections

- **Package README:** an overview table (commands, Python API, inputs, outputs); prerequisites
  (what to install, and what needs no cloud account); copy-paste examples; a complete
  argument reference that matches `--help`.
- **MIP README:** an overview; data sources with DOIs and licenses; the protocol rules in plain
  words; a map of regime IDs to folders.
- **Model README:** follow `skills/contribute-model/references/model_readme_template.md`.
- Steps that only maintainers can do (bucket writes, deployments) are labeled **Maintainers only**.

## 4. Branding and names

- Use **HydrologyMIP** and <https://hydrologymip.org> in prose. Use `hydromip` only for the Python
  package (`import hydromip`) and for `HYDROMIP_*` environment variables.
- The repository is `grey-nearing/camels-us-mip`. The active benchmark is `camels_us_daily`.
  Regime IDs are `gauged`, `pub_12fold`, and `pur_18huc02`; the PUR folder is
  `splits/pur_18fold/`.

## 5. Rules CI checks (`python -m hydromip.tools.ci_guardrails --mode all`)

- Every relative Markdown link resolves inside the repository. Paths inside backticks are not
  checked, so link real files only.
- No machine-specific paths, home directories, internal host names, private environment names,
  credentials, or tokens. This applies to code blocks and "do not" examples too; the guardrail tool
  lists the exact patterns.
- No identifiers that do not exist in the code. Check every function name, flag, and file path
  against the current code before you write it.

## 6. Skill format (Layer 2)

- One folder per skill: `skills/<name>/SKILL.md`, where `<name>` is lowercase kebab-case. Optional
  supporting files go in `skills/<name>/references/`. Every folder directly under `skills/` must be
  a skill.
- Start with YAML frontmatter:

```yaml
---
name: <name>            # must equal the folder name
description: >-
  What the skill covers, then "Use when ..." (at most 1024 characters).
license: Apache-2.0
metadata:
  audience: contributor # or reviewer, or maintainer
---
```

- The body has: the rules or steps; exact, checked commands; a **Verify** section; and a **Stop and
  ask a human if** section.
- In skills, write repository paths in backticks relative to the root. Do not use relative Markdown
  links, because skills are also read through the `.agents/skills` and `.claude/skills` symlinks.
- When you add, rename, or remove a skill, update the table in `AGENTS.md` and `skills/README.md` in
  the same PR. CI checks that `AGENTS.md` lists every `skills/<name>/SKILL.md`.
