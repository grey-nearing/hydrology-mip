# Agent Skills (`skills/`)

This folder is the second documentation layer of HydrologyMIP. It holds step-by-step skills for AI
coding agents (and for humans who like checklists). The first layer, for humans, is the root
[README.md](../README.md), [CONTRIBUTING.md](../CONTRIBUTING.md), and the package and MIP READMEs.
Agents load the always-on rules in [AGENTS.md](../AGENTS.md) first.

## How agents find these skills

Each skill is a folder with a `SKILL.md` that starts with YAML frontmatter (`name`, `description`).
This follows the open Agent Skills format.

| Tool | Reads rules from | Finds skills in |
| :--- | :--- | :--- |
| Claude Code | `CLAUDE.md` (imports `AGENTS.md`) | `.claude/skills/` (symlink to `skills/`) |
| OpenAI Codex | `AGENTS.md` | `.agents/skills/` (symlink to `skills/`) |
| Gemini CLI | `GEMINI.md` (imports `AGENTS.md`) | `.agents/skills/` |
| GitHub Copilot, Cursor | `AGENTS.md` | `.agents/skills/` or `.claude/skills/` |
| Other tools | `AGENTS.md` | open the paths listed in `AGENTS.md` |

On Windows, git needs symlink support (`core.symlinks=true`). Without it, open the files directly
from `skills/`.

## Skills

| Skill | Audience | Use it to |
| :--- | :--- | :--- |
| [repo-structure](repo-structure/SKILL.md) | everyone | find where things live and which files never to edit |
| [protocol-guardrails](protocol-guardrails/SKILL.md) | everyone | follow the benchmark protocol, anti-leakage rules, and data-integrity rules |
| [contribute-model](contribute-model/SKILL.md) | contributors | add a runnable model under `models/<name>/`, run it, and score it locally |
| [contribute-hydrographs](contribute-hydrographs/SKILL.md) | contributors | package hydrographs from external code and submit them |
| [contribute-mip](contribute-mip/SKILL.md) | contributors | add a new benchmark under `mips/<mip_id>/` |
| [leaderboard-norms](leaderboard-norms/SKILL.md) | contributors, reviewers | label provenance, citations, ensembles, and leaderboard routing |
| [data-schemas](data-schemas/SKILL.md) | everyone | look up store, split, leaderboard, registry, and output schemas |
| [testing](testing/SKILL.md) | everyone | write and run tests that CI accepts |
| [docs-style](docs-style/SKILL.md) | everyone | write docs and skills that follow the two-layer rules |
| [pr-review](pr-review/SKILL.md) | reviewers, contributors | review a PR, or check your own before opening it |
| [maintainer-gcs-bucket](maintainer-gcs-bucket/SKILL.md) | maintainers only | manage the private archive bucket safely |
| [maintainer-ingest-submission](maintainer-ingest-submission/SKILL.md) | maintainers only | turn an accepted submission into leaderboard rows |
| [maintainer-audit](maintainer-audit/SKILL.md) | maintainers only | audit models and leaderboards, and prepare releases |

## Adding or changing a skill

Follow [docs-style](docs-style/SKILL.md):

1. Create one folder per skill, and make `name` equal to the folder name.
2. Write a `description` that says when to use the skill.
3. Set `metadata.audience`.
4. Update this table and the table in `AGENTS.md` in the same PR.

`python -m hydromip.tools.ci_guardrails --mode all` checks the layout.
