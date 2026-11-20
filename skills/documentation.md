---
name: documentation
description: >-
  Core documentation rules (focus strictly on what exists and how to use it;
  write concisely for a high-school reading level, non-experts, and ESL
  readers) and two-tier documentation architecture (root README.md and package
  README.md guides) for camels-us-mip. Use whenever adding, updating, or
  reviewing README files, docstrings, or reports.
---

# Documentation Strategy & Standards (`camels-us-mip`)

## 1. Core High-Level Rules (Mandatory)

### Rule 1 — Document Only What Exists and How to Use It
- **Focus strictly on the present system and how to use it:** Every README, usage guide, docstring, and report must describe **only what currently exists in the repository and how a user runs or interacts with it**.
- **Never expose development history or design decisions:** Do **not** explain the decisions made to arrive at the current design, the path or iterations taken to get here, what approaches were previously tried or replaced, or why alternative designs were rejected.
- **Never expose internal coding rules to users:** Do **not** clutter user-facing documentation with internal engineering or agent rules (for example, no user cares that we enforce a rule against `try`/`except` blocks—that belongs in [`skills/algorithm-rules-and-norms.md`](./algorithm-rules-and-norms.md), not in user documentation). Users only need to know what the software does, what inputs it requires, and what outputs or errors it produces.

### Rule 2 — Target Human Readability Over Technical Proficiency
- **Target reader profile:** Write all documentation for a reader at a **high school reading level** who is a **non-expert** in computer science, machine learning, mathematics, and hydrology.
- **Write for English as a Second Language (ESL) readers:** Many or most readers of this repository have **English as a second language**. Use short, direct sentences, active voice, and plain, everyday words. Avoid idioms, colloquialisms, complex nested clauses, or dense academic/software jargon. When a technical or hydrological term is required, define it simply in plain language at point of use.
- **Be strictly concise (never verbose):** Keep explanations as short and direct as possible. Cut filler paragraphs, repetitive explanations, and unnecessary theoretical background. Prefer brief bullet points, tables, and copy-pasteable commands over long blocks of text.

---

## 2. Two-Tier Documentation Architecture

### Tier 1 — Concise Top-Level [`README.md`](../README.md)
- Keep the root `README.md` short, scannable, and focused on:
  - Brief overview of the CAMELS-US Model Intercomparison Project (`camels-us-mip`) and the $3 \times 4 = 12$ leaderboards on `hydromip.org`.
  - Installation (`pip install -e .`).
  - Copy-pasteable quick-start commands for:
    1. Running any model via `models/run_model.py`,
    2. Calculating metrics for a model and leaderboard from `gs://camels-us-mip/` via `metrics/`, and
    3. Launching the leaderboard web application in `frontend/`.
  - Direct links to each package `README.md` (`splits/README.md`, `metrics/README.md`, `models/README.md`, `frontend/README.md`, `benchmarks/README.md`) and `LICENSE_ASSESSMENT.md`.

### Tier 2 — Self-Contained Package `README.md`s
Every top-level code package (`splits/README.md`, `metrics/README.md`, `models/README.md`, `frontend/README.md`, `benchmarks/README.md`) must have its own self-contained `README.md` containing:
1. **Overview & Summary Table:** Summarize the CLI commands, Python API, inputs, and outputs.
2. **Prerequisites & Credentials:** Show environment setup and Google Cloud Storage authentication for `gs://camels-us-mip/`.
3. **Copy-Pasteable Quick Start Examples:** Provide short, copy-pasteable CLI and Python examples.
4. **Complete Arguments Reference:** Document every CLI flag and public API argument, its type, and what it does in plain language.

---

## 3. Formatting & Terminology Norms

- **State Units and Formats Explicitly:** Always state physical units and formats clearly (for example, `mm/day`, $\text{km}^2$, UTC/local dates in `YYYY-MM-DD`, 8-digit USGS gauge IDs).
- **Use Standard Domain Terms With Plain Explanations:** Use standard hydrological terms (NSE, KGE, FHV, PUB, PUR, HUC-02), and briefly state what they mean in plain words so non-experts can follow.
- **Clean Visual Formatting:** Avoid excessive inline bolding in paragraphs. Use Markdown tables, code blocks, and short headings to organize information cleanly.
