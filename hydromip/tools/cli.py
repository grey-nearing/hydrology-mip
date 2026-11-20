# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``hydromip`` CLI dispatcher (e.g. ``hydromip package``).

Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_SUBCOMMAND_MODULES: dict[str, str] = {
    "package": "hydromip.tools.package_submission",
    "process-submission": "hydromip.tools.process_submission",
    "guardrails": "hydromip.tools.ci_guardrails",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``hydromip <subcommand>`` calls lazily so ``--help`` needs zero heavy deps."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(
            "usage: hydromip {package,process-submission,guardrails} [args...]\n\n"
            "Subcommands:\n"
            "  package             Package hydrograph predictions into canonical CAMELS-US Zarr v2\n"
            "  process-submission  Score a community submission manifest and update leaderboards\n"
            "  guardrails          Run AST, security, docs, and registry CI checks"
        )
        return 0

    subcmd, rest = args[0], args[1:]
    mod_path = _SUBCOMMAND_MODULES.get(subcmd)
    if mod_path is not None:
        mod = importlib.import_module(mod_path)
        return int(mod.main(rest))

    raise ValueError(
        f"Unknown hydromip subcommand {subcmd!r}. Expected one of: package, process-submission, guardrails."
    )


if __name__ == "__main__":
    raise SystemExit(main())
