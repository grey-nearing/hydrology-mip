# Copyright 2026 HydrologyMIP Contributors
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``hydromip`` CLI dispatcher (e.g. ``hydromip package``).

Zero ``try``/``except`` blocks.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from hydromip.tools import ci_guardrails, package_submission, process_submission


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``hydromip <subcommand>`` calls to the matching tool module."""
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
    if subcmd == "package":
        return package_submission.main(rest)
    if subcmd == "process-submission":
        return process_submission.main(rest)
    if subcmd == "guardrails":
        return ci_guardrails.main(rest)

    raise ValueError(
        f"Unknown hydromip subcommand {subcmd!r}. Expected one of: package, process-submission, guardrails."
    )


if __name__ == "__main__":
    raise SystemExit(main())
