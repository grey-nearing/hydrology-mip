"""Convenience entrypoint in `figures/` that invokes `leaderboards.generate_figures`."""

from __future__ import annotations

from leaderboards.generate_figures import generate_all_paper_figures, main

__all__ = ["generate_all_paper_figures", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
