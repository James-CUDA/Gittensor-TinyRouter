#!/usr/bin/env python3
"""Audit an on-disk LLM response cache ($TRINITY_LLM_CACHE): value + pollution.

Walks a cache directory and reports, per model, the entry count / tokens and the dollar
value if every entry is re-served once, plus how many entries are polluted (error/blank
completions that would re-serve a 0 score). Zero API cost.

    python scripts/cache_audit_report.py --cache .cache/llm
    TRINITY_LLM_CACHE=.cache/llm python scripts/cache_audit_report.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.llm.cache import CACHE_ENV_VAR  # noqa: E402
from trinity.llm.cache_audit import audit_cache, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the cache audit; exit non-zero when the cache holds polluted entries."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=None,
                    help=f"cache directory (default: ${CACHE_ENV_VAR})")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = args.cache or os.environ.get(CACHE_ENV_VAR, "").strip()
    if not root:
        ap.error(f"provide --cache DIR or set ${CACHE_ENV_VAR}")
    audit = audit_cache(root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2))
    else:
        print(render(audit), end="")
    return 1 if audit.n_polluted else 0


if __name__ == "__main__":
    raise SystemExit(main())
