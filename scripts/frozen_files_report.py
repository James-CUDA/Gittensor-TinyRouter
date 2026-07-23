#!/usr/bin/env python3
"""Flag frozen-file modifications in a submission PR. Zero API cost.

``docs/COMPETITION_RULES.md`` enforces every anti-cheat row with a numbered gate
except one — *"Modifying frozen files in a submission PR | Rejected by
maintainer"*. This runs that check mechanically over a changed-file list.

Feed it ``git diff --name-only``::

    git diff --name-only origin/main... | python scripts/frozen_files_report.py --changed -
    python scripts/frozen_files_report.py --changed changed.txt --json

Set ``TINYROUTER_BENCHMARK_DIR`` to also cover the encrypted hidden benchmarks;
without it that rule is inert (there is no directory to protect).

Exits 1 when any frozen file was modified, 0 when clean, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.submission.frozen_files import (  # noqa: E402
    audit_frozen_files,
    frozen_violations,
)


def _read_paths(source: str) -> list[str]:
    if source == "-":
        text = sys.stdin.read()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        text = path.read_text()
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    """Print the frozen-file report; exit 1 when any frozen file was touched."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--changed", required=True,
        help="file with one changed path per line, or '-' for stdin",
    )
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = ap.parse_args(argv)

    try:
        paths = _read_paths(args.changed)
    except FileNotFoundError as exc:
        print(f"no such file: {exc}", file=sys.stderr)
        return 2

    hits = frozen_violations(paths)
    message = audit_frozen_files(paths)

    if args.as_json:
        print(json.dumps({
            "checked": len(paths),
            "violations": [h.to_dict() for h in hits],
            "message": message,
        }, indent=2))
    elif hits:
        print(f"FROZEN FILES MODIFIED ({len(hits)} of {len(paths)} changed paths)")
        for h in hits:
            print(f"  {h.path}")
            print(f"      matched {h.rule.pattern} — {h.rule.reason}")
        print()
        print("COMPETITION_RULES.md lists these as frozen; modifying them in a")
        print("submission PR is cheating. In a feature PR this is expected — the")
        print("rule is scoped to submission PRs.")
    else:
        print(f"no frozen files touched ({len(paths)} changed paths checked)")

    return 1 if hits else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
