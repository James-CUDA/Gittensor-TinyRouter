#!/usr/bin/env python3
"""Check that docs/MILESTONE1.md matches the trinity.m1 code. Zero API cost.

MILESTONE1.md is the miner-facing contract — the ``< 1,000,000`` param budget,
``1``/day rate, ``≥ 0.02`` win margin, pack layout, and ``0.7·domain +
0.3·difficulty`` composite. Those numbers also live in ``trinity.m1.constants``
and the pack/metrics code. Nothing checked the two agree; a doc edit or a
constant change could split them, and a miner following the document would then
build a pack the validator rejects.

    python scripts/milestone1_conformance_report.py
    python scripts/milestone1_conformance_report.py --doc docs/MILESTONE1.md --json

Exits 1 on any mismatch, 0 when doc and code agree, 2 when the document cannot
be read or parsed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.m1.conformance import check, default_doc_path, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the conformance report; exit non-zero on mismatch."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doc", type=Path, default=None,
                    help="path to MILESTONE1.md (default: the repo's docs/MILESTONE1.md)")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = ap.parse_args(argv)

    doc_path = args.doc if args.doc is not None else default_doc_path()
    try:
        report = check(doc_path=doc_path)
    except FileNotFoundError:
        print(f"no such file: {doc_path}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"could not parse MILESTONE1.md: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2) if args.as_json else render(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
