#!/usr/bin/env python3
"""Check the role prompts still match SPEC §4.4 word-for-word. Zero API cost.

``roles/prompts.py`` claims its instruction text is "preserved word-for-word from
SPEC §4.4", with the ``{Q}`` / ``{C_prev}`` scaffold moved into the user message.
Nothing verified that claim, and the role prompts define what Thinker, Worker and
Verifier actually do — silent drift would make every published number stop
describing the documented system.

    python scripts/prompt_conformance_report.py
    python scripts/prompt_conformance_report.py --spec docs/SPEC.md --json

Exits 1 when any role's system prompt has drifted from the document (or the
scaffold labels are no longer delivered), 0 when conformant, 2 when the SPEC
cannot be read or §4.4 cannot be parsed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.roles.conformance import check, default_spec_path, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the conformance report; exit non-zero on drift."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, default=None,
                    help="path to SPEC.md (default: the repo's docs/SPEC.md)")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = ap.parse_args(argv)

    spec_path = args.spec if args.spec is not None else default_spec_path()
    try:
        report = check(spec_path=spec_path)
    except FileNotFoundError:
        print(f"no such file: {spec_path}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"could not parse SPEC §4.4: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2) if args.as_json else render(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
