#!/usr/bin/env python3
"""Measure what SPEC §4.5 transcript truncation costs a run. Zero API cost.

``roles.postprocess`` caps each turn's output and biases the kept text toward the
tail "so the final answer / verdict is preserved". This report checks that claim
against real turns and evaluates SPEC §4.5's own revisit trigger — *"revisit only
if transcripts overflow the SLM context"* — which nothing else measures.

Input JSON (``--turns``) is either a flat list of turn records::

    [{"role": "verifier", "raw_output": "...", "processed_output": "..."}, ...]

or an object mapping benchmark -> list of turn records, which reports per
benchmark plus a pooled ``all``::

    {"math500": [ ... ], "drop": [ ... ]}

Turn records use the ``TurnRecord`` field names (``role``, ``raw_output``,
``processed_output``); ``raw`` / ``processed`` are accepted as aliases.

    python scripts/transcript_budget_report.py --turns turns.json
    python scripts/transcript_budget_report.py --turns turns.json --json

Exits non-zero when transcripts overflow the context (SPEC §4.5 says revisit) or
when any verifier turn lost its committed verdict to truncation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.transcript_budget import (  # noqa: E402
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MAX_CHARS,
    analyze,
    analyze_benchmarks,
    render,
)


def _load(path: Path) -> Any:
    data = json.loads(path.read_text())
    if not isinstance(data, (list, dict)):
        raise ValueError(f"{path}: expected a list of turns or an object of benchmark -> turns")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the transcript-budget report; exit non-zero on a problem."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", type=Path, required=True, help="turn records JSON")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    ap.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"post-processing budget the run used (default {DEFAULT_MAX_CHARS})",
    )
    ap.add_argument(
        "--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS,
        help=f"SLM context window (default {DEFAULT_CONTEXT_TOKENS})",
    )
    ap.add_argument(
        "--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN,
        help=f"token estimate divisor (default {DEFAULT_CHARS_PER_TOKEN})",
    )
    args = ap.parse_args(argv)

    if not args.turns.exists():
        print(f"no such file: {args.turns}", file=sys.stderr)
        return 2

    try:
        data = _load(args.turns)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"could not read {args.turns}: {exc}", file=sys.stderr)
        return 2

    opts = dict(
        max_chars=args.max_chars,
        context_tokens=args.context_tokens,
        chars_per_token=args.chars_per_token,
    )

    if isinstance(data, dict):
        reports = analyze_benchmarks(data, **opts)
        if args.as_json:
            print(json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=2))
        else:
            for name, rep in reports.items():
                print(f"== {name} ==")
                print(render(report=rep))
                print()
        bad = any(r.revisit_recommended or r.verdict_losses for r in reports.values())
    else:
        rep = analyze(data, **opts)
        print(json.dumps(rep.to_dict(), indent=2) if args.as_json else render(report=rep))
        bad = bool(rep.revisit_recommended or rep.verdict_losses)

    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
