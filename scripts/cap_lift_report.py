#!/usr/bin/env python3
"""Verify SPEC R6 offline: does lifting the token cap jump accuracy past every constituent?

Reads a JSON token-cap sweep -- the routed accuracy at several per-turn output-token
caps on a benchmark, plus the constituent single-model accuracies at the lifted cap --
and reports the lift jump, whether accuracy is monotone non-decreasing in the cap, the
margin over the best constituent, and the R6 verdict. Zero API cost.

Input JSON (``--sweep``):

    {"benchmark": "livecodebench",
     "caps": {"4096": 0.615, "uncapped": 0.862},
     "constituents": {"gpt5": 0.838, "gemini": 0.672, "claude": 0.465}}

A cap key may be a number or "uncapped"/"inf"/null for the lifted (removed) budget.

    python scripts/cap_lift_report.py --sweep cap_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.cap_lift import analyze_cap_lift, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "caps" not in data:
        raise ValueError(f"{path}: expected an object with a 'caps' mapping")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R6 report; exit non-zero when R6 is not shown."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", required=True, type=Path,
                    help="JSON with 'caps' {cap: accuracy} and optional 'constituents'")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    data = _load(args.sweep)
    benchmark = str(data.get("benchmark", "livecodebench"))
    caps = data["caps"]
    constituents = data.get("constituents")
    r = analyze_cap_lift(caps, constituents, benchmark=benchmark)
    if args.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(render(caps, constituents, benchmark=benchmark), end="")
    return 0 if r.holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
