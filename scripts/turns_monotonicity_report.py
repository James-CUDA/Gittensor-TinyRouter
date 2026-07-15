#!/usr/bin/env python3
"""Verify SPEC R7 offline: does accuracy rise monotonically with the turn budget?

Reads a JSON turn sweep -- the accuracy of one trained head evaluated at several
``max_turns`` budgets, per benchmark -- and reports whether accuracy is monotone
non-decreasing in the budget (never falling), the net gain from the smallest to
the largest budget, and any downward step (an R7 violation). Zero API cost.

Input JSON (``--sweeps``):

    {"livecodebench": {"2": 0.823, "4": 0.845, "6": 0.863},
     "math500":       {"2": 0.50,  "6": 0.55}}

    python scripts/turns_monotonicity_report.py --sweeps turns_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.turns_monotonicity import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of benchmark -> {{turns: accuracy}}")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R7 report; exit non-zero when R7 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweeps", required=True, type=Path,
                    help="JSON of benchmark -> {max_turns: accuracy}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    sweeps = _load(args.sweeps)
    report = analyze_benchmarks(sweeps)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(sweeps))
    return 0 if report["r7_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
