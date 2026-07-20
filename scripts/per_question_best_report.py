#!/usr/bin/env python3
"""Verify SPEC R5 offline: does TRINITY reach the Per-Question-Best ceiling?

Reads a JSON of per-benchmark TRINITY accuracy and the Per-Question-Best (PQB,
"union of correct answers across the pool") ceiling, and reports the gap
``PQB - TRINITY`` per benchmark, whether TRINITY is within tolerance of the
ceiling ("≈"), and the R5 verdict across benchmarks. Zero API cost.

The PQB ceiling is the routing/union oracle that ``trinity.analysis.union_oracle``
/ ``scripts/oracle_ceiling.py`` already estimate from the cached solve matrix; this
report consumes it and adds the TRINITY comparison R5 needs.

Input JSON (``--targets``):

    {"livecodebench": {"trinity": 0.845, "per_question_best": 0.862, "baseline": 0.61},
     "math500":       {"trinity": 0.55,  "per_question_best": 0.60},
     "mmlu":          {"trinity": 0.70,  "per_question_best": 0.71}}

    python scripts/per_question_best_report.py --targets r5_targets.json

Exits non-zero when R5 is violated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.per_question_best import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected an object of benchmark -> {{trinity, per_question_best}}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R5 report; exit non-zero when R5 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", required=True, type=Path,
                    help="JSON of benchmark -> {trinity, per_question_best[, baseline]}")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="'≈' tolerance: TRINITY reaches the ceiling if PQB - TRINITY <= tol")
    ap.add_argument("--min-pass", type=int, default=None, dest="min_pass",
                    help="benchmarks that must reach the ceiling (default: all but one)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    targets = _load(args.targets)
    report = analyze_benchmarks(targets, tol=args.tol, min_pass=args.min_pass)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(targets, tol=args.tol, min_pass=args.min_pass))
    return 0 if report["r5_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
