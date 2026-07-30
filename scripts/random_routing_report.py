#!/usr/bin/env python3
"""Verify SPEC R4 offline: does TRINITY beat random routing?

Reads a JSON of per-benchmark TRINITY accuracy and the random-routing baseline
accuracy, and reports whether TRINITY strictly beats random routing per benchmark
plus the R4 verdict. Zero API cost.

Input JSON (``--accuracies``), the baseline key is ``random_routing`` or ``random``:

    {"math500": {"trinity": 0.88, "random_routing": 0.30},
     "rlpr":    {"trinity": 0.41, "random_routing": 0.32}}

    python scripts/random_routing_report.py --accuracies r4.json

Exits non-zero when R4 is violated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.random_routing import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected an object of benchmark -> {{trinity, random_routing}}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R4 report; exit non-zero when R4 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracies", required=True, type=Path,
                    help="JSON of benchmark -> {trinity, random_routing}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    accs = _load(args.accuracies)
    report = analyze_benchmarks(accs)
    print(json.dumps(report, indent=2) if args.json else render(accs))
    return 0 if report["r4_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
