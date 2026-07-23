#!/usr/bin/env python3
"""Verify SPEC R3 offline: does TRINITY beat the best multi-agent baseline?

Reads a JSON of per-benchmark TRINITY accuracy and the multi-agent baseline
accuracies (MoA / MasRouter / RouterDC / Smoothie), and reports whether TRINITY
strictly beats the strongest baseline per benchmark plus the R3 verdict. Zero API
cost.

Input JSON (``--accuracies``):

    {"math500": {"trinity": 0.88, "baselines": {"MoA": 0.66, "MasRouter": 0.68}},
     "mmlu":    {"trinity": 0.92, "baselines": {"RouterDC": 0.60, "Smoothie": 0.65}}}

    python scripts/multi_agent_baseline_report.py --accuracies r3.json

Exits non-zero when R3 is violated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.multi_agent_baseline import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected an object of benchmark -> {{trinity, baselines}}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R3 report; exit non-zero when R3 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracies", required=True, type=Path,
                    help="JSON of benchmark -> {trinity, baselines: {name: score}}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    accs = _load(args.accuracies)
    report = analyze_benchmarks(accs)
    print(json.dumps(report, indent=2) if args.json else render(accs))
    return 0 if report["r3_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
