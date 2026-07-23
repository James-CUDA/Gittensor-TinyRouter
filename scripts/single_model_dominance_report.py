#!/usr/bin/env python3
"""Verify SPEC R1/R2 offline: does TRINITY beat the single models, on average and per task?

Reads a JSON of per-task TRINITY accuracy and the single-model accuracies, and
reports **R2** (TRINITY beats every single model on every task) and **R1** (TRINITY's
mean > the best single model's mean). Zero API cost.

Input JSON (``--accuracies``):

    {"math500": {"trinity": 0.90, "singles": {"gpt5": 0.85, "gemini": 0.80}},
     "mmlu":    {"trinity": 0.92, "singles": {"gpt5": 0.88, "gemini": 0.86}}}

    python scripts/single_model_dominance_report.py --accuracies r1r2.json

Exits non-zero unless both R1 and R2 hold; ``--r2-only`` / ``--r1-only`` gate on one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.single_model_dominance import analyze_tasks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of task -> {{trinity, singles}}")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R1/R2 report; exit non-zero when the selected invariant is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracies", required=True, type=Path,
                    help="JSON of task -> {trinity, singles: {model: score}}")
    ap.add_argument("--r1-only", action="store_true", help="gate on R1 (average) only")
    ap.add_argument("--r2-only", action="store_true", help="gate on R2 (per task) only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    accs = _load(args.accuracies)
    report = analyze_tasks(accs)
    print(json.dumps(report, indent=2) if args.json else render(accs))
    if args.r1_only:
        holds = report["r1_holds"]
    elif args.r2_only:
        holds = report["r2_holds"]
    else:
        holds = report["r1r2_holds"]
    return 0 if holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
