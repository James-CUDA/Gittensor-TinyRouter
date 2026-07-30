#!/usr/bin/env python3
"""Verify SPEC R13 offline: TRINITY's relative-error-reduction over the best single agent.

Reads a JSON of per-task TRINITY accuracy and the best single-agent accuracy, and
reports the relative error reduction RER = (Z - S*) / (1 - S*) per task, the mean,
and whether TRINITY closes positive error on every task (the R13 invariant). The
absolute mean is ~21.9% in the paper (ballpark, pool-dependent). Zero API cost.

Input JSON (``--accuracies``), the single-agent bar is ``best_single`` (a scalar) or
``singles`` (a ``{model: score}`` map whose max is taken):

    {"math500": {"trinity": 0.88, "best_single": 0.80},
     "mmlu":    {"trinity": 0.92, "singles": {"gpt5": 0.90, "gemini": 0.88}}}

    python scripts/relative_error_reduction_report.py --accuracies r13.json

Exits non-zero when R13 is violated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.relative_error_reduction import analyze_tasks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected an object of task -> {{trinity, best_single}}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R13 report; exit non-zero when R13 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracies", required=True, type=Path,
                    help="JSON of task -> {trinity, best_single | singles}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    accs = _load(args.accuracies)
    report = analyze_tasks(accs)
    print(json.dumps(report, indent=2) if args.json else render(accs))
    return 0 if report["r13_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
