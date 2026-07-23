#!/usr/bin/env python3
"""Report per-model routed accuracy — is the head sending questions to the right model?

Complements `trinity.analysis.sampling` (which measures each model's *intrinsic* solve
rate over all questions): this reads a run's `(routed_model, correct)` outcomes — the
model the head routed the answering turn to, and whether the trajectory was correct —
and reports, per model, how many questions it was routed (share), its routed accuracy
on them, and its contribution (share x accuracy) to overall accuracy, plus the model
the head over-uses relative to its routed accuracy. Zero API cost.

Input JSON (`--outcomes`), either shape:

    {"records": [["glm-5p2", true], ["deepseek-v4-pro", false],
                 {"model": "minimax-m3", "correct": 1}]}

or a bare list of records. `correct` may be a bool, a 0/1, or a score (>0 => correct).

    python scripts/routed_accuracy_report.py --outcomes outcomes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.routed_accuracy import analyze, render  # noqa: E402


def _load(path: Path) -> list[Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of records or an object with 'records'")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the routed-accuracy report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outcomes", required=True, type=Path,
                    help="JSON list of (routed_model, correct) records")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    records = _load(args.outcomes)
    if args.json:
        print(json.dumps(analyze(records).to_dict(), indent=2))
    else:
        print(render(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
