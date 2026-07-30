#!/usr/bin/env python3
"""Verify SPEC R8 offline: sep-CMA-ES > SFT > RS > REINFORCE on every task?

Reads a JSON of per-task final fitness for each optimizer and reports, per task,
whether the observed best->worst order matches the expected chain plus the R8
verdict. Zero API cost. SPEC §5.4 flags R8 as "a hypothesis to test, not a given".

Input JSON (``--scores``), optimizer keys are canonicalized (``Sep-CMA-ES``,
``random search``, ``policy_gradient`` all resolve):

    {"math500": {"sep_cmaes": 0.72, "sft": 0.66, "rs": 0.60, "reinforce": 0.55},
     "mmlu":    {"sep_cmaes": 0.68, "sft": 0.60, "rs": 0.58, "reinforce": 0.50}}

    python scripts/optimizer_ranking_report.py --scores r8.json

Exits non-zero when R8 is violated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.optimizer_ranking import analyze_tasks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of task -> {{optimizer: fitness}}")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R8 report; exit non-zero when R8 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", required=True, type=Path,
                    help="JSON of task -> {optimizer: final_fitness}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    scores = _load(args.scores)
    report = analyze_tasks(scores)
    print(json.dumps(report, indent=2) if args.json else render(scores))
    return 0 if report["r8_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
