#!/usr/bin/env python3
"""Report what the coordinator actually routes to — per model and role, offline.

Complements the submission gate `audit_head_routing_diversity` (which inspects head
*weights*): this reads a coordinator's **decision log** — the `(model, role)` picks
it made across each benchmark's turns — and reports each model's and role's usage
share, the normalized entropy of the model distribution, unused pool models/roles,
and a collapse flag when one model dominates. Zero API cost.

Input JSON (`--decisions`), either shape:

    {"math500": [["glm-5p2", "worker"], ["deepseek-v4-pro", "verifier"], ...],
     "mmlu":    [{"model": "glm-5p2", "role": "thinker"}, ...]}

Optionally pass `--pool glm-5p2,deepseek-v4-pro,minimax-m3` to surface pool models
that were never routed to.

    python scripts/routing_composition_report.py --decisions decisions.json

Exits non-zero when any benchmark's router is collapsed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.routing_composition import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of benchmark -> [decisions]")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the routing-composition report; exit non-zero if any router collapsed."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decisions", required=True, type=Path,
                    help="JSON of benchmark -> list of (model, role) decisions")
    ap.add_argument("--pool", default=None,
                    help="comma-separated full model pool (to flag never-used models)")
    ap.add_argument("--collapse-threshold", type=float, default=0.90,
                    dest="collapse_threshold",
                    help="a model's share at/above this flags the router as collapsed")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    per_benchmark = _load(args.decisions)
    pool = args.pool.split(",") if args.pool else None
    report = analyze_benchmarks(per_benchmark, pool_models=pool,
                                collapse_threshold=args.collapse_threshold)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(per_benchmark, pool_models=pool,
                     collapse_threshold=args.collapse_threshold))
    return 1 if report["any_collapsed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
