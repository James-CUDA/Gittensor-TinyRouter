#!/usr/bin/env python3
"""Did a fugu run cost what we projected? Reconcile a projection against actual spend.

Reprojects the API cost from the eval parameters (via fugu.cost.estimate_eval_cost) and
compares it to the actual metered spend in a run's cost block (a fugu_baseline_*.json with
a "cost" block, or a bare CostMeter.report() dict). Reports per-field over/under-run. Zero
API cost.

    python scripts/fugu_cost_reconcile_report.py --actual fugu_baseline_math500.json \
        --workers deepseek-v4-flash,qwen3.5-35b-a3b --n-tasks 120 --reps 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.fugu.cost import estimate_eval_cost  # noqa: E402
from trinity.fugu.cost_reconcile import reconcile_projection, render  # noqa: E402


def _actual_cost_block(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    cost = data.get("cost")
    return cost if isinstance(cost, dict) else data     # accept a run file or a bare block


def main(argv: list[str] | None = None) -> int:
    """Print the reconciliation; exit non-zero when actual diverged beyond the tolerance."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--actual", required=True, type=Path,
                    help="run JSON with a 'cost' block (or a bare CostMeter.report dict)")
    ap.add_argument("--workers", required=True, help="comma-separated worker model names")
    ap.add_argument("--n-tasks", required=True, type=int)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    estimate = estimate_eval_cost(worker_names=workers, n_tasks=args.n_tasks, reps=args.reps)
    rec = reconcile_projection(estimate, _actual_cost_block(args.actual), tol=args.tol)
    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(render(rec), end="")
    return 0 if rec.within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
