#!/usr/bin/env python3
"""Baseline: random routing — pick a random model + random role each turn.

This is the FLOOR baseline. Any useful router must beat random routing to
demonstrate routing intelligence. The eval pipeline also runs this internally
(100 seeds, mean ± std), but this script lets you check it standalone.

Usage:
    python baselines/random_router.py --benchmark math500 --seeds 100
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


async def run(benchmark: str, max_items: int, n_seeds: int) -> None:
    from trinity.adapters import get_adapter
    from trinity.eval_harness import RandomPolicy, score_policy
    from trinity.llm.openrouter_client import OpenRouterPool

    adapter = get_adapter(benchmark)
    tasks = adapter.load_tasks("test", max_items=max_items, seed=42)
    pool = OpenRouterPool(str(_REPO / "configs" / "models.yaml"))
    pool_models = list(pool.models)
    n_models = len(pool_models)
    print(f"[baseline] random routing on {benchmark}: {len(tasks)} tasks, {n_seeds} seeds")

    scores = []
    for s in range(n_seeds):
        rand = RandomPolicy(n_models, seed=42 * 10000 + s)
        s_r = await score_policy(tasks, rand, pool, pool_models, sample=False,
                                  max_turns=5, max_tokens=4096, reasoning="minimal")
        scores.append(s_r)

    avg = float(mean(scores))
    std = (sum((x - avg) ** 2 for x in scores) / n_seeds) ** 0.5
    print(f"[baseline] random routing on {benchmark}: {avg:.4f} ± {std:.4f}  (n={n_seeds} seeds)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Random-routing baseline")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    asyncio.run(run(args.benchmark, args.max_items, args.seeds))


if __name__ == "__main__":
    main()
