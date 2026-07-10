"""Entrypoint: evaluate a trained coordinator + baselines on a benchmark.

Reports the relative invariants from SPEC §1.3:
  - TRINITY (trained coordinator, argmax) vs
  - each single model alone (one direct Worker turn) [R1, R2] vs
  - random routing (random agent+role each turn) [R4].

Usage:
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python -m trinity.eval --benchmark math500 \
        --theta experiments/math500/run/best_theta.npy
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from .adapters import get_adapter
from .eval_harness import (
    RandomPolicy,
    load_and_configure_policy,
    score_policy,
    score_random_routing,
    score_single_model,
    task_rng,
)
from .llm.openrouter_client import OpenRouterPool

__all__ = [
    "REPRODUCIBILITY_SEED",
    "RandomPolicy",
    "task_rng",
    "evaluate",
    "main",
]

_REPO = Path(__file__).resolve().parents[2]

# Locked reproducibility seed — committed, never changed.
# Using a custom seed prints a warning: cherry-picking seeds to find a lucky
# eval split undermines result trustworthiness.
REPRODUCIBILITY_SEED: int = 42


async def evaluate(args) -> dict:
    pool = OpenRouterPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    # Resolve the benchmark to an adapter ONCE; the rest of the evaluator drives
    # the adapter interface and never branches on the benchmark name (#9).
    adapter = get_adapter(args.benchmark)
    tasks = adapter.load_tasks("test", max_items=args.max_items, seed=args.seed)
    print(f"[eval] benchmark={args.benchmark}  {len(tasks)} test tasks  pool={pool_models}")
    run_kwargs = dict(max_turns=args.max_turns, max_tokens=args.max_tokens, reasoning=args.reasoning)

    results: dict[str, float] = {}

    # --- single-model baselines (R1/R2) ---
    for m in pool_models:
        reps = [await score_single_model(tasks, pool, m, adapter,
                                          max_tokens=args.max_tokens, reasoning=args.reasoning)
                for _ in range(max(1, args.single_reps))]
        s = float(mean(reps))
        results[f"single::{m}"] = s
        if len(reps) > 1:
            sd = (sum((x - s) ** 2 for x in reps) / len(reps)) ** 0.5
            results[f"single_std::{m}"] = sd
            print(f"  single  {m:20s} = {s:.4f} ± {sd:.4f}  (reps={reps})")
        else:
            print(f"  single  {m:20s} = {s:.4f}")

    # --- TRINITY trained coordinator (argmax) ---
    print("[eval] building coordinator on GPU...")
    policy, _spec = load_and_configure_policy(args.config, n_models, args.theta)
    s_trinity = await score_policy(
        tasks, policy, pool, pool_models, adapter=adapter,
        sample=False, label="TRINITY", **run_kwargs,
    )
    results["TRINITY"] = s_trinity
    print(f"  TRINITY (trained)        = {s_trinity:.4f}")

    # --- random routing (R4) — multi-seed baseline to remove run-to-run noise ---
    # The paper reports random routing as a single draw per run, but with
    # small eval sets (~120 q's) the variance is large (0.733–0.792 in
    # practice).  Reporting the mean over 100 seeds gives an honest baseline.
    rand_seeds = max(1, args.rand_seeds)
    s_rand, rand_std = await score_random_routing(
        tasks, pool, pool_models, n_models, adapter=adapter,
        n_seeds=rand_seeds, base_seed=args.seed, **run_kwargs,
    )
    results["random_routing"] = s_rand
    if rand_std is not None:
        results["random_routing_std"] = rand_std
        print(f"  random routing           = {s_rand:.4f} ± {rand_std:.4f}  (n={rand_seeds} seeds)")
    else:
        print(f"  random routing           = {s_rand:.4f}")

    best_single = max(results[k] for k in results if k.startswith("single::"))
    invariants = {
        "R1/R2 TRINITY > best single model": s_trinity > best_single,
        "R4 TRINITY > random routing": s_trinity > s_rand,
        "best_single": best_single,
    }
    out = {"benchmark": args.benchmark, "results": results, "invariants": invariants}
    print("[eval] invariants:", json.dumps(invariants, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TRINITY + baselines")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--theta", required=True, help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=100, dest="max_items")
    ap.add_argument("--single-reps", type=int, default=1, dest="single_reps",
                    help="average each single-model baseline over K runs (cuts nondeterminism noise)")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--seed", type=int, default=REPRODUCIBILITY_SEED,
                    help=f"random seed (default: {REPRODUCIBILITY_SEED} — locked for reproducibility; "
                         "overriding prints a warning)")
    ap.add_argument("--rand-seeds", type=int, default=100, dest="rand_seeds",
                    help="number of random seeds for the random-routing baseline (default: 100)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.seed != REPRODUCIBILITY_SEED:
        print(
            f"[eval] ⚠ seed={args.seed} differs from locked REPRODUCIBILITY_SEED="
            f"{REPRODUCIBILITY_SEED}. Non-default seeds weaken reproducibility "
            f"and should be justified in any reported results.",
            flush=True,
        )
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
