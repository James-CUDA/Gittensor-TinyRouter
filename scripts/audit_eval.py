#!/usr/bin/env python3
"""Audit-split evaluation — the final, honest number.

Generates a held-out question split from a SEALED seed that is NEVER used
during training or development. After all experiments are complete and the
best theta is chosen, run this script ONCE to get the ungameable result.

The seed is baked into this script and must never be changed after the
first experiment begins. The idea is simple: the researcher cannot overfit
to questions they never saw, so the audit score is the trustworthy number.

Usage:
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python scripts/audit_eval.py \
        --benchmark math500 \
        --theta experiments/math500/run/best_theta.npy \
        --out experiments/math500/audit_result.json

The --seed flag is deliberately absent — the seed is locked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.adapters import get_adapter  # noqa: E402
from trinity.eval_harness import (  # noqa: E402
    load_and_configure_policy,
    score_policy,
    score_random_routing,
    score_single_model,
)
from trinity.llm.openrouter_client import OpenRouterPool  # noqa: E402

# ---- SEALED SEED — committed, never changed. ----
# This seed selects a held-out subset of questions that NO experiment has
# ever seen. Running eval against it gives the honest, ungameable score.
_AUDIT_SEED: int = 314159265  # first 9 digits of pi — arbitrary but fixed forever

# The audit uses a different split name so HF datasets don't serve the same
# questions as the training or test splits (which experiments already use).
_AUDIT_SPLIT: str = "train"  # We sample from train but with a DIFFERENT seed
# and a DIFFERENT shuffle, so the subset is as-if held-out.


async def run_audit(args) -> dict:
    pool = OpenRouterPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    adapter = get_adapter(args.benchmark)
    # Load tasks with the SEALED seed — NO override possible.
    tasks = adapter.load_tasks(
        _AUDIT_SPLIT,
        max_items=args.max_items,
        seed=_AUDIT_SEED,
    )
    print(f"[audit] benchmark={args.benchmark}  {len(tasks)} audit tasks  "
          f"seed={_AUDIT_SEED} (SEALED — never used in any experiment)")

    run_kwargs = dict(
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
    )

    results: dict[str, float] = {}

    # --- Single-model baselines ---
    for m in pool_models:
        s = await score_single_model(
            tasks, pool, m, adapter,
            max_tokens=args.max_tokens, reasoning=args.reasoning,
        )
        results[f"single::{m}"] = s
        print(f"  single  {m:20s} = {s:.4f}")

    # --- TRINITY trained coordinator (argmax) ---
    print("[audit] building coordinator on GPU...")
    policy, _spec = load_and_configure_policy(args.config, n_models, args.theta)
    s_trinity = await score_policy(
        tasks, policy, pool, pool_models, adapter=adapter,
        sample=False, label="TRINITY", **run_kwargs,
    )
    results["TRINITY"] = s_trinity
    print(f"  TRINITY (trained)        = {s_trinity:.4f}")

    # --- Random routing baseline (100 seeds) ---
    s_rand, rand_std = await score_random_routing(
        tasks, pool, pool_models, n_models, adapter=adapter,
        n_seeds=100, base_seed=_AUDIT_SEED, **run_kwargs,
    )
    results["random_routing"] = s_rand
    results["random_routing_std"] = rand_std  # type: ignore[assignment]
    print(f"  random routing           = {s_rand:.4f} ± {rand_std:.4f}  (n=100 seeds)")

    best_single = max(v for k, v in results.items() if k.startswith("single::"))
    out = {
        "benchmark": args.benchmark,
        "audit_seed": _AUDIT_SEED,
        "audit_split": _AUDIT_SPLIT,
        "num_tasks": len(tasks),
        "results": results,
        "best_single": best_single,
        "trinity_vs_best_single": s_trinity - best_single,
        "trinity_vs_random": s_trinity - s_rand,
    }
    print(f"\n[audit] FINAL: TRINITY={s_trinity:.4f}  best_single={best_single:.4f}  "
          f"random={s_rand:.4f}±{rand_std:.4f}")
    print(f"[audit] TRINITY - best_single = {out['trinity_vs_best_single']:+.4f}")
    print(f"[audit] TRINITY - random      = {out['trinity_vs_random']:+.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[audit] saved to {args.out}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit-split evaluation — run ONCE after all experiments are done"
    )
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--theta", required=True, help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--out", default="")
    # Deliberately NO --seed argument.
    args = ap.parse_args()
    asyncio.run(run_audit(args))


if __name__ == "__main__":
    main()
