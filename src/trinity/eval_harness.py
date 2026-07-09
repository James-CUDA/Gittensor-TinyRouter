"""Shared evaluation helpers for trinity.eval and scripts/audit_eval."""
from __future__ import annotations

import asyncio
import random
from statistics import mean

import numpy as np

from .coordinator.config import build_policy_from_config, load_coordinator_section
from .orchestration.session import run_trajectory
from .roles.prompts import build_messages
from .types import ROLE_ORDER, Role

__all__ = [
    "RandomPolicy",
    "task_rng",
    "reduce_scores",
    "load_and_configure_policy",
    "score_policy",
    "score_single_model",
    "score_random_routing",
]


class RandomPolicy:
    """Random (agent, role) each turn — the R4 routing baseline (no GPU).

    Draws from the caller-supplied ``rng`` when one is given, so each trajectory
    can own a deterministically-seeded stream. Falling back to a single shared
    ``self.rng`` across trajectories running under ``asyncio.gather`` would make
    the draws depend on network completion order rather than on the seed.
    """

    def __init__(self, n_models: int, seed: int = 0, *, rng: random.Random | None = None) -> None:
        self.n_models = n_models
        self.rng = rng if rng is not None else random.Random(seed)

    def decide(
        self,
        transcript_text: str,
        *,
        sample: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, Role]:
        r = self.rng if rng is None else rng
        return r.randrange(self.n_models), r.choice(ROLE_ORDER)


def task_rng(seed: int, task_id: str) -> random.Random:
    """Build a per-task RNG whose stream depends only on ``seed`` and ``task_id``."""
    return random.Random(f"{seed}:{task_id}")


def reduce_scores(scores: list, *, label: str) -> float:
    """Average per-task scores, counting a failed trajectory as ``0.0``."""
    n_failed = sum(isinstance(s, BaseException) for s in scores)
    if scores and n_failed == len(scores):
        raise RuntimeError(
            f"{label}: all {len(scores)} trajectories failed (last error: "
            f"{type(scores[-1]).__name__}: {scores[-1]}); refusing to report 0.0."
        )
    if n_failed:
        print(
            f"  [warn] {label}: {n_failed}/{len(scores)} trajectories failed "
            "(counted as 0.0); the reported score is degraded.",
            flush=True,
        )
    return float(mean(0.0 if isinstance(s, BaseException) else s for s in scores))


def load_and_configure_policy(config_path: str, n_models: int, theta_path: str):
    """Build coordinator from YAML and install trained theta."""
    cc = load_coordinator_section(config_path)
    policy, spec = build_policy_from_config(cc, n_models=n_models)
    policy.configure(np.load(theta_path), spec)
    return policy, spec


async def score_policy(
    tasks,
    policy,
    pool,
    pool_models,
    *,
    adapter,
    sample: bool,
    rng_seed: int | None = None,
    label: str = "routing",
    client=None,
    **run_kwargs,
) -> float:
    import httpx

    if client is None:
        async with httpx.AsyncClient() as cli:
            return await score_policy(
                tasks, policy, pool, pool_models,
                adapter=adapter, sample=sample, rng_seed=rng_seed, label=label,
                client=cli, **run_kwargs,
            )
    trajs = await asyncio.gather(
        *[
            run_trajectory(
                t, policy, pool, pool_models, adapter=adapter, sample=sample, client=client,
                rng=None if rng_seed is None else task_rng(rng_seed, t.task_id),
                **run_kwargs,
            )
            for t in tasks
        ],
        return_exceptions=True,
    )
    scores = [t if isinstance(t, BaseException) else adapter.score_trajectory(t) for t in trajs]
    return reduce_scores(scores, label=label)


async def score_single_model(
    tasks, pool, model, adapter, *, max_tokens, reasoning, client=None
) -> float:
    import httpx

    if client is None:
        async with httpx.AsyncClient() as cli:
            return await score_single_model(
                tasks, pool, model, adapter,
                max_tokens=max_tokens, reasoning=reasoning, client=cli,
            )

    async def one(task):
        msgs = build_messages(Role.WORKER, adapter.build_prompt(task), [])
        res = await pool.chat(
            model, msgs, max_tokens=max_tokens, temperature=0.0,
            reasoning=reasoning, client=client,
        )
        return adapter.score_output(res.text, task.answer)

    scores = await asyncio.gather(*[one(t) for t in tasks], return_exceptions=True)
    return reduce_scores(scores, label=f"single::{model}")


async def score_random_routing(
    tasks, pool, pool_models, n_models, *, adapter, n_seeds, base_seed, **run_kwargs
) -> tuple[float, float | None]:
    """Mean score over multiple random-routing seeds; std when n_seeds > 1."""
    rand_scores: list[float] = []
    for s in range(n_seeds):
        seed_s = base_seed * 10000 + s
        rand = RandomPolicy(n_models, seed=seed_s)
        s_r = await score_policy(
            tasks, rand, pool, pool_models, adapter=adapter,
            sample=False, rng_seed=seed_s, label="random routing", **run_kwargs,
        )
        rand_scores.append(s_r)
    s_rand = float(mean(rand_scores))
    if n_seeds > 1:
        rand_std = (sum((x - s_rand) ** 2 for x in rand_scores) / n_seeds) ** 0.5
        return s_rand, rand_std
    return s_rand, None
