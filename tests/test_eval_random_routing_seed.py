"""Tests that the R4 random-routing baseline depends on the seed, not on call latency.

`_score_policy` fans every task out through `asyncio.gather`. When all trajectories
share one `RandomPolicy.rng`, turn-2+ draws are consumed in network-completion order,
so the routing (and the resulting `random_routing` score) varies run to run under a
fixed seed. These tests pin the per-trajectory RNG contract that prevents that.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.eval import RandomPolicy, task_rng
from trinity.orchestration.session import run_trajectory
from trinity.types import ROLE_ORDER, Task

_MODELS = ["m0", "m1", "m2"]


@dataclass
class _ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _LatencyPool:
    """Stub pool whose per-task delay controls the order trajectories resume in."""

    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    async def chat(self, model, messages, *, temperature=0.0, top_p=1.0, max_tokens=0):
        blob = " ".join(m["content"] for m in messages)
        task_id = next(k for k in self.delays if f"Q{k}" in blob)
        await asyncio.sleep(self.delays[task_id])
        return _ChatResult(text="a derivation with no extractable answer")


def _tasks() -> list[Task]:
    return [
        Task(task_id=str(i), benchmark="math500", prompt=f"Q{i}", answer="1") for i in range(3)
    ]


async def _routes(delays: dict[str, float], seed: int = 42) -> list[list[tuple[str, str]]]:
    """Route 3 tasks concurrently through one shared policy, as `_score_policy` does."""
    policy = RandomPolicy(n_models=len(_MODELS), seed=seed)
    pool = _LatencyPool(delays)
    trajs = await asyncio.gather(
        *[
            run_trajectory(
                t, policy, pool, _MODELS, max_turns=3, reasoning=None,
                rng=task_rng(seed, t.task_id),
            )
            for t in _tasks()
        ]
    )
    return [[(tr.agent_name, tr.role.value) for tr in tj.turns] for tj in trajs]


def test_routing_is_invariant_to_call_completion_order():
    """Same seed, same tasks, opposite latency orders -> identical routing."""
    fast_first = asyncio.run(_routes({"0": 0.001, "1": 0.010, "2": 0.020}))
    slow_first = asyncio.run(_routes({"0": 0.020, "1": 0.010, "2": 0.001}))
    assert fast_first == slow_first


def test_routing_changes_with_the_seed():
    """The baseline is still random — a different seed must move the draws."""
    delays = {"0": 0.001, "1": 0.001, "2": 0.001}
    assert asyncio.run(_routes(delays, seed=42)) != asyncio.run(_routes(delays, seed=43))


def test_task_rng_streams_are_independent_and_reproducible():
    seed = 42
    assert [task_rng(seed, "a").random() for _ in range(2)] == [task_rng(seed, "a").random()] * 2
    assert task_rng(seed, "a").random() != task_rng(seed, "b").random()
    assert task_rng(seed, "a").random() != task_rng(seed + 1, "a").random()


def test_decide_falls_back_to_instance_rng_when_no_rng_supplied():
    """Existing callers that never pass `rng=` keep the old seeded-instance behaviour."""
    a = [RandomPolicy(3, seed=7).decide("", rng=None) for _ in range(1)]
    b = [RandomPolicy(3, seed=7).decide("", rng=None) for _ in range(1)]
    assert a == b
    idx, role = a[0]
    assert 0 <= idx < 3
    assert role in ROLE_ORDER
