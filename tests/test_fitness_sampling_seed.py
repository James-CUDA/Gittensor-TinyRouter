"""Tests that train-time policy sampling is reproducible from the seed (issue #130).

``evaluate_candidate`` fans a candidate's minibatch out through ``asyncio.gather``.
Before the fix, ``head.select`` sampled ``(agent_idx, role)`` from the *process-global*
torch RNG (``generator=None``): nothing seeded it, so two runs at the same ``--seed``
routed differently and the recorded seed did not determine the training trajectory.

The fix threads a per-trajectory generator, seeded stably from ``(rng_seed, task_id)``,
down to ``head.select``. These tests pin that contract at the plumbing level (mock
policy/pool, no torch/network), mirroring ``test_eval_random_routing_seed.py``, plus a
torch-guarded check that the ``head.select`` generator hook the fix now feeds is itself
reproducible.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import trinity.optim.fitness as F
from trinity.optim.fitness import _trajectory_seed
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


class _RngMockPolicy:
    """Mock CoordinatorPolicy: routes purely from the supplied per-trajectory rng.

    ``make_rng`` returns a plain ``random.Random`` (no torch needed offline); the
    real policy returns a ``torch.Generator`` instead, but the plumbing under test
    -- seed derivation, per-task generator identity, threading into ``decide`` --
    is identical. ``decide`` draws from that rng so routing is a direct readout of
    whether the seed reached the sampler.
    """

    def __init__(self, n_models: int = 3) -> None:
        self.n_models = n_models

    def configure(self, theta, spec) -> None:  # test stub
        return None

    def make_rng(self, seed: int) -> random.Random:
        return random.Random(seed)

    def decide(self, transcript_text: str, *, sample: bool, rng=None):
        r = rng if rng is not None else random.Random(0)
        return r.randrange(self.n_models), ROLE_ORDER[r.randrange(len(ROLE_ORDER))]


def _tasks() -> list[Task]:
    return [
        Task(task_id=str(i), benchmark="math500", prompt=f"Q{i}", answer="1") for i in range(3)
    ]


async def _routes(delays: dict[str, float], *, rng_seed):
    """Score one candidate over 3 concurrent tasks; return per-task routing."""
    policy = _RngMockPolicy(n_models=len(_MODELS))
    pool = _LatencyPool(delays)
    _fit, trajs, _per_task = await F.evaluate_candidate(
        None, None, policy, pool, _MODELS, _tasks(),
        sample=True, client=object(),  # non-None -> skip httpx; dropped for this pool
        return_trajectories=True, return_per_task=True,
        max_turns=3, rng_seed=rng_seed, reasoning=None,
    )
    return [[(tr.agent_name, tr.role.value) for tr in tj.turns] for tj in trajs]


def test_sampling_is_invariant_to_call_completion_order():
    """Same seed, same tasks, opposite latency orders -> identical routing."""
    fast_first = asyncio.run(_routes({"0": 0.001, "1": 0.010, "2": 0.020}, rng_seed=42))
    slow_first = asyncio.run(_routes({"0": 0.020, "1": 0.010, "2": 0.001}, rng_seed=42))
    assert fast_first == slow_first


def test_sampling_is_reproducible_under_a_fixed_rng_seed():
    """Two runs at the same rng_seed produce byte-identical routing."""
    delays = {"0": 0.001, "1": 0.001, "2": 0.001}
    assert asyncio.run(_routes(delays, rng_seed=7)) == asyncio.run(_routes(delays, rng_seed=7))


def test_sampling_changes_with_the_rng_seed():
    """It is still a stochastic policy -- a different seed must move the draws."""
    delays = {"0": 0.001, "1": 0.001, "2": 0.001}
    assert asyncio.run(_routes(delays, rng_seed=42)) != asyncio.run(_routes(delays, rng_seed=43))


def test_no_rng_seed_passes_no_generator(monkeypatch):
    """rng_seed=None keeps the old behavior: run_trajectory is called with rng=None."""
    seen: list[object] = []

    async def fake_run(task, policy, pool, pool_models, *, rng=None, **kwargs):
        seen.append(rng)
        from trinity.types import Trajectory

        traj = Trajectory(task=task, turns=[])
        traj.final_answer = ""
        return traj

    monkeypatch.setattr(F, "run_trajectory", fake_run)

    seen.clear()
    asyncio.run(F.evaluate_candidate(
        None, None, _RngMockPolicy(), None, _MODELS, _tasks(),
        sample=True, client=object(), return_per_task=True, max_turns=2, rng_seed=None,
    ))
    assert seen == [None, None, None]

    seen.clear()
    asyncio.run(F.evaluate_candidate(
        None, None, _RngMockPolicy(), None, _MODELS, _tasks(),
        sample=True, client=object(), return_per_task=True, max_turns=2, rng_seed=5,
    ))
    assert all(isinstance(r, random.Random) for r in seen) and len(seen) == 3


def test_trajectory_seed_is_stable_and_pythonhashseed_independent():
    # Hardcoded expected value: proves the derivation is sha256, not the builtin
    # hash (which is PYTHONHASHSEED-dependent and would not match across processes).
    assert _trajectory_seed(0, "a") == 11381658363930578919
    # Independent, reproducible streams per (seed, task_id).
    assert _trajectory_seed(0, "a") == _trajectory_seed(0, "a")
    assert _trajectory_seed(0, "a") != _trajectory_seed(0, "b")
    assert _trajectory_seed(0, "a") != _trajectory_seed(1, "a")


# The torch-path check (head.select + CoordinatorPolicy.make_rng) lives in
# tests/test_torch_head_rng.py: importing torch must not pollute sys.modules for
# the torch-free canaries (test_shaped_fitness.py::test_no_torch_imported), so it
# has to sit in a test_torch_* file that sorts after them.


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
