"""Regression: HERO self-consistency must resolve the livecodebench_v6 alias.

A frozen v6 item carries ``benchmark == "livecodebench_v6"`` (the adapter identity).
reward.has_answer / score_text resolve that to the ``livecodebench`` code checker,
but the HERO agreement test did not, so v6 trajectories fell through to raw-text
equality and two turns with identical code but different prose wrongly "disagreed".
No network, no GPU, no torch.
"""
from __future__ import annotations

from trinity.optim.fitness import _answers_agree, hero_quality
from trinity.orchestration import reward as R
from trinity.types import Role, Task, Trajectory, TurnRecord

# Same code block, different surrounding prose.
_A = "Here is my solution:\n```python\ndef f():\n    return 1\n```"
_B = "Final answer:\n```python\ndef f():\n    return 1\n```"


def _traj(benchmark: str) -> Trajectory:
    task = Task(task_id="t", benchmark=benchmark, prompt="q", answer="x")
    turns = [
        TurnRecord(turn=1, agent_name="m", role=Role.WORKER, raw_output=_A, processed_output=_A),
        TurnRecord(turn=2, agent_name="m", role=Role.WORKER, raw_output=_B, processed_output=_B),
    ]
    return Trajectory(task=task, turns=turns, final_answer=_B)


def test_answers_agree_resolves_the_v6_alias():
    # Identical code -> agree, whether the benchmark is the family key or the v6 identity.
    assert _answers_agree("livecodebench", _A, _B) is True
    assert _answers_agree("livecodebench_v6", _A, _B) is True   # was False before the fix


def test_hero_quality_is_alias_invariant_for_identical_code():
    # The same trajectory content must score the same under the family key and the
    # versioned identity; previously v6 scored 0.5 vs 1.0.
    assert hero_quality(_traj("livecodebench")) == 1.0
    assert hero_quality(_traj("livecodebench_v6")) == 1.0


def test_v6_disagreement_is_still_detected():
    # The fix must not make everything agree: genuinely different code disagrees.
    other = "```python\ndef f():\n    return 2\n```"
    assert _answers_agree("livecodebench_v6", _A, other) is False


def test_alias_resolves_to_the_same_key_the_fix_relies_on():
    assert R.resolve_benchmark("livecodebench_v6") == "livecodebench"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
