"""An all-Verifier trajectory commits NO answer — its critique must never be scored.

``session._final_answer`` prefers the last Worker output, then any non-verifier
output. Its old last resort returned ``traj.turns[-1].processed_output`` — a
**Verifier** turn whenever every turn in the trajectory is a Verifier. That text
is a critique (post-processing is pass-through) that routinely names a choice
letter or number it is only *discussing*, and ``reward._committed_answer``
trusts ``final_answer`` whenever it parses (``has_answer(final)`` runs before
any role filtering), so the turn-scan's "Verifier turns are never eligible"
rule could not save it: the checker's words were graded.

Reachable paths: train-time categorical sampling picks VERIFIER every turn for
a slice of trajectories on every uniform-init candidate, and a
Verifier-collapsed head does it on every query at argmax eval — ~1/n_options
expected spurious reward on MCQ benchmarks for runs that never produced an
answer, corrupting the CMA-ES fitness signal and inflating eval accuracy.

Pure / offline: mock policy + stub pool, no network, no torch.
"""
from __future__ import annotations

import asyncio

from trinity.orchestration.reward import committed_answer, score
from trinity.orchestration.session import _final_answer, run_trajectory
from trinity.types import Role, Task, Trajectory, TurnRecord


def _turn(role: Role, text: str, *, verdict: str | None = None, turn: int = 1) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        agent_name="pool-model",
        role=role,
        raw_output=text,
        processed_output=text,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# _final_answer directly
# ---------------------------------------------------------------------------
def test_all_verifier_trajectory_has_no_final_answer():
    task = Task(task_id="q", benchmark="mmlu", prompt="p", answer="B")
    traj = Trajectory(
        task=task,
        turns=[
            _turn(Role.VERIFIER, "There is no solution yet; the answer is B would "
                                 "need support. VERDICT: REVISE", verdict="REVISE"),
            _turn(Role.VERIFIER, "Still nothing to check. Answer: B. VERDICT: REVISE",
                  verdict="REVISE", turn=2),
        ],
    )
    assert _final_answer(traj) == ""


def test_empty_trajectory_still_yields_empty_string():
    task = Task(task_id="q", benchmark="mmlu", prompt="p", answer="B")
    assert _final_answer(Trajectory(task=task, turns=[])) == ""


def test_worker_and_thinker_preference_is_unchanged():
    task = Task(task_id="q", benchmark="mmlu", prompt="p", answer="B")
    turns = [
        _turn(Role.THINKER, "plan"),
        _turn(Role.WORKER, "Answer: B", turn=2),
        _turn(Role.VERIFIER, "Looks right. VERDICT: ACCEPT", verdict="ACCEPT", turn=3),
    ]
    assert _final_answer(Trajectory(task=task, turns=turns)) == "Answer: B"
    # No Worker -> the last non-verifier (Thinker) output.
    assert _final_answer(Trajectory(task=task, turns=[turns[0], turns[2]])) == "plan"


# ---------------------------------------------------------------------------
# The scoring path end-to-end: the critique's letter must not be graded.
# ---------------------------------------------------------------------------
def test_verifier_critique_letter_is_not_scored():
    task = Task(task_id="q", benchmark="mmlu", prompt="p", answer="B")
    traj = Trajectory(
        task=task,
        turns=[_turn(Role.VERIFIER, "No worker output to verify, though the "
                                    "answer is B seems plausible. VERDICT: REVISE",
                     verdict="REVISE")],
        terminated_by="max_turns",
    )
    traj.final_answer = _final_answer(traj)
    assert traj.final_answer == ""
    assert committed_answer("mmlu", traj) == ""
    assert score(traj) == 0.0  # was ~gold-letter roulette when the critique was graded


# ---------------------------------------------------------------------------
# run_trajectory end-to-end with an always-VERIFIER policy.
# ---------------------------------------------------------------------------
class _VerifierPolicy:
    """A policy collapsed onto VERIFIER — argmax eval of a degenerate head."""

    def decide(self, transcript_text: str, *, sample: bool = False, rng=None):
        return 0, Role.VERIFIER


class _CritiquePool:
    """Stub pool whose 'model' critiques by naming the gold letter."""

    async def chat(self, model, messages, **kwargs):
        class _Res:
            text = "Nothing to check yet, but the answer is B. VERDICT: REVISE"
            prompt_tokens = 0
            completion_tokens = 0

        return _Res()


def test_run_trajectory_all_verifier_commits_no_answer():
    task = Task(task_id="q", benchmark="mmlu", prompt="p", answer="B")
    traj = asyncio.run(
        run_trajectory(
            task, _VerifierPolicy(), _CritiquePool(), ["m0"], max_turns=3,
        )
    )
    assert [t.role for t in traj.turns] == [Role.VERIFIER] * 3
    assert traj.final_answer == ""       # was the turn-3 critique text
    assert score(traj) == 0.0            # the critique's gold letter never scores
