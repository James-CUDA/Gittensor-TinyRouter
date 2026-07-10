"""The committed answer is chosen from a non-verifier turn.

A Verifier-ACCEPT trajectory terminates *on* a Verifier turn whose
post-processed output keeps its full critique (post-processing is pass-through).
That critique routinely names a choice letter or number it is only *discussing*,
so scoring it would credit or penalise the run for the checker's words instead of
an answer the solver committed. ``reward._committed_answer`` must therefore skip
Verifier turns and read the last non-verifier ``O_k`` — matching
``session._final_answer`` and the ``_terminating_role`` contract.

Pure / offline / numpy-free — imports only ``trinity.orchestration.reward``.
"""
from __future__ import annotations

from trinity.orchestration import reward as R
from trinity.orchestration.reward import committed_answer, score
from trinity.types import Role, Task, Trajectory, TurnRecord


def _turn(role: Role, text: str, *, verdict: str | None = None) -> TurnRecord:
    return TurnRecord(
        turn=0,
        agent_name="pool-model",
        role=role,
        raw_output=text,
        processed_output=text,
        verdict=verdict,
    )


def _traj(benchmark: str, answer: str, turns: list[TurnRecord], final: str) -> Trajectory:
    task = Task(task_id="q", benchmark=benchmark, prompt="p", answer=answer)
    return Trajectory(task=task, turns=turns, final_answer=final, terminated_by="accept")


# ---------------------------------------------------------------------------
# The bug: a Verifier that discusses a DIFFERENT answer must not be scored.
# ---------------------------------------------------------------------------
def test_verifier_wrong_letter_does_not_override_worker_answer():
    # Worker committed C (gold); a later Worker turn rephrased without a letter,
    # so final_answer is unparseable; the terminal Verifier mentions B.
    traj = _traj(
        "mmlu",
        "C",
        turns=[
            _turn(Role.WORKER, "Answer: C"),
            _turn(Role.WORKER, "let me reconsider the whole approach"),
            _turn(Role.VERIFIER, "Answer: B. VERDICT: ACCEPT", verdict="ACCEPT"),
        ],
        final="let me reconsider the whole approach",
    )
    # The committed answer is the Worker's C, never the Verifier's B.
    assert committed_answer("mmlu", traj) == "Answer: C"
    assert score(traj) == 1.0  # was 0.0 when the Verifier turn was scored


def test_verifier_gold_letter_does_not_inflate_a_run_with_no_committed_answer():
    # No worker turn ever produced a parseable letter; only the Verifier names
    # the gold letter. Reading the Verifier would falsely credit the run.
    traj = _traj(
        "mmlu",
        "C",
        turns=[
            _turn(Role.WORKER, "let me reconsider the whole approach"),
            _turn(Role.VERIFIER, "Answer: C. VERDICT: ACCEPT", verdict="ACCEPT"),
        ],
        final="let me reconsider the whole approach",
    )
    assert committed_answer("mmlu", traj) == "let me reconsider the whole approach"
    assert score(traj) == 0.0  # was 1.0 (false positive) when the Verifier was scored


def test_math_verifier_number_is_ignored():
    # Worker boxed 4 (gold); final rephrase drops it; Verifier muses "7".
    traj = _traj(
        "math500",
        "4",
        turns=[
            _turn(Role.WORKER, r"so \boxed{4}"),
            _turn(Role.WORKER, "on reflection I am unsure"),
            _turn(Role.VERIFIER, "I would have said 7. VERDICT: ACCEPT", verdict="ACCEPT"),
        ],
        final="on reflection I am unsure",
    )
    assert committed_answer("math500", traj) == r"so \boxed{4}"
    assert score(traj) == 1.0


# ---------------------------------------------------------------------------
# Fallback ordering: prefer Worker, then any other non-verifier turn.
# ---------------------------------------------------------------------------
def test_falls_back_to_thinker_when_no_worker_answer():
    # Only the Thinker carried a parseable answer; Worker/Verifier did not.
    traj = _traj(
        "math500",
        "9",
        turns=[
            _turn(Role.THINKER, r"so \boxed{9}"),
            _turn(Role.WORKER, "restating the setup only"),
            _turn(Role.VERIFIER, "sounds right, VERDICT: ACCEPT", verdict="ACCEPT"),
        ],
        final="restating the setup only",
    )
    assert committed_answer("math500", traj) == r"so \boxed{9}"
    assert score(traj) == 1.0


def test_recovers_worker_answer_past_unparseable_final():
    # Regression: the original recovery still works when the Verifier has no answer.
    traj = _traj(
        "math500",
        "4",
        turns=[
            _turn(Role.WORKER, r"\boxed{4}"),
            _turn(Role.VERIFIER, "Looks good. VERDICT: ACCEPT", verdict="ACCEPT"),
        ],
        final="Looks good. VERDICT: ACCEPT",
    )
    assert committed_answer("math500", traj) == r"\boxed{4}"
    assert score(traj) == 1.0


def test_parseable_final_answer_is_used_directly():
    # When the final answer already parses, it wins with no turn scan.
    traj = _traj(
        "mmlu",
        "A",
        turns=[_turn(Role.WORKER, "Answer: A")],
        final="Answer: A",
    )
    assert committed_answer("mmlu", traj) == "Answer: A"
    assert score(traj) == 1.0


# ---------------------------------------------------------------------------
# The shared selector directly.
# ---------------------------------------------------------------------------
def test_last_answerful_output_skips_verifier_and_honours_role():
    turns = [
        _turn(Role.THINKER, "Answer: A"),
        _turn(Role.WORKER, "Answer: B"),
        _turn(Role.VERIFIER, "Answer: D. VERDICT: ACCEPT", verdict="ACCEPT"),
    ]
    # role=WORKER -> the Worker's B (Verifier D is never eligible).
    assert R._last_answerful_output("mmlu", turns, role=Role.WORKER) == "Answer: B"
    # role=None -> newest non-verifier turn with an answer, i.e. the Worker's B.
    assert R._last_answerful_output("mmlu", turns, role=None) == "Answer: B"
    # A verifier-only set yields nothing.
    only_verifier = [_turn(Role.VERIFIER, "Answer: D. VERDICT: ACCEPT", verdict="ACCEPT")]
    assert R._last_answerful_output("mmlu", only_verifier, role=None) is None
