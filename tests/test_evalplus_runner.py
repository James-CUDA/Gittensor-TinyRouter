"""Offline tests for the sandboxed EvalPlus runner (issue #254).

Exercises the real grading mechanism — assembling a solution with its base/plus
``check`` harness and running it in a subprocess — with the base+plus rule at the
centre. No network, no GPU, no torch.
"""
from __future__ import annotations

import pytest

from trinity.adapters.evalplus import EvalPlusAdapter
from trinity.adapters.evalplus_runner import (
    EvalPlusResult,
    build_harness,
    evaluate_solution,
    score_evalplus,
)

_BASE = "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(0) == 1\n"
_PLUS = (
    "def check(candidate):\n"
    "    assert candidate(1) == 2\n"
    "    assert candidate(0) == 1\n"
    "    assert candidate(-5) == -4\n"
)
_REF = {"entry_point": "add_one", "plus_test": _PLUS, "base_test": _BASE}

_GOOD = "```python\ndef add_one(x):\n    return x + 1\n```"
_BASE_ONLY = "```python\ndef add_one(x):\n    return x + 1 if x >= 0 else 0\n```"  # passes base, fails plus
_BADSYNTAX = "```python\ndef add_one(x)\n    return x + 1\n```"                    # missing colon


def test_build_harness_calls_check_on_entry_point():
    h = build_harness("def add_one(x): return x + 1", _PLUS, "add_one")
    assert "def add_one" in h and "def check" in h and h.rstrip().endswith("check(add_one)")


# --- the base+plus rule (the point of EvalPlus) ---


def test_correct_solution_passes_base_and_plus():
    res = evaluate_solution(_GOOD, _REF)
    assert isinstance(res, EvalPlusResult)
    assert res.passed and res.reason == "passed" and res.base_pass and res.plus_pass
    assert res.reward == 1.0


def test_base_passing_plus_failing_scores_zero():
    res = evaluate_solution(_BASE_ONLY, _REF)
    assert res.passed is False and res.reason == "plus_failed"
    assert res.base_pass is True and res.plus_pass is False and res.reward == 0.0


def test_base_failure_short_circuits():
    wrong = "```python\ndef add_one(x):\n    return x + 2\n```"   # fails even base
    res = evaluate_solution(wrong, _REF)
    assert res.passed is False and res.reason == "base_failed" and res.base_pass is False


def test_reference_without_base_test_still_grades_plus():
    ref = {"entry_point": "add_one", "plus_test": _PLUS}   # no base_test
    assert evaluate_solution(_GOOD, ref).reason == "passed"
    assert evaluate_solution(_BASE_ONLY, ref).reason == "plus_failed"


# --- clean failure reasons ---


def test_unparseable_solution_is_harness_error():
    assert evaluate_solution(_BADSYNTAX, _REF).reason == "harness_error"


def test_no_code_and_no_tests():
    assert evaluate_solution("I cannot solve this.", _REF).reason == "no_code_found"
    assert evaluate_solution(_GOOD, {"entry_point": "add_one", "plus_test": ""}).reason == "no_tests"


def test_score_evalplus_binary_and_none():
    assert score_evalplus(_GOOD, _REF) == 1.0
    assert score_evalplus(_BASE_ONLY, _REF) == 0.0
    assert score_evalplus(_GOOD, {"entry_point": "add_one"}) is None   # no harness -> None


def test_adapter_with_runner_end_to_end():
    a = EvalPlusAdapter("humaneval_plus", runner=score_evalplus)
    assert a.score_output(_GOOD, _REF) == 1.0
    assert a.score_output(_BASE_ONLY, _REF) == 0.0


def test_timeout_is_a_clean_failure():
    slow = "```python\ndef add_one(x):\n    while True:\n        pass\n```"
    assert evaluate_solution(slow, _REF, timeout=2).passed is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
