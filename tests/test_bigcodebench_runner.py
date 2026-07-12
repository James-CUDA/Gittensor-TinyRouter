"""Offline tests for the sandboxed BigCodeBench runner (issue #212).

Exercises the real grading mechanism — assembling a solution with its unittest
harness and running it in a subprocess — against a throwaway harness built in the
test. No network, no GPU, no torch.
"""
from __future__ import annotations

import pytest

from trinity.adapters.bigcodebench import BigCodeBenchAdapter
from trinity.adapters.bigcodebench_runner import (
    BigCodeBenchResult,
    build_harness,
    evaluate_solution,
    run_module,
    score_bigcodebench,
    _sandbox_env,
)

_TEST = (
    "import unittest\n\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_sum(self):\n"
    "        self.assertEqual(add_positive([1, -2, 3]), 4)\n"
    "    def test_empty(self):\n"
    "        self.assertEqual(add_positive([]), 0)\n"
)
_REF = {"entry_point": "add_positive", "test": _TEST}

_GOOD = "```python\ndef add_positive(nums):\n    return sum(n for n in nums if n > 0)\n```"
_WRONG = "```python\ndef add_positive(nums):\n    return sum(nums)\n```"          # counts negatives too
_BADSYNTAX = "```python\ndef add_positive(nums)\n    return 1\n```"               # missing colon


# --- build_harness ---


def test_build_harness_appends_unittest_main_once():
    h = build_harness("def f(): pass", _TEST)
    assert h.count("unittest.main()") == 1
    assert "def f(): pass" in h and "class TestCases" in h


def test_build_harness_does_not_double_main():
    test_with_main = _TEST + "\nif __name__ == '__main__':\n    unittest.main()\n"
    h = build_harness("def f(): pass", test_with_main)
    assert h.count("unittest.main(") == 1


# --- evaluate_solution / score_bigcodebench (real subprocess) ---


def test_correct_solution_passes():
    res = evaluate_solution(_GOOD, _REF)
    assert isinstance(res, BigCodeBenchResult)
    assert res.passed is True and res.reason == "passed" and res.reward == 1.0


def test_wrong_solution_fails_tests():
    res = evaluate_solution(_WRONG, _REF)
    assert res.passed is False and res.reason == "tests_failed" and res.reward == 0.0


def test_unparseable_solution_is_a_clean_harness_error():
    res = evaluate_solution(_BADSYNTAX, _REF)
    assert res.passed is False and res.reason == "harness_error"


def test_no_code_found():
    res = evaluate_solution("I could not solve this.", _REF)
    assert res.passed is False and res.reason == "no_code_found"


def test_score_bigcodebench_binary_and_none():
    assert score_bigcodebench(_GOOD, _REF) == 1.0
    assert score_bigcodebench(_WRONG, _REF) == 0.0
    # No harness -> cannot execute -> None (adapter falls back to placeholder).
    assert score_bigcodebench(_GOOD, {"entry_point": "add_positive"}) is None


def test_adapter_with_runner_executes_end_to_end():
    adapter = BigCodeBenchAdapter(runner=score_bigcodebench)
    assert adapter.score_output(_GOOD, _REF) == 1.0
    assert adapter.score_output(_WRONG, _REF) == 0.0


def test_timeout_is_a_clean_failure():
    slow = "```python\ndef add_positive(nums):\n    while True:\n        pass\n```"
    res = evaluate_solution(slow, _REF, timeout=2)
    assert res.passed is False and res.reason == "tests_failed"


# --- sandbox / isolation (issue #214 review: secrets must not leak) ---


def test_sandbox_env_drops_host_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("BENCHMARK_PASSWORD", "hunter2")
    env = _sandbox_env()
    assert "OPENROUTER_API_KEY" not in env
    assert "BENCHMARK_PASSWORD" not in env
    assert env.get("PYTHONDONTWRITEBYTECODE") == "1"


def test_graded_code_cannot_read_host_secret(monkeypatch):
    """A secret set in the parent env is invisible to the executed harness."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    ref = {"entry_point": "f", "test": (
        "import os, unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_no_secret(self):\n"
        "        self.assertIsNone(os.environ.get('OPENROUTER_API_KEY'))\n"
    )}
    res = evaluate_solution("```python\ndef f():\n    return 1\n```", ref)
    assert res.passed is True and res.reason == "passed"


def test_child_runs_in_isolated_mode():
    """`python -I` is in effect: sys.flags.isolated is set in the child."""
    ok, _ = run_module(
        "import sys\nassert sys.flags.isolated == 1\n", timeout=30
    )
    assert ok is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
