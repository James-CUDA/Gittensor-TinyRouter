"""Offline tests for the BigCodeBench adapter (issue #212).

No network, no GPU, no torch (loaders import `datasets` lazily; the HF path is
faked). Covers the schema round-trip, the guarded loader + toy fallback, prompt
shape, code extraction/placeholder scoring, and registry integration.
"""
from __future__ import annotations

import sys
import types

import pytest

from trinity.adapters import available_adapters, get_adapter
from trinity.adapters.bigcodebench import (
    BENCHMARK,
    BigCodeBenchAdapter,
    BigCodeBenchReference,
    build_bigcodebench_prompt,
    extract_solution_code,
    load_bigcodebench_tasks,
    normalize_code,
    score_solution_exact,
)
from trinity.adapters.base import ScoringMode, TaskType


def _fake_datasets(monkeypatch, rows):
    module = types.ModuleType("datasets")

    def load_dataset(path, name=None, split=None):
        return rows

    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --- registry ---


def test_adapter_is_registered_under_name_and_alias():
    assert BENCHMARK in available_adapters()
    assert "bigcode" in available_adapters()
    assert isinstance(get_adapter("bigcodebench"), BigCodeBenchAdapter)
    assert get_adapter("bigcode").name == BENCHMARK


def test_task_type_and_scoring_modes():
    a = BigCodeBenchAdapter()
    assert a.task_type() is TaskType.CODE
    assert a.scoring_modes() == frozenset({ScoringMode.CACHED, ScoringMode.EXECUTION})


# --- schema ---


def test_reference_roundtrips():
    ref = BigCodeBenchReference(
        entry_point="f", test="import unittest\n", canonical_solution="def f(): return 1\n",
        code_prompt="def f():\n", libs=["numpy", "pandas"],
    )
    assert BigCodeBenchReference.from_dict(ref.to_dict()) == ref
    assert ref.is_valid()
    assert not BigCodeBenchReference(entry_point="f", test="  ").is_valid()


def test_libs_coercion_from_json_string(monkeypatch):
    _fake_datasets(monkeypatch, [{
        "task_id": "BigCodeBench/0",
        "complete_prompt": "def f():\n    \"\"\"doc\"\"\"\n",
        "test": "import unittest\nclass T(unittest.TestCase):\n    def test(self): pass\n",
        "entry_point": "f",
        "canonical_solution": "def f():\n    return 1\n",
        "libs": '["numpy"]',
    }])
    tasks = load_bigcodebench_tasks("v0.1.0_hf", max_items=None, seed=0)
    assert len(tasks) == 1
    assert tasks[0].task_id == "BigCodeBench/0"
    assert tasks[0].benchmark == BENCHMARK
    assert tasks[0].meta["libs"] == ["numpy"]
    assert tasks[0].answer["entry_point"] == "f"


# --- loader / toy fallback ---


def test_toy_fallback_when_datasets_missing(monkeypatch):
    # Import of `datasets` raises -> guarded loader returns None -> toy set.
    monkeypatch.setitem(sys.modules, "datasets", None)
    tasks = load_bigcodebench_tasks("test", max_items=None, seed=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.benchmark == BENCHMARK
    assert t.answer["test"].strip() and t.answer["entry_point"] == "add_positive"
    assert t.meta["task_type"] == TaskType.CODE.value


def test_loader_is_deterministic_and_truncates(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    a = load_bigcodebench_tasks("test", max_items=1, seed=5)
    b = load_bigcodebench_tasks("test", max_items=1, seed=5)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert len(a) == 1


def test_rows_missing_prompt_or_test_are_skipped(monkeypatch):
    _fake_datasets(monkeypatch, [
        {"task_id": "a", "complete_prompt": "", "test": "x", "entry_point": "f"},  # no prompt
        {"task_id": "b", "complete_prompt": "p", "test": "", "entry_point": "f"},  # no test
    ])
    # Both rows skipped -> _hf returns None -> toy fallback (1 task).
    tasks = load_bigcodebench_tasks("v0.1.0_hf", max_items=None, seed=0)
    assert len(tasks) == 1 and tasks[0].task_id.startswith("bigcodebench-toy")


# --- prompt / extraction / placeholder ---


def test_prompt_includes_entry_point_and_response_instruction():
    p = build_bigcodebench_prompt("def f():\n    pass", "f")
    assert "def f()" in p and "```python" in p and "`f`" in p


def test_extract_prefers_fenced_block_and_ignores_prose():
    out = "Sure!\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_solution_code(out).strip() == "def f():\n    return 1"
    assert extract_solution_code("just some prose, no code") == ""
    assert extract_solution_code("") == ""
    # Bare code (no fence) that looks like Python is still extracted.
    assert extract_solution_code("import os\nx = 1").startswith("import os")


def test_normalize_code_strips_fences_and_blank_lines():
    assert normalize_code("```python\n\ndef f():\n    return 1\n\n```") == "def f():\n    return 1"


def test_placeholder_exact_match_against_canonical():
    ref = {"canonical_solution": "def f():\n    return 1\n"}
    assert score_solution_exact("```python\ndef f():\n    return 1\n```", ref) == 1.0
    assert score_solution_exact("```python\ndef f():\n    return 2\n```", ref) == 0.0
    assert score_solution_exact("def f():\n    return 1", {}) == 0.0   # no gold -> 0


def test_default_adapter_uses_placeholder_no_runner():
    ref = {"canonical_solution": "def f():\n    return 1\n"}
    a = BigCodeBenchAdapter()   # no runner -> never executes model code
    assert a.score_output("```python\ndef f():\n    return 1\n```", ref) == 1.0
    assert a.score_output("```python\ndef f():\n    return 9\n```", ref) == 0.0


def test_injected_runner_is_used():
    seen = {}

    def fake_runner(output, reference):
        seen["called"] = True
        return 1.0

    a = BigCodeBenchAdapter(runner=fake_runner)
    assert a.score_output("anything", {"test": "x"}) == 1.0
    assert seen.get("called") is True
    # A runner returning None (cannot execute) scores 0, not a crash.
    assert BigCodeBenchAdapter(runner=lambda o, r: None).score_output("x", {}) == 0.0


def test_serialize_task_shape(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    task = load_bigcodebench_tasks("test", max_items=1, seed=0)[0]
    d = BigCodeBenchAdapter().serialize_task(task)
    assert set(d) == {"task_id", "benchmark", "prompt", "reference", "task_type", "meta"}
    assert d["task_type"] == TaskType.CODE.value
    assert d["reference"]["entry_point"] == "add_positive"


def test_registered_adapter_executes_via_runner(monkeypatch):
    """Issue #214 review: the registered adapter must RUN the tests, not exact-match.

    A correct solution written differently from the canonical scores 0 under the
    placeholder but 1.0 when actually executed -- proving the runner is wired in.
    """
    from trinity.adapters import get_adapter

    monkeypatch.setitem(sys.modules, "datasets", None)   # force the deterministic toy task
    adapter = get_adapter("bigcodebench")
    assert adapter._runner is not None
    tasks = load_bigcodebench_tasks("test", None, seed=0)
    ref = tasks[0].answer   # entry_point add_positive
    different_but_correct = (
        "```python\n"
        "def add_positive(nums):\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        if n > 0:\n"
        "            total += n\n"
        "    return total\n"
        "```"
    )
    # Exact-match placeholder would give 0.0 (source differs from canonical).
    assert score_solution_exact(different_but_correct, ref) == 0.0
    # The wired runner executes and passes the tests.
    assert adapter.score_output(different_but_correct, ref) == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
