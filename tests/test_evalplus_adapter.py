"""Offline tests for the EvalPlus adapters (HumanEval+ / MBPP+) — issue #254.

No network, no GPU, no torch (the HF path is faked). Covers schema round-trip, the
guarded loader + toy fallback, prompt shape, extraction/placeholder scoring, and
registry integration for both benchmarks.
"""
from __future__ import annotations

import sys
import types

import pytest

from trinity.adapters import available_adapters, get_adapter
from trinity.adapters.base import ScoringMode, TaskType
from trinity.adapters.evalplus import (
    DATASETS,
    HUMANEVAL_PLUS,
    MBPP_PLUS,
    EvalPlusAdapter,
    EvalPlusReference,
    build_evalplus_prompt,
    extract_solution_code,
    load_evalplus_tasks,
    normalize_code,
    score_solution_exact,
)


def _fake_datasets(monkeypatch, rows):
    module = types.ModuleType("datasets")
    module.load_dataset = lambda path, name=None, split=None: rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --- registry ---


def test_both_benchmarks_registered_with_aliases():
    for name in (HUMANEVAL_PLUS, MBPP_PLUS):
        assert name in available_adapters()
        assert get_adapter(name).name == name
    assert get_adapter("humaneval").name == HUMANEVAL_PLUS
    assert get_adapter("mbpp").name == MBPP_PLUS


def test_task_type_and_scoring_modes():
    a = EvalPlusAdapter(HUMANEVAL_PLUS)
    assert a.task_type() is TaskType.CODE
    assert a.scoring_modes() == frozenset({ScoringMode.CACHED, ScoringMode.EXECUTION})


def test_unknown_benchmark_rejected():
    with pytest.raises(ValueError):
        EvalPlusAdapter("not_a_benchmark")
    with pytest.raises(ValueError):
        load_evalplus_tasks("nope", "test", None)


# --- schema ---


def test_reference_roundtrips_and_validates():
    ref = EvalPlusReference(
        entry_point="f", prompt="def f(): ...", canonical_solution="def f(): return 1\n",
        plus_test="def check(c):\n    assert c() == 1\n", base_test="def check(c):\n    assert c() == 1\n",
    )
    assert EvalPlusReference.from_dict(ref.to_dict()) == ref
    assert ref.is_valid()
    assert not EvalPlusReference(entry_point="f", plus_test="  ").is_valid()   # no harness
    assert not EvalPlusReference(entry_point="", plus_test="x").is_valid()      # no entry point


# --- loader / toy fallback ---


@pytest.mark.parametrize("bench", [HUMANEVAL_PLUS, MBPP_PLUS])
def test_toy_fallback_when_datasets_missing(monkeypatch, bench):
    monkeypatch.setitem(sys.modules, "datasets", None)   # import raises -> toy
    tasks = load_evalplus_tasks(bench, "test", None, seed=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.benchmark == bench and t.answer["entry_point"] == "add_one"
    assert t.answer["plus_test"].strip() and t.meta["task_type"] == TaskType.CODE.value
    assert t.meta["source"] == DATASETS[bench][0]


def test_loader_parses_hf_rows_and_is_deterministic(monkeypatch):
    _fake_datasets(monkeypatch, [{
        "task_id": "HumanEval/0",
        "prompt": "def f():\n    \"\"\"doc\"\"\"\n",
        "entry_point": "f",
        "canonical_solution": "def f():\n    return 1\n",
        "test": "def check(candidate):\n    assert candidate() == 1\n",
    }])
    a = load_evalplus_tasks(HUMANEVAL_PLUS, "test", None, seed=3)
    b = load_evalplus_tasks(HUMANEVAL_PLUS, "test", None, seed=3)
    assert [t.task_id for t in a] == [t.task_id for t in b] == ["HumanEval/0"]
    assert a[0].answer["plus_test"].startswith("def check")


def test_rows_missing_prompt_entry_or_test_are_skipped(monkeypatch):
    _fake_datasets(monkeypatch, [
        {"task_id": "a", "prompt": "", "entry_point": "f", "test": "x"},         # no prompt
        {"task_id": "b", "prompt": "p", "entry_point": "", "test": "x"},         # no entry point
        {"task_id": "c", "prompt": "p", "entry_point": "f", "test": ""},         # no test
    ])
    tasks = load_evalplus_tasks(HUMANEVAL_PLUS, "test", None, seed=0)
    assert len(tasks) == 1 and tasks[0].task_id.endswith("toy-0")   # all skipped -> toy


# --- prompt / extraction / placeholder ---


def test_prompt_includes_entry_point_and_instruction():
    p = build_evalplus_prompt("def f():\n    pass", "f")
    assert "def f()" in p and "```python" in p and "`f`" in p


def test_extract_prefers_fence_and_ignores_prose():
    assert extract_solution_code("x\n```python\ndef f():\n    return 1\n```\ny").strip() == "def f():\n    return 1"
    assert extract_solution_code("no code here, just prose") == ""
    assert extract_solution_code("import os\nx=1").startswith("import os")


def test_normalize_and_placeholder():
    assert normalize_code("```python\n\ndef f():\n    return 1\n\n```") == "def f():\n    return 1"
    ref = {"canonical_solution": "def f():\n    return 1\n"}
    assert score_solution_exact("```python\ndef f():\n    return 1\n```", ref) == 1.0
    assert score_solution_exact("```python\ndef f():\n    return 2\n```", ref) == 0.0
    assert score_solution_exact("def f(): return 1", {}) == 0.0   # no gold -> 0


def test_default_adapter_uses_placeholder_and_injected_runner():
    ref = {"canonical_solution": "def f():\n    return 1\n", "plus_test": "x"}
    assert EvalPlusAdapter(HUMANEVAL_PLUS).score_output("```python\ndef f():\n    return 1\n```", ref) == 1.0
    called = {}
    a = EvalPlusAdapter(MBPP_PLUS, runner=lambda o, r: (called.setdefault("x", 1), 1.0)[1])
    assert a.score_output("code", ref) == 1.0 and called
    assert EvalPlusAdapter(MBPP_PLUS, runner=lambda o, r: None).score_output("x", ref) == 0.0


def test_serialize_task_shape(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    task = load_evalplus_tasks(HUMANEVAL_PLUS, "test", 1, 0)[0]
    d = EvalPlusAdapter(HUMANEVAL_PLUS).serialize_task(task)
    assert set(d) == {"task_id", "benchmark", "prompt", "reference", "task_type", "meta"}
    assert d["reference"]["entry_point"] == "add_one"


def test_registered_adapter_executes_via_runner(monkeypatch):
    """Issue #255 review: the registered adapter must RUN the harness, not exact-match.

    A correct solution written differently from the canonical scores 0 under the
    placeholder but 1.0 when actually executed -- proving the runner is wired in.
    """
    monkeypatch.setitem(sys.modules, "datasets", None)   # force the deterministic toy task
    adapter = get_adapter(HUMANEVAL_PLUS)
    assert adapter._runner is not None
    task = load_evalplus_tasks(HUMANEVAL_PLUS, "test", None, seed=0)[0]
    ref = task.answer   # entry_point add_one
    different_but_correct = "```python\ndef add_one(x):\n    y = x\n    return y + 1\n```"
    # Exact-match placeholder would give 0.0 (source differs from canonical).
    assert score_solution_exact(different_but_correct, ref) == 0.0
    # The wired runner executes the base+plus harness and passes.
    assert adapter.score_output(different_but_correct, ref) == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
