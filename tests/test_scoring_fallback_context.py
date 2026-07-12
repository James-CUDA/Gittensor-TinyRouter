"""Regression: the execution-only fallback in score_item must be context-free.

When a context was supplied but execution returned None ("could not execute"),
score_item's final fallback must call score_execution WITHOUT a context (as its
own comment says) -- not re-pass the failing context, which both drops the
adapter's context-free score and re-runs the (expensive) execution a second time.
No network, no GPU.
"""
from __future__ import annotations

from trinity.adapters.base import BenchmarkAdapter, ScoringMode, TaskType
from trinity.adapters.scoring import score_item


class _ExecOnly(BenchmarkAdapter):
    """Execution-only adapter that can only score WITHOUT a context.

    Mirrors an adapter whose prepared executor failed (returns None with a
    context) but which still has a context-free fallback path (returns 1.0).
    """

    name = "exec-only"

    def __init__(self) -> None:
        self.contexts: list[object] = []

    def load_tasks(self, split, max_items, seed=0):  # noqa: D102
        return []

    def build_prompt(self, task):  # noqa: D102
        return ""

    def score_output(self, output, reference):  # noqa: D102
        return 0.0

    def task_type(self):  # noqa: D102
        return TaskType.PATCH

    def serialize_task(self, task):  # noqa: D102
        return {}

    def scoring_modes(self):  # noqa: D102
        return frozenset({ScoringMode.EXECUTION})

    def score_execution(self, output, reference, *, context=None):  # noqa: D102
        self.contexts.append(context)
        return 1.0 if context is None else None


def test_fallback_is_context_free_and_returns_the_score():
    adapter = _ExecOnly()
    out = score_item(adapter, "x", None, execution_context={"broken": True})
    # The context-free fallback score is used, not dropped to 0.0.
    assert out.reward == 1.0
    assert out.mode is ScoringMode.EXECUTION


def test_fallback_does_not_re_use_the_failed_context():
    adapter = _ExecOnly()
    score_item(adapter, "x", None, execution_context={"broken": True})
    # Two calls: the first attempt (with context) failed, the fallback is None.
    assert adapter.contexts == [{"broken": True}, None]


def test_successful_execution_short_circuits_without_the_fallback():
    # If the first execution succeeds, the fallback is never reached.
    class _Ok(_ExecOnly):
        def score_execution(self, output, reference, *, context=None):
            self.contexts.append(context)
            return 1.0

    adapter = _Ok()
    out = score_item(adapter, "x", None, execution_context={"ready": True})
    assert out.reward == 1.0 and adapter.contexts == [{"ready": True}]


def test_no_context_supplied_still_scores_context_free():
    adapter = _ExecOnly()
    out = score_item(adapter, "x", None, execution_context=None)
    assert out.reward == 1.0 and adapter.contexts == [None]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
