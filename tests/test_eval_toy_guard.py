"""Evaluation must not silently run on the offline toy set.

``build_benchmark`` and ``trinity.train`` already escalate
:class:`~trinity.adapters.split_policy.ToyFallbackWarning` to a hard error.
``trinity.eval`` had the same exposure: a failed HuggingFace load substituted
2-3 toy questions and reported R1/R2/R4 as if real — the JOURNAL follow-up from
the hidden-benchmark guard (2026-07-10).

Offline: no network. ``datasets`` is faked (or removed) per test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.adapters import get_adapter  # noqa: E402
from trinity.adapters.split_policy import ToyFallbackWarning  # noqa: E402
from trinity.eval import _load_eval_tasks  # noqa: E402


def _install_missing_datasets(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)


def test_toy_fallback_aborts_evaluation(monkeypatch):
    _install_missing_datasets(monkeypatch)
    adapter = get_adapter("mmlu")
    with pytest.raises(RuntimeError, match="Refusing to evaluate on the offline toy set"):
        _load_eval_tasks(adapter, "mmlu", max_items=10, seed=0)


def test_abort_message_names_the_benchmark(monkeypatch):
    _install_missing_datasets(monkeypatch)
    adapter = get_adapter("gpqa")
    with pytest.raises(RuntimeError, match="'gpqa'"):
        _load_eval_tasks(adapter, "gpqa", max_items=10, seed=0)


def test_abort_chains_the_original_warning(monkeypatch):
    _install_missing_datasets(monkeypatch)
    adapter = get_adapter("mmlu")
    with pytest.raises(RuntimeError) as excinfo:
        _load_eval_tasks(adapter, "mmlu", max_items=10, seed=0)
    assert isinstance(excinfo.value.__cause__, ToyFallbackWarning)
