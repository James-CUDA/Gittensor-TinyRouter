"""Tests for DROP eval-split resolution and the loud toy fallback.

``ucinlp/drop`` publishes only ``train`` and ``validation`` — DROP's ``test``
set is hidden (leaderboard-only). Before the fix, every eval-path request
(``load_tasks("test", ...)``) raised inside ``load_dataset``, was swallowed,
and the loader silently served the 2-item toy set: eval reported
leaderboard-grade numbers computed from two toy questions, and — unlike
MMLU (#35), MMLU-Pro (#50), GPQA (#95), SWE-bench (#196) and AIME — DROP never
emitted :class:`ToyFallbackWarning`, so the eval/train toy guards that exist
precisely to refuse such runs could not fire.

Same layout as ``test_mmlu_pro_train_split.py`` (the #50 regression tests).
"""
from __future__ import annotations

import sys
import types
import warnings

from trinity.adapters.drop import load_drop_tasks
from trinity.adapters.split_policy import ToyFallbackWarning, resolve_split


def _install_fake_drop_datasets(monkeypatch) -> None:
    """Simulate ucinlp/drop, which has only ``train`` and ``validation`` splits."""
    real_splits = {"ucinlp/drop": {"train", "validation"}}

    def load_dataset(path, name=None, split=None, **kwargs):
        if split not in real_splits[path]:
            raise ValueError(
                f"Unknown split {split!r}. Should be one of {sorted(real_splits[path])}."
            )
        return [
            {
                "passage": f"Passage {split}-{i}: 12 houses in 2018, 9 in 2019.",
                "question": f"q-{split}-{i}?",
                "answers_spans": {"spans": [f"a-{split}-{i}"], "types": ["span"]},
            }
            for i in range(50)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_resolve_split_maps_test_to_validation():
    assert resolve_split("drop", "test") == "validation"
    assert resolve_split("drop", "eval") == "validation"
    # The upstream ``train`` split is real; it must pass through untouched.
    assert resolve_split("drop", "train") == "train"
    assert resolve_split("drop", "validation") == "validation"


def test_test_split_loads_validation_not_toy_set(monkeypatch):
    _install_fake_drop_datasets(monkeypatch)
    tasks = load_drop_tasks("test", max_items=5, seed=0)
    assert len(tasks) == 5
    # Real rows, not the built-in 2-item toy set.
    assert all(t.answer["gold_answers"][0].startswith("a-validation-") for t in tasks)


def test_train_split_still_reads_upstream_train(monkeypatch):
    _install_fake_drop_datasets(monkeypatch)
    tasks = load_drop_tasks("train", max_items=3, seed=0)
    assert len(tasks) == 3
    assert all(t.answer["gold_answers"][0].startswith("a-train-") for t in tasks)


def test_toy_fallback_emits_warning_when_hf_missing(monkeypatch):
    monkeypatch.setattr("trinity.adapters.drop._hf_drop", lambda _split: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ToyFallbackWarning)
        tasks = load_drop_tasks("test", max_items=None, seed=0)
    assert len(tasks) == 2
    assert any(isinstance(w.message, ToyFallbackWarning) for w in caught)


def test_real_load_does_not_warn(monkeypatch):
    _install_fake_drop_datasets(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ToyFallbackWarning)  # guard mode, as in eval
        tasks = load_drop_tasks("test", max_items=2, seed=0)
    assert len(tasks) == 2
