"""Regression: the MMLU loader must bound the answer index by the row's options.

_load_mmlu_hf validated ``answer`` against len(_CHOICE_LETTERS) (10, widened for
MMLU-Pro) rather than the row's own number of choices, so a malformed row -- e.g.
4 choices with answer=7 -- produced a gold letter ("H") indexing a non-existent
option. No network, no GPU (the HF loader is stubbed).
"""
from __future__ import annotations

import trinity.adapters.loaders as L


def _load(rows, monkeypatch):
    # _load_mmlu_hf iterates whatever _try_load_hf returns and reads rows via
    # _row_get (dict access), so a list of dicts is a faithful stand-in for the
    # HF dataset -- no network / datasets needed.
    monkeypatch.setattr(L, "_try_load_hf", lambda *a, **k: rows)
    return L._load_mmlu_hf("test") or []


def test_answer_beyond_the_rows_options_is_dropped(monkeypatch):
    rows = [
        {"question": "Q1", "choices": ["a", "b", "c", "d"], "answer": 1},   # valid -> B
        {"question": "Q2", "choices": ["a", "b", "c", "d"], "answer": 7},   # malformed
    ]
    tasks = _load(rows, monkeypatch)
    # Only the valid row survives; the out-of-range one is skipped, not mislabeled.
    assert [t.answer for t in tasks] == ["B"]
    # And no surviving task carries a letter past its own options.
    for t in tasks:
        assert t.answer in "ABCD"[: len(t.meta["choices"])]


def test_in_range_answers_map_to_the_right_letter(monkeypatch):
    rows = [{"question": "Q", "choices": ["a", "b", "c", "d"], "answer": idx}
            for idx in range(4)]
    tasks = _load(rows, monkeypatch)
    assert [t.answer for t in tasks] == ["A", "B", "C", "D"]


def test_negative_answer_is_dropped(monkeypatch):
    rows = [{"question": "Q", "choices": ["a", "b"], "answer": -1}]
    assert _load(rows, monkeypatch) == []


def test_all_malformed_yields_none_not_bogus_tasks(monkeypatch):
    rows = [{"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 9}]
    monkeypatch.setattr(L, "_try_load_hf", lambda *a, **k: rows)
    assert L._load_mmlu_hf("test") is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
