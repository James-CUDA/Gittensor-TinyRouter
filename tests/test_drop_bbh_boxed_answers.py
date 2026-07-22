"""DROP/BBH must accept \\boxed answers that format_hint requests (issue #437)."""
from __future__ import annotations

from trinity.adapters.bbh import score_bbh
from trinity.adapters.drop import score_drop


def test_drop_accepts_boxed_answer():
    assert score_drop(r"Answer: \boxed{21}", {"gold_answers": ["21"]}) == 1.0
    assert score_drop(r"Answer: 21", {"gold_answers": ["21"]}) == 1.0


def test_bbh_exact_accepts_boxed_answer():
    assert score_bbh(r"\boxed{True}", {"answer": "True", "answer_type": "exact_match"}) == 1.0
    assert score_bbh("True", {"answer": "True", "answer_type": "exact_match"}) == 1.0
