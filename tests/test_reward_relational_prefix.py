"""Relational prefixes must not grade as bare magnitudes (#479)."""
from __future__ import annotations
from trinity.orchestration import reward as R

def test_inequality_does_not_match_bare_gold():
    for cand in (r"\neq 5", r"\ne 5", r"\le 5", r"\leq 5", r"\ge 5", r"\geq 5", r"\lt 5", r"\gt 5", "≠5", "≤5", "≥5"):
        assert R.score_text("math500", cand, "5") == 0.0, cand

def test_same_relation_still_matches():
    assert R.score_text("math500", r"\neq 5", r"\neq 5") == 1.0
    assert R.score_text("math500", r"\le 5", r"\le 5") == 1.0

def test_bare_number_unaffected():
    assert R.score_text("math500", "5", "5") == 1.0
    assert R.score_text("math500", "The answer is 5.", "5") == 1.0
    assert R.extract_last_number(r"\neq 5") == r"\neq5"
