"""Thousands-comma stripper must not merge set/tuple/list elements (issue #296).

The grader strips digit-grouping commas so ``2,000`` matches ``2000``, but the
same regex over-reached on structured answers like ``{2, 100}`` and ``(5, 120)``,
grading them equal to the concatenated scalar — a false positive in the reward.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_set_with_three_digit_element_is_not_merged_to_scalar():
    assert normalize_math_answer("{2, 100}") == "2,100"
    assert math_equal("{2, 100}", "2100") is False
    assert score_text("math500", r"\boxed{\{2, 100\}}", "2100") == 0.0


def test_tuple_with_three_digit_element_is_not_merged_to_scalar():
    assert normalize_math_answer("(5, 120)") == "(5,120)"
    assert math_equal("(5, 120)", "5120") is False
    assert score_text("math500", r"\boxed{(5, 120)}", "5120") == 0.0


def test_real_thousands_separator_still_matches():
    assert math_equal("2,000", "2000") is True
    assert score_text("math500", r"\boxed{2,000}", "2000") == 1.0


def test_small_element_set_stays_unequal():
    assert score_text("math500", r"\boxed{\{2, 10\}}", "210") == 0.0


def test_unicode_degree_glyph_matches_plain_reference():
    assert normalize_math_answer("90°") == "90"
    assert score_text("math500", r"\boxed{90°}", "90") == 1.0
