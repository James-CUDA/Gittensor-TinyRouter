"""Nested braced \\frac operands must unwrap (issue #409)."""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_frac_with_sqrt_numerator_matches_slash_form():
    assert "\\frac" not in normalize_math_answer(r"\frac{\sqrt{2}}{2}")
    assert math_equal(r"\frac{\sqrt{2}}{2}", r"\sqrt{2}/2")
    # End-to-end grader path extracts \\boxed before normalize/compare.
    assert score_text("math500", r"\boxed{\frac{\sqrt{2}}{2}}", r"\sqrt{2}/2") == 1.0


def test_plain_frac_still_matches():
    assert math_equal(r"\frac{1}{2}", "1/2")
    assert math_equal(r"\dfrac{3}{4}", r"\tfrac{3}{4}")


def test_frac_with_superscript_operand():
    # Braced superscript in the numerator is the same [^{}]+ failure mode.
    # Exact normalize must agree — do not rely on the sympy fallback.
    assert "\\frac" not in normalize_math_answer(r"\frac{x^{2}}{2}")
    assert normalize_math_answer(r"\frac{x^{2}}{2}") == normalize_math_answer(r"x^{2}/2")
    assert math_equal(r"\frac{x^{2}}{2}", r"x^{2}/2")
    assert normalize_math_answer(r"\frac{2^{10}}{2}") == normalize_math_answer(r"2^{10}/2")
    assert math_equal(r"\frac{2^{10}}{2}", r"2^{10}/2")
