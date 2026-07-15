"""Regression: a "solve for x" answer that carries the variable must still match (#348).

MATH-500 "solve for x" problems have a bare-value reference (e.g. `5`) while models
commonly commit the equation they solved (`\\boxed{x=5}`). ``normalize_math_answer``
already strips a leading bare `=` but not a leading single-letter assignment, so
`x=5` never matched `5` -- a false negative. The strip must be conservative: only a
single leading letter directly followed by `=` is removed, so multi-char tokens
(`log=2`) and genuinely-different answers are unaffected.

Pure / offline -- no network, no GPU.
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


@pytest.mark.parametrize(
    "candidate,reference",
    [
        (r"\boxed{x=5}", "5"),
        (r"\boxed{x = 5}", "5"),
        (r"\boxed{n=10}", "10"),
        (r"\boxed{y = -3}", "-3"),
    ],
)
def test_equation_form_answer_matches_bare_value(candidate, reference):
    assert score_text("math500", candidate, reference) == 1.0


def test_normalizer_strips_single_variable_assignment():
    assert normalize_math_answer("x=5") == "5"
    assert normalize_math_answer("x = 5") == "5"
    # Consistent with the pre-existing leading bare "=" strip.
    assert normalize_math_answer("=5") == "5"


# --- must NOT over-strip / create false positives ---------------------------


def test_multichar_token_is_not_stripped():
    # Only a SINGLE leading letter+"=" is removed; "log=2" keeps its left side.
    assert math_equal("log=2", "2") is False


def test_wrong_equation_answer_still_fails():
    assert math_equal("x=5", "6") is False


def test_bare_value_control_unchanged():
    assert score_text("math500", r"\boxed{5}", "5") == 1.0


def test_expression_left_side_preserved_on_self_compare():
    # Stripping is symmetric, so an equation still equals itself.
    assert math_equal("y=2x+3", "y=2x+3") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
