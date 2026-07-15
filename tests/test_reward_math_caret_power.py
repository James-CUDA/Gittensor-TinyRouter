"""Regression: the caret ``^`` must mean exponentiation, not bitwise XOR (#342).

The symbolic-equality fallback parsed ``^`` with Python semantics (XOR) because it
omitted sympy's ``convert_xor`` transformation, so ``2^6`` evaluated to ``2 XOR 6 == 4``
instead of ``64``. That both credited wrong answers (``\\boxed{2^6}`` graded equal to
``4`` -- a false positive) and rejected correct ones (``2^3`` != ``8`` -- a false
negative) on MATH-500, where ``^`` universally reads "to the power of".

Pure / offline -- no network, no GPU. Requires sympy (a base dependency; CI has it).
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import math_equal, score_text

sympy = pytest.importorskip("sympy")


# --- false positives: XOR coincidences must NOT grade equal -----------------


@pytest.mark.parametrize(
    "candidate,reference",
    [
        ("2^6", "4"),    # 2 XOR 6 == 4, but 2**6 == 64
        ("5^1", "4"),    # 5 XOR 1 == 4, but 5**1 == 5
        ("6^3", "5"),    # 6 XOR 3 == 5, but 6**3 == 216
        ("2^3", "1"),    # 2 XOR 3 == 1, but 2**3 == 8
    ],
)
def test_xor_coincidence_is_not_equal(candidate, reference):
    assert math_equal(candidate, reference) is False


def test_end_to_end_wrong_power_scores_zero():
    # A wrong boxed answer must score 0.0, not 1.0.
    assert score_text("math500", r"\boxed{2^6}", "4") == 0.0


# --- false negatives: a real power must grade equal to its value -------------


@pytest.mark.parametrize(
    "candidate,reference",
    [
        ("2^3", "8"),
        ("2^6", "64"),
        ("10^6", "1000000"),
        ("3^4", "81"),
    ],
)
def test_power_equals_its_value(candidate, reference):
    assert math_equal(candidate, reference) is True


def test_end_to_end_correct_power_scores_one():
    assert score_text("math500", r"\boxed{2^3}", "8") == 1.0


# --- controls: unrelated forms unchanged ------------------------------------


@pytest.mark.parametrize(
    "candidate,reference,expected",
    [
        ("2^{10}", "2^{10}", True),   # identical form
        ("1/2", "0.5", True),
        (r"\sqrt{4}", "2", True),
        ("7", "8", False),
    ],
)
def test_controls_unchanged(candidate, reference, expected):
    assert math_equal(candidate, reference) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
