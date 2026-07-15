"""Regression for math_equal leading-zero / empty-string handling (#319/#331).

``_is_zero_padded`` must reject AIME-style zero-padded integers without crashing
on empty answers (adapter conformance probes ``score_output("")``).
"""
from trinity.orchestration.reward import math_equal, score_text


def test_math_equal_empty_strings_do_not_raise():
    assert math_equal("", "") is False
    assert math_equal("", "42") is False
    assert math_equal("42", "") is False
    assert math_equal(None, "42") is False
    assert math_equal("42", None) is False


def test_math_equal_rejects_zero_padded_integer_vs_unpadded():
    assert math_equal("005", "5") is False
    assert math_equal("5", "005") is False
    assert math_equal("005", "009") is False
    assert math_equal("005", "005") is True


def test_math_equal_preserves_decimal_leading_zero():
    assert math_equal("0.5", "1/2") is True


def test_score_text_math500_empty_probe_is_binary_zero():
    # Mirrors adapters.conformance._BINARY_PROBES[0] against a real math answer.
    assert score_text("math500", "", "42") == 0.0
