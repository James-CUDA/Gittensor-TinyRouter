"""Absolute-value bars ``|x|`` must peel to the absolute value (issue #490)."""

from trinity.orchestration.reward import extract_last_number, normalize_math_answer, score_text


def test_extract_abs_as_whole_term():
    assert extract_last_number("|-3|") == "|-3|"
    assert extract_last_number("the value is |-3|") == "|-3|"


def test_normalize_abs_bars():
    assert "|" not in normalize_math_answer("|-3|")
    assert "abs" in normalize_math_answer("|-3|")


def test_unboxed_abs_grades_positive_not_signed():
    assert score_text("math500", "|-3|", "3") == 1.0
    assert score_text("math500", "|-3|", "-3") == 0.0


def test_boxed_abs():
    assert score_text("math500", r"\boxed{|-3|}", "3") == 1.0
    assert score_text("math500", r"\boxed{|5|}", "5") == 1.0
