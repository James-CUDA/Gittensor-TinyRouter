"""Indexed roots ``\\sqrt[n]{x}`` must not grade as the radicand digit (issue #483)."""

from trinity.orchestration.reward import extract_last_number, normalize_math_answer, score_text


def test_extract_takes_whole_indexed_sqrt():
    assert extract_last_number(r"\sqrt[3]{8}") == r"\sqrt[3]{8}"
    assert extract_last_number(r"the cube root is $\sqrt[3]{8}$") == r"\sqrt[3]{8}"


def test_normalize_indexed_sqrt_to_power():
    assert "sqrt" not in normalize_math_answer(r"\sqrt[3]{8}")
    assert "\\sqrt" not in normalize_math_answer(r"\sqrt[3]{8}")


def test_unboxed_nth_root_grades_value_not_operand():
    assert score_text("math500", r"the answer is $\sqrt[3]{8}$", "2") == 1.0
    assert score_text("math500", r"the answer is $\sqrt[3]{8}$", "8") == 0.0


def test_boxed_nth_root_equals_simplified_value():
    assert score_text("math500", r"\boxed{\sqrt[3]{8}}", "2") == 1.0
    assert score_text("math500", r"\boxed{\sqrt[4]{16}}", "2") == 1.0
