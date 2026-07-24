"""Classic TeX ``\\over`` fractions must not grade as the denominator (issue #484)."""

from trinity.orchestration.reward import extract_last_number, normalize_math_answer, score_text


def test_extract_over_as_whole_term():
    assert extract_last_number(r"1\over 2") == r"1\over 2"
    assert extract_last_number(r"{1\over 2}") == r"{1\over 2}"
    assert extract_last_number(r"answer is $1\over 2$") == r"1\over 2"


def test_normalize_over_to_slash():
    assert "\\over" not in normalize_math_answer(r"1\over 2")
    assert "\\over" not in normalize_math_answer(r"{1\over 2}")
    assert normalize_math_answer(r"1\over 2") == "1/2"


def test_unboxed_over_grades_as_fraction_not_denominator():
    assert score_text("math500", r"the answer is $1\over 2$", "1/2") == 1.0
    assert score_text("math500", r"the answer is $1\over 2$", "2") == 0.0
    assert score_text("math500", r"${1\over 2}$", r"\frac{1}{2}") == 1.0


def test_boxed_over_equals_frac_gold():
    assert score_text("math500", r"\boxed{1\over 2}", "1/2") == 1.0
    assert score_text("math500", r"\boxed{{1\over 2}}", r"\frac{1}{2}") == 1.0
