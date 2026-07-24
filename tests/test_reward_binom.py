"""Binomial ``\\binom`` / ``\\choose`` must not grade as the lower index (issue #489)."""

from trinity.orchestration.reward import extract_last_number, normalize_math_answer, score_text


def test_extract_binom_as_whole_term():
    assert extract_last_number(r"\binom{5}{2}") == r"\binom{5}{2}"
    assert extract_last_number(r"{5 \choose 2}") == r"{5 \choose 2}"


def test_normalize_binom_to_sympy():
    assert "binomial(5,2)" in normalize_math_answer(r"\binom{5}{2}").replace(" ", "")
    assert "\\binom" not in normalize_math_answer(r"\binom{5}{2}")


def test_unboxed_binom_grades_value_not_index():
    assert score_text("math500", r"answer is $\binom{5}{2}$", "10") == 1.0
    assert score_text("math500", r"answer is $\binom{5}{2}$", "2") == 0.0


def test_boxed_binom_and_choose():
    assert score_text("math500", r"\boxed{\binom{5}{2}}", "10") == 1.0
    assert score_text("math500", r"\boxed{{5 \choose 2}}", "10") == 1.0
