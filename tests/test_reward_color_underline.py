"""Presentation color/underline wrappers must not affect math equality (issue #497)."""

from trinity.orchestration.reward import normalize_math_answer, score_text


def test_color_unwrap():
    assert normalize_math_answer(r"\color{red}{5}") == "5"
    assert score_text("math500", r"\boxed{\color{red}{5}}", "5") == 1.0
    assert score_text("math500", r"\boxed{\color{red}{6}}", "5") == 0.0


def test_textcolor_unwrap():
    assert normalize_math_answer(r"\textcolor{blue}{42}") == "42"
    assert score_text("math500", r"\boxed{\textcolor{blue}{42}}", "42") == 1.0


def test_underline_unwrap():
    assert normalize_math_answer(r"\underline{5}") == "5"
    assert score_text("math500", r"\boxed{\underline{5}}", "5") == 1.0
