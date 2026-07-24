"""Unicode Greek letters must equal LaTeX spellings (issue #498)."""

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_theta_latex_equals_unicode():
    assert normalize_math_answer(r"2\theta") == normalize_math_answer("2θ")
    assert math_equal(r"2\theta", "2θ")
    assert score_text("math500", r"\boxed{2\theta}", "2θ") == 1.0


def test_alpha_phi():
    assert math_equal(r"\alpha", "α")
    assert math_equal(r"\phi", "φ")
    assert score_text("math500", r"\boxed{\alpha}", "α") == 1.0


def test_wrong_greek_still_fails():
    assert score_text("math500", r"\boxed{\theta}", "α") == 0.0
