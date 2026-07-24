"""\\displaystyle and \\tag must not break math equality (issue #506)."""
from trinity.orchestration.reward import normalize_math_answer, score_text

def test_displaystyle_frac():
    assert "displaystyle" not in normalize_math_answer(r"\displaystyle\frac{1}{2}")
    assert score_text("math500", r"\boxed{\displaystyle\frac{1}{2}}", "1/2") == 1.0

def test_tag_stripped():
    assert "tag" not in normalize_math_answer(r"5\tag{1}")
    assert score_text("math500", r"\boxed{5\tag{1}}", "5") == 1.0
    assert score_text("math500", r"\boxed{5\tag{1}}", "6") == 0.0
