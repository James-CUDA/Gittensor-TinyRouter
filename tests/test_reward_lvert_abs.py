"""LaTeX \\lvert/\\rvert absolute value must peel (issue #505)."""
from trinity.orchestration.reward import normalize_math_answer, score_text

def test_lvert_boxed():
    assert "|" not in normalize_math_answer(r"\lvert -4\rvert") or "abs" in normalize_math_answer(r"\lvert -4\rvert")
    assert score_text("math500", r"\boxed{\lvert -4\rvert}", "4") == 1.0
    assert score_text("math500", r"\boxed{\lvert -4\rvert}", "-4") == 0.0

def test_left_right_bars():
    assert score_text("math500", r"\boxed{\left|-4\right|}", "4") == 1.0

def test_ascii_bars_too():
    assert score_text("math500", r"\boxed{|-4|}", "4") == 1.0
