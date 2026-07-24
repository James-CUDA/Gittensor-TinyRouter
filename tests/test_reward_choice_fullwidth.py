"""Fullwidth （B）, boxed B., 答案是B (issue #508)."""
from trinity.orchestration.reward import extract_choice_letter, score_text

def test_fullwidth_parens():
    assert extract_choice_letter("（B）") == "B"
    assert score_text("mmlu", "（B）", "B") == 1.0

def test_boxed_with_period():
    assert extract_choice_letter(r"\boxed{B.}") == "B"
    assert score_text("mmlu", r"\boxed{B.}", "B") == 1.0

def test_answer_is_chinese():
    assert extract_choice_letter("答案是B") == "B"
    assert score_text("mmlu", "答案是B", "B") == 1.0
    assert score_text("mmlu", "答案是B", "C") == 0.0
