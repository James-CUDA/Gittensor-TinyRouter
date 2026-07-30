"""Choice-letter gaps: [B], ***B***, 答案：B (issue #491)."""

from trinity.orchestration.reward import extract_choice_letter, score_text


def test_bracketed_choice_letter():
    assert extract_choice_letter("[B]") == "B"
    assert score_text("mmlu", "[B]", "B") == 1.0
    assert score_text("mmlu", "[B]", "C") == 0.0


def test_triple_star_md_emphasis():
    assert extract_choice_letter("***B***") == "B"
    assert score_text("mmlu", "***B***", "B") == 1.0
    # existing double-star still works
    assert extract_choice_letter("**B**") == "B"


def test_chinese_answer_cue():
    assert extract_choice_letter("答案：B") == "B"
    assert extract_choice_letter("答案:C") == "C"
    assert score_text("mmlu", "答案：B", "B") == 1.0
