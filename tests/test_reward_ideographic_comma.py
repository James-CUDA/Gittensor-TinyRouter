"""Ideographic thousands comma ， must behave like ASCII , (issue #500)."""

from trinity.orchestration.reward import extract_last_number, score_text


def test_extract_ideographic_thousands():
    assert extract_last_number("1，000") == "1000"
    assert extract_last_number("总计 12，345") == "12345"


def test_score_ideographic_thousands():
    assert score_text("math500", "1，000", "1000") == 1.0
    assert score_text("math500", r"\boxed{1，000}", "1000") == 1.0
    assert score_text("math500", "1，000", "000") == 0.0
