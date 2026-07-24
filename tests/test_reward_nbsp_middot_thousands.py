"""NBSP and middle-dot thousands must equal 1000 (issue #507)."""
from trinity.orchestration.reward import extract_last_number, score_text

def test_nbsp_thousands():
    assert extract_last_number("1\u00a0000") == "1000"
    assert score_text("math500", "1\u00a0000", "1000") == 1.0

def test_middot_thousands():
    assert extract_last_number("1·000") == "1000"
    assert score_text("math500", "1·000", "1000") == 1.0
    assert score_text("math500", "1·000", "000") == 0.0
