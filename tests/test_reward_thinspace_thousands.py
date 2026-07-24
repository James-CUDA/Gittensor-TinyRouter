"""LaTeX thin-space thousands must extract as one number (#478)."""
from __future__ import annotations
from trinity.orchestration import reward as R

def test_extract_thinspace_thousands():
    assert R.extract_last_number(r"1\,000") == "1000"
    assert R.extract_last_number(r"1\,234\,567") == "1234567"
    assert R.extract_last_number(r"1\:000") == "1000"
    assert R.extract_last_number(r"1\;000") == "1000"

def test_score_thinspace_thousands():
    assert R.score_text("math500", r"1\,000", "1000") == 1.0
    assert R.score_text("math500", r"The answer is 1\,000.", "1000") == 1.0
