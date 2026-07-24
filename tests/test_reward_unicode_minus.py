"""Unicode minus U+2212 must grade like ASCII hyphen (#460)."""

from __future__ import annotations

from trinity.orchestration import reward as R


def test_extract_last_number_keeps_unicode_minus_sign() -> None:
    assert R.extract_last_number("The answer is −3.") == "-3"


def test_score_text_math500_unicode_minus() -> None:
    c = "The answer is −3."
    assert R.score_text("math500", c, "-3") == 1.0
    assert R.score_text("math500", c, "3") == 0.0
    assert R.score_text("math500", "\boxed{−3}", "-3") == 1.0


def test_normalize_folds_unicode_minus() -> None:
    assert R.normalize_math_answer("−3") == R.normalize_math_answer("-3")
