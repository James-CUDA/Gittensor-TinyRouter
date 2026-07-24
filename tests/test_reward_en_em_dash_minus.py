"""Unary en/em dashes must grade like ASCII minus (#473)."""

from __future__ import annotations

from trinity.orchestration import reward as R


def test_extract_keeps_en_dash_sign() -> None:
    assert R.extract_last_number("The answer is –3.") == "-3"
    assert R.extract_last_number("The answer is —3.") == "-3"


def test_extract_keeps_year_range_unsigned() -> None:
    # Interior en dash is a range separator, not a unary minus.
    assert R.extract_last_number("from 1994–1995 inclusive") == "1995"


def test_score_text_en_em_dash_minus() -> None:
    for dash in ("–", "—"):
        c = f"The answer is {dash}3."
        assert R.score_text("math500", c, "-3") == 1.0
        assert R.score_text("math500", c, "3") == 0.0


def test_normalize_folds_unary_en_dash() -> None:
    assert R.normalize_math_answer("–3") == "-3"
    assert R.normalize_math_answer("—3") == "-3"
