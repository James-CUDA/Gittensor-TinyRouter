"""Leading-decimal DROP tokens must keep their value (issue #423)."""
from __future__ import annotations

from trinity.adapters.drop import _normalize_tokens, score_drop


def test_leading_decimal_normalizes_like_zero_point_five():
    assert _normalize_tokens(".5") == ["0.5"]
    assert _normalize_tokens("0.5") == ["0.5"]
    assert _normalize_tokens("-.5") == ["-0.5"]
    assert _normalize_tokens(".5.") == ["0.5"]


def test_leading_decimal_matches_zero_point_five_gold():
    assert score_drop("Answer: .5", {"gold_answers": ["0.5"]}) == 1.0
    assert score_drop("Answer: .5.", {"gold_answers": ["0.5"]}) == 1.0
    assert score_drop("Answer: .5", {"gold_answers": ["5"]}) == 0.0


def test_currency_and_trailing_point_still_normalize():
    assert _normalize_tokens("$16") == ["16.0"]
    assert _normalize_tokens("$.5") == ["0.5"]
    assert _normalize_tokens("16.") == ["16.0"]
    assert _normalize_tokens("1,234.5") == ["1234.5"]
    assert _normalize_tokens("-5") == ["-5.0"]
