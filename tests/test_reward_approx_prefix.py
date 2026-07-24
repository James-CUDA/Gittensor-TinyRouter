"""Approximation prefixes in boxed math answers are presentation (#474)."""

from __future__ import annotations

from trinity.orchestration import reward as R


def test_normalize_strips_approx_family() -> None:
    assert R.normalize_math_answer(r"\approx 12") == "12"
    assert R.normalize_math_answer(r"\sim 3") == "3"
    assert R.normalize_math_answer(r"\simeq 3") == "3"
    assert "≈" not in R.normalize_math_answer("≈12")
    assert R.normalize_math_answer("≈12") == "12"


def test_boxed_approx_matches_gold_number() -> None:
    assert R.score_text("math500", r"\boxed{\approx 12}", "12") == 1.0
    assert R.score_text("math500", r"\boxed{\sim 3}", "3") == 1.0


def test_pm_is_not_stripped() -> None:
    # \pm changes meaning — must not collapse to the magnitude alone.
    assert R.score_text("math500", r"\boxed{\pm 5}", "5") == 0.0
