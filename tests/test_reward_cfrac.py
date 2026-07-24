"""\\cfrac must unwrap like \\frac / \\dfrac / \\tfrac (#477)."""
from __future__ import annotations
from trinity.orchestration import reward as R

def test_cfrac_normalizes_like_frac():
    assert "\\cfrac" not in R.normalize_math_answer(r"\cfrac{1}{2}")
    assert R.math_equal(r"\cfrac{1}{2}", "1/2")
    assert R.score_text("math500", r"\boxed{\cfrac{1}{2}}", "1/2") == 1.0
