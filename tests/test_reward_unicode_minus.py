"""The Unicode minus sign U+2212 ("−") must grade like the ASCII hyphen.

Rendered math uses U+2212 for negation, and models that echo rendered output
emit it. Neither the extraction regexes nor ``normalize_math_answer`` knew the
glyph, so a negative answer written with it was mis-graded in both directions:

* un-boxed ``"The answer is −3."`` extracted the unsigned ``"3"`` — scoring
  1.0 against a reference of ``3`` (sign-flip false positive) and 0.0 against
  the true ``-3`` (false negative);
* boxed ``\\boxed{−3}`` reached the normalizer intact but never equalled a
  plain ``-3`` — ``float()`` and sympy both reject the glyph (false negative).

The fix folds U+2212 to ``-`` at the top of both ``extract_last_number`` and
``normalize_math_answer`` (mirroring the existing ``π`` glyph fold). En/em
dashes are deliberately NOT folded — only U+2212 is unambiguously a minus.
"""
from __future__ import annotations

from trinity.orchestration import reward as R


# --- extraction ---


def test_extracts_unicode_minus_as_signed_number():
    assert R.extract_last_number("The answer is \u22123.") == "-3"


def test_extracts_unicode_minus_decimal():
    assert R.extract_last_number("we get \u22120.5 overall") == "-0.5"


# --- normalization ---


def test_normalizes_unicode_minus_to_ascii():
    assert R.normalize_math_answer("\u22123") == "-3"
    assert R.math_equal("\u22123", "-3") is True


# --- grading: end-to-end through score_text ---


def test_unboxed_unicode_minus_scores_both_directions():
    cand = "The answer is \u22123."
    assert R.score_text("math500", cand, "-3") == 1.0
    # Sign-flip false positive before the fix: unsigned "3" was extracted.
    assert R.score_text("math500", cand, "3") == 0.0


def test_boxed_unicode_minus_matches_plain_reference():
    assert R.score_text("math500", "\\boxed{\u22123}", "-3") == 1.0


def test_unicode_minus_fraction():
    assert R.score_text("math500", "\\boxed{\u2212\\frac{1}{2}}", "-1/2") == 1.0


def test_ascii_minus_behavior_unchanged():
    assert R.extract_last_number("The answer is -3.") == "-3"
    assert R.score_text("math500", "The answer is -3.", "-3") == 1.0
    assert R.score_text("math500", "The answer is -3.", "3") == 0.0
