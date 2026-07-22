"""Un-boxed LaTeX ``\\frac``/``\\sqrt`` final answers must be extracted whole.

``extract_last_number`` is the math fallback when no ``\\boxed{...}`` is
present. Its digit-only alternatives read an inline-LaTeX answer such as
``$\\frac{1}{2}$`` as its operand digits and return the denominator (``2``) —
and ``$2\\sqrt{3}$`` as the radicand (``3``). The grader then compares a value
the model never answered: the true reference fails (false negative) and a
reference that happens to equal the operand digit passes (false positive).
This also poisons ``has_answer``, committed-answer selection, and the HERO
self-consistency vote, which all share the extractor.

These tests pin the fix: LaTeX fraction/radical terms are captured as one
token, while the "last plain number still wins" contract is preserved.
"""
from __future__ import annotations

from trinity.orchestration import reward as R


# --- extraction: the LaTeX term is one token ---


def test_extracts_unboxed_frac_whole():
    assert R.extract_last_number(r"So the answer is $\frac{1}{2}$.") == r"\frac{1}{2}"


def test_extracts_unboxed_sqrt_with_coefficient_whole():
    assert R.extract_last_number(r"Therefore the area is $2\sqrt{3}$.") == r"2\sqrt{3}"


def test_extracts_frac_with_nested_operand_whole():
    text = r"Thus x = \frac{\sqrt{2}}{2} is the final value."
    assert R.extract_last_number(text) == r"\frac{\sqrt{2}}{2}"


def test_extracts_dfrac_and_digit_pair_spellings():
    assert R.extract_last_number(r"answer: $\dfrac{3}{4}$") == r"\dfrac{3}{4}"
    assert R.extract_last_number(r"answer: $\frac12$") == r"\frac12"


def test_extracts_negative_frac_with_sign():
    assert R.extract_last_number(r"we get $-\frac{1}{2}$") == r"-\frac{1}{2}"


def test_last_plain_number_still_wins():
    # The "final answers come last" contract is unchanged: a plain number
    # stated after a fraction is still the committed answer.
    text = r"halfway we had \frac{1}{2}, so the count is 42."
    assert R.extract_last_number(text) == "42"


def test_plain_numbers_unaffected():
    assert R.extract_last_number("first 1/2 then finally 42") == "42"
    assert R.extract_last_number("The total is 1,234.50 dollars.") == "1234.50"
    assert R.extract_last_number("no digits") is None


# --- grading: end-to-end through score_text ---


def test_unboxed_frac_scores_correct_reference():
    cand = r"So the answer is $\frac{1}{2}$."
    assert R.score_text("math500", cand, r"\frac{1}{2}") == 1.0
    assert R.score_text("math500", cand, "1/2") == 1.0


def test_unboxed_frac_no_longer_matches_its_denominator():
    # Before the fix the denominator "2" was extracted, so a WRONG answer
    # scored 1.0 against a reference of 2.
    assert R.score_text("math500", r"So the answer is $\frac{1}{2}$.", "2") == 0.0


def test_unboxed_sqrt_scores_both_directions():
    cand = r"Therefore the area is $2\sqrt{3}$."
    assert R.score_text("math500", cand, r"2\sqrt{3}") == 1.0
    assert R.score_text("math500", cand, "3") == 0.0
