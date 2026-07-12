"""Regression: extract_choice_letter must honour recency ACROSS phrasings.

The within-pattern last-match fix (#1) only took the last occurrence of a *single*
pattern, then returned the first matching pattern in priority order. So a model
that states "the answer is A" (pattern 0) and then commits to a different letter
via a lower-priority form (\\boxed{C}, "Option C", a bare final line) had its
commitment ignored — the discarded first guess was returned. No network, no GPU.
"""
from __future__ import annotations

from trinity.orchestration.reward import extract_choice_letter, score_text


def test_boxed_revision_beats_an_earlier_answer_phrase():
    txt = r"I think the answer is A. On reflection, \boxed{C}."
    assert extract_choice_letter(txt) == "C"          # was "A"


def test_revised_to_correct_is_not_a_false_negative():
    txt = r"The answer is A. But wait: \boxed{C}."
    assert score_text("mmlu", txt, "C") == 1.0        # was 0.0


def test_discarded_first_guess_is_not_a_false_positive():
    # The dangerous case: a stale guess equal to the gold letter must NOT score.
    txt = r"The answer is A. But wait: \boxed{C}."
    assert score_text("mmlu", txt, "A") == 0.0        # was 1.0


def test_option_form_commitment_also_wins():
    assert extract_choice_letter("Answer: A at first. Final: Option D.") == "D"


def test_final_bare_line_commitment_wins_over_earlier_phrase():
    assert extract_choice_letter("the answer is A\n...\nB.") == "B"


# --- guards that must remain unchanged ---
def test_same_pattern_revision_still_resolves_to_last():
    assert extract_choice_letter("At first the answer is A. Wait, the answer is C.") == "C"


def test_prose_letter_is_still_not_a_choice():
    assert extract_choice_letter("A nice approach to this question.") is None


def test_font_wrapped_boxed_letter_still_extracted():
    assert extract_choice_letter(r"So \boxed{\text{B}}.") == "B"
    assert extract_choice_letter(r"\boxed{\textbf{D}}") == "D"


def test_single_clean_answer_unchanged():
    assert extract_choice_letter("The answer is B.") == "B"


def test_tie_break_prefers_higher_priority_pattern():
    # If two forms sit at the same position, pattern order breaks the tie; here
    # there is one unambiguous final commitment.
    assert extract_choice_letter("Option A.\nThe answer is C") == "C"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
