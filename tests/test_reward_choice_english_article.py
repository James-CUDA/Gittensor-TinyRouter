"""A letter-shaped English word is prose, not a committed multiple-choice answer.

``[A-J]`` under ``re.I`` also matches the article ``a`` and the pronoun ``I``, and
the committed-answer patterns accepted a zero-width ``\\b`` as their delimiter —
which the space after an article satisfies. So "the answer is a decrease in
pressure" was read as choice A (issue #413), which misgraded MMLU both ways:

* false positive — a wrong answer scored 1.0 against reference ``A``;
* false negative — because "last committed wins" ranks by match position, a
  trailing prose sentence outranked a genuine earlier ``\\boxed{C}``.

The rule pinned here: a committed letter is always delimited (punctuation, a
closing brace, or end of line); an article runs on into its noun phrase.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import extract_choice_letter, score_text


# ---------------------------------------------------------------------------
# The article is not a choice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "The answer is a decrease in the equilibrium constant.",
        "Thus, the answer is a shift toward the products.",
        "The correct answer is a stronger acid.",
        "In summary the answer is a linear relationship.",
        "Final answer: a monopoly has no close substitutes.",
    ],
)
def test_indefinite_article_is_not_choice_a(text):
    assert extract_choice_letter(text) is None


def test_pronoun_i_is_not_choice_i():
    # "I" is the other single-letter English word, and I is inside A-J.
    assert extract_choice_letter("The answer is I believe option C") == "C"


# ---------------------------------------------------------------------------
# Direction 1: the false positive is gone
# ---------------------------------------------------------------------------
def test_article_answer_no_longer_scores_against_reference_a():
    assert score_text("mmlu", "The answer is a decrease in pressure.", "A") == 0.0
    assert score_text("mmlu", "Final answer: a monopoly has no substitutes.", "A") == 0.0


# ---------------------------------------------------------------------------
# Direction 2: a genuine commitment is no longer overridden by trailing prose
# ---------------------------------------------------------------------------
def test_trailing_article_prose_does_not_override_a_committed_answer():
    boxed = "\\boxed{C}\n\nSo the answer is a straightforward application of the rule."
    assert extract_choice_letter(boxed) == "C"
    assert score_text("mmlu", boxed, "C") == 1.0

    final = "Final answer: D\nIn short, the answer is a consequence of conservation."
    assert extract_choice_letter(final) == "D"
    assert score_text("mmlu", final, "D") == 1.0


# ---------------------------------------------------------------------------
# A delimited lowercase letter is still a real commitment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is a.", "A"),
        ("The answer is a)", "A"),
        ("The answer is a,", "A"),
        ("The answer is I.", "I"),
        (r"\boxed{a}", "A"),
        (r"\boxed{ (b) }", "B"),
        ("the answer is c.", "C"),
    ],
)
def test_delimited_lowercase_letter_still_counts(text, expected):
    assert extract_choice_letter(text) == expected


# ---------------------------------------------------------------------------
# Genuine commitments followed by an explanation must not regress. Requiring a
# hard delimiter for EVERY letter would drop these, which is why the rule keys
# on the letter's spelling instead.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is A because it has the lowest ionization energy.", "A"),
        ("The answer is B because it has the lowest ionization energy.", "B"),
        ("The answer is C since the reaction is exothermic.", "C"),
        ("The answer is D as shown above.", "D"),
        ("The answer is A.", "A"),
        ("The answer is (C)", "C"),
        ("The answer is D", "D"),
        ("Answer: C", "C"),
        ("Final answer: B", "B"),
        (r"\boxed{D}", "D"),
        ("Option B is correct.", "B"),
    ],
)
def test_genuine_commitments_are_unchanged(text, expected):
    assert extract_choice_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The answer is an increase in entropy.",
        "The answer is the third option.",
    ],
)
def test_non_committal_prose_still_yields_nothing(text):
    assert extract_choice_letter(text) is None


# ---------------------------------------------------------------------------
# The issue #267 contract (last committed wins) is untouched
# ---------------------------------------------------------------------------
def test_last_committed_answer_still_wins():
    assert extract_choice_letter("I think the answer is B. But wait.\n\\boxed{D}") == "D"
    assert extract_choice_letter("\\boxed{B}\nFinal answer: D") == "D"
