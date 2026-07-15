"""Regression: the MMLU choice extractor must see through Markdown emphasis (#335).

Bolding the final choice in Markdown (``**B**``, ``*C*``, ``__B__``) is one of the
most common LLM output styles, but the emphasis delimiter sits between the
"answer is" cue and the letter, so the choice patterns never reached the letter and
``extract_choice_letter`` returned ``None`` -- a false negative that graded a correct
MMLU answer wrong. The extractor already unwraps LaTeX emphasis (``\\textbf{B}``);
it must unwrap the Markdown twin too, without fabricating a choice from a lone
``*``/``_`` in ordinary prose and without changing any answer that already extracts.

Pure / offline -- no network, no GPU.
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import extract_choice_letter, score_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("**D**", "D"),
        ("**B**", "B"),
        ("The answer is **B**.", "B"),
        ("My final answer is **A**.", "A"),
        ("*C*", "C"),               # single-asterisk italic
        ("__B__", "B"),             # underscore bold
        ("_D_", "D"),               # underscore italic
        ("**_B_**", "B"),           # nested bold+italic collapses fully
    ],
)
def test_markdown_emphasised_letter_is_extracted(text, expected):
    assert extract_choice_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "This is a * bullet with no answer here",
        "use snake_case variable names",
        "2 * 3 = 6, so the total is six",
        "nothing choice-like in this sentence at all",
    ],
)
def test_lone_delimiter_does_not_fabricate_a_choice(text):
    assert extract_choice_letter(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is B.", "B"),          # plain, already worked
        (r"\boxed{\text{B}}", "B"),         # LaTeX emphasis, already worked
        (r"\textbf{C}", "C"),
    ],
)
def test_existing_forms_unchanged(text, expected):
    assert extract_choice_letter(text) == expected


def test_score_text_mmlu_end_to_end():
    # The whole point: a correct bolded answer now scores 1.0, not 0.0.
    assert score_text("mmlu", "The answer is **B**.", "B") == 1.0
    assert score_text("mmlu", "The answer is B.", "B") == 1.0
    # A genuinely-wrong bolded answer still scores 0.0.
    assert score_text("mmlu", "The answer is **A**.", "B") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
